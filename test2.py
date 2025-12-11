import os
import json
import re
import logging
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes

# ==========================================
# 0. CONFIGURATION & ENVIRONMENT
# ==========================================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    raise ValueError("CRITICAL: Token tidak ditemukan di .env")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ==========================================
# 1. DATA ENGINE (STRICT PREPROCESSING)
# ==========================================

try:
    with open("tourism_jogja.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info(f"Database loaded: {len(raw_data)} items.")
except FileNotFoundError:
    logger.error("File tourism_jogja.json tidak ditemukan!")
    raw_data = []

df = pd.DataFrame(raw_data)

# 1. Normalisasi Data
# Pastikan kolom string lowercase untuk pencarian akurat
df['name_clean'] = df['name'].astype(str).str.lower()
df['kategori_clean'] = df['kategori'].astype(str).str.lower()
df['deskripsi_clean'] = df['deskripsi'].astype(str).str.lower()

# Soup text untuk TF-IDF (Pencarian isi)
df["search_text"] = (
    df["name_clean"] + " " + 
    df["kategori_clean"] + " " + 
    df["deskripsi_clean"]
)

# 2. ICONIC ENTITY MAPPING (Pencocokan Cerdas)
# Format: "Kata Kunci User": ["Frasa Wajib Ada di Nama Tempat"]
# Ini mencegah "Taman Kelinci Borobudur" muncul saat user cari "Borobudur" (Target: Candi Borobudur)
ICONIC_MAPPING = {
    "borobudur": ["candi borobudur"],
    "prambanan": ["candi prambanan"],
    "malioboro": ["malioboro", "jalan malioboro"],
    "parangtritis": ["pantai parangtritis"],
    "kraton": ["kraton", "keraton"],
    "tugu": ["tugu yogyakarta", "tugu jogja", "tugu pal"],
    "breksi": ["tebing breksi"],
    "merapi": ["gunung merapi", "museum gunungapi", "bunker kaliadem"],
    "tamansari": ["taman sari", "tamansari", "water castle"],
    "heha": ["heha sky", "heha ocean"],
    "pine": ["hutan pinus"],
    "mangunan": ["kebun buah mangunan", "hutan pinus mangunan"]
}

# 3. Vectorizer Initialization
vectorizer = TfidfVectorizer(stop_words=None) 
tfidf_matrix = vectorizer.fit_transform(df['search_text'])

# ==========================================
# 2. INTELLIGENT RETRIEVAL SYSTEM (STRICT)
# ==========================================

class RetrievalEngine:
    
    # KATA KUNCI KATEGORI (MAPPING LOGIC)
    CATEGORY_MAPPING = {
        'pantai': ['pantai', 'laut', 'samudra', 'pasir', 'bahari'],
        'candi': ['candi', 'sejarah', 'budaya', 'situs', 'arkeologi'],
        'alam': ['alam', 'gunung', 'bukit', 'hutan', 'air terjun', 'goa', 'gua', 'kebun'],
        'kuliner': ['makan', 'kuliner', 'restoran', 'cafe', 'kafe', 'warung', 'jajan'],
        'belanja': ['belanja', 'mall', 'pasar', 'oleh-oleh', 'shop'],
        'keluarga': ['keluarga', 'anak', 'taman', 'rekreasi', 'edukasi', 'wahana'],
        'malam': ['malam', 'night', 'bintang', 'lampu']
    }

    @staticmethod
    def extract_user_intent(query):
        """
        Membedah query user dengan Strict Parsing.
        """
        q_lower = query.lower()
        params = {
            "top_n": 3,
            "is_comparison": False,
            "is_popular_search": False,
            "price_filter": None, # 'free', 'cheap', None
            "category_intent": None, # Akan diisi list kategori yang cocok
            "exclude_ids": []
        }

        # 1. Deteksi Jumlah
        number_match = re.search(r'\b(\d+)\b', q_lower)
        if number_match:
            params["top_n"] = min(int(number_match.group(1)), 10)

        # 2. Deteksi Perbandingan
        if any(x in q_lower for x in ["banding", "beda", "vs", "bagusan mana"]):
            params["is_comparison"] = True
            params["top_n"] = 2 

        # 3. Deteksi Popularitas
        popular_keywords = ['terkenal', 'hits', 'terbaik', 'ikon', 'populer', 'wajib', 'paling bagus', 'rekomendasi']
        if any(w in q_lower for w in popular_keywords) or (len(q_lower.split()) <= 2 and "wisata" in q_lower):
            params["is_popular_search"] = True

        # 4. Deteksi Harga
        if any(x in q_lower for x in ["gratis", "free", "0 rupiah", "tanpa biaya"]):
            params["price_filter"] = 'free'
        elif any(x in q_lower for x in ["murah", "terjangkau", "hemat", "low budget"]):
            params["price_filter"] = 'cheap'

        # 5. DETEKSI KATEGORI (STRICT)
        detected_cats = []
        for key, keywords in RetrievalEngine.CATEGORY_MAPPING.items():
            if any(k in q_lower for k in keywords):
                detected_cats.append(key)
        
        if detected_cats:
            params["category_intent"] = detected_cats
            # Jika user spesifik minta kategori, kita kurangi agresivitas popularitas umum
            params["is_popular_search"] = False 

        return params

    @staticmethod
    def get_recommendations(query, user_history_ids, intent_params):
        """
        Logika Filter Bertingkat (Layered Filtering).
        """
        filtered_df = df.copy()

        # --- LAYER 1: STRICT CATEGORY FILTER ---
        if intent_params['category_intent']:
            target_keywords = []
            for cat in intent_params['category_intent']:
                target_keywords.extend(RetrievalEngine.CATEGORY_MAPPING[cat])
            
            pattern = '|'.join(target_keywords)
            filtered_df = filtered_df[filtered_df['search_text'].str.contains(pattern, regex=True)]
            
            if filtered_df.empty:
                filtered_df = df.copy()

        # --- LAYER 2: PRICE FILTER ---
        if intent_params['price_filter'] == 'free':
            filtered_df = filtered_df[filtered_df['htm'] == 0]
        elif intent_params['price_filter'] == 'cheap':
            filtered_df = filtered_df[filtered_df['htm'] <= 20000]

        # --- LAYER 3: HISTORY FILTER ---
        if user_history_ids:
            filtered_df = filtered_df[~filtered_df['place_id'].isin(user_history_ids)]
        
        if filtered_df.empty:
            if intent_params['price_filter'] == 'free': 
                filtered_df = df[df['htm'] == 0]
            else:
                filtered_df = df.copy()

        # --- LAYER 4: SCORING & RANKING ---
        
        # A. TF-IDF Similarity
        query_vec = vectorizer.transform([query.lower()])
        sim_scores_all = cosine_similarity(query_vec, tfidf_matrix).flatten()
        subset_indices = filtered_df.index
        sim_scores = sim_scores_all[subset_indices]

        # B. Rating Score
        ratings = filtered_df['rating'].values
        norm_ratings = ratings / 5.0

        # C. ENTITY PRIORITY SCORE (LOGIKA BARU - ANTI SALAH KAPRAH)
        # Skor ini khusus untuk memastikan jika user cari "Borobudur", 
        # yang naik adalah "Candi Borobudur" (5.0), bukan "Taman Kelinci Borobudur" (0.0).
        
        entity_scores = np.zeros(len(filtered_df))
        q_lower = query.lower()
        
        # 1. Cek Specific Entity Request
        for key, targets in ICONIC_MAPPING.items():
            if key in q_lower:
                # User menyebut keyword ikonik (misal: "borobudur")
                for target in targets:
                    # Cari baris yang namanya mengandung target spesifik (misal: "candi borobudur")
                    # Kita pakai str.contains agar "Kawasan Malioboro" tetap kena jika targetnya "malioboro"
                    # Tapi "Taman Kelinci Borobudur" TIDAK kena karena targetnya "candi borobudur"
                    match_mask = filtered_df['name_clean'].str.contains(target, regex=False)
                    
                    # Beri skor SUPER TINGGI (5.0) agar menang mutlak lawan skor lain
                    entity_scores[match_mask] = 5.0

        # 2. General VIP Boost (Untuk query "wisata hits" tanpa menyebut nama tempat)
        # Ini fallback agar tempat ikonik tetap muncul kalau query-nya umum
        general_vip_scores = np.zeros(len(filtered_df))
        if intent_params['is_popular_search']:
            # Kumpulkan semua target ikonik
            all_iconic_targets = [t for targets in ICONIC_MAPPING.values() for t in targets]
            for target in all_iconic_targets:
                match_mask = filtered_df['name_clean'].str.contains(target, regex=False)
                general_vip_scores[match_mask] = 1.0

        # D. PEMBOBOTAN DINAMIS
        if np.max(entity_scores) > 0:
            # Skenario 1: User mencari Entitas Spesifik (Borobudur, Malioboro, dll)
            # Entity Score (5.0) mendominasi total skor.
            final_scores = (entity_scores * 1.0) + (sim_scores * 0.1) + (norm_ratings * 0.1)
        
        elif intent_params['is_popular_search']:
            # Skenario 2: User cari "Wisata Hits/Terkenal" (General)
            # Kombinasi Rating tinggi + General VIP Boost
            final_scores = (general_vip_scores * 0.5) + (norm_ratings * 0.3) + (sim_scores * 0.2)
        
        elif intent_params['category_intent']:
            # Skenario 3: User cari Kategori (Pantai, Goa)
            # Relevansi Teks & Rating jadi prioritas
            # General VIP tetap bantu sedikit agar pantai terkenal naik
            final_scores = (sim_scores * 0.6) + (norm_ratings * 0.3) + (general_vip_scores * 0.1)
        
        else:
            # Skenario 4: Pencarian Spesifik Lainnya (Long tail)
            final_scores = (sim_scores * 0.8) + (norm_ratings * 0.2)

        # Sorting Top N
        top_n = intent_params['top_n']
        sorted_indices = final_scores.argsort()[::-1][:top_n]
        
        final_results = filtered_df.iloc[sorted_indices]
        return final_results

# ==========================================
# 3. GENERATIVE AI ENGINE (STRICT RAG)
# ==========================================

async def generate_response(query, recommendations, intent_params):
    # Siapkan Context Data (JSON Values)
    # Ini memastikan Gemini HANYA baca dari sini
    context_data = ""
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        context_data += (
            f"### DATA_WISATA_{i} ###\n"
            f"Nama Resmi: {row['name']}\n"
            f"Kategori JSON: {row['kategori']}\n"
            f"Rating: {row['rating']}\n"
            f"Harga Tiket: Rp {int(row['htm'])}\n"
            f"Deskripsi Database: {row['deskripsi']}\n\n"
        )

    # Mode Instruksi
    if intent_params['is_comparison']:
        task_instruction = "BANDINGKAN data di atas (apple-to-apple) dari segi harga, rating, dan pengalaman."
    else:
        task_instruction = "REKOMENDASIKAN tempat di atas sesuai format wajib."

    # PROMPT SANGAT STRICT
    system_prompt = f"""
    Bertindaklah sebagai 'Travel-O', pemandu wisata Jogja.
    
    PERTANYAAN USER: "{query}"
    
    TUGAS: {task_instruction}

    SUMBER DATA (MUTLAK):
    {context_data}

    ATURAN HUKUM (JANGAN DILANGGAR):
    1.  DILARANG menambah/mengarang tempat wisata di luar "SUMBER DATA". Jika data cuma 2, jawab 2.
    2.  DILARANG menggunakan Markdown (bold, italic, list, dll). Teks polos saja.
    3.  Gunakan Bahasa Indonesia yang ramah, sopan, dan mengalir (natural).
    4.  Jangan menyebut "berdasarkan database" atau "berdasarkan JSON", ucapkan seolah kamu tahu sendiri.
    
    FORMAT OUTPUT WAJIB (Copy-paste struktur ini):
    [Nama Tempat]
    Kategori: [Kategori] | Rating: [Rating] | HTM: [Harga]
    Alasan: [Satu kalimat padat kenapa ini cocok untuk user berdasarkan deskripsi]

    (Spasi antar item)
    """

    try:
        response = model.generate_content(system_prompt)
        text = response.text
        
        # FINAL CLEANING (Python Regex Cleaner)
        # Menghapus paksa simbol markdown yang mungkin lolos
        text = re.sub(r'[*_#`]', '', text) 
        text = re.sub(r'\n{3,}', '\n\n', text) # Hapus spasi berlebih
        
        return text.strip()

    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return "Maaf, server sedang sibuk. Coba tanya lagi ya."

# ==========================================
# 4. TELEGRAM HANDLERS
# ==========================================

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['shown_ids'] = [] 
    await update.message.reply_text(
        "Halo! Travel-O siap bantu. Mau cari wisata apa?\n"
        "Contoh: 'Pantai murah', 'Candi terpopuler', 'Tempat makan keluarga'"
    )

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    if 'shown_ids' not in context.user_data: context.user_data['shown_ids'] = []
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    # 1. Analisis & Filter
    intent = RetrievalEngine.extract_user_intent(user_query)
    
    # 2. Ambil Data
    recs = RetrievalEngine.get_recommendations(user_query, context.user_data['shown_ids'], intent)
    
    # 3. Update History
    if not recs.empty:
        context.user_data['shown_ids'].extend(recs['place_id'].tolist())
        if len(context.user_data['shown_ids']) > 50: # Keep memory small
            context.user_data['shown_ids'] = context.user_data['shown_ids'][-50:]

    # 4. Generate Jawaban
    if recs.empty:
        reply = "Waduh, Travel-O belum nemu tempat yang cocok sama kriteria itu. Coba kata kunci lain ya?"
    else:
        reply = await generate_response(user_query, recs, intent)

    await update.message.reply_text(reply)

async def reset_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['shown_ids'] = []
    await update.message.reply_text("Riwayat rekomendasi di-reset!")

# ==========================================
# 5. MAIN APP
# ==========================================

def main():
    if not TELEGRAM_TOKEN: return
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("reset", reset_history)) 
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    print("Bot Travel-O (Strict Mode) Running...")
    app.run_polling()

if __name__ == "__main__":
    main()