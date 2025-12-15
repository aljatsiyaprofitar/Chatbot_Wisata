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
# 0. CONFIGURATION & ENV
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

# MENGGUNAKAN GEMINI 2.5 FLASH (TANPA FAILSAFE)
model = genai.GenerativeModel("gemini-2.5-flash")
logger.info("System Status: Running on Gemini 2.5 Flash")

# ==========================================
# 1. DATA ENGINE
# ==========================================

try:
    with open("tourism_jogja.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info(f"Database loaded: {len(raw_data)} items.")
except FileNotFoundError:
    logger.error("File tourism_jogja.json tidak ditemukan!")
    raw_data = []

df = pd.DataFrame(raw_data)

# Normalisasi Data
df['name_clean'] = df['name'].astype(str).str.lower()
df['kategori_clean'] = df['kategori'].astype(str).str.lower()
df['deskripsi_clean'] = df['deskripsi'].astype(str).str.lower()

# Search Soup
df["search_text"] = (
    df["name_clean"] + " " + 
    df["kategori_clean"] + " " + 
    df["deskripsi_clean"]
)

# === VIP SCORING RULES (TIERED PRIORITY) ===
# Mengatur prioritas spesifik: 
# Tier 1 (Skor 5000): Borobudur, Prambanan, Parangtritis (Top Priority)
# Tier 2 (Skor 4000): Malioboro
# Tier 3 (Skor 3000): Kraton, Tugu
# Tier 4 (Skor 1000): Taman Sari (Lower Priority)
VIP_SCORE_MAPPING = {
    "candi borobudur": 5000.0,
    "candi prambanan": 5000.0,
    "pantai parangtritis": 5000.0,
    
    "malioboro": 4000.0,
    "jalan malioboro": 4000.0,
    
    "kraton": 3000.0,
    "keraton": 3000.0,
    "tugu yogyakarta": 3000.0,
    "tugu jogja": 3000.0,
    
    "taman sari": 1000.0,
    "tamansari": 1000.0,
    "pangandaran": 1000.0
}

# Init Vectorizer
vectorizer = TfidfVectorizer(stop_words=None) 
tfidf_matrix = vectorizer.fit_transform(df['search_text'])

# ==========================================
# 2. INTELLIGENT RETRIEVAL SYSTEM (TIERED)
# ==========================================

class RetrievalEngine:
    
    # Mapping Kategori yang diperluas
    CATEGORY_MAPPING = {
        'pantai': ['pantai', 'laut', 'samudra', 'pasir', 'bahari', 'coast'],
        'candi': ['candi', 'temple', 'sejarah', 'budaya', 'situs', 'arkeologi', 'warisan'],
        'alam': ['alam', 'gunung', 'bukit', 'hutan', 'air terjun', 'goa', 'gua', 'kebun', 'nature'],
        'kuliner': ['makan', 'kuliner', 'restoran', 'cafe', 'kafe', 'warung', 'jajan', 'food'],
        'belanja': ['belanja', 'mall', 'pasar', 'oleh-oleh', 'shop', 'souvenir'],
        'keluarga': ['keluarga', 'anak', 'taman', 'rekreasi', 'edukasi', 'wahana', 'play'],
        'sejarah': ['museum', 'monumen', 'benteng', 'sejarah', 'history'],
        'kota': ['kota', 'alun-alun', 'tugu', 'nol kilometer', 'city']
    }

    @staticmethod
    def extract_user_intent(query):
        q_lower = query.lower()
        params = {
            "top_n": 3,
            "is_comparison": False,
            "is_popular_search": False, # Default False, aktif jika ada keyword
            "price_filter": None,
            "category_intent": None,
            "exclude_ids": []
        }

        # 1. Deteksi Jumlah
        number_match = re.search(r'\b(\d+)\b', q_lower)
        if number_match:
            params["top_n"] = min(int(number_match.group(1)), 10)

        # 2. Deteksi Perbandingan
        if any(x in q_lower for x in ["banding", "beda", "vs"]):
            params["is_comparison"] = True
            params["top_n"] = 2 

        # 3. Deteksi Popularitas (Explicit)
        popular_keywords = ['terkenal', 'hits', 'terbaik', 'ikon', 'populer', 'wajib', 'rekomendasi', 'wisata jogja', 'bagus']
        
        # Jika query mengandung kata populer ATAU query sangat pendek (umum)
        if any(w in q_lower for w in popular_keywords) or len(q_lower.split()) <= 3:
            params["is_popular_search"] = True

        # 4. Deteksi Harga
        if any(x in q_lower for x in ["gratis", "free", "0 rupiah"]):
            params["price_filter"] = 'free'
        elif any(x in q_lower for x in ["murah", "terjangkau", "hemat"]):
            params["price_filter"] = 'cheap'

        # 5. Deteksi Kategori
        detected_cats = []
        for key, keywords in RetrievalEngine.CATEGORY_MAPPING.items():
            if any(k in q_lower for k in keywords):
                detected_cats.append(key)
        
        if detected_cats:
            params["category_intent"] = detected_cats

        return params

    @staticmethod
    def get_recommendations(query, user_history_ids, intent_params):
        filtered_df = df.copy()

        # --- LAYER 1: STRICT CATEGORY FILTER ---
        if intent_params['category_intent']:
            target_keywords = []
            for cat in intent_params['category_intent']:
                target_keywords.extend(RetrievalEngine.CATEGORY_MAPPING[cat])
            
            pattern = '|'.join(target_keywords)
            filtered_df = filtered_df[filtered_df['search_text'].str.contains(pattern, regex=True)]
            
            # Fallback
            if filtered_df.empty:
                filtered_df = df.copy()

        # --- LAYER 2: PRICE FILTER ---
        if intent_params['price_filter'] == 'free':
            filtered_df = filtered_df[filtered_df['htm'] == 0]
        elif intent_params['price_filter'] == 'cheap':
            filtered_df = filtered_df[filtered_df['htm'] <= 25000]

        # --- LAYER 3: HISTORY FILTER ---
        if user_history_ids:
            filtered_df = filtered_df[~filtered_df['place_id'].isin(user_history_ids)]
        
        if filtered_df.empty:
            filtered_df = df.copy() # Reset jika habis

        # --- LAYER 4: SCORING (LOGIKA BARU DENGAN NEGATIVE FILTER) ---
        
        # A. Hitung Basic Relevance (0.0 - 1.0)
        query_vec = vectorizer.transform([query.lower()])
        sim_scores_all = cosine_similarity(query_vec, tfidf_matrix).flatten()
        subset_indices = filtered_df.index
        base_sim_scores = sim_scores_all[subset_indices]
        
        # B. Hitung Rating Score (0.0 - 1.0)
        ratings = filtered_df['rating'].values / 5.0
        
        # C. TIERED VIP BOOST (STRICTER)
        # Menambahkan NEGATIVE FILTER untuk membuang "Unit Office", "Tour", dll.
        NEGATIVE_KEYWORDS = ["unit", "office", "kantor", "tour", "travel", "kelinci", "homestay", "hotel"]
        
        vip_boosts = np.zeros(len(filtered_df))
        names_lower = filtered_df['name_clean'].values
        
        for i, name in enumerate(names_lower):
            boost = 0.0
            
            # 1. Cek Negative Keywords dulu
            # Jika nama tempat mengandung kata terlarang, skip boost VIP
            is_clean_name = True
            for neg in NEGATIVE_KEYWORDS:
                if neg in name:
                    is_clean_name = False
                    break
            
            if not is_clean_name:
                vip_boosts[i] = 0.0
                continue # Skip ke tempat selanjutnya, tempat ini tidak berhak dapat boost

            # 2. Cek setiap aturan VIP
            for phrase, score in VIP_SCORE_MAPPING.items():
                if phrase in name:
                    # Ambil skor tertinggi jika cocok beberapa
                    if score > boost:
                        boost = score
            vip_boosts[i] = boost

        # KOMPOSISI SKOR AKHIR
        if intent_params['is_popular_search']:
             # Mode populer: VIP Boost dominan mutlak
             final_scores = vip_boosts + (base_sim_scores * 2.0) + (ratings * 1.0)
        else:
             # Pencarian spesifik: Tetap beri prioritas VIP tapi perhatikan relevansi teks
             final_scores = (vip_boosts * 0.5) + (base_sim_scores * 5.0) + (ratings * 1.0)

        # Sorting Index sementara
        sorted_indices = final_scores.argsort()[::-1]
        
        # --- LAYER 5: DEDUPLIKASI & FINAL SELECTION ---
        # Pastikan tidak ada tempat dengan keyword utama yang sama muncul berulang
        
        DEDUP_KEYWORDS = ["malioboro", "borobudur", "prambanan", "parangtritis", "kraton", "tugu"]
        seen_keywords = set()
        final_indices = []
        
        top_n_needed = intent_params['top_n']
        
        for idx in sorted_indices:
            name = filtered_df.iloc[idx]['name_clean']
            
            skip = False
            for keyword in DEDUP_KEYWORDS:
                if keyword in name:
                    if keyword in seen_keywords:
                        skip = True # Keyword ini sudah ada yang mewakili di list final
                    else:
                        seen_keywords.add(keyword) # Tandai keyword ini sudah terpakai
                    break # Pindah ke cek nama berikutnya
            
            if not skip:
                final_indices.append(idx)
                if len(final_indices) >= top_n_needed:
                    break
        
        final_results = filtered_df.iloc[final_indices]
        return final_results

# ==========================================
# 3. GENERATIVE AI ENGINE
# ==========================================

async def generate_response(query, recommendations, intent_params):
    context_data = ""
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        context_data += (
            f"### DATA_WISATA_{i} ###\n"
            f"Nama: {row['name']}\n"
            f"Kategori: {row['kategori']}\n"
            f"Rating: {row['rating']}\n"
            f"HTM: {int(row['htm'])}\n"
            f"Deskripsi: {row['deskripsi']}\n\n"
        )

    task_instruction = "REKOMENDASIKAN tempat di atas. Urutan Nomor 1 adalah yang paling direkomendasikan."
    if intent_params['is_comparison']:
        task_instruction = "BANDINGKAN tempat di atas."

    system_prompt = f"""
    Kamu adalah Travel-O, asisten wisata Jogja.
    
    PERTANYAAN USER: "{query}"
    TUGAS: {task_instruction}

    DATA FAKTA (Gunakan HANYA ini):
    {context_data}

    ATURAN STRICT:
    1.  DILARANG menyebutkan tempat wisata yang TIDAK ada di DATA FAKTA di atas.
    2.  JANGAN gunakan format Markdown (*, _, #). Plain text saja.
    3.  Gunakan Bahasa Indonesia natural.
    4.  Jelaskan kenapa tempat ini ikonik/cocok (berdasarkan data).

    FORMAT WAJIB:
    [Nama Tempat]
    Kategori: ... | Rating: ... | HTM: ...
    Alasan: ...

    (Beri jarak antar tempat)
    """

    try:
        response = model.generate_content(system_prompt)
        text = response.text
        # Clean Markdown
        text = re.sub(r'[*_#`]', '', text) 
        return text.strip()
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return "Maaf, server sedang sibuk."

# ==========================================
# 4. HANDLERS
# ==========================================

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['shown_ids'] = [] 
    await update.message.reply_text("Halo! Travel-O siap bantu cari wisata hits di Jogja.")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    if 'shown_ids' not in context.user_data: context.user_data['shown_ids'] = []
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    intent = RetrievalEngine.extract_user_intent(user_query)
    recs = RetrievalEngine.get_recommendations(user_query, context.user_data['shown_ids'], intent)
    
    if not recs.empty:
        context.user_data['shown_ids'].extend(recs['place_id'].tolist())
        if len(context.user_data['shown_ids']) > 50: 
            context.user_data['shown_ids'] = context.user_data['shown_ids'][-50:]

    if recs.empty:
        reply = "Belum nemu yang pas nih. Coba cari yang lain?"
    else:
        reply = await generate_response(user_query, recs, intent)

    await update.message.reply_text(reply)

async def reset_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['shown_ids'] = []
    await update.message.reply_text("History reset!")

def main():
    if not TELEGRAM_TOKEN: return
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("reset", reset_history)) 
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    print("Bot Travel-O (VIP TIER MODE) Running...")
    app.run_polling()

if __name__ == "__main__":
    main()