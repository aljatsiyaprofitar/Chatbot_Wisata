import os
import json
import pandas as pd
import logging
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, CallbackQueryHandler, filters, ContextTypes

# ======================
# 0. SETUP LOGGING & ENV
# ======================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Pastikan TELEGRAM_BOT_TOKEN dan GEMINI_API_KEY ada di .env")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")

# ======================
# 1. LOAD & PREP DATA
# ======================
try:
    with open("tourism_jogja.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
except FileNotFoundError:
    # Data Dummy
    raw_data = [
        {"name": "Pantai Parangtritis", "kategori": "Pantai", "deskripsi": "Pantai ikonik dengan ombak besar dan sunset indah.", "rating": 4.5, "htm": 10000},
        {"name": "Tebing Breksi", "kategori": "Alam", "deskripsi": "Bekas tambang kapur yang estetik untuk foto.", "rating": 4.4, "htm": 10000},
        {"name": "HeHa Sky View", "kategori": "Modern", "deskripsi": "Restoran dengan pemandangan kota Jogja dari atas.", "rating": 4.6, "htm": 20000},
        {"name": "Malioboro", "kategori": "Belanja", "deskripsi": "Pusat belanja dan jalan-jalan malam hari.", "rating": 4.8, "htm": 0},
    ]

df = pd.DataFrame(raw_data)

# Preprocessing Text
df["combined_text"] = (
    df["name"] + " " + df["kategori"] + " " + df["deskripsi"]
).str.lower()

# Init Vectorizer
vectorizer = TfidfVectorizer(stop_words=None)
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# ==========================
# 2. LOGIC PENCARIAN CERDAS
# ==========================

def filter_data(df, query):
    """Melakukan filter data berdasarkan kata kunci harga/rating."""
    filtered_df = df.copy()
    query_lower = query.lower()
    
    # Filter Harga
    if "gratis" in query_lower or "free" in query_lower:
        filtered_df = filtered_df[filtered_df['htm'] == 0]
    elif "murah" in query_lower:
        filtered_df = filtered_df[filtered_df['htm'] <= 15000]
    
    # Filter Rating
    if "terbaik" in query_lower or "populer" in query_lower:
        filtered_df = filtered_df[filtered_df['rating'] >= 4.5]

    return filtered_df

def get_recommendations(query, top_n=3):
    """
    Kombinasi Filter, Semantic Search, DAN Boosting Rating.
    Ini menggantikan fungsi 'poin 5, 7, 11' di prompt Gemini,
    agar data yang dikirim ke Gemini SUDAH data yang terbaik/ikonik.
    """
    # 1. Filter Dasar
    subset_df = filter_data(df, query)
    
    if subset_df.empty:
        # Jika kosong, kembalikan top rated (tempat paling hits)
        return df.sort_values(by='rating', ascending=False).head(top_n)

    # 2. Semantic Search (Kecocokan Teks)
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Ambil skor hanya untuk subset data yang sudah difilter
    subset_indices = subset_df.index
    subset_scores = similarity_scores[subset_indices]
    
    # --- LOGIKA BARU: BOOSTING RATING ---
    # Kita gabungkan skor kemiripan teks dengan skor rating.
    # Rumus: 70% Kecocokan Teks + 30% Rating Tinggi.
    # Ini memastikan tempat "Ikonik" (rating tinggi) lebih prioritas muncul.
    
    ratings = subset_df['rating'].values
    # Normalisasi rating ke skala 0-1 (misal rating 5 jadi 1.0)
    normalized_ratings = ratings / 5.0 
    
    # Final Score
    final_scores = (subset_scores * 0.7) + (normalized_ratings * 0.3)
    
    # Urutkan berdasarkan skor tertinggi
    top_relative_indices = final_scores.argsort()[-top_n:][::-1]
    top_df_indices = subset_indices[top_relative_indices]
    
    return df.loc[top_df_indices]

# Update 2: Prompt Gemini (Strict tapi Tetap Luwes)
async def generate_gemini_response(query, history, recommendations):
    # 1. Siapkan Context (Sama seperti sebelumnya)
    context_str = ""
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        context_str += (
            f"urutan_ke_{i}: {row['name']}\n"
            f"- Kategori: {row['kategori']} | Rating: {row['rating']} | HTM: Rp{int(row['htm'])}\n"
            f"- Deskripsi: {row['deskripsi']}\n\n"
        )

    # 2. Siapkan History
    history_str = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in history[-3:]])

    # 3. Prompt
    # KITA PERTAHANKAN INSTRUKSI "IKONIK" TAPI DENGAN CARA AMAN:
    # Kita beritahu Gemini bahwa data yang dikirim INI ADALAH tempat ikonik tersebut.
    system_prompt = f"""
    Kamu adalah Travel-O, asisten wisata Jogja.
    
    DATA WISATA TERPILIH (Gunakan data ini sebagai sumber kebenaran mutlak):
    {context_str}

    RIWAYAT CHAT:
    {history_str}

    PERTANYAAN USER: "{query}"

    INSTRUKSI UTAMA:
    1.  Tugasmu adalah menjelaskan tempat-tempat di 'DATA WISATA TERPILIH' kepada user.
    2.  Data di atas sudah dipilihkan oleh sistem sebagai tempat yang paling cocok, ikonik, atau terbaik sesuai request user.
    3.  JANGAN menambahkan tempat wisata lain di luar daftar di atas (agar tombol Maps di aplikasi tidak error).
    4.  Jelaskan dengan gaya bahasa yang asik, menyakinkan, dan informatif seolah-olah ini memang rekomendasi terbaik darimu.

    FORMAT JAWABAN (Tanpa Markdown):
    [Nomor]. [Nama Tempat Sesuai Data]
    Kategori: ... | Rating: ... | HTM: ...
    Alasan: [Jelaskan kenapa tempat ini menarik/ikonik berdasarkan deskripsi data]

    PENTING:
    Nama tempat harus persis sama dengan Data Wisata agar user tidak nyasar.
    """

    try:
        response = model.generate_content(system_prompt)
        text = response.text.strip()
        clean_text = text.replace("*", "").replace("#", "").replace("__", "")
        return clean_text

    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return "Maaf, sistem sedang sibuk."
    
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['history'] = []
    await update.message.reply_text(
        "Halo! Saya Travel-O. Silakan tanya tentang wisata Jogja."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    
    if 'history' not in context.user_data:
        context.user_data['history'] = []
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    # --- PERUBAHAN UTAMA DI SINI ---
    
    # 1. Kita ambil HANYA 3 rekomendasi teratas secara teknis (Python)
    # Jangan ambil 5, biar Gemini tidak 'pilih kasih'.
    top_3_places = get_recommendations(user_query, top_n=3) 
    
    # 2. Kirim HANYA 3 data ini ke Gemini untuk dibuatkan narasi
    bot_reply = await generate_gemini_response(
        user_query, 
        context.user_data['history'], 
        top_3_places  # <--- Pastikan yang dikirim adalah variabel yang sama
    )

    context.user_data['history'].append({"user": user_query, "bot": bot_reply})

    # 3. Buat tombol dari 3 data yang SAMA PERSIS
    keyboard = []
    for _, row in top_3_places.iterrows():
        place_name = row['name']
        # Trik google maps search query
        maps_url = f"https://www.google.com/maps/search/?api=1&query={place_name.replace(' ', '+')}+Yogyakarta"
        
        keyboard.append([InlineKeyboardButton(f"📍 Peta {place_name}", url=maps_url)])

    keyboard.append([InlineKeyboardButton("🗑️ Hapus Chat", callback_data="clear_history")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Kirim
    await update.message.reply_text(bot_reply, reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "clear_history":
        context.user_data['history'] = []
        await query.edit_message_text(text="Riwayat chat telah dihapus.")

# ====================
# 4. MAIN LOOP
# ====================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log error yang disebabkan oleh Update."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    
    # Opsional: Jika error terjadi saat user chat, beritahu mereka
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Maaf, koneksi ke server Telegram sedang lambat. Mohon coba lagi."
            )
        except:
            # Jika mengirim pesan error pun gagal, abaikan saja
            pass

def main():
    # Mengatur timeout menjadi lebih lama (30 detik) agar tidak mudah putus
    # Defaultnya biasanya hanya 5-10 detik
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(30) 
        .read_timeout(30)
        .write_timeout(30)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button_handler))
    
    # Daftarkan error handler
    app.add_error_handler(error_handler)

    print("Bot sedang berjalan...")
    
    # Gunakan allowed_updates agar bot hanya memproses pesan yang diperlukan (lebih hemat data)
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()