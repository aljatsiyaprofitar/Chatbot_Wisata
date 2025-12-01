import os
import json
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes

# ======================
# 1. LOAD ENV & GEMINI
# ======================
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-pro")

# ======================
# 2. LOAD JSON DATABASE
# ======================
with open("tourism_jogja.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)

df["name"] = df["name"].astype(str)
df["kategori"] = df["kategori"].astype(str)
df["deskripsi"] = df["deskripsi"].astype(str)
df["rating"] = df["rating"].astype(float)
df["htm"] = df["htm"].astype(float)

df["combined_text"] = (
    df["name"] + " - " +
    df["kategori"] + " - " +
    df["deskripsi"] + " - Rating: " +
    df["rating"].astype(str) +
    " - HTM: " + df["htm"].astype(int).astype(str)
).str.lower()

# ==========================
# 3. TF-IDF SEMANTIC SEARCH
# ==========================
vectorizer = TfidfVectorizer(stop_words=None)
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

def ambil_data_relevan(query, top_n=5):
    """Kembalikan N baris paling relevan berdasarkan kemiripan semantik."""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

def jawaban_gemini(query):
    """Generator jawaban natural dari Gemini dengan konteks RAG."""
    subset = ambil_data_relevan(query)

    context = "\n".join(
        f"- {row['name']} ({row['kategori']}), Rating {row['rating']:.1f}, "
        f"HTM Rp{int(row['htm'])}: {row['deskripsi'][:200]}..."
        for _, row in subset.iterrows()
    )

    prompt = f"""
Kamu adalah asisten wisata Yogyakarta yang ramah dan informatif.
Gunakan data berikut untuk menjawab:

📘 DATA WISATA RELEVAN:
{context}

❓ Pertanyaan pengguna:
"{query}"

INSTRUKSI PENTING:
    1. Jawab dengan bahasa Indonesia yang natural dan ramah.
    2. JANGAN gunakan format Markdown sama sekali (jangan ada huruf tebal/bold, jangan ada tanda ** atau __).
    3. Ikuti format tampilan berikut untuk rekomendasi:
       
       1. Nama Tempat
          Kategori: ... | Rating: ... | HTM: ...
          Alasan: (Ceritakan kenapa tempat ini cocok, ringkas saja)

    4. Berikan 3 rekomendasi terbaik jika user meminta saran.
    """


    response = model.generate_content(prompt)
    return response.text.strip()

# ====================
# 4. TELEGRAM HANDLER
# ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Halo! Saya adalah Travel-O, Asisten Wisata Jogja 🧭\n"
        "Tanya saja: 'rekomendasi pantai', 'wisata rating 4.5', pantai dengan htm murah, atau apa pun!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    reply = jawaban_gemini(user_input)
    await update.message.reply_text(reply)

# ====================
# 5. MAIN BOT
# ====================
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot sedang berjalan...")
    app.run_polling()

if __name__ == "__main__":
    main()