import google.generativeai as genai
import os

# masukkan langsung API key sementara untuk tes
genai.configure(api_key="API_key_gemini")

for m in genai.list_models():
    print(m.name)
