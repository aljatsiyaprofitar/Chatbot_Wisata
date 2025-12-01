import google.generativeai as genai
import os

# masukkan langsung API key sementara untuk tes
genai.configure(api_key="AIzaSyA3MUd9nASEw9_FKbagBWv8Mnz275DpZ_Y")

for m in genai.list_models():
    print(m.name)
