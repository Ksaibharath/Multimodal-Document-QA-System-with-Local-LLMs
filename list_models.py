import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Available models:\n")

for model in genai.list_models():
    print(model.name, "->", model.supported_generation_methods)