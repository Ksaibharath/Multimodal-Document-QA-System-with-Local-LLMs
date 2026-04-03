import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("API KEY FOUND:", bool(api_key))

genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-flash-latest")
response = model.generate_content("Say hello in one line.")
print(response.text)