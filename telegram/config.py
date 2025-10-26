import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000/predict")

if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN не найден в .env")
