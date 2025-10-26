import os
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000/predict")

# --- команды ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎧 Привет! Отправь мне аудиосообщение — я попробую определить,"
        " разговорный это стиль или официальный."
    )

# --- обработка аудио ---
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice or update.message.audio
    if not voice:
        await update.message.reply_text("Отправь, пожалуйста, аудиосообщение.")
        return

    file = await context.bot.get_file(voice.file_id)
    file_bytes = await file.download_as_bytearray()

    files = {'file': ('audio.ogg', file_bytes, 'audio/ogg')}
    try:
        response = requests.post(SERVER_URL, files=files)
        data = response.json()
        label = data.get('label')
        class_name = data.get('class_name')
        await update.message.reply_text(f"🗣 Стиль речи: {class_name}\n📊 Код: {label}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при обработке: {e}")

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверка доступности сервера"""
    try:
        r = requests.get(SERVER_URL.replace("/predict", "/docs"))
        if r.status_code == 200:
            await update.message.reply_text("✅ Сервер в сети!")
        else:
            await update.message.reply_text("⚠️ Сервер недоступен.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

# --- запуск ---
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    print("🤖 Telegram-бот запущен и ждёт сообщений...")
    app.run_polling()

if __name__ == "__main__":
    main()
