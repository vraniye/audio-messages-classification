import os
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000/predict")
MIN_WORDS = os.getenv("MIN_WORDS")
MIN_DURATION = os.getenv("MIN_DURATION")

# --- команды ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎧 Привет! Отправь мне аудиосообщение — я попробую определить,"
        " разговорный это стиль или официальный."
    )

# --- обработка аудио ---
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Определяем тип сообщения
    voice = update.message.voice
    audio = update.message.audio

    if not voice and not audio:
        await update.message.reply_text("Отправь, пожалуйста, голосовое сообщение или аудиофайл.")
        return

    # Быстрая проверка длительности для голосовых сообщений
    if voice and voice.duration < MIN_DURATION:
        await update.message.reply_text(
            f"❌ Слишком короткое голосовое сообщение!\n"
            f"Минимальная длительность: *{MIN_DURATION} секунды*\n\n"
            f"Отправьте сообщение подлиннее.",
            parse_mode='Markdown'
        )
        return

    # Показываем индикатор набора текста
    await update.message.chat.send_action(action="typing")

    # Скачиваем файл
    file = await context.bot.get_file(voice.file_id if voice else audio.file_id)
    file_bytes = await file.download_as_bytearray()

    # Отправляем на сервер
    files = {'file': ('audio.ogg', file_bytes, 'audio/ogg')}
    try:
        response = requests.post(f"{SERVER_URL}/classify", files=files, timeout=60)

        if response.status_code != 200:
            await update.message.reply_text(f"❌ Ошибка сервера: {response.status_code}")
            return

        data = response.json()

        # Обрабатываем ответ от сервера
        if data.get('success'):
            label = data.get('label')
            class_name = data.get('label_name', 'Неизвестно')
            confidence = data.get('confidence', 0)
            text = data.get('text', '')
            duration = data.get('duration')
            word_count = data.get('word_count')

            # Формируем детализированный ответ
            response_parts = [
                f"🏷 *Стиль речи:* {class_name}",
                f"📊 *Код:* {label}",
                f"🎯 *Уверенность:* {confidence:.2f}"
            ]

            # Добавляем дополнительную информацию если есть
            if duration:
                response_parts.append(f"⏱ *Длительность:* {duration:.1f}с")
            if word_count:
                response_parts.append(f"📝 *Слов распознано:* {word_count}")

            response_parts.append(f"\n*Текст:*\n{text[:400]}{'...' if len(text) > 400 else ''}")

            await update.message.reply_text(
                "\n".join(response_parts),
                parse_mode='Markdown'
            )

        else:
            # Обработка ошибок от сервера
            error_msg = data.get('error', 'Неизвестная ошибка')
            if "слишком короткое" in error_msg.lower():
                await update.message.reply_text(
                    f"❌ {error_msg}\n\n"
                    f"Попробуйте отправить сообщение короче (*{MIN_DURATION}+ секунд*).",
                    parse_mode='Markdown'
                )
            elif "слов" in error_msg.lower():
                await update.message.reply_text(
                    f"❌ {error_msg}\n\n"
                    f"Попробуйте сказать больше (*{MIN_WORDS} слов*).",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text(f"❌ Ошибка обработки: {error_msg}")

    except requests.exceptions.Timeout:
        await update.message.reply_text(
            "⏱ Сервер не отвечает в течение минуты.\n"
            "Попробуйте повторить запрос позже."
        )
    except requests.exceptions.ConnectionError:
        await update.message.reply_text(
            "🔌 Не удается подключиться к серверу.\n"
            "Убедитесь, что сервер запущен и доступен."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Неожиданная ошибка: {e}")


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
