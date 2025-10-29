import os
import requests
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000/predict")
MIN_WORDS = os.getenv("MIN_WORDS")
MIN_DURATION = os.getenv("MIN_DURATION")

# Словарь моделей
MODELS = {
    "Classic ML": "logistic",
    "Нейросеть": "cnn",
    "Transformer": "bert",
    "FastText": "fasttext",
    "Ансамбль": "ensemble",
    "Все": "all",
}
MODEL_DISPLAY = {value: key for key, value in MODELS.items()}
KEYBOARD_LAYOUT = [
    ["Classic ML", "Нейросеть", "Transformer"],
    ["FastText", "Ансамбль", "Все"],
]


# --- команды ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начало работы с выбором модели."""
    try:
        # Простая клавиатура без сложных параметров
        reply_markup = ReplyKeyboardMarkup(
            KEYBOARD_LAYOUT,
            resize_keyboard=True,
            one_time_keyboard=True
        )

        # Более простое сообщение
        welcome_text = (
            "🎧 Привет! Выбери модель для классификации:\n\n"
            "• Classic ML - классическая ML модель\n"
            "• Нейросеть - нейросетевая модель\n"
            "• Transformer - трансформерная модель\n"
            "• FastText - fastText-бейзлайн\n"
            "• Ансамбль - логрег + TextCNN\n"
            "• Все - запустить всё сразу и получить сводку\n\n"
            "После выбора отправь аудиосообщение для анализа."
        )

        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup
        )
    except Exception as e:
        print(f"Ошибка в start: {e}")
        # Резервный вариант - отправить простое сообщение без клавиатуры
        await update.message.reply_text(
            "🎧 Привет! Отправь мне аудиосообщение для классификации стиля речи."
        )


async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора модели из кнопок."""
    user_choice = update.message.text
    model_id = MODELS.get(user_choice)

    if model_id:
        # Сохраняем выбранную модель в контексте пользователя
        context.user_data['selected_model'] = model_id
        await update.message.reply_text(
            f"✅ Выбрана модель: *{user_choice}*\n\n"
            f"Теперь отправь мне аудиосообщение или аудиофайл для классификации.",
            parse_mode='Markdown',
            reply_markup=None  # Убираем клавиатуру после выбора
        )
    else:
        await update.message.reply_text(
            "Пожалуйста, выберите модель из предложенных вариантов.",
            reply_markup=ReplyKeyboardMarkup(
                KEYBOARD_LAYOUT,
                one_time_keyboard=True,
                resize_keyboard=True
            )
        )


# --- обработка аудио ---
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Проверяем, выбрана ли модель
    if 'selected_model' not in context.user_data:
        await update.message.reply_text(
            "❌ Сначала выберите модель для классификации!",
            reply_markup=ReplyKeyboardMarkup(
                KEYBOARD_LAYOUT,
                one_time_keyboard=True,
                resize_keyboard=True
            )
        )
        return

    # Определяем тип сообщения
    voice = update.message.voice
    audio = update.message.audio

    if not voice and not audio:
        await update.message.reply_text("Отправь, пожалуйста, голосовое сообщение или аудиофайл.")
        return

    # Быстрая проверка длительности для голосовых сообщений
    if voice and voice.duration < int(MIN_DURATION):
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

    # Отправляем на сервер с выбранной моделью
    files = {'file': ('audio.ogg', file_bytes, 'audio/ogg')}
    data = {'model': context.user_data['selected_model']}

    try:
        response = requests.post(f"{SERVER_URL}/classify", files=files, data=data, timeout=60)

        if response.status_code != 200:
            await update.message.reply_text(f"❌ Ошибка сервера: {response.status_code}")
            return

        response_data = response.json()

        # Обрабатываем ответ от сервера
        if response_data.get('success'):
            label = response_data.get('label')
            class_name = response_data.get('label_name', 'Неизвестно')
            confidence = float(response_data.get('confidence', 0))
            text = response_data.get('text', '')
            duration = response_data.get('duration')
            word_count = response_data.get('word_count')
            model_used = response_data.get('model') or context.user_data['selected_model']
            best_model = response_data.get('best_model')
            model_name = MODEL_DISPLAY.get(model_used, model_used)

            details = response_data.get('details') or []
            summary_entries = response_data.get('summary') or []

            # Формируем детализированный ответ
            response_parts = [
                f"🏷 *Стиль речи:* {class_name}",
                f"📊 *Код:* {label}",
                f"🎯 *Уверенность:* {confidence:.2f}",
                f"🤖 *Модель:* {model_name}"
            ]

            if model_used == "all" and best_model:
                best_model_name = MODEL_DISPLAY.get(best_model, best_model)
                response_parts.append(f"⭐ *Лучшая модель:* {best_model_name}")

            # Добавляем дополнительную информацию если есть
            if duration:
                response_parts.append(f"⏱ *Длительность:* {duration:.1f}с")
            if word_count:
                response_parts.append(f"📝 *Слов распознано:* {word_count}")

            if details:
                response_parts.append("📋 *Сводка по моделям:*")
                for item in details:
                    detail_model = item.get('model')
                    detail_name = MODEL_DISPLAY.get(detail_model, detail_model)
                    detail_label = item.get('label_name', 'Неизвестно')
                    detail_conf = float(item.get('confidence', 0))
                    response_parts.append(f"• {detail_name}: {detail_label} ({detail_conf:.2f})")
            elif summary_entries:
                response_parts.append("📋 *Сводка:*")
                for entry in summary_entries:
                    response_parts.append(f"• {entry}")

            response_parts.append(f"\n*Текст:*\n{text[:400]}{'...' if len(text) > 400 else ''}")

            await update.message.reply_text(
                "\n".join(response_parts),
                parse_mode='Markdown'
            )

        else:
            # Обработка ошибок от сервера
            error_msg = response_data.get('error', 'Неизвестная ошибка')
            if "слишком короткое" in error_msg.lower():
                await update.message.reply_text(
                    f"❌ {error_msg}\n\n"
                    f"Попробуйте отправить сообщение подлиннее (*{MIN_DURATION}+ секунд*).",
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


async def change_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для смены модели"""
    reply_markup = ReplyKeyboardMarkup(
        KEYBOARD_LAYOUT,
        one_time_keyboard=True,
        resize_keyboard=True,
        input_field_placeholder="Выберите модель..."
    )

    await update.message.reply_text(
        "Выберите модель для классификации:",
        reply_markup=reply_markup
    )


# --- запуск ---
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("model", change_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_model_selection))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    print("🤖 Telegram-бот запущен и ждёт сообщений...")
    app.run_polling()


if __name__ == "__main__":
    main()
