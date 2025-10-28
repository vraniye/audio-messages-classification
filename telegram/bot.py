import os
import requests
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes, ConversationHandler

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000")
MIN_WORDS = int(os.getenv("MIN_WORDS", 5))
MIN_DURATION = int(os.getenv("MIN_DURATION", 3))

# Состояние для выбора модели
CHOOSING_MODEL = 1

# Доступные модели
MODELS = {
    "classic": "Classic ML",
    "neural": "Нейросеть",
    "transformer": "Transformer"
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начало работы с выбором модели."""
    keyboard = [
        ["Classic ML", "Нейросеть", "Transformer"]
    ]
    reply_markup = ReplyKeyboardMarkup(
        keyboard,
        one_time_keyboard=True,
        resize_keyboard=True,
        input_field_placeholder="Выберите модель..."
    )

    await update.message.reply_text(
        "🎧 Привет! Выбери модель для классификации:\n\n"
        "• *Classic ML* - классическая ML модель\n"
        "• *Нейросеть* - нейросетевая модель\n"
        "• *Transformer* - трансформерная модель\n\n"
        "После выбора модели отправь мне аудиосообщение — я попробую определить, "
        "разговорный это стиль или официальный.",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

    return CHOOSING_MODEL


async def choose_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора модели."""
    user_choice = update.message.text

    model_mapping = {
        "Classic ML": "classic",
        "Нейросеть": "neural",
        "Transformer": "transformer"
    }

    chosen_model = model_mapping.get(user_choice)

    if chosen_model:
        context.user_data['chosen_model'] = chosen_model
        await update.message.reply_text(
            f"✅ Выбрана модель: *{MODELS[chosen_model]}*\n\n"
            f"Теперь отправь мне аудиосообщение для анализа.",
            reply_markup=ReplyKeyboardRemove(),
            parse_mode='Markdown'
        )
        return ConversationHandler.END
    else:
        keyboard = [["Classic ML", "Нейросеть", "Transformer"]]
        reply_markup = ReplyKeyboardMarkup(
            keyboard,
            one_time_keyboard=True,
            resize_keyboard=True
        )
        await update.message.reply_text(
            "❌ Пожалуйста, выбери модель из предложенных вариантов:",
            reply_markup=reply_markup
        )
        return CHOOSING_MODEL


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена выбора модели."""
    await update.message.reply_text(
        "Операция отменена. Используй /start чтобы начать заново.",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка аудиосообщений."""
    # Проверяем, выбрана ли модель
    if 'chosen_model' not in context.user_data:
        await update.message.reply_text(
            "❌ Сначала выбери модель для классификации с помощью /start"
        )
        return

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

    # Отправляем на сервер с выбранной моделью
    files = {'file': ('audio.ogg', file_bytes, 'audio/ogg')}
    data = {'model': context.user_data['chosen_model']}  # Передаем выбранную модель

    try:
        response = requests.post(f"{SERVER_URL}/classify", files=files, data=data, timeout=60)

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
            model_used = data.get('model', context.user_data['chosen_model'])

            # Формируем детализированный ответ
            response_parts = [
                f"🤖 *Модель:* {MODELS.get(model_used, model_used)}",
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
        r = requests.get(SERVER_URL.replace("/predict", "/health"))
        if r.status_code == 200:
            health_data = r.json()
            models_status = health_data.get('models_loaded', {})

            status_text = "✅ Сервер в сети!\n\n🤖 Статус моделей:\n"
            for model, loaded in models_status.items():
                status_text += f"• {MODELS.get(model, model)}: {'✅' if loaded else '❌'}\n"

            await update.message.reply_text(status_text)
        else:
            await update.message.reply_text("⚠️ Сервер недоступен.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда помощи"""
    help_text = (
        "🤖 *Доступные команды:*\n"
        "/start - выбрать модель и начать работу\n"
        "/ping - проверить статус сервера и моделей\n"
        "/help - показать эту справку\n\n"
        "🎧 *Доступные модели:*\n"
        "• Classic ML - классические алгоритмы\n"
        "• Нейросеть - нейросетевые архитектуры\n"
        "• Transformer - трансформерные модели\n\n"
        "📋 *Требования к аудио:*\n"
        f"• Минимальная длительность: {MIN_DURATION} секунд\n"
        f"• Минимальное количество слов: {MIN_WORDS}"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')


# --- запуск ---
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # ConversationHandler для выбора модели
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSING_MODEL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, choose_model)
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    print("🤖 Telegram-бот запущен и ждёт сообщений...")
    app.run_polling()


if __name__ == "__main__":
    main()