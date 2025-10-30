# Speech Style Classifier

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue?logo=telegram)](https://telegram.org/)
[![ML](https://img.shields.io/badge/Machine-Learning-orange)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-Transformers-yellow)](https://huggingface.co/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Audio%20Processing-green)](https://ffmpeg.org/)

**AI-powered system for classifying speech style from voice messages**

</div>

## О проекте

Система для автоматического определения стиля речи (разговорный vs официально-деловой) из голосовых сообщений. Проект объединяет современные подходы машинного обучения с удобным Telegram-интерфейсом.

### Основные возможности

- **🎙️ Распознавание речи** — первично Salute Speech (HTTP/2 API), автоматически переключается на Google Speech в качестве резервного варианта
- **🤖 Классификация текста** - multiple ML подходы для точного определения стиля речи
- **🧠 Набор моделей** — logistic regression, TextCNN, BERT, fastText и ансамбль; режим «Все» запускает все модели и собирает сводку
- **📱 Telegram бот** - удобный интерфейс для взаимодействия
- **⚡ REST API** - готовый бэкенд для интеграции
- **🔊 Поддержка аудиоформатов** - работа с различными аудиоформатами через FFmpeg

## ⚙️ Конфигурация

Основные переменные окружения перечислены в `env.example`:

- `SALUTE_AUTH_KEY` — обязательный Base64(ClientID:ClientSecret) для Salute Speech.
- `SALUTE_SCOPE` — область доступа (по умолчанию `SALUTE_SPEECH_PERS`).
- `FFMPEG_BIN`, `FFPROBE_BIN` — пути к утилитам FFmpeg/FFprobe, если они не лежат в `PATH`.
- `MIN_WORDS`, `MIN_DURATION` — ограничения на длину распознанного текста и голосового сообщения.
- `SERVER_URL` — базовый URL FastAPI-сервера для Telegram-бота.

Для классификации доступны модели `logistic`, `cnn`, `bert`, `fasttext`, `ensemble` и специальный режим `all`, в котором финальная метка выбирается по большинству голосов (при равенстве — по средней и максимальной уверенности), а ответ содержит сводку и перечень индивидуальных прогнозов.

### Структура артефактов моделей

```
server/models/
├── lr/                 # Логистическая регрессия
│   ├── lr_model.pkl
│   └── tfidf_vectorizer.pkl
├── cnn/                # TextCNN
│   ├── cnn_best.pth
│   └── vocab.pkl
├── bert/               # Классификатор на основе RuBERT
│   └── bert_classifier_best.pth
├── fasttext/           # fastText baseline
│   ├── fasttext_ensemble.bin
│   └── label_index.json
└── ensemble/           # Ансамбль логрег + TextCNN
    ├── feature_union.joblib
    ├── logreg_elasticnet.joblib
    ├── textcnn_state.pt
    ├── sequence_vocab.json
    └── label_index.json
```

Крупные бинарники исключены из git (`.gitignore`), поэтому после клонирования репозитория необходимо поместить их вручную.

### Локальная проверка без бэкенда

Для быстрого теста моделей без запуска сервера и бота используйте CLI:

```bash
python tools/classify_text.py "Добрый день, коллеги. Прошу согласовать документ."
python tools/classify_text.py "Привет! Как дела?" -m logistic cnn
python tools/classify_text.py --list  # показать доступные модели
```

По умолчанию скрипт запускает режим `all` и выводит сводку по всем моделям вместе с голосованием.

## 🏗️ Архитектура

```mermaid
graph LR
    A[Пользователь] --> B[Telegram Bot]
    B --> C[FastAPI Server]
    C --> D[Audio Processor]
    D --> E[FFmpeg/FFprobe]
    E --> F[Speech Recognition]
    F --> G[Text Classifier]
    G --> H[Response]
    H --> B --> A
