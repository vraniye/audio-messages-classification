# 🎙️ Speech Style Classifier

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue?logo=telegram)](https://telegram.org/)
[![ML](https://img.shields.io/badge/Machine-Learning-orange)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-Transformers-yellow)](https://huggingface.co/)

**AI-powered system for classifying speech style from voice messages**

</div>

## 📖 О проекте

Система для автоматического определения стиля речи (разговорный vs официально-деловой) из голосовых сообщений. Проект объединяет современные подходы машинного обучения с удобным Telegram-интерфейсом.

### 🎯 Основные возможности

- **🎙️ Распознавание речи** - конвертация аудио в текст с поддержкой русского и английского языков
- **🤖 Классификация текста** - multiple ML подходы для точного определения стиля речи
- **📱 Telegram бот** - удобный интерфейс для взаимодействия
- **⚡ REST API** - готовый бэкенд для интеграции

## 🏗️ Архитектура

```mermaid
graph LR
    A[Пользователь] --> B[Telegram Bot]
    B --> C[Flask Server]
    C --> D[ASR Module]
    D --> E[Text Classifier]
    E --> F[Response]
    F --> B --> A