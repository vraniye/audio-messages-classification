import os
import logging
import joblib
import numpy as np
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classifier")


class TextClassifier:
    """
    Класс для классификации стиля речи.
    Поддерживает различные типы моделей.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.vectorizer = None
        self.label_names = {
            0: "Разговорная речь",
            1: "Официально-деловая речь"
        }

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Загружает модель и векторизатор.

        Args:
            model_path: Путь к директории с моделью

        Returns:
            bool: Успех загрузки
        """
        try:
            model_file = os.path.join(model_path, "classifier.pkl")
            vectorizer_file = os.path.join(model_path, "vectorizer.pkl")

            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
                logger.info(f"Модель загружена из {model_file}")
            else:
                logger.error(f"Файл модели не найден: {model_file}")
                return False

            if os.path.exists(vectorizer_file):
                self.vectorizer = joblib.load(vectorizer_file)
                logger.info(f"Векторизатор загружен из {vectorizer_file}")
            else:
                logger.warning("Векторизатор не найден, используется по умолчанию")

            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Классифицирует текст.

        Args:
            text: Входной текст для классификации

        Returns:
            Dict с результатами классификации
        """
        if not self.model:
            return self._fallback_prediction(text)

        try:
            # Предобработка текста
            processed_text = self._preprocess_text(text)

            # Векторизация
            if self.vectorizer:
                features = self.vectorizer.transform([processed_text])
            else:
                # Простая фичеризация по длине и другим признакам
                features = self._basic_features(processed_text)

            # Предсказание
            prediction = self.model.predict(features)[0]
            probability = self._get_probability(features)

            label_name = self.label_names.get(prediction, "Неизвестный")

            return {
                "success": True,
                "label": int(prediction),
                "label_name": label_name,
                "confidence": probability,
                "text_length": len(text)
            }

        except Exception as e:
            logger.error(f"Ошибка классификации: {e}")
            return self._fallback_prediction(text)

    def _preprocess_text(self, text: str) -> str:
        """Базовая предобработка текста."""
        # Удаляем лишние пробелы и приводим к нижнему регистру
        return ' '.join(text.strip().lower().split())

    def _basic_features(self, text: str):
        """Базовая фичеризация для fallback."""
        # Здесь должна быть реализация фичеризации
        # Временно возвращаем пустые фичи
        from scipy.sparse import csr_matrix
        return csr_matrix((1, 1))

    def _get_probability(self, features) -> float:
        """Получает вероятность предсказания."""
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features)[0]
                return float(np.max(proba))
        except:
            pass
        return 0.8  # Значение по умолчанию

    def _fallback_prediction(self, text: str) -> Dict[str, Any]:
        """Fallback классификация когда модель не загружена."""
        # Простая эвристика на основе длины текста и ключевых слов
        official_keywords = ['уважаемый', 'заявление', 'документ', 'официальный', 'служебный']
        informal_keywords = ['привет', 'пока', 'спасибо', 'пожалуйста', 'дружище']

        text_lower = text.lower()
        official_count = sum(1 for word in official_keywords if word in text_lower)
        informal_count = sum(1 for word in informal_keywords if word in text_lower)

        if official_count > informal_count:
            label = 1
        elif informal_count > official_count:
            label = 0
        else:
            # Если ключевых слов нет, используем длину текста
            label = 1 if len(text) > 100 else 0

        return {
            "success": True,
            "label": label,
            "label_name": self.label_names.get(label, "Неизвестный"),
            "confidence": 0.6,
            "text_length": len(text),
            "fallback": True
        }


# Синглтон экземпляр
text_classifier = TextClassifier()