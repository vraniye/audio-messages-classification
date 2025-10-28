import os
import logging
import joblib
import numpy as np
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classifier")


class TextClassifier:
    """
    Класс для классификации стиля речи.
    Поддерживает различные типы моделей.
    """
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.label_names = {
            0: "Разговорная речь",
            1: "Официально-деловая речь"
        }

        # Конфигурация путей к моделям
        self.model_config = {
            "classic": {
                "model_path": "models/classic/classifier.pkl",
                "vectorizer_path": "models/classic/vectorizer.pkl"
            },
            "neural": {
                "model_path": "models/neural/model.pkl",
                "vectorizer_path": "models/neural/vectorizer.pkl"
            },
            "transformer": {
                "model_path": "models/transformer/model.pkl",
                "vectorizer_path": "models/transformer/vectorizer.pkl"
            }
        }

    def load_model(self, model_type: str = "classic") -> bool:
        if model_type in self.models:
            logger.info(f"Модель {model_type} уже загружена")
            return True

        try:
            config = self.model_config.get(model_type, self.model_config["classic"])

            model_file = config["model_path"]
            vectorizer_file = config["vectorizer_path"]

            # Загрузка модели
            if os.path.exists(model_file):
                self.models[model_type] = joblib.load(model_file)
                logger.info(f"Модель {model_type} загружена из {model_file}")
            else:
                logger.error(f"Файл модели не найден: {model_file}")
                return False

            if os.path.exists(vectorizer_file):
                self.vectorizers[model_type] = joblib.load(vectorizer_file)
                logger.info(f"Векторизатор {model_type} загружен из {vectorizer_file}")
            else:
                logger.warning(f"Векторизатор для {model_type} не найден, будет использован fallback")
                self.vectorizers[model_type] = None

            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_type}: {e}")
            return False

    def predict(self, text: str, model: str = "classic") -> Dict[str, Any]:
        if model not in self.models:
            success = self.load_model(model)
            if not success:
                logger.warning(f"Не удалось загрузить модель {model}, используется fallback")
                return self._fallback_prediction(text)

        try:
            processed_text = self._preprocess_text(text)

            vectorizer = self.vectorizers.get(model)
            if vectorizer:
                features = vectorizer.transform([processed_text])
            else:
                features = self._basic_features(processed_text)

            current_model = self.models[model]
            prediction = current_model.predict(features)[0]
            probability = self._get_probability(current_model, features)

            label_name = self.label_names.get(prediction, "Неизвестный")

            return {
                "success": True,
                "label": int(prediction),
                "label_name": label_name,
                "confidence": probability,
                "text_length": len(text),
                "model": model
            }

        except Exception as e:
            logger.error(f"Ошибка классификации моделью {model}: {e}")
            return self._fallback_prediction(text, model)

    def _preprocess_text(self, text: str) -> str:
        """Базовая предобработка текста."""
        return ' '.join(text.strip().lower().split())

    def _basic_features(self, text: str):
        """Базовая фичеризация для fallback."""
        from scipy.sparse import csr_matrix
        return csr_matrix((1, 1))

    def _get_probability(self, model, features) -> float:
        """Получает вероятность предсказания."""
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                return float(np.max(proba))
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(features)[0]
                return float(1 / (1 + np.exp(-abs(decision))))
        except Exception as e:
            logger.warning(f"Не удалось получить вероятность: {e}")

        return 0.8  # Значение по умолчанию

    def _fallback_prediction(self, text: str, model: str = "classic") -> Dict[str, Any]:
        """Fallback классификация когда модель не работает."""
        official_keywords = ['уважаемый', 'заявление', 'документ', 'официальный', 'служебный', 'прошу', 'предоставить']
        informal_keywords = ['привет', 'пока', 'спасибо', 'пожалуйста', 'дружище', 'норм', 'окей', 'класс']

        text_lower = text.lower()
        official_count = sum(1 for word in official_keywords if word in text_lower)
        informal_count = sum(1 for word in informal_keywords if word in text_lower)

        if official_count > informal_count:
            label = 1
            confidence = min(0.7 + official_count * 0.1, 0.95)
        elif informal_count > official_count:
            label = 0
            confidence = min(0.7 + informal_count * 0.1, 0.95)
        else:
            label = 1 if len(text) > 100 else 0
            confidence = 0.6

        return {
            "success": True,
            "label": label,
            "label_name": self.label_names.get(label, "Неизвестный"),
            "confidence": confidence,
            "text_length": len(text),
            "fallback": True,
            "model": model
        }

    def get_available_models(self) -> list:
        available = []
        for model_type, config in self.model_config.items():
            if os.path.exists(config["model_path"]):
                available.append(model_type)
        return available

    def preload_models(self, model_types: list = None):
        if model_types is None:
            model_types = ["classic", "neural", "transformer"]

        for model_type in model_types:
            try:
                self.load_model(model_type)
                logger.info(f"Модель {model_type} предзагружена")
            except Exception as e:
                logger.error(f"Ошибка предзагрузки модели {model_type}: {e}")


text_classifier = TextClassifier()
