import os
import tempfile
import logging
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .audio_processor import audio_processor
from .classifier import text_classifier

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# Конфигурация
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(current_dir, "models"))
MIN_WORDS_REQUIRED = int(os.getenv("MIN_WORDS", "5"))  # Значение по умолчанию 5
MIN_DURATION = int(os.getenv("MIN_DURATION", "3"))  # Значение по умолчанию 3

app = FastAPI(
    title="Speech Style Classifier API",
    description="API для классификации стиля речи из аудиосообщений"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Модели ответов
class ModelDetail(BaseModel):
    model: str
    label: int
    label_name: str
    confidence: float
    success: bool
    fallback: Optional[bool] = None


class ClassificationResponse(BaseModel):
    success: bool
    label: int
    label_name: str
    confidence: float
    text: str
    text_length: int
    model: str
    word_count: Optional[int] = None
    best_model: Optional[str] = None
    summary: Optional[List[str]] = None
    details: Optional[List[ModelDetail]] = None
    error: Optional[str] = None
    duration: Optional[float] = None
    fallback: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Загружаем модель при запуске сервера."""
    available_for_preload = [name for name in text_classifier.get_available_models() if name != "all"]
    if available_for_preload:
        text_classifier.preload_models(available_for_preload)
        logger.info("Предзагружены модели: %s", ", ".join(available_for_preload))
    else:
        logger.warning(
            "Не найдено доступных моделей для предзагрузки. Проверьте артефакты в %s",
            MODEL_PATH,
        )


@app.get("/")
async def root():
    """Корневой endpoint."""
    return {
        "message": "Speech Style Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "classify": "/classify",
            "formats": "/supported_formats"
        }
    }


@app.get("/health")
async def health_check() -> HealthResponse:
    """Проверка здоровья сервера."""
    return HealthResponse(
        status="ok",
        model_loaded=bool(text_classifier.models)
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify_audio(
    file: UploadFile = File(...),
    model: str = Form("default")  # Добавляем параметр модели из формы
):
    """
    Классифицирует стиль речи из аудиофайла с использованием выбранной модели.

    Поддерживаемые форматы: WAV, MP3, M4A, AAC, FLAC, OGG

    Args:
        file: Аудиофайл для обработки
        model: Идентификатор модели для классификации

    Returns:
        ClassificationResponse: Результат классификации
    """
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an audio file. Received content type: {}".format(file.content_type)
        )

    # Нормализуем идентификатор модели
    model = (model or "logistic").lower()
    if model == "default":
        model = "logistic"

    # Валидация модели
    valid_models = set(text_classifier.model_config.keys()) | {"all"}
    if model not in valid_models:
        available = text_classifier.get_available_models()
        available_display = ", ".join(available or sorted(valid_models))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model}. Available models: {available_display}"
        )
    try:
        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(file.filename)[1] or ".ogg"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        logger.info(f"Обработка файла: {file.filename}, размер: {len(content)} байт, модель: {model}")

        success, text = audio_processor.convert_audio_to_text(temp_path)

        os.unlink(temp_path)

        if not success:
            logger.warning(f"Ошибка распознавания речи: {text}")
            return ClassificationResponse(
                success=False,
                label=-1,
                label_name="Ошибка",
                confidence=0.0,
                text="",
                text_length=0,
                error=text,
                model=model  # Добавляем информацию о модели в ответ
            )

        words = text.strip().split()
        word_count = len(words)

        if word_count < MIN_WORDS_REQUIRED:
            error_message = f"Слишком мало слов распознано: {word_count}. Минимум требуется: {MIN_WORDS_REQUIRED} слов"
            logger.warning(error_message)
            return ClassificationResponse(
                success=False,
                label=-1,
                label_name="Ошибка",
                confidence=0.0,
                text=text,
                text_length=len(text),
                word_count=word_count,
                error=error_message,
                model=model
            )

        classification_result = text_classifier.predict(text, model=model)

        detail_items = classification_result.get("details") or []
        details: Optional[List[ModelDetail]] = None
        if detail_items:
            details = [
                ModelDetail(
                    model=item.get("model", ""),
                    label=int(item.get("label", 0)),
                    label_name=item.get("label_name", "Неизвестный"),
                    confidence=float(item.get("confidence", 0.0)),
                    success=bool(item.get("success", False)),
                    fallback=item.get("fallback"),
                )
                for item in detail_items
            ]

        model_used = classification_result.get("model", model)
        logger.info(
            "Успешная классификация: %s (уверенность: %.2f), слов: %s, модель: %s",
            classification_result.get("label_name"),
            classification_result.get("confidence", 0.0),
            word_count,
            model_used,
        )

        return ClassificationResponse(
            success=classification_result.get("success", True),
            label=int(classification_result.get("label", 0)),
            label_name=classification_result.get("label_name", "Неизвестный"),
            confidence=float(classification_result.get("confidence", 0.0)),
            text=text,
            text_length=int(classification_result.get("text_length", len(text))),
            word_count=word_count,
            model=model_used,
            best_model=classification_result.get("best_model"),
            summary=classification_result.get("summary"),
            details=details,
            duration=classification_result.get("duration"),
            fallback=classification_result.get("fallback"),
        )

    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/supported_formats")
async def get_supported_formats():
    """Возвращает список поддерживаемых форматов аудио."""
    return {
        "audio_formats": audio_processor.get_supported_formats(),
        "languages": audio_processor.get_supported_languages(),
        "status": "ok"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
