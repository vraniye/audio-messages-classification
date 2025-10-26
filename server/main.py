import os
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .audio_processor import audio_processor
from .classifier import text_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# Конфигурация
MODEL_PATH = os.getenv("MODEL_PATH", "./models")

app = FastAPI(
    title="Speech Style Classifier API",
    description="API для классификации стиля речи из аудиосообщений"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшне укажите конкретные домены
    allow_methods=["*"],
    allow_headers=["*"],
)


# Модели ответов
class ClassificationResponse(BaseModel):
    success: bool
    label: int
    label_name: str
    confidence: float
    text: str
    text_length: int
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Загрузка модели при старте
@app.on_event("startup")
async def startup_event():
    """Загружаем модель при запуске сервера."""
    if os.path.exists(MODEL_PATH):
        text_classifier.load_model(MODEL_PATH)
        logger.info("Модель успешно загружена")
    else:
        logger.warning(f"Директория с моделью не найдена: {MODEL_PATH}")


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
        model_loaded=text_classifier.model is not None
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify_audio(file: UploadFile = File(...)):
    """
    Классифицирует стиль речи из аудиофайла.

    Поддерживаемые форматы: WAV, MP3, M4A, AAC, FLAC, OGG

    Returns:
        ClassificationResponse: Результат классификации
    """
    # Проверка типа файла
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an audio file. Received content type: {}".format(file.content_type)
        )

    try:
        # Сохраняем временный файл
        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(file.filename)[1] or ".ogg"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        logger.info(f"Обработка файла: {file.filename}, размер: {len(content)} байт")

        # Конвертируем аудио в текст
        success, text = audio_processor.convert_audio_to_text(temp_path)

        # Очищаем временный файл
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
                error=text
            )

        # Классифицируем текст
        classification_result = text_classifier.predict(text)

        logger.info(f"Успешная классификация: {classification_result['label_name']} "
                    f"(уверенность: {classification_result['confidence']:.2f})")

        return ClassificationResponse(
            success=True,
            label=classification_result["label"],
            label_name=classification_result["label_name"],
            confidence=classification_result["confidence"],
            text=text,
            text_length=classification_result["text_length"]
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