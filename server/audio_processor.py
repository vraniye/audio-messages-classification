import os
import tempfile
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import speech_recognition as sr
from pydub import AudioSegment

from salutespeech.salute_transcribe import (
    SaluteTranscribeError,
    transcribe_file,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio-processor")

FFMPEG_BIN = os.getenv("FFMPEG_BIN")
FFPROBE_BIN = os.getenv("FFPROBE_BIN")

if FFMPEG_BIN:
    AudioSegment.converter = FFMPEG_BIN
if FFPROBE_BIN:
    AudioSegment.ffprobe = FFPROBE_BIN


@dataclass
class AudioToTextConverter:
    """
    Конвертация аудио в текст с использованием Salute Speech и резервом Google Speech.
    """

    default_language: str = "ru-RU"
    supported_languages: Dict[str, str] = field(default_factory=lambda: {
        "ru": "ru-RU",
        "en": "en-US",
        "de": "de-DE",
        "fr": "fr-FR",
    })
    noise_adjustment_duration: float = 0.5

    def __post_init__(self) -> None:
        self.recognizer = sr.Recognizer()

    def convert_audio_to_text(self, audio_path: str, language: Optional[str] = None) -> Tuple[bool, str]:
        """
        Конвертирует аудиофайл в текст. Пытается Salute Speech, затем Google Speech.
        """
        lang_code = self.supported_languages.get(language or "", self.default_language)

        if self._salute_configured():
            logger.info("Пропускаем Salute Speech и используем Google Speech (отключено).")

        return self._transcribe_with_google(audio_path, lang_code)

    @staticmethod
    def _salute_configured() -> bool:
        return bool(os.getenv("SALUTE_AUTH_KEY"))

    def _transcribe_with_salute(self, audio_path: str, lang_code: str) -> str:
        logger.info("Распознавание через Salute Speech (%s)", lang_code)
        text = transcribe_file(audio_path, lang=lang_code, convert=True)
        logger.info("Salute Speech распознал текст: %s...", text[:100] if text else "")
        if not text:
            raise SaluteTranscribeError("Salute Speech вернул пустую транскрипцию.")
        return text.strip()

    def _transcribe_with_google(self, audio_path: str, lang_code: str) -> Tuple[bool, str]:
        wav_path: Optional[str] = None
        try:
            wav_path = self._convert_to_wav(audio_path)
            if not wav_path:
                return False, "Ошибка конвертации аудио в WAV"

            with sr.AudioFile(wav_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=self.noise_adjustment_duration)
                audio_data = self.recognizer.record(source)

            text = self.recognizer.recognize_google(audio_data, language=lang_code, show_all=False)
            logger.info("Google Speech распознал текст: %s...", text[:200] if text else "")
            return True, text.strip()

        except sr.UnknownValueError:
            logger.warning("Google Speech не распознал речь")
            return False, "Речь не распознана. Проверьте качество аудио."
        except sr.RequestError as exc:
            logger.error("Ошибка Google Speech: %s", exc)
            return False, f"Ошибка сервиса распознавания: {exc}"
        except Exception as exc:
            logger.error("Неожиданная ошибка при распознавании: %s", exc)
            return False, f"Ошибка обработки аудио: {exc}"
        finally:
            if wav_path and wav_path != audio_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    logger.warning("Не удалось удалить временный файл %s", wav_path)

    def _convert_to_wav(self, audio_path: str) -> Optional[str]:
        """
        Конвертирует аудиофайл в WAV формат для распознавания.
        """
        try:
            if audio_path.lower().endswith(".wav"):
                return audio_path

            extension = os.path.splitext(audio_path)[1].lower().lstrip(".")
            if extension in {"mp3", "m4a", "aac", "flac", "ogg", "wma"}:
                audio = AudioSegment.from_file(audio_path, format=extension)
            else:
                audio = AudioSegment.from_file(audio_path)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name

            audio.export(wav_path, format="wav")
            return wav_path

        except Exception as exc:
            logger.error("Ошибка конвертации в WAV: %s", exc)
            return None

    def get_supported_formats(self) -> list:
        return ["wav", "mp3", "m4a", "aac", "flac", "ogg", "wma"]

    def get_supported_languages(self) -> Dict[str, str]:
        return self.supported_languages


audio_processor = AudioToTextConverter()
