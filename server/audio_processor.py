import os
import tempfile
import logging
import speech_recognition as sr
from pydub import AudioSegment
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio-processor")


class AudioToTextConverter:
    """
    Класс для преобразования аудиофайлов в текст.
    Поддерживает различные форматы аудио и языки.
    """

    def __init__(self, default_language: str = "ru-RU"):
        self.recognizer = sr.Recognizer()
        self.default_language = default_language
        self.supported_languages = {
            "ru": "ru-RU",
            "en": "en-US",
            "de": "de-DE",
            "fr": "fr-FR"
        }

    def convert_audio_to_text(self,
                              audio_path: str,
                              language: Optional[str] = None) -> Tuple[bool, str]:
        """
        Конвертирует аудиофайл в текст.

        Args:
            audio_path: Путь к аудиофайлу
            language: Язык распознавания (ru, en, etc.)

        Returns:
            Tuple[bool, str]: (успех, текст или сообщение об ошибке)
        """
        try:
            # Определяем язык
            lang_code = self.supported_languages.get(language, self.default_language)

            # Конвертируем аудио в WAV если нужно
            wav_path = self._convert_to_wav(audio_path)
            if not wav_path:
                return False, "Ошибка конвертации аудио в WAV"

            # Распознаем речь
            with sr.AudioFile(wav_path) as source:
                # Убираем шум и адаптируем к уровню звука
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)

                # Распознаем текст
                text = self.recognizer.recognize_google(
                    audio_data,
                    language=lang_code,
                    show_all=False
                )

            # Очищаем временные файлы
            if wav_path != audio_path:
                os.unlink(wav_path)

            logger.info(f"Успешно распознан текст: {text[:100]}...")
            return True, text.strip()

        except sr.UnknownValueError:
            logger.warning("Речь не распознана")
            return False, "Речь не распознана. Проверьте качество аудио."

        except sr.RequestError as e:
            logger.error(f"Ошибка сервиса распознавания: {e}")
            return False, f"Ошибка сервиса распознавания: {e}"

        except Exception as e:
            logger.error(f"Неожиданная ошибка при распознавании: {e}")
            return False, f"Ошибка обработки аудио: {str(e)}"

    def _convert_to_wav(self, audio_path: str) -> Optional[str]:
        """
        Конвертирует аудиофайл в WAV формат для распознавания.

        Args:
            audio_path: Путь к исходному аудиофайлу

        Returns:
            Optional[str]: Путь к WAV файлу или None при ошибке
        """
        try:
            # Если файл уже в WAV формате
            if audio_path.lower().endswith('.wav'):
                return audio_path

            # Определяем формат по расширению
            ext = os.path.splitext(audio_path)[1].lower().lstrip('.')

            # Загружаем аудио
            if ext in ['mp3', 'm4a', 'aac', 'flac', 'ogg', 'wma']:
                audio = AudioSegment.from_file(audio_path, format=ext)
            else:
                # Пытаемся автоматически определить формат
                audio = AudioSegment.from_file(audio_path)

            # Создаем временный WAV файл
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wav_path = tmp.name

            # Экспортируем в WAV
            audio.export(wav_path, format="wav")
            return wav_path

        except Exception as e:
            logger.error(f"Ошибка конвертации в WAV: {e}")
            return None

    def get_supported_formats(self) -> list:
        """Возвращает список поддерживаемых форматов аудио."""
        return ['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg', 'wma']

    def get_supported_languages(self) -> dict:
        """Возвращает словарь поддерживаемых языков."""
        return self.supported_languages


# Синглтон экземпляр для использования в приложении
audio_processor = AudioToTextConverter()