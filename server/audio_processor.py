import os
import tempfile
import logging
import speech_recognition as sr
from pydub import AudioSegment
from typing import Optional, Tuple

ffmpeg_path = "C:\\fmpeg\\fmpeg.exe"
ffprobe_path = "C:\\fmpeg\\fprobe.exe"

if os.path.exists(ffmpeg_path):
    AudioSegment.converter = ffmpeg_path
if os.path.exists(ffprobe_path):
    AudioSegment.ffprobe = ffprobe_path

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
            lang_code = self.supported_languages.get(language, self.default_language)

            wav_path = self._convert_to_wav(audio_path)
            if not wav_path:
                return False, "Ошибка конвертации аудио в WAV"

            # Распознаем речь
            with sr.AudioFile(wav_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)

                text = self.recognizer.recognize_google(
                    audio_data,
                    language=lang_code,
                    show_all=False
                )

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
            if audio_path.lower().endswith('.wav'):
                return audio_path

            ext = os.path.splitext(audio_path)[1].lower().lstrip('.')

            if ext in ['mp3', 'm4a', 'aac', 'flac', 'ogg', 'wma']:
                audio = AudioSegment.from_file(audio_path, format=ext)
            else:
                audio = AudioSegment.from_file(audio_path)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wav_path = tmp.name

            audio.export(wav_path, format="wav")
            return wav_path

        except Exception as e:
            logger.error(f"Ошибка конвертации в WAV: {e}")
            return None

    def get_supported_formats(self) -> list:
        return ['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg', 'wma']

    def get_supported_languages(self) -> dict:
        return self.supported_languages


audio_processor = AudioToTextConverter()
