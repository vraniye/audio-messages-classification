from __future__ import annotations

"""
Example integration:

    from salute_transcribe import transcribe_file

    # С вариантом конвертации (OGG/Opus → WAV PCM16 16 kHz mono)
    text = transcribe_file("/path/to/audio.ogg", lang="ru-RU")

    # Если аудио уже приведено к нужному формату, можно отключить конвертацию
    text = transcribe_file("/path/to/already_converted.wav", lang="ru-RU", convert=False)
    print(text)
"""

import base64
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

# Базовая настройка логгера; основную конфигурацию оставляем приложению.
logger = logging.getLogger("salute_transcribe")


class SaluteTranscribeError(Exception):
    """Base exception for Salute Speech transcription failures."""


class AuthError(SaluteTranscribeError):
    """Authentication or authorization error."""


class RateLimitError(SaluteTranscribeError):
    """API rate limit reached (HTTP 429)."""


class PayloadTooLarge(SaluteTranscribeError):
    """Submitted payload rejected as too large (HTTP 413)."""


class NetError(SaluteTranscribeError):
    """Non-recoverable networking error talking to Salute Speech."""


class ValidationError(SaluteTranscribeError):
    """Input validation failure."""


class ConversionError(SaluteTranscribeError):
    """Audio conversion failure."""


# Параметры доступа к Salute Speech.
DEFAULT_SCOPE = os.getenv("SALUTE_SCOPE", "SALUTE_SPEECH_PERS")
OAUTH_URL = "https://salute.online.sberbank.ru:9443/api/v2/oauth"
RECOGNIZE_URL = "https://salute.online.sberbank.ru:9443/api/v2/speech:recognize"

# Пути до утилит ffmpeg/ffprobe можно переопределить через окружение.
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

# Пороговые значения для диагностики входного файла.
WARN_DURATION_SECONDS = 60.0
WARN_FILE_SIZE_BYTES = 2 * 1024 * 1024

# Настройки HTTP-клиента и алгоритма ретраев.
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
HTTP_LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=10)

MAX_RETRIES = 5
BACKOFF_BASE = 0.6
BACKOFF_CAP = 8.0


@dataclass
class AudioMetadata:
    path: Path
    codec: str
    channels: int
    sample_rate: int
    duration: float
    size_bytes: int


@dataclass
class TokenInfo:
    value: str
    expires_at: float

    def is_valid(self) -> bool:
        return time.time() < self.expires_at - 10  # keep a small buffer


class SaluteSpeechClient:
    def __init__(self, scope: str = DEFAULT_SCOPE) -> None:
        # Настраиваем клиент: запоминаем scope и открываем HTTP/2-соединения.
        self._scope = scope
        self._token: Optional[TokenInfo] = None
        self._http = httpx.Client(
            base_url=None,
            http2=True,
            timeout=REQUEST_TIMEOUT,
            limits=HTTP_LIMITS,
            headers={"User-Agent": "salute-transcribe/1.0"},
        )

    def close(self) -> None:
        self._http.close()

    def _get_basic_auth(self) -> str:
        # Читаем ключ авторизации из окружения и убеждаемся, что он корректно закодирован.
        key = os.getenv("SALUTE_AUTH_KEY")
        if not key:
            raise AuthError("Environment variable SALUTE_AUTH_KEY is required.")
        # Validate base64 format (raise early if malformed)
        try:
            base64.b64decode(key, validate=True)
        except Exception as exc:
            raise AuthError("SALUTE_AUTH_KEY must be a valid Base64(ClientID:ClientSecret).") from exc
        return key

    def _fetch_token(self) -> TokenInfo:
        # Получаем новый access token по клиентским кредам.
        headers = {
            "Authorization": f"Basic {self._get_basic_auth()}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"scope": self._scope, "grant_type": "client_credentials"}
        response = self._request_with_retry(OAUTH_URL, data=data, headers=headers, expect_json=True)
        access_token = response.get("access_token")
        expires_in = response.get("expires_in")

        if not access_token:
            raise AuthError("Salute Speech OAuth response is missing access_token.")

        expires_at = time.time() + float(expires_in or 300)
        token = TokenInfo(value=access_token, expires_at=expires_at)
        self._token = token
        logger.debug("Obtained new access token (expires in %.0fs)", expires_at - time.time())
        return token

    def _get_token(self) -> str:
        if self._token and self._token.is_valid():
            return self._token.value
        return self._fetch_token().value

    def recognize(self, audio_bytes: bytes, lang: str) -> str:
        # Отправляем WAV и обрабатываем частый случай с истёкшим токеном.
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "audio/wav",
            "Accept": "application/json",
        }
        params = {"format": "text", "lang": lang}
        try:
            response = self._request_with_retry(
                RECOGNIZE_URL,
                params=params,
                content=audio_bytes,
                headers=headers,
                expect_json=True,
            )
        except AuthError:
            # Token could be expired/inv.
            self._token = None
            headers["Authorization"] = f"Bearer {self._get_token()}"
            response = self._request_with_retry(
                RECOGNIZE_URL,
                params=params,
                content=audio_bytes,
                headers=headers,
                expect_json=True,
            )

        return self._extract_transcription(response)

    def _request_with_retry(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        expect_json: bool = False,
    ) -> Any:
        # Универсальный POST с ретраями по сетевым и временным ошибкам.
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._http.post(
                    url,
                    params=params,
                    data=data,
                    content=content,
                    headers=headers,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "Network error talking to Salute Speech (attempt %s/%s): %s",
                    attempt,
                    MAX_RETRIES,
                    exc,
                )
                if attempt == MAX_RETRIES:
                    raise NetError("Unable to reach Salute Speech API.") from exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 401:
                raise AuthError("Salute Speech rejected credentials (401).")
            if response.status_code == 413:
                raise PayloadTooLarge("Audio payload exceeds Salute Speech limit (413).")
            if response.status_code == 429:
                logger.warning("Salute Speech rate limit hit (429), attempt %s/%s.", attempt, MAX_RETRIES)
                if attempt == MAX_RETRIES:
                    raise RateLimitError("Rate limit reached, retry later.")
                self._sleep_backoff(attempt, respect_retry_after=response.headers.get("Retry-After"))
                continue
            if response.status_code == 403:
                raise AuthError("Salute Speech rejected the request (403).")
            if 500 <= response.status_code < 600:
                logger.warning(
                    "Salute Speech server error %s, attempt %s/%s.",
                    response.status_code,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt == MAX_RETRIES:
                    raise NetError(f"Salute Speech server error {response.status_code}.")
                self._sleep_backoff(attempt)
                continue
            if 400 <= response.status_code < 500:
                raise NetError(
                    f"Salute Speech client error {response.status_code}: {response.text.strip() or 'no details'}"
                )

            if expect_json:
                try:
                    return response.json()
                except json.JSONDecodeError as exc:
                    raise NetError("Unexpected non-JSON response from Salute Speech.") from exc
            return response

        raise NetError("Exhausted retries talking to Salute Speech.") from last_exc

    @staticmethod
    def _sleep_backoff(attempt: int, respect_retry_after: Optional[str] = None) -> None:
        # Экспоненциальный бэкофф с поддержкой Retry-After.
        delay = min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_CAP)
        if respect_retry_after:
            try:
                delay = max(delay, float(respect_retry_after))
            except ValueError:
                pass
        time.sleep(delay)

    @staticmethod
    def _extract_transcription(payload: Dict[str, Any]) -> str:
        # Ответы у Salute Speech бывают разной формы, пытаемся вытащить первый текст.
        if not isinstance(payload, dict):
            raise NetError("Unexpected response payload type.")

        result = payload.get("result") or payload.get("results")
        if isinstance(result, dict):
            texts = result.get("texts") or result.get("alternatives") or result.get("hypotheses")
            if isinstance(texts, list) and texts:
                first = texts[0]
                if isinstance(first, dict):
                    text = first.get("text") or first.get("utterance")
                    if text:
                        return str(text).strip()
                elif isinstance(first, str):
                    return first.strip()
            transcript = result.get("text") or result.get("utterance")
            if transcript:
                return str(transcript).strip()
        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, dict):
                text = item.get("text") or item.get("utterance")
                if text:
                    return str(text).strip()
            elif isinstance(item, str):
                return item.strip()

        text = payload.get("text") or payload.get("transcription")
        if text:
            return str(text).strip()

        raise NetError("Unable to extract transcription from Salute Speech response.")


def _run_ffprobe(path: Path) -> AudioMetadata:
    # Считываем метаданные аудио и предупреждаем о потенциальных проблемах.
    if not path.exists():
        raise ValidationError(f"Audio file not found: {path}")
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ValidationError(f"ffprobe failed ({proc.returncode}): {proc.stderr.strip()}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValidationError("Unable to parse ffprobe JSON output.") from exc

    streams = data.get("streams") or []
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if not audio_stream:
        raise ValidationError("ffprobe did not find an audio stream.")

    format_info = data.get("format") or {}
    size_bytes = int(format_info.get("size") or path.stat().st_size)
    duration = float(format_info.get("duration") or audio_stream.get("duration") or 0.0)
    codec = str(audio_stream.get("codec_name") or format_info.get("format_name") or "").lower()
    channels = int(audio_stream.get("channels") or 0)
    sample_rate = int(float(audio_stream.get("sample_rate") or 0))

    if size_bytes > WARN_FILE_SIZE_BYTES:
        logger.warning("Audio file %s is larger than 2 MB (%.2f MB).", path, size_bytes / (1024 * 1024))
    if duration > WARN_DURATION_SECONDS:
        logger.warning("Audio file %s is longer than 60 seconds (%.1f s).", path, duration)
    if codec and codec not in {"opus", "pcm_s16le", "wav"}:
        logger.info("Audio codec detected: %s", codec)

    return AudioMetadata(
        path=path,
        codec=codec,
        channels=channels,
        sample_rate=sample_rate,
        duration=duration,
        size_bytes=size_bytes,
    )


def _convert_to_wav(meta: AudioMetadata) -> Path:
    # Принудительно конвертируем входной файл в WAV PCM16 mono 16 kHz.
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file_path = Path(temp_file.name)
    temp_file.close()

    downmix_required = meta.channels != 1
    if downmix_required and meta.channels and meta.channels > 1:
        logger.debug("Downmixing %s from %s channels to mono for Salute Speech.", meta.path, meta.channels)

    channels_arg = "1"  # Salute Speech ожидает mono; всегда приводим к одному каналу.
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(meta.path),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        channels_arg,
        "-vn",
        str(temp_file_path),
    ]

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        temp_file_path.unlink(missing_ok=True)
        raise ConversionError(f"ffmpeg failed ({proc.returncode}): {stderr}")

    if not temp_file_path.exists() or temp_file_path.stat().st_size == 0:
        temp_file_path.unlink(missing_ok=True)
        raise ConversionError("ffmpeg produced an empty output file.")

    return temp_file_path


def _validate_target_format(meta: AudioMetadata) -> None:
    # Убеждаемся, что файл уже соответствует формату WAV PCM16 mono 16 kHz.
    if meta.codec != "pcm_s16le":
        raise ValidationError(
            "convert=False допустимо только для WAV PCM16 (pcm_s16le)."
        )
    if meta.sample_rate != 16000:
        raise ValidationError("convert=False требует частоту дискретизации 16 кГц.")
    if meta.channels != 1:
        raise ValidationError("convert=False требует моно-канал (1).")


_default_client: Optional[SaluteSpeechClient] = None


def _get_client() -> SaluteSpeechClient:
    # Ленивое создание Singleton-клиента для повторного использования соединений.
    global _default_client
    if _default_client is None:
        _default_client = SaluteSpeechClient()
    return _default_client


def transcribe_file(path: str, *, lang: str = "ru-RU", convert: bool = True) -> str:
    """
    Convert an audio file to text using Salute Speech API.

    Args:
        path: Path to the audio file (.ogg with Opus is supported).
        lang: Recognition language code, defaults to "ru-RU".
        convert: When True (default) run ffmpeg conversion to target format.

    Returns:
        Recognized transcription as plain text.
    """
    audio_path = Path(path).expanduser().resolve()
    # Собираем информацию о файле перед отправкой.
    meta = _run_ffprobe(audio_path)

    wav_path: Optional[Path] = None
    try:
        if convert:
            wav_path = _convert_to_wav(meta)
            audio_path_for_upload = wav_path
        else:
            _validate_target_format(meta)
            audio_path_for_upload = meta.path
            logger.debug("Используем уже подготовленный WAV без повторной конвертации: %s", audio_path_for_upload)

        with audio_path_for_upload.open("rb") as fh:
            audio_bytes = fh.read()

        client = _get_client()
        text = client.recognize(audio_bytes, lang=lang)
        logger.info("Transcription complete for %s", audio_path)
        return text
    finally:
        if convert and wav_path and wav_path.exists():
            try:
                wav_path.unlink()
            except OSError:
                logger.warning("Failed to remove temporary WAV file %s", wav_path)


def _shutdown() -> None:
    # Гарантируем закрытие HTTP-клиента при завершении работы скрипта.
    global _default_client
    if _default_client is not None:
        _default_client.close()
        _default_client = None


def _main() -> None:
    file_path = os.getenv("FILE_PATH")
    if not file_path:
        raise SystemExit("Set FILE_PATH environment variable pointing to audio file.")
    try:
        # Для быстрой проверки: транскрибируем и выводим текст.
        text = transcribe_file(file_path)
        print(text)
    finally:
        _shutdown()


if __name__ == "__main__":
    _main()

