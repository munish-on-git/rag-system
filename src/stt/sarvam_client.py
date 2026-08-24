import mimetypes
import time
from dataclasses import dataclass
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = "saaras:v3"
SARVAM_TIMEOUT = 15.0
SARVAM_MAX_RETRIES = 3

_EXTENSION_TO_CONTENT_TYPE = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/x-m4a",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    ".amr": "audio/amr",
    ".aiff": "audio/aiff",
}

_FALLBACK_CONTENT_TYPE = "application/octet-stream"


def resolve_content_type(filename: str, content_type: Optional[str] = None) -> str:
    if content_type:
        base_type = content_type.split(";")[0].strip()
        if base_type and base_type != "application/octet-stream":
            return base_type

    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return guessed

    for ext, ctype in _EXTENSION_TO_CONTENT_TYPE.items():
        if filename.lower().endswith(ext):
            return ctype

    return _FALLBACK_CONTENT_TYPE


class SarvamSTTError(Exception):
    pass


@dataclass
class TranscriptionResponse:
    transcript: str
    language_code: Optional[str]
    mode: str
    model: str
    request_id: Optional[str]
    latency_ms: float


class SarvamSTTClient:
    def __init__(self, api_key: str):
        self._session = requests.Session()

        retry_strategy = Retry(
            total=SARVAM_MAX_RETRIES,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )

        self._session.mount(
            "https://",
            HTTPAdapter(max_retries=retry_strategy),
        )

        self._headers = {"api-subscription-key": api_key}

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        language_code: str = "unknown",
        mode: str = "transcribe",
        model: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> TranscriptionResponse:
        model = model or SARVAM_MODEL
        resolved_content_type = resolve_content_type(filename, content_type)

        files = {
            "file": (filename, audio_bytes, resolved_content_type)
        }
        data = {
            "model": model,
            "mode": mode,
            "language_code": language_code,
        }

        start = time.perf_counter()

        try:
            response = self._session.post(
                SARVAM_STT_URL,
                headers=self._headers,
                files=files,
                data=data,
                timeout=SARVAM_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise SarvamSTTError(f"Network error calling Sarvam STT: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            raise SarvamSTTError(
                f"Sarvam STT returned {response.status_code}: {response.text}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise SarvamSTTError(
                f"Sarvam STT returned non-JSON response: {response.text}"
            ) from exc

        transcript = (payload.get("transcript") or "").strip()

        if not transcript:
            raise SarvamSTTError("Sarvam STT returned an empty transcript.")

        return TranscriptionResponse(
            transcript=transcript,
            language_code=payload.get("language_code", language_code),
            mode=mode,
            model=model,
            request_id=payload.get("request_id"),
            latency_ms=round(latency_ms, 2),
        )