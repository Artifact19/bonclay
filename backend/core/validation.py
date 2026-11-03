"""Input sanitising and upload validation."""
from __future__ import annotations

from typing import Optional

from fastapi import UploadFile

from .errors import AppError

MAX_TEXT_CHARS = 2000
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac")
# Browser microphone recordings; only enabled on routes that take live recordings.
RECORDING_EXTENSIONS = (".webm", ".ogg")


def sanitize_text(value: Optional[str], field: str = "text", max_len: int = MAX_TEXT_CHARS,
                  required: bool = True) -> str:
    """Strip NUL bytes/control chars and enforce length. Raises AppError on bad input."""
    if value is None:
        value = ""
    cleaned = value.replace("\x00", "")
    cleaned = "".join(ch for ch in cleaned if ch in "\n\r\t" or ord(ch) >= 32)
    cleaned = cleaned.strip()
    if required and not cleaned:
        raise AppError("EMPTY_TEXT", f"{field} cannot be empty", 400, {"field": field})
    if len(cleaned) > max_len:
        raise AppError("TEXT_TOO_LONG", f"{field} exceeds the maximum of {max_len} characters",
                       422, {"field": field, "max_length": max_len, "length": len(cleaned)})
    return cleaned


def sniff_audio_kind(data: bytes) -> Optional[str]:
    """Identify audio container from magic bytes: wav/mp3/m4a/flac/webm/ogg or None."""
    h = data[:16]
    if h[:4] == b"RIFF" and h[8:12] == b"WAVE":
        return "wav"
    if h[:4] == b"fLaC":
        return "flac"
    if h[:3] == b"ID3" or (len(h) >= 2 and h[0] == 0xFF and (h[1] & 0xE0) == 0xE0):
        return "mp3"
    if h[4:8] == b"ftyp":
        return "m4a"
    if h[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if h[:4] == b"OggS":
        return "ogg"
    return None


_EXT_TO_KINDS = {
    ".wav": {"wav"}, ".mp3": {"mp3"}, ".m4a": {"m4a"}, ".flac": {"flac"},
    ".webm": {"webm"}, ".ogg": {"ogg"},
}


async def read_limited(upload: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise AppError("FILE_TOO_LARGE", f"File exceeds the {max_bytes // (1024 * 1024)}MB limit",
                           413, {"max_bytes": max_bytes})
        chunks.append(chunk)
    return b"".join(chunks)


async def validate_audio_upload(upload: Optional[UploadFile], allow_recording: bool = False,
                                field: str = "audio file") -> bytes:
    """Validate extension, size and magic bytes; return the file bytes."""
    if upload is None:
        raise AppError("FILE_REQUIRED", f"{field} is required", 400)
    name = (upload.filename or "").lower()
    allowed = AUDIO_EXTENSIONS + (RECORDING_EXTENSIONS if allow_recording else ())
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in allowed:
        raise AppError("INVALID_FILE_TYPE",
                       f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(allowed)}",
                       415, {"allowed": list(allowed)})
    data = await read_limited(upload)
    if not data:
        raise AppError("EMPTY_FILE", "Uploaded file is empty", 400)
    kind = sniff_audio_kind(data)
    permitted = {k for e in allowed for k in _EXT_TO_KINDS[e]}
    if kind is None or kind not in permitted:
        raise AppError("INVALID_FILE_CONTENT",
                       "File content does not look like a supported audio file", 415,
                       {"detected": kind})
    if kind not in _EXT_TO_KINDS[ext]:
        raise AppError("FILE_TYPE_MISMATCH",
                       f"File extension '{ext}' does not match its content ({kind})", 415,
                       {"detected": kind})
    return data


async def validate_pdf_upload(upload: Optional[UploadFile], max_bytes: int = 20 * 1024 * 1024) -> bytes:
    if upload is None or not (upload.filename or "").lower().endswith(".pdf"):
        raise AppError("INVALID_FILE_TYPE", "Only PDF files are supported.", 415, {"allowed": [".pdf"]})
    data = await read_limited(upload, max_bytes)
    if data.lstrip()[:5] != b"%PDF-":
        raise AppError("INVALID_FILE_CONTENT", "File content is not a PDF", 415)
    return data
