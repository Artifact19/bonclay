import asyncio
import io

import pytest
from fastapi import UploadFile

validation = pytest.importorskip("core.validation")
from core.errors import AppError  # noqa: E402


def upload(name, data):
    return UploadFile(filename=name, file=io.BytesIO(data))


def run(coro):
    return asyncio.run(coro)


def test_sanitize_strips_control_chars_and_nul():
    assert validation.sanitize_text("  he\x00llo\x07 world \n") == "hello world"


def test_sanitize_keeps_newlines_tabs_unicode():
    assert validation.sanitize_text("a\tb\nनमस्ते") == "a\tb\nनमस्ते"


def test_sanitize_empty_raises():
    with pytest.raises(AppError) as e:
        validation.sanitize_text("   \x00 ")
    assert e.value.code == "EMPTY_TEXT" and e.value.status == 400
    assert validation.sanitize_text("", required=False) == ""
    assert validation.sanitize_text(None, required=False) == ""


def test_sanitize_too_long():
    with pytest.raises(AppError) as e:
        validation.sanitize_text("a" * 11, max_len=10)
    assert e.value.code == "TEXT_TOO_LONG" and e.value.status == 422


def test_valid_wav_accepted(ref_wav):
    data = ref_wav.read_bytes()
    assert run(validation.validate_audio_upload(upload("ref.wav", data))) == data


def test_extension_rejected():
    with pytest.raises(AppError) as e:
        run(validation.validate_audio_upload(upload("evil.exe", b"MZ....")))
    assert e.value.code == "INVALID_FILE_TYPE" and e.value.status == 415


def test_content_mismatch_rejected(ref_wav):
    with pytest.raises(AppError) as e:
        run(validation.validate_audio_upload(upload("fake.wav", b"this is not audio at all")))
    assert e.value.code == "INVALID_FILE_CONTENT"
    with pytest.raises(AppError) as e:
        run(validation.validate_audio_upload(upload("ref.mp3", ref_wav.read_bytes())))
    assert e.value.code in ("FILE_TYPE_MISMATCH", "INVALID_FILE_CONTENT")


def test_empty_and_missing_file():
    with pytest.raises(AppError) as e:
        run(validation.validate_audio_upload(upload("a.wav", b"")))
    assert e.value.code == "EMPTY_FILE"
    with pytest.raises(AppError) as e:
        run(validation.validate_audio_upload(None))
    assert e.value.code == "FILE_REQUIRED"


def test_oversize_rejected():
    with pytest.raises(AppError) as e:
        run(validation.read_limited(upload("a.wav", b"x" * 2048), max_bytes=1024))
    assert e.value.code == "FILE_TOO_LARGE" and e.value.status == 413


def test_recording_formats_gated():
    webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 32
    with pytest.raises(AppError):
        run(validation.validate_audio_upload(upload("rec.webm", webm)))
    assert run(validation.validate_audio_upload(upload("rec.webm", webm), allow_recording=True)) == webm


def test_pdf_validation():
    pdf = b"%PDF-1.4\n..."
    assert run(validation.validate_pdf_upload(upload("a.pdf", pdf))) == pdf
    with pytest.raises(AppError):
        run(validation.validate_pdf_upload(upload("a.pdf", b"not pdf")))
    with pytest.raises(AppError):
        run(validation.validate_pdf_upload(upload("a.txt", pdf)))
