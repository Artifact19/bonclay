# Extra pip deps: langdetect, indic_transliteration, librosa (for resample_to_24k), numpy
"""Indic language utilities: detection, transliteration, sample phrases."""
import asyncio

from fastapi import APIRouter
from pydantic import BaseModel, Field

try:
    from services import hinglish as hinglish_svc
    from services import indic as indic_svc
    from services import indic_speak as speak_svc
    from services.common import api_error
except ImportError:
    from backend.services import hinglish as hinglish_svc
    from backend.services import indic as indic_svc
    from backend.services import indic_speak as speak_svc
    from backend.services.common import api_error

router = APIRouter(prefix="/api/indic", tags=["indic"])


class DetectRequest(BaseModel):
    text: str = Field(..., max_length=20000)


class TransliterateRequest(BaseModel):
    text: str = Field(..., max_length=20000)
    target: str = "devanagari"


@router.post("/detect")
async def detect(req: DetectRequest):
    if not req.text.strip():
        raise api_error(400, "Text is empty", "EMPTY_TEXT", None)
    res = indic_svc.detect_language(req.text)
    res["is_indic"] = res["language"] in indic_svc.INDIC_LANGS
    res["hinglish"] = hinglish_svc.looks_hinglish(req.text)  # Hindi typed in Roman letters
    return res


@router.post("/transliterate")
async def transliterate(req: TransliterateRequest):
    if req.target.lower() != "devanagari":
        raise api_error(400, "Only 'devanagari' target is supported", "UNSUPPORTED_TARGET", {"target": req.target})
    if not req.text.strip():
        raise api_error(400, "Text is empty", "EMPTY_TEXT", None)
    if hinglish_svc.looks_hinglish(req.text):
        # Casual Roman Hindi: a language model reads it in context, with the dictionary as the offline fallback.
        res = await asyncio.to_thread(hinglish_svc.to_devanagari, req.text)
        return {"text": res["text"], "scheme": "hinglish", "engine": res["provider"] or "dictionary",
                "method": res["method"], "provider": res["provider"]}
    res = indic_svc.transliterate_to_devanagari(req.text)
    return {**res, "method": "rules"}  # formal ITRANS / Harvard-Kyoto input keeps the scheme-based converter


@router.get("/samples")
async def samples():
    return {"samples": [{"language": k, "text": v} for k, v in indic_svc.SAMPLES.items()]}


@router.get("/voices")
async def voices():
    """Indic-Speak preset voices (22 languages, one female and one male each) and whether the engine can run here.

    `attribution` is the notice the Indic Open Model License requires wherever its output reaches a user interface."""
    reason = speak_svc.indic_speak_unavailable_reason()
    return {
        "available": reason is None,
        "reason": reason,
        "voices": speak_svc.VOICES,
        "max_chars": speak_svc.MAX_TEXT_CHARS,
        "attribution": speak_svc.ATTRIBUTION,
        "model_url": speak_svc.MODEL_URL,
    }
