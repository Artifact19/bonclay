# Extra pip deps: kokoro>=0.9 (KPipeline), soundfile, numpy; system: espeak-ng (needed for non-English voices)
"""Kokoro-82M preset voices and generation."""
import asyncio
import importlib.util
import io
import sys
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field

try:
    from services.common import api_error
    from services.auth import get_current_user_id
    from services import provenance
    from services import mlx_kokoro
    from services import pron_lexicon
except ImportError:
    from backend.services.common import api_error
    from backend.services.auth import get_current_user_id
    from backend.services import provenance
    from backend.services import mlx_kokoro
    from backend.services import pron_lexicon

router = APIRouter(prefix="/api/kokoro", tags=["kokoro"])

# prefix letter -> (kokoro lang_code, language name, accent)
_LANG = {
    "a": ("a", "English", "American"), "b": ("b", "English", "British"),
    "j": ("j", "Japanese", "Japanese"), "z": ("z", "Mandarin Chinese", "Chinese"),
    "e": ("e", "Spanish", "Spanish"), "f": ("f", "French", "French"),
    "h": ("h", "Hindi", "Indian"), "i": ("i", "Italian", "Italian"),
    "p": ("p", "Brazilian Portuguese", "Brazilian"),
}
_IDS = ["af_heart", "af_bella", "af_sky", "af_nicole", "am_adam", "am_michael", "bf_emma", "bf_isabella",
        "bm_george", "bm_lewis", "jf_alpha", "zf_xiaobei", "ef_dora", "ff_siwis", "hf_alpha", "if_sara", "pf_dora"]


def _voice(vid: str) -> dict:
    code, language, accent = _LANG[vid[0]]
    return {
        "id": vid, "name": vid.split("_", 1)[1].replace("_", " ").title(), "language": language,
        "language_code": code, "gender": "female" if vid[1] == "f" else "male", "accent": accent,
        "sample_url": f"https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/samples/{vid}.wav",
    }


VOICES = [_voice(v) for v in _IDS]
_VOICE_MAP = {v["id"]: v for v in VOICES}
_pipelines = {}

# Text front-ends that need optional packages: lang_code -> (ISO language, modules that must all import,
# modules of which at least one must import, how to install them). A voice whose packages are missing is listed
# as unavailable, and generating with it gives a clear 503, instead of an opaque 500 ("No module named ...").
_REQUIRES = {
    "j": ("ja", ("pyopenjtalk", "fugashi", "jaconv", "mojimoji"), ("unidic_lite", "unidic"),
          "pip install -r requirements-ja.txt"),
    "z": ("zh", ("jieba", "pypinyin", "cn2an", "ordered_set"), (), 'pip install "misaki[zh]"'),
}


def missing_packages(lang_code: str) -> list:
    """Optional packages this voice's language needs but this server does not have (empty when it can speak)."""
    need = _REQUIRES.get(lang_code)
    if need is None:
        return []
    _, required, one_of, _ = need
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if one_of and not any(importlib.util.find_spec(m) is not None for m in one_of):
        missing.append("/".join(one_of))
    return missing


def unavailable_languages() -> set:
    """ISO codes of Kokoro languages that cannot be spoken here (for the engine list and the UI)."""
    return {need[0] for code, need in _REQUIRES.items() if missing_packages(code)}


class KokoroRequest(BaseModel):
    voice: str = "af_heart"
    text: str = Field(..., min_length=1, max_length=5000)
    speed: float = Field(1.0, ge=0.5, le=2.0)


@router.get("/voices")
async def voices():
    """Preset voices. `available` is false for a voice whose language needs packages this server lacks."""
    out = []
    for v in VOICES:
        missing = missing_packages(v["language_code"])
        out.append({**v, "available": not missing,
                    "unavailable_reason": f"{v['language']} needs: {_REQUIRES[v['language_code']][3]}" if missing else None})
    return {"voices": out}


def _synthesize(voice: dict, text: str, speed: float) -> bytes:
    try:
        from services.espeak import configure_espeak
    except ImportError:
        from backend.services.espeak import configure_espeak
    configure_espeak()
    from kokoro import KPipeline  # lazy
    import numpy as np
    import soundfile as sf
    code = voice["language_code"]
    if code not in _pipelines:
        _pipelines[code] = KPipeline(lang_code=code)
    parts = []
    for _, _, audio in _pipelines[code](text, voice=voice["id"], speed=speed):
        if audio is not None:
            parts.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
    if not parts:
        raise RuntimeError("Kokoro produced no audio")
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(parts), 24000, format="WAV")
    return buf.getvalue()


def _backend():
    """The backend module (imported lazily to avoid an import cycle; also works when backend.py runs as __main__)."""
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "advanced_manager"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "advanced_manager"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


async def _synthesize_mlx(voice: dict, text: str, speed: float):
    """WAV bytes from the MLX Kokoro worker, or None when PyTorch must speak (MLX off, worker down, voice unsupported)."""
    if not mlx_kokoro.wants_mlx():
        return None
    try:
        handle = await _backend().advanced_manager.load_model("kokoro")
        if not mlx_kokoro.is_handle(handle):
            if getattr(handle, "lang_code", None) == "a":  # the worker could not start: reuse the PyTorch pipeline the
                _pipelines.setdefault("a", handle)         # manager just built instead of loading a second copy
            return None
        audio = await asyncio.to_thread(handle.generate, text, voice["id"], voice["language_code"], speed)
    except Exception as e:  # noqa: BLE001  never fail the request because the fast engine is unavailable
        mlx_kokoro.log_once("router", "MLX Kokoro failed in /api/kokoro/generate (%s); using PyTorch Kokoro", e)
        return None
    if audio is None:
        return None
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    return buf.getvalue()


@router.post("/generate")
async def generate(req: KokoroRequest, user_id: str = Depends(get_current_user_id)):
    voice = _VOICE_MAP.get(req.voice)
    if voice is None:
        raise api_error(400, "Unknown Kokoro voice", "UNKNOWN_VOICE", {"voice": req.voice, "available": _IDS})
    missing = missing_packages(voice["language_code"])
    if missing:
        raise api_error(503, f"{voice['language']} voices need packages that are not installed on this server",
                        "ENGINE_UNAVAILABLE", {"missing": missing, "hint": _REQUIRES[voice["language_code"]][3]})
    # Spelling hints for the Hindi voice (services/pron_lexicon.py, off unless PRON_LEXICON=on). Only the text handed to
    # the engine changes; the receipt below still records what the caller wrote.
    spoken = pron_lexicon.apply(req.text, "hi" if voice["language_code"] == "h" else None, "kokoro")
    wav = await _synthesize_mlx(voice, spoken, req.speed)  # Apple Silicon: the MLX worker; None means PyTorch speaks
    if wav is None:
        try:
            import kokoro  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError as e:
            raise api_error(503, "Kokoro engine is not installed", "ENGINE_UNAVAILABLE",
                            {"hint": "pip install kokoro soundfile  (and install espeak-ng for non-English voices)",
                             "missing": str(e)})
        try:
            wav = await asyncio.to_thread(_synthesize, voice, spoken, req.speed)
        except Exception as e:
            raise api_error(500, "Kokoro generation failed", "GENERATION_FAILED", str(e))
    # Every clip BonClay makes carries an inaudible watermark and a signed receipt (no consent: a preset voice).
    wav, receipt = await asyncio.to_thread(
        lambda: provenance.apply_provenance(
            kind="generation", subject_id=str(uuid.uuid4()), wav_bytes=wav, text=req.text, engine="kokoro",
            voice_id=None, voice_name=req.voice, consent_id=None, consent_verified=None, user_id=user_id))
    headers = {"Content-Disposition": f'inline; filename="{req.voice}.wav"'}
    if receipt.get("receipt_id"):
        headers["X-Receipt-Id"] = receipt["receipt_id"]
    return Response(content=wav, media_type="audio/wav", headers=headers)
