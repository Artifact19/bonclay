"""Indic-Speak (Bodhan AI / AI4Bharat): preset voices for 22 Indian languages, behind a small client.

Indic-Speak is a 3.8B-parameter model that needs transformers 5.x (this backend pins 4.x for qwen-tts), so it runs
in its own virtualenv. This module starts services/indic_speak_worker.py in that environment and talks to it over
127.0.0.1, the same way services/mlx_clone.py does for Apple-GPU voice cloning.

It has preset voices only, no cloning. Enabled when a worker python exists: $INDIC_SPEAK_PYTHON, else
backend/.venv-indic/bin/python (created by scripts/setup_indic_speak.sh), and the weights are downloaded.
Set INDIC_SPEAK=off to opt out.

Indic Open Model License v1.0: wherever this engine's output reaches a user interface the notice in ATTRIBUTION
must be shown, and offering the model to third parties through an API needs Bodhan AI's approval unless the
use is academic or non-profit (see the model card and README).
"""
import io
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

log = logging.getLogger("bonclay.indic_speak")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "bodhan-ai/indic-speak"
SAMPLE_RATE = 24000
ATTRIBUTION = "Built with Indic-Speak from Bodhan AI / AI4Bharat."
MODEL_URL = "https://huggingface.co/bodhan-ai/indic-speak"

MAX_TEXT_CHARS = int(os.getenv("INDIC_SPEAK_MAX_CHARS", "600"))  # per request; ~12 characters of speech per second
CHUNK_CHARS = 160  # longest single generation; longer sentences are split at a space
MIN_SENTENCE_CHARS = 12  # a shorter fragment ("हाँ।") is merged into the sentence before it
PAUSE_SECONDS = 0.25  # silence between stitched chunks
# The model emits ~82 audio tokens per second of speech. Real speech runs at 8-15 characters/second, so a clip
# that needs more than text / MIN_CHARS_PER_SECOND seconds has derailed (it never emitted its end-of-speech token).
TOKENS_PER_SECOND = 82
MIN_CHARS_PER_SECOND = 6.0
MAX_SPLIT_DEPTH = 2

# Must match services/indic_speak_worker.py: control-token look-alikes (<|speaker>, <speaker|>, <|eot_id|> ...).
CONTROL_TOKEN_RE = re.compile(r"<\|[^<>|]{0,48}\|?>|<[^<>|]{1,48}\|>")

# language code -> (language name, female voice, male voice), from the model's voices.md. Any voice can speak any
# language (conditioning is cross-lingual), but each has a native language where diction is strongest.
_CATALOG: Dict[str, Tuple[str, str, str]] = {
    "as": ("Assamese", "Prastuti", "Ankur"), "bn": ("Bengali", "Ishita", "Sourav"),
    "brx": ("Bodo", "Gwrbw", "Sansuma"), "doi": ("Dogri", "Preeti", "Sham"),
    "gu": ("Gujarati", "Dhara", "Parth"), "hi": ("Hindi", "Kavya", "Amit"),
    "kn": ("Kannada", "Deepika", "Adarsh"), "ks": ("Kashmiri", "Zoon", "Ishfaq"),
    "kok": ("Konkani", "Anjali", "Sandeep"), "mai": ("Maithili", "Vaidehi", "Madhukar"),
    "ml": ("Malayalam", "Lakshmi", "Kiran"), "mni": ("Manipuri", "Thoibi", "Chaoba"),
    "mr": ("Marathi", "Anagha", "Chinmay"), "ne": ("Nepali", "Srijana", "Sagar"),
    "or": ("Odia", "Itishree", "Akash"), "pa": ("Punjabi", "Kaur", "Manpreet"),
    "sa": ("Sanskrit", "Bharati", "Aryaman"), "sat": ("Santali", "Phulmani", "Sibu"),
    "sd": ("Sindhi", "Moomal", "Rano"), "ta": ("Tamil", "Anitha", "Arun"),
    "te": ("Telugu", "Sravani", "Vamsi"), "ur": ("Urdu", "Saba", "Zaid"),
}
# The model card's "Production" tier; the rest are marked "preview" by its authors.
_PRODUCTION = {"hi", "bn", "mr", "te", "ta", "gu", "kn", "ml", "or", "pa", "as", "ur"}
DEFAULT_LANGUAGE = "hi"

LANGUAGES = sorted(_CATALOG)
VOICES: List[dict] = [
    {"id": voice, "name": voice, "language": name, "language_code": code, "gender": gender,
     "tier": "production" if code in _PRODUCTION else "preview"}
    for code, (name, female, male) in _CATALOG.items()
    for voice, gender in ((female, "female"), (male, "male"))
]
_SPEAKERS = {v["id"] for v in VOICES}


def is_known_speaker(name: str) -> bool:
    return name in _SPEAKERS


def default_speaker(language: Optional[str], gender: str = "female") -> str:
    """The recommended voice for a language code; Hindi's when the language is not in the catalog."""
    _, female, male = _CATALOG.get((language or "").split("-")[0].lower(), _CATALOG[DEFAULT_LANGUAGE])
    return male if gender == "male" else female


class IndicSpeakError(Exception):
    """A failure the API layer turns into a structured error (stable code + HTTP status)."""

    def __init__(self, message: str, code: str = "GENERATION_FAILED", status: int = 500):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


# -- availability ---------------------------------------------------------------------------------------------

def quantized_model_dir() -> Optional[str]:
    """INDIC_SPEAK_MLX_MODEL: folder with the MLX 4-bit language model (scripts/setup_indic_speak_q4.sh). When it is set and
    complete, the worker runs it on the Apple GPU: about 2 GB instead of 7 GB, 3.7x faster, +0.01 read-back CER."""
    path = os.getenv("INDIC_SPEAK_MLX_MODEL", "").strip()
    if path and os.path.isfile(os.path.join(os.path.expanduser(path), "model.safetensors")):
        return os.path.expanduser(path)
    return None


def indic_speak_python() -> Optional[str]:
    """The python of the Indic-Speak virtualenv (the 4-bit one when INDIC_SPEAK_MLX_MODEL is set), or None when it has not
    been set up."""
    venv = ".venv-indic-q4" if quantized_model_dir() else ".venv-indic"
    for path in (os.getenv("INDIC_SPEAK_PYTHON"),
                 os.path.join(BACKEND_DIR, venv, "bin", "python"),
                 os.path.join(BACKEND_DIR, venv, "Scripts", "python.exe")):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def weights_cached(model: Optional[str] = None) -> bool:
    """True when the model, the vocoder and the model's inference code are all in the Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        repo = model or os.getenv("INDIC_SPEAK_MODEL", DEFAULT_MODEL)
        return all(isinstance(try_to_load_from_cache(repo, f), str)
                   for f in ("config.json", "model.safetensors", "vocos/best.pt", "inference.py"))
    except Exception:  # noqa: BLE001  huggingface_hub missing or the cache is unreadable
        return False


def indic_speak_unavailable_reason() -> Optional[str]:
    """Why Indic-Speak cannot run here, or None when it can."""
    if os.getenv("INDIC_SPEAK", "").strip().lower() in ("0", "off", "false", "disabled"):
        return "Indic-Speak is turned off (INDIC_SPEAK=off)."
    if indic_speak_python() is None:
        return "Indic-Speak is not set up. Run: bash backend/scripts/setup_indic_speak.sh"
    if not weights_cached():
        return "The Indic-Speak weights are not downloaded. Run: bash backend/scripts/setup_indic_speak.sh"
    return None


# -- text handling --------------------------------------------------------------------------------------------

def clean_text(s: str) -> str:
    """Remove control-token look-alikes, repeating until stable (mirrors the worker)."""
    prev = None
    while prev != s:
        prev, s = s, CONTROL_TOKEN_RE.sub("", s or "")
    return s.strip()


_SENTENCE_END = re.compile(r"(?<=[।॥.!?…])\s+|\n+")


def split_for_speech(text: str, max_chars: int = CHUNK_CHARS) -> List[str]:
    """One chunk per sentence (Devanagari danda included), each at most max_chars.

    Sentences are generated separately on purpose: the model derails on repeated text (the same sentence twice in
    one request made it babble to its length limit), and separate requests cannot loop on each other."""
    text = re.sub(r"[ \t]+", " ", clean_text(text))
    chunks: List[str] = []
    for piece in (p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()):
        while len(piece) > max_chars:  # one very long sentence: break at a space near the limit
            cut = piece.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if not piece:
            continue
        if chunks and len(piece) < MIN_SENTENCE_CHARS and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return chunks


# -- worker client --------------------------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class IndicSpeakModel:
    """Owns the worker process. The API server's model manager loads, idles out and closes it like any engine."""

    def __init__(self, python: str, model: Optional[str] = None, start_timeout: float = 900.0):
        self.model_name = model or os.getenv("INDIC_SPEAK_MODEL", DEFAULT_MODEL)
        self.request_timeout = float(os.getenv("INDIC_SPEAK_TIMEOUT", "600"))
        self._active = 0  # requests in flight; the model manager must not evict a busy worker
        self._active_lock = threading.Lock()
        self.port = _free_port()
        worker = os.path.join(BACKEND_DIR, "services", "indic_speak_worker.py")
        cmd = [python, worker, "--port", str(self.port), "--model", self.model_name]
        if quantized_model_dir():
            cmd += ["--mlx-model", quantized_model_dir()]
        self.proc = subprocess.Popen(cmd)
        deadline = time.time() + start_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Indic-Speak worker exited with code {self.proc.returncode} "
                                   "(see the log above; the weights may not be downloaded)")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if json.load(r).get("ok"):
                        log.info("Indic-Speak worker ready: %s", self.model_name)
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(1)
        self.close()
        raise RuntimeError("Indic-Speak worker did not become ready in time")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def busy(self) -> bool:
        """True while a generation is running. At ~7x real time (measured RTF 6.96 on an Apple M4) one can outlast the manager's idle timeout."""
        return self._active > 0

    def eval(self):  # parity with torch modules (the manager calls _set_eval)
        return self

    def generate(self, text: str, speaker: str, **params) -> Tuple[np.ndarray, int, bool]:
        """One utterance (the worker caps it near 30 s). Returns (mono float32 audio, sample rate, truncated)."""
        body = json.dumps({"text": text, "speaker": speaker, **params}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with self._active_lock:
            self._active += 1
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as r:
                data, truncated = r.read(), r.headers.get("X-Truncated") == "1"
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise IndicSpeakError(f"Indic-Speak worker: {detail}", "GENERATION_FAILED",
                                  422 if e.code == 400 else 500) from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if not self.alive():
                raise IndicSpeakError("The Indic-Speak worker stopped unexpectedly. Try again.",
                                      "ENGINE_UNAVAILABLE", 503) from e
            raise IndicSpeakError("Indic-Speak did not finish in time. Try a shorter text.",
                                  "GENERATION_TIMEOUT", 504) from e
        finally:
            with self._active_lock:
                self._active -= 1
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        return audio, int(sr), truncated

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    __del__ = close


def token_cap(text: str) -> int:
    """How many tokens a clip of this text may use: room for slow speech, but not for a runaway."""
    return int(min(2520, max(120, TOKENS_PER_SECOND * (len(text) / MIN_CHARS_PER_SECOND + 1.5))))


def _halve(text: str) -> Tuple[str, str]:
    """Split at the space nearest the middle (or the middle itself when there is none)."""
    mid = len(text) // 2
    left = text.rfind(" ", 0, mid + 1)
    right = text.find(" ", mid)
    cut = min((c for c in (left, right) if c > 0), key=lambda c: abs(c - mid), default=mid)
    return text[:cut].strip(), text[cut:].strip()


def _speak(model: IndicSpeakModel, text: str, speaker: str, depth: int = 0) -> List[np.ndarray]:
    """One chunk. A clip that hits its token cap is a runaway, never returned: the chunk is split and retried, and
    when splitting does not help the request fails with a clear error rather than returning babble."""
    audio, _, runaway = model.generate(text, speaker, max_new_tokens=token_cap(text))
    if not runaway:
        return [audio]
    log.warning("Indic-Speak ran away on %d characters (depth %d); %s", len(text), depth,
                "splitting" if depth < MAX_SPLIT_DEPTH and len(text) >= 30 else "giving up")
    if depth >= MAX_SPLIT_DEPTH or len(text) < 30:
        raise IndicSpeakError("Indic-Speak could not speak this text cleanly. Try rewording it.",
                              "GENERATION_FAILED", 502)
    left, right = _halve(text)
    return _speak(model, left, speaker, depth + 1) + _speak(model, right, speaker, depth + 1)


def synthesize(model: IndicSpeakModel, text: str, speaker: str) -> Tuple[np.ndarray, int]:
    """Speak text of any length (up to MAX_TEXT_CHARS), one sentence per generation, stitched with short pauses."""
    chunks = split_for_speech(text)
    if not chunks:
        raise IndicSpeakError("There is no text to speak.", "EMPTY_TEXT", 400)
    pause = np.zeros(int(SAMPLE_RATE * PAUSE_SECONDS), dtype=np.float32)
    parts: List[np.ndarray] = []
    for chunk in chunks:
        for audio in _speak(model, chunk, speaker):
            if parts:
                parts.append(pause)
            parts.append(audio)
    return np.concatenate(parts), SAMPLE_RATE
