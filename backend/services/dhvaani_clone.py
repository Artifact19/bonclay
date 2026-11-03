"""DhVaani-0.5 (ARTPARK-IISc): voice cloning in Indian languages.

A consented voice, even from an English sample, speaks Hindi, Marathi, Tamil, Bengali, Telugu, Kannada, Malayalam,
Gujarati, Punjabi, Odia, Urdu, Assamese and more. It is a small model (about 0.1B parameters, ZipVoice-based, 491 MB)
that runs on the Apple GPU, an NVIDIA GPU or the CPU. Measured on an M4 with synthetic references (docs/TRIAL_REPORT.md,
docs/decisions/indian-language-cloning.md): no failures in 50 renders, read-back character error rate Hindi 5-8 %, Tamil
4-6 %, Telugu 7-11 %, Marathi 9-18 %, Bengali 12-18 %, speaker similarity about 0.69 from a Hindi reference and about 0.41
from an English one, about 1.3-3x slower than real time under load. That is fine for Studio renders, Say it once and
pre-rendered clips; a live phone or agent turn keeps the fast preset voice.

Unlike the other cloning engines it needs the TRANSCRIPT of the reference clip as well as the clip, and the clip should
be a clean 3-10 s of one voice. `select_reference` cuts such a clip out of a longer sample at a pause (the cut clip is
transcribed by the caller; when that is impossible a proportional slice of the stored transcript is the fallback).

Like Chatterbox it runs in its own virtualenv: this module starts services/dhvaani_worker.py there and talks to it over
127.0.0.1, and exposes the subset of the Qwen3TTSModel interface the API server uses for cloning
(create_voice_clone_prompt() and generate_voice_clone()).

Enabled when a worker python exists ($DHVAANI_PYTHON, else backend/.venv-dhvaani/bin/python, created by
scripts/setup_dhvaani.sh) and the weights are downloaded. Set DHVAANI=off to opt out. DHVAANI_STEPS (default 8; 16 was the
default until 20 Oct 2026 and is a little slower) sets the number of sampling steps. A worker is not started when the machine has less than DHVAANI_MIN_FREE_MB
(default 1500) of memory available.
"""
import hashlib
import io
import json
import logging
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf

log = logging.getLogger("bonclay.dhvaani")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_KEY = "dhvaani"
DEFAULT_MODEL = "ARTPARK-IISc/DhVaani-0.5"
PROMPT_BACKEND = "dhvaani"
PROMPT_SAMPLE_RATE = 24000
SAMPLE_RATE = 24000
SETUP_HINT = "./bonclay setup --dhvaani"
ENGINE_NAME = "DhVaani (ARTPARK-IISc)"
STUDIO_LABEL = "Your voice (DhVaani · ARTPARK-IISc)"
CREDIT = "Voice cloning in Indian languages by DhVaani from ARTPARK-IISc."
# The languages routed to DhVaani: the model's well and fairly represented ones. Sanskrit, Dogri, Kashmiri and
# Rajasthani are left out (about a tenth of the training data each): they stay on preset voices.
LANGUAGES = frozenset({"hi", "mr", "ta", "bn", "te", "kn", "ml", "gu", "pa", "or", "ur", "as", "ne", "mai", "mag",
                       "hne", "bho", "kok", "sd", "mni", "sat", "brx"})
MAX_TEXT_CHARS = int(os.getenv("DHVAANI_MAX_CHARS", "1000"))  # per request: ~2x real time on a laptop
MIN_FREE_MB = int(os.getenv("DHVAANI_MIN_FREE_MB", "1500"))
_WEIGHT_FILES = ("model.safetensors", "model.json", "tokens.txt")

# The reference clip the model likes: clean, one voice, 3 to 10 seconds.
MIN_REF_S = 3.0
MAX_REF_S = 10.0
TARGET_REF_S = 8.0
HARD_MIN_REF_S = 1.5   # below this the sample is refused
_FRAME_S = 0.02
_MIN_PAUSE_S = 0.15
_EDGE_KEEP_S = 0.15


# -- availability ---------------------------------------------------------------------------------------------

def dhvaani_python() -> Optional[str]:
    """The python of the DhVaani virtualenv, or None when it has not been set up."""
    for path in (os.getenv("DHVAANI_PYTHON"),
                 os.path.join(BACKEND_DIR, ".venv-dhvaani", "bin", "python"),
                 os.path.join(BACKEND_DIR, ".venv-dhvaani", "Scripts", "python.exe")):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def weights_cached(model: Optional[str] = None) -> bool:
    """True when the model files are in the Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        repo = model or os.getenv("DHVAANI_MODEL", DEFAULT_MODEL)
        return all(isinstance(try_to_load_from_cache(repo, f), str) for f in _WEIGHT_FILES)
    except Exception:  # noqa: BLE001  huggingface_hub missing or the cache is unreadable
        return False


def turned_off() -> bool:
    return os.getenv("DHVAANI", "").strip().lower() in ("0", "off", "false", "disabled")


def unavailable_reason() -> Optional[str]:
    """Why DhVaani cannot run here, or None when it can."""
    if turned_off():
        return "DhVaani is turned off (DHVAANI=off)."
    if dhvaani_python() is None:
        return f"DhVaani is not set up. Run: {SETUP_HINT}"
    if not weights_cached():
        return f"The DhVaani weights are not downloaded. Run: {SETUP_HINT}"
    return None


def available() -> bool:
    return unavailable_reason() is None


def supports(language: Optional[str]) -> bool:
    return (language or "").strip().lower().split("-")[0] in LANGUAGES


def free_memory_mb() -> Optional[float]:
    """Memory the machine could hand out now, in MB (None when it cannot be read)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 20
    except Exception:  # noqa: BLE001
        return None


def memory_problem(free_mb: Optional[float] = None, minimum_mb: Optional[int] = None) -> Optional[str]:
    """A clear message when there is too little memory to start the worker, else None. Reading failures do not block."""
    free = free_memory_mb() if free_mb is None else free_mb
    need = MIN_FREE_MB if minimum_mb is None else minimum_mb
    if free is None or need <= 0 or free >= need:
        return None
    return (f"Only {free / 1024:.1f} GB of memory is free and DhVaani needs about {need / 1024:.1f} GB to start. "
            "Close other apps (or stop a model that is loaded) and try again.")


# -- the reference clip ---------------------------------------------------------------------------------------

class ReferenceError(ValueError):
    """The voice sample cannot serve as a DhVaani reference (too short, silent...)."""


@dataclass
class ReferenceChoice:
    start: float          # seconds into the recording
    end: float
    whole: bool           # True when the clip is the whole recording (edge silence trimmed): its transcript is the stored one
    clean_cut: bool       # False when no pause was found and the cut falls mid-speech

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ReferenceSelection:
    audio: np.ndarray                 # mono float32 clip
    sample_rate: int
    choice: ReferenceChoice
    text: Optional[str]               # the transcript when it is known without recognising speech, else None
    approximate_text: Optional[str]   # a proportional slice of the stored transcript: the fallback when recognition fails

    @property
    def needs_transcription(self) -> bool:
        return self.text is None


def _frame_db(audio: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(sr * _FRAME_S))
    frames = len(audio) // n
    if frames == 0:
        return np.full(1, -120.0)
    x = audio[: frames * n].reshape(frames, n).astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-6))


def _silence_threshold(db: np.ndarray) -> float:
    peak = float(np.percentile(db, 98))
    floor = float(np.percentile(db, 10))
    return max(peak - 35.0, floor + 6.0)


def speech_bounds(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """(start, end) seconds of the speech in a recording, with a little silence kept each side."""
    db = _frame_db(audio, sr)
    loud = np.where(db > _silence_threshold(db))[0]
    if loud.size == 0 or float(np.percentile(db, 98)) < -60.0:
        raise ReferenceError("The voice sample is silent. Record it again, closer to the microphone.")
    total = len(audio) / float(sr)
    return (max(0.0, loud[0] * _FRAME_S - _EDGE_KEEP_S), min(total, (loud[-1] + 1) * _FRAME_S + _EDGE_KEEP_S))


def find_pauses(audio: np.ndarray, sr: int, start: float, end: float, min_pause_s: float = _MIN_PAUSE_S) -> List[float]:
    """The middle (seconds) of every silence of at least `min_pause_s` strictly inside (start, end)."""
    db = _frame_db(audio, sr)
    quiet = db <= _silence_threshold(db)
    mids: List[float] = []
    i, n = 0, len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * _FRAME_S >= min_pause_s:
                mid = (i + j) / 2.0 * _FRAME_S
                if start + 0.5 < mid < end - 0.5:
                    mids.append(mid)
            i = j
        else:
            i += 1
    return mids


def choose_reference(audio: np.ndarray, sr: int) -> ReferenceChoice:
    """The clip to use as the reference: the whole recording when it is at most 10 s (edge silence trimmed), else the
    4-10 s stretch between two pauses closest to 8 s, else (no usable pauses) a cut at the quietest moment."""
    start, end = speech_bounds(audio, sr)
    if end - start < HARD_MIN_REF_S:
        raise ReferenceError("The voice sample is too short to use as a reference. Record at least 3 seconds of speech.")
    if end - start <= MAX_REF_S:
        return ReferenceChoice(start, end, whole=True, clean_cut=True)
    bounds = [start] + find_pauses(audio, sr, start, end) + [end]
    best: Optional[Tuple[float, float, float]] = None
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            dur = bounds[j] - bounds[i]
            if dur > MAX_REF_S:
                break
            if dur < MIN_REF_S + 1.0:  # prefer 4 s or more
                continue
            score = abs(dur - TARGET_REF_S)
            if best is None or score < best[0] - 1e-6:
                best = (score, bounds[i], bounds[j])
    if best is not None:
        return ReferenceChoice(best[1], best[2], whole=False, clean_cut=True)
    # no pause pair gives a 4-10 s clip: cut at the quietest frame between 5 s and 10 s from the start
    db = _frame_db(audio, sr)
    lo, hi = int((start + 5.0) / _FRAME_S), min(len(db) - 1, int((start + MAX_REF_S) / _FRAME_S))
    cut = (lo + int(np.argmin(db[lo:hi + 1]))) * _FRAME_S if hi > lo else start + TARGET_REF_S
    return ReferenceChoice(start, cut, whole=False, clean_cut=False)


def proportional_words(transcript: str, choice: ReferenceChoice, speech_start: float, speech_end: float) -> str:
    """The share of `transcript` that falls in the chosen clip, assuming words are spread evenly over the speech. An
    approximation (a boundary can be a word off): used only when the clip cannot be transcribed."""
    words = (transcript or "").split()
    span = max(speech_end - speech_start, 1e-6)
    a = round(len(words) * max(0.0, choice.start - speech_start) / span)
    b = round(len(words) * min(span, choice.end - speech_start) / span)
    return " ".join(words[a:max(a + 1, b)])


def select_reference(audio: np.ndarray, sr: int, transcript: Optional[str] = None) -> ReferenceSelection:
    """Pick the reference clip from a voice sample (mono float array at rate `sr`) and what is known of its words.

    A clip that is the whole sample keeps the stored `transcript`; a cut clip has no known transcript (the caller
    transcribes it) but carries a proportional slice of the stored one as the fallback."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    choice = choose_reference(audio, sr)
    clip = audio[int(choice.start * sr): int(choice.end * sr)]
    transcript = (transcript or "").strip()
    if choice.whole:
        return ReferenceSelection(clip, sr, choice, transcript or None, None)
    speech_start, speech_end = speech_bounds(audio, sr)
    approx = proportional_words(transcript, choice, speech_start, speech_end) if transcript else None
    return ReferenceSelection(clip, sr, choice, None, approx)


def prompt_from_clip(audio: np.ndarray, sr: int, text: str) -> dict:
    """The clone prompt DhVaani uses: the reference clip (as WAV bytes) and its transcript."""
    text = (text or "").strip()
    if not text:
        raise ReferenceError("DhVaani needs the transcript of the reference clip.")
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), int(sr), format="WAV")
    return {"backend": PROMPT_BACKEND, "ref_wav": buf.getvalue(), "ref_text": text}


def is_dhvaani_prompt(prompt: Any) -> bool:
    return isinstance(prompt, dict) and prompt.get("backend") == PROMPT_BACKEND


def source_fingerprint(wav_bytes: bytes, transcript: Optional[str]) -> str:
    """Identifies the (recording, stored transcript) a cached reference pair was made from."""
    h = hashlib.sha256(wav_bytes)
    h.update(b"\0" + (transcript or "").strip().encode("utf-8"))
    return h.hexdigest()


# -- worker client --------------------------------------------------------------------------------------------

class DhVaaniError(Exception):
    """A failure the API layer turns into a structured error (stable code + HTTP status)."""

    def __init__(self, message: str, code: str = "GENERATION_FAILED", status: int = 500):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DhVaaniClone:
    """Owns the worker process. Drop-in for the subset of Qwen3TTSModel the API server uses for cloning."""

    prompt_backend = PROMPT_BACKEND
    prompt_sample_rate = PROMPT_SAMPLE_RATE
    wants_language_code = True  # generate_voice_clone gets a real language code, not "Auto"
    engine_key = ENGINE_KEY

    def __init__(self, python: str, model: Optional[str] = None, start_timeout: float = 300.0):
        self.proc = None
        problem = memory_problem()
        if problem:
            log.warning("DhVaani worker not started: %s", problem)
            raise DhVaaniError(problem, "INSUFFICIENT_MEMORY", 503)
        self.model_name = model or os.getenv("DHVAANI_MODEL", DEFAULT_MODEL)
        self.request_timeout = float(os.getenv("DHVAANI_TIMEOUT", "900"))
        self._active = 0
        self._active_lock = threading.Lock()
        self._ref_dir = tempfile.mkdtemp(prefix="bonclay_dhvaani_refs_")
        self.port = _free_port()
        worker = os.path.join(BACKEND_DIR, "services", "dhvaani_worker.py")
        env = {**os.environ, "HF_HUB_DISABLE_XET": "1", "TOKENIZERS_PARALLELISM": "false"}
        self.proc = subprocess.Popen([python, worker, "--port", str(self.port), "--model", self.model_name], env=env)
        deadline = time.time() + start_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"DhVaani worker exited with code {self.proc.returncode} "
                                   f"(see the log above; run {SETUP_HINT} if the environment is incomplete)")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    info = json.load(r)
                    if info.get("ok"):
                        self.device = info.get("device")
                        log.info("DhVaani worker ready: %s on %s", self.model_name, self.device)
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(1)
        self.close()
        raise RuntimeError("DhVaani worker did not become ready in time")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def busy(self) -> bool:
        """True while a generation is running: the model manager must not stop a busy worker."""
        return self._active > 0

    def eval(self):  # parity with torch modules (the manager calls _set_eval)
        return self

    # -- Qwen3TTSModel-compatible surface --------------------------------------------------------

    def create_voice_clone_prompt(self, ref_audio: List[Tuple[np.ndarray, int]], ref_text: Optional[str] = None,
                                  x_vector_only_mode: bool = False) -> dict:
        """The reference clip and its transcript. DhVaani has no embedding step and no transcript-free mode."""
        if not (ref_text or "").strip():
            raise ReferenceError("DhVaani needs the transcript of the reference clip.")
        audio, sr = ref_audio[0]
        return prompt_from_clip(audio, sr, ref_text)

    def generate_voice_clone(self, text: str, language: str = "Auto", voice_clone_prompt: Any = None,
                             **_unused) -> Tuple[List[np.ndarray], int]:
        p = voice_clone_prompt or {}
        if not is_dhvaani_prompt(p):
            raise ValueError("Voice prompt was not built for the DhVaani engine")
        body = {"text": text, "ref_audio_path": self._ref_path(p["ref_wav"]), "ref_text": p["ref_text"]}
        code = (language or "").strip().lower().split("-")[0]
        if code in LANGUAGES:
            body["language"] = code
        for key in ("num_step", "speed", "seed"):
            if key in _unused:
                body[key] = _unused[key]
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with self._active_lock:
            self._active += 1
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as r:
                audio, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise DhVaaniError(f"DhVaani worker: {detail}", "GENERATION_FAILED", 422 if e.code == 400 else 500) from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if not self.alive():
                raise DhVaaniError("The DhVaani worker stopped unexpectedly. Try again.", "ENGINE_UNAVAILABLE", 503) from e
            raise DhVaaniError("DhVaani did not finish in time. Try a shorter text.", "GENERATION_TIMEOUT", 504) from e
        finally:
            with self._active_lock:
                self._active -= 1
        return [audio], int(sr)

    def _ref_path(self, wav: bytes) -> str:
        """Reference audio as a file the worker can read; cached by content hash."""
        path = os.path.join(self._ref_dir, hashlib.sha1(wav).hexdigest() + ".wav")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(wav)
        return path

    def close(self) -> None:
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        ref_dir = getattr(self, "_ref_dir", None)
        if ref_dir and os.path.isdir(ref_dir):  # the reference clips are the voice owner's: never leave them in /tmp
            for name in os.listdir(ref_dir):
                try:
                    os.unlink(os.path.join(ref_dir, name))
                except OSError:
                    pass
            try:
                os.rmdir(ref_dir)
            except OSError:
                pass

    __del__ = close
