"""Chatterbox Multilingual (Resemble AI, MIT licence): voice cloning for Hindi and 22 other languages.

It is the engine behind Hindi voice cloning: an English or a Hindi voice sample speaks intelligible Hindi in the
same voice (bake-off on a Mac: character error rate about 0.27 and speaker similarity about 0.80 from an English
sample, CER about 0.21 from a Hindi one). Of the Indian languages it clones Hindi only.

It is slow: on the CPU about 4-6 seconds of compute per second of speech (the Apple GPU path was slower still and
sometimes failed with NaNs, so the worker always uses the CPU). It is for Studio renders and pre-rendered clips,
never for live agent turns.

chatterbox-tts pins its own torch / numpy / transformers, so it runs in its own virtualenv. This module starts
services/chatterbox_worker.py in that environment and talks to it over 127.0.0.1, the same way services/indic_speak.py
does, and exposes the subset of the Qwen3TTSModel interface the API server uses for cloning:
create_voice_clone_prompt() and generate_voice_clone().

Enabled when a worker python exists ($CHATTERBOX_PYTHON, else backend/.venv-chatterbox/bin/python, created by
scripts/setup_chatterbox.sh) and the weights are downloaded. Set CHATTERBOX=off to opt out.
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
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf

log = logging.getLogger("bonclay.chatterbox")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "ResembleAI/chatterbox"
PROMPT_BACKEND = "chatterbox"
PROMPT_SAMPLE_RATE = 24000  # reference audio is stored at the model's rate (it resamples to 16 kHz itself)
SAMPLE_RATE = 24000
SETUP_HINT = "./bonclay setup --chatterbox"
# Hindi first: it is the only Indian language Chatterbox clones. The rest are the package's other languages.
LANGUAGES = ("hi", "en", "ar", "da", "de", "el", "es", "fi", "fr", "he", "it", "ja", "ko", "ms", "nl", "no", "pl",
             "pt", "ru", "sv", "sw", "tr", "zh")
MAX_TEXT_CHARS = int(os.getenv("CHATTERBOX_MAX_CHARS", "1000"))  # per request; ~5x real time on a laptop CPU
_WEIGHT_FILES = ("ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json")


# -- availability ---------------------------------------------------------------------------------------------

def chatterbox_python() -> Optional[str]:
    """The python of the Chatterbox virtualenv, or None when it has not been set up."""
    for path in (os.getenv("CHATTERBOX_PYTHON"),
                 os.path.join(BACKEND_DIR, ".venv-chatterbox", "bin", "python"),
                 os.path.join(BACKEND_DIR, ".venv-chatterbox", "Scripts", "python.exe")):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def weights_cached(model: Optional[str] = None) -> bool:
    """True when the model files are in the Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        repo = model or os.getenv("CHATTERBOX_MODEL", DEFAULT_MODEL)
        return all(isinstance(try_to_load_from_cache(repo, f), str) for f in _WEIGHT_FILES)
    except Exception:  # noqa: BLE001  huggingface_hub missing or the cache is unreadable
        return False


def unavailable_reason() -> Optional[str]:
    """Why Chatterbox cannot run here, or None when it can."""
    if os.getenv("CHATTERBOX", "").strip().lower() in ("0", "off", "false", "disabled"):
        return "Chatterbox is turned off (CHATTERBOX=off)."
    if chatterbox_python() is None:
        return f"Chatterbox is not set up. Run: {SETUP_HINT}"
    if not weights_cached():
        return f"The Chatterbox weights are not downloaded. Run: {SETUP_HINT}"
    return None


def resolve_language(language: Optional[str], text: str = "") -> str:
    """A Chatterbox language code. "Auto" (what the Qwen3 callers pass) is read from the script of the text."""
    code = (language or "").strip().lower().split("-")[0]
    if code in LANGUAGES:
        return code
    return "hi" if re.search(r"[ऀ-ॿ]", text or "") else "en"


# -- worker client --------------------------------------------------------------------------------------------

class ChatterboxError(Exception):
    """A failure the API layer turns into a structured error (stable code + HTTP status)."""

    def __init__(self, message: str, code: str = "GENERATION_FAILED", status: int = 500):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ChatterboxClone:
    """Owns the worker process. Drop-in for the subset of Qwen3TTSModel the API server uses for cloning."""

    prompt_backend = PROMPT_BACKEND
    prompt_sample_rate = PROMPT_SAMPLE_RATE
    wants_language_code = True  # generate_voice_clone needs a real language, not "Auto"

    def __init__(self, python: str, model: Optional[str] = None, start_timeout: float = 900.0):
        self.proc = None
        self.model_name = model or os.getenv("CHATTERBOX_MODEL", DEFAULT_MODEL)
        self.request_timeout = float(os.getenv("CHATTERBOX_TIMEOUT", "1800"))
        self._active = 0  # requests in flight; the model manager must not evict a busy worker
        self._active_lock = threading.Lock()
        self._ref_dir = tempfile.mkdtemp(prefix="bonclay_chatterbox_refs_")
        self.port = _free_port()
        worker = os.path.join(BACKEND_DIR, "services", "chatterbox_worker.py")
        env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1", "TOKENIZERS_PARALLELISM": "false"}
        self.proc = subprocess.Popen([python, worker, "--port", str(self.port), "--model", self.model_name], env=env)
        deadline = time.time() + start_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Chatterbox worker exited with code {self.proc.returncode} "
                                   f"(see the log above; run {SETUP_HINT} if the environment is incomplete)")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if json.load(r).get("ok"):
                        log.info("Chatterbox worker ready: %s", self.model_name)
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(1)
        self.close()
        raise RuntimeError("Chatterbox worker did not become ready in time")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def busy(self) -> bool:
        """True while a generation is running: at ~5x real time one can outlast the manager's idle timeout."""
        return self._active > 0

    def eval(self):  # parity with torch modules (the manager calls _set_eval)
        return self

    # -- Qwen3TTSModel-compatible surface --------------------------------------------------------

    def create_voice_clone_prompt(self, ref_audio: List[Tuple[np.ndarray, int]], ref_text: Optional[str] = None,
                                  x_vector_only_mode: bool = False) -> dict:
        """The reference audio itself: Chatterbox conditions on the sample, it has no separate embedding step.
        (ref_text and x_vector_only_mode are accepted for parity and unused.)"""
        audio, sr = ref_audio[0]
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), int(sr), format="WAV")
        return {"backend": PROMPT_BACKEND, "ref_wav": buf.getvalue()}

    def generate_voice_clone(self, text: str, language: str = "Auto", voice_clone_prompt: Any = None,
                             **_unused) -> Tuple[List[np.ndarray], int]:
        p = voice_clone_prompt or {}
        if p.get("backend") != PROMPT_BACKEND:
            raise ValueError("Voice prompt was not built for the Chatterbox engine")
        params = {k: _unused[k] for k in ("exaggeration", "cfg_weight", "temperature", "seed") if k in _unused}
        body = json.dumps({"text": text, "language": resolve_language(language, text),
                           "ref_audio": self._ref_path(p["ref_wav"]), **params}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with self._active_lock:
            self._active += 1
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as r:
                audio, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise ChatterboxError(f"Chatterbox worker: {detail}", "GENERATION_FAILED",
                                  422 if e.code == 400 else 500) from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if not self.alive():
                raise ChatterboxError("The Chatterbox worker stopped unexpectedly. Try again.",
                                      "ENGINE_UNAVAILABLE", 503) from e
            raise ChatterboxError("Chatterbox did not finish in time. Try a shorter text.",
                                  "GENERATION_TIMEOUT", 504) from e
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

    __del__ = close


def is_chatterbox_prompt(prompt: Any) -> bool:
    return isinstance(prompt, dict) and prompt.get("backend") == PROMPT_BACKEND
