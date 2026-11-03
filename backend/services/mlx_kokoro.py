"""Kokoro-82M on Apple Silicon through MLX, with automatic fallback to PyTorch Kokoro.

On a Mac the MLX build of Kokoro (mlx-audio, mlx-community/Kokoro-82M-bf16) speaks the same voices as the PyTorch
package and was measured 10-47% faster per sentence on an M4 (English 0.39 s vs 0.66 s, Hindi 0.45 s vs 0.83 s) with
identical read-back scores. It runs as a worker process (services/mlx_kokoro_worker.py) in backend/.venv-mlx, started
and owned by the API server like the other MLX workers, and talks HTTP over 127.0.0.1.

KOKORO_ENGINE (default auto):
    auto   MLX on Apple Silicon when backend/.venv-mlx (or $MLX_PYTHON) exists, PyTorch everywhere else
    mlx    the same, but names the MLX engine explicitly (PyTorch is still the fallback; the turn never fails)
    torch  never start the worker; PyTorch Kokoro only

What the rest of the server sees: advanced_manager.load_model("kokoro") returns a `MlxKokoro` handle instead of a
PyTorch KPipeline. backend._kokoro_generate / _kokoro_with_voice ask the handle first; when it returns None (the worker
failed, or it cannot speak that voice or language) they take the handle's lazily built PyTorch pipeline for that request
and log the reason once. So PyTorch Kokoro (about 1 GB of RAM) is only loaded when a fallback actually happens.
The audio is the same 24 kHz mono float32 as the PyTorch path, so the watermark, receipt and file formats are untouched.
"""
import io
import json
import logging
import os
import platform
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

import numpy as np
import soundfile as sf

log = logging.getLogger("bonclay.mlx_kokoro")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
SAMPLE_RATE = 24000
MODES = ("auto", "mlx", "torch")
RETRY_AFTER_S = 60.0  # after a failed start, use PyTorch for this long before trying the worker again


class Unsupported(Exception):
    """The worker cannot speak this voice or language (a missing optional package): not a failure of the worker."""


class WorkerError(Exception):
    """The worker failed or could not be reached."""


# ------------------------------------------------------------------------------------------------ routing
_state_lock = threading.Lock()
_logged: set = set()
_failed_at: Optional[float] = None


def log_once(key: str, message: str, *args, level: int = logging.WARNING) -> None:
    """Log a given problem once per process, so a broken worker does not repeat itself on every sentence."""
    with _state_lock:
        if key in _logged:
            return
        _logged.add(key)
    log.log(level, message, *args)


def engine_mode() -> str:
    mode = os.getenv("KOKORO_ENGINE", "auto").strip().lower()
    if mode not in MODES:
        log_once("bad-mode", "KOKORO_ENGINE=%r is not one of %s; using auto", mode, "/".join(MODES))
        return "auto"
    return mode


def mlx_python() -> Optional[str]:
    """The python of the MLX virtualenv, or None when there is none or this is not an Apple Silicon Mac."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    for path in (os.getenv("MLX_PYTHON"), os.path.join(BACKEND_DIR, ".venv-mlx", "bin", "python")):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def unavailable_reason() -> Optional[str]:
    """Why the MLX voice will not be used (None when it will be tried)."""
    if engine_mode() == "torch":
        return "KOKORO_ENGINE=torch"
    if mlx_python() is None:
        return "needs Apple Silicon and the MLX environment (bash backend/scripts/setup_mlx.sh)"
    with _state_lock:
        if _failed_at is not None and time.time() - _failed_at < RETRY_AFTER_S:
            return "the MLX worker failed to start a moment ago"
    return None


def wants_mlx() -> bool:
    return unavailable_reason() is None


def note_start_failure(error: Any) -> None:
    global _failed_at
    with _state_lock:
        _failed_at = time.time()
    log_once("start", "MLX Kokoro could not start (%s); using PyTorch Kokoro", error)


def is_handle(obj: Any) -> bool:
    return isinstance(obj, MlxKokoro)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------------------------------------ the handle
class MlxKokoro:
    """What advanced_manager keeps for "kokoro" when the MLX engine is in use: a worker process plus a PyTorch fallback."""

    proc: Any = None  # class defaults so close() (also run by __del__) is safe when __init__ failed early
    _torch: Any = None

    def __init__(self, python: Optional[str] = None, model: str = DEFAULT_MODEL, start_timeout: float = 900.0):
        python = python or mlx_python()
        if not python:
            raise RuntimeError("no MLX python found")
        self.model_name = os.getenv("MLX_KOKORO_MODEL", model)
        self.port = _free_port()
        self.last_engine: Optional[str] = None   # "mlx" or "torch": what served the latest request
        self.last_error: Optional[str] = None
        self._inflight = 0
        self._lock = threading.Lock()
        self._torch: Any = None
        self._torch_lock = threading.Lock()
        worker = os.path.join(BACKEND_DIR, "services", "mlx_kokoro_worker.py")
        self.proc = subprocess.Popen([python, worker, "--port", str(self.port), "--model", self.model_name])
        deadline = time.time() + start_timeout  # the first run downloads the weights and voices
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"MLX Kokoro worker exited with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if json.load(r).get("ok"):
                        log.info("MLX Kokoro worker ready: %s", self.model_name)
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(0.5)
        self.close()
        raise RuntimeError("MLX Kokoro worker did not become ready in time")

    # -- manager hooks (AdvancedModelManager) ----------------------------------------------------
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def busy(self) -> bool:
        return self._inflight > 0

    def eval(self):  # parity with torch modules
        return self

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._torch = None

    __del__ = close

    # -- synthesis -------------------------------------------------------------------------------
    def synthesize(self, text: str, voice: str, lang_code: str, speed: float = 1.0, timeout: float = 120.0) -> np.ndarray:
        """24 kHz mono float32 from the worker. Raises Unsupported or WorkerError."""
        body = json.dumps({"text": text, "voice": voice, "lang_code": lang_code, "speed": speed}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with self._lock:
            self._inflight += 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                audio, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 422:
                raise Unsupported(detail) from e
            raise WorkerError(f"HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise WorkerError(f"{type(e).__name__}: {e}") from e
        finally:
            with self._lock:
                self._inflight -= 1
        if int(sr) != SAMPLE_RATE:
            raise WorkerError(f"worker returned {sr} Hz audio, expected {SAMPLE_RATE}")
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def generate(self, text: str, voice: str, lang_code: str, speed: float = 1.0) -> Optional[np.ndarray]:
        """Audio from MLX, or None when this request must be spoken by PyTorch Kokoro (the reason is logged once)."""
        try:
            audio = self.synthesize(text, voice, lang_code, speed)
        except Unsupported as e:
            self.last_engine, self.last_error = "torch", f"MLX cannot speak {voice}: {e}"
            log_once(f"unsupported:{lang_code}", "MLX Kokoro cannot speak language '%s' (%s); using PyTorch Kokoro for it",
                     lang_code, e)
            return None
        except WorkerError as e:
            self.last_engine, self.last_error = "torch", f"MLX worker failed: {e}"
            log_once("worker", "MLX Kokoro worker failed (%s); using PyTorch Kokoro", e)
            return None
        if audio.size == 0:
            self.last_engine, self.last_error = "torch", "MLX returned no audio"
            log_once("empty", "MLX Kokoro returned no audio; using PyTorch Kokoro")
            return None
        self.last_engine, self.last_error = "mlx", None
        return audio

    def torch_pipeline(self) -> Any:
        """The PyTorch American-English KPipeline, built the first time a request has to fall back to it."""
        with self._torch_lock:
            if self._torch is None:
                from services.espeak import configure_espeak
                configure_espeak()
                from kokoro import KPipeline
                log.info("Loading PyTorch Kokoro as the fallback voice")
                self._torch = KPipeline(lang_code="a")
            return self._torch
