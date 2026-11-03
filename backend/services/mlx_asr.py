"""Client for services/mlx_asr_worker.py: Whisper on the Apple GPU, started on first use and kept warm.

Used by the API's speech recognition when no Groq key is set (before falling back to CPU Whisper). Enabled on Apple
Silicon when backend/.venv-mlx has mlx-whisper and both models are downloaded (scripts/setup_mlx.sh does both).
Set ASR_BACKEND=whisper to force the CPU path, and MLX_ASR_STRATEGY=hybrid|accurate|auto|fast|single to choose how the two
models are used (ASR_STRATEGY is the older name and still works). The strategies, and the measurements behind the
default, are described in services/mlx_asr_worker.py and docs/RESULTS.md ("Speech recognition strategy").
"""
import glob
import json
import logging
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

log = logging.getLogger("bonclay.mlx_asr")

FAST = os.getenv("MLX_ASR_FAST_MODEL", "mlx-community/whisper-small-mlx")
ACCURATE = os.getenv("MLX_ASR_ACCURATE_MODEL", "mlx-community/whisper-large-v3-turbo")
_HERE = os.path.dirname(os.path.abspath(__file__))
STRATEGIES = ("hybrid", "accurate", "auto", "fast", "single")
# "single": one Whisper pass over a short encoder window (10 s, 15 s and so on up to 30 s, picked from the clip's length), with
# a fall back to the full 30 s window when the result looks unsure. Against "auto" on 40 synthetic clips and on phone-line
# copies of them (8 kHz band, 25 dB noise): Hindi speech recognition 1.28 s -> 0.48 s with +0.14 points error, English the
# same speed with 0.5 points fewer errors; in the live turn bench Hindi recognition went 1,162 -> 432 ms (docs/RESULTS.md,
# "Speech recognition on a shorter encoder window"). MLX_ASR_STRATEGY=auto restores the old behaviour.
DEFAULT_STRATEGY = "single"


def resolve_strategy() -> str:
    """MLX_ASR_STRATEGY (or the older ASR_STRATEGY), case-insensitive; an unknown or empty value means the default."""
    for name in ("MLX_ASR_STRATEGY", "ASR_STRATEGY"):
        value = os.getenv(name, "").strip().lower()
        if value in STRATEGIES:
            return value
        if value:
            log.warning("%s=%r is not one of %s; using %s", name, value, "|".join(STRATEGIES), DEFAULT_STRATEGY)
            break
    return DEFAULT_STRATEGY


def resident_roles(strategy: Optional[str] = None) -> tuple:
    """The Whisper models the worker loads and keeps: only what the strategy can reach ("accurate" for the default
    `single`; both for every other strategy). MLX_ASR_RESIDENT=both loads both whatever the strategy
    (the behaviour before 27 Oct 2026: the small model sat idle, about 0.5 GB, under the default strategy)."""
    if os.getenv("MLX_ASR_RESIDENT", "auto").strip().lower() == "both":
        return ("fast", "accurate")
    strategy = strategy or resolve_strategy()
    if strategy in ("single", "accurate"):
        return ("accurate",)
    return ("fast", "accurate")   # hybrid and auto by design; fast too, since an Indian-language hint runs the large model


def resident_repos(strategy: Optional[str] = None) -> tuple:
    return tuple(FAST if r == "fast" else ACCURATE for r in resident_roles(strategy))


def _mlx_python() -> Optional[str]:
    try:
        from services.mlx_clone import mlx_python
    except ImportError:
        from backend.services.mlx_clone import mlx_python
    return mlx_python()


def _cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


def unavailable_reason() -> Optional[str]:
    if os.getenv("ASR_BACKEND", "").lower() == "whisper":
        return "ASR_BACKEND=whisper"
    py = _mlx_python()
    if not py:
        return "no MLX environment (Apple Silicon only; run bash backend/scripts/setup_mlx.sh)"
    venv = os.path.dirname(os.path.dirname(py))
    if not glob.glob(os.path.join(venv, "lib", "python3*", "site-packages", "mlx_whisper")):
        return "mlx-whisper is not installed (run bash backend/scripts/setup_mlx.sh)"
    missing = [r for r in resident_repos() if not _cached(r)]
    if missing:
        return f"not downloaded: {', '.join(missing)} (run bash backend/scripts/setup_mlx.sh)"
    return None


class _Worker:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.last_used: float = 0.0   # time.time() of the latest ensure() or request; read by stop_if_idle

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure(self, timeout: float = 600.0) -> int:
        with self.lock:
            self.last_used = time.time()
            if self.running:
                return self.port
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                self.port = s.getsockname()[1]
            self.proc = subprocess.Popen(
                [_mlx_python(), os.path.join(_HERE, "mlx_asr_worker.py"), "--port", str(self.port),
                 "--fast", FAST, "--accurate", ACCURATE, "--load", ",".join(resident_roles())],
                env={**os.environ, "HF_HUB_OFFLINE": "1"})
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    self.proc = None
                    raise RuntimeError("MLX speech-recognition worker exited during start-up")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2):
                        return self.port
                except (urllib.error.URLError, ConnectionError, OSError):
                    time.sleep(0.5)
            self.stop()
            raise RuntimeError("MLX speech-recognition worker did not start in time")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


_worker = _Worker()


def transcribe(path: str, language: Optional[str] = None, timeout: float = 120.0, prompt: Optional[str] = None) -> dict:
    """Blocking. Returns {"text", "language", "model", "seconds"} (and "segments" with avg_logprob / no_speech_prob
    when the worker reports them); raises on failure. `prompt` is Whisper's initial_prompt (an agent's vocabulary)."""
    port = _worker.ensure()
    body = json.dumps({"path": path, "language": language, "strategy": resolve_strategy(),
                       **({"prompt": prompt} if prompt else {})}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/transcribe", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            _worker.last_used = time.time()
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"MLX ASR: {e.read().decode(errors='replace')[:300]}") from e


def status() -> dict:
    reason = unavailable_reason()
    return {"available": reason is None, "reason": reason, "running": _worker.running, "fast": FAST,
            "accurate": ACCURATE, "strategy": resolve_strategy(), "resident": list(resident_roles())}


def stop_if_idle(idle_s: float, now: Optional[float] = None) -> bool:
    """Stop the worker (it frees its Whisper models, about 1.6 GB) when it has not been asked for `idle_s` seconds.
    True when it was stopped. It starts again, and loads again, on the next request. idle_s <= 0: never."""
    if idle_s <= 0 or not _worker.running:
        return False
    if (time.time() if now is None else now) - _worker.last_used < idle_s:
        return False
    with _worker.lock:
        _worker.stop()
    log.info("MLX speech recognition worker stopped after %.0f s idle", idle_s)
    return True


def warm() -> None:
    if unavailable_reason() is None:
        _worker.ensure()
