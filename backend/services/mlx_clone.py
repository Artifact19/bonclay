"""Qwen3-TTS voice cloning on Apple Silicon through MLX, behind the same interface as qwen_tts.Qwen3TTSModel.

On a Mac, PyTorch runs Qwen3-TTS on the CPU (~10x slower than real time) and its MPS path is slower
still, while MLX runs it on the GPU in well under real time. MLX lives in its own virtualenv
(mlx-audio needs transformers 5.x, qwen-tts pins 4.x), so this module starts services/mlx_worker.py
in that environment and talks to it over 127.0.0.1.

Enabled when the machine is macOS on arm64 and an MLX python is found: $MLX_PYTHON, else
backend/.venv-mlx/bin/python (created by scripts/setup_mlx.sh). Set CLONE_BACKEND=torch to opt out.
"""
import hashlib
import io
import json
import logging
import os
import platform
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf

log = logging.getLogger("bonclay.mlx")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 8-bit weights (affine, group size 64): about the same similarity to the reference as bf16, ~1.1 s faster per sentence
# on an M4 and 0.3 GB smaller on disk (23 Oct trial). MLX_CLONE_MODEL=mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 goes back.
DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
PROMPT_BACKEND = "mlx-qwen3"


def mlx_python() -> Optional[str]:
    """The python of the MLX virtualenv, or None when MLX cloning is unavailable on this machine."""
    if os.getenv("CLONE_BACKEND", "").lower() == "torch":
        return None
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    for path in (os.getenv("MLX_PYTHON"), os.path.join(BACKEND_DIR, ".venv-mlx", "bin", "python")):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MlxQwen3Clone:
    """Drop-in for the subset of Qwen3TTSModel the API server uses for cloning."""

    # Reference clips are decoded at the model's own rate (backend._create_voice_profile reads this). 16 kHz, the
    # default for the other engines, is resampled up again inside the model and measured 0.06 worse in speaker
    # similarity on one synthetic voice. Prompts stored at 16 kHz before this change still work: the worker resamples
    # whatever WAV it is given.
    prompt_sample_rate = 24000

    def __init__(self, python: str, model: str = DEFAULT_MODEL, start_timeout: float = 900.0):
        self.model_name = os.getenv("MLX_CLONE_MODEL", model)
        self.port = _free_port()
        self._ref_dir = tempfile.mkdtemp(prefix="bonclay_mlx_refs_")
        worker = os.path.join(BACKEND_DIR, "services", "mlx_worker.py")
        self.proc = subprocess.Popen([python, worker, "--port", str(self.port), "--model", self.model_name])
        deadline = time.time() + start_timeout  # first run downloads the weights
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"MLX worker exited with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if json.load(r).get("ok"):
                        log.info("MLX clone worker ready: %s", self.model_name)
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(1)
        self.close()
        raise RuntimeError("MLX worker did not become ready in time")

    # -- Qwen3TTSModel-compatible surface --------------------------------------------------------

    def create_voice_clone_prompt(self, ref_audio: List[Tuple[np.ndarray, int]], ref_text: Optional[str] = None,
                                  x_vector_only_mode: bool = False) -> dict:
        audio, sr = ref_audio[0]
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), int(sr), format="WAV")
        prompt = {"backend": PROMPT_BACKEND, "ref_wav": buf.getvalue(),
                  "ref_text": None if x_vector_only_mode else (ref_text or None)}
        self.prepare(prompt)   # the worker encodes the reference now, so the first generation does not
        return prompt

    def prepare(self, voice_clone_prompt: Any, wait: bool = False) -> None:
        """Ask the worker to encode this voice's reference ahead of its first generation (POST /prepare; the worker
        keeps the result in its reference cache). In the background unless `wait`; never raises."""
        p = voice_clone_prompt or {}
        if p.get("backend") != PROMPT_BACKEND or not p.get("ref_wav"):
            return
        body = json.dumps({"ref_audio": self._ref_path(p["ref_wav"]), "ref_text": p.get("ref_text")}).encode()

        def post() -> None:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/prepare", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    out = json.load(r)
                    log.info("MLX clone worker prepared a reference in %s ms", out.get("ms"))
            except Exception as e:  # noqa: BLE001 - a warm cache is a bonus; the generation still works without it
                log.debug("MLX clone worker /prepare failed: %s", e)

        if wait:
            post()
        else:
            import threading  # noqa: PLC0415
            threading.Thread(target=post, name="mlx-prepare", daemon=True).start()

    def generate_voice_clone(self, text: str, language: str = "Auto", voice_clone_prompt: Any = None,
                             **_unused) -> Tuple[List[np.ndarray], int]:
        """One sentence in the prompt's voice. Sampling settings from the caller (`temperature`, `top_p`, ... from the
        API) are dropped on purpose: the worker's environment rules (MLX_CLONE_TEMPERATURE, default 0.5;
        MLX_CLONE_STREAM and MLX_CLONE_STREAM_INTERVAL, see services/mlx_worker.py), because those are the values
        that were measured."""
        p = voice_clone_prompt or {}
        if p.get("backend") != PROMPT_BACKEND:
            raise ValueError("Voice prompt was not built for the MLX clone engine")
        body = json.dumps({"text": text, "ref_audio": self._ref_path(p["ref_wav"]),
                           "ref_text": p.get("ref_text"), "language": language}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                audio, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"MLX worker: {e.read().decode(errors='replace')[:300]}") from e
        return [audio], int(sr)

    def eval(self):  # parity with torch modules (the manager calls _set_eval)
        return self

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    __del__ = close

    def _ref_path(self, wav: bytes) -> str:
        """Reference audio as a file the worker can read; cached by content hash."""
        path = os.path.join(self._ref_dir, hashlib.sha1(wav).hexdigest() + ".wav")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(wav)
        return path


def is_mlx_prompt(prompt: Any) -> bool:
    return isinstance(prompt, dict) and prompt.get("backend") == PROMPT_BACKEND
