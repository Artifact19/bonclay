"""Kokoro-82M speech on Apple's MLX (Apple Silicon GPU): a small worker the API server starts and owns.

Why its own process, and not an endpoint of mlx_worker.py (the Qwen3 voice-clone worker): Kokoro is the live voice of
every conversation turn and is kept warm, while the clone worker holds the much larger Qwen3-TTS model and is only
loaded when somebody clones a voice. One process for both would load Qwen3 (~2 GB) for every Kokoro turn, and a crash
or a long clone job would stall the live voice. The Kokoro worker adds about 0.8 GB of process memory and 1.7 GB of
MLX memory (measured), and exits with the API server (parent_watch) with a bounded MLX buffer cache.

Runs in backend/.venv-mlx (mlx-audio + misaki + the spaCy English model + espeak-ng from Homebrew).
Only listens on 127.0.0.1.

    POST /generate  {"text", "voice": "af_heart", "lang_code": "a", "speed": 1.0}
                    -> audio/wav, mono float32, 24 kHz (header X-Sample-Rate)
                    a language this worker cannot speak (a missing optional package) answers 422 with
                    {"unsupported": true, "error": ...} so the API server can use PyTorch Kokoro for that request.
    GET  /health    -> {"ok": true, "model": "...", "loaded": ["a", "h"], "mlx_mb": {"active", "cache", "peak"}}

Usage: python mlx_kokoro_worker.py --port 8766 --model mlx-community/Kokoro-82M-bf16
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
SAMPLE_RATE = 24000
LANG_CODES = set("abefhijpz")  # Kokoro's single-letter language codes (the first letter of every voice id)


def _setup_path() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def synthesize(model, pipelines: dict, text: str, voice: str, lang_code: str, speed: float) -> np.ndarray:
    """Text -> 24 kHz mono float32, through one cached KokoroPipeline per language.

    Calls the pipeline directly instead of model.generate(): that method empties the pipeline's voice cache on every
    call, which re-reads the voice file for each sentence (measured: it is a large share of a short sentence's time)."""
    if lang_code not in pipelines:
        pipelines[lang_code] = model._get_pipeline(lang_code)
    pipe = pipelines[lang_code]
    parts = []
    for result in pipe(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        audio = result.audio
        if audio is not None:
            parts.append(np.array(audio, dtype=np.float32).reshape(-1))  # np.array forces the MLX evaluation
    if not parts:
        raise RuntimeError("Kokoro produced no audio")
    return np.concatenate(parts)


def _mlx_memory_mb() -> dict:
    """MLX's own accounting (active buffers, the cache of freed buffers, the peak), in MB; empty if unavailable."""
    try:
        import mlx.core as mx
        return {"active": round(mx.get_active_memory() / 2 ** 20), "cache": round(mx.get_cache_memory() / 2 ** 20),
                "peak": round(mx.get_peak_memory() / 2 ** 20)}
    except Exception:  # noqa: BLE001
        return {}


def main() -> None:
    _setup_path()
    from parent_watch import limit_mlx_cache, watch_parent
    watch_parent()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    from espeak import configure_espeak  # services/espeak.py: Homebrew's espeak-ng, the bundled one crashes on macOS
    configure_espeak()
    from mlx_audio.tts.utils import load_model

    limit_mlx_cache()  # cap MLX's pool of freed GPU buffers (MLX_CACHE_LIMIT_MB, default 256)
    started = time.time()
    model = load_model(args.model)
    lock = threading.Lock()  # one generation at a time on the GPU
    pipelines: dict = {}
    print(f"[mlx-kokoro] {args.model} ready in {time.time() - started:.1f}s on port {args.port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True, "model": args.model, "loaded": sorted(pipelines), "mlx_mb": _mlx_memory_mb()})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                self._json(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                text, voice = str(req["text"]), str(req.get("voice") or "af_heart")
                code = str(req.get("lang_code") or voice[:1]).lower()
                speed = float(req.get("speed") or 1.0)
                if code not in LANG_CODES:
                    self._json(422, {"unsupported": True, "error": f"unknown Kokoro language code '{code}'"})
                    return
                try:
                    with lock:
                        audio = synthesize(model, pipelines, text, voice, code, speed)
                except ImportError as e:  # a language whose optional packages are not installed in this venv
                    pipelines.pop(code, None)
                    self._json(422, {"unsupported": True, "error": f"{type(e).__name__}: {e}"[:300]})
                    return
                buf = io.BytesIO()
                sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="FLOAT")
                data = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("X-Sample-Rate", str(SAMPLE_RATE))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"[:500]})

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
