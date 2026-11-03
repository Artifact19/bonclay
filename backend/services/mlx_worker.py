"""Qwen3-TTS voice-clone worker on Apple's MLX (Apple Silicon GPU).

Runs in its own virtualenv (mlx-audio needs transformers 5.x; the PyTorch qwen-tts package pins 4.x),
started and owned by the API server through services/mlx_clone.py. Only listens on 127.0.0.1.

    POST /generate  {"text", "ref_audio": "<wav path>", "ref_text": str|null, "language": "auto"}
                    -> audio/wav (mono float), header X-Sample-Rate
    POST /prepare   {"ref_audio": "<wav path>", "ref_text": str|null}  -> {"ok": true, "ms": ...}
                    encodes the reference ahead of the first generation (see install_reference_cache)
    GET  /health    -> {"ok": true, "model": "...", "ref_cache": n}

Generation settings come from the worker's environment (the API's per-request sampling settings are not forwarded):
    MLX_CLONE_STREAM            on (default) | off. on decodes the audio in streaming chunks (only the new codes, not the
                                reference's again) and joins them; the response is still one whole WAV. off = one decode.
    MLX_CLONE_STREAM_INTERVAL   seconds of audio per streamed chunk (default 0.32, which also gives the earliest first chunk)
    MLX_CLONE_TEMPERATURE       sampling temperature (default 0.5; mlx-audio's own default is 0.9)

Usage: python mlx_worker.py --port 8765 --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit
"""
import argparse
import os
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

LANG_NAMES = {"en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean", "de": "german",
              "fr": "french", "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian"}



def _watch_parent() -> None:
    """Exit if the API server that started this worker dies (see parent_watch.py)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parent_watch import watch_parent
    watch_parent()


def install_reference_cache(model, size: int = 8, evaluate=None) -> dict:
    """Memoise the two reference-only steps of a voice-clone generation on `model`, in place.

    Qwen3-TTS's in-context cloning encodes the reference clip through the speech tokenizer and extracts its speaker
    embedding on every request, although both depend only on the clip: for a 10 s reference that is roughly a tenth
    of a short generation, paid again for each sentence of a Studio render. Both results are kept per clip (keyed by
    a hash of the samples and the call arguments), `size` clips deep, least recently used out first. `/prepare` fills
    the cache before the first generation of a new voice. Returns the cache (an OrderedDict) for inspection.
    `evaluate` is called on each fresh result (mx.eval by default) so the work happens once, not lazily per use."""
    import hashlib
    from collections import OrderedDict

    cache: "OrderedDict[tuple, object]" = OrderedDict()
    size = max(1, int(size))
    if evaluate is None:
        try:
            import mlx.core as mx  # noqa: PLC0415
            evaluate = mx.eval
        except ImportError:  # pragma: no cover - the worker always has MLX; tests pass their own
            evaluate = lambda *_: None  # noqa: E731

    def key_of(name, array, args, kwargs):
        a = np.asarray(array)
        digest = hashlib.sha1(a.tobytes()).hexdigest()
        return (name, digest, tuple(a.shape), args, tuple(sorted(kwargs.items())))

    def memo(fn, name):
        def wrapped(array, *args, **kwargs):
            k = key_of(name, array, args, kwargs)
            hit = cache.get(k)
            if hit is not None:
                cache.move_to_end(k)
                return hit
            out = fn(array, *args, **kwargs)
            evaluate(out)
            cache[k] = out
            while len(cache) > size:
                cache.popitem(last=False)
            return out
        wrapped.__name__ = getattr(fn, "__name__", name)
        return wrapped

    tokenizer = getattr(model, "speech_tokenizer", None)
    if tokenizer is not None and callable(getattr(tokenizer, "encode", None)):
        tokenizer.encode = memo(tokenizer.encode, "codes")
    if callable(getattr(model, "extract_speaker_embedding", None)):
        model.extract_speaker_embedding = memo(model.extract_speaker_embedding, "speaker")
    return cache

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def generation_settings() -> dict:
    """The extra keyword arguments for `model.generate`, read from the environment on every call (see the module doc).

    Streaming decode took a sentence from 4.35 s to 3.3 s and the MLX peak from 7.3 GB to 3.2 GB in a 23 Oct trial (same
    tokens, same quality): the non-streaming path re-decodes the whole reference with every sentence. Temperature 0.5
    raised the ECAPA similarity to the reference by 0.07 on a synthetic voice (unconfirmed on real recordings).
    No `mx.clear_cache()` is added between streamed chunks (MLX issue 3450: segfault with a streaming decoder);
    mlx-audio clears the cache itself when a generation ends."""
    kw: dict = {"temperature": _env_float("MLX_CLONE_TEMPERATURE", 0.5)}
    if (os.getenv("MLX_CLONE_STREAM", "on") or "on").strip().lower() not in ("off", "0", "false", "no"):
        kw["stream"] = True
        kw["streaming_interval"] = _env_float("MLX_CLONE_STREAM_INTERVAL", 0.32)
    return kw


def synthesize(model, text: str, ref_audio: str, ref_text, lang: str):
    """One generation: the chunks `model.generate` yields, joined in order into one float32 array. -> (audio, rate)."""
    results = list(model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text or None, lang_code=lang,
                                  **generation_settings()))
    audio = np.concatenate([np.asarray(r.audio, dtype=np.float32) for r in results])
    return audio, int(results[0].sample_rate)


def main() -> None:
    _watch_parent()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
    args = ap.parse_args()

    from mlx_audio.tts.utils import load_model
    from parent_watch import limit_mlx_cache

    # Cap MLX's pool of freed GPU buffers (MLX_CACHE_LIMIT_MB, default 256). Generating audio of different lengths
    # allocates differently sized buffers, so without a cap the pool swelled to 5-7 GB during a clone session.
    limit_mlx_cache()
    started = time.time()
    model = load_model(args.model)
    ref_cache = install_reference_cache(model, int(os.getenv("MLX_REF_CACHE", "8") or 8))
    lock = threading.Lock()  # one generation at a time on the GPU
    print(f"[mlx-worker] {args.model} ready in {time.time() - started:.1f}s on port {args.port}", flush=True)

    def prepare(ref_audio_path: str, ref_text) -> None:
        """Run the reference-only steps once so the cache is warm for the first real generation of this voice."""
        from mlx_audio.utils import load_audio  # noqa: PLC0415
        ref = load_audio(ref_audio_path, sample_rate=model.sample_rate)
        if hasattr(model, "_prepare_icl_generation_inputs") and ref_text:
            model._prepare_icl_generation_inputs(text="नमस्ते", ref_audio=ref, ref_text=ref_text, language="auto")
        elif callable(getattr(model, "extract_speaker_embedding", None)):
            model.extract_speaker_embedding(ref)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the API server's log readable
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
                self._json(200, {"ok": True, "model": args.model, "ref_cache": len(ref_cache)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/prepare":
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    t0 = time.time()
                    with lock:
                        prepare(req["ref_audio"], req.get("ref_text") or None)
                    self._json(200, {"ok": True, "ms": int((time.time() - t0) * 1000), "ref_cache": len(ref_cache)})
                except Exception as e:  # noqa: BLE001
                    self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return
            if self.path != "/generate":
                self._json(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                lang = LANG_NAMES.get((req.get("language") or "auto").split("-")[0].lower(), "auto")
                with lock:
                    audio, sr = synthesize(model, req["text"], req["ref_audio"], req.get("ref_text"), lang)
                buf = io.BytesIO()
                sf.write(buf, audio, sr, format="WAV", subtype="FLOAT")
                data = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("X-Sample-Rate", str(sr))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
