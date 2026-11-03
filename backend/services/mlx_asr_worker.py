"""Speech recognition on the Apple GPU with mlx-whisper, run in backend/.venv-mlx and owned by the API server.

Strategies (see transcribe_plan; measured in docs/RESULTS.md, "Speech recognition strategy"):
  hybrid    a small, fast Whisper transcribes first. If it hears an Indian language, a larger Whisper re-transcribes the
            clip, because the small model's Hindi is noticeably worse while its English is fine.
  accurate  one pass of the large model, language auto-detected by it.
  auto      the small model only detects the language (one encoder pass, no text decoding); an Indian language goes to a
            single large-model pass, anything else to the small model.
  fast      the small model only.
  single    (opt-in, MLX_ASR_STRATEGY=single) one large-model pass on an encoder window of 10/15/20/30 s sized to the clip,
            language detected in the same pass, the small model not used (services/asr_window.py). A clip too long for
            the largest window, or a result that looks wrong, goes to the 30 s window / the `accurate` path.
A language hint that is an Indian language always means one large-model pass, whatever the strategy.

    POST /transcribe {"path": "<audio file>", "language": "hi"|null, "strategy": "hybrid"|"accurate"|"auto"|"fast"}
                     -> {"text", "language", "model", "seconds"}
    GET  /health     -> {"ok": true}
"""
import argparse
import gc
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

INDIAN = {"hi", "mr", "bn", "gu", "kn", "ml", "or", "pa", "ta", "te", "as", "ur", "ne", "sa", "sd"}


def _watch_parent() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parent_watch import watch_parent
    watch_parent()


STRATEGIES = ("hybrid", "accurate", "auto", "fast", "single")


def _num(v):
    """A JSON-safe float (MLX / numpy scalars included), or None."""
    try:
        return None if v is None else float(v.item() if hasattr(v, "item") else v)
    except (TypeError, ValueError):
        return None


def transcribe_plan(strategy: str, language, path, run, detect, fast: str, accurate: str, window=None):
    """Decide which model(s) transcribe `path` and run them. Returns (result dict, model repo used).

    run(path, repo, language) -> {"text", "language", ...}; detect(path, repo) -> language code. `path` is passed through
    untouched (a file path, or a waveform the caller already loaded). Both callables are injected so this logic is
    testable without MLX. window(path, repo, language) -> result dict, or None when the clip does not fit a window; only
    the `single` strategy uses it."""
    if strategy not in STRATEGIES:
        strategy = "hybrid"
    if strategy == "single":
        out = window(path, accurate, language) if window is not None else None
        return (out if out is not None else run(path, accurate, language)), accurate
    if language in INDIAN or strategy == "accurate":
        return run(path, accurate, language), accurate
    if strategy == "auto" and not language:
        detected = detect(path, fast)
        if detected in INDIAN:
            return run(path, accurate, detected), accurate
        return run(path, fast, detected), fast
    out = run(path, fast, language)
    if strategy == "hybrid" and not language and out.get("language") in INDIAN:
        return run(path, accurate, out["language"]), accurate
    return out, fast


def roles_for_strategy(strategy: str) -> tuple:
    """Which Whisper models a strategy can reach: ("accurate",) or ("fast", "accurate"). `single` (the default) and
    `accurate` only ever run the large model. Every other strategy can reach both: `hybrid` and `auto` by design, `fast`
    because an Indian-language hint always means one large-model pass whatever the strategy (so a worker holding only the
    small model would swap models on every Hindi request). Kept in step with transcribe_plan (a test checks it)."""
    if strategy in ("single", "accurate"):
        return ("accurate",)
    return ("fast", "accurate")


def keep_models_resident(max_resident=None) -> None:
    """Make mlx_whisper keep every model it has loaded, not just the last one. `max_resident` (None: no limit) caps how
    many stay in memory: loading one more drops the oldest, so a worker that only needs one size never holds two.

    mlx_whisper holds a single model: asking for a different one (the hybrid strategy does this for every Indian-language
    clip, small model first and then the large one, and back again for the next English clip) reloads weights from disk and
    leaves the old ones in MLX's buffer pool. That made this worker's memory climb and every Hindi request pay a model
    load. Two models fit comfortably (about 2 GB together), so both stay in memory."""
    import importlib
    from mlx_whisper.load_models import load_model
    holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
    models = {}

    def get_model(cls, model_path, dtype):
        key = (model_path, str(dtype))
        if key not in models:
            while max_resident and len(models) >= max_resident:
                models.pop(next(iter(models)))   # oldest first; the weights are freed once nothing runs on them
            models[key] = load_model(model_path, dtype=dtype)
        return models[key]

    holder.get_model = classmethod(get_model)


def main() -> None:
    _watch_parent()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--fast", default="mlx-community/whisper-small-mlx")
    ap.add_argument("--accurate", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--load", default="fast,accurate",
                    help="models to load at start and keep resident: fast, accurate or fast,accurate. A model outside "
                         "this list is still loaded when a request needs it, replacing the oldest one")
    args = ap.parse_args()
    roles = [r for r in args.load.split(",") if r in ("fast", "accurate")] or ["fast", "accurate"]

    import mlx.core as mx
    import mlx_whisper
    from parent_watch import limit_mlx_cache

    limit_mlx_cache()          # cap MLX's pool of freed GPU buffers (MLX_CACHE_LIMIT_MB, default 256)
    keep_models_resident(max_resident=len(roles))
    lock = threading.Lock()  # one transcription at a time on the GPU

    state = {"prompt": None}   # the current request's initial_prompt (requests run one at a time, under `lock`)

    def run(path: str, repo: str, language):
        extra = {"initial_prompt": state["prompt"]} if state["prompt"] else {}
        return mlx_whisper.transcribe(path, path_or_hf_repo=repo, language=language, fp16=True, **extra)

    def detect(audio, repo: str) -> str:
        """Language id only: one encoder pass over the first 30 s and one decoder step, no text is decoded.
        `audio` is a path or an already-loaded waveform."""
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.transcribe import ModelHolder
        model = ModelHolder.get_model(repo, mx.float16)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        _, probs = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
        return max(probs, key=probs.get)

    def window(audio, repo: str, language):
        """One pass on a short encoder window; None when the clip is too long for one (the caller then uses `run`)."""
        import asr_window
        if len(audio) > (asr_window.BUCKETS[-1] - asr_window.MARGIN_S) * asr_window.SAMPLE_RATE:
            return None
        return asr_window.transcribe_window(repo, audio, language, prompt=state["prompt"])

    # Load the models the strategy uses now so the first request is not slow (the API passes --load).
    started = time.time()
    silence = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mlx_asr_warmup.wav")
    try:
        import numpy as np
        import soundfile as sf
        sf.write(silence, np.zeros(16000, dtype=np.float32), 16000)
        for role in roles:
            run(silence, args.fast if role == "fast" else args.accurate, "en")
        if "accurate" in roles:
            import asr_window
            asr_window.transcribe_window(args.accurate, np.zeros(16000, dtype=np.float32), None)  # compile the short windows
    finally:
        if os.path.exists(silence):
            os.remove(silence)
    print(f"[mlx-asr] ready in {time.time() - started:.1f}s on port {args.port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._json(200 if self.path == "/health" else 404, {"ok": self.path == "/health"})

        def do_POST(self):
            if self.path != "/transcribe":
                self._json(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                language = req.get("language") or None
                strategy = req.get("strategy") or "auto"
                t = time.time()
                with lock:
                    state["prompt"] = req.get("prompt") or None
                    audio = req["path"]
                    if strategy == "single" or (strategy == "auto" and not language):
                        from mlx_whisper.audio import load_audio
                        audio = load_audio(audio)  # decode the file once, not once per model
                    out, model = transcribe_plan(strategy, language, audio, run, detect, args.fast, args.accurate, window)
                result = {"text": (out.get("text") or "").strip(), "language": out.get("language") or language,
                          "model": model, "seconds": round(time.time() - t, 2)}
                segs = [{k: (str(s.get(k) or "") if k == "text" else _num(s.get(k)))
                         for k in ("start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio")}
                        for s in (out.get("segments") or []) if isinstance(s, dict)]
                if segs:   # per-segment confidence for the turn (services/query_understanding.asr_confidence)
                    result["segments"] = segs
                del out
                gc.collect()
                mx.clear_cache()  # hand this request's scratch buffers back to the OS
                self._json(200, result)
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
