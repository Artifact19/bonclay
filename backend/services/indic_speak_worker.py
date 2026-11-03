"""Indic-Speak text-to-speech worker (Bodhan AI / AI4Bharat): preset voices in 22 Indian languages and English.

Runs in its own virtualenv (the model needs transformers 5.x; the rest of this backend pins 4.x), started and
owned by the API server through services/indic_speak.py. Only listens on 127.0.0.1.

    POST /generate  {"text", "speaker", "style"?, "temperature"?, "top_p"?, "top_k"?, "max_new_tokens"?, "seed"?}
                    -> audio/wav (mono float32, 24 kHz); header X-Truncated: 1 when the audio hit the token cap
    GET  /health    -> {"ok": true, "model": "...", "device": "..."}

    python indic_speak_worker.py --port 8766                 serve (offline: weights must already be cached)
    python indic_speak_worker.py --download                  fetch the weights + SNAC quantizer, then exit
                                                             (needs HF_TOKEN and accepted terms on the model page)

The model's own inference code (inference.py, vocos/) is imported from the downloaded snapshot rather than
copied into this repository: it ships under the Indic Open Model License, which travels with the weights.
"""
import argparse
import io
import json
import os
import re
import sys
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

DEFAULT_MODEL = "bodhan-ai/indic-speak"
SNAC_REPO = "hubertsiuzdak/snac_24khz"
SAMPLE_RATE = 24000
MAX_TEXT_CHARS = 600

# The model's control tokens look like <|speaker>, <speaker|>, <|eot_id|>. The tokenizer turns such text into the
# real control token, so user text must not be able to contain them. services/indic_speak.py keeps an identical copy.
CONTROL_TOKEN_RE = re.compile(r"<\|[^<>|]{0,48}\|?>|<[^<>|]{1,48}\|>")


def clean(s: str) -> str:
    """Remove control-token look-alikes, repeating until stable so '<<|eot_id|>|eot_id|>' cannot rebuild one."""
    prev = None
    while prev != s:
        prev, s = s, CONTROL_TOKEN_RE.sub("", s or "")
    return s.strip()


def _clamp(value, lo, hi, default):
    try:
        return min(hi, max(lo, float(value)))
    except (TypeError, ValueError):
        return default


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download(model: str) -> int:
    """Fetch everything the worker needs at request time, so serving can run fully offline."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        path = snapshot_download(model, token=os.getenv("HF_TOKEN") or None, ignore_patterns=["*.png", ".gitattributes"])
    except GatedRepoError:
        print(f"[indic-speak] Access to {model} is gated. Open https://huggingface.co/{model} while signed in, "
              "request access (it is approved automatically), then run this again with HF_TOKEN set.", file=sys.stderr)
        return 2
    print(f"[indic-speak] weights ready: {path}", flush=True)
    from snac import SNAC
    SNAC.from_pretrained(SNAC_REPO)
    print("[indic-speak] SNAC quantizer ready", flush=True)
    return 0


# What serving needs from the downloaded snapshot (the rest of the repo is docs and images).
REQUIRED_FILES = ("config.json", "tokenizer.json", "model.safetensors", "inference.py", "vocos/best.pt", "vocos/load.py")


def snapshot_dir(model: str) -> str:
    """The cached snapshot folder, checked for the files serving needs.

    snapshot_download(local_files_only=True) is not used: newer huggingface_hub versions insist that every file of
    the repo is cached, and the download skips images and .gitattributes on purpose."""
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(model, "config.json")
    if not isinstance(config, str):
        raise SystemExit(f"[indic-speak] {model} is not downloaded. Run: bash backend/scripts/setup_indic_speak.sh")
    snapshot = os.path.dirname(config)
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(snapshot, f))]
    if missing:
        raise SystemExit(f"[indic-speak] the downloaded copy of {model} is incomplete (missing {', '.join(missing)}). "
                         "Run: bash backend/scripts/setup_indic_speak.sh")
    return snapshot


def load_tts(model: str, device: str, mlx_model: str = ""):
    # Never download at request time; `--download` (scripts/setup_indic_speak.sh) does that once.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops Apple's GPU lacks run on the CPU instead of failing
    import torch

    # The model's loader asks for torch.load(weights_only=False), i.e. arbitrary pickle execution. Its checkpoints
    # contain only tensors and plain values, so force the restricted unpickler for every load in this process.
    real_load = torch.load
    torch.load = lambda *a, **k: real_load(*a, **{**k, "weights_only": True})

    snapshot = snapshot_dir(model)
    sys.path.insert(0, snapshot)
    if mlx_model:  # 4-bit language model on the Apple GPU (services/indic_speak_quant.py); same interface as TTS
        from indic_speak_quant import QuantTTS
        return QuantTTS(snapshot, mlx_model)
    from inference import TTS  # the model's own code, from the snapshot

    return TTS(snapshot, device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--mlx-model", default="", help="folder with the MLX 4-bit language model (see indic_speak_quant.py)")
    ap.add_argument("--download", action="store_true", help="fetch the weights and exit")
    args = ap.parse_args()

    if args.download:
        return download(args.model)

    # This process holds ~7.5 GB (about 3 GB with --mlx-model). If the API server that started it dies without cleaning up (a crash, kill -9),
    # exit instead of lingering as an orphan.
    parent = os.getppid()

    def watch_parent() -> None:
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                print("[indic-speak-worker] the API server is gone; exiting", flush=True)
                os._exit(0)

    threading.Thread(target=watch_parent, daemon=True).start()

    started = time.time()
    device = "mlx" if args.mlx_model else pick_device(args.device)
    tts = load_tts(args.model, device, args.mlx_model)
    lock = threading.Lock()  # one generation at a time on the accelerator
    print(f"[indic-speak-worker] {args.model} ready in {time.time() - started:.1f}s on {device}, port {args.port}",
          flush=True)

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
                self._json(200, {"ok": True, "model": args.model, "device": device})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                self._json(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                text, speaker, style = clean(req.get("text", "")), clean(req.get("speaker", "")), clean(req.get("style", ""))
                if not text:
                    self._json(400, {"error": "text is empty"})
                    return
                if len(text) > MAX_TEXT_CHARS:
                    self._json(400, {"error": f"text is longer than {MAX_TEXT_CHARS} characters"})
                    return
                seed = req.get("seed")
                started_gen = time.time()
                with lock, warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    wav = tts(text, speaker=speaker, style=style,
                              temperature=_clamp(req.get("temperature"), 0.1, 1.2, 0.6),
                              top_p=_clamp(req.get("top_p"), 0.1, 1.0, 0.9),
                              top_k=int(_clamp(req.get("top_k"), 1, 200, 50)),
                              max_new_tokens=int(_clamp(req.get("max_new_tokens"), 50, 2520, 2520)),
                              seed=int(seed) if seed is not None else None)
                truncated = any("max_new_tokens" in str(w.message) for w in caught)
                elapsed = time.time() - started_gen
                secs = len(wav) / SAMPLE_RATE
                print(f"[indic-speak-worker] {speaker or '-'}: {len(text)} chars -> {secs:.1f}s audio in {elapsed:.1f}s "
                      f"({elapsed / max(secs, 1e-6):.1f}x real time){' TRUNCATED' if truncated else ''}", flush=True)
                buf = io.BytesIO()
                sf.write(buf, np.asarray(wav, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="FLOAT")
                data = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("X-Sample-Rate", str(SAMPLE_RATE))
                self.send_header("X-Truncated", "1" if truncated else "0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
