"""Chatterbox Multilingual text-to-speech worker (Resemble AI, MIT): zero-shot voice cloning in 23 languages.

Runs in its own virtualenv (chatterbox-tts pins its own torch / numpy / transformers), started and owned by the API
server through services/chatterbox_clone.py. Only listens on 127.0.0.1. It runs on the CPU on purpose: on Apple's
GPU (MPS) it is slower and sometimes fails with "probability tensor contains inf/nan".

    POST /generate  {"text", "language": "hi", "ref_audio": "<wav path>", "exaggeration"?, "cfg_weight"?,
                     "temperature"?, "seed"?}
                    -> audio/wav (mono float32 at the model's rate, 24 kHz)
    GET  /health    -> {"ok": true, "model": "...", "device": "cpu", "languages": [...]}

    python chatterbox_worker.py --port 8767                 serve (weights must already be cached)
    python chatterbox_worker.py --download                  fetch the weights, then exit

Long text is split into sentences that are generated one by one and joined with short pauses. The package fixes the
generation length at 1000 speech tokens (40 s) and has no argument for it, so the worker wraps the model's token
generator to cap each sentence relative to its text length: a sentence that runs on (the model repeating itself
instead of stopping) is cut off, split in half and retried rather than returned as babble.
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

DEFAULT_MODEL = "ResembleAI/chatterbox"
WEIGHT_FILES = ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt",
                "Cangjie5_TC.json"]  # what ChatterboxMultilingualTTS.from_pretrained downloads
MAX_TEXT_CHARS = 2000
CHUNK_CHARS = 180  # longest single generation; longer sentences are split at a space
MIN_SENTENCE_CHARS = 12  # a shorter fragment is merged into the sentence before it
PAUSE_SECONDS = 0.25
TOKENS_PER_SECOND = 25  # the speech tokenizer runs at 25 Hz
MIN_CHARS_PER_SECOND = 5.0  # slower than this is not speech: a clip needing more time than that has run on
HARD_TOKEN_LIMIT = 1000  # the package's own fixed ceiling (40 s)
MAX_SPLIT_DEPTH = 2


def token_cap(text: str) -> int:
    """How many speech tokens a sentence of this text may use: room for slow speech, but not for a runaway."""
    return int(min(HARD_TOKEN_LIMIT, max(60, TOKENS_PER_SECOND * (len(text) / MIN_CHARS_PER_SECOND + 1.5))))


_SENTENCE_END = re.compile(r"(?<=[।॥.!?…])\s+|\n+")


def split_for_speech(text: str, max_chars: int = CHUNK_CHARS):
    """One chunk per sentence (Devanagari danda included), each at most max_chars."""
    text = re.sub(r"[ \t]+", " ", (text or "").strip())
    chunks = []
    for piece in (p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()):
        while len(piece) > max_chars:
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


def _halve(text: str):
    mid = len(text) // 2
    left, right = text.rfind(" ", 0, mid + 1), text.find(" ", mid)
    cut = min((c for c in (left, right) if c > 0), key=lambda c: abs(c - mid), default=mid)
    return text[:cut].strip(), text[cut:].strip()


def _clamp(value, lo, hi, default):
    try:
        return min(hi, max(lo, float(value)))
    except (TypeError, ValueError):
        return default


def _watch_parent() -> None:
    """Exit if the API server that started this worker dies (see parent_watch.py)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parent_watch import watch_parent
    watch_parent()


def snapshot_dir(model: str, local_only: bool):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model, repo_type="model", allow_patterns=WEIGHT_FILES,
                             token=os.getenv("HF_TOKEN"), local_files_only=local_only)


class Engine:
    """The loaded model. One generation at a time: the model keeps its reference conditioning as state."""

    def __init__(self, model_id: str):
        warnings.filterwarnings("ignore")
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        try:
            path = snapshot_dir(model_id, local_only=True)  # no network when the weights are already cached
        except Exception:  # noqa: BLE001  not cached yet: download
            path = snapshot_dir(model_id, local_only=False)
        self.model = ChatterboxMultilingualTTS.from_local(path, "cpu")
        self.languages = sorted(SUPPORTED_LANGUAGES)
        self.sr = int(self.model.sr)
        self.lock = threading.Lock()
        self._cap = HARD_TOKEN_LIMIT
        self._last_tokens = 0
        self._ref = None  # (path, mtime, exaggeration) the model is currently conditioned on
        orig = self.model.t3.inference

        def capped(*args, **kwargs):
            kwargs["max_new_tokens"] = self._cap
            out = orig(*args, **kwargs)
            self._last_tokens = int(out.shape[-1])
            return out

        self.model.t3.inference = capped

    def _condition(self, ref_audio: str, exaggeration: float) -> None:
        key = (ref_audio, os.path.getmtime(ref_audio), exaggeration)
        if self._ref != key:
            self.model.prepare_conditionals(ref_audio, exaggeration=exaggeration)
            self._ref = key

    def _speak(self, text, language, exaggeration, cfg_weight, temperature, depth=0):
        """One chunk, or its halves when it hit the token cap."""
        self._cap = token_cap(text)
        wav = self.model.generate(text, language_id=language, exaggeration=exaggeration, cfg_weight=cfg_weight,
                                  temperature=temperature)
        if self._last_tokens < self._cap:
            return [wav.squeeze().cpu().numpy().astype(np.float32)]
        if depth >= MAX_SPLIT_DEPTH or len(text) < 30:
            raise ValueError("could not speak this text cleanly (the model ran on); try rewording it")
        left, right = _halve(text)
        return (self._speak(left, language, exaggeration, cfg_weight, temperature, depth + 1)
                + self._speak(right, language, exaggeration, cfg_weight, temperature, depth + 1))

    def generate(self, text, language, ref_audio, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, seed=None):
        import torch
        language = (language or "en").lower()
        if language not in self.languages:
            raise ValueError(f"unsupported language '{language}'")
        if not ref_audio or not os.path.isfile(ref_audio):
            raise ValueError("ref_audio must be a readable wav file")
        chunks = split_for_speech(text)
        if not chunks:
            raise ValueError("there is no text to speak")
        with self.lock:
            if seed is not None:
                torch.manual_seed(int(seed))
            self._condition(ref_audio, exaggeration)
            pause = np.zeros(int(self.sr * PAUSE_SECONDS), dtype=np.float32)
            parts = []
            for chunk in chunks:
                for audio in self._speak(chunk, language, exaggeration, cfg_weight, temperature):
                    if parts:
                        parts.append(pause)
                    parts.append(audio)
        return np.concatenate(parts)


ENGINE = None
MODEL_NAME = DEFAULT_MODEL


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, json.dumps({"ok": True, "model": MODEL_NAME, "device": "cpu",
                                        "languages": ENGINE.languages}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/generate":
            return self._send(404, b'{"error":"not found"}')
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            text = str(req.get("text") or "")
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(f"text is longer than {MAX_TEXT_CHARS} characters")
            started = time.time()
            audio = ENGINE.generate(
                text, req.get("language") or "hi", req.get("ref_audio"),
                exaggeration=_clamp(req.get("exaggeration"), 0.25, 2.0, 0.5),
                cfg_weight=_clamp(req.get("cfg_weight"), 0.0, 1.0, 0.5),
                temperature=_clamp(req.get("temperature"), 0.05, 2.0, 0.8), seed=req.get("seed"))
            buf = io.BytesIO()
            sf.write(buf, audio, ENGINE.sr, format="WAV", subtype="FLOAT")
            print(f"[chatterbox] {len(text)} chars -> {len(audio) / ENGINE.sr:.1f}s audio in {time.time() - started:.1f}s",
                  flush=True)
            self._send(200, buf.getvalue(), "audio/wav")
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode())
        except Exception as e:  # noqa: BLE001
            print(f"[chatterbox] generation failed: {type(e).__name__}: {e}", flush=True)
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"[:300]}).encode())


def main() -> None:
    global ENGINE, MODEL_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--download", action="store_true", help="fetch the weights and exit")
    args = ap.parse_args()
    MODEL_NAME = args.model

    if args.download:
        try:
            path = snapshot_dir(args.model, local_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"Download failed: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Weights are in {path}")
        return

    _watch_parent()
    started = time.time()
    ENGINE = Engine(args.model)
    print(f"[chatterbox] loaded {args.model} on cpu in {time.time() - started:.1f}s", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
