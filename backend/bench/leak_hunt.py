#!/usr/bin/env python3
"""Short soak of the API process itself, to find what grows: the real app and routes in this process (FastAPI TestClient),
the slow or external parts faked (language model, speech recognition, Kokoro), everything else real (conversation and
turn storage, retrieval with the real embedding model, AudioSeal watermark, receipts, performance fields).

    HF_HUB_OFFLINE=1 backend/.venv/bin/python backend/bench/leak_hunt.py --minutes 12
    HF_HUB_OFFLINE=1 backend/.venv/bin/python backend/bench/leak_hunt.py --minutes 12 --tracemalloc
    backend/.venv/bin/python backend/bench/leak_hunt.py --minutes 2 --fake-embedder --no-watermark     # quick check

One request at a time, weighted mix (spoken turn blocking 30, spoken turn streaming 30, Kokoro generate 25,
transcribe 15), with unique questions so per-text caches fill the way real traffic fills them. Every --sample-every seconds
(default 30) it records this process's RSS and macOS physical footprint (before and after gc.collect()) and the number of
requests made. The memory slope is a linear fit over the samples after the first --warmup minutes.

With --tracemalloc it takes a snapshot after the warm-up and another at the end and prints the allocation sites that grew
the most (by traceback), and the Python object types that grew the most. tracemalloc slows the run about 3x and adds its own
memory: use it to find the cause, not to measure the slope.

Raw results go to backend/bench/results/leak_hunt-<stamp>.json (git-ignored).
"""
import argparse
import collections
import datetime as dt
import gc
import io
import json
import os
import random
import sys
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path

os.environ["BONCLAY_SKIP_DOTENV"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "leak_hunt"   # the app then does not open its phone-only public port (8010) or the heartbeat
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WARM_ON_START"] = "off"
os.environ["MEMORY_GUARD"] = "off"
os.environ["KOKORO_ENGINE"] = "torch"
os.environ["ASR_BACKEND"] = "whisper"
os.environ["LLM_PROVIDERS"] = "mlx"
os.environ.setdefault("RERANKER", "off")
os.environ.setdefault("M3_MODES", "off")
os.environ.setdefault("FACT_SHEET", "off")
os.environ.setdefault("CALLER_MEMORY_LLM", "off")

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
sys.path.insert(0, HERE)
RESULTS_DIR = Path(HERE) / "results"
LOCAL = {"storage-mode": "local"}
MB = 2 ** 20

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import soundfile as sf  # noqa: E402

from model_memory import footprint_mb  # noqa: E402

QUESTIONS = ["What time does the clinic open on {d}?", "How much is a dentist consultation, {n} people?",
             "Is the clinic open on {d}?", "Can I book an appointment for {d} at {h}?", "Where do I park my car, {n} cars?",
             "क्लिनिक का फोन नंबर क्या है, {n}?", "अपॉइंटमेंट कैसे बुक करूं {d}?", "Do you take insurance number {n}?"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
REPLY = ["The clinic opens at nine in the morning. ", "It closes at eight in the evening. ", "Please arrive ten minutes early."]
DOC = ("The clinic is open from nine in the morning until eight in the evening, Monday to Saturday. Sunday is closed. "
       "A dentist consultation costs five hundred rupees. Parking is free for patients behind the building. "
       "Appointments can be booked by phone or at the front desk. We accept most insurance plans. ") * 12


def complete_bge_snapshot():
    """The hub cache can hold BGE-M3's weights in one snapshot and its tokenizer in another; offline, the name then does not
    load. The snapshot that has the tokenizer is the same model by path (None when there is none)."""
    try:
        import glob
        from huggingface_hub import constants
        for d in glob.glob(os.path.join(constants.HF_HUB_CACHE, "models--BAAI--bge-m3", "snapshots", "*")):
            if os.path.exists(os.path.join(d, "tokenizer.json")):
                return d
    except Exception:  # noqa: BLE001
        pass
    return None


def wav_bytes(seconds=2.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


class Rig:
    def __init__(self, args):
        import backend
        from fastapi.testclient import TestClient
        from services import llm
        from services.local_store import LocalStore
        self.b = backend
        self.tmp = tempfile.mkdtemp(prefix="bonclay_leak_")
        os.environ["EMBEDDING_FP16_DIR"] = os.path.join(self.tmp, "fp16")
        backend.local_db = LocalStore(self.tmp)
        backend.LOCAL_AUDIO_DIR = self.tmp
        backend.LOCAL_VOICES_FILE = os.path.join(self.tmp, "voices.json")
        if args.fake_embedder:
            import hashlib

            class Fake:
                loaded = True

                def encode(self, texts, *a, **k):
                    out = []
                    for t in texts:
                        v = np.zeros(64, dtype=np.float32)
                        for w in t.lower().split():
                            v[int(hashlib.md5(w.strip(".,?!").encode()).hexdigest(), 16) % 64] += 1.0
                        out.append(v)
                    return np.stack(out)
            backend.embedding_model = Fake()
        self.n = 0
        self.asr_text = QUESTIONS[0]

        async def fake_asr(audio_bytes, name, language=None):
            return self.asr_text

        async def fake_local_transcribe(audio_bytes, ext, language=None, prompt=None):
            return {"text": self.asr_text, "language": "en", "model": "fake", "segments": None}

        async def fake_load(key):
            return object()

        def fake_kokoro(model, text, lang="en"):
            return np.random.default_rng(len(text)).normal(0, 0.05, 24000 * 2).astype(np.float32)

        class FakeStream:
            def __init__(self, deltas):
                self.deltas, self.info, self.result, self.closed = deltas, {}, None, False

            def __iter__(self):
                text = ""
                for d in self.deltas:
                    text += d
                    yield d
                self.result = llm.StreamResult(text, "fake", "fake-model", 7, {}, 3)

            def close(self):
                self.closed = True

        backend._DISCLOSURE_CLIPS.clear()
        backend._speech_to_text = fake_asr
        backend._local_transcribe = fake_local_transcribe
        backend._kokoro_generate = fake_kokoro
        backend.advanced_manager.load_model = fake_load
        llm.chat_stream = lambda messages, **kw: FakeStream(REPLY)
        persona = json.dumps({"name": "Ria", "greeting": "Hi, I'm Ria.", "system_instructions": "a friendly clinic assistant",
                              "suggested_languages": ["en", "hi"]})
        llm.chat = lambda messages, json_mode=False, **kw: llm.ChatResult(
            persona if json_mode else " ".join(REPLY) + f" ({self.n})", "fake", "fake-model", 5, {})
        if hasattr(backend.limiter, "reset"):
            backend.limiter.limit = lambda *a, **k: (lambda fn: fn)
        self.client = TestClient(backend.app, raise_server_exceptions=False)
        self.client.__enter__()
        if args.no_watermark:
            from services import provenance
            provenance.watermark = lambda w, sr, bits: w
            provenance.watermark_ex = lambda w, sr, bits: (w, {})
        self.wav = wav_bytes()
        r = self.client.post("/api/agents", headers=LOCAL, json={"name": "Clinic", "role_description": "clinic receptionist",
                                                                  "disclose_ai": False})
        assert r.status_code == 200, r.text
        self.agent = r.json()["id"]
        r = self.client.post("/api/knowledge/upload", headers=LOCAL, data={"agent_id": self.agent},
                             files={"file": ("clinic.txt", DOC.encode(), "text/plain")})
        assert r.status_code == 200, r.text

    def question(self):
        self.n += 1
        self.asr_text = random.choice(QUESTIONS).format(d=random.choice(DAYS), n=self.n, h=f"{random.randint(1, 12)} o'clock")

    def turn(self):
        self.question()
        r = self.client.post(f"/api/conversations/{self.agent}/turn", headers=LOCAL,
                             files={"audio": ("q.wav", self.wav, "audio/wav")})
        return r.status_code

    def turn_stream(self):
        self.question()
        r = self.client.post(f"/api/conversations/{self.agent}/turn/stream", headers=LOCAL,
                             files={"audio": ("q.wav", self.wav, "audio/wav")})
        return r.status_code

    def kokoro(self):
        self.question()
        r = self.client.post("/api/kokoro/generate", headers=LOCAL, json={"voice": "af_heart", "text": self.asr_text})
        return r.status_code

    def transcribe(self):
        self.question()
        r = self.client.post("/api/transcribe", headers=LOCAL, files={"audio_file": ("q.wav", self.wav, "audio/wav")})
        return r.status_code


def sample(t0, requests):
    pid = os.getpid()
    p = psutil.Process(pid)
    raw = p.memory_info().rss / MB
    fp_raw = footprint_mb(pid)
    gc.collect()
    return {"t_s": round(time.time() - t0, 1), "requests": requests, "rss_mb": round(raw, 1),
            "footprint_mb": None if fp_raw is None else round(fp_raw, 1), "rss_after_gc_mb": round(p.memory_info().rss / MB, 1),
            "threads": threading.active_count(), "gc_objects": len(gc.get_objects())}


def slope(samples, key, after_s):
    pts = [(s["t_s"] / 60.0, s[key]) for s in samples if s["t_s"] >= after_s and s.get(key) is not None]
    if len(pts) < 3:
        return None
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    return round(float(np.polyfit(x, y, 1)[0]), 2)


def type_counts():
    c = collections.Counter(type(o).__name__ for o in gc.get_objects())
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--warmup", type=float, default=2.0, help="minutes before the slope fit and the first tracemalloc snapshot")
    ap.add_argument("--sample-every", type=float, default=30.0)
    ap.add_argument("--tracemalloc", action="store_true")
    ap.add_argument("--fake-embedder", action="store_true", help="a 64-dim hashing embedder instead of BGE-M3")
    ap.add_argument("--no-watermark", action="store_true", help="skip AudioSeal")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--label", default="", help="free text stored in the result (for example 'before fix')")
    ap.add_argument("--out")
    args = ap.parse_args()
    random.seed(args.seed)
    if not args.fake_embedder and not os.getenv("EMBEDDING_MODEL") and complete_bge_snapshot():
        os.environ["EMBEDDING_MODEL"] = complete_bge_snapshot()
    if args.tracemalloc:
        tracemalloc.start(12)
    rig = Rig(args)
    mix = [(rig.turn, 30), (rig.turn_stream, 30), (rig.kokoro, 25), (rig.transcribe, 15)]
    funcs = [f for f, w in mix for _ in range(w)]
    names = {rig.turn: "turn", rig.turn_stream: "turn_stream", rig.kokoro: "kokoro", rig.transcribe: "transcribe"}
    t0 = time.time()
    end = t0 + args.minutes * 60
    requests, failures, by_status = 0, 0, collections.Counter()
    samples = [sample(t0, 0)]
    next_sample = t0 + args.sample_every
    snap1 = counts1 = None
    print(f"leak hunt: {args.minutes} min, tracemalloc={args.tracemalloc}, embedder={'fake' if args.fake_embedder else 'real'}, "
          f"watermark={'off' if args.no_watermark else 'real'}", flush=True)
    print(f"  t={samples[0]['t_s']:6.1f}s  rss {samples[0]['rss_mb']} MB  footprint {samples[0]['footprint_mb']} MB", flush=True)
    while time.time() < end:
        f = random.choice(funcs)
        try:
            status = f()
        except Exception as e:  # noqa: BLE001
            status = f"exc:{type(e).__name__}"
        requests += 1
        by_status[f"{names[f]}:{status}"] += 1
        if status != 200:
            failures += 1
        now = time.time()
        if args.tracemalloc and snap1 is None and now - t0 >= args.warmup * 60:
            gc.collect()
            snap1, counts1 = tracemalloc.take_snapshot(), type_counts()
        if now >= next_sample:
            s = sample(t0, requests)
            samples.append(s)
            next_sample = now + args.sample_every
            print(f"  t={s['t_s']:6.1f}s  requests {requests:5d}  rss {s['rss_mb']:7.1f} (gc {s['rss_after_gc_mb']:7.1f}) MB  "
                  f"footprint {s['footprint_mb']} MB  objects {s['gc_objects']}  threads {s['threads']}", flush=True)
    gc.collect()
    result = {"label": args.label, "args": vars(args), "date": dt.datetime.now().isoformat(timespec="seconds"),
              "requests": requests, "failures": failures, "by_status": dict(by_status), "samples": samples,
              "rss_mb_per_min": slope(samples, "rss_mb", args.warmup * 60),
              "rss_after_gc_mb_per_min": slope(samples, "rss_after_gc_mb", args.warmup * 60),
              "footprint_mb_per_min": slope(samples, "footprint_mb", args.warmup * 60),
              "objects_per_min": slope(samples, "gc_objects", args.warmup * 60)}
    names_now = collections.Counter(t.name.rstrip("0123456789_-") or t.name for t in threading.enumerate())
    result["threads"] = dict(names_now)
    print("threads at the end:", dict(names_now))
    print(f"\nrequests {requests}, failures {failures}: {dict(by_status)}")
    print(f"slope after the first {args.warmup:g} min: RSS {result['rss_mb_per_min']} MB/min, RSS after gc "
          f"{result['rss_after_gc_mb_per_min']} MB/min, footprint {result['footprint_mb_per_min']} MB/min, "
          f"objects {result['objects_per_min']}/min")
    if args.tracemalloc and snap1 is not None:
        snap2, counts2 = tracemalloc.take_snapshot(), type_counts()
        diff = snap2.compare_to(snap1, "traceback")
        print("\nTop allocation sites by growth (traceback, innermost app frame last):")
        rows = []
        for st in diff[:15]:
            frames = [f for f in st.traceback.format() if "site-packages" not in f]
            tb = " <- ".join(x.strip().replace(BACKEND + "/", "") for x in frames[-6:][::-1] if x.strip().startswith("File"))
            print(f"  +{st.size_diff / 1024:9.1f} KiB  +{st.count_diff:7d} blocks  {tb[:400]}")
            rows.append({"kib": round(st.size_diff / 1024, 1), "blocks": st.count_diff, "where": tb[:600]})
        result["tracemalloc_top"] = rows
        print("\nTop allocation sites by line:")
        for st in snap2.compare_to(snap1, "lineno")[:15]:
            print(f"  +{st.size_diff / 1024:9.1f} KiB  +{st.count_diff:7d}  {str(st.traceback[0]).replace(BACKEND + '/', '')}")
        grew = sorted(((counts2[k] - counts1.get(k, 0), k) for k in counts2), reverse=True)[:12]
        print("\nObject types that grew:", ", ".join(f"{k} +{n}" for n, k in grew))
        result["types_grew"] = {k: n for n, k in grew}
    RESULTS_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / f"leak_hunt-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"written {out}")
    rig.client.__exit__(None, None, None)


if __name__ == "__main__":
    main()
