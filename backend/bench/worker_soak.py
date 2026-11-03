#!/usr/bin/env python3
"""Short soak of the MLX model workers (the other half of the 45-minute soak's growth): Kokoro, speech recognition and the
local language model are started the way the app starts them and given a steady mix of real requests; each worker's
physical footprint and RSS are sampled every --sample-every seconds and fitted (after the first --warmup minutes).

    HF_HUB_OFFLINE=1 backend/.venv/bin/python backend/bench/worker_soak.py --minutes 8

No API, no data directory. Results: backend/bench/results/worker_soak-<stamp>.json (git-ignored).
"""
import argparse
import datetime as dt
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import model_memory as mm  # noqa: E402  (sets HF_HUB_OFFLINE, LLM_PROVIDERS=mlx, MEMORY_GUARD=off)

TEXTS = ["The clinic opens at nine in the morning.", "Please arrive ten minutes early and bring your reports.",
         "Your appointment is confirmed for {d} at {n} o'clock."]
HI = ["क्लिनिक सुबह नौ बजे खुलता है।", "कृपया दस मिनट पहले आएं।", "आपका अपॉइंटमेंट {n} बजे तय हो गया है।"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def slope(points, after_min):
    pts = [(t / 60, v) for t, v in points if t / 60 >= after_min and v is not None]
    if len(pts) < 3:
        return None
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    return round(float(np.polyfit(x, y, 1)[0]), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--warmup", type=float, default=1.5)
    ap.add_argument("--sample-every", type=float, default=30.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    from services import llm, mlx_asr, mlx_kokoro
    kok = mlx_kokoro.MlxKokoro()
    wav = mm.speech_wav_16k(os.path.join(mm.tempfile_dir(), "probe.wav"))
    mlx_asr.warm()
    llm.warm_local()
    pids = {"kokoro": kok.proc.pid, "asr": mlx_asr._worker.proc.pid, "llm": llm._mlx.proc.pid}
    series = {k: {"footprint": [], "rss": []} for k in pids}
    t0 = time.time()
    stop = threading.Event()

    def sampler():
        while not stop.wait(args.sample_every):
            line = []
            for k, pid in pids.items():
                m = mm.tree_memory(pid)
                if m:
                    t = time.time() - t0
                    series[k]["footprint"].append((t, m["footprint_mb"]))
                    series[k]["rss"].append((t, m["rss_mb"]))
                    line.append(f"{k} {m['footprint_mb']:.0f}/{m['rss_mb']:.0f}")
            print(f"  t={time.time() - t0:6.0f}s  footprint/rss MB: " + "  ".join(line), flush=True)

    for k, pid in pids.items():
        m = mm.tree_memory(pid)
        series[k]["footprint"].append((0.0, m["footprint_mb"]))
        series[k]["rss"].append((0.0, m["rss_mb"]))
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    n = 0
    end = t0 + args.minutes * 60
    random.seed(3)
    counts = {"kokoro": 0, "asr": 0, "llm": 0}
    while time.time() < end:
        n += 1
        what = random.choices(["kokoro_en", "kokoro_hi", "asr", "llm"], [30, 20, 15, 35])[0]
        try:
            if what == "kokoro_en":
                kok.synthesize(random.choice(TEXTS).format(d=random.choice(DAYS), n=n), "af_heart", "a")
                counts["kokoro"] += 1
            elif what == "kokoro_hi":
                kok.synthesize(random.choice(HI).format(n=n), "hf_alpha", "h")
                counts["kokoro"] += 1
            elif what == "asr":
                mlx_asr.transcribe(wav, random.choice([None, "en", "hi"]))
                counts["asr"] += 1
            else:
                llm.chat([{"role": "system", "content": "You are a clinic receptionist. Answer in one short sentence."},
                          {"role": "user", "content": f"What time does the clinic open on {random.choice(DAYS)}? Ticket {n}."}],
                         temperature=0.3, max_tokens=40, timeout=120.0)
                counts["llm"] += 1
        except Exception as e:  # noqa: BLE001
            print("  request failed:", type(e).__name__, str(e)[:100])
    stop.set()
    th.join(timeout=2)
    result = {"label": args.label, "date": dt.datetime.now().isoformat(timespec="seconds"), "minutes": args.minutes,
              "requests": counts, "slopes_mb_per_min": {k: {"footprint": slope(v["footprint"], args.warmup),
                                                              "rss": slope(v["rss"], args.warmup)} for k, v in series.items()},
              "series": series, "env": {k: os.environ.get(k) for k in ("MLX_CACHE_LIMIT_MB", "MLX_ASR_STRATEGY")}}
    print("requests:", counts)
    print("slopes (MB/min after warm-up):", json.dumps(result["slopes_mb_per_min"]))
    out = Path(args.out) if args.out else mm.RESULTS_DIR / f"worker_soak-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    mm.RESULTS_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    kok.close()
    mlx_asr._worker.stop()
    llm._mlx.stop()


if __name__ == "__main__":
    main()
