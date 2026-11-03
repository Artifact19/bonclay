#!/usr/bin/env python3
"""Runs OmniVoice (Xiaomi / k2-fsa, arXiv 2604.00688) for bench/indic_benchmark.py. RESEARCH USE ONLY: the weights are CC-BY-NC.

Own virtualenv (backend/.venv-omni: torch 2.8.0, torchaudio 2.8.0, `pip install omnivoice`); Apple GPU (mps, float16).
    .venv-omni/bin/python bench/omnivoice_worker.py JOBS.json OUT.json --ref REF.wav --ref-text "transcript"
JOBS.json: [{"id", "text", "language", "wav"}]. OUT.json: {id: {"gen_seconds", "audio_seconds"} or {"error"}, "_load_seconds": s}.
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs"), ap.add_argument("out")
    ap.add_argument("--ref", required=True), ap.add_argument("--ref-text", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice
    jobs = json.load(open(a.jobs))
    t0 = time.time()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=a.device, dtype=torch.float16)
    res = {"_load_seconds": round(time.time() - t0, 1), "_device": a.device}

    def one(job):
        t = time.time()
        audio = model.generate(text=job["text"], language=job["language"], ref_audio=a.ref, ref_text=a.ref_text)[0]
        os.makedirs(os.path.dirname(job["wav"]), exist_ok=True)
        sf.write(job["wav"], audio, 24000)
        return {"gen_seconds": round(time.time() - t, 2), "audio_seconds": round(len(audio) / 24000, 2)}

    try:
        one({**jobs[0], "wav": jobs[0]["wav"] + ".warmup.wav"})
        os.remove(jobs[0]["wav"] + ".warmup.wav")
    except Exception as e:  # noqa: BLE001
        print("warm-up failed:", e, flush=True)
    for job in jobs:
        try:
            res[job["id"]] = one(job)
        except Exception as e:  # noqa: BLE001
            res[job["id"]] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print("  omnivoice", job["id"], res[job["id"]], flush=True)
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
