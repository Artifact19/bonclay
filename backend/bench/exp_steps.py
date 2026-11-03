#!/usr/bin/env python3
"""Experiment: DhVaani sampling steps vs speed and read-back CER (the flow-matching 'fewer solver steps' idea).

One worker session, steps [6, 8, 12, 16] on Hindi, Marathi and Tamil (28 sentences), synthetic Hindi Kokoro reference, then one
read-back pass after the worker is closed. Run on an otherwise idle machine: the timings are the point.

    backend/.venv/bin/python backend/bench/exp_steps.py --out DIR
"""
import argparse
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
sys.path.insert(0, HERE)

import soundfile as sf  # noqa: E402

import indic_benchmark as ib  # noqa: E402
from services import dhvaani_clone as dh  # noqa: E402

STEPS = [6, 8, 12, 16]
LANGS = ["hi", "mr", "ta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = ib.load_sentences(LANGS)
    refs = ib.make_refs(a.out)
    client = dh.DhVaaniClone(dh.dhvaani_python())
    recs = []
    try:
        audio, sr = sf.read(refs["hi"]["path"], dtype="float32")
        sel = dh.select_reference(audio, sr, refs["hi"]["text"])
        prompt = dh.prompt_from_clip(sel.audio, sr, sel.text)
        client.generate_voice_clone(ib.read_text(items[0]), "hi", prompt, num_step=16)  # warm-up
        for steps in STEPS:
            for it in items:
                dst = os.path.join(a.out, "wavs", f"s{steps}", f"{it['id']}.wav")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                t0 = time.time()
                wavs, osr = client.generate_voice_clone(ib.read_text(it), it["language"], prompt, num_step=steps)
                gen = time.time() - t0
                sf.write(dst, wavs[0], osr)
                recs.append({"steps": steps, "id": it["id"], "language": it["language"], "wav": dst, "gen_seconds": round(gen, 2),
                             "audio_seconds": round(len(wavs[0]) / osr, 2), "item": it})
            print(f"steps {steps} rendered", flush=True)
    finally:
        client.close()
    hyps = ib.run_asr([{"wav": r["wav"], "language": r["language"], "error": None} for r in recs], a.out)
    out = {}
    for r in recs:
        sc = ib.score_item(r.pop("item"), hyps[r["wav"]])
        r["cer"] = sc["cer"]
        r["rtf"] = round(r["gen_seconds"] / max(r["audio_seconds"], 1e-6), 3)
    for steps in STEPS:
        g = [r for r in recs if r["steps"] == steps]
        out[steps] = {"mean_cer": round(st.mean(r["cer"] for r in g), 3), "mean_gen_s": round(st.mean(r["gen_seconds"] for r in g), 2),
                      "mean_rtf": round(st.mean(r["rtf"] for r in g), 3),
                      **{l: round(st.mean(r["cer"] for r in g if r["language"] == l), 3) for l in LANGS}}
    json.dump({"summary": out, "items": recs}, open(os.path.join(a.out, "results.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
