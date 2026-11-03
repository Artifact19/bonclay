#!/usr/bin/env python3
"""Before/after check of candidate Marathi spelling hints for DhVaani (services/pron_lexicon.py).

Starts only the DhVaani worker (never the API server), references are synthetic Kokoro voices. For each candidate rule
set the ळ / ऱ्ह sentences (and one plain control) are rendered, read back with mlx-whisper, and scored against the
ORIGINAL spelling with leak_scan. Whisper biases Marathi toward Hindi spelling, so a drop in "ळ lost" is weak evidence
and an unchanged count proves nothing: the wavs are written for a listening check, which is the real test.

    backend/.venv/bin/python backend/bench/pron_lexicon_check.py --out <dir> [--steps 16]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from dhvaani_live_check import asr_usable, make_refs  # noqa: E402
from dhvaani_marathi_check import SENTENCES, leak_scan  # noqa: E402
from indic_clone_trial import ASR_MODEL, MLX_PY, _ASR_SCRIPT  # noqa: E402
from scoring import cer, normalise_hyp  # noqa: E402
from services import dhvaani_clone as dh  # noqa: E402
from services.pron_lexicon import apply_rules, rule  # noqa: E402

PICK = ("la1", "la2", "rha", "plain")
LA, RHA = "ळ", "ऱ्ह"
CANDIDATES = {
    "baseline": [],
    "la_nukta": [rule("la->ल़", LA, "ल़")],
    "la_dda": [rule("la->ड़", LA, "ड़")],
    "rha_ra_ha": [rule("rha->र्ह", RHA, "र्ह")],
    "rha_raha": [rule("rha->रह", RHA, "रह")],
    "combo_a": [rule("la->ल़", LA, "ल़"), rule("rha->र्ह", RHA, "र्ह")],
    "combo_b": [rule("la->ड़", LA, "ड़"), rule("rha->रह", RHA, "रह")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for why in (dh.unavailable_reason(), asr_usable()):
        if why:
            sys.exit(why)
    refs = make_refs(a.out)
    client = dh.DhVaaniClone(dh.dhvaani_python())
    items = []
    try:
        m = refs["hi"]
        audio, sr = sf.read(m["path"], dtype="float32")
        sel = dh.select_reference(audio, sr, m["text"])
        prompt = dh.prompt_from_clip(sel.audio, sr, sel.text)
        client.generate_voice_clone(SENTENCES[0][2], "mr", prompt, num_step=a.steps)  # warm-up
        for cname, rules in CANDIDATES.items():
            for sid, kind, text, nums in SENTENCES:
                if sid not in PICK:
                    continue
                spoken = apply_rules(text, rules)
                dst = os.path.join(a.out, f"{sid}__{cname}.wav")
                wavs, sr = client.generate_voice_clone(spoken, "mr", prompt, num_step=a.steps)
                sf.write(dst, wavs[0], sr)
                items.append({"candidate": cname, "id": sid, "text": text, "spoken": spoken, "nums": nums, "wav": dst})
                print(cname, sid, flush=True)
    finally:
        client.close()
    jf, rf = os.path.join(a.out, "asr_jobs.json"), os.path.join(a.out, "asr_out.json")
    json.dump([{"wav": i["wav"], "lang": "mr"} for i in items], open(jf, "w"))
    subprocess.run([MLX_PY, "-c", _ASR_SCRIPT, jf, rf, ASR_MODEL], check=True, stdout=subprocess.DEVNULL)
    hyps = json.load(open(rf))
    for f in (jf, rf):
        os.remove(f)
    for it in items:
        it["hyp"] = hyps[it["wav"]]
        norm = normalise_hyp(it["hyp"], it["nums"])
        it["cer"] = cer(it["text"], norm)
        pairs, _subs, flags = leak_scan(it["text"], norm)
        it["leak_pairs"], it["leak_flags"] = pairs, flags
    summary = {}
    for cname in CANDIDATES:
        g = [i for i in items if i["candidate"] == cname]
        la = sum(1 for i in g for f in i["leak_flags"] if f.startswith("ळ lost"))
        rh = sum(1 for i in g for f in i["leak_flags"] if f.startswith("ऱ्ह lost"))
        ctrl = [i["cer"] for i in g if i["id"] == "plain"]
        summary[cname] = {"la_lost": la, "rha_lost": rh, "flags_total": sum(len(i["leak_flags"]) for i in g),
                          "mean_cer": round(float(np.mean([i["cer"] for i in g])), 3),
                          "plain_cer": round(float(ctrl[0]), 3) if ctrl else None}
    json.dump({"steps": a.steps, "asr": ASR_MODEL, "summary": summary, "items": items},
              open(os.path.join(a.out, "results.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{'candidate':<12} {'ळ lost':>6} {'ऱ्ह lost':>8} {'flags':>5} {'CER':>6} {'plain':>6}")
    for c, s in summary.items():
        print(f"{c:<12} {s['la_lost']:>6} {s['rha_lost']:>8} {s['flags_total']:>5} {s['mean_cer']:>6} {s['plain_cer']:>6}")


if __name__ == "__main__":
    main()
