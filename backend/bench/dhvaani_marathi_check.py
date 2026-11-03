#!/usr/bin/env python3
"""Marathi-focused check of DhVaani-0.5 voice cloning. Starts only the DhVaani worker (the app's own client,
services/dhvaani_clone.py), never the API server. References are two synthetic Kokoro voices (am_michael speaking
English, hf_alpha speaking Hindi); no real person's recording and nothing from backend/data is used.

12 varied Marathi sentences x 2 reference types (hi, en) x 2 step settings (16, 32) = 48 renders. Measures seconds per
sentence, RTF, read-back CER (mlx-whisper large-v3-turbo in backend/.venv-mlx; Whisper can mishear Marathi and often
writes it Hindi-style, so CER is an upper bound on the engine's error and a lower bound on Hindi leakage), ECAPA speaker
similarity to the reference, and a Hindi-leakage scan (ref word vs heard word, character substitutions).

    backend/.venv/bin/python backend/bench/dhvaani_marathi_check.py --out video/dhvaani_marathi [--steps 16 32]

Writes <out>/s<NN>_<ref>_<steps>.wav, ref_*.wav, results.json and README.md.
"""
import argparse
import collections
import difflib
import json
import os
import subprocess
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from dhvaani_live_check import asr_usable, make_refs  # noqa: E402
from indic_clone_trial import ASR_MODEL, MLX_PY, _ASR_SCRIPT, cer, normalise_hyp  # noqa: E402
from services import dhvaani_clone as dh  # noqa: E402

# (id, kind, text, number words that Whisper's digits are mapped back to)
SENTENCES = [
    ("greeting", "greeting", "नमस्कार, आमच्या दवाखान्यात तुमचे स्वागत आहे.", []),
    ("greeting2", "greeting", "सुप्रभात, आज मी तुम्हाला कशी मदत करू शकते?", []),
    ("date", "number/date", "तुमची भेट पंधरा नोव्हेंबरला सकाळी दहा वाजता ठरली आहे.", ["पंधरा", "दहा"]),
    ("fee", "number", "पहिल्या तपासणीची फी चारशे पन्नास रुपये आहे.", ["चारशे पन्नास"]),
    ("question", "question", "तुम्हाला हा त्रास किती दिवसांपासून होत आहे?", []),
    ("long", "long (23 words)",
     "कृपया तुमचे जुने रिपोर्ट, औषधांची चिठ्ठी आणि आधार कार्ड सोबत घेऊन या, कारण डॉक्टरांना तुमच्या आजाराचा "
     "संपूर्ण इतिहास पाहायचा आहे आणि त्यानंतरच ते पुढचे उपचार ठरवतील.", []),
    ("loan", "loanwords", "तुमची अपॉइंटमेंट क्लिनिकमध्ये कन्फर्म झाली आहे, कृपया रिसेप्शनवर या.", []),
    ("la1", "ळ", "सकाळी आणि संध्याकाळी जेवणानंतर ही गोळी पाण्याबरोबर घ्या.", []),
    ("la2", "ळ", "आम्ही तुम्हाला वेळेवर कळवू, तुम्ही अजिबात काळजी करू नका.", []),
    ("rha", "ऱ्ह", "वऱ्हाडी मंडळी कऱ्हाडहून आली आणि पुण्यातील नऱ्हे गावात थांबली.", []),
    ("mixed", "Hinglish-ish", "आज माझा शेड्यूल खूप बिझी आहे, पण मी तुम्हाला संध्याकाळी कॉल करतो.", []),
    ("plain", "Marathi-only forms", "मला माहीत नाही, पण तुम्ही उद्या परत या, मी बघतो आणि सांगतो.", []),
]


def _nfc(s):
    return unicodedata.normalize("NFC", s)


def words(s):
    out = []
    for w in _nfc(s).replace("।", " ").split():
        w = "".join(ch for ch in w if unicodedata.category(ch)[0] in "LM")
        if w:
            out.append(w)
    return out


def leak_scan(ref, hyp):
    """Word-level alignment of the reference against what Whisper heard. Returns (mismatched word pairs, character
    substitutions inside aligned word pairs, flags for known Hindi-leak patterns)."""
    rw, hw = words(ref), words(hyp)
    pairs, subs, flags = [], collections.Counter(), []
    sm = difflib.SequenceMatcher(None, rw, hw, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        r, h = rw[i1:i2], hw[j1:j2]
        pairs.append((" ".join(r), " ".join(h)))
        if tag == "replace" and len(r) == len(h):
            for a, b in zip(r, h):
                for t2, a1, a2, b1, b2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                    if t2 != "equal":
                        subs[(a[a1:a2] or "-", b[b1:b2] or "-")] += 1
                if "ळ" in a and "ळ" not in b:
                    flags.append(f"ळ lost: {a} -> {b}")
                if "ऱ्ह" in a and "ऱ्ह" not in b:
                    flags.append(f"ऱ्ह lost: {a} -> {b}")
                if "ण" in a and "ण" not in b:
                    flags.append(f"ण -> न: {a} -> {b}")
                if a.endswith("ी") and b.endswith("ि") or a.endswith("ळी") and b.endswith("ली"):
                    flags.append(f"ending: {a} -> {b}")
    return pairs, subs, flags


def similarity_fn():
    try:
        from services import speaker
        import torch
        import torchaudio

        def load16(p):
            x, sr = sf.read(p, dtype="float32")
            x = x.mean(axis=1) if x.ndim > 1 else x
            return torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy() if sr != 16000 else x

        def emb(p):
            return speaker.embed(load16(p))
        return emb, None
    except Exception as e:  # noqa: BLE001
        return None, f"skipped: {type(e).__name__}: {e}"[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[16, 32])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reason = dh.unavailable_reason()
    if reason:
        sys.exit(f"DhVaani is not available: {reason}")
    problem = asr_usable()
    if problem:
        sys.exit(f"read-back is not available: {problem}")

    print("making the synthetic references with Kokoro ...", flush=True)
    refs = make_refs(a.out)
    results = {"refs": {k: {x: v for x, v in m.items() if x != "path"} for k, m in refs.items()},
               "free_mb_before": dh.free_memory_mb(), "loadavg_start": os.getloadavg(), "items": []}
    t0 = time.time()
    client = dh.DhVaaniClone(dh.dhvaani_python())
    results["worker_start_seconds"] = round(time.time() - t0, 1)
    results["device"] = getattr(client, "device", None)
    try:
        prompts = {}
        for k, m in refs.items():
            audio, sr = sf.read(m["path"], dtype="float32")
            sel = dh.select_reference(audio, sr, m["text"])
            prompts[k] = dh.prompt_from_clip(sel.audio, sr, sel.text)
        t0 = time.time()
        client.generate_voice_clone(SENTENCES[0][2], "mr", prompts["hi"], num_step=16)
        results["first_request_seconds"] = round(time.time() - t0, 1)
        print(f"warm-up request {results['first_request_seconds']} s", flush=True)
        for steps in a.steps:
            for ref in ("hi", "en"):
                for n, (sid, kind, text, nums) in enumerate(SENTENCES, 1):
                    dst = os.path.join(a.out, f"s{n:02d}_{sid}_{ref}_{steps}steps.wav")
                    t0 = time.time()
                    try:
                        wavs, sr = client.generate_voice_clone(text, "mr", prompts[ref], num_step=steps)
                        secs = time.time() - t0
                        sf.write(dst, wavs[0], sr)
                        item = {"gen_seconds": round(secs, 2), "audio_seconds": round(len(wavs[0]) / sr, 2),
                                "rtf": round(secs / max(len(wavs[0]) / sr, 1e-6), 2), "error": None}
                    except Exception as e:  # noqa: BLE001
                        item = {"gen_seconds": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {e}"[:300]}
                    item.update({"n": n, "id": sid, "kind": kind, "text": text, "nums": nums, "ref": ref,
                                 "steps": steps, "wav": dst, "loadavg": round(os.getloadavg()[0], 2)})
                    results["items"].append(item)
                    print(f"  {steps} steps from_{ref} s{n:02d} {sid}: {item['gen_seconds']} s / "
                          f"{item.get('audio_seconds')} s audio RTF {item.get('rtf')} {item['error'] or ''}", flush=True)
        results["free_mb_after"] = dh.free_memory_mb()
    finally:
        proc = client.proc
        client.close()
        results["worker_stopped"] = proc.poll() is not None
        print(f"worker stopped: {results['worker_stopped']}", flush=True)

    ok = [it for it in results["items"] if not it["error"]]
    jf, rf = os.path.join(a.out, "asr_jobs.json"), os.path.join(a.out, "asr_out.json")
    json.dump([{"wav": it["wav"], "lang": "mr"} for it in ok], open(jf, "w"))
    t0 = time.time()
    subprocess.run([MLX_PY, "-c", _ASR_SCRIPT, jf, rf, ASR_MODEL], check=True, stdout=subprocess.DEVNULL)
    results["asr"] = {"model": ASR_MODEL, "seconds": round(time.time() - t0, 1)}
    hyps = json.load(open(rf))
    emb, why = similarity_fn()
    ref_emb = {k: emb(m["path"]) for k, m in refs.items()} if emb else None
    if why:
        results["similarity"] = why
    for it in ok:
        hyp = hyps[it["wav"]]
        it["hyp"] = hyp
        norm = normalise_hyp(hyp, it["nums"])
        it["cer"] = cer(it["text"], norm)
        it["leak_pairs"], subs, it["leak_flags"] = leak_scan(it["text"], norm)
        it["leak_subs"] = [[x, y, c] for (x, y), c in subs.items()]
        if ref_emb:
            e = emb(it["wav"])
            r = ref_emb[it["ref"]]
            it["sim"] = round(float(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r))), 3)
    # the audio files are the evidence; the job files are scratch
    for f in (jf, rf):
        os.remove(f)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), ensure_ascii=False, indent=1)
    write_readme(a.out, results)

    print(f"\n{'steps':>5} {'ref':<3} {'gen s':>6} {'RTF':>5} {'CER':>6} {'sim':>5}")
    for steps in a.steps:
        for ref in ("hi", "en"):
            g = [i for i in ok if i["steps"] == steps and i["ref"] == ref]
            if g:
                m = lambda k: np.mean([i[k] for i in g if k in i])  # noqa: E731
                print(f"{steps:>5} {ref:<3} {m('gen_seconds'):>6.1f} {m('rtf'):>5.2f} {m('cer'):>6.3f} {m('sim'):>5.3f}")


def write_readme(out, results):
    L = ["# DhVaani Marathi listening set", "",
         "Every voice here is synthetic: references are Kokoro `hf_alpha` (Hindi speech, `ref_hi.wav`) and Kokoro "
         "`am_michael` (English speech, `ref_en.wav`). File name: `sNN_<id>_<ref>_<steps>steps.wav`.", "",
         "| NN | id | type | Marathi sentence |", "| --- | --- | --- | --- |"]
    for n, (sid, kind, text, _) in enumerate(SENTENCES, 1):
        L.append(f"| {n:02d} | {sid} | {kind} | {text} |")
    L += ["", "Per sentence there are four files: `_hi_16steps`, `_hi_32steps`, `_en_16steps`, `_en_32steps`. "
          "Numbers, CER, similarity and the Hindi-leakage list are in `results.json` and `docs/RESULTS.md`."]
    open(os.path.join(out, "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
