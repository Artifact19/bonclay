#!/usr/bin/env python3
"""Experiment A: does DhVaani speak numbers correctly when the input has digits instead of number words?

Takes every benchmark sentence (hi mr ta bn te kn gu ml pa; not hinglish) with a non-empty number_words, renders
  words   = the original text,
  digits  = number words replaced by ASCII digits (all languages),
  native  = number words replaced by native-script digits (hi, mr only),
with the DhVaani worker (Hindi synthetic Kokoro reference, 16 steps), reads each back with mlx-whisper and scores
against the ORIGINAL words text with scoring.cer / normalise_hyp. Also a number-correctness check that the digit
mapping in normalise_hyp would otherwise hide (the scorer turns any digit run back into the expected word).

    backend/.venv/bin/python backend/bench/exp_digits_vs_words.py --out <dir> [--stage all|gen|asr]
"""
import argparse
import glob
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

from dhvaani_live_check import make_refs  # noqa: E402
from indic_clone_trial import ASR_MODEL, MLX_PY, _ASR_SCRIPT  # noqa: E402
from scoring import _DIGITS, cer, normalise_hyp  # noqa: E402
from services import dhvaani_clone as dh  # noqa: E402

LANGS = ("hi", "mr", "ta", "bn", "te", "kn", "gu", "ml", "pa")
VALUE = {
    "hi": {"पंद्रह": "15", "दस": "10", "चार सौ पचास": "450"},
    "mr": {"पंधरा": "15", "दहा": "10", "चारशे पन्नास": "450"},
    "bn": {"দশ": "10", "পাঁচ": "5", "চারশো": "400"},
    "gu": {"દસ": "10", "પાંચ": "5", "ચારસો": "400"},
    "kn": {"ಹತ್ತು": "10", "ಐದು": "5", "ನಾನೂರು": "400"},
    "ml": {"പത്തു": "10", "അഞ്ചു": "5", "നാനൂറു": "400"},
    "pa": {"ਦਸ": "10", "ਪੰਜ": "5", "ਚਾਰ ਸੌ": "400"},
    "ta": {"பத்து": "10", "ஐந்து": "5", "நானூறு": "400"},
    "te": {"పది": "10", "ఐదు": "5", "నాలుగు వందల": "400"},
}
NATIVE = {"hi": "०१२३४५६७८९", "mr": "०१२३४५६७८९"}
LOCK_NOTES = "/Users/ayushchougula/Library/Application Support/Claude/side-session-notes/local_6e55bbb9-b8fe-413f-a039-b3201b264e3f"


def to_ascii(s):
    out = []
    for ch in s:
        if ch.isdigit():
            import unicodedata
            out.append(str(unicodedata.digit(ch)))
        else:
            out.append(ch)
    return "".join(out)


def digit_runs(s):
    return [to_ascii(m.group(0)) for m in _DIGITS.finditer(s or "")]


def build_items():
    items = []
    for lang in LANGS:
        for it in json.load(open(os.path.join(HERE, "data", "indic_benchmark", f"{lang}.json"))):
            nums = it.get("number_words") or []
            if not nums:
                continue
            vals = [VALUE[lang][w] for w in nums]
            ascii_t, native_t = it["text"], it["text"]
            # replace in order, one occurrence each (left to right) so repeated words stay aligned
            pos = 0
            a_parts, n_parts = [], []
            for w, v in zip(nums, vals):
                i = it["text"].index(w, pos)
                a_parts.append(it["text"][pos:i])
                a_parts.append(v)
                n_parts.append(it["text"][pos:i])
                n_parts.append("".join(NATIVE[lang][int(c)] for c in v) if lang in NATIVE else v)
                pos = i + len(w)
            a_parts.append(it["text"][pos:])
            n_parts.append(it["text"][pos:])
            ascii_t, native_t = "".join(a_parts), "".join(n_parts)
            variants = {"words": it["text"], "digits": ascii_t}
            if lang in NATIVE:
                variants["native"] = native_t
            items.append({"id": it["id"], "lang": lang, "text": it["text"], "nums": nums, "values": vals,
                          "variants": variants})
    return items


def wait_memory(mb=4500, max_s=600):
    import psutil
    t0 = time.time()
    while True:
        free = psutil.virtual_memory().available / 2 ** 20
        if free >= mb:
            print(f"memory ok: {free:.0f} MB available", flush=True)
            return free
        if time.time() - t0 > max_s:
            sys.exit(f"gave up waiting for memory: {free:.0f} MB available")
        print(f"waiting for memory: {free:.0f} MB < {mb}", flush=True)
        time.sleep(30)


def stage_gen(out, items):
    wait_memory()
    print("making synthetic references ...", flush=True)
    refs = make_refs(out)
    client = dh.DhVaaniClone(dh.dhvaani_python())
    renders = []
    try:
        audio, sr = sf.read(refs["hi"]["path"], dtype="float32")
        sel = dh.select_reference(audio, sr, refs["hi"]["text"])
        prompt = dh.prompt_from_clip(sel.audio, sr, sel.text)
        t0 = time.time()
        client.generate_voice_clone(items[0]["variants"]["words"], "hi", prompt, num_step=16)
        print(f"warm-up {time.time() - t0:.1f} s", flush=True)
        for it in items:
            for var, text in it["variants"].items():
                dst = os.path.join(out, f"{it['id']}_{var}.wav")
                t0 = time.time()
                rec = {"id": it["id"], "lang": it["lang"], "variant": var, "text_in": text, "wav": dst, "error": None}
                try:
                    wavs, sr2 = client.generate_voice_clone(text, it["lang"], prompt, num_step=16)
                    sf.write(dst, wavs[0], sr2)
                    rec["audio_seconds"] = round(len(wavs[0]) / sr2, 2)
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                rec["gen_seconds"] = round(time.time() - t0, 2)
                renders.append(rec)
                print(f"  {it['id']} {var}: {rec['gen_seconds']} s {rec['error'] or ''}", flush=True)
    finally:
        proc = client.proc
        client.close()
        print(f"worker stopped: {proc.poll() is not None}", flush=True)
    json.dump(renders, open(os.path.join(out, "renders.json"), "w"), ensure_ascii=False, indent=1)


def acquire_lock(max_s=1200):
    path = os.path.join(LOCK_NOTES, "asr.lock")
    t0 = time.time()
    while True:
        try:
            os.mkdir(path)
            print("asr lock acquired", flush=True)
            return path
        except FileExistsError:
            if time.time() - t0 > max_s:
                sys.exit("could not get the ASR lock in 20 minutes")
            print("asr lock busy, waiting", flush=True)
            time.sleep(10)


def stage_asr(out):
    renders = [r for r in json.load(open(os.path.join(out, "renders.json"))) if not r["error"]]
    jf, rf = os.path.join(out, "asr_jobs.json"), os.path.join(out, "asr_out.json")
    json.dump([{"wav": r["wav"], "lang": r["lang"]} for r in renders], open(jf, "w"))
    lock = acquire_lock()
    try:
        t0 = time.time()
        subprocess.run([MLX_PY, "-c", _ASR_SCRIPT, jf, rf, ASR_MODEL], check=True, stdout=subprocess.DEVNULL)
        print(f"asr done in {time.time() - t0:.0f} s", flush=True)
    finally:
        try:
            os.rmdir(lock)
            print("asr lock released", flush=True)
        except OSError as e:
            print(f"could not remove lock: {e}", flush=True)
    json.dump(json.load(open(rf)), open(os.path.join(out, "transcripts.json"), "w"), ensure_ascii=False, indent=1)
    os.remove(jf)
    os.remove(rf)


def num_ok(hyp, nums, values):
    """True when every expected number is in the transcript: as digits (ASCII or native, in order) or as its word."""
    runs = digit_runs(hyp)
    if runs:
        return runs == values
    return all(w in hyp for w in nums)


def stage_score(out, items):
    renders = json.load(open(os.path.join(out, "renders.json")))
    hyps = json.load(open(os.path.join(out, "transcripts.json")))
    by_id = {it["id"]: it for it in items}
    for r in renders:
        if r["error"]:
            continue
        it = by_id[r["id"]]
        hyp = hyps[r["wav"]]
        r["hyp"] = hyp
        r["cer"] = cer(it["text"], normalise_hyp(hyp, it["nums"]))
        r["hyp_digit_runs"] = digit_runs(hyp)
        r["num_ok"] = num_ok(hyp, it["nums"], it["values"])
    json.dump(renders, open(os.path.join(out, "results.json"), "w"), ensure_ascii=False, indent=1)
    write_report(out, items, renders)


def write_report(out, items, renders):
    ok = [r for r in renders if not r["error"]]
    g = lambda i, v: next((r for r in ok if r["id"] == i and r["variant"] == v), None)  # noqa: E731
    L = ["# Experiment A: digits vs number words, DhVaani", ""]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    summ = {}
    for v in ("words", "digits", "native"):
        rs = [r for r in ok if r["variant"] == v]
        summ[v] = {"n": len(rs), "mean_cer": mean([r["cer"] for r in rs]), "num_ok": sum(r["num_ok"] for r in rs)}
    L += ["## Per variant (all scored against the original words text)", "",
          "| variant | n | mean CER | numbers heard correctly |", "| --- | --- | --- | --- |"]
    for v, s in summ.items():
        L.append(f"| {v} | {s['n']} | {s['mean_cer']:.3f} | {s['num_ok']}/{s['n']} |")
    # words baseline restricted to the same sentences as native (hi, mr)
    nat_ids = {r["id"] for r in ok if r["variant"] == "native"}
    wn = [g(i, "words")["cer"] for i in nat_ids if g(i, "words")]
    L += ["", f"Words baseline on the hi+mr sentences only (same as native): {mean(wn):.3f}", ""]
    L += ["## Per sentence CER (number heard correctly in brackets)", "",
          "| id | words | digits | native | digits - words | native - words |", "| --- | --- | --- | --- | --- | --- |"]
    deltas = []
    for it in items:
        row = [it["id"]]
        w = g(it["id"], "words")
        cells = {}
        for v in ("words", "digits", "native"):
            r = g(it["id"], v)
            cells[v] = r
            row.append(f"{r['cer']:.3f} ({'ok' if r['num_ok'] else 'WRONG'})" if r else "-")
        for v in ("digits", "native"):
            r = cells[v]
            if r and w:
                d = r["cer"] - w["cer"]
                deltas.append((d, it["id"], v, r, w))
                row.append(f"{d:+.3f}")
            else:
                row.append("-")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    for v in ("digits", "native"):
        ds = [d for d in deltas if d[2] == v]
        worse = [d for d in ds if d[0] > 0.10]
        L.append(f"- {v}: {len(worse)} of {len(ds)} renders have CER more than 0.10 above the words render "
                 f"({', '.join(d[1] for d in worse) or 'none'}).")
    L += ["", "## Worst 3 digit/native renders by CER difference to the words render", ""]
    for d, i, v, r, w in sorted(deltas, key=lambda x: -x[0])[:3]:
        L += [f"- {i} {v} (delta {d:+.3f}, number heard {'ok' if r['num_ok'] else 'WRONG'})",
              f"  - input text: {r['text_in']}", f"  - original:   {w['text_in']}",
              f"  - Whisper (digit render): {r['hyp']}", f"  - Whisper (words render): {w['hyp']}"]
    L += ["", "## All transcripts", ""]
    for it in items:
        for v in it["variants"]:
            r = g(it["id"], v)
            if r:
                L.append(f"- {it['id']} {v}: in `{r['text_in']}` heard `{r['hyp']}` CER {r['cer']:.3f}")
    errs = [r for r in renders if r["error"]]
    if errs:
        L += ["", "Render errors: " + "; ".join(f"{r['id']} {r['variant']}: {r['error']}" for r in errs)]
    open(os.path.join(out, "report_raw.md"), "w").write("\n".join(L) + "\n")
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("\n".join(L[:30]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "gen", "asr", "score"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = build_items()
    json.dump(items, open(os.path.join(a.out, "items.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{len(items)} sentences, {sum(len(i['variants']) for i in items)} renders", flush=True)
    if a.stage in ("all", "gen"):
        stage_gen(a.out, items)  # worker is closed before ASR
    if a.stage in ("all", "asr"):
        stage_asr(a.out)
    if a.stage in ("all", "score"):
        stage_score(a.out, items)


if __name__ == "__main__":
    main()
