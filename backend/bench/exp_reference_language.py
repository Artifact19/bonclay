#!/usr/bin/env python3
"""Experiment B: does the language of the DhVaani reference voice change the read-back CER?

Languages ml, pa (the bad ones) and ta, kn (controls). Three reference conditions, all synthetic (Kokoro voices, no real
person): hi = Hindi speech (the benchmark baseline), en = English speech, boot = a SAME-LANGUAGE reference bootstrapped
from the engine's own hi-reference render of one short sentence of that language (cut with select_reference, its known
text as the prompt text). The bootstrap sentence (id -02, "the visit is fixed for tomorrow ten o'clock") is excluded
from the boot condition and from the 'comparable' means.

    backend/.venv/bin/python backend/bench/exp_reference_language.py --out <dir> [--skip-render]

Steps: wait for memory, render everything in one DhVaani worker, close it, take the ASR lock (a mkdir lock shared with
other jobs), transcribe with mlx-whisper in backend/.venv-mlx, score with bench/scoring.py, write results.json and
tables.md. Never starts the API server, never touches backend/data.
"""
import argparse
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

from dhvaani_live_check import make_refs  # noqa: E402
from indic_clone_trial import ASR_MODEL, MLX_PY, _ASR_SCRIPT  # noqa: E402
from scoring import cer, normalise_hyp  # noqa: E402
from services import dhvaani_clone as dh  # noqa: E402

LANGS = ("ml", "pa", "ta", "kn")
BOOT_ID_SUFFIX = "-02"
STEPS = 16
MIN_FREE_MB = 4500
SCRIPT_PREFIX = {"ml": "MALAYALAM", "pa": "GURMUKHI", "ta": "TAMIL", "kn": "KANNADA", "hi": "DEVANAGARI"}


def load_items(lang):
    return json.load(open(os.path.join(HERE, "data", "indic_benchmark", f"{lang}.json")))


def wait_memory(timeout_s=600, poll_s=30):
    import psutil
    t0 = time.time()
    while True:
        free = psutil.virtual_memory().available / 2 ** 20
        if free >= MIN_FREE_MB:
            print(f"free memory {free:.0f} MB, starting", flush=True)
            return free
        if time.time() - t0 > timeout_s:
            sys.exit(f"gave up waiting for memory ({free:.0f} MB free < {MIN_FREE_MB} MB)")
        print(f"waiting for memory: {free:.0f} MB free < {MIN_FREE_MB}", flush=True)
        time.sleep(poll_s)


def script_share(text, lang):
    """Share of the letters of `text` that belong to the language's own script, and the dominant script name."""
    counts = {}
    for c in unicodedata.normalize("NFC", text or ""):
        if unicodedata.category(c)[0] == "L":
            try:
                name = unicodedata.name(c).split()[0]
            except ValueError:
                name = "?"
            counts[name] = counts.get(name, 0) + 1
    tot = sum(counts.values())
    if not tot:
        return 0.0, "none"
    return round(counts.get(SCRIPT_PREFIX[lang], 0) / tot, 3), max(counts, key=counts.get)


def get_refs(out):
    """The two Kokoro references; reused when already written (so a restart does not hold Kokoro in memory)."""
    from indic_clone_trial import REFS
    meta_path = os.path.join(out, "refs_meta.json")
    if os.path.isfile(meta_path) and all(os.path.isfile(os.path.join(out, f"ref_{k}.wav")) for k in REFS):
        return json.load(open(meta_path))
    refs = make_refs(out)
    json.dump(refs, open(meta_path, "w"), ensure_ascii=False)
    import gc
    gc.collect()
    return refs


def render(out):
    wait_memory()
    refs = get_refs(out)
    reason = dh.unavailable_reason()
    if reason:
        sys.exit(f"DhVaani is not available: {reason}")
    wait_memory()
    results = {"refs": {k: {x: v for x, v in m.items() if x != "path"} for k, m in refs.items()}, "renders": [],
               "boot": {}, "steps": STEPS}
    client = dh.DhVaaniClone(dh.dhvaani_python())
    try:
        prompts = {}
        for k, m in refs.items():
            audio, sr = sf.read(m["path"], dtype="float32")
            sel = dh.select_reference(audio, sr, m["text"])
            prompts[k] = dh.prompt_from_clip(sel.audio, sr, sel.text)

        def gen(lang, text, prompt, dst):
            t0 = time.time()
            wavs, sr = client.generate_voice_clone(text, lang, prompt, num_step=STEPS)
            sf.write(dst, wavs[0], sr)
            return {"gen_seconds": round(time.time() - t0, 2), "audio_seconds": round(len(wavs[0]) / sr, 2)}

        client.generate_voice_clone(load_items("ta")[0]["text"], "ta", prompts["hi"], num_step=STEPS)  # warm-up
        for lang in LANGS:
            items = load_items(lang)
            os.makedirs(os.path.join(out, "wavs", lang), exist_ok=True)
            for cond in ("hi", "en"):
                for it in items:
                    dst = os.path.join(out, "wavs", lang, f"{it['id']}_{cond}.wav")
                    rec = {"lang": lang, "cond": cond, "id": it["id"], "text": it["text"], "nums": it["number_words"],
                           "wav": dst, "error": None}
                    try:
                        rec.update(gen(lang, it["text"], prompts[cond], dst))
                    except Exception as e:  # noqa: BLE001
                        rec["error"] = f"{type(e).__name__}: {e}"[:300]
                    results["renders"].append(rec)
                    print(f"  {lang} {cond} {it['id']}: {rec.get('gen_seconds')} s {rec['error'] or ''}", flush=True)
            # bootstrap: cut the engine's own hi-reference render of the chosen sentence
            src = next(r for r in results["renders"] if r["lang"] == lang and r["cond"] == "hi"
                       and r["id"].endswith(BOOT_ID_SUFFIX))
            ref_item = next(i for i in items if i["id"].endswith(BOOT_ID_SUFFIX))
            audio, sr = sf.read(src["wav"], dtype="float32")
            sel = dh.select_reference(audio, sr, ref_item["text"])
            clip_path = os.path.join(out, "wavs", lang, f"bootref_{lang}.wav")
            sf.write(clip_path, sel.audio, sr)
            text = sel.text or sel.approximate_text
            results["boot"][lang] = {"source_id": ref_item["id"], "text": ref_item["text"], "prompt_text": text,
                                     "whole_clip": sel.choice.whole, "clip_seconds": round(len(sel.audio) / sr, 2),
                                     "source_audio_seconds": src["audio_seconds"], "clip": clip_path}
            print(f"  boot {lang}: clip {len(sel.audio) / sr:.2f} s whole={sel.choice.whole}", flush=True)
            bp = dh.prompt_from_clip(sel.audio, sr, text)
            for it in items:
                if it["id"].endswith(BOOT_ID_SUFFIX):
                    continue
                dst = os.path.join(out, "wavs", lang, f"{it['id']}_boot.wav")
                rec = {"lang": lang, "cond": "boot", "id": it["id"], "text": it["text"], "nums": it["number_words"],
                       "wav": dst, "error": None}
                try:
                    rec.update(gen(lang, it["text"], bp, dst))
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                results["renders"].append(rec)
                print(f"  {lang} boot {it['id']}: {rec.get('gen_seconds')} s {rec['error'] or ''}", flush=True)
    finally:
        client.close()
        print("worker closed", flush=True)
    json.dump(results, open(os.path.join(out, "renders.json"), "w"), ensure_ascii=False, indent=1)
    return results


def asr_locked(jobs, out, lock_dir):
    deadline = time.time() + 20 * 60
    while True:
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            if time.time() > deadline:
                sys.exit("could not get the ASR lock in 20 minutes")
            print("ASR lock busy, waiting 10 s", flush=True)
            time.sleep(10)
    try:
        jf, rf = os.path.join(out, "asr_jobs.json"), os.path.join(out, "asr_out.json")
        json.dump(jobs, open(jf, "w"))
        t0 = time.time()
        subprocess.run([MLX_PY, "-c", _ASR_SCRIPT, jf, rf, ASR_MODEL], check=True, stdout=subprocess.DEVNULL)
        print(f"ASR done in {time.time() - t0:.0f} s", flush=True)
        return json.load(open(rf))
    finally:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def score_and_tables(results, hyps, out):
    ok = [r for r in results["renders"] if not r["error"]]
    for r in ok:
        r["hyp"] = hyps[r["wav"]]
        r["cer"] = cer(r["text"], normalise_hyp(r["hyp"], r["nums"]))
        r["script_share"], r["script"] = script_share(r["hyp"], r["lang"])
    results["scored"] = ok
    L = ["# Reference-language experiment: tables", "",
         f"num_step={results['steps']}. Mean read-back CER. 'excl' = the 7 sentences other than the bootstrap sentence "
         "(id -02), comparable across all three conditions; 'all8' only exists for hi and en.", "",
         "| lang | hi all8 | hi excl | en all8 | en excl | boot excl | boot clip s | boot source CER (hi) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    summary = {}
    for lang in LANGS:
        g = lambda c, excl: [r["cer"] for r in ok if r["lang"] == lang and r["cond"] == c  # noqa: E731
                             and not (excl and r["id"].endswith(BOOT_ID_SUFFIX))]
        src = next(r for r in ok if r["lang"] == lang and r["cond"] == "hi" and r["id"].endswith(BOOT_ID_SUFFIX))
        s = {"hi_all8": mean(g("hi", False)), "hi_excl": mean(g("hi", True)), "en_all8": mean(g("en", False)),
             "en_excl": mean(g("en", True)), "boot_excl": mean(g("boot", True)), "boot_src_cer": src["cer"],
             "boot_clip_s": results["boot"][lang]["clip_seconds"]}
        summary[lang] = s
        L.append(f"| {lang} | {s['hi_all8']:.3f} | {s['hi_excl']:.3f} | {s['en_all8']:.3f} | {s['en_excl']:.3f} | "
                 f"{s['boot_excl']:.3f} | {s['boot_clip_s']} | {s['boot_src_cer']:.3f} |")
    results["summary"] = summary
    for lang in LANGS:
        L += ["", f"## {lang}: per sentence CER (script share of the transcript in brackets)", "",
              "| id | hi | en | boot |", "| --- | --- | --- | --- |"]
        for it in load_items(lang):
            cells = []
            for c in ("hi", "en", "boot"):
                r = next((x for x in ok if x["lang"] == lang and x["cond"] == c and x["id"] == it["id"]), None)
                cells.append("-" if r is None else f"{r['cer']:.3f} ({r['script_share']:.2f})")
            L.append(f"| {it['id']} | " + " | ".join(cells) + " |")
    for lang in ("ml", "pa"):
        L += ["", f"## {lang}: five worst renders with the Whisper transcript", ""]
        for r in sorted([x for x in ok if x["lang"] == lang], key=lambda x: -x["cer"])[:5]:
            L += [f"- {r['id']} {r['cond']} CER {r['cer']:.3f}, script {r['script']} {r['script_share']:.2f}, "
                  f"audio {r['audio_seconds']} s", f"  - text: {r['text']}", f"  - heard: {r['hyp']}"]
    open(os.path.join(out, "tables.md"), "w").write("\n".join(L) + "\n")
    json.dump(results, open(os.path.join(out, "results.json"), "w"), ensure_ascii=False, indent=1)
    print("\n".join(L[:12]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-render", action="store_true", help="reuse renders.json and the wavs in --out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.skip_render:
        results = json.load(open(os.path.join(a.out, "renders.json")))
    else:
        results = render(a.out)
    jobs = [{"wav": r["wav"], "lang": r["lang"]} for r in results["renders"] if not r["error"]]
    hyps = asr_locked(jobs, a.out, os.path.join(a.out, "..", "asr.lock"))
    for f in ("asr_jobs.json", "asr_out.json"):
        try:
            os.remove(os.path.join(a.out, f))
        except OSError:
            pass
    score_and_tables(results, hyps, a.out)


if __name__ == "__main__":
    main()
