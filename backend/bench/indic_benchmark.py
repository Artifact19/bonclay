#!/usr/bin/env python3
"""Open Indian-language voice benchmark: original sentences (bench/data/indic_benchmark/*.json, CC-BY-4.0) read back
through an ASR, one engine at a time.

    ./bonclay bench indic [--quick] [--engines dhvaani kokoro_hi] [--langs mr hi ...] [--limit N] [--refs hi en]
    backend/.venv/bin/python backend/bench/indic_benchmark.py --listening ratings.csv --report-only RESULT.json

Method (docs/BENCHMARK_INDIC.md has the full text and the limits):
  * Engines run strictly sequentially, each started and stopped before the next; before each start the app's own memory
    guard (core/memory_guard.py) is asked for headroom and the engine is skipped, with the reason, when it refuses.
    The ASR runs last, after every engine is down.
  * Voices are synthetic Kokoro references (hf_alpha speaking Hindi, am_michael speaking English) made fresh for each
    run. No real person's recording and nothing from backend/data is read.
  * Score: character error rate of the read-back transcript against the sentence (bench/scoring.py), digits mapped back
    to the sentence's number words. Marathi also gets the leak scan (bench/dhvaani_marathi_check.leak_scan).
  * ASR CONFOUND: the only ASR here is mlx-whisper large-v3-turbo, which writes Marathi (and sometimes Gujarati,
    Punjabi, Bengali) toward Hindi-like spelling. A CER is therefore an upper bound on the engine's error for words the
    ASR can misspell, and a lower bound on dialect leakage. Every item carries a listening-check slot: the run writes
    listening_sheet.csv (engine, id, wav, rating); fill `rating` with ok / bad by ear and pass --listening to publish
    the column. Nothing is marked listened until a human did.
"""
import argparse
import csv
import datetime as dt
import gc
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
sys.path.insert(0, HERE)

from scoring import cer, normalise_hyp  # noqa: E402

DATA_DIR = Path(HERE) / "data" / "indic_benchmark"
REPO_ROOT = Path(BACKEND).parent
DOC_PATH = REPO_ROOT / "docs" / "BENCHMARK_INDIC.md"
LANGS = ["hi", "hinglish", "mr", "ta", "bn", "te", "kn", "gu", "ml", "pa"]
LANG_NAMES = {"hi": "Hindi", "hinglish": "Hinglish", "mr": "Marathi", "ta": "Tamil", "bn": "Bengali", "te": "Telugu",
              "kn": "Kannada", "gu": "Gujarati", "ml": "Malayalam", "pa": "Punjabi"}
ENGINES = {  # key -> (memory-guard engine name, languages it can read, clones a voice)
    "dhvaani": ("dhvaani", set(LANGS), True),
    "kokoro_hi": ("kokoro", {"hi", "hinglish"}, False),
    # Hindi is the only Indian language the app's Chatterbox worker clones (services/chatterbox_clone.LANGUAGES)
    "chatterbox": ("chatterbox_ml", {"hi", "hinglish"}, True),
    # community fine-tune of Kyutai Pocket TTS; own virtualenv (backend/.venv-pocket); no Bengali, Gujarati or Punjabi
    "pocket": ("pocket", {"hi", "hinglish", "mr", "ta", "te", "kn", "ml"}, True),
    # Indic-Speak with its language model quantized by MLX (bench/indic_speak_quant.py): same sentences, same speakers
    "indic_speak_q4": ("chatterbox_ml", {l for l in LANGS}, False),
    "indic_speak_q8": ("chatterbox_ml", {l for l in LANGS}, False),
    # OmniVoice (Xiaomi): 600+ languages, weights CC-BY-NC = research use only; own virtualenv backend/.venv-omni; Apple GPU
    "omnivoice": ("chatterbox_ml", {l for l in LANGS}, True),
    # preset voices (no cloning): the engine's default female speaker for each language, so it answers "which engine
    # reads the language better", not "which clones better"
    "indic_speak": ("indic_speak", {l for l in LANGS if l != "hinglish"} | {"hinglish"}, False),
}
REFS = {"hi": ("h", "hf_alpha", "नमस्ते, आप रिसेप्शन पर पहुंचे हैं, कृपया अपना नाम और मिलने का दिन बताइए।"),
        "en": ("a", "am_michael", "Good morning, you have reached the front desk, please tell me your name and the "
                                  "day you would like to visit.")}
ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
REQUIRED_FIELDS = ("language", "id", "text", "number_words", "kind")


# --------------------------------------------------------------------------------------------- data

def load_sentences(langs=None, data_dir=DATA_DIR):
    """All benchmark items for `langs` (default all), validated. `spoken` (Devanagari rendering of a Hinglish line) is
    what an engine reads; `text` is what is scored against for every other language."""
    out = []
    for lang in langs or LANGS:
        rows = json.loads((Path(data_dir) / f"{lang}.json").read_text(encoding="utf-8"))
        for r in rows:
            missing = [k for k in REQUIRED_FIELDS if k not in r]
            if missing or r["language"] != lang or not r["text"].strip():
                raise ValueError(f"{lang}.json item {r.get('id')}: bad or missing {missing or 'language/text'}")
            out.append(r)
    ids = [r["id"] for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sentence ids")
    return out


def read_text(item):
    return item.get("spoken") or item["text"]


def asr_language(lang):
    return "hi" if lang == "hinglish" else lang


# --------------------------------------------------------------------------------------------- scoring / report

def score_item(item, transcript):
    """CER of one read-back (against the Devanagari reading for Hinglish), plus the Marathi leak scan."""
    ref = read_text(item)
    norm = normalise_hyp(transcript, item["number_words"])
    out = {"hyp": transcript, "cer": cer(ref, norm)}
    if item["language"] == "mr":
        from dhvaani_marathi_check import leak_scan
        pairs, _subs, flags = leak_scan(ref, norm)
        out["leak_pairs"], out["leak_flags"] = pairs, flags
    return out


def load_ratings(path):
    """listening_sheet.csv filled in by a person: {(engine, id): 'ok' | 'bad'}. Other values are ignored."""
    ratings = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = (row.get("rating") or "").strip().lower()
            if r in ("ok", "bad"):
                ratings[(row["engine"], row["id"])] = r
    return ratings


def summarise(items, ratings=None):
    """{(engine, lang): {n, cer_mean, cer_median, rtf_mean, listened, listened_ok}}"""
    import statistics
    groups = {}
    for it in items:
        if it.get("error") or it.get("cer") is None:
            continue
        groups.setdefault((it["engine"], it["language"]), []).append(it)
    out = {}
    for key, g in groups.items():
        cers = [i["cer"] for i in g]
        rated = [(ratings or {}).get((i["engine"], i["id"])) for i in g]
        rated = [r for r in rated if r]
        out[key] = {"n": len(g), "cer_mean": round(statistics.mean(cers), 3), "cer_median": round(statistics.median(cers), 3),
                    "rtf_mean": round(statistics.mean([i["rtf"] for i in g if i.get("rtf")] or [0]), 2) or None,
                    "listened": len(rated), "listened_ok": sum(r == "ok" for r in rated)}
    return out


def render_markdown(result, ratings=None):
    items, st = result["items"], result["settings"]
    summ = summarise(items, ratings)
    engines = list(dict.fromkeys(k[0] for k in summ))
    L = ["# Indian-language voice benchmark", "",
         f"Generated by `backend/bench/indic_benchmark.py` on {result['timestamp']} (commit {result['git'].get('commit')}"
         f"{', dirty tree' if result['git'].get('dirty_working_tree') else ''}), {result['machine'].get('cpu')}, "
         f"{result['machine'].get('ram_gb')} GB RAM. Raw numbers: `{result.get('json_name', 'bench/results/')}`."
         + (" **Quick smoke run: the numbers are not meaningful.**" if result.get("quick") else ""), "",
         "## What this measures", "",
         "Each engine reads original clinic and front-desk sentences (`backend/bench/data/indic_benchmark/*.json`, "
         "data licence CC-BY-4.0). The audio is transcribed by an ASR and the transcript is compared with the sentence: "
         "character error rate (CER) over letters and vowel signs (`backend/bench/scoring.py`; digits are mapped back "
         "to the sentence's number words; spaces and punctuation ignored). Lower is better. Voices are synthetic "
         "(Kokoro `hf_alpha` Hindi, `am_michael` English), never a real person.", "",
         "## Limits you must read before quoting a number", "",
         f"1. **The ASR is part of the score.** Only `{st['asr_model']}` is used. It writes Marathi (and at times "
         "Gujarati, Punjabi and Bengali) toward Hindi-like spelling, so a high CER can be the ASR and a low CER can hide "
         "an accent that sounds Hindi. For Marathi the leak scan lists words where ळ or ऱ्ह was heard as something else; "
         "the same caveat applies to it.",
         "2. **Listening check.** The `Listened` column counts sentences a person rated by ear (ok / total rated). "
         "`0 of 0` means nobody has listened yet; treat the CER as unconfirmed.",
         "3. One run on one 16 GB Apple M-series laptop, one synthetic reference per type, 16 sampling steps. "
         "No confidence intervals; with 6 to 12 sentences per language a difference of a few points is noise.",
         "4. The sentences were written for this repo and not reviewed by a native speaker of each language. "
         "Corrections are welcome (open an issue or PR against the JSON files).",
         "5. CER is only comparable between engines within one language, not across scripts.",
         "6. Engines covered: " + ", ".join(f"`{e}`" for e in dict.fromkeys(i["engine"] for i in items)) + ". Others (Indic-Speak, Chatterbox, IndicF5) "
         "are not in this run. Engines were run one at a time; an engine the memory guard refused is listed under "
         "Skipped.",
         "7. **Preset vs cloned voices.** `indic_speak` speaks its default female preset voice per language; `dhvaani` "
         "clones a synthetic reference. Same sentences, same ASR, but not the same task, so speaker similarity is not compared.",
         "8. Engines were run in separate sessions when merged (see `merged_from` in the JSON); the machine state differed "
         "slightly, so real-time factors are indicative only.", ""]
    L += ["## Results (CER, lower is better)", ""]
    head = "| Language | n | " + " | ".join(f"{e} CER (median) | {e} Listened" for e in engines) + " |"
    L += [head, "| --- | ---: | " + " | ".join("---: | ---:" for _ in engines) + " |"]
    for lang in LANGS:
        cells, n = [], 0
        for e in engines:
            s = summ.get((e, lang))
            if s:
                n = max(n, s["n"])
                cells += [f"{s['cer_mean']:.3f} ({s['cer_median']:.3f})", f"{s['listened_ok']} of {s['listened']}"]
            else:
                cells += ["n/a", "n/a"]
        if n:
            L.append(f"| {LANG_NAMES[lang]} | {n} | " + " | ".join(cells) + " |")
    skipped = result.get("skipped") or {}
    if skipped:
        L += ["", "## Skipped", ""] + [f"- `{k}`: {v}" for k, v in skipped.items()]
    mr = [i for i in items if i["language"] == "mr" and i.get("leak_flags")]
    if mr:
        L += ["", "## Marathi leak scan", "", "Flags from `leak_scan` (ळ or ऱ्ह missing, ण heard as न, ending changed):", ""]
        for i in mr[:20]:
            L.append(f"- `{i['engine']}` {i['id']}: " + "; ".join(i["leak_flags"][:4]))
    L += ["", "## Marathi lexicon", "",
          "DhVaani reads characters, so Marathi ळ and ऱ्ह can come out Hindi-like. `backend/services/pron_lexicon.py` holds "
          "spelling hints keyed by (language, engine). It is **off** (`PRON_LEXICON=on` enables it) until the owner "
          "confirms by ear that the defect is real and the hints fix it; the read-back numbers for the candidates are in "
          "`docs/RESULTS.md` (\"Marathi spelling hints\"). This table is measured without any hint.", ""]
    hindi_lexicon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hindi_lexicon_section.md")
    if os.path.exists(hindi_lexicon):  # measured by hindi_lexicon_check.py, kept as a file so a regenerated report keeps it
        L += ["", open(hindi_lexicon, encoding="utf-8").read().rstrip(), ""]
    L += ["", "## Reproduce", "", "```bash", "./bonclay bench indic            # all engines, all languages (about 20 minutes)",
          "./bonclay bench indic --quick    # smoke test", "```", "",
          "Then listen to the files named in `listening_sheet.csv`, fill the `rating` column, and run with "
          "`--listening listening_sheet.csv --report-only <result.json>`.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------------------------- engines

def memory_ok(engine_name):
    """Ask the app's memory guard for headroom; returns a reason string when it refuses, None when fine."""
    try:
        from core import memory_guard
        memory_guard.ensure_headroom(engine_name, [], unload=lambda k: None)
    except Exception as e:  # noqa: BLE001 (AppError: INSUFFICIENT_MEMORY)
        return f"{type(e).__name__}: {getattr(e, 'message', e)}"[:300]
    return None


def make_refs(workdir):
    from dhvaani_live_check import make_refs as _make
    return _make(workdir)


def render_dhvaani(items, refs, refs_to_use, work, steps):
    import soundfile as sf
    from services import dhvaani_clone as dh
    why = dh.unavailable_reason()
    if why:
        raise RuntimeError(why)
    client = dh.DhVaaniClone(dh.dhvaani_python())
    out = []
    try:
        prompts = {}
        for k in refs_to_use:
            audio, sr = sf.read(refs[k]["path"], dtype="float32")
            sel = dh.select_reference(audio, sr, refs[k]["text"])
            prompts[k] = dh.prompt_from_clip(sel.audio, sr, sel.text)
        client.generate_voice_clone(read_text(items[0]), "hi", prompts[refs_to_use[0]], num_step=steps)  # warm-up
        for ref in refs_to_use:
            for it in items:
                if it["language"] == "hinglish":
                    code = "hi"
                else:
                    code = it["language"]
                dst = os.path.join(work, "wavs", "dhvaani", ref, f"{it['id']}.wav")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                rec = {"engine": "dhvaani" if len(refs_to_use) == 1 else f"dhvaani_{ref}", "base_engine": "dhvaani",
                       "ref": ref, "language": it["language"], "id": it["id"], "kind": it["kind"], "wav": dst,
                       "error": None, "item": it}
                t0 = time.time()
                try:
                    wavs, sr = client.generate_voice_clone(read_text(it), code, prompts[ref], num_step=steps)
                    sf.write(dst, wavs[0], sr)
                    gen, dur = time.time() - t0, len(wavs[0]) / sr
                    rec.update({"gen_seconds": round(gen, 2), "audio_seconds": round(dur, 2), "rtf": round(gen / max(dur, 1e-6), 2)})
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                out.append(rec)
                print(f"  dhvaani/{ref} {it['id']}: {rec.get('gen_seconds')} s {rec['error'] or ''}", flush=True)
    finally:
        client.close()
    return out


def render_kokoro_hi(items, work):
    import numpy as np
    import soundfile as sf
    from services.espeak import configure_espeak
    configure_espeak()
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M")
    out = []
    for it in items:
        dst = os.path.join(work, "wavs", "kokoro_hi", f"{it['id']}.wav")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        rec = {"engine": "kokoro_hi", "base_engine": "kokoro_hi", "ref": None, "language": it["language"], "id": it["id"],
               "kind": it["kind"], "wav": dst, "error": None, "item": it}
        t0 = time.time()
        try:
            parts = [x.numpy() if hasattr(x, "numpy") else np.asarray(x) for _, _, x in pipe(read_text(it), voice="hf_alpha", speed=1.0)
                     if x is not None]
            wav = np.concatenate(parts)
            sf.write(dst, wav, 24000)
            gen, dur = time.time() - t0, len(wav) / 24000
            rec.update({"gen_seconds": round(gen, 2), "audio_seconds": round(dur, 2), "rtf": round(gen / max(dur, 1e-6), 2)})
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        out.append(rec)
        print(f"  kokoro_hi {it['id']}: {rec.get('gen_seconds')} s {rec['error'] or ''}", flush=True)
    del pipe
    gc.collect()
    return out


def render_chatterbox(items, refs, ref, work):
    """Chatterbox Multilingual (Resemble AI) through the app's own worker client; Hindi/Hinglish only, on CPU."""
    import soundfile as sf
    from services import chatterbox_clone as cb
    why = cb.unavailable_reason()
    if why:
        raise RuntimeError(why)
    client = cb.ChatterboxClone(cb.chatterbox_python())
    out = []
    try:
        audio, sr = sf.read(refs[ref]["path"], dtype="float32")
        prompt = client.create_voice_clone_prompt([(audio, sr)])
        client.generate_voice_clone(read_text(items[0]), "hi", prompt)  # warm-up (a one-word text is rejected)
        for it in items:
            dst = os.path.join(work, "wavs", "chatterbox", f"{it['id']}.wav")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            rec = {"engine": "chatterbox", "base_engine": "chatterbox", "ref": ref, "language": it["language"],
                   "id": it["id"], "kind": it["kind"], "wav": dst, "error": None, "item": it}
            t0 = time.time()
            try:
                wavs, osr = client.generate_voice_clone(read_text(it), "hi", prompt)
                sf.write(dst, wavs[0], osr)
                gen, dur = time.time() - t0, len(wavs[0]) / osr
                rec.update({"gen_seconds": round(gen, 2), "audio_seconds": round(dur, 2), "rtf": round(gen / max(dur, 1e-6), 2)})
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            out.append(rec)
            print(f"  chatterbox {it['id']}: {rec.get('gen_seconds')} s {rec['error'] or ''}", flush=True)
    finally:
        client.close()
    return out


def render_pocket(items, refs, ref, work, model_dir):
    """Pocket TTS multilingual in its own virtualenv through bench/pocket_worker.py (CPU)."""
    py = os.path.join(BACKEND, ".venv-pocket", "bin", "python")
    if not os.path.isfile(py) or not model_dir or not os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        raise RuntimeError("Pocket TTS is not set up: need backend/.venv-pocket (pip install pocket-tts) and --pocket-dir "
                           "with config.yaml, model.safetensors, tokenizer.model from Thanushttk/pocket-tts-multilingual")
    jobs = [{"id": it["id"], "text": read_text(it), "wav": os.path.join(work, "wavs", "pocket", f"{it['id']}.wav")} for it in items]
    jf, rf = os.path.join(work, "pocket_jobs.json"), os.path.join(work, "pocket_out.json")
    json.dump(jobs, open(jf, "w"), ensure_ascii=False)
    subprocess.run([py, os.path.join(HERE, "pocket_worker.py"), jf, rf, "--model-dir", model_dir, "--ref", refs[ref]["path"]],
                   check=True)
    res = json.load(open(rf))
    out = []
    for it, job in zip(items, jobs):
        r = res.get(it["id"], {"error": "no result"})
        rec = {"engine": "pocket", "base_engine": "pocket", "ref": ref, "language": it["language"], "id": it["id"],
               "kind": it["kind"], "wav": job["wav"], "error": r.get("error"), "item": it}
        if not rec["error"]:
            rec.update({"gen_seconds": r["gen_seconds"], "audio_seconds": r["audio_seconds"],
                        "rtf": round(r["gen_seconds"] / max(r["audio_seconds"], 1e-6), 2)})
        out.append(rec)
    return out


def render_indic_quant(items, work, variant, quant_dir):
    """Indic-Speak with an MLX-quantized language model: generate tokens in .venv-mlx, decode audio in .venv-indic."""
    from services import indic_speak as isp
    snaps = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--bodhan-ai--indic-speak/snapshots/*"))
    mlx_model = os.path.join(quant_dir or "", variant)
    py_mlx, py_indic = (os.path.join(BACKEND, ".venv-mlx", "bin", "python"), os.path.join(BACKEND, ".venv-indic", "bin", "python"))
    if not (snaps and os.path.isdir(mlx_model) and os.path.isfile(py_mlx) and os.path.isfile(py_indic)):
        raise RuntimeError(f"quantized Indic-Speak needs the cached model, .venv-mlx, .venv-indic and --quant-dir/{variant} "
                           "(made with: python -m mlx_lm convert --hf-path SNAPSHOT --mlx-path DIR/q4 -q --q-bits 4)")
    d = os.path.join(work, f"indic_speak_{variant}")
    os.makedirs(d, exist_ok=True)
    jobs = [{"id": it["id"], "text": read_text(it), "wav": os.path.join(d, "wavs", f"{it['id']}.wav"),
             "speaker": isp.default_speaker("hi" if it["language"] == "hinglish" else it["language"])} for it in items]
    jf, ids, rf = (os.path.join(d, n) for n in ("jobs.json", "ids.json", "out.json"))
    json.dump(jobs, open(jf, "w"), ensure_ascii=False)
    script = os.path.join(HERE, "indic_speak_quant.py")
    subprocess.run([py_mlx, script, "gen", jf, ids, "--mlx-model", mlx_model, "--snapshot", snaps[0]], check=True,
                   env={**os.environ, "PYTHONPATH": BACKEND})
    subprocess.run([py_indic, script, "decode", ids, rf, "--snapshot", snaps[0], "--jobs-for-paths", jf], check=True,
                   env={**os.environ, "PYTHONPATH": BACKEND})
    res = json.load(open(rf))
    out = []
    for it, job in zip(items, jobs):
        r = res.get(it["id"], {"error": "no result"})
        rec = {"engine": f"indic_speak_{variant}", "base_engine": f"indic_speak_{variant}", "ref": None,
               "speaker": job["speaker"], "language": it["language"], "id": it["id"], "kind": it["kind"],
               "wav": job["wav"], "error": r.get("error"), "item": it}
        if not rec["error"]:
            total = r["lm_seconds"] + r["decode_seconds"]
            rec.update({"gen_seconds": round(total, 2), "audio_seconds": r["audio_seconds"], "peak_gb": r["peak_gb"],
                        "rtf": round(total / max(r["audio_seconds"], 1e-6), 2)})
        out.append(rec)
    return out


def render_omnivoice(items, refs, ref, work):
    """OmniVoice in its own virtualenv through bench/omnivoice_worker.py (Apple GPU, float16). Research use only."""
    py = os.path.join(BACKEND, ".venv-omni", "bin", "python")
    if not os.path.isfile(py):
        raise RuntimeError("OmniVoice is not set up: need backend/.venv-omni (torch==2.8.0 torchaudio==2.8.0, pip install omnivoice)")
    jobs = [{"id": it["id"], "text": read_text(it), "language": "hi" if it["language"] == "hinglish" else it["language"],
             "wav": os.path.join(work, "wavs", "omnivoice", f"{it['id']}.wav")} for it in items]
    jf, rf = os.path.join(work, "omni_jobs.json"), os.path.join(work, "omni_out.json")
    json.dump(jobs, open(jf, "w"), ensure_ascii=False)
    subprocess.run([py, os.path.join(HERE, "omnivoice_worker.py"), jf, rf, "--ref", refs[ref]["path"], "--ref-text", refs[ref]["text"]],
                   check=True)
    res = json.load(open(rf))
    out = []
    for it, job in zip(items, jobs):
        r = res.get(it["id"], {"error": "no result"})
        rec = {"engine": "omnivoice", "base_engine": "omnivoice", "ref": ref, "language": it["language"], "id": it["id"],
               "kind": it["kind"], "wav": job["wav"], "error": r.get("error"), "item": it}
        if not rec["error"]:
            rec.update({"gen_seconds": r["gen_seconds"], "audio_seconds": r["audio_seconds"],
                        "rtf": round(r["gen_seconds"] / max(r["audio_seconds"], 1e-6), 2)})
        out.append(rec)
    return out


def render_indic_speak(items, work):
    """Indic-Speak (Bodhan AI / AI4Bharat), the app's own worker client, default female preset voice per language."""
    import soundfile as sf
    from services import indic_speak as isp
    why = isp.indic_speak_unavailable_reason()
    if why:
        raise RuntimeError(why)
    model = isp.IndicSpeakModel(isp.indic_speak_python())
    out = []
    try:
        isp.synthesize(model, "नमस्ते।", isp.default_speaker("hi"))  # warm-up
        for it in items:
            code = "hi" if it["language"] == "hinglish" else it["language"]
            speaker = isp.default_speaker(code)
            dst = os.path.join(work, "wavs", "indic_speak", f"{it['id']}.wav")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            rec = {"engine": "indic_speak", "base_engine": "indic_speak", "ref": None, "speaker": speaker,
                   "language": it["language"], "id": it["id"], "kind": it["kind"], "wav": dst, "error": None, "item": it}
            t0 = time.time()
            try:
                wav, sr = isp.synthesize(model, read_text(it), speaker)
                sf.write(dst, wav, sr)
                gen, dur = time.time() - t0, len(wav) / sr
                rec.update({"gen_seconds": round(gen, 2), "audio_seconds": round(dur, 2), "rtf": round(gen / max(dur, 1e-6), 2)})
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            out.append(rec)
            print(f"  indic_speak {it['id']}: {rec.get('gen_seconds')} s {rec['error'] or ''}", flush=True)
    finally:
        model.close()
    return out


def run_asr(records, work):
    from indic_clone_trial import MLX_PY, _ASR_SCRIPT
    ok = [r for r in records if not r["error"]]
    jf, rf = os.path.join(work, "asr_jobs.json"), os.path.join(work, "asr_out.json")
    json.dump([{"wav": r["wav"], "lang": asr_language(r["language"])} for r in ok], open(jf, "w"))
    subprocess.run([MLX_PY, "-c", _ASR_SCRIPT, jf, rf, ASR_MODEL], check=True, stdout=subprocess.DEVNULL)
    hyps = json.load(open(rf))
    for f in (jf, rf):
        os.remove(f)
    return hyps


def asr_problem():
    from dhvaani_live_check import asr_usable
    return asr_usable()


# --------------------------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", nargs="+", default=list(ENGINES), choices=list(ENGINES))
    ap.add_argument("--langs", nargs="+", default=LANGS, choices=LANGS)
    ap.add_argument("--limit", type=int, default=0, help="at most N sentences per language (0 = all)")
    ap.add_argument("--refs", nargs="+", default=["hi"], choices=["hi", "en"], help="reference types for cloning engines")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="2 sentences per language, 8 steps (numbers are not meaningful)")
    ap.add_argument("--work", help="folder for audio and the listening sheet (default: bench/results/indic-<stamp>)")
    ap.add_argument("--listening", help="listening_sheet.csv with the rating column filled in")
    ap.add_argument("--report-only", metavar="RESULT_JSON", help="re-render the markdown from a saved result")
    ap.add_argument("--md", default=str(DOC_PATH))
    ap.add_argument("--pocket-dir", default=os.getenv("POCKET_TTS_DIR"), help="folder with the Pocket TTS multilingual files")
    ap.add_argument("--quant-dir", default=os.getenv("INDIC_SPEAK_MLX_DIR"), help="folder holding the MLX q4/q8 conversions")
    ap.add_argument("--merge", metavar="RESULT_JSON", help="add this run's engines to an earlier result (same sentences, "
                    "same ASR) and write one combined report; used to run engines in separate sessions")
    a = ap.parse_args(argv)

    ratings = load_ratings(a.listening) if a.listening else None
    if a.report_only:
        result = json.loads(Path(a.report_only).read_text(encoding="utf-8"))
        Path(a.md).write_text(render_markdown(result, ratings), encoding="utf-8")
        print(f"wrote {a.md}")
        return 0

    from common import RESULTS_DIR, git_info, machine_info, now_stamp, write_result
    if a.quick:
        a.limit, a.steps = 2, 8
    problem = asr_problem()
    if problem:
        sys.exit(f"read-back is not available: {problem}")
    items = load_sentences(a.langs)
    if a.limit:
        kept, seen = [], {}
        for it in items:
            seen[it["language"]] = seen.get(it["language"], 0) + 1
            if seen[it["language"]] <= a.limit:
                kept.append(it)
        items = kept
    work = a.work or str(RESULTS_DIR / f"indic-{now_stamp()}")
    os.makedirs(work, exist_ok=True)
    print(f"{len(items)} sentences, engines {a.engines}, one at a time; work folder {work}", flush=True)

    refs = make_refs(work)
    gc.collect()
    records, skipped = [], {}
    for eng in a.engines:  # strictly one at a time; each is fully stopped before the next starts
        guard_name, langs, _clones = ENGINES[eng]
        mine = [i for i in items if i["language"] in langs]
        if not mine:
            skipped[eng] = "no selected language is supported"
            continue
        reason = memory_ok(guard_name)
        if reason:
            skipped[eng] = reason
            print(f"skipping {eng}: {reason}", flush=True)
            continue
        print(f"== {eng} ==", flush=True)
        try:
            if eng == "dhvaani":
                records += render_dhvaani(mine, refs, a.refs, work, a.steps)
            elif eng == "chatterbox":
                records += render_chatterbox(mine, refs, a.refs[0], work)
            elif eng.startswith("indic_speak_q"):
                records += render_indic_quant(mine, work, eng.rsplit("_", 1)[1], a.quant_dir)
            elif eng == "omnivoice":
                records += render_omnivoice(mine, refs, a.refs[0], work)
            elif eng == "pocket":
                records += render_pocket(mine, refs, a.refs[0], work, a.pocket_dir)
            elif eng == "indic_speak":
                records += render_indic_speak(mine, work)
            else:
                records += render_kokoro_hi(mine, work)
        except Exception as e:  # noqa: BLE001
            skipped[eng] = f"{type(e).__name__}: {e}"[:300]
            print(f"{eng} failed: {skipped[eng]}", flush=True)
    t0 = time.time()
    hyps = run_asr(records, work) if any(not r["error"] for r in records) else {}
    result_items = []
    for r in records:
        it = r.pop("item")
        row = {k: v for k, v in r.items() if k != "base_engine"}
        if not r["error"]:
            row.update(score_item(it, hyps[r["wav"]]))
        row["listening_check"] = (ratings or {}).get((row["engine"], row["id"]))
        result_items.append(row)
    result = {"name": "indic_benchmark", "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "git": git_info(),
              "machine": machine_info(), "quick": bool(a.quick), "skipped": skipped, "items": result_items,
              "settings": {"engines": a.engines, "langs": a.langs, "refs": a.refs, "steps": a.steps, "limit": a.limit,
                           "asr_model": ASR_MODEL, "asr_seconds": round(time.time() - t0, 1),
                           "references": {k: {"voice": v[1], "synthetic": True} for k, v in REFS.items()}},
          }
    if a.merge:
        prior = json.loads(Path(a.merge).read_text(encoding="utf-8"))
        mine = {(r["engine"], r["id"]) for r in result_items}
        result["items"] = [i for i in prior["items"] if (i["engine"], i["id"]) not in mine] + result_items
        result["skipped"] = {**prior.get("skipped", {}), **skipped}
        result["settings"]["engines"] = list(dict.fromkeys(prior["settings"]["engines"] + a.engines))
        result["merged_from"] = {"file": Path(a.merge).name, "timestamp": prior["timestamp"], "commit": prior["git"].get("commit")}
        result_items = result["items"]
    path = write_result(result)
    result["json_name"] = f"bench/results/{path.name}"
    texts = {i["id"]: i["text"] for i in load_sentences()}
    with open(os.path.join(work, "listening_sheet.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["engine", "id", "language", "wav", "text", "rating"])
        for r in result_items:
            if not r.get("error"):
                w.writerow([r["engine"], r["id"], r["language"], r["wav"], texts.get(r["id"], ""), ""])
    if not a.quick:
        Path(a.md).write_text(render_markdown(result, ratings), encoding="utf-8")
        print(f"wrote {a.md}")
    else:
        print(render_markdown(result, ratings))
    print(f"result {path}; listening sheet {os.path.join(work, 'listening_sheet.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
