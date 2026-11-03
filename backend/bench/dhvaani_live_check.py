#!/usr/bin/env python3
"""Live check of the DhVaani engine WITHOUT the API server: starts only the DhVaani worker (through the same client class
the app uses, services/dhvaani_clone.py), clones two synthetic voices (Kokoro am_michael speaking English, hf_alpha
speaking Hindi: never a real person's voice) and speaks one sentence in each of Marathi, Tamil, Bengali, Telugu, Kannada,
Gujarati, Malayalam and Punjabi. Then it stops the worker, transcribes every clip with mlx-whisper (backend/.venv-mlx,
skipped with a message when that environment is not usable) and reports seconds per sentence, the real-time factor and
the read-back character error rate (and ECAPA speaker similarity to the reference when the model is cached).

    backend/.venv/bin/python backend/bench/dhvaani_live_check.py --out /path/to/dhvaani_live [--steps 16] [--skip-asr]

Writes <out>/<lang>_from_hi.wav, <out>/<lang>_from_en.wav, <out>/ref_en.wav, <out>/ref_hi.wav and <out>/results.json.
Needs backend/.venv-dhvaani (./bonclay setup --dhvaani) and the Kokoro model of the main environment. It never touches
backend/data.
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

from indic_clone_trial import ASR_MODEL, MLX_PY, REFS, SENTENCES, _ASR_SCRIPT, cer, normalise_hyp  # noqa: E402
from services import dhvaani_clone as dh  # noqa: E402

# One clinic sentence per language. mr/ta/bn/te are the trial's first sentence; the rest are new. The number words are
# what Whisper's digits are mapped back to before scoring.
LIVE = {lang: SENTENCES[lang][0] for lang in ("mr", "ta", "bn", "te")}
LIVE.update({
    "kn": ("ವೈದ್ಯರು ನಾಳೆ ಬೆಳಿಗ್ಗೆ ಹತ್ತು ಗಂಟೆಗೆ ನಿಮ್ಮನ್ನು ನೋಡುತ್ತಾರೆ.", ["ಹತ್ತು"]),
    "gu": ("ડૉક્ટર આવતીકાલે સવારે દસ વાગ્યે તમને ક્લિનિકમાં જોશે.", ["દસ"]),
    "ml": ("ഡോക്ടർ നാളെ രാവിലെ പത്ത് മണിക്ക് നിങ്ങളെ ക്ലിനിക്കിൽ കാണും.", ["പത്ത്"]),
    "pa": ("ਡਾਕਟਰ ਕੱਲ੍ਹ ਸਵੇਰੇ ਦਸ ਵਜੇ ਤੁਹਾਨੂੰ ਕਲੀਨਿਕ ਵਿੱਚ ਦੇਖਣਗੇ।", ["ਦਸ"]),
})
ORDER = ("mr", "ta", "bn", "te", "kn", "gu", "ml", "pa")


def make_refs(out):
    """The two synthetic references (Kokoro): written once, 6-8 s each, with their exact transcripts."""
    from services.espeak import configure_espeak
    configure_espeak()
    from kokoro import KPipeline
    meta = {}
    for key, (code, voice, text) in REFS.items():
        path = os.path.join(out, f"ref_{key}.wav")
        pipe = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M")
        parts = [x.numpy() if hasattr(x, "numpy") else np.asarray(x) for _, _, x in pipe(text, voice=voice, speed=1.0)
                 if x is not None]
        wav = np.concatenate(parts)
        sf.write(path, wav, 24000)
        meta[key] = {"voice": voice, "text": text, "seconds": round(len(wav) / 24000, 2), "path": path}
    return meta


def asr_usable():
    if not os.path.isfile(MLX_PY):
        return "backend/.venv-mlx does not exist"
    if not glob.glob(os.path.join(BACKEND, ".venv-mlx", "lib", "python3*", "site-packages", "mlx_whisper")):
        return "mlx-whisper is not installed in backend/.venv-mlx yet"
    if subprocess.run([MLX_PY, "-c", "import mlx_whisper"], capture_output=True).returncode != 0:
        return "mlx_whisper does not import in backend/.venv-mlx"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--skip-asr", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    reason = dh.unavailable_reason()
    if reason:
        sys.exit(f"DhVaani is not available: {reason}")
    print("making the synthetic references with Kokoro ...", flush=True)
    refs = make_refs(a.out)
    for k, m in refs.items():
        print(f"  ref {k}: {m['voice']} {m['seconds']} s  {m['text'][:50]}")

    results = {"steps": a.steps, "refs": {k: {x: v for x, v in m.items() if x != "path"} for k, m in refs.items()},
               "free_mb_before": dh.free_memory_mb(), "loadavg_start": os.getloadavg(), "items": []}
    t0 = time.time()
    client = dh.DhVaaniClone(dh.dhvaani_python())
    results["worker_start_seconds"] = round(time.time() - t0, 1)
    results["device"] = getattr(client, "device", None)
    print(f"worker ready in {results['worker_start_seconds']} s on {results['device']}", flush=True)
    try:
        prompts = {}
        for k, m in refs.items():
            audio, sr = sf.read(m["path"], dtype="float32")
            sel = dh.select_reference(audio, sr, m["text"])  # the app's own reference selection
            assert sel.choice.whole and sel.text == m["text"], "a 6-8 s reference must be used whole"
            prompts[k] = dh.prompt_from_clip(sel.audio, sr, sel.text)
        # one throw-away sentence: the first request pays for warm-up (kernel compilation), which is not a render
        t0 = time.time()
        client.generate_voice_clone(LIVE["mr"][0], "mr", prompts["hi"], num_step=a.steps)
        results["first_request_seconds"] = round(time.time() - t0, 1)
        print(f"first (warm-up) request {results['first_request_seconds']} s", flush=True)
        for ref in ("hi", "en"):
            for lang in ORDER:
                text, nums = LIVE[lang]
                t0 = time.time()
                try:
                    wavs, sr = client.generate_voice_clone(text, lang, prompts[ref], num_step=a.steps)
                    secs = time.time() - t0
                    wav = wavs[0]
                    dst = os.path.join(a.out, f"{lang}_from_{ref}.wav")
                    sf.write(dst, wav, sr)
                    item = {"ref": ref, "lang": lang, "text": text, "nums": nums, "wav": dst,
                            "gen_seconds": round(secs, 2), "audio_seconds": round(len(wav) / sr, 2),
                            "rtf": round(secs / max(len(wav) / sr, 1e-6), 2), "error": None,
                            "loadavg": round(os.getloadavg()[0], 2)}
                except Exception as e:  # noqa: BLE001
                    item = {"ref": ref, "lang": lang, "text": text, "nums": nums, "error": f"{type(e).__name__}: {e}"[:300],
                            "gen_seconds": round(time.time() - t0, 2)}
                results["items"].append(item)
                print(f"  from_{ref} {lang}: {item.get('gen_seconds')} s for {item.get('audio_seconds')} s of audio "
                      f"(RTF {item.get('rtf')}) {item['error'] or ''}", flush=True)
        results["free_mb_after"] = dh.free_memory_mb()
    finally:
        proc = client.proc
        client.close()
        results["worker_stopped"] = proc.poll() is not None
        print(f"worker stopped: {results['worker_stopped']}", flush=True)

    ok = [it for it in results["items"] if not it["error"]]
    problem = None if a.skip_asr else asr_usable()
    if a.skip_asr or problem:
        results["asr"] = f"skipped: {problem or '--skip-asr'}"
        print(f"read-back skipped: {problem or '--skip-asr'}")
    else:
        jobs = [{"wav": it["wav"], "lang": it["lang"]} for it in ok]
        jf, rf = os.path.join(a.out, "asr_jobs.json"), os.path.join(a.out, "asr_out.json")
        json.dump(jobs, open(jf, "w"))
        t0 = time.time()
        subprocess.run([MLX_PY, "-c", _ASR_SCRIPT, jf, rf, ASR_MODEL], check=True, stdout=subprocess.DEVNULL)
        results["asr"] = {"model": ASR_MODEL, "seconds": round(time.time() - t0, 1)}
        hyps = json.load(open(rf))
        try:
            from services import speaker
            import torch
            import torchaudio

            def load16(p):
                x, sr = sf.read(p, dtype="float32")
                x = x.mean(axis=1) if x.ndim > 1 else x
                return torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy() if sr != 16000 else x

            ref_emb = {k: speaker.embed(load16(m["path"])) for k, m in refs.items()}
        except Exception as e:  # noqa: BLE001
            ref_emb = None
            results["similarity"] = f"skipped: {type(e).__name__}: {e}"[:200]
        for it in ok:
            hyp = hyps[it["wav"]]
            it["hyp"] = hyp
            it["cer"] = cer(it["text"], normalise_hyp(hyp, it["nums"]))
            if ref_emb and all(v is not None for v in ref_emb.values()):
                emb = speaker.embed(load16(it["wav"]))
                r = ref_emb[it["ref"]]
                it["sim"] = round(float(np.dot(emb, r) / (np.linalg.norm(emb) * np.linalg.norm(r))), 3)

    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), ensure_ascii=False, indent=1)
    print()
    print(f"{'ref':<4} {'lang':<4} {'gen s':>6} {'audio s':>8} {'RTF':>5} {'CER':>6} {'sim':>5}")
    for it in results["items"]:
        if it["error"]:
            print(f"{it['ref']:<4} {it['lang']:<4} ERROR {it['error']}")
            continue
        c = it.get("cer")
        print(f"{it['ref']:<4} {it['lang']:<4} {it['gen_seconds']:>6} {it['audio_seconds']:>8} {it['rtf']:>5} "
              f"{'' if c is None else f'{c:.2f}':>6} {it.get('sim', ''):>5}")


if __name__ == "__main__":
    main()
