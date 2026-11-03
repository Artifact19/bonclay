#!/usr/bin/env python3
"""40-clip accuracy and latency check: today's speech recognition vs one Whisper pass with a shorter encoder window.

Idea 3 in docs/OPTIMIZATION_RESEARCH.md: whisper-large-v3-turbo's encoder always looks at 30 s (1,500 frames), so a 5 s
question pays for 30 s. Slicing the positional embedding lets it run on an 8-10 s window, and decode(language=None)
detects the language from the encoder output it already has, so the whisper-small language-id pass is not needed.
The research measured two clips; this script is the 40-clip check it asked for.

Clips: 20 English (Kokoro af_heart) + 20 Hindi (Kokoro hf_alpha), fixed sentences from bench/readback.py, 16 kHz mono,
no real person's voice. --noisy adds a 20 dB SNR copy of every second clip.

Variants (one process, warm models, audio already decoded, as the worker does for `auto`):
  baseline   what production runs today (MLX_ASR_STRATEGY=auto): whisper-small language id, then one turbo pass for an
             Indian language or one small pass otherwise (services/mlx_asr_worker.transcribe_plan)
  turbo30    one turbo pass, language detected in the decoder, full 30 s window (isolates dropping the small model)
  trim8      one turbo pass, language detected in the decoder, window = smallest of 8/10/15/20/30 s that holds the clip
             plus 1.5 s (the research's proposal); falls back to the 30 s window when the result looks wrong
             (compression ratio > 2.4, average log-probability < -1.0, or empty text). Implementation: services/asr_window.py
  trim10     the same with windows 10/15/20/30 s (no 8 s window)

Metrics: WER (English: lowercase, punctuation removed) and CER (Hindi: Devanagari letters and marks only) with jiwer,
as corpus-level rates (total edits / total reference length); language-id correctness; fallbacks; latency (median over
--rounds interleaved rounds per clip).

Three steps, because Kokoro and jiwer live in backend/.venv and mlx-whisper in backend/.venv-mlx:
    backend/.venv/bin/python backend/bench/asr_window_check.py all --noisy          # runs the three below
    backend/.venv/bin/python backend/bench/asr_window_check.py gen --out DIR --noisy
    backend/.venv-mlx/bin/python backend/bench/asr_window_check.py run --clips DIR --json OUT.json
    backend/.venv/bin/python backend/bench/asr_window_check.py score --json OUT.json
Offline (HF_HUB_OFFLINE=1); needs the models from bash backend/scripts/setup_mlx.sh.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "services"))
sys.path.insert(0, HERE)
VARIANTS = ("baseline", "turbo30", "trim8", "trim10")
TRIM_BUCKETS = {"trim8": (8, 10, 15, 20, 30), "trim10": (10, 15, 20, 30)}


# --- gen (backend/.venv) ----------------------------------------------------------------------------------------------

def cmd_gen(a):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from readback import SENTENCES_EN, SENTENCES_HI
    from services.espeak import configure_espeak
    configure_espeak()
    from kokoro import KPipeline

    os.makedirs(a.out, exist_ok=True)
    en = KPipeline(lang_code="a")
    hi = KPipeline(lang_code="h", model=en.model)
    rng = np.random.default_rng(7)
    plan = ([("en", t, en, "af_heart") for t in SENTENCES_EN[:20]]
            + [("hi", t, hi, "hf_alpha") for t in SENTENCES_HI[:20]])
    clips, counts = [], {}
    for kind, text, pipe, voice in plan:
        i = counts[kind] = counts.get(kind, 0) + 1
        chunks = [x.numpy() if hasattr(x, "numpy") else np.asarray(x) for _, _, x in pipe(text, voice=voice) if x is not None]
        audio = resample_poly(np.concatenate(chunks).astype(np.float32), 2, 3).astype(np.float32)
        base = f"{kind}{i:02d}"
        sf.write(os.path.join(a.out, base + ".wav"), audio, 16000)
        clips.append({"id": base, "kind": kind, "ref": text, "file": base + ".wav", "noisy": False,
                      "seconds": round(len(audio) / 16000, 2)})
        if (a.noisy or a.phone) and i % 2 == 0:
            degraded, snr = audio, 20
            if a.phone:   # a phone line: band-limited to 8 kHz (what Exotel sends), plus line noise at 25 dB
                degraded = resample_poly(resample_poly(audio, 1, 2), 2, 1).astype(np.float32)[:len(audio)]
                degraded = np.pad(degraded, (0, len(audio) - len(degraded)))
                snr = 25
            noise = rng.standard_normal(len(audio)).astype(np.float32)
            noise *= np.sqrt(np.mean(degraded ** 2) / (10 ** (snr / 10)) / np.mean(noise ** 2))
            sf.write(os.path.join(a.out, base + "_noisy.wav"), np.clip(degraded + noise, -1, 1), 16000)
            clips.append({"id": base + "_noisy", "kind": kind, "ref": text, "file": base + "_noisy.wav", "noisy": True,
                          "seconds": round(len(audio) / 16000, 2)})
    with open(os.path.join(a.out, "clips.json"), "w") as f:
        json.dump(clips, f, ensure_ascii=False, indent=1)
    secs = [c["seconds"] for c in clips if not c["noisy"]]
    print(f"{len(clips)} clips in {a.out}; clean clip length {min(secs):.1f}-{max(secs):.1f} s")


# --- run (backend/.venv-mlx) ------------------------------------------------------------------------------------------

def cmd_run(a):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.audio import load_audio
    import mlx_asr_worker as worker
    import asr_window

    fast, accurate = "mlx-community/whisper-small-mlx", "mlx-community/whisper-large-v3-turbo"
    worker.keep_models_resident()
    clips = json.load(open(os.path.join(a.clips, "clips.json")))
    for c in clips:
        c["audio"] = load_audio(os.path.join(a.clips, c["file"]))

    def run(audio, repo, language):
        return mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language=language, fp16=True)

    def detect(audio, repo):
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.transcribe import ModelHolder
        model = ModelHolder.get_model(repo, mx.float16)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        _, probs = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
        return max(probs, key=probs.get)

    def v_baseline(audio):
        out, _ = worker.transcribe_plan("auto", None, audio, run, detect, fast, accurate)
        return {"text": (out.get("text") or "").strip(), "language": out.get("language")}

    def v_turbo30(audio):
        r = asr_window.transcribe_window(accurate, audio, None, buckets=(30,), fallback=False)
        return {"text": r["text"], "language": r["language"]}

    def v_trim(name):
        def fn(audio):
            r = asr_window.transcribe_window(accurate, audio, None, buckets=TRIM_BUCKETS[name])
            return {"text": r["text"], "language": r["language"], "window": r["window"], "fallback": r["fallback"]}
        return fn

    fns = {"baseline": v_baseline, "turbo30": v_turbo30, "trim8": v_trim("trim8"), "trim10": v_trim("trim10")}
    for c in clips[:2] + clips[-2:]:          # warm-up: both models, both languages
        for fn in fns.values():
            fn(c["audio"])

    samples = {}
    for rnd in range(a.rounds):
        for ci, c in enumerate(clips):
            k = (ci + rnd) % len(VARIANTS)                # rotate the order so background load hits every variant alike
            for v in list(VARIANTS[k:]) + list(VARIANTS[:k]):
                t = time.perf_counter()
                out = fns[v](c["audio"])
                out["wall"] = time.perf_counter() - t
                samples.setdefault((v, c["id"]), []).append(out)
        print(f"round {rnd + 1}/{a.rounds}", flush=True)
    rows = []
    for (v, cid), outs in samples.items():
        c = next(x for x in clips if x["id"] == cid)
        rows.append({"variant": v, "clip": cid, "kind": c["kind"], "noisy": c["noisy"], "seconds_audio": c["seconds"],
                     "ref": c["ref"], "text": outs[0]["text"], "language": outs[0]["language"],
                     "wall": statistics.median(o["wall"] for o in outs), "window": outs[0].get("window"),
                     "fallback": outs[0].get("fallback")})
    with open(a.json, "w") as f:
        json.dump({"rows": rows, "load": os.getloadavg(), "rounds": a.rounds}, f, ensure_ascii=False, indent=1)
    print(f"wrote {a.json}")


# --- score (backend/.venv) --------------------------------------------------------------------------------------------

def norm_en(s):
    s = unicodedata.normalize("NFKC", s).lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s).replace("'", "")
    return re.sub(r"\s+", " ", s).strip()


def norm_hi(s):
    s = unicodedata.normalize("NFC", s)
    return "".join(c for c in s if 0x0900 <= ord(c) <= 0x097F and unicodedata.category(c)[0] in "LM")


def errors(kind, ref, hyp):
    """(edits, reference length): words for English, characters for Hindi, so group rates are corpus-level."""
    import jiwer
    if kind == "hi":
        r, h = norm_hi(ref), norm_hi(hyp)
        if not h:
            return len(r), len(r)
        m = jiwer.process_characters(r, h)
    else:
        r, h = norm_en(ref), norm_en(hyp)
        if not h:
            return len(r.split()), len(r.split())
        m = jiwer.process_words(r, h)
    return m.substitutions + m.deletions + m.insertions, m.substitutions + m.deletions + m.hits


def cmd_score(a):
    data = json.load(open(a.json))
    rows = data["rows"]
    for r in rows:
        r["edits"], r["n"] = errors(r["kind"], r["ref"], r["text"])
    print(f"Interleaved rounds per clip: {data['rounds']}; load average at the end {data['load'][0]:.1f}\n")
    print("| Clips | Variant | n | Error rate | Clips with errors | Language id ok | Fallbacks | Latency median (s) | mean (s) | p95 (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    groups = [("English clean (WER)", lambda r: r["kind"] == "en" and not r["noisy"]),
              ("English noisy (WER)", lambda r: r["kind"] == "en" and r["noisy"]),
              ("Hindi clean (CER)", lambda r: r["kind"] == "hi" and not r["noisy"]),
              ("Hindi noisy (CER)", lambda r: r["kind"] == "hi" and r["noisy"])]
    for name, sel in groups:
        for v in VARIANTS:
            sub = [r for r in rows if r["variant"] == v and sel(r)]
            if not sub:
                continue
            lat = sorted(r["wall"] for r in sub)
            want = sub[0]["kind"]
            idok = sum(1 for r in sub if r["language"] == want)
            fb = sum(1 for r in sub if r.get("fallback")) if v.startswith("trim") else "-"
            rate = sum(r["edits"] for r in sub) / sum(r["n"] for r in sub)
            print(f"| {name} | {v} | {len(sub)} | {rate * 100:.2f}% | {sum(1 for r in sub if r['edits'])} | {idok}/{len(sub)} | {fb} | "
                  f"{statistics.median(lat):.3f} | {statistics.mean(lat):.3f} | {lat[min(len(lat) - 1, round(.95 * (len(lat) - 1)))]:.3f} |")
    print()
    for kind, label in (("en", "English"), ("hi", "Hindi")):
        for v in VARIANTS[1:]:
            base = {r["clip"]: r for r in rows if r["variant"] == "baseline" and r["kind"] == kind}
            pr = [(base[r["clip"]], r) for r in rows if r["variant"] == v and r["kind"] == kind]
            de = sum(r["edits"] for _, r in pr) / sum(r["n"] for _, r in pr) - sum(b["edits"] for b, _ in pr) / sum(b["n"] for b, _ in pr)
            dl = statistics.mean(r["wall"] - b["wall"] for b, r in pr)
            same = sum(1 for b, r in pr if b["text"].strip() == r["text"].strip())
            print(f"{label}, {v} minus baseline over {len(pr)} clips: error {de * 100:+.2f} points, latency {dl:+.3f} s, "
                  f"identical transcript on {same}/{len(pr)}")
    for v in ("trim8", "trim10"):
        print(f"\n{v} windows used (clean clips):", {w: sum(1 for r in rows if r["variant"] == v and r.get("window") == w and not r["noisy"])
                                                     for w in (8, 10, 15, 20, 30)})
    print("\nClips where a variant has more edits than the baseline:")
    base = {r["clip"]: r for r in rows if r["variant"] == "baseline"}
    for v in VARIANTS[1:]:
        for r in rows:
            if r["variant"] == v and r["edits"] > base[r["clip"]]["edits"]:
                print(f"  [{v}] {r['clip']}: edits {r['edits']} vs baseline {base[r['clip']]['edits']}  hyp={r['text']!r}")


def cmd_all(a):
    tmp = tempfile.mkdtemp(prefix="asr_window_")
    out = a.json or os.path.join(tmp, "result.json")
    me = os.path.abspath(__file__)
    mlx_py = os.path.join(BACKEND, ".venv-mlx", "bin", "python")
    subprocess.check_call([sys.executable, me, "gen", "--out", tmp] + (["--noisy"] if a.noisy else []) + (["--phone"] if a.phone else []))
    subprocess.check_call([mlx_py, me, "run", "--clips", tmp, "--json", out, "--rounds", str(a.rounds)],
                          env={**os.environ, "HF_HUB_OFFLINE": "1"})
    subprocess.check_call([sys.executable, me, "score", "--json", out])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--noisy", action="store_true", help="add a 20 dB SNR copy of every second clip")
    g.add_argument("--phone", action="store_true", help="instead add a phone-line copy (8 kHz band, 25 dB noise) of every second clip; scored as the 'noisy' rows")
    r = sub.add_parser("run")
    r.add_argument("--clips", required=True)
    r.add_argument("--json", required=True)
    r.add_argument("--rounds", type=int, default=3)
    s = sub.add_parser("score")
    s.add_argument("--json", required=True)
    al = sub.add_parser("all")
    al.add_argument("--json")
    al.add_argument("--rounds", type=int, default=3)
    al.add_argument("--noisy", action="store_true")
    al.add_argument("--phone", action="store_true", help="the 'noisy' rows are phone-line copies (8 kHz band, 25 dB noise)")
    a = ap.parse_args()
    {"gen": cmd_gen, "run": cmd_run, "score": cmd_score, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    main()
