#!/usr/bin/env python3
"""Open TTS engines vs the app's Kokoro, measured INTERLEAVED so machine load hits every engine alike.

Standalone trial (it does not call the BonClay API and changes nothing in the repo). Every engine runs in its own
subprocess (its own venv), loaded once and warmed up. Then, for each round, for each sentence, each engine speaks the
same sentence in turn (engine order rotates each round), so a load spike from another job lands on all engines.
Per sentence the median over rounds is taken; the table reports the median and mean of those medians.

  time to first audio (TTFA)  seconds from the request until the first audio chunk exists. Streaming engines (Kokoro's
                              generator) report the first chunk; others report the whole synthesis, so TTFA = total.
  RTF                         total synthesis seconds / audio seconds (below 1.0 is faster than real time).
  read-back WER / CER         the audio is transcribed with mlx-whisper (whisper-small for English, large-v3-turbo for
                              Hindi, as BonClay does) and scored with the same normalisation as backend/bench/readback.py
                              (English: lowercase words, punctuation removed; Hindi: Devanagari letters and marks only).
  memory                      worker process RSS (ps) after the model is loaded and the peak over the run.

Run with the backend venv (it has jiwer); engine and ASR interpreters are passed in:

    backend/.venv/bin/python backend/bench/tts_compare.py \
        --engines kokoro-torch,kokoro-mlx,supertonic3,supertonic3-2step,omnivoice \
        --py-mlx <venv with mlx-audio+misaki>/bin/python --py-supertonic <venv with supertonic>/bin/python \
        --py-asr backend/.venv-mlx/bin/python --out <work dir>

Internal modes (the driver starts these): --worker ENGINE, --asr-worker.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)

# engine -> settings. "py" picks the interpreter key. Voices are synthetic presets only.
ENGINES = {
    "kokoro-torch": {"py": "kokoro", "label": "Kokoro-82M PyTorch (the app)",
                     "voice": {"en": "af_heart", "hi": "hf_alpha"}, "lang_code": {"en": "a", "hi": "h"}},
    "kokoro-mlx": {"py": "mlx", "label": "Kokoro-82M MLX bf16 (mlx-audio)",
                   "voice": {"en": "af_heart", "hi": "hf_alpha"}, "lang_code": {"en": "a", "hi": "h"}},
    "supertonic3": {"py": "supertonic", "label": "Supertonic 3 ONNX CPU, 8 steps (default)", "steps": 8,
                    "voice": {"en": "F1", "hi": "F1"}},
    "supertonic3-2step": {"py": "supertonic", "label": "Supertonic 3 ONNX CPU, 2 steps", "steps": 2,
                          "voice": {"en": "F1", "hi": "F1"}},
    "supertonic3-4step": {"py": "supertonic", "label": "Supertonic 3 ONNX CPU, 4 steps", "steps": 4,
                          "voice": {"en": "F1", "hi": "F1"}},
    "pocket-tts": {"py": "mlx", "label": "Pocket TTS ~100M MLX (streaming, English only)", "langs": ["en"],
                   "repo": "mlx-community/pocket-tts", "voice": {"en": "alba"}},
    "soprano": {"py": "mlx", "label": "Soprano 1.1 80M MLX bf16 (English only)", "langs": ["en"],
                "repo": "mlx-community/Soprano-1.1-80M-bf16"},
    "omnivoice": {"py": "mlx", "label": "OmniVoice 0.6B MLX bf16, 32 steps (default)", "steps": 32,
                  "repo": "mlx-community/OmniVoice-bfloat16", "language": {"en": "en", "hi": "hi"}},
    "omnivoice-16": {"py": "mlx", "label": "OmniVoice 0.6B MLX bf16, 16 steps", "steps": 16,
                     "repo": "mlx-community/OmniVoice-bfloat16", "language": {"en": "en", "hi": "hi"}},
}
LANGS = ("en", "hi")


# ----------------------------------------------------------------------------------------------- workers
def _proto_stream():
    """Library chatter goes to stderr; JSON lines go on the original stdout."""
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    return proto


def _configure_espeak():
    """Same idea as backend/services/espeak.py: use Homebrew's espeak-ng, the bundled wheel crashes on macOS arm64."""
    lib, data = "/opt/homebrew/lib/libespeak-ng.dylib", "/opt/homebrew/share/espeak-ng-data"
    if os.path.isfile(lib) and os.path.isdir(data):
        import misaki.espeak  # noqa: F401
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(lib)
        EspeakWrapper.set_data_path(data)


def build_engine(name):
    """Returns (speak, extra). speak(lang, text) -> (audio float32 mono ndarray, sample rate, ttfa seconds or None)."""
    import numpy as np
    cfg = ENGINES[name]
    if name == "kokoro-torch":
        sys.path.insert(0, BACKEND)
        try:
            from services.espeak import configure_espeak
            configure_espeak()
        except ImportError:
            _configure_espeak()
        from kokoro import KPipeline
        pipes = {}

        def speak(lang, text):
            if lang not in pipes:
                pipes[lang] = KPipeline(lang_code=cfg["lang_code"][lang])
            parts, first = [], None
            t0 = time.perf_counter()
            for _, _, audio in pipes[lang](text, voice=cfg["voice"][lang], speed=1.0):
                if audio is not None:
                    if first is None:
                        first = time.perf_counter() - t0
                    parts.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
            return np.concatenate(parts), 24000, first
        return speak, {}

    if name == "kokoro-mlx":
        _configure_espeak()
        from mlx_audio.tts.utils import load_model
        model = load_model("mlx-community/Kokoro-82M-bf16")

        def speak(lang, text):
            parts, first = [], None
            t0 = time.perf_counter()
            for r in model.generate(text=text, voice=cfg["voice"][lang], speed=1.0, lang_code=cfg["lang_code"][lang]):
                a = np.array(r.audio)  # forces evaluation
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(a.reshape(-1))
            return np.concatenate(parts), model.config.sample_rate, first
        return speak, {}

    if name.startswith("supertonic3"):
        from supertonic import TTS
        tts = TTS(model="supertonic-3", auto_download=True)
        styles = {}

        def speak(lang, text):
            v = cfg["voice"][lang]
            if v not in styles:
                styles[v] = tts.get_voice_style(voice_name=v)
            wav, _dur = tts.synthesize(text, voice_style=styles[v], total_steps=cfg["steps"], lang=lang)
            return np.asarray(wav).reshape(-1).astype("float32"), tts.sample_rate, None
        return speak, {}

    if name == "pocket-tts":
        from mlx_audio.tts.utils import load_model
        model = load_model(cfg["repo"])

        def speak(lang, text):
            parts, first = [], None
            t0 = time.perf_counter()
            # stream=True yields a chunk as soon as ~0.1 s of audio exists; first chunk time is the TTFA
            for r in model.generate(text=text, voice=cfg["voice"][lang], stream=True, streaming_interval=0.1):
                a = np.array(r.audio).reshape(-1)
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(a)
            return np.concatenate(parts), model.sample_rate, first
        return speak, {}

    if name == "soprano":
        from mlx_audio.tts.utils import load_model
        model = load_model(cfg["repo"])

        def speak(lang, text):
            parts, first = [], None
            t0 = time.perf_counter()
            for r in model.generate(text=text):
                a = np.array(r.audio).reshape(-1)
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(a)
            return np.concatenate(parts), r.sample_rate, first
        return speak, {}

    if name.startswith("omnivoice"):
        from mlx_audio.tts.utils import load_model
        model = load_model(cfg["repo"])  # sets the audio and text tokenizers on the model

        def speak(lang, text):
            r = next(model.generate(text=text, language=cfg["language"][lang], num_steps=cfg["steps"]))
            return np.array(r.audio).reshape(-1), r.sample_rate, None
        return speak, {}
    raise SystemExit(f"unknown engine {name}")


def worker(name):
    import numpy as np
    import soundfile as sf
    proto = _proto_stream()
    t0 = time.perf_counter()
    speak, _ = build_engine(name)
    speak("en", "Warming up the voice before the real measurement starts.")
    if "hi" in ENGINES[name].get("langs", LANGS):
        speak("hi", "यह मापन शुरू होने से पहले आवाज़ को गर्म करने का वाक्य है।")
    proto.write(json.dumps({"ready": True, "load_s": round(time.perf_counter() - t0, 2)}) + "\n")
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("quit"):
            break
        out = {"id": req["id"]}
        try:
            t = time.perf_counter()
            audio, sr, first = speak(req["lang"], req["text"])
            total = time.perf_counter() - t
            audio = np.asarray(audio, dtype="float32")
            out.update(total=total, ttfa=first if first is not None else total, audio_s=len(audio) / sr)
            sf.write(req["out"], audio, sr)
            try:
                import mlx.core as mx
                out["mlx_peak_mb"] = round(mx.get_peak_memory() / 1e6, 1)
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
        proto.write(json.dumps(out) + "\n")


def asr_worker():
    """Runs under backend/.venv-mlx. Reads {"path","lang"} lines, replies {"text"}. Same models/params as the app."""
    proto = _proto_stream()
    import mlx_whisper
    fast, accurate = "mlx-community/whisper-small-mlx", "mlx-community/whisper-large-v3-turbo"
    proto.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("quit"):
            break
        try:
            repo = accurate if req["lang"] == "hi" else fast
            r = mlx_whisper.transcribe(req["path"], path_or_hf_repo=repo, language=req["lang"], fp16=True)
            proto.write(json.dumps({"text": (r.get("text") or "").strip(), "model": repo}, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            proto.write(json.dumps({"text": "", "error": f"{type(e).__name__}: {e}"}) + "\n")


# ----------------------------------------------------------------------------------------------- driver
class Proc:
    def __init__(self, cmd, env=None, log=None):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log or subprocess.DEVNULL,
                                  text=True, bufsize=1, env={**os.environ, **(env or {})})

    def read(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited (code {self.p.poll()})")
        return json.loads(line)

    def ask(self, obj):
        self.p.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        return self.read()

    def rss_mb(self):
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.p.pid)], capture_output=True, text=True).stdout
            return int(out.strip()) / 1024
        except Exception:  # noqa: BLE001
            return None

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"quit": True}) + "\n")
            self.p.stdin.flush()
            self.p.wait(timeout=20)
        except Exception:  # noqa: BLE001
            self.p.kill()


def pick_sentences(n):
    sys.path.insert(0, HERE)
    import readback
    out = {}
    for lang, lst in (("en", readback.SENTENCES_EN), ("hi", readback.SENTENCES_HI)):
        ok = [s for s in lst if 8 <= len(s.replace("।", " ").split()) <= 15]
        out[lang] = ok[:n]
    return out, readback


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def fmt(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker")
    ap.add_argument("--asr-worker", action="store_true")
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--langs", default="en,hi")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("-n", "--sentences", type=int, default=10)
    ap.add_argument("--out", default="tts_compare_out")
    ap.add_argument("--py-kokoro", default=os.path.join(BACKEND, ".venv/bin/python"))
    ap.add_argument("--py-mlx", default=None, help="interpreter with mlx-audio + misaki (for kokoro-mlx, omnivoice)")
    ap.add_argument("--py-supertonic", default=None, help="interpreter with the supertonic package")
    ap.add_argument("--py-asr", default=os.path.join(BACKEND, ".venv-mlx/bin/python"))
    ap.add_argument("--hf-home-mlx", default=None, help="HF_HOME for the mlx-audio engines (default: shared cache)")
    ap.add_argument("--no-asr", action="store_true")
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker)
    if a.asr_worker:
        return asr_worker()

    os.makedirs(a.out, exist_ok=True)
    pys = {"kokoro": a.py_kokoro, "mlx": a.py_mlx, "supertonic": a.py_supertonic}
    names = [e for e in a.engines.split(",") if e]
    langs = [l for l in a.langs.split(",") if l]
    sents, readback = pick_sentences(a.sentences)
    me = os.path.abspath(__file__)
    procs, info, skipped = {}, {}, {}
    print(f"load average at start: {os.getloadavg()}", flush=True)
    for e in names:
        py = pys[ENGINES[e]["py"]]
        if not py:
            skipped[e] = "no interpreter given"
            continue
        log = open(os.path.join(a.out, f"{e}.log"), "w")
        try:
            # VIRTUAL_ENV: some libraries (spaCy via misaki) shell out to `uv pip install` for a missing model
            env = {"VIRTUAL_ENV": os.path.dirname(os.path.dirname(py))}
            if ENGINES[e]["py"] == "mlx" and a.hf_home_mlx:  # keep new downloads out of the shared Hugging Face cache
                env["HF_HOME"] = a.hf_home_mlx
            pr = Proc([py, me, "--worker", e], log=log, env=env)
            r = pr.read()
            procs[e] = pr
            info[e] = {"load_s": r["load_s"], "rss_loaded_mb": pr.rss_mb(), "rss_peak_mb": pr.rss_mb(), "mlx_peak_mb": None}
            print(f"{e}: ready, load+warmup {r['load_s']}s, RSS {info[e]['rss_loaded_mb']:.0f} MB", flush=True)
        except Exception as ex:  # noqa: BLE001
            skipped[e] = f"failed to start: {ex} (see {log.name})"
            print(f"{e}: SKIPPED {skipped[e]}", flush=True)
    if not procs:
        sys.exit("no engine started")

    # runs[(engine, lang, idx)] = list over rounds of dicts
    runs, errors = {}, {}
    wav_dir = os.path.join(a.out, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    load = []
    for rnd in range(a.rounds):
        order = list(procs)
        order = order[rnd % len(order):] + order[:rnd % len(order)]
        for lang in langs:
            for i, text in enumerate(sents[lang]):
                for e in order:  # same sentence, engines alternate back to back
                    if lang not in ENGINES[e].get("langs", LANGS):
                        continue
                    outp = os.path.join(wav_dir, f"{e}_{lang}_{i:02d}.wav")
                    try:
                        r = procs[e].ask({"id": f"{rnd}-{lang}-{i}", "lang": lang, "text": text, "out": outp})
                    except Exception as ex:  # noqa: BLE001
                        errors.setdefault((e, lang), []).append(str(ex))
                        continue
                    if "error" in r:
                        errors.setdefault((e, lang), []).append(r["error"])
                        continue
                    rss = procs[e].rss_mb()
                    if rss:
                        info[e]["rss_peak_mb"] = max(info[e]["rss_peak_mb"] or 0, rss)
                    if r.get("mlx_peak_mb"):
                        info[e]["mlx_peak_mb"] = r["mlx_peak_mb"]
                    runs.setdefault((e, lang, i), []).append(r)
        load.append(os.getloadavg()[0])
        print(f"round {rnd + 1}/{a.rounds} done, 1-min load {load[-1]:.2f}", flush=True)
    for pr in procs.values():
        pr.close()

    # read-back
    texts = {}
    if not a.no_asr:
        asr = Proc([a.py_asr, me, "--asr-worker"], log=open(os.path.join(a.out, "asr.log"), "w"))
        asr.read()
        for (e, lang, i) in sorted(runs):
            path = os.path.join(wav_dir, f"{e}_{lang}_{i:02d}.wav")
            r = asr.ask({"path": path, "lang": lang})
            texts[(e, lang, i)] = r.get("text", "")
        asr.close()

    # tables
    result = {"settings": vars(a) | {"sentences": sents}, "load_avg_start": None, "load_per_round": load,
              "skipped": skipped, "engines": {}}
    lines = []
    for lang in langs:
        metric = "WER" if lang == "en" else "CER"
        lines.append(f"\n### {'English' if lang == 'en' else 'Hindi'} ({len(sents[lang])} sentences x {a.rounds} rounds, "
                     f"interleaved; medians per sentence, then median / mean over sentences)\n")
        lines.append(f"| Engine | TTFA med s | TTFA mean s | total med s | RTF med | audio s (mean) | {metric} mean | "
                     f"{metric} median | RSS loaded MB | RSS peak MB |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for e in procs:
            if lang not in ENGINES[e].get("langs", LANGS):
                continue
            ttfa, tot, rtf, aud, err = [], [], [], [], []
            for i in range(len(sents[lang])):
                rs = runs.get((e, lang, i), [])
                if not rs:
                    continue
                ttfa.append(med([r["ttfa"] for r in rs]))
                tot.append(med([r["total"] for r in rs]))
                aud.append(med([r["audio_s"] for r in rs]))
                rtf.append(med([r["total"] / r["audio_s"] for r in rs if r["audio_s"]]))
                if (e, lang, i) in texts:
                    ref, hyp = sents[lang][i], texts[(e, lang, i)]
                    err.append(readback.score(lang, ref, hyp)[0])
            if not tot:
                lines.append(f"| {e} | failed: {errors.get((e, lang), ['?'])[0][:80]} |||||||||")
                continue
            lines.append(f"| {e} | {fmt(med(ttfa))} | {fmt(mean(ttfa))} | {fmt(med(tot))} | {fmt(med(rtf), 3)} | "
                         f"{fmt(mean(aud), 1)} | {fmt(mean(err) * 100 if err else None, 1)}% | "
                         f"{fmt(med(err) * 100 if err else None, 1)}% | {fmt(info[e]['rss_loaded_mb'], 0)} | "
                         f"{fmt(info[e]['rss_peak_mb'], 0)} |")
            result["engines"].setdefault(e, {})[lang] = {
                "ttfa_median": med(ttfa), "ttfa_mean": mean(ttfa), "total_median": med(tot), "rtf_median": med(rtf),
                "audio_mean": mean(aud), "error_rate_mean": mean(err), "error_rate_median": med(err),
                "per_sentence": [{"text": sents[lang][i], "transcript": texts.get((e, lang, i)),
                                  "total_s": [r["total"] for r in runs.get((e, lang, i), [])],
                                  "audio_s": [r["audio_s"] for r in runs.get((e, lang, i), [])]}
                                 for i in range(len(sents[lang]))]}
    lines.append("\n### Median total seconds per sentence, by round (shows how much load moved the numbers)\n")
    lines.append("| Engine | lang | " + " | ".join(f"round {r + 1}" for r in range(a.rounds)) + " |")
    lines.append("|---|---|" + "---|" * a.rounds)
    for e in procs:
        for lang in langs:
            if lang not in ENGINES[e].get("langs", LANGS):
                continue
            cells = []
            for rd in range(a.rounds):
                xs = [runs[(e, lang, i)][rd]["total"] for i in range(len(sents[lang]))
                      if len(runs.get((e, lang, i), [])) > rd]
                cells.append(fmt(med(xs)))
            lines.append(f"| {e} | {lang} | " + " | ".join(cells) + " |")
    lines.append(f"\nWorker load+warmup seconds: " + ", ".join(f"{e} {info[e]['load_s']}" for e in procs))
    lines.append("MLX peak GPU memory (MB): " + ", ".join(f"{e} {info[e]['mlx_peak_mb']}" for e in procs if info[e]["mlx_peak_mb"]))
    lines.append(f"1-minute load average at each round end: {', '.join(f'{x:.2f}' for x in load)}")
    for k, v in skipped.items():
        lines.append(f"SKIPPED {k}: {v}")
    for (e, lang), v in errors.items():
        lines.append(f"ERRORS {e}/{lang}: {len(v)} e.g. {v[0][:120]}")
    result["engine_info"] = info
    result["errors"] = {f"{e}/{l}": v for (e, l), v in errors.items()}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    md = "\n".join(lines)
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
