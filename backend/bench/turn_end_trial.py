"""Trial: would Pipecat Smart Turn v3 end caller turns sooner than the current silence rule, without cutting people off?

Not part of the app and not wired into anything. Synthetic voices only (Kokoro). Two stages, two environments:

  gen    (backend/.venv: needs kokoro)      build the test set: WAVs + manifest.json in --work
  bench  (scratch venv: onnxruntime, transformers)   time feature extraction + inference on an 8 s window
  eval   (scratch venv: + silero-vad, torch) replay every file through the "browser" Silero state machine, run Smart
                                            Turn at the silence checkpoints, write results.json
  report (any)                              print markdown tables from results.json

Current rule being compared (frontend/src/lib/vad.ts, SILERO_DEFAULTS): Silero v5, positive 0.5 / negative 0.35,
redemption 700 ms (end of speech after ~700 ms of accumulated silent frames), minSpeech 400 ms.
Smart Turn policy: when 200 ms of silence has been seen, run Smart Turn on the turn audio so far (last 8 s); end the
turn at once if P(complete) >= threshold, otherwise keep waiting and fall back to the 700 ms rule.

  python turn_end_trial.py gen    --work DIR            # backend/.venv/bin/python
  python turn_end_trial.py bench  --work DIR --model M  # scratch venv
  python turn_end_trial.py eval   --work DIR --model M
  python turn_end_trial.py report --work DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 512                    # Silero v5 frame at 16 kHz, 32 ms (what @ricky0123/vad-web feeds it)
FRAME_MS = FRAME / SR * 1000
POS, NEG = 0.5, 0.35           # positiveSpeechThreshold / negativeSpeechThreshold
REDEMPTION_FRAMES = int(np.ceil(700 / FRAME_MS))     # 22 frames = 704 ms
MIN_SPEECH_FRAMES = int(np.ceil(400 / FRAME_MS))
PRE_PAD_MS = 200
CHECKPOINTS = [7, 10, 13, 16, 19]                    # silent frames seen: 224, 320, 416, 512, 608 ms
PAUSES_MS = [300, 600, 900]
LEAD_MS, TAIL_MS = 600, 2500

# ---------------------------------------------------------------------------------------------------------- texts
# (language, voice, complete texts, incomplete (head, tail) pairs). Hinglish = code-mixed Devanagari through Kokoro's
# Hindi voice (Kokoro has no Latin-script Hinglish path), so it is Hindi phonetics with English loanwords.
COMPLETE = [
    ("en", "af_bella", "Can you tell me what time the clinic opens tomorrow morning?"),
    ("en", "am_adam", "I would like to book an appointment for next Tuesday."),
    ("en", "af_heart", "I ordered a blue jacket last week and it still has not arrived."),
    ("en", "am_michael", "Please cancel my subscription effective immediately."),
    ("en", "af_bella", "What are your opening hours on Saturdays?"),
    ("en", "am_adam", "Thanks, that is all I needed for today."),
    ("en", "af_heart", "I need to reschedule my dentist appointment to Friday afternoon."),
    ("hi", "hf_alpha", "क्या आप मुझे बता सकते हैं कि क्लिनिक कल कितने बजे खुलेगा?"),
    ("hi", "hf_alpha", "मुझे अगले मंगलवार के लिए एक समय चाहिए।"),
    ("hi", "hf_alpha", "मेरा सामान अभी तक नहीं पहुँचा है।"),
    ("hi", "hf_alpha", "कृपया मेरी सदस्यता अभी रद्द कर दीजिए।"),
    ("hi", "hf_alpha", "शनिवार को आपकी दुकान कितने बजे बंद होती है?"),
    ("hi", "hf_alpha", "धन्यवाद, मुझे आज बस इतना ही चाहिए था।"),
    ("hinglish", "hf_alpha", "मेरा अपॉइंटमेंट कल सुबह का है।"),
    ("hinglish", "hm_omega", "प्लीज़ मेरा ऑर्डर कैंसल कर दीजिए।"),
    ("hinglish", "hf_alpha", "क्या आप मेरा डिलीवरी स्टेटस चेक कर सकते हैं?"),
    ("hinglish", "hm_omega", "मुझे नेक्स्ट वीक का स्लॉट बुक करना है।"),
    ("hinglish", "hf_alpha", "मेरा पेमेंट फेल हो गया है लेकिन पैसे कट गए हैं।"),
    ("hinglish", "hm_omega", "ओके थैंक यू बस इतना ही चाहिए था।"),
    ("hinglish", "hf_alpha", "आपकी वेबसाइट पर लॉगिन नहीं हो रहा है।"),
]
INCOMPLETE = [
    ("en", "af_bella", "I wanted to ask about my appointment", "which is scheduled for tomorrow at ten o'clock."),
    ("en", "am_adam", "So I ordered a blue jacket last week and", "it still has not arrived at my address."),
    ("en", "af_heart", "Could you please check if", "the doctor is available on Thursday afternoon?"),
    ("en", "am_michael", "My phone number is nine eight seven six", "five four three two one zero."),
    ("en", "af_bella", "I was thinking that maybe", "we could move the meeting to next week."),
    ("en", "am_adam", "The problem is that the app keeps", "crashing every time I open the payment screen."),
    ("en", "af_heart", "Yes I would like to book a table for", "four people at seven thirty tonight."),
    ("hi", "hf_alpha", "मुझे डॉक्टर से", "कल सुबह मिलना है।"),
    ("hi", "hf_alpha", "मैंने पिछले हफ़्ते एक नीली जैकेट मँगवाई थी और", "वह अभी तक मेरे पते पर नहीं आई है।"),
    ("hi", "hf_alpha", "क्या आप देख सकते हैं कि", "गुरुवार दोपहर को डॉक्टर उपलब्ध हैं या नहीं?"),
    ("hi", "hf_alpha", "मेरा फ़ोन नंबर नौ आठ सात छह", "पाँच चार तीन दो एक शून्य है।"),
    ("hi", "hf_alpha", "मैं सोच रहा था कि शायद", "हम बैठक अगले हफ़्ते कर सकते हैं।"),
    ("hi", "hf_alpha", "दिक्कत यह है कि ऐप बार-बार", "बंद हो जाता है जब मैं भुगतान खोलता हूँ।"),
    ("hinglish", "hf_alpha", "मेरा अपॉइंटमेंट", "कल सुबह का है।"),
    ("hinglish", "hm_omega", "भैया मेरा ऑर्डर तीन दिन से", "ट्रैकिंग में स्टक दिखा रहा है।"),
    ("hinglish", "hf_alpha", "क्या आप चेक कर सकते हैं कि", "मेरा रिफंड प्रोसेस हुआ या नहीं?"),
    ("hinglish", "hm_omega", "मुझे नेक्स्ट वीक का", "एक स्लॉट बुक करना है।"),
    ("hinglish", "hf_alpha", "मेरा पेमेंट फेल हो गया लेकिन", "पैसे अकाउंट से कट गए हैं।"),
    ("hinglish", "hm_omega", "मैं सोच रहा था कि अगर", "आप डिलीवरी आज ही कर दें तो अच्छा रहेगा।"),
    ("hinglish", "hf_alpha", "मेरा ईमेल आईडी ए आर जे यू एन", "एट जीमेल डॉट कॉम है।"),
]


# ---------------------------------------------------------------------------------------------------------- gen
def _trim(x: np.ndarray, thr_ratio: float = 0.02) -> np.ndarray:
    """Drop Kokoro's near-silent lead/tail so the true speech end is known to ~10 ms."""
    win = int(0.01 * SR)
    n = len(x) // win
    if n == 0:
        return x
    rms = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1))
    loud = np.where(rms > thr_ratio * rms.max())[0]
    if not len(loud):
        return x
    return x[loud[0] * win:(loud[-1] + 1) * win]


def _split_at_dip(x, frac):
    """Cut x at the quietest 10 ms window within +-250 ms of the point `frac` of the way through; 8 ms fades."""
    win = int(0.01 * SR)
    n = len(x) // win
    rms = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1))
    c = int(frac * n)
    lo, hi = max(1, c - 25), min(n - 1, c + 25)
    k = lo + int(np.argmin(rms[lo:hi]))
    cut = k * win + win // 2
    fade = int(0.008 * SR)
    a, b = x[:cut].copy(), x[cut:].copy()
    a[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    b[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    return a, b


def cmd_gen(a):
    import soundfile as sf
    from scipy.signal import resample_poly
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # backend/, read-only import of espeak helper
    from services.espeak import configure_espeak
    configure_espeak()
    from kokoro import KPipeline
    pipes = {}

    def synth(text, voice):
        code = voice[0]
        if code not in pipes:
            pipes[code] = KPipeline(lang_code=code)
        parts = [np.asarray(au, dtype=np.float32) for _, _, au in pipes[code](text, voice=voice, speed=1.0)
                 if au is not None]
        x = np.concatenate(parts)
        return _trim(resample_poly(x, 2, 3).astype(np.float32))      # 24 kHz -> 16 kHz

    work = Path(a.work)
    (work / "audio").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    noise = lambda ms: (rng.standard_normal(int(ms / 1000 * SR)) * 0.0015).astype(np.float32)   # about -56 dBFS
    sil = noise
    manifest = []

    def emit(name, lang, kind, pause_ms, voice, text, pieces, set_):
        """pieces: list of ('s', audio) speech / ('p', ms) silence; the first speech piece gets LEAD_MS before it."""
        out, t, ends = [noise(LEAD_MS)], LEAD_MS / 1000, []
        for kind_, v in pieces:
            seg = v if kind_ == "s" else sil(v)
            out.append(seg)
            t += len(seg) / SR
            if kind_ == "s":
                ends.append(t)
        out.append(noise(TAIL_MS))
        wav = np.concatenate(out)
        wav = wav / max(1e-6, np.abs(wav).max()) * 0.5
        sf.write(work / "audio" / f"{name}.wav", wav, SR, subtype="PCM_16")
        manifest.append(dict(name=name, lang=lang, kind=kind, set=set_, pause_ms=pause_ms, voice=voice, text=text,
                             speech_ends=ends, true_end=ends[-1], dur=len(wav) / SR))

    for i, (lang, voice, text) in enumerate(COMPLETE):
        emit(f"c{i:02d}_{lang}", lang, "complete", 0, voice, text, [("s", synth(text, voice))], "complete")
    for i, (lang, voice, head, tail) in enumerate(INCOMPLETE):
        h, t = synth(head, voice), synth(tail, voice)
        for p in PAUSES_MS:
            emit(f"i{i:02d}_{lang}_p{p}", lang, "incomplete", p, voice, f"{head} [{p} ms] {tail}",
                 [("s", h), ("p", p), ("s", t)], "tts")
    # second set: synthesise the whole sentence once and cut it at the energy dip nearest the head/tail character
    # split, so the first half keeps mid-sentence (continuing) prosody instead of a re-synthesised sentence-final fall
    for i, (lang, voice, head, tail) in enumerate(INCOMPLETE):
        full = synth(f"{head} {tail}", voice)
        h, t = _split_at_dip(full, len(head) / (len(head) + len(tail) + 1))
        for p in PAUSES_MS:
            emit(f"k{i:02d}_{lang}_p{p}", lang, "incomplete", p, voice, f"{head} [{p} ms cut] {tail}",
                 [("s", h), ("p", p), ("s", t)], "cut")
    (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"wrote {len(manifest)} files to {work/'audio'}")


# ---------------------------------------------------------------------------------------------------------- Smart Turn
class SmartTurn:
    def __init__(self, model_path, threads=0):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.fe = WhisperFeatureExtractor(chunk_length=8)

    def features(self, audio):
        n = 8 * SR
        audio = audio[-n:] if len(audio) > n else np.pad(audio, (n - len(audio), 0))   # zero-pad at the front
        f = self.fe(audio, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=n,
                    truncation=True, do_normalize=True).input_features.astype(np.float32)
        return f

    def run(self, feats):
        return float(self.sess.run(None, {"input_features": feats})[0].reshape(-1)[0])

    def prob(self, audio):
        return self.run(self.features(audio))


def cmd_bench(a):
    import onnxruntime as ort
    audio_files = sorted((Path(a.work) / "audio").glob("c0[0-9]_*.wav"))
    import soundfile as sf
    xs = [sf.read(f, dtype="float32")[0] for f in audio_files][:6]
    res = {}
    for threads in (1, 2, 4, 0):
        st = SmartTurn(a.model, threads)
        for x in xs[:2]:
            st.prob(x[: 8 * SR])
        fe_ms, ort_ms = [], []
        for i in range(60):
            x = xs[i % len(xs)][: 8 * SR]
            t0 = time.perf_counter(); f = st.features(x); t1 = time.perf_counter(); st.run(f); t2 = time.perf_counter()
            fe_ms.append((t1 - t0) * 1e3); ort_ms.append((t2 - t1) * 1e3)
        q = lambda v, p: float(np.percentile(v, p))
        res[str(threads or "default")] = dict(feat_p50=q(fe_ms, 50), feat_p95=q(fe_ms, 95), ort_p50=q(ort_ms, 50),
                                              ort_p95=q(ort_ms, 95), total_p50=q(np.add(fe_ms, ort_ms), 50),
                                              total_p95=q(np.add(fe_ms, ort_ms), 95))
        print("threads", threads or "default", {k: round(v, 1) for k, v in res[str(threads or 'default')].items()})
    (Path(a.work) / "bench.json").write_text(json.dumps(dict(model=str(a.model), ort=ort.__version__, **res), indent=1))


# ---------------------------------------------------------------------------------------------------------- eval
def silero_probs(model, wav):
    import torch
    model.reset_states()
    n = len(wav) // FRAME
    return np.array([float(model(torch.from_numpy(wav[i * FRAME:(i + 1) * FRAME]), SR)) for i in range(n)])


def episodes(probs):
    """Replay the vad-web frame processor with an endless redemption, returning silence episodes inside speech.

    vad-web: speech starts at prob >= POS; once speaking, every frame < NEG adds 1 to a counter, a frame >= POS resets
    it, frames in between change nothing; the segment ends when the counter hits REDEMPTION_FRAMES. An episode is the
    stretch between two resets; `frames` maps counter value -> frame index at which it was reached.
    """
    eps, speaking, counter, start, cur = [], False, 0, None, None
    for i, p in enumerate(probs):
        if p >= POS:
            if not speaking:
                speaking, start = True, i
            counter, cur = 0, None
        elif p < NEG and speaking:
            counter += 1
            if cur is None:
                cur = dict(first=i, frames={})
                eps.append(cur)
            cur["frames"][counter] = i
    return start, eps


def cmd_eval(a):
    import soundfile as sf
    from silero_vad import load_silero_vad
    work = Path(a.work)
    man = json.loads((work / "manifest.json").read_text())
    vad = load_silero_vad(onnx=True)
    st = SmartTurn(a.model)
    out = []
    for m in man:
        wav, _ = sf.read(work / "audio" / f"{m['name']}.wav", dtype="float32")
        probs = silero_probs(vad, wav)
        start, eps = episodes(probs)
        seg_start = max(0, int(start - PRE_PAD_MS / FRAME_MS))
        rec = dict(m, eps=[])
        for ep in eps:
            e = dict(first_t=(ep["first"]) * FRAME_MS / 1000, cps={}, end_t=None)
            for c in CHECKPOINTS:
                if c in ep["frames"]:
                    f = ep["frames"][c]
                    audio = wav[seg_start * FRAME:(f + 1) * FRAME]
                    t0 = time.perf_counter()
                    p = st.prob(audio)
                    e["cps"][c] = dict(t=(f + 1) * FRAME_MS / 1000, p=p, ms=(time.perf_counter() - t0) * 1e3)
            if REDEMPTION_FRAMES in ep["frames"]:
                e["end_t"] = (ep["frames"][REDEMPTION_FRAMES] + 1) * FRAME_MS / 1000
            rec["eps"].append(e)
        out.append(rec)
        print(m["name"], [(round(e["first_t"], 2), {c: round(v["p"], 2) for c, v in e["cps"].items()}, e["end_t"])
                          for e in rec["eps"]])
    (work / f"results_{Path(a.model).stem}.json").write_text(json.dumps(out, ensure_ascii=False))


# ---------------------------------------------------------------------------------------------------------- report
def decide(rec, mode, th):
    """Return (end_time, episode_index) of the first turn end for one file under a policy.

    mode: 'silence' (current 700 ms rule), 'sil13' / 'sil16' (silence-only, 13 / 16 silent frames = 416 / 512 ms), 'st1' (Smart Turn once at the first checkpoint, then the 700 ms rule),
          'st5' (Smart Turn at every checkpoint 224..608 ms, then the 700 ms rule).
    """
    for k, e in enumerate(rec["eps"]):
        if mode.startswith("sil") and mode != "silence":          # silence-only rule with a shorter redemption
            v = e["cps"].get(str(int(mode[3:]))) or e["cps"].get(int(mode[3:]))
            if v:
                return v["t"], k
        elif mode != "silence":
            cps = [CHECKPOINTS[0]] if mode == "st1" else CHECKPOINTS
            for c in cps:
                v = e["cps"].get(c) or e["cps"].get(str(c))
                if v and v["p"] >= th:
                    return v["t"], k
        if e["end_t"] is not None:
            return e["end_t"], k
    return None, None


def classify(rec, mode, th):
    t, k = decide(rec, mode, th)
    if t is None:
        return dict(kind="none")
    # true end of the turn = end of the last speech piece; an end decided before it is a cut-off.
    if t < rec["true_end"] - 0.05:
        return dict(kind="cut", at=t)
    return dict(kind="end", delay=t - rec["true_end"])


def cmd_report(a):
    work = Path(a.work)
    res = {}
    for f in sorted(work.glob("results_*.json")):
        res[f.stem.replace("results_", "")] = json.loads(f.read_text())
    langs = ["en", "hi", "hinglish"]
    for name, recs in res.items():
        print(f"\n## model {name}  ({len(recs)} files)")
        # false cut-offs on incomplete variants: first end happened before the true end of speech
        pols = [("silence 700 ms (current)", "silence", 0.5), ("silence 512 ms", "sil16", 0.5),
                ("silence 416 ms", "sil13", 0.5)] + [(f"SmartTurn x1 th={th}", "st1", th) for th in (0.5, 0.8, 0.9)] \
            + [(f"SmartTurn x5 th={th}", "st5", th) for th in (0.5, 0.8)]
        sets = [x for x in ("tts", "cut") if any(r.get("set") == x for r in recs)]
        for sname in sets:
            inc = [r for r in recs if r["kind"] == "incomplete" and r["set"] == sname]
            print(f"\n### [{sname}] false cut-offs (turn ended during the mid-thought pause) / n")
            print("| policy | pause | " + " | ".join(langs) + " | all |")
            print("|---|---|" + "---|" * (len(langs) + 1))
            for label, mode, th in pols:
                for p in PAUSES_MS:
                    row, tot = [], [0, 0]
                    for lg in langs:
                        rs = [r for r in inc if r["pause_ms"] == p and r["lang"] == lg]
                        n = sum(classify(r, mode, th)["kind"] == "cut" for r in rs)
                        row.append(f"{n}/{len(rs)}"); tot[0] += n; tot[1] += len(rs)
                    print(f"| {label} | {p} ms | " + " | ".join(row) + f" | {tot[0]}/{tot[1]} |")
        print("\n### end-of-turn delay after true end of speech (ms): mean [p50 / max]")
        print("| policy | set | " + " | ".join(langs) + " | all |")
        print("|---|---|" + "---|" * (len(langs) + 1))
        for label, mode, th in pols:
            for setname, sel in [("complete", lambda r: r["kind"] == "complete")] + \
                    [(f"end of B, {x}", (lambda x: lambda r: r["kind"] == "incomplete" and r["set"] == x)(x)) for x in sets]:
                row, allv = [], []
                for lg in langs:
                    vals = [classify(r, mode, th) for r in recs if sel(r) and r["lang"] == lg]
                    d = [v["delay"] * 1000 for v in vals if v["kind"] == "end"]
                    allv += d
                    row.append(f"{np.mean(d):.0f} [{np.median(d):.0f}/{np.max(d):.0f}]" if d else "-")
                print(f"| {label} | {setname} | " + " | ".join(row)
                      + (f" | {np.mean(allv):.0f} [{np.median(allv):.0f}/{np.max(allv):.0f}] |" if allv else " | - |"))
        # how often does Smart Turn fire on the terminal end at all (otherwise the 700 ms fallback applies)
        print("\n### Smart Turn P(complete) at the 224 ms checkpoint")
        print("| point | lang | n | mean P | min | share >= 0.5 | share >= 0.8 |")
        print("|---|---|---|---|---|---|---|")
        mid = lambda sn, pm: (lambda r, k, e: r["kind"] == "incomplete" and r["set"] == sn and r["pause_ms"] == pm
                              and abs(e["first_t"] - r["speech_ends"][0]) < 0.5)
        end_ = lambda sn: (lambda r, k, e: r["kind"] == "incomplete" and r["set"] == sn and k == len(r["eps"]) - 1)
        for label, sel in ([("true end (complete)", lambda r, k, e: r["kind"] == "complete" and k == len(r["eps"]) - 1)]
                           + [(f"true end after pause [{sn}]", end_(sn)) for sn in sets]
                           + [(f"mid pause {pm} [{sn}]", mid(sn, pm)) for sn in sets for pm in PAUSES_MS]):
            for lg in langs + ["all"]:
                ps = []
                for r in recs:
                    if lg != "all" and r["lang"] != lg:
                        continue
                    for k, e in enumerate(r["eps"]):
                        v = (e["cps"].get(str(CHECKPOINTS[0])) or e["cps"].get(CHECKPOINTS[0]))
                        if v and sel(r, k, e):
                            ps.append(v["p"])
                if ps:
                    print(f"| {label} | {lg} | {len(ps)} | {np.mean(ps):.2f} | {np.min(ps):.2f} | "
                          f"{np.mean(np.array(ps) >= .5):.0%} | {np.mean(np.array(ps) >= .8):.0%} |")
        ms = [v["ms"] for r in recs for e in r["eps"] for v in e["cps"].values()]
        print(f"\nSmart Turn inference inside eval (features+ORT, concurrent load): n={len(ms)} p50={np.median(ms):.0f} ms "
              f"p95={np.percentile(ms, 95):.0f} ms")
        # spurious in-speech episodes that reached a checkpoint (not the planted pause, not the terminal end)
        inc = 0
        for r in recs:
            for k, e in enumerate(r["eps"]):
                planted = r["kind"] == "incomplete" and abs(e["first_t"] - r["speech_ends"][0]) < 0.5
                if not planted and k != len(r["eps"]) - 1 and e["cps"]:
                    inc += 1
        print(f"incidental silent stretches inside speech that reached a checkpoint: {inc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gen", "bench", "eval", "report"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", help="path to smart-turn ONNX (bench/eval)")
    a = ap.parse_args()
    {"gen": cmd_gen, "bench": cmd_bench, "eval": cmd_eval, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
