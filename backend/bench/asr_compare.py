#!/usr/bin/env python3
"""Compare local speech-recognition strategies on synthetic clips, without the API.

Starts its own MLX ASR worker (backend/.venv-mlx, the same command services/mlx_asr.py uses) on its own port, makes
test clips with Kokoro, runs every clip through every strategy and stops the worker at the end.

Clips (all synthetic, Kokoro speech, 16 kHz mono): 20 Hindi sentences (hf_alpha), 10 Hinglish (Roman Hindi read by the
English voice af_heart) and 10 English (af_heart). Every second clip also gets a noisy copy (white noise, SNR 20 dB).

Strategies (what the worker does; see services/mlx_asr_worker.py transcribe_plan):
  hybrid    small model first, large model again if it hears an Indian language (the previous default)
  accurate  one large-v3-turbo pass, language auto-detected
  hinted    the caller knows the language: Hindi -> one large pass with language=hi, English -> small with language=en
            (Hinglish has no honest hint, so it is not run)
  auto      small model detects the language only, then one large pass (Indian) or one small pass (other)

Metrics: latency per clip (client wall time, after one warm-up request per strategy; mean/median/p95), WER (English and
Hinglish: lowercase, punctuation removed) or CER (Hindi: Devanagari letters and marks only) with jiwer, and language-id
correctness (Hindi must come back as "hi", English as "en"; Hinglish is reported as a distribution because either answer
is defensible). Hinglish references are Roman, so a transcript in Devanagari scores badly: the share of Devanagari
transcripts is reported too.

    backend/.venv/bin/python backend/bench/asr_compare.py
    backend/.venv/bin/python backend/bench/asr_compare.py --quick        # 4+2+2 clips, for a smoke test
    backend/.venv/bin/python backend/bench/asr_compare.py --strategies hybrid,accurate --keep-audio
Needs the models downloaded (bash backend/scripts/setup_mlx.sh). About 2 GB of memory for the worker plus Kokoro while
the clips are made (Kokoro is freed before the worker starts).
"""
import argparse
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)

from services import asr_scoring  # noqa: E402  (script-normalised Hinglish scoring: wer_raw, wer_folded, cer_folded)

HINGLISH = [
    "Mera naam Rahul hai aur main Pune mein rehta hoon.",
    "Kal subah nau baje doctor se appointment hai.",
    "Aap please apni report kal tak bhej dijiye.",
    "Mujhe ek glass paani aur do chai chahiye.",
    "Yeh train Mumbai kab pahunchegi, kuch pata hai?",
    "Hum shaam ko market jaa rahe hain, aap bhi chaliye.",
    "Meri behen ne aaj naya phone kharida hai.",
    "Kya aap mujhe apna phone number bata sakte hain?",
    "Bahut shukriya, aapki madad se mera kaam ho gaya.",
    "Kal ki meeting ka time badal gaya hai, ab teen baje hogi.",
]
LABELS = {"hybrid": "(a) hybrid", "accurate": "(b) accurate", "hinted": "(c) hinted", "auto": "(d) auto"}


def mlx_python() -> str:
    from services import mlx_asr
    py = mlx_asr._mlx_python()
    if not py:
        sys.exit("No MLX environment (backend/.venv-mlx). Run bash backend/scripts/setup_mlx.sh")
    return py


def free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return s.getsockname()[1]
    raise RuntimeError("no free port")


# --- clips -----------------------------------------------------------------------------------------------------------

def make_clips(out_dir: str, n_hi: int, n_hinglish: int, n_en: int, snr_db: float, seed: int = 7) -> list:
    """Kokoro speech -> 16 kHz wav files. Returns [{id, kind, ref, path, noisy}]."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from readback import SENTENCES_EN, SENTENCES_HI
    from services.espeak import configure_espeak
    configure_espeak()
    from kokoro import KPipeline

    en = KPipeline(lang_code="a")
    hi = KPipeline(lang_code="h", model=en.model)
    rng = np.random.default_rng(seed)
    plan = ([("hi", t, hi, "hf_alpha") for t in SENTENCES_HI[:n_hi]]
            + [("hinglish", t, en, "af_heart") for t in HINGLISH[:n_hinglish]]
            + [("en", t, en, "af_heart") for t in SENTENCES_EN[:n_en]])
    clips, counts = [], {}
    for kind, text, pipe, voice in plan:
        i = counts[kind] = counts.get(kind, 0) + 1
        chunks = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(text, voice=voice, speed=1.0)
                  if a is not None]
        audio = resample_poly(np.concatenate(chunks).astype(np.float32), 2, 3).astype(np.float32)  # 24 kHz -> 16 kHz
        base = f"{kind}{i:02d}"
        path = os.path.join(out_dir, base + ".wav")
        sf.write(path, audio, 16000)
        clips.append({"id": base, "kind": kind, "ref": text, "path": path, "noisy": False,
                      "seconds": len(audio) / 16000})
        if i % 2 == 0:  # every second clip gets a noisy copy
            noise = rng.standard_normal(len(audio)).astype(np.float32)
            noise *= np.sqrt(np.mean(audio ** 2) / (10 ** (snr_db / 10)) / np.mean(noise ** 2))
            noisy = np.clip(audio + noise, -1.0, 1.0)
            npath = os.path.join(out_dir, base + "_noisy.wav")
            sf.write(npath, noisy, 16000)
            clips.append({"id": base + "_noisy", "kind": kind, "ref": text, "path": npath, "noisy": True,
                          "seconds": len(noisy) / 16000})
    del en, hi
    return clips


# --- worker ----------------------------------------------------------------------------------------------------------

class Worker:
    """The same command and environment as services/mlx_asr._Worker.ensure, on a fixed (or any free) port."""

    def __init__(self, port: int):
        self.port = free_port(port)
        self.proc = None

    def start(self, timeout: float = 600.0) -> float:
        from services import mlx_asr
        t0 = time.time()
        self.proc = subprocess.Popen(
            [mlx_python(), os.path.join(BACKEND, "services", "mlx_asr_worker.py"), "--port", str(self.port),
             "--fast", mlx_asr.FAST, "--accurate", mlx_asr.ACCURATE],
            env={**os.environ, "HF_HUB_OFFLINE": "1"})
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("MLX ASR worker exited during start-up")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2):
                    return time.time() - t0
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.5)
        raise RuntimeError("MLX ASR worker did not start in time")

    def transcribe(self, path: str, language, strategy: str) -> dict:
        body = json.dumps({"path": path, "language": language, "strategy": strategy}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/transcribe", data=body,
                                     headers={"Content-Type": "application/json"})
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.load(r)
        out["wall"] = time.perf_counter() - t
        return out

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


# --- scoring ---------------------------------------------------------------------------------------------------------

def norm_en(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s).replace("'", "")
    return re.sub(r"\s+", " ", s).strip()


def norm_hi(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return "".join(c for c in s if 0x0900 <= ord(c) <= 0x097F and unicodedata.category(c)[0] in "LM")


def error_rate(kind: str, ref: str, hyp: str) -> float:
    import jiwer
    if kind == "hi":
        r, h = norm_hi(ref), norm_hi(hyp)
        return 1.0 if not h else float(jiwer.cer(r, h))
    r, h = norm_en(ref), norm_en(hyp)
    return 1.0 if not h else float(jiwer.wer(r, h))


def has_devanagari(s: str) -> bool:
    return any(0x0900 <= ord(c) <= 0x097F for c in s)


def request_for(strategy: str, clip: dict):
    """(language hint, worker strategy), or None when the strategy does not apply to this clip."""
    if strategy == "hinted":
        if clip["kind"] == "hinglish":
            return None
        return ("hi" if clip["kind"] == "hi" else "en"), "hybrid"
    return None, strategy


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summarize(rows: list) -> dict:
    lat = [r["wall"] for r in rows]
    err = [r["err"] for r in rows]
    out = {"n": len(rows), "lat_mean": statistics.mean(lat), "lat_median": statistics.median(lat),
           "lat_p95": pct(lat, 0.95), "err_mean": statistics.mean(err), "err_median": statistics.median(err)}
    ided = [r for r in rows if r["id_ok"] is not None]
    out["id_ok"] = sum(1 for r in ided if r["id_ok"]) if ided else None
    out["id_n"] = len(ided)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8767, help="worker port (default 8767, or any free port)")
    ap.add_argument("--strategies", default="hybrid,accurate,hinted,auto")
    ap.add_argument("--quick", action="store_true", help="4 Hindi + 2 Hinglish + 2 English clips")
    ap.add_argument("--rounds", type=int, default=3, help="repeat the whole set this many times (default 3)")
    ap.add_argument("--snr", type=float, default=20.0, help="noise level of the noisy copies, dB")
    ap.add_argument("--keep-audio", action="store_true", help="keep the generated clips (prints the folder)")
    ap.add_argument("--json", help="also write the raw per-clip results to this file")
    ap.add_argument("--rescore", help="recompute the Hinglish scores of a saved --json file and exit (no models)")
    a = ap.parse_args()
    if a.rescore:
        rescore(a.rescore)
        return
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in LABELS:
            sys.exit(f"Unknown strategy {s!r}; choose from {', '.join(LABELS)}")
    n_hi, n_hg, n_en = (4, 2, 2) if a.quick else (20, 10, 10)

    tmp = tempfile.mkdtemp(prefix="asr_compare_")
    worker = Worker(a.port)
    try:
        print(f"Making clips in {tmp} ...", flush=True)
        t0 = time.time()
        clips = make_clips(tmp, n_hi, n_hg, n_en, a.snr)
        print(f"  {len(clips)} clips ({sum(c['noisy'] for c in clips)} noisy) in {time.time() - t0:.0f}s", flush=True)
        print(f"Starting the MLX ASR worker on port {worker.port} ...", flush=True)
        print(f"  ready in {worker.start():.0f}s", flush=True)

        warm = {k: next(c for c in clips if c["kind"] == k and not c["noisy"]) for k in ("hi", "en")}
        for strat in strategies:
            for c in warm.values():  # warm-up: one Hindi and one English request per strategy, not timed
                req = request_for(strat, c)
                worker.transcribe(c["path"], req[0], req[1])
        # Strategies are interleaved clip by clip, in a rotating order, and the whole set is repeated `rounds` times, so
        # background load on the machine hits every strategy alike. A clip's latency is its median over the rounds.
        load_before = os.getloadavg()
        samples = {}
        for rnd in range(a.rounds):
            for ci, c in enumerate(clips):
                order = strategies[(ci + rnd) % len(strategies):] + strategies[:(ci + rnd) % len(strategies)]
                for strat in order:
                    req = request_for(strat, c)
                    if req is None:
                        continue
                    samples.setdefault((strat, c["id"]), []).append(worker.transcribe(c["path"], req[0], req[1]))
            print(f"  round {rnd + 1}/{a.rounds} done", flush=True)
        load_after = os.getloadavg()
        rows = []
        for (strat, cid), outs in samples.items():
            c = next(x for x in clips if x["id"] == cid)
            out = outs[0]  # greedy decoding is deterministic; the text is the same each round
            lang = out.get("language")
            id_ok = {"hi": lang == "hi", "en": lang == "en"}.get(c["kind"])
            if strat == "hinted":
                id_ok = None  # the language was given, not detected
            row = {"strategy": strat, "clip": cid, "kind": c["kind"], "noisy": c["noisy"],
                   "seconds_audio": round(c["seconds"], 2), "wall": statistics.median(o["wall"] for o in outs),
                   "server": out.get("seconds"), "model": out.get("model"), "language": lang,
                   "text": out["text"], "err": error_rate(c["kind"], c["ref"], out["text"]), "id_ok": id_ok,
                   "devanagari": has_devanagari(out["text"]), "ref": c["ref"],
                   "texts_differ_between_rounds": len({o["text"] for o in outs}) > 1}
            if c["kind"] == "hinglish":   # the raw WER punishes a Devanagari transcript of a Roman reference
                row.update(asr_scoring.hinglish_scores(c["ref"], out["text"]))
            rows.append(row)
    finally:
        worker.stop()
        if a.keep_audio:
            print(f"Clips kept in {tmp}")
        else:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)

    print()
    print(f"ASR strategy comparison, {datetime.now(timezone.utc).strftime('%Y-%m-%d')}, {platform.platform()}, "
          f"{platform.machine()}; clips are synthetic (Kokoro); latency = client wall time, warm, median of {a.rounds} "
          f"interleaved rounds per clip; load average {load_before[0]:.1f} before, {load_after[0]:.1f} after.")
    print()
    print("| Clips | Strategy | n | Latency mean (s) | median (s) | p95 (s) | Error mean | Error median | Language id |")
    print("|---|---|---|---|---|---|---|---|---|")
    groups = [("Hindi, all clips (CER)", lambda r: r["kind"] == "hi"),
              ("Hindi, noisy copies only (CER)", lambda r: r["kind"] == "hi" and r["noisy"]),
              ("Hinglish, all (WER)", lambda r: r["kind"] == "hinglish"),
              ("English, all clips (WER)", lambda r: r["kind"] == "en"),
              ("English, noisy copies only (WER)", lambda r: r["kind"] == "en" and r["noisy"])]
    for name, sel in groups:
        for strat in strategies:
            sub = [r for r in rows if r["strategy"] == strat and sel(r)]
            if not sub:
                continue
            m = summarize(sub)
            ident = f"{m['id_ok']}/{m['id_n']}" if m["id_n"] else "given" if strat == "hinted" else "n/a"
            print(f"| {name} | {LABELS[strat]} | {m['n']} | {m['lat_mean']:.2f} | {m['lat_median']:.2f} | "
                  f"{m['lat_p95']:.2f} | {m['err_mean']:.3f} | {m['err_median']:.3f} | {ident} |")
    print()
    base = strategies[0]
    for kind, name in (("hi", "Hindi"), ("en", "English")):
        ref = {r["clip"]: r for r in rows if r["strategy"] == base and r["kind"] == kind}
        for strat in strategies[1:]:
            pairs = [(ref[r["clip"]], r) for r in rows if r["strategy"] == strat and r["kind"] == kind and r["clip"] in ref]
            if pairs:
                dl = statistics.mean(r["wall"] - b["wall"] for b, r in pairs)
                de = statistics.mean(r["err"] - b["err"] for b, r in pairs)
                print(f"{name}, {LABELS[strat]} vs {LABELS[base]}, paired over {len(pairs)} clips: "
                      f"latency {dl:+.2f} s, error {de:+.4f}")
    print()
    print_hinglish(rows, strategies)


def print_hinglish(rows: list, strategies: list) -> None:
    """The Hinglish rows three ways: raw WER (Roman reference against the transcript as it came), WER after both sides
    are brought to one Roman spelling, and CER on that (services/asr_scoring.py). The folded numbers are the honest ones."""
    for strat in strategies:
        sub = [r for r in rows if r["strategy"] == strat and r["kind"] == "hinglish"]
        if sub:
            langs = {}
            for r in sub:
                langs[r["language"]] = langs.get(r["language"], 0) + 1
            scored = [r for r in sub if "wer_folded" in r]
            folded = (f"; WER raw {statistics.mean(r['wer_raw'] for r in scored):.3f} -> folded "
                      f"{statistics.mean(r['wer_folded'] for r in scored):.3f}, CER folded "
                      f"{statistics.mean(r['cer_folded'] for r in scored):.3f} (n={len(scored)})") if scored else ""
            print(f"Hinglish {LABELS[strat]}: languages {langs}; "
                  f"{sum(r['devanagari'] for r in sub)}/{len(sub)} transcripts in Devanagari{folded}")


def rescore(path: str) -> None:
    """Recompute the Hinglish scores of a saved --json result file (no models needed) and print them."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    for r in rows:
        if r.get("kind") == "hinglish" and "ref" in r and "text" in r:
            r.update(asr_scoring.hinglish_scores(r["ref"], r["text"]))
    strategies = sorted({r["strategy"] for r in rows if r.get("kind") == "hinglish"}, key=list(LABELS).index)
    print_hinglish(rows, strategies)
    for r in rows:
        if r.get("kind") == "hinglish" and r["strategy"] == strategies[0]:
            print(f"  {r['clip']:14s} raw {r['wer_raw']:.2f} folded {r['wer_folded']:.2f} cer {r['cer_folded']:.2f} | "
                  f"{r['text'][:60]}")


if __name__ == "__main__":
    main()
