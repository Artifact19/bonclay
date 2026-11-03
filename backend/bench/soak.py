#!/usr/bin/env python3
"""Stability soak: a weighted mix of real requests for a fixed time, with memory and health sampling.

Mix (weights): Kokoro English 30, Kokoro Hindi 20, spoken agent turn 20, transcription 15, and (with --voice-id)
cloned-voice generation 15, with small random pauses between requests. One request at a time.
Every --sample-interval seconds (default 60) it records the API process's RSS and its children's RSS (model workers)
and /api/health. It counts requests, failures (by status code or exception), restarts (API unreachable, then back,
or the server's process id changed) and per-type latency.

Per-route limits of the API (transcribe 10/min, voices/generate 10/min) are respected client side; a type whose
budget is used up is skipped for that pick. A 429 would be counted separately and is not part of the failure rate.

    backend/.venv/bin/python backend/bench/soak.py                 # 45 minutes
    backend/.venv/bin/python backend/bench/soak.py --minutes 120 --voice-id <id>
    backend/.venv/bin/python backend/bench/soak.py --quick         # 2 minutes
"""
import argparse
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from urllib.parse import urlparse

import numpy as np
import psutil
import requests

from common import (LIMITS, Api, RateGate, add_common_args, delete_agent, ensure_agent, err_text, fmt, kokoro_voices,
                    kokoro_wav, new_result, pick_voice, print_header, require_api, spoken_turn, stats, write_result)

TEXTS_EN = ["Your appointment is confirmed for Thursday at four in the afternoon.",
            "The clinic is open from nine in the morning until eight in the evening.",
            "Please arrive ten minutes early and bring your old reports.",
            "The pharmacy inside the clinic closes at half past seven.",
            "Thank you for calling Sunrise Clinic, how can I help you today?"]
TEXTS_HI = ["आपका अपॉइंटमेंट गुरुवार को शाम चार बजे तय हो गया है।",
            "क्लिनिक सुबह नौ बजे से रात आठ बजे तक खुला रहता है।",
            "कृपया दस मिनट पहले आएं और अपनी पुरानी रिपोर्ट साथ लाएं।",
            "क्लिनिक के अंदर की दवा की दुकान साढ़े सात बजे बंद होती है।"]
Q_EN = ["What time does the clinic open on Monday?", "How much is a dentist consultation?",
        "Is the clinic open on Sunday?"]
Q_HI = ["क्लिनिक का फोन नंबर क्या है?", "अपॉइंटमेंट कैसे बुक करूं?"]
WEIGHTS = {"kokoro_en": 30, "kokoro_hi": 20, "turn": 20, "transcribe": 15, "clone_generate": 15}


def listener_pid(port: int):
    """Process id listening on the TCP port (lsof first: psutil needs root for this on macOS)."""
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True,
                             timeout=10).stdout.split()
        if out:
            return int(out[0])
    except Exception:  # noqa: BLE001
        pass
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
                return c.pid
    except Exception:  # noqa: BLE001
        pass
    return None


def footprint_mb(pid):
    """macOS physical footprint (dirty + compressed + swapped memory, the figure Activity Monitor calls Memory), in MB.

    RSS counts only pages that are resident right now, so on a Mac under memory pressure it falls when pages are
    compressed or swapped out and rises when they are touched again: it can show growth or shrinkage that is not
    real. The footprint does not move with that. None off macOS or when it cannot be read."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("proc"))
        names = ("ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins ri_wired_size "
                 "ri_resident_size ri_phys_footprint").split()

        class Rusage(ctypes.Structure):  # struct rusage_info_v4, padded: the kernel writes the full struct
            _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in names] + \
                       [(f"pad{i}", ctypes.c_uint64) for i in range(40)]
        ru = Rusage()
        if lib.proc_pid_rusage(int(pid), 4, ctypes.byref(ru)) != 0:
            return None
        return round(ru.ri_phys_footprint / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        return None


def memory_sample(pid):
    if not pid:
        return None
    try:
        p = psutil.Process(pid)
        mb = lambda x: round(x / 2 ** 20, 1)  # noqa: E731
        kids = []
        for c in p.children(recursive=True):
            try:
                kids.append({"pid": c.pid, "name": " ".join(c.cmdline()[:3])[:90], "rss_mb": mb(c.memory_info().rss),
                             "footprint_mb": footprint_mb(c.pid)})
            except psutil.Error:
                pass
        api_mb = mb(p.memory_info().rss)
        kids_mb = round(sum(k["rss_mb"] for k in kids), 1)
        out = {"pid": pid, "api_rss_mb": api_mb, "children_rss_mb": kids_mb, "total_rss_mb": round(api_mb + kids_mb, 1),
               "children": kids}
        api_fp = footprint_mb(pid)
        if api_fp is not None:
            kids_fp = round(sum(k["footprint_mb"] or 0 for k in kids), 1)
            out.update({"api_footprint_mb": api_fp, "children_footprint_mb": kids_fp,
                        "total_footprint_mb": round(api_fp + kids_fp, 1)})
        return out
    except psutil.Error:
        return None


class Sampler(threading.Thread):
    def __init__(self, base: str, port: int, local: bool, interval: float):
        super().__init__(daemon=True)
        self.api, self.port, self.local, self.interval = Api(base), port, local, interval
        self.samples, self.stop_evt = [], threading.Event()

    def sample_once(self, t0: float):
        s = {"t_s": round(time.time() - t0, 1)}
        try:
            r, secs, _ = self.api.get("/api/health", timeout=15)
            h = r.json() if r.status_code == 200 else {}
            s.update({"health_ok": r.status_code == 200, "health_s": round(secs, 3),
                      "models_loaded": h.get("models_loaded"), "llm_running": (h.get("llm") or {}).get("local_running")})
        except Exception as e:  # noqa: BLE001
            s.update({"health_ok": False, "health_error": type(e).__name__})
        try:  # context for reading RSS: macOS compresses and swaps pages under memory pressure, which lowers RSS
            vm, sw = psutil.virtual_memory(), psutil.swap_memory()
            s.update({"system_mem_used_pct": vm.percent, "swap_used_mb": round(sw.used / 2 ** 20)})
        except Exception:  # noqa: BLE001
            pass
        if self.local:
            m = memory_sample(listener_pid(self.port))
            s.update(m or {"pid": None})
        self.samples.append(s)
        return s

    def run(self):
        t0 = self.t0
        while not self.stop_evt.wait(self.interval):
            s = self.sample_once(t0)
            print(f"  [sample {s['t_s'] / 60:5.1f} min] health {'ok' if s.get('health_ok') else 'FAIL'}  "
                  f"api {fmt(s.get('api_rss_mb'))} MB  children {fmt(s.get('children_rss_mb'))} MB  "
                  f"(footprint: api {fmt(s.get('api_footprint_mb'))}, children {fmt(s.get('children_footprint_mb'))})",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--minutes", type=float, default=None, help="duration (default 45; quick 2)")
    ap.add_argument("--voice-id", help="cloned voice id: adds cloned-voice generation to the mix")
    ap.add_argument("--agent-id", help="use this agent instead of creating one from the clinic_receptionist template")
    ap.add_argument("--keep-agent", action="store_true")
    ap.add_argument("--sample-interval", type=float, default=None, help="seconds between samples (default 60; quick 15)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    minutes = a.minutes if a.minutes is not None else (2 if a.quick else 45)
    interval = a.sample_interval if a.sample_interval is not None else (15 if a.quick else 60)
    rng = random.Random(a.seed)

    api = Api(a.api)
    health = require_api(api)
    u = urlparse(api.base)
    port = u.port or (443 if u.scheme == "https" else 80)
    local = u.hostname in ("127.0.0.1", "localhost", "::1")
    weights = dict(WEIGHTS)
    if not a.voice_id:
        weights.pop("clone_generate")
    settings = {"quick": a.quick, "minutes": minutes, "sample_interval_s": interval, "weights": weights,
                "voice_id": a.voice_id, "pause_s": [0.2, 1.5] if a.quick else [0.5, 4.0], "seed": a.seed,
                "memory_sampled": local, "port": port}
    res = new_result("soak", api, health, settings)
    print_header(f"Soak ({minutes:g} min)", res)

    agent_id, created = ensure_agent(api, a.agent_id)
    res["agent_id"] = agent_id
    sampler = Sampler(api.base, port, local, interval)
    log, restarts, down_events = [], 0, []
    try:
        avail = kokoro_voices(api)
        v_en = pick_voice(avail, ["af_heart"]) or "af_heart"
        v_hi = pick_voice(avail, ["hf_alpha"]) or "hf_alpha"
        print("rendering test audio (not timed)...")
        q_wavs = [kokoro_wav(api, v_en, t)[0] for t in Q_EN] + [kokoro_wav(api, v_hi, t)[0] for t in Q_HI]
        asr_wavs = [kokoro_wav(api, v_en, t)[0] for t in TEXTS_EN[:3]]
        gates = {"transcribe": RateGate(LIMITS["transcribe"]), "clone_generate": RateGate(LIMITS["voices_generate"])}

        def do(kind: str):
            """One request. Returns (status, seconds, extra)."""
            if kind == "kokoro_en":
                r, s, _ = api.post("/api/kokoro/generate", json={"voice": v_en, "text": rng.choice(TEXTS_EN)}, timeout=180)
            elif kind == "kokoro_hi":
                r, s, _ = api.post("/api/kokoro/generate", json={"voice": v_hi, "text": rng.choice(TEXTS_HI)}, timeout=180)
            elif kind == "turn":
                r, s = spoken_turn(api, agent_id, rng.choice(q_wavs), None, timeout=300)
            elif kind == "transcribe":
                r, s, _ = api.post("/api/transcribe", files={"audio_file": ("a.wav", rng.choice(asr_wavs), "audio/wav")},
                                   data={"language": "en"}, timeout=300, max_429=0)
            else:
                r, s, _ = api.post("/api/voices/generate", data={"text": rng.choice(TEXTS_EN), "voice_id": a.voice_id,
                                                                 "language": "en"}, timeout=600, max_429=0)
            extra = {}
            if kind == "turn" and r.status_code == 200:
                extra["fallback"] = bool(r.json().get("fallback"))
            return r, s, extra

        t0 = time.time()
        sampler.t0 = t0
        deadline = t0 + minutes * 60
        sampler.sample_once(t0)
        sampler.start()
        down_since = None
        n = 0
        while time.time() < deadline:
            kinds = [k for k in weights if k not in gates or gates[k].seconds_until_free() <= 0]
            if not kinds:
                time.sleep(2)
                continue
            kind = rng.choices(kinds, weights=[weights[k] for k in kinds])[0]
            if kind in gates:
                gates[kind].try_acquire()
            rec = {"t_s": round(time.time() - t0, 1), "kind": kind}
            try:
                r, secs, extra = do(kind)
                rec.update({"status": r.status_code, "seconds": round(secs, 3), **extra})
                if r.status_code != 200:
                    rec["error"] = err_text(r)
                if down_since is not None:
                    restarts += 1
                    down_events.append({"down_at_s": round(down_since - t0, 1), "back_at_s": rec["t_s"]})
                    print(f"  API is back after {time.time() - down_since:.0f}s (counted as a restart)", flush=True)
                    down_since = None
            except requests.ConnectionError as e:
                rec.update({"status": None, "exception": type(e).__name__})
                if down_since is None:
                    down_since = time.time()
                    print(f"  API UNREACHABLE at {rec['t_s']}s", flush=True)
            except Exception as e:  # noqa: BLE001  (timeouts, etc.)
                rec.update({"status": None, "exception": type(e).__name__})
            log.append(rec)
            n += 1
            if n % 10 == 0:
                print(f"  {n} requests, {(time.time() - t0) / 60:.1f}/{minutes:g} min, "
                      f"{sum(1 for x in log if x.get('status') != 200)} not-OK", flush=True)
            if down_since is not None:  # do not hammer a dead server: poll health until it is back
                while time.time() < deadline:
                    time.sleep(2)
                    try:
                        if Api(api.base).get("/api/health", timeout=5)[0].status_code == 200:
                            break
                    except Exception:  # noqa: BLE001
                        pass
                continue
            time.sleep(rng.uniform(*settings["pause_s"]))
        sampler.stop_evt.set()
        sampler.join(timeout=5)
        sampler.sample_once(t0)
        res["still_down_at_end"] = down_since is not None
        res["elapsed_s"] = round(time.time() - t0, 1)
    except KeyboardInterrupt:
        print("interrupted; writing what was collected")
        res["interrupted"] = True
        sampler.stop_evt.set()
    finally:
        sampler.stop_evt.set()
        if created and not a.keep_agent:
            delete_agent(api, agent_id)
        res["items"] = log
        res["memory_samples"] = sampler.samples
        res["restart_events"] = down_events
        res["restarts_detected_by_reachability"] = restarts
        summarize(res)
        print_summary(res)
        print(f"\nraw results: {write_result(res)}")


def trend(samples, key):
    pts = [(s["t_s"] / 60.0, s[key]) for s in samples if s.get(key) is not None]
    if len(pts) < 2:
        return None
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    slope = float(np.polyfit(x, y, 1)[0]) if x.max() > x.min() else 0.0
    return {"samples": len(pts), "start_mb": float(y[0]), "end_mb": float(y[-1]), "min_mb": float(y.min()),
            "max_mb": float(y.max()), "slope_mb_per_min": slope}


def summarize(res: dict) -> None:
    items = res["items"]
    fails, per_type = Counter(), defaultdict(list)
    rate_limited = 0
    for r in items:
        if r.get("status") == 200:
            per_type[r["kind"]].append(r["seconds"] * 1000)
        elif r.get("status") == 429:
            rate_limited += 1
        else:
            fails[str(r["status"]) if r.get("status") else r.get("exception", "unknown")] += 1
    samples = res.get("memory_samples", [])
    pids = [s["pid"] for s in samples if s.get("pid")]
    pid_changes = sum(1 for a, b in zip(pids, pids[1:]) if a != b)
    restarts = max(res.get("restarts_detected_by_reachability", 0), pid_changes)
    crashes = restarts + (1 if res.get("still_down_at_end") else 0)
    nfail = sum(fails.values())
    non_rl = len(items) - rate_limited
    res["summary"] = {
        "requests": len(items), "ok": sum(len(v) for v in per_type.values()), "failures": nfail,
        "failure_rate": (nfail / non_rl) if non_rl else None, "failures_by_cause": dict(fails),
        "rate_limited_429": rate_limited, "restarts": restarts, "server_pid_changes": pid_changes,
        "still_down_at_end": res.get("still_down_at_end", False), "crashes": crashes,
        "fallback_replies": sum(1 for r in items if r.get("fallback")),
        "health_failures": sum(1 for s in samples if s.get("health_ok") is False),
        "latency_ms": {k: stats(v) for k, v in per_type.items()},
        "count_by_type": dict(Counter(r["kind"] for r in items)),
        "memory": {"api_rss": trend(samples, "api_rss_mb"), "children_rss": trend(samples, "children_rss_mb"),
                   "total_rss": trend(samples, "total_rss_mb"),
                   "api_footprint": trend(samples, "api_footprint_mb"),
                   "children_footprint": trend(samples, "children_footprint_mb"),
                   "total_footprint": trend(samples, "total_footprint_mb")},
        "elapsed_s": res.get("elapsed_s"),
    }


def print_summary(res: dict) -> None:
    s = res["summary"]
    print(f"\nrequests {s['requests']}, ok {s['ok']}, failures {s['failures']} "
          f"(rate {fmt(None if s['failure_rate'] is None else s['failure_rate'] * 100, 2)}%), "
          f"429s {s['rate_limited_429']}, fallback replies {s['fallback_replies']}")
    if s["failures_by_cause"]:
        print("  failures by cause:", s["failures_by_cause"])
    print(f"crashes/restarts: {s['crashes']} (target 0); health failures in samples: {s['health_failures']}")
    for k, v in s["latency_ms"].items():
        print(f"  {k:15s} n={v['n']:<4d} p50 {fmt(v['p50'])} ms  p95 {fmt(v['p95'])} ms")
    for k, m in s["memory"].items():
        if m:
            print(f"  memory {k:13s} start {m['start_mb']:.0f} MB, end {m['end_mb']:.0f} MB, max {m['max_mb']:.0f} MB, "
                  f"trend {m['slope_mb_per_min']:+.2f} MB/min over {m['samples']} samples")


if __name__ == "__main__":
    main()
