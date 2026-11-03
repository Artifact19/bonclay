"""Shared helpers for the BonClay measurement scripts.

Every script talks to the running API over HTTP (BONCLAY_API, default http://127.0.0.1:8000) with the header
`storage-mode: local`, so it measures the real system, not a mock. Nothing here imports the backend code.
"""
import argparse
import collections
import datetime as dt
import io
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import soundfile as sf

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
REPO_ROOT = BENCH_DIR.parent.parent
DEFAULT_API = os.getenv("BONCLAY_API", "http://127.0.0.1:8000")

# Per-route limits the API enforces per client IP (slowapi, see backend.py). The scripts stay under them.
LIMITS = {"transcribe": 10, "voices_generate": 10, "voices_clone": 5, "indic_generate": 6}  # requests per minute


class ApiError(RuntimeError):
    def __init__(self, status: Optional[int], message: str):
        super().__init__(f"HTTP {status}: {message}" if status else message)
        self.status = status
        self.message = message


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default: BONCLAY_API or {DEFAULT_API})")
    p.add_argument("--quick", action="store_true", help="tiny smoke-test version (numbers are not meaningful)")


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


# ----------------------------------------------------------------------------------------------------- HTTP

class Api:
    """requests wrapper: base URL, `storage-mode: local`, and a polite retry when the API answers 429.

    Time spent waiting out a 429 is returned separately (`waited`) so it never ends up in a latency number."""

    def __init__(self, base: str = DEFAULT_API):
        self.base = base.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"storage-mode": "local"})

    def request(self, method: str, path: str, *, timeout: float = 180, max_429: int = 6, **kw):
        """Returns (response, seconds, waited_seconds). `seconds` is the last attempt only, request start until the
        full response body is received. Raises requests exceptions on network failure; never on HTTP status."""
        waited = 0.0
        for attempt in range(max_429 + 1):
            t0 = time.perf_counter()
            r = self.s.request(method, self.base + path, timeout=timeout, **kw)
            _ = r.content  # make sure the body is read inside the timed span
            dt_s = time.perf_counter() - t0
            if r.status_code != 429 or attempt == max_429:
                return r, dt_s, waited
            try:
                wait = float(r.headers.get("Retry-After", "60"))
            except ValueError:
                wait = 60.0
            wait = min(max(wait, 1.0), 70.0)
            print(f"    (rate limited on {path}, waiting {wait:.0f}s)", flush=True)
            time.sleep(wait)
            waited += wait
        raise AssertionError("unreachable")

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, **kw):
        return self.request("POST", path, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)


def err_text(r: requests.Response) -> str:
    try:
        j = r.json()
        err = j.get("error", j) if isinstance(j, dict) else j
        return json.dumps(err)[:300]
    except ValueError:
        return (r.text or "")[:300]


def require_api(api: Api) -> dict:
    """Health check up front; exit with a hint when the API is not running."""
    try:
        r, _, _ = api.get("/api/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Cannot reach the BonClay API at {api.base} ({type(e).__name__}). "
                 "Start it first with `./bonclay start` from the repo root.")


class RateGate:
    """Client-side sliding window so a script stays under a per-minute API limit (which is per client IP)."""

    def __init__(self, per_minute: int, window: float = 60.0, margin: int = 1):
        self.cap = max(1, per_minute - margin)
        self.window = window
        self.t: collections.deque = collections.deque()

    def _trim(self, now: float) -> None:
        while self.t and now - self.t[0] >= self.window:
            self.t.popleft()

    def seconds_until_free(self) -> float:
        now = time.monotonic()
        self._trim(now)
        return 0.0 if len(self.t) < self.cap else self.window - (now - self.t[0]) + 0.5

    def wait(self) -> float:
        """Block until a request is allowed, then record it. Returns the seconds waited."""
        w = self.seconds_until_free()
        if w > 0:
            print(f"    (pacing: waiting {w:.0f}s to stay under the API rate limit)", flush=True)
            time.sleep(w)
        self.t.append(time.monotonic())
        return w

    def try_acquire(self) -> bool:
        if self.seconds_until_free() > 0:
            return False
        self.t.append(time.monotonic())
        return True


# --------------------------------------------------------------------------------------------- audio / TTS

def wav_info(data: bytes) -> Dict[str, Any]:
    info = sf.info(io.BytesIO(data))
    return {"seconds": info.frames / float(info.samplerate), "sample_rate": info.samplerate, "bytes": len(data)}


def kokoro_voices(api: Api) -> List[str]:
    r, _, _ = api.get("/api/kokoro/voices")
    if r.status_code != 200:
        return []
    return [v["id"] for v in r.json().get("voices", []) if v.get("available", True)]


def pick_voice(available: List[str], preferred: List[str]) -> Optional[str]:
    for v in preferred:
        if v in available:
            return v
    return None


def kokoro_wav(api: Api, voice: str, text: str, timeout: float = 120):
    """(wav bytes, seconds). Raises ApiError on failure."""
    r, secs, _ = api.post("/api/kokoro/generate", json={"voice": voice, "text": text}, timeout=timeout)
    if r.status_code != 200:
        raise ApiError(r.status_code, err_text(r))
    return r.content, secs


# ------------------------------------------------------------------------------------------------- stats

def pct(values: List[float], q: float) -> Optional[float]:
    v = [x for x in values if x is not None]
    return float(np.percentile(v, q)) if v else None


def stats(values: List[float]) -> Dict[str, Any]:
    v = [x for x in values if x is not None]
    if not v:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {"n": len(v), "mean": float(np.mean(v)), "p50": pct(v, 50), "p95": pct(v, 95),
            "min": float(min(v)), "max": float(max(v))}


def fmt(x: Optional[float], digits: int = 0, unit: str = "") -> str:
    return "n/a" if x is None else f"{x:.{digits}f}{unit}"


# ----------------------------------------------------------------------------------- machine / run metadata

def _sh(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=REPO_ROOT).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def git_info() -> Dict[str, Any]:
    dirty = _sh(["git", "status", "--porcelain"])
    return {"commit": _sh(["git", "rev-parse", "--short", "HEAD"]), "dirty_working_tree": bool(dirty)}


def machine_info() -> Dict[str, Any]:
    import psutil
    cpu = None
    if sys.platform == "darwin":
        cpu = _sh(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    try:
        load = [round(x, 2) for x in psutil.getloadavg()]
    except (AttributeError, OSError):
        load = None
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu": cpu or platform.processor(),
            "cpu_count": psutil.cpu_count(logical=True), "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
            "python": platform.python_version(), "load_avg_at_start": load}


def health_summary(health: dict) -> Dict[str, Any]:
    """The parts of /api/health that say what was being measured."""
    return {"platform": health.get("platform"), "llm": health.get("llm"), "models_loaded": health.get("models_loaded"),
            "gpu": health.get("gpu")}


def new_result(name: str, api: Api, health: dict, settings: dict) -> Dict[str, Any]:
    return {"name": name, "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "git": git_info(),
            "machine": machine_info(), "api": api.base, "health": health_summary(health), "settings": settings,
            "quick": bool(settings.get("quick")), "items": [], "summary": {}}


def write_result(result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result['name']}-{now_stamp()}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def print_header(title: str, result: dict) -> None:
    m, g = result["machine"], result["git"]
    llm = (result["health"].get("llm") or {})
    print(f"\n=== {title} ===")
    print(f"machine: {m['cpu']}, {m['ram_gb']} GB RAM, {m['platform']}; commit {g['commit']}"
          f"{' (dirty)' if g['dirty_working_tree'] else ''}")
    print(f"llm: {llm.get('active')} {llm.get('model')}   api: {result['api']}")
    if result["quick"]:
        print("QUICK SMOKE RUN: numbers are not meaningful")


# --------------------------------------------------------------------------------------------- agents

def ensure_agent(api: Api, agent_id: Optional[str], template: str = "clinic_receptionist"):
    """(agent_id, created_by_us). Creates an agent from the template when no id is given."""
    if agent_id:
        r, _, _ = api.get(f"/api/agents/{agent_id}")
        if r.status_code != 200:
            sys.exit(f"Agent {agent_id} not found: {err_text(r)}")
        return agent_id, False
    r, _, _ = api.post("/api/agents/from-template", json={"template_id": template}, timeout=300)
    if r.status_code != 200:
        sys.exit(f"Could not create an agent from template {template!r}: HTTP {r.status_code} {err_text(r)}")
    return r.json()["id"], True


def delete_agent(api: Api, agent_id: str) -> None:
    try:
        r, _, _ = api.delete(f"/api/agents/{agent_id}", timeout=60)
        print(f"deleted agent {agent_id}: HTTP {r.status_code}")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not delete agent {agent_id}: {e}")


def spoken_turn(api: Api, agent_id: str, wav: bytes, conversation_id: Optional[str] = None, timeout: float = 240):
    """POST a spoken turn. Returns (response, client_seconds)."""
    data = {"conversation_id": conversation_id} if conversation_id else None
    r, secs, _ = api.post(f"/api/conversations/{agent_id}/turn", files={"audio": ("question.wav", wav, "audio/wav")},
                          data=data, timeout=timeout)
    return r, secs
