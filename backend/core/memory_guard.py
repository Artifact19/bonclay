"""Memory guard: check free memory before a heavy model is loaded, unload idle models first, refuse rather than swap.

The model manager calls ensure_headroom() before it starts an engine. The decision is logged on the "bonclay.memory"
logger (one line per load: ok / freed / refused) so it can be read after the fact.

Settings (all optional, see backend/.env.example):
  MEMORY_GUARD=off            switch the guard off (default on)
  MEMORY_GUARD_RESERVE_MB     memory to leave free for the OS and the app on top of the model (default 1500)
  MEMORY_NEED_<ENGINE>_MB     override the estimate for one engine, e.g. MEMORY_NEED_INDIC_SPEAK_MB=7000
  MEMORY_GUARD_HEAVY_MB       engines needing at least this much (default 1500; Kokoro and DhVaani stay below it) are "heavy": refused outright while macOS
                              reports critical memory pressure unless an idle engine was unloaded first

The estimates in DEFAULT_NEED_MB are measured (backend/bench/model_memory.py, docs/RESULTS.md "Model memory, measured
27 Oct 2026") plus 10 %, rounded.

macOS: psutil's "available" already counts inactive and purgeable pages but not the pages the compressor could still
take, so it is read as is. On top of it the guard reads the kernel's memory-pressure level (sysctl
kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical). At "warn" the reserve is multiplied by 1.5 (less room
than the page counts suggest: the system is already compressing hard). At "critical" a heavy engine is not started
unless an idle engine was unloaded to make room. Linux and Windows have no such level (pressure_level() is None), so
their behaviour is unchanged.

When room has to be made, idle engines are unloaded in the order of (measured size x seconds idle), largest first, so one
big idle model goes before several small warm ones (equal sizes: least recently used first).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from time import time as _wall_time   # separate name: tests replace this module's `time` with a stub that only sleeps
from typing import Callable, Dict, List, Optional, Tuple

from core.errors import AppError

log = logging.getLogger("bonclay.memory")

# Resident memory of each engine once loaded and warm, in MB: the measured value plus 10 %, rounded (see the module
# docstring). Engines not listed are not checked.
DEFAULT_NEED_MB: Dict[str, int] = {
    # measured steady footprint (MB) -> x1.1 -> rounded to 100; the measured value is in the comment
    "indic_speak": 8400,     # Indic-Speak 3.8B worker: 7658 (was 7000)
    "chatterbox_ml": 4500,   # Chatterbox worker, CPU: 4074 (was 3000)
    "dhvaani": 1000,         # DhVaani worker: 874 (was 1500); it peaks near 2 GB while a request runs, the reserve covers that
    "qwen3_0_6b": 2600,      # Qwen3-TTS clone worker (MLX, 8-bit): 2393 (was 1500; the cause of "2540 MB free, 3000 needed")
    "indic_f5": 3000,        # not measured: not installed on the measuring machine
    "kokoro": 1000,          # Kokoro MLX worker, English and Hindi both used: 937 (was 600)
    "background_llm": 5000,  # Gemma 4 E4B (4-bit) for background jobs: 4565 (was 3500)
}
DEFAULT_RESERVE_MB = 1500
DEFAULT_HEAVY_MB = 1500
WARN_RESERVE_FACTOR = 1.5
SIZE_FLOOR_MB = 200      # an engine with no estimate counts as this big when choosing what to unload first

PRESSURE_NORMAL, PRESSURE_WARN, PRESSURE_CRITICAL = 1, 2, 4
_PRESSURE_NAMES = {PRESSURE_NORMAL: "normal", PRESSURE_WARN: "warn", PRESSURE_CRITICAL: "critical"}


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(float(os.getenv(name, "").strip())))
    except ValueError:
        return default


def enabled() -> bool:
    return os.getenv("MEMORY_GUARD", "on").strip().lower() not in ("off", "0", "false", "no")


QUANTIZED_NEED_MB: Dict[str, int] = {"indic_speak": 3500}  # INDIC_SPEAK_MLX_MODEL set: 4-bit model; the whole worker process measured 3.1 GB


def need_mb(engine: str) -> int:
    default = DEFAULT_NEED_MB.get(engine, 0)
    if engine == "indic_speak" and os.getenv("INDIC_SPEAK_MLX_MODEL", "").strip():
        default = QUANTIZED_NEED_MB[engine]
    return _env_int(f"MEMORY_NEED_{engine.upper()}_MB", default)


def reserve_mb() -> int:
    return _env_int("MEMORY_GUARD_RESERVE_MB", DEFAULT_RESERVE_MB)


def available_mb() -> Optional[float]:
    """Memory the machine could hand out now without swapping, in MB (None when it cannot be read)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 20
    except Exception:  # noqa: BLE001
        return None


_pressure_probe_failed = False


def pressure_level() -> Optional[int]:
    """macOS memory-pressure level from the kernel: 1 normal, 2 warn, 4 critical. None off macOS or when it cannot be read
    (then nothing about the guard's behaviour changes). Cheap: one sysctl call, no process is started when ctypes works."""
    global _pressure_probe_failed
    if sys.platform != "darwin" or _pressure_probe_failed:
        return None
    name = "kern.memorystatus_vm_pressure_level"
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
            return int(value.value)
    except Exception:  # noqa: BLE001
        pass
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2).stdout.strip()
        return int(out)
    except Exception:  # noqa: BLE001
        _pressure_probe_failed = True   # no sysctl here: do not try again on every load
        return None


def pressure_name(level: Optional[int] = None) -> str:
    """"normal", "warn", "critical" or "unknown" for a level (default: the current one)."""
    level = pressure_level() if level is None else level
    if level is None:
        return "unknown"
    if level >= PRESSURE_CRITICAL:
        return "critical"
    return "warn" if level >= PRESSURE_WARN else "normal"


def heavy_mb() -> int:
    """Engines estimated at this much or more are "heavy" (MEMORY_GUARD_HEAVY_MB, default 1500)."""
    return _env_int("MEMORY_GUARD_HEAVY_MB", DEFAULT_HEAVY_MB)


def effective_reserve_mb(level: Optional[int] = None) -> int:
    """The reserve for a pressure level: x1.5 at warn or critical, as configured at normal or when unknown."""
    reserve = reserve_mb()
    if level is not None and level >= PRESSURE_WARN:
        return int(round(reserve * WARN_RESERVE_FACTOR))
    return reserve


def pressure_note(level: Optional[int] = None) -> str:
    """A short phrase for a refusal message ("" at normal pressure and off macOS)."""
    level = pressure_level() if level is None else level
    if level is None or level < PRESSURE_WARN:
        return ""
    return f" (the system reports {pressure_name(level)} memory pressure)"


def status() -> dict:
    """What the guard sees now: {"available_mb": float|None, "pressure": "normal|warn|critical|unknown", "pressure_level":
    int|None, "reserve_mb": int, "effective_reserve_mb": int, "enabled": bool}."""
    level = pressure_level()
    avail = available_mb()
    return {"available_mb": None if avail is None else round(avail, 1), "pressure": pressure_name(level),
            "pressure_level": level, "reserve_mb": reserve_mb(), "effective_reserve_mb": effective_reserve_mb(level),
            "enabled": enabled()}


def eviction_order(candidates: List[Tuple[str, float]], now: Optional[float] = None) -> List[str]:
    """Keys of `candidates` ((key, last_used) pairs) in the order they should be unloaded: largest (estimated size x
    seconds idle) first. A model with no estimate counts as SIZE_FLOOR_MB. With equal sizes this is least recently used
    first; a big model idle for a while goes before several small warm ones."""
    now = _wall_time() if now is None else now

    def score(c: Tuple[str, float]) -> Tuple[float, float]:
        size = max(need_mb(c[0]), SIZE_FLOOR_MB)
        idle = max(1.0, now - c[1])
        return (size * idle, -c[1])

    return [k for k, _ in sorted(candidates, key=score, reverse=True)]


def fits(engine: str, read_available: Optional[Callable[[], Optional[float]]] = None) -> Tuple[bool, Optional[float], int]:
    """Would `engine` fit now? (ok, available MB or None, required MB). Unlike ensure_headroom it unloads nothing and
    raises nothing: background jobs just wait and ask again later. Always ok with MEMORY_GUARD=off or an unreadable memory.
    On macOS the reserve is x1.5 at "warn" pressure, and a heavy engine is not ok at "critical" (nothing is unloaded here)."""
    need = need_mb(engine)
    level = pressure_level()
    required = need + effective_reserve_mb(level)
    if not enabled() or need <= 0:
        return True, None, required
    avail = (read_available or available_mb)()
    if level == PRESSURE_CRITICAL and need >= heavy_mb():
        return False, avail, required
    return (avail is None or avail >= required), avail, required


def ensure_headroom(
    engine: str,
    candidates: List[Tuple[str, float]],
    unload: Callable[[str], None],
    read_available: Optional[Callable[[], Optional[float]]] = None,
    settle_s: float = 0.3,
) -> None:
    """Make sure `engine` fits in memory, or raise AppError INSUFFICIENT_MEMORY (503).

    candidates: (key, last_used) of models that may be unloaded. The caller leaves out busy and pinned models, so a model
    in the middle of a request, or held by a live call, is never touched. They are unloaded in the order of estimated size
    x idle time, largest first (eviction_order), re-reading memory after each, and only as many as needed.

    macOS pressure: at "warn" the reserve is x1.5; at "critical" a heavy engine (see heavy_mb) is refused unless at least
    one idle model was unloaded for it."""
    if not enabled():
        return
    read_available = read_available or available_mb
    need = need_mb(engine)
    if need <= 0:
        return
    heavy = need >= heavy_mb()
    level = pressure_level()
    required = need + effective_reserve_mb(level)
    avail = read_available()
    critical_blocked = level == PRESSURE_CRITICAL and heavy   # lifted once something idle has been unloaded
    note = (lambda lv: f", pressure {pressure_name(lv)}" if lv is not None and lv >= PRESSURE_WARN else "")  # noqa: E731
    if avail is None and not critical_blocked:
        log.info("memory guard: %s allowed, memory could not be read", engine)
        return
    if avail is not None and avail >= required and not critical_blocked:
        log.info("memory guard: %s ok (%.0f MB available, %d MB needed incl. reserve%s)", engine, avail, required,
                 note(level))
        return
    freed: List[str] = []
    for key in eviction_order([c for c in candidates if c[0] != engine]):
        log.warning("memory guard: %s available, %s needs %d MB%s; unloading idle model %s (about %d MB)",
                    "?" if avail is None else f"{avail:.0f} MB", engine, required, note(level), key, need_mb(key))
        unload(key)
        freed.append(key)
        critical_blocked = False   # something idle went: the kernel's level lags a little, free memory decides now
        if settle_s:
            time.sleep(settle_s)  # a worker process gives its memory back a moment after it exits
        avail = read_available()
        level = pressure_level()
        required = need + effective_reserve_mb(level)
        if avail is None or avail >= required:
            break
    if (avail is None or avail >= required) and not critical_blocked:
        log.info("memory guard: %s ok after unloading %s (%s MB available)", engine, freed,
                 "?" if avail is None else f"{avail:.0f}")
        return
    shown = 0.0 if avail is None else avail
    log.error("memory guard: refusing to load %s: %.0f MB available, %d MB needed%s; unloaded %s",
              engine, shown, required, pressure_note(level), freed or "nothing")
    raise AppError(
        "INSUFFICIENT_MEMORY",
        f"Not enough free memory to load this voice engine ({shown / 1024:.1f} GB free, about {required / 1024:.1f} GB "
        f"needed{pressure_note(level)}). Close other apps or wait a moment for idle engines to unload, then try again.",
        503,
        {"engine": engine, "available_mb": int(shown), "required_mb": required, "unloaded": freed,
         "pressure": pressure_name(level)},
    )
