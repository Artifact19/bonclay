"""Which AI models are loaded right now, what each one takes, and how the machine's memory is doing.

    GET   /api/system/resources              the snapshot the dashboard's "Models and memory" indicator shows
    POST  /api/system/models/{key}/unload    unload an idle model (409 with a plain message when it is busy or in a live call)

Reading never loads or touches a model: it looks at what the model manager and the services already hold in memory.
Memory is "measured" for a model that runs as its own worker process (the process and its children, read with psutil)
and "estimated" for one that lives inside this API process (the size of its weights when they can be read without
touching the model, otherwise the memory guard's estimate or a small fixed figure), because the API process's
resident memory cannot be split per model. The whole response is cached for 1 s so that polling is cheap; the macOS
compressed-memory figure (vm_stat) for 2 s.
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform as _platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends

try:
    from core import memory_guard
    from core.errors import AppError
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover - imported as backend.routers.system_resources
    from backend.core import memory_guard
    from backend.core.errors import AppError
    from backend.services.auth import get_current_user_id

log = logging.getLogger("bonclay.system_resources")
router = APIRouter(prefix="/api/system", tags=["system"])

MB = 2 ** 20
SNAPSHOT_TTL_S = 1.0
VM_STAT_TTL_S = 2.0
LOADED_UNTIL_IDLE_S = 60.0      # a model not used for this long is shown as idle

# kern.memorystatus_vm_pressure_level
PRESSURE_LEVELS = {1: "normal", 2: "warn", 4: "critical"}

# What the engines the model manager holds are called, and what they are for.
ENGINE_INFO: Dict[str, Tuple[str, str]] = {
    "kokoro": ("Standard voice (Kokoro)", "voice"),
    "qwen3_0_6b": ("Voice cloning (Qwen3-TTS)", "clone"),
    "indic_speak": ("Indian-language voices (Indic-Speak)", "voice"),
    "chatterbox_ml": ("Multilingual voice cloning (Chatterbox)", "clone"),
    "dhvaani": ("Indian-language voice cloning (DhVaani)", "clone"),
    "indic_f5": ("Indian-language voice cloning (IndicF5)", "clone"),
}
# Small fixed estimates (MB) for in-app models when neither the weights nor the memory guard say anything.
FALLBACK_ESTIMATE_MB: Dict[str, int] = {
    "kokoro": 600, "qwen3_0_6b": 1500, "indic_f5": 3000, "embedder": 1100, "reranker": 600, "audioseal": 300,
    "ecapa": 150, "answer_audit": 2200, "whisper": 1000,
}
WHISPER_MB = {"tiny": 150, "base": 300, "small": 1000, "medium": 2600, "large": 4800, "turbo": 3000}
# Models whose engine is unloaded through the model manager (everything else says "cannot be unloaded from here").
CANNOT_UNLOAD = "This model cannot be unloaded from here."


def _b():
    """The backend module (imported lazily; also works when backend.py runs as __main__)."""
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "advanced_manager"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "advanced_manager"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


def _loaded_module(name: str):
    """A module only if it is already imported: reading the indicator must not import a heavy package."""
    return sys.modules.get(name) or sys.modules.get("backend." + name)


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - the indicator must never take the page down
        return default


# ------------------------------------------------------------------------------------------------------ the machine

def parse_vm_stat(text: str) -> Optional[float]:
    """Compressed memory in MB from `vm_stat` output (pages occupied by the compressor x page size), or None."""
    import re
    page = re.search(r"page size of (\d+) bytes", text or "")
    pages = re.search(r"Pages occupied by compressor:\s+(\d+)", text or "")
    if not page or not pages:
        return None
    return int(pages.group(1)) * int(page.group(1)) / MB


def pressure_from_level(level: Any) -> Optional[str]:
    try:
        return PRESSURE_LEVELS.get(int(str(level).strip()))
    except (TypeError, ValueError):
        return None


_cache_lock = threading.Lock()
_vm_cache: Tuple[float, Optional[float]] = (0.0, None)
_snapshot_cache: Tuple[float, Optional[dict]] = (0.0, None)


def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
    except Exception:  # noqa: BLE001
        return None
    return out.stdout if out.returncode == 0 else None


def compressed_mb(now: Optional[float] = None) -> Optional[float]:
    """macOS only, cached for 2 s."""
    global _vm_cache
    now = time.monotonic() if now is None else now
    if _platform.system() != "Darwin":
        return None
    at, value = _vm_cache
    if at and now - at < VM_STAT_TTL_S:
        return value
    out = _run(["vm_stat"])
    value = parse_vm_stat(out) if out else None
    _vm_cache = (now, value)
    return value


def memory_pressure() -> Optional[str]:
    if _platform.system() != "Darwin":
        return None
    out = _run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    return pressure_from_level(out) if out else None


def _device_and_gpu(total_mb: float) -> Tuple[str, Optional[dict]]:
    torch = sys.modules.get("torch")        # already imported by the app; never imported here
    if torch is None:
        return "cpu", None
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return "cuda", {"name": torch.cuda.get_device_name(0), "total_mb": round(total / MB),
                            "used_mb": round((total - free) / MB)}
    except Exception:  # noqa: BLE001
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            used = None
            try:
                used = round(torch.mps.current_allocated_memory() / MB)
            except Exception:  # noqa: BLE001
                pass
            # Apple GPUs share the machine's memory, so the GPU's budget is the machine's total.
            return "mps", {"name": "Apple GPU (shared memory)", "total_mb": round(total_mb), "used_mb": used}
    except Exception:  # noqa: BLE001
        pass
    return "cpu", None


def system_block() -> dict:
    import psutil
    vm = psutil.virtual_memory()
    total = vm.total / MB
    available = vm.available / MB
    swap = _safe(psutil.swap_memory)
    device, gpu = _device_and_gpu(total)
    system = {"darwin": "darwin", "linux": "linux", "windows": "windows"}.get(_platform.system().lower(), "linux")
    comp = _safe(compressed_mb)
    return {
        "total_mb": round(total),
        "used_mb": round(max(0.0, total - available)),
        "available_mb": round(available),
        "compressed_mb": None if comp is None else round(comp),
        "swap_used_mb": round(swap.used / MB) if swap else 0,
        "swap_total_mb": round(swap.total / MB) if swap else 0,
        "pressure": _safe(memory_pressure),
        "platform": system,
        "device": device,
        "gpu": gpu,
    }


# ------------------------------------------------------------------------------------------------------ models

_RUSAGE = None   # (proc_pid_rusage, struct class), loaded once on macOS


def _footprint_bytes(pid: int) -> Optional[int]:
    """macOS physical footprint of one process (what Activity Monitor calls Memory), or None. On Apple Silicon the models
    on the GPU live in unified memory that RSS leaves out (an MLX worker holding a 2.4 GB model shows ~0.1 GB RSS), so the
    footprint is the figure that matches reality. Same reader as bench/model_memory.py."""
    global _RUSAGE
    if sys.platform != "darwin":
        return None
    try:
        if _RUSAGE is None:
            import ctypes
            import ctypes.util
            lib = ctypes.CDLL(ctypes.util.find_library("proc"))
            names = ("ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins ri_wired_size "
                     "ri_resident_size ri_phys_footprint").split()

            class Rusage(ctypes.Structure):
                _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in names] + \
                           [(f"pad{i}", ctypes.c_uint64) for i in range(40)]
            _RUSAGE = (lib.proc_pid_rusage, Rusage, ctypes)
        call, Rusage, ctypes = _RUSAGE
        ru = Rusage()
        if call(int(pid), 4, ctypes.byref(ru)) != 0:
            return None
        return int(ru.ri_phys_footprint)
    except Exception:  # noqa: BLE001
        return None


def _proc_bytes(proc: Any) -> int:
    """One process's memory: the macOS footprint for a real process, else its RSS (Linux, Windows, test fakes)."""
    try:
        import psutil
        if isinstance(proc, psutil.Process):
            fp = _footprint_bytes(proc.pid)
            if fp is not None:
                return fp
    except Exception:  # noqa: BLE001
        pass
    return proc.memory_info().rss


def tree_rss_mb(pid: Optional[int]) -> Optional[float]:
    """Memory of a process and all its children, in MB (None when it cannot be read): the physical footprint on macOS
    (includes Apple GPU memory), the resident set elsewhere."""
    if not pid:
        return None
    try:
        import psutil
        proc = psutil.Process(pid)
        total = _proc_bytes(proc)
        for child in proc.children(recursive=True):
            try:
                total += _proc_bytes(child)
            except Exception:  # noqa: BLE001 - a child that has just exited
                pass
        return total / MB
    except Exception:  # noqa: BLE001
        return None


def _pid_of(handle: Any) -> Optional[int]:
    proc = getattr(handle, "proc", None)
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int):
        return None
    try:
        if proc.poll() is not None:     # the worker has exited
            return None
    except Exception:  # noqa: BLE001
        pass
    return pid


def _weights_mb(obj: Any) -> Optional[float]:
    """Size of a torch model's weights, read without running it (None when it is not a torch module)."""
    for candidate in (obj, getattr(obj, "model", None)):
        params = getattr(candidate, "parameters", None)
        if not callable(params):
            continue
        try:
            total = sum(p.numel() * p.element_size() for p in params())
        except Exception:  # noqa: BLE001
            continue
        if total > 0:
            return total / MB
    return None


def _guard_estimate(key: str) -> Optional[int]:
    need = _safe(lambda: memory_guard.need_mb(key), 0)
    return need if need and need > 0 else None


def _device_cached() -> str:
    torch = sys.modules.get("torch")
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _device_name(raw: Any) -> str:
    s = str(raw or "").lower()
    if s.startswith("cuda"):
        return "cuda"
    if s in ("mps", "metal", "mlx"):
        return s if s == "mlx" else "mps"
    return "cpu"


def _manager_device(key: str, model: Any, is_worker: bool) -> str:
    if is_worker:
        name = type(model).__name__
        if name in ("MlxKokoro", "MlxQwen3Clone"):
            return "mlx"
        if key == "indic_speak":
            if os.getenv("INDIC_SPEAK_MLX_MODEL", "").strip():     # the 4-bit model runs on the Apple GPU
                return "mlx"
            return _device_name(getattr(model, "device", None) or _device_cached())
        if key == "chatterbox_ml":
            return "cpu"
        return _device_name(getattr(model, "device", None) or _device_cached())
    if key == "qwen3_0_6b":
        b = _loaded_module("backend")
        return _device_name(_safe(lambda: b.clone_engine_device(), "cpu"))
    if key == "indic_f5":
        b = _loaded_module("backend")
        return _device_name(_safe(lambda: b.DEVICE, "cpu"))
    return _device_name(getattr(model, "device", None)) if getattr(model, "device", None) else "cpu"


def _state(*, pinned: bool = False, busy: bool = False, loading: bool = False, idle_s: Optional[float] = None) -> str:
    if pinned:
        return "pinned (live call)"
    if busy:
        return "busy"
    if loading:
        return "loading"
    if idle_s is not None and idle_s >= LOADED_UNTIL_IDLE_S:
        return "idle"
    return "loaded"


def _entry(key: str, label: str, kind: str, *, where: str, device: str, state: str, memory_mb: Optional[float],
           source: str, idle_s: Optional[float] = None, pid: Optional[int] = None, can_unload: bool = False) -> dict:
    return {"key": key, "label": label, "kind": kind, "where": where, "device": device, "state": state,
            "memory_mb": round(memory_mb or 0), "memory_source": source,
            "idle_s": None if idle_s is None else round(max(0.0, idle_s)), "pid": pid, "can_unload": can_unload}


def _worker_entry(key: str, label: str, kind: str, handle: Any, device: str, *, state: str,
                  idle_s: Optional[float] = None, can_unload: bool = False, pid: Optional[int] = None) -> dict:
    pid = pid or _pid_of(handle)
    rss = _safe(lambda: tree_rss_mb(pid))
    return _entry(key, label, kind, where="worker process", device=device, state=state,
                  memory_mb=rss if rss is not None else _guard_estimate(key) or FALLBACK_ESTIMATE_MB.get(key, 500),
                  source="measured" if rss is not None else "estimated", idle_s=idle_s, pid=pid,
                  can_unload=can_unload)


def _manager_models(mgr: Any, now: float) -> List[dict]:
    out: List[dict] = []
    for key, value in list(mgr.loaded_models.items()):
        try:
            model, _loaded_at, last_used = value
        except (TypeError, ValueError):
            continue
        label, kind = ENGINE_INFO.get(key, (key, "voice"))
        pinned = bool(_safe(lambda: mgr.is_pinned(key), False))
        busy = bool(getattr(model, "busy", False))
        idle_s = now - last_used
        state = _state(pinned=pinned, busy=busy, idle_s=idle_s)
        can_unload = not pinned and not busy
        if _pid_of(model) is not None or getattr(model, "proc", None) is not None:
            out.append(_worker_entry(key, label, kind, model, _manager_device(key, model, True), state=state,
                                     idle_s=idle_s, can_unload=can_unload))
        else:
            est = _weights_mb(model) or _guard_estimate(key) or FALLBACK_ESTIMATE_MB.get(key, 500)
            out.append(_entry(key, label, kind, where="in app", device=_manager_device(key, model, False),
                              state=state, memory_mb=est, source="estimated", idle_s=idle_s,
                              can_unload=can_unload))
    return out


def _other_models(now: float) -> List[dict]:
    """The models the model manager does not hold: workers and in-app singletons the services keep. Each part is
    read defensively and only if its module is already imported."""
    out: List[dict] = []

    mlx_asr = _loaded_module("services.mlx_asr")
    worker = getattr(mlx_asr, "_worker", None)
    if worker is not None and _safe(lambda: worker.running, False):
        out.append(_worker_entry("asr_mlx", "Speech recognition (Whisper on Apple GPU)", "speech_recognition",
                                 worker, "mlx", state="loaded"))

    b = _loaded_module("backend")
    whisper = getattr(b, "whisper_model", None)
    if whisper is not None and _safe(lambda: whisper.loaded, False):
        size = str(getattr(whisper, "size", "small"))
        est = _weights_mb(getattr(whisper, "_model", None)) or WHISPER_MB.get(size) or FALLBACK_ESTIMATE_MB["whisper"]
        out.append(_entry("whisper", f"Speech recognition (Whisper {size})", "speech_recognition",
                          where="in app", device="cpu", state="loaded", memory_mb=est, source="estimated"))

    llm = _loaded_module("services.llm")
    if llm is not None:
        live = getattr(llm, "_mlx", None)
        if live is not None and _safe(lambda: live.running, False):
            out.append(_worker_entry("llm_local", "Language model, offline replies", "language_model", live, "mlx",
                                     state=_state(busy=bool(_safe(lambda: live.request_lock.locked(), False)))))
        bg = getattr(llm, "_bg", None)
        if bg is not None and _safe(lambda: llm.background_running(), False):
            out.append(_worker_entry("background_llm", "Language model, background jobs", "language_model", bg, "mlx",
                                     state=_state(busy=bool(_safe(lambda: bg.request_lock.locked(), False))),
                                     can_unload=not _safe(lambda: bg.request_lock.locked(), False)))

    retrieval = _loaded_module("services.retrieval")
    if retrieval is not None:
        embedders = dict(getattr(retrieval, "_embedders", {}) or {})
        half = set(getattr(retrieval, "_half_models", set()) or set())
        for name, model in embedders.items():
            est = _weights_mb(model) or (FALLBACK_ESTIMATE_MB["embedder"] if name in half else
                                         FALLBACK_ESTIMATE_MB["embedder"] * 2)
            short = str(name).split("/")[-1]
            out.append(_entry("embedder:" + short, f"Search model ({short})", "search",
                              where="in app", device=_device_name(_safe(retrieval.embedding_device, "cpu")),
                              state="loaded", memory_mb=est, source="estimated"))
        reranker = getattr(retrieval, "_reranker", None)
        if reranker is not None or getattr(retrieval, "_reranker_loading", False):
            loading = reranker is None
            est = (_weights_mb(reranker) if reranker is not None else None) or FALLBACK_ESTIMATE_MB["reranker"]
            out.append(_entry("reranker", "Answer re-ranker", "reranker", where="in app",
                              device=_device_name(_safe(retrieval.reranker_device, "cpu")),
                              state="loading" if loading else "loaded", memory_mb=est, source="estimated"))

    prov = _loaded_module("services.provenance")
    models = getattr(prov, "_models", None)
    if models:
        est = sum(_weights_mb(m) or 0 for m in models) or FALLBACK_ESTIMATE_MB["audioseal"]
        out.append(_entry("audioseal", "Audio watermark (AudioSeal)", "watermark", where="in app", device="cpu",
                          state="loaded", memory_mb=est, source="estimated"))

    speaker = _loaded_module("services.speaker")
    clf = getattr(speaker, "_classifier", None)
    if clf is not None:
        est = _weights_mb(getattr(clf, "mods", None)) or _weights_mb(clf) or FALLBACK_ESTIMATE_MB["ecapa"]
        out.append(_entry("ecapa", "Speaker check (ECAPA)", "speaker_check", where="in app", device="cpu",
                          state="loaded", memory_mb=est, source="estimated"))

    audit = _loaded_module("services.answer_audit")
    verifier = getattr(audit, "_verifier", None)
    if verifier is not None and getattr(verifier, "loaded", False):
        spec = _safe(lambda: audit.model_spec(), {}) or {}
        label = spec.get("label") or getattr(audit, "MODEL_LABEL", "answer checker")
        est = _weights_mb(getattr(verifier, "_model", None)) or spec.get("need_mb") \
            or FALLBACK_ESTIMATE_MB["answer_audit"]
        last = getattr(verifier, "last_used", 0.0)
        idle = None
        if last:
            clock = getattr(verifier, "_clock", time.monotonic)
            idle = _safe(lambda: clock() - last)
        out.append(_entry("answer_audit", f"Answer check ({label})", "audit", where="in app", device="cpu",
                          state=_state(idle_s=idle), memory_mb=est, source="estimated", idle_s=idle))
    return out


def models_block(now: Optional[float] = None) -> List[dict]:
    now = time.time() if now is None else now
    mgr = _b().advanced_manager
    models = _safe(lambda: _manager_models(mgr, now), []) + _safe(lambda: _other_models(now), [])
    return models


def app_block(models: List[dict]) -> dict:
    def own_rss() -> float:
        import psutil
        return _proc_bytes(psutil.Process(os.getpid())) / MB
    api = _safe(own_rss, 0.0)
    seen = set()
    workers = 0.0
    for m in models:
        pid = m.get("pid")
        if m["where"] == "worker process" and m["memory_source"] == "measured" and pid not in seen:
            seen.add(pid)
            workers += m["memory_mb"]
    return {"api_rss_mb": round(api), "workers_rss_mb": round(workers), "total_rss_mb": round(api + workers)}


def guard_block() -> dict:
    keys = list(memory_guard.DEFAULT_NEED_MB)
    return {"enabled": bool(_safe(memory_guard.enabled, False)), "reserve_mb": _safe(memory_guard.reserve_mb, 0),
            "estimates_mb": {k: _safe(lambda k=k: memory_guard.need_mb(k), 0) for k in keys}}


def build_snapshot() -> dict:
    models = models_block()
    return {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "system": system_block(),
            "app": app_block(models), "models": models, "guard": guard_block()}


def snapshot(now: Optional[float] = None) -> dict:
    """The response, cached for 1 s."""
    global _snapshot_cache
    now = time.monotonic() if now is None else now
    with _cache_lock:
        at, value = _snapshot_cache
        if value is not None and now - at < SNAPSHOT_TTL_S:
            return value
        value = build_snapshot()
        _snapshot_cache = (now, value)
        return value


def clear_caches() -> None:
    global _vm_cache, _snapshot_cache
    with _cache_lock:
        _vm_cache = (0.0, None)
        _snapshot_cache = (0.0, None)


@router.get("/resources")
async def system_resources(user_id: str = Depends(get_current_user_id)):
    return await asyncio.to_thread(snapshot)


# ------------------------------------------------------------------------------------------------------ unload

async def _unload_manager_model(mgr: Any, key: str) -> None:
    # The manager's public unload (refuses busy and pinned engines); older managers only have the private path.
    unload = getattr(mgr, "unload", None) or getattr(mgr, "_unload_model", None)
    if not callable(unload):
        raise AppError("CANNOT_UNLOAD", CANNOT_UNLOAD, 409)
    async with mgr.lock:
        entry = mgr.loaded_models.get(key)
        if entry is None:
            raise AppError("NOT_LOADED", "This model is not loaded any more.", 409)
        if mgr.is_pinned(key):
            raise AppError("MODEL_IN_USE", "This model is in use by a live call, so it cannot be unloaded now.", 409)
        if bool(getattr(entry[0], "busy", False)):
            raise AppError("MODEL_BUSY", "This model is working on a request right now. Try again in a moment.", 409)
        done = await asyncio.to_thread(unload, key)
        if done is False:   # the public unload refused: it became busy or was pinned in the meantime
            raise AppError("MODEL_BUSY", "This model is working on a request right now. Try again in a moment.", 409)


@router.post("/models/{key}/unload")
async def unload_model(key: str, user_id: str = Depends(get_current_user_id)):
    mgr = _b().advanced_manager
    if key in mgr.loaded_models:
        await _unload_manager_model(mgr, key)
    elif key == "background_llm":
        llm = _loaded_module("services.llm")
        bg = getattr(llm, "_bg", None)
        if bg is None or not _safe(lambda: llm.background_running(), False):
            raise AppError("NOT_LOADED", "This model is not loaded any more.", 409)
        if _safe(lambda: bg.request_lock.locked(), False):
            raise AppError("MODEL_BUSY", "This model is working on a request right now. Try again in a moment.", 409)
        await asyncio.to_thread(llm.unload_background)
    elif key in ENGINE_INFO:        # a model the manager can hold, but it was unloaded in the meantime
        raise AppError("NOT_LOADED", "This model is not loaded any more.", 409)
    else:
        raise AppError("CANNOT_UNLOAD", CANNOT_UNLOAD, 409)
    clear_caches()
    return {"key": key, "unloaded": True}
