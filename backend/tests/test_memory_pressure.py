"""macOS memory pressure in the memory guard, the order idle engines are unloaded in, the idle timeouts per engine, and
what the speech-recognition warm-up loads. Nothing here reads the real machine: pressure, free memory and the clock are
replaced."""
import asyncio
import os
import sys
import types

import pytest

SERVICES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "services")
if SERVICES not in sys.path:       # services/mlx_asr_worker.py is a script folder module, as in test_memory_bounds.py
    sys.path.insert(0, SERVICES)

from core import memory_guard as mg
from core.errors import AppError


def reader(values):
    it = iter(values)
    last = [values[-1]]

    def read():
        last[0] = next(it, last[0])
        return last[0]
    return read


@pytest.fixture(autouse=True)
def _guard_env(monkeypatch):
    for k in ("MEMORY_GUARD", "MEMORY_GUARD_RESERVE_MB", "MEMORY_GUARD_HEAVY_MB", "MEMORY_NEED_INDIC_SPEAK_MB",
              "MEMORY_NEED_KOKORO_MB", "MEMORY_NEED_DHVAANI_MB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MEMORY_GUARD_RESERVE_MB", "1000")
    monkeypatch.setitem(mg.DEFAULT_NEED_MB, "indic_speak", 6000)
    monkeypatch.setitem(mg.DEFAULT_NEED_MB, "dhvaani", 2000)
    monkeypatch.setitem(mg.DEFAULT_NEED_MB, "kokoro", 600)


def pressure(monkeypatch, *levels):
    """pressure_level() returns the levels in turn, then the last one for ever."""
    seq = list(levels)
    monkeypatch.setattr(mg, "pressure_level", lambda: seq.pop(0) if len(seq) > 1 else seq[0])


# ------------------------------------------------------------------------------------------------ pressure level

def test_pressure_is_unknown_off_macos(monkeypatch):
    monkeypatch.undo()          # the conftest stub would hide the real function
    monkeypatch.setattr(sys, "platform", "linux")
    assert mg.pressure_level() is None
    assert mg.pressure_name(None) == "unknown"


def test_pressure_falls_back_to_the_sysctl_command(monkeypatch):
    import ctypes
    monkeypatch.undo()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(mg, "_pressure_probe_failed", False)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: (_ for _ in ()).throw(OSError("no libc")))
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(stdout="2\n")
    monkeypatch.setattr("subprocess.run", fake_run)
    assert mg.pressure_level() == 2
    assert calls == [["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]]


def test_a_missing_sysctl_is_remembered_and_means_unknown(monkeypatch):
    import ctypes
    monkeypatch.undo()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(mg, "_pressure_probe_failed", False)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: (_ for _ in ()).throw(OSError("no libc")))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("sysctl")))
    assert mg.pressure_level() is None
    assert mg._pressure_probe_failed is True
    monkeypatch.setattr(mg, "_pressure_probe_failed", False)   # (state restored for the other tests)


def test_names_and_reserve_factor():
    assert [mg.pressure_name(n) for n in (1, 2, 4, None)] == ["normal", "warn", "critical", "unknown"]
    assert mg.effective_reserve_mb(None) == 1000 and mg.effective_reserve_mb(1) == 1000
    assert mg.effective_reserve_mb(2) == 1500 and mg.effective_reserve_mb(4) == 1500
    assert mg.pressure_note(1) == "" and "warn" in mg.pressure_note(2)


def test_status_reports_what_the_guard_sees(monkeypatch):
    pressure(monkeypatch, 2)
    monkeypatch.setattr(mg, "available_mb", lambda: 4321.0)
    assert mg.status() == {"available_mb": 4321.0, "pressure": "warn", "pressure_level": 2, "reserve_mb": 1000,
                           "effective_reserve_mb": 1500, "enabled": True}


# ------------------------------------------------------------------------------------------------ warn and critical

def test_normal_pressure_changes_nothing(monkeypatch):
    pressure(monkeypatch, 1)
    mg.ensure_headroom("dhvaani", [], lambda k: None, reader([3000]), settle_s=0)      # needs 2000 + 1000
    with pytest.raises(AppError):
        mg.ensure_headroom("dhvaani", [], lambda k: None, reader([2999]), settle_s=0)


def test_warn_asks_for_one_and_a_half_times_the_reserve(monkeypatch):
    pressure(monkeypatch, 2)
    mg.ensure_headroom("dhvaani", [], lambda k: None, reader([3500]), settle_s=0)      # 2000 + 1500
    with pytest.raises(AppError) as e:
        mg.ensure_headroom("dhvaani", [], lambda k: None, reader([3499]), settle_s=0)
    assert e.value.details["required_mb"] == 3500 and e.value.details["pressure"] == "warn"
    assert "warn memory pressure" in e.value.message
    assert mg.fits("dhvaani", lambda: 3499.0) == (False, 3499.0, 3500)
    assert mg.fits("dhvaani", lambda: 3500.0)[0] is True


def test_warn_makes_the_guard_unload_what_normal_pressure_would_have_left_alone(monkeypatch):
    pressure(monkeypatch, 2)
    unloaded = []
    mg.ensure_headroom("dhvaani", [("idle", 1.0)], unloaded.append, reader([3200, 9000]), settle_s=0)
    assert unloaded == ["idle"]        # 3200 MB is enough at normal pressure (3000) but not at warn (3500)


def test_critical_refuses_a_heavy_engine_when_nothing_idle_can_go(monkeypatch):
    pressure(monkeypatch, 4)
    with pytest.raises(AppError) as e:
        mg.ensure_headroom("dhvaani", [], lambda k: None, reader([50000]), settle_s=0)   # plenty of memory
    assert e.value.code == "INSUFFICIENT_MEMORY" and e.value.status == 503
    assert e.value.details["pressure"] == "critical" and e.value.details["unloaded"] == []
    assert "critical memory pressure" in e.value.message
    assert mg.fits("dhvaani", lambda: 50000.0)[0] is False     # fits() unloads nothing, so it says no


def test_critical_allows_a_heavy_engine_once_an_idle_one_was_unloaded(monkeypatch):
    pressure(monkeypatch, 4)        # the kernel's level lags: still critical after the unload
    unloaded = []
    mg.ensure_headroom("dhvaani", [("idle", 1.0)], unloaded.append, reader([50000, 50000]), settle_s=0)
    assert unloaded == ["idle"]


def test_critical_still_refuses_when_the_unload_did_not_free_enough(monkeypatch):
    pressure(monkeypatch, 4)
    with pytest.raises(AppError) as e:
        mg.ensure_headroom("dhvaani", [("idle", 1.0)], lambda k: None, reader([50000, 1000]), settle_s=0)
    assert e.value.details["unloaded"] == ["idle"] and e.value.details["available_mb"] == 1000


def test_critical_does_not_block_a_light_engine(monkeypatch):
    pressure(monkeypatch, 4)
    mg.ensure_headroom("kokoro", [], lambda k: None, reader([5000]), settle_s=0)       # 600 MB: below the heavy line
    assert mg.fits("kokoro", lambda: 5000.0)[0] is True


def test_the_heavy_line_is_a_setting(monkeypatch):
    pressure(monkeypatch, 4)
    monkeypatch.setenv("MEMORY_GUARD_HEAVY_MB", "300")
    with pytest.raises(AppError):
        mg.ensure_headroom("kokoro", [], lambda k: None, reader([5000]), settle_s=0)


def test_unreadable_pressure_leaves_linux_and_windows_as_they_were(monkeypatch):
    pressure(monkeypatch, None)
    mg.ensure_headroom("dhvaani", [], lambda k: None, reader([3000]), settle_s=0)
    assert mg.fits("dhvaani", lambda: 3000.0) == (True, 3000.0, 3000)
    assert mg.fits("dhvaani", lambda: 2999.0)[0] is False


def test_the_guard_off_ignores_pressure_too(monkeypatch):
    pressure(monkeypatch, 4)
    monkeypatch.setenv("MEMORY_GUARD", "off")
    mg.ensure_headroom("dhvaani", [], lambda k: None, reader([1]), settle_s=0)
    assert mg.fits("dhvaani", lambda: 1.0)[0] is True


# ------------------------------------------------------------------------------------------------ eviction order

def test_a_big_idle_model_goes_before_several_small_warm_ones():
    now = 10_000.0
    cands = [("kokoro", now - 400), ("dhvaani", now - 300), ("indic_speak", now - 200)]
    # sizes 600 / 2000 / 6000 MB: scores 240k / 600k / 1200k. The least recently used (kokoro) is not first.
    assert mg.eviction_order(cands, now) == ["indic_speak", "dhvaani", "kokoro"]


def test_idle_time_beats_size_when_it_is_long_enough():
    now = 10_000.0
    # dhvaani (2000 MB) idle 1000 s = 2.0M beats indic_speak (6000 MB) idle 100 s = 0.6M
    assert mg.eviction_order([("indic_speak", now - 100), ("dhvaani", now - 1000)], now) == ["dhvaani", "indic_speak"]


def test_equal_sizes_are_least_recently_used_first_and_unknown_engines_have_a_floor():
    now = 10_000.0
    assert mg.eviction_order([("new", now - 10), ("old", now - 900), ("mid", now - 300)], now) == ["old", "mid", "new"]
    assert mg.SIZE_FLOOR_MB == 200
    # an unknown engine counts as SIZE_FLOOR_MB (200) and loses to the 600 MB kokoro idle for as long
    assert mg.eviction_order([("mystery", now - 500), ("kokoro", now - 500)], now) == ["kokoro", "mystery"]


def test_ensure_headroom_unloads_in_that_order_and_stops_when_enough(monkeypatch):
    now = 10_000.0
    monkeypatch.setattr(mg, "_wall_time", lambda: now)
    unloaded = []
    cands = [("kokoro", now - 400), ("dhvaani", now - 300), ("indic_speak", now - 200)]
    monkeypatch.setitem(mg.DEFAULT_NEED_MB, "chatterbox_ml", 3000)
    # needs 3000 + 1000: 1000 free; after indic_speak 7000 free -> done, the two small ones stay
    mg.ensure_headroom("chatterbox_ml", cands, unloaded.append, reader([1000, 7000]), settle_s=0)
    assert unloaded == ["indic_speak"]


def test_the_manager_hands_the_guard_its_idle_models_and_the_guard_picks_the_biggest(model_manager_cls, monkeypatch):
    import time
    from core import memory_guard

    class Idle:
        busy = False

    m = model_manager_cls(max_gpu_memory_gb=12.0, idle_timeout_sec=300.0, min_residence_sec=60.0)
    m._instantiate_model_sync = lambda key: object()
    m._get_current_gpu_usage = lambda: 0
    now = time.time()
    m.loaded_models = {"kokoro": (Idle(), now - 900, now - 800), "dhvaani": (Idle(), now - 900, now - 600),
                       "indic_speak": (Idle(), now - 900, now - 500)}
    free = {"mb": 1000.0}
    order = []
    real_unload = m._unload_model

    def unload(key):
        order.append(key)
        real_unload(key)
        free["mb"] += {"indic_speak": 6000, "dhvaani": 2000, "kokoro": 600}[key]
    m._unload_model = unload
    monkeypatch.setenv("MEMORY_GUARD", "on")
    monkeypatch.setattr(memory_guard, "available_mb", lambda: free["mb"])
    monkeypatch.setattr(memory_guard.time, "sleep", lambda s: None)
    monkeypatch.setitem(memory_guard.DEFAULT_NEED_MB, "chatterbox_ml", 3000)
    m._ensure_memory("chatterbox_ml")
    assert order == ["indic_speak"] and set(m.loaded_models) == {"kokoro", "dhvaani"}


# ------------------------------------------------------------------------------------------------ idle timeouts

def _manager(model_manager_cls):
    """The real class from backend.py (model_manager_cls builds a separate copy that cannot see backend's functions)."""
    import backend

    class M:
        busy = False
    m = backend.AdvancedModelManager(idle_timeout_sec=300.0, min_residence_sec=0.0)
    m.loaded_models = {k: (M(), 0.0, 0.0) for k in ("kokoro", "indic_speak", "dhvaani", "chatterbox_ml", "qwen3_0_6b")}
    return m


def test_every_engine_unloads_when_idle_by_default(model_manager_cls):
    m = _manager(model_manager_cls)
    assert m.idle_timeout_for("indic_speak") == 300.0 and m.idle_timeout_for("kokoro") == 300.0
    assert sorted(m._idle_candidates(301.0)) == sorted(m.loaded_models)
    assert m._idle_candidates(299.0) == []


def test_live_call_engines_stay_loaded_while_the_phone_channel_is_set_up(model_manager_cls, monkeypatch):
    import backend
    m = _manager(model_manager_cls)
    monkeypatch.setenv("KEEP_LIVE_ENGINES_WARM", "auto")
    monkeypatch.setattr(backend, "_exotel_configured", lambda: True)
    assert m.keeps_live_engines_warm() and m.idle_timeout_for("kokoro") is None
    assert "kokoro" not in m._idle_candidates(10 ** 6)
    assert "indic_speak" in m._idle_candidates(10 ** 6) and "qwen3_0_6b" in m._idle_candidates(10 ** 6)
    monkeypatch.setattr(backend, "_exotel_configured", lambda: False)
    assert not m.keeps_live_engines_warm() and m.idle_timeout_for("kokoro") == 300.0


def test_keep_live_engines_warm_can_be_forced_either_way(model_manager_cls, monkeypatch):
    import backend
    m = _manager(model_manager_cls)
    monkeypatch.setattr(backend, "_exotel_configured", lambda: True)
    monkeypatch.setenv("KEEP_LIVE_ENGINES_WARM", "off")
    assert m.idle_timeout_for("kokoro") == 300.0
    monkeypatch.setattr(backend, "_exotel_configured", lambda: False)
    monkeypatch.setenv("KEEP_LIVE_ENGINES_WARM", "on")
    assert m.idle_timeout_for("kokoro") is None


def test_idle_minutes_can_be_set_for_all_engines_or_for_one(model_manager_cls, monkeypatch):
    m = _manager(model_manager_cls)
    monkeypatch.setenv("ENGINE_IDLE_UNLOAD_MIN", "10")
    assert m.idle_timeout_for("dhvaani") == 600.0
    monkeypatch.setenv("ENGINE_IDLE_UNLOAD_MIN_INDIC_SPEAK", "1")
    assert m.idle_timeout_for("indic_speak") == 60.0 and m.idle_timeout_for("dhvaani") == 600.0
    monkeypatch.setenv("ENGINE_IDLE_UNLOAD_MIN", "not a number")
    assert m.idle_timeout_for("dhvaani") == 300.0     # an unreadable value falls back to the manager's own


def test_a_pinned_engine_is_not_idle_even_without_a_timeout(model_manager_cls, monkeypatch):
    m = _manager(model_manager_cls)
    m.pin("indic_speak")
    assert "indic_speak" not in m._idle_candidates(10 ** 6)


def test_asr_worker_idle_limit_follows_the_groq_key(monkeypatch):
    import backend
    monkeypatch.setattr(backend, "groq_client", object())
    assert backend.AdvancedModelManager.asr_idle_unload_s() == 600.0
    monkeypatch.setattr(backend, "groq_client", None)
    assert backend.AdvancedModelManager.asr_idle_unload_s() == 0.0      # local ASR is the live path: never stopped
    monkeypatch.setenv("MLX_ASR_IDLE_UNLOAD_S", "90")
    assert backend.AdvancedModelManager.asr_idle_unload_s() == 90.0


def test_the_asr_worker_is_stopped_only_when_idle_long_enough(monkeypatch):
    from services import mlx_asr

    class Proc:
        def poll(self):
            return None
    stopped = []
    w = mlx_asr._worker
    monkeypatch.setattr(w, "proc", Proc())
    monkeypatch.setattr(w, "stop", lambda: stopped.append(1))
    monkeypatch.setattr(w, "last_used", 1000.0)
    assert mlx_asr.stop_if_idle(0, now=5000.0) is False            # 0: never
    assert mlx_asr.stop_if_idle(600, now=1500.0) is False          # 500 s idle
    assert mlx_asr.stop_if_idle(600, now=1700.0) is True and stopped == [1]


# ------------------------------------------------------------------------------------------------ ASR models and warm-up

def test_the_models_loaded_follow_the_strategy(monkeypatch):
    from services import mlx_asr, mlx_asr_worker
    monkeypatch.delenv("MLX_ASR_RESIDENT", raising=False)
    expected = {"single": ("accurate",), "accurate": ("accurate",), "fast": ("fast", "accurate"),
                "auto": ("fast", "accurate"), "hybrid": ("fast", "accurate")}
    for strategy, roles in expected.items():
        assert mlx_asr.resident_roles(strategy) == roles == mlx_asr_worker.roles_for_strategy(strategy)
    assert mlx_asr.DEFAULT_STRATEGY == "single" and mlx_asr.resident_roles() == ("accurate",)
    assert mlx_asr.resident_repos("fast") == (mlx_asr.FAST, mlx_asr.ACCURATE)   # a Hindi hint reaches the large one
    monkeypatch.setenv("MLX_ASR_RESIDENT", "both")      # the old behaviour on request
    assert mlx_asr.resident_roles("single") == ("fast", "accurate")


def test_the_roles_cover_what_transcribe_plan_really_runs():
    """A strategy's roles must contain every model transcribe_plan can reach (else a request would load one lazily)."""
    from services.mlx_asr_worker import INDIAN, roles_for_strategy, transcribe_plan
    for strategy in ("single", "accurate", "fast", "auto", "hybrid"):
        used = set()

        def run(path, repo, language, used=used):
            used.add(repo)
            return {"text": "", "language": "hi" if language in (None, "hi") else language}

        for language in (None, "en", "hi"):
            transcribe_plan(strategy, language, "x.wav", run, lambda p, r, used=used: (used.add(r), "hi")[1], "FAST",
                            "ACCURATE", window=lambda p, r, l, used=used: (used.add(r), {"text": ""})[1])
        roles = {"FAST": "fast", "ACCURATE": "accurate"}
        assert {roles[r] for r in used} <= set(roles_for_strategy(strategy)), strategy


def test_the_asr_worker_is_started_with_the_models_it_needs(monkeypatch):
    from services import mlx_asr
    seen = {}

    class Dead:
        def poll(self):
            return 1

    def popen(cmd, env=None):
        seen["cmd"] = cmd
        raise RuntimeError("stop here")
    monkeypatch.setattr(mlx_asr.subprocess, "Popen", popen)
    monkeypatch.setattr(mlx_asr, "_mlx_python", lambda: "/x/python")
    monkeypatch.setattr(mlx_asr._worker, "proc", None)
    with pytest.raises(RuntimeError):
        mlx_asr._worker.ensure()
    assert seen["cmd"][seen["cmd"].index("--load") + 1] == "accurate"
    monkeypatch.setenv("MLX_ASR_STRATEGY", "auto")
    with pytest.raises(RuntimeError):
        mlx_asr._worker.ensure()
    assert seen["cmd"][seen["cmd"].index("--load") + 1] == "fast,accurate"


def test_a_worker_for_one_model_never_holds_two(monkeypatch):
    import importlib
    loads = []

    class ModelHolder:
        @classmethod
        def get_model(cls, model_path, dtype):
            raise AssertionError("the original loader must not be used")

    load_models = types.ModuleType("mlx_whisper.load_models")
    load_models.load_model = lambda path, dtype=None: loads.append(path) or object()
    transcribe = types.ModuleType("mlx_whisper.transcribe")
    transcribe.ModelHolder = ModelHolder
    package = types.ModuleType("mlx_whisper")
    package.__path__ = []
    for name, mod in (("mlx_whisper", package), ("mlx_whisper.load_models", load_models),
                      ("mlx_whisper.transcribe", transcribe)):
        monkeypatch.setitem(sys.modules, name, mod)
    worker = importlib.import_module("mlx_asr_worker")
    worker.keep_models_resident(max_resident=1)
    a = ModelHolder.get_model("large", "f16")
    assert ModelHolder.get_model("large", "f16") is a
    ModelHolder.get_model("small", "f16")               # needed once: replaces large, never both
    ModelHolder.get_model("large", "f16")               # reloaded, since it was dropped
    assert loads == ["large", "small", "large"]


class _Groq:
    def __init__(self):
        self.listed = 0
        self.models = types.SimpleNamespace(list=self._list)

    def _list(self):
        self.listed += 1


def _warm_asr_env(monkeypatch, groq, local_ready=True):
    import backend
    from routers import warmup
    from services import mlx_asr
    calls = []
    monkeypatch.setattr(backend, "groq_client", groq)
    monkeypatch.setattr(mlx_asr, "unavailable_reason", lambda: None if local_ready else "no MLX environment")
    monkeypatch.setattr(mlx_asr, "warm", lambda: calls.append("warm"))
    monkeypatch.setattr(mlx_asr, "transcribe", lambda *a, **k: calls.append("transcribe") or {"text": ""})
    return warmup, calls


def test_with_a_groq_key_the_warm_up_does_not_load_local_whisper(monkeypatch):
    groq = _Groq()
    warmup, calls = _warm_asr_env(monkeypatch, groq)
    detail = asyncio.run(warmup._warm_asr())
    assert calls == [] and groq.listed == 1 and "Groq" in detail and "not loaded" in detail


def test_without_a_groq_key_it_loads_the_one_model_the_strategy_needs(monkeypatch):
    warmup, calls = _warm_asr_env(monkeypatch, None)
    detail = asyncio.run(warmup._warm_asr())
    assert calls == ["warm", "transcribe"]
    assert "large-v3-turbo" in detail and "small" not in detail


def test_warm_local_asr_on_loads_it_even_with_a_groq_key(monkeypatch):
    groq = _Groq()
    warmup, calls = _warm_asr_env(monkeypatch, groq)
    monkeypatch.setenv("WARM_LOCAL_ASR", "on")
    asyncio.run(warmup._warm_asr())
    assert calls == ["warm", "transcribe"] and groq.listed == 0


def test_warm_local_asr_off_skips_it_without_a_groq_key(monkeypatch):
    warmup, calls = _warm_asr_env(monkeypatch, None)
    monkeypatch.setenv("WARM_LOCAL_ASR", "off")
    with pytest.raises(warmup.Skipped):
        asyncio.run(warmup._warm_asr())
    assert calls == []


def test_the_live_warm_up_connects_to_groq_and_loads_local_only_on_request(monkeypatch):
    groq = _Groq()
    warmup, calls = _warm_asr_env(monkeypatch, groq)
    assert "Groq" in asyncio.run(warmup._warm_asr_connection()) and calls == [] and groq.listed == 1
    monkeypatch.setenv("WARM_LOCAL_ASR", "on")
    asyncio.run(warmup._warm_asr_connection())
    assert calls == ["warm", "transcribe"]


def test_both_sizes_are_reported_when_the_strategy_uses_both(monkeypatch):
    warmup, calls = _warm_asr_env(monkeypatch, None)
    monkeypatch.setenv("MLX_ASR_STRATEGY", "auto")
    assert "small + large-v3-turbo" in asyncio.run(warmup._warm_asr())


def test_the_local_language_model_is_warmed_only_when_it_will_answer(monkeypatch):
    from routers import warmup
    from services import llm
    started = []
    monkeypatch.setattr(llm._mlx, "unavailable_reason", lambda: None)
    monkeypatch.setattr(llm, "warm_local", lambda: started.append(1))
    monkeypatch.setattr(llm, "chat", lambda *a, **k: llm.ChatResult("OK", "groq", "m", 1, {}))
    monkeypatch.setattr(llm, "status", lambda: {"active": "groq"})
    asyncio.run(warmup._warm_llm())
    assert started == []                                   # Groq answers: the local server waits for a failure
    monkeypatch.setenv("WARM_LOCAL_LLM", "on")
    asyncio.run(warmup._warm_llm())
    assert started == [1]
    monkeypatch.delenv("WARM_LOCAL_LLM")
    monkeypatch.setattr(llm, "status", lambda: {"active": "mlx"})
    asyncio.run(warmup._warm_llm())
    assert started == [1, 1]                               # nothing else can answer: warm it
    monkeypatch.setenv("WARM_LOCAL_LLM", "off")
    asyncio.run(warmup._warm_llm())
    assert started == [1, 1]
