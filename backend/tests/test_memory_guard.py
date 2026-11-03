import asyncio
import time

import pytest

from core import memory_guard as mg
from core.errors import AppError


def reader(values):
    it = iter(values)
    last = [values[-1]]

    def read():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]
    return read


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("MEMORY_GUARD", "MEMORY_GUARD_RESERVE_MB", "MEMORY_NEED_INDIC_SPEAK_MB"):
        monkeypatch.delenv(k, raising=False)
    # these tests were written with the 7000 MB estimate; the measured default is 8400 (27 Oct 2026), so pin the old one
    monkeypatch.setitem(mg.DEFAULT_NEED_MB, "indic_speak", 7000)


def test_enough_memory_unloads_nothing():
    unloaded = []
    mg.ensure_headroom("indic_speak", [("a", 1.0)], unloaded.append, reader([20000]), settle_s=0)
    assert unloaded == []


def test_unloads_lru_first_and_only_as_many_as_needed():
    unloaded = []
    cands = [("new", 300.0), ("old", 100.0), ("mid", 200.0)]
    # 3000 -> 5000 after "old" -> 9000 after "mid": needs 8500
    mg.ensure_headroom("indic_speak", cands, unloaded.append, reader([3000, 5000, 9000, 9000]), settle_s=0)
    assert unloaded == ["old", "mid"]


def test_refuses_with_app_error_when_still_short():
    unloaded = []
    with pytest.raises(AppError) as e:
        mg.ensure_headroom("indic_speak", [("a", 1.0)], unloaded.append, reader([2000, 2500]), settle_s=0)
    assert e.value.code == "INSUFFICIENT_MEMORY" and e.value.status == 503
    assert e.value.details["required_mb"] == 8500 and e.value.details["unloaded"] == ["a"]
    assert "free memory" in e.value.message


def test_unreadable_memory_allows_load():
    mg.ensure_headroom("indic_speak", [], lambda k: None, lambda: None, settle_s=0)


def test_unknown_engine_and_disabled_guard_skip_check(monkeypatch):
    boom = reader([0])
    mg.ensure_headroom("some_new_engine", [], lambda k: None, boom, settle_s=0)
    monkeypatch.setenv("MEMORY_GUARD", "off")
    mg.ensure_headroom("indic_speak", [], lambda k: None, boom, settle_s=0)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_NEED_INDIC_SPEAK_MB", "1000")
    monkeypatch.setenv("MEMORY_GUARD_RESERVE_MB", "500")
    mg.ensure_headroom("indic_speak", [], lambda k: None, reader([1600]), settle_s=0)
    with pytest.raises(AppError):
        mg.ensure_headroom("indic_speak", [], lambda k: None, reader([1400]), settle_s=0)


def test_logs_decision(caplog):
    caplog.set_level("INFO", logger="bonclay.memory")
    mg.ensure_headroom("dhvaani", [], lambda k: None, reader([9000]), settle_s=0)
    assert "memory guard: dhvaani ok" in caplog.text


def test_manager_unloads_idle_never_busy_then_loads(model_manager_cls, monkeypatch):
    class Busy:
        busy = True
    m = model_manager_cls(max_gpu_memory_gb=12.0, idle_timeout_sec=300.0, min_residence_sec=60.0)
    m._instantiate_model_sync = lambda key: object()
    m._get_current_gpu_usage = lambda: 0
    now = time.time()
    m.loaded_models = {"busy": (Busy(), now - 500, now - 900), "idle": (object(), now - 500, now - 400)}
    state = {"free": 2000}
    orig_unload = m._unload_model

    def unload(key):
        orig_unload(key)
        state["free"] = 12000
    m._unload_model = unload
    monkeypatch.setattr(mg, "available_mb", lambda: state["free"])
    monkeypatch.setattr(mg.time, "sleep", lambda s: None)
    asyncio.run(m.load_model("indic_speak"))
    assert set(m.loaded_models) == {"busy", "indic_speak"}


def test_manager_refuses_load_when_nothing_can_be_freed(model_manager_cls, monkeypatch):
    class Busy:
        busy = True
    m = model_manager_cls(max_gpu_memory_gb=12.0, idle_timeout_sec=300.0, min_residence_sec=60.0)
    loaded = []
    m._instantiate_model_sync = lambda key: loaded.append(key)
    m._get_current_gpu_usage = lambda: 0
    now = time.time()
    m.loaded_models = {"busy": (Busy(), now - 500, now - 900)}
    monkeypatch.setattr(mg, "available_mb", lambda: 1000)
    with pytest.raises(AppError) as e:
        asyncio.run(m.load_model("indic_speak"))
    assert e.value.code == "INSUFFICIENT_MEMORY"
    assert loaded == [] and "busy" in m.loaded_models
