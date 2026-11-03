import asyncio
import time

import pytest

GB = 1024 ** 3


def make(cls, **kw):
    m = cls(max_gpu_memory_gb=12.0, idle_timeout_sec=300.0, min_residence_sec=kw.get("min_res", 60.0))
    m._instantiate_model_sync = lambda key: object()
    return m


def test_lru_eviction_under_vram_pressure(model_manager_cls):
    m = make(model_manager_cls)
    now = time.time()
    # three resident models, all past min residence; "b" is least recently used
    m.loaded_models = {
        "a": (object(), now - 500, now - 100),
        "b": (object(), now - 500, now - 400),
        "c": (object(), now - 500, now - 50),
    }
    # headroom < 1 GB: usage reports above the cap until one model is evicted
    state = {"over": True}
    m._get_current_gpu_usage = lambda: int(m.max_gpu_memory + 0.5 * GB) if len(m.loaded_models) >= 3 else int(m.max_gpu_memory - 3 * GB)
    asyncio.run(m.load_model("d"))
    assert "b" not in m.loaded_models
    assert {"a", "c", "d"} == set(m.loaded_models)


def test_evicts_multiple_until_pressure_clears(model_manager_cls):
    m = make(model_manager_cls)
    now = time.time()
    m.loaded_models = {k: (object(), now - 500, now - (400 - i * 100)) for i, k in enumerate("abc")}
    m._get_current_gpu_usage = lambda: int(m.max_gpu_memory + GB) if len(m.loaded_models) > 1 else 0
    asyncio.run(m.load_model("d"))
    assert set(m.loaded_models) == {"c", "d"}  # a (oldest) then b evicted


def test_min_residence_protects_recent_models(model_manager_cls):
    m = make(model_manager_cls)
    now = time.time()
    m.loaded_models = {"fresh": (object(), now - 5, now - 5)}
    m._get_current_gpu_usage = lambda: int(m.max_gpu_memory + GB)
    asyncio.run(m.load_model("new"))
    assert "fresh" in m.loaded_models and "new" in m.loaded_models


def test_cache_hit_updates_last_used_and_no_reload(model_manager_cls):
    m = make(model_manager_cls)
    calls = []
    m._instantiate_model_sync = lambda key: calls.append(key) or object()
    first = asyncio.run(m.load_model("x"))
    t0 = m.loaded_models["x"][2]
    time.sleep(0.01)
    second = asyncio.run(m.load_model("x"))
    assert first is second and calls == ["x"]
    assert m.loaded_models["x"][2] > t0


def test_no_eviction_without_pressure(model_manager_cls):
    m = make(model_manager_cls)
    now = time.time()
    m.loaded_models = {"a": (object(), now - 500, now - 400)}
    m._get_current_gpu_usage = lambda: int(m.max_gpu_memory - 2 * GB)
    asyncio.run(m.load_model("b"))
    assert set(m.loaded_models) == {"a", "b"}
