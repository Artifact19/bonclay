"""The clone step of the warm-up (routers/warmup.py `_warm_clone`): start the clone engine and, when a cloned voice is
stored, render one short sentence with the newest one and throw it away. Everything heavy is faked; nothing may be
written (no generation row, no receipt, no file in the audio folder)."""
import asyncio
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_offline_agents import LOCAL, env  # noqa: F401  (env is a fixture)
from tests.test_warmup import IDS, by_id, fx, wait_done  # noqa: F401  (fx is a fixture)


@pytest.fixture()
def clone(fx, monkeypatch, tmp_path):  # noqa: F811
    """The route's fakes plus a fake clone engine, a spy on `_clone_chunks`, and spies on everything that stores."""
    import backend
    from core import memory_guard
    from routers import warmup
    from services import provenance

    monkeypatch.delenv("WARM_CLONE", raising=False)
    monkeypatch.setattr(warmup, "_clone_rendered", None)
    monkeypatch.setattr(backend, "mlx_python", lambda: "/fake/python")
    monkeypatch.setattr(backend, "LOCAL_AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(memory_guard, "fits", lambda engine: (True, 9000.0, 3000))
    monkeypatch.setattr(backend, "_is_clone_prompt", lambda prompt, model=None: True)

    state = {"voices": [], "model": object(), "load_error": None, "render_error": None, "rendered": [],
             "persisted": [], "stored": []}

    async def load_model(key):
        fx.calls.append(f"load:{key}")
        if key == "qwen3_0_6b" and state["load_error"]:
            raise state["load_error"]
        return state["model"] if key == "qwen3_0_6b" else object()

    async def load_voice_embedding(voice_id, user_id, storage_mode, row=None):
        fx.calls.append(f"prompt:{voice_id}")
        return {"prompt_of": voice_id}

    async def clone_chunks(model, text, prompt, language="Auto", **kw):
        if state["render_error"]:
            raise state["render_error"]
        state["rendered"].append((model, text, prompt, language))
        return np.zeros(12000, dtype=np.float32), 24000

    async def persist_generation(*a, **k):
        state["persisted"].append((a, k))
        raise AssertionError("a warm-up must not store a generation")

    monkeypatch.setattr(backend.advanced_manager, "load_model", load_model)
    monkeypatch.setattr(backend, "get_local_voices", lambda: list(state["voices"]))
    monkeypatch.setattr(backend, "load_voice_embedding", load_voice_embedding)
    monkeypatch.setattr(backend, "_clone_chunks", clone_chunks)
    monkeypatch.setattr(backend, "persist_generation", persist_generation)
    monkeypatch.setattr(provenance, "apply_provenance", lambda **k: state["stored"].append(k))
    monkeypatch.setattr(backend, "_voice_file_write", lambda *a, **k: state["stored"].append(a))
    state["dir"] = tmp_path
    return state


def voice(voice_id, created, engine="qwen3_0_6b", **extra):
    return {"id": voice_id, "display_name": voice_id.title(), "engine": engine, "created_at": created,
            "user_id": "local-user", "consent_verified": True, **extra}


def run_warmup(path="/api/warmup"):
    import backend

    with TestClient(backend.app, raise_server_exceptions=False) as c:
        c.post(path, headers=LOCAL)
        return wait_done(c)


def test_the_result_includes_the_clone_step_by_default(clone):
    done = run_warmup()
    assert [c["id"] for c in done["components"]] == IDS + ["clone"]
    comp = by_id(done)["clone"]
    assert comp["state"] == "ready" and comp["seconds"] is not None and "engine started" in comp["detail"]
    assert done["state"] == "ready"


def test_without_a_cloned_voice_only_the_engine_starts(clone, fx):  # noqa: F811
    clone["voices"] = [voice("kokoro-preset", "2026-10-01", engine="kokoro")]   # not a cloned voice
    done = run_warmup()
    assert by_id(done)["clone"]["state"] == "ready" and "no cloned voice" in by_id(done)["clone"]["detail"]
    assert "load:qwen3_0_6b" in fx.calls and clone["rendered"] == []
    assert not any(c.startswith("prompt:") for c in fx.calls)


def test_with_a_cloned_voice_it_renders_once_and_writes_nothing(clone, fx):  # noqa: F811
    clone["voices"] = [voice("old", "2026-10-01"), voice("new", "2026-10-20"), voice("middle", "2026-10-10"),
                       voice("withdrawn", "2026-10-21", consent_verified=False)]
    done = run_warmup()
    comp = by_id(done)["clone"]
    assert comp["state"] == "ready" and "rendered 0.5 s" in comp["detail"] and "New" in comp["detail"]
    assert [(r[1], r[2], r[3]) for r in clone["rendered"]] == [("Hello, how can I help you today?",
                                                                 {"prompt_of": "new"}, "en")]
    assert clone["persisted"] == [] and clone["stored"] == []        # no generation row, no receipt, no voice file
    assert os.listdir(clone["dir"]) == []                            # nothing in the audio folder either
    assert fx.calls.index("load:qwen3_0_6b") < fx.calls.index("prompt:new")


def test_a_second_warm_up_does_not_render_again(clone):
    clone["voices"] = [voice("new", "2026-10-20")]
    run_warmup()
    done = run_warmup()
    assert len(clone["rendered"]) == 1 and "already rendered" in by_id(done)["clone"]["detail"]


def test_warm_clone_off_skips_the_step(clone, fx, monkeypatch):  # noqa: F811
    monkeypatch.setenv("WARM_CLONE", "off")
    clone["voices"] = [voice("new", "2026-10-20")]
    done = run_warmup()
    assert [c["id"] for c in done["components"]] == IDS
    assert "load:qwen3_0_6b" not in fx.calls and clone["rendered"] == []


def test_include_clone_overrides_warm_clone_off(clone, fx, monkeypatch):  # noqa: F811
    monkeypatch.setenv("WARM_CLONE", "off")
    monkeypatch.setattr("backend.mlx_python", lambda: None)          # even where cloning runs on PyTorch
    done = run_warmup("/api/warmup?include=clone")
    assert by_id(done)["clone"]["state"] == "ready" and "load:qwen3_0_6b" in fx.calls


def test_an_engine_that_fails_to_start_is_reported_and_the_rest_still_warm(clone):
    from core.errors import AppError

    clone["load_error"] = AppError("MODEL_LOAD_FAILED", "Could not load model 'qwen3_0_6b': boom", 503)
    done = run_warmup()
    comps = by_id(done)
    assert comps["clone"]["state"] == "failed" and "MODEL_LOAD_FAILED" in comps["clone"]["detail"]
    assert all(comps[i]["state"] == "ready" for i in IDS)


def test_a_failing_render_is_reported_not_raised(clone):
    clone["voices"] = [voice("new", "2026-10-20")]
    clone["render_error"] = RuntimeError("MLX worker: out of memory")
    done = run_warmup()
    comp = by_id(done)["clone"]
    assert comp["state"] == "failed" and "out of memory" in comp["detail"]
    assert clone["persisted"] == [] and os.listdir(clone["dir"]) == []


def test_no_headroom_skips_the_step_with_a_reason(clone, fx, monkeypatch):  # noqa: F811
    from core import memory_guard

    monkeypatch.setattr(memory_guard, "fits", lambda engine: (False, 1200.0, 4500))
    clone["voices"] = [voice("new", "2026-10-20")]
    done = run_warmup()
    comp = by_id(done)["clone"]
    assert comp["state"] == "skipped" and "1200 MB free" in comp["detail"] and "4500 MB" in comp["detail"]
    assert "load:qwen3_0_6b" not in fx.calls and done["state"] == "ready"


def test_on_a_machine_without_mlx_the_default_step_is_skipped(clone, fx, monkeypatch):  # noqa: F811
    monkeypatch.setattr("backend.mlx_python", lambda: None)
    done = run_warmup()
    comp = by_id(done)["clone"]
    assert comp["state"] == "skipped" and "PyTorch" in comp["detail"] and "load:qwen3_0_6b" not in fx.calls


# ------------------------------------------------------------------------------------------------ startup warm-up

def test_the_startup_warm_up_has_a_clone_step(clone, env, monkeypatch):  # noqa: F811
    from routers import warmup

    clone["voices"] = [voice("new", "2026-10-20")]
    for name in ("_warm_llm_connection", "_warm_asr_connection", "_warm_kokoro_en", "_warm_kokoro_hi", "_warm_clips",
                 "_warm_embedding", "_warm_watermark"):
        monkeypatch.setattr(warmup, name, lambda: asyncio.sleep(0, "ok"))
    monkeypatch.setattr(warmup, "_warm_knowledge", lambda agent_id: asyncio.sleep(0, "ok"))
    monkeypatch.setattr(warmup, "_live_task", None)
    out = asyncio.run(env.warm_for_live_turns("startup"))
    assert list(out)[-1] == "clone" and out["clone"]["state"] == "ready" and "rendered" in out["clone"]["detail"]
    assert len(clone["rendered"]) == 1 and clone["persisted"] == []


def test_the_startup_clone_step_starts_only_after_every_other_step_finished(clone, env, monkeypatch):  # noqa: F811
    """Kokoro and the rest of the live path win the memory: the clone engine loads after all of them are done."""
    import time

    from routers import warmup

    events = []

    def fake(name, delay):
        async def run(*_a):
            events.append(("start", name, time.time()))
            await asyncio.sleep(delay)
            events.append(("end", name, time.time()))
            return "ok"
        return run

    delays = {"llm": 0.05, "asr": 0.15, "kokoro_en": 0.1, "kokoro_hi": 0.1, "clips": 0.1, "embedding": 0.05,
              "watermark": 0.02}
    for fn, name in (("_warm_llm_connection", "llm"), ("_warm_asr_connection", "asr"), ("_warm_kokoro_en", "kokoro_en"),
                     ("_warm_kokoro_hi", "kokoro_hi"), ("_warm_clips", "clips"), ("_warm_embedding", "embedding"),
                     ("_warm_watermark", "watermark")):
        monkeypatch.setattr(warmup, fn, fake(name, delays[name]))
    monkeypatch.setattr(warmup, "_warm_knowledge", fake("knowledge", 0.2))
    monkeypatch.setattr(warmup, "_warm_clone", fake("clone", 0.01))
    monkeypatch.setattr(warmup, "_live_task", None)
    out = asyncio.run(env.warm_for_live_turns("startup"))
    assert list(out)[-1] == "clone" and out["clone"]["state"] == "ready"
    clone_start = next(t for k, n, t in events if k == "start" and n == "clone")
    others_end = [t for k, n, t in events if k == "end" and n != "clone"]
    assert len(others_end) == 8 and clone_start >= max(others_end)


def test_the_startup_warm_up_reports_a_failing_clone_engine_and_does_not_raise(clone, env, monkeypatch):  # noqa: F811
    from routers import warmup

    clone["load_error"] = RuntimeError("worker died")
    for name in ("_warm_llm_connection", "_warm_asr_connection", "_warm_kokoro_en", "_warm_kokoro_hi", "_warm_clips",
                 "_warm_embedding", "_warm_watermark"):
        monkeypatch.setattr(warmup, name, lambda: asyncio.sleep(0, "ok"))
    monkeypatch.setattr(warmup, "_warm_knowledge", lambda agent_id: asyncio.sleep(0, "ok"))
    monkeypatch.setattr(warmup, "_live_task", None)
    out = asyncio.run(env.warm_for_live_turns("startup"))
    assert out["clone"]["state"] == "failed" and "worker died" in out["clone"]["detail"]
    assert out["watermark"]["state"] == "ready"


def test_the_startup_warm_up_honours_warm_clone_off_and_a_ringing_phone_skips_the_step(clone, fx, env, monkeypatch):  # noqa: F811
    from routers import warmup

    for name in ("_warm_llm_connection", "_warm_asr_connection", "_warm_kokoro_en", "_warm_kokoro_hi", "_warm_clips",
                 "_warm_embedding", "_warm_watermark"):
        monkeypatch.setattr(warmup, name, lambda: asyncio.sleep(0, "ok"))
    monkeypatch.setattr(warmup, "_warm_knowledge", lambda agent_id: asyncio.sleep(0, "ok"))
    monkeypatch.setattr(warmup, "_live_task", None)
    monkeypatch.setenv("WARM_CLONE", "off")
    out = asyncio.run(env.warm_for_live_turns("startup"))
    assert out["clone"] == {"state": "skipped", "seconds": 0.0, "detail": "WARM_CLONE=off"}
    monkeypatch.delenv("WARM_CLONE")
    monkeypatch.setattr(warmup, "_live_task", None)
    out = asyncio.run(env.warm_for_live_turns("phone", "agent-1"))
    assert "clone" not in out and "load:qwen3_0_6b" not in fx.calls
