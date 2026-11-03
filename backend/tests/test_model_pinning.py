"""A live phone call pins the engines its turns need (backend.AdvancedModelManager.pin): pinned engines are never idle
candidates, never evicted under GPU pressure and never offered to the memory guard; the pin is released when the call
ends, and a count so two calls on one engine release it only when both are over."""
import json

import pytest
from starlette.websockets import WebSocketDisconnect

import backend
from core import memory_guard
from routers import phone_exotel as px
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_phone_exotel import rig, start_msg, token_for  # noqa: F401  (rig is a fixture)


class _Model:
    busy = False


def _manager(model_manager_cls):
    m = model_manager_cls(idle_timeout_sec=0.0, min_residence_sec=0.0)
    m.loaded_models = {"kokoro": (_Model(), 0.0, 0.0), "qwen3_0_6b": (_Model(), 0.0, 0.0)}
    return m


def test_pinned_engines_are_neither_idle_nor_evictable(model_manager_cls):
    m = _manager(model_manager_cls)
    assert sorted(m._idle_candidates(100.0)) == ["kokoro", "qwen3_0_6b"]
    assert m.pin("kokoro") == 1
    assert m._idle_candidates(100.0) == ["qwen3_0_6b"]
    assert list(m._evictable_models(100.0)) == ["qwen3_0_6b"]
    assert m.is_pinned("kokoro") and not m.is_pinned("qwen3_0_6b")


def test_the_pin_is_a_count_and_releases_cleanly(model_manager_cls):
    m = _manager(model_manager_cls)
    assert m.pin("kokoro") == 1 and m.pin("kokoro") == 2        # two calls
    assert m.unpin("kokoro") == 1 and m.is_pinned("kokoro")     # one ended: still held
    assert m.unpin("kokoro") == 0 and not m.is_pinned("kokoro") # both ended: free
    assert m.unpin("kokoro") == 0 and "kokoro" not in m.pinned  # extra unpins are harmless
    assert sorted(m._idle_candidates(100.0)) == ["kokoro", "qwen3_0_6b"]


def test_a_pin_may_be_placed_before_the_engine_is_loaded(model_manager_cls):
    m = model_manager_cls(idle_timeout_sec=0.0, min_residence_sec=0.0)
    m.pin("kokoro")
    m.loaded_models["kokoro"] = (_Model(), 0.0, 0.0)
    assert m._idle_candidates(100.0) == []


def test_the_memory_guard_is_never_offered_a_pinned_engine(model_manager_cls, monkeypatch):
    m = _manager(model_manager_cls)
    seen = {}
    monkeypatch.setattr(memory_guard, "ensure_headroom", lambda key, candidates, unload: seen.update(c=candidates))
    m.pin("kokoro")
    m._ensure_memory("dhvaani")
    assert [k for k, _ in seen["c"]] == ["qwen3_0_6b"]


def test_a_phone_call_pins_kokoro_for_its_duration(rig, monkeypatch):
    mgr = backend.advanced_manager
    mgr.pinned.clear()
    monkeypatch.delenv("PHONE_CLONE_VOICE", raising=False)
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        for _ in range(30):                                   # the greeting's first media frame means on_start ran
            if json.loads(ws.receive_text()).get("event") == "media":
                break
        assert mgr.pinned.get("kokoro") == 1
        assert "qwen3_0_6b" not in mgr.pinned                  # Kokoro-only calls do not hold the clone engine
    for _ in range(50):                                        # the session's finally runs as the socket closes
        if not mgr.pinned:
            break
        import time
        time.sleep(0.02)
    assert mgr.pinned.get("kokoro", 0) == 0


def test_a_clone_voice_call_pins_the_clone_engine_too(monkeypatch):
    monkeypatch.setenv("PHONE_CLONE_VOICE", "1")
    session = px.CallSession.__new__(px.CallSession)
    session.b, session.agent = backend, {"id": "agent-1", "user_id": "local-user", "voice_id": "voice-123"}
    assert session._phone_engines() == ["kokoro", backend.CLONE_ENGINE]
    monkeypatch.setenv("PHONE_CLONE_VOICE", "0")
    assert session._phone_engines() == ["kokoro"]


def test_health_lists_pinned_engines(app_client):
    mgr = backend.advanced_manager
    mgr.pinned.clear()
    mgr.pin("kokoro")
    try:
        assert app_client.get("/api/health").json()["models_pinned"] == {"kokoro": 1}
    finally:
        mgr.unpin("kokoro")
