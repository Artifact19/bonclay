"""A live phone call keeps what it needs warm (routers/phone_exotel.py, services/phone_greeting.py, services/retrieval.py).

The call pins its speech engines (Kokoro, the clone engine when the agent uses one), the embedding model, the reranker
when RERANKER=on and speech recognition when it is local. Pins are counts: two overlapping calls release them only when
both have ended, and while a call holds a pin neither the idle sweeper nor the memory guard may unload the engine. A call
start warms the embedding model and the reranker in the background, once, and the reranker warm-up never runs twice at
once. Fakes only: nothing is downloaded or loaded."""
import asyncio
import json
import logging
import threading
import time
import types

import pytest

import backend
from core import memory_guard
from core.errors import AppError
from routers import phone_exotel as px
from services import phone_greeting, retrieval
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_phone_exotel import rig, start_msg, token_for  # noqa: F401  (rig is a fixture)

LOG = "bonclay.phone.exotel"


class _Model:
    busy = False


def _fake_backend(mgr, reranker=True, groq=True):
    """The parts of the backend module a call session reads when it pins (no models, no network)."""
    return types.SimpleNamespace(
        advanced_manager=mgr,
        groq_client=object() if groq else None,
        retrieval=types.SimpleNamespace(reranker_enabled=lambda: reranker),
        CLONE_ENGINE="qwen3_0_6b",
    )


def _session(b, voice_id=None):
    s = px.CallSession.__new__(px.CallSession)
    s.b = b
    s.agent = {"id": "agent-1", "user_id": "local-user", "voice_id": voice_id}
    return s


def _manager(cls):
    m = cls(idle_timeout_sec=0.0, min_residence_sec=0.0)
    m.loaded_models = {"kokoro": (_Model(), 0.0, 0.0), "qwen3_0_6b": (_Model(), 0.0, 0.0)}
    return m


def _evictable_now(m):
    """What the idle sweeper (30 s loop) and the GPU-pressure eviction would unload at this moment."""
    return sorted(m._idle_candidates(100.0)), sorted(m._evictable_models(100.0))


# ------------------------------------------------------------------------------------- pins held by a call

def test_pinned_engines_are_not_evicted_by_the_idle_sweeper_or_the_memory_guard_during_a_call(model_manager_cls, monkeypatch):
    m = _manager(model_manager_cls)
    session = _session(_fake_backend(m, reranker=True, groq=False))
    session._pin_models()
    assert m.is_pinned("kokoro")
    idle, evictable = _evictable_now(m)
    assert idle == ["qwen3_0_6b"] and evictable == ["qwen3_0_6b"]          # kokoro is held by the call

    offered = {}
    monkeypatch.setattr(memory_guard, "ensure_headroom", lambda key, cands, unload: offered.update(c=cands))
    m._ensure_memory("dhvaani")
    assert [k for k, _ in offered["c"]] == ["qwen3_0_6b"]                  # the guard is never offered kokoro


def test_the_memory_guard_with_no_free_memory_unloads_only_unpinned_engines(model_manager_cls, monkeypatch):
    m = _manager(model_manager_cls)
    session = _session(_fake_backend(m, reranker=False, groq=True))
    session._pin_models()
    unloaded = []
    monkeypatch.setattr(m, "_unload_model", lambda key: unloaded.append(key))
    monkeypatch.setenv("MEMORY_GUARD", "on")
    monkeypatch.setattr(memory_guard, "available_mb", lambda: 100.0)      # never enough for dhvaani (needs 3 GB)
    monkeypatch.setattr(memory_guard, "time", types.SimpleNamespace(sleep=lambda s: None))   # no settle pause
    with pytest.raises(AppError) as err:
        m._ensure_memory("dhvaani")
    assert err.value.code == "INSUFFICIENT_MEMORY"
    assert unloaded == ["qwen3_0_6b"] and "kokoro" in m.loaded_models


def test_gpu_pressure_eviction_skips_a_pinned_engine(model_manager_cls, monkeypatch):
    m = _manager(model_manager_cls)
    _session(_fake_backend(m, reranker=False, groq=True))._pin_models()
    m._get_current_gpu_usage = lambda: int(m.max_gpu_memory + 1)      # always over the cap
    monkeypatch.setattr(m, "_ensure_memory", lambda key: None)         # memory is not what this test is about
    monkeypatch.setattr(m, "_instantiate_model_sync", lambda key: _Model())
    asyncio.run(m.load_model("new"))
    assert "kokoro" in m.loaded_models and "qwen3_0_6b" not in m.loaded_models and "new" in m.loaded_models


def test_pins_release_when_the_call_ends_and_idle_eviction_resumes(model_manager_cls):
    m = _manager(model_manager_cls)
    session = _session(_fake_backend(m, reranker=True, groq=False))
    session._pin_models()
    session._unpin_models()
    assert m.pinned == {}
    idle, evictable = _evictable_now(m)
    assert idle == ["kokoro", "qwen3_0_6b"] and evictable == ["kokoro", "qwen3_0_6b"]


def test_two_overlapping_calls_unpin_only_after_both_end(model_manager_cls, caplog):
    m = _manager(model_manager_cls)
    b = _fake_backend(m, reranker=True, groq=False)
    first, second = _session(b), _session(b)
    with caplog.at_level(logging.INFO, logger=LOG):
        first._pin_models()
        second._pin_models()
        assert m.pinned == {"kokoro": 2, "embedding": 2, "reranker": 2, "asr": 2}
        first._unpin_models()
        assert m.pinned == {"kokoro": 1, "embedding": 1, "reranker": 1, "asr": 1}
        assert m.is_pinned("kokoro") and _evictable_now(m)[0] == ["qwen3_0_6b"]   # the second call still holds it
        second._unpin_models()
        assert m.pinned == {} and _evictable_now(m)[0] == ["kokoro", "qwen3_0_6b"]
    lines = [r.getMessage() for r in caplog.records if r.name == LOG]
    assert lines.count("call warm: pinned [kokoro, embedding, reranker, asr]") == 2
    assert lines.count("call warm: unpinned") == 2


def test_pinned_keys_follow_the_call_and_the_settings(model_manager_cls, monkeypatch):
    m = model_manager_cls(idle_timeout_sec=0.0, min_residence_sec=0.0)
    monkeypatch.delenv("PHONE_CLONE_VOICE", raising=False)
    assert _session(_fake_backend(m, reranker=False, groq=True))._pin_keys() == ["kokoro", "embedding"]
    assert _session(_fake_backend(m, reranker=True, groq=True))._pin_keys() == ["kokoro", "embedding", "reranker"]
    assert _session(_fake_backend(m, reranker=True, groq=False))._pin_keys() == ["kokoro", "embedding", "reranker", "asr"]
    monkeypatch.setenv("PHONE_CLONE_VOICE", "1")
    clone = _session(_fake_backend(m, reranker=False, groq=True), voice_id="voice-1")
    assert clone._pin_keys() == ["kokoro", "qwen3_0_6b", "embedding"]


# ------------------------------------------------------------------------------------ reranker warm-up

class _CountingReranker:
    def __init__(self, delay=0.0):
        self.calls, self.pairs, self.inside, self.max_inside = 0, [], 0, 0
        self._lock = threading.Lock()
        self.delay = delay

    def predict(self, pairs, **kw):
        with self._lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
            self.calls += 1
            self.pairs.append(list(pairs))
        time.sleep(self.delay)
        with self._lock:
            self.inside -= 1
        return [0.5] * len(pairs)


@pytest.fixture()
def reranker_state(monkeypatch):
    monkeypatch.setattr(retrieval, "_reranker_failed", None)
    monkeypatch.setattr(retrieval, "_reranker", None)
    monkeypatch.setenv("RERANKER", "on")
    return monkeypatch


def test_reranker_warm_up_scores_one_tiny_pair_when_enabled(reranker_state):
    fake = _CountingReranker()
    reranker_state.setattr(retrieval, "_reranker", fake)
    assert retrieval.warm_reranker() == retrieval.reranker_model_name()
    assert fake.calls == 1 and [len(p) for p in fake.pairs] == [1]


def test_reranker_warm_up_never_runs_when_the_reranker_is_off(reranker_state):
    fake = _CountingReranker()
    reranker_state.setattr(retrieval, "_reranker", fake)
    reranker_state.setenv("RERANKER", "off")
    assert retrieval.warm_reranker() is None
    assert fake.calls == 0


def test_reranker_warm_up_does_not_retry_a_reranker_that_failed_to_load(reranker_state):
    fake = _CountingReranker()
    reranker_state.setattr(retrieval, "_reranker", fake)
    reranker_state.setattr(retrieval, "_reranker_failed", "OSError: no weights")
    assert retrieval.warm_reranker() is None and fake.calls == 0


def test_reranker_warm_up_never_runs_concurrently_with_itself(reranker_state):
    fake = _CountingReranker(delay=0.02)
    reranker_state.setattr(retrieval, "_reranker", fake)
    threads = [threading.Thread(target=retrieval.warm_reranker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake.calls == 4        # each call start asked for its warm-up ...
    assert fake.max_inside == 1   # ... and they ran one at a time


def test_the_call_start_warm_scores_the_embedder_and_the_reranker_once(reranker_state):
    reranker_state.setenv("PHONE_PREWARM", "on")
    fake = _CountingReranker()
    reranker_state.setattr(retrieval, "_reranker", fake)
    encoded = []
    b = types.SimpleNamespace(embedding_model=types.SimpleNamespace(encode=lambda texts: encoded.append(list(texts))),
                              retrieval=retrieval)

    async def run():
        task = phone_greeting.schedule_call_warm(b)
        assert task is not None
        return await task

    assert asyncio.run(run()) == {"embedding": "ok", "reranker": "ok"}
    assert encoded == [["warm-up"]] and fake.calls == 1


def test_the_call_start_warm_is_off_with_phone_prewarm_off(reranker_state):
    reranker_state.setenv("PHONE_PREWARM", "off")
    b = types.SimpleNamespace(embedding_model=types.SimpleNamespace(encode=lambda t: pytest.fail("must not run")),
                              retrieval=retrieval)
    assert phone_greeting.schedule_call_warm(b) is None


def test_a_failing_reranker_warm_up_is_reported_not_raised(reranker_state):
    def boom(*a):
        raise OSError("disk gone")

    b = types.SimpleNamespace(embedding_model=types.SimpleNamespace(encode=lambda t: None),
                              retrieval=types.SimpleNamespace(reranker_enabled=lambda: True, warm_reranker=boom))
    out = asyncio.run(phone_greeting.warm_call_models(b))
    assert out["embedding"] == "ok" and out["reranker"].startswith("failed")


# ------------------------------------------------------------------- a real call start over the socket

class _FakeEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kw):
        self.calls.append(list(texts))
        return [[0.0]] * len(texts)


@pytest.mark.parametrize("mode", ["on", "off"])
def test_a_call_start_pins_the_call_models_and_warms_the_reranker_once(rig, monkeypatch, caplog, mode):
    mgr = backend.advanced_manager
    mgr.pinned.clear()
    monkeypatch.delenv("PHONE_CLONE_VOICE", raising=False)
    monkeypatch.setenv("PHONE_PREWARM", "on")
    monkeypatch.setenv("RERANKER", mode)
    monkeypatch.setattr(phone_greeting, "schedule_warm", lambda *a, **k: None)   # no real speech warm-up in a test
    embedder = _FakeEmbedder()
    monkeypatch.setattr(backend, "embedding_model", embedder)
    warms = []
    monkeypatch.setattr(backend.retrieval, "warm_reranker", lambda: warms.append(1) or "fake-reranker")
    monkeypatch.setattr(backend.retrieval, "reranker_enabled", lambda: mode == "on")
    a = make_agent(rig.client)
    with caplog.at_level(logging.INFO, logger=LOG):
        with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
            ws.send_text(start_msg())
            for _ in range(30):
                if json.loads(ws.receive_text()).get("event") == "media":
                    break
            for _ in range(100):                              # the warm-up runs in the background
                if embedder.calls and (len(warms) == (1 if mode == "on" else 0)):
                    break
                time.sleep(0.02)
            time.sleep(0.1)                                   # and nothing else runs it a second time
            assert embedder.calls.count(["warm-up"]) == 1
            assert len(warms) == (1 if mode == "on" else 0)
            assert "kokoro" in mgr.pinned and "embedding" in mgr.pinned
            assert ("reranker" in mgr.pinned) == (mode == "on")
    def lines():
        return [r.getMessage() for r in caplog.records if r.name == LOG]

    for _ in range(100):                                      # the session's finally unpins as the socket closes
        if not mgr.pinned and "call warm: unpinned" in lines():
            break
        time.sleep(0.02)
    assert mgr.pinned == {}
    assert any(line.startswith("call warm: pinned [kokoro, embedding") for line in lines())
    assert "call warm: unpinned" in lines()
