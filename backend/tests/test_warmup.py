"""POST/GET /api/warmup with every model helper faked (no model is loaded): state transitions, idempotent start,
skipped components, a failing component, and ?include=clone (WARM_CLONE is off here; see test_warm_clone.py)."""
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_offline_agents import LOCAL

IDS = ["llm", "asr", "kokoro_en", "kokoro_hi", "embedding", "watermark", "speaker"]


class Fakes:
    def __init__(self):
        self.calls = []
        self.gate = None  # a threading.Event: the language-model warm-up blocks until it is set
        self.llm_error = None
        self.speaker_ok = True
        self.watermark_reason = None
        self.asr_error = None


@pytest.fixture()
def fx(monkeypatch):
    import backend
    from routers import warmup
    from services import llm, mlx_asr, provenance, speaker

    f = Fakes()
    monkeypatch.setenv("WARM_CLONE", "off")  # the clone step has its own tests (test_warm_clone.py)
    monkeypatch.setattr(warmup, "_state", {"state": "idle", "started_at": None, "finished_at": None,
                                           "components": []})
    monkeypatch.setattr(warmup, "_has", lambda module: True)

    monkeypatch.setattr(llm._mlx, "unavailable_reason", lambda: None)
    monkeypatch.setattr(llm, "warm_local", lambda: f.calls.append("llm.warm_local"))

    def chat(messages, **kw):
        f.calls.append("llm.chat")
        if f.gate is not None:
            assert f.gate.wait(10)
        if f.llm_error:
            raise f.llm_error
        return llm.ChatResult("OK", "mlx", "fake-llama", 5, {})

    monkeypatch.setattr(llm, "chat", chat)

    monkeypatch.setattr(mlx_asr, "unavailable_reason", lambda: None)
    monkeypatch.setattr(mlx_asr, "warm", lambda: f.calls.append("asr.warm"))

    def transcribe(path, language=None, timeout=0):
        f.calls.append("asr.transcribe")
        if f.asr_error:
            raise f.asr_error
        return {"text": ""}

    monkeypatch.setattr(mlx_asr, "transcribe", transcribe)

    async def load_model(key):
        f.calls.append(f"load:{key}")
        return object()

    monkeypatch.setattr(backend.advanced_manager, "load_model", load_model)

    def kokoro(pipeline, text, language="en"):
        f.calls.append(f"kokoro:{language}")
        if language == "hi":
            backend._kokoro_lang_pipelines["h"] = object()
        return np.zeros(24000, dtype=np.float32)

    monkeypatch.setattr(backend, "_kokoro_generate", kokoro)
    monkeypatch.delitem(backend._kokoro_lang_pipelines, "h", raising=False)
    monkeypatch.setattr(backend, "kokoro_unavailable_languages", lambda: set())

    class Embed:
        def encode(self, texts, *a, **k):
            f.calls.append("embed")
            return np.zeros((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(backend, "embedding_model", Embed())

    monkeypatch.setattr(provenance, "unavailable_reason", lambda: f.watermark_reason)
    monkeypatch.setattr(provenance, "watermark", lambda wav, sr, bits: f.calls.append("wm") or wav)
    monkeypatch.setattr(provenance, "detect", lambda wav, sr: f.calls.append("detect") or {})

    monkeypatch.setattr(speaker, "enabled", lambda: True)
    monkeypatch.setattr(speaker, "available", lambda: f.speaker_ok)
    monkeypatch.setattr(speaker, "embed", lambda audio: f.calls.append("speaker.embed"))
    return f


def wait_done(client, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get("/api/warmup", headers=LOCAL).json()
        if body["state"] != "running":
            return body
        time.sleep(0.02)
    raise AssertionError("warm-up did not finish")


def by_id(body):
    return {c["id"]: c for c in body["components"]}


def test_idle_lists_pending_components(app_client, fx):
    body = app_client.get("/api/warmup", headers=LOCAL).json()
    assert body["state"] == "idle" and body["started_at"] is None and body["finished_at"] is None
    assert [c["id"] for c in body["components"]] == IDS
    assert all(c["state"] == "pending" and c["seconds"] is None and c["detail"] is None for c in body["components"])
    assert "Indic-Speak" in body["not_warmed"] and "Chatterbox" in body["not_warmed"]


def test_everything_warms_in_order(app_client, fx):
    import backend

    with TestClient(backend.app, raise_server_exceptions=False) as c:
        fx.gate = threading.Event()
        first = c.post("/api/warmup", headers=LOCAL).json()
        assert first["state"] == "running" and first["started_at"] is not None and first["finished_at"] is None
        assert [x["id"] for x in first["components"]] == IDS

        # while the language model is still warming: running, the first is "warming", the rest "pending"
        time.sleep(0.2)
        mid = c.get("/api/warmup", headers=LOCAL).json()
        assert mid["state"] == "running"
        states = [x["state"] for x in mid["components"]]
        assert states[0] == "warming" and set(states[1:]) == {"pending"}

        fx.gate.set()
        done = wait_done(c)

    assert done["state"] == "ready" and done["finished_at"] >= done["started_at"]
    comps = by_id(done)
    assert all(v["state"] == "ready" and v["seconds"] is not None for v in comps.values()), comps
    assert comps["llm"]["detail"] == "mlx · fake-llama"
    assert fx.calls == ["llm.warm_local", "llm.chat", "asr.warm", "asr.transcribe", "load:kokoro", "kokoro:en",
                        "load:kokoro", "kokoro:hi", "embed", "wm", "detect", "speaker.embed"]
    assert "load:qwen3_0_6b" not in fx.calls


def test_post_is_idempotent_while_running(app_client, fx):
    import backend

    with TestClient(backend.app, raise_server_exceptions=False) as c:
        fx.gate = threading.Event()
        a = c.post("/api/warmup", headers=LOCAL).json()
        b = c.post("/api/warmup", headers=LOCAL).json()
        assert a["started_at"] == b["started_at"] and b["state"] == "running"
        fx.gate.set()
        wait_done(c)
    assert fx.calls.count("llm.chat") == 1 and fx.calls.count("embed") == 1


def test_unavailable_components_are_skipped_with_a_reason(app_client, fx, monkeypatch):
    import backend
    from routers import warmup
    from services import llm, mlx_asr

    fx.speaker_ok = False
    fx.watermark_reason = "AudioSeal unavailable: not installed"
    monkeypatch.setattr(mlx_asr, "unavailable_reason", lambda: "no MLX environment")
    monkeypatch.setattr(backend.whisper_model, "_model", None)
    monkeypatch.setattr(type(backend.whisper_model), "available", property(lambda self: False))
    monkeypatch.setattr(llm._mlx, "unavailable_reason", lambda: "no MLX environment")

    def no_llm(messages, **kw):
        raise llm.LLMUnavailable({"groq": "GROQ_API_KEY is not set", "mlx": "no MLX environment"})

    monkeypatch.setattr(llm, "chat", no_llm)
    monkeypatch.setattr(warmup, "_has", lambda module: module != "sentence_transformers")

    with TestClient(backend.app, raise_server_exceptions=False) as c:
        c.post("/api/warmup", headers=LOCAL)
        done = wait_done(c)

    comps = by_id(done)
    assert done["state"] == "ready"  # nothing failed: skipped is not a failure
    assert comps["llm"]["state"] == "skipped" and "GROQ_API_KEY" in comps["llm"]["detail"]
    assert comps["asr"]["state"] == "skipped" and "no MLX environment" in comps["asr"]["detail"]
    assert comps["embedding"]["state"] == "skipped" and "sentence-transformers" in comps["embedding"]["detail"]
    assert comps["watermark"]["state"] == "skipped" and "AudioSeal" in comps["watermark"]["detail"]
    assert comps["speaker"]["state"] == "skipped"
    assert comps["kokoro_en"]["state"] == "ready" and comps["kokoro_hi"]["state"] == "ready"
    assert "llm.warm_local" not in fx.calls  # the local server is not started when it is unavailable


def test_hindi_voice_without_its_pipeline_is_skipped(app_client, fx, monkeypatch):
    import backend

    def kokoro(pipeline, text, language="en"):  # the Hindi pipeline cannot be built: Kokoro falls back to English
        return np.zeros(24000, dtype=np.float32)

    monkeypatch.setattr(backend, "_kokoro_generate", kokoro)
    with TestClient(backend.app, raise_server_exceptions=False) as c:
        c.post("/api/warmup", headers=LOCAL)
        done = wait_done(c)
    comps = by_id(done)
    assert comps["kokoro_en"]["state"] == "ready"
    assert comps["kokoro_hi"]["state"] == "skipped" and "espeak" in comps["kokoro_hi"]["detail"]


def test_failing_component_gives_partial_and_the_rest_still_warm(app_client, fx):
    import backend

    fx.asr_error = RuntimeError("MLX speech-recognition worker exited during start-up")
    with TestClient(backend.app, raise_server_exceptions=False) as c:
        c.post("/api/warmup", headers=LOCAL)
        done = wait_done(c)

        assert done["state"] == "partial"
        comps = by_id(done)
        assert comps["asr"]["state"] == "failed"
        assert "RuntimeError: MLX speech-recognition worker exited" in comps["asr"]["detail"]
        assert all(comps[i]["state"] == "ready" for i in IDS if i != "asr")

        fx.asr_error = None  # a new warm-up after a partial one starts afresh and can succeed
        c.post("/api/warmup", headers=LOCAL)
        assert wait_done(c)["state"] == "ready"


def test_include_clone_adds_the_clone_engine(app_client, fx, monkeypatch):
    import backend
    from core import memory_guard

    monkeypatch.setattr(memory_guard, "fits", lambda engine: (True, 9000.0, 3000))
    monkeypatch.setattr(backend, "get_local_voices", lambda: [])
    with TestClient(backend.app, raise_server_exceptions=False) as c:
        r = c.post("/api/warmup?include=clone", headers=LOCAL).json()
        assert [x["id"] for x in r["components"]] == IDS + ["clone"]
        done = wait_done(c)
    assert done["state"] == "ready" and by_id(done)["clone"]["state"] == "ready"
    assert fx.calls[-1] == "load:qwen3_0_6b"


def test_requires_auth_when_configured(app_client, fx, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "x")
    assert app_client.get("/api/warmup").status_code == 401
    assert app_client.post("/api/warmup").status_code == 401
