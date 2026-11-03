"""Agent Studio without Supabase or Groq: the whole agent lifecycle in local mode, with the language model, speech
recognition, embeddings and TTS faked so this runs on CPU in seconds."""
import hashlib
import json

import numpy as np
import pytest

LOCAL = {"storage-mode": "local"}


def fake_embed(texts):
    """Deterministic bag-of-words vectors: texts sharing words are close in cosine similarity."""
    out = []
    for t in texts:
        v = np.zeros(64, dtype=np.float32)
        for w in t.lower().split():
            v[int(hashlib.md5(w.strip(".,?!").encode()).hexdigest(), 16) % 64] += 1.0
        out.append(v)
    return np.stack(out)


class FakeEmbedding:
    loaded = True

    def encode(self, texts, *a, **k):
        return fake_embed(list(texts))


PERSONA = {"name": "Ria", "greeting": "Hi, I'm Ria.", "system_instructions": "a friendly refund assistant",
           "suggested_languages": ["en", "hi"]}


@pytest.fixture()
def env(app_client, monkeypatch, tmp_path):
    import backend
    from services import llm
    from services.local_store import LocalStore

    monkeypatch.setattr(backend, "local_db", LocalStore(str(tmp_path)))
    monkeypatch.setattr(backend, "LOCAL_AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(backend, "LOCAL_VOICES_FILE", str(tmp_path / "voices.json"))
    monkeypatch.setattr(backend, "embedding_model", FakeEmbedding())
    if hasattr(backend.limiter, "reset"):
        backend.limiter.reset()

    calls = []

    def fake_chat(messages, *, json_mode=False, **kw):
        calls.append({"messages": messages, "json_mode": json_mode, **kw})
        if json_mode and "grader" in messages[-1]["content"]:
            text = json.dumps({"followed_expected_behaviour": "yes", "invented_facts": "no", "broke_rules": "no",
                               "reasoning": "ok"})
        elif json_mode:
            text = json.dumps(PERSONA)
        else:
            text = "Refunds are possible within thirty days."
        return llm.ChatResult(text, "fake", "fake-model", 5, {})

    monkeypatch.setattr(llm, "chat", fake_chat)
    backend.chat_calls = calls
    return backend


def make_agent(client, **extra):
    r = client.post("/api/agents", headers=LOCAL, json={"name": "Support", "role_description": "refund helper",
                                                         **extra})
    assert r.status_code == 200, r.text
    return r.json()


def upload_txt(client, agent_id, text="The refund window is thirty days from purchase. " * 5, name="policy.txt"):
    return client.post("/api/knowledge/upload", headers=LOCAL, data={"agent_id": agent_id},
                       files={"file": (name, text.encode(), "text/plain")})


def test_agent_crud_and_cascade(app_client, env, ref_wav):
    a = make_agent(app_client)
    assert set(a) == {"id", "name", "greeting", "system_instructions", "supported_languages", "voice_id",
                      "disclose_ai", "tools", "created_at", "document_count"}
    assert a["name"] == "Ria" and a["supported_languages"] == ["en", "hi"] and a["voice_id"] is None
    assert a["document_count"] == 0
    assert env.chat_calls[0]["json_mode"] is True

    b = make_agent(app_client)
    ids = [x["id"] for x in app_client.get("/api/agents", headers=LOCAL).json()["agents"]]
    assert ids == [b["id"], a["id"]]  # newest first

    got = app_client.get(f"/api/agents/{a['id']}", headers=LOCAL)
    assert got.status_code == 200 and got.json()["id"] == a["id"]
    miss = app_client.get("/api/agents/nope", headers=LOCAL)
    assert miss.status_code == 404 and miss.json()["code"] == "AGENT_NOT_FOUND"

    r = app_client.put(f"/api/agents/{a['id']}", headers=LOCAL,
                       json={"greeting": "Hello!", "supported_languages": ["hi"]})
    assert r.status_code == 200
    assert r.json()["greeting"] == "Hello!" and r.json()["supported_languages"] == ["hi"]
    assert r.json()["name"] == "Ria"  # untouched fields stay

    doc = upload_txt(app_client, a["id"]).json()["document"]
    assert app_client.get(f"/api/agents/{a['id']}", headers=LOCAL).json()["document_count"] == 1
    assert doc["chunk_count"] >= 1

    # a conversation turn so there is something to cascade
    env.local_db.insert("conversations", {"agent_id": a["id"], "user_id": "local-user"})
    d = app_client.delete(f"/api/agents/{a['id']}", headers=LOCAL)
    assert d.status_code == 200 and d.json() == {"deleted": True}
    assert app_client.get(f"/api/agents/{a['id']}", headers=LOCAL).status_code == 404
    assert env.local_db.select("documents", lambda r: r["agent_id"] == a["id"]) == []
    assert env.local_db.select("document_chunks") == []
    assert env.local_db.select("conversations", lambda r: r["agent_id"] == a["id"]) == []
    assert app_client.delete(f"/api/agents/{a['id']}", headers=LOCAL).status_code == 404
    assert [x["id"] for x in app_client.get("/api/agents", headers=LOCAL).json()["agents"]] == [b["id"]]


def test_foreign_agents_are_invisible(app_client, env):
    a = make_agent(app_client)
    env.local_db.update("agents", a["id"], {"user_id": "someone-else"})
    assert app_client.get(f"/api/agents/{a['id']}", headers=LOCAL).status_code == 404
    assert app_client.get("/api/agents", headers=LOCAL).json() == {"agents": []}


def test_knowledge_upload_and_retrieve(app_client, env):
    import asyncio
    a = make_agent(app_client)
    r = upload_txt(app_client, a["id"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["document"]["filename"] == "policy.txt" and body["document"]["id"] == body["document_id"]
    chunks = asyncio.run(env.retrieve_context(a["id"], "refund window", user_id="local-user", storage_mode="local"))
    assert chunks and "refund window" in chunks[0]

    docs = app_client.get(f"/api/agents/{a['id']}/documents", headers=LOCAL).json()["documents"]
    assert [d["id"] for d in docs] == [body["document_id"]] and docs[0]["chunk_count"] == body["chunks_created"]

    assert app_client.delete(f"/api/agents/{a['id']}/documents/{body['document_id']}",
                             headers=LOCAL).json() == {"deleted": True}
    assert app_client.get(f"/api/agents/{a['id']}/documents", headers=LOCAL).json() == {"documents": []}
    assert env.local_db.select("document_chunks") == []
    assert app_client.delete(f"/api/agents/{a['id']}/documents/{body['document_id']}",
                             headers=LOCAL).status_code == 404


def test_knowledge_upload_validation(app_client, env):
    a = make_agent(app_client)
    bad = app_client.post("/api/knowledge/upload", headers=LOCAL, data={"agent_id": a["id"]},
                          files={"file": ("x.docx", b"hello", "application/octet-stream")})
    assert bad.status_code == 415 and bad.json()["code"] == "INVALID_FILE_TYPE"
    assert upload_txt(app_client, a["id"], text="x", name="a.md").status_code == 200  # markdown is accepted
    r = app_client.post("/api/knowledge/upload", headers=LOCAL, data={"agent_id": a["id"]},
                        files={"file": ("a.txt", b"\xff\xfe\x00bad", "text/plain")})
    assert r.status_code == 415
    big = app_client.post("/api/knowledge/upload", headers=LOCAL, data={"agent_id": a["id"]},
                          files={"file": ("a.txt", b"a " * (1024 * 1024 + 1), "text/plain")})
    assert big.status_code == 413


def test_turn_returns_provider_and_playable_audio(app_client, env, monkeypatch, ref_wav):
    async def fake_asr(audio_bytes, name, language=None):
        return "What is the refund window?"

    class FakeKokoro:
        pass

    async def fake_load(key):
        return FakeKokoro()

    monkeypatch.setattr(env, "_speech_to_text", fake_asr)
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": np.zeros(2400, dtype=np.float32))
    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)

    a = make_agent(app_client)
    upload_txt(app_client, a["id"])
    wav = ref_wav.read_bytes()
    r = app_client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL,
                        files={"audio": ("rec.wav", wav, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["llm_provider"] == "fake" and body["llm_model"] == "fake-model"
    assert body["fallback"] is False and body["grounded"] is True and body["engine"] == "kokoro"
    assert body["user_text"] == "What is the refund window?"
    assert body["agent_text"] == "This is an AI assistant. Refunds are possible within thirty days."
    assert body["agent_audio_url"].startswith("/api/conversations/turns/")
    assert {"asr_ms", "retrieval_ms", "llm_ms", "tts_ms"} <= set(body["timings"])

    audio = app_client.get(body["agent_audio_url"], headers=LOCAL)
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert audio.content[:4] == b"RIFF"
    assert app_client.get("/api/conversations/turns/unknown/audio", headers=LOCAL).status_code == 404

    # second turn keeps the conversation and sees history
    r2 = app_client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL, data={
        "conversation_id": body["conversation_id"]}, files={"audio": ("rec.wav", wav, "audio/wav")})
    assert r2.status_code == 200 and r2.json()["conversation_id"] == body["conversation_id"]
    assert len(env.chat_calls[-1]["messages"]) == 4  # system + user + agent + user

    stats = app_client.get(f"/api/analytics/agents/{a['id']}", headers=LOCAL).json()
    assert stats["total_turns"] == 2 and "en" in stats["language_distribution"]

    # the audio is private to the owner
    env.local_db.update("agents", a["id"], {"user_id": "someone-else"})
    assert app_client.get(body["agent_audio_url"], headers=LOCAL).status_code == 404


def _turn_setup(env, monkeypatch, user_text="What is the refund window?"):
    async def fake_asr(audio_bytes, name, language=None):
        return user_text

    async def fake_load(key):
        return object()

    spoken = []

    def fake_kokoro(model, text, lang="en"):
        spoken.append(text)
        return np.zeros(2400, dtype=np.float32)

    monkeypatch.setattr(env, "_speech_to_text", fake_asr)
    monkeypatch.setattr(env, "_kokoro_generate", fake_kokoro)
    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
    return spoken


def _turn(client, agent_id, wav, conversation_id=None):
    r = client.post(f"/api/conversations/{agent_id}/turn", headers=LOCAL,
                    data={"conversation_id": conversation_id} if conversation_id else None,
                    files={"audio": ("rec.wav", wav, "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


def test_agent_says_it_is_an_ai_on_the_first_reply_only(app_client, env, monkeypatch, ref_wav):
    spoken = _turn_setup(env, monkeypatch)
    a = make_agent(app_client)
    assert a["disclose_ai"] is True
    wav = ref_wav.read_bytes()
    first = _turn(app_client, a["id"], wav)
    assert first["agent_text"].startswith("This is an AI assistant. ")
    assert spoken[-1] == first["agent_text"]  # spoken, not only displayed
    second = _turn(app_client, a["id"], wav, first["conversation_id"])
    assert second["agent_text"] == "Refunds are possible within thirty days."
    # a new conversation starts with the disclosure again
    assert _turn(app_client, a["id"], wav)["agent_text"].startswith("This is an AI assistant.")


def test_disclosure_can_be_switched_off(app_client, env, monkeypatch, ref_wav):
    _turn_setup(env, monkeypatch)
    a = make_agent(app_client, disclose_ai=False)
    assert a["disclose_ai"] is False
    assert _turn(app_client, a["id"], ref_wav.read_bytes())["agent_text"] == "Refunds are possible within thirty days."
    r = app_client.put(f"/api/agents/{a['id']}", headers=LOCAL, json={"disclose_ai": True})
    assert r.status_code == 200 and r.json()["disclose_ai"] is True
    assert app_client.get(f"/api/agents/{a['id']}", headers=LOCAL).json()["disclose_ai"] is True
    assert _turn(app_client, a["id"], ref_wav.read_bytes())["agent_text"].startswith("This is an AI assistant.")


def test_disclosure_follows_the_reply_language(env):
    assert env.ai_disclosure("hi") == "यह एक AI सहायक है।"
    assert env.ai_disclosure("mr") == "हा एक AI सहाय्यक आहे।"
    assert env.ai_disclosure("en") == env.ai_disclosure("ta") == "This is an AI assistant."


def test_turn_falls_back_without_llm(app_client, env, monkeypatch, ref_wav):
    from services import llm

    def boom(*a, **k):
        raise llm.LLMUnavailable({"groq": "GROQ_API_KEY is not set"})

    async def fake_asr(audio_bytes, name, language=None):
        return "hello"

    async def fake_load(key):
        return object()

    monkeypatch.setattr(llm, "chat", boom)
    monkeypatch.setattr(env, "_speech_to_text", fake_asr)
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": np.zeros(2400, dtype=np.float32))
    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
    a = make_agent_direct(env)
    r = app_client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL,
                        files={"audio": ("rec.wav", ref_wav.read_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fallback"] is True and body["llm_provider"] is None and body["llm_model"] is None
    assert body["agent_text"] == "This is an AI assistant. I'm having trouble responding right now."


def make_agent_direct(env):
    return env.local_db.insert("agents", {"user_id": "local-user", "project_id": "local_project", "name": "A",
                                          "system_instructions": "x", "greeting": "", "supported_languages": ["en"]})


def test_turn_requires_some_speech_recognition(app_client, env, monkeypatch, ref_wav):
    async def no_asr(audio_bytes, name, language=None):
        return None

    monkeypatch.setattr(env, "_speech_to_text", no_asr)
    a = make_agent(app_client)
    r = app_client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL,
                        files={"audio": ("rec.wav", ref_wav.read_bytes(), "audio/wav")})
    assert r.status_code == 503 and r.json()["code"] == "ASR_UNAVAILABLE"


def test_evaluation_run_is_stored_and_listed(app_client, env):
    a = make_agent(app_client)
    assert app_client.get(f"/api/agents/{a['id']}/evaluations", headers=LOCAL).json() == {"runs": []}
    r = app_client.post("/api/evaluations/run", headers=LOCAL, json={"agent_id": a["id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["personas_tested"] == 6 and body["overall_score"] == 100 and len(body["results"]) == 6
    assert body["results"][0]["persona"] == "confused_customer"
    assert body["results"][0]["agent_reply"] == "Refunds are possible within thirty days."
    runs = app_client.get(f"/api/agents/{a['id']}/evaluations", headers=LOCAL).json()["runs"]
    assert len(runs) == 6 and {r["persona"] for r in runs} == {x["persona"] for x in body["results"]}
    assert all(r["run_id"] == body["run_id"] and r["agent_reply"] and r["task_completion"] for r in runs)
    assert app_client.get("/api/agents/nope/evaluations", headers=LOCAL).status_code == 404


def test_create_agent_without_llm_is_503(app_client, env, monkeypatch):
    from services import llm

    def boom(*a, **k):
        raise llm.LLMUnavailable({"groq": "GROQ_API_KEY is not set", "mlx": "no MLX environment"})

    monkeypatch.setattr(llm, "chat", boom)
    r = app_client.post("/api/agents", headers=LOCAL, json={"name": "X", "role_description": "y"})
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "LLM_UNAVAILABLE" and "GROQ_API_KEY" in body["error"]
    assert set(body["details"]["reasons"]) == {"groq", "mlx"}
    assert app_client.get("/api/agents", headers=LOCAL).json() == {"agents": []}

    a = make_agent_direct(env)
    r = app_client.post("/api/evaluations/run", headers=LOCAL, json={"agent_id": a["id"]})
    assert r.status_code == 503 and r.json()["code"] == "LLM_UNAVAILABLE"
    assert env.local_db.select("evaluation_runs") == []


def test_health_and_warm_report_llm(app_client, env, monkeypatch):
    from services import llm
    monkeypatch.setattr(llm, "warm_local", lambda: None)
    assert "llm" in app_client.get("/api/health").json()
    assert "llm" in app_client.get("/api/models/status").json()
    r = app_client.post("/api/llm/warm", headers=LOCAL)
    assert r.status_code == 200 and r.json()["ok"] is True and "providers" in r.json()["llm"]
