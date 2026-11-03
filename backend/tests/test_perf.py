"""GET /api/perf/turns: per-turn reply-time numbers stored by the conversation turn endpoint (local mode, with ASR,
LLM and TTS faked like tests/test_offline_agents.py)."""
import numpy as np
import pytest

from services import provenance
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)


@pytest.fixture(autouse=True)
def no_watermark_model(monkeypatch):
    """The AudioSeal model cold-loads on its first use, which would land inside the timed turn and make the reply-time
    numbers depend on whether an earlier test warmed it. These tests measure the turn plumbing, so skip the model."""
    monkeypatch.setattr(provenance, "unavailable_reason", lambda: "stubbed out: not part of what is timed here")


def fake_speech(env, monkeypatch):
    async def fake_asr(audio_bytes, name, language=None):
        return "What is the refund window?"

    async def fake_load(key):
        return object()

    monkeypatch.setattr(env, "_speech_to_text", fake_asr)
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": np.zeros(2400, dtype=np.float32))
    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)


def speak(client, agent_id, wav):
    r = client.post(f"/api/conversations/{agent_id}/turn", headers=LOCAL, files={"audio": ("rec.wav", wav, "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


def test_empty_state(app_client, env):
    body = app_client.get("/api/perf/turns", headers=LOCAL).json()
    assert body["turns"] == [] and body["count"] == 0
    assert body["p50_ms"] is None and body["p95_ms"] is None and body["within_target_pct"] is None
    assert body["target_ms"] == 3000 and set(body["stage_p50"]) == {"asr_ms", "retrieval_ms", "llm_ms", "tts_ms"}


def test_turns_are_listed_with_percentiles(app_client, env, monkeypatch, ref_wav):
    fake_speech(env, monkeypatch)
    a = make_agent(app_client)
    wav = ref_wav.read_bytes()
    responses = [speak(app_client, a["id"], wav) for _ in range(3)]
    assert all(r["total_ms"] == r["latency_ms"] for r in responses)

    body = app_client.get("/api/perf/turns", headers=LOCAL).json()
    assert body["count"] == 3 and len(body["turns"]) == 3
    t = body["turns"][0]
    assert t["agent_id"] == a["id"] and t["agent_name"] == "Ria"
    assert set(t["timings"]) == {"asr_ms", "retrieval_ms", "llm_ms", "tts_ms"}
    assert t["engine"] == "kokoro" and t["llm_provider"] == "fake" and t["fallback"] is False and t["language"]
    assert [x["total_ms"] for x in body["turns"]] == [r["total_ms"] for r in reversed(responses)]  # newest first
    totals = sorted(x["total_ms"] for x in body["turns"])
    assert body["p50_ms"] == totals[1] and body["p95_ms"] == totals[2]  # nearest rank over three values
    assert body["within_target_pct"] == 100.0  # faked stages are instant
    assert all(isinstance(v, int) for v in body["stage_p50"].values())

    assert len(app_client.get("/api/perf/turns?limit=2", headers=LOCAL).json()["turns"]) == 2
    assert app_client.get("/api/perf/turns?limit=0", headers=LOCAL).status_code == 422


def test_agent_filter(app_client, env, monkeypatch, ref_wav):
    fake_speech(env, monkeypatch)
    a, b = make_agent(app_client), make_agent(app_client)
    wav = ref_wav.read_bytes()
    speak(app_client, a["id"], wav)
    speak(app_client, a["id"], wav)
    speak(app_client, b["id"], wav)
    assert app_client.get("/api/perf/turns", headers=LOCAL).json()["count"] == 3
    only_a = app_client.get(f"/api/perf/turns?agent_id={a['id']}", headers=LOCAL).json()
    assert only_a["count"] == 2 and {t["agent_id"] for t in only_a["turns"]} == {a["id"]}
    assert app_client.get("/api/perf/turns?agent_id=nope", headers=LOCAL).json()["count"] == 0


def test_other_users_turns_are_invisible(app_client, env, monkeypatch, ref_wav):
    fake_speech(env, monkeypatch)
    a = make_agent(app_client)
    speak(app_client, a["id"], ref_wav.read_bytes())
    for c in env.local_db.select("conversations"):
        env.local_db.update("conversations", c["id"], {"user_id": "someone-else"})
    body = app_client.get("/api/perf/turns", headers=LOCAL).json()
    assert body["count"] == 0 and body["p50_ms"] is None


def test_slow_and_old_turns_count_against_target():
    from routers.perf import _summary, _turn_json
    rows = [{"id": str(i), "agent_id": "a", "timings": {"llm_ms": ms}, "total_ms": ms} for i, ms in
            enumerate([1000, 2000, 3000, 4000, 9000])]
    s = _summary([_turn_json(r, {}) for r in rows])
    assert s["p50_ms"] == 3000 and s["p95_ms"] == 9000 and s["within_target_pct"] == 60.0
    assert _turn_json({"id": "x", "timings": None}, {}) is None  # turns stored before metrics existed are skipped
