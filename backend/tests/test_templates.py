"""Agent templates and demo mode, with embeddings faked and no language model."""
import hashlib
import time
from pathlib import Path

import numpy as np
import pytest

LOCAL = {"storage-mode": "local"}
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "agents"


class FakeEmbedding:
    loaded = True

    def encode(self, texts, *a, **k):
        out = []
        for t in texts:
            v = np.zeros(64, dtype=np.float32)
            for w in t.lower().split():
                v[int(hashlib.md5(w.strip(".,?!").encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v)
        return np.stack(out)


@pytest.fixture()
def env(app_client, monkeypatch, tmp_path):
    import backend
    from services import llm
    from services.local_store import LocalStore
    from routers import templates as tpl_router

    monkeypatch.setattr(backend, "local_db", LocalStore(str(tmp_path)))
    monkeypatch.setattr(backend, "LOCAL_AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(backend, "LOCAL_VOICES_FILE", str(tmp_path / "voices.json"))
    monkeypatch.setattr(backend, "embedding_model", FakeEmbedding())
    if hasattr(backend.limiter, "reset"):
        backend.limiter.reset()

    calls = []

    def fake_chat(messages, **kw):  # any call here means a template used the LLM
        calls.append(messages)
        raise AssertionError("no LLM call expected")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(backend, "chat_calls", calls, raising=False)
    tpl_router._eval_state.clear()
    return backend


def _raw(tid):
    import json
    return json.loads((TEMPLATES_DIR / f"{tid}.json").read_text(encoding="utf-8"))


def test_list_templates(app_client, env):
    r = app_client.get("/api/agent-templates", headers=LOCAL)
    assert r.status_code == 200
    templates = r.json()["templates"]
    assert len(templates) == 8
    assert len({t["id"] for t in templates}) == 8
    for t in templates:
        assert set(t) == {"id", "name", "role", "languages", "greeting", "document", "suggested_questions", "sample_pdf",
                          "audience", "org_label", "org_placeholder", "personalise"}
        assert t["audience"].startswith("For ") and t["org_label"]
        # every sample organisation name really appears in the template, so swapping it out works
        for n in t["personalise"]["sample_orgs"]:
            assert n in t["greeting"] or n in t["name"] or n in _raw(t["id"])["system_instructions"]
        doc = TEMPLATES_DIR / t["document"]
        assert doc.is_file() and doc.read_text(encoding="utf-8").strip()
        assert "Sample data for a fictional organisation" in doc.read_text(encoding="utf-8")
        assert t["languages"] and t["greeting"] and len(t["suggested_questions"]) >= 2


def test_create_from_template_local(app_client, env):
    r = app_client.post("/api/agents/from-template", headers=LOCAL, json={"template_id": "clinic_receptionist"})
    assert r.status_code == 200, r.text
    a = r.json()
    assert set(a) == {"id", "name", "greeting", "system_instructions", "supported_languages", "voice_id",
                      "disclose_ai", "tools", "created_at", "document_count"}
    assert "Sunrise" in a["name"] and a["supported_languages"] == ["en", "hi"]
    assert a["tools"] == ["appointments"]  # the clinic template can book appointments through MCP
    assert a["document_count"] >= 1 and a["voice_id"] is None
    assert env.chat_calls == []
    got = app_client.get(f"/api/agents/{a['id']}", headers=LOCAL).json()
    assert got["document_count"] == a["document_count"]
    docs = app_client.get(f"/api/agents/{a['id']}/documents", headers=LOCAL).json()["documents"]
    assert docs[0]["filename"] == "Sample - clinic_receptionist.md" and docs[0]["chunk_count"] >= 1


def _post(app_client, **body):
    return app_client.post("/api/agents/from-template", headers=LOCAL, json={"template_id": "clinic_receptionist", **body})


def test_personalised_clinic(app_client, env):
    r = _post(app_client, org_name="  Lotus Care Clinic ", city="Nashik",
              details="Open 8 to 5. Callers ask about vaccines.", languages=["en", "mr"])
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["name"] == "Lotus Care Clinic Receptionist"
    assert "Lotus Care Clinic" in a["greeting"] and "Sunrise" not in a["greeting"]
    assert "Lotus Care Clinic in Nashik" in a["system_instructions"]
    assert "Sunrise" not in a["system_instructions"] and "Pune" not in a["system_instructions"]
    assert a["supported_languages"] == ["en", "mr"] and a["tools"] == ["appointments"]
    docs = app_client.get(f"/api/agents/{a['id']}/documents", headers=LOCAL).json()["documents"]
    assert sorted(d["filename"] for d in docs) == ["About Lotus Care Clinic.md", "Sample - clinic_receptionist.md"]
    assert env.chat_calls == []


def test_personalised_without_city_drops_the_sample_city(app_client, env):
    a = _post(app_client, org_name="Lotus Care Clinic").json()
    assert "Lotus Care Clinic." in a["system_instructions"] and "Pune" not in a["system_instructions"]
    docs = app_client.get(f"/api/agents/{a['id']}/documents", headers=LOCAL).json()["documents"]
    assert [d["filename"] for d in docs] == ["Sample - clinic_receptionist.md"]  # no About file without details


def test_sample_document_is_labelled_and_optional(app_client, env):
    a = _post(app_client, org_name="Lotus Care Clinic", include_sample=False).json()
    assert a["document_count"] == 0
    b = _post(app_client, org_name="Lotus Care Clinic").json()
    chunks = env.local_db.select("document_chunks", lambda r: True)
    assert any("[SAMPLE DOCUMENT" in c["chunk_text"] for c in chunks)
    assert b["document_count"] == 1


@pytest.mark.parametrize("tid", ["clinic_receptionist", "college_admissions", "emi_reminder", "farmer_helpline",
                                 "hindi_tutor", "hod_office", "school_fee_reminder"])
def test_every_template_personalises(app_client, env, tid):
    raw = _raw(tid)
    r = app_client.post("/api/agents/from-template", headers=LOCAL,
                        json={"template_id": tid, "org_name": "Acme Place", "include_sample": False})
    assert r.status_code == 200, r.text
    a = r.json()
    assert "Acme Place" in a["name"]
    for n in raw["personalise"]["sample_orgs"]:
        for text in (a["name"], a["greeting"], a["system_instructions"]):
            assert n not in text, (tid, n)
    for phrase in raw["personalise"]["strip"]:
        assert phrase not in a["system_instructions"]
    assert "Acme Place" in a["system_instructions"] or "Acme Place" in a["greeting"]


def test_hod_name_is_not_rescanned(app_client, env):
    # the user's own name contains part of the sample name; it must not be swapped a second time
    r = app_client.post("/api/agents/from-template", headers=LOCAL, json={
        "template_id": "hod_office", "org_name": "Prof. Jayashri Rao", "include_sample": False}).json()
    assert r["name"] == "HOD Office Assistant: Prof. Jayashri Rao"
    assert "office of Prof. Jayashri Rao," in r["greeting"]


def test_template_without_org_name_is_unchanged(app_client, env):
    a = _post(app_client).json()
    assert a["name"] == "Sunrise Family Clinic Receptionist" and "Sunrise" in a["greeting"]


@pytest.mark.parametrize("body", [
    {"org_name": "   "}, {"org_name": "x" * 81}, {"org_name": "A", "city": "c" * 61},
    {"org_name": "A", "details": "d" * 1001}, {"org_name": "A", "languages": []},
    {"org_name": "A", "languages": ["english"]}, {"city": "Pune"},
])
def test_bad_overrides_are_rejected_before_anything_is_created(app_client, env, body):
    r = _post(app_client, **body)
    assert r.status_code == 422 and r.json()["code"] == "INVALID_TEMPLATE_INPUT", r.text
    assert env.local_db.select("agents", lambda r: True) == []


def test_unknown_template_404(app_client, env):
    r = app_client.post("/api/agents/from-template", headers=LOCAL, json={"template_id": "nope"})
    assert r.status_code == 404 and r.json()["code"] == "TEMPLATE_NOT_FOUND"


def test_demo_round_trip(app_client, env, monkeypatch):
    from fastapi.testclient import TestClient
    from routers import templates as tpl_router

    started = []

    async def fake_eval(request, agent_id, user_id):
        started.append(agent_id)
        env.local_db.insert("evaluation_runs", {"agent_id": agent_id, "user_id": user_id, "run_id": "r1",
                                                "persona": "x", "task_completion": True})

    monkeypatch.setattr(tpl_router, "_run_eval", fake_eval)

    with TestClient(env.app, raise_server_exceptions=False) as c:  # keeps one event loop for the background task
        s = c.get("/api/demo/status", headers=LOCAL).json()
        assert s["loaded"] is False and s["agents"] == [] and s["evaluation_ready"] is False

        r = c.post("/api/demo/load", headers=LOCAL)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["loaded"] is True and len(body["agents"]) == 2 and body["evaluation"] == "running"

        for _ in range(50):
            s = c.get("/api/demo/status", headers=LOCAL).json()
            if s["evaluation_ready"]:
                break
            time.sleep(0.05)
        assert s["loaded"] is True and s["agents"] == body["agents"] and s["evaluation_ready"] is True
        assert started == [body["agents"][0]]

        agents = c.get("/api/agents", headers=LOCAL).json()["agents"]
        assert len(agents) == 2 and all(a["document_count"] >= 1 for a in agents)
        assert env.local_db.select("agents", lambda r: r.get("demo") is True).__len__() == 2

        again = c.post("/api/demo/load", headers=LOCAL).json()  # no duplicates
        assert again["agents"] == body["agents"] and again["evaluation"] == "ready"
        assert len(c.get("/api/agents", headers=LOCAL).json()["agents"]) == 2
        assert len(started) == 1

        # a user's own agent survives a reset
        own = c.post("/api/agents/from-template", headers=LOCAL, json={"template_id": "hindi_tutor"}).json()
        reset = c.post("/api/demo/reset", headers=LOCAL).json()
        assert reset == {"reset": True, "removed": 2}
        assert [a["id"] for a in c.get("/api/agents", headers=LOCAL).json()["agents"]] == [own["id"]]
        assert c.get("/api/demo/status", headers=LOCAL).json() == {
            "loaded": False, "agents": [], "evaluation_ready": False, "evaluation_running": False}
        assert env.local_db.select("evaluation_runs") == []
        assert c.post("/api/demo/reset", headers=LOCAL).json() == {"reset": True, "removed": 0}


def test_demo_cloud_mode_is_refused(app_client, env, monkeypatch):
    monkeypatch.setattr(env, "supabase", object())
    h = {"storage-mode": "cloud"}
    for method, path in (("post", "/api/demo/load"), ("get", "/api/demo/status"), ("post", "/api/demo/reset")):
        r = getattr(app_client, method)(path, headers=h)
        assert r.status_code == 409 and r.json()["code"] == "DEMO_LOCAL_ONLY", (path, r.text)


def test_demo_runs_the_real_evaluation_path(app_client, env, monkeypatch):
    """No fake runner: the background task goes through POST /api/evaluations/run's code with a faked LLM."""
    import json
    from fastapi.testclient import TestClient
    from services import llm

    def chat(messages, *, json_mode=False, **kw):
        text = (json.dumps({"followed_expected_behaviour": "yes", "invented_facts": "no", "broke_rules": "no",
                            "reasoning": "ok"}) if json_mode else "We open at 9 AM.")
        return llm.ChatResult(text, "fake", "fake-model", 5, {})

    monkeypatch.setattr(llm, "chat", chat)
    with TestClient(env.app, raise_server_exceptions=False) as c:
        clinic = c.post("/api/demo/load", headers=LOCAL).json()["agents"][0]
        for _ in range(100):
            if c.get("/api/demo/status", headers=LOCAL).json()["evaluation_ready"]:
                break
            time.sleep(0.05)
        runs = c.get(f"/api/agents/{clinic}/evaluations", headers=LOCAL).json()["runs"]
        assert runs and all(r["policy_compliance"] for r in runs)
