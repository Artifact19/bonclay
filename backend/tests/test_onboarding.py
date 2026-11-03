"""Onboarding checklist and setup hints, in local mode with the language model faked."""
import json

import pytest

LOCAL = {"storage-mode": "local"}


@pytest.fixture()
def env(app_client, monkeypatch, tmp_path):
    import backend
    from core import generations as generations_store
    from services import llm
    from services.local_store import LocalStore

    monkeypatch.setattr(backend, "local_db", LocalStore(str(tmp_path)))
    monkeypatch.setattr(backend, "LOCAL_VOICES_FILE", str(tmp_path / "voices.json"))
    monkeypatch.setattr(generations_store, "_data_dir", str(tmp_path))
    monkeypatch.setattr(backend, "supabase", None)
    if hasattr(backend.limiter, "reset"):
        backend.limiter.reset()

    persona = {"name": "Ria", "greeting": "Hi.", "system_instructions": "a refund helper",
               "suggested_languages": ["en"]}
    monkeypatch.setattr(llm, "chat", lambda messages, **kw: llm.ChatResult(json.dumps(persona), "fake", "m", 1, {}))
    return backend


def steps(client):
    r = client.get("/api/onboarding", headers=LOCAL)
    assert r.status_code == 200, r.text
    return r.json()


def test_empty_data_is_zero_of_five(app_client, env):
    body = steps(app_client)
    assert body["completed"] == 0 and body["total"] == 5
    assert [s["id"] for s in body["steps"]] == ["clone_voice", "speak_indian", "create_agent", "add_knowledge",
                                                "run_attack_test"]
    assert all(s["done"] is False and s["href"] and s["hint"] and s["title"] for s in body["steps"])


def test_creating_an_agent_completes_that_step(app_client, env):
    r = app_client.post("/api/agents", headers=LOCAL, json={"name": "Support", "role_description": "refunds"})
    assert r.status_code == 200, r.text
    body = steps(app_client)
    done = {s["id"]: s["done"] for s in body["steps"]}
    assert done["create_agent"] is True
    assert done["add_knowledge"] is False and done["run_attack_test"] is False
    assert body["completed"] == 1


def test_voice_and_indian_generation_steps(app_client, env):
    from core import generations as gs
    from services.auth import LOCAL_USER

    env.save_local_voice({"id": "v1", "name": "Me", "user_id": LOCAL_USER})
    gs.record_generation(LOCAL_USER, "v1", "Me", "hello", "kokoro", "en", 1.0, None, use_supabase=False)
    assert {s["id"]: s["done"] for s in steps(app_client)["steps"]}["speak_indian"] is False
    gs.record_generation(LOCAL_USER, "v1", "Me", "namaste", "kokoro", "hi", 1.0, None, use_supabase=False)
    done = {s["id"]: s["done"] for s in steps(app_client)["steps"]}
    assert done["clone_voice"] is True and done["speak_indian"] is True


def test_setup_hints_shape_and_no_llm_error(app_client, env, monkeypatch):
    from services import llm

    monkeypatch.setattr(llm, "status", lambda: {"active": None, "providers": []})
    r = app_client.get("/api/setup-hints", headers=LOCAL)
    assert r.status_code == 200, r.text
    hints = r.json()["hints"]
    assert isinstance(hints, list)
    ids = [h["id"] for h in hints]
    assert len(ids) == len(set(ids))
    for h in hints:
        assert h["severity"] in ("error", "warning", "info")
        assert h["title"] and h["detail"] and (h["command"] is None or isinstance(h["command"], str))
    assert any(h["id"] == "no_llm" and h["severity"] == "error" for h in hints)


def test_no_llm_hint_absent_when_a_model_is_active(app_client, env, monkeypatch):
    from services import llm

    monkeypatch.setattr(llm, "status", lambda: {"active": "groq", "providers": []})
    ids = [h["id"] for h in app_client.get("/api/setup-hints", headers=LOCAL).json()["hints"]]
    assert "no_llm" not in ids


# -- the six-step business setup checklist ----------------------------------------------------------------------

SETUP_IDS = ["create_agent", "add_documents", "talk_to_it", "pick_voice", "connect_phone", "practice_call"]


def setup(client):
    body = steps(client)
    return body, {s["id"]: s for s in body["setup"]}


@pytest.fixture()
def no_phone(monkeypatch):
    from routers import phone_exotel

    monkeypatch.setattr(phone_exotel, "missing_settings", lambda: ["EXOTEL_API_KEY"])


def test_setup_checklist_is_empty_and_complete_in_shape(app_client, env, no_phone):
    body, by_id = setup(app_client)
    assert [s["id"] for s in body["setup"]] == SETUP_IDS
    assert body["setup_completed"] == 0 and body["setup_total"] == 6
    for s in body["setup"]:
        assert s["done"] is False
        assert s["title"] and s["why"] and s["cta"] and s["href"].startswith("/dashboard/")
        assert isinstance(s["minutes"], int) and s["minutes"] > 0
    assert by_id["connect_phone"]["href"] == "/dashboard/studio/agents"
    assert body["has_agent"] is False and body["agent_id"] is None and body["phone_connected"] is False
    # the legacy keys are still there for older clients
    assert body["total"] == 5 and len(body["steps"]) == 5


def test_agent_documents_voice_and_deep_links(app_client, env, no_phone):
    r = app_client.post("/api/agents", headers=LOCAL, json={"name": "Support", "role_description": "refunds"})
    assert r.status_code == 200, r.text
    agent_id = r.json()["id"]
    body, by_id = setup(app_client)
    assert by_id["create_agent"]["done"] is True
    assert body["has_agent"] is True and body["agent_id"] == agent_id
    assert by_id["connect_phone"]["href"] == f"/dashboard/studio/agents?agent={agent_id}&tab=phone"
    assert by_id["add_documents"]["href"] == f"/dashboard/studio/agents?agent={agent_id}&tab=knowledge"
    assert by_id["add_documents"]["done"] is False and by_id["pick_voice"]["done"] is False

    env.local_db.insert("documents", {"agent_id": agent_id, "user_id": "x", "filename": "fees.pdf"})
    env.local_db.update("agents", agent_id, {"voice_id": "v9"})
    _, by_id = setup(app_client)
    assert by_id["add_documents"]["done"] is True and by_id["pick_voice"]["done"] is True


def test_cloned_voice_counts_as_picking_a_voice(app_client, env, no_phone):
    from services.auth import LOCAL_USER

    env.save_local_voice({"id": "v1", "name": "Me", "user_id": LOCAL_USER})
    assert setup(app_client)[1]["pick_voice"]["done"] is True


def test_talk_and_practice_call_come_from_conversations(app_client, env, no_phone):
    agent_id = app_client.post("/api/agents", headers=LOCAL, json={"name": "S", "role_description": "r"}).json()["id"]
    env.local_db.insert("conversations", {"agent_id": agent_id, "user_id": "u"})
    _, by_id = setup(app_client)
    assert by_id["talk_to_it"]["done"] is True and by_id["practice_call"]["done"] is False
    env.local_db.insert("conversations", {"agent_id": agent_id, "user_id": "u", "is_test": True})
    assert setup(app_client)[1]["practice_call"]["done"] is True


def test_a_conversation_of_another_users_agent_does_not_count(app_client, env, no_phone):
    app_client.post("/api/agents", headers=LOCAL, json={"name": "S", "role_description": "r"})
    env.local_db.insert("conversations", {"agent_id": "someone-elses-agent", "user_id": "u", "is_test": True})
    _, by_id = setup(app_client)
    assert by_id["talk_to_it"]["done"] is False and by_id["practice_call"]["done"] is False


def test_phone_step_follows_the_phone_status(app_client, env, monkeypatch):
    from routers import phone_exotel

    monkeypatch.setattr(phone_exotel, "missing_settings", lambda: [])
    body, by_id = setup(app_client)
    assert by_id["connect_phone"]["done"] is True and body["phone_connected"] is True
    monkeypatch.setattr(phone_exotel, "missing_settings", lambda: ["PUBLIC_BASE_URL"])
    body, by_id = setup(app_client)
    assert by_id["connect_phone"]["done"] is False and body["phone_connected"] is False


def test_approvals_waiting_counts_pending_meeting_requests(app_client, env, no_phone, monkeypatch):
    from mcp_servers import meetings

    monkeypatch.setattr(meetings, "list_requests", lambda status=None: [{"status": "pending"}] * 3)
    assert steps(app_client)["approvals_waiting"] == 3

    def boom(status=None):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(meetings, "list_requests", boom)
    assert steps(app_client)["approvals_waiting"] == 0
