"""The approvals routes (routers/approvals.py): list, settings, approve (also at another time), decline with a suggestion,
and the refusals as 409s."""
from datetime import datetime

import pytest

from mcp_servers import meetings as m
from services import gcal
from tests.test_offline_agents import LOCAL, env  # noqa: F401  (env is a fixture)

NOW = datetime(2026, 10, 15, 9, 0)
MONDAY, TUESDAY = "2026-10-19", "2026-10-20"


@pytest.fixture(autouse=True)
def office(monkeypatch, tmp_path):
    monkeypatch.setenv("BONCLAY_MEETINGS_FILE", str(tmp_path / "mcp" / "meetings.json"))
    monkeypatch.setattr(m, "now", lambda: NOW)
    monkeypatch.setitem(m.DEFAULT_SETTINGS, "sample_schedule", False)   # the made-up busy slots have their own tests
    monkeypatch.delenv("GOOGLE_CALENDAR_ID", raising=False)


def ask(**kw):
    args = {"name": "Riya Patil", "role": "student", "reason": "project guide approval", "date": MONDAY, "time": "10:30",
            "phone": "9876543210", "class_or_ref": "TY AI-A"}
    args.update(kw)
    return m.request_meeting(**args)


def test_pending_requests_are_listed_with_counts(app_client):
    ask()
    ask(name="Sanjay Rao", role="faculty", phone="9000000002", time="14:00", class_or_ref="")
    body = app_client.get("/api/meetings", headers=LOCAL).json()
    assert [r["request_id"] for r in body["requests"]] == ["M10", "M11"] and body["counts"]["pending"] == 2
    assert body["requests"][0]["name"] == "Riya Patil" and body["requests"][0]["on_calendar"] is False
    assert app_client.get("/api/meetings?status=bogus", headers=LOCAL).status_code == 422


def test_approve_and_decline_change_the_request_and_the_agent_sees_it(app_client):
    ask()
    ask(name="Sanjay Rao", role="parent", phone="9000000002", time="14:00", class_or_ref="Neha, SY AI-B")
    ok = app_client.post("/api/meetings/M10/approve", headers=LOCAL, json={"note": "Bring the report."})
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    no = app_client.post("/api/meetings/M11/decline", headers=LOCAL,
                         json={"note": "In an exam meeting.", "suggest_date": TUESDAY, "suggest_time": "15:00"})
    assert no.json()["status"] == "declined" and no.json()["suggested"]["time"] == "15:00"
    said = m.meeting_status(phone="9000000002")["requests"][0]
    assert said["decision_note"] == "In an exam meeting." and said["suggested"]["weekday"] == "Tuesday"
    assert app_client.get("/api/meetings?status=all", headers=LOCAL).json()["counts"] == \
        {"pending": 0, "approved": 1, "declined": 1, "cancelled": 0}


def test_a_decided_request_cannot_be_decided_again(app_client):
    ask()
    app_client.post("/api/meetings/M10/approve", headers=LOCAL, json={})
    r = app_client.post("/api/meetings/M10/decline", headers=LOCAL, json={})
    assert r.status_code == 409 and r.json()["code"] == "MEETING_REFUSED" and "already approved" in r.json()["error"]
    assert app_client.post("/api/meetings/M99/approve", headers=LOCAL, json={}).status_code == 409


def test_approving_at_another_time_and_a_calendar_clash(app_client, monkeypatch):
    ask()
    assert app_client.post("/api/meetings/M10/approve", headers=LOCAL,
                           json={"date": TUESDAY, "time": "15:00"}).json()["time"] == "15:00"
    ask(name="Sanjay Rao", role="faculty", phone="9000000002", time="14:00", class_or_ref="")
    monkeypatch.setattr(gcal, "configured", lambda: True)
    monkeypatch.setattr(gcal, "busy", lambda a, b: [(a, b)])
    r = app_client.post("/api/meetings/M11/approve", headers=LOCAL, json={})
    assert r.status_code == 409 and "Calendar is busy" in r.json()["error"]


def test_settings_show_the_rules_and_the_calendar_status(app_client):
    body = app_client.get("/api/meetings/settings", headers=LOCAL).json()
    assert body["settings"]["slot_minutes"] == 20 and body["settings"]["owner"] == "Prof. Jayashri Prasad"
    assert body["calendar"]["configured"] is False and "service_account_email" in body["calendar"]
