"""GET /api/analytics/results: the agent-centred "is it working?" numbers (local mode, temp store, owner scoping)."""
from datetime import datetime, timedelta, timezone

import pytest

from mcp_servers import meetings
from routers import analytics
from tests.test_offline_agents import LOCAL, env  # noqa: F401  (env is a fixture)

NO_INFO = "I'm sorry, I don't have that information. I can connect you with a member of staff."
OK = "We are open nine to five."


@pytest.fixture(autouse=True)
def office(monkeypatch, tmp_path):
    monkeypatch.setenv("BONCLAY_MEETINGS_FILE", str(tmp_path / "mcp" / "meetings.json"))


def seed(env, user, agent_id, qa, days_ago=1, language="en", is_test=False, ms=1000):
    """One conversation of (question, reply) turns; every agent turn carries a reply time of `ms` (a list per turn works)."""
    db = env.local_db
    conv = db.insert("conversations", {"agent_id": agent_id, "user_id": user, **({"is_test": True} if is_test else {})})
    base = datetime.now(timezone.utc) - timedelta(days=days_ago)
    for i, (q, r) in enumerate(qa):
        t = base + timedelta(minutes=i)
        total = ms[i] if isinstance(ms, list) else ms
        db.insert("conversation_turns", {"conversation_id": conv["id"], "role": "user", "text_content": q,
                                         "created_at": t.isoformat()})
        db.insert("conversation_turns", {"conversation_id": conv["id"], "role": "agent", "text_content": r,
                                         "created_at": (t + timedelta(seconds=1)).isoformat(), "language": language,
                                         "fallback": False, "latency_ms": total, "total_ms": total,
                                         "timings": {"llm_ms": total}})
    return conv["id"]


def get(client, qs=""):
    r = client.get(f"/api/analytics/results{qs}", headers=LOCAL)
    assert r.status_code == 200, r.text
    return r.json()


def test_empty_state_has_zeros_and_no_reply_time(app_client, env):
    body = get(app_client)
    assert body["conversations"] == 0 and body["questions_asked"] == 0
    assert body["questions_answered"] == 0 and body["questions_unanswered"] == 0
    assert body["reply_time"]["p50_ms"] is None and body["reply_time"]["p95_ms"] is None
    assert body["reply_time"]["count"] == 0
    assert len(body["daily"]) == 7 and body["by_language"] == []
    assert body["range"] == "7d" and body["agent_id"] is None


def test_counts_answers_gaps_and_reply_times(app_client, env):
    a = env.local_db.insert("agents", {"user_id": "local-user", "name": "Ria"})
    seed(env, "local-user", a["id"], [("What are your hours please?", OK), ("Do you have parking space?", NO_INFO)],
         ms=[1000, 5000])
    seed(env, "local-user", a["id"], [("Where is the clinic located?", OK)], language="hi", ms=2000)
    seed(env, "local-user", a["id"], [("Old question about fees?", NO_INFO)], days_ago=20)  # outside 7 days
    body = get(app_client)
    assert body["conversations"] == 2
    assert (body["questions_asked"], body["questions_answered"], body["questions_unanswered"]) == (3, 2, 1)
    rt = body["reply_time"]
    assert rt["count"] == 3 and rt["p50_ms"] == 2000 and rt["p95_ms"] == 5000
    assert rt["within_target_pct"] == 66.7 and rt["target_ms"] == 3000
    assert {x["language"]: x["count"] for x in body["by_language"]} == {"en": 2, "hi": 1}
    assert sum(d["asked"] for d in body["daily"]) == 3 and sum(d["unanswered"] for d in body["daily"]) == 1
    wide = get(app_client, "?range=30d")
    assert wide["conversations"] == 3 and wide["questions_unanswered"] == 2 and len(wide["daily"]) == 30


def test_test_calls_are_not_counted_but_their_reply_times_are(app_client, env):
    a = env.local_db.insert("agents", {"user_id": "local-user", "name": "Ria"})
    seed(env, "local-user", a["id"], [("Practice question about fees?", NO_INFO)], is_test=True, ms=4000)
    body = get(app_client)
    assert body["conversations"] == 0 and body["questions_asked"] == 0
    assert body["reply_time"]["p50_ms"] == 4000  # same turns as the Overview page's reply time


def test_agent_filter_and_owner_scoping(app_client, env):
    mine = env.local_db.insert("agents", {"user_id": "local-user", "name": "Mine"})
    mine2 = env.local_db.insert("agents", {"user_id": "local-user", "name": "Mine2"})
    theirs = env.local_db.insert("agents", {"user_id": "someone-else", "name": "Theirs"})
    seed(env, "local-user", mine["id"], [("Do you sell gift cards?", NO_INFO)])
    seed(env, "local-user", mine2["id"], [("Where is the parking lot?", OK), ("Is there wheelchair access?", OK)])
    seed(env, "someone-else", theirs["id"], [("Secret competitor question here", NO_INFO)])
    assert get(app_client)["questions_asked"] == 3
    one = get(app_client, f"?agent_id={mine2['id']}")
    assert one["questions_asked"] == 2 and one["questions_unanswered"] == 0 and one["agent_name"] == "Mine2"
    foreign = get(app_client, f"?agent_id={theirs['id']}")
    assert foreign["questions_asked"] == 0 and foreign["agent_name"] is None


def test_bookings_come_from_the_meeting_requests_for_all_agents_only(app_client, env):
    day = datetime.now().isoformat(timespec="seconds")
    with meetings.transaction() as data:
        base = {"name": "X", "role": "student", "date": "2026-10-20", "time": "10:00", "minutes": 15, "reason": "r",
                "phone": "9876543210", "created_at": day}
        data["requests"] += [
            {**base, "request_id": "M10", "status": "pending"},
            {**base, "request_id": "M11", "status": "approved", "decided_at": day},
            {**base, "request_id": "M12", "status": "approved", "decided_at": day, "created_at": "2020-01-01T09:00:00"},
            {**base, "request_id": "M13", "status": "declined", "decided_at": day},
        ]
    a = env.local_db.insert("agents", {"user_id": "local-user", "name": "Ria"})
    assert get(app_client)["bookings"] == {"requested": 3, "booked": 2, "waiting": 1}
    assert get(app_client, f"?agent_id={a['id']}")["bookings"] is None  # not split by agent: not tracked, not faked


def test_bookings_are_null_when_the_meetings_file_cannot_be_read(app_client, env, monkeypatch):
    def boom(status=None):
        raise OSError("unreadable")
    monkeypatch.setattr(meetings, "list_requests", boom)
    assert get(app_client)["bookings"] is None


def test_validates_range(app_client, env):
    assert app_client.get("/api/analytics/results?range=90d", headers=LOCAL).status_code == 422


def test_the_existing_summary_fields_are_unchanged(app_client, env):
    body = app_client.get("/api/analytics/summary", headers=LOCAL).json()
    assert set(body) == {"range", "by_engine", "by_language", "daily", "latency"}
    assert analytics.router.prefix == "/api/analytics"
