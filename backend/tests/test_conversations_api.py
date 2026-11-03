"""Calls and bookings (routers/conversations.py): channel, masked caller, derived outcome and the transcript, from the
local store, owner-scoped and read-only."""
from datetime import datetime, timedelta, timezone

import pytest

from mcp_servers import meetings as m
from routers import conversations as c
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)

NO_INFO = "I'm sorry, I don't have that information. I can connect you with a member of staff."
NOW = datetime(2026, 10, 15, 9, 0)


@pytest.fixture(autouse=True)
def office(monkeypatch, tmp_path):
    monkeypatch.setenv("BONCLAY_MEETINGS_FILE", str(tmp_path / "mcp" / "meetings.json"))
    monkeypatch.setattr(m, "now", lambda: NOW)
    monkeypatch.delenv("GOOGLE_CALENDAR_ID", raising=False)


def iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def talk(env, agent_id, *lines, source=None, minutes_ago=10, user="local-user", is_test=False, **conv):
    """Store a conversation. Each line is (role, text) or (role, text, extra turn fields)."""
    row = {"agent_id": agent_id, "user_id": user, "created_at": iso(minutes_ago),
           **({"source": source} if source else {}), **({"is_test": True} if is_test else {}), **conv}
    cid = env.local_db.insert("conversations", row)["id"]
    for i, line in enumerate(lines):
        role, text, *extra = line
        env.local_db.insert("conversation_turns", {"conversation_id": cid, "role": role, "text_content": text,
                                                   "created_at": iso(minutes_ago - 0.1 * (i + 1)), **(extra[0] if extra else {})})
    return cid


def calls(client, **params):
    r = client.get("/api/calls", headers=LOCAL, params=params)
    assert r.status_code == 200, r.text
    return r.json()["calls"]


GREETING = ("agent", "This is an AI assistant. How can I help?")


def test_channels_and_masked_callers(app_client, env):
    agent = make_agent(app_client)
    a = agent["id"]
    talk(env, a, ("user", "What are your timings today?"), ("agent", "We open at nine."), minutes_ago=50)
    talk(env, a, ("user", "Fees please for the course"), ("agent", "The fee is 500."), source="widget", minutes_ago=40)
    talk(env, a, ("user", "Do you open on Sunday?"), ("agent", "Closed on Sunday."), source="whatsapp",
         external_id="919876543210", minutes_ago=30)
    talk(env, a, ("user", "Hello"), ("agent", "Hi"), is_test=True, minutes_ago=20)
    by = {x["channel"]: x for x in calls(app_client)}
    assert set(by) == {"playground", "website", "whatsapp"}
    assert by["whatsapp"]["caller"] == "••••3210" and "9876543210" not in str(by["whatsapp"])
    assert by["website"]["caller"] is None and by["website"]["duration_s"] is None
    rows = calls(app_client)
    assert [r["practice"] for r in rows] == [True, False, False, False]  # newest first
    assert rows[0]["agent_name"] == agent["name"]


def test_outcomes_are_derived_from_what_is_stored(app_client, env):
    a = make_agent(app_client)["id"]
    book = {"tool_calls": [{"name": "book_appointment", "arguments": {"phone": "9876543210"}, "ok": True,
                            "result": "booking A12 with Dr. Meera Kulkarni on Saturday 2026-10-17 12:30"}]}
    failed = {"tool_calls": [{"name": "book_appointment", "arguments": {}, "ok": False, "result": "failed: nope"}]}
    ids = {
        "booked": talk(env, a, ("user", "I want to book with Dr Meera"), ("agent", "Booked.", book), minutes_ago=90),
        "answered": talk(env, a, ("user", "When do you open today?"), ("agent", "At nine."), minutes_ago=80),
        "couldnt": talk(env, a, ("user", "Do you treat dogs here?"), ("agent", NO_INFO), minutes_ago=70),
        "failed_tool": talk(env, a, ("user", "Book me please now"), ("agent", "I could not.", failed), minutes_ago=60),
        "ended": talk(env, a, ("user", "Hello"), GREETING, source="phone", minutes_ago=50),
        "silent": talk(env, a, source="phone", minutes_ago=40),
        "handoff": talk(env, a, ("user", "I want a person"), ("agent", "Sure."), source="whatsapp",
                        external_id="911234567890", needs_human=True, minutes_ago=30),
    }
    out = {cid: x["outcome"]["key"] for cid, x in ((x["id"], x) for x in calls(app_client))}
    assert out[ids["booked"]] == "booked"
    assert out[ids["answered"]] == "answered"
    assert out[ids["couldnt"]] == "couldnt_answer"
    assert out[ids["failed_tool"]] == "answered"  # a failed tool is not a booking and not a gap
    assert out[ids["ended"]] == "ended_early" and out[ids["silent"]] == "ended_early"
    assert out[ids["handoff"]] == "handed_off"
    detail = next(x for x in calls(app_client) if x["id"] == ids["couldnt"])["outcome"]["detail"]
    assert "dogs" in detail


def test_meeting_requests_follow_the_approvals_store(app_client, env):
    a = make_agent(app_client)["id"]
    req = m.request_meeting(name="Riya Patil", role="student", reason="project guide approval", date="2026-10-19",
                            time="10:30", phone="9876543210")
    rid = req["request_id"]
    made = {"tool_calls": [{"name": "request_meeting", "arguments": {}, "ok": True,
                            "result": f"meeting request {rid} pending for Monday 2026-10-19 10:30"}]}
    cid = talk(env, a, ("user", "I need to meet the HOD"), ("agent", "Requested.", made))
    row = calls(app_client)[0]
    assert row["outcome"]["key"] == "needs_approval" and row["outcome"]["request_id"] == rid
    m.approve(rid)
    assert calls(app_client)[0]["outcome"]["key"] == "booked"
    # a request that was withdrawn in the same conversation does not count
    cancelled = {"tool_calls": [{"name": "cancel_meeting", "arguments": {}, "ok": True, "result": f"meeting request {rid} cancelled"}]}
    env.local_db.insert("conversation_turns", {"conversation_id": cid, "role": "agent", "text_content": "Withdrawn.",
                                               "created_at": iso(1), **cancelled})
    assert calls(app_client)[0]["outcome"]["key"] == "answered"


def test_phone_duration_is_only_shown_for_calls(app_client, env):
    a = make_agent(app_client)["id"]
    talk(env, a, ("user", "I would like to book an appointment"), ("agent", "Sure, which doctor?"),
         ("user", "Dr Meera on Saturday morning"), ("agent", "Done."), source="phone", minutes_ago=5)
    row = calls(app_client)[0]
    assert row["channel"] == "phone" and row["caller"] is None  # the number of a caller on the phone is not stored
    assert 0 < row["duration_s"] < 60 and row["outcome"]["key"] == "answered" and row["replies"] == 2


def test_transcript_masks_numbers_and_hides_tool_arguments(app_client, env):
    a = make_agent(app_client)["id"]
    book = {"tool_calls": [{"name": "book_appointment", "arguments": {"phone": "9876543210"}, "ok": True,
                            "result": "booking A5 with Dr. Meera on Saturday 2026-10-17 12:30"}]}
    cid = talk(env, a, ("user", "My number is 9876543210 and also 878899625"), ("agent", "Booked for 2026-10-17.", book))
    r = app_client.get(f"/api/calls/{cid}", headers=LOCAL)
    assert r.status_code == 200
    body = r.json()
    t = body["transcript"]
    assert t[0] == {"role": "caller", "text": "My number is ••••3210 and also ••••9625", "at": t[0]["at"], "actions": []}
    assert t[1]["text"] == "Booked for 2026-10-17."  # a date is not a number
    assert t[1]["actions"] == [{"name": "book_appointment", "ok": True,
                                "summary": "booking A5 with Dr. Meera on Saturday 2026-10-17 12:30"}]
    assert "9876543210" not in r.text and "arguments" not in r.text
    assert body["outcome"]["key"] == "booked"


def test_owner_scoping_filters_window_and_unknown_ids(app_client, env):
    a = make_agent(app_client)["id"]
    b = make_agent(app_client)["id"]
    mine = talk(env, a, ("user", "How much is the fee?"), ("agent", "Five hundred."))
    talk(env, a, ("user", "someone else's chat"), ("agent", "ok"), user="another-user")
    talk(env, b, ("user", "an old chat with b"), ("agent", "ok"), minutes_ago=60 * 24 * 40)
    talk(env, b, ("user", "a new chat with b"), ("agent", "ok"), minutes_ago=3)
    assert len(calls(app_client)) == 2  # the 40 day old one is outside the default 30 days
    assert len(calls(app_client, days=60)) == 3
    assert [x["id"] for x in calls(app_client, agent_id=a)] == [mine]
    assert len(calls(app_client, limit=1)) == 1
    assert app_client.get("/api/calls/does-not-exist", headers=LOCAL).status_code == 404
    other = env.local_db.select("conversations", lambda r: r["user_id"] == "another-user")[0]["id"]
    assert app_client.get(f"/api/calls/{other}", headers=LOCAL).status_code == 404  # not the owner's
    assert app_client.get("/api/calls?days=0", headers=LOCAL).status_code == 422


def test_empty_store_and_nothing_is_written(app_client, env):
    assert calls(app_client) == []
    make_agent(app_client)
    before = (env.local_db.select("conversations"), env.local_db.select("conversation_turns"))
    calls(app_client)
    assert (env.local_db.select("conversations"), env.local_db.select("conversation_turns")) == before


def test_masking_helpers():
    assert c.mask_number("+91 98765 43210") == "••••3210" and c.mask_number("") is None and c.mask_number("12") == "••••"
    assert c.mask_text("call 98765 43210 or +919876543210") == "call ••••3210 or ••••3210"
    assert c.mask_text("slot 2026-10-17 at 10:30, booking A12") == "slot 2026-10-17 at 10:30, booking A12"
