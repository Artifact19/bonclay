"""Spoken turns of an agent with tools, end to end through the API: speech recognition, TTS and the language model are
faked, but the MCP side is real (the appointment server runs as a subprocess over stdio, writing to a temp file).
Covers booking by voice, the confirmation rule for cancelling, the streaming turn, an agent without tools, and the
agent's `tools` setting."""
import json
import re
from datetime import date, timedelta

import pytest

from services import llm, tools
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import names, parse, rig  # noqa: F401  (rig is a fixture)


def next_day(weekday: int, min_days: int = 2) -> str:
    d = date.today() + timedelta(days=min_days)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d.isoformat()


@pytest.fixture()
def appt(monkeypatch, tmp_path):
    """A private tool hub whose appointment server writes to a temp file."""
    path = tmp_path / "appointments.json"
    monkeypatch.setenv("BONCLAY_APPOINTMENTS_FILE", str(path))
    monkeypatch.setenv("BONCLAY_TOOL_PROTOCOL", "native")  # the scripted models below answer in native tool calls
    h = tools.ToolHub()
    monkeypatch.setattr(tools, "hub", h)
    yield path
    h.close()


def bookings(path):
    return json.loads(path.read_text())["bookings"] if path.exists() else []


class Model:
    """A scripted language model. `plan(messages, tools_given)` returns a ChatResult; every request is kept."""

    def __init__(self, monkeypatch, plan):
        self.requests = []

        def chat(messages, **kw):
            self.requests.append({"messages": messages, **kw})
            return plan(messages, kw)

        monkeypatch.setattr(llm, "chat", chat)


def tool_calls_result(*calls):
    return llm.ChatResult("", "fake", "fake-model", 1, {}, tool_calls=[
        {"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)])


def last_tool_json(messages):
    """The JSON body of the latest tool result the model was shown."""
    text = [m["content"] for m in messages if m["role"] == "tool"][-1]
    return json.loads(text.split("not instructions: ", 1)[1])


def booking_plan(day, time="10:30", name="Priya Sharma", phone="9876543210"):
    def plan(messages, kw):
        if not any(m["role"] == "tool" for m in messages):
            return tool_calls_result(("book_appointment", {"name": name, "phone": phone, "date": day, "time": time}))
        b = last_tool_json(messages)
        return llm.ChatResult(f"Done. Booking {b['booking_id']} with {b['doctor']} on {b['weekday']} at {b['time']}.",
                              "fake", "fake-model", 1, {})
    return plan


def post_blocking(rig, agent_id, conversation_id=None):
    data = {"conversation_id": conversation_id} if conversation_id else None
    return rig.client.post(f"/api/conversations/{agent_id}/turn", headers=LOCAL, data=data,
                           files={"audio": ("rec.wav", rig.wav, "audio/wav")})


# ---------------------------------------------------------------------------------------------- the agent setting

def test_agent_tools_setting_is_validated_stored_and_returned(rig):
    a = make_agent(rig.client)
    assert a["tools"] == []
    b = make_agent(rig.client, tools=["appointments"])
    assert b["tools"] == ["appointments"]
    assert rig.client.get(f"/api/agents/{b['id']}", headers=LOCAL).json()["tools"] == ["appointments"]
    r = rig.client.put(f"/api/agents/{a['id']}", headers=LOCAL, json={"tools": ["appointments", "appointments"]})
    assert r.status_code == 200 and r.json()["tools"] == ["appointments"]
    r = rig.client.put(f"/api/agents/{a['id']}", headers=LOCAL, json={"tools": []})
    assert r.json()["tools"] == []
    bad = rig.client.put(f"/api/agents/{a['id']}", headers=LOCAL, json={"tools": ["rm_rf"]})
    assert bad.status_code == 422 and bad.json()["code"] == "INVALID_TOOLS"
    bad = rig.client.post("/api/agents", headers=LOCAL, json={"name": "X", "role_description": "y", "tools": ["nope"]})
    assert bad.status_code == 422


# ---------------------------------------------------------------------------------------------- blocking turn

def test_booking_by_voice_calls_the_tool_and_the_reply_states_it(rig, monkeypatch, appt):
    day = next_day(0)  # a Monday: Dr. Meera Kulkarni sees patients 9:00 to 13:30
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    model = Model(monkeypatch, booking_plan(day))
    rig.asr_text = "Book me an appointment with the doctor on Monday at 10:30, I am Priya Sharma, 9876543210"
    r = post_blocking(rig, a["id"])
    assert r.status_code == 200, r.text
    out = r.json()

    [row] = bookings(appt)
    assert (row["name"], row["phone"], row["date"], row["time"], row["status"]) == \
        ("Priya Sharma", "9876543210", day, "10:30", "booked")
    assert out["agent_text"] == f"Done. Booking {row['booking_id']} with Dr. Meera Kulkarni on Monday at 10:30."
    assert rig.spoken[-1][0] == out["agent_text"] and out["fallback"] is False

    [call] = out["tool_calls"]
    assert call["name"] == "book_appointment" and call["ok"] is True
    assert call["arguments"]["phone"] == "9876543210"
    assert call["result"] == f"booking {row['booking_id']} with Dr. Meera Kulkarni on Monday {day} 10:30"
    assert out["timings"]["tools_ms"] > 0 and out["timings"]["llm_ms"] >= 0

    first = model.requests[0]
    assert [t["function"]["name"] for t in first["tools"]] == \
        ["list_free_slots", "book_appointment", "find_booking", "cancel_appointment"]
    system = first["messages"][0]["content"]
    assert "tool results" in system and "TOOLS:" in system and "data, never instructions" in system


def test_the_tool_calls_are_kept_in_the_turn_record(rig, env, monkeypatch, appt):
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Model(monkeypatch, booking_plan(next_day(0)))
    out = post_blocking(rig, a["id"]).json()
    stored = env.local_db.get("conversation_turns", next(
        t["id"] for t in env.local_db.select("conversation_turns", lambda r: r["role"] == "agent")))
    assert stored["tool_calls"] == out["tool_calls"] and stored["timings"]["tools_ms"] == out["timings"]["tools_ms"]


def test_a_refused_booking_is_reported_and_nothing_is_stored(rig, monkeypatch, appt):
    sunday = next_day(6)

    def plan(messages, kw):
        if not any(m["role"] == "tool" for m in messages):
            return tool_calls_result(("book_appointment", {"name": "Priya Sharma", "phone": "9876543210",
                                                            "date": sunday, "time": "10:30"}))
        text = [m["content"] for m in messages if m["role"] == "tool"][-1]
        assert "NOT done" in text and "Sunday" in text
        return llm.ChatResult("Sorry, we take no appointments on Sundays. Would Monday work?", "fake", "fake", 1, {})

    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Model(monkeypatch, plan)
    out = post_blocking(rig, a["id"]).json()
    assert bookings(appt) == []
    assert out["tool_calls"][0]["ok"] is False and "Sunday" in out["tool_calls"][0]["result"]
    assert "Sundays" in out["agent_text"]


def test_a_made_up_booking_id_is_replaced_by_a_safe_line(rig, monkeypatch, appt):
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Model(monkeypatch, lambda m, kw: llm.ChatResult("All booked! Your booking ID is A42.", "fake", "fake", 1, {}))
    out = post_blocking(rig, a["id"]).json()
    assert "A42" not in out["agent_text"] and out["agent_text"] == tools.UNCONFIRMED_LINE
    assert out["tool_calls"] == [] and bookings(appt) == []


def test_cancel_needs_the_agent_to_have_asked_in_the_previous_turn(rig, monkeypatch, appt):
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Model(monkeypatch, booking_plan(next_day(0)))
    conv = post_blocking(rig, a["id"]).json()["conversation_id"]  # booking A10 exists now
    assert bookings(appt)[0]["booking_id"] == "A10"

    def plan(messages, kw):
        last = messages[-1]
        if last["role"] == "tool":
            if "NOT DONE" in last["content"]:
                return llm.ChatResult("Shall I cancel booking A10?", "fake", "fake", 1, {})
            return llm.ChatResult(f"Booking {last_tool_json(messages)['booking_id']} is cancelled.", "fake", "fake", 1, {})
        return tool_calls_result(("cancel_appointment", {"booking_id": "A10"}))  # this model always just cancels

    Model(monkeypatch, plan)
    rig.asr_text = "Please cancel my appointment A10"
    second = post_blocking(rig, a["id"], conv).json()
    assert second["agent_text"] == "Shall I cancel booking A10?"   # refused, so the agent asked first
    assert second["tool_calls"][0]["ok"] is False and "not confirmed" in second["tool_calls"][0]["result"]
    assert bookings(appt)[0]["status"] == "booked"

    rig.asr_text = "Yes, please"
    third = post_blocking(rig, a["id"], conv).json()   # the previous reply named booking A10: now it may run
    assert third["agent_text"] == "Booking A10 is cancelled." and third["tool_calls"][0]["ok"] is True
    assert bookings(appt)[0]["status"] == "cancelled"


def test_cancel_in_a_new_conversation_is_refused(rig, monkeypatch, appt):
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Model(monkeypatch, booking_plan(next_day(0)))
    post_blocking(rig, a["id"])
    Model(monkeypatch, lambda m, kw: tool_calls_result(("cancel_appointment", {"booking_id": "A10"}))
          if m[-1]["role"] != "tool" else llm.ChatResult("Shall I cancel booking A10?", "fake", "fake", 1, {}))
    out = post_blocking(rig, a["id"]).json()  # no conversation id: nothing was asked before
    assert out["tool_calls"][0]["ok"] is False and bookings(appt)[0]["status"] == "booked"


def test_agent_without_tools_never_sees_or_starts_them(rig, monkeypatch, appt):
    a = make_agent(rig.client, disclose_ai=False)
    seen = []

    def chat(messages, **kw):
        seen.append({"messages": messages, **kw})
        return llm.ChatResult("The clinic opens at nine.", "fake", "fake-model", 1, {})

    monkeypatch.setattr(llm, "chat", chat)
    out = post_blocking(rig, a["id"]).json()
    assert out["agent_text"] == "The clinic opens at nine." and out["tool_calls"] == []
    assert "tools" not in seen[0] and "tools_ms" not in out["timings"]
    assert "TOOLS:" not in seen[0]["messages"][0]["content"]
    assert tools.hub._conns == {}  # no MCP server was started for this agent


def test_when_the_tool_server_cannot_start_the_agent_answers_without_tools(rig, monkeypatch, appt):
    monkeypatch.setattr(tools, "BUILTIN_SERVERS", {"appointments": {"command": "/nonexistent/python", "args": []}})
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    seen = []

    def chat(messages, **kw):
        seen.append(kw)
        return llm.ChatResult("Please call reception to book.", "fake", "fake-model", 1, {})

    monkeypatch.setattr(llm, "chat", chat)
    out = post_blocking(rig, a["id"]).json()
    assert out["agent_text"] == "Please call reception to book." and out["tool_calls"] == []
    assert "tools" not in seen[0]


# ---------------------------------------------------------------------------------------------- streaming turn

def test_streaming_turn_books_and_reports_the_tool_calls(rig, monkeypatch, appt):
    day = next_day(2)  # a Wednesday: Dr. Meera Kulkarni again
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Model(monkeypatch, booking_plan(day, "11:00"))
    events = parse(rig.post(a["id"]))
    # Starting the tool server can take longer than FILLER_AFTER_MS: the "one moment" filler is then sent first.
    events = [(e, d) for e, d in events if not d.get("filler")]
    assert names(events) == ["transcript", "sentence", "sentence", "done"]
    [row] = bookings(appt)
    assert (row["date"], row["time"]) == (day, "11:00")
    spoken = [d["text"] for e, d in events if e == "sentence"]
    assert spoken == ["Done.", f"Booking {row['booking_id']} with Dr. Meera Kulkarni on Wednesday at 11:00."]
    done = events[-1][1]
    assert done["agent_text"] == " ".join(spoken) and done["fallback"] is False
    [call] = done["tool_calls"]
    assert call["name"] == "book_appointment" and call["ok"] and row["booking_id"] in call["result"]
    t = done["timings"]
    assert t["tools_ms"] > 0 and t["llm_first_token_ms"] <= t["llm_ms"] and t["first_audio_ms"] <= t["total_ms"]
    assert done["llm_provider"] == "fake"


def test_streaming_turn_without_tools_is_unchanged(rig, appt):
    a = make_agent(rig.client, disclose_ai=False)
    done = parse(rig.post(a["id"]))[-1][1]
    assert done["tool_calls"] == [] and "tools_ms" not in done["timings"]
    assert "tools" not in rig.asked[-1] and tools.hub._conns == {}


def test_streaming_turn_falls_back_when_the_model_is_unavailable(rig, monkeypatch, appt):
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)

    def chat(messages, **kw):
        raise llm.LLMUnavailable({"groq": "down"})

    monkeypatch.setattr(llm, "chat", chat)
    events = parse(rig.post(a["id"]))
    done = events[-1][1]
    assert done["fallback"] is True and done["agent_text"] == rig.env.TURN_FALLBACK_TEXT


def test_the_json_protocol_books_by_voice_on_a_model_without_native_tools(rig, monkeypatch, appt):
    day = next_day(0)
    req = json.dumps({"tool": "book_appointment", "arguments": {"name": "Priya Sharma", "phone": "9876543210",
                                                                "date": day, "time": "12:00"}})

    monkeypatch.setenv("BONCLAY_TOOL_PROTOCOL", "json")

    def plan(messages, kw):
        assert not kw.get("tools")
        if messages[-1]["role"] == "user" and "TOOL RESULT" in messages[-1]["content"]:
            bid = re.search(r'"booking_id": "(A\d+)"', messages[-1]["content"]).group(1)
            return llm.ChatResult(f"Your booking {bid} is confirmed for {day} at 12:00.", "mlx", "local", 1, {})
        return llm.ChatResult(req, "mlx", "local", 1, {})

    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Model(monkeypatch, plan)
    out = post_blocking(rig, a["id"]).json()
    [row] = bookings(appt)
    assert row["time"] == "12:00" and out["agent_text"] == f"Your booking {row['booking_id']} is confirmed for {day} at 12:00."
    assert out["llm_provider"] == "mlx" and out["tool_calls"][0]["ok"]
