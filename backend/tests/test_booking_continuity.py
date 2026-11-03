"""A booking spoken over many turns: the agent must not ask again for what the caller already said.

The scripted model is a perfect receptionist that knows only what is in the messages it is sent: it asks for the
name, then the number, then the day and time, skipping whatever already appears in a caller message, and books through
the tool once it has all four. Before the fix the model saw only the last six rows of the conversation, so a caller who
asked a question or two first (or gave the number in pieces) had the name scroll out and was asked for it again."""
from services import llm
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import rig  # noqa: F401  (rig is a fixture)
from tests.test_turn_tools import Model, appt, bookings, last_tool_json, next_day, tool_calls_result  # noqa: F401


def receptionist(day):
    asked = []

    def plan(messages, kw):
        if any(m["role"] == "tool" for m in messages):
            b = last_tool_json(messages)
            return llm.ChatResult(f"Booked {b['booking_id']}.", "fake", "fake-model", 1, {})
        said = " ".join(m["content"] for m in messages if m["role"] == "user")
        name = "Priya Sharma" in said
        digits = "9876543210" in said.replace(" ", "")
        slot = "10:30" in said
        if not name:
            reply = "May I have the patient's name?"
        elif not digits:
            reply = "And the phone number?"
        elif not slot:
            reply = "Which day and time would you like?"
        else:
            return tool_calls_result(("book_appointment", {"name": "Priya Sharma", "phone": "9876543210",
                                                           "date": day, "time": "10:30"}))
        asked.append(reply)
        return llm.ChatResult(reply, "fake", "fake-model", 1, {})
    return plan, asked


def say(rig, agent_id, text, conv=None):
    r = rig.client.post(f"/api/conversations/{agent_id}/turn", headers=LOCAL,
                        data={"text": text, **({"conversation_id": conv} if conv else {})})
    assert r.status_code == 200, r.text
    return r.json()


def test_name_and_number_given_once_are_remembered_through_a_long_booking(rig, monkeypatch, appt):
    day = next_day(0)
    plan, asked = receptionist(day)
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Model(monkeypatch, plan)
    turns = ["I want to book an appointment", "Priya Sharma", "Is Monday okay for the doctor?", "Yes please",
             "98765", "43210", "What is the fee?", "OK", "Monday 10:30"]
    conv = None
    for text in turns:
        out = say(rig, a["id"], text, conv)
        conv = out["conversation_id"]
    assert asked.count("May I have the patient's name?") == 1  # asked once, at the start, never again
    assert asked.count("And the phone number?") >= 1 and "Which day and time would you like?" in asked
    [row] = bookings(appt)
    assert (row["name"], row["phone"], row["time"]) == ("Priya Sharma", "9876543210", "10:30")
    assert out["tool_calls"][0]["name"] == "book_appointment"


def test_an_agent_without_tools_still_sees_only_the_short_window(rig, monkeypatch):
    a = make_agent(rig.client, disclose_ai=False)
    seen = Model(monkeypatch, lambda m, kw: llm.ChatResult("Fine.", "fake", "fake-model", 1, {}))
    conv = None
    for i in range(10):
        conv = say(rig, a["id"], f"question {i}", conv)["conversation_id"]
    assert len(seen.requests[-1]["messages"]) == 1 + rig.env.HISTORY_TURNS + 1
