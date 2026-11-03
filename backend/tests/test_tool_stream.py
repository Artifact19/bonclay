"""The tool loop with a streamed answer (services/tools.py ToolStream) and the booking-id guard on streamed text, then
the same thing through the streaming turn endpoint with a scripted model and the real appointment server."""
import json
import re

import pytest

from services import llm, tools
from tests.test_tools_client import BOOK, BOOKED, CANCEL, MESSAGES, FakeHub
from tests.test_turn_stream import FakeStream, names, parse, rig  # noqa: F401  (rig is a fixture)
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401
from tests.test_turn_tools import appt, bookings, next_day  # noqa: F401  (appt is a fixture)

BOOK_JSON = '{"tool": "book_appointment", "arguments": {"name": "P", "phone": "1"}}'


class Script:
    """llm.chat_stream answers from a list of delta lists, one per model call. Every request is kept."""

    def __init__(self, monkeypatch, *replies):
        self.requests, self.queue, self.streams = [], list(replies), []

        def chat_stream(messages, **kw):
            self.requests.append({"messages": [dict(m) for m in messages], **kw})
            reply = self.queue.pop(0)
            if isinstance(reply, Exception):
                raise reply
            s = FakeStream(reply)
            self.streams.append(s)
            return s

        monkeypatch.setattr(llm, "chat_stream", chat_stream)


@pytest.fixture()
def json_protocol(monkeypatch):
    monkeypatch.setenv("BONCLAY_TOOL_PROTOCOL", "json")


def stream_of(**kw):
    return tools.ToolStream(MESSAGES, [BOOK, CANCEL], **kw)


# ---------------------------------------------------------------------------------------------- the loop

def test_a_plain_answer_streams_as_it_arrives(monkeypatch, json_protocol):
    script = Script(monkeypatch, ["Sure, ", "we open ", "at nine."])
    s = stream_of()
    it = iter(s)
    assert next(it) == "Sure, "                    # the first delta reaches the caller before the model is done
    assert script.streams[0].result is None
    assert "".join([next(it), *it]) == "we open at nine."
    assert s.turn.text == "Sure, we open at nine." and s.turn.protocol == "json" and s.turn.calls == []
    assert s.result.text == s.turn.text and s.first_token_ms is not None
    assert s.provider == "fake" and s.model == "fake-model"


def test_a_tool_request_is_executed_and_never_spoken(monkeypatch, json_protocol):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    script = Script(monkeypatch, [BOOK_JSON[:20], BOOK_JSON[20:]], ["Booked: ", "A12 on Monday."])
    s = stream_of()
    spoken = list(s)
    assert "".join(spoken) == "Booked: A12 on Monday."
    assert fake.calls == [("book_appointment", {"name": "P", "phone": "1"})]
    assert [c.name for c in s.turn.calls] == ["book_appointment"]
    # the model saw the result as data, then answered
    last = script.requests[1]["messages"]
    assert last[-1]["role"] == "user" and "TOOL RESULT" in last[-1]["content"] and "A12" in last[-1]["content"]
    assert last[-2]["role"] == "assistant" and last[-2]["content"].startswith('{"tool"')


def test_a_fenced_json_request_is_a_request(monkeypatch, json_protocol):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    Script(monkeypatch, ["```json\n", BOOK_JSON, "\n```"], ["Booked A12."])
    assert "".join(stream_of()) == "Booked A12."
    assert len(fake.calls) == 1


def test_prose_before_a_request_is_spoken_but_the_json_is_not(monkeypatch, json_protocol):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    Script(monkeypatch, ["Let me check. ", BOOK_JSON], ["Booked A12."])
    s = stream_of()
    out = "".join(s)
    assert out == "Let me check. Booked A12."          # one space between the two parts, no JSON
    assert "{" not in out and len(fake.calls) == 1


def test_braces_inside_an_ordinary_answer_are_still_text(monkeypatch, json_protocol):
    Script(monkeypatch, ["Use the code {A} ", "at the desk."])
    assert "".join(stream_of()) == "Use the code {A} at the desk."


def test_tool_json_that_does_not_parse_is_never_spoken(monkeypatch, json_protocol):
    Script(monkeypatch, ["Okay. ", '{"tool": "book_appointment", "arguments": {'])
    assert "".join(stream_of()) == "Okay. "


def test_a_made_up_tool_is_recorded_and_the_model_gets_another_go(monkeypatch, json_protocol):
    monkeypatch.setattr(tools, "hub", FakeHub())
    script = Script(monkeypatch, ['{"tool": "cancel_all", "arguments": {}}'], ["I cannot do that."])
    s = stream_of()
    assert "".join(s) == "I cannot do that."
    assert [(c.name, c.ok) for c in s.turn.calls] == [("cancel_all", False)]
    assert "There is no tool called 'cancel_all'" in script.requests[1]["messages"][-1]["content"]


def test_after_three_calls_the_model_must_answer(monkeypatch, json_protocol):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    Script(monkeypatch, *([[BOOK_JSON]] * 4), ["Done."])
    s = stream_of()
    assert "".join(s) == "Done." and len(fake.calls) == 3


def test_an_empty_reply_raises(monkeypatch, json_protocol):
    Script(monkeypatch, [""])
    with pytest.raises(ValueError, match="empty reply"):
        list(stream_of())


def test_a_model_that_is_unavailable_raises(monkeypatch, json_protocol):
    Script(monkeypatch, llm.LLMUnavailable({"mlx": "down"}))
    with pytest.raises(llm.LLMUnavailable):
        list(stream_of())


def test_the_native_protocol_answers_as_one_delta(monkeypatch):
    monkeypatch.setenv("BONCLAY_TOOL_PROTOCOL", "native")
    monkeypatch.setattr(llm, "chat", lambda messages, **kw: llm.ChatResult("Hello there.", "fake", "m", 1, {}))
    s = stream_of()
    assert list(s) == ["Hello there."] and s.turn.protocol == "native" and s.result.text == "Hello there."


def test_the_system_prompt_and_the_clock_are_prepared_once(monkeypatch, json_protocol):
    script = Script(monkeypatch, ["Hi."])
    list(stream_of())
    msgs = script.requests[0]["messages"]
    assert msgs[0]["content"].count("TOOLS:") == 1 and "Today is" not in msgs[0]["content"]
    assert msgs[-1]["content"].startswith(tools.NOW_MARKER) and msgs[-1]["content"].endswith("Book me in.")


def test_prepare_messages_leaves_a_prepared_prompt_alone():
    section = tools.tools_prompt([BOOK], json_protocol=True)
    user = f"{tools.now_line()}\nBook me in."
    msgs = [{"role": "system", "content": f"You are Ria.\n\n{section}\n\nReply in English."},
            {"role": "user", "content": user}]
    assert tools.prepare_messages(msgs, [BOOK], True) == msgs
    assert tools.prepare_messages(MESSAGES, [BOOK], True)[0]["content"].count("TOOLS:") == 1


def test_the_system_text_does_not_depend_on_the_time():
    from datetime import datetime
    a = tools.tools_prompt([BOOK], json_protocol=True)
    assert a == tools.tools_prompt([BOOK], json_protocol=True)
    assert "Today is" not in a and tools.NOW_MARKER not in a
    assert "2026-10-19" in tools.tools_prompt([BOOK], json_protocol=True, now=datetime(2026, 10, 19, 9, 5))


# ---------------------------------------------------------------------------------------------- the guard

def test_guard_sentence_passes_known_ids_and_replaces_invented_ones():
    ok = tools.ToolCall("book_appointment", {}, True, "booking A12", 1, BOOKED[1])
    assert tools.guard_sentence("Booking A12 is confirmed.", [ok]) is None
    assert tools.guard_sentence("Booking A 12 is confirmed.", [ok]) is None
    assert tools.guard_sentence("Booking A99 is confirmed.", [ok]) == tools.UNCONFIRMED_LINE
    assert tools.guard_sentence("Booking A99 is confirmed.", []) == tools.UNCONFIRMED_LINE
    assert tools.guard_sentence("As I said, A55.", [], "Your booking is A55.") is None
    failed = tools.ToolCall("book_appointment", {}, False, "failed", 1, "no")
    assert tools.guard_sentence("Booking A12 is set.", [failed]) == tools.UNCONFIRMED_LINE


# ---------------------------------------------------------------------------------------------- the streaming turn

def test_streaming_turn_speaks_the_answer_while_it_is_written(rig, monkeypatch, appt, json_protocol):
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Script(monkeypatch, ["The clinic opens at 9 a.m. ", "on Monday. ", "It closes at eight."])
    events = [(e, d) for e, d in parse(rig.post(a["id"])) if not d.get("filler")]
    assert names(events) == ["transcript", "sentence", "sentence", "done"]
    spoken = [d["text"] for e, d in events if e == "sentence"]
    assert spoken == ["The clinic opens at 9 a.m. on Monday.", "It closes at eight."]
    done = events[-1][1]
    assert done["agent_text"] == " ".join(spoken) and done["tool_calls"] == [] and not done["fallback"]
    assert done["llm_provider"] == "fake"


def test_streaming_turn_books_through_the_json_protocol_and_streams_the_confirmation(rig, monkeypatch, appt,
                                                                                      json_protocol):
    day = next_day(2)
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    req = json.dumps({"tool": "book_appointment", "arguments": {"name": "Priya Sharma", "phone": "9876543210",
                                                                "date": day, "time": "11:00"}})
    seen = []

    def chat_stream(messages, **kw):
        seen.append([dict(m) for m in messages])
        if messages[-1]["role"] == "user" and "TOOL RESULT" in messages[-1]["content"]:
            b = re.search(r'"booking_id": "(A\d+)"', messages[-1]["content"]).group(1)
            return FakeStream(["Done. ", f"Booking {b} is set ", f"for {day} at 11:00."])
        return FakeStream([req[:30], req[30:]])

    monkeypatch.setattr(llm, "chat_stream", chat_stream)
    events = [(e, d) for e, d in parse(rig.post(a["id"])) if not d.get("filler")]
    [row] = bookings(appt)
    spoken = [d["text"] for e, d in events if e == "sentence"]
    assert spoken == ["Done.", f"Booking {row['booking_id']} is set for {day} at 11:00."]
    done = events[-1][1]
    assert [c["name"] for c in done["tool_calls"]] == ["book_appointment"] and done["fallback"] is False
    assert done["timings"]["tools_ms"] > 0 and done["timings"]["llm_first_token_ms"] <= done["timings"]["llm_ms"]
    system = seen[0][0]["content"]
    assert system.index("TOOLS:") < system.index("Language policy:") and "Context from the documents:" not in system
    assert "Today is" not in system and "Reply in " not in system   # the reply language is in the caller's message
    last = seen[0][-1]["content"]
    assert last.startswith("Context from the documents:") and "Right now it is" in last and last.endswith(
        "Caller says: When does the clinic open?")
    assert last.index("Right now it is") < last.index("Reply in English, even if") < last.index("Caller says")


def test_an_invented_booking_id_is_replaced_and_the_reply_stops(rig, monkeypatch, appt, json_protocol):
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Script(monkeypatch, ["Sure, I can help. ", "Your booking A77 is confirmed. ", "See you on Monday."])
    events = [(e, d) for e, d in parse(rig.post(a["id"])) if not d.get("filler")]
    spoken = [d["text"] for e, d in events if e == "sentence"]
    assert spoken == ["Sure, I can help.", tools.UNCONFIRMED_LINE]
    done = events[-1][1]
    assert "A77" not in done["agent_text"] and done["agent_text"].endswith(tools.UNCONFIRMED_LINE)
    assert bookings(appt) == []


def test_streaming_tool_turn_falls_back_when_the_model_fails_before_speaking(rig, monkeypatch, appt, json_protocol):
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    Script(monkeypatch, llm.LLMUnavailable({"mlx": "down"}))
    done = parse(rig.post(a["id"]))[-1][1]
    assert done["fallback"] is True and done["agent_text"] == rig.env.TURN_FALLBACK_TEXT


def test_the_blocking_tool_turn_can_be_switched_back_on(rig, monkeypatch, appt, json_protocol):
    monkeypatch.setenv("BONCLAY_TOOL_STREAM", "off")
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    calls = []

    def chat(messages, **kw):
        calls.append(kw)
        return llm.ChatResult("The clinic opens at nine. It closes at eight.", "fake", "m", 1, {})

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "chat_stream", lambda *a, **k: pytest.fail("the model must not be streamed"))
    events = [(e, d) for e, d in parse(rig.post(a["id"])) if not d.get("filler")]
    assert [d["text"] for e, d in events if e == "sentence"] == ["The clinic opens at nine.", "It closes at eight."]
    assert calls and events[-1][1]["fallback"] is False
