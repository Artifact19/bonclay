"""Read back, then commit (services/tools.py READBACK_TOOLS): a booking or a meeting request runs only after the agent
has read the resolved day, date and time back to the caller and asked. Also the slot memory (known_slots) that keeps
the model from asking for the same detail twice."""
import pytest

from services import tools
from tests.test_tools_client import BOOKED, FakeHub

BOOK = tools.ToolSpec("appointments", "book_appointment", "Book.", {
    "type": "object", "properties": {"name": {"type": "string"}, "phone": {"type": "string"}, "date": {"type": "string"},
                                     "time": {"type": "string"}, "doctor": {"type": "string"}},
    "required": ["name", "phone", "date", "time"]})
MEET = tools.ToolSpec("meetings", "request_meeting", "Request.", {
    "type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "reason": {"type": "string"},
                                     "date": {"type": "string"}, "time": {"type": "string"}, "phone": {"type": "string"}},
    "required": ["name", "role", "reason", "date", "time", "phone"]})
FIND = tools.ToolSpec("appointments", "find_booking", "Find.", {"type": "object", "properties": {"phone": {"type": "string"}}})
ARGS = {"name": "Priya Sharma", "phone": "9876543210", "date": "2026-10-26", "time": "11:00"}   # a Monday
READ_BACK = "Monday 26 October at 11 am for Priya Sharma with the General Physician. Shall I book it?"


@pytest.fixture(autouse=True)
def readback_on(monkeypatch):
    monkeypatch.setenv("TOOLS_READBACK", "on")


@pytest.fixture()
def hub(monkeypatch):
    fake = FakeHub({"book_appointment": BOOKED, "request_meeting": (True, {"request_id": "M3", "status": "pending"})})
    monkeypatch.setattr(tools, "hub", fake)
    return fake


def test_the_first_call_is_refused_with_the_sentence_to_read_back(hub):
    c = tools.execute(BOOK, ARGS, previous_agent_text="Sure, one moment.")
    assert c.ok is False and hub.calls == []
    assert c.value.startswith("NOT DONE")
    assert "Monday 26 October 2026 at 11:00 (11 am) for Priya Sharma" in c.value
    assert "Shall I book it?" in c.value and "after the caller says yes" in c.value


def test_it_runs_once_the_previous_turn_read_the_slot_back_and_asked(hub):
    c = tools.execute(BOOK, ARGS, previous_agent_text=READ_BACK)
    assert c.ok is True and hub.calls == [("book_appointment", ARGS)]


def test_a_read_back_in_hindi_or_hinglish_counts(hub):
    for prev in ("सोमवार 26 अक्टूबर, सुबह 11 बजे, प्रिया शर्मा जी। क्या मैं बुक कर दूँ?",
                 "सोमवार २६ अक्टूबर सुबह ११ बजे। बुक कर दूँ?",
                 "Somvaar 26 October, subah 11 baje, Priya Sharma. Book kar doon?"):
        hub.calls.clear()
        c = tools.execute(BOOK, ARGS, previous_agent_text=prev)
        assert c.ok is True and hub.calls, prev


def test_the_wrong_day_or_hour_in_the_read_back_does_not_count(hub):
    for prev in ("Tuesday 27 October at 11 am for Priya Sharma. Shall I book it?",
                 "Monday 26 October at 4 pm for Priya Sharma. Shall I book it?",
                 "Monday at 11 am for Priya Sharma. Shall I book it?"):            # the day number is missing
        c = tools.execute(BOOK, ARGS, previous_agent_text=prev)
        assert c.ok is False and c.value.startswith("NOT DONE"), prev
    assert hub.calls == []


def test_a_statement_without_a_question_does_not_count(hub):
    c = tools.execute(BOOK, ARGS, previous_agent_text="Monday 26 October at 11 am for Priya Sharma, booking it now.")
    assert c.ok is False and hub.calls == []


def test_the_callers_words_are_resolved_before_the_read_back(hub):
    c = tools.execute(BOOK, {**ARGS, "date": "agle somvaar", "time": "11 baje"}, previous_agent_text=None)
    assert c.ok is False and c.value.startswith("NOT DONE")
    assert "Monday" in c.value and "at 11:00 (11 am)" in c.value
    assert len(c.arguments["date"]) == 10 and c.arguments["time"] == "11:00"      # the record holds ISO values


def test_meeting_requests_are_read_back_too(hub):
    args = {"name": "Rahul", "role": "student", "reason": "fee receipt", "date": "2026-10-27", "time": "15:30",
            "phone": "9876543210"}
    c = tools.execute(MEET, args, previous_agent_text="Let me check.")
    assert c.ok is False and "Shall I send the request?" in c.value
    assert "Tuesday 27 October 2026 at 15:30 (3:30 pm) for Rahul (student) about fee receipt" in c.value
    c = tools.execute(MEET, args, previous_agent_text="Tuesday 27 October at 3:30 pm for Rahul about the fee receipt. "
                                                      "Shall I send the request?")
    assert c.ok is True and hub.calls[-1][0] == "request_meeting"


def test_reads_and_cancels_are_not_affected(hub):
    assert tools.execute(FIND, {"phone": "9876543210"}, previous_agent_text=None).ok is True


def test_the_switch_restores_one_step_writes(hub, monkeypatch):
    monkeypatch.setenv("TOOLS_READBACK", "off")
    assert tools.execute(BOOK, ARGS, previous_agent_text=None).ok is True


# ---------------------------------------------------------------------------------------------- slot memory

HISTORY = [
    {"role": "user", "text_content": "Book me tomorrow", "tool_calls": None},
    {"role": "agent", "text_content": "What is your name?", "tool_calls": [
        {"name": "book_appointment", "arguments": {"name": "Priya", "phone": "98765", "date": "2026-10-23", "time": "11:00"},
         "ok": False, "result": "NOT DONE"}]},
    {"role": "user", "text_content": "Priya Sharma", "tool_calls": None},
    {"role": "agent", "text_content": "Friday 23 October at 11 am?", "tool_calls": [
        {"name": "book_appointment", "arguments": {"name": "Priya Sharma", "phone": "98765", "date": "2026-10-23",
                                                   "time": "11:00"}, "ok": False, "result": "NOT DONE"}]},
]


def test_known_slots_come_from_earlier_tool_calls_latest_first():
    slots = tools.known_slots(HISTORY)
    assert slots == {"name": "Priya Sharma", "phone": "98765", "date": "2026-10-23", "time": "11:00"}
    assert tools.known_slots([]) == {} and tools.known_slots(None) == {}
    assert tools.known_slots([{"role": "agent", "text_content": "hi"}]) == {}


def test_the_known_details_are_added_to_the_callers_message_once():
    slots = tools.known_slots(HISTORY)
    messages = [{"role": "system", "content": "You are Ria."}, {"role": "user", "content": "Yes please"}]
    out = tools.prepare_messages(messages, [BOOK], True, known_slots=slots)
    line = tools.slots_line(slots)
    assert line.startswith(tools.SLOTS_MARKER) and "date 2026-10-23" in line and "name Priya Sharma" in line
    assert out[-1]["content"].count(tools.SLOTS_MARKER) == 1 and out[-1]["content"].endswith(line)
    again = tools.prepare_messages(out, [BOOK], True, known_slots=slots)
    assert again[-1]["content"].count(tools.SLOTS_MARKER) == 1       # never added twice
    assert messages[-1]["content"] == "Yes please"                     # the caller's own messages are untouched
    assert tools.slots_line({}) == "" and tools.slots_line(None) == ""
