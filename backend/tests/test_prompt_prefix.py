"""The prompt prefix a language model can reuse across the turns of one agent (Groq caches a prompt by its exact start):
the system message is byte-identical from turn to turn, and everything that differs per turn (the question, the clock,
the retrieved passages, the key facts, the reply language of the turn, the caller's details) is in the final user
message. Built with the real assembly (backend._turn_messages, services.tools.prepare_messages); the clock, the fact
sheet and the passages are fixed here."""
from datetime import datetime

import pytest

import backend
from services import fact_sheet
from services import tools as tools_mod

AGENT = {"name": "Ria", "system_instructions": "the front desk of Sahyadri Clinic, Pune.",
         "supported_languages": ["en", "hi"]}
BOOK = tools_mod.ToolSpec(server="appointments", tool="book_appointment", description="Book an appointment.",
                          schema={"type": "object", "required": ["name", "phone", "date", "time"],
                                  "properties": {"name": {"type": "string"}, "phone": {"type": "string"},
                                                 "date": {"type": "string"}, "time": {"type": "string"}}})
PASSAGE_FEES = {"header": "Sahyadri Clinic > Fees", "content": "Consultation fee: Rs. 500 (general physician)."}
PASSAGE_DOCTORS = {"header": "Sahyadri Clinic > Doctors", "content": "Dr. Mehta (Cardiology) sits on Tuesday and Friday."}
FACTS = {"fee": ["Consultation fee, general physician: Rs. 500"], "doctor": ["Dr. Mehta (Cardiology) days: Tuesday and Friday"]}


@pytest.fixture()
def clock(monkeypatch):
    """A fixed clock: `now_line()` (called by the turn code without arguments) reads the time set here."""
    real = tools_mod.now_line
    at = {"when": datetime(2026, 10, 23, 10, 14)}
    monkeypatch.setattr(tools_mod, "now_line", lambda now=None: real(at["when"]))
    return at


@pytest.fixture()
def facts(monkeypatch):
    monkeypatch.setattr(fact_sheet, "turn_facts",
                        lambda run: FACTS["fee"] if "fee" in run.user_text.lower() else FACTS["doctor"])


def turn(question, chunks, *, history, language="en", decision=None, specs=(), at=None, clock=None):
    run = backend._TurnRun(None, "agent-prefix", "user-prefix", storage_mode="local")
    run.agent = AGENT
    run.user_text = question
    run.conversation_id = "conv-prefix"
    run.history = list(history)
    run.history_known = True
    run.chunks = chunks
    run.tool_specs = list(specs)
    run.language = language
    run.language_decision = decision
    if clock is not None and at is not None:
        clock["when"] = at
    return run


def send(run):
    messages = backend._turn_messages(run)
    if run.tool_specs:   # the tool loop's first request (services/tools.py)
        messages = tools_mod.prepare_messages(messages, run.tool_specs, True,
                                              known_slots=tools_mod.known_slots(run.history))
    return messages


def history_after(question, answer):
    return [{"role": "user", "text_content": question}, {"role": "agent", "text_content": answer}]


@pytest.mark.parametrize("specs", [(), (BOOK,)], ids=["no-tools", "tools"])
def test_two_turns_have_byte_identical_system_messages(clock, facts, specs):
    first = send(turn("What is the consultation fee?", [PASSAGE_FEES], history=[], specs=specs,
                      at=datetime(2026, 10, 23, 10, 14), clock=clock))
    second = send(turn("Is Dr. Mehta free on Friday?", [PASSAGE_DOCTORS],
                       history=history_after("What is the consultation fee?", "It is Rs. 500."),
                       specs=specs, at=datetime(2026, 10, 23, 10, 19), clock=clock))
    assert first[0]["role"] == second[0]["role"] == "system"
    assert first[0] == second[0]                                             # byte for byte, role and content


def test_the_per_turn_values_are_only_in_the_final_user_message(clock, facts):
    history = history_after("What is the consultation fee?", "It is Rs. 500.")
    msgs = send(turn("Is Dr. Mehta free on Friday?", [PASSAGE_DOCTORS], history=history, specs=(BOOK,),
                     at=datetime(2026, 10, 23, 10, 19), clock=clock))
    system, final = msgs[0]["content"], msgs[-1]
    assert final["role"] == "user" and final["content"].startswith("Context from the documents:")
    for per_turn in ("Dr. Mehta (Cardiology) sits on Tuesday and Friday.", "Dr. Mehta (Cardiology) days",
                     "Right now it is Friday 2026-10-23, 10:19.", "Is Dr. Mehta free on Friday?",
                     "Reply in English, even if the context is in another language."):
        assert per_turn in final["content"]
        assert per_turn not in system
        assert sum(per_turn in m["content"] for m in msgs) == 1           # only in the final message
    assert "10:14" not in final["content"]                                  # the earlier turn's clock is not repeated
    assert "Today is" not in system and "Right now" not in system and "CONTEXT:" not in system


def test_the_clock_and_passages_never_reach_the_system_message(clock, facts):
    msgs = send(turn("What is the consultation fee?", [PASSAGE_FEES], history=[], specs=(BOOK,),
                     at=datetime(2026, 10, 23, 9, 5), clock=clock))
    assert "09:05" not in msgs[0]["content"] and "Right now" not in msgs[0]["content"]
    assert "Consultation fee: Rs. 500" not in msgs[0]["content"]           # the passage itself
    assert "Context from the documents:" not in msgs[0]["content"]         # the header (the rules only name it)
    assert "Right now it is Friday 2026-10-23, 09:05." in msgs[-1]["content"]


def test_a_language_switch_leaves_the_system_message_alone(clock, facts, monkeypatch):
    from services import language_policy
    history = history_after("What is the consultation fee?", "It is Rs. 500.")
    hindi = language_policy.decide("ओपीडी कब खुलती है?", AGENT["supported_languages"], None, "It is Rs. 500.")
    english_turn = send(turn("Is Dr. Mehta free on Friday?", [PASSAGE_DOCTORS], history=history, specs=(BOOK,),
                             language="en", decision=None, at=datetime(2026, 10, 23, 10, 19), clock=clock))
    hindi_turn = send(turn("ओपीडी कब खुलती है?", [PASSAGE_DOCTORS], history=history, specs=(BOOK,),
                           language=hindi.target, decision=hindi, at=datetime(2026, 10, 23, 10, 19), clock=clock))
    assert english_turn[0] == hindi_turn[0]                                 # the reply language is not in the system
    assert "Reply in English, even if" in english_turn[-1]["content"]
    assert "Reply in Hindi, written in Devanagari script, even if" in hindi_turn[-1]["content"]
    assert "Reply in " not in hindi_turn[0]["content"] and "Language policy:" in hindi_turn[0]["content"]


def test_the_language_policy_and_the_rules_are_in_the_system_message(clock, facts):
    msgs = send(turn("What is the consultation fee?", [PASSAGE_FEES], history=[], clock=clock,
                     at=datetime(2026, 10, 23, 10, 14)))
    system = msgs[0]["content"]
    assert "Rules you always follow" in system and "Never invent facts, prices, times or policies." in system
    assert "Language policy: answer in the language and script the caller uses in this message;" in system
    assert "You can speak English, Hindi and Hinglish." in system
    assert system.endswith("Keep names, numbers and times as they are.")
