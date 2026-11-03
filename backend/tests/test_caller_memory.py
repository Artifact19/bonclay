"""What the caller has told the agent in this conversation (services/caller_memory.py): the rules that read it, how it is
merged and kept, how the model is told on every turn (with or without tools), the small model that reads what the rules
cannot, and the format example the Whisper prompt gets after a request for a name, a number, a date or a time."""
import json
import os
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import pytest

from services import caller_memory as cm
from services import tools
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import Rig, parse

NOW = datetime(2026, 10, 23, 10, 0)      # a Friday


@pytest.fixture()
def rig(app_client, env, monkeypatch, ref_wav):
    return Rig(env, monkeypatch, app_client, ref_wav)


def read(text, agent=None):
    return cm.extract(text, agent, NOW)


# ================================================================================================ the rules

@pytest.mark.parametrize("said", [
    "98765 43210",
    "my number is 9876543210",
    "+91 98765 43210",
    "+919876543210",
    "09876543210",
    "nine eight seven six five four three two one zero",
    "nau aath saat chhe paanch char teen do ek shunya",
    "९८७६५ ४३२१०",
    "नौ आठ सात छह पाँच चार तीन दो एक शून्य",
    "98765 43210 25 October",
])
def test_a_phone_number_is_read_however_it_is_said(said):
    assert read(said)["phone"] == "9876543210"


def test_double_and_triple_in_a_number():
    assert read("double nine eight double seven six five four three two")["phone"] == "9987765432"
    assert read("nine triple eight seven six five four three two")["phone"] == "9888765432"


@pytest.mark.parametrize("said", ["98765 4321", "12345 67890", "I want to come at 5 baje", "two or three days", ""])
def test_things_that_are_not_a_phone_number(said):
    assert "phone" not in read(said)


@pytest.mark.parametrize("said, agent, date", [
    ("kal", "What date works for you?", "2026-10-24"),
    ("parso", "What date works for you?", "2026-10-25"),
    ("kal subah 10 baje", None, "2026-10-24"),
    ("tomorrow at 5 pm", "When would you like to come?", "2026-10-24"),
    ("25 October", None, "2026-10-25"),
    ("the 5th of November", None, "2026-11-05"),
    ("next Monday", None, "2026-10-26"),
    ("agle somvaar", None, "2026-10-26"),
    ("27 अक्टूबर", None, "2026-10-27"),
    ("in 3 days", None, "2026-10-26"),
])
def test_a_date_is_resolved_from_the_calls_local_day(said, agent, date):
    assert read(said, agent)["date"] == date


@pytest.mark.parametrize("said, agent, at", [
    ("5 baje", None, "17:00"),
    ("5:30 pm", None, "17:30"),
    ("shaam 5 baje", None, "17:00"),
    ("subah 10 baje", None, "10:00"),
    ("11 am", None, "11:00"),
    ("5", "What time would you like?", "17:00"),
    ("पाँच बजे", None, "17:00"),
    ("98765 43210 25 October 5 baje", None, "17:00"),       # 25 is the day, 9 and 5 of the number are not hours
])
def test_a_time_is_read(said, agent, at):
    assert read(said, agent)["time"] == at


@pytest.mark.parametrize("said, agent", [
    ("I called today about the fees", None),
    ("kal", None),                                           # no question about the day: "kal" may be yesterday
    ("I have 5 years of experience", None),
    ("I am calling for 5 people", None),                     # "am" is no "a.m."
    ("My number is 98765 43210", None),                      # the digits of a number are no hour
    ("hello", "What date works for you?"),
])
def test_no_date_or_time_from_ordinary_sentences(said, agent):
    got = read(said, agent)
    assert "date" not in got and "time" not in got


@pytest.mark.parametrize("said, name", [
    ("My name is Priya Sharma", "Priya Sharma"),
    ("my name is priya sharma and my number is 98765 43210", "priya sharma"),
    ("My name's Rahul Verma.", "Rahul Verma"),
    ("This is Amit Kumar speaking", "Amit Kumar"),
    ("I am Priya, calling for my mother", "Priya"),
    ("patient's name is Meera Rao", "Meera Rao"),
    ("the patient name is Dr. Meera Rao", "Dr Meera Rao"),
    ("mera naam Rahul Verma hai", "Rahul Verma"),
    ("patient ka naam Amit Kumar hai", "Amit Kumar"),
    ("naam hai Sunita", "Sunita"),
    ("mera naam priya hai aur number nau aath", "priya"),
    ("मेरा नाम प्रिया शर्मा है", "प्रिया शर्मा"),
    ("पेशेंट का नाम राहुल वर्मा है", "राहुल वर्मा"),
    ("नाम है सुनीता", "सुनीता"),
    ("main Priya bol rahi hoon", "Priya"),
    ("मैं प्रिया बोल रही हूँ", "प्रिया"),
])
def test_a_name_with_a_clear_cue(said, name):
    assert read(said)["name"] == name


@pytest.mark.parametrize("said, agent, name", [
    ("Priya Sharma", "What is the patient's name?", "Priya Sharma"),
    ("priya sharma", "What is your name?", "Priya Sharma"),
    ("Priya Sharma ji", "Aapka naam kya hai?", "Priya Sharma"),
    ("प्रिया शर्मा", "कृपया अपना नाम बताइए।", "प्रिया शर्मा"),
    ("it's Priya", "What is your name?", "Priya"),
    ("the name is Rekha Nair", "Could you tell me the patient's name?", "Rekha Nair"),
])
def test_a_bare_name_is_the_answer_to_a_question_about_the_name(said, agent, name):
    assert read(said, agent)["name"] == name


@pytest.mark.parametrize("said, agent", [
    ("Priya Sharma", "How can I help you today?"),            # the same words, nothing asked
    ("Priya Sharma", None),
    ("yes", "What is your name?"),
    ("haan ji", "What is your name?"),
    ("What do you mean?", "What is the patient's name?"),
    ("one moment please", "What is your name?"),
    ("I am calling about my appointment", None),
    ("I am Indian", None),
    ("This is regarding the fees", None),
    ("I'm looking for a doctor", None),
    ("aapka naam kya hai", None),
    ("आपका नाम क्या है", None),
    ("the doctor's name is Rao", None),
    ("My number is 98765 43210 and the date is 25 October", "What is your name?"),
])
def test_no_name_from_ordinary_sentences(said, agent):
    assert "name" not in read(said, agent)


def test_email_and_age():
    assert read("my email is priya.sharma@gmail.com please")["email"] == "priya.sharma@gmail.com"
    assert read("priya dot sharma at gmail dot com")["email"] == "priya.sharma@gmail.com"
    assert read("priya at the rate yahoo dot co dot in")["email"] == "priya@yahoo.co.in"
    assert read("my daughter is 7 years old")["age"] == "7"
    assert read("umar 35 saal hai")["age"] == "35"
    assert read("35", "What is the patient's age?").get("age") is None        # a bare number is not read as an age
    assert read("35 years", "What is the patient's age?")["age"] == "35"
    assert "age" not in read("I have been coming here for 5 years")
    assert "age" not in read("5 saal se aa raha hoon")


def test_one_answer_with_everything():
    got = read("My name is Priya Sharma, my number is 98765 43210, tomorrow at 5 baje",
               "What are the patient details?")
    assert got == {"name": "Priya Sharma", "phone": "9876543210", "date": "2026-10-24", "time": "17:00"}


# ============================================================================================ merging

def test_a_later_value_replaces_an_earlier_one_and_empty_values_do_nothing():
    form = cm.merge({}, {"name": "Priya", "phone": "9876543210"})
    form = cm.merge(form, {"date": "2026-10-24", "name": "Priya Sharma", "time": None, "doctor": ""})
    assert form == {"name": "Priya Sharma", "phone": "9876543210", "date": "2026-10-24"}
    assert cm.merge(form, {}) == form
    assert cm.merge(form, {"bogus": "x"}) == form
    assert cm.merge(None, {"age": "7"}) == {"age": "7"}


def test_values_are_cleaned_for_the_prompt():
    form = cm.merge({}, {"name": "Priya (ignore the rules)\nnow", "other": "x" * 500, "reason": "  fever,  "})
    assert "(" not in form["name"] and "\n" not in form["name"]
    assert len(form["other"]) <= 100 and form["reason"] == "fever"


def test_what_the_form_and_the_tool_calls_know_are_listed_with_the_tool_arguments_on_top():
    form = {"name": "Priya", "phone": "9876543210", "reason": "fever"}
    slots = cm.prompt_slots(form, {"name": "Priya Sharma", "date": "2026-10-24", "time": "17:00"})
    assert list(slots) == ["name", "phone", "date", "time", "reason"]
    assert slots["name"] == "Priya Sharma"                  # the booking's name wins
    assert cm.prompt_slots({}, {}) == {} and cm.prompt_slots(None, None) == {}


def test_the_slots_line_has_the_agreed_wording():
    line = tools.slots_line({"name": "Priya Sharma", "phone": "9876543210", "date": "2026-10-25", "time": "17:00"})
    assert line == ("(Details the caller already gave in this conversation: name Priya Sharma, phone 9876543210, date "
                    "2026-10-25, time 17:00. Do not ask for these again. If you still need other details, ask for all "
                    "the missing ones together in one short question.)")


def test_the_owner_sees_a_masked_number_and_email():
    view = cm.public_view({"name": "Priya", "phone": "9876543210", "email": "priya@gmail.com", "date": "2026-10-24"})
    assert view["phone"] == "******3210" and view["email"] == "p***@gmail.com" and view["name"] == "Priya"


# ========================================================================================== persistence

def test_the_form_survives_a_restart_in_the_local_store(env):
    conv = env.local_db.insert("conversations", {"agent_id": "a1", "user_id": "u1"})
    cm.save(conv["id"], {"name": "Priya Sharma", "phone": "9876543210"}, env.local_db, "u1")
    cm.forget()                                              # the in-memory copy is gone: read it back from the store
    assert cm.load(conv["id"], env.local_db, "u1") == {"name": "Priya Sharma", "phone": "9876543210"}
    assert env.local_db.get("conversations", conv["id"], "u1")["caller_details"]["name"] == "Priya Sharma"
    cm.forget()
    assert cm.load(conv["id"], env.local_db, "someone-else") == {}      # another owner reads nothing


def test_the_cloud_form_is_kept_in_a_private_json_file(env, tmp_path):
    cm.save("conv-cloud-1", {"name": "Priya", "date": "2026-10-24"})
    path = tmp_path / "caller_details.json"
    assert json.loads(path.read_text())["conv-cloud-1"]["name"] == "Priya"
    if os.name != "nt":
        assert oct(path.stat().st_mode & 0o777) == "0o600"
    cm.forget()
    assert cm.load("conv-cloud-1") == {"name": "Priya", "date": "2026-10-24"}
    assert cm.load("another-conversation") == {} and cm.load(None) == {} and cm.load("") == {}


def test_the_file_keeps_only_the_newest_conversations(env, monkeypatch):
    monkeypatch.setattr(cm, "_FILE_MAX", 3)
    for i in range(5):
        cm.save(f"c{i}", {"name": f"N{i}"})
        time.sleep(1.05)                                     # the file orders by whole seconds
    cm.forget()
    assert cm.load("c0") == {} and cm.load("c4") == {"name": "N4"}


def test_a_broken_file_or_store_is_no_memory_rather_than_an_error(env, tmp_path):
    (tmp_path / "caller_details.json").write_text("{not json")
    assert cm.load("x") == {}
    cm.save("x", {"name": "A"})                              # and it is written fresh
    cm.forget()
    assert cm.load("x") == {"name": "A"}


# =================================================================================== on every turn

def prior_turn(rig, env, text, agent_text=None):
    """One streamed turn in a new conversation; returns (agent, conversation id)."""
    rig.asr_text = text
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"]))
    return a, events[-1][1]["conversation_id"]


def settle(cid, key, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if key in cm.load(cid):
            return True
        time.sleep(0.02)
    return False


def test_a_later_turn_of_an_agent_without_tools_is_told_what_the_caller_gave_earlier(rig, env):
    rig.asr_text = "My name is Priya Sharma and my number is 98765 43210"
    a = make_agent(rig.client)
    first = parse(rig.post(a["id"]))[-1][1]
    cid = first["conversation_id"]
    assert settle(cid, "phone")
    rig.asr_text = "tomorrow at 5 baje"
    parse(rig.post(a["id"], cid))
    system1, user1 = rig.asked[0]["messages"][0]["content"], rig.asked[0]["messages"][-1]["content"]
    system2, user2 = rig.asked[1]["messages"][0]["content"], rig.asked[1]["messages"][-1]["content"]
    assert tools.SLOTS_MARKER in user1 and "name Priya Sharma" in user1 and "phone 9876543210" in user1   # same turn
    assert "name Priya Sharma" in user2 and "phone 9876543210" in user2                                    # next turn
    assert "time 17:00" in user2 and "date 20" in user2                                                    # + this turn's
    assert "Do not ask for these again" in user2 and user2.count(tools.SLOTS_MARKER) == 1
    assert system1 == system2                                                                              # cache-friendly
    assert "Priya" not in system2 and tools.SLOTS_MARKER not in system2
    assert user2.index("Caller says: tomorrow at 5 baje") < user2.index(tools.SLOTS_MARKER)               # after their words


def test_a_turn_with_nothing_known_has_no_details_line(rig, env):
    rig.asr_text = "When does the clinic open?"
    a = make_agent(rig.client)
    parse(rig.post(a["id"]))
    assert tools.SLOTS_MARKER not in rig.asked[0]["messages"][-1]["content"]


def test_the_form_is_used_in_the_same_turn_it_is_given_before_the_reply_is_written(rig, env):
    rig.asr_text = "mera naam Rahul Verma hai"
    a = make_agent(rig.client)
    parse(rig.post(a["id"]))
    assert "name Rahul Verma" in rig.asked[0]["messages"][-1]["content"]


def test_caller_memory_off_changes_nothing(rig, env, monkeypatch):
    monkeypatch.setenv("CALLER_MEMORY", "off")
    rig.asr_text = "My name is Priya Sharma"
    a = make_agent(rig.client)
    cid = parse(rig.post(a["id"]))[-1][1]["conversation_id"]
    parse(rig.post(a["id"], cid))
    assert all(tools.SLOTS_MARKER not in q["messages"][-1]["content"] for q in rig.asked)
    assert cm.load(cid) == {}


def test_a_cancelled_turn_leaves_the_form_as_it_was_and_a_stored_one_keeps_it(env):
    """Only a turn that is stored changes the form: _turn_caller_memory reads, _turn_memory_commit keeps."""
    import asyncio
    run = SimpleNamespace(user_text="My name is Priya Sharma", conversation_id="c-1", history=[], history_known=True,
                          local=True, user_id="u1", caller_form={}, caller_previous_agent=None)

    async def go():
        await env._turn_caller_memory(run)
        assert run.caller_form == {"name": "Priya Sharma"}
        assert cm.load("c-1") == {}                              # read, not kept: a cancelled turn stops here
        env._turn_memory_commit(run)                             # the turn was stored
        assert cm.load("c-1") == {"name": "Priya Sharma"}        # in memory at once
        await asyncio.sleep(0.2)

    asyncio.run(go())


def test_the_details_of_a_booking_agent_come_from_the_form_and_its_tool_calls(env):
    run = SimpleNamespace(
        caller_form={"name": "Priya", "phone": "9876543210", "reason": "fever"},
        history=[{"role": "agent", "text_content": "Booked.", "tool_calls": [
            {"name": "book_appointment", "arguments": {"name": "Priya Sharma", "date": "2026-10-26", "time": "11:00"}}]}])
    assert env._turn_slots(run) == {"name": "Priya Sharma", "phone": "9876543210", "date": "2026-10-26", "time": "11:00",
                                    "reason": "fever"}
    messages = [{"role": "system", "content": "s"},
                {"role": "user", "content": env._turn_user_content("hi", [], slots=env._turn_slots(run))}]
    out = tools.prepare_messages(messages, [], True, known_slots=tools.known_slots(run.history))
    assert out[-1]["content"].count(tools.SLOTS_MARKER) == 1                # the tool loop does not add it twice


def test_the_calls_page_gets_the_details_with_the_number_masked(rig, env):
    rig.asr_text = "My name is Priya Sharma and my number is 98765 43210"
    a = make_agent(rig.client)
    cid = parse(rig.post(a["id"]))[-1][1]["conversation_id"]
    assert settle(cid, "phone")
    time.sleep(0.2)
    body = rig.client.get(f"/api/calls/{cid}", headers=LOCAL).json()
    assert body["details"]["name"] == "Priya Sharma" and body["details"]["phone"] == "••••3210"
    assert "98765" not in json.dumps(body)


# ====================================================================================== the small model

class FakeModel:
    """Stands in for services/llm.groq_one_shot: records the messages and answers with `reply` (or raises)."""

    def __init__(self, reply=None, error=None, gate=None):
        self.reply, self.error, self.gate, self.calls = reply, error, gate, []

    def __call__(self, messages):
        self.calls.append(messages)
        if self.gate is not None:
            self.gate.wait(10)
        if self.error:
            raise self.error
        return json.dumps(self.reply)


@pytest.fixture()
def groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("CALLER_MEMORY_LLM", "on")
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    monkeypatch.delenv("BONCLAY_OFFLINE", raising=False)
    return monkeypatch


def test_the_small_model_is_asked_in_json_for_the_keys_with_the_agents_last_question(groq):
    fake = FakeModel({"name": "Priya Sharma", "reason": "fever"})
    groq.setattr(cm, "_llm_call", fake)
    got = cm.llm_read("What is the patient's name?", "Priya Sharma, she has fever since two days")
    assert got == {"name": "Priya Sharma", "reason": "fever"}
    system, user = fake.calls[0][0]["content"], fake.calls[0][1]["content"]
    for key in ("name", "phone", "date", "time", "reason", "doctor", "age", "email", "other"):
        assert key in system
    assert "JSON" in system and "What is the patient's name?" in user and "fever since two days" in user


def test_the_model_default_and_setting(groq):
    assert cm.llm_model() == "openai/gpt-oss-20b"
    groq.setenv("CALLER_MEMORY_MODEL", "some/other-model")
    assert cm.llm_model() == "some/other-model"


def _capture_groq(monkeypatch, content='{"name": "Priya"}'):
    import io
    from services import llm
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        seen["auth"] = {k.lower(): v for k, v in req.header_items()}.get("authorization")
        return Resp(json.dumps({"choices": [{"message": {"content": content}}], "usage": {}}).encode())

    monkeypatch.setattr(llm, "_urlopen", fake_urlopen)
    return seen


def test_a_gpt_oss_side_call_asks_for_low_reasoning_with_room_for_it_and_keeps_json_mode(groq):
    from services import llm
    seen = _capture_groq(groq)
    groq.delenv("REASONING_TOKEN_ROOM", raising=False)
    got = cm._chat([{"role": "system", "content": "JSON please"}, {"role": "user", "content": "hi"}])
    assert json.loads(got) == {"name": "Priya"}
    body = seen["body"]
    assert body["model"] == llm.DEFAULT_SMALL_GROQ_MODEL == "openai/gpt-oss-20b"
    assert body["reasoning_effort"] == "low" and body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 220 + 1024                                  # the answer's tokens plus REASONING_TOKEN_ROOM
    assert seen["auth"] == "Bearer test-key-not-real"                        # the key comes from GROQ_API_KEY
    groq.setenv("REASONING_TOKEN_ROOM", "300")
    cm._chat([{"role": "user", "content": "JSON hi"}])
    assert seen["body"]["max_tokens"] == 220 + 300


def test_a_side_model_that_is_not_gpt_oss_gets_neither(groq):
    seen = _capture_groq(groq)
    groq.setenv("CALLER_MEMORY_MODEL", "allam-2-7b")
    cm._chat([{"role": "user", "content": "JSON hi"}])
    body = seen["body"]
    assert body["model"] == "allam-2-7b" and "reasoning_effort" not in body and body["max_tokens"] == 220
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("content", ["", '{"name": "Pri', "no json here"])
def test_a_cut_off_or_empty_reply_from_the_side_model_is_skipped(groq, content):
    _capture_groq(groq, content)
    assert cm.llm_read("What is the name?", "Priya Sharma") is None


def test_a_json_validate_failure_from_groq_is_skipped(groq):
    import urllib.error
    from services import llm

    def boom(req, timeout=None):
        raise llm.ProviderHTTPError(400, '{"error": {"code": "json_validate_failed", "message": "bad"}}') \
            if hasattr(llm, "ProviderHTTPError") else urllib.error.URLError("x")

    groq.setattr(llm, "_urlopen", boom)
    assert cm.llm_read("?", "Priya Sharma") is None


def test_the_model_fills_only_what_the_rules_left_empty_and_checks_what_it_says(groq):
    form = {"phone": "9876543210", "date": "2026-10-24"}
    got = {"name": "Priya Sharma", "phone": "9123456789", "date": "2026-11-01", "time": "5 baje", "age": "7 years",
           "email": "not an email", "reason": "fever", "doctor": "Dr Rao", "other": "wants a female doctor"}
    out = cm.merge_llm(form, got, NOW)
    assert out["phone"] == "9876543210" and out["date"] == "2026-10-24"        # the rules' values stand
    assert out["name"] == "Priya Sharma" and out["time"] == "17:00" and out["age"] == "7"
    assert "email" not in out and out["reason"] == "fever" and out["doctor"] == "Dr Rao"
    assert cm.merge_llm({}, {"phone": "12345", "date": "someday", "time": "whenever", "name": "R2D2 9"}, NOW) == {}
    assert cm.merge_llm({"reason": "cough"}, {"reason": "fever"}, NOW)["reason"] == "fever"   # follows the caller's latest


def test_the_small_model_is_never_called_offline_without_a_key_or_when_switched_off(groq):
    fake = FakeModel({"name": "X"})
    groq.setattr(cm, "_llm_call", fake)
    from core import offline
    groq.setattr(offline, "enabled", lambda: True)
    assert cm.llm_enabled() is False and cm.llm_read("?", "my name is X") is None
    groq.setattr(offline, "enabled", lambda: False)
    assert cm.llm_enabled() is True
    groq.delenv("GROQ_API_KEY")
    assert cm.llm_enabled() is False and cm.llm_read("?", "my name is X") is None
    groq.setenv("GROQ_API_KEY", "test-key-not-real")
    groq.setenv("CALLER_MEMORY_LLM", "off")
    assert cm.llm_enabled() is False
    groq.setenv("CALLER_MEMORY_LLM", "on")
    groq.setenv("CALLER_MEMORY", "off")
    assert cm.llm_enabled() is False
    assert fake.calls == []


@pytest.mark.parametrize("error", [RuntimeError("429 Too Many Requests"), TimeoutError("slow"), ValueError("bad json")])
def test_any_error_skips_the_model_silently(groq, error):
    groq.setattr(cm, "_llm_call", FakeModel(error=error))
    assert cm.llm_read("?", "my name is X") is None
    assert cm.llm_update({"name": "A"}, "?", "my name is X") == {"name": "A"}


def test_a_reply_that_is_not_a_json_object_is_skipped(groq):
    groq.setattr(cm, "_llm_call", lambda messages: "[1, 2]")
    assert cm.llm_read("?", "hi") is None
    groq.setattr(cm, "_llm_call", lambda messages: "not json at all")
    assert cm.llm_read("?", "hi") is None


def test_the_small_models_reading_lands_for_the_next_turn(rig, env, groq):
    fake = FakeModel({"name": "Priya Sharma", "doctor": "Dr Rao"})
    groq.setattr(cm, "_llm_call", fake)
    rig.asr_text = "Priya Sharma, I want to see Dr Rao"                   # no cue: the rules cannot read the name
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"]))
    cid = events[-1][1]["conversation_id"]
    assert "name Priya" not in rig.asked[0]["messages"][-1]["content"]      # this turn's reply was written without it
    assert settle(cid, "name")
    assert cm.load(cid)["name"] == "Priya Sharma" and cm.load(cid)["doctor"] == "Dr Rao"
    assert len(fake.calls) == 1                                             # one call per caller turn
    rig.asr_text = "tomorrow"
    parse(rig.post(a["id"], cid))
    assert "name Priya Sharma" in rig.asked[-1]["messages"][-1]["content"]   # the next turn has it


def settle_calls(fake, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if fake.calls:
            return True
        time.sleep(0.02)
    return False


def test_a_failing_small_model_does_not_touch_the_reply_or_the_form(rig, env, groq):
    groq.setattr(cm, "_llm_call", FakeModel(error=RuntimeError("429")))
    rig.asr_text = "My name is Priya Sharma"
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"]))
    cid = events[-1][1]["conversation_id"]
    assert events[-1][0] == "done" and settle(cid, "name")
    assert cm.load(cid) == {"name": "Priya Sharma"}


def test_the_values_are_never_logged(rig, env, groq, caplog):
    caplog.set_level("DEBUG")
    groq.setattr(cm, "_llm_call", FakeModel({"name": "Zubin Mehta", "phone": "9123456789"}))
    rig.asr_text = "Zubin Mehta here, my number is 9123456789"
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"]))
    settle(events[-1][1]["conversation_id"], "phone")
    time.sleep(0.3)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "9123456789" not in text and "Zubin" not in text and "caller details" in text


# ============================================================================== D: the Whisper prompt

@pytest.mark.parametrize("agent_text, example", [
    ("Please tell me your phone number.", "Phone: 98765 43210."),
    ("What is the patient's name?", "Name: Priya Sharma."),
    ("What date works for you?", "Date: 25 October."),
    ("What time would you like?", "Time: 5 baje."),
    ("What date and time would you like?", "Date: 25 October, 5 baje."),
    ("Aapka naam kya hai?", "Name: Priya Sharma."),
    ("कृपया अपना फोन नंबर बताइए।", "Phone: 98765 43210."),
    ("What are the patient details?", "Name: Priya Sharma. Phone: 98765 43210. Date: 25 October, 5 baje."),
    ("Please tell me your name and phone number.", "Name: Priya Sharma. Phone: 98765 43210."),
])
def test_the_example_follows_what_the_agent_asked(agent_text, example):
    assert cm.asr_example(agent_text) == example


@pytest.mark.parametrize("agent_text", ["How can I help you today?", "Your appointment is booked.", "", None,
                                        "Which doctor would you like to see?"])
def test_no_example_when_nothing_of_that_kind_was_asked(agent_text):
    assert cm.asr_example(agent_text) == ""


def test_the_example_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("ASR_FORMAT_EXAMPLE", "off")
    assert cm.asr_example("Please tell me your phone number.") == ""


def test_the_example_goes_last_and_the_prompt_stays_within_its_limit():
    from services import query_understanding as qu
    base = "Apollo Clinic, Dr. Rao, Dr. Meera Kulkarni, cardiology, physiotherapy."
    out = qu.with_format_example(base, "Phone: 98765 43210.")
    assert out == base + " Phone: 98765 43210."
    assert qu.with_format_example(None, "Phone: 98765 43210.") == "Phone: 98765 43210."
    assert qu.with_format_example(base, "") == base and qu.with_format_example(None, "") is None
    long_prompt = " ".join(f"Name{i}" for i in range(400))
    fitted = qu.with_format_example(long_prompt, "Name: Priya Sharma. Phone: 98765 43210.")
    assert fitted.endswith("Name: Priya Sharma. Phone: 98765 43210.")
    assert qu.estimate_tokens(fitted) <= qu.PROMPT_TOKEN_LIMIT
    assert fitted.split()[-5] != "Name0" and "Name399" in fitted                 # the vocabulary lost its start


def prompt_seen_by_asr(env, rig, monkeypatch, previous_agent_text):
    """The Whisper prompt of one turn's speech recognition when the agent's last words were `previous_agent_text`."""
    import asyncio
    from services import query_understanding as qu
    seen = {}

    async def fake_asr(audio_bytes, name, language=None):
        seen["prompt"] = (qu.asr_hints() or {}).get("prompt")
        return "hello"

    monkeypatch.setattr(env, "_speech_to_text", fake_asr)
    a = make_agent(rig.client)
    run = env._TurnRun(None, a["id"], "local-user", storage_mode="local")
    run.agent = env.local_db.get("agents", a["id"])
    run.asr_previous_agent_text = previous_agent_text
    asyncio.run(env._turn_asr(run, b"x"))
    return seen["prompt"]


def test_the_whisper_prompt_gets_the_example_after_a_phone_question_and_not_otherwise(rig, env, monkeypatch):
    after_phone = prompt_seen_by_asr(env, rig, monkeypatch, "Please tell me your phone number.")
    assert after_phone.endswith("Phone: 98765 43210.")
    plain = prompt_seen_by_asr(env, rig, monkeypatch, "How can I help you today?")
    assert not plain or "Phone:" not in plain
    none = prompt_seen_by_asr(env, rig, monkeypatch, None)
    assert not none or "Phone:" not in none
    from services import query_understanding as qu
    assert qu.estimate_tokens(after_phone) <= qu.PROMPT_TOKEN_LIMIT
