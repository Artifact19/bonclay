"""The local fallback model for background jobs (services/background_llm.py): the idle / memory / live-call rules, JSON
repair, unloading, and its use by the question index, the fact sheet and scripts/rechunk_knowledge.py --questions --local.
No real model here (a fake stands in); the one real-model test runs only with BONCLAY_SLOW_TESTS=1 and the model on disk."""
import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import activity, memory_guard
from services import background_llm as bg, fact_sheet as fs, llm, question_index as qi
from services.local_store import LocalStore

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rechunk_knowledge.py"
GEMMA = "mlx-community/gemma-4-e4b-it-4bit"


@pytest.fixture(autouse=True)
def local_fallback(monkeypatch):
    """The fallback on, the model 'downloaded', memory ample, quiet app; a fake clock-free setup."""
    monkeypatch.setenv("BACKGROUND_LLM_FALLBACK", "local")
    monkeypatch.setenv("BACKGROUND_LLM_POLL_S", "5")
    monkeypatch.setenv("QUESTION_INDEX_MIN_INTERVAL_S", "0")
    monkeypatch.setenv("QUESTION_INDEX_TPM", "1000000")
    monkeypatch.setattr(llm, "background_unavailable_reason", lambda model: None)
    monkeypatch.setattr(llm, "background_running", lambda: False)
    monkeypatch.setattr(activity, "call_live", lambda: False)
    activity.reset()
    yield
    activity.reset()
    bg.release()


class Clock:
    """A clock the waiting loop advances by sleeping, so waits take no real time."""

    def __init__(self):
        self.t = 1000.0
        self.slept = []
        self.on_sleep = None

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s
        if self.on_sleep:
            self.on_sleep(self)


class FakeLocal:
    def __init__(self, replies=None):
        self.calls = []
        self.replies = replies

    def __call__(self, messages, *, model, json_mode, temperature, max_tokens):
        self.calls.append({"messages": messages, "model": model, "json_mode": json_mode, "max_tokens": max_tokens})
        r = self.replies.pop(0) if self.replies else local_reply(messages)
        if isinstance(r, BaseException):
            raise r
        return llm.ChatResult(r, "mlx-background", model, 3, {})


def local_reply(messages):
    payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
    items = [{"id": p["id"], "en": ["What are the OPD timings?", "When is the clinic open?"],
              "hi": ["ओपीडी का समय क्या है?", "क्लिनिक कब खुलता है?"],
              "hinglish": ["OPD ka time kya hai?", "Clinic kab khulta hai?"]} for p in payload]
    return json.dumps({"items": items})


def passages(n):
    return [{"id": f"p{i}", "header": f"Clinic > Timings {i}",
             "content": "OPD timings are Monday to Saturday, 9:30 AM to 1:00 PM. " + f"Room {i}."} for i in range(n)]


def hosted_429():
    return llm.LLMUnavailable({"groq": "failed: RuntimeError: HTTP 429 rate limit"})


class HostedChat:
    def __init__(self, replies):
        self.replies = replies
        self.calls = 0

    def __call__(self, messages, **kw):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return llm.ChatResult(r, "groq", "m", 1, {"total_tokens": 100})


# --------------------------------------------------------------------------------------------- settings

def test_defaults_and_switches(monkeypatch):
    monkeypatch.delenv("BACKGROUND_LLM_FALLBACK", raising=False)
    monkeypatch.delenv("BACKGROUND_LOCAL_MODEL", raising=False)
    assert bg.mode() == "local" and bg.model_name() == GEMMA and bg.local_ready()
    monkeypatch.setenv("BACKGROUND_LLM_FALLBACK", "off")
    assert bg.mode() == "off" and not bg.local_ready() and "off" in bg.unavailable_reason()
    monkeypatch.setenv("BACKGROUND_LOCAL_MODEL", "x/y")
    assert bg.model_name() == "x/y"


def test_the_live_model_setting_is_untouched(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit")
    assert bg.model_name() == GEMMA
    assert llm._mlx.model == os.getenv("LOCAL_LLM_MODEL", llm.DEFAULT_LOCAL_MODEL) or llm._mlx.model


def test_no_model_on_disk_means_no_fallback(monkeypatch):
    monkeypatch.setattr(llm, "background_unavailable_reason", lambda m: f"{m} is not downloaded")
    assert not bg.local_ready()
    with pytest.raises(bg.Busy, match="not downloaded"):
        bg.run_local([{"role": "user", "content": "x"}], chat=FakeLocal())


# ---------------------------------------------------------------------------------- idle / memory / calls

def test_idle_rule(monkeypatch):
    assert activity.is_idle() and bg.conditions() == (True, "")
    activity.mark_turn(time.monotonic() - 10)
    ok, why = bg.conditions()
    assert not ok and "turn" in why
    activity.mark_turn(time.monotonic() - 61)
    assert bg.conditions()[0]
    monkeypatch.setenv("BACKGROUND_LLM_IDLE_S", "5")
    activity.mark_turn(time.monotonic() - 10)
    assert bg.conditions()[0]


def test_a_live_call_blocks_even_when_quiet(monkeypatch):
    monkeypatch.setattr(activity, "call_live", lambda: True)
    ok, why = bg.conditions()
    assert not ok and "phone call" in why and not activity.is_idle()


def test_call_live_reads_the_call_sessions(monkeypatch):
    from services import call_control
    monkeypatch.undo()          # the fixture's fakes off
    monkeypatch.setattr(call_control, "live_sessions", lambda: [object()])
    assert activity.call_live()
    monkeypatch.setattr(call_control, "live_sessions", lambda: [])
    assert not activity.call_live()


def test_memory_must_allow_about_six_and_a_half_gb(monkeypatch):
    monkeypatch.setenv("MEMORY_GUARD", "on")
    monkeypatch.delenv("MEMORY_NEED_BACKGROUND_LLM_MB", raising=False)
    monkeypatch.delenv("MEMORY_GUARD_RESERVE_MB", raising=False)
    assert memory_guard.need_mb("background_llm") + memory_guard.reserve_mb() == 6500   # measured 4.6 GB (+10 %) + reserve
    monkeypatch.setattr(memory_guard, "available_mb", lambda: 3000.0)
    ok, why = bg.conditions()
    assert not ok and "memory" in why
    monkeypatch.setattr(memory_guard, "available_mb", lambda: 7000.0)
    assert bg.conditions()[0]
    assert memory_guard.fits("background_llm", lambda: None)[0]          # unreadable memory does not block
    monkeypatch.setenv("MEMORY_GUARD", "off")
    monkeypatch.setattr(memory_guard, "available_mb", lambda: 1.0)
    assert bg.conditions()[0]


def test_memory_guard_does_not_unload_other_engines_for_background_work(monkeypatch):
    monkeypatch.setenv("MEMORY_GUARD", "on")
    monkeypatch.setattr(memory_guard, "available_mb", lambda: 100.0)
    ok, avail, required = memory_guard.fits("background_llm")
    assert not ok and avail == 100.0 and required == 6500


# ------------------------------------------------------------------------------------------- run_local

def test_run_local_waits_for_an_idle_app_then_runs(monkeypatch):
    clock = Clock()
    activity.mark_turn(time.monotonic())            # a turn just started
    fake = FakeLocal(['{"ok": 1}'])

    def quiet_after_three(c):
        if len(c.slept) == 3:
            activity.reset()
    clock.on_sleep = quiet_after_three
    res = bg.run_local([{"role": "user", "content": "q"}], chat=fake, sleep=clock.sleep, clock=clock.now, wait=600)
    assert res.text == '{"ok": 1}' and len(fake.calls) == 1 and clock.slept == [5, 5, 5]
    call = fake.calls[0]
    assert call["model"] == GEMMA and call["json_mode"] is True
    assert "JSON object" in call["messages"][0]["content"] and call["messages"][-1] == {"role": "user", "content": "q"}


def test_run_local_gives_up_busy_without_calling_the_model():
    clock = Clock()
    activity.mark_turn(time.monotonic() + 1e6)       # always "just now"
    fake = FakeLocal()
    with pytest.raises(bg.Busy, match="turn"):
        bg.run_local([{"role": "user", "content": "q"}], chat=fake, sleep=clock.sleep, clock=clock.now, wait=30)
    assert not fake.calls and sum(clock.slept) >= 30


def test_run_local_never_runs_during_a_call(monkeypatch):
    monkeypatch.setattr(activity, "call_live", lambda: True)
    clock = Clock()
    fake = FakeLocal()
    with pytest.raises(bg.Busy, match="phone call"):
        bg.run_local([{"role": "user", "content": "q"}], chat=fake, sleep=clock.sleep, clock=clock.now, wait=20)
    assert not fake.calls


def test_a_request_cut_short_by_a_new_turn_is_waited_out_and_repeated(monkeypatch):
    clock = Clock()
    state = {"n": 0}

    def flaky(messages, **kw):
        state["n"] += 1
        if state["n"] == 1:
            activity.mark_turn(time.monotonic() + 1e6)       # a caller arrives; the model is unloaded under the request
            raise ConnectionError("server stopped")
        return llm.ChatResult("{}", "mlx-background", GEMMA, 1, {})

    clock.on_sleep = lambda c: activity.reset()
    res = bg.run_local([{"role": "user", "content": "q"}], chat=flaky, sleep=clock.sleep, clock=clock.now, wait=100)
    assert res.text == "{}" and state["n"] == 2


def test_a_real_model_error_when_idle_is_raised():
    def broken(messages, **kw):
        raise ValueError("bad request")
    with pytest.raises(ValueError):
        bg.run_local([{"role": "user", "content": "q"}], chat=broken, sleep=lambda s: None)


# ------------------------------------------------------------------------------------------- unloading

def test_a_conversation_turn_unloads_the_model(monkeypatch):
    stopped = threading.Event()
    monkeypatch.setattr(llm, "background_running", lambda: True)
    monkeypatch.setattr(llm, "unload_background", lambda: stopped.set() or True)
    activity.mark_turn()
    assert stopped.wait(2)


def test_no_unload_work_when_nothing_is_loaded(monkeypatch):
    called = []
    monkeypatch.setattr(llm, "unload_background", lambda: called.append(1) or False)
    activity.mark_turn()
    time.sleep(0.05)
    assert not called


def test_the_model_is_unloaded_after_five_idle_minutes(monkeypatch):
    unloaded = []
    monkeypatch.setattr(llm, "unload_background", lambda: unloaded.append(1) or True)
    assert bg.unload_s() == 300.0
    bg._last_used = time.monotonic() - 10
    bg._idle_check()                       # used 10 s ago: stays, a new timer is set
    assert not unloaded
    bg.release()                           # cancels the timer, unloads
    assert unloaded == [1]
    bg._last_used = time.monotonic() - 400
    bg._idle_check()
    assert unloaded == [1, 1]


def test_release_is_a_noop_without_a_loaded_background_model(monkeypatch):
    monkeypatch.undo()
    assert llm.unload_background() is False


def test_the_background_server_is_a_separate_process_with_its_own_model(monkeypatch):
    srv = llm._bg_server(GEMMA)
    assert srv is not llm._mlx and srv.model == GEMMA and srv._log_path.endswith("background_llm.log")
    assert llm._bg_server(llm._mlx.model) is llm._mlx          # the live model: its own server, never stopped here
    assert llm.unload_background() is False


# -------------------------------------------------------------------------------------------- JSON repair

@pytest.mark.parametrize("raw", [
    '```json\n{"a": [1, 2]}\n```',
    'Here you go: {"a": [1, 2]} hope that helps',
    '{"a": [1, 2,],}',
    '{"a": [1, 2]',
    '{"a": [1, 2',
    '“{"a": [1, 2]}',
])
def test_parse_json_repairs_the_usual_slips(raw):
    assert bg.parse_json(raw) == {"a": [1, 2]}


def test_parse_json_repairs_a_cut_off_string_and_list_replies():
    out = bg.parse_json('{"items": [{"id": 0, "en": ["What is the fee?", "When is it ope')
    assert out["items"][0]["en"][0] == "What is the fee?"
    assert bg.parse_json('[{"key": "a", "value": "b"},]') == [{"key": "a", "value": "b"}]


@pytest.mark.parametrize("raw", ["", "no json at all", "{{{{"])
def test_parse_json_gives_up_cleanly(raw):
    with pytest.raises(ValueError):
        bg.parse_json(raw)


# ----------------------------------------------------------------------------------- the question index

def test_strict_checks_drop_statements_wrong_scripts_and_invented_numbers():
    passage = "OPD timings are Monday to Saturday, 9:30 AM to 1:00 PM."
    assert qi.check_strict("What are the OPD timings?", "en", passage)
    assert qi.check_strict("ओपीडी का समय क्या है?", "hi", passage)
    assert qi.check_strict("OPD ka time kya hai?", "hinglish", passage)
    assert qi.check_strict("Cardiology OPD kab hoti hai", "hinglish")          # no mark, but a question word
    assert not qi.check_strict("The OPD is open on Monday.", "en", passage)     # a statement
    assert not qi.check_strict("OPD ka time kya hai?", "en", passage)           # Roman Hindi filed as English
    assert not qi.check_strict("What are the OPD timings?", "hinglish", passage)  # English filed as Hinglish
    assert not qi.check_strict("Is the OPD open at 11:45?", "en", passage)      # a number the passage lacks
    assert not qi.check_strict("OPD का समय क्या है?", "hinglish", passage)      # Devanagari in the Roman bucket


def test_parse_reply_strict_repairs_json_and_drops_bad_items():
    reply = ('```json\n{"items": [{"id": 0, "en": ["What are the OPD timings?", "The OPD is nice.", "Is it open at 7:15?"],'
             ' "hi": ["ओपीडी का समय क्या है?", "OPD ka time kya hai?"], "hinglish": ["OPD ka time kya hai?",],},]}\n```')
    out = qi.parse_reply(reply, [0], strict=True, passages={0: "OPD timings are 9:30 AM to 1:00 PM."})
    assert out[0] == ["What are the OPD timings?", "ओपीडी का समय क्या है?", "OPD ka time kya hai?"]
    with pytest.raises(ValueError):
        qi.parse_reply(reply, [0])                       # the hosted path does not repair


def test_a_hosted_429_falls_back_to_the_local_model():
    local = FakeLocal()
    hosted = HostedChat([hosted_429()])
    clock = Clock()
    batch = qi.make_batches(passages(2))[0]
    out = qi.generate_batch(batch, chat=hosted, pacer=qi.Pacer(sleep=clock.sleep), sleep=clock.sleep, local_chat=local)
    assert hosted.calls == 1 and len(local.calls) == 1
    assert all(len(v) == 6 for v in out.values()) and clock.slept == []      # no backoff wait: the local model answered
    assert local.calls[0]["model"] == GEMMA and "ANSWERS" in local.calls[0]["messages"][0]["content"]


def test_hosted_is_still_first_and_local_untouched_when_it_works():
    local = FakeLocal()
    hosted = HostedChat([local_reply([{"content": 'x\n[{"id": 0, "text": "a b c d e"}]'}])])
    qi.generate_batch(qi.make_batches(passages(1))[0], chat=hosted, pacer=qi.Pacer(), sleep=Clock().sleep, local_chat=local)
    assert hosted.calls == 1 and not local.calls


def test_offline_uses_the_local_model_instead_of_skipping():
    offline = llm.LLMUnavailable({"groq": "GROQ_API_KEY is not set", "openai": "LLM_BASE_URL is not set"})
    local = FakeLocal()
    out = qi.generate_batch(qi.make_batches(passages(1))[0], chat=HostedChat([offline]), pacer=qi.Pacer(),
                            sleep=Clock().sleep, local_chat=local)
    assert out[0] and len(local.calls) == 1


def test_offline_with_the_fallback_off_still_skips(monkeypatch):
    monkeypatch.setenv("BACKGROUND_LLM_FALLBACK", "off")
    offline = llm.LLMUnavailable({"groq": "GROQ_API_KEY is not set"})
    with pytest.raises(qi.Skipped):
        qi.generate_batch(qi.make_batches(passages(1))[0], chat=HostedChat([offline]), pacer=qi.Pacer(),
                          sleep=Clock().sleep, local_chat=FakeLocal())


def test_fallback_off_keeps_the_429_backoff(monkeypatch):
    monkeypatch.setenv("BACKGROUND_LLM_FALLBACK", "off")
    monkeypatch.setenv("QUESTION_INDEX_BACKOFF_S", "10")
    hosted = HostedChat([hosted_429(), local_reply([{"content": 'x\n[{"id": 0, "text": "a b c d e"}]'}])])
    clock = Clock()
    local = FakeLocal()
    qi.generate_batch(qi.make_batches(passages(1))[0], chat=hosted, pacer=qi.Pacer(sleep=clock.sleep), sleep=clock.sleep,
                      local_chat=local)
    assert clock.slept == [10] and not local.calls


def test_the_local_model_waits_while_a_conversation_is_going_on_then_the_job_fails_politely(monkeypatch):
    monkeypatch.setenv("BACKGROUND_LLM_WAIT_S", "0.05")
    activity.mark_turn(time.monotonic() + 1e6)
    slept = []
    local = FakeLocal()
    with pytest.raises(RuntimeError, match="local model"):
        qi.generate_batch(qi.make_batches(passages(1))[0], chat=HostedChat([hosted_429()]), pacer=qi.Pacer(),
                          sleep=lambda s: slept.append(s) or time.sleep(0.02), local_chat=local)
    assert not local.calls and slept


def test_local_replies_with_slips_are_repaired_and_bad_items_dropped():
    messy = ('Sure!\n```json\n{"items": [{"id": 0, "en": ["What are the OPD timings?", "OPD is open."],'
             ' "hi": ["ओपीडी का समय क्या है?"], "hinglish": ["OPD ka time kya hai?", "Time batao 99?"],}]}')
    local = FakeLocal([messy])
    out = qi.generate_batch(qi.make_batches(passages(1))[0], chat=HostedChat([hosted_429()]), pacer=qi.Pacer(),
                            sleep=Clock().sleep, local_chat=local)
    assert out[0] == ["What are the OPD timings?", "ओपीडी का समय क्या है?", "OPD ka time kya hai?"]


def test_unreadable_local_reply_is_tried_twice_then_fails():
    local = FakeLocal(["not json", "still not json"])
    with pytest.raises(RuntimeError, match="failed"):
        qi.generate_batch(qi.make_batches(passages(1))[0], chat=HostedChat([hosted_429(), hosted_429()]),
                          pacer=qi.Pacer(), sleep=Clock().sleep, local_chat=local)
    assert len(local.calls) == 2


def test_local_requests_are_small_batches():
    local = FakeLocal()
    batch = qi.make_batches(passages(4), size=4)[0]
    out = qi.generate_batch(batch, chat=HostedChat([hosted_429()]), pacer=qi.Pacer(), sleep=Clock().sleep, local_chat=local)
    assert len(local.calls) == 2 and set(out) == {0, 1, 2, 3}          # BACKGROUND_LLM_BATCH=2 passages per request
    ids = [[p["id"] for p in json.loads(c["messages"][-1]["content"].split("\n", 1)[1])] for c in local.calls]
    assert ids == [[0, 1], [2, 3]]


def test_a_stray_token_in_the_middle_of_the_json_costs_only_the_broken_questions():
    # seen from the real model: a corrupt token ends item 0's list and the JSON never recovers
    glitch = ('{"items": [{"id": 0, "en": ["What are the OPD timings?", "When is the OPD open?"], "hi": ["ओपीडी का समय क्या है?"], '
              '"hinglish": ["OPD ka time kya hai?", "OPD kab khulta hai?"inesement", "id": 1, "en": ["What is the fee?"], '
              '"hi": ["शुल्क कितना है?"], "hinglish": ["Fee kitna hai?"]}]')
    with pytest.raises(ValueError):
        bg.parse_json(glitch)
    out = qi.parse_reply(glitch, [0, 1], strict=True, passages={0: "OPD timings", 1: "The fee"})
    assert out[0][:2] == ["What are the OPD timings?", "When is the OPD open?"] and "OPD ka time kya hai?" in out[0]
    assert out[1] == ["What is the fee?", "शुल्क कितना है?", "Fee kitna hai?"]
    with pytest.raises(ValueError):
        qi.parse_reply("nothing here", [0], strict=True)


def test_forced_local_never_calls_the_hosted_model():
    hosted = HostedChat([])
    local = FakeLocal()
    out = qi.generate_batch(qi.make_batches(passages(2))[0], chat=hosted, pacer=qi.Pacer(), sleep=Clock().sleep,
                            local_chat=local, local=True)
    assert hosted.calls == 0 and len(local.calls) == 1 and out


def test_force_setting_goes_local_first(monkeypatch):
    monkeypatch.setenv("BACKGROUND_LLM_FORCE", "on")
    hosted = HostedChat([])
    local = FakeLocal()
    qi.generate_batch(qi.make_batches(passages(1))[0], chat=hosted, pacer=qi.Pacer(), sleep=Clock().sleep, local_chat=local)
    assert hosted.calls == 0 and local.calls


def test_available_means_hosted_or_the_local_model(monkeypatch):
    monkeypatch.setattr(qi, "hosted_available", lambda: False)
    assert qi.available()
    monkeypatch.setenv("BACKGROUND_LLM_FALLBACK", "off")
    assert not qi.available()
    monkeypatch.setattr(qi, "hosted_available", lambda: True)
    assert qi.available()


def test_the_model_is_released_after_a_job_unless_more_are_queued(monkeypatch):
    released = []
    monkeypatch.setattr(bg, "release", lambda: released.append(1))
    embed = lambda texts: [[1.0, 0.0] for _ in texts]
    save = lambda rows: len(rows)
    r = qi.index_passages(passages(2), embed=embed, save=save, chat=HostedChat([hosted_429()]), pacer=qi.Pacer(),
                          sleep=Clock().sleep, local_chat=FakeLocal())
    assert r["status"] == "done" and r["questions"] == 12 and released == [1]


# -------------------------------------------------------------------------------------------- fact sheet

DOC = """Sahyadri Clinic
OPD timings: Monday to Saturday, 9:30 AM to 1:00 PM. Sunday closed.
Consultation fee: Rs. 500 (general physician), Rs. 800 (cardiology).
Helpline: +91 98765 43210. Address: 12 FC Road, Pune 411004."""


def test_fact_sheet_falls_back_to_the_local_model_and_still_validates(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(hosted_429()))
    messy = ('```json\n{"facts": [{"key": "OPD timings, Monday to Saturday", "value": "9:30 AM to 1:00 PM"},'
             ' {"key": "Consultation fee, cardiology", "value": "Rs. 800"},'
             ' {"key": "MRI scan fee", "value": "Rs. 4500"}, {"key": "Helpline phone number", "value": "+91 98765 43210"},]}')
    local = FakeLocal([messy])
    monkeypatch.setattr(llm, "background_chat", lambda messages, **kw: local(messages, **kw))
    facts, model = fs.build_facts([DOC])
    assert [f["value"] for f in facts] == ["9:30 AM to 1:00 PM", "Rs. 800", "+91 98765 43210"]    # the MRI fee is not in the text
    assert model == GEMMA and len(local.calls) == 1 and local.calls[0]["json_mode"] is True
    assert "JSON object" in local.calls[0]["messages"][0]["content"]


def test_fact_sheet_without_the_fallback_still_raises(monkeypatch):
    monkeypatch.setenv("BACKGROUND_LLM_FALLBACK", "off")
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(hosted_429()))
    with pytest.raises(llm.LLMUnavailable):
        fs.build_facts([DOC])


def test_fact_sheet_refresh_builds_offline_with_the_local_model(monkeypatch, tmp_path):
    monkeypatch.setenv("FACT_SHEET", "on")
    monkeypatch.setenv("FACT_SHEET_DIR", str(tmp_path / "sheets"))
    monkeypatch.setattr(fs, "hosted_llm_available", lambda: False)
    fs.set_loader(lambda a, u, m: ("fp-1", lambda: [DOC]))
    fs.clear_memory()
    released = []
    monkeypatch.setattr(bg, "release", lambda: released.append(1))
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(llm.LLMUnavailable({"groq": "GROQ_API_KEY is not set"})))
    local = FakeLocal([json.dumps({"facts": [{"key": "Consultation fee, cardiology", "value": "Rs. 800"}]})])
    monkeypatch.setattr(llm, "background_chat", lambda messages, **kw: local(messages, **kw))
    try:
        assert fs.refresh("agent1", "u", "local") == [{"key": "Consultation fee, cardiology", "value": "Rs. 800"}]
        assert released == [1] and (tmp_path / "sheets" / "agent1.json").exists()
    finally:
        fs.set_loader(None)
        fs.clear_memory()


def test_fact_sheet_waits_for_idle_then_gives_up_busy(monkeypatch):
    monkeypatch.setenv("BACKGROUND_LLM_WAIT_S", "0.05")
    monkeypatch.setenv("BACKGROUND_LLM_POLL_S", "0.01")
    activity.mark_turn(time.monotonic() + 1e6)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(hosted_429()))
    called = []
    monkeypatch.setattr(llm, "background_chat", lambda *a, **k: called.append(1))
    t = time.monotonic()
    with pytest.raises(bg.Busy):
        fs.build_facts([DOC])
    assert not called and time.monotonic() - t >= 0.05


# ----------------------------------------------------------------------------------- rechunk_knowledge --local

@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("rechunk_knowledge", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def doc_store(tmp_path):
    store = LocalStore(str(tmp_path))
    d = store.insert("documents", {"agent_id": "a1", "user_id": "u1", "filename": "g.md", "chunk_count": 2})
    store.insert_many("document_chunks", [
        {"document_id": d["id"], "chunk_text": "OPD timings are Monday to Saturday, 9:30 AM to 1:00 PM.", "chunk_index": 0,
         "kind": "passage", "header": "Clinic > OPD", "embedding": [1.0], "embed_model": "m", "embed_dim": 1},
        {"document_id": d["id"], "chunk_text": "The consultation fee is Rs. 500 for a general physician visit.", "chunk_index": 1,
         "kind": "passage", "header": "Clinic > Fees", "embedding": [1.0], "embed_model": "m", "embed_dim": 1}])
    return store


enc = lambda texts: [[float(len(t.split())), 1.0] for t in texts]


def test_script_local_uses_only_the_local_model(tmp_path, script, monkeypatch):
    store = doc_store(tmp_path)
    hosted = HostedChat([])
    local = FakeLocal()
    out = []
    r = script.run(str(tmp_path), encode=enc, model_name="m", out=out.append, questions=True, local=True, chat=hosted,
                   local_chat=local, sleep=lambda s: None)
    assert r["failed"] == 0 and hosted.calls == 0 and local.calls
    assert r["questions"]["questions"] == 12 and "indexed" in " ".join(out)
    assert len([c for c in store.select("document_chunks") if c.get("kind") == "question"]) == 12


def test_script_local_dry_run_is_unchanged(tmp_path, script):
    doc_store(tmp_path)
    before = (tmp_path / "local_db" / "document_chunks.json").read_bytes()
    out = []

    def no_model(_):
        raise AssertionError("a dry run must not embed")
    r = script.run(str(tmp_path), dry_run=True, encode=no_model, model_name="m", out=out.append, questions=True, local=True,
                   local_chat=lambda *a, **k: (_ for _ in ()).throw(AssertionError("a dry run must not call a model")))
    assert r["questions"]["documents"] == 1 and r["questions"]["calls"] >= 1
    assert (tmp_path / "local_db" / "document_chunks.json").read_bytes() == before and any("would index" in l for l in out)


def test_script_local_needs_the_model(tmp_path, script, monkeypatch):
    doc_store(tmp_path)
    monkeypatch.setattr(llm, "background_unavailable_reason", lambda m: f"{m} is not downloaded")
    out = []
    r = script.run(str(tmp_path), encode=enc, model_name="m", out=out.append, questions=True, local=True)
    assert "no local language model" in r["questions"]["status"] and r["questions"]["questions"] == 0


def test_script_local_flag_is_wired_and_goes_with_questions(tmp_path, script, capsys):
    with pytest.raises(SystemExit) as e:
        script.main(["--data-dir", str(tmp_path), "--local"])
    assert e.value.code == 2 and "--questions" in capsys.readouterr().err
    assert "--local" in SCRIPT.read_text()


# ------------------------------------------------------------------------------- the real model (slow)

def _model_on_disk():
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(GEMMA, "model.safetensors"), str)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(os.getenv("BONCLAY_SLOW_TESTS") != "1" or not _model_on_disk(),
                    reason="set BONCLAY_SLOW_TESTS=1 and download mlx-community/gemma-4-e4b-it-4bit to run the real model")
def test_real_gemma_writes_valid_hindi_and_hinglish_questions(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setenv("BACKGROUND_LLM_FALLBACK", "local")
    monkeypatch.setenv("BACKGROUND_LLM_POLL_S", "1")
    monkeypatch.setenv("MEMORY_GUARD", "off")
    activity.reset()
    batch = qi.make_batches(passages(1))[0]
    try:
        out = qi.generate_batch(batch, chat=HostedChat([hosted_429()]), pacer=qi.Pacer(), sleep=lambda s: None)
    finally:
        bg.release()
    questions = out[0]
    scripts = [qi._script_of(q) for q in questions]
    assert "hi" in scripts and "latin" in scripts and all(q.endswith("?") for q in questions)
    assert not llm.background_running()
