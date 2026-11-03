"""Groq's gpt-oss calls a tool natively when the JSON tool protocol describes tools in the prompt (no `tools` in the
request, so tool_choice is "none"), and Groq rejects the reply: HTTP 400 tool_use_failed for a blocking request, HTTP
200 then an `event: error` for a streamed one. The attempted call is in `failed_generation`; these tests replay the
exact shapes Groq sent on 2026-10-19 and check the tool turn carries on instead of the caller hearing
"I'm having trouble responding right now"."""
import io
import json
import urllib.error

import pytest

from services import llm, tools

# What Groq answered (blocking) when gpt-oss-20b got the JSON tool prompt
BLOCKING_400 = {"error": {"message": "Tool choice is none, but model called a tool", "type": "invalid_request_error",
                          "code": "tool_use_failed",
                          "failed_generation": json.dumps({"name": "book_appointment", "arguments": {
                              "name": "Priya", "phone": "9876543210", "date": "2026-10-20", "time": "11:00"}})}}
# ... and streamed: the JSON protocol wrapped inside the native call's arguments
STREAM_ERROR = {"error": {"message": "Tool choice is none, but model called a tool", "type": "invalid_request_error",
                          "code": "tool_use_failed", "status_code": 400,
                          "failed_generation": json.dumps({"name": "book_appointment", "arguments": {
                              "tool": "book_appointment", "arguments": {"name": "Priya", "phone": "9876543210",
                                                                        "date": "2026-10-20", "time": "11:00"}}})}}
ARGS = {"name": "Priya", "phone": "9876543210", "date": "2026-10-20", "time": "11:00"}


class Resp(io.BytesIO):
    status = 200
    headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(req, code, body, headers=None):
    return urllib.error.HTTPError(req.full_url, code, "error", headers or {}, io.BytesIO(json.dumps(body).encode()))


def ok(text):
    return Resp(json.dumps({"choices": [{"message": {"content": text}}], "usage": {}}).encode())


def sse(*events):
    """A streamed reply: `events` are dicts (data lines) or ("error", dict) for an `event: error`."""
    out = b""
    for e in events:
        if isinstance(e, tuple):
            out += b"event: error\n" + b"data: " + json.dumps(e[1]).encode() + b"\n\n"
        else:
            out += b"data: " + json.dumps(e).encode() + b"\n\n"
    return Resp(out)


def delta(text):
    return {"choices": [{"delta": {"content": text}}]}


@pytest.fixture()
def gpt_oss(monkeypatch):
    """Groq only, with gpt-oss-20b as the model (qwen out of quota)."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    monkeypatch.setenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "")
    llm._dead_models.clear()
    llm._quota_until.clear()
    sent = []
    yield sent
    llm._dead_models.clear()
    llm._quota_until.clear()


# ---------------------------------------------------------------------------------------------- parsing

def test_failed_generation_shapes_are_read():
    assert llm.failed_tool_call(json.dumps(BLOCKING_400)) == {"name": "book_appointment", "arguments": ARGS}
    assert llm.failed_tool_call(STREAM_ERROR) == {"name": "book_appointment", "arguments": ARGS}
    as_string = {"error": {"failed_generation": json.dumps({"name": "functions.find_booking",
                                                            "arguments": json.dumps({"phone": "98"})})}}
    assert llm.failed_tool_call(as_string) == {"name": "find_booking", "arguments": {"phone": "98"}}
    listed = {"error": {"failed_generation": json.dumps([{"name": "cancel_appointment",
                                                          "arguments": {"booking_id": "A12"}}])}}
    assert llm.failed_tool_call(listed)["name"] == "cancel_appointment"
    harmony = {"error": {"failed_generation": '<|channel|>commentary to=functions.find_booking <|constrain|>json'
                                              '<|message|>{"phone": "9876543210"}'}}
    assert llm.failed_tool_call(harmony) == {"name": "find_booking", "arguments": {"phone": "9876543210"}}
    assert llm.failed_tool_call({"error": {"message": "Tool choice is none"}}) is None
    assert llm.failed_tool_call({"error": {"failed_generation": "I think I should book"}}) is None
    assert llm.failed_tool_call("not json") is None


# ---------------------------------------------------------------------------------------------- blocking path

def test_blocking_json_protocol_call_becomes_the_reply(gpt_oss, monkeypatch):
    def fake(req, timeout=None):
        gpt_oss.append(json.loads(req.data))
        raise http_error(req, 400, BLOCKING_400)

    monkeypatch.setattr(llm, "_urlopen", fake)
    res = llm.chat([{"role": "user", "content": "book"}], json_tools=["book_appointment", "cancel_appointment"])
    assert json.loads(res.text) == {"tool": "book_appointment", "arguments": ARGS}
    assert res.model == "openai/gpt-oss-20b" and len(gpt_oss) == 1          # no second request
    assert "tools" not in gpt_oss[0] and "json_tools" not in gpt_oss[0]


def test_blocking_without_json_tools_asks_once_more_for_words(gpt_oss, monkeypatch):
    def fake(req, timeout=None):
        body = json.loads(req.data)
        gpt_oss.append(body)
        if len(gpt_oss) == 1:
            raise http_error(req, 400, BLOCKING_400)
        return ok("Dr. Meera Kulkarni sits on Monday, Wednesday and Friday.")

    monkeypatch.setattr(llm, "_urlopen", fake)
    res = llm.chat([{"role": "system", "content": "You are Ria."}, {"role": "user", "content": "When?"}])
    assert res.text.startswith("Dr. Meera") and len(gpt_oss) == 2
    retry = gpt_oss[1]
    assert retry["messages"][0] == {"role": "system", "content": "You are Ria."}
    assert retry["messages"][-1]["content"].startswith("When?") and llm.PLAIN_TEXT_NOTE in retry["messages"][-1]["content"]


def test_blocking_call_to_a_tool_the_prompt_does_not_offer_is_not_used(gpt_oss, monkeypatch):
    """A made-up or builtin tool is never turned into an action: the model is asked for words instead."""
    def fake(req, timeout=None):
        gpt_oss.append(json.loads(req.data))
        if len(gpt_oss) == 1:
            raise http_error(req, 400, BLOCKING_400)
        return ok("Sure.")

    monkeypatch.setattr(llm, "_urlopen", fake)
    assert llm.chat([{"role": "user", "content": "hi"}], json_tools=["find_booking"]).text == "Sure."
    assert len(gpt_oss) == 2


def test_blocking_retry_that_fails_again_moves_to_the_next_groq_model(gpt_oss, monkeypatch):
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-120b")

    def fake(req, timeout=None):
        body = json.loads(req.data)
        gpt_oss.append(body["model"])
        if body["model"] == "openai/gpt-oss-20b":
            raise http_error(req, 400, {"error": {"code": "tool_use_failed",
                                                  "message": "Tool choice is none, but model called a tool"}})
        return ok("Hello.")

    monkeypatch.setattr(llm, "_urlopen", fake)
    res = llm.chat([{"role": "user", "content": "hi"}])
    assert res.text == "Hello." and res.model == "openai/gpt-oss-120b"
    assert gpt_oss == ["openai/gpt-oss-20b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]


def test_native_auto_malformed_call_is_recovered_as_a_tool_call(gpt_oss, monkeypatch):
    err = {"error": {"code": "tool_use_failed", "message": "Failed to call a function. Please adjust your prompt.",
                     "failed_generation": json.dumps({"name": "book_appointment", "arguments": ARGS})}}

    def fake(req, timeout=None):
        gpt_oss.append(json.loads(req.data))
        raise http_error(req, 400, err)

    monkeypatch.setattr(llm, "_urlopen", fake)
    native = [{"type": "function", "function": {"name": "book_appointment", "parameters": {"type": "object"}}}]
    res = llm.chat([{"role": "user", "content": "book"}], tools=native, tool_choice="auto")
    assert res.tool_calls == [{"id": "call_failed_generation", "name": "book_appointment", "arguments": ARGS}]
    assert len(gpt_oss) == 1


def test_native_final_round_still_gets_words(gpt_oss, monkeypatch):
    """tool_choice "none" (out of calls): a call in failed_generation is NOT used; the plain retry answers."""
    def fake(req, timeout=None):
        body = json.loads(req.data)
        gpt_oss.append(body)
        if "tools" in body:
            raise http_error(req, 400, BLOCKING_400)
        return ok("Booked, A12.")

    monkeypatch.setattr(llm, "_urlopen", fake)
    native = [{"type": "function", "function": {"name": "book_appointment", "parameters": {"type": "object"}}}]
    res = llm.chat([{"role": "user", "content": "book"}], tools=native, tool_choice="none")
    assert res.text == "Booked, A12." and not res.tool_calls
    assert "tools" not in gpt_oss[1] and "tool_choice" not in gpt_oss[1]


# ---------------------------------------------------------------------------------------------- streaming path

def test_stream_event_error_becomes_the_json_protocol_call(gpt_oss, monkeypatch):
    def fake(req, timeout=None):
        gpt_oss.append(json.loads(req.data))
        return sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]}, ("error", STREAM_ERROR))

    monkeypatch.setattr(llm, "_urlopen", fake)
    s = llm.chat_stream([{"role": "user", "content": "book"}], json_tools=("book_appointment",))
    text = "".join(s)
    assert json.loads(text) == {"tool": "book_appointment", "arguments": ARGS}
    assert s.result.model == "openai/gpt-oss-20b" and len(gpt_oss) == 1


def test_stream_http_400_becomes_the_json_protocol_call(gpt_oss, monkeypatch):
    def fake(req, timeout=None):
        raise http_error(req, 400, BLOCKING_400)

    monkeypatch.setattr(llm, "_urlopen", fake)
    text = "".join(llm.chat_stream([{"role": "user", "content": "book"}], json_tools=["book_appointment"]))
    assert json.loads(text)["tool"] == "book_appointment"


def test_stream_without_json_tools_retries_once_for_words(gpt_oss, monkeypatch):
    def fake(req, timeout=None):
        body = json.loads(req.data)
        gpt_oss.append(body)
        if len(gpt_oss) == 1:
            return sse(("error", STREAM_ERROR))
        return Resp(sse(delta("We open "), delta("at nine.")).getvalue() + b"data: [DONE]\n\n")

    monkeypatch.setattr(llm, "_urlopen", fake)
    assert "".join(llm.chat_stream([{"role": "user", "content": "When do you open?"}])) == "We open at nine."
    assert len(gpt_oss) == 2 and llm.PLAIN_TEXT_NOTE in gpt_oss[1]["messages"][-1]["content"]


# ---------------------------------------------------------------------------------------------- the tool turn, end to end

class FakeHub:
    def __init__(self):
        self.calls = []

    def call(self, spec, arguments, timeout):
        self.calls.append((spec.tool, arguments))
        return True, {"booking_id": "A12", "doctor": "Dr. Meera Kulkarni", "date": "2026-10-20",
                      "weekday": "Tuesday", "time": "11:00", "status": "booked"}


BOOK = tools.ToolSpec("appointments", "book_appointment", "Book an appointment.", {
    "type": "object", "properties": {k: {"type": "string"} for k in ("name", "phone", "date", "time")},
    "required": ["name", "phone", "date", "time"]})
MESSAGES = [{"role": "system", "content": "You are Ria."},
            {"role": "user", "content": "Book me tomorrow at 11, I am Priya, 9876543210."}]


def test_blocking_tool_turn_on_gpt_oss_books_instead_of_failing(gpt_oss, monkeypatch):
    monkeypatch.setenv("BONCLAY_TOOL_PROTOCOL", "json")
    hub = FakeHub()
    monkeypatch.setattr(tools, "hub", hub)

    def fake(req, timeout=None):
        body = json.loads(req.data)
        gpt_oss.append(body)
        if len(gpt_oss) == 1:
            raise http_error(req, 400, BLOCKING_400)
        return ok("Booked: A12 with Dr. Meera Kulkarni, Tuesday 20 October at 11:00.")

    monkeypatch.setattr(llm, "_urlopen", fake)
    out = tools.run_tool_turn(MESSAGES, [BOOK])
    assert hub.calls == [("book_appointment", ARGS)]
    assert out.text.startswith("Booked: A12") and out.protocol == "json" and out.model == "openai/gpt-oss-20b"
    # the model then sees its call in the JSON protocol and the tool's result
    assert json.loads(gpt_oss[1]["messages"][-2]["content"]) == {"tool": "book_appointment", "arguments": ARGS}
    assert "TOOL RESULT from book_appointment" in gpt_oss[1]["messages"][-1]["content"]


def test_streamed_tool_turn_on_gpt_oss_books_instead_of_failing(gpt_oss, monkeypatch):
    monkeypatch.setenv("BONCLAY_TOOL_PROTOCOL", "json")
    hub = FakeHub()
    monkeypatch.setattr(tools, "hub", hub)

    def fake(req, timeout=None):
        gpt_oss.append(json.loads(req.data))
        if len(gpt_oss) == 1:
            return sse(("error", STREAM_ERROR))
        return Resp(b'data: {"choices":[{"delta":{"content":"Booked: A12 for Tuesday at 11:00."}}]}\n\n'
                    b'data: [DONE]\n\n')

    monkeypatch.setattr(llm, "_urlopen", fake)
    stream = tools.ToolStream(MESSAGES, [BOOK])
    spoken = "".join(stream)
    assert hub.calls == [("book_appointment", ARGS)]
    assert spoken == "Booked: A12 for Tuesday at 11:00." and "{" not in spoken
    assert stream.turn.model == "openai/gpt-oss-20b"
