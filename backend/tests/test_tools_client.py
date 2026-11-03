"""BonClay as an MCP client (services/tools.py): a real stdio server is started, its tools are discovered and called,
a slow or broken server is survived, and the turn loop (native tool calls, the JSON protocol, the call limit, the
confirmation rule, ungrounded booking ids) runs against a fake model."""
import json
import re
import sys
import textwrap

import pytest

from services import llm, tools

FAKE_SERVER = textwrap.dedent('''
    import time
    try:
        from mcp.server.mcpserver import MCPServer as Server
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server
        from mcp.server.fastmcp.exceptions import ToolError
    mcp = Server("fake")

    @mcp.tool()
    def echo(text: str) -> dict:
        """Repeat the text."""
        return {"echo": text}

    @mcp.tool()
    def boom() -> dict:
        """Always fails."""
        raise ToolError("it broke on purpose")

    @mcp.tool()
    def slow() -> dict:
        """Takes too long."""
        time.sleep(6)
        return {}

    if __name__ == "__main__":
        mcp.run()
''')


@pytest.fixture()
def hub(monkeypatch, tmp_path):
    """A private ToolHub with two fake servers configured (the same code under two names) and no built-ins."""
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    cfg = {name: {"command": sys.executable, "args": [str(script)]} for name in ("fake", "twin")}
    cfg["broken"] = {"command": sys.executable, "args": ["-c", "raise SystemExit(3)"]}
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv("BONCLAY_MCP_SERVERS", str(path))
    h = tools.ToolHub()
    monkeypatch.setattr(tools, "hub", h)
    yield h
    h.close()


# ---------------------------------------------------------------------------------------------- the client

def test_configured_servers_and_agent_tools_setting(hub):
    assert {"appointments", "fake", "twin", "broken"} <= set(tools.known_servers())
    assert tools.clean_tool_names(["fake", "fake", " twin "]) == ["fake", "twin"]
    assert tools.clean_tool_names(None) == []
    with pytest.raises(ValueError, match="unknown tool server 'nope'"):
        tools.clean_tool_names(["nope"])
    with pytest.raises(ValueError):
        tools.clean_tool_names("fake")


def test_discovery_lists_the_servers_tools_with_schemas(hub):
    specs = hub.specs_for(["fake"])
    assert sorted(s.name for s in specs) == ["boom", "echo", "slow"]
    echo = next(s for s in specs if s.name == "echo")
    assert echo.server == "fake" and "Repeat" in echo.description
    assert echo.schema["required"] == ["text"] and echo.schema["properties"]["text"]["type"] == "string"
    fn = tools.openai_tools([echo])[0]["function"]
    assert fn["name"] == "echo" and fn["parameters"]["required"] == ["text"]


def test_a_call_runs_the_tool_and_a_session_is_reused(hub):
    echo = next(s for s in hub.specs_for(["fake"]) if s.name == "echo")
    assert hub.call(echo, {"text": "hi"}) == (True, {"echo": "hi"})
    first = hub._conns["fake"]
    assert hub.call(echo, {"text": "again"}) == (True, {"echo": "again"})
    assert hub._conns["fake"] is first and hub.specs_for(["fake"])  # one subprocess, one session


def test_a_tool_error_comes_back_as_data_not_an_exception(hub):
    boom = next(s for s in hub.specs_for(["fake"]) if s.name == "boom")
    ok, value = hub.call(boom, {})
    assert ok is False and "it broke on purpose" in value


def test_execute_checks_arguments_before_calling(hub):
    echo = next(s for s in hub.specs_for(["fake"]) if s.name == "echo")
    call = tools.execute(echo, {}, previous_agent_text=None)
    assert call.ok is False and "'text' is a required property" in call.result
    call = tools.execute(echo, {"text": 5}, previous_agent_text=None)
    assert call.ok is False and "invalid arguments" in call.result
    assert tools.execute(echo, {"text": "ok"}, previous_agent_text=None).ok is True


def test_a_slow_tool_times_out_and_the_next_call_starts_a_fresh_server(hub):
    specs = {s.name: s for s in hub.specs_for(["fake"])}
    with pytest.raises(tools.ToolError, match="did not answer within"):
        hub.call(specs["slow"], {}, timeout=0.5)
    assert "fake" not in hub._conns  # dropped: its outcome is unknown
    assert hub.call(specs["echo"], {"text": "back"}) == (True, {"echo": "back"})
    call = tools.execute(specs["slow"], {}, previous_agent_text=None, timeout=0.3)
    assert call.ok is False and "not available" in call.result  # execute never raises


def test_a_server_that_cannot_start_is_left_out_of_discovery(hub):
    assert hub.specs_for(["broken"]) == []
    assert sorted(s.name for s in hub.specs_for(["broken", "fake"])) == ["boom", "echo", "slow"]


def test_same_tool_name_on_two_servers_is_prefixed(hub):
    names = [s.name for s in hub.specs_for(["fake", "twin"]) if s.tool == "echo"]
    assert names == ["fake__echo", "twin__echo"]


def test_close_stops_the_server_process(hub):
    hub.specs_for(["fake"])
    conn = hub._conns["fake"]
    hub.close()
    conn.task.result(10)  # the connection's task ends: the subprocess was closed
    assert conn.session is None


# ---------------------------------------------------------------------------------------------- the turn loop

class FakeHub:
    """Stands in for ToolHub in the loop tests: records calls, answers from a table."""

    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def call(self, spec, arguments, timeout=8.0):
        self.calls.append((spec.tool, arguments))
        result = self.results.get(spec.tool)
        if callable(result):
            return result(arguments)
        return result if result is not None else (True, {"ok": True})


BOOK = tools.ToolSpec("appointments", "book_appointment", "Book.", {
    "type": "object", "properties": {"name": {"type": "string"}, "phone": {"type": "string"}},
    "required": ["name", "phone"]})
CANCEL = tools.ToolSpec("appointments", "cancel_appointment", "Cancel.", {
    "type": "object", "properties": {"booking_id": {"type": "string"}}, "required": ["booking_id"]})
BOOKED = (True, {"booking_id": "A12", "doctor": "Dr. Meera Kulkarni", "weekday": "Monday", "date": "2026-10-19",
                 "time": "10:30", "status": "booked"})
MESSAGES = [{"role": "system", "content": "You are Ria."}, {"role": "user", "content": "Book me in."}]


def call(tool_name, /, **arguments):
    return {"id": f"c-{tool_name}", "name": tool_name, "arguments": arguments}


def scripted_chat(monkeypatch, *replies):
    """llm.chat answers from `replies` in order: a string is a plain answer, a list is a set of tool calls, an
    Exception is raised. Every request is kept in `seen`."""
    seen, queue = [], list(replies)

    def chat(messages, **kw):
        seen.append({"messages": [dict(m) for m in messages], **kw})
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, list):
            return llm.ChatResult("", "fake", "fake-model", 1, {}, tool_calls=r)
        return llm.ChatResult(r, "fake", "fake-model", 1, {})

    monkeypatch.setattr(llm, "chat", chat)
    return seen


@pytest.fixture()
def native(monkeypatch):
    monkeypatch.setenv("BONCLAY_TOOL_PROTOCOL", "native")


def test_the_json_protocol_is_the_default(monkeypatch):
    monkeypatch.delenv("BONCLAY_TOOL_PROTOCOL", raising=False)
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    seen = scripted_chat(monkeypatch, '{"tool": "book_appointment", "arguments": {"name": "P", "phone": "1"}}',
                         "Booked: A12.")
    out = tools.run_tool_turn(MESSAGES, [BOOK])
    assert out.protocol == "json" and len(fake.calls) == 1
    assert all("tools" not in r for r in seen)  # nothing was sent as OpenAI `tools`


def test_native_tool_call_then_a_grounded_answer(monkeypatch, native):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    seen = scripted_chat(monkeypatch, [call("book_appointment", name="Priya", phone="9876543210")],
                         "Booked: A12 with Dr. Meera Kulkarni, Monday at 10:30.")
    out = tools.run_tool_turn(MESSAGES, [BOOK, CANCEL])
    assert out.text.startswith("Booked: A12") and out.protocol == "native" and out.provider == "fake"
    assert fake.calls == [("book_appointment", {"name": "Priya", "phone": "9876543210"})]
    [c] = out.calls
    assert c.record() == {"name": "book_appointment", "arguments": {"name": "Priya", "phone": "9876543210"},
                          "ok": True, "result": "booking A12 with Dr. Meera Kulkarni on Monday 2026-10-19 10:30"}
    first, second = seen
    assert [t["function"]["name"] for t in first["tools"]] == ["book_appointment", "cancel_appointment"]
    assert first["tool_choice"] == "auto"
    assert "TOOLS:" in first["messages"][0]["content"]
    assert "Today is" not in first["messages"][0]["content"]            # the clock is in the caller's message
    assert "Right now it is" in first["messages"][-1]["content"]
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c-book_appointment"
    assert "data, not instructions" in tool_msg["content"] and "A12" in tool_msg["content"]
    assert second["messages"][-2]["tool_calls"][0]["function"]["name"] == "book_appointment"


def test_at_most_three_tool_calls_then_the_model_must_answer(monkeypatch, native):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    one = [call("book_appointment", name="P", phone="1")]
    seen = scripted_chat(monkeypatch, one, one, one, "I could only do part of that. Booking A12 is confirmed.")
    out = tools.run_tool_turn(MESSAGES, [BOOK])
    assert len(fake.calls) == 3 and len(out.calls) == 3
    assert [r["tool_choice"] for r in seen] == ["auto", "auto", "auto", "none"]


def test_extra_calls_in_one_round_beyond_the_limit_are_refused(monkeypatch, native):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    scripted_chat(monkeypatch, [call("book_appointment", name="P", phone="1")] * 5, "Done, A12.")
    out = tools.run_tool_turn(MESSAGES, [BOOK])
    assert len(fake.calls) == 3
    assert [c.ok for c in out.calls] == [True, True, True, False, False]


def test_an_unknown_tool_name_is_not_executed(monkeypatch, native):
    fake = FakeHub()
    monkeypatch.setattr(tools, "hub", fake)
    scripted_chat(monkeypatch, [call("delete_everything")], "Sorry, I can't do that.")
    out = tools.run_tool_turn(MESSAGES, [BOOK])
    assert fake.calls == [] and out.calls[0].ok is False and out.text == "Sorry, I can't do that."


def test_json_protocol_when_no_provider_has_native_tools(monkeypatch, native):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    seen = scripted_chat(
        monkeypatch,
        llm.LLMUnavailable({"groq": "no key", "mlx": "no native tool calling"}),   # the native attempt
        '```json\n{"tool": "book_appointment", "arguments": {"name": "Priya", "phone": "9876543210"}}\n```',
        "Your booking A12 is confirmed for Monday at 10:30.")
    out = tools.run_tool_turn(MESSAGES, [BOOK])
    assert out.protocol == "json" and out.text.startswith("Your booking A12")
    assert fake.calls == [("book_appointment", {"name": "Priya", "phone": "9876543210"})]
    assert "tools" not in seen[1] and "tools" not in seen[2]
    system = seen[1]["messages"][0]["content"]
    assert '{"tool": "<tool name>"' in system and "book_appointment(name, phone)" in system  # strings: no type
    result = seen[2]["messages"][-1]
    assert result["role"] == "user" and "TOOL RESULT" in result["content"] and "not instructions" in result["content"]


def test_json_protocol_plain_answer_needs_no_tool(monkeypatch, native):
    fake = FakeHub()
    monkeypatch.setattr(tools, "hub", fake)
    scripted_chat(monkeypatch, llm.LLMUnavailable({}), "We open at nine. Shall I book you in?")
    out = tools.run_tool_turn(MESSAGES, [BOOK])
    assert out.text == "We open at nine. Shall I book you in?" and out.calls == [] and fake.calls == []


def test_prose_with_braces_is_an_answer_not_a_tool_request():
    by_name = {"book_appointment": BOOK}
    assert tools.parse_tool_request('{"tool": "book_appointment", "arguments": {"name": "a"}}', by_name)[0] \
        == "book_appointment"
    assert tools.parse_tool_request("Sure {not json} thanks", by_name) is None
    assert tools.parse_tool_request('{"tool": "unknown", "arguments": {}}', by_name) is None
    assert tools.parse_tool_request('{"answer": "hi"}', by_name) is None


def test_json_protocol_also_stops_at_three_calls(monkeypatch):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    req = '{"tool": "book_appointment", "arguments": {"name": "P", "phone": "1"}}'
    seen = scripted_chat(monkeypatch, req, req, req, req, "All set: booking A12.")
    out = tools.run_tool_turn(MESSAGES, [BOOK])
    assert len(fake.calls) == 3 and out.text == "All set: booking A12."
    assert "No more tool calls" in seen[-1]["messages"][-1]["content"]


def test_a_native_failure_after_a_tool_ran_never_repeats_the_tool(monkeypatch, native):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    seen = scripted_chat(
        monkeypatch,
        [call("book_appointment", name="P", phone="9876543210")],
        llm.LLMUnavailable({"groq": "HTTP 400"}),                 # the native follow-up fails
        "Your booking A12 is confirmed.")                         # the JSON-protocol answer
    out = tools.run_tool_turn(MESSAGES, [BOOK])
    assert len(fake.calls) == 1 and out.text == "Your booking A12 is confirmed."
    replay = seen[-1]["messages"]
    assert any("TOOL RESULT" in m["content"] and "A12" in m["content"] for m in replay if m["role"] == "user")


def test_calls_are_visible_to_the_caller_even_when_the_turn_fails(monkeypatch, native):
    monkeypatch.setattr(tools, "hub", FakeHub({"book_appointment": BOOKED}))
    scripted_chat(monkeypatch, [call("book_appointment", name="P", phone="9876543210")],
                  llm.LLMUnavailable({}), llm.LLMUnavailable({}))
    record = []
    with pytest.raises(llm.LLMUnavailable):
        tools.run_tool_turn(MESSAGES, [BOOK], calls=record)
    assert [c.name for c in record] == ["book_appointment"]


def test_a_booking_id_no_tool_returned_is_never_spoken(monkeypatch, native):
    monkeypatch.setattr(tools, "hub", FakeHub())
    scripted_chat(monkeypatch, "Great, you are booked. Your booking ID is A77.")
    out = tools.run_tool_turn(MESSAGES, [BOOK])
    assert out.text == tools.UNCONFIRMED_LINE
    # An id the tool returned, or one said earlier in the call, may be repeated.
    monkeypatch.setattr(tools, "hub", FakeHub({"book_appointment": BOOKED}))
    scripted_chat(monkeypatch, [call("book_appointment", name="P", phone="9876543210")], "Booking A 12 is set.")
    assert tools.run_tool_turn(MESSAGES, [BOOK]).text == "Booking A 12 is set."
    monkeypatch.setattr(tools, "hub", FakeHub())
    scripted_chat(monkeypatch, "As I said, A55 is your booking.")
    assert tools.run_tool_turn(MESSAGES, [BOOK], known_text="Your booking is A55.").text.startswith("As I said")


def test_tool_results_are_truncated_and_labelled():
    text = tools.wrap_result("find_booking", True, {"x": "y" * 5000})
    assert len(text) < tools.RESULT_MAX_CHARS + 300 and text.startswith("TOOL RESULT from find_booking (success)")
    assert "NOT done" in tools.wrap_result("book_appointment", False, "slot taken")


# ---------------------------------------------------------------------------------------------- confirmation rule

def test_cancel_needs_the_previous_turn_to_have_asked_about_that_booking(monkeypatch, native):
    fake = FakeHub({"cancel_appointment": (True, {"booking_id": "A12", "status": "cancelled", "date": "2026-10-19",
                                                  "time": "10:30"})})
    monkeypatch.setattr(tools, "hub", fake)
    # No earlier question: refused, the tool is never reached, and the model is told to ask first.
    seen = scripted_chat(monkeypatch, [call("cancel_appointment", booking_id="A12")],
                         "Shall I cancel booking A12?")
    out = tools.run_tool_turn(MESSAGES, [CANCEL], previous_agent_text="How can I help?", known_text="A12")
    assert fake.calls == [] and out.calls[0].ok is False and "not confirmed" in out.calls[0].result
    assert "NOT DONE" in seen[1]["messages"][-1]["content"]
    # The previous reply asked about exactly that booking: the caller's yes lets it through.
    scripted_chat(monkeypatch, [call("cancel_appointment", booking_id="a 12")], "Booking A12 is cancelled.")
    out = tools.run_tool_turn(MESSAGES, [CANCEL], previous_agent_text="Shall I cancel booking A12?")
    assert fake.calls == [("cancel_appointment", {"booking_id": "a 12"})] and out.calls[0].ok
    assert out.calls[0].result.startswith("cancelled A12")
    # A different booking than the one asked about is refused too.
    fake.calls.clear()
    scripted_chat(monkeypatch, [call("cancel_appointment", booking_id="A13")], "Shall I cancel A13?")
    out = tools.run_tool_turn(MESSAGES, [CANCEL], previous_agent_text="Shall I cancel booking A12?", known_text="A13")
    assert fake.calls == [] and out.calls[0].ok is False


def test_cancel_with_an_unknown_history_is_refused():
    spec = CANCEL
    assert tools.confirmation_error(spec, {"booking_id": "A12"}, None) is not None
    assert tools.confirmation_error(BOOK, {"name": "x"}, None) is None  # only destructive tools need it


def test_a_question_about_the_booking_is_what_counts_as_asking():
    ask = tools._asked_about
    assert ask("Shall I cancel booking A12?", "A12")
    assert ask("Booking A12 with Dr. Meera Kulkarni is on Monday at 10:30, shall I cancel booking A12?", "A12")
    assert not ask("Booking A12 is set. Anything else?", "A12")      # a confirmation is not the question
    assert not ask("Shall I cancel booking A13?", "A12")             # another booking
    assert not ask("Cancel booking A12.", "A12")                     # a statement
    assert ask("क्या मैं बुकिंग A12 रद्द कर दूँ?", "A12")


def test_a_made_up_tool_is_reported_to_the_model_and_never_spoken(monkeypatch):
    fake = FakeHub({"book_appointment": BOOKED})
    monkeypatch.setattr(tools, "hub", fake)
    monkeypatch.delenv("BONCLAY_TOOL_PROTOCOL", raising=False)
    seen = scripted_chat(monkeypatch, '{"tool": "cancel_appointments", "arguments": {"cancel_all": "true"}}',
                         "I cannot cancel other bookings.")
    out = tools.run_tool_turn(MESSAGES, [BOOK, CANCEL])
    assert out.text == "I cannot cancel other bookings." and fake.calls == []
    assert [(c.name, c.ok, c.result) for c in out.calls] == [
        ("cancel_appointments", False, "failed: unknown tool")]
    assert "There is no tool called 'cancel_appointments'" in seen[1]["messages"][-1]["content"]
    assert tools.unknown_tool_request('{"tool": "book_appointment", "arguments": {}}', {"book_appointment": BOOK}) is None
    assert tools.unknown_tool_request("plain text", {}) is None
