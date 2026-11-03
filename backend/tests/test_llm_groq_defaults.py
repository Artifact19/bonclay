"""Groq calls must carry a real User-Agent (Cloudflare blocks Python's default) and use a model Groq still serves."""
import io
import json

from services import llm


def _capture(monkeypatch):
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["body"] = json.loads(req.data)
        return Resp(json.dumps({"choices": [{"message": {"content": "OK"}}], "usage": {}}).encode())

    monkeypatch.setattr(llm, "_urlopen", fake_urlopen)
    return seen


def test_groq_request_has_user_agent_and_current_default_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_CHAT_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    seen = _capture(monkeypatch)
    res = llm.chat([{"role": "user", "content": "hi"}], max_tokens=4)
    assert res.provider == "groq" and res.text == "OK"
    assert seen["headers"]["user-agent"].startswith("BonClay/")
    assert seen["body"]["model"] == llm.DEFAULT_GROQ_MODEL != "llama-3.3-70b-versatile"


def test_failed_provider_is_remembered_until_it_answers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDERS", "groq")

    def boom(req, timeout=None):
        raise OSError("blocked")

    monkeypatch.setattr(llm, "_urlopen", boom)
    try:
        llm.chat([{"role": "user", "content": "hi"}], max_tokens=4)
    except llm.LLMUnavailable:
        pass
    assert "blocked" in llm.last_failures["groq"]
    assert next(p for p in llm.status()["providers"] if p["name"] == "groq")["last_error"]
    _capture(monkeypatch)
    llm.chat([{"role": "user", "content": "hi"}], max_tokens=4)
    assert "groq" not in llm.last_failures


def test_no_other_module_hard_codes_a_groq_chat_model():
    """The default lives in llm.DEFAULT_GROQ_MODEL only; the document Q&A route reads it from there (ASR's
    whisper-large-v3-turbo is a different, still served model)."""
    import re
    from pathlib import Path
    backend_dir = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"qwen/qwen|llama-3\.\d|llama3-|mixtral|-versatile|-instant")
    offenders = [str(p.relative_to(backend_dir)) for p in [backend_dir / "backend.py", *backend_dir.glob("routers/*.py"),
                 *backend_dir.glob("services/*.py"), *backend_dir.glob("core/*.py")]
                 if pattern.search(p.read_text(encoding="utf-8"))
                 and p.name != "llm.py"]
    assert offenders == []


def test_rag_route_uses_the_shared_default(monkeypatch):
    import importlib
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_CHAT_MODEL", raising=False)
    from routers import rag
    importlib.reload(rag)
    assert rag.GROQ_MODEL == llm.DEFAULT_GROQ_MODEL


def test_rate_limited_groq_model_falls_back_to_production_model(monkeypatch):
    import urllib.error
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_CHAT_MODEL", raising=False)
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b,openai/gpt-oss-120b")
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    sent = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        sent.append(body)
        if body["model"] == llm.DEFAULT_GROQ_MODEL:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"rate_limit_exceeded"))
        return Resp(json.dumps({"choices": [{"message": {"content": "OK"}}], "usage": {}}).encode())

    monkeypatch.setattr(llm, "_urlopen", fake_urlopen)
    res = llm.chat([{"role": "user", "content": "hi"}], max_tokens=4)
    assert res.text == "OK" and res.provider == "groq" and res.model == "openai/gpt-oss-20b"
    assert [b["model"] for b in sent] == [llm.DEFAULT_GROQ_MODEL, "openai/gpt-oss-20b"]
    assert sent[1]["reasoning_effort"] == "low" and sent[0]["reasoning_effort"] == "none"  # Qwen3: no thinking


def test_non_retryable_groq_error_does_not_switch_model(monkeypatch):
    import urllib.error
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    sent = []

    def fake_urlopen(req, timeout=None):
        sent.append(json.loads(req.data)["model"])
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"invalid_api_key"))

    monkeypatch.setattr(llm, "_urlopen", fake_urlopen)
    try:
        llm.chat([{"role": "user", "content": "hi"}], max_tokens=4)
    except llm.LLMUnavailable:
        pass
    assert sent == [llm.DEFAULT_GROQ_MODEL]


def test_fallback_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "")
    assert llm._groq_models() == [llm.DEFAULT_GROQ_MODEL]


def test_tool_choice_none_rejection_is_retried_without_tools_as_plain_text(monkeypatch):
    """Groq's gpt-oss calls a tool even with tool_choice "none" and the server answers HTTP 400: ask again with no
    tools and the tool history as plain text, so the final round still gets an answer."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    bodies = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        bodies.append(body)
        if "tools" in body:
            raise llm.urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                             io.BytesIO(b'{"error":{"message":"Tool choice is none, but model called a tool"}}'))
        return Resp(json.dumps({"choices": [{"message": {"content": "Booked for 11."}}], "usage": {}}).encode())

    monkeypatch.setattr(llm, "_urlopen", fake_urlopen)
    tools = [{"type": "function", "function": {"name": "book", "parameters": {"type": "object", "properties": {}}}}]
    msgs = [{"role": "user", "content": "book 11"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "type": "function", "function": {"name": "book", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"}]
    res = llm.chat(msgs, tools=tools, tool_choice="none", max_tokens=20)
    assert res.text == "Booked for 11." and not res.tool_calls
    assert len(bodies) == 2 and "tools" in bodies[0]
    retry = bodies[1]
    assert "tools" not in retry and "tool_choice" not in retry
    assert all(m["role"] in ("user", "assistant") and not m.get("tool_calls") for m in retry["messages"])
    assert "Tool result: ok" in retry["messages"][-1]["content"]
