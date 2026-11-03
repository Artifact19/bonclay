"""Generation settings for grounded answers (services/llm.py): Qwen3 thinking off for answers (Groq reasoning_effort
"none"), tool requests left alone, a low answer temperature, and a configurable fallback list that starts with the
best-grounded model and skips a model Groq no longer serves."""
import io
import json
import urllib.error

import pytest

from services import llm


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    monkeypatch.delenv("GROQ_CHAT_MODEL", raising=False)
    monkeypatch.delenv("GROQ_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("LLM_THINKING", raising=False)
    llm._dead_models.clear()
    sent = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        sent.append(body)
        if body.get("stream"):
            lines = [b'data: {"choices":[{"delta":{"content":"OK."}}]}\n', b"data: [DONE]\n"]
            return Resp(b"".join(lines))
        return Resp(json.dumps({"choices": [{"message": {"content": "OK"}}], "usage": {}}).encode())

    monkeypatch.setattr(llm, "_urlopen", fake_urlopen)
    yield sent
    llm._dead_models.clear()


def test_answer_request_turns_qwen_thinking_off_and_sends_the_temperature(groq):
    llm.chat([{"role": "user", "content": "hi"}], temperature=llm.answer_temperature(), max_tokens=4)
    body = groq[-1]
    assert body["model"] == "qwen/qwen3.8-27b" and body["reasoning_effort"] == "none" and body["temperature"] == 0.2
    assert "reasoning_format" not in body and "include_reasoning" not in body


def test_streamed_answer_also_has_thinking_off(groq):
    s = llm.chat_stream([{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=4)
    assert "".join(s) == "OK."
    assert groq[-1]["stream"] is True and groq[-1]["reasoning_effort"] == "none" and groq[-1]["temperature"] == 0.1


def test_tool_requests_are_left_as_they_were(groq):
    tools = [{"type": "function", "function": {"name": "book", "parameters": {"type": "object", "properties": {}}}}]
    llm.chat([{"role": "user", "content": "book 11"}], tools=tools, max_tokens=4)
    assert "reasoning_effort" not in groq[-1] and groq[-1]["tools"]


def test_thinking_can_be_switched_back_on(groq, monkeypatch):
    monkeypatch.setenv("LLM_THINKING", "on")
    llm.chat([{"role": "user", "content": "hi"}], max_tokens=4)
    assert "reasoning_effort" not in groq[-1]


def test_gpt_oss_keeps_its_own_reasoning_settings():
    assert llm._groq_extra("openai/gpt-oss-20b") == {"reasoning_effort": "low", "include_reasoning": False}
    assert llm._groq_extra("llama-3.3-70b-versatile") == {}


def test_answer_temperature_env_is_clamped(monkeypatch):
    monkeypatch.setenv("LLM_ANSWER_TEMPERATURE", "0.05")
    assert llm.answer_temperature() == 0.05
    monkeypatch.setenv("LLM_ANSWER_TEMPERATURE", "7")
    assert llm.answer_temperature() == 1.0
    monkeypatch.setenv("LLM_ANSWER_TEMPERATURE", "nonsense")
    assert llm.answer_temperature() == 0.2


def test_a_server_that_rejects_reasoning_effort_is_asked_again_without_it(groq, monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        calls.append(body)
        if "reasoning_effort" in body:
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                         io.BytesIO(b'{"error":{"message":"reasoning_effort is not supported"}}'))
        if body.get("stream"):
            return Resp(b'data: {"choices":[{"delta":{"content":"Hi."}}]}\n' + b"data: [DONE]\n")
        return Resp(json.dumps({"choices": [{"message": {"content": "OK"}}], "usage": {}}).encode())

    monkeypatch.setattr(llm, "_urlopen", fake_urlopen)
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "")
    assert llm.chat([{"role": "user", "content": "hi"}], max_tokens=4).text == "OK"
    assert "".join(llm.chat_stream([{"role": "user", "content": "hi"}], max_tokens=4)) == "Hi."
    assert [("reasoning_effort" in c) for c in calls] == [True, False, True, False]


def test_default_fallbacks_are_models_a_free_groq_account_has(monkeypatch):
    monkeypatch.delenv("GROQ_CHAT_MODEL", raising=False)
    monkeypatch.delenv("GROQ_FALLBACK_MODELS", raising=False)
    llm._dead_models.clear()
    assert llm._groq_models() == ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-120b, openai/gpt-oss-20b")
    assert llm._groq_models() == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def test_a_decommissioned_fallback_is_paid_for_once(groq, monkeypatch):
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "llama-3.3-70b-versatile,openai/gpt-oss-20b,openai/gpt-oss-120b")
    calls = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        calls.append(body["model"])
        if body["model"] == "qwen/qwen3.8-27b":
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"rate_limit_exceeded"))
        if body["model"] == "llama-3.3-70b-versatile":
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(
                b'{"error":{"code":"model_decommissioned","message":"The model has been decommissioned"}}'))
        return Resp(json.dumps({"choices": [{"message": {"content": "OK"}}], "usage": {}}).encode())

    monkeypatch.setattr(llm, "_urlopen", fake_urlopen)
    assert llm.chat([{"role": "user", "content": "hi"}], max_tokens=4).model == "openai/gpt-oss-20b"
    assert calls == ["qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
    calls.clear()
    llm.chat([{"role": "user", "content": "hi"}], max_tokens=4)
    assert calls == ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]          # the gone model is skipped for an hour
    monkeypatch.setenv("GROQ_DEAD_MODEL_S", "0")
    calls.clear()
    llm.chat([{"role": "user", "content": "hi"}], max_tokens=4)
    assert "llama-3.3-70b-versatile" in calls


def test_a_reasoning_model_gets_room_for_its_reasoning(monkeypatch):
    monkeypatch.delenv("REASONING_TOKEN_ROOM", raising=False)
    assert llm._with_reasoning_room(300, {"reasoning_effort": "low", "include_reasoning": False}) == 1324
    assert llm._with_reasoning_room(300, {"reasoning_effort": "none"}) == 300
    assert llm._with_reasoning_room(300, None) == 300
    monkeypatch.setenv("REASONING_TOKEN_ROOM", "0")
    assert llm._with_reasoning_room(300, {"reasoning_effort": "low"}) == 300


def test_an_invalid_json_reply_moves_on_to_the_next_model():
    assert llm._groq_retryable(RuntimeError('HTTP 400: {"error":{"code":"json_validate_failed"}}'))
