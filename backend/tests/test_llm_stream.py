"""llm.chat_stream: SSE parsing against a real local HTTP server, and the provider fallback chain (a provider that
fails before its first token is skipped; one that fails after it stops the stream)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from services import llm


def sse(*events) -> bytes:
    out = b""
    for e in events:
        out += (e if isinstance(e, str) else "data: " + json.dumps(e)).encode() + b"\n\n"
    return out


def chunk(text=None, **extra):
    delta = {} if text is None else {"content": text}
    return {"choices": [{"index": 0, "delta": delta}], **extra}


class Server:
    """Serves one canned response per POST (a status and a body); records the request bodies."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                status, payload = outer.responses.pop(0) if outer.responses else outer.responses_last
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream" if status == 200 else "application/json")
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()

            def log_message(self, *a):
                pass

        self.responses_last = (500, b"{}")
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server():
    servers = []

    def make(*responses):
        s = Server(responses)
        servers.append(s)
        return s
    yield make
    for s in servers:
        s.close()


@pytest.fixture()
def only_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "openai")
    monkeypatch.setenv("LLM_MODEL", "tiny")
    monkeypatch.setenv("LLM_API_KEY", "secret")


MSGS = [{"role": "user", "content": "hi"}]


def test_sse_parsing_deltas_usage_and_summary(server, monkeypatch, only_openai):
    body = sse(": keep-alive", {"choices": [{"delta": {"role": "assistant"}}]}, chunk("Hel"), "event: ping",
               chunk("lo"), chunk(" wor"), chunk("ld."), "data: not json", {"choices": []},
               chunk(None, usage={"total_tokens": 12, "completion_tokens": 4}), "data: [DONE]",
               chunk("ignored after done"))
    s = server((200, body))
    monkeypatch.setenv("LLM_BASE_URL", s.url)
    stream = llm.chat_stream(MSGS, temperature=0.1, max_tokens=50)
    assert stream.result is None
    deltas = list(stream)
    assert deltas == ["Hel", "lo", " wor", "ld."]
    r = stream.result
    assert r.text == "Hello world." and r.provider == "openai" and r.model == "tiny"
    assert r.usage == {"total_tokens": 12, "completion_tokens": 4}
    assert r.first_token_ms is not None and r.first_token_ms <= r.latency_ms
    req = s.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer secret"
    assert req["body"]["stream"] is True and req["body"]["messages"] == MSGS and req["body"]["max_tokens"] == 50
    assert llm.usage_today()["openai"]["tokens"] >= 12


def test_groq_style_usage_and_stream_options_retry(server, monkeypatch, only_openai):
    rejected = json.dumps({"error": {"message": "unknown field stream_options"}}).encode()
    ok = sse(chunk("ok"), {"choices": [], "x_groq": {"usage": {"total_tokens": 3}}}, "data: [DONE]")
    s = server((400, rejected), (200, ok))
    monkeypatch.setenv("LLM_BASE_URL", s.url)
    stream = llm.chat_stream(MSGS)
    assert "".join(stream) == "ok" and stream.result.usage == {"total_tokens": 3}
    assert "stream_options" in s.requests[0]["body"] and "stream_options" not in s.requests[1]["body"]


def test_server_error_event_before_text_is_a_failure(server, monkeypatch, only_openai):
    s = server((200, sse({"error": {"message": "model overloaded"}})))
    monkeypatch.setenv("LLM_BASE_URL", s.url)
    with pytest.raises(llm.LLMUnavailable) as e:
        list(llm.chat_stream(MSGS))
    assert "model overloaded" in e.value.reasons["openai"]


def test_http_error_and_empty_reply_are_failures(server, monkeypatch, only_openai):
    s = server((500, b'{"error":"boom"}'))
    monkeypatch.setenv("LLM_BASE_URL", s.url)
    with pytest.raises(llm.LLMUnavailable) as e:
        list(llm.chat_stream(MSGS))
    assert "HTTP 500" in e.value.reasons["openai"]
    s2 = server((200, sse({"choices": [{"delta": {"role": "assistant"}}]}, "data: [DONE]")))
    monkeypatch.setenv("LLM_BASE_URL", s2.url)
    with pytest.raises(llm.LLMUnavailable) as e:
        list(llm.chat_stream(MSGS))
    assert "empty reply" in e.value.reasons["openai"]


def test_unconfigured_providers_are_reported(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "groq,openai")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(llm.LLMUnavailable) as e:
        list(llm.chat_stream(MSGS))
    assert set(e.value.reasons) == {"groq", "openai"}


def fake_chain(monkeypatch, **providers):
    """providers: name -> generator function (messages, info, **kw) -> deltas."""
    monkeypatch.setenv("LLM_PROVIDERS", ",".join(providers))
    monkeypatch.setattr(llm, "PROVIDERS", {n: (lambda: None, None) for n in providers})
    monkeypatch.setattr(llm, "STREAM_PROVIDERS", providers)


def test_fallback_to_next_provider_before_first_token(monkeypatch):
    def first(messages, info, **kw):
        raise ConnectionError("refused")
        yield  # pragma: no cover

    def second(messages, info, **kw):
        info["model"] = "m2"
        yield "from "
        yield "second"
    fake_chain(monkeypatch, groq=first, openai=second)
    stream = llm.chat_stream(MSGS)
    assert "".join(stream) == "from second"
    assert stream.result.provider == "openai" and stream.result.model == "m2"


def test_provider_failing_after_first_token_stops_and_reports(monkeypatch):
    tried = []

    def first(messages, info, **kw):
        tried.append("first")
        info["model"] = "m1"
        yield "Half a sen"
        raise ConnectionResetError("connection reset")

    def second(messages, info, **kw):
        tried.append("second")
        yield "never"
    fake_chain(monkeypatch, groq=first, openai=second)
    stream = llm.chat_stream(MSGS)
    got = []
    with pytest.raises(llm.LLMStreamInterrupted) as e:
        for d in stream:
            got.append(d)
    assert got == ["Half a sen"] and tried == ["first"]  # the second provider was not asked to start over
    assert e.value.partial == "Half a sen" and e.value.provider == "groq" and e.value.model == "m1"
    assert stream.result is None


def test_all_failed_raises_with_every_reason(monkeypatch):
    def bad(msg):
        def gen(messages, info, **kw):
            raise RuntimeError(msg)
            yield  # pragma: no cover
        return gen
    fake_chain(monkeypatch, groq=bad("one"), openai=bad("two"))
    with pytest.raises(llm.LLMUnavailable) as e:
        list(llm.chat_stream(MSGS))
    assert "one" in e.value.reasons["groq"] and "two" in e.value.reasons["openai"]


def test_total_timeout_interrupts_a_slow_reply(monkeypatch):
    import time

    def slow(messages, info, **kw):
        for _ in range(50):
            time.sleep(0.03)
            yield "x"
    fake_chain(monkeypatch, groq=slow)
    with pytest.raises(llm.LLMStreamInterrupted) as e:
        list(llm.chat_stream(MSGS, total_timeout=0.1))
    assert isinstance(e.value.cause, TimeoutError) and e.value.partial


def test_close_stops_the_provider_and_releases_resources(monkeypatch):
    state = {"closed": False}

    def gen(messages, info, **kw):
        try:
            for i in range(100):
                yield str(i)
        finally:
            state["closed"] = True
    fake_chain(monkeypatch, groq=gen)
    stream = llm.chat_stream(MSGS)
    it = iter(stream)
    assert next(it) == "0"
    stream.close()
    assert state["closed"] is True


def test_chat_is_unchanged_and_non_streaming(server, monkeypatch, only_openai):
    s = server((200, json.dumps({"choices": [{"message": {"content": "plain"}}], "usage": {"total_tokens": 2}})
                .encode()))
    monkeypatch.setenv("LLM_BASE_URL", s.url)
    r = llm.chat(MSGS)
    assert r.text == "plain" and s.requests[0]["body"]["stream"] is False
