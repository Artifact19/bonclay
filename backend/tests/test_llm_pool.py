"""The language-model client reuses one keep-alive connection (services/llm.py _urlopen) and can open it ahead of a
reply (prewarm). A real local HTTP/1.1 server counts connections."""
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from services import llm

MSGS = [{"role": "user", "content": "hi"}]


class Server:
    """HTTP/1.1 keep-alive server: counts accepted connections and requests. `mode`: ok | stream | status | hang |
    close_after (answers, then closes the connection without announcing it)."""

    def __init__(self, mode="ok"):
        self.mode, self.connections, self.requests, self.heads = mode, 0, 0, 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                outer.connections += 1

            def do_HEAD(self):
                outer.heads += 1
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests += 1
                if outer.mode == "hang":
                    time.sleep(3)
                    return
                if outer.mode == "status":
                    data = b'{"error":{"message":"rate limited"}}'
                    self.send_response(429)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if outer.mode == "stream":
                    data = (b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
                            b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\ndata: [DONE]\n\n')
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                data = json.dumps({"choices": [{"message": {"content": "OK"}}], "usage": {"total_tokens": 2},
                                   "echo": json.loads(body)["model"]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                if outer.mode == "close_after":
                    self.close_connection = True      # the next request on this socket finds it dead

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server():
    made = []

    def make(mode="ok"):
        s = Server(mode)
        made.append(s)
        return s

    llm.close_pool()
    yield make
    for s in made:
        s.close()
    llm.close_pool()


def chat(url, **kw):
    return llm._openai_chat(url, "m", MSGS, api_key=None, json_mode=False, temperature=0, max_tokens=4,
                            timeout=kw.pop("timeout", 5), **kw)


def test_sequential_requests_share_one_connection(server):
    s = server()
    for _ in range(4):
        assert chat(s.url)["choices"][0]["message"]["content"] == "OK"
    assert s.requests == 4 and s.connections == 1


def test_streamed_replies_reuse_the_connection_too(server):
    s = server("stream")
    for _ in range(3):
        info = {}
        out = "".join(llm._openai_stream(s.url, "m", MSGS, api_key=None, temperature=0, max_tokens=4, timeout=5,
                                         info=info))
        assert out == "hello"
    assert s.requests == 3 and s.connections == 1


def test_a_stream_closed_early_does_not_poison_the_pool(server):
    s = server("stream")
    gen = llm._openai_stream(s.url, "m", MSGS, api_key=None, temperature=0, max_tokens=4, timeout=5, info={})
    assert next(gen) == "he"
    gen.close()
    assert "".join(llm._openai_stream(s.url, "m", MSGS, api_key=None, temperature=0, max_tokens=4, timeout=5,
                                      info={})) == "hello"


def test_a_connection_the_server_closed_is_replaced_without_an_error(server):
    s = server("close_after")
    assert chat(s.url)["echo"] == "m"
    assert chat(s.url)["echo"] == "m"      # the pooled socket is dead: a new one is opened, the caller never knows
    assert s.requests == 2 and s.connections == 2


def test_an_http_error_status_raises_like_urlopen_with_a_readable_body(server):
    s = server("status")
    req = llm.urllib.request.Request(s.url + "/chat/completions", data=b"{}", headers={"Content-Type": "x"})
    with pytest.raises(urllib.error.HTTPError) as e:
        llm._urlopen(req, timeout=5)
    assert e.value.code == 429 and b"rate limited" in e.value.read()
    with pytest.raises(RuntimeError, match="HTTP 429"):
        chat(s.url)


def test_the_error_does_not_leave_a_broken_connection_in_the_pool(server):
    s = server("status")
    for _ in range(2):
        with pytest.raises(RuntimeError):
            chat(s.url)
    s.mode = "ok"
    assert chat(s.url)["choices"]


def test_a_stalled_server_times_out_as_a_timeout_error(server):
    s = server("hang")
    t = time.time()
    with pytest.raises(TimeoutError):
        chat(s.url, timeout=0.3)
    assert time.time() - t < 2


def test_a_refused_connection_is_a_url_error():
    import socket
    llm.close_pool()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    req = llm.urllib.request.Request(f"http://127.0.0.1:{port}/x", data=b"{}")
    with pytest.raises((urllib.error.URLError, TimeoutError)):
        llm._urlopen(req, timeout=1)


def test_request_headers_and_body_are_sent_unchanged(server):
    s = server()
    seen = {}
    orig = llm._client().send

    def spy(request, **kw):
        seen["ua"] = request.headers["user-agent"]
        seen["auth"] = request.headers.get("authorization")
        seen["enc"] = request.headers["accept-encoding"]
        return orig(request, **kw)
    llm._pool.send = spy
    chat(s.url, tools=None)
    assert seen["ua"] == llm.USER_AGENT and seen["auth"] is None and seen["enc"] == "identity"


# ------------------------------------------------------------------------------------------------ prewarm

@pytest.fixture()
def warm(monkeypatch):
    monkeypatch.setattr(llm, "_last_prewarm", None)
    monkeypatch.setenv("LLM_PROVIDERS", "openai")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm.close_pool()
    yield
    llm.close_pool()


def test_prewarm_opens_a_connection_that_the_next_request_reuses(server, warm, monkeypatch):
    s = server()
    monkeypatch.setenv("LLM_BASE_URL", s.url)
    llm.prewarm(wait=True)
    assert s.heads == 1 and s.connections == 1 and s.requests == 0      # nothing sent but a HEAD: no tokens
    assert chat(s.url)["choices"]
    assert s.connections == 1 and s.requests == 1


def test_prewarm_is_rate_limited_to_once_a_minute(server, warm, monkeypatch):
    s = server()
    monkeypatch.setenv("LLM_BASE_URL", s.url)
    llm.prewarm(wait=True)
    llm.prewarm(wait=True)
    llm.prewarm()
    assert s.heads == 1
    monkeypatch.setattr(llm, "_last_prewarm", time.monotonic() - llm.PREWARM_MIN_INTERVAL_S - 1)
    llm.prewarm(wait=True)
    assert s.heads == 2


def test_prewarm_returns_at_once_and_works_in_the_background(server, warm, monkeypatch):
    s = server()
    monkeypatch.setenv("LLM_BASE_URL", s.url)
    t = time.time()
    llm.prewarm()
    assert time.time() - t < 0.5
    deadline = time.time() + 5
    while s.heads == 0 and time.time() < deadline:
        time.sleep(0.02)
    assert s.heads == 1


def test_prewarm_never_raises_and_skips_what_is_not_configured(monkeypatch, warm):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    llm.prewarm(wait=True)                                              # nothing configured: nothing to do
    monkeypatch.setattr(llm, "_last_prewarm", None)
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")         # nobody listens there
    llm.prewarm(wait=True)
    monkeypatch.setattr(llm, "_last_prewarm", None)
    monkeypatch.setattr(llm, "_prewarm_targets", lambda: 1 / 0)
    llm.prewarm(wait=True)


def test_prewarm_does_not_start_the_local_model(monkeypatch, warm):
    monkeypatch.setenv("LLM_PROVIDERS", "mlx")
    monkeypatch.setattr(llm._mlx, "ensure", lambda *a, **k: pytest.fail("the local model must not be started"))
    assert llm._prewarm_targets() == []
    llm.prewarm(wait=True)


def test_prewarm_targets_the_hosted_providers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "groq,openai,mlx")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    assert llm._prewarm_targets() == ["https://api.groq.com/", "http://localhost:11434/"]
