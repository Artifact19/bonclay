"""The local-model request lock: a request that times out, fails, or is abandoned by its consumer must free the slot,
and a call queued behind a stuck one must give up instead of waiting forever. The server is a real HTTP server that
stalls."""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from services import llm
from tests.test_llm_stream import chunk, sse


class SlowServer:
    """Answers `stall` POSTs by sending nothing for `hang` seconds, then replies normally."""

    def __init__(self, stalls=1, hang=3.0):
        self.stalls, self.hang = stalls, hang
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                if outer.stalls > 0:
                    outer.stalls -= 1
                    time.sleep(outer.hang)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(sse(chunk("ok"), "data: [DONE]"))

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
def local(monkeypatch):
    server = SlowServer()
    monkeypatch.setattr(llm._mlx, "ensure", lambda *a, **k: server.url)
    monkeypatch.setattr(llm, "LOCAL_REQUEST_TIMEOUT", 0.3)
    yield server
    server.close()
    assert not llm._mlx.request_lock.locked()


def drain(it):
    return "".join(it)


def test_stream_that_times_out_releases_the_lock_for_the_next_request(local):
    info = {}
    with pytest.raises(Exception):
        drain(llm._mlx_stream([{"role": "user", "content": "hi"}], info, temperature=0, max_tokens=5))
    assert not llm._mlx.request_lock.locked() and info["lock_held"] is False
    assert drain(llm._mlx_stream([{"role": "user", "content": "hi"}], {}, temperature=0, max_tokens=5)) == "ok"


def test_blocking_chat_that_times_out_releases_the_lock(local):
    with pytest.raises(Exception):
        llm._mlx_chat([{"role": "user", "content": "hi"}], json_mode=False, temperature=0, max_tokens=5)
    assert not llm._mlx.request_lock.locked()


def test_closing_a_stream_early_releases_the_lock(local):
    local.stalls = 0
    gen = llm._mlx_stream([{"role": "user", "content": "hi"}], {}, temperature=0, max_tokens=5)
    assert next(gen) == "ok"
    assert llm._mlx.request_lock.locked()
    gen.close()  # GeneratorExit at the yield
    assert not llm._mlx.request_lock.locked()


def test_a_request_queued_behind_a_stuck_one_gives_up_instead_of_hanging(local, monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_QUEUE_S", "0.2")
    assert llm._mlx.request_lock.acquire(timeout=1)  # another request holds the slot
    try:
        t = time.time()
        with pytest.raises(TimeoutError):
            drain(llm._mlx_stream([{"role": "user", "content": "hi"}], {}, temperature=0, max_tokens=5))
        assert time.time() - t < 2
    finally:
        llm._mlx.request_lock.release()
    local.stalls = 0
    assert drain(llm._mlx_stream([{"role": "user", "content": "hi"}], {}, temperature=0, max_tokens=5)) == "ok"


def test_chat_stream_total_timeout_frees_the_local_model(local, monkeypatch):
    monkeypatch.setattr(llm, "_order", lambda: ["mlx"])
    monkeypatch.setitem(llm.PROVIDERS, "mlx", (lambda: None, llm.PROVIDERS["mlx"][1]))
    stream = llm.chat_stream([{"role": "user", "content": "hi"}], total_timeout=0.1)
    with pytest.raises(llm.LLMUnavailable):  # the stalled server fails the only provider
        drain(stream)
    assert not llm._mlx.request_lock.locked()
    local.stalls = 0
    assert drain(llm.chat_stream([{"role": "user", "content": "hi"}])) == "ok"
