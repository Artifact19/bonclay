"""Outbound integrations: the signed webhook and Google Sheets logging (services/integrations.py, routers/integrations.py).

The receiver is a real HTTP server on 127.0.0.1 (allowed because the tests set BONCLAY_ALLOW_LOCAL_WEBHOOKS=on); Google
is replaced at its single HTTP function with a fake that checks the sign-in JWT and records the appends."""
import asyncio
import json
import os
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from services import integrations as svc
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import rig  # noqa: F401  (rig is a fixture)
from tests.test_turn_tools import Model, appt, booking_plan, next_day, post_blocking  # noqa: F401

DEFAULT_BACKOFF, DEFAULT_TIMEOUT = svc.BACKOFF_S, svc.WEBHOOK_TIMEOUT_S  # before any fixture patches them
SHEET_ID = "1" + "a" * 40
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit#gid=0"


# ------------------------------------------------------------------------------------------------ fixtures

@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("BONCLAY_INTEGRATIONS_DIR", str(tmp_path / "integrations"))
    monkeypatch.setenv("BONCLAY_ALLOW_LOCAL_WEBHOOKS", "on")
    monkeypatch.delenv("BONCLAY_SECRETS_KEY", raising=False)
    monkeypatch.setattr(svc, "BACKOFF_S", (0.02, 0.02, 0.02))
    svc.dispatcher.clear()
    svc._log_cache.clear()
    svc._token_cache.clear()
    svc._sheets_times.clear()
    yield
    svc.dispatcher.wait_idle(5)
    svc.dispatcher.clear()


class Receiver:
    """A webhook receiver. `script` lists what to do with request 1, 2, ...: an int status, "redirect" or "slow"."""

    def __init__(self, script=()):
        self.requests, self.script = [], list(script)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.requests.append({"headers": {k.lower(): v for k, v in self.headers.items()}, "body": body,
                                       "path": self.path})
                i = len(outer.requests) - 1
                action = outer.script[i] if i < len(outer.script) else 200
                if action == "slow":
                    time.sleep(1.0)
                    action = 200
                if action == "redirect":
                    self.send_response(302)
                    self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
                    self.end_headers()
                    return
                self.send_response(action)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/hook"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def json(self, i=-1):
        return json.loads(self.requests[i]["body"])


@pytest.fixture()
def receiver():
    r = Receiver()
    yield r
    r.close()


@pytest.fixture()
def make_receiver():
    made = []

    def make(script=()):
        r = Receiver(script)
        made.append(r)
        return r
    yield make
    for r in made:
        r.close()


@pytest.fixture(scope="module")
def sa_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    public = key.public_key()
    info = {"type": "service_account", "project_id": "demo", "private_key_id": "kid1", "private_key": pem,
            "client_email": "logger@demo.iam.gserviceaccount.com"}
    return SimpleNamespace(json=json.dumps(info), info=info, public=public, pem=pem)


class FakeGoogle:
    """Stands in for services.integrations._google_http."""

    def __init__(self, sa_key, monkeypatch):
        self.sa, self.appends, self.tokens, self.error = sa_key, [], 0, None
        monkeypatch.setattr(svc, "_google_http", self)

    def __call__(self, method, url, *, headers=None, json_body=None, data=None, timeout=10):
        if url == svc.GOOGLE_TOKEN_URL:
            assert data["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
            claims = jwt.decode(data["assertion"], self.sa.public, algorithms=["RS256"], audience=svc.GOOGLE_TOKEN_URL)
            assert claims["iss"] == self.sa.info["client_email"] and claims["scope"] == svc.SHEETS_SCOPE
            self.tokens += 1
            return 200, {"access_token": f"tok{self.tokens}", "expires_in": 3600}
        if self.error:
            return self.error
        assert headers["Authorization"].startswith("Bearer tok")
        self.appends.append({"url": url, "rows": json_body["values"]})
        return 200, {"updates": {"updatedRows": len(json_body["values"])}}


def fake_run(agent, *, test=False, tool_objs=(), text="Book me for Monday, my number is 9876543210"):
    return SimpleNamespace(agent=agent, agent_id="agent-1", user_id="local-user", local=True, is_test=test,
                           conversation_id="conv-1", language="en", user_text=text, timings={"llm_ms": 5},
                           tool_objs=list(tool_objs))


def integrations_row(hook_url, **wh):
    cfg = svc.default_cfg()
    cfg["webhook"].update({"enabled": True, "url": hook_url, "secret_set": True,
                           "secret_created_at": "2026-01-01T00:00:00+00:00", **wh})
    return {"name": "Reception", "integrations": cfg}


# ------------------------------------------------------------------------------------------- the signature

def test_signature_covers_timestamp_and_raw_body():
    body = b'{"event":"turn.completed"}'
    sig = svc.sign_body("whsec_x", 1700000000, body)
    assert sig.startswith("sha256=") and len(sig) == 7 + 64
    assert sig == svc.sign_body("whsec_x", 1700000000, body)          # deterministic
    assert svc.verify_signature("whsec_x", body, 1700000000, sig, now=1700000100)
    assert not svc.verify_signature("whsec_y", body, 1700000000, sig, now=1700000100)           # wrong secret
    assert not svc.verify_signature("whsec_x", body + b" ", 1700000000, sig, now=1700000100)    # body changed
    assert not svc.verify_signature("whsec_x", body, 1700000001, sig, now=1700000100)           # timestamp changed
    assert not svc.verify_signature("whsec_x", body, "soon", sig, now=1700000100)
    assert not svc.verify_signature("whsec_x", body, 1700000000, "", now=1700000100)


def test_replay_window_is_five_minutes():
    body = b"{}"
    sig = svc.sign_body("s", 1000, body)
    assert svc.verify_signature("s", body, 1000, sig, now=1000 + 299)
    assert svc.verify_signature("s", body, 1000, sig, now=1000 + 300)
    assert not svc.verify_signature("s", body, 1000, sig, now=1000 + 301)      # captured request replayed later
    assert not svc.verify_signature("s", body, 1000, sig, now=1000 - 301)      # from the future


def test_the_signature_matches_what_a_receiver_would_compute(receiver):
    """HMAC by hand, exactly as docs/INTEGRATIONS.md tells a receiver to do it."""
    import hashlib
    import hmac
    job = svc.Job("webhook", "a1", "u", True, None, "turn.completed", {"turn_id": "t1", "receipt_id": "r"}, url=receiver.url)
    body, headers = svc.webhook_request(job, "whsec_abc")
    expected = hmac.new(b"whsec_abc", headers["X-BonClay-Timestamp"].encode() + b"." + body, hashlib.sha256).hexdigest()
    assert headers["X-BonClay-Signature"] == "sha256=" + expected
    assert headers["X-BonClay-Event"] == "turn.completed" and headers["X-BonClay-Delivery"] == job.delivery_id


# -------------------------------------------------------------------------------------------------- SSRF

@pytest.mark.parametrize("ip", ["127.0.0.1", "::1", "10.0.0.5", "172.16.3.4", "192.168.1.1", "169.254.169.254",
                                "100.64.0.1", "0.0.0.0", "224.0.0.1", "fe80::1", "fd00::1", "::ffff:10.0.0.1",
                                "::ffff:169.254.169.254", "240.0.0.1", "198.18.0.1"])
def test_non_public_addresses_are_blocked(ip):
    assert svc.ip_block_reason(ip) is not None


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"])
def test_public_addresses_are_allowed(ip):
    assert svc.ip_block_reason(ip) is None


def test_loopback_is_only_allowed_when_asked():
    assert svc.ip_block_reason("127.0.0.1", allow_loopback=True) is None
    assert svc.ip_block_reason("10.0.0.1", allow_loopback=True) is not None   # never a LAN address


@pytest.mark.parametrize("url", ["http://example.com/hook", "ftp://example.com/x", "file:///etc/passwd", "example.com/hook",
                                 "https://user:pw@example.com/hook", "https://example.com/hook#frag", "",
                                 "https://exa mple.com", "javascript:alert(1)", "http://169.254.169.254/latest/meta-data/"])
def test_bad_urls_are_refused(url):
    with pytest.raises(svc.IntegrationError):
        svc.parse_webhook_url(url)


def test_http_localhost_needs_the_local_switch(monkeypatch):
    assert svc.parse_webhook_url("http://localhost:9000/x")[1:] == ("localhost", 9000)
    monkeypatch.setenv("BONCLAY_ALLOW_LOCAL_WEBHOOKS", "off")
    with pytest.raises(svc.IntegrationError) as e:
        svc.parse_webhook_url("http://localhost:9000/x")
    assert "BONCLAY_ALLOW_LOCAL_WEBHOOKS" in e.value.message


def test_local_webhooks_default_follows_the_mode(monkeypatch):
    monkeypatch.delenv("BONCLAY_ALLOW_LOCAL_WEBHOOKS")
    monkeypatch.delenv("NEXT_PUBLIC_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    assert svc.local_webhooks_allowed() is True                      # local mode
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    assert svc.local_webhooks_allowed() is False                     # cloud deployment: off unless switched on
    monkeypatch.setenv("BONCLAY_ALLOW_LOCAL_WEBHOOKS", "on")
    assert svc.local_webhooks_allowed() is True


def fake_dns(monkeypatch, mapping):
    def getaddrinfo(host, port, *a, **k):
        if host not in mapping:
            try:
                import ipaddress
                ipaddress.ip_address(host)            # the HTTP client connects to the literal IP: no DNS involved
                return [(2, 1, 6, "", (host, port))]
            except ValueError:
                raise OSError("unknown host") from None
        return [(2, 1, 6, "", (ip, port)) for ip in mapping[host]]
    monkeypatch.setattr(svc.socket, "getaddrinfo", getaddrinfo)


def test_a_name_that_resolves_to_a_private_address_is_blocked(monkeypatch):
    fake_dns(monkeypatch, {"evil.example": ["10.1.2.3"], "meta.example": ["169.254.169.254"],
                           "mixed.example": ["8.8.8.8", "192.168.0.9"], "rebind.example": ["127.0.0.1"],
                           "good.example": ["93.184.216.34"], "numeric": ["127.0.0.1"]})
    for host in ("evil.example", "meta.example", "mixed.example", "rebind.example"):
        with pytest.raises(svc.IntegrationError) as e:
            svc.validate_webhook_url(f"https://{host}/hook")
        assert e.value.code == "WEBHOOK_URL_BLOCKED", host
    assert svc.validate_webhook_url("https://good.example/hook") == "https://good.example/hook"
    with pytest.raises(svc.IntegrationError) as e:
        svc.validate_webhook_url("https://unresolvable.example/hook")
    assert e.value.code == "WEBHOOK_UNREACHABLE"
    with pytest.raises(svc.IntegrationError):
        svc.validate_webhook_url("https://2130706433/hook")      # 127.0.0.1 written as one number
    with pytest.raises(svc.IntegrationError):
        svc.validate_webhook_url("https://[::ffff:169.254.169.254]/hook")


def test_delivery_re_resolves_and_refuses_a_name_that_changed(monkeypatch):
    fake_dns(monkeypatch, {"hook.example": ["93.184.216.34"]})
    assert svc.validate_webhook_url("https://hook.example/x")
    fake_dns(monkeypatch, {"hook.example": ["10.0.0.7"]})        # DNS rebinding after the URL was saved
    code, err, retryable = svc._post_pinned("https://hook.example/x", b"{}", {}, 1.0)
    assert code is None and "Blocked" in err and retryable is False


def test_the_connection_goes_to_the_vetted_ip_not_a_second_lookup(monkeypatch, receiver):
    fake_dns(monkeypatch, {"hook.test": ["127.0.0.1"]})
    monkeypatch.setattr(svc, "LOCAL_HOSTS", ("localhost", "127.0.0.1", "::1", "hook.test"))
    port = receiver.server.server_address[1]
    code, err, _ = svc._post_pinned(f"http://hook.test:{port}/p", b"{}", {"X-Test": "1"}, 2.0)
    assert code == 200 and err is None, err
    assert receiver.requests[0]["headers"]["host"] == f"hook.test:{port}"


def test_redirects_are_never_followed(make_receiver):
    r = make_receiver(["redirect"])
    code, err, retryable = svc._post_pinned(r.url, b"{}", {}, 2.0)
    assert code == 302 and "redirect" in err and retryable is False
    assert len(r.requests) == 1                      # the metadata address was never requested


def test_proxy_environment_is_ignored(monkeypatch, receiver):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    code, err, _ = svc._post_pinned(receiver.url, b"{}", {}, 2.0)
    assert code == 200 and err is None


# ------------------------------------------------------------------------- delivery, retries and the log

def run_dispatch(row, **kw):
    run = fake_run(row, **kw)
    svc.save_secrets("agent-1", {"webhook_secret": "whsec_test"}, local=True)
    svc.dispatch_turn(run, "turn-1", "Booked for Monday.", 321)
    assert svc.dispatcher.wait_idle(10)


def test_a_turn_is_delivered_signed_with_the_documented_payload(receiver):
    run_dispatch(integrations_row(receiver.url))
    [req] = receiver.requests
    h = req["headers"]
    assert h["x-bonclay-event"] == "turn.completed" and h["x-bonclay-delivery"]
    assert svc.verify_signature("whsec_test", req["body"], h["x-bonclay-timestamp"], h["x-bonclay-signature"])
    body = receiver.json()
    assert body["event"] == "turn.completed" and body["agent_id"] == "agent-1" and body["turn_id"] == "turn-1"
    assert body["conversation_id"] == "conv-1" and body["language"] == "en" and body["is_test"] is False
    assert body["caller_text"].endswith("9876543210") and body["agent_text"] == "Booked for Monday."
    assert body["tool_calls"] == [] and body["booking"] is None and body["created_at"]
    assert body["timings"] == {"llm_ms": 5, "total_ms": 321}
    [entry] = svc.get_log("agent-1", local=True)
    assert entry["status"] == "delivered" and entry["code"] == 200 and entry["attempts"] == 1
    assert entry["latency_ms"] is not None and entry["target"] == "webhook"
    assert receiver.url not in json.dumps(entry) and "9876543210" not in json.dumps(entry)  # no payload, no URL in the log


def test_a_failing_receiver_is_retried_with_backoff_then_succeeds(make_receiver):
    r = make_receiver([500, 503, 200])
    run_dispatch(integrations_row(r.url))
    assert len(r.requests) == 3
    assert len({q["headers"]["x-bonclay-delivery"] for q in r.requests}) == 1   # same delivery id: receivers can de-duplicate
    assert len({q["body"] for q in r.requests}) == 1                            # same body each time
    [entry] = svc.get_log("agent-1", local=True)
    assert entry["status"] == "delivered" and entry["attempts"] == 3


def test_retries_stop_after_three_and_the_failure_is_logged(make_receiver):
    r = make_receiver([500] * 10)
    run_dispatch(integrations_row(r.url))
    assert len(r.requests) == 4                       # the first try and three retries
    [entry] = svc.get_log("agent-1", local=True)
    assert entry["status"] == "failed" and entry["code"] == 500 and entry["attempts"] == 4
    assert entry["error"] == "HTTP 500"


def test_backoff_schedule_is_two_eight_thirty():
    assert DEFAULT_BACKOFF == (2.0, 8.0, 30.0) and DEFAULT_TIMEOUT == 5.0 and svc.LOG_KEEP == 50


def test_a_client_error_is_not_retried(make_receiver):
    r = make_receiver([410, 200])
    run_dispatch(integrations_row(r.url))
    assert len(r.requests) == 1
    [entry] = svc.get_log("agent-1", local=True)
    assert entry["status"] == "failed" and entry["code"] == 410 and entry["attempts"] == 1


def test_a_redirect_is_a_failure_without_retry(make_receiver):
    r = make_receiver(["redirect", 200])
    run_dispatch(integrations_row(r.url))
    assert len(r.requests) == 1
    [entry] = svc.get_log("agent-1", local=True)
    assert entry["status"] == "failed" and entry["code"] == 302 and "redirect" in entry["error"]


def test_a_slow_receiver_times_out_and_is_retried(make_receiver, monkeypatch):
    monkeypatch.setattr(svc, "WEBHOOK_TIMEOUT_S", 0.2)
    r = make_receiver(["slow", 200])
    run_dispatch(integrations_row(r.url))
    assert len(r.requests) == 2
    assert svc.get_log("agent-1", local=True)[0]["status"] == "delivered"


def test_a_dead_port_fails_and_logs_the_error_without_the_url():
    run_dispatch(integrations_row("http://127.0.0.1:9/secret-token-path"))
    [entry] = svc.get_log("agent-1", local=True)
    assert entry["status"] == "failed" and entry["attempts"] == 4 and entry["error"]
    assert "secret-token-path" not in json.dumps(entry)


def test_dispatch_returns_at_once_even_when_the_receiver_is_slow(make_receiver, monkeypatch):
    monkeypatch.setattr(svc, "WEBHOOK_TIMEOUT_S", 3.0)
    r = make_receiver(["slow"])
    svc.save_secrets("agent-1", {"webhook_secret": "s"}, local=True)
    started = time.time()
    svc.dispatch_turn(fake_run(integrations_row(r.url)), "t", "hi", 1)
    assert time.time() - started < 0.2                 # the turn does not wait for the receiver
    assert svc.dispatcher.wait_idle(10)
    assert len(r.requests) == 1


def test_the_log_keeps_the_newest_fifty():
    for i in range(60):
        svc.log_update("agent-1", "u", True, {"delivery_id": f"d{i}", "target": "webhook", "event": "turn.completed",
                                              "status": "delivered", "at": svc._iso(), "created_at": svc._iso()})
    rows = svc.get_log("agent-1", local=True)
    assert len(rows) == 50 and rows[0]["delivery_id"] == "d59" and rows[-1]["delivery_id"] == "d10"
    svc._log_cache.clear()                              # and it survives a restart (local mode keeps a file)
    assert len(svc.get_log("agent-1", local=True)) == 50


# ----------------------------------------------------------------------------- what is sent, and when

def test_test_calls_are_skipped_unless_included(receiver):
    run_dispatch(integrations_row(receiver.url), test=True)
    assert receiver.requests == [] and svc.get_log("agent-1", local=True) == []
    run_dispatch(integrations_row(receiver.url, include_test=True), test=True)
    assert len(receiver.requests) == 1 and receiver.json()["is_test"] is True


def test_nothing_is_sent_when_off_or_incomplete(receiver):
    for change in ({"enabled": False}, {"secret_set": False}):
        run_dispatch(integrations_row(receiver.url, **change))
    run_dispatch(integrations_row(""))
    assert receiver.requests == []
    svc.dispatch_turn(fake_run({"name": "x"}), "t", "hi", 1)       # an agent without integrations
    svc.dispatch_turn(SimpleNamespace(), "t", "hi", 1)             # a broken run: never raises
    assert svc.dispatcher.pending() == 0


def test_event_selection(receiver):
    run_dispatch(integrations_row(receiver.url, events=["tool.called"]))
    assert receiver.requests == []                                 # no tool call: nothing to send
    run_dispatch(integrations_row(receiver.url, events=["turn.completed"]))
    assert len(receiver.requests) == 1


def booking_call(**extra):
    value = {"booking_id": "A10", "name": "Priya", "phone": "9876543210", "doctor": "Dr. Meera Kulkarni",
             "date": "2026-10-19", "time": "10:30", "status": "booked", **extra}
    call = SimpleNamespace(name="book_appointment", ok=True, value=value, arguments={"phone": "9876543210", "name": "Priya"},
                           result="booking A10 with Dr. Meera Kulkarni on 2026-10-19 10:30")
    call.record = lambda: {"name": call.name, "arguments": call.arguments, "ok": True, "result": call.result}
    return call


def test_a_booking_sends_tool_called_and_turn_completed_with_the_booking(receiver):
    bad = SimpleNamespace(name="cancel_appointment", ok=False, value="no", arguments={}, result="failed: no")
    bad.record = lambda: {"name": bad.name, "arguments": {}, "ok": False, "result": bad.result}
    run_dispatch(integrations_row(receiver.url), tool_objs=[booking_call(), bad])
    events = sorted(receiver.json(i)["event"] for i in range(len(receiver.requests)))
    assert events == ["tool.called", "turn.completed"]               # the failed call produced no event
    by_event = {receiver.json(i)["event"]: receiver.json(i) for i in range(2)}
    assert by_event["tool.called"]["tool_calls"][0]["name"] == "book_appointment"
    assert by_event["tool.called"]["booking"]["booking_id"] == "A10"
    done = by_event["turn.completed"]
    assert len(done["tool_calls"]) == 2 and done["booking"]["date"] == "2026-10-19" and done["booking"]["time"] == "10:30"


def test_redact_phones_covers_text_tool_arguments_and_the_booking(receiver):
    run_dispatch(integrations_row(receiver.url, redact_phones=True), tool_objs=[booking_call()])
    for i in range(len(receiver.requests)):
        raw = receiver.requests[i]["body"].decode()
        assert "9876543210" not in raw
    done = [receiver.json(i) for i in range(2) if receiver.json(i)["event"] == "turn.completed"][0]
    assert "[phone]" in done["caller_text"] and done["booking"]["phone"] == "[phone]"
    assert done["booking"]["date"] == "2026-10-19"                  # dates and times survive
    assert done["tool_calls"][0]["arguments"]["phone"] == "[phone]"


@pytest.mark.parametrize("text,expected", [
    ("call me on 9876543210 tomorrow", "call me on [phone] tomorrow"),
    ("+91 98765 43210 please", "[phone] please"),
    ("my number is +44 20 7946 0958", "my number is [phone]"),
    ("book 2026-10-19 at 10:30", "book 2026-10-19 at 10:30"),
    ("booking A12 on 19 October", "booking A12 on 19 October"),
])
def test_phone_redaction(text, expected):
    assert svc.redact_phones(text) == expected


def test_redaction_is_off_by_default(receiver):
    run_dispatch(integrations_row(receiver.url))
    assert "9876543210" in receiver.requests[0]["body"].decode()


def test_the_test_event_is_signed_logged_and_not_retried(make_receiver):
    r = make_receiver([500, 200])
    svc.save_secrets("agent-1", {"webhook_secret": "whsec_t"}, local=True)
    cfg = svc.normalize_cfg(integrations_row(r.url)["integrations"])
    entry = svc.send_test_event("agent-1", "u", True, None, cfg, "Reception")
    assert entry["status"] == "failed" and entry["code"] == 500 and len(r.requests) == 1
    entry = svc.send_test_event("agent-1", "u", True, None, cfg, "Reception")
    assert entry["status"] == "delivered"
    q = r.requests[1]
    assert q["headers"]["x-bonclay-event"] == "test.ping"
    assert svc.verify_signature("whsec_t", q["body"], q["headers"]["x-bonclay-timestamp"], q["headers"]["x-bonclay-signature"])
    assert len(svc.get_log("agent-1", local=True)) == 2


# ---------------------------------------------------------------------------------------------- secrets

def test_local_secrets_are_a_private_file_and_gone_with_the_agent(tmp_path):
    svc.save_secrets("agent-1", {"webhook_secret": "whsec_zzz"}, local=True)
    path = svc.secrets_dir() / "agent-1.secrets.json"
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(svc.secrets_dir()).st_mode) == 0o700
    assert svc.load_secrets("agent-1", local=True) == {"webhook_secret": "whsec_zzz"}
    svc.log_update("agent-1", "u", True, {"delivery_id": "d", "target": "webhook", "event": "x", "status": "failed",
                                          "at": svc._iso(), "created_at": svc._iso()})
    svc.forget_agent("agent-1")
    assert not path.exists() and not (svc.secrets_dir() / "agent-1.log.json").exists()
    assert svc.load_secrets("agent-1", local=True) == {}


def test_agent_ids_cannot_escape_the_secrets_directory():
    with pytest.raises(svc.IntegrationError):
        svc.save_secrets("../../etc/passwd", {"a": "b"}, local=True)


def test_cloud_mode_refuses_to_store_secrets_without_an_encryption_key(monkeypatch):
    assert svc.secrets_storage_status(False)["available"] is False
    with pytest.raises(svc.SecretsUnavailable) as e:
        svc.save_secrets("agent-1", {"webhook_secret": "x"}, local=False)
    assert "BONCLAY_SECRETS_KEY" in e.value.message
    assert svc.save_secrets("agent-1", {}, local=False) is None        # deleting needs no key
    assert not (svc.secrets_dir() / "agent-1.secrets.json").exists()   # and nothing was written to disk


def test_cloud_mode_encrypts_secrets_at_rest(monkeypatch):
    monkeypatch.setenv("BONCLAY_SECRETS_KEY", "a long random passphrase")
    blob = svc.save_secrets("agent-1", {"webhook_secret": "whsec_plaintext"}, local=False)
    assert blob and "whsec_plaintext" not in blob
    assert svc.load_secrets("agent-1", local=False, enc=blob) == {"webhook_secret": "whsec_plaintext"}
    monkeypatch.setenv("BONCLAY_SECRETS_KEY", "another key")
    assert svc.load_secrets("agent-1", local=False, enc=blob) == {}    # a wrong key reads nothing


def test_a_cloud_turn_signs_with_the_decrypted_secret(monkeypatch, receiver):
    monkeypatch.setenv("BONCLAY_SECRETS_KEY", "k")
    row = integrations_row(receiver.url)
    row["integrations"]["secrets_enc"] = svc.save_secrets("agent-1", {"webhook_secret": "whsec_cloud"}, local=False)
    run = fake_run(row)
    run.local = False
    svc.dispatch_turn(run, "t", "hi", 1)
    assert svc.dispatcher.wait_idle(10)
    q = receiver.requests[0]
    assert svc.verify_signature("whsec_cloud", q["body"], q["headers"]["x-bonclay-timestamp"], q["headers"]["x-bonclay-signature"])


# --------------------------------------------------------------------------------------------- Sheets

def test_service_account_validation(sa_key):
    ok = svc.parse_service_account(sa_key.json)
    assert ok["client_email"] == "logger@demo.iam.gserviceaccount.com"
    for bad in ("not json", "[]", json.dumps({**sa_key.info, "type": "authorized_user"}),
                json.dumps({**sa_key.info, "client_email": "nope"}),
                json.dumps({**sa_key.info, "private_key": "-----BEGIN PRIVATE KEY-----\nzzz\n-----END PRIVATE KEY-----"})):
        with pytest.raises(svc.IntegrationError) as e:
            svc.parse_service_account(bad)
        assert e.value.code == "INVALID_SERVICE_ACCOUNT"
        assert "BEGIN" not in e.value.message


def test_spreadsheet_id_from_a_link_or_an_id():
    assert svc.spreadsheet_id_from(SHEET_URL) == SHEET_ID
    assert svc.spreadsheet_id_from(f"  {SHEET_ID} ") == SHEET_ID
    for bad in ("", "short", "https://example.com/nothing", "../../x" * 5):
        with pytest.raises(svc.IntegrationError):
            svc.spreadsheet_id_from(bad)
    assert svc.sheets_range("") == "A1" and svc.sheets_range("Bookings") == "'Bookings'!A1"
    assert svc.sheets_range("It's") == "'It''s'!A1"
    with pytest.raises(svc.IntegrationError):
        svc.clean_sheet_name("a!b")


def sheets_row_cfg(**over):
    cfg = svc.default_cfg()
    cfg["sheets"].update({"enabled": True, "spreadsheet_id": SHEET_ID, "credentials_set": True,
                          "service_account_email": "logger@demo.iam.gserviceaccount.com", **over})
    return {"name": "Reception", "integrations": cfg}


def test_a_turn_appends_a_row_with_a_signed_google_login(sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    svc.save_secrets("agent-1", {"service_account": svc.parse_service_account(sa_key.json)}, local=True)
    svc.dispatch_turn(fake_run(sheets_row_cfg(sheet_name="Calls"), tool_objs=[booking_call()]), "turn-1", "Done.", 99)
    assert svc.dispatcher.wait_idle(10)
    [a] = g.appends
    assert f"/{SHEET_ID}/values/" in a["url"] and a["url"].endswith(":append?valueInputOption=RAW&insertDataOption=INSERT_ROWS")
    assert "%27Calls%27%21A1" in a["url"]
    [row] = a["rows"]
    assert len(row) == len(svc.SHEETS_COLUMNS) == 9
    ts, agent, lang, caller, reply, bid, bdate, btime, receipt = row
    assert (agent, lang, reply, bid, bdate, btime) == ("Reception", "en", "Done.", "A10", "2026-10-19", "10:30")
    assert "9876543210" in caller and ts.endswith("+00:00")
    assert svc.get_log("agent-1", local=True)[0]["status"] == "delivered"


def test_values_are_sent_raw_so_a_caller_cannot_inject_a_formula(sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    svc.save_secrets("agent-1", {"service_account": svc.parse_service_account(sa_key.json)}, local=True)
    svc.dispatch_turn(fake_run(sheets_row_cfg(), text='=IMPORTXML("http://evil","//a")'), "t", "ok", 1)
    assert svc.dispatcher.wait_idle(10)
    assert "valueInputOption=RAW" in g.appends[0]["url"] and g.appends[0]["rows"][0][3].startswith("=")


def test_rows_due_together_are_one_batched_request(sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    svc.save_secrets("agent-1", {"service_account": svc.parse_service_account(sa_key.json)}, local=True)
    d = svc.Dispatcher(workers=1)
    # All five rows get the same due time (a slow machine would otherwise spread the submits across the batch window).
    frozen = svc._now()
    monkeypatch.setattr(svc, "_now", lambda: frozen)
    for i in range(5):
        job = svc.Job("sheets", "agent-1", "u", True, None, "turn.completed",
                      {"turn_id": f"t{i}", "created_at": svc._iso(), "caller_text": f"q{i}", "agent_text": "a"},
                      "Reception", sheet={"spreadsheet_id": SHEET_ID, "sheet_name": ""})
        d.submit(job, delay=0.15)
    monkeypatch.setattr(svc, "_now", lambda: frozen + 1.0)  # every row is now due
    assert d.wait_idle(10)
    assert len(g.appends) == 1 and [r[3] for r in g.appends[0]["rows"]] == [f"q{i}" for i in range(5)]
    assert g.tokens == 1
    assert [e["status"] for e in svc.get_log("agent-1", local=True)] == ["delivered"] * 5


def test_the_access_token_is_cached_across_appends(sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    creds = svc.parse_service_account(sa_key.json)
    for _ in range(3):
        svc.append_rows("agent-1", creds, SHEET_ID, "", [["x"]])
    assert g.tokens == 1 and len(g.appends) == 3


def test_an_expired_token_is_refreshed_once(sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    creds = svc.parse_service_account(sa_key.json)
    real = g.__call__
    seen = {"n": 0}

    def flaky(method, url, **kw):
        if url != svc.GOOGLE_TOKEN_URL:
            seen["n"] += 1
            if seen["n"] == 1:
                return 401, {"error": {"message": "expired"}}
        return real(method, url, **kw)
    monkeypatch.setattr(svc, "_google_http", flaky)
    assert svc.append_rows("agent-1", creds, SHEET_ID, "", [["x"]]) == 1 and g.tokens == 2


def test_not_shared_reports_googles_exact_error_and_what_to_do(sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    g.error = (403, {"error": {"code": 403, "message": "The caller does not have permission", "status": "PERMISSION_DENIED"}})
    svc.save_secrets("agent-1", {"service_account": svc.parse_service_account(sa_key.json)}, local=True)
    cfg = svc.normalize_cfg(sheets_row_cfg()["integrations"])
    out = svc.check_sheets_connection("agent-1", "u", True, None, cfg, "Reception")
    assert out["ok"] is False and out["status"] == 403
    assert "The caller does not have permission" in out["error"]
    assert "logger@demo.iam.gserviceaccount.com" in out["hint"] and "Editor" in out["hint"]
    assert svc.get_log("agent-1", local=True)[0]["status"] == "failed"
    g.error = (404, {"error": {"message": "Requested entity was not found."}})
    assert "not found" in svc.check_sheets_connection("agent-1", "u", True, None, cfg, "R")["error"]
    g.error = (403, {"error": {"message": "Google Sheets API has not been used in project 1 before or it is disabled."}})
    assert "Enable the Google Sheets API" in svc.check_sheets_connection("agent-1", "u", True, None, cfg, "R")["hint"]


def test_connection_test_appends_a_marked_row(sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    svc.save_secrets("agent-1", {"service_account": svc.parse_service_account(sa_key.json)}, local=True)
    out = svc.check_sheets_connection("agent-1", "u", True, None, svc.normalize_cfg(sheets_row_cfg()["integrations"]), "Reception")
    assert out == {"ok": True, "error": None, "hint": "", "status": 200}
    assert g.appends[0]["rows"][0][3] == "BonClay test row"


def test_a_rejected_key_is_reported(sa_key, monkeypatch):
    monkeypatch.setattr(svc, "_google_http", lambda *a, **k: (400, {"error": "invalid_grant", "error_description": "Invalid JWT Signature."}))
    with pytest.raises(svc.SheetsError) as e:
        svc.append_rows("agent-1", svc.parse_service_account(sa_key.json), SHEET_ID, "", [["x"]])
    assert "Invalid JWT Signature" in e.value.message and e.value.retryable is False


def test_a_google_outage_is_retried_and_a_429_waits(sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    svc.save_secrets("agent-1", {"service_account": svc.parse_service_account(sa_key.json)}, local=True)
    real = g.__call__
    calls = {"n": 0}

    def outage(method, url, **kw):
        if url != svc.GOOGLE_TOKEN_URL:
            calls["n"] += 1
            if calls["n"] == 1:
                return 503, {"error": {"message": "backend error"}}
            if calls["n"] == 2:
                return 429, {"error": {"message": "Quota exceeded"}}
        return real(method, url, **kw)
    monkeypatch.setattr(svc, "_google_http", outage)
    monkeypatch.setattr(svc, "_sheets_error", lambda status, body, email, retry=None: svc.SheetsError(
        f"HTTP {status}", status=status, retryable=True, wait=0.02))
    svc.dispatch_turn(fake_run(sheets_row_cfg()), "t", "ok", 1)
    assert svc.dispatcher.wait_idle(10)
    assert len(g.appends) == 1
    assert svc.get_log("agent-1", local=True)[0]["status"] == "delivered" and svc.get_log("agent-1", local=True)[0]["attempts"] == 3


def test_appends_are_paced_below_googles_write_limit(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(svc, "_now", lambda: now[0])
    waits = [svc._sheets_pace("a@b.c") for _ in range(svc.SHEETS_APPENDS_PER_MIN)]
    assert waits == [0.0] * svc.SHEETS_APPENDS_PER_MIN
    assert svc._sheets_pace("a@b.c") > 0                      # the 51st in a minute has to wait
    assert svc._sheets_pace("other@b.c") == 0.0               # per service account
    now[0] += 61
    assert svc._sheets_pace("a@b.c") == 0.0


def test_sheets_skips_test_calls_and_can_redact(sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    svc.save_secrets("agent-1", {"service_account": svc.parse_service_account(sa_key.json)}, local=True)
    svc.dispatch_turn(fake_run(sheets_row_cfg(), test=True), "t1", "ok", 1)
    assert svc.dispatcher.wait_idle(10) and g.appends == []
    svc.dispatch_turn(fake_run(sheets_row_cfg(redact_phones=True)), "t2", "ok", 1)
    assert svc.dispatcher.wait_idle(10)
    assert "9876543210" not in json.dumps(g.appends) and "[phone]" in g.appends[0]["rows"][0][3]


# ----------------------------------------------------------------------------------------------- the API

def put(client, agent_id, section, **body):
    return client.put(f"/api/agents/{agent_id}/integrations/{section}", headers=LOCAL, json=body)


def test_webhook_settings_secret_shown_once_and_never_again(app_client, env, receiver):
    a = make_agent(app_client)
    base = f"/api/agents/{a['id']}/integrations"
    first = app_client.get(base, headers=LOCAL).json()
    assert first["webhook"]["enabled"] is False and first["webhook"]["secret_set"] is False
    assert first["privacy_note"] and first["secrets_storage"]["available"] is True

    r = put(app_client, a["id"], "webhook", url=receiver.url, enabled=True, redact_phones=True)
    assert r.status_code == 200, r.text
    out = r.json()
    secret = out["secret"]
    assert secret.startswith("whsec_") and len(secret) > 30
    assert out["webhook"]["enabled"] and out["webhook"]["secret_set"] and out["webhook"]["redact_phones"]
    assert out["webhook"]["url"] == receiver.url

    for resp in (app_client.get(base, headers=LOCAL), put(app_client, a["id"], "webhook", include_test=True),
                 app_client.get(f"/api/agents/{a['id']}", headers=LOCAL), app_client.get("/api/agents", headers=LOCAL),
                 app_client.get(base + "/deliveries", headers=LOCAL)):
        assert resp.status_code == 200 and secret not in resp.text and "secrets_enc" not in resp.text
    assert "secret" not in app_client.get(base, headers=LOCAL).json()
    assert svc.load_secrets(a["id"], local=True)["webhook_secret"] == secret

    rotated = app_client.post(base + "/webhook/rotate-secret", headers=LOCAL).json()
    assert rotated["secret"] != secret and svc.load_secrets(a["id"], local=True)["webhook_secret"] == rotated["secret"]
    assert rotated["secret"] not in app_client.get(base, headers=LOCAL).text


def test_webhook_url_validation_through_the_api(app_client, env, monkeypatch):
    a = make_agent(app_client)
    fake_dns(monkeypatch, {"intranet.example": ["10.0.0.9"], "ok.example": ["93.184.216.34"]})
    for url, code in (("http://example.com/hook", "INVALID_WEBHOOK_URL"), ("https://intranet.example/hook", "WEBHOOK_URL_BLOCKED"),
                      ("https://169.254.169.254/latest", "WEBHOOK_URL_BLOCKED"), ("https://u:p@ok.example/", "INVALID_WEBHOOK_URL")):
        r = put(app_client, a["id"], "webhook", url=url)
        assert r.status_code == 422 and r.json()["code"] == code, (url, r.text)
    assert app_client.get(f"/api/agents/{a['id']}/integrations", headers=LOCAL).json()["webhook"]["url"] == ""
    assert put(app_client, a["id"], "webhook", enabled=True).json()["code"] == "WEBHOOK_URL_REQUIRED"
    assert put(app_client, a["id"], "webhook", url="https://ok.example/hook", events=["nope"]).json()["code"] == "INVALID_EVENTS"
    ok = put(app_client, a["id"], "webhook", url="https://ok.example/hook", events=["tool.called"])
    assert ok.status_code == 200 and ok.json()["webhook"]["events"] == ["tool.called"] and ok.json()["secret"]
    cleared = put(app_client, a["id"], "webhook", url="")
    assert cleared.json()["webhook"]["secret_set"] is False and svc.load_secrets(a["id"], local=True) == {}


def test_other_users_agents_and_unknown_agents_are_not_reachable(app_client, env):
    for method, path in (("get", ""), ("put", "/webhook"), ("post", "/webhook/test"), ("post", "/sheets/test"),
                         ("get", "/deliveries")):
        kw = {"json": {}} if method == "put" else {}
        r = getattr(app_client, method)(f"/api/agents/nope/integrations{path}", headers=LOCAL, **kw)
        assert r.status_code == 404 and r.json()["code"] == "AGENT_NOT_FOUND", path
    a = make_agent(app_client)
    env.local_db.update("agents", a["id"], {"user_id": "someone-else"})
    assert app_client.get(f"/api/agents/{a['id']}/integrations", headers=LOCAL).status_code == 404


def test_send_test_event_through_the_api(app_client, env, receiver):
    a = make_agent(app_client)
    base = f"/api/agents/{a['id']}/integrations"
    assert app_client.post(base + "/webhook/test", headers=LOCAL).json()["code"] == "WEBHOOK_URL_REQUIRED"
    secret = put(app_client, a["id"], "webhook", url=receiver.url).json()["secret"]
    out = app_client.post(base + "/webhook/test", headers=LOCAL).json()
    assert out["ok"] is True and out["delivery"]["status"] == "delivered" and out["delivery"]["code"] == 200
    q = receiver.requests[0]
    assert q["headers"]["x-bonclay-event"] == "test.ping"
    assert svc.verify_signature(secret, q["body"], q["headers"]["x-bonclay-timestamp"], q["headers"]["x-bonclay-signature"])
    log = app_client.get(base + "/deliveries", headers=LOCAL).json()["deliveries"]
    assert len(log) == 1 and log[0]["event"] == "test.ping"


def test_the_key_is_write_only_and_stored_in_a_private_file(app_client, env, sa_key):
    a = make_agent(app_client)
    base = f"/api/agents/{a['id']}/integrations"
    r = put(app_client, a["id"], "sheets", service_account_json=sa_key.json, spreadsheet=SHEET_URL, sheet_name="Calls", enabled=True)
    assert r.status_code == 200, r.text
    sheets = r.json()["sheets"]
    assert sheets["credentials_set"] and sheets["service_account_email"] == "logger@demo.iam.gserviceaccount.com"
    assert sheets["spreadsheet_id"] == SHEET_ID and sheets["sheet_name"] == "Calls" and sheets["enabled"]
    for resp in (r, app_client.get(base, headers=LOCAL), app_client.get(f"/api/agents/{a['id']}", headers=LOCAL),
                 app_client.get("/api/agents", headers=LOCAL)):
        assert "BEGIN PRIVATE KEY" not in resp.text and "private_key" not in resp.text and "kid1" not in resp.text
    path = svc.secrets_dir() / f"{a['id']}.secrets.json"
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert "BEGIN PRIVATE KEY" not in json.dumps(env.local_db.get("agents", a["id"]))   # not in the agent store either
    deleted = app_client.delete(base + "/sheets", headers=LOCAL).json()
    assert deleted["sheets"]["credentials_set"] is False and svc.load_secrets(a["id"], local=True) == {}


def test_sheets_settings_are_validated(app_client, env, sa_key):
    a = make_agent(app_client)
    for body, code in (({"service_account_json": "{}"}, "INVALID_SERVICE_ACCOUNT"), ({"spreadsheet": "nope"}, "INVALID_SPREADSHEET"),
                       ({"sheet_name": "a!b"}, "INVALID_SHEET_NAME"), ({"enabled": True}, "SHEETS_INCOMPLETE")):
        r = put(app_client, a["id"], "sheets", **body)
        assert r.status_code == 422 and r.json()["code"] == code, (body, r.text)


def test_cloud_mode_without_a_key_refuses_to_store_secrets(app_client, env, sa_key, monkeypatch):
    a = make_agent(app_client)
    local_load = env._load_agent
    monkeypatch.setattr(env, "_storage_mode", lambda request: "cloud")

    async def load(request, agent_id, user_id, columns="*"):   # the cloud lookup needs Supabase: use the local row
        return await asyncio.to_thread(env.local_db.get, "agents", agent_id, user_id)
    monkeypatch.setattr(env, "_load_agent", load)
    r = put(app_client, a["id"], "sheets", service_account_json=sa_key.json)
    assert r.status_code == 409 and r.json()["code"] == "SECRETS_STORAGE_UNAVAILABLE"
    assert "BONCLAY_SECRETS_KEY" in r.json()["error"]
    assert "private_key" not in r.text and not (svc.secrets_dir() / f"{a['id']}.secrets.json").exists()
    r = put(app_client, a["id"], "webhook", url="http://127.0.0.1:9/x")
    assert r.status_code == 409 and r.json()["code"] == "SECRETS_STORAGE_UNAVAILABLE"


def test_sheets_test_connection_through_the_api(app_client, env, sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    a = make_agent(app_client)
    base = f"/api/agents/{a['id']}/integrations"
    assert app_client.post(base + "/sheets/test", headers=LOCAL).json()["ok"] is False
    put(app_client, a["id"], "sheets", service_account_json=sa_key.json, spreadsheet=SHEET_ID)
    g.error = (403, {"error": {"message": "The caller does not have permission"}})
    out = app_client.post(base + "/sheets/test", headers=LOCAL).json()
    assert out["ok"] is False and "does not have permission" in out["error"] and "logger@demo" in out["hint"]
    g.error = None
    assert app_client.post(base + "/sheets/test", headers=LOCAL).json()["ok"] is True and len(g.appends) == 1


def test_deleting_an_agent_removes_its_secrets_and_log(app_client, env, receiver):
    a = make_agent(app_client)
    put(app_client, a["id"], "webhook", url=receiver.url)
    app_client.post(f"/api/agents/{a['id']}/integrations/webhook/test", headers=LOCAL)
    assert (svc.secrets_dir() / f"{a['id']}.secrets.json").exists() and (svc.secrets_dir() / f"{a['id']}.log.json").exists()
    assert app_client.delete(f"/api/agents/{a['id']}", headers=LOCAL).status_code == 200
    assert not (svc.secrets_dir() / f"{a['id']}.secrets.json").exists() and not (svc.secrets_dir() / f"{a['id']}.log.json").exists()


def test_the_agent_object_is_unchanged(app_client, env):
    """Integrations live on their own endpoints: agent create/update/get keep their shape and ignore the field."""
    a = make_agent(app_client)
    assert "integrations" not in a
    r = app_client.put(f"/api/agents/{a['id']}", headers=LOCAL, json={"integrations": {"webhook": {"url": "https://x"}}})
    assert r.status_code == 200 and "integrations" not in r.json()


# --------------------------------------------------------------------------------- end to end: real turns

def configure(client, agent_id, receiver, **wh):
    r = put(client, agent_id, "webhook", url=receiver.url, enabled=True, **wh)
    assert r.status_code == 200, r.text
    return r.json()["secret"]


def test_a_booking_turn_reaches_the_webhook_with_receipt_and_timings(rig, appt, monkeypatch, receiver):
    a = make_agent(rig.client, tools=["appointments"], disclose_ai=False)
    secret = configure(rig.client, a["id"], receiver)
    Model(monkeypatch, booking_plan(next_day(0)))
    rig.asr_text = "Book me for Monday at 10:30, I am Priya Sharma, 9876543210"
    out = post_blocking(rig, a["id"]).json()
    assert svc.dispatcher.wait_idle(10)
    got = {receiver.json(i)["event"]: (receiver.json(i), receiver.requests[i]) for i in range(len(receiver.requests))}
    assert set(got) == {"tool.called", "turn.completed"}
    done, req = got["turn.completed"]
    assert svc.verify_signature(secret, req["body"], req["headers"]["x-bonclay-timestamp"], req["headers"]["x-bonclay-signature"])
    assert done["receipt_id"] == out["receipt_id"] and done["conversation_id"] == out["conversation_id"]
    assert done["agent_id"] == a["id"] and done["agent_text"] == out["agent_text"] and done["language"] == "en"
    assert done["caller_text"] == rig.asr_text and done["timings"]["total_ms"] == out["total_ms"]
    assert done["booking"]["booking_id"] == out["tool_calls"][0]["result"].split()[1] and done["booking"]["time"] == "10:30"
    assert got["tool.called"][0]["tool_calls"][0]["name"] == "book_appointment"
    [entry_a, entry_b] = svc.get_log(a["id"], local=True)
    assert {entry_a["status"], entry_b["status"]} == {"delivered"}


def test_a_test_call_is_not_sent_unless_included(rig, appt, monkeypatch, receiver):
    a = make_agent(rig.client, disclose_ai=False)
    configure(rig.client, a["id"], receiver)
    data = {"test_call": "true"}
    r = rig.client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL, data=data,
                        files={"audio": ("rec.wav", rig.wav, "audio/wav")})
    assert r.status_code == 200
    assert svc.dispatcher.wait_idle(10) and receiver.requests == []
    put(rig.client, a["id"], "webhook", include_test=True)
    rig.client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL, data=data,
                    files={"audio": ("rec.wav", rig.wav, "audio/wav")})
    assert svc.dispatcher.wait_idle(10) and len(receiver.requests) == 1 and receiver.json()["is_test"] is True


def test_a_streamed_turn_is_delivered_too(rig, receiver):
    a = make_agent(rig.client, disclose_ai=False)
    configure(rig.client, a["id"], receiver)
    r = rig.post(a["id"])
    assert r.status_code == 200
    assert svc.dispatcher.wait_idle(10) and len(receiver.requests) == 1
    body = receiver.json()
    assert body["event"] == "turn.completed" and body["agent_text"].startswith("The clinic opens")
    assert body["receipt_id"] and body["timings"]["total_ms"] > 0


def test_a_dead_receiver_never_breaks_or_delays_the_turn(rig, monkeypatch):
    a = make_agent(rig.client, disclose_ai=False)
    put(rig.client, a["id"], "webhook", url="http://127.0.0.1:9/dead", enabled=True)
    monkeypatch.setattr(svc, "BACKOFF_S", (5.0, 5.0, 5.0))
    started = time.time()
    r = rig.post(a["id"])
    assert r.status_code == 200 and "event: done" in r.text
    assert time.time() - started < 5                          # the retries wait in the background
    svc.dispatcher.clear()


def test_sheets_rows_from_a_real_turn(rig, sa_key, monkeypatch):
    g = FakeGoogle(sa_key, monkeypatch)
    a = make_agent(rig.client, disclose_ai=False)
    r = put(rig.client, a["id"], "sheets", service_account_json=sa_key.json, spreadsheet=SHEET_URL, enabled=True)
    assert r.status_code == 200
    assert rig.post(a["id"]).status_code == 200
    assert svc.dispatcher.wait_idle(10)
    [row] = g.appends[0]["rows"]
    assert row[1] == "Ria" and row[2] == "en" and row[3] == "When does the clinic open?" and row[8]   # receipt id
