"""Phone calls over Exotel (routers/phone_exotel.py): signed stream tokens, the outbound-call request, and a whole call
run against the real streaming turn with the same fakes as test_turn_stream.py (no model, no network): the agent speaks
first, answers what the caller says, and stops talking and clears Exotel's queue when the caller talks over it."""
import base64
import json
import os
import time

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from routers import phone_exotel as px
from services import phone_audio as pa
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import Rig

STREAM_SID = "MZ0000000000000000000000000000001"


@pytest.fixture()
def rig(app_client, env, monkeypatch, ref_wav, tmp_path, request):
    monkeypatch.setenv("WIDGET_TOKEN_SECRET", "test-secret")
    monkeypatch.setenv("PHONE_VAD", "energy")          # deterministic, no ONNX model needed
    monkeypatch.setenv("PHONE_PACE_SPEED", "8")        # the pacing clock runs 8 times faster than real time
    monkeypatch.setenv("PHONE_END_SILENCE_MS", "400")
    for name in ("PHONE_ALLOWED_NUMBERS", "PHONE_MAX_CALL_S", "EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_ACCOUNT_SID",
                 "EXOTEL_CALLER_ID", "EXOTEL_SUBDOMAIN", "PUBLIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)       # a real backend/.env must not leak into the tests
    monkeypatch.setenv("BONCLAY_PHONE_NUMBERS_FILE", str(tmp_path / "phone_numbers.json"))   # never the real file
    monkeypatch.setenv("PHONE_GREETING_DIR", str(tmp_path / "phone_greetings"))           # never the real cache
    monkeypatch.setenv("PHONE_PREWARM", "off")         # no background renders and warm-ups (tests of them turn it on)
    monkeypatch.delenv("PHONE_SPECULATIVE", raising=False)
    monkeypatch.delenv("PHONE_EARLY_SILENCE_MS", raising=False)
    from services import phone_greeting, provenance
    phone_greeting._mem.clear()
    phone_greeting._warm_until.clear()
    monkeypatch.setattr(provenance, "_data_dir", str(tmp_path))   # receipts of the test clips stay in tmp_path
    provenance._store.reset()
    request.addfinalizer(provenance._store.reset)                 # re-read from the real folder after the test
    monkeypatch.setenv("BONCLAY_CALL_RECORDS_FILE", str(tmp_path / "call_records.jsonl"))   # never the real log
    monkeypatch.setenv("BONCLAY_CALL_SETTINGS_FILE", str(tmp_path / "call_settings.json"))   # never the real settings
    px._calls_by_owner.clear()
    px._polls_by_owner.clear()
    px._known_calls.clear()
    return Rig(env, monkeypatch, app_client, ref_wav)


def token_for(agent, ttl=300, user="local-user"):
    return px.make_token(agent["id"], user, "local", ttl)


def pcm_tone(ms, amp=0.3, rate=8000, freq=220.0):
    t = np.arange(int(rate * ms / 1000)) / rate
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype("<i2").tobytes()


def pcm_silence(ms, rate=8000):
    return b"\x00\x00" * int(rate * ms / 1000)


def media(pcm, seq=1):
    return json.dumps({"event": "media", "sequence_number": str(seq), "stream_sid": STREAM_SID,
                       "media": {"chunk": str(seq), "timestamp": "0", "payload": base64.b64encode(pcm).decode()}})


def start_msg(rate="8000"):
    return json.dumps({"event": "start", "sequence_number": "1", "stream_sid": STREAM_SID,
                       "start": {"stream_sid": STREAM_SID, "call_sid": "CA1", "account_sid": "AC1",
                                 "from": "+919876543210", "to": "+918047491899", "custom_parameters": {},
                                 "media_format": {"encoding": "audio/x-raw", "sample_rate": rate, "bit_rate": "16"}}})


def feed(ws, pcm, frame_ms=100):
    step = int(8000 * 2 * frame_ms / 1000)
    for i in range(0, len(pcm), step):
        ws.send_text(media(pcm[i:i + step]))


def read_until(ws, stop, limit=400):
    """Messages from the server until `stop(message)` is true (the matching one is last). A watchdog closes the socket after
    20 s, so a feature that broke makes the test fail instead of waiting for a message that never comes."""
    import threading
    watchdog = threading.Timer(20.0, lambda: ws.close())
    watchdog.daemon = True
    watchdog.start()
    out = []
    for _ in range(limit):
        try:
            m = json.loads(ws.receive_text())
        except WebSocketDisconnect:
            raise AssertionError(f"the call went quiet; saw {[x['event'] for x in out][-10:]}") from None
        out.append(m)
        if stop(m):
            watchdog.cancel()
            return out
    watchdog.cancel()
    raise AssertionError(f"never saw the expected message in {limit}: {[x['event'] for x in out][-10:]}")


def is_mark(m):
    return m["event"] == "mark"


# ------------------------------------------------------------------------------------------------ tokens

def test_a_token_round_trips_and_names_the_agent(rig):
    t = px.make_token("a1", "u1", "local", 60, now=1000)
    assert px.read_token(t, now=1010) == {"a": "a1", "u": "u1", "m": "local", "x": 1060}


def test_expired_tampered_and_foreign_tokens_are_refused(rig, monkeypatch):
    t = px.make_token("a1", "u1", "local", 60, now=1000)
    assert px.read_token(t, now=1061) is None                       # expired
    payload, mac = t.split(".")
    other = px.make_token("a2", "u1", "local", 60, now=1000).split(".")[0]
    assert px.read_token(f"{other}.{mac}", now=1010) is None         # another payload, the first one's signature
    assert px.read_token(payload + ".AAAA", now=1010) is None
    assert px.read_token("", now=1) is None and px.read_token("nonsense", now=1) is None
    monkeypatch.setenv("WIDGET_TOKEN_SECRET", "another-secret")
    import routers.widget as w
    monkeypatch.setattr(w, "_secret", lambda: b"another-secret")
    assert px.read_token(t, now=1010) is None                       # signed with a different secret


# ------------------------------------------------------------------------------------- owner routes

CONFIG = {"EXOTEL_API_KEY": "k", "EXOTEL_API_TOKEN": "t", "EXOTEL_ACCOUNT_SID": "acme",
          "EXOTEL_CALLER_ID": "08047491899", "PUBLIC_BASE_URL": "https://abc.ngrok.app",
          "PHONE_ALLOWED_NUMBERS": "+919876543210"}


def configure(monkeypatch, **extra):
    for k, v in {**CONFIG, **extra}.items():
        monkeypatch.setenv(k, v)


def test_status_lists_what_is_missing(rig, monkeypatch):
    for k in CONFIG:
        monkeypatch.delenv(k, raising=False)
    body = rig.client.get("/api/phone/exotel/status", headers=LOCAL).json()
    assert body["configured"] is False and "EXOTEL_API_KEY" in body["missing"] and "PUBLIC_BASE_URL" in body["missing"]
    configure(monkeypatch)
    assert rig.client.get("/api/phone/exotel/status", headers=LOCAL).json()["configured"] is True


def test_the_stream_url_is_wss_and_carries_a_valid_token(rig, monkeypatch):
    configure(monkeypatch)
    a = make_agent(rig.client)
    r = rig.client.post("/api/phone/exotel/inbound-link", headers=LOCAL, json={"agent_id": a["id"]})
    assert r.status_code == 200, r.text
    url = r.json()["stream_url"]
    assert url.startswith("wss://abc.ngrok.app/api/phone/exotel/stream/")
    claims = px.read_token(url.rsplit("/", 1)[1])
    assert claims["a"] == a["id"] and claims["x"] > time.time() + 20 * 86400


class FakeHttp:
    sent = []
    status = 200
    body = {"Call": {"Sid": "CAabc123"}}
    gets = []
    get_status = 200
    get_body = {"Call": {"Sid": "CAabc123", "Status": "ringing"}}
    get_error = False

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, auth=None, data=None):
        FakeHttp.sent.append({"url": url, "auth": auth, "data": data})
        import httpx
        return httpx.Response(FakeHttp.status, json=FakeHttp.body)

    async def get(self, url, auth=None):
        FakeHttp.gets.append({"url": url, "auth": auth})
        import httpx
        if FakeHttp.get_error:
            raise httpx.ConnectError("down")
        if isinstance(FakeHttp.get_body, str):
            return httpx.Response(FakeHttp.get_status, text=FakeHttp.get_body)
        return httpx.Response(FakeHttp.get_status, json=FakeHttp.get_body)


@pytest.fixture()
def exotel(monkeypatch):
    import httpx
    FakeHttp.sent, FakeHttp.status, FakeHttp.body = [], 200, {"Call": {"Sid": "CAabc123"}}
    FakeHttp.gets, FakeHttp.get_status, FakeHttp.get_error = [], 200, False
    FakeHttp.get_body = {"Call": {"Sid": "CAabc123", "Status": "ringing"}}
    monkeypatch.setattr(httpx, "AsyncClient", FakeHttp)
    return FakeHttp


def call(rig, agent, **over):
    body = {"agent_id": agent["id"], "to": "+91 98765 43210", "caller_agreed": True, **over}
    return rig.client.post("/api/phone/exotel/call", headers=LOCAL, json=body)


def test_a_call_asks_exotel_to_stream_to_us(rig, exotel, monkeypatch):
    configure(monkeypatch)
    a = make_agent(rig.client)
    r = call(rig, a)
    assert r.status_code == 200, r.text
    assert r.json()["call_sid"] == "CAabc123" and "98765" not in json.dumps(r.json())   # the number is masked
    sent = exotel.sent[0]
    assert sent["url"] == "https://api.in.exotel.com/v1/Accounts/acme/Calls/connect"
    assert sent["auth"] == ("k", "t")
    d = sent["data"]
    assert d["From"] == "+919876543210" and d["CallerId"] == "08047491899" and d["StreamType"] == "bidirectional"
    assert d["StreamUrl"].startswith("wss://abc.ngrok.app/api/phone/exotel/stream/")
    claims = px.read_token(d["StreamUrl"].rsplit("/", 1)[1])
    assert claims["a"] == a["id"] and claims["x"] < time.time() + 16 * 60    # a call token lives minutes, not days


def test_calling_needs_consent_a_real_number_and_a_configured_account(rig, exotel, monkeypatch):
    a = make_agent(rig.client)
    configure(monkeypatch)
    add(rig, label="Saved", number="+91 90000 11111")
    assert call(rig, a, to="+919000011111", caller_agreed=False).json()["code"] == "CONSENT_REQUIRED"   # saved: per call
    assert call(rig, a, to="98765").json()["code"] == "BAD_NUMBER"
    assert call(rig, a, to="+0123456789").json()["code"] == "BAD_NUMBER"
    assert call(rig, a, to="abc").json()["code"] == "BAD_NUMBER"
    monkeypatch.delenv("EXOTEL_API_TOKEN")
    r = call(rig, a)
    assert r.status_code == 503 and "EXOTEL_API_TOKEN" in r.json()["error"]
    assert exotel.sent == []   # nothing reached Exotel in any of these


def test_the_allow_list_and_the_hourly_limit_hold(rig, exotel, monkeypatch):
    configure(monkeypatch, PHONE_ALLOWED_NUMBERS="+919876543210, +919000000000")
    a = make_agent(rig.client)
    assert call(rig, a, to="+919111111111").json()["code"] == "NUMBER_NOT_ALLOWED"
    for _ in range(px.CALLS_PER_HOUR):
        assert call(rig, a).status_code == 200
    assert call(rig, a).status_code == 429


def test_someone_elses_agent_cannot_be_called_and_exotel_errors_are_passed_on(rig, exotel, monkeypatch):
    configure(monkeypatch)
    assert call(rig, {"id": "00000000-0000-0000-0000-000000000000"}).status_code == 404
    a = make_agent(rig.client)
    exotel.status, exotel.body = 401, {"message": "bad key"}
    r = call(rig, a)
    assert r.status_code == 502 and r.json()["code"] == "EXOTEL_REFUSED"


def test_an_empty_allow_list_lets_any_valid_number_be_called(rig, exotel, monkeypatch):
    configure(monkeypatch)
    monkeypatch.delenv("PHONE_ALLOWED_NUMBERS")
    a = make_agent(rig.client)
    assert call(rig, a, to="09111111111").status_code == 200
    assert exotel.sent[0]["data"]["From"] == "+919111111111"
    assert call(rig, a, caller_agreed=False).json()["code"] == "CONSENT_REQUIRED"   # consent is still needed
    assert call(rig, a, to="98765").json()["code"] == "BAD_NUMBER"


@pytest.mark.parametrize("raw", ["+919876543210", "919876543210", "09876543210", "9876543210", "+91 98765-43210",
                                 "(+91) 98765 43210", "0091 98765 43210", "098765 43210"])
def test_indian_numbers_normalise_to_the_same_e164(raw):
    assert px.normalise_number(raw) == "+919876543210"


def test_normalise_number_keeps_foreign_numbers_and_rejects_junk():
    assert px.normalise_number("+1 (415) 555-0132") == "+14155550132"
    for bad in ("", "98765", "+0123456789", "abc", "12345678901234567", "+91"):
        assert px.normalise_number(bad) is None, bad


def test_the_allow_list_and_the_request_are_compared_after_normalising(rig, exotel, monkeypatch):
    configure(monkeypatch, PHONE_ALLOWED_NUMBERS="09876543210, 91 90000-00000")
    a = make_agent(rig.client)
    for to in ("+919876543210", "9876543210", "919876543210", "+91 (98765) 43210"):
        r = call(rig, a, to=to)
        assert r.status_code == 200, (to, r.text)
    assert {m["data"]["From"] for m in exotel.sent} == {"+919876543210"}
    assert call(rig, a, to="09000000000").status_code == 200
    r = call(rig, a, to="+919111111111")
    assert r.status_code == 403 and r.json()["code"] == "NUMBER_NOT_ALLOWED"


def test_status_offers_the_allow_list_masked_and_inbound_readiness(rig, monkeypatch):
    configure(monkeypatch, PHONE_ALLOWED_NUMBERS="09876543210, +919000000000, +919876543210")
    body = rig.client.get("/api/phone/exotel/status", headers=LOCAL).json()
    assert [{k: e[k] for k in ("number", "masked", "source")} for e in body["allowed_numbers"]] == [
        {"number": "+919876543210", "masked": "+91•••••43210", "source": "env"},
        {"number": "+919000000000", "masked": "+91•••••00000", "source": "env"}]   # normalised, deduplicated
    assert body["allowed_numbers_set"] is True and body["inbound_ready"] is True
    assert body["caller_id_masked"] == "+91•••••91899" and "8047" not in body["caller_id_masked"]
    monkeypatch.delenv("PHONE_ALLOWED_NUMBERS")
    monkeypatch.delenv("PUBLIC_BASE_URL")
    body = rig.client.get("/api/phone/exotel/status", headers=LOCAL).json()
    assert body["allowed_numbers"] == [] and body["allowed_numbers_set"] is False and body["inbound_ready"] is False


# ------------------------------------------------------------------------------------------- call status

def placed(rig, exotel, monkeypatch, **extra):
    configure(monkeypatch, **extra)
    r = call(rig, make_agent(rig.client))
    assert r.status_code == 200, r.text
    return r.json()["call_sid"]


def poll(rig, sid, headers=LOCAL):
    return rig.client.get(f"/api/phone/exotel/call/{sid}", headers=headers)


def test_call_status_reads_exotels_json(rig, exotel, monkeypatch):
    sid = placed(rig, exotel, monkeypatch)
    exotel.get_body = {"Call": {"Sid": sid, "Status": "completed", "Duration": "42", "StartTime": "2026-10-16 10:00:00",
                                "EndTime": "2026-10-16 10:00:42", "From": "+919876543210"}}
    r = poll(rig, sid)
    assert r.status_code == 200, r.text
    assert r.json() == {"call_sid": sid, "status": "completed", "duration_s": 42, "started_at": "2026-10-16 10:00:00",
                        "ended_at": "2026-10-16 10:00:42", "message": None}
    assert exotel.gets[0] == {"url": f"https://api.in.exotel.com/v1/Accounts/acme/Calls/{sid}.json", "auth": ("k", "t")}
    assert "98765" not in r.text


@pytest.mark.parametrize("raw,expected", [("queued", "queued"), ("ringing", "ringing"), ("in-progress", "in-progress"),
                                          ("In Progress", "in-progress"), ("no_answer", "no-answer"), ("busy", "busy"),
                                          ("cancelled", "canceled"), ("failed", "failed"), ("weird", "unknown"), (None, "unknown")])
def test_call_status_words_are_normalised(rig, exotel, monkeypatch, raw, expected):
    sid = placed(rig, exotel, monkeypatch)
    exotel.get_body = {"Call": {"Sid": sid, "Status": raw}}
    assert poll(rig, sid).json()["status"] == expected


def test_call_status_tolerates_xml_and_carries_the_failure_message(rig, exotel, monkeypatch):
    sid = placed(rig, exotel, monkeypatch)
    exotel.get_body = (f"<TwilioResponse><Call><Sid>{sid}</Sid><Status>failed</Status><Duration>0</Duration>"
                       "<StartTime>2026-10-16 10:00:00</StartTime><DetailedStatus>The number is switched off</DetailedStatus>"
                       "</Call></TwilioResponse>")
    body = poll(rig, sid).json()
    assert body["status"] == "failed" and body["duration_s"] == 0 and body["ended_at"] is None
    assert body["started_at"] == "2026-10-16 10:00:00" and body["message"] == "The number is switched off"
    exotel.get_body = "<html>nonsense</html>"
    assert poll(rig, sid).json()["status"] == "unknown"


def test_call_status_passes_on_exotel_trouble(rig, exotel, monkeypatch):
    sid = placed(rig, exotel, monkeypatch)
    exotel.get_status = 500
    r = poll(rig, sid)
    assert r.status_code == 502 and r.json()["code"] == "EXOTEL_REFUSED"
    exotel.get_error = True
    r = poll(rig, sid)
    assert r.status_code == 502 and r.json()["code"] == "EXOTEL_UNREACHABLE"


def test_only_the_owner_who_placed_a_call_can_read_it(rig, exotel, monkeypatch):
    sid = placed(rig, exotel, monkeypatch)
    assert poll(rig, "CAnotplaced").status_code == 404                     # never placed from here
    px._known_calls[sid]["owner"] = "someone-else"                         # placed by another user
    r = poll(rig, sid)
    assert r.status_code == 404 and r.json()["code"] == "CALL_NOT_FOUND"
    assert exotel.gets == []                                               # Exotel was never asked


def test_polling_is_rate_limited(rig, exotel, monkeypatch):
    sid = placed(rig, exotel, monkeypatch)
    for _ in range(px.POLLS_PER_MINUTE):
        assert poll(rig, sid).status_code == 200
    r = poll(rig, sid)
    assert r.status_code == 429 and r.json()["code"] == "RATE_LIMITED"
    assert len(exotel.gets) == px.POLLS_PER_MINUTE                         # the refused poll did not reach Exotel
    px._polls_by_owner["local-user"] = type(px._polls_by_owner["local-user"])(t - 61 for t in px._polls_by_owner["local-user"])
    assert poll(rig, sid).status_code == 200                               # a minute later it is allowed again


def test_the_call_register_forgets_the_oldest_calls(rig, exotel, monkeypatch):
    monkeypatch.setattr(px, "KNOWN_CALLS_MAX", 2)
    configure(monkeypatch)
    a = make_agent(rig.client)
    sids = []
    for n in range(3):
        exotel.body = {"Call": {"Sid": f"CA{n}"}}
        sids.append(call(rig, a).json()["call_sid"])
    assert list(px._known_calls) == sids[1:]


# ------------------------------------------------------------------------------------------- a whole call

def test_a_bad_token_or_a_foreign_agent_is_closed_before_accepting(rig):
    a = make_agent(rig.client)
    with pytest.raises(WebSocketDisconnect) as e:
        with rig.client.websocket_connect(f"{px.STREAM_PATH}/not-a-token"):
            pass
    assert e.value.code == 4401
    with pytest.raises(WebSocketDisconnect) as e:
        with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a, user='someone-else')}"):
            pass
    assert e.value.code == 4404


def test_the_agent_speaks_first_with_the_ai_disclosure_in_valid_phone_audio(rig):
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(start_msg())
        msgs = read_until(ws, is_mark)
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
    audio = [m for m in msgs if m["event"] == "media"]
    assert audio, "the agent said nothing"
    for m in audio:
        pcm = base64.b64decode(m["media"]["payload"])
        assert m["stream_sid"] == STREAM_SID
        assert len(pcm) % pa.SLIN_FRAME_BYTES == 0 and len(pcm) >= pa.MIN_CHUNK_BYTES   # what Exotel accepts
    assert any("AI" in t[0] or "assistant" in t[0].lower() for t in rig.spoken)           # the disclosure was spoken
    assert msgs[-1]["mark"]["name"] == "turn-1-end"


def test_the_agent_answers_what_the_caller_says(rig):
    rig.asr_text = "When does the clinic open?"
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)                                   # the greeting is done
        feed(ws, pcm_silence(300) + pcm_tone(1200) + pcm_silence(900))
        msgs = read_until(ws, is_mark)
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
    assert any(m["event"] == "media" for m in msgs)
    assert msgs[-1]["mark"]["name"] == "turn-2-end"
    asked = [q["messages"][-1]["content"] for q in rig.asked]
    assert any("When does the clinic open?" in c for c in asked)   # the caller's words reached the language model


def test_talking_over_the_agent_cuts_it_off_and_clears_the_line(rig, env, monkeypatch):
    long_sentence = np.full(24000 * 6, 0.05, dtype=np.float32)    # 6 s of audio per sentence
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": long_sentence)
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, lambda m: m["event"] == "media")           # the agent is talking
        feed(ws, pcm_tone(800))                                   # the caller talks over it
        msgs = read_until(ws, lambda m: m["event"] == "clear")
        assert msgs[-1] == {"event": "clear", "stream_sid": STREAM_SID}
        # nothing more is sent for the cancelled reply: the next message is only for the caller's new turn
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))


def test_hanging_up_ends_the_call_and_stops_the_turn(rig, env, monkeypatch):
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": np.full(24000 * 6, 0.05, dtype=np.float32))
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, lambda m: m["event"] == "media")
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
    # leaving the `with` block without a hang or an error is the assertion: the running turn was cancelled cleanly


def test_a_hello_while_the_agent_is_still_thinking_does_not_cancel_its_answer(rig, env, monkeypatch):
    import time as _t

    def slow_voice(model, text, lang="en"):
        _t.sleep(1.2)                                    # a slow voice, like a cloned one
        return np.full(2400, 0.05, dtype=np.float32)

    monkeypatch.setattr(env, "_kokoro_generate", slow_voice)
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        feed(ws, pcm_tone(500) + pcm_silence(600))          # "hello?" before the agent has said anything
        msgs = read_until(ws, is_mark)
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
    assert not any(m["event"] == "clear" for m in msgs)       # nothing to cut off yet
    assert any(m["event"] == "media" for m in msgs)
    assert msgs[-1]["mark"]["name"] == "turn-1-end"           # the first turn finished: it was not replaced


# ------------------------------------------------------------------------------------- numbers saved in the app

def add(rig, label="Arnav", number="+91 98765 43210", agreed=True):
    return rig.client.post("/api/phone/exotel/numbers", headers=LOCAL, json={"label": label, "number": number, "agreed": agreed})


def test_a_saved_number_is_listed_normalised_and_masked(rig):
    r = add(rig)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["number"] == "+919876543210" and body["label"] == "Arnav" and body["source"] == "saved" and "98765" not in body["masked"]
    status = rig.client.get("/api/phone/exotel/status", headers=LOCAL).json()
    [entry] = status["allowed_numbers"]
    assert entry["number"] == "+919876543210" and entry["source"] == "saved" and entry["id"] == body["id"] and entry["label"] == "Arnav"
    assert status["allowed_numbers_set"] is True
    assert rig.client.get("/api/phone/exotel/numbers", headers=LOCAL).json()["numbers"][0]["id"] == body["id"]


def test_saving_a_number_needs_consent_a_real_number_and_a_name(rig):
    assert add(rig, agreed=False).json()["code"] == "CONSENT_REQUIRED"
    assert add(rig, number="12345").json()["code"] == "BAD_NUMBER"
    assert add(rig, label="   ").json()["code"] == "BAD_LABEL"
    assert add(rig, label="x" * 41).json()["code"] == "BAD_LABEL"
    assert rig.client.get("/api/phone/exotel/numbers", headers=LOCAL).json()["numbers"] == []


def test_the_same_number_cannot_be_saved_twice_and_the_list_has_a_limit(rig, monkeypatch):
    assert add(rig).status_code == 200
    assert add(rig, label="Again", number="9876543210").json()["code"] == "NUMBER_EXISTS"          # the same number, written another way
    monkeypatch.setattr(px, "SAVED_NUMBERS_MAX", 2)
    assert add(rig, label="B", number="9000000001").status_code == 200
    assert add(rig, label="C", number="9000000002").json()["code"] == "TOO_MANY_NUMBERS"


def test_a_number_from_env_is_listed_but_cannot_be_removed_or_saved_again(rig, monkeypatch):
    monkeypatch.setenv("PHONE_ALLOWED_NUMBERS", "+919111111111")
    [entry] = rig.client.get("/api/phone/exotel/status", headers=LOCAL).json()["allowed_numbers"]
    assert entry["source"] == "env" and entry["id"] is None
    assert add(rig, number="9111111111").json()["code"] == "NUMBER_EXISTS"
    assert rig.client.delete("/api/phone/exotel/numbers/env", headers=LOCAL).json()["code"] == "NUMBER_FROM_ENV"
    saved = add(rig).json()
    assert [e["source"] for e in rig.client.get("/api/phone/exotel/numbers", headers=LOCAL).json()["numbers"]] == ["env", "saved"]
    assert saved["source"] == "saved"


def test_removing_a_saved_number_stops_calls_to_it(rig, exotel, monkeypatch):
    configure(monkeypatch)
    a = make_agent(rig.client)
    saved = add(rig, number="9222222222").json()
    assert saved["id"], saved
    assert call(rig, a, to="+919222222222").status_code == 200                       # saved: allowed
    r = call(rig, a, to="+919000000000")
    assert r.status_code == 403 and r.json()["code"] == "NUMBER_NOT_ALLOWED" and "Add it first" in r.json()["error"]
    assert rig.client.delete(f"/api/phone/exotel/numbers/{saved['id']}", headers=LOCAL).json() == {"removed": True}
    assert rig.client.delete(f"/api/phone/exotel/numbers/{saved['id']}", headers=LOCAL).json()["code"] == "NUMBER_NOT_FOUND"
    assert call(rig, a, to="+919222222222").json()["code"] == "NUMBER_NOT_ALLOWED"    # removed: no longer allowed


def test_the_saved_numbers_file_is_private_and_survives_a_restart(rig, tmp_path):
    add(rig)
    path = tmp_path / "phone_numbers.json"
    if os.name != "nt":
        assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert [r["number"] for r in px.saved_numbers()] == ["+919876543210"]


# ---- call records and the status callback -------------------------------------------------------------------------

def test_a_placed_call_is_recorded_and_listed_with_a_masked_number(rig, exotel, monkeypatch):
    configure(monkeypatch)
    a = make_agent(rig.client)
    assert call(rig, a).status_code == 200
    rows = rig.client.get("/api/phone/exotel/calls", headers=LOCAL).json()["calls"]
    assert len(rows) == 1 and rows[0]["call_sid"] == "CAabc123"
    assert rows[0]["direction"] == "outbound" and rows[0]["status"] == "queued" and rows[0]["agent_id"] == a["id"]
    assert "98765" not in json.dumps(rows) and rows[0]["to_masked"]          # the number is never stored in full


def test_the_call_does_not_send_a_status_callback_unless_asked(rig, exotel, monkeypatch):
    configure(monkeypatch)
    monkeypatch.delenv("EXOTEL_STATUS_CALLBACK", raising=False)
    call(rig, make_agent(rig.client))
    assert "StatusCallback" not in exotel.sent[0]["data"]      # a real Exotel account answered 500 with it


def test_the_call_asks_exotel_to_report_status_to_a_signed_url(rig, exotel, monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv("EXOTEL_STATUS_CALLBACK", "1")
    a = make_agent(rig.client)
    call(rig, a)
    url = exotel.sent[0]["data"]["StatusCallback"]
    assert url.startswith("https://abc.ngrok.app" + px.STATUS_CALLBACK_PATH + "/")
    claims = px.read_token(url.rsplit("/", 1)[1])
    assert claims["a"] == a["id"] and claims["u"] == "local-user"


def test_the_status_callback_updates_the_record_and_ignores_forged_or_late_updates(rig, exotel, monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv("EXOTEL_STATUS_CALLBACK", "1")
    a = make_agent(rig.client)
    call(rig, a)
    token = exotel.sent[0]["data"]["StatusCallback"].rsplit("/", 1)[1]
    url = f"{px.STATUS_CALLBACK_PATH}/{token}"
    assert rig.client.post(url, data={"CallSid": "CAabc123", "Status": "completed", "Duration": "42"}).json() == {"ok": True}
    row = rig.client.get("/api/phone/exotel/calls", headers=LOCAL).json()["calls"][0]
    assert row["status"] == "completed" and row["duration_s"] == 42
    rig.client.post(url, data={"CallSid": "CAabc123", "Status": "in-progress"})        # a late, older update
    assert rig.client.get("/api/phone/exotel/calls", headers=LOCAL).json()["calls"][0]["status"] == "completed"
    assert rig.client.post(f"{px.STATUS_CALLBACK_PATH}/forged.token", data={"CallSid": "CAx", "Status": "failed"}
                           ).json() == {"ok": False}
    assert len(rig.client.get("/api/phone/exotel/calls", headers=LOCAL).json()["calls"]) == 1
    stranger = px.make_token(a["id"], "someone-else", "local", 300)
    rig.client.post(f"{px.STATUS_CALLBACK_PATH}/{stranger}", data={"CallSid": "CAabc123", "Status": "failed"})
    assert rig.client.get("/api/phone/exotel/calls", headers=LOCAL).json()["calls"][0]["status"] == "completed"


def test_a_call_that_connects_is_recorded_with_its_conversation_and_how_it_ended(rig):
    from services import call_records
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
    time.sleep(0.3)
    rows = call_records.list_for("local-user")
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "in-progress" and row["hangup_reason"] == "caller_hung_up"
    assert row["direction"] == "inbound" and row["agent_id"] == a["id"] and row.get("conversation_id")


# ---- answering machines and key presses ---------------------------------------------------------------------------

def _outbound_call(rig, a, sid="CA1"):
    px._known_calls[sid] = {"owner": "local-user", "agent_id": a["id"]}


def _talk(ws):
    ws.send_text(start_msg())
    read_until(ws, is_mark)                                   # the greeting is done
    feed(ws, pcm_silence(300) + pcm_tone(1200) + pcm_silence(900))


def test_an_answering_machine_on_an_outbound_call_is_hung_up_on_and_recorded(rig):
    from services import call_records
    rig.asr_text = "Hello, the person you are calling is not available. Please leave a message after the beep."
    a = make_agent(rig.client)
    _outbound_call(rig, a)
    try:                                                       # the server ends the stream mid-way
        with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
            _talk(ws)
            for _ in range(200):
                ws.receive_text()
    except Exception:  # noqa: BLE001 - WebSocketDisconnect or a closed test stream: both mean the server hung up
        pass
    time.sleep(0.3)
    row = call_records.get("CA1")
    assert row["outcome"] == "voicemail" and row["hangup_reason"] == "voicemail" and row["direction"] == "outbound"
    assert rig.asked == []                                      # the language model was never asked to answer a machine


def test_the_same_words_on_an_inbound_call_are_answered_normally(rig):
    rig.asr_text = "Please leave a message after the beep"
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        _talk(ws)
        msgs = read_until(ws, is_mark)
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
    assert any(m["event"] == "media" for m in msgs) and rig.asked


def test_a_person_answering_an_outbound_call_is_not_treated_as_a_machine(rig):
    from services import call_records
    rig.asr_text = "Hello? Who is this, I am not available to talk for long"
    a = make_agent(rig.client)
    _outbound_call(rig, a)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        _talk(ws)
        msgs = read_until(ws, is_mark)
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
    assert any(m["event"] == "media" for m in msgs)
    time.sleep(0.3)
    assert call_records.get("CA1").get("outcome") != "voicemail"


def test_key_presses_are_counted_but_never_stored_and_zero_asks_for_a_person(rig):
    import sys
    from services import call_records
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        for digit in ("4", "2", "0"):
            ws.send_text(json.dumps({"event": "dtmf", "stream_sid": STREAM_SID, "dtmf": {"digit": digit, "duration": "250"}}))
        ws.send_text(json.dumps({"event": "dtmf", "stream_sid": STREAM_SID, "dtmf": {}}))   # malformed: ignored
        time.sleep(0.5)
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
    time.sleep(0.3)
    row = call_records.get("CA1")
    assert row["dtmf_presses"] == 3 and row["outcome"] == "human_requested"
    assert not any("digit" in k or "dtmf_digits" in k for k in row)            # only a count is kept
    conv = sys.modules["backend"].local_db.get("conversations", row["conversation_id"])
    assert conv.get("needs_human") is True and conv.get("handoff_at")


# ---- end_call and transfer_call --------------------------------------------------------------------------------------

def _live_session_id():
    from services import call_control
    ids = list(call_control._sessions)
    assert len(ids) == 1, ids
    return ids[0]


def _run_tool(tool, cid, agent_id):
    from types import SimpleNamespace
    from services import call_control
    return call_control.run(tool, {}, SimpleNamespace(conversation_id=cid, agent_id=agent_id))


def _closes_after_the_next_reply(ws):
    feed(ws, pcm_silence(300) + pcm_tone(1200) + pcm_silence(900))
    try:
        read_until(ws, is_mark)
        for _ in range(100):
            ws.receive_text()
    except Exception:  # noqa: BLE001 - the server ends the stream after the reply: that is what is expected
        return True
    return False


def test_the_call_tools_exist_only_on_a_phone_turn_and_transfer_only_with_a_number(env, tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from services import call_control, tools
    monkeypatch.setenv("BONCLAY_CALL_SETTINGS_FILE", str(tmp_path / "call_settings.json"))
    assert "call" in tools.known_servers() and tools.clean_tool_names(["call"]) == ["call"]
    assert tools.hub.specs_for(["call"]) == []                       # never a process: bound per phone turn

    def specs(source):
        run = SimpleNamespace(agent={"tools": ["call"]}, source=source, agent_id="ag1", conversation_id=None, tool_specs=[])
        asyncio.run(env._turn_tools(run))
        return [s.name for s in run.tool_specs]
    assert specs(None) == [] and specs("widget") == []
    assert specs("phone") == ["end_call"]
    call_control.set_transfer_number("ag1", "+919876543210")
    assert specs("phone") == ["end_call", "transfer_call"]
    call_control.set_transfer_number("ag1", None)
    assert specs("phone") == ["end_call"]


def test_a_tool_call_outside_a_live_call_says_so(env):
    from types import SimpleNamespace
    from services import call_control, tools
    spec = call_control.specs_for_run(SimpleNamespace(agent_id="x", conversation_id="nope"), tools)[0]
    out = tools.execute(spec, {}, previous_agent_text=None)
    assert out.ok is False and "not a live phone call" in str(out.value)


def test_end_call_hangs_up_after_the_agents_reply(rig):
    from services import call_records
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        ok, value = _run_tool("end_call", _live_session_id(), a["id"])
        assert ok is True and "goodbye" in value["message"]
        assert _closes_after_the_next_reply(ws)
    time.sleep(0.3)
    row = call_records.get("CA1")
    assert row["outcome"] == "ended_by_agent" and row["hangup_reason"] == "ended_by_agent"


def test_transfer_needs_a_number_and_never_takes_one_from_the_model(rig):
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        ok, value = _run_tool("transfer_call", _live_session_id(), a["id"])
        assert ok is False and "no transfer number" in value
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))


def test_transfer_in_callback_mode_promises_a_call_back_and_flags_a_person(rig, tmp_path, monkeypatch):
    import sys
    from services import call_control, call_records
    monkeypatch.setenv("BONCLAY_CALL_SETTINGS_FILE", str(tmp_path / "call_settings.json"))
    a = make_agent(rig.client)
    r = rig.client.put(f"/api/agents/{a['id']}/call-settings", headers=LOCAL, json={"transfer_number": "98765 00000"})
    assert r.status_code == 200 and r.json()["transfer_number"] == "+919876500000"
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        ok, value = _run_tool("transfer_call", _live_session_id(), a["id"])
        assert ok is True and "call them back" in value["message"]
        assert _closes_after_the_next_reply(ws)
    time.sleep(0.3)
    row = call_records.get("CA1")
    assert row["outcome"] == "transfer_requested" and row["hangup_reason"] == "callback_promised"
    assert row["transfer_to"] and "98765" not in row["transfer_to"] and "00000" not in row["transfer_to"].replace("*", "")[:6]
    conv = sys.modules["backend"].local_db.get("conversations", row["conversation_id"])
    assert conv.get("needs_human") is True


def test_transfer_in_flow_mode_ends_the_stream_so_the_exotel_flow_can_dial(rig, tmp_path, monkeypatch):
    from services import call_records
    monkeypatch.setenv("BONCLAY_CALL_SETTINGS_FILE", str(tmp_path / "call_settings.json"))
    monkeypatch.setenv("PHONE_TRANSFER_MODE", "flow")
    a = make_agent(rig.client)
    rig.client.put(f"/api/agents/{a['id']}/call-settings", headers=LOCAL, json={"transfer_number": "+919876500000"})
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        ok, value = _run_tool("transfer_call", _live_session_id(), a["id"])
        assert ok is True and "connected to a person" in value["message"]
        assert _closes_after_the_next_reply(ws)
    time.sleep(0.3)
    assert call_records.get("CA1")["hangup_reason"] == "transferred"


def test_call_settings_validate_clear_and_belong_to_the_owner(rig, tmp_path, monkeypatch):
    monkeypatch.setenv("BONCLAY_CALL_SETTINGS_FILE", str(tmp_path / "call_settings.json"))
    a = make_agent(rig.client)
    url = f"/api/agents/{a['id']}/call-settings"
    assert rig.client.get(url, headers=LOCAL).json()["transfer_number"] is None
    assert rig.client.put(url, headers=LOCAL, json={"transfer_number": "abc"}).json()["code"] == "BAD_NUMBER"
    assert rig.client.put(url, headers=LOCAL, json={"transfer_number": "9876543210"}).json()["transfer_number"] == "+919876543210"
    assert "9876543210" not in rig.client.get(url, headers=LOCAL).json()["transfer_number_masked"]
    assert rig.client.put(url, headers=LOCAL, json={"transfer_number": ""}).json()["transfer_number"] is None
    assert rig.client.get("/api/agents/00000000-0000-0000-0000-000000000000/call-settings", headers=LOCAL).status_code == 404


# ---- live monitoring and takeover ------------------------------------------------------------------------------------

def _wait_for_live(rig, n=1, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        calls = rig.client.get("/api/phone/exotel/live", headers=LOCAL).json()["calls"]
        if len(calls) == n:
            return calls
        time.sleep(0.05)
    raise AssertionError("no live call appeared")


def _monitor_url(rig, conversation_id):
    tok = rig.client.post(f"/api/phone/exotel/live/{conversation_id}/monitor-token", headers=LOCAL).json()["token"]
    return f"/api/phone/exotel/monitor/{tok}"


def _read_monitor(mon, want, limit=200):
    """Messages from the monitor socket until one matches `want(msg)`; returns (that message, all messages)."""
    seen = []
    for _ in range(limit):
        msg = json.loads(mon.receive_text())
        seen.append(msg)
        if want(msg):
            return msg, seen
    raise AssertionError(f"never saw the message, got {[m.get('type') for m in seen][:20]}")


def test_a_live_call_is_listed_and_listening_in_needs_the_owners_short_lived_token(rig):
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        live = _wait_for_live(rig)
        assert live[0]["agent_id"] == a["id"] and live[0]["human_active"] is False and live[0]["call_sid"] == "CA1"
        cid = live[0]["conversation_id"]
        assert rig.client.post("/api/phone/exotel/live/nope/monitor-token", headers=LOCAL).json()["code"] == "CALL_NOT_LIVE"
        good = px.make_monitor_token(cid, "local-user")
        assert px.read_monitor_token(good)["c"] == cid
        assert px.read_monitor_token(px.make_monitor_token(cid, "local-user", ttl=-5)) is None      # expired
        assert px.read_monitor_token(token_for(a)) is None                                         # a stream token is no monitor token
        assert px.read_token(good) is None                                                          # and the reverse
        with pytest.raises(WebSocketDisconnect):
            with rig.client.websocket_connect("/api/phone/exotel/monitor/not-a-token"):
                pass
        with pytest.raises(WebSocketDisconnect):
            with rig.client.websocket_connect(f"/api/phone/exotel/monitor/{px.make_monitor_token(cid, 'someone-else')}"):
                pass
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))


def test_a_listener_hears_both_sides_and_reads_the_transcript(rig):
    rig.asr_text = "When does the clinic open?"
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        cid = _wait_for_live(rig)[0]["conversation_id"]
        with rig.client.websocket_connect(_monitor_url(rig, cid)) as mon:
            hello = json.loads(mon.receive_text())
            assert hello["type"] == "hello" and hello["rate"] == 8000 and hello["human"] is False
            feed(ws, pcm_silence(300) + pcm_tone(1200) + pcm_silence(900))
            heard, _ = _read_monitor(mon, lambda m: m["type"] == "transcript" and m["role"] == "caller")
            assert "clinic" in heard["text"]
            _, seen = _read_monitor(mon, lambda m: m["type"] == "transcript" and m["role"] == "agent")
            kinds = {(m["type"], m.get("from")) for m in seen}
            assert ("audio", "agent") in kinds
            pcm = base64.b64decode(next(m["pcm"] for m in seen if m["type"] == "audio" and m["from"] == "agent"))
            assert len(pcm) % 2 == 0 and len(pcm) > 0
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))


def test_a_person_can_take_the_call_over_talk_and_hand_it_back(rig):
    from services import call_records
    import sys
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        cid = _wait_for_live(rig)[0]["conversation_id"]
        with rig.client.websocket_connect(_monitor_url(rig, cid)) as mon:
            mon.receive_text()                                                            # hello
            mon.send_text(json.dumps({"type": "audio", "pcm": base64.b64encode(pcm_tone(200)).decode()}))   # too early: ignored
            mon.send_text(json.dumps({"type": "takeover"}))
            state, _ = _read_monitor(mon, lambda m: m["type"] == "state")
            assert state["human"] is True
            assert _wait_for_live(rig)[0]["human_active"] is True
            asked_before = len(rig.asked)
            # the caller talks: the agent must stay silent
            feed(ws, pcm_silence(300) + pcm_tone(1200) + pcm_silence(900))
            _read_monitor(mon, lambda m: m["type"] == "audio" and m["from"] == "caller")
            time.sleep(0.5)
            assert len(rig.asked) == asked_before
            # the person speaks: it reaches the caller, labelled for other listeners
            voice = pcm_tone(400, rate=8000)
            mon.send_text(json.dumps({"type": "audio", "pcm": base64.b64encode(voice).decode()}))
            got = read_until(ws, lambda m: m["event"] == "media")
            assert base64.b64decode(got[-1]["media"]["payload"])
            _read_monitor(mon, lambda m: m["type"] == "audio" and m["from"] == "human")
            # a second person cannot take it
            with rig.client.websocket_connect(_monitor_url(rig, cid)) as mon2:
                mon2.receive_text()
                mon2.send_text(json.dumps({"type": "takeover"}))
                err, _ = _read_monitor(mon2, lambda m: m["type"] == "error")
                assert err["code"] == "ALREADY_TAKEN"
            # handing back: the agent listens and answers again
            mon.send_text(json.dumps({"type": "release"}))
            _read_monitor(mon, lambda m: m["type"] == "state" and m["human"] is False)
            feed(ws, pcm_silence(300) + pcm_tone(1200) + pcm_silence(900))
            _read_monitor(mon, lambda m: m["type"] == "transcript" and m["role"] == "agent")
            assert len(rig.asked) > asked_before
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
    time.sleep(0.3)
    row = call_records.get("CA1")
    assert row["outcome"] == "human_took_over"
    conv = sys.modules["backend"].local_db.get("conversations", row["conversation_id"])
    assert conv.get("needs_human") is True


def test_when_the_person_who_took_over_disconnects_the_agent_takes_the_call_back(rig):
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        cid = _wait_for_live(rig)[0]["conversation_id"]
        with rig.client.websocket_connect(_monitor_url(rig, cid)) as mon:
            mon.receive_text()
            mon.send_text(json.dumps({"type": "takeover"}))
            _read_monitor(mon, lambda m: m["type"] == "state")
        end = time.time() + 3
        while time.time() < end and _wait_for_live(rig)[0]["human_active"]:
            time.sleep(0.05)
        assert _wait_for_live(rig)[0]["human_active"] is False
        feed(ws, pcm_silence(300) + pcm_tone(1200) + pcm_silence(900))
        msgs = read_until(ws, is_mark)
        assert any(m["event"] == "media" for m in msgs)               # the agent answers again
        ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))


def test_listeners_are_told_when_the_call_ends_and_the_live_list_empties(rig):
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        cid = _wait_for_live(rig)[0]["conversation_id"]
        with rig.client.websocket_connect(_monitor_url(rig, cid)) as mon:
            mon.receive_text()
            ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
            _read_monitor(mon, lambda m: m["type"] == "ended")
    _wait_for_live(rig, n=0)


def test_exotel_errors_are_passed_on_as_a_plain_sentence():
    xml = ('<?xml version="1.0" encoding="UTF-8"?><TwilioResponse><RestException><Status>403</Status>'
           '<Message>Your account is not yet KYC compliant. This is mandatory before making outbound calls.</Message>'
           '</RestException></TwilioResponse>')
    out = px.exotel_reason(xml)
    assert out.startswith("Your account is not yet KYC compliant. This is mandatory before making outbound calls")
    assert "Finish KYC in the Exotel dashboard" in out and "<" not in out
    assert px.exotel_reason('{"RestException": {"Status": 401, "Message": "Unauthorized; "}}') == "Unauthorized"
    assert px.exotel_reason("<html><b>Bad gateway</b></html>") == "Bad gateway"
    assert px.exotel_reason("") == "no reason given"


def test_a_number_in_phone_allowed_numbers_is_pre_approved(rig, exotel, monkeypatch):
    """The owner confirmed consent once by adding the number to backend/.env: no caller_agreed is needed for it, while a
    number saved in the app still needs it on every call."""
    configure(monkeypatch, PHONE_ALLOWED_NUMBERS="+919876543210")
    a = make_agent(rig.client)
    assert call(rig, a, to="09876543210", caller_agreed=False).status_code == 200
    add(rig, label="Saved", number="+91 90000 11111")
    assert call(rig, a, to="+919000011111", caller_agreed=False).json()["code"] == "CONSENT_REQUIRED"
