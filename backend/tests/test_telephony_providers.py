"""Each phone provider's HTTP side (services/telephony), against a fake HTTP client: the request that places a call (URL,
authentication, body, stream or answer address), the call id read from the answer, the call status, the hangup, and what each
provider's refusals mean. No network."""
import asyncio
import json

import httpx
import pytest

from services.telephony import registry
from services.telephony.base import ProviderHTTPError, ProviderUnreachable
from services.telephony.exotel import PROVIDER as EXOTEL
from services.telephony.plivo import PLIVO, VOBIZ
from services.telephony.telnyx import PROVIDER as TELNYX
from services.telephony.twilio import PROVIDER as TWILIO

WSS = "wss://abc.ngrok.app/api/phone/x/stream/TOKEN"
STATUS = "https://abc.ngrok.app/api/phone/x/status-callback/TOKEN"
ANSWER = "https://abc.ngrok.app/api/phone/x/answer/TOKEN"
TO = "+919876543210"


class Http:
    """Records every request; answers from `queue` (httpx.Response objects), else 200 with `{}`."""
    sent: list = []
    queue: list = []
    boom = False

    def __init__(self, *a, **k):
        self.timeout = k.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def _do(self, method, url, **kw):
        Http.sent.append({"method": method, "url": url, **kw})
        if Http.boom:
            raise httpx.ConnectError("down")
        return Http.queue.pop(0) if Http.queue else httpx.Response(200, json={})

    async def post(self, url, **kw):
        return await self._do("post", url, **kw)

    async def get(self, url, **kw):
        return await self._do("get", url, **kw)

    async def delete(self, url, **kw):
        return await self._do("delete", url, **kw)


@pytest.fixture()
def http(monkeypatch):
    Http.sent, Http.queue, Http.boom = [], [], False
    monkeypatch.setattr(httpx, "AsyncClient", Http)
    return Http


def run(coro):
    return asyncio.run(coro)


def reply(status=200, body=None, text=None):
    return httpx.Response(status, text=text) if text is not None else httpx.Response(status, json=body if body is not None else {})


# ------------------------------------------------------------------------------------------------------- Twilio

TW = {"label": "1", "sid": "AC123", "token": "tok", "caller_id": "+15005550006"}


def test_twilio_places_the_call_with_inline_twiml_that_connects_a_stream(http):
    http.queue = [reply(201, {"sid": "CA999", "status": "queued"})]
    cid = run(TWILIO.place_call(TW, TO, WSS, STATUS))
    assert cid == "CA999"
    r = http.sent[0]
    assert r["method"] == "post" and r["url"] == "https://api.twilio.com/2010-04-01/Accounts/AC123/Calls.json"
    assert r["auth"] == ("AC123", "tok")
    assert r["data"]["To"] == TO and r["data"]["From"] == "+15005550006"
    assert r["data"]["Twiml"] == ('<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="' + WSS
                                  + '" /></Connect></Response>')
    assert r["data"]["StatusCallback"] == STATUS and "completed" in r["data"]["StatusCallbackEvent"]


def test_twilio_escapes_the_stream_address_in_the_twiml(http):
    http.queue = [reply(201, {"sid": "CA1"})]
    run(TWILIO.place_call(TW, TO, 'wss://x/a?b=1&c="2"', STATUS))
    import xml.etree.ElementTree as ET
    stream = ET.fromstring(http.sent[0]["data"]["Twiml"]).find("Connect/Stream")
    assert stream.get("url") == 'wss://x/a?b=1&c="2"'                 # well-formed XML that reads back as sent


def test_twilio_refusal_is_an_http_error_with_the_bodys_own_words(http):
    http.queue = [reply(400, {"code": 21211, "message": "The 'To' number is not a valid phone number.", "status": 400})]
    with pytest.raises(ProviderHTTPError) as e:
        run(TWILIO.place_call(TW, TO, WSS, STATUS))
    assert e.value.status == 400 and TWILIO.reason(e.value.body) == "The 'To' number is not a valid phone number"


def test_a_network_error_is_unreachable(http):
    http.boom = True
    with pytest.raises(ProviderUnreachable) as e:
        run(TWILIO.place_call(TW, TO, WSS, STATUS))
    assert e.value.reason == "ConnectError"


@pytest.mark.parametrize("status,body,kind", [
    (401, {"code": 20003, "message": "Authenticate", "status": 401}, "auth_failed"),
    (403, {"code": 20003, "message": "Permission Denied"}, "auth_failed"),
    (400, {"code": 21215, "message": "Geo Permission configuration is not permitting call"}, "not_allowed"),
    (400, {"code": 21219, "message": "The number +91 is unverified. Trial accounts may only make calls to verified numbers."}, "not_allowed"),
    (402, {"message": "Payment required"}, "exhausted"),
    (403, {"code": 10003, "message": "Incoming call rejected due to inactive account"}, "exhausted"),
    (400, {"code": 21211, "message": "Invalid 'To' number"}, None),
    (500, {"message": "Internal error"}, None),
])
def test_twilio_refusals(status, body, kind):
    assert TWILIO.classify_refusal(status, json.dumps(body)) == kind


def test_twilio_call_status_hangup_and_callback(http):
    http.queue = [reply(200, {"sid": "CA9", "status": "completed", "duration": "42", "start_time": "Tue, 27 Oct 2026 10:00:00 +0000",
                              "end_time": "Tue, 27 Oct 2026 10:00:42 +0000"})]
    s = run(TWILIO.call_status(TW, "CA9"))
    assert http.sent[0]["url"] == "https://api.twilio.com/2010-04-01/Accounts/AC123/Calls/CA9.json"
    assert s["status"] == "completed" and s["duration_s"] == 42 and s["ended_at"]
    http.queue = [reply(200, {})]
    assert run(TWILIO.hangup(TW, "CA9")) is True
    assert http.sent[1]["data"] == {"Status": "completed"}
    http.boom = True
    assert run(TWILIO.hangup(TW, "CA9")) is False                    # best effort: never raises
    cb = TWILIO.parse_status_callback({"CallSid": "CA9", "CallStatus": "no-answer", "CallDuration": "0"})
    assert cb["sid"] == "CA9" and cb["status"] == "no-answer" and cb["duration_s"] == 0
    assert TWILIO.parse_status_callback({"CallStatus": "completed"}) is None


def test_twilio_answer_document_for_inbound_calls():
    doc = TWILIO.answer_document(WSS)
    assert "<Connect><Stream" in doc and WSS in doc


# -------------------------------------------------------------------------------------------------- Plivo / Vobiz

PL = {"label": "1", "auth_id": "MAXXXX", "token": "ptok", "caller_id": "+14155550000"}
VB = {"label": "1", "auth_id": "VBID", "token": "vtok", "caller_id": "+912240000000", "base_url": "https://api.vobiz.ai/api/v1"}


def test_plivo_places_the_call_with_an_answer_url_and_basic_auth(http):
    http.queue = [reply(201, {"message": "call fired", "request_uuid": "uuid-1", "api_id": "x"})]
    cid = run(PLIVO.place_call(PL, TO, WSS, STATUS, ANSWER))
    assert cid == "uuid-1"
    r = http.sent[0]
    assert r["method"] == "post" and r["url"] == "https://api.plivo.com/v1/Account/MAXXXX/Call/"
    assert r["auth"] == ("MAXXXX", "ptok")
    assert r["json"] == {"from": "+14155550000", "to": TO, "answer_url": ANSWER, "answer_method": "POST",
                         "hangup_url": STATUS, "hangup_method": "POST"}


def test_plivo_answer_document_is_a_bidirectional_stream():
    doc = PLIVO.answer_document(WSS)
    assert doc == ('<?xml version="1.0" encoding="UTF-8"?><Response><Stream bidirectional="true" keepCallAlive="true" '
                   'contentType="audio/x-mulaw;rate=8000">' + WSS + "</Stream></Response>")


def test_vobiz_uses_its_own_base_url_and_auth_headers(http):
    http.queue = [reply(200, {"api_id": "a", "message": "Call fired", "request_uuid": "vb-1"})]
    cid = run(VOBIZ.place_call(VB, TO, WSS, STATUS, ANSWER))
    assert cid == "vb-1"
    r = http.sent[0]
    assert r["url"] == "https://api.vobiz.ai/api/v1/Account/VBID/Call/" and "auth" not in r
    assert r["headers"]["X-Auth-ID"] == "VBID" and r["headers"]["X-Auth-Token"] == "vtok"
    assert r["json"]["answer_url"] == ANSWER and r["json"]["from"] == "+912240000000"


def test_vobiz_base_url_can_be_changed_per_account(http):
    http.queue = [reply(200, {"request_uuid": "x"})]
    run(VOBIZ.place_call({**VB, "base_url": "https://eu.vobiz.example/api/v1/"}, TO, WSS, STATUS, ANSWER))
    assert http.sent[0]["url"] == "https://eu.vobiz.example/api/v1/Account/VBID/Call/"


def test_plivo_family_hangup_is_a_delete(http):
    http.queue = [reply(204, text="")]
    assert run(PLIVO.hangup(PL, "uuid-1")) is True
    assert http.sent[0]["method"] == "delete" and http.sent[0]["url"] == "https://api.plivo.com/v1/Account/MAXXXX/Call/uuid-1/"
    http.queue = [reply(204, text="")]
    assert run(VOBIZ.hangup(VB, "vb-1")) is True
    assert http.sent[1]["url"] == "https://api.vobiz.ai/api/v1/Account/VBID/Call/vb-1/"


def test_plivo_status_reads_the_live_call_then_the_record(http):
    http.queue = [reply(200, {"call_status": "in-progress", "session_start": "2026-10-27 10:00:00"})]
    assert run(PLIVO.call_status(PL, "u"))["status"] == "in-progress"
    assert http.sent[0]["url"].endswith("/Call/u/?status=live")
    http.queue = [reply(404, {"error": "not found"}), reply(200, {"call_duration": "30", "answer_time": "t0", "end_time": "t1",
                                                                  "hangup_cause_name": "Normal Hangup"})]
    s = run(PLIVO.call_status(PL, "u"))
    assert s["status"] == "completed" and s["duration_s"] == 30 and s["ended_at"] == "t1"
    http.queue = [reply(404, {}), reply(200, {"call_duration": "0", "hangup_cause_name": "Busy Line"})]
    assert run(PLIVO.call_status(PL, "u"))["status"] == "busy"
    http.queue = [reply(404, {}), reply(404, {})]
    assert run(PLIVO.call_status(PL, "u"))["status"] == "queued"


@pytest.mark.parametrize("status,body,kind", [
    (401, {"error": "Authentication credentials are not valid"}, "auth_failed"),
    (402, {"error": "Insufficient balance"}, "exhausted"),
    (400, {"error": "From number not owned by account"}, "not_allowed"),
    (400, {"error": "Destination is not a sandbox number"}, "not_allowed"),
    (400, {"error": "Invalid destination"}, None),
    (429, {"error": "Concurrent call limit exceeded"}, None),
])
def test_plivo_and_vobiz_refusals(status, body, kind):
    assert PLIVO.classify_refusal(status, json.dumps(body)) == kind
    assert VOBIZ.classify_refusal(status, json.dumps(body)) == kind


def test_plivo_family_callbacks():
    cb = PLIVO.parse_status_callback({"CallUUID": "u1", "CallStatus": "completed", "Duration": "12"})
    assert cb == {"sid": "u1", "status": "completed", "duration_s": 12, "ended_at": None, "message": None}
    # Vobiz's documented Hangup payload: Event, Status, Duration, CallUUID
    cb = VOBIZ.parse_status_callback({"CallUUID": "u2", "Event": "Hangup", "Status": "completed", "Duration": "7",
                                      "EndTime": "2026-10-27 10:00:07"})
    assert cb["status"] == "completed" and cb["duration_s"] == 7 and cb["ended_at"] == "2026-10-27 10:00:07"
    assert PLIVO.parse_status_callback({"CallUUID": "u3", "CallStatus": "timeout"})["status"] == "no-answer"


# ------------------------------------------------------------------------------------------------------- Telnyx

TX = {"label": "1", "api_key": "KEY0123", "connection_id": "conn-1", "caller_id": "+13125550000"}


def test_telnyx_dials_with_a_bidirectional_pcmu_stream(http):
    http.queue = [reply(200, {"data": {"call_control_id": "cc-1", "call_leg_id": "l", "call_session_id": "s", "is_alive": False}})]
    cid = run(TELNYX.place_call(TX, TO, WSS, STATUS))
    assert cid == "cc-1"
    r = http.sent[0]
    assert r["method"] == "post" and r["url"] == "https://api.telnyx.com/v2/calls"
    assert r["headers"]["Authorization"] == "Bearer KEY0123"
    assert r["json"] == {"connection_id": "conn-1", "to": TO, "from": "+13125550000", "stream_url": WSS,
                         "stream_track": "inbound_track", "stream_bidirectional_mode": "rtp",
                         "stream_bidirectional_codec": "PCMU", "webhook_url": STATUS, "webhook_url_method": "POST"}


def test_telnyx_answers_an_inbound_call_with_the_stream_and_hangs_up_through_the_api(http):
    http.queue = [reply(200, {"data": {"result": "ok"}}), reply(200, {"data": {"result": "ok"}})]
    assert run(TELNYX.answer_call(TX, "cc-9", WSS, STATUS)) is True
    assert http.sent[0]["url"] == "https://api.telnyx.com/v2/calls/cc-9/actions/answer"
    assert http.sent[0]["json"]["stream_url"] == WSS and http.sent[0]["json"]["stream_bidirectional_mode"] == "rtp"
    assert run(TELNYX.hangup(TX, "cc-9")) is True
    assert http.sent[1]["url"] == "https://api.telnyx.com/v2/calls/cc-9/actions/hangup"
    assert TELNYX.api_hangup is True and not any(p.api_hangup for p in (EXOTEL, TWILIO, PLIVO, VOBIZ))


def test_telnyx_call_status(http):
    http.queue = [reply(200, {"data": {"is_alive": True, "start_time": "t0"}})]
    assert run(TELNYX.call_status(TX, "cc-1"))["status"] == "in-progress"
    assert http.sent[0]["url"] == "https://api.telnyx.com/v2/calls/cc-1"
    http.queue = [reply(200, {"data": {"is_alive": False, "end_time": "t1", "call_duration": 12}})]
    s = run(TELNYX.call_status(TX, "cc-1"))
    assert s["status"] == "completed" and s["duration_s"] == 12


@pytest.mark.parametrize("status,body,kind", [
    (401, {"errors": [{"code": "10009", "title": "Authentication failed", "detail": "Invalid API key"}]}, "auth_failed"),
    (403, {"errors": [{"code": "20100", "title": "Insufficient Funds", "detail": "Your balance is too low"}]}, "exhausted"),
    (403, {"errors": [{"code": "10010", "title": "Forbidden", "detail": "The destination number is not verified"}]}, "not_allowed"),
    (422, {"errors": [{"code": "90041", "title": "Invalid number", "detail": "to is not a valid number"}]}, None),
])
def test_telnyx_refusals(status, body, kind):
    assert TELNYX.classify_refusal(status, json.dumps(body)) == kind
    assert TELNYX.reason(json.dumps(body)) == body["errors"][0]["detail"]


def test_telnyx_webhooks():
    def hook(event, **payload):
        return {"data": {"event_type": event, "payload": {"call_control_id": "cc-1", **payload}}}
    assert TELNYX.parse_status_callback(hook("call.initiated"))["status"] == "queued"
    assert TELNYX.parse_status_callback(hook("call.answered"))["status"] == "in-progress"
    assert TELNYX.parse_status_callback(hook("call.hangup", hangup_cause="user_busy"))["status"] == "busy"
    assert TELNYX.parse_status_callback(hook("call.hangup", hangup_cause="normal_clearing", end_time="t"))["status"] == "completed"
    assert TELNYX.parse_status_callback({"data": {"event_type": "call.initiated", "payload": {}}}) is None
    assert TELNYX.parse_status_callback({}) is None


# ------------------------------------------------------------------------------------------------------- Exotel

def test_exotel_provider_sends_exactly_what_it_always_did(http, monkeypatch):
    monkeypatch.delenv("EXOTEL_STATUS_CALLBACK", raising=False)
    http.queue = [reply(200, {"Call": {"Sid": "CAx"}})]
    acct = {"label": "1", "key": "k", "token": "t", "sid": "acme", "caller_id": "08047491899", "subdomain": "api.in.exotel.com"}
    assert run(EXOTEL.place_call(acct, TO, WSS, STATUS)) == "CAx"
    r = http.sent[0]
    assert r["url"] == "https://api.in.exotel.com/v1/Accounts/acme/Calls/connect" and r["auth"] == ("k", "t")
    assert r["data"] == {"From": TO, "CallerId": "08047491899", "StreamUrl": WSS, "StreamType": "bidirectional"}
    monkeypatch.setenv("EXOTEL_STATUS_CALLBACK", "1")
    http.queue = [reply(200, text="<Response><Call><Sid>CAy</Sid></Call></Response>")]
    assert run(EXOTEL.place_call(acct, TO, WSS, STATUS)) == "CAy"          # XML answers are read too
    assert http.sent[1]["data"]["StatusCallback"] == STATUS


def test_every_provider_names_itself_and_whether_it_has_been_run_for_real():
    assert [registry.provider(n).name for n in registry.NAMES] == list(registry.NAMES)
    assert EXOTEL.verified is True
    assert all(registry.provider(n).verified is False and registry.provider(n).verification_note
               for n in ("twilio", "plivo", "vobiz", "telnyx"))
