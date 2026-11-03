"""Twilio: Calls API plus bidirectional Media Streams (`<Connect><Stream>`), mu-law 8 kHz.

Sources (read 27 Oct 2026):
  https://www.twilio.com/docs/voice/media-streams                       bidirectional streams, one stream per call
  https://www.twilio.com/docs/voice/media-streams/websocket-messages    connected / start / media / dtmf / mark / stop in, media / mark / clear out
  https://www.twilio.com/docs/voice/twiml/stream                        `<Connect><Stream url="wss://...">`: wss only, no query string
  https://www.twilio.com/docs/voice/api/call-resource                   POST /2010-04-01/Accounts/{Sid}/Calls.json, statuses, fetch
Hangup (POST Status=completed on the call resource) is from Twilio's well-known Update-a-call API; that page was not
readable in full, so it is the one unconfirmed call here (it is best effort, and a failure is only logged).
The adapter has not been run against a real Twilio account: "unverified" in docs/PHONE_AND_MEETINGS.md."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional
from xml.sax.saxutils import quoteattr

from .. import phone_audio as pa
from .base import (AUTH_FAILED, EXHAUSTED, NOT_ALLOWED, Event, Provider, ProviderHTTPError, ProviderUnreachable,
                   StreamCodec, duration_of, field, http_request, json_message, normalise_status)

API = "https://api.twilio.com/2010-04-01"
EXHAUSTED_WORDS = ("balance", "insufficient", "out of funds", "add funds", "inactive account", "account is suspended", "payment")
NOT_ALLOWED_WORDS = ("unverified", "verified numbers", "geo permission", "not authorized to call", "not authorised to call",
                     "not permitted", "dialing permission")
NOT_ALLOWED_CODES = {21215, 21219, 21608, 13225, 21408, 21214}
AUTH_CODES = {20003, 20005}
EXHAUSTED_CODES = {10003}


class TwilioCodec(StreamCodec):
    """Twilio Media Streams: JSON events with `streamSid`; audio is base64 audio/x-mulaw at 8000 Hz, mono."""
    name = "twilio"

    def parse(self, text: str) -> Optional[Event]:
        ev = self.load(text)
        if ev is None:
            return None
        kind = ev.get("event")
        if kind == "start":
            start = ev.get("start") or {}
            params = start.get("customParameters")
            return Event("start", stream_id=start.get("streamSid") or ev.get("streamSid"), call_id=start.get("callSid"),
                         caller=str(start.get("from", "")), rate=8000, params=params if isinstance(params, dict) else {})
        if kind == "media":
            media = ev.get("media") or {}
            if media.get("track") == "outbound":
                return None
            raw = self.b64decode(media.get("payload"))
            return Event("media", pcm=pa.ulaw_decode(raw)) if raw is not None else None
        if kind == "dtmf":
            return Event("dtmf", digit=self.digit_of(ev.get("dtmf")))
        if kind == "mark":
            return Event("mark", name=str((ev.get("mark") or {}).get("name") or ""))
        if kind == "stop":
            return Event("stop")
        return None                      # "connected" and anything new

    def media(self, pcm: bytes, stream_id: Optional[str]) -> dict:
        return {"event": "media", "streamSid": stream_id, "media": {"payload": self.b64(pa.ulaw_encode(pcm))}}

    def clear(self, stream_id: Optional[str]) -> dict:
        return {"event": "clear", "streamSid": stream_id}

    def mark(self, name: str, stream_id: Optional[str]) -> dict:
        return {"event": "mark", "streamSid": stream_id, "mark": {"name": name}}


def twiml(stream_url: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url=' + quoteattr(stream_url)
            + " /></Connect></Response>")


class Twilio(Provider):
    name = "twilio"
    display = "Twilio"
    env_prefix = "TWILIO"
    verified = False
    verification_note = "Written from Twilio's docs; not yet run against a real account."
    VARS = (("sid", "TWILIO_ACCOUNT_SID"), ("token", "TWILIO_AUTH_TOKEN"), ("caller_id", "TWILIO_CALLER_ID"))
    inbound_how = ("In the Twilio console open the number, Voice Configuration, 'A call comes in': Webhook, HTTP POST, "
                   "and paste the address.")

    def account_id(self, account: Dict[str, str]) -> str:
        return str(account["sid"])

    def codec(self) -> StreamCodec:
        return TwilioCodec()

    def answer_document(self, stream_url: str) -> str:
        return twiml(stream_url)

    async def place_call(self, account, to, stream_url, status_url, answer_url=None):
        url = f"{API}/Accounts/{account['sid']}/Calls.json"
        data: Dict[str, Any] = {"To": to, "From": account["caller_id"], "Twiml": twiml(stream_url)}
        if status_url:
            data["StatusCallback"] = status_url
            data["StatusCallbackMethod"] = "POST"
            data["StatusCallbackEvent"] = ["initiated", "ringing", "answered", "completed"]
        r = await http_request("post", url, 20.0, auth=(account["sid"], account["token"]), data=data)
        if r.status_code >= 300:
            raise ProviderHTTPError(r.status_code, r.text or "")
        try:
            return (r.json() or {}).get("sid")
        except Exception:  # noqa: BLE001
            return None

    async def call_status(self, account, call_id):
        url = f"{API}/Accounts/{account['sid']}/Calls/{call_id}.json"
        r = await http_request("get", url, 10.0, auth=(account["sid"], account["token"]))
        if r.status_code >= 300:
            raise ProviderHTTPError(r.status_code, r.text or "")
        try:
            f = r.json() or {}
        except Exception:  # noqa: BLE001
            f = {}
        status = normalise_status(f.get("status"))
        return {"status": status, "duration_s": duration_of(f.get("duration")), "started_at": f.get("start_time") or None,
                "ended_at": f.get("end_time") or None, "message": None}

    async def hangup(self, account, call_id) -> bool:
        try:
            r = await http_request("post", f"{API}/Accounts/{account['sid']}/Calls/{call_id}.json", 10.0,
                                   auth=(account["sid"], account["token"]), data={"Status": "completed"})
            return r.status_code < 300
        except (ProviderUnreachable, Exception):  # noqa: BLE001 - best effort
            return False

    def reason(self, body: str) -> str:
        return json_message(body, "message", "detail")[:300]

    def classify_refusal(self, status: int, body: str) -> Optional[str]:
        code = None
        try:
            data = json.loads(body)
            code = int(data.get("code")) if isinstance(data, dict) and data.get("code") is not None else None
        except (ValueError, TypeError):
            code = None
        low = self.reason(body).lower()
        if code in EXHAUSTED_CODES or status == 402 or any(w in low for w in EXHAUSTED_WORDS):
            return EXHAUSTED
        if code in NOT_ALLOWED_CODES or any(w in low for w in NOT_ALLOWED_WORDS):
            return NOT_ALLOWED
        if code in AUTH_CODES or status in (401, 403):
            return AUTH_FAILED
        return None

    def parse_status_callback(self, form):
        sid = field(form, "CallSid")
        if not sid:
            return None
        return {"sid": str(sid), "status": normalise_status(field(form, "CallStatus")),
                "duration_s": duration_of(field(form, "CallDuration", "Duration")),
                "ended_at": field(form, "Timestamp"), "message": field(form, "ErrorMessage")}


PROVIDER = Twilio()
