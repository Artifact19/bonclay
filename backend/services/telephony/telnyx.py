"""Telnyx: Call Control Dial with bidirectional media streaming (RTP mode, PCMU 8 kHz).

Sources (read 27 Oct 2026):
  https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming    stream_url, stream_track, stream_bidirectional_mode ("rtp"),
                                                                                 stream_bidirectional_codec; events start / media / dtmf / stop / error in,
                                                                                 media / clear / mark out; PCMU 8 kHz default; RTP chunks 20 ms to 30 s
  https://developers.telnyx.com/api-reference/call-commands/dial                 POST https://api.telnyx.com/v2/calls (connection_id, to, from, webhook_url, stream_*)
  https://developers.telnyx.com/api-reference/call-commands/answer-call          POST /v2/calls/{call_control_id}/actions/answer (same stream_* fields)
  https://developers.telnyx.com/api-reference/call-commands/hangup-call          POST /v2/calls/{call_control_id}/actions/hangup
  https://developers.telnyx.com/api-reference/call-information/retrieve-a-call-status   GET /v2/calls/{call_control_id}: is_alive, call_duration, start_time, end_time
  https://developers.telnyx.com/docs/voice/programmable-voice/receiving-webhooks  webhook envelope data.event_type / data.payload.call_control_id
  https://developers.telnyx.com/api/errors/20100                                 20100 Insufficient Funds
Not confirmed from the docs, so assumed: that an RTP-mode `media` payload for PCMU is the bare mu-law bytes (no RTP header), the
exact `start` field names beyond `media_format`, `stream_id` and `call_control_id`, the hangup-cause names in call.hangup
webhooks, and which HTTP codes Telnyx uses for a restricted account. The adapter is "unverified" until a real call has run.

Unlike the other carriers, closing the stream does not end a Telnyx call: when the agent ends the call, the call is hung up
through the API (`api_hangup`)."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .. import phone_audio as pa
from .base import (AUTH_FAILED, EXHAUSTED, NOT_ALLOWED, Event, Provider, ProviderHTTPError, StreamCodec, duration_of,
                   http_request, json_message)

API = "https://api.telnyx.com/v2"
EXHAUSTED_WORDS = ("insufficient funds", "insufficient balance", "balance", "out of credit", "payment required")
NOT_ALLOWED_WORDS = ("not verified", "unverified", "whitelist", "not allowed", "restricted destination", "geo permission")
EXHAUSTED_CODES = {"20100"}
AUTH_CODES = {"10009"}
HANGUP_STATUS = {"user_busy": "busy", "no_answer": "no-answer", "timeout": "no-answer", "originator_cancel": "canceled",
                 "call_rejected": "failed", "unallocated_number": "failed", "normal_clearing": "completed",
                 "normal_unspecified": "completed"}


class TelnyxCodec(StreamCodec):
    """Telnyx media streaming: JSON events with a top-level `stream_id`; audio is base64 PCMU (mu-law 8 kHz) in RTP mode."""
    name = "telnyx"

    def parse(self, text: str) -> Optional[Event]:
        ev = self.load(text)
        if ev is None:
            return None
        kind = ev.get("event")
        if kind == "start":
            start = ev.get("start") or {}
            params = start.get("custom_parameters") or start.get("client_state")
            return Event("start", stream_id=ev.get("stream_id"), call_id=start.get("call_control_id"),
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
        return None                      # "connected", "error" and anything new

    def media(self, pcm: bytes, stream_id: Optional[str]) -> dict:
        return {"event": "media", "media": {"payload": self.b64(pa.ulaw_encode(pcm))}}

    def clear(self, stream_id: Optional[str]) -> dict:
        return {"event": "clear"}

    def mark(self, name: str, stream_id: Optional[str]) -> dict:
        return {"event": "mark", "mark": {"name": name}}


def stream_fields(stream_url: str) -> Dict[str, Any]:
    return {"stream_url": stream_url, "stream_track": "inbound_track", "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_codec": "PCMU"}


class Telnyx(Provider):
    name = "telnyx"
    display = "Telnyx"
    env_prefix = "TELNYX"
    verified = False
    verification_note = ("Written from Telnyx's docs; not yet run against a real account. A Telnyx account in the "
                         "free tier is limited to verified numbers.")
    VARS = (("api_key", "TELNYX_API_KEY"), ("connection_id", "TELNYX_CONNECTION_ID"), ("caller_id", "TELNYX_CALLER_ID"))
    api_hangup = True
    inbound_how = ("In the Telnyx portal create a Voice API (Call Control) application whose webhook URL is the address, and "
                   "assign the number to it. The address answers the call and starts the stream.")

    def account_id(self, account: Dict[str, str]) -> str:
        return str(account["api_key"])

    def codec(self) -> StreamCodec:
        return TelnyxCodec()

    @staticmethod
    def _headers(account: Dict[str, str]) -> Dict[str, str]:
        return {"Authorization": f"Bearer {account['api_key']}", "Content-Type": "application/json"}

    async def place_call(self, account, to, stream_url, status_url, answer_url=None):
        body: Dict[str, Any] = {"connection_id": account["connection_id"], "to": to, "from": account["caller_id"],
                                **stream_fields(stream_url)}
        if status_url:
            body["webhook_url"] = status_url
            body["webhook_url_method"] = "POST"
        r = await http_request("post", f"{API}/calls", 20.0, headers=self._headers(account), json=body)
        if r.status_code >= 300:
            raise ProviderHTTPError(r.status_code, r.text or "")
        try:
            return ((r.json() or {}).get("data") or {}).get("call_control_id")
        except Exception:  # noqa: BLE001
            return None

    async def answer_call(self, account, call_control_id: str, stream_url: str, status_url: str = "") -> bool:
        """Inbound: answer the call (the webhook told us about it) and start the stream. Best effort."""
        body: Dict[str, Any] = dict(stream_fields(stream_url))
        if status_url:
            body["webhook_url"] = status_url
        try:
            r = await http_request("post", f"{API}/calls/{call_control_id}/actions/answer", 10.0,
                                   headers=self._headers(account), json=body)
            return r.status_code < 300
        except Exception:  # noqa: BLE001
            return False

    async def call_status(self, account, call_id):
        r = await http_request("get", f"{API}/calls/{call_id}", 10.0, headers=self._headers(account))
        if r.status_code >= 300:
            raise ProviderHTTPError(r.status_code, r.text or "")
        try:
            d = (r.json() or {}).get("data") or {}
        except Exception:  # noqa: BLE001
            d = {}
        if d.get("is_alive"):
            status = "in-progress"
        elif d.get("end_time"):
            status = "completed"
        else:
            status = "ringing"       # a Dial answers is_alive=false until the callee picks up
        return {"status": status, "duration_s": duration_of(d.get("call_duration")), "started_at": d.get("start_time") or None,
                "ended_at": d.get("end_time") or None, "message": None}

    async def hangup(self, account, call_id) -> bool:
        try:
            r = await http_request("post", f"{API}/calls/{call_id}/actions/hangup", 10.0,
                                   headers=self._headers(account), json={})
            return r.status_code < 300
        except Exception:  # noqa: BLE001 - best effort
            return False

    def reason(self, body: str) -> str:
        return json_message(body, "detail", "title", "message")[:300]

    def classify_refusal(self, status: int, body: str) -> Optional[str]:
        code = ""
        try:
            data = json.loads(body)
            errs = data.get("errors") if isinstance(data, dict) else None
            code = str((errs[0] or {}).get("code") or "") if isinstance(errs, list) and errs else ""
        except (ValueError, AttributeError, IndexError):
            code = ""
        low = self.reason(body).lower()
        if code in EXHAUSTED_CODES or status == 402 or any(w in low for w in EXHAUSTED_WORDS):
            return EXHAUSTED
        if any(w in low for w in NOT_ALLOWED_WORDS) and status != 401:
            return NOT_ALLOWED
        if code in AUTH_CODES or status in (401, 403):
            return AUTH_FAILED
        return None

    def parse_status_callback(self, form):
        """Telnyx posts JSON webhooks: {"data": {"event_type": "call.hangup", "payload": {"call_control_id": ...}}}."""
        data = form.get("data") if isinstance(form, dict) else None
        if not isinstance(data, dict):
            return None
        payload = data.get("payload") or {}
        sid = payload.get("call_control_id")
        if not sid:
            return None
        event = str(data.get("event_type") or "")
        if event == "call.initiated":
            status = "queued"
        elif event == "call.answered":
            status = "in-progress"
        elif event == "call.hangup":
            status = HANGUP_STATUS.get(str(payload.get("hangup_cause") or "").lower(), "completed")
        else:
            return {"sid": str(sid), "status": "unknown", "duration_s": None, "ended_at": None, "message": None}
        return {"sid": str(sid), "status": status, "duration_s": None,
                "ended_at": payload.get("end_time") if status not in ("queued", "in-progress") else None,
                "message": payload.get("hangup_cause") if event == "call.hangup" else None}


PROVIDER = Telnyx()
