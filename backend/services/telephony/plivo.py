"""Plivo, and Vobiz (a Plivo-style API with its own base URL and auth headers): Call API plus bidirectional Audio Streams.

Sources (read 27 Oct 2026):
  Plivo   https://www.plivo.com/docs/voice/api/call/make-a-call        POST /v1/Account/{auth_id}/Call/ (JSON, Basic auth), retrieve, live, DELETE hangup
          https://docs.plivo.com/docs/voice-agents/audio-streaming/concepts/audio-streaming-reference.md
                                                                       <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://..</Stream>;
                                                                       start / media / dtmf / playedStream / clearedAudio in, playAudio / clearAudio / checkpoint out
  Vobiz   https://www.vobiz.ai/docs/call/make-call.md                  POST https://api.vobiz.ai/api/v1/Account/{auth_id}/Call/, headers X-Auth-ID and X-Auth-Token
          https://www.vobiz.ai/docs/call/hangup-call.md                DELETE .../Call/{call_uuid}/ (204)
          https://www.vobiz.ai/docs/call/retrieve-live-call.md         GET .../Call/{call_uuid}/?status=live
          https://www.vobiz.ai/docs/xml/stream.md                      the same <Stream> element; contentType audio/x-mulaw;rate=8000 or audio/x-l16;rate=8000|16000
          https://www.vobiz.ai/docs/integrations/websockets.md         start (callId, streamId, mediaFormat), playAudio / clearAudio / checkpoint

What is and is not confirmed:
  * Plivo and Vobiz: the call request, the answer XML, `start`, `playAudio`, `clearAudio`, `checkpoint`, hangup are from the pages above.
  * Plivo's `media`, `dtmf` and `playedStream` shapes are from its reference page. Vobiz does not document `media` (only
    `media.payload`), `dtmf`, `playedStream` or `clearedAudio` shapes, nor its call-detail (CDR) endpoint: this adapter
    assumes they match Plivo's. Both adapters are "unverified" until a real call has run through them.
  * Plivo's refusal wording (credit, sandbox, caller id) is not documented; the words below are an educated guess that is
    cheap to extend, and any refusal not recognised stops the call with the provider's own message."""
from __future__ import annotations

from typing import Any, Dict, Optional
from xml.sax.saxutils import escape

from .. import phone_audio as pa
from .base import (AUTH_FAILED, EXHAUSTED, NOT_ALLOWED, Event, Provider, ProviderHTTPError, StreamCodec, duration_of,
                   field, http_request, json_message, normalise_status)

PLIVO_BASE = "https://api.plivo.com/v1"
VOBIZ_BASE = "https://api.vobiz.ai/api/v1"
EXHAUSTED_WORDS = ("insufficient", "balance", "credit", "recharge", "payment required", "low balance")
NOT_ALLOWED_WORDS = ("sandbox", "not verified", "unverified", "whitelist", "not owned by account", "not authorised",
                     "not authorized to call", "geo permission", "destination not")
STATUS_WORDS = {"cancel": "canceled", "timeout": "no-answer", "rejected": "failed", "busyline": "busy", "hangup": "completed"}


class PlivoCodec(StreamCodec):
    """Plivo/Vobiz Audio Streams. We ask for audio/x-mulaw at 8000 Hz; `start.mediaFormat` says what really arrives, and
    audio goes back in the same format (`playAudio.media.contentType` and `sampleRate` must match the stream)."""
    name = "plivo"
    include_stream_id_in_media = False       # Vobiz's playAudio carries the streamId at the top level, Plivo's does not

    def __init__(self) -> None:
        self.encoding = "mulaw"              # "mulaw" | "l16"
        self.rate = 8000

    def parse(self, text: str) -> Optional[Event]:
        ev = self.load(text)
        if ev is None:
            return None
        kind = ev.get("event")
        if kind == "start":
            start = ev.get("start") or {}
            fmt = start.get("mediaFormat") or start.get("media_format") or {}
            enc = str(fmt.get("encoding") or "audio/x-mulaw").lower()
            self.encoding = "l16" if "l16" in enc else "mulaw"
            try:
                self.rate = int(fmt.get("sampleRate") or fmt.get("sample_rate") or 8000)
            except (TypeError, ValueError):
                self.rate = 8000
            if self.encoding == "mulaw":
                self.rate = 8000
            return Event("start", stream_id=start.get("streamId") or start.get("stream_id") or ev.get("streamId"),
                         call_id=start.get("callId") or start.get("call_id") or start.get("callUuid"),
                         caller=str(start.get("from", "")), rate=self.rate)
        if kind == "media":
            raw = self.b64decode((ev.get("media") or {}).get("payload"))
            if raw is None:
                return None
            return Event("media", pcm=pa.ulaw_decode(raw) if self.encoding == "mulaw" else raw[: len(raw) - len(raw) % 2])
        if kind == "dtmf":
            return Event("dtmf", digit=self.digit_of(ev.get("dtmf")))
        if kind == "playedStream":
            return Event("mark", name=str(ev.get("name") or ""))
        if kind == "stop":
            return Event("stop")
        return None                            # "clearedAudio" and anything new

    def media(self, pcm: bytes, stream_id: Optional[str]) -> dict:
        audio = pa.ulaw_encode(pcm) if self.encoding == "mulaw" else pcm
        msg: Dict[str, Any] = {"event": "playAudio"}
        if self.include_stream_id_in_media:
            msg["streamId"] = stream_id
        msg["media"] = {"contentType": "audio/x-mulaw" if self.encoding == "mulaw" else "audio/x-l16",
                        "sampleRate": self.rate, "payload": self.b64(audio)}
        return msg

    def clear(self, stream_id: Optional[str]) -> dict:
        return {"event": "clearAudio", "streamId": stream_id}

    def mark(self, name: str, stream_id: Optional[str]) -> dict:
        return {"event": "checkpoint", "streamId": stream_id, "name": name}


class VobizCodec(PlivoCodec):
    name = "vobiz"
    include_stream_id_in_media = True


def stream_xml(stream_url: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?><Response><Stream bidirectional="true" keepCallAlive="true" '
            'contentType="audio/x-mulaw;rate=8000">' + escape(stream_url) + "</Stream></Response>")


class Plivo(Provider):
    name = "plivo"
    display = "Plivo"
    env_prefix = "PLIVO"
    verified = False
    verification_note = ("Written from Plivo's docs; not yet run against a real account. Plivo trial accounts can only "
                         "call numbers verified in the console (sandbox).")
    VARS = (("auth_id", "PLIVO_AUTH_ID"), ("token", "PLIVO_AUTH_TOKEN"), ("caller_id", "PLIVO_CALLER_ID"))
    default_base = PLIVO_BASE
    codec_cls = PlivoCodec
    needs_answer_url = True
    inbound_how = ("In the Plivo console create an Application whose Answer URL is the address (method POST), and attach it to "
                   "the number (Phone Numbers, the number, Application).")

    def account_id(self, account: Dict[str, str]) -> str:
        return str(account["auth_id"])

    def codec(self) -> StreamCodec:
        return self.codec_cls()

    def answer_document(self, stream_url: str) -> str:
        return stream_xml(stream_url)

    # ---- HTTP -----------------------------------------------------------------------------------------------
    def base_url(self, account: Dict[str, str]) -> str:
        return (account.get("base_url") or self.default_base).rstrip("/")

    def auth_kwargs(self, account: Dict[str, str]) -> Dict[str, Any]:
        return {"auth": (account["auth_id"], account["token"])}

    def call_url(self, account: Dict[str, str], call_id: str = "") -> str:
        return f"{self.base_url(account)}/Account/{account['auth_id']}/Call/" + (f"{call_id}/" if call_id else "")

    async def place_call(self, account, to, stream_url, status_url, answer_url=None):
        body: Dict[str, Any] = {"from": account["caller_id"], "to": to, "answer_url": answer_url, "answer_method": "POST"}
        if status_url:
            body["hangup_url"] = status_url
            body["hangup_method"] = "POST"
        r = await http_request("post", self.call_url(account), 20.0, json=body, **self.auth_kwargs(account))
        if r.status_code >= 300:
            raise ProviderHTTPError(r.status_code, r.text or "")
        try:
            return (r.json() or {}).get("request_uuid")
        except Exception:  # noqa: BLE001
            return None

    async def call_status(self, account, call_id):
        """A call that is live answers `?status=live`; one that is over is in the call detail record. Neither: still queued."""
        live = await http_request("get", self.call_url(account, call_id) + "?status=live", 10.0, **self.auth_kwargs(account))
        if live.status_code < 300:
            try:
                f = live.json() or {}
            except Exception:  # noqa: BLE001
                f = {}
            return {"status": normalise_status(f.get("call_status") or "in-progress"), "duration_s": None,
                    "started_at": f.get("session_start") or None, "ended_at": None, "message": None}
        if live.status_code != 404:
            raise ProviderHTTPError(live.status_code, live.text or "")
        cdr = await http_request("get", self.call_url(account, call_id), 10.0, **self.auth_kwargs(account))
        if cdr.status_code == 404:
            return {"status": "queued", "duration_s": None, "started_at": None, "ended_at": None, "message": None}
        if cdr.status_code >= 300:
            raise ProviderHTTPError(cdr.status_code, cdr.text or "")
        try:
            f = cdr.json() or {}
        except Exception:  # noqa: BLE001
            f = {}
        cause = str(f.get("hangup_cause_name") or "")
        low = cause.lower()
        duration = duration_of(f.get("call_duration"))
        if "busy" in low:
            status = "busy"
        elif "no answer" in low or "no_answer" in low or "timeout" in low:
            status = "no-answer"
        elif "cancel" in low:
            status = "canceled"
        elif f.get("answer_time") or (duration or 0) > 0:
            status = "completed"
        else:
            status = "failed"
        return {"status": status, "duration_s": duration, "started_at": f.get("answer_time") or f.get("initiation_time") or None,
                "ended_at": f.get("end_time") or None, "message": cause or None}

    async def hangup(self, account, call_id) -> bool:
        try:
            r = await http_request("delete", self.call_url(account, call_id), 10.0, **self.auth_kwargs(account))
            return r.status_code < 300
        except Exception:  # noqa: BLE001 - best effort
            return False

    # ---- refusals and callbacks -------------------------------------------------------------------------------
    def reason(self, body: str) -> str:
        return json_message(body, "error", "message", "detail")[:300]

    def classify_refusal(self, status: int, body: str) -> Optional[str]:
        low = self.reason(body).lower()
        if status == 402 or any(w in low for w in EXHAUSTED_WORDS):
            return EXHAUSTED
        if any(w in low for w in NOT_ALLOWED_WORDS):
            return NOT_ALLOWED
        if status in (401, 403):
            return AUTH_FAILED
        return None

    def parse_status_callback(self, form):
        sid = field(form, "CallUUID", "call_uuid", "RequestUUID")
        if not sid:
            return None
        raw = field(form, "CallStatus", "Status")
        status = normalise_status(raw)
        if status == "unknown":
            word = str(raw or field(form, "Event") or "").lower().replace(" ", "").replace("-", "").replace("_", "")
            status = STATUS_WORDS.get(word, "unknown")
            if status == "unknown" and "hangup" in str(field(form, "Event") or "").lower():
                status = "completed"
        return {"sid": str(sid), "status": status, "duration_s": duration_of(field(form, "Duration", "BillDuration")),
                "ended_at": field(form, "EndTime"), "message": field(form, "HangupCauseName", "HangupCause")}


class Vobiz(Plivo):
    name = "vobiz"
    display = "Vobiz"
    env_prefix = "VOBIZ"
    default_base = VOBIZ_BASE
    codec_cls = VobizCodec
    VARS = (("auth_id", "VOBIZ_AUTH_ID"), ("token", "VOBIZ_AUTH_TOKEN"), ("caller_id", "VOBIZ_CALLER_ID"))
    OPTIONAL = (("base_url", "VOBIZ_BASE_URL", VOBIZ_BASE),)
    verification_note = ("Written from Vobiz's docs (call API, headers, <Stream> XML, start / playAudio / clearAudio / checkpoint "
                         "and hangup are documented); the media, dtmf and call-detail shapes are assumed to match Plivo's. "
                         "Not yet run against a real account.")
    inbound_how = ("In the Vobiz console create an Application whose Answer URL is the address (method POST), and attach it to "
                   "the number.")

    def auth_kwargs(self, account: Dict[str, str]) -> Dict[str, Any]:
        return {"headers": {"X-Auth-ID": account["auth_id"], "X-Auth-Token": account["token"],
                            "Content-Type": "application/json"}}


PLIVO = Plivo()
VOBIZ = Vobiz()
