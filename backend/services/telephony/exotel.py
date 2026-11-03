"""Exotel: the Voicebot applet's WebSocket (raw PCM, 8 kHz by default), the Calls/connect API and Exotel's refusals.

Moved here from routers/phone_exotel.py without a change in behaviour; that router keeps the old names as re-exports.
https://developer.exotel.com/docs/agentstream/websocket-protocol (verified live on a real account, 22-23 Oct 2026)."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from .. import phone_audio as pa
from . import base
from .base import (Event, Provider, ProviderHTTPError, StreamCodec, duration_of, env, field, http_request,
                   normalise_status)
from .numbers import mask_caller_id
from .state import AccountBook

DEFAULT_SUBDOMAIN = "api.in.exotel.com"
EXHAUSTED_WORDS = ("balance", "credit", "insufficient", "recharge", "quota", "limit exceeded")
NOT_ALLOWED_WORDS = ("whitelist", "not verified")
ACCOUNT_VARS = (("key", "EXOTEL_API_KEY"), ("token", "EXOTEL_API_TOKEN"), ("sid", "EXOTEL_ACCOUNT_SID"),
                ("caller_id", "EXOTEL_CALLER_ID"))


def classify_refusal(status: int, message: str) -> Optional[str]:
    """What an Exotel refusal means: "exhausted" (out of credit), "not_allowed" (this number cannot be called from this
    account: try another, mark nothing), "kyc" (this account has not finished KYC, so it may place no outbound call: try
    another, mark nothing), "auth_failed" (credentials refused) or None (any other refusal: stop, the next account would
    refuse it the same way)."""
    low = (message or "").lower()
    if status == 402 or any(w in low for w in EXHAUSTED_WORDS):
        return "exhausted"
    if any(w in low for w in NOT_ALLOWED_WORDS):
        return "not_allowed"
    if "kyc" in low:   # 403 "Your account is not yet KYC compliant..." (seen on two trial accounts, 23 Oct 2026)
        return "kyc"
    if status in (401, 403):
        return "auth_failed"
    return None


def exotel_message(text: str) -> str:
    """The human sentence in an Exotel error body (XML `<Message>...</Message>` or JSON), else the start of the body with
    tags removed: Exotel's own words with nothing added (refusals are classified on this)."""
    body = text or ""
    found = re.search(r"<Message>(.*?)</Message>", body, re.S | re.I)
    reason = found.group(1) if found else None
    if reason is None:
        try:
            data = json.loads(body)
            err = data.get("RestException", data) if isinstance(data, dict) else {}
            reason = (err.get("Message") or err.get("message") or err.get("error")) if isinstance(err, dict) else None
        except ValueError:
            reason = None
    reason = re.sub(r"<[^>]+>", " ", reason or body)
    return re.sub(r"\s+", " ", reason).strip(" ;.") or "no reason given"


def exotel_reason(text: str) -> str:
    """Exotel's own words for a refusal, passed on as they are, with a hint when the account is not KYC-approved."""
    reason = exotel_message(text)
    if "kyc" in reason.lower():
        reason += ". Finish KYC in the Exotel dashboard, or call a number you have whitelisted on a trial account"
    return reason[:300]


def parse_call(text: str) -> Dict[str, Any]:
    """The fields of an Exotel call from its JSON ({"Call": {...}}) or, failing that, its XML answer."""
    try:
        data = json.loads(text)
        call = data.get("Call") if isinstance(data, dict) else None
        if isinstance(call, dict):
            return {k: ("" if v is None else str(v)) for k, v in call.items() if not isinstance(v, (dict, list))}
    except ValueError:
        pass
    out: Dict[str, Any] = {}
    for tag in ("Status", "Duration", "StartTime", "EndTime", "DetailedStatus", "Message"):
        m = re.search(rf"<{tag}>([^<]*)</{tag}>", text)
        if m:
            out[tag] = m.group(1).strip()
    return out


# ------------------------------------------------------------------------------------------------------------ codec

class ExotelCodec(StreamCodec):
    """Exotel's Voicebot WebSocket: JSON events with `stream_sid`; audio is base64 raw PCM16 at the rate named in `start`
    (8, 16 or 24 kHz), in both directions."""
    name = "exotel"

    def parse(self, text: str) -> Optional[Event]:
        ev = self.load(text)
        if ev is None:
            return None
        kind = ev.get("event")
        if kind == "start":
            start = ev.get("start") or {}
            fmt = start.get("media_format") or {}
            try:
                rate = int(fmt.get("sample_rate") or pa.PHONE_RATE)
            except (TypeError, ValueError):
                rate = pa.PHONE_RATE
            return Event("start", stream_id=ev.get("stream_sid") or start.get("stream_sid"),
                         call_id=start.get("call_sid"), caller=str(start.get("from", "")), rate=rate,
                         params=start.get("custom_parameters") if isinstance(start.get("custom_parameters"), dict) else {})
        if kind == "media":
            pcm = self.b64decode((ev.get("media") or {}).get("payload"))
            return Event("media", pcm=pcm) if pcm is not None else None
        if kind == "dtmf":
            # (Exotel's exact dtmf event shape is not verified against a live account: the digit is read from
            # `dtmf.digit`, `dtmf` or `digit`.)
            d = ev.get("dtmf")
            return Event("dtmf", digit=self.digit_of(d if isinstance(d, dict) or isinstance(d, (str, int)) else ev.get("digit")))
        if kind == "mark":
            return Event("mark", name=str((ev.get("mark") or {}).get("name") or ""))
        if kind == "stop":
            return Event("stop")
        return None

    def media(self, pcm: bytes, stream_id: Optional[str]) -> dict:
        return {"event": "media", "stream_sid": stream_id, "media": {"payload": self.b64(pcm)}}

    def clear(self, stream_id: Optional[str]) -> dict:
        return {"event": "clear", "stream_sid": stream_id}

    def mark(self, name: str, stream_id: Optional[str]) -> dict:
        return {"event": "mark", "stream_sid": stream_id, "mark": {"name": name}}


# ---------------------------------------------------------------------------------------------------------- provider

def state_path() -> str:
    return os.getenv("BONCLAY_EXOTEL_STATE_FILE", "").strip() or os.path.join(base.backend_dir(), "data", "exotel_accounts.json")


def _account_key(account: Dict[str, str]) -> str:
    import hashlib  # noqa: PLC0415
    return hashlib.sha256(str(account["sid"]).encode("utf-8")).hexdigest()[:16]


# Exotel's state book. routers/phone_exotel.py binds its module globals to it (bind_namespace) and re-exports its pieces.
BOOK = AccountBook("exotel", "Exotel", "EXOTEL", state_path, _account_key)


class Exotel(Provider):
    name = "exotel"
    display = "Exotel"
    env_prefix = "EXOTEL"
    verified = True
    verification_note = "Verified on a real account (calls placed and answered, 22-23 Oct 2026)."
    VARS = ACCOUNT_VARS
    OPTIONAL = (("subdomain", "EXOTEL_SUBDOMAIN", DEFAULT_SUBDOMAIN),)
    inbound_how = ("Paste the address into the Voicebot applet of the ExoPhone's call flow (Exotel dashboard, App Bazaar, "
                   "Voicebot), then a Connect applet after it.")

    def account_id(self, account: Dict[str, str]) -> str:
        return str(account["sid"])

    def codec(self) -> StreamCodec:
        return ExotelCodec()

    async def place_call(self, account, to, stream_url, status_url, answer_url=None):
        url = f"https://{account['subdomain']}/v1/Accounts/{account['sid']}/Calls/connect"
        form = {"From": to, "CallerId": account["caller_id"], "StreamUrl": stream_url, "StreamType": "bidirectional"}
        if os.getenv("EXOTEL_STATUS_CALLBACK") == "1":  # off by default: a real Exotel account answered 500 (code 35000) with it
            form["StatusCallback"] = status_url
        r = await http_request("post", url, 20.0, auth=(account["key"], account["token"]), data=form)
        text = r.text or ""
        if r.status_code >= 300:
            raise ProviderHTTPError(r.status_code, text)
        sid = None
        try:
            sid = ((r.json().get("Call") or {}).get("Sid"))
        except Exception:  # noqa: BLE001 - Exotel answers XML unless asked for JSON
            m = re.search(r"<Sid>([^<]+)</Sid>", text)
            sid = m.group(1) if m else None
        return sid

    async def call_status(self, account, call_id):
        url = f"https://{account['subdomain']}/v1/Accounts/{account['sid']}/Calls/{call_id}.json"
        r = await http_request("get", url, 10.0, auth=(account["key"], account["token"]))
        if r.status_code >= 300:
            raise ProviderHTTPError(r.status_code, r.text or "")
        f = parse_call(r.text or "")
        return {"status": normalise_status(f.get("Status")), "duration_s": duration_of(f.get("Duration")),
                "started_at": f.get("StartTime") or None, "ended_at": f.get("EndTime") or None,
                "message": f.get("DetailedStatus") or f.get("Message") or None}

    def reason(self, body: str) -> str:
        return exotel_reason(body)

    def classify_refusal(self, status: int, body: str) -> Optional[str]:
        return classify_refusal(status, exotel_message(body))

    def parse_status_callback(self, form):
        """Field names are read leniently (CallSid or Sid, Status or CallStatus, Duration or ConversationDuration); check
        them against Exotel's current documentation before relying on this in production."""
        sid = field(form, "CallSid", "Sid", "call_sid")
        if not sid:
            return None
        status = normalise_status(field(form, "Status", "CallStatus", "call_status"))
        return {"sid": str(sid), "status": status,
                "duration_s": duration_of(field(form, "Duration", "ConversationDuration", "DialCallDuration")),
                "ended_at": field(form, "EndTime", "DateUpdated"), "message": field(form, "DetailedStatus", "Message")}


PROVIDER = Exotel()
