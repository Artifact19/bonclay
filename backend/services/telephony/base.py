"""What a phone provider is, and what its audio WebSocket looks like.

A `Provider` knows its accounts (read from backend/.env with the suffix scheme Exotel always had: account 1 uses the plain
names, accounts 2..9 the same names ending _2 .. _9), how to place a call on one of them, how to read the status of a call,
how to hang one up, and how to tell a refusal about the account (out of credit, bad credentials, number not allowed, KYC)
from any other refusal. A `StreamCodec` is the carrier's WebSocket dialect: it turns one text frame into an `Event`
and builds the frames we send back. Inside BonClay audio is always 16-bit PCM (8 kHz on every carrier except an Exotel
stream set to 16 or 24 kHz); a codec converts mu-law at its edge.

The call logic itself does not live here: routers/phone_exotel.py runs one `CallSession` for every provider.
"""
from __future__ import annotations

import base64
import dataclasses
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

MAX_ACCOUNTS = 9
EXHAUSTED = "exhausted"
NOT_ALLOWED = "not_allowed"
KYC = "kyc"
AUTH_FAILED = "auth_failed"
REFUSAL_KINDS = (EXHAUSTED, NOT_ALLOWED, KYC, AUTH_FAILED)
# How a refusal reads in the log and in the "all failed" message.
REFUSAL_WORDS = {EXHAUSTED: "out of credit", AUTH_FAILED: "credentials refused", NOT_ALLOWED: "does not allow this number",
                 KYC: "not KYC compliant"}

_warned_partial: set = set()     # (provider, label, missing names) already warned about: a half-filled account logs once


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def backend_dir() -> str:
    """backend/ (this file is backend/services/telephony/base.py)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------------------------------------------------ errors

class ProviderHTTPError(Exception):
    """The provider answered with an HTTP error. `status` and `body` are its own words, unchanged."""

    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body or ""


class ProviderUnreachable(Exception):
    """The provider could not be reached (network error, timeout). The call may or may not have been placed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def http_request(method: str, url: str, timeout: float = 20.0, **kwargs: Any):
    """One HTTP request. `httpx.AsyncClient` is looked up at call time (tests replace it); a network error becomes
    ProviderUnreachable. The response is returned whatever its status: the caller decides what a refusal means."""
    import httpx  # noqa: PLC0415
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await getattr(client, method)(url, **kwargs)
    except httpx.HTTPError as e:
        raise ProviderUnreachable(e.__class__.__name__) from e


def json_message(body: str, *keys: str) -> str:
    """The human sentence in a JSON error body: the first of `keys` found at the top level or one level down, else the
    start of the body with tags and extra spaces removed."""
    import json  # noqa: PLC0415
    text = body or ""
    found: Optional[str] = None
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        stack: List[Any] = [data]
        for key in ("error", "errors", "RestException"):
            inner = data.get(key)
            if isinstance(inner, dict):
                stack.append(inner)
            elif isinstance(inner, list) and inner and isinstance(inner[0], dict):
                stack.append(inner[0])
        for d in stack:
            for k in keys:
                v = d.get(k)
                if isinstance(v, str) and v.strip():
                    found = v
                    break
            if found:
                break
        if found is None and isinstance(data.get("error"), str):
            found = data["error"]
    reason = re.sub(r"<[^>]+>", " ", found or text)
    return re.sub(r"\s+", " ", reason).strip(" ;.") or "no reason given"


# ------------------------------------------------------------------------------------------------------------ codec

@dataclasses.dataclass
class Event:
    """One thing the carrier told us, in our terms.

    kind: "start" (stream_id, call_id, caller, rate, params), "media" (pcm: PCM16 mono at the call's rate), "dtmf"
    (digit), "mark" (name: what we asked to be told when it has played) or "stop". A frame we do not use gives None."""
    kind: str
    stream_id: Optional[str] = None
    call_id: Optional[str] = None
    caller: str = ""
    rate: int = 8000
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    pcm: bytes = b""
    digit: str = ""
    name: str = ""


class StreamCodec:
    """A carrier's WebSocket dialect. One instance per call (the Plivo family learns its audio format from `start`).

    `parse(text)` -> Event | None. The builders return the JSON object to send (the call serialises it): `media(pcm)`
    plays PCM16 to the caller, `clear()` drops what the carrier has queued (barge-in), `mark(name)` asks to be told when
    everything sent so far has played."""
    name = "base"
    rate = 8000
    chunk_bytes = 3200     # PCM bytes per media frame (200 ms at 8 kHz); every carrier accepts any size, this paces well

    def parse(self, text: str) -> Optional[Event]:  # pragma: no cover - interface
        raise NotImplementedError

    def media(self, pcm: bytes, stream_id: Optional[str]) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def clear(self, stream_id: Optional[str]) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def mark(self, name: str, stream_id: Optional[str]) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------ shared helpers
    @staticmethod
    def load(text: str) -> Optional[dict]:
        import json  # noqa: PLC0415
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            return None
        return obj if isinstance(obj, dict) else None

    @staticmethod
    def b64decode(payload: Any) -> Optional[bytes]:
        try:
            return base64.b64decode(payload or "")
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def digit_of(d: Any) -> str:
        v = d.get("digit") if isinstance(d, dict) else d
        return str(v if v is not None else "").strip()


# ---------------------------------------------------------------------------------------------------------- provider

class Provider:
    """One telephony carrier. Subclasses set the class attributes and implement the methods below."""
    name = "base"                 # the id used in settings, tokens and URLs: exotel | twilio | plivo | vobiz | telnyx
    display = "Base"              # how it reads in messages: Exotel, Twilio, ...
    env_prefix = "BASE"           # TWILIO -> TWILIO_ROTATION, TWILIO_EXHAUSTED_RESET_HOURS
    verified = True               # False: written from the carrier's docs, not yet run against a real account
    verification_note = ""
    VARS: Tuple[Tuple[str, str], ...] = ()                       # (account key, env name): all needed for an account
    OPTIONAL: Tuple[Tuple[str, str, str], ...] = ()              # (account key, env name, default)
    api_hangup = False            # the call must be ended through the API when the agent ends it (closing the stream is not enough)
    needs_answer_url = False      # the call request names an Answer URL (Plivo family) instead of carrying the answer itself
    answer_content_type = "application/xml"
    inbound_how = ""              # one line for the inbound-link route: where to paste the address

    # ------------------------------------------------------------------------------------------------- accounts
    def account_env(self, label: str) -> Dict[str, str]:
        """The raw values of one account: account 1 uses the plain names, accounts 2..9 the names with `_2` .. `_9`."""
        suffix = "" if label == "1" else f"_{label}"
        out = {k: env(name + suffix) for k, name in self.VARS}
        for k, name, default in self.OPTIONAL:
            out[k] = env(name + suffix, default) or default
        return out

    def accounts(self) -> List[Dict[str, str]]:
        """Every complete account in backend/.env, in order: [{label "1".."9", <credentials>, caller_id}]. An account
        counts only when all of VARS are set; a half-filled suffixed account is ignored with one warning (names only)."""
        out: List[Dict[str, str]] = []
        log = logging.getLogger(f"bonclay.phone.{self.name}")
        for n in range(1, MAX_ACCOUNTS + 1):
            label = str(n)
            a = self.account_env(label)
            have = [k for k, _ in self.VARS if a[k]]
            if len(have) == len(self.VARS):
                out.append({"label": label, **a})
            elif have and n > 1:
                missing = [name + f"_{label}" for k, name in self.VARS if not a[k]]
                warn_key = (self.name, label, tuple(missing))
                if warn_key not in _warned_partial:
                    _warned_partial.add(warn_key)
                    log.warning("%s account %s is ignored: set %s", self.display, label, ", ".join(missing))
        return out

    def configured(self) -> bool:
        return bool(self.accounts())

    def missing_vars(self) -> List[str]:
        """The names of account 1's settings that are not set (what to tell an owner who has nothing configured)."""
        a = self.account_env("1")
        return [name for k, name in self.VARS if not a[k]]

    def account_id(self, account: Dict[str, str]) -> str:
        """What identifies the account in the saved state (hashed there, never stored as it is)."""
        raise NotImplementedError

    # -------------------------------------------------------------------------------------------------- audio
    def codec(self) -> StreamCodec:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------ calls
    async def place_call(self, account: Dict[str, str], to: str, stream_url: str, status_url: str,
                         answer_url: Optional[str] = None) -> Optional[str]:
        """Ring `to` and connect the call to `stream_url` (a wss:// address). Returns the carrier's id for the call (None
        when the answer holds none). Raises ProviderHTTPError (the carrier refused) or ProviderUnreachable."""
        raise NotImplementedError

    async def call_status(self, account: Dict[str, str], call_id: str) -> Dict[str, Any]:
        """{"status": queued|ringing|in-progress|completed|failed|busy|no-answer|canceled|unknown, "duration_s",
        "started_at", "ended_at", "message"}. Raises ProviderHTTPError or ProviderUnreachable."""
        raise NotImplementedError

    async def hangup(self, account: Dict[str, str], call_id: str) -> bool:
        """End the call, best effort: True when the carrier took the request, False on any failure. Never raises."""
        return False

    # --------------------------------------------------------------------------------------------- refusals
    def reason(self, body: str) -> str:
        """The carrier's own words in an error body."""
        return json_message(body, "message", "detail", "title", "Message", "error")

    def classify_refusal(self, status: int, body: str) -> Optional[str]:
        """What a refusal means: "exhausted" (out of credit), "not_allowed" (this number cannot be called from this
        account: try another, mark nothing), "kyc" (this account has not finished verification: try another, mark
        nothing), "auth_failed" (credentials refused) or None (any other refusal: stop, the next account would refuse
        it the same way)."""
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------- callbacks
    def parse_status_callback(self, form: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """A status callback body (form fields or JSON) -> {"sid", "status", "duration_s", "ended_at", "message"}, or None."""
        return None

    def answer_document(self, stream_url: str) -> str:
        """The XML the carrier fetches when a call is answered, pointing the call at `stream_url`."""
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------- shared helpers
    def stream_codec_name(self) -> str:
        return self.codec().name


# ------------------------------------------------------------------------------------------------ status helpers

CALL_STATUSES = ("queued", "ringing", "in-progress", "completed", "failed", "busy", "no-answer", "canceled")


def normalise_status(raw: Any) -> str:
    s = re.sub(r"[\s_]+", "-", str(raw or "").strip().lower())
    s = {"inprogress": "in-progress", "noanswer": "no-answer", "cancelled": "canceled"}.get(s, s)
    return s if s in CALL_STATUSES else "unknown"


def duration_of(value: Any) -> Optional[int]:
    try:
        return max(0, int(float(str(value))))
    except (TypeError, ValueError):
        return None


def field(form: Dict[str, Any], *names: str) -> Any:
    """The first non-empty value among `names` in a form or JSON body, names compared without regard to case."""
    lower = {str(k).lower(): v for k, v in form.items()}
    for n in names:
        v = lower.get(n.lower())
        if v not in (None, ""):
            return v
    return None
