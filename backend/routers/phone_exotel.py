"""Phone calls over Exotel: the agent answers (or places) a real phone call and talks, with barge-in.

How it works. Exotel's Voicebot applet opens a WebSocket to us for every call and streams the caller's voice as raw
16-bit PCM (8 kHz by default, a frame about every 100 ms); whatever PCM we send back is played to the caller
(https://developer.exotel.com/docs/agentstream/websocket-protocol). This router is the other end of that socket:

    pickup (`start`)  -> the pre-rendered opening (AI disclosure + the agent's greeting, services/phone_greeting.py),
                         sent at once: no speech recognition, knowledge search or language model
    caller's audio -> voice activity detection (services/phone_audio.py) -> one utterance
                   -> the SAME streaming turn the playground and the website widget use (`_turn_event_stream`:
                      speech recognition, knowledge search, tools, language policy, AI disclosure, watermark, receipt)
                   -> each sentence as soon as it is ready -> 8 kHz PCM -> paced out to the caller
    caller pauses (PHONE_EARLY_SILENCE_MS) -> the turn starts speculatively; its audio is held until the full
                      PHONE_END_SILENCE_MS confirms the end of the sentence, and it is cancelled if the caller goes on
    caller talks over the agent -> `clear` (Exotel drops what it has queued) and the running turn is cancelled
    a turn cancelled before it was stored (the caller spoke again) hands its caller audio to the next turn, so a name given
                      in one piece and a date in the next are answered together (PHONE_CARRY_UNANSWERED)
    the wait for the end of the caller's sentence follows what the agent just asked for: a name, a number, a date or a time
                      is dictated slowly (PHONE_ADAPTIVE_END; services/turn_taking.py), and an answer that ends in "and" or
                      holds 7 digits of a number is waited for once more (PHONE_INCOMPLETE_EXTRA_MS)

Other providers (Twilio, Plivo, Vobiz, Telnyx; services/telephony/, docs/PHONE_AND_MEETINGS.md "Phone providers"): the same call
runs over their WebSocket through a codec. PHONE_PROVIDERS and PHONE_ROUTES choose the provider, with failover across providers
and accounts. Their routes are /api/phone/{provider}/stream/{token} (WS), .../status-callback/{token}, .../answer/{token}
(public, signed token, inert unless the provider is in use) and POST .../inbound-link (owner).

Routes
    WS   /api/phone/exotel/stream/{token}      Exotel connects here. The token (signed, expiring) names the agent.
    GET  /api/phone/exotel/status              owner: is Exotel configured, what is missing
    POST /api/phone/exotel/call                owner: place an outbound call to a phone that agreed to be called
    GET  /api/phone/exotel/call/{call_sid}     owner: live status of a call this owner placed (polled by the Call dialog)
    POST /api/phone/exotel/inbound-link        owner: a long-lived wss:// URL for the Voicebot applet of an ExoPhone

Settings (backend/.env): EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_ACCOUNT_SID, EXOTEL_CALLER_ID (the ExoPhone),
PUBLIC_BASE_URL (the https:// address of this server that Exotel can reach, for example an ngrok or cloudflared
tunnel), and optionally EXOTEL_SUBDOMAIN (default api.in.exotel.com), PHONE_MAX_CALL_S (default 600), PHONE_END_SILENCE_MS (default 500),
PHONE_EARLY_SILENCE_MS (default 250), PHONE_SPECULATIVE (on | asr | off, default on), PHONE_CARRY_UNANSWERED (default on),
PHONE_ADAPTIVE_END, PHONE_DETAIL_END_SILENCE_MS, PHONE_DETAILS_END_SILENCE_MS and PHONE_INCOMPLETE_EXTRA_MS (see docs/PHONE_AND_MEETINGS.md). The greeting cache and warm-up
settings (PHONE_PREWARM, PHONE_GREETING_DIR, ...) are in services/phone_greeting.py.
Several accounts: account 1 is the names above; accounts 2..9 are the same names ending _2 .. _9 (EXOTEL_API_KEY_2, ...,
optional EXOTEL_SUBDOMAIN_2). Outbound calls use the first usable account (EXOTEL_ROTATION=failover, default) or take turns
(round_robin) and move to the next account when one is out of credit or its credentials are refused; the state is kept
in data/exotel_accounts.json and an account marked exhausted is tried only after the active ones (and shown active again after EXOTEL_EXHAUSTED_RESET_HOURS, 24).
PHONE_ALLOWED_NUMBERS (comma separated) is optional: when set, only these numbers can be called; when empty, any valid
number can be called (with caller_agreed). Numbers are compared after normalising to E.164, so 09876543210,
919876543210 and +91 98765-43210 are the same Indian number.
"""
from __future__ import annotations

import asyncio
import base64
import collections
import dataclasses
import hashlib
import hmac
import io
import json
import logging
import os
import re
import sys
import threading
import time
import types
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

try:
    from core import offline
    from core.errors import AppError
    from services import call_control, call_records, caller_sheet, voicemail
    from services import phone_audio as pa
    from services import phone_greeting, turn_taking
    from services.auth import get_current_user_id
    from services.telephony import base as tbase
    from services.telephony import exotel as _exotel
    from services.telephony import registry
    from services.telephony.numbers import E164_RE, mask, mask_caller_id, mask_e164, normalise_number
except ImportError:  # pragma: no cover - imported as backend.routers.phone_exotel
    from backend.core import offline
    from backend.core.errors import AppError
    from backend.services import call_control, call_records, caller_sheet, voicemail
    from backend.services import phone_audio as pa
    from backend.services import phone_greeting, turn_taking
    from backend.services.auth import get_current_user_id
    from backend.services.telephony import base as tbase
    from backend.services.telephony import exotel as _exotel
    from backend.services.telephony import registry
    from backend.services.telephony.numbers import E164_RE, mask, mask_caller_id, mask_e164, normalise_number

log = logging.getLogger("bonclay.phone.exotel")

router = APIRouter(tags=["phone"])
# The two routes Exotel itself calls, authenticated by the signed token in their URL. They are also served alone on the
# phone-only public port (phone_public.py), which is all the tunnel exposes; the owner routes on `router` never are.
public_router = APIRouter(tags=["phone-public"])

STREAM_PATH = "/api/phone/exotel/stream"
OUTBOUND_TOKEN_TTL_S = 15 * 60
INBOUND_TOKEN_TTL_S = 30 * 24 * 3600
CALLS_PER_HOUR = 10                      # outbound calls per owner
_active_calls = 0                        # live call streams in this process; see max_calls()
BUSY_CLOSE_CODE = 4429
BUSY_FALLBACK = ("In the Exotel flow, put a Connect applet that dials your mobile (or the agent's transfer number) after the "
                 "Voicebot applet: the call reaches a person when the agent is busy, the computer is asleep or the tunnel is "
                 "down. To be verified on a real account (docs/PHONE_AND_MEETINGS.md).")


def max_calls() -> int:
    """PHONE_MAX_CALLS (default 2): live calls this computer takes at once. A further call is refused before its stream
    is accepted (close code 4429), so the Exotel flow moves on to its next applet, a Connect step to the owner's mobile.
    On a 16 GB laptop two calls already share one model worker each; a third would slow both."""
    try:
        return max(1, int(_env("PHONE_MAX_CALLS", "2") or 2))
    except ValueError:
        return 2
IMPATIENT_MS = 1500                      # a shorter utterance while the agent is still thinking is not a new question
_CLIP_URL_RE = re.compile(r"^/api/conversations/turns/([0-9a-f-]{36})/sentences/(\d+)/audio$")
POLLS_PER_MINUTE = 120                   # call-status polls per owner (the dialog polls every ~2 s: 30 a minute)
KNOWN_CALLS_MAX = 500
_calls_by_owner: "collections.defaultdict[str, collections.deque]" = collections.defaultdict(collections.deque)
_polls_by_owner: "collections.defaultdict[str, collections.deque]" = collections.defaultdict(collections.deque)
# call_sid -> {"owner", "agent_id"}: only the owner who placed a call can read its status (newest kept, oldest dropped)
_known_calls: "collections.OrderedDict[str, Dict[str, str]]" = collections.OrderedDict()


def _b():
    """The backend module (imported lazily so this router never creates an import cycle)."""
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "local_db"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "local_db"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


def _widget():
    try:
        from routers import widget  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from backend.routers import widget  # noqa: PLC0415
    return widget


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


# ------------------------------------------------------------------------------------------------ settings
# Providers (Exotel, Twilio, Plivo, Vobiz, Telnyx) live in services/telephony/. Exotel's settings, account state and
# refusal rules are defined there (services/telephony/exotel.py) and re-exported here under their old names.

def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


DEFAULT_SUBDOMAIN = _exotel.DEFAULT_SUBDOMAIN
MAX_ACCOUNTS = tbase.MAX_ACCOUNTS
_ACCOUNT_VARS = _exotel.ACCOUNT_VARS
_warned_partial = tbase._warned_partial   # the same set: a half-filled account logs once


def _account_env(label: str) -> Dict[str, str]:
    """The raw values of one Exotel account: account 1 uses the plain names, accounts 2..9 the names with `_2` .. `_9`."""
    return _exotel.PROVIDER.account_env(label)


def exotel_accounts() -> List[Dict[str, str]]:
    """Every complete Exotel account in backend/.env, in order: [{label "1".."9", key, token, sid, caller_id, subdomain}].
    Account 1 uses EXOTEL_API_KEY ...; account N (2..9) the same names ending _N. An account counts only when its key,
    token, SID and caller id are all set; a half-filled suffixed account is ignored with one warning (names only)."""
    return _exotel.PROVIDER.accounts()


def exotel_settings() -> Dict[str, str]:
    """The first complete account (account 1 when none is complete) with the public address: the old single-account shape."""
    accounts = exotel_accounts()
    a = accounts[0] if accounts else _account_env("1")
    return {"key": a["key"], "token": a["token"], "sid": a["sid"], "caller_id": a["caller_id"],
            "subdomain": a["subdomain"], "public": _env("PUBLIC_BASE_URL").rstrip("/")}


def missing_settings() -> List[str]:
    """[] when at least one complete account (of any provider in PHONE_PROVIDERS / PHONE_ROUTES) exists and PUBLIC_BASE_URL
    is set; otherwise the names still to be set (the first provider's when nothing is configured: Exotel by default)."""
    return registry.missing_settings()


def _not_set_up(miss: List[str]) -> str:
    names = registry.enabled()
    who = registry.provider(names[0]).display if names == ["exotel"] else "Phone calling"
    return f"{who} is not set up: add " + ", ".join(miss) + " to backend/.env."


# ------------------------------------------------------------------------------------ account state and rotation
# Each account is `active`, `exhausted` (out of credit) or `auth_failed` (credentials refused); see
# services/telephony/state.py for the rules. Exotel's state lives in data/exotel_accounts.json, keyed by a hash of the
# account SID. This module's `_now_ts`, `_rr_counter` and `_state_loaded_from` are the clock, the round-robin counter and
# the file marker of Exotel's book (bound below), where the tests set them.

_STATE_LOCK = _exotel.BOOK.lock
_account_state = _exotel.BOOK.states          # sha256(sid)[:16] -> {"state", "since" (epoch seconds)}
_state_loaded_from: Optional[str] = None
_rr_counter = 0
EXHAUSTED_WORDS = _exotel.EXHAUSTED_WORDS
NOT_ALLOWED_WORDS = _exotel.NOT_ALLOWED_WORDS
classify_refusal = _exotel.classify_refusal


def _now_ts() -> float:
    return time.time()


_exotel.BOOK.bind_namespace(globals())


def accounts_state_path() -> str:
    return _exotel.state_path()


def _account_key(account: Dict[str, str]) -> str:
    return _exotel._account_key(account)


def _load_state() -> None:
    _exotel.BOOK._load()


def _save_state() -> None:
    _exotel.BOOK._save()


def _reset_seconds() -> float:
    return _exotel.BOOK.reset_seconds()


def account_state(account: Dict[str, str]) -> Dict[str, Any]:
    """{"state": "active" | "exhausted" | "auth_failed", "since": epoch seconds or None}. A refused account counts as
    active again once the reset window has passed."""
    return _exotel.BOOK.state(account)


def _set_account_state(account: Dict[str, str], state: str) -> None:
    _exotel.BOOK.set(account, state)


def _dial_order(accounts: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """The Exotel accounts to try for one call, each at most once: the active accounts first (in order, or rotated by one
    each call with EXOTEL_ROTATION=round_robin), then the exhausted / auth_failed ones in their configured order."""
    return _exotel.BOOK.dial_order(accounts)


def account_by_label(label: Optional[str]) -> Optional[Dict[str, str]]:
    """The configured Exotel account with this label; else the first one (the call was placed before a restart that
    removed its account); else None."""
    return registry.account_by_label("exotel", label)


def accounts_public() -> List[Dict[str, Any]]:
    """Per Exotel account, for the status route: label, masked caller id, state and since. Never the key, token or SID."""
    return _exotel.BOOK.public(exotel_accounts(), mask_caller_id)


SAVED_NUMBERS_MAX = 20
LABEL_MAX = 40
_saved_lock = threading.Lock()


def numbers_path() -> str:
    return os.getenv("BONCLAY_PHONE_NUMBERS_FILE", "").strip() or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "phone_numbers.json")


def saved_numbers() -> List[dict]:
    """The numbers the owner added in the app: [{id, label, number (E.164), added_at}]."""
    try:
        with open(numbers_path(), "r", encoding="utf-8") as f:
            rows = json.load(f)
    except (OSError, ValueError):
        return []
    return [r for r in rows if isinstance(r, dict) and normalise_number(r.get("number")) and r.get("id")] \
        if isinstance(rows, list) else []


def _write_saved(rows: List[dict]) -> None:
    path = numbers_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)   # phone numbers are personal data
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _allowed_env_raw() -> str:
    """PHONE_ALLOWED_NUMBERS, re-read from backend/.env on every use, so a number the owner adds there can be called
    without restarting the API (everything else in .env is read once at start). The process environment wins when
    the file does not set it; the test suite never reads the real .env (BONCLAY_SKIP_DOTENV)."""
    if not os.getenv("BONCLAY_SKIP_DOTENV"):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        try:
            from dotenv import dotenv_values  # noqa: PLC0415
            value = dotenv_values(path).get("PHONE_ALLOWED_NUMBERS") if os.path.exists(path) else None
        except Exception:  # noqa: BLE001 - an unreadable file falls back to the environment
            value = None
        if value is not None:
            return value.strip()
    return _env("PHONE_ALLOWED_NUMBERS")


def allowed_entries() -> List[dict]:
    """Everyone who may be called: the numbers from PHONE_ALLOWED_NUMBERS in backend/.env, then the ones saved in the app.
    Each is {number (E.164), label, source ("env" or "saved"), id (saved only)}. Duplicates are dropped (the first wins)."""
    out: List[dict] = []
    seen = set()
    for raw in _allowed_env_raw().split(","):
        n = normalise_number(raw)
        if n and n not in seen:
            seen.add(n)
            out.append({"number": n, "label": "Allowed in backend/.env", "source": "env", "id": None})
    for r in saved_numbers():
        n = normalise_number(r["number"])
        if n not in seen:
            seen.add(n)
            out.append({"number": n, "label": str(r.get("label") or "Saved number"), "source": "saved", "id": r["id"]})
    return out


def allowed_numbers() -> List[str]:
    """E.164 numbers that may be called (see allowed_entries)."""
    return [e["number"] for e in allowed_entries()]


def _public_base() -> str:
    return _env("PUBLIC_BASE_URL").rstrip("/")


STATUS_CALLBACK_PATH = "/api/phone/exotel/status-callback"
PROVIDER_NAMES = tuple(n for n in registry.NAMES if n != "exotel")      # the providers with routes of their own


def provider_path(provider: str, what: str) -> str:
    """/api/phone/{provider}/{stream|status-callback|answer|inbound-link}. Exotel's own paths are the ones it always had."""
    return f"/api/phone/{provider}/{what}"


def stream_url(token: str, provider: str = "exotel") -> str:
    """The wss:// address the carrier connects to, built from PUBLIC_BASE_URL. Exotel keeps its own path
    (/api/phone/exotel/stream/{token}); every other provider has /api/phone/{provider}/stream/{token}."""
    base = _public_base()
    if not base:
        who = registry.provider(provider).display
        raise AppError("PHONE_NOT_CONFIGURED", "Set PUBLIC_BASE_URL in backend/.env to the https address of this "
                       f"server that {who} can reach (for example an ngrok tunnel).", 503)
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return f"{base}{STREAM_PATH if provider == 'exotel' else provider_path(provider, 'stream')}/{token}"


def status_callback_url(token: str, provider: str = "exotel") -> str:
    """The https address the carrier posts call-status changes to. The token proves it is the URL we handed out."""
    base = _public_base()
    path = STATUS_CALLBACK_PATH if provider == "exotel" else provider_path(provider, "status-callback")
    return f"{base}{path}/{token}" if base else ""


def answer_url(token: str, provider: str) -> str:
    """The https address a carrier fetches the answer document from (TwiML, Plivo XML) when a call is answered."""
    base = _public_base()
    if not base:
        raise AppError("PHONE_NOT_CONFIGURED", "Set PUBLIC_BASE_URL in backend/.env to the https address of this server "
                       f"that {registry.provider(provider).display} can reach (for example an ngrok tunnel).", 503)
    return f"{base}{provider_path(provider, 'answer')}/{token}"


# -------------------------------------------------------------------------------------------------- tokens

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _mac(payload: str) -> str:
    return _b64(hmac.new(_widget()._secret(), ("phone|" + payload).encode("utf-8"), hashlib.sha256).digest())[:32]


def make_token(agent_id: str, user_id: str, mode: str, ttl: int, now: Optional[float] = None,
               provider: Optional[str] = None, kind: Optional[str] = None) -> str:
    """A signed, expiring token naming the agent. `provider` (a name other than exotel) binds it to that provider's routes;
    `kind="ans"` makes it an answer-document token, which opens no stream. Exotel's stream tokens carry neither."""
    claims = {"a": agent_id, "u": user_id, "m": mode, "x": int((now if now is not None else time.time()) + ttl)}
    if provider and provider != "exotel":
        claims["p"] = provider
    if kind:
        claims["k"] = kind
    payload = _b64(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    return f"{payload}.{_mac(payload)}"


def read_token(token: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """The claims of a valid, unexpired token, else None."""
    try:
        payload, _, mac = str(token).partition(".")
        if not payload or not hmac.compare_digest(mac, _mac(payload)):
            return None
        claims = json.loads(_unb64(payload))
        if not isinstance(claims, dict) or claims.get("x", 0) < (now if now is not None else time.time()):
            return None
        if not all(isinstance(claims.get(k), str) and claims[k] for k in ("a", "u", "m")):
            return None
        return claims
    except Exception:  # noqa: BLE001 - garbage in, no claims out
        return None


# ------------------------------------------------------------------------------------------ audio helpers

def wav_to_float(data: bytes):
    """A WAV (any PCM or float format) -> (mono float32, sample rate)."""
    try:
        import soundfile as sf  # noqa: PLC0415
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        import wave  # noqa: PLC0415
        with wave.open(io.BytesIO(data), "rb") as w:
            sr, ch, frames = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
        x = pa.pcm16_to_float(frames)
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, int(sr)


def _event_parts(chunk: bytes):
    try:
        head, _, rest = chunk.decode("utf-8").partition("\n")
        if not head.startswith("event: ") or not rest.startswith("data: "):
            return None
        return head[7:].strip(), json.loads(rest[6:].strip())
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ the call

MAX_MONITORS = 3
MONITOR_TOKEN_TTL_S = 5 * 60
MONITOR_AUDIO_MAX_B64 = 48_000      # one chunk of a person's voice from the browser (a few hundred ms)


class _Monitor:
    """One listener. Messages for it wait in a bounded queue, so a slow browser can never slow down the call: when the
    queue is full its oldest news is simply dropped."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()      # the listener's loop (the call's loop may be another thread's)
        self.queue: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=400)

    def put(self, msg: dict) -> None:
        try:
            self.loop.call_soon_threadsafe(self._put, msg)
        except RuntimeError:       # the listener's loop is gone
            pass

    def _put(self, msg: dict) -> None:
        try:
            self.queue.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(msg)
            except Exception:  # noqa: BLE001
                pass


@dataclasses.dataclass
class _Speculation:
    """A turn started at the caller's early pause (PHONE_EARLY_SILENCE_MS) whose audio is held until the end of the
    utterance is confirmed (PHONE_END_SILENCE_MS), or cancelled when the caller goes on talking.

    `heard` is what speech recognition made of the audio so far (None until it has). When it looks unfinished
    (services/turn_taking.is_incomplete) the confirmation waits once more (`extended`; `deferred` while it waits): the
    caller going on in that time cancels the turn and its audio is carried into the next one."""
    task: asyncio.Task
    gate: asyncio.Event
    turn_id: str
    n: int
    started: float
    pieces: List[bytes] = dataclasses.field(default_factory=list)
    heard: Optional[str] = None
    extended: bool = False
    deferred: bool = False


@dataclasses.dataclass
class _Inflight:
    """The running turn's caller audio, for CARRY: `pieces` are the PCM pieces (at the call's rate) the turn answers,
    in the order said: pieces of earlier turns that were cancelled before they were stored, then the new utterance.
    `stored` is True once the turn is in the conversation's history (the `done` event, or the turn's own flag)."""
    task: Optional[asyncio.Task]
    n: int
    pieces: List[bytes]
    run: Any = None
    done: bool = False
    noise: bool = False
    heard: bool = False

    @property
    def stored(self) -> bool:
        return self.done or bool(getattr(self.run, "stored", False))


@dataclasses.dataclass
class _Precheck:
    """The same wait for an unfinished answer as _Speculation has, for a call that does not start replies early (an agent
    with tools, or PHONE_SPECULATIVE=off) after the agent asked for details. At the caller's pause (or, without a pause,
    at the end silence) the recording goes to speech recognition on its own: no reply is started, nothing is spoken or
    stored. When the caller has finished, that result is the turn's (`asr_pre`: no second recognition); when it ends in
    "and" or holds part of a number, the turn is started only after one more wait, and the caller going on in that time
    cancels it (its audio is carried into the next one)."""
    pieces: List[bytes]
    wav: bytes
    task: "asyncio.Task"
    at: float
    deciding: bool = False       # the end silence was reached: the turn is being decided (or waited for)
    extended: bool = False


CARRY_MAX_PIECES = 3          # cancelled pieces kept for the next turn
CARRY_MAX_S = 25.0            # the clip handed to speech recognition (carried pieces and the new one) stays under this
CARRY_GAP_MS = 300            # silence put between two pieces


def speculative_mode() -> str:
    """PHONE_SPECULATIVE: "on" (default: the whole turn runs during the pause, only the audio waits), "asr" (only speech
    recognition runs during the pause: the turn waits right after the transcript, so a cancelled one has done nothing
    but recognise speech), or "off"."""
    v = _env("PHONE_SPECULATIVE", "on").lower()
    if v in ("off", "0", "false", "no"):
        return "off"
    return "asr" if v == "asr" else "on"


class CallSession:
    """One phone call. `ws` is the socket the carrier opened; `claims` say which agent answers and through which provider.
    The carrier's WebSocket dialect is `codec` (services/telephony: Exotel's by default); everything else is the same for
    every provider."""
    codec: "tbase.StreamCodec" = _exotel.ExotelCodec()      # (an instance of its own replaces it in __init__)
    provider_name = "exotel"

    def __init__(self, ws: WebSocket, claims: Dict[str, Any], agent: dict, accepted_at: Optional[float] = None,
                 codec: Optional["tbase.StreamCodec"] = None):
        self.ws = ws
        self.claims = claims
        self.provider_name = (claims.get("p") if isinstance(claims, dict) else None) or "exotel"
        self.codec = codec if codec is not None else _exotel.ExotelCodec()
        self.agent = agent
        self.b = _b()
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None
        self.rate = pa.PHONE_RATE
        self.detector: Optional[pa.UtteranceDetector] = None
        self.pacer = pa.Pacer(speed=float(_env("PHONE_PACE_SPEED", "1") or 1))   # > 1 only in tests (a faster clock)
        self.conversation_id: Optional[str] = None
        self.turn_task: Optional[asyncio.Task] = None
        self.turns = 0
        self.turn_audio = False     # has the running turn sent any audio yet
        self.turn_speculative = False
        self.barges = 0
        self.started_at = time.time()
        self.end_reason: Optional[str] = None
        self.outbound = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None   # the loop this call runs on (set in run())
        self.monitors: List["_Monitor"] = []          # people listening in from the dashboard (see monitor_socket)
        self.human_active = False                    # a person has taken the call over: the agent is silent
        self.human_monitor: Optional["_Monitor"] = None
        self.pending_action: Optional[str] = None     # "end" | "transfer": done once the agent's reply has been spoken
        self._first_heard = True          # the callee's first utterance is checked for an answering machine
        self._utter_ms = 0                # speech length of the utterance the running turn answers
        self.dtmf_presses = 0
        self.human_requested = False
        self.accepted_at = accepted_at if accepted_at is not None else time.monotonic()
        self._send_lock = asyncio.Lock()
        self.sent_audio_s = 0.0
        self.log: List[dict] = []   # what happened, for tests and the log line at the end
        self.tm: Dict[str, float] = {}   # monotonic marks of the running turn, for the per-turn latency line
        self.end_silence_ms = 500          # the base (PHONE_END_SILENCE_MS); the detector's own varies, see end_wait
        self.end_wait = turn_taking.EndWait(self.end_silence_ms, "normal")   # for the caller's next utterance
        self.last_agent_text = ""          # what the agent last said (its stored reply, or the greeting)
        self._wait_label = ""              # the end wait of the turn that is running, for its latency line
        self._wait_ms = 0
        self._carry: List[bytes] = []      # caller audio of cancelled, never-stored turns, for the next turn
        self._inflight: Optional[_Inflight] = None
        self._defer_task: Optional[asyncio.Task] = None
        self._pre: Optional[_Precheck] = None
        self._pre_task: Optional[asyncio.Task] = None
        self.spec_mode = speculative_mode()
        self.spec: Optional[_Speculation] = None
        self.spec_stats = {"started": 0, "confirmed": 0, "cancelled": 0}
        self.greeting: Optional[Any] = None          # phone_greeting.Greeting played on this call
        self._greeting_task: Optional[asyncio.Task] = None
        self._greeting_first_at: Optional[float] = None
        self._greeting_cut_at: Optional[float] = None
        self._greeting_cut = asyncio.Event()
        self._conv_ready = asyncio.Event()
        self._conv_ready.set()                        # set again once `start` has opened the conversation
        self._tasks: set = set()

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ---------------------------------------------------------------------------------------------- sending
    async def _send(self, obj: dict) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(obj))

    def _mark(self, key: str) -> None:
        self.tm.setdefault(key, time.monotonic())

    def _report_turn(self, n: int) -> None:
        """One line per turn: ms from the end-of-speech event to each stage. `end` is the utterance event, which fires
        end_silence_ms after the caller's last word, so the caller waited that much longer. A speculative turn
        started before `end` (its `start` is negative)."""
        t = self.tm
        base = t.get("end")
        if base is None:
            return
        row: Dict[str, Any] = {"event": "latency", "turn": n, "speculative": self.turn_speculative,
                               "end_wait": self._wait_label}
        for key in ("start", "heard", "sentence", "audio"):
            if key in t:
                row[key + "_ms"] = int((t[key] - base) * 1000)
        self.log.append(row)
        log.info("phone turn %d latency (ms after the end-of-speech event): %s | speculative start: %s | end_wait=%s | "
                 "first audio %s ms after the caller's last word", n,
                 ", ".join(f"{k[:-3]}={v}" for k, v in row.items() if k.endswith("_ms")),
                 "yes" if self.turn_speculative else "no", self._wait_label or "?",
                 row["audio_ms"] + (self._wait_ms or self.end_silence_ms) if "audio_ms" in row else "?")

    def tap(self, msg: dict) -> None:
        for m in list(self.monitors):
            m.put(msg)

    def _tap_audio(self, who: str, pcm: bytes) -> None:
        if self.monitors and pcm:
            self.tap({"type": "audio", "from": who, "rate": self.rate, "pcm": base64.b64encode(pcm).decode("ascii")})

    async def takeover(self, monitor: "_Monitor") -> bool:
        """A person takes the call: the agent stops talking and stops listening until `release`. Flags the
        conversation for a person and records it. Returns False when somebody else already has the call."""
        if self.human_active:
            return False
        self.human_active, self.human_monitor = True, monitor
        if self.spec is not None:
            await self._cancel_speculation()
        self._drop_pre()
        await self.stop_turn()
        await self.clear()
        self.log.append({"event": "takeover"})
        call_records.upsert(self.call_sid, outcome="human_took_over")
        await self._flag_human()
        self.tap({"type": "state", "human": True})
        return True

    async def release(self) -> None:
        """The person hands the call back: the agent listens again."""
        if not self.human_active:
            return
        self.human_active, self.human_monitor = False, None
        reset = getattr(self.detector, "reset", None)
        if callable(reset):
            reset()            # whatever the caller said while the person talked is not an utterance for the agent
        self.log.append({"event": "released"})
        self.tap({"type": "state", "human": False})

    async def send_pcm(self, pcm: bytes, on_first=None, who: str = "agent") -> None:
        """PCM16 at the call's rate to the caller, paced just ahead of real time. `on_first` runs once the first chunk
        is out. `who` only labels the audio for people listening in ("agent", or "human" after a takeover)."""
        if "audio" not in self.tm:
            self._mark("audio")
            self._report_turn(self.turns)
        self.turn_audio = True
        for chunk in pa.chunk_pcm(pcm, self.codec.chunk_bytes):
            await self.pacer.wait()
            await self._send(self.codec.media(chunk, self.stream_sid))
            self._tap_audio(who, chunk)
            if on_first is not None:
                on_first()
                on_first = None
            seconds = pa.pcm_seconds(len(chunk), self.rate)
            self.pacer.sent(seconds)
            self.sent_audio_s += seconds

    async def clear(self) -> None:
        self.pacer.reset()
        await self._send(self.codec.clear(self.stream_sid))
        self.log.append({"event": "clear"})

    # --------------------------------------------------------------------------------------------- greeting
    async def _greet(self) -> None:
        """The opening (AI disclosure + greeting), pre-rendered by services/phone_greeting.py: sent the moment the
        call starts. No speech recognition, knowledge search or language model; a cold cache renders just this text."""
        t0 = time.monotonic()
        try:
            g, pcm, source = await phone_greeting.for_call(self.b, self.agent, self.claims["u"], self.claims["m"],
                                                           self.rate)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - no greeting: the first reply discloses (its history has no agent turn)
            log.exception("phone greeting unavailable (agent %s); the first reply will say the AI disclosure",
                          self.agent.get("id"))
            self.log.append({"event": "greeting_failed"})
            return
        self.greeting = g
        self.last_agent_text = g.text
        self._apply_end_wait()   # a greeting that asks for a name or a number is answered more slowly

        def first_out() -> None:
            now = time.monotonic()
            self._greeting_first_at = now
            row = {"event": "greeting", "source": source, "clip_id": g.clip_id, "receipt_id": g.receipt_id,
                   "accept_to_start_ms": int((t0 - self.accepted_at) * 1000),
                   "start_to_first_byte_ms": int((now - t0) * 1000)}
            self.log.append(row)
            log.info("phone call %s greeting: accept->start %d ms, start->first byte %d ms (%s; clip %s, receipt %s, %s)",
                     self.call_sid, row["accept_to_start_ms"], row["start_to_first_byte_ms"], source, g.clip_id,
                     g.receipt_id, g.engine)
            self._spawn(self._record_greeting(g, now, row["start_to_first_byte_ms"]))

        await self.send_pcm(pcm, on_first=first_out)
        await self._send(self.codec.mark("turn-1-end", self.stream_sid))

    async def _open_conversation(self) -> None:
        try:
            self.conversation_id = await phone_greeting.create_conversation(self.b, self.agent, self.claims["u"],
                                                                            self.claims["m"])
            if self.conversation_id:
                call_control.register(self.conversation_id, self)
        except Exception as e:  # noqa: BLE001 - the first turn then creates it, as before
            log.warning("could not open the call's conversation: %r", e)
        finally:
            self._conv_ready.set()

    async def _record_greeting(self, g: Any, first_at: float, first_ms: int) -> None:
        """Store the greeting as the conversation's first turn once the caller has heard the AI disclosure. Cut off by
        the caller before the disclosure finished: nothing is stored, so the first reply says the disclosure."""
        try:
            await self._conv_ready.wait()
            if not self.conversation_id:
                return
            if g.discloses:
                wait = g.disclosure_s - (time.monotonic() - first_at)
                if wait > 0:
                    try:   # until the disclosure has played, or the caller cut it
                        await asyncio.wait_for(self._greeting_cut.wait(), timeout=wait)
                    except asyncio.TimeoutError:
                        pass
                cut = self._greeting_cut_at
                if cut is not None and cut - first_at < g.disclosure_s:
                    log.info("phone call %s: the caller cut the greeting before the AI disclosure ended; the first "
                             "reply says it", self.call_sid)
                    self.log.append({"event": "greeting_not_recorded"})
                    return
            turn_id = await phone_greeting.record_turn(self.b, g, self.agent, self.claims["u"], self.claims["m"],
                                                       self.conversation_id, first_ms)
            self.log.append({"event": "greeting_recorded", "turn_id": turn_id})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("could not store the greeting turn: %r", e)

    # ------------------------------------------------------------------------------------------------ turns
    @property
    def thinking(self) -> bool:
        """A turn is running and has not said anything yet (speech recognition, the model, a slow voice)."""
        return self.turn_task is not None and not self.turn_task.done() and not self.turn_audio

    @property
    def speaking(self) -> bool:
        """The caller can hear the agent: audio of this turn is going out or still queued at Exotel."""
        running = self.turn_task is not None and not self.turn_task.done()
        return (running and self.turn_audio) or self.pacer.playing

    def start_turn(self, wav: bytes, *, speculative: bool = False, end: Optional[float] = None,
                   pieces: Optional[List[bytes]] = None, asr_pre: Optional["asyncio.Future"] = None) -> None:
        """Start a turn on `wav` (the caller's audio for speech recognition). `pieces` are the PCM pieces that make
        it up (see _Inflight): kept so that the turn, when it is cancelled before it is stored, hands them on."""
        self.turn_audio = False
        self.turns += 1
        self.tm = {"end": end} if end is not None else {}
        self.turn_speculative = speculative
        self._mark("start")
        gate = asyncio.Event()
        if not speculative:
            gate.set()
            self._note_wait()
        turn_id = str(uuid.uuid4())
        inflight = _Inflight(None, self.turns, list(pieces or []))
        self._inflight = inflight
        self.turn_task = inflight.task = asyncio.create_task(self._turn(wav, self.turns, gate, turn_id, inflight,
                                                                        asr_pre))
        if speculative:
            self.spec = _Speculation(self.turn_task, gate, turn_id, self.turns, time.monotonic(),
                                     pieces=list(pieces or []))
            self.spec_stats["started"] += 1

    def _note_wait(self, extra_ms: int = 0, reason: str = "") -> None:
        """What the caller waited for the end of this utterance, for the turn's latency line: the end silence the
        detector used (services/turn_taking.end_wait) and, once, the extra wait for an unfinished answer."""
        ms = self.detector.end_silence_ms if self.detector is not None else self.end_silence_ms
        label = f"{ms}({self.end_wait.kind})"
        if extra_ms:
            label += f"+{extra_ms}(incomplete:{reason})"
        self._wait_ms, self._wait_label = ms + extra_ms, label

    # ------------------------------------------------------------------------------------ the caller's words
    def _fit_pieces(self, pieces: List[bytes]) -> List[bytes]:
        """`pieces` (the caller's audio, oldest first) as they go to speech recognition: at most CARRY_MAX_S seconds in
        all with the gaps. The oldest part goes first, whole pieces, then the start of the oldest piece left."""
        gap = int(self.rate * CARRY_GAP_MS / 1000) * 2
        limit = int(CARRY_MAX_S * self.rate) * 2
        out = list(pieces)
        while len(out) > 1 and sum(len(x) for x in out) + gap * (len(out) - 1) > limit:
            out.pop(0)
        if out and len(out[0]) > limit:
            out[0] = out[0][len(out[0]) - limit:]
            out[0] = out[0][len(out[0]) % 2:]
        return out

    def _joined(self, pieces: List[bytes]) -> bytes:
        """The pieces as one clip, a short silence between them."""
        silence = b"\x00\x00" * int(self.rate * CARRY_GAP_MS / 1000)
        return silence.join(pieces)

    def _pieces_for(self, audio: bytes) -> List[bytes]:
        """The carried pieces (when PHONE_CARRY_UNANSWERED is on) followed by the new utterance's `audio`, fitted."""
        carried = list(self._carry) if turn_taking.carry_enabled() else []
        return self._fit_pieces(carried + [audio])

    def _clip_for(self, pieces: List[bytes]) -> bytes:
        return pa.to_asr_wav(self._joined(pieces), self.rate)

    def _keep_words(self, inflight: Optional[_Inflight], task: Optional[asyncio.Task]) -> bool:
        """The turn that was just stopped was cancelled before it was stored: its caller audio is carried into the next
        turn. True when something was carried. A stored turn is never carried (its words are in the history), and
        neither is audio that held no speech."""
        if (inflight is None or inflight.task is not task or inflight.stored or inflight.noise
                or not turn_taking.carry_enabled() or not inflight.pieces):
            return False
        self._carry = inflight.pieces[-CARRY_MAX_PIECES:]
        ms = int(sum(pa.pcm_seconds(len(x), self.rate) for x in self._carry) * 1000)
        log.info("phone turn %d: carried unanswered caller audio (%d ms%s) into the next turn", inflight.n, ms,
                 ", already transcribed" if inflight.heard else "")
        self.log.append({"event": "carried", "turn": inflight.n, "ms": ms, "pieces": len(self._carry)})
        return True

    def _keep_unanswered(self, inflight: Optional[_Inflight]) -> None:
        """A turn that failed (an error event, an exception) answered nobody and stored nothing: the next turn has its words
        too, so what the caller said is not lost to a failure either."""
        if (inflight is not None and self._inflight is inflight and not inflight.stored and not inflight.noise
                and turn_taking.carry_enabled() and inflight.pieces):
            self._carry = inflight.pieces[-CARRY_MAX_PIECES:]
            self.log.append({"event": "carried", "turn": inflight.n, "pieces": len(self._carry), "after": "failure"})

    async def stop_turn_keeping_words(self) -> None:
        """stop_turn(), for a caller who spoke again: the words of the turn it cancels are not lost."""
        task, inflight = self.turn_task, self._inflight
        running = task is not None and not task.done()
        await self.stop_turn()
        if running:
            self._keep_words(inflight, task)

    async def stop_turn(self) -> None:
        task = self.turn_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            if task is self._greeting_task:
                self._greeting_cut_at = time.monotonic()
                self._greeting_cut.set()

    async def _turn(self, wav: bytes, n: int, gate: asyncio.Event, turn_id: str,
                    inflight: Optional[_Inflight] = None, asr_pre: Optional["asyncio.Future"] = None) -> None:
        """One reply. `gate` is set for a normal turn; a speculative turn's audio (and, with PHONE_SPECULATIVE=asr,
        everything after speech recognition) waits for it."""
        b = self.b
        gen = None
        try:
            try:
                await asyncio.wait_for(self._conv_ready.wait(), timeout=5.0)   # the call's conversation, from `start`
            except asyncio.TimeoutError:
                pass
            run = await b._turn_begin(None, self.agent["id"], self.agent["user_id"], storage_mode=self.claims["m"],
                                      agent=self.agent, source="phone")
            run.commit_gate = gate   # for backend._turn_event_stream: store nothing before the turn is confirmed
            run.asr_previous_agent_text = self.last_agent_text   # for the Whisper prompt (services/caller_memory.py)
            run.asr_pre = asr_pre     # this recording's speech recognition, when the call already started it
            if inflight is not None:
                inflight.run = run
            b._remember_stream_turn(turn_id, run.user_id)
            gen = b._turn_event_stream(run, wav or b"", self.conversation_id, turn_id)
            async for chunk in gen:
                parts = _event_parts(chunk)
                if parts is None:
                    continue
                name, data = parts
                if name == "sentence":
                    self._mark("sentence")
                    pcm = await self._sentence_pcm(str(data.get("audio_url") or ""))
                    if pcm:
                        await gate.wait()   # a speculative turn speaks only once the caller has really finished
                        await self.send_pcm(pcm)
                elif name == "transcript":
                    self._mark("heard")
                    if inflight is not None:
                        inflight.heard = True
                    spec = self.spec
                    if spec is not None and spec.turn_id == turn_id:
                        spec.heard = str(data.get("user_text") or "")
                    self.log.append({"event": "heard", "text": data.get("user_text", "")})
                    if data.get("user_text"):
                        self.tap({"type": "transcript", "role": "caller", "text": data["user_text"]})
                    if self.spec_mode == "asr":
                        await gate.wait()   # PHONE_SPECULATIVE=asr: nothing past speech recognition before the end
                    if self._first_heard and data.get("user_text"):
                        self._first_heard = False
                        if self.outbound and voicemail.enabled():
                            verdict = voicemail.classify(data.get("user_text"), self._utter_ms)
                            if verdict["voicemail"]:
                                await gate.wait()   # the callee has really finished
                                await self._hang_up_on_voicemail(verdict)
                                return
                elif name == "done":
                    if not gate.is_set():   # the whole turn finished inside the pause and has been stored
                        log.warning("phone turn %d: a speculative turn was stored before the caller's end of speech was "
                                    "confirmed (turn %s)", n, turn_id)
                        self.log.append({"event": "speculative_stored_early", "turn_id": turn_id})
                    self.conversation_id = data.get("conversation_id") or self.conversation_id
                    if inflight is not None:
                        inflight.done = True
                        if self._inflight is inflight:
                            self._carry = []          # its words are in the history now: nothing to carry
                    self.last_agent_text = str(data.get("agent_text") or "")
                    self._apply_end_wait()
                    timings = data.get("timings") or {}
                    self.log.append({"event": "said", "text": data.get("agent_text", ""), "ms": timings.get("total_ms")})
                    if data.get("agent_text"):
                        self.tap({"type": "transcript", "role": "agent", "text": data["agent_text"]})
                    log.info("phone turn %d pipeline (ms): %s | model %s/%s", n,
                             ", ".join(f"{k.removesuffix('_ms')}={v}" for k, v in timings.items()
                                       if k.endswith("_ms") and isinstance(v, (int, float))),
                             data.get("llm_provider"), data.get("llm_model"))
                elif name == "error":
                    self.log.append({"event": "error", "code": data.get("code")})
                    if data.get("code") == "NO_SPEECH":
                        if inflight is not None:
                            inflight.noise = True
                            if self._inflight is inflight:
                                self._carry = []      # nothing but noise in it: nothing to carry
                        break   # the caller coughed: say nothing
                    self._keep_unanswered(inflight)   # any other error: the caller was not answered, nothing was stored
            await gate.wait()
            await self._send(self.codec.mark(f"turn-{n}-end", self.stream_sid))
            await self._after_reply()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one bad turn must not end the call
            log.exception("phone turn failed (agent %s)", self.agent.get("id"))
            self.log.append({"event": "turn_failed"})
            self._keep_unanswered(inflight)
        finally:
            close = getattr(gen, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001
                    pass

    async def control(self, tool: str, args: dict) -> Tuple[bool, Any]:
        """end_call and transfer_call (services/call_control.py), run on this call's loop. They only note what to do;
        the reply that goes with them is spoken first and `_after_reply` then acts."""
        if tool == "end_call":
            self.pending_action = self.pending_action or "end"
            call_records.upsert(self.call_sid, outcome="ended_by_agent")
            return True, {"message": "The call will end after your reply. Say a short goodbye now."}
        if tool == "transfer_call":
            number = call_control.transfer_number(self.agent.get("id", ""))
            if not number:
                return False, "no transfer number is set for this agent, so offer to take a message instead"
            self.pending_action = "transfer"
            call_records.upsert(self.call_sid, outcome="transfer_requested", transfer_to=mask(number))
            await self._flag_human()
            if call_control.transfer_mode() == "flow":
                return True, {"message": "The caller is being connected to a person after your reply. Tell them to hold on."}
            return True, {"message": "Nobody can take the call right now. Tell the caller a person will call them back "
                                     "on this number and thank them."}
        return False, f"unknown call control {tool}"

    async def _after_reply(self) -> None:
        """Once the reply that came with end_call or transfer_call has been spoken: end the stream. Ending it hangs up,
        or (flow transfer mode) lets the Exotel call flow carry on to the step that dials the transfer number."""
        action, self.pending_action = self.pending_action, None
        if action is None:
            return
        await asyncio.sleep(1.2)       # the last words are still in the phone's buffer
        self.end_reason = "ended_by_agent" if action == "end" else (
            "transferred" if call_control.transfer_mode() == "flow" else "callback_promised")
        self.log.append({"event": action, "mode": call_control.transfer_mode() if action == "transfer" else None})
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001 - the stream may already be gone
            pass

    async def _sentence_pcm(self, url: str) -> bytes:
        """The audio of one reply sentence as PCM16 at the call's rate."""
        data: Optional[bytes] = None
        match = _CLIP_URL_RE.match(url)
        if match:
            path = os.path.join(self.b.LOCAL_AUDIO_DIR, f"turn_{match.group(1)}_{match.group(2)}.wav")
            data = await asyncio.to_thread(lambda: open(path, "rb").read())
        elif url.startswith("data:audio/wav;base64,"):
            data = base64.b64decode(url.split(",", 1)[1])
        if not data:
            return b""

        def convert() -> bytes:
            x, sr = wav_to_float(data)
            return pa.to_phone_pcm(x, sr, self.rate)

        return await asyncio.to_thread(convert)

    # --------------------------------------------------------------------------------------- speculation
    def _may_speculate(self, ev: "pa.VadEvent") -> bool:
        if self.spec_mode == "off" or self.spec is not None:
            return False
        if self.agent.get("tools"):
            return False   # a tool call (a booking) is an action: it must never run on a sentence that was not finished
        if not self.conversation_id:
            return False   # the turn would create the conversation, and a cancelled one would leave it behind
        if self._ignorable(ev.speech_ms):
            return False   # "hello?" while the agent thinks: ignored at the end of the utterance anyway
        return True

    def _ignorable(self, speech_ms: int) -> bool:
        """A short utterance while the agent is still thinking is "hello?", not a new question. Not when the agent has
        just asked for details: a short piece then ("5 baje") is the next item of the answer."""
        return self.thinking and speech_ms < IMPATIENT_MS and self.end_wait.kind == "normal"

    async def _speculate(self, ev: "pa.VadEvent") -> None:
        await self.stop_turn_keeping_words()
        pieces = self._pieces_for(ev.audio)
        self.start_turn(self._clip_for(pieces), speculative=True, pieces=pieces)
        self.log.append({"event": "speculative_start", "ms": ev.speech_ms})

    # ------------------------------------------------------------------- an unfinished answer, without speculation
    def _precheck_wanted(self) -> bool:
        """The agent asked for details and does not reply early: look at the answer before starting its turn."""
        return (turn_taking.adaptive_enabled() and turn_taking.incomplete_extra_ms() > 0
                and self.end_wait.kind != "normal")

    def _start_precheck(self, pieces: List[bytes], wav: bytes) -> _Precheck:
        run = types.SimpleNamespace(agent=self.agent, user_id=self.claims["u"], storage_mode=self.claims["m"],
                                    asr_previous_agent_text=self.last_agent_text)
        task = asyncio.ensure_future(self.b._asr_for_turn(run, wav))
        task.add_done_callback(lambda t: t.cancelled() or t.exception())   # (never "exception was never retrieved")
        self._pre = _Precheck(list(pieces), wav, task, time.monotonic())
        self.log.append({"event": "precheck_start", "ms": int(sum(pa.pcm_seconds(len(x), self.rate) for x in pieces) * 1000)})
        return self._pre

    def _drop_pre(self) -> None:
        """Forget the precheck: its recognition is cancelled unless a turn has taken it over."""
        pre, self._pre = self._pre, None
        task, self._pre_task = self._pre_task, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
        if pre is not None and not pre.task.done():
            pre.task.cancel()

    def _keep_pre_words(self, pre: _Precheck) -> None:
        if not turn_taking.carry_enabled() or not pre.pieces:
            return
        self._carry = pre.pieces[-CARRY_MAX_PIECES:]
        ms = int(sum(pa.pcm_seconds(len(x), self.rate) for x in self._carry) * 1000)
        log.info("phone turn %d: carried unanswered caller audio (%d ms%s) into the next turn", self.turns + 1, ms,
                 ", already transcribed" if pre.task.done() else "")
        self.log.append({"event": "carried", "turn": self.turns + 1, "ms": ms, "pieces": len(self._carry)})

    def _decide_precheck(self, pre: _Precheck) -> None:
        pre.deciding = True
        self._pre_task = self._spawn(self._pre_decide(pre))

    async def _pre_decide(self, pre: _Precheck) -> None:
        """The end silence was reached for an answer to a request for details. Wait for its recognition (it has been
        running since the pause), then either start the turn on it, or, when the answer looks unfinished, wait once more
        first. The caller going on (a pause or an utterance event) drops this: see on_media."""
        try:
            try:
                text, _hints = await asyncio.wait_for(asyncio.shield(pre.task), 4.0)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - no transcript (error, timeout): the turn recognises it itself
                text = None
            extra = 0
            reason = turn_taking.is_incomplete(text, self.last_agent_text) if text else None
            if reason and not pre.extended:
                pre.extended = True
                extra = turn_taking.incomplete_extra_ms()
                self.log.append({"event": "end_extended", "reason": reason, "ms": extra})
                log.info("phone turn %d: the answer looks unfinished (%s); waiting %d ms more before replying",
                         self.turns + 1, reason, extra)
                await asyncio.sleep(extra / 1000.0)
                while self.detector is not None and self.detector.speaking:
                    await asyncio.sleep(0.05)     # the caller is talking again: their pause or utterance drops this
            if self._pre is not pre:
                return
            self._pre = None
            self._pre_task = None
            self._utter_ms = int(sum(len(x) for x in pre.pieces) / 2 / self.rate * 1000)
            self.start_turn(pre.wav, end=pre.at if not extra else time.monotonic(), pieces=pre.pieces, asr_pre=pre.task)
            if extra:
                self._note_wait(extra, reason or "")
        except asyncio.CancelledError:
            raise

    def _apply_end_wait(self) -> None:
        """After each agent reply: the end-of-turn silence for the caller's next utterance, from what that reply asked
        for (services/turn_taking.end_wait: several details 1500 ms, one name/number/date/time 1100 ms, else the base
        PHONE_END_SILENCE_MS). Applies at once, also to an utterance the caller has already started."""
        self.end_wait = turn_taking.end_wait(self.last_agent_text)
        if self.detector is not None:
            self.detector.set_end_silence(self.end_wait.ms)

    def _unfinished(self, spec: _Speculation) -> Optional[str]:
        """Why the speculative turn's transcript looks like a half-said answer (services/turn_taking.is_incomplete), or
        None. Needs the transcript: a turn that has not been through speech recognition yet is confirmed as before."""
        if not turn_taking.adaptive_enabled() or spec.extended or spec.heard is None:
            return None
        if turn_taking.incomplete_extra_ms() <= 0:
            return None
        return turn_taking.is_incomplete(spec.heard, self.last_agent_text)

    def _defer(self, spec: _Speculation, reason: str) -> None:
        """The caller's end of speech is not confirmed yet: wait once more, then confirm, unless they go on."""
        extra = turn_taking.incomplete_extra_ms()
        spec.extended = spec.deferred = True
        self.log.append({"event": "end_extended", "reason": reason, "ms": extra})
        log.info("phone turn %d: the answer looks unfinished (%s); waiting %d ms more before replying", spec.n, reason, extra)
        self._defer_task = self._spawn(self._confirm_later(spec, extra, reason))

    async def _confirm_later(self, spec: _Speculation, extra_ms: int, reason: str) -> None:
        try:
            await asyncio.sleep(extra_ms / 1000.0)
            while (self.spec is spec and spec.deferred and self.detector is not None
                   and self.detector.speaking and not spec.task.done()):
                await asyncio.sleep(0.05)   # the caller is talking again: their utterance settles it (a new turn)
            if self.spec is not spec or not spec.deferred:
                return
            spec.deferred = False
            if spec.task.done():             # it ended on its own (an error): answer again from its audio
                self.spec = None
                self.turns -= 1
                self.start_turn(self._clip_for(spec.pieces), end=time.monotonic(), pieces=spec.pieces)
                self._note_wait(extra_ms, reason)
                return
            self._note_wait(extra_ms, reason)
            self._confirm(spec, time.monotonic())
        except asyncio.CancelledError:
            raise

    def _confirm(self, spec: _Speculation, end: float) -> None:
        self.spec = None
        self.tm["end"] = end
        self.spec_stats["confirmed"] += 1
        self.log.append({"event": "speculative_confirmed", "head_start_ms": int((end - spec.started) * 1000)})
        spec.gate.set()

    async def _cancel_speculation(self) -> None:
        """The caller went on talking: drop the speculative turn. It sent nothing; it is cancelled before it stores its
        turn, and its sentence clips (temporary files) are removed."""
        spec = self.spec
        if spec is None:
            return
        self.spec = None
        spec.deferred = False
        task, self._defer_task = self._defer_task, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
        if self.turn_task is spec.task:
            await self.stop_turn()
        self.turns -= 1
        self.spec_stats["cancelled"] += 1
        self.log.append({"event": "speculative_cancelled"})
        getattr(self.b, "_STREAM_TURN_OWNERS", {}).pop(spec.turn_id, None)
        directory = getattr(self.b, "LOCAL_AUDIO_DIR", None)

        def remove_clips() -> None:
            if not directory or not os.path.isdir(directory):
                return
            prefix = f"turn_{spec.turn_id}_"
            for name in os.listdir(directory):
                if name.startswith(prefix) and name.endswith(".wav"):
                    try:
                        os.unlink(os.path.join(directory, name))
                    except OSError:
                        pass
        await asyncio.to_thread(remove_clips)

    def _keep_spec_words(self, spec: _Speculation) -> None:
        """The deferred speculative turn is dropped for the caller's next part: its pieces are carried."""
        if not turn_taking.carry_enabled() or not spec.pieces:
            return
        self._carry = spec.pieces[-CARRY_MAX_PIECES:]
        ms = int(sum(pa.pcm_seconds(len(x), self.rate) for x in self._carry) * 1000)
        log.info("phone turn %d: carried unanswered caller audio (%d ms%s) into the next turn", spec.n, ms,
                 ", already transcribed" if spec.heard is not None else "")
        self.log.append({"event": "carried", "turn": spec.n, "ms": ms, "pieces": len(self._carry)})

    # ---------------------------------------------------------------------------------------------- events
    def _phone_engines(self) -> List[str]:
        """The speech engines a turn of this call may need: Kokoro (preset replies, fillers, the fallback line) and,
        with PHONE_CLONE_VOICE=1 and a cloned voice on the agent, the clone engine (see backend._turn_voice)."""
        keys = ["kokoro"]
        if _env("PHONE_CLONE_VOICE", "0") == "1" and self.agent.get("voice_id"):
            keys.append(str(getattr(self.b, "CLONE_ENGINE", "qwen3_0_6b")))
        return keys

    def _pin_keys(self) -> List[str]:
        """What a call keeps loaded: its speech engines (_phone_engines, managed by advanced_manager), and the models
        that live outside it (services/retrieval.py and backend.py keep them for the process): the embedding model, the
        reranker when RERANKER=on, and speech recognition when it is local (no Groq key). Their pins are a record and a
        guard for any future unload path: nothing unloads them today."""
        keys = self._phone_engines() + ["embedding"]
        if self.b.retrieval.reranker_enabled():
            keys.append("reranker")
        if getattr(self.b, "groq_client", None) is None:
            keys.append("asr")
        return keys

    def _pin_models(self) -> None:
        mgr = getattr(self.b, "advanced_manager", None)
        if mgr is None or not hasattr(mgr, "pin"):
            return
        self._pinned = self._pin_keys()
        for key in self._pinned:
            mgr.pin(key)
        log.info("call warm: pinned [%s]", ", ".join(self._pinned))

    def _unpin_models(self) -> None:
        mgr = getattr(self.b, "advanced_manager", None)
        keys = getattr(self, "_pinned", []) or []
        for key in keys:
            if mgr is not None and hasattr(mgr, "unpin"):
                mgr.unpin(key)
        self._pinned = []
        if keys:
            log.info("call warm: unpinned")

    async def on_start(self, ev: "tbase.Event") -> None:
        self.stream_sid = ev.stream_id
        self.call_sid = ev.call_id
        self.rate = ev.rate if ev.rate in pa.SUPPORTED_RATES else pa.PHONE_RATE
        # The agent speaks first, at once: the pre-rendered AI disclosure and greeting (services/phone_greeting.py).
        self.turns = 1
        self.turn_audio = False
        self.turn_speculative = False
        self.tm = {}
        self._conv_ready.clear()
        self._greeting_task = self.turn_task = asyncio.create_task(self._greet())
        self._spawn(self._open_conversation())
        self.log.append({"event": "start", "rate": self.rate})
        self._pin_models()   # the call's engines stay loaded until it ends (no 5-13 s cold turn mid-call)
        phone_greeting.schedule_call_warm(self.b)   # embedding model + reranker warm, in the background (PHONE_PREWARM)
        direction = "outbound" if (self.call_sid in _known_calls or (call_records.get(self.call_sid or "") or {})
                                   .get("direction") == "outbound") else "inbound"
        self.outbound = direction == "outbound"
        call_records.upsert(self.call_sid, owner=self.claims["u"], agent_id=self.agent["id"], direction=direction,
                            status="in-progress", started_at=_now_iso(), provider=self.provider_name)
        log.info("phone call started: agent %s, call %s, from %s, %s Hz, via %s", self.agent["id"], self.call_sid,
                 mask(ev.caller), self.rate, self.provider_name)
        self.end_silence_ms = int(_env("PHONE_END_SILENCE_MS", "500") or 500)
        self.end_wait = turn_taking.end_wait(self.last_agent_text)   # (the greeting may have been sent, and asked, already)
        early = int(_env("PHONE_EARLY_SILENCE_MS", "250") or 250) if self.spec_mode != "off" else 0
        cfg = pa.DetectorConfig(end_silence_ms=self.end_silence_ms,
                                early_silence_ms=early if 0 < early < self.end_silence_ms else 0)
        # built off the event loop: the first Silero session loads a model, and the greeting is already going out
        self.detector = await asyncio.to_thread(pa.UtteranceDetector, self.rate, None, cfg)
        self.detector.set_end_silence(self.end_wait.ms)   # the greeting may already have asked for something

    async def _hang_up_on_voicemail(self, verdict: dict) -> None:
        """An answering machine picked up: do not talk to it. Record that, stop speaking and end the stream (campaigns
        treat this as a call to retry, not as a conversation)."""
        log.info("answering machine on call %s (%s): hanging up", self.call_sid, verdict.get("reason"))
        self.log.append({"event": "voicemail", "reason": verdict.get("reason")})
        self.end_reason = "voicemail"
        call_records.upsert(self.call_sid, outcome="voicemail")
        try:
            await self.clear()
            await self.ws.close()
        except Exception:  # noqa: BLE001 - the stream may already be gone
            pass

    async def on_dtmf(self, ev: "tbase.Event") -> None:
        """A key press on the callee's phone. The digit itself is never stored or logged (it may be a card number or a
        code); only how many keys were pressed is. Pressing 0 asks for a person: the conversation is flagged
        `needs_human`, which shows on the Calls page. (Exotel's exact dtmf event shape is not verified against a live
        account: the codec reads the digit from `dtmf.digit`, `dtmf` or `digit`.)"""
        digit = (ev.digit or "").strip()
        if not digit or len(digit) > 2:
            return
        self.dtmf_presses += 1
        self.log.append({"event": "dtmf"})
        if digit == "0" and not self.human_requested:
            self.human_requested = True
            self.log.append({"event": "human_requested"})
            call_records.upsert(self.call_sid, outcome="human_requested")
            await self._flag_human()
        call_records.upsert(self.call_sid, dtmf_presses=self.dtmf_presses)

    async def _flag_human(self) -> None:
        if not self.conversation_id:
            try:
                await asyncio.wait_for(self._conv_ready.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass
        if not self.conversation_id:
            return
        try:
            from routers import whatsapp as _wa  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            from backend.routers import whatsapp as _wa  # noqa: PLC0415
        await _wa._touch_conversation(self.claims["m"], self.agent, self.conversation_id,
                                      {"needs_human": True, "handoff_at": _now_iso()})

    async def on_media(self, ev: "tbase.Event") -> None:
        if self.detector is None:
            return
        pcm = ev.pcm
        self._tap_audio("caller", pcm)
        if self.human_active:
            return          # a person has the call: the agent neither listens nor answers
        for vad_event in self.detector.feed(pcm):
            if vad_event.kind == "barge":
                if self.speaking:   # talking over audio the caller can hear: stop it. While the agent is only thinking
                    self.barges += 1  # there is nothing to stop, and cancelling would throw the work away
                    await self.stop_turn_keeping_words()   # (a reply that was not stored yet: its words are carried)
                    await self.clear()
            elif vad_event.kind == "pause":
                if self.spec is not None and self.spec.deferred:   # the caller went on after the extra wait
                    self._keep_spec_words(self.spec)
                    await self._cancel_speculation()
                if self._pre is not None and self._pre.deciding:   # the caller went on after the extra wait
                    self._keep_pre_words(self._pre)
                    self._drop_pre()
                if self._may_speculate(vad_event):
                    await self._speculate(vad_event)
                elif self._pre is None and self.spec is None and self._precheck_wanted():
                    await self.stop_turn_keeping_words()          # as a speculative start does
                    pieces = self._pieces_for(vad_event.audio)
                    self._start_precheck(pieces, self._clip_for(pieces))   # recognise it now, say nothing yet
            elif vad_event.kind == "resume":
                await self._cancel_speculation()
                self._drop_pre()
            elif vad_event.kind == "utterance":
                now = time.monotonic()
                spec = self.spec
                if self._pre is not None and self._pre.deciding:
                    self._keep_pre_words(self._pre)       # the caller went on while their answer was being waited for
                    self._drop_pre()
                if spec is not None and spec.deferred:
                    # the caller went on while the end of their answer was being waited for: that answer was not
                    # finished. The turn on the first part is dropped and its audio goes with the new part.
                    self._keep_spec_words(spec)
                    await self._cancel_speculation()
                    spec = None
                if spec is not None and not spec.task.done():
                    reason = self._unfinished(spec)
                    if reason:
                        self._defer(spec, reason)   # an unfinished answer ("... and", 7 digits of a number): wait once more
                        continue
                    self._note_wait()
                    self._confirm(spec, now)   # the speculative turn was right: release its audio
                    continue
                if spec is not None:            # it ended on its own (an error): start again with the full audio
                    self.spec = None
                    self.turns -= 1
                if self._ignorable(vad_event.speech_ms):
                    self.log.append({"event": "ignored", "ms": vad_event.speech_ms})   # "hello?" while it thinks
                    self._drop_pre()
                    continue
                await self.stop_turn_keeping_words()
                self._utter_ms = vad_event.speech_ms
                pieces = self._pieces_for(vad_event.audio)
                wav = self._clip_for(pieces)
                if self._precheck_wanted():
                    # after a request for details, an agent that does not reply early (tools, speculation off) looks at
                    # the answer first: unfinished, it is waited for once more; finished, its recognition is the turn's
                    pre = self._pre
                    if pre is None or pre.wav != wav:
                        self._drop_pre()
                        pre = self._start_precheck(pieces, wav)
                    pre.at = now
                    self._note_wait()
                    self._decide_precheck(pre)
                    continue
                self.start_turn(wav, end=now, pieces=pieces)

    async def _api_hangup(self) -> None:
        """A provider whose call outlives its stream (Telnyx) is hung up through its API when the agent ended the call:
        the agent said goodbye, an answering machine, the time limit, a promised callback. Best effort, never raises."""
        if self.end_reason not in ("ended_by_agent", "voicemail", "time_limit", "callback_promised") or not self.call_sid:
            return
        try:
            provider = registry.provider(self.provider_name)
            if not provider.api_hangup:
                return
            known = _known_calls.get(self.call_sid) or call_records.get(self.call_sid) or {}
            account = registry.account_by_label(self.provider_name, known.get("account"))
            if account is not None and await provider.hangup(account, self.call_sid):
                self.log.append({"event": "api_hangup"})
        except Exception as e:  # noqa: BLE001
            log.warning("could not hang up call %s through %s: %s", self.call_sid, self.provider_name, e.__class__.__name__)

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        max_s = float(_env("PHONE_MAX_CALL_S", "600") or 600)
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(self.ws.receive_text(), timeout=max(1.0, max_s - (time.time() - self.started_at)))
                except asyncio.TimeoutError:
                    log.info("phone call hit the %ss limit", max_s)
                    self.end_reason = "time_limit"
                    break
                ev = self.codec.parse(raw)
                if ev is None:
                    continue
                kind = ev.kind
                if kind == "start":
                    await self.on_start(ev)
                elif kind == "media":
                    await self.on_media(ev)
                elif kind == "dtmf":
                    await self.on_dtmf(ev)
                elif kind == "mark":
                    self.pacer.finished()
                elif kind == "stop":
                    self.end_reason = self.end_reason or "caller_hung_up"
                    break
        except WebSocketDisconnect:
            self.end_reason = self.end_reason or "caller_hung_up"
        finally:
            if self.spec is not None:
                await self._cancel_speculation()
            self._drop_pre()
            await self.stop_turn()
            await self._api_hangup()
            self._unpin_models()
            g = self.greeting
            self.tap({"type": "ended"})
            call_control.unregister(self.conversation_id)
            call_records.upsert(self.call_sid, hangup_reason=self.end_reason or "ended",
                                conversation_id=getattr(self, "conversation_id", None),
                                duration_s=int(time.time() - self.started_at), ended_at=_now_iso())
            caller_sheet.on_phone_call_end(self.agent, user_id=self.claims["u"], local=self.claims["m"] == "local",
                                           conversation_id=getattr(self, "conversation_id", None),
                                           duration_s=int(time.time() - self.started_at))   # queued, never blocks
            log.info("phone call ended: call %s, %d turns, %d barge-ins, %.0fs, greeting clip %s, speculative turns "
                     "%d started / %d confirmed / %d cancelled", self.call_sid, self.turns, self.barges,
                     time.time() - self.started_at, g.clip_id if g is not None else None,
                     self.spec_stats["started"], self.spec_stats["confirmed"], self.spec_stats["cancelled"])


def install(app) -> None:
    """Called by backend.py when it includes this router. Adds the public routes (stream, status callback) to the main
    app too, so a call still works on the loopback port without the phone-only port. Then, a few seconds after startup,
    render (or check) the greetings of the agents that take phone calls (services/phone_greeting.prewarm_all), in the
    background.

    Registered with `app.on_event("startup")`, like backend.py's own startup work: FastAPI keeps that API (deprecated in
    favour of lifespan, which backend.py does not use), while Starlette 1.x removed `add_event_handler` from the app."""
    app.include_router(public_router)

    async def _phone_greetings_at_startup() -> None:
        phone_greeting._spawn(phone_greeting.prewarm_all_later(_b()))

    app.on_event("startup")(_phone_greetings_at_startup)


@public_router.websocket(STREAM_PATH + "/{token}")
async def exotel_stream(ws: WebSocket, token: str):
    await _serve_stream(ws, token, "exotel")


def provider_active(name: str) -> bool:
    """Exotel's routes always answer. Another provider's routes answer only while it is in use (named in PHONE_PROVIDERS or in a
    PHONE_ROUTES entry): otherwise they are inert (the stream is closed, the answer is a 404), so the surface the tunnel
    exposes follows the configuration."""
    return name == "exotel" or name in registry.enabled() or name in registry.routed_providers()


def active_public_paths() -> List[str]:
    """The public route patterns that currently do something (the health probe lists them)."""
    paths = [STREAM_PATH + "/{token}", STATUS_CALLBACK_PATH + "/{token}"]
    for name in PROVIDER_NAMES:
        if provider_active(name):
            paths += [provider_path(name, w) + "/{token}" for w in ("stream", "status-callback", "answer")]
    return sorted(paths)


def _make_stream_route(name: str):
    async def provider_stream(ws: WebSocket, token: str):
        await _serve_stream(ws, token, name)
    provider_stream.__name__ = f"{name}_stream"
    return provider_stream


async def _serve_stream(ws: WebSocket, token: str, provider_name: str) -> None:
    """The call's WebSocket, for any provider. The token (signed, expiring) names the agent and the provider; a token for
    another provider, or an answer-document token, opens nothing."""
    claims = read_token(token)
    if claims is None or (claims.get("p") or "exotel") != provider_name or claims.get("k"):
        await ws.close(code=4401)
        return
    if not provider_active(provider_name):
        await ws.close(code=4404)
        return
    agent = await _widget()._load_agent_row(claims["m"], claims["a"])
    if agent is None or agent.get("user_id") != claims["u"]:
        await ws.close(code=4404)
        return
    global _active_calls
    if _active_calls >= max_calls():
        # Busy: refuse before accepting, so Exotel's flow continues to its next applet (a Connect to the owner's mobile)
        # instead of the caller hearing dead air or two calls slowing each other down.
        log.warning("phone call refused: %d live call(s) already (PHONE_MAX_CALLS=%d); the %s flow falls back",
                    _active_calls, max_calls(), registry.provider(provider_name).display)
        await ws.close(code=BUSY_CLOSE_CODE)
        return
    await ws.accept()
    accepted = time.monotonic()
    phone_greeting.schedule_warm(_b(), agent, claims["u"], claims["m"])   # no-op when the outbound call already warmed
    _active_calls += 1
    try:
        if provider_name == "exotel":
            session = CallSession(ws, claims, agent, accepted_at=accepted)
        else:
            session = CallSession(ws, claims, agent, accepted_at=accepted, codec=registry.provider(provider_name).codec())
        await session.run()
    finally:
        _active_calls = max(0, _active_calls - 1)


def _make_answer_route(name: str):
    async def provider_answer(token: str, request: Request):
        return await _answer(token, request, name)
    provider_answer.__name__ = f"{name}_answer"
    return provider_answer


async def _answer(token: str, request: Request, name: str):
    """What the carrier fetches when a call is answered (outbound: the Answer URL we gave it; inbound: the URL of the number's
    voice webhook, see POST /api/phone/{provider}/inbound-link). It returns the carrier's own document (TwiML for Twilio,
    Plivo XML for Plivo and Vobiz) pointing the call at a fresh signed stream URL for the agent. Telnyx's Call Control sends
    JSON webhooks instead: a new inbound call is answered through the API with the stream attached. Authenticated only by
    the signed token in the URL, which opens no stream itself (kind "ans")."""
    from fastapi.responses import JSONResponse, Response  # noqa: PLC0415
    claims = read_token(token)
    if claims is None or (claims.get("p") or "exotel") != name or claims.get("k") != "ans" or not provider_active(name):
        return Response(status_code=404)
    provider = registry.provider(name)
    fresh = make_token(claims["a"], claims["u"], claims["m"], OUTBOUND_TOKEN_TTL_S, provider=name)
    try:
        wss = stream_url(fresh, name)
    except AppError:
        return Response(status_code=503)
    if name != "telnyx":
        return Response(content=provider.answer_document(wss), media_type=provider.answer_content_type)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    data = body.get("data") if isinstance(body, dict) else None
    payload = (data or {}).get("payload") or {}
    if (data or {}).get("event_type") == "call.initiated" and payload.get("direction") == "incoming":
        accounts = provider.accounts()
        account = next((a for a in accounts if a["connection_id"] == payload.get("connection_id")), None) or (
            accounts[0] if accounts else None)
        cid = payload.get("call_control_id")
        if account is not None and cid:
            call_records.upsert(cid, owner=claims["u"], agent_id=claims["a"], direction="inbound", status="queued",
                                provider=name, account=account["label"])
            ok = await provider.answer_call(account, cid, wss, status_callback_url(token, name))
            if not ok:
                log.warning("Telnyx: could not answer the inbound call %s", cid)
    else:
        _record_status(claims, provider.parse_status_callback(body if isinstance(body, dict) else {}))
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------------------------- owner routes

class CallRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64)
    to: str = Field(..., description="The phone to call, in international format, for example +919876543210")
    caller_agreed: bool = Field(False, description="The person being called agreed to receive this call")


class LinkRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64)


async def _owned_agent(request: Request, agent_id: str, user_id: str) -> dict:
    return await _b()._load_agent(request, agent_id, user_id)


@router.get("/api/phone/exotel/status")
async def phone_status(user_id: str = Depends(get_current_user_id)):
    miss = missing_settings()
    allowed = allowed_numbers()
    s = exotel_settings()
    caller_id = normalise_number(s["caller_id"]) if s["caller_id"] else None
    return {"configured": not miss, "missing": miss, "subdomain": s["subdomain"],
            "public_base_url": s["public"] or None, "vad": pa.make_vad(pa.PHONE_RATE).name,
            "resampler": pa.resampler_name(), "allowed_numbers_set": bool(allowed),
            # owner-only route, so the full numbers are fine here: the Call dialog offers them in a select
            "allowed_numbers": [{**e, "masked": mask_e164(e["number"])} for e in allowed_entries()],
            "caller_id_masked": mask_e164(caller_id) if caller_id else (mask(s["caller_id"]) or None),
            "inbound_ready": bool(s["public"]), "accounts": accounts_public(),
            # every provider (Exotel, Twilio, Plivo, Vobiz, Telnyx): configured, in use (PHONE_PROVIDERS) and its accounts
            "providers": registry.public_status(), "provider_order": registry.enabled(),
            "max_call_seconds": float(_env("PHONE_MAX_CALL_S", "600") or 600),
            "max_calls": max_calls(), "active_calls": _active_calls, "busy_fallback": BUSY_FALLBACK}


class NumberBody(BaseModel):
    label: str = Field(..., max_length=200)
    number: str = Field(..., max_length=40)
    agreed: bool = Field(False, description="The person agreed to be called on this number")


@router.get("/api/phone/exotel/numbers")
async def list_numbers(user_id: str = Depends(get_current_user_id)):
    return {"numbers": [{**e, "masked": mask_e164(e["number"])} for e in allowed_entries()], "max_saved": SAVED_NUMBERS_MAX}


@router.post("/api/phone/exotel/numbers")
async def add_number(body: NumberBody, user_id: str = Depends(get_current_user_id)):
    """Save a number so it can be called. The person must have agreed; the number is stored in backend/data/phone_numbers.json
    (private to this computer, not committed)."""
    if not body.agreed:
        raise AppError("CONSENT_REQUIRED", "Only save a number whose owner agreed to be called.", 400)
    label = re.sub(r"[\x00-\x1f\x7f]", " ", body.label)
    label = re.sub(r"\s+", " ", label).strip()
    if not 1 <= len(label) <= LABEL_MAX:
        raise AppError("BAD_LABEL", f"Give the person a name of 1 to {LABEL_MAX} characters.", 422)
    number = normalise_number(body.number)
    if number is None:
        raise AppError("BAD_NUMBER", "That does not look like a phone number. Try +91 98765 43210.", 422)
    with _saved_lock:
        rows = saved_numbers()
        if number in allowed_numbers():
            raise AppError("NUMBER_EXISTS", "That number is already in the list.", 409)
        if len(rows) >= SAVED_NUMBERS_MAX:
            raise AppError("TOO_MANY_NUMBERS", f"You can save up to {SAVED_NUMBERS_MAX} numbers. Remove one first.", 409)
        entry = {"id": uuid.uuid4().hex[:12], "label": label, "number": number,
                 "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        await asyncio.to_thread(_write_saved, [*rows, entry])
    log.info("phone number saved: %s", mask(number))
    return {"id": entry["id"], "label": label, "number": number, "masked": mask_e164(number), "source": "saved"}


@router.delete("/api/phone/exotel/numbers/{number_id}")
async def remove_number(number_id: str, user_id: str = Depends(get_current_user_id)):
    if number_id in ("env", "null", "None"):
        raise AppError("NUMBER_FROM_ENV", "That number comes from backend/.env. Remove it from PHONE_ALLOWED_NUMBERS there.", 400)
    with _saved_lock:
        rows = saved_numbers()
        keep = [r for r in rows if r["id"] != number_id]
        if len(keep) == len(rows):
            raise AppError("NUMBER_NOT_FOUND", "That number is not saved.", 404)
        await asyncio.to_thread(_write_saved, keep)
    return {"removed": True}


@router.post("/api/phone/exotel/inbound-link")
async def inbound_link(body: LinkRequest, request: Request, user_id: str = Depends(get_current_user_id)):
    """The wss:// address to paste into the Voicebot applet of an ExoPhone so that calls to it reach this agent."""
    b = _b()
    agent = await _owned_agent(request, body.agent_id, user_id)
    mode = b._storage_mode(request)
    token = make_token(agent["id"], user_id, mode, INBOUND_TOKEN_TTL_S)
    url = stream_url(token)
    phone_greeting.schedule_prepare(b, agent, user_id, mode, warm=False)   # the greeting is ready for the first call
    return {"stream_url": url, "valid_days": INBOUND_TOKEN_TTL_S // 86400,
            "flow": {"applet": "Voicebot", "then": "Connect (your mobile or the transfer number)",
                     "fallback": BUSY_FALLBACK, "max_calls": max_calls()}}


exotel_message = _exotel.exotel_message
exotel_reason = _exotel.exotel_reason


@router.post("/api/phone/exotel/call")
async def place_call(body: CallRequest, request: Request, user_id: str = Depends(get_current_user_id)):
    """Ring `to` and let the agent talk. Only for people who agreed: the request must say so (a number in
    PHONE_ALLOWED_NUMBERS is pre-approved), at most CALLS_PER_HOUR calls an hour, and with PHONE_ALLOWED_NUMBERS set only
    the allowed and saved numbers can be called."""
    b = _b()
    to = normalise_number(body.to)
    # A number in PHONE_ALLOWED_NUMBERS (backend/.env) is pre-approved: the owner confirmed that person's consent once,
    # when adding it there. Every other number needs caller_agreed on each call.
    preapproved = to is not None and any(e["number"] == to and e["source"] == "env" for e in allowed_entries())
    if not body.caller_agreed and not preapproved:
        raise AppError("CONSENT_REQUIRED", "Only call someone who agreed to receive this call: set caller_agreed.", 400)
    if to is None:
        raise AppError("BAD_NUMBER", "Give the number in international format, for example +919876543210.", 422)
    allowed = allowed_numbers()
    if allowed and to not in allowed:
        raise AppError("NUMBER_NOT_ALLOWED", "That number is not saved. Add it first.", 403)
    miss = missing_settings()
    if miss:
        raise AppError("PHONE_NOT_CONFIGURED", _not_set_up(miss), 503)
    now = time.time()
    recent = _calls_by_owner[user_id]
    while recent and recent[0] < now - 3600:
        recent.popleft()
    if len(recent) >= CALLS_PER_HOUR:
        raise AppError("RATE_LIMITED", "Too many calls this hour.", 429)
    agent = await _owned_agent(request, body.agent_id, user_id)
    mode = b._storage_mode(request)
    result = await dial(b, agent, user_id, mode, to)
    recent.append(now)
    return result


async def dial(b, agent: dict, user_id: str, mode: str, to: str) -> Dict[str, Any]:
    """Place one outbound call and record it, through the first provider and account that takes it (services/telephony/
    registry.attempts: PHONE_ROUTES for this number, then PHONE_PROVIDERS in order, each with its accounts active-first).
    The caller has already checked consent, the allow-list and the rate limits (the owner route above; a campaign does its own
    checks). Raises AppError when no provider can be reached or every one refuses; returns {"status", "to", "call_sid",
    "agent_id", "provider", "account"}."""
    # While the carrier dials and the phone rings: render the greeting and warm up the first reply, in the background.
    phone_greeting.schedule_prepare(b, agent, user_id, mode, warm=True)
    plan = registry.attempts(to)
    if not plan:
        first = registry.provider(registry.enabled()[0])
        raise AppError("PHONE_NOT_CONFIGURED", _not_set_up(missing_settings() or [first.VARS[0][1]]), 503)
    providers_used = {att.provider.name for att in plan}
    offline.block(f"phone calls through {plan[0].provider.display}" if len(providers_used) == 1 else "phone calls")
    # One try per (provider, account) per call. A refusal about the account (credit, credentials, number not allowed on it,
    # KYC) moves on to the next; any other refusal, and any network error (the call may already have been placed), stops here.
    outcome: Dict[Tuple[str, str], str] = {}
    sid: Optional[str] = None
    chosen = plan[0]
    for i, att in enumerate(plan):
        prov, account = att.provider, att.account
        book = registry.book(prov.name)
        tag = None if prov.name == "exotel" else prov.name
        token = make_token(agent["id"], user_id, mode, OUTBOUND_TOKEN_TTL_S, provider=tag)
        answer = (answer_url(make_token(agent["id"], user_id, mode, OUTBOUND_TOKEN_TTL_S, provider=tag, kind="ans"), prov.name)
                  if prov.needs_answer_url else None)
        try:
            sid = await prov.place_call(account, to, stream_url(token, prov.name), status_callback_url(token, prov.name), answer)
        except tbase.ProviderUnreachable as e:
            raise AppError(f"{prov.name.upper()}_UNREACHABLE", f"Could not reach {prov.display}: {e.reason}", 502) from e
        except tbase.ProviderHTTPError as e:
            text = e.body
            log.warning("%s refused the call (%s) on account %s: %s", prov.display, e.status, account["label"], text[:300])
            kind = prov.classify_refusal(e.status, text)
            if kind is None or len(plan) == 1:   # one account configured: the plain refusal, as before
                if kind in (tbase.EXHAUSTED, tbase.AUTH_FAILED):
                    book.set(account, kind)
                raise AppError(f"{prov.name.upper()}_REFUSED",
                               f"{prov.display} refused the call (HTTP {e.status}): {prov.reason(text)}", 502) from e
            if kind in (tbase.EXHAUSTED, tbase.AUTH_FAILED):
                book.set(account, kind)
            outcome[(prov.name, account["label"])] = tbase.REFUSAL_WORDS[kind]
            if i + 1 < len(plan):
                nxt = plan[i + 1]
                log.warning("%s account %s %s; trying %s", prov.display, account["label"],
                            outcome[(prov.name, account["label"])],
                            f"account {nxt.account['label']}" if nxt.provider is prov
                            else f"{nxt.provider.display} account {nxt.account['label']}")
                continue
            raise _all_refused(plan, outcome) from e
        book.set(account, "active")
        chosen = att
        break
    prov, account = chosen.provider, chosen.account
    if sid:
        _known_calls[sid] = {"owner": user_id, "agent_id": agent["id"], "account": account["label"], "provider": prov.name}
        while len(_known_calls) > KNOWN_CALLS_MAX:
            _known_calls.popitem(last=False)
        extra = {"exotel_account": account["label"]} if prov.name == "exotel" else {}
        call_records.upsert(sid, owner=user_id, agent_id=agent["id"], direction="outbound", status="queued",
                            to_masked=mask(to), provider=prov.name, account=account["label"], **extra)
    log.info("outbound call placed to %s for agent %s (call %s) via %s account %s", mask(to), agent["id"], sid,
             prov.display, account["label"])
    return {"status": "calling", "to": mask(to), "call_sid": sid, "agent_id": agent["id"], "provider": prov.name,
            "account": account["label"]}


def _all_refused(plan, outcome: Dict[Tuple[str, str], str]) -> AppError:
    """The error when every (provider, account) in `plan` refused for a reason about the account."""
    names = {att.provider.name for att in plan}
    if len(names) == 1:
        prov = plan[0].provider
        accounts = prov.accounts()
        book = registry.book(prov.name)
        states = [f"{a['label']} {outcome.get((prov.name, a['label'])) or book.state(a)['state'].replace('_', ' ')}"
                  for a in accounts]
        if outcome and all(v == tbase.REFUSAL_WORDS[tbase.KYC] for v in outcome.values()):
            return AppError(f"{prov.name.upper()}_KYC_REQUIRED",
                            f"{prov.display} refused the call on all {len(accounts)} accounts: none has finished KYC, which "
                            f"{prov.display} requires before any outbound call. Finish KYC in the {prov.display} dashboard "
                            "(Settings, KYC).", 403)
        return AppError(f"{prov.name.upper()}_ALL_ACCOUNTS_EXHAUSTED",
                        f"All {len(accounts)} {prov.display} accounts are out of credit or refused ({', '.join(states)}). "
                        "Recharge one or add another account in backend/.env.", 503)
    parts = [f"{att.provider.name} {att.account['label']} {outcome.get((att.provider.name, att.account['label'])) or 'refused'}"
             for att in plan]
    return AppError("PHONE_ALL_ACCOUNTS_FAILED",
                    f"Every phone account refused the call ({', '.join(parts)}). Recharge or fix one, or add another "
                    "account in backend/.env.", 503)


CALL_STATUSES = tbase.CALL_STATUSES
_call_status = tbase.normalise_status
_parse_call = _exotel.parse_call
_duration = tbase.duration_of
_field = tbase.field


@router.get("/api/phone/exotel/call/{call_sid}")
async def call_status(call_sid: str, user_id: str = Depends(get_current_user_id)):
    """Where a call this owner placed stands: queued, ringing, in-progress, then completed, failed, busy, no-answer or
    canceled, through whichever provider placed it. Asks the carrier on every poll (the dialog polls about every 2 s), at
    most POLLS_PER_MINUTE polls a minute."""
    known = _known_calls.get(call_sid)
    if known is None or known["owner"] != user_id:
        raise AppError("CALL_NOT_FOUND", "No call with that id was placed from this account.", 404)
    miss = missing_settings()
    if miss:
        raise AppError("PHONE_NOT_CONFIGURED", _not_set_up(miss), 503)
    now = time.time()
    polls = _polls_by_owner[user_id]
    while polls and polls[0] < now - 60:
        polls.popleft()
    if len(polls) >= POLLS_PER_MINUTE:
        raise AppError("RATE_LIMITED", "Checking the call too often. Wait a moment.", 429)
    polls.append(now)
    name = known.get("provider") or "exotel"
    prov = registry.provider(name)
    s = registry.account_by_label(name, known.get("account"))   # the account that placed the call
    if s is None:
        raise AppError("PHONE_NOT_CONFIGURED", f"{prov.display} is not set up any more: its account is gone from backend/.env.", 503)
    offline.block(f"phone calls through {prov.display}")
    try:
        f = await prov.call_status(s, call_sid)
    except tbase.ProviderUnreachable as e:
        raise AppError(f"{name.upper()}_UNREACHABLE", f"Could not reach {prov.display}: {e.reason}", 502) from e
    except tbase.ProviderHTTPError as e:
        log.warning("%s call lookup failed (%s) for call %s", prov.display, e.status, call_sid)
        raise AppError(f"{name.upper()}_REFUSED", f"{prov.display} could not give the call status (HTTP {e.status}).", 502) from e
    status = f["status"]
    bad = status in ("failed", "busy", "no-answer", "canceled")
    call_records.upsert(call_sid, owner=user_id, status=status if status != "unknown" else None,
                        duration_s=f.get("duration_s"), started_at=f.get("started_at") or None,
                        ended_at=f.get("ended_at") or None, message=f.get("message") if bad else None)
    return {"call_sid": call_sid, "status": status, "duration_s": f.get("duration_s"),
            "started_at": f.get("started_at") or None, "ended_at": f.get("ended_at") or None,
            "message": f.get("message") if bad else None}


def _record_status(claims: Dict[str, Any], p: Optional[Dict[str, Any]]) -> bool:
    """File a parsed status callback under the token's owner and agent. False when it names no call, or the call is
    another owner's."""
    if not p or not p.get("sid"):
        return False
    sid, status = str(p["sid"]), p["status"]
    known = call_records.get(sid) or {}
    if known.get("owner") not in (None, claims["u"]):
        return False
    call_records.upsert(sid, owner=claims["u"], agent_id=claims["a"], direction=known.get("direction") or "outbound",
                        status=status if status != "unknown" else None, duration_s=p.get("duration_s"),
                        ended_at=p.get("ended_at") if status in call_records.TERMINAL else None, message=p.get("message"))
    return True


async def _status_callback(token: str, request: Request, name: str):
    claims = read_token(token)
    if claims is None or (claims.get("p") or "exotel") != name or claims.get("k") or not provider_active(name):
        return {"ok": False}
    try:
        form = dict(await request.form())
    except Exception:  # noqa: BLE001 - JSON or empty body
        form = {}
    if not form:
        try:
            body = await request.json()
            form = body if isinstance(body, dict) else {}
        except Exception:  # noqa: BLE001
            form = {}
    return {"ok": _record_status(claims, registry.provider(name).parse_status_callback(form))}


@public_router.post(STATUS_CALLBACK_PATH + "/{token}")
async def status_callback(token: str, request: Request):
    """Exotel tells us a call changed state. Authenticated only by the signed token in the URL (Exotel does not sign its
    callbacks), so the token's owner and agent are what the record is filed under. Always answers 200 so Exotel does not
    retry a call we cannot place; unknown tokens are ignored.

    Field names are read leniently (CallSid or Sid, Status or CallStatus, Duration or ConversationDuration); check them
    against Exotel's current documentation before relying on this in production. The other providers post to
    /api/phone/{provider}/status-callback/{token} (their own field names: services/telephony)."""
    return await _status_callback(token, request, "exotel")


def _make_status_route(name: str):
    async def provider_status_callback(token: str, request: Request):
        return await _status_callback(token, request, name)
    provider_status_callback.__name__ = f"{name}_status_callback"
    return provider_status_callback


def _make_link_route(name: str):
    async def provider_inbound_link(body: LinkRequest, request: Request, user_id: str = Depends(get_current_user_id)):
        """The address to give this provider so that calls to its number reach this agent (see docs/PHONE_AND_MEETINGS.md)."""
        b = _b()
        agent = await _owned_agent(request, body.agent_id, user_id)
        mode = b._storage_mode(request)
        prov = registry.provider(name)
        token = make_token(agent["id"], user_id, mode, INBOUND_TOKEN_TTL_S, provider=name, kind="ans")
        url = answer_url(token, name)
        phone_greeting.schedule_prepare(b, agent, user_id, mode, warm=False)   # the greeting is ready for the first call
        return {"provider": name, "answer_url": url, "method": "POST", "valid_days": INBOUND_TOKEN_TTL_S // 86400,
                "how": prov.inbound_how, "max_calls": max_calls(), "configured": prov.configured()}
    provider_inbound_link.__name__ = f"{name}_inbound_link"
    return provider_inbound_link


# The routes of every provider but Exotel (whose own routes are above): the stream, the status callback and the answer
# document are public (signed token in the URL, served on the phone-only port too); the inbound link is the owner's.
for _pname in PROVIDER_NAMES:
    public_router.add_api_websocket_route(provider_path(_pname, "stream") + "/{token}", _make_stream_route(_pname))
    public_router.add_api_route(provider_path(_pname, "status-callback") + "/{token}", _make_status_route(_pname), methods=["POST"])
    public_router.add_api_route(provider_path(_pname, "answer") + "/{token}", _make_answer_route(_pname), methods=["GET", "POST"])
    router.add_api_route(provider_path(_pname, "inbound-link"), _make_link_route(_pname), methods=["POST"])


@router.get("/api/phone/exotel/calls")
async def my_calls(limit: int = 50, user_id: str = Depends(get_current_user_id)):
    """The owner's recent phone calls from the call records (newest first; numbers masked)."""
    return {"calls": call_records.list_for(user_id, limit)}


class CallSettingsIn(BaseModel):
    transfer_number: Optional[str] = Field(None, description="Where transfer_call hands the caller, in international "
                                                              "format; empty or null removes it")


@router.get("/api/agents/{agent_id}/call-settings")
async def get_call_settings(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    agent = await _owned_agent(request, agent_id, user_id)
    n = call_control.transfer_number(agent["id"])
    return {"transfer_number": n, "transfer_number_masked": mask(n) if n else None,
            "transfer_mode": call_control.transfer_mode()}


@router.put("/api/agents/{agent_id}/call-settings")
async def put_call_settings(agent_id: str, body: CallSettingsIn, request: Request,
                            user_id: str = Depends(get_current_user_id)):
    """Set (or clear) the number transfer_call hands callers to. The agent also needs the "call" tool server enabled."""
    agent = await _owned_agent(request, agent_id, user_id)
    raw = (body.transfer_number or "").strip()
    number = normalise_number(raw) if raw else None
    if raw and number is None:
        raise AppError("BAD_NUMBER", "Give the number in international format, for example +919876543210.", 422)
    call_control.set_transfer_number(agent["id"], number)
    return {"transfer_number": number, "transfer_number_masked": mask(number) if number else None,
            "transfer_mode": call_control.transfer_mode(), "tool_enabled": "call" in (agent.get("tools") or [])}


# ------------------------------------------------------------------------------------------------ live monitoring

def make_monitor_token(conversation_id: str, user_id: str, ttl: int = MONITOR_TOKEN_TTL_S,
                       now: Optional[float] = None) -> str:
    claims = {"c": conversation_id, "u": user_id, "k": "mon", "x": int((now if now is not None else time.time()) + ttl)}
    payload = _b64(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    return f"{payload}.{_mac(payload)}"


def read_monitor_token(token: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """The claims of a valid, unexpired monitor token, else None. A call's stream token is not accepted here."""
    try:
        payload, _, mac = str(token).partition(".")
        if not payload or not hmac.compare_digest(mac, _mac(payload)):
            return None
        claims = json.loads(_unb64(payload))
        if not isinstance(claims, dict) or claims.get("k") != "mon" or claims.get("x", 0) < (now if now is not None else time.time()):
            return None
        return claims if all(isinstance(claims.get(k), str) and claims[k] for k in ("c", "u")) else None
    except Exception:  # noqa: BLE001
        return None


def _live_view(s: "CallSession") -> Dict[str, Any]:
    rec = call_records.get(s.call_sid or "") or {}
    return {"conversation_id": s.conversation_id, "call_sid": s.call_sid, "agent_id": s.agent.get("id"),
            "agent_name": s.agent.get("name") or "Agent", "direction": rec.get("direction") or ("outbound" if s.outbound else "inbound"),
            "started_at": int(s.started_at), "seconds": int(time.time() - s.started_at), "human_active": s.human_active,
            "listeners": len(s.monitors)}


@router.get("/api/phone/exotel/live")
async def live_calls(user_id: str = Depends(get_current_user_id)):
    """The owner's phone calls that are happening right now."""
    return {"calls": [_live_view(s) for s in call_control.live_sessions() if s.claims.get("u") == user_id and s.conversation_id]}


@router.post("/api/phone/exotel/live/{conversation_id}/monitor-token")
async def monitor_token(conversation_id: str, user_id: str = Depends(get_current_user_id)):
    """A short-lived token for the listen-in socket of one live call (a browser cannot send headers on a WebSocket)."""
    s = call_control.session_by_conversation(conversation_id)
    if s is None or s.claims.get("u") != user_id:
        raise AppError("CALL_NOT_LIVE", "That call is not live.", 404)
    return {"token": make_monitor_token(conversation_id, user_id), "path": f"/api/phone/exotel/monitor/",
            "expires_in": MONITOR_TOKEN_TTL_S}


@router.websocket("/api/phone/exotel/monitor/{token}")
async def monitor_socket(ws: WebSocket, token: str):
    """Listen in on a live call, and take it over.

    Server to browser (JSON text): hello {type, rate, human, call_sid, agent}; audio {from: caller|agent|human, rate,
    pcm: base64 PCM16}; transcript {role, text}; state {human}; ended.
    Browser to server: {type: "takeover"} (the agent goes quiet), {type: "release"} (the agent listens again),
    {type: "audio", pcm: base64 PCM16 at the call's rate} (your voice, only after takeover). If the person who took the
    call over disconnects, the agent takes it back."""
    claims = read_monitor_token(token)
    if claims is None:
        await ws.close(code=4401)
        return
    session = call_control.session_by_conversation(claims["c"])
    if session is None or session.claims.get("u") != claims["u"]:
        await ws.close(code=4404)
        return
    if len(session.monitors) >= MAX_MONITORS:
        await ws.close(code=4429)
        return
    await ws.accept()
    mon = _Monitor()
    session.monitors.append(mon)
    await ws.send_text(json.dumps({"type": "hello", "rate": session.rate, "human": session.human_active,
                                   "call_sid": session.call_sid, "agent": session.agent.get("name") or "Agent"}))

    async def on_call(coro):
        """Run a call-session coroutine on the call's own loop (the same loop in the server; another thread's in tests)."""
        if session.loop is None or session.loop is asyncio.get_running_loop():
            return await coro
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, session.loop))

    async def pump() -> None:
        while True:
            msg = await mon.queue.get()
            await ws.send_text(json.dumps(msg))
            if msg.get("type") == "ended":
                return

    async def listen() -> None:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except ValueError:
                continue
            kind = msg.get("type") if isinstance(msg, dict) else None
            if kind == "takeover":
                if not await on_call(session.takeover(mon)):
                    mon.put({"type": "error", "code": "ALREADY_TAKEN", "message": "Someone else has this call."})
            elif kind == "release" and session.human_monitor is mon:
                await on_call(session.release())
            elif kind == "audio" and session.human_active and session.human_monitor is mon:
                raw = msg.get("pcm")
                if isinstance(raw, str) and 0 < len(raw) <= MONITOR_AUDIO_MAX_B64:
                    try:
                        pcm = base64.b64decode(raw)
                    except Exception:  # noqa: BLE001
                        continue
                    await on_call(session.send_pcm(pcm[: len(pcm) - len(pcm) % 2], who="human"))

    tasks = [asyncio.create_task(pump()), asyncio.create_task(listen())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception:  # noqa: BLE001
        pass
    finally:
        for t in tasks:
            t.cancel()
        if mon in session.monitors:
            session.monitors.remove(mon)
        if session.human_monitor is mon:       # the person left: the agent takes the call back
            await on_call(session.release())
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
