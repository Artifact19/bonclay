"""WhatsApp Cloud API plumbing for the WhatsApp voice-note channel (routers/whatsapp.py runs the conversation).

Everything here is transport and parsing, with no dependency on the web app:

  * webhook signature check (`X-Hub-Signature-256`, HMAC-SHA256 of the raw body keyed with the Meta app secret)
  * parsing of the webhook payload into plain `Inbound` messages (statuses are ignored)
  * human-handoff detection in English, Hindi (Devanagari) and Hinglish (Roman Hindi)
  * audio conversion: the caller's voice note (OGG/Opus) to 16 kHz WAV for speech recognition, and the agent's reply
    (WAV) to a WhatsApp voice note (OGG/Opus, mono, 48 kHz, about 24 kbps) with ffmpeg
  * a small Graph API client (download media, upload media, send a message, mark as read) and a dry-run twin that
    records what would have been sent, for the owner's "Simulate a message" button
  * the per-agent secrets (access token, app secret): a 0600 file in local mode, an BONCLAY_SECRETS_KEY encrypted blob
    on the agent row in cloud mode. They are never returned by any API and never logged.

Meta's rules this respects: replies only go to people who wrote first (the 24 hour customer service window; there are
no outbound-initiated messages and no templates here), and the agent says it is an AI at the start of a conversation
(done by the shared turn pipeline).
"""
from __future__ import annotations

import asyncio
import base64
import collections
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets as _secrets
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import httpx
import numpy as np

try:
    from services import audio_decode
except ImportError:  # pragma: no cover - imported as backend.services.whatsapp
    from backend.services import audio_decode

try:
    from core import offline
except ImportError:  # pragma: no cover - running from a different cwd
    from backend.core import offline

log = logging.getLogger("bonclay.whatsapp")

DEFAULT_GRAPH_URL = "https://graph.facebook.com/v23.0"
MAX_MEDIA_BYTES = 16 * 1024 * 1024         # WhatsApp's own limit for audio is 16 MB
MAX_VOICE_NOTE_S = 180.0                   # longer voice notes get a polite "please send a shorter one"
MAX_BODY_BYTES = 1024 * 1024               # a webhook call is a few KB; reject anything near this
MAX_TEXT_CHARS = 4096                      # WhatsApp text message limit
WINDOW_S = 23 * 3600                       # replies only inside the 24 h window (an hour of margin)
OGG_MIME = "audio/ogg; codecs=opus"
OPUS_BITRATE = "24k"
HTTP_TIMEOUT_S = 25.0
PHONE_ID_RE = re.compile(r"^\d{5,24}$")


class WhatsAppError(Exception):
    """A problem with a Graph API call or the audio. `message` is safe to show (never contains a secret)."""

    def __init__(self, code: str, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


# ------------------------------------------------------------------------------------------------- environment

def graph_base() -> str:
    return (os.getenv("WHATSAPP_GRAPH_URL") or DEFAULT_GRAPH_URL).strip().rstrip("/")


def global_verify_token() -> str:
    return os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()


def global_app_secret() -> str:
    return os.getenv("WHATSAPP_APP_SECRET", "").strip()


def global_access_token() -> str:
    return os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()


def mask_phone(wa_id: str) -> str:
    """`91••••••3210` for logs: the country code and the last four digits."""
    s = re.sub(r"\D", "", str(wa_id or ""))
    return s if len(s) <= 6 else f"{s[:2]}{'•' * (len(s) - 6)}{s[-4:]}"


# ------------------------------------------------------------------------------------------------- settings

DEFAULT_HANDOFF = {
    "en": "I'm an AI assistant, so I've passed your request to our team. A person will message you here soon.",
    "hi": "मैं एक AI असिस्टेंट हूँ, इसलिए मैंने आपका अनुरोध हमारी टीम को भेज दिया है। जल्द ही कोई व्यक्ति यहीं आपसे संपर्क करेगा।",
    "hinglish": "Main ek AI assistant hoon, isliye maine aapki request hamari team ko bhej di hai. Jaldi hi koi insaan "
                "yahin aapse baat karega.",
}
UNSUPPORTED_REPLY = {
    "en": "I can read text and voice notes. Please send one of those.",
    "hi": "मैं टेक्स्ट और वॉइस नोट समझ सकती हूँ। कृपया इनमें से कुछ भेजें।",
    "hinglish": "Main text aur voice note samajh sakti hoon. Kripya inmein se kuch bhejiye.",
}
UNHEARD_REPLY = {
    "en": "I couldn't hear anything in that voice note. Could you send it again, or type your question?",
    "hi": "मुझे उस वॉइस नोट में कुछ सुनाई नहीं दिया। कृपया दोबारा भेजें, या अपना सवाल लिखें।",
    "hinglish": "Mujhe us voice note mein kuch sunai nahi diya. Dobara bhejiye, ya apna sawal likh dijiye.",
}
TOO_LONG_REPLY = {
    "en": "That voice note is too long for me. Could you send a shorter one, or type your question?",
    "hi": "वह वॉइस नोट मेरे लिए बहुत लंबा है। कृपया छोटा भेजें, या अपना सवाल लिखें।",
    "hinglish": "Woh voice note mere liye bahut lamba hai. Chhota bhejiye, ya apna sawal likh dijiye.",
}
FAILED_REPLY = {
    "en": "Sorry, something went wrong on my side. Please try again in a moment.",
    "hi": "क्षमा करें, मेरी तरफ़ से कुछ गड़बड़ हो गई। कृपया थोड़ी देर में फिर कोशिश करें।",
    "hinglish": "Maaf kijiye, meri taraf se kuch gadbad ho gayi. Thodi der mein phir koshish kijiye.",
}


def canned(table: Dict[str, str], language: str) -> str:
    return table.get(language) or table["en"]


def new_verify_token() -> str:
    return "avt_" + _secrets.token_urlsafe(24)


def normalize_phone_number_id(value: Any) -> str:
    s = re.sub(r"\s+", "", str(value or ""))
    if s and not PHONE_ID_RE.match(s):
        raise ValueError("The phone number ID is the long number from Meta's dashboard (digits only), "
                         "not the phone number itself.")
    return s


def clean_config(raw: Any) -> dict:
    """The saved settings with defaults filled in; only known keys are kept (so `secrets_enc` and anything stray never
    leaves through here)."""
    raw = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(raw.get("enabled")),
        "phone_number_id": str(raw.get("phone_number_id") or ""),
        "handoff_text": str(raw.get("handoff_text") or ""),
        "voice_for_text": raw.get("voice_for_text") is not False,
        "verify_token": str(raw.get("verify_token") or ""),
        "access_token_hint": str(raw.get("access_token_hint") or ""),
        "access_token_set": bool(raw.get("access_token_set")),
        "app_secret_set": bool(raw.get("app_secret_set")),
        "created_at": raw.get("created_at"),
    }


# ------------------------------------------------------------------------------------------------- secrets

_AGENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_secrets_lock = threading.RLock()


def secrets_dir() -> Path:
    override = os.getenv("BONCLAY_WHATSAPP_DIR")
    return Path(override) if override else Path(__file__).resolve().parents[1] / "data" / "whatsapp"


def _secrets_path(agent_id: str) -> Path:
    if not _AGENT_ID_RE.match(str(agent_id or "")):
        raise WhatsAppError("INVALID_AGENT", "Bad agent id.")
    return secrets_dir() / f"{agent_id}.secrets.json"


def _fernet():
    key = os.getenv("BONCLAY_SECRETS_KEY", "").strip()
    if not key:
        return None
    from cryptography.fernet import Fernet
    try:
        return Fernet(key.encode())
    except Exception:  # noqa: BLE001 - any passphrase works: derive a Fernet key from it
        return Fernet(base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest()))


def secrets_storage_ok(local: bool) -> Tuple[bool, Optional[str]]:
    if local or _fernet() is not None:
        return True, None
    return False, ("Cloud mode stores the access token and app secret encrypted, and BONCLAY_SECRETS_KEY is not set "
                   "on the server. Set it (any long random string), or set WHATSAPP_ACCESS_TOKEN and "
                   "WHATSAPP_APP_SECRET in the server environment.")


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    os.chmod(path, 0o600)


def load_secrets(agent_id: str, *, local: bool, enc: Optional[str] = None) -> Dict[str, str]:
    """{"access_token"?, "app_secret"?} of an agent."""
    try:
        if local:
            path = _secrets_path(agent_id)
            with _secrets_lock:
                return json.loads(path.read_text("utf-8")) if path.is_file() else {}
        f = _fernet()
        return json.loads(f.decrypt(enc.encode()).decode("utf-8")) if (enc and f) else {}
    except Exception as e:  # noqa: BLE001
        log.warning("could not read WhatsApp secrets of agent %s: %s", agent_id, e.__class__.__name__)
        return {}


def save_secrets(agent_id: str, secrets: Dict[str, str], *, local: bool) -> Optional[str]:
    """Persist. Local: the 0600 file (returns None). Cloud: returns the encrypted blob for the agent row."""
    if local:
        path = _secrets_path(agent_id)
        with _secrets_lock:
            if secrets:
                _write_private(path, json.dumps(secrets).encode("utf-8"))
            elif path.exists():
                path.unlink()
        return None
    if not secrets:
        return None
    f = _fernet()
    if f is None:
        raise WhatsAppError("SECRETS_STORAGE_UNAVAILABLE", secrets_storage_ok(False)[1] or "")
    return f.encrypt(json.dumps(secrets).encode("utf-8")).decode("ascii")


def forget_agent(agent_id: str) -> None:
    try:
        p = _secrets_path(agent_id)
        if p.exists():
            p.unlink()
    except Exception as e:  # noqa: BLE001
        log.warning("could not remove WhatsApp secrets of agent %s: %s", agent_id, e.__class__.__name__)


def token_hint(token: str) -> str:
    """What the owner sees instead of the token: the last four characters."""
    t = (token or "").strip()
    return f"••••{t[-4:]}" if len(t) >= 8 else "••••"


# ------------------------------------------------------------------------------------------------- webhook

def signature_valid(app_secret: str, body: bytes, header: Optional[str]) -> bool:
    """`X-Hub-Signature-256: sha256=<hex>` is HMAC-SHA256(app secret, raw body). Constant-time compare."""
    if not app_secret or not header:
        return False
    header = header.strip()
    if not header.lower().startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[7:].strip().lower())


def sign(app_secret: str, body: bytes) -> str:
    """The header value Meta would send (used by the tests and the live check)."""
    return "sha256=" + hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def token_matches(candidate: str, known: List[str]) -> bool:
    """True when `candidate` equals any of `known` (compares all of them, in constant time)."""
    ok = False
    for k in known:
        if k and hmac.compare_digest(str(candidate or "").encode(), k.encode()):
            ok = True
    return ok


@dataclass
class Inbound:
    message_id: str
    wa_id: str                 # the sender's WhatsApp ID (their phone number, digits only)
    phone_number_id: str       # the business number it was sent to
    kind: str                  # "text" | "audio" | "other"
    timestamp: float = 0.0
    name: str = ""
    text: str = ""
    media_id: str = ""
    mime: str = ""
    raw_type: str = ""


def _values(payload: Any):
    if not isinstance(payload, dict):
        return
    for entry in payload.get("entry") or []:
        for change in (entry.get("changes") or []) if isinstance(entry, dict) else []:
            if isinstance(change, dict) and change.get("field", "messages") == "messages" \
                    and isinstance(change.get("value"), dict):
                yield change["value"]


def phone_number_ids(payload: Any) -> List[str]:
    """The business phone number IDs a payload talks about (untrusted until the signature is checked)."""
    seen: List[str] = []
    for value in _values(payload):
        meta = value.get("metadata")
        pid = str(meta.get("phone_number_id") or "") if isinstance(meta, dict) else ""
        if pid and pid not in seen:
            seen.append(pid)
    return seen


def parse_messages(payload: Any) -> Tuple[List[Inbound], int]:
    """(inbound messages, number of status updates ignored). Delivery/read statuses are not messages."""
    out: List[Inbound] = []
    statuses = 0
    for value in _values(payload):
        statuses += len(value.get("statuses") or [])
        meta = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        pid = str(meta.get("phone_number_id") or "")
        names: Dict[str, str] = {}
        for c in value.get("contacts") or []:
            if isinstance(c, dict):
                prof = c.get("profile") if isinstance(c.get("profile"), dict) else {}
                names[str(c.get("wa_id") or "")] = str(prof.get("name") or "")[:80]
        for m in value.get("messages") or []:
            if not isinstance(m, dict) or not m.get("id") or not m.get("from"):
                continue
            typ = str(m.get("type") or "")
            wa_id = str(m["from"])
            try:
                ts = float(m.get("timestamp") or 0)
            except (TypeError, ValueError):
                ts = 0.0
            msg = Inbound(message_id=str(m["id"]), wa_id=wa_id, phone_number_id=pid, kind="other", timestamp=ts,
                          name=names.get(wa_id, ""), raw_type=typ)
            if typ == "text" and isinstance(m.get("text"), dict):
                msg.kind, msg.text = "text", str(m["text"].get("body") or "")
            elif typ == "audio" and isinstance(m.get("audio"), dict) and m["audio"].get("id"):
                msg.kind, msg.media_id = "audio", str(m["audio"]["id"])
                msg.mime = str(m["audio"].get("mime_type") or "")
            elif typ == "button" and isinstance(m.get("button"), dict):
                msg.kind, msg.text = "text", str(m["button"].get("text") or "")
            elif typ == "interactive" and isinstance(m.get("interactive"), dict):
                it = m["interactive"]
                reply = it.get("button_reply") or it.get("list_reply") or {}
                msg.kind, msg.text = "text", (str(reply.get("title") or "") if isinstance(reply, dict) else "")
            if msg.kind == "text" and not msg.text.strip():
                msg.kind = "other"
            out.append(msg)
    return out, statuses


class SeenMessages:
    """Message IDs already handled (Meta retries a delivery it thinks failed). In memory, bounded, 48 h."""

    def __init__(self, cap: int = 5000, ttl_s: float = 48 * 3600):
        self.cap, self.ttl = cap, ttl_s
        self._ids: "collections.OrderedDict[str, float]" = collections.OrderedDict()
        self._lock = threading.Lock()

    def first_time(self, message_id: str) -> bool:
        """True the first time an ID is seen (and remembers it); False for a repeat."""
        now = time.time()
        with self._lock:
            while self._ids and (len(self._ids) >= self.cap or next(iter(self._ids.values())) < now - self.ttl):
                self._ids.popitem(last=False)
            if message_id in self._ids:
                return False
            self._ids[message_id] = now
            return True

    def clear(self) -> None:
        with self._lock:
            self._ids.clear()


class SenderLimiter:
    """At most `limit` messages per sender and hour: a stranger cannot spend the owner's quota."""

    def __init__(self, limit: int = 40, window_s: float = 3600.0):
        self.limit, self.window = limit, window_s
        self._hits: Dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            q = self._hits.setdefault(key, collections.deque())
            while q and q[0] < now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self._hits) > 5000:  # forget idle senders
                for k in [k for k, v in self._hits.items() if not v or v[-1] < now - self.window][:1000]:
                    self._hits.pop(k, None)
            return True

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


# ------------------------------------------------------------------------------------------------- handoff

def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    t = re.sub(r"[​-‍﻿]", "", t)
    t = re.sub(r"[^\w\sऀ-ॿ']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_EN_PERSON = (r"(?:human|humans|real person|live person|live agent|human agent|real human|person|agent|manager|"
              r"supervisor|representative|rep|operator|someone|somebody|staff|executive|customer (?:service|care|support)|"
              r"support team|your team)")
_EN_ADJ = r"(?:a |an |the |your |one |real |live |human |senior )*"
_EN_BARE = re.compile(r"^(?:(?:a|an|the|your|real|live|human|senior|customer)\s+)*"
                      r"(?:human|agent|manager|supervisor|representative|operator|person|someone|executive|"
                      r"customer care|customer service|customer support|live chat|live support)"
                      r"(?:\s+(?:please|pls|plz|now|needed|required))?$")
_EN_PATTERNS = [
    re.compile(rf"\b(?:talk|speak|chat|connect|transfer|escalate|forward|pass|put|get|patch|hand|switch)\b.{{0,30}}"
               rf"\b(?:to|with|over to|me to)\b\s+{_EN_ADJ}{_EN_PERSON}\b"),
    re.compile(rf"\b(?:i want|i need|i would like|i'd like|id like|get me|give me|can i get|could i get|may i have|"
               rf"i'm looking for|let me (?:talk|speak)|can i (?:talk|speak)|could i (?:talk|speak)|may i (?:talk|speak))"
               rf"\b.{{0,20}}\b{_EN_ADJ}"
               rf"(?:human|real person|live agent|human agent|manager|supervisor|representative|agent|person|operator)\b"),
    re.compile(r"\b(?:human|real person|live agent|live person|manager|supervisor|representative|operator)\b"
               r"\s+(?:please|pls|plz|now)\b"),
    re.compile(r"\b(?:not|no)\b.{0,12}\b(?:a )?(?:bot|robot|machine|ai)\b.{0,25}\b(?:human|person|agent|manager)\b"),
]
_HI_PERSON = (r"(?:इंसान|इन्सान|आदमी|ह्यूमन|मैनेजर|एजेंट|एजेन्ट|सुपरवाइज़र|सुपरवाइजर|प्रतिनिधि|"
              r"असली व्यक्ति|किसी व्यक्ति|किसी इंसान|किसी से|कर्मचारी|स्टाफ|ऑपरेटर|अधिकारी)")
_HI_VERB = r"(?:बात|मिला|मिलवा|जोड़|जोड|कनेक्ट|ट्रांसफर|बुला|चाहिए|चाहिये|कॉल)"
_HI_PATTERNS = [re.compile(rf"{_HI_PERSON}.{{0,25}}{_HI_VERB}"), re.compile(rf"{_HI_VERB}.{{0,10}}{_HI_PERSON}")]
_HI_BARE = re.compile(rf"^{_HI_PERSON}(?:\s+(?:कृपया|प्लीज़|प्लीज))?$")
_HG_PERSON = (r"(?:insaan|insan|inshaan|aadmi|admi|banda|human|manager|agent|supervisor|representative|operator|staff|"
              r"asli insaan|real person|koi insaan|koi aadmi|koi banda|koi human|kisi insaan|kisi aadmi|kisi human)")
_HG_VERB = r"(?:baat|bat|connect|milao|milwao|milwa|transfer|jodo|jod|bulao|bula|chahiye|chahie|chaiye|karao|karwao|kara)"
_HG_PATTERNS = [
    re.compile(rf"\b{_HG_PERSON}\b.{{0,25}}\b{_HG_VERB}\b"),
    re.compile(r"\bkisi (?:se|ke saath) (?:baat|connect)"),
    re.compile(r"\b(?:mujhe|mereko|muje|hume|humein)\b.{0,15}\b(?:insaan|insan|aadmi|banda|human|manager|agent|"
               r"supervisor|representative|real person)\b"),
]
_HG_BARE = re.compile(r"^(?:human|insaan|insan|manager|agent|supervisor|representative|aadmi|banda)"
                      r"(?:\s+(?:please|pls|plz|chahiye|se baat|ko bulao))?$")


def wants_human(text: str) -> bool:
    """True when the caller asks for a person: "I want to talk to a human", "agent please", "मुझे मैनेजर से बात करनी है",
    "kisi insaan se baat karao", "manager chahiye". A passing mention ("what do your agents charge?", "I am a travel
    agent") does not count: a handoff word has to come with a request (talk to, connect me, want, need) or be
    the whole message."""
    t = _norm(text)
    if not t:
        return False
    if _EN_BARE.match(t) or _HI_BARE.match(t) or _HG_BARE.match(t):
        return True
    return any(p.search(t) for p in (*_EN_PATTERNS, *_HI_PATTERNS, *_HG_PATTERNS))


# ------------------------------------------------------------------------------------------------- audio

def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise WhatsAppError("FFMPEG_MISSING", "ffmpeg is needed for WhatsApp voice notes (brew install ffmpeg).")
    return path


def wav_to_voice_note(wav_bytes: bytes) -> bytes:
    """A WhatsApp voice note: OGG container, Opus, mono, 48 kHz, about 24 kbps (what the Cloud API renders as a voice
    message). Blocking: run it in a thread."""
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-nostdin", "-i", "pipe:0", "-vn", "-map_metadata", "-1", "-c:a", "libopus",
         "-b:a", OPUS_BITRATE, "-vbr", "on", "-application", "voip", "-ac", "1", "-ar", "48000", "-f", "ogg", "pipe:1"],
        input=wav_bytes, capture_output=True, timeout=90)
    if proc.returncode != 0 or not proc.stdout:
        raise WhatsAppError("ENCODE_FAILED", "Could not encode the reply as a voice note: "
                            + proc.stderr.decode(errors="replace")[:160])
    return proc.stdout


def voice_note_to_wav(data: bytes, *, max_seconds: float = MAX_VOICE_NOTE_S) -> Tuple[bytes, float]:
    """(16 kHz mono 16-bit WAV, seconds) of a received voice note (OGG/Opus, or anything ffmpeg reads). Blocking.
    Raises WhatsAppError("TOO_LONG") above `max_seconds` and WhatsAppError("UNDECODABLE") for bad audio."""
    suffix = ".ogg" if data[:4] == b"OggS" else ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            samples = audio_decode.decode_file(path, 16000)
        except ValueError as e:
            raise WhatsAppError("UNDECODABLE", str(e)) from e
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    seconds = samples.size / 16000.0
    if seconds > max_seconds:
        raise WhatsAppError("TOO_LONG", f"voice note is {seconds:.0f} s long")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue(), seconds


# ------------------------------------------------------------------------------------------------- Graph API

_META_MEDIA_HOSTS = (".fbcdn.net", ".facebook.com", ".whatsapp.net", ".whatsapp.com", ".fbsbx.com")


def media_url_ok(url: str) -> bool:
    """The media URL Meta returns is fetched with the access token, so it must be https on a Meta host (or the same
    origin as the configured Graph URL, which is how the tests and the live check use a fake server)."""
    try:
        u, base = urlsplit(url), urlsplit(graph_base())
        port, base_port = u.port, base.port
    except ValueError:
        return False
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    if (u.scheme, u.hostname, port) == (base.scheme, base.hostname, base_port):
        return True
    host = u.hostname.lower()
    return u.scheme == "https" and any(host == h[1:] or host.endswith(h) for h in _META_MEDIA_HOSTS)


def _graph_error(resp: httpx.Response) -> WhatsAppError:
    """A WhatsAppError from a failed Graph response: Meta's own error message and code, nothing else."""
    msg, code = f"HTTP {resp.status_code}", ""
    try:
        err = resp.json().get("error") or {}
        if isinstance(err, dict):
            msg = str(err.get("message") or msg)[:300]
            code = str(err.get("code") or "")
    except Exception:  # noqa: BLE001
        pass
    return WhatsAppError("GRAPH_ERROR", f"WhatsApp said: {msg}" + (f" (code {code})" if code else ""),
                         resp.status_code)


class GraphClient:
    """The few Graph API calls the channel needs. One short-lived instance per message (the token is a field)."""

    dry_run = False

    def __init__(self, access_token: str, phone_number_id: str, *, base: Optional[str] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        self._token = access_token
        self.phone_number_id = phone_number_id
        self.base = (base or graph_base()).rstrip("/")
        self._transport = transport

    def __repr__(self) -> str:  # never show the token
        return f"GraphClient(phone_number_id={self.phone_number_id!r})"

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, follow_redirects=False, transport=self._transport)

    @property
    def _auth(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    async def _request(self, method: str, url: str, *, retry: bool = True, **kw) -> httpx.Response:
        if offline.enabled():
            raise WhatsAppError("OFFLINE", offline.reason("WhatsApp"))
        last: Optional[Exception] = None
        for attempt in range(2 if retry else 1):
            try:
                async with self._client() as c:
                    resp = await c.request(method, url, headers=self._auth, **kw)
                if resp.status_code >= 500 and attempt == 0 and retry:
                    await asyncio.sleep(0.5)
                    continue
                return resp
            except httpx.HTTPError as e:
                last = e
                if attempt == 0 and retry:
                    await asyncio.sleep(0.5)
        raise WhatsAppError("NETWORK", f"Could not reach WhatsApp ({last.__class__.__name__}).")

    async def fetch_media(self, media_id: str, cap: int = MAX_MEDIA_BYTES) -> Tuple[bytes, str]:
        """(bytes, mime) of a media item: GET /{media-id} gives a short-lived URL, then GET the URL with the token."""
        if not re.match(r"^[A-Za-z0-9_-]{1,100}$", media_id or ""):
            raise WhatsAppError("BAD_MEDIA_ID", "Bad media id.")
        meta = await self._request("GET", f"{self.base}/{media_id}")
        if meta.status_code != 200:
            raise _graph_error(meta)
        info = meta.json()
        url = str(info.get("url") or "")
        try:
            size = int(info.get("file_size") or 0)
        except (TypeError, ValueError):
            size = 0
        if size > cap:
            raise WhatsAppError("MEDIA_TOO_LARGE", "That media file is too large.")
        if not media_url_ok(url):
            raise WhatsAppError("BAD_MEDIA_URL", "WhatsApp returned a media URL that is not on a Meta host.")
        got = bytearray()
        try:
            async with self._client() as c:
                async with c.stream("GET", url, headers=self._auth) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        raise _graph_error(resp)
                    async for chunk in resp.aiter_bytes():
                        got.extend(chunk)
                        if len(got) > cap:
                            raise WhatsAppError("MEDIA_TOO_LARGE", "That media file is too large.")
        except httpx.HTTPError as e:
            raise WhatsAppError("NETWORK", f"Could not download the media ({e.__class__.__name__}).") from e
        return bytes(got), str(info.get("mime_type") or "")

    async def upload_media(self, data: bytes, *, mime: str = OGG_MIME, filename: str = "reply.ogg") -> str:
        """POST /{phone-number-id}/media (multipart); returns the media id."""
        resp = await self._request("POST", f"{self.base}/{self.phone_number_id}/media",
                                   data={"messaging_product": "whatsapp", "type": "audio/ogg"},
                                   files={"file": (filename, data, mime)})
        if resp.status_code != 200 or not resp.json().get("id"):
            raise _graph_error(resp)
        return str(resp.json()["id"])

    async def _send(self, payload: dict) -> dict:
        resp = await self._request("POST", f"{self.base}/{self.phone_number_id}/messages", json=payload)
        if resp.status_code not in (200, 201):
            raise _graph_error(resp)
        return resp.json()

    async def send_text(self, to: str, body: str) -> dict:
        return await self._send({"messaging_product": "whatsapp", "recipient_type": "individual", "to": to,
                                 "type": "text", "text": {"preview_url": False, "body": body[:MAX_TEXT_CHARS]}})

    async def send_audio(self, to: str, media_id: str) -> dict:
        return await self._send({"messaging_product": "whatsapp", "recipient_type": "individual", "to": to,
                                 "type": "audio", "audio": {"id": media_id}})

    async def mark_read(self, message_id: str) -> None:
        """Blue ticks (best effort: never fails the turn)."""
        try:
            await self._request("POST", f"{self.base}/{self.phone_number_id}/messages", retry=False,
                                json={"messaging_product": "whatsapp", "status": "read", "message_id": message_id})
        except WhatsAppError:
            pass

    async def phone_info(self) -> dict:
        """GET /{phone-number-id}: proves the token and the phone number id belong together."""
        resp = await self._request("GET", f"{self.base}/{self.phone_number_id}",
                                   params={"fields": "display_phone_number,verified_name"})
        if resp.status_code != 200:
            raise _graph_error(resp)
        return resp.json()


@dataclass
class DryRunClient:
    """Same calls as GraphClient, but nothing leaves the process: what would be sent is recorded in `sent`."""
    phone_number_id: str = "dry-run"
    sent: List[dict] = field(default_factory=list)
    dry_run: bool = True
    _media: Dict[str, bytes] = field(default_factory=dict)

    async def fetch_media(self, media_id: str, cap: int = MAX_MEDIA_BYTES) -> Tuple[bytes, str]:
        raise WhatsAppError("DRY_RUN", "The simulator has no media to download.")

    async def upload_media(self, data: bytes, *, mime: str = OGG_MIME, filename: str = "reply.ogg") -> str:
        media_id = f"dry-{len(self._media) + 1}"
        self._media[media_id] = data
        self.sent.append({"call": "upload", "mime": mime, "bytes": len(data), "media_id": media_id})
        return media_id

    async def send_text(self, to: str, body: str) -> dict:
        self.sent.append({"call": "message", "type": "text", "to": to, "body": body[:MAX_TEXT_CHARS]})
        return {}

    async def send_audio(self, to: str, media_id: str) -> dict:
        self.sent.append({"call": "message", "type": "audio", "to": to, "media_id": media_id,
                          "bytes": len(self._media.get(media_id, b""))})
        return {}

    async def mark_read(self, message_id: str) -> None:
        return None
