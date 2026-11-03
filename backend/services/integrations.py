"""Outbound integrations of an agent: a signed webhook and Google Sheets logging.

After every finished turn (and after every successful tool call) the backend hands one small job to a background
dispatcher. A turn never waits for it: `dispatch_turn` only builds the payloads and queues them; worker threads deliver
them, retry with backoff and write a delivery log the owner can read in the agent editor.

Webhook
    POST <owner's https URL>, JSON body, headers
        X-BonClay-Event       turn.completed | tool.called | test.ping
        X-BonClay-Delivery    uuid of the delivery (the same on every retry: use it to de-duplicate)
        X-BonClay-Timestamp   unix seconds of this attempt
        X-BonClay-Signature   sha256=<hex HMAC-SHA256(secret, "<timestamp>." + raw body)>
    Signing the timestamp together with the body means a captured request cannot be replayed later: the receiver
    rejects a timestamp older than five minutes (`verify_signature`, docs/INTEGRATIONS.md).
    SSRF: https only (http only for localhost, and only when local webhooks are allowed); the host is resolved, every
    address must be public (loopback only for the literal names localhost / 127.x / ::1 in local mode), the request
    goes to the resolved IP (no second DNS lookup an attacker could change), proxies are ignored and redirects are
    never followed.

Google Sheets
    Each completed turn appends one row to the owner's spreadsheet through the Sheets API v4 `values:append`
    (valueInputOption=RAW, so nothing a caller says can become a formula). The owner pastes a service-account key; it
    is stored server-side only. Rows of one agent that are due together go out as one batched request, and appends are
    paced below Google's 60 writes per minute per user. The service-account sign-in is a signed JWT (PyJWT with the
    `cryptography` package, both already dependencies) exchanged at Google's token endpoint: no google-auth needed.

Secrets (the webhook signing secret and the service-account key) never leave the server: local mode keeps them in a
0600 file under backend/data/integrations/, cloud mode keeps them encrypted (Fernet, BONCLAY_SECRETS_KEY) in the agent
row and refuses to store them when that key is missing. API responses carry only `*_set` flags; the signing secret is
shown once, when it is created or rotated.
"""
from __future__ import annotations

import base64
import collections
import hashlib
import hmac
import heapq
import ipaddress
import itertools
import json
import logging
import os
import re
import secrets as _secrets
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit

try:
    import httpx
except Exception:  # noqa: BLE001
    httpx = None

try:
    from core import offline
except ImportError:  # pragma: no cover - running from a different cwd
    from backend.core import offline

log = logging.getLogger("bonclay.integrations")

# ------------------------------------------------------------------------------------------------------ settings

EVENTS = ("turn.completed", "tool.called")
TEST_EVENT = "test.ping"
SIGNATURE_TOLERANCE_S = 300          # a receiver should reject a timestamp older than this
BACKOFF_S: Tuple[float, ...] = (2.0, 8.0, 30.0)  # waits before retry 1, 2, 3 (so at most 4 tries per delivery)
WEBHOOK_TIMEOUT_S = 5.0
SHEETS_TIMEOUT_S = 10.0
LOG_KEEP = 50
MAX_QUEUE = 1000
WORKERS = 4
TEXT_MAX = 4000                      # caller / agent text in a payload
SHEETS_APPENDS_PER_MIN = 50          # Google allows 60 write requests per minute per user
SHEETS_BATCH_MAX = 50
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"
SHEETS_COLUMNS = ["timestamp", "agent", "language", "caller_text", "agent_text", "booking_id", "booking_date",
                  "booking_time", "receipt_id"]
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
_AGENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_URL_IN_TEXT = re.compile(r"https?://\S+")


class IntegrationError(ValueError):
    """A setting the owner can fix. `code` is the API error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class SecretsUnavailable(IntegrationError):
    def __init__(self, message: str):
        super().__init__("SECRETS_STORAGE_UNAVAILABLE", message)


def _now() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(_now() if ts is None else ts, tz=timezone.utc).isoformat(timespec="seconds")


def _flag(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def local_webhooks_allowed() -> bool:
    """http://localhost targets are allowed when BONCLAY_ALLOW_LOCAL_WEBHOOKS=on; unset means "only in local mode"
    (Supabase auth is not configured, so this is a single-user machine)."""
    explicit = _flag("BONCLAY_ALLOW_LOCAL_WEBHOOKS")
    if explicit is not None:
        return explicit
    try:
        from services.auth import auth_configured
    except ImportError:  # pragma: no cover
        from backend.services.auth import auth_configured
    return not auth_configured()


# -------------------------------------------------------------------------------------------------- signature

def sign_body(secret: str, timestamp: Any, body: bytes) -> str:
    """`sha256=<hex>`: HMAC-SHA256 over "<timestamp>.<raw body>" keyed with the signing secret."""
    mac = hmac.new(secret.encode("utf-8"), f"{timestamp}.".encode("utf-8") + body, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


def verify_signature(secret: str, body: bytes, timestamp: Any, signature: str, *,
                     tolerance: float = SIGNATURE_TOLERANCE_S, now: Optional[float] = None) -> bool:
    """What a receiver does: constant-time compare, and a timestamp within `tolerance` seconds (replay window)."""
    try:
        ts = int(str(timestamp))
    except (TypeError, ValueError):
        return False
    if abs((_now() if now is None else now) - ts) > tolerance:
        return False
    return hmac.compare_digest(sign_body(secret, ts, body), str(signature or ""))


def new_webhook_secret() -> str:
    return "whsec_" + _secrets.token_urlsafe(32)


# -------------------------------------------------------------------------------------------------------- SSRF

def ip_block_reason(ip: str, allow_loopback: bool = False) -> Optional[str]:
    """Why a resolved address must not be called (None: fine). Everything that is not a public unicast address is
    blocked: private, link-local (including the 169.254.169.254 metadata address), CGNAT, multicast, reserved,
    unspecified, unique-local IPv6, and IPv4 addresses hidden in IPv6 form."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return "not an IP address"
    if isinstance(addr, ipaddress.IPv6Address):
        inner = addr.ipv4_mapped or addr.sixtofour
        if inner is not None:
            addr = inner
    if addr.is_loopback:
        return None if allow_loopback else "loopback address"
    if addr.is_unspecified:
        return "unspecified address"
    if addr.is_link_local:
        return "link-local address (cloud metadata range)"
    if addr.is_multicast:
        return "multicast address"
    if addr.is_private or addr.is_reserved or not addr.is_global:
        return "private or reserved address"
    return None


def resolve_host(host: str, port: int) -> List[str]:
    """All addresses of a host (the literal itself for an IP)."""
    try:
        ipaddress.ip_address(host)
        return [host]
    except ValueError:
        pass
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    seen: List[str] = []
    for info in infos:
        ip = info[4][0].split("%", 1)[0]
        if ip not in seen:
            seen.append(ip)
    return seen


def parse_webhook_url(url: Any) -> Tuple[str, str, int]:
    """(clean url, host, port) for a syntactically acceptable webhook URL, else IntegrationError. No network access."""
    text = str(url or "").strip()
    if not text:
        raise IntegrationError("INVALID_WEBHOOK_URL", "Enter the URL that should receive the events.")
    if len(text) > 500 or any(ord(c) < 33 for c in text):
        raise IntegrationError("INVALID_WEBHOOK_URL", "That URL is too long or contains spaces.")
    try:
        parts = urlsplit(text)
        host, port = parts.hostname, parts.port
    except ValueError as e:
        raise IntegrationError("INVALID_WEBHOOK_URL", "That is not a valid URL.") from e
    if parts.scheme not in ("https", "http") or not host:
        raise IntegrationError("INVALID_WEBHOOK_URL", "Use a full https:// URL.")
    if parts.username is not None or parts.password is not None:
        raise IntegrationError("INVALID_WEBHOOK_URL", "The URL must not contain a user name or password.")
    if parts.fragment:
        raise IntegrationError("INVALID_WEBHOOK_URL", "Remove the #fragment from the URL.")
    host = host.lower()
    if parts.scheme == "http":
        if host not in LOCAL_HOSTS:
            raise IntegrationError("INVALID_WEBHOOK_URL", "Only https:// URLs are allowed (http only for localhost).")
        if not local_webhooks_allowed():
            raise IntegrationError("INVALID_WEBHOOK_URL", "http://localhost is only allowed for local testing "
                                   "(set BONCLAY_ALLOW_LOCAL_WEBHOOKS=on). Use an https:// URL.")
    port = port or (443 if parts.scheme == "https" else 80)
    return text, host, port


def check_target(host: str, port: int) -> List[str]:
    """Resolve and vet a host: returns the addresses to use, or raises IntegrationError (blocked / unresolvable)."""
    allow_loopback = host in LOCAL_HOSTS and local_webhooks_allowed()
    try:
        ips = resolve_host(host, port)
    except OSError as e:
        raise IntegrationError("WEBHOOK_UNREACHABLE", f"Could not resolve the host name ({e.__class__.__name__}).") from e
    if not ips:
        raise IntegrationError("WEBHOOK_UNREACHABLE", "The host name did not resolve.")
    for ip in ips:
        reason = ip_block_reason(ip, allow_loopback)
        if reason:
            raise IntegrationError("WEBHOOK_URL_BLOCKED", f"Blocked: the URL points to a {reason}. Webhooks may only "
                                   "call public internet addresses.")
    return ips


def validate_webhook_url(url: Any) -> str:
    """Save-time check: syntax plus DNS/IP vetting. Blocking (DNS): call off the event loop."""
    clean, host, port = parse_webhook_url(url)
    check_target(host, port)
    return clean


def _clean_error(text: str) -> str:
    """An error line for the log: no URLs (they may carry a token in the path), no control characters, short."""
    t = _URL_IN_TEXT.sub("<url>", str(text))
    return "".join(ch if ch >= " " else " " for ch in t)[:300]


def _post_pinned(url: str, body: bytes, headers: Dict[str, str], timeout: float) -> Tuple[Optional[int], Optional[str], bool]:
    """POST once. Returns (status code or None, error or None, retryable). The connection goes to the vetted IP."""
    if httpx is None:
        return None, "httpx is not installed", False
    if offline.enabled():
        return None, offline.reason("outbound webhooks"), False
    try:
        clean, host, port = parse_webhook_url(url)
        ips = check_target(host, port)
    except IntegrationError as e:
        return None, e.message, e.code == "WEBHOOK_UNREACHABLE"
    parts = urlsplit(clean)
    ip = ips[0]
    netloc = f"[{ip}]" if ":" in ip else ip
    default_port = 443 if parts.scheme == "https" else 80
    if port != default_port:
        netloc += f":{port}"
    target = parts._replace(netloc=netloc).geturl()
    send_headers = {**headers, "Host": host if port == default_port else f"{host}:{port}"}
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=False, trust_env=False) as client:
            with client.stream("POST", target, content=body, headers=send_headers,
                               extensions={"sni_hostname": host}) as resp:
                code = resp.status_code
    except httpx.TimeoutException:
        return None, f"timed out after {timeout:g} s", True
    except Exception as e:  # noqa: BLE001 - network errors of every kind
        return None, _clean_error(f"{e.__class__.__name__}: {e}"), True
    if 200 <= code < 300:
        return code, None, False
    if 300 <= code < 400:
        return code, f"HTTP {code}: redirects are not followed", False
    return code, f"HTTP {code}", code in (408, 425, 429) or code >= 500


# ---------------------------------------------------------------------------------------------------- redaction

try:  # the same phone pattern as the knowledge-gaps report
    from services import knowledge_gaps as _kg
except ImportError:  # pragma: no cover
    try:
        from backend.services import knowledge_gaps as _kg
    except Exception:  # noqa: BLE001
        _kg = None
_PHONE_FALLBACK = re.compile(r"(?<![\w])(?:\+\s?91[\s-]?|91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_PHONE_INTL = re.compile(r"(?<![\w])\+\d{1,3}[\s-]?\(?\d{1,4}\)?(?:[\s-]?\d{2,4}){2,4}(?!\d)")
_PHONE_RUN = re.compile(r"(?<![\d-])\d{10,15}(?![\d-])")
_PHONE_KEYS = ("phone", "phone_number", "mobile")


def redact_phones(text: str) -> str:
    """Replace phone numbers by [phone] (Indian mobile numbers with or without +91, international +CC numbers and
    bare runs of 10-15 digits). Dates and times are left alone."""
    t = str(text)
    t = (getattr(_kg, "_PHONE_IN", None) or _PHONE_FALLBACK).sub("[phone]", t)
    t = _PHONE_INTL.sub("[phone]", t)
    return _PHONE_RUN.sub("[phone]", t)


def _redact_value(value: Any, key: str = "") -> Any:
    if key.lower() in _PHONE_KEYS and isinstance(value, (str, int)) and not isinstance(value, bool):
        return "[phone]" if str(value).strip() else value
    if isinstance(value, str):
        return redact_phones(value)
    if isinstance(value, dict):
        return {k: _redact_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, key) for v in value]
    return value


def redact_payload(payload: dict) -> dict:
    return _redact_value(payload)


# ----------------------------------------------------------------------------------------------- settings model

def default_cfg() -> dict:
    return {
        "webhook": {"enabled": False, "url": "", "include_test": False, "redact_phones": False,
                    "events": list(EVENTS), "secret_set": False, "secret_created_at": None},
        "sheets": {"enabled": False, "spreadsheet_id": "", "sheet_name": "", "include_test": False,
                   "redact_phones": False, "service_account_email": "", "credentials_set": False},
        # one row per conversation with what the caller told the agent (services/caller_sheet.py)
        "caller_sheet": {"enabled": False, "spreadsheet_id": "", "sheet_name": "BonClay", "include_phone": True,
                         "include_test": False},
    }


def normalize_cfg(raw: Any) -> dict:
    """The settings of an agent row, with defaults filled in and only known keys kept (never `secrets_enc`)."""
    out = default_cfg()
    if not isinstance(raw, dict):
        return out
    for section in ("webhook", "sheets", "caller_sheet"):
        saved = raw.get(section)
        if not isinstance(saved, dict):
            continue
        for k, default in out[section].items():
            if k in saved and saved[k] is not None and isinstance(default, (bool, str)):
                out[section][k] = type(default)(saved[k])
        if section == "webhook":
            if isinstance(saved.get("events"), list):
                out[section]["events"] = [e for e in saved["events"] if e in EVENTS]
            out[section]["secret_created_at"] = saved.get("secret_created_at")
    return out


def public_cfg(raw: Any) -> dict:
    """What the API returns: the settings and flags, never a secret."""
    cfg = normalize_cfg(raw)
    url = cfg["webhook"]["url"]
    cfg["webhook"]["url_host"] = (urlsplit(url).hostname or "") if url else ""
    return cfg


def spreadsheet_id_from(value: Any) -> str:
    """The id from a Google Sheets link or a bare id."""
    text = str(value or "").strip()
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]{20,})", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,100}", text):
        return text
    raise IntegrationError("INVALID_SPREADSHEET", "Paste the spreadsheet's link (https://docs.google.com/spreadsheets/"
                           "d/...) or its id.")


def clean_sheet_name(value: Any) -> str:
    """A tab name ('' = the first tab)."""
    name = str(value or "").strip()
    if len(name) > 100 or any(c in name for c in "[]*?:/\\!") or any(ord(c) < 32 for c in name):
        raise IntegrationError("INVALID_SHEET_NAME", "The tab name has characters Google Sheets does not allow.")
    return name


def parse_service_account(raw: Any) -> dict:
    """Validate a pasted service-account key. Returns the key fields the Sheets client needs."""
    try:
        data = raw if isinstance(raw, dict) else json.loads(str(raw or ""))
    except (TypeError, ValueError) as e:
        raise IntegrationError("INVALID_SERVICE_ACCOUNT", "That is not valid JSON. Paste the whole key file you "
                               "downloaded from Google Cloud.") from e
    if not isinstance(data, dict) or data.get("type") != "service_account":
        raise IntegrationError("INVALID_SERVICE_ACCOUNT", 'This is not a service-account key (its "type" must be '
                               '"service_account").')
    email, key = str(data.get("client_email") or ""), str(data.get("private_key") or "")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or "BEGIN" not in key:
        raise IntegrationError("INVALID_SERVICE_ACCOUNT", 'The key must contain "client_email" and "private_key".')
    try:
        from cryptography.hazmat.primitives import serialization
        serialization.load_pem_private_key(key.encode(), password=None)
    except Exception as e:  # noqa: BLE001
        raise IntegrationError("INVALID_SERVICE_ACCOUNT", "The private key in that file could not be read.") from e
    return {"type": "service_account", "client_email": email, "private_key": key,
            "private_key_id": str(data.get("private_key_id") or ""), "project_id": str(data.get("project_id") or "")}


# ---------------------------------------------------------------------------------------------------- secrets

def secrets_dir() -> Path:
    override = os.getenv("BONCLAY_INTEGRATIONS_DIR")
    return Path(override) if override else Path(__file__).resolve().parents[1] / "data" / "integrations"


def _agent_path(agent_id: str, suffix: str) -> Path:
    if not _AGENT_ID_RE.match(str(agent_id or "")):
        raise IntegrationError("INVALID_AGENT", "Bad agent id.")
    return secrets_dir() / f"{agent_id}{suffix}"


def _fernet():
    key = os.getenv("BONCLAY_SECRETS_KEY", "").strip()
    if not key:
        return None
    from cryptography.fernet import Fernet
    try:
        return Fernet(key.encode())
    except Exception:  # noqa: BLE001 - any passphrase works: derive a Fernet key from it
        return Fernet(base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest()))


def secrets_storage_status(local: bool) -> dict:
    if local or _fernet() is not None:
        return {"available": True, "reason": None}
    return {"available": False, "reason": "Cloud mode stores secrets encrypted, and BONCLAY_SECRETS_KEY is not set on "
            "the server. Set it (any long random string) to use the signing secret and Google Sheets."}


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


_secrets_lock = threading.RLock()


def load_secrets(agent_id: str, *, local: bool, enc: Optional[str] = None) -> dict:
    """{"webhook_secret"?, "service_account"?} of an agent. Local mode reads the 0600 file; cloud mode decrypts `enc`."""
    try:
        if local:
            path = _agent_path(agent_id, ".secrets.json")
            with _secrets_lock:
                return json.loads(path.read_text("utf-8")) if path.is_file() else {}
        if not enc:
            return {}
        f = _fernet()
        return json.loads(f.decrypt(enc.encode()).decode("utf-8")) if f else {}
    except Exception as e:  # noqa: BLE001
        log.warning("could not read integration secrets of agent %s: %s", agent_id, e.__class__.__name__)
        return {}


def save_secrets(agent_id: str, secrets: dict, *, local: bool) -> Optional[str]:
    """Persist the secrets. Local: the file (returns None). Cloud: returns the encrypted blob for the agent row, or
    raises SecretsUnavailable when no BONCLAY_SECRETS_KEY is configured."""
    if local:
        path = _agent_path(agent_id, ".secrets.json")
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
        raise SecretsUnavailable(secrets_storage_status(False)["reason"])
    return f.encrypt(json.dumps(secrets).encode("utf-8")).decode("ascii")


def forget_agent(agent_id: str) -> None:
    """The agent was deleted: remove its secrets, log and cached token (best effort, never raises)."""
    try:
        for suffix in (".secrets.json", ".log.json"):
            p = _agent_path(agent_id, suffix)
            if p.exists():
                p.unlink()
        with _log_lock:
            _log_cache.pop(agent_id, None)
        _token_cache.pop(agent_id, None)
        _token_cache.pop("caller-sheet:" + agent_id, None)
        try:
            from services import caller_sheet
        except ImportError:  # pragma: no cover
            from backend.services import caller_sheet
        caller_sheet.forget_agent(agent_id)
    except Exception as e:  # noqa: BLE001
        log.warning("could not clean up integrations of agent %s: %s", agent_id, e)


# ---------------------------------------------------------------------------------------------- delivery log

_log_lock = threading.RLock()
_log_cache: Dict[str, List[dict]] = {}
_cloud_writes = itertools.count(1)


def _backend():
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "supabase"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "supabase"):
            return main
    return mod


def _log_load(agent_id: str) -> List[dict]:
    if agent_id in _log_cache:
        return _log_cache[agent_id]
    rows: List[dict] = []
    try:
        path = _agent_path(agent_id, ".log.json")
        if path.is_file():
            rows = json.loads(path.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        rows = []
    _log_cache[agent_id] = rows
    return rows


def _entry_row(entry: dict, agent_id: str, user_id: Optional[str]) -> dict:
    return {"delivery_id": entry["delivery_id"], "agent_id": agent_id, "user_id": user_id, "target": entry["target"],
            "event": entry["event"], "status": entry["status"], "code": entry.get("code"),
            "latency_ms": entry.get("latency_ms"), "error": entry.get("error"), "attempts": entry.get("attempts", 1),
            "updated_at": entry["at"]}


def log_update(agent_id: str, user_id: Optional[str], local: bool, entry: dict) -> None:
    """Insert or update (by delivery_id) one entry of the agent's delivery log; keeps the newest LOG_KEEP."""
    try:
        with _log_lock:
            rows = _log_load(agent_id)
            for i, r in enumerate(rows):
                if r["delivery_id"] == entry["delivery_id"]:
                    rows[i] = {**r, **entry}
                    break
            else:
                rows.insert(0, entry)
            del rows[LOG_KEEP:]
            if local:
                _write_private(_agent_path(agent_id, ".log.json"), json.dumps(rows).encode("utf-8"))
        if not local:
            _cloud_log_write(agent_id, user_id, entry)
    except Exception as e:  # noqa: BLE001 - the log must never break a delivery
        log.warning("delivery log write failed: %s", e.__class__.__name__)


def _cloud_log_write(agent_id: str, user_id: Optional[str], entry: dict) -> None:
    sb = getattr(_backend(), "supabase", None)
    if sb is None:
        return
    try:
        sb.table("integration_deliveries").upsert(_entry_row(entry, agent_id, user_id),
                                                  on_conflict="delivery_id").execute()
        if next(_cloud_writes) % 20 == 0:  # keep the table small: drop everything past the newest LOG_KEEP
            old = (sb.table("integration_deliveries").select("delivery_id").eq("agent_id", agent_id)
                   .order("created_at", desc=True).range(LOG_KEEP, LOG_KEEP + 500).execute().data or [])
            if old:
                sb.table("integration_deliveries").delete().in_("delivery_id", [r["delivery_id"] for r in old]).execute()
    except Exception as e:  # noqa: BLE001 - migration 015 may not be applied; the in-memory log still works
        log.warning("could not write the delivery log (apply migration 015_agent_integrations.sql): %s",
                    e.__class__.__name__)


def get_log(agent_id: str, *, local: bool) -> List[dict]:
    """The newest deliveries of an agent (newest first). Blocking."""
    if not local:
        sb = getattr(_backend(), "supabase", None)
        if sb is not None:
            try:
                data = (sb.table("integration_deliveries").select("*").eq("agent_id", agent_id)
                        .order("updated_at", desc=True).limit(LOG_KEEP).execute().data or [])
                return [{"delivery_id": r["delivery_id"], "target": r["target"], "event": r["event"],
                         "status": r["status"], "code": r.get("code"), "latency_ms": r.get("latency_ms"),
                         "error": r.get("error"), "attempts": r.get("attempts") or 1,
                         "at": r.get("updated_at"), "created_at": r.get("created_at")} for r in data]
            except Exception as e:  # noqa: BLE001
                log.warning("could not read the delivery log from the database: %s", e.__class__.__name__)
    with _log_lock:
        return [dict(r) for r in _log_load(agent_id)]


# -------------------------------------------------------------------------------------------------- payloads

def _booking_of(tool_calls: List[Any]) -> Optional[dict]:
    """The booking a turn made or changed: the last successful tool call whose result carries a booking_id."""
    found = None
    for c in tool_calls:
        value = getattr(c, "value", None)
        if getattr(c, "ok", False) and isinstance(value, dict) and value.get("booking_id"):
            found = {k: value.get(k) for k in ("booking_id", "name", "phone", "doctor", "specialty", "date", "weekday",
                                               "time", "status", "reason") if value.get(k) not in (None, "")}
    return found


def _clip(text: Any) -> str:
    return str(text or "")[:TEXT_MAX]


def turn_payload(run: Any, turn_id: str, agent_text: str, latency_ms: int) -> dict:
    """The event body shared by the turn's webhook events and the Sheets row (before redaction)."""
    calls = list(getattr(run, "tool_objs", None) or [])
    return {
        "agent_id": run.agent_id,
        "conversation_id": getattr(run, "conversation_id", None),
        "turn_id": turn_id,
        "created_at": _iso(),
        "language": getattr(run, "language", None),
        "caller_text": _clip(getattr(run, "user_text", "")),
        "agent_text": _clip(agent_text),
        "tool_calls": [c.record() for c in calls],
        "booking": _booking_of(calls),
        "receipt_id": None,  # looked up by the worker (the receipt is written before the turn is stored)
        "is_test": bool(getattr(run, "is_test", False)),
        "timings": {**dict(getattr(run, "timings", None) or {}), "total_ms": int(latency_ms)},
    }


@dataclass
class Job:
    kind: str                         # "webhook" | "sheets" | "caller_sheet"
    agent_id: str
    user_id: str
    local: bool
    enc: Optional[str]                # cloud: the agent row's encrypted secrets
    event: str
    payload: dict
    agent_name: str = ""
    url: str = ""
    sheet: Dict[str, str] = field(default_factory=dict)
    delivery_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attempt: int = 0
    body: Optional[bytes] = None
    created_at: str = field(default_factory=_iso)
    rate_waits: int = 0


def _find_receipt_id(turn_id: str) -> Optional[str]:
    try:
        try:
            from services import provenance
        except ImportError:  # pragma: no cover
            from backend.services import provenance
        rows = provenance._store.find(turn_id=turn_id)
        return rows[-1].get("receipt_id") if rows else None
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------- dispatcher

class Dispatcher:
    """A bounded, time-ordered job queue served by a few daemon threads. Retries re-enter the queue with a delay, so a
    receiver that is down never occupies a worker while it waits."""

    def __init__(self, workers: int = WORKERS):
        self._cv = threading.Condition()
        self._heap: List[Tuple[float, int, Job]] = []
        self._seq = itertools.count()
        self._workers = workers
        self._threads: List[threading.Thread] = []
        self._inflight = 0
        self._stopped = False

    def submit(self, job: Job, delay: float = 0.0) -> bool:
        with self._cv:
            if len(self._heap) >= MAX_QUEUE:
                return False
            heapq.heappush(self._heap, (_now() + delay, next(self._seq), job))
            self._ensure_threads()
            self._cv.notify()
            return True

    def _ensure_threads(self) -> None:
        self._threads = [t for t in self._threads if t.is_alive()]
        while len(self._threads) < self._workers and not self._stopped:
            t = threading.Thread(target=self._run, name=f"bonclay-integrations-{len(self._threads)}", daemon=True)
            self._threads.append(t)
            t.start()

    def pending(self) -> int:
        with self._cv:
            return len(self._heap) + self._inflight

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """For tests and shutdown: wait until nothing is queued or running."""
        end = _now() + timeout
        with self._cv:
            while self._heap or self._inflight:
                left = end - _now()
                if left <= 0:
                    return False
                self._cv.wait(min(left, 0.05))
        return True

    def clear(self) -> None:
        with self._cv:
            self._heap.clear()
            self._cv.notify_all()

    def _take(self) -> Optional[List[Job]]:
        """Block until a job is due; a Sheets job takes the other due Sheets jobs of its agent along (one request)."""
        with self._cv:
            while not self._stopped:
                if not self._heap:
                    self._cv.wait(1.0)
                    continue
                wait = self._heap[0][0] - _now()
                if wait > 0:
                    self._cv.wait(min(wait, 1.0))
                    continue
                due, seq, job = heapq.heappop(self._heap)
                batch = [job]
                if job.kind == "sheets":
                    keep, taken, now = [], [(due, seq, job)], _now()
                    for item in self._heap:
                        j = item[2]
                        if (j.kind == "sheets" and j.agent_id == job.agent_id and item[0] <= now
                                and len(taken) < SHEETS_BATCH_MAX):
                            taken.append(item)
                        else:
                            keep.append(item)
                    if len(taken) > 1:
                        heapq.heapify(keep)
                        self._heap = keep
                        batch = [t[2] for t in sorted(taken, key=lambda t: (t[0], t[1]))]
                self._inflight += 1
                return batch
        return None

    def _run(self) -> None:
        while True:
            batch = self._take()
            if batch is None:
                return
            try:
                if batch[0].kind == "sheets":
                    _process_sheets(self, batch)
                elif batch[0].kind == "caller_sheet":
                    try:
                        from services import caller_sheet
                    except ImportError:  # pragma: no cover
                        from backend.services import caller_sheet
                    caller_sheet.process(self, batch[0])
                else:
                    _process_webhook(self, batch[0])
            except Exception:  # noqa: BLE001 - a bug in one delivery must not kill the worker
                log.exception("integration delivery crashed")
                for j in batch:
                    log_update(j.agent_id, j.user_id, j.local, _entry(j, "failed", error="internal error"))
            finally:
                with self._cv:
                    self._inflight -= 1
                    self._cv.notify_all()


dispatcher = Dispatcher()


def _entry(job: Job, status: str, *, code: Optional[int] = None, latency_ms: Optional[int] = None,
           error: Optional[str] = None) -> dict:
    return {"delivery_id": job.delivery_id, "target": job.kind, "event": job.event, "status": status, "code": code,
            "latency_ms": latency_ms, "error": error, "attempts": job.attempt + 1, "at": _iso(),
            "created_at": job.created_at}


def _retry_or_fail(d: Dispatcher, job: Job, *, code: Optional[int], latency_ms: int, error: str, retryable: bool,
                   wait: Optional[float] = None) -> None:
    if retryable and job.attempt < len(BACKOFF_S):
        delay = BACKOFF_S[job.attempt] if wait is None else wait
        log_update(job.agent_id, job.user_id, job.local,
                   _entry(job, "retrying", code=code, latency_ms=latency_ms,
                          error=f"{error} - retrying in {delay:g} s"))
        job.attempt += 1
        if d.submit(job, delay):
            return
        job.attempt -= 1
        error = "queue full"
    log_update(job.agent_id, job.user_id, job.local, _entry(job, "failed", code=code, latency_ms=latency_ms, error=error))


# --------------------------------------------------------------------------------------------------- webhook

def webhook_request(job: Job, secret: str) -> Tuple[bytes, Dict[str, str]]:
    """The exact body and headers of one attempt (a fresh timestamp and signature each time)."""
    if job.body is None:
        job.body = json.dumps({"event": job.event, **job.payload}, ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8")
    ts = int(_now())
    return job.body, {
        "Content-Type": "application/json", "User-Agent": "BonClay-Webhooks/1",
        "X-BonClay-Event": job.event, "X-BonClay-Delivery": job.delivery_id, "X-BonClay-Timestamp": str(ts),
        "X-BonClay-Signature": sign_body(secret, ts, job.body)}


def attempt_webhook(job: Job) -> Tuple[Optional[int], Optional[str], bool, int]:
    """One delivery attempt: (code, error, retryable, latency ms)."""
    secret = load_secrets(job.agent_id, local=job.local, enc=job.enc).get("webhook_secret")
    if not secret:
        return None, "no signing secret is stored for this agent", False, 0
    if job.payload.get("receipt_id") is None and job.body is None and job.payload.get("turn_id"):
        job.payload["receipt_id"] = _find_receipt_id(job.payload["turn_id"])
    body, headers = webhook_request(job, secret)
    started = time.time()
    code, error, retryable = _post_pinned(job.url, body, headers, WEBHOOK_TIMEOUT_S)
    return code, error, retryable, int((time.time() - started) * 1000)


def _process_webhook(d: Dispatcher, job: Job) -> None:
    code, error, retryable, ms = attempt_webhook(job)
    if error is None:
        log_update(job.agent_id, job.user_id, job.local, _entry(job, "delivered", code=code, latency_ms=ms))
    else:
        _retry_or_fail(d, job, code=code, latency_ms=ms, error=error, retryable=retryable)


def send_test_event(agent_id: str, user_id: str, local: bool, enc: Optional[str], cfg: dict, agent_name: str) -> dict:
    """The "Send test event" button: one synchronous attempt (no retries), logged like any delivery. Blocking."""
    wh = cfg["webhook"]
    payload = {"agent_id": agent_id, "conversation_id": None, "turn_id": None, "created_at": _iso(),
               "language": "en", "caller_text": "This is a test event from BonClay.",
               "agent_text": "If you can read this, your endpoint received it and the signature can be checked.",
               "tool_calls": [], "booking": None, "receipt_id": None, "is_test": True,
               "timings": {"total_ms": 0}}
    job = Job("webhook", agent_id, user_id, local, enc, TEST_EVENT, payload, agent_name, url=wh["url"])
    code, error, _retry, ms = attempt_webhook(job)
    entry = _entry(job, "delivered" if error is None else "failed", code=code, latency_ms=ms, error=error)
    log_update(agent_id, user_id, local, entry)
    return entry


# --------------------------------------------------------------------------------------------------- sheets

_token_cache: Dict[str, Tuple[str, float, str]] = {}
_sheets_times: Dict[str, collections.deque] = {}
_sheets_lock = threading.Lock()


def _google_http(method: str, url: str, *, headers: Optional[dict] = None, json_body: Any = None,
                 data: Optional[dict] = None, timeout: float = SHEETS_TIMEOUT_S) -> Tuple[int, Any]:
    """The only place the Google APIs are called (tests replace it). No redirects, no proxies."""
    if httpx is None:
        raise RuntimeError("httpx is not installed")
    offline.block("Google Sheets")
    with httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=False, trust_env=False) as client:
        resp = client.request(method, url, headers=headers, json=json_body, data=data)
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, resp.text[:500]


class SheetsError(Exception):
    def __init__(self, message: str, *, status: Optional[int] = None, retryable: bool = False,
                 wait: Optional[float] = None, hint: str = ""):
        super().__init__(message)
        self.message, self.status, self.retryable, self.wait, self.hint = message, status, retryable, wait, hint


def _jwt_assertion(creds: dict, now: float) -> str:
    import jwt
    claims = {"iss": creds["client_email"], "scope": SHEETS_SCOPE, "aud": GOOGLE_TOKEN_URL, "iat": int(now) - 5,
              "exp": int(now) + 3300}
    headers = {"kid": creds["private_key_id"]} if creds.get("private_key_id") else None
    return jwt.encode(claims, creds["private_key"], algorithm="RS256", headers=headers)


def _access_token(agent_id: str, creds: dict) -> str:
    fingerprint = hashlib.sha256(creds["private_key"].encode()).hexdigest()[:16]
    cached = _token_cache.get(agent_id)
    if cached and cached[2] == fingerprint and cached[1] - 60 > _now():
        return cached[0]
    try:
        status, body = _google_http("POST", GOOGLE_TOKEN_URL, data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": _jwt_assertion(creds, _now())})
    except Exception as e:  # noqa: BLE001
        raise SheetsError(_clean_error(f"Could not reach Google to sign in: {e.__class__.__name__}"),
                          retryable=True) from e
    if status != 200 or not isinstance(body, dict) or not body.get("access_token"):
        desc = ""
        if isinstance(body, dict):
            desc = str(body.get("error_description") or body.get("error") or "")
        raise SheetsError(f"Google rejected the service-account key: {desc or f'HTTP {status}'}", status=status,
                          retryable=status >= 500 or status == 429,
                          hint="Check that the key was not deleted or disabled in Google Cloud.")
    _token_cache[agent_id] = (str(body["access_token"]), _now() + float(body.get("expires_in") or 3600), fingerprint)
    return str(body["access_token"])


def _sheets_error(status: int, body: Any, email: str, retry_after: Optional[float] = None) -> SheetsError:
    message = ""
    if isinstance(body, dict):
        err = body.get("error")
        message = str(err.get("message") if isinstance(err, dict) else err or "")
    message = message or (body if isinstance(body, str) else "") or f"HTTP {status}"
    low = message.lower()
    hint = ""
    if status == 403 and ("has not been used" in low or "is disabled" in low or "service_disabled" in low):
        hint = "Enable the Google Sheets API for the Google Cloud project of this service account."
    elif status in (403, 401):
        hint = f"Share the spreadsheet with {email} (Editor access) and try again."
    elif status == 404:
        hint = "Check the spreadsheet link or id (and that it is shared with the service account)."
    elif status == 400 and "range" in low:
        hint = "Check the tab name: it must match a tab of the spreadsheet exactly, or leave it empty."
    return SheetsError(f"Google Sheets said ({status}): {message}"[:400], status=status,
                       retryable=status in (408, 429) or status >= 500, wait=retry_after, hint=hint)


def sheets_range(sheet_name: str) -> str:
    """A1 notation that appends after the last row of a tab ('' = the first tab)."""
    if not sheet_name:
        return "A1"
    return "'" + sheet_name.replace("'", "''") + "'!A1"


def append_rows(agent_id: str, creds: dict, spreadsheet_id: str, sheet_name: str, rows: List[List[Any]]) -> int:
    """One values:append request. Returns the number of rows Google says it updated; raises SheetsError."""
    email = creds["client_email"]
    for attempt in (0, 1):
        token = _access_token(agent_id, creds)
        url = (f"{SHEETS_API}/{quote(spreadsheet_id, safe='')}/values/{quote(sheets_range(sheet_name), safe='')}:append"
               "?valueInputOption=RAW&insertDataOption=INSERT_ROWS")
        try:
            status, body = _google_http("POST", url, headers={"Authorization": f"Bearer {token}"},
                                        json_body={"majorDimension": "ROWS", "values": rows})
        except Exception as e:  # noqa: BLE001
            raise SheetsError(_clean_error(f"Could not reach Google Sheets: {e.__class__.__name__}"),
                              retryable=True) from e
        if status == 401 and attempt == 0:  # the cached token expired early: sign in again once
            _token_cache.pop(agent_id, None)
            continue
        if status == 200:
            updates = body.get("updates", {}) if isinstance(body, dict) else {}
            return int(updates.get("updatedRows") or len(rows))
        raise _sheets_error(status, body, email, 30.0 if status == 429 else None)
    raise SheetsError("Google Sheets did not accept the sign-in.", status=401, hint="Check the service-account key.")


def _sheets_pace(email: str) -> float:
    """Seconds to wait before another append is allowed for this service account (0: go now); counts the append."""
    now = _now()
    with _sheets_lock:
        q = _sheets_times.setdefault(email, collections.deque())
        while q and q[0] <= now - 60:
            q.popleft()
        if len(q) >= SHEETS_APPENDS_PER_MIN:
            return max(0.5, q[0] + 60 - now)
        q.append(now)
        return 0.0


def sheet_row(payload: dict, agent_name: str) -> List[Any]:
    b = payload.get("booking") or {}
    return [payload.get("created_at") or _iso(), agent_name, payload.get("language") or "",
            payload.get("caller_text") or "", payload.get("agent_text") or "", b.get("booking_id") or "",
            b.get("date") or "", b.get("time") or "", payload.get("receipt_id") or ""]


def _process_sheets(d: Dispatcher, batch: List[Job]) -> None:
    first = batch[0]
    creds = load_secrets(first.agent_id, local=first.local, enc=first.enc).get("service_account")
    if not creds:
        for j in batch:
            log_update(j.agent_id, j.user_id, j.local,
                       _entry(j, "failed", error="no Google service-account key is stored"))
        return
    wait = _sheets_pace(creds["client_email"])
    if wait:  # Google's per-minute write limit: wait, without using up a retry
        for j in batch:
            if j.rate_waits >= 20 or not d.submit(j, wait):
                log_update(j.agent_id, j.user_id, j.local, _entry(j, "failed", error="Sheets write limit reached"))
            else:
                j.rate_waits += 1
        return
    for j in batch:
        if j.payload.get("receipt_id") is None and j.payload.get("turn_id"):
            j.payload["receipt_id"] = _find_receipt_id(j.payload["turn_id"])
    rows = [sheet_row(j.payload, j.agent_name) for j in batch]
    started = time.time()
    try:
        append_rows(first.agent_id, creds, first.sheet["spreadsheet_id"], first.sheet.get("sheet_name", ""), rows)
    except SheetsError as e:
        ms = int((time.time() - started) * 1000)
        for j in batch:
            _retry_or_fail(d, j, code=e.status, latency_ms=ms, error=e.message + (f" {e.hint}" if e.hint else ""),
                           retryable=e.retryable, wait=e.wait)
        return
    ms = int((time.time() - started) * 1000)
    for j in batch:
        log_update(j.agent_id, j.user_id, j.local, _entry(j, "delivered", code=200, latency_ms=ms))


def check_sheets_connection(agent_id: str, user_id: str, local: bool, enc: Optional[str], cfg: dict,
                            agent_name: str) -> dict:
    """The "Test connection" button: append one clearly marked row and report exactly what Google answered. Blocking."""
    sh = cfg["sheets"]
    creds = load_secrets(agent_id, local=local, enc=enc).get("service_account")
    job = Job("sheets", agent_id, user_id, local, enc, TEST_EVENT, {}, agent_name)
    if not creds or not sh["spreadsheet_id"]:
        return {"ok": False, "error": "Add the service-account key and the spreadsheet first.", "hint": "",
                "status": None}
    row = [_iso(), agent_name, "en", "BonClay test row", "You can delete this row.", "", "", "", ""]
    started = time.time()
    try:
        append_rows(agent_id, creds, sh["spreadsheet_id"], sh["sheet_name"], [row])
    except SheetsError as e:
        log_update(agent_id, user_id, local, _entry(job, "failed", code=e.status,
                                                    latency_ms=int((time.time() - started) * 1000), error=e.message))
        return {"ok": False, "error": e.message, "hint": e.hint, "status": e.status}
    log_update(agent_id, user_id, local, _entry(job, "delivered", code=200,
                                                latency_ms=int((time.time() - started) * 1000)))
    return {"ok": True, "error": None, "hint": "", "status": 200}


# ----------------------------------------------------------------------------------------------- the hook

def dispatch_turn(run: Any, turn_id: str, agent_text: str, latency_ms: int) -> None:
    """Called once per stored turn (from backend._turn_store). Builds the events and queues them; returns at once and
    never raises. An agent without integrations costs one dict lookup."""
    try:
        agent = getattr(run, "agent", None) or {}
        raw = agent.get("integrations")
        if not raw:
            return
        cfg = normalize_cfg(raw)
        wh, sh = cfg["webhook"], cfg["sheets"]
        want_wh = bool(wh["enabled"] and wh["url"] and wh["secret_set"])
        want_sh = bool(sh["enabled"] and sh["spreadsheet_id"] and sh["credentials_set"])
        if not (want_wh or want_sh):
            return
        is_test = bool(getattr(run, "is_test", False))
        enc = raw.get("secrets_enc") if isinstance(raw, dict) else None
        local, user_id, agent_id = bool(getattr(run, "local", True)), str(run.user_id), str(run.agent_id)
        name = str(agent.get("name") or "")
        base = turn_payload(run, turn_id, agent_text, latency_ms)
        if want_wh and (wh["include_test"] or not is_test):
            payload = redact_payload(base) if wh["redact_phones"] else base
            events: List[Tuple[str, dict]] = []
            if "tool.called" in wh["events"]:
                calls = list(getattr(run, "tool_objs", None) or [])
                for i, call in enumerate(calls):
                    if getattr(call, "ok", False):
                        booked = isinstance(call.value, dict) and bool(call.value.get("booking_id"))
                        events.append(("tool.called", {**payload, "tool_calls": [payload["tool_calls"][i]],
                                                       "booking": payload["booking"] if booked else None}))
            if "turn.completed" in wh["events"]:
                events.append(("turn.completed", payload))
            for event, body in events:
                if not dispatcher.submit(Job("webhook", agent_id, user_id, local, enc, event, dict(body), name,
                                             url=wh["url"])):
                    log.warning("integrations queue is full: dropped a %s event", event)
        if want_sh and (sh["include_test"] or not is_test):
            payload = redact_payload(base) if sh["redact_phones"] else dict(base)
            job = Job("sheets", agent_id, user_id, local, enc, "turn.completed", payload, name,
                      sheet={"spreadsheet_id": sh["spreadsheet_id"], "sheet_name": sh["sheet_name"]})
            if not dispatcher.submit(job):
                log.warning("integrations queue is full: dropped a Sheets row")
    except Exception as e:  # noqa: BLE001 - a turn must never fail because of an integration
        log.warning("could not queue integration events: %s: %s", e.__class__.__name__, e)
