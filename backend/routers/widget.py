"""Website widget: one script tag puts an BonClay agent on a business's own site.

    <script src="https://<bonclay-host>/widget/v1/bonclay-agent.js" defer></script>
    <bonclay-agent agent="PUBLIC_WIDGET_KEY"></bonclay-agent>

Owner side (normal auth, the agent must be the caller's):
    GET    /api/agents/{id}/widget          the widget settings of an agent (key, allowed origins, limits)
    PUT    /api/agents/{id}/widget          publish / unpublish, edit origins, limits and the business name
    POST   /api/agents/{id}/widget/rotate   a new key; the old one stops working at once
    DELETE /api/agents/{id}/widget          remove the widget and its key

Public side (no owner auth; the website visitor is anonymous):
    GET  /widget/v1/bonclay-agent.js        the Web Component (static, cacheable)
    POST /api/widget/session                {key, previous?} -> a signed 10 minute session token
    POST /api/widget/turn/stream            multipart `audio` + the session token -> the same SSE events as the owner's
                                            /api/conversations/{agent}/turn/stream (the same internal function runs it)
    GET  /api/widget/clip/{turn}/{index}    one sentence clip of a turn that session started

Security model:
  * The widget key is public (it sits in page HTML). It names an agent, nothing else. The owner can switch it off or
    replace it (rotate) at any time; every turn re-reads the agent, so a revoked key stops working immediately.
  * The browser's `Origin` header must be exactly one of the owner's allowed origins (scheme + host + port; no
    wildcards; localhost only when listed). Session tokens are HMAC-signed and bound to key + origin + a random
    visitor id; a turn must come from the same origin. Non-browser clients can fake `Origin`, so the limits below are
    what protects the owner's quota.
  * CORS headers are sent only for the origin a request was validated for (never `*`, never the dashboard's list).
  * Rate limits (in memory, per process): per visitor IP and per visitor id per hour, per key per day, and a cap on
    sessions per IP. 429 carries Retry-After.
  * The token only allows turns for its agent. Visitors cannot choose a conversation: the server keeps one per
    visitor. Replies are cut down to what the page needs (no receipts, timings, models, retrieval or tool details).
  * Turns run as the agent's owner with conversations marked source="widget".
"""
import asyncio
import base64
import collections
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.datastructures import MutableHeaders

try:
    from core.errors import AppError, error_response
    from core.validation import sanitize_text
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover - imported as backend.routers.widget
    from backend.core.errors import AppError, error_response
    from backend.core.validation import sanitize_text
    from backend.services.auth import get_current_user_id

log = logging.getLogger("bonclay.widget")

router = APIRouter(tags=["widget"])

# ----------------------------------------------------------------------------------------------------- settings

SESSION_TTL_S = 600
RENEW_GRACE_S = 24 * 3600          # an expired token still proves who the visitor was for this long
DEFAULT_PER_VISITOR_HOUR = 30
DEFAULT_PER_KEY_DAY = 300
MAX_PER_VISITOR_HOUR = 600
MAX_PER_KEY_DAY = 10000
SESSIONS_PER_IP_HOUR = 60          # per key and IP
SESSIONS_PER_IP_ALL_KEYS_HOUR = 120
MAX_ORIGINS = 20
MAX_UPLOAD_BYTES = 6 * 1024 * 1024  # a 30 s recording is well under 1 MB
KEY_PREFIX = "awk_"
KEY_RE = re.compile(r"^awk_[A-Za-z0-9_-]{16,64}$")
SCRIPT_PATH = "/widget/v1/bonclay-agent.js"
SCRIPT_FILE = Path(__file__).resolve().parent.parent / "widget" / "v1" / "bonclay-agent.js"
FORBIDDEN = ("WIDGET_FORBIDDEN", "This assistant is not available on this site.", 403)


def _now() -> float:
    return time.time()


def _b():
    """The backend module (imported lazily so this router never creates an import cycle)."""
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "local_db"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "local_db"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


# --------------------------------------------------------------------------------------------------- origins

_DEFAULT_PORTS = {"http": 80, "https": 443}
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def normalize_origin(value: Optional[str], *, lenient: bool = False) -> Optional[str]:
    """`scheme://host[:port]`, lower case, default port dropped; None when it is not a plain origin.

    `lenient` (owner input) also accepts a bare host ("example.com" -> https, "localhost:3000" -> http) and a
    trailing slash. The browser's own `Origin` header is parsed strictly. Paths, queries, credentials and wildcards
    are never accepted."""
    if not value:
        return None
    text = value.strip()
    if not text or text.lower() == "null" or "*" in text or len(text) > 200:
        return None
    if "://" not in text:
        if not lenient:
            return None
        host_part = text.split("/", 1)[0].split(":", 1)[0].lower()
        text = ("http://" if host_part in ("localhost", "127.0.0.1") else "https://") + text
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return None
    scheme = (parts.scheme or "").lower()
    host = (parts.hostname or "").lower()
    if scheme not in ("http", "https") or not host or parts.username or parts.password:
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        return None
    if not re.fullmatch(r"[a-z0-9.\-]+|\[[0-9a-f:]+\]", host if ":" not in host else f"[{host}]"):
        return None
    shown = f"[{host}]" if ":" in host else host
    if port is not None and port != _DEFAULT_PORTS[scheme]:
        shown += f":{port}"
    return f"{scheme}://{shown}"


def _clean_origins(values: List[str]) -> List[str]:
    if len(values) > MAX_ORIGINS:
        raise AppError("TOO_MANY_ORIGINS", f"At most {MAX_ORIGINS} allowed sites.", 422, {"field": "origins"})
    out: List[str] = []
    for raw in values:
        origin = normalize_origin(raw, lenient=True)
        if origin is None:
            raise AppError("INVALID_ORIGIN", f"'{str(raw)[:80]}' is not a site address. Use https://example.com "
                           "(no path, no wildcard).", 422, {"field": "origins", "value": str(raw)[:80]})
        if origin not in out:
            out.append(origin)
    return out


def origin_allowed(origin: Optional[str], allowed: List[str]) -> bool:
    """Exact match only. localhost is allowed only because the owner listed it."""
    norm = normalize_origin(origin)
    return norm is not None and norm in {normalize_origin(a) for a in allowed or []}


# ----------------------------------------------------------------------------------------------- session tokens

_secret_lock = threading.Lock()
_secret_cache: Optional[bytes] = None


def _secret() -> bytes:
    """WIDGET_TOKEN_SECRET, else a random secret created on first use and kept in data/widget_secret (mode 600)."""
    global _secret_cache
    env = os.getenv("WIDGET_TOKEN_SECRET", "").strip()
    if env:
        return env.encode("utf-8")
    with _secret_lock:
        if _secret_cache is not None:
            return _secret_cache
        data_dir = getattr(_b(), "LOCAL_DATA_DIR", None) or os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data")
        path = os.path.join(data_dir, "widget_secret")
        value = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = f.read().strip()
        except OSError:
            pass
        if not value:
            value = secrets.token_hex(32)
            try:
                os.makedirs(data_dir, exist_ok=True)
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(value)
            except FileExistsError:
                with open(path, "r", encoding="utf-8") as f:
                    value = f.read().strip()
            except OSError as e:
                log.warning("could not persist the widget secret (%s); sessions will not survive a restart", e)
        _secret_cache = value.encode("utf-8")
        return _secret_cache


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _mac(payload: str) -> str:
    return _b64(hmac.new(_secret(), b"bonclay-widget-session-v1." + payload.encode("ascii"), hashlib.sha256).digest())


def sign_session(claims: Dict[str, Any]) -> str:
    payload = _b64(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{payload}.{_mac(payload)}"


def read_session(token: Optional[str], *, allow_expired_for: float = 0.0) -> Dict[str, Any]:
    """The claims of a validly signed token. Raises AppError WIDGET_SESSION_INVALID (bad / forged) or
    WIDGET_SESSION_EXPIRED. `allow_expired_for` accepts tokens expired less than that many seconds ago."""
    try:
        payload, mac = (token or "").split(".")
        if not hmac.compare_digest(mac, _mac(payload)):
            raise ValueError("signature")
        claims = json.loads(_unb64(payload))
        if not isinstance(claims, dict) or not all(k in claims for k in ("k", "a", "m", "o", "v", "exp")):
            raise ValueError("claims")
    except Exception:  # noqa: BLE001 - anything malformed is just an invalid token
        raise AppError("WIDGET_SESSION_INVALID", "This session is not valid. Start again.", 401) from None
    if float(claims["exp"]) + allow_expired_for < _now():
        err = AppError("WIDGET_SESSION_EXPIRED", "This session has expired. Start again.", 401)
        err.claims = claims  # for the caller only: never part of the response
        raise err
    return claims


# ------------------------------------------------------------------------------------------------- rate limits

class SlidingLimiter:
    """Counts hits in a sliding window per bucket, in memory. `consume` checks several buckets at once and records
    a hit in all of them only when none is over its limit (so a refused turn is not charged)."""

    def __init__(self, max_buckets: int = 50000):
        self._hits: Dict[str, "collections.deque[float]"] = {}
        self._lock = threading.Lock()
        self._max = max_buckets

    def consume(self, rules: List[Tuple[str, int, int]]) -> int:
        """rules: (bucket, limit, window seconds). Returns 0 when allowed, else seconds until a slot frees up."""
        now = _now()
        with self._lock:
            wait = 0
            for bucket, limit, window in rules:
                q = self._hits.get(bucket)
                if q is None:
                    continue
                while q and q[0] <= now - window:
                    q.popleft()
                if len(q) >= limit:
                    wait = max(wait, int(q[0] + window - now) + 1)
            if wait:
                return wait
            if len(self._hits) > self._max:
                self._prune(now)
            for bucket, _limit, window in rules:
                self._hits.setdefault(bucket, collections.deque()).append(now)
            return 0

    def _prune(self, now: float) -> None:
        for bucket in [b for b, q in self._hits.items() if not q or q[-1] <= now - 86400]:
            del self._hits[bucket]
        while len(self._hits) > self._max:  # still too many: forget the oldest buckets
            del self._hits[next(iter(self._hits))]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingLimiter()
_TURN_OWNERS: "collections.OrderedDict[str, str]" = collections.OrderedDict()  # turn id -> visitor id
_VISITOR_CONV: "collections.OrderedDict[str, str]" = collections.OrderedDict()  # visitor id -> conversation id


def _remember(store: "collections.OrderedDict[str, str]", key: str, value: str, cap: int = 4000) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > cap:
        store.popitem(last=False)


# --------------------------------------------------------------------------------------------------- storage

def _widget_cfg(row: dict) -> dict:
    """The agent's widget settings with defaults filled in."""
    cfg = row.get("widget") if isinstance(row.get("widget"), dict) else {}
    return {
        "enabled": bool(cfg.get("enabled")) and bool(row.get("widget_key")),
        "origins": [str(o) for o in (cfg.get("origins") or [])],
        "per_visitor_hour": int(cfg.get("per_visitor_hour") or DEFAULT_PER_VISITOR_HOUR),
        "per_key_day": int(cfg.get("per_key_day") or DEFAULT_PER_KEY_DAY),
        "business_name": str(cfg.get("business_name") or ""),
        "created_at": cfg.get("created_at"),
    }


def _owner_json(row: dict) -> dict:
    cfg = _widget_cfg(row)
    return {**cfg, "key": row.get("widget_key") or None, "script_path": SCRIPT_PATH}


async def _find_agent_by_key(key: str) -> Optional[Tuple[str, dict]]:
    """(storage mode, agent row) for a widget key, or None. Looks in the local store and, when configured, Supabase."""
    if not key or not KEY_RE.match(key):
        return None
    b = _b()
    rows = await asyncio.to_thread(b.local_db.select, "agents", lambda r: r.get("widget_key") == key)
    if rows:
        return "local", rows[0]
    if b.supabase is not None:
        try:
            res = await asyncio.to_thread(
                lambda: b.supabase.table("agents").select("*").eq("widget_key", key).limit(1).execute())
            if res.data:
                return "cloud", res.data[0]
        except Exception as e:  # noqa: BLE001 - migration 013 not applied, or the database is down
            log.warning("widget key lookup in Supabase failed (apply migration 013_agent_widget.sql?): %s", e)
    return None


async def _load_agent_row(mode: str, agent_id: str) -> Optional[dict]:
    b = _b()
    if mode == "local":
        return await asyncio.to_thread(b.local_db.get, "agents", agent_id)
    if b.supabase is None:
        return None
    try:
        res = await asyncio.to_thread(
            lambda: b.supabase.table("agents").select("*").eq("id", agent_id).limit(1).execute())
        return res.data[0] if res.data else None
    except Exception as e:  # noqa: BLE001
        log.warning("widget agent lookup failed: %s", e)
        return None


async def _save_widget(request: Request, agent_id: str, user_id: str, cfg: Optional[dict], key: Optional[str]) -> dict:
    """Write the widget settings and key of an owned agent (local file or Supabase); returns the updated row."""
    b = _b()
    fields = {"widget": cfg, "widget_key": key}
    if b._storage_mode(request) == "local":
        row = await asyncio.to_thread(b.local_db.update, "agents", agent_id, fields, user_id)
        if row is None:
            raise AppError("AGENT_NOT_FOUND", "Agent not found", 404)
        return row
    try:
        await asyncio.to_thread(lambda: b.supabase.table("agents").update(fields).eq("id", agent_id)
                                .eq("user_id", user_id).execute())
    except Exception as e:  # noqa: BLE001
        log.warning("saving widget settings failed: %s", e)
        raise AppError("DB_ERROR", "Could not save the widget settings. Apply "
                       "backend/supabase/migrations/013_agent_widget.sql and try again.", 500) from e
    return await b._owned_agent(agent_id, user_id)


def _new_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(18)


# --------------------------------------------------------------------------------------------- owner endpoints

class WidgetUpdate(BaseModel):
    enabled: Optional[bool] = None
    origins: Optional[List[str]] = None
    per_visitor_hour: Optional[int] = Field(None, ge=1, le=MAX_PER_VISITOR_HOUR)
    per_key_day: Optional[int] = Field(None, ge=1, le=MAX_PER_KEY_DAY)
    business_name: Optional[str] = None


@router.get("/api/agents/{agent_id}/widget")
async def get_widget(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    row = await _b()._load_agent(request, agent_id, user_id)
    return _owner_json(row)


@router.put("/api/agents/{agent_id}/widget")
async def put_widget(agent_id: str, body: WidgetUpdate, request: Request,
                     user_id: str = Depends(get_current_user_id)):
    """Publish (creates the key on first use), unpublish (the key is kept, but rejected), change origins and limits."""
    row = await _b()._load_agent(request, agent_id, user_id)
    cfg = _widget_cfg(row)
    key = row.get("widget_key") or None
    sent = body.model_fields_set
    if "origins" in sent and body.origins is not None:
        cfg["origins"] = _clean_origins(body.origins)
    if "per_visitor_hour" in sent and body.per_visitor_hour is not None:
        cfg["per_visitor_hour"] = body.per_visitor_hour
    if "per_key_day" in sent and body.per_key_day is not None:
        cfg["per_key_day"] = body.per_key_day
    if "business_name" in sent and body.business_name is not None:
        cfg["business_name"] = sanitize_text(body.business_name, "business_name", 80, required=False)
    if "enabled" in sent and body.enabled is not None:
        cfg["enabled"] = bool(body.enabled)
        if body.enabled and not key:
            key = _new_key()
            cfg["created_at"] = datetime.now(timezone.utc).isoformat()
    if cfg["enabled"] and not cfg["origins"]:
        raise AppError("ORIGINS_REQUIRED", "Add at least one allowed site (for example https://example.com) "
                       "before publishing.", 422, {"field": "origins"})
    saved = await _save_widget(request, agent_id, user_id, {k: v for k, v in cfg.items()}, key)
    return _owner_json(saved)


@router.post("/api/agents/{agent_id}/widget/rotate")
async def rotate_widget(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    """Replace the key. Pages that still carry the old key stop working immediately."""
    row = await _b()._load_agent(request, agent_id, user_id)
    cfg = _widget_cfg(row)
    if not row.get("widget_key"):
        raise AppError("WIDGET_NOT_PUBLISHED", "Publish the widget first.", 409)
    cfg["created_at"] = datetime.now(timezone.utc).isoformat()
    saved = await _save_widget(request, agent_id, user_id, cfg, _new_key())
    return _owner_json(saved)


@router.delete("/api/agents/{agent_id}/widget")
async def delete_widget(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    await _b()._load_agent(request, agent_id, user_id)
    saved = await _save_widget(request, agent_id, user_id, None, None)
    return _owner_json(saved)


# ---------------------------------------------------------------------------------------------- public: script

@router.api_route(SCRIPT_PATH, methods=["GET", "HEAD"], include_in_schema=False)
async def widget_script():
    if not SCRIPT_FILE.is_file():
        raise AppError("NOT_FOUND", "Widget bundle not found", 404)
    return FileResponse(SCRIPT_FILE, media_type="application/javascript; charset=utf-8", headers={
        "Cache-Control": "public, max-age=300", "X-Content-Type-Options": "nosniff",
        "Cross-Origin-Resource-Policy": "cross-origin"})


# ---------------------------------------------------------------------------------------------- public: session

class SessionRequest(BaseModel):
    key: str = Field(..., max_length=100)
    previous: Optional[str] = Field(None, max_length=2000)  # an earlier token: keeps the visitor (and conversation)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _request_origin(request: Request) -> Optional[str]:
    return normalize_origin(request.headers.get("origin"))


def _allow_cors(request: Request, origin: Optional[str]) -> None:
    """Mark this request as validated for `origin`: the CORS middleware then echoes exactly that origin."""
    if origin:
        request.scope.setdefault("state", {})["widget_cors_origin"] = origin


def _too_many(wait: int, what: str) -> JSONResponse:
    return error_response("WIDGET_RATE_LIMITED", f"Too many {what} right now. Try again in a few minutes.", 429,
                          {"retry_after": wait}, headers={"Retry-After": str(wait)})


@router.post("/api/widget/session")
async def widget_session(body: SessionRequest, request: Request):
    """A short-lived session token for the page that sent this request. Origin must be on the key's allowlist."""
    origin = _request_origin(request)
    ip = _client_ip(request)
    wait = limiter.consume([(f"sess:{body.key[:12]}:{ip}", SESSIONS_PER_IP_HOUR, 3600),
                            (f"sess:{ip}", SESSIONS_PER_IP_ALL_KEYS_HOUR, 3600)])
    if wait:
        _allow_cors(request, origin)  # (a rate-limited page still gets to read why)
        return _too_many(wait, "visitors")
    found = await _find_agent_by_key(body.key)
    if found is None or origin is None:
        raise AppError(*FORBIDDEN)
    mode, agent = found
    cfg = _widget_cfg(agent)
    if not cfg["enabled"] or not origin_allowed(origin, cfg["origins"]):
        raise AppError(*FORBIDDEN)
    _allow_cors(request, origin)
    visitor = None
    if body.previous:
        try:
            old = read_session(body.previous, allow_expired_for=RENEW_GRACE_S)
            if old["k"] == body.key and old["o"] == origin:
                visitor = str(old["v"])
        except AppError:
            pass
    visitor = visitor or secrets.token_hex(12)
    now = int(_now())
    token = sign_session({"k": body.key, "a": agent["id"], "m": mode, "o": origin, "v": visitor, "iat": now,
                          "exp": now + SESSION_TTL_S})
    return {"token": token, "expires_in": SESSION_TTL_S, "agent": {
        "name": str(agent.get("name") or "Assistant"),
        "business_name": cfg["business_name"] or str(agent.get("name") or ""),
        "greeting": str(agent.get("greeting") or "")}}


def _bearer(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    return auth.split(" ", 1)[1].strip() if auth.lower().startswith("bearer ") else None


def _check_session(request: Request, token: Optional[str]) -> Dict[str, Any]:
    """Verified claims for a request from the page the session was issued to. CORS is allowed once the signature is
    good and the Origin is the token's, so even a 'session expired' answer can be read by the widget."""
    try:
        claims = read_session(token)
    except AppError as e:
        claims = getattr(e, "claims", None)
        if claims and claims.get("o") == _request_origin(request):
            _allow_cors(request, claims["o"])
        raise
    origin = _request_origin(request)
    if origin is None or origin != claims["o"]:
        raise AppError(*FORBIDDEN)
    _allow_cors(request, origin)
    return claims


# ------------------------------------------------------------------------------------------------ public: turn

_SAFE_ERRORS = {
    "NO_SPEECH": "I did not catch that. Please try again.",
    "ASR_UNAVAILABLE": "The assistant cannot hear right now. Please try again later.",
    "TURN_FAILED": "Something went wrong. Please try again.",
    "AGENT_NOT_FOUND": "This assistant is not available.",
}
_CLIP_URL_RE = re.compile(r"^/api/conversations/turns/([0-9a-f-]{36})/sentences/(\d+)/audio$")
_DONE_FIELDS = ("turn_id", "user_text", "agent_text", "language", "fallback", "interrupted")


def _event_parts(chunk: bytes) -> Optional[Tuple[str, dict]]:
    try:
        head, _, rest = chunk.decode("utf-8").partition("\n")
        if not head.startswith("event: ") or not rest.startswith("data: "):
            return None
        return head[7:].strip(), json.loads(rest[6:].strip())
    except Exception:  # noqa: BLE001
        return None


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def clean_event(chunk: bytes, visitor: str, clip_base: str = "/api/widget/clip") -> bytes:
    """One server-sent event of the internal turn, cut down for a website visitor. `clip_base` is where the visitor
    fetches a local clip from (the web call link uses its own, link-scoped route)."""
    parts = _event_parts(chunk)
    if parts is None:
        return b""  # anything unexpected is not forwarded
    name, data = parts
    if name == "transcript":
        return _sse(name, {"user_text": data.get("user_text", ""), "language": data.get("language")})
    if name == "sentence":
        url = str(data.get("audio_url") or "")
        match = _CLIP_URL_RE.match(url)
        if match:  # a local clip: served by this router's session-checked route
            _remember(_TURN_OWNERS, match.group(1), visitor)
            url = f"{clip_base}/{match.group(1)}/{match.group(2)}"
        elif not url.startswith("data:audio/"):
            url = ""
        out = {"index": data.get("index"), "text": data.get("text", ""), "audio_url": url}
        if data.get("filler"):
            out["filler"] = True
        return _sse(name, out)
    if name == "done":
        if data.get("conversation_id"):
            _remember(_VISITOR_CONV, visitor, str(data["conversation_id"]))
        return _sse(name, {k: data.get(k) for k in _DONE_FIELDS})
    if name == "error":
        code = str(data.get("code") or "TURN_FAILED")
        return _sse(name, {"code": code, "message": _SAFE_ERRORS.get(code, _SAFE_ERRORS["TURN_FAILED"])})
    return b""


async def _visitor_events(body_iterator, visitor: str, clip_base: str = "/api/widget/clip"):
    try:
        async for chunk in body_iterator:
            piece = clean_event(chunk, visitor, clip_base)
            if piece:
                yield piece
    finally:
        close = getattr(body_iterator, "aclose", None)
        if close is not None:
            await close()


@router.post("/api/widget/turn/stream")
async def widget_turn_stream(request: Request, audio: UploadFile = File(...), session: str = Form(None)):
    """One spoken turn from a website visitor. The pipeline is the owner's streaming turn (`_stream_turn_response`);
    this route checks the session, applies the limits, and trims what goes back."""
    claims = _check_session(request, _bearer(request) or session)
    key, visitor = claims["k"], claims["v"]
    found = await _load_agent_row(claims["m"], claims["a"])
    cfg = _widget_cfg(found) if found else None
    if found is None or not cfg["enabled"] or found.get("widget_key") != key \
            or not origin_allowed(claims["o"], cfg["origins"]):
        raise AppError("WIDGET_SESSION_INVALID", "This session is not valid. Start again.", 401)
    wait = limiter.consume([(f"turn-ip:{key}:{_client_ip(request)}", cfg["per_visitor_hour"], 3600),
                            (f"turn-v:{key}:{visitor}", cfg["per_visitor_hour"], 3600),
                            (f"turn-key:{key}", cfg["per_key_day"], 86400)])
    if wait:
        return _too_many(wait, "questions")
    if audio.size is not None and audio.size > MAX_UPLOAD_BYTES:
        raise AppError("FILE_TOO_LARGE", "That recording is too long.", 413)
    b = _b()
    run = await b._turn_begin(request, found["id"], found["user_id"], storage_mode=claims["m"], agent=found,
                              source="widget")
    response = await b._stream_turn_response(run, audio, _VISITOR_CONV.get(visitor))
    response.body_iterator = _visitor_events(response.body_iterator, visitor)
    return response


@router.get("/api/widget/clip/{turn_id}/{index}")
async def widget_clip(turn_id: str, index: int, request: Request):
    """One sentence clip of a turn started by the caller's own session (token in the Authorization header)."""
    claims = _check_session(request, _bearer(request))
    b = _b()
    b._check_id(turn_id, "turn id")
    if claims["m"] != "local" or index < 0 or index > 999 or _TURN_OWNERS.get(turn_id) != claims["v"]:
        raise AppError("NOT_FOUND", "Clip not found", 404)
    path = os.path.join(b.LOCAL_AUDIO_DIR, f"turn_{turn_id}_{index}.wav")
    if not os.path.isfile(path):
        raise AppError("NOT_FOUND", "Clip not found", 404)
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------------------------------------------- CORS

class WidgetCORSMiddleware:
    """CORS for /api/widget/* only. The app-wide CORS list (the dashboard) never applies to these routes; a response
    gets `Access-Control-Allow-Origin` only for the origin its route validated against the key's allowlist.

    A preflight cannot carry the key, so it is answered for any origin: it only says which methods and headers the
    real request may use. The real request is still checked, and without an allow header the browser will not let the
    page read the answer."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/widget/"):
            return await self.app(scope, receive, send)
        headers = MutableHeaders(scope=scope)
        origin = normalize_origin(headers.get("origin"))
        if scope["method"] == "OPTIONS" and origin and headers.get("access-control-request-method"):
            res = Response(status_code=204, headers={
                "Access-Control-Allow-Origin": origin, "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Authorization, Content-Type", "Access-Control-Max-Age": "600",
                "Vary": "Origin"})
            return await res(scope, receive, send)
        length = headers.get("content-length")
        if scope["method"] == "POST" and length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 65536:
            res = error_response("FILE_TOO_LARGE", "That recording is too long.", 413)
            return await res(scope, receive, send)
        scope.setdefault("state", {})

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                out = MutableHeaders(scope=message)
                for name in [h for h in out.keys() if h.lower().startswith("access-control-")]:
                    del out[name]  # whatever the app-wide CORS middleware added does not apply here
                allowed = scope["state"].get("widget_cors_origin")
                if allowed:
                    out["Access-Control-Allow-Origin"] = allowed
                    out["Access-Control-Expose-Headers"] = "Retry-After"
                out.append("Vary", "Origin")
            await send(message)

        await self.app(scope, receive, send_wrapper)


def install(app) -> None:
    """Called by backend.py after the router is included: adds the widget CORS rules (outermost middleware)."""
    app.add_middleware(WidgetCORSMiddleware)
