"""Web call link: a shareable page where anyone talks to one agent by voice in their browser. No phone network, no
KYC, no login, nothing to install.

Owner side (main API, normal auth; the agent must be the caller's):
    GET    /api/agents/{id}/web-call-links                 the agent's links with their counters, and the public base URL
    POST   /api/agents/{id}/web-call-links                 create a link (expiry, session length, minutes per day / visitor)
    POST   /api/agents/{id}/web-call-links/{link}/revoke   switch a link off at once (its record and counters stay)
    DELETE /api/agents/{id}/web-call-links/{link}          remove a link record

Public side (no login; the visitor is anonymous). These routes, and only these, are added to the phone-only public app
(phone_public.py), which is what the tunnel exposes; the main API is never reachable from the internet:
    GET  /call/{link}                    the page (HTML; the script and stylesheet are same-origin files)
    GET  /call/assets/call.js|call.css   the bundle (static, cacheable)
    POST /call/api/{link}/session        a signed session token for this page (same origin only)
    POST /call/api/{link}/turn           one spoken turn: multipart `audio`; the same SSE events the website widget sends
    GET  /call/api/{link}/clip/{turn}/{n} one sentence clip of a turn this session started
    POST /call/api/{link}/end            the visitor hung up (settles the minutes used)

Security model:
  * The link id (`awc_` + 192 random bits) is a bearer capability for ONE agent's conversation and nothing else: it
    names the link, the link names the agent, and no route takes an agent, conversation or turn id from the visitor.
    The link is re-read on every request, so revoking, expiry or deleting the agent takes effect on the next turn.
  * Session tokens are the website widget's HMAC tokens (routers/widget.py), bound to link + origin + a random
    visitor id; a token made for a link is refused on any other link, and the widget routes refuse it too.
  * POSTs must come from the page's own origin (the Origin header must name the Host the request arrived for, or the
    PUBLIC_BASE_URL host). No CORS header is ever sent, so a page on another site cannot read an answer. Non-browser
    clients can fake Origin: the limits below protect the owner's quota.
  * Limits (in memory per process, except the daily total that is kept with the link): page and session starts per IP,
    turns per visitor and per IP, a session length cap (default 10 minutes, inside the signed token), minutes per day
    for the link (default 60) and, when the owner sets it, minutes per visitor per day. Minutes are server-side time
    between a visitor's turns (at least 5 s and at most 60 s per turn), not a client's claim.
  * The visitor's IP is the connection's peer, or, when that peer is this computer (the tunnel), the client address the
    tunnel reports (CF-Connecting-IP, else the last X-Forwarded-For entry).
  * Visitors get what the widget's visitors get (transcript, sentence text and clips, final text): no receipts, models,
    retrieval, tool calls, agent id or owner id. Turns run as the agent's owner and are stored with source="widget".
  * The page is sent with a strict Content-Security-Policy (same-origin script, style and fetch only, no framing), no
    referrer and no caching of anything per link.
"""
import asyncio
import collections
import html
import ipaddress
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

try:
    from core.errors import AppError, error_response
    from core.validation import sanitize_text
    from routers import widget
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover - imported as backend.routers.web_call
    from backend.core.errors import AppError, error_response
    from backend.core.validation import sanitize_text
    from backend.routers import widget
    from backend.services.auth import get_current_user_id

log = logging.getLogger("bonclay.webcall")

router = APIRouter(tags=["web-call"])         # owner routes: included in the main API
public_router = APIRouter(tags=["web-call-public"])  # visitor routes: included ONLY in the phone-only public app

# ----------------------------------------------------------------------------------------------------- settings

LINK_PREFIX = "awc_"
LINK_RE = re.compile(r"^awc_[A-Za-z0-9_-]{24,64}$")
DEFAULT_SESSION_MIN = 10
MAX_SESSION_MIN = 30
DEFAULT_DAY_MIN = 60
MAX_DAY_MIN = 1440
MAX_LINKS_PER_AGENT = 20
EXPIRIES = {"1d": 86400, "7d": 7 * 86400, "never": None}
DEFAULT_EXPIRY = "7d"
MIN_TURN_CHARGE_S = 5.0     # every turn costs at least this much of the daily minutes (so rapid-fire turns run out)
MAX_TURN_CHARGE_S = 60.0    # a long gap between turns (silence) is not counted beyond this
HITS_PER_IP_MINUTE = 90
SESSIONS_PER_IP_LINK_HOUR = 10
SESSIONS_PER_IP_HOUR = 30
SESSIONS_PER_LINK_DAY = 300
TURNS_PER_VISITOR_MINUTE = 20
TURNS_PER_IP_HOUR = 300
MAX_UPLOAD_BYTES = widget.MAX_UPLOAD_BYTES
ASSET_DIR = Path(__file__).resolve().parent.parent / "webcall" / "v1"
FOOTER_VERIFY = "bonclay.ayushchougula.in/verify"
SAFE_ERRORS = widget._SAFE_ERRORS

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob: data:; "
       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}

limiter = widget.SlidingLimiter()
_LAST_SEEN: "collections.OrderedDict[str, float]" = collections.OrderedDict()      # session id -> last server activity
_VISITOR_USED: "collections.OrderedDict[str, float]" = collections.OrderedDict()   # link:ip:day -> seconds used


def _now() -> float:
    return widget._now()


def _b():
    return widget._b()


def _today(now: Optional[float] = None) -> str:
    return datetime.fromtimestamp(_now() if now is None else now).strftime("%Y-%m-%d")


def _iso(epoch: Optional[float]) -> Optional[str]:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat() if epoch else None


# ---------------------------------------------------------------------------------------------------- link store

class LinkStore:
    """The links, in one JSON file next to the local database (`<data dir>/web_call_links.json`, mode 0600):
    {"links": {id: {agent_id, user_id, mode, label, created, expires, revoked, limits, counters}}}. Only ids, limits and
    counters: no names, no audio, no visitor data. Writes are atomic (temp file, then rename); every call takes one lock
    (the main API and the public app run in one process)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @staticmethod
    def path() -> str:
        return os.path.join(os.path.dirname(_b().local_db.root), "web_call_links.json")

    def _load(self) -> Dict[str, dict]:
        try:
            with open(self.path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            links = data.get("links") if isinstance(data, dict) else None
            return links if isinstance(links, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, links: Dict[str, dict]) -> None:
        path = self.path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")   # mkstemp creates it as 0600
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"links": links}, f, ensure_ascii=False)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _roll(rec: dict, today: str) -> None:
        if rec.get("day") != today:
            rec["day"] = today
            rec["minutes_today"] = 0.0

    def get(self, link_id: str) -> Optional[dict]:
        with self._lock:
            rec = self._load().get(link_id)
            if rec is None:
                return None
            rec = dict(rec)
            self._roll(rec, _today())
            return rec

    def for_agent(self, agent_id: str, user_id: str) -> List[dict]:
        with self._lock:
            today = _today()
            out = []
            for link_id, rec in self._load().items():
                if rec.get("agent_id") == agent_id and rec.get("user_id") == user_id:
                    rec = dict(rec)
                    self._roll(rec, today)
                    out.append({**rec, "id": link_id})
            return sorted(out, key=lambda r: r.get("created") or 0, reverse=True)

    def create(self, rec: dict) -> dict:
        with self._lock:
            links = self._load()
            link_id = LINK_PREFIX + secrets.token_urlsafe(24)
            links[link_id] = rec
            self._save(links)
            return {**rec, "id": link_id}

    def mutate(self, link_id: str, fn) -> Optional[dict]:
        """Apply `fn(rec)` to a stored link and save it; returns the updated copy (None when there is no such link)."""
        with self._lock:
            links = self._load()
            rec = links.get(link_id)
            if rec is None:
                return None
            self._roll(rec, _today())
            fn(rec)
            self._save(links)
            return dict(rec)

    def revoke(self, link_id: str, agent_id: str, user_id: str) -> Optional[dict]:
        def mark(rec: dict) -> None:
            if not rec.get("revoked"):
                rec["revoked"] = _now()

        owned = self.get(link_id)
        if not owned or owned.get("agent_id") != agent_id or owned.get("user_id") != user_id:
            return None
        return self.mutate(link_id, mark)

    def remove(self, link_id: str, agent_id: str, user_id: str) -> bool:
        with self._lock:
            links = self._load()
            rec = links.get(link_id)
            if not rec or rec.get("agent_id") != agent_id or rec.get("user_id") != user_id:
                return False
            del links[link_id]
            self._save(links)
            return True

    def record_session(self, link_id: str) -> Optional[dict]:
        def bump(rec: dict) -> None:
            rec["sessions"] = int(rec.get("sessions") or 0) + 1
            rec["last_used"] = _now()

        return self.mutate(link_id, bump)

    def charge(self, link_id: str, seconds: float, turns: int = 0) -> Optional[dict]:
        def add(rec: dict) -> None:
            rec["minutes_today"] = float(rec.get("minutes_today") or 0) + seconds / 60.0
            rec["minutes_total"] = float(rec.get("minutes_total") or 0) + seconds / 60.0
            rec["turns"] = int(rec.get("turns") or 0) + turns
            rec["last_used"] = _now()

        return self.mutate(link_id, add)


store = LinkStore()


def link_status(rec: dict, now: Optional[float] = None) -> str:
    now = _now() if now is None else now
    if rec.get("revoked"):
        return "revoked"
    if rec.get("expires") and now >= float(rec["expires"]):
        return "expired"
    return "active"


def _link_limits(rec: dict) -> Dict[str, Optional[int]]:
    return {"max_session_minutes": int(rec.get("max_session_minutes") or DEFAULT_SESSION_MIN),
            "max_minutes_per_day": int(rec.get("max_minutes_per_day") or DEFAULT_DAY_MIN),
            "max_minutes_per_visitor": int(rec["max_minutes_per_visitor"]) if rec.get("max_minutes_per_visitor") else None}


# --------------------------------------------------------------------------------------------- owner endpoints

def public_base() -> Dict[str, Any]:
    """Where visitors reach the page: PUBLIC_BASE_URL (the tunnel), else this computer's public-app port."""
    configured = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured.startswith(("http://", "https://")):
        return {"base_url": configured, "public": configured.startswith("https://")}
    try:
        from phone_public import port as public_port  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from backend.phone_public import port as public_port  # noqa: PLC0415
    p = public_port()
    return {"base_url": f"http://localhost:{p}" if p else "", "public": False}


def link_url(link_id: str) -> str:
    base = public_base()["base_url"]
    return f"{base}/call/{link_id}" if base else f"/call/{link_id}"


def _owner_json(rec: dict) -> dict:
    now = _now()
    limits = _link_limits(rec)
    return {
        "id": rec["id"], "url": link_url(rec["id"]), "label": rec.get("label") or "", "status": link_status(rec, now),
        "created_at": _iso(rec.get("created")), "expires_at": _iso(rec.get("expires")),
        "revoked_at": _iso(rec.get("revoked")), "last_used_at": _iso(rec.get("last_used")), **limits,
        "sessions": int(rec.get("sessions") or 0), "turns": int(rec.get("turns") or 0),
        "minutes_today": round(float(rec.get("minutes_today") or 0), 1),
        "minutes_total": round(float(rec.get("minutes_total") or 0), 1),
    }


class LinkCreate(BaseModel):
    label: Optional[str] = Field(None, max_length=80)
    expires_in: str = DEFAULT_EXPIRY            # "1d" | "7d" | "never"
    max_session_minutes: int = Field(DEFAULT_SESSION_MIN, ge=1, le=MAX_SESSION_MIN)
    max_minutes_per_day: int = Field(DEFAULT_DAY_MIN, ge=1, le=MAX_DAY_MIN)
    max_minutes_per_visitor: Optional[int] = Field(None, ge=1, le=MAX_DAY_MIN)   # per visitor (IP) per day; None = no cap


@router.get("/api/agents/{agent_id}/web-call-links")
async def list_links(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    await _b()._load_agent(request, agent_id, user_id)
    rows = await asyncio.to_thread(store.for_agent, agent_id, user_id)
    return {"links": [_owner_json(r) for r in rows], **public_base(),
            "defaults": {"expires_in": DEFAULT_EXPIRY, "max_session_minutes": DEFAULT_SESSION_MIN,
                         "max_minutes_per_day": DEFAULT_DAY_MIN}}


@router.post("/api/agents/{agent_id}/web-call-links")
async def create_link(agent_id: str, body: LinkCreate, request: Request, user_id: str = Depends(get_current_user_id)):
    await _b()._load_agent(request, agent_id, user_id)
    if body.expires_in not in EXPIRIES:
        raise AppError("INVALID_EXPIRY", "Expiry must be 1d, 7d or never.", 422, {"field": "expires_in"})
    label = sanitize_text(body.label, "label", 80, required=False) if body.label else ""
    existing = await asyncio.to_thread(store.for_agent, agent_id, user_id)
    if sum(1 for r in existing if link_status(r) == "active") >= MAX_LINKS_PER_AGENT:
        raise AppError("TOO_MANY_LINKS", f"At most {MAX_LINKS_PER_AGENT} active links per agent. Revoke one first.", 409)
    now = _now()
    ttl = EXPIRIES[body.expires_in]
    rec = {"agent_id": agent_id, "user_id": user_id, "mode": _b()._storage_mode(request), "label": label,
           "created": now, "expires": now + ttl if ttl else None, "revoked": None,
           "max_session_minutes": body.max_session_minutes, "max_minutes_per_day": body.max_minutes_per_day,
           "max_minutes_per_visitor": body.max_minutes_per_visitor, "sessions": 0, "turns": 0,
           "minutes_total": 0.0, "day": _today(now), "minutes_today": 0.0, "last_used": None}
    created = await asyncio.to_thread(store.create, rec)
    return _owner_json(created)


@router.post("/api/agents/{agent_id}/web-call-links/{link_id}/revoke")
async def revoke_link(agent_id: str, link_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    await _b()._load_agent(request, agent_id, user_id)
    rec = await asyncio.to_thread(store.revoke, link_id, agent_id, user_id) if LINK_RE.match(link_id) else None
    if rec is None:
        raise AppError("LINK_NOT_FOUND", "Link not found.", 404)
    return _owner_json({**rec, "id": link_id})


@router.delete("/api/agents/{agent_id}/web-call-links/{link_id}")
async def delete_link(agent_id: str, link_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    await _b()._load_agent(request, agent_id, user_id)
    if not LINK_RE.match(link_id) or not await asyncio.to_thread(store.remove, link_id, agent_id, user_id):
        raise AppError("LINK_NOT_FOUND", "Link not found.", 404)
    return {"ok": True}


# ------------------------------------------------------------------------------------------------- public helpers

LOOPBACK_PEERS = ("127.0.0.1", "::1", "localhost", "testclient")


def client_ip(request: Request) -> str:
    """The visitor's address: the peer, or, when the peer is this computer (the tunnel), what the tunnel reports."""
    peer = request.client.host if request.client else "unknown"
    if peer in LOOPBACK_PEERS:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[-1]
        for raw in (request.headers.get("cf-connecting-ip", ""), forwarded):
            try:
                return str(ipaddress.ip_address(raw.strip()))
            except ValueError:
                continue
    return peer


def _page_origin(request: Request) -> Optional[str]:
    """The request's Origin when it is this page's own origin (same host the request was made to), else None."""
    origin = widget.normalize_origin(request.headers.get("origin"))
    if origin is None:
        return None
    scheme = origin.split("://", 1)[0]
    hosts = [request.headers.get("host", ""), request.headers.get("x-forwarded-host", "").split(",")[0]]
    base = os.getenv("PUBLIC_BASE_URL", "").strip()
    if base:
        hosts.append(urlsplit(base).netloc)
    for host in hosts:
        if host.strip() and widget.normalize_origin(f"{scheme}://{host.strip()}") == origin:
            return origin
    return None


def _forbidden() -> AppError:
    return AppError("CALL_FORBIDDEN", "This page cannot start a call from here.", 403)


def _inactive() -> AppError:
    return AppError("CALL_LINK_INACTIVE", "This call link is no longer active.", 410)


def _too_many(wait: int, message: str, code: str = "CALL_RATE_LIMITED") -> JSONResponse:
    return error_response(code, message, 429, {"retry_after": wait}, headers={"Retry-After": str(wait)})


def _hit(request: Request) -> Optional[int]:
    """Count this request against the per-IP request rate; returns the seconds to wait when it is over."""
    wait = limiter.consume([(f"hit:{client_ip(request)}", HITS_PER_IP_MINUTE, 60)])
    return wait or None


async def _live_link(link_id: str) -> dict:
    """The stored link when it exists and is switched on and its agent still exists; else CALL_LINK_INACTIVE (410) or
    404 for an id that was never a link."""
    rec = await asyncio.to_thread(store.get, link_id) if LINK_RE.match(link_id or "") else None
    if rec is None:
        raise AppError("NOT_FOUND", "Not found", 404)
    if link_status(rec) != "active":
        raise _inactive()
    return {**rec, "id": link_id}


async def _agent_for(link: dict) -> dict:
    row = await widget._load_agent_row(link.get("mode") or "local", link["agent_id"])
    if not row or row.get("user_id") != link.get("user_id"):
        raise _inactive()
    return row


def _visitor_key(link_id: str, ip: str) -> str:
    return f"{link_id}:{ip}:{_today()}"


def _visitor_seconds(link_id: str, ip: str) -> float:
    return _VISITOR_USED.get(_visitor_key(link_id, ip), 0.0)


def _add_visitor_seconds(link_id: str, ip: str, seconds: float) -> None:
    key = _visitor_key(link_id, ip)
    _VISITOR_USED[key] = _VISITOR_USED.get(key, 0.0) + seconds
    _VISITOR_USED.move_to_end(key)
    while len(_VISITOR_USED) > 5000:
        _VISITOR_USED.popitem(last=False)


def _check_budgets(link: dict, ip: str) -> None:
    """Raises 429 when the link's day total, or this visitor's day total, is used up."""
    limits = _link_limits(link)
    if float(link.get("minutes_today") or 0) >= limits["max_minutes_per_day"] - 1e-6:
        raise AppError("CALL_DAY_LIMIT", "This line has reached its limit for today. Please try again tomorrow.", 429)
    cap = limits["max_minutes_per_visitor"]
    if cap and _visitor_seconds(link["id"], ip) >= cap * 60 - 1e-3:
        raise AppError("CALL_VISITOR_LIMIT", "You have used your time on this line for today. Please try again "
                                              "tomorrow.", 429)


def _claims(request: Request, link_id: str, *, grace: float = 0.0) -> Dict[str, Any]:
    """Verified session claims for this link: a valid signature, the call kind, this link, and this page's origin."""
    token = widget._bearer(request)
    try:
        claims = widget.read_session(token, allow_expired_for=grace)
    except AppError as e:
        if e.code == "WIDGET_SESSION_EXPIRED":
            raise AppError("CALL_SESSION_EXPIRED", "Your call reached its time limit. Thanks for talking!", 401) from None
        raise AppError("CALL_SESSION_INVALID", "This session is not valid. Start again.", 401) from None
    if claims.get("kind") != "call" or claims.get("k") != link_id:
        raise AppError("CALL_SESSION_INVALID", "This session is not valid. Start again.", 401)
    origin = _page_origin(request)
    if origin is None or origin != claims["o"]:
        raise _forbidden()
    return claims


def _charge_seconds(sid: str, now: float, iat: float, *, minimum: float) -> float:
    last = _LAST_SEEN.get(sid, iat)
    return max(minimum, min(MAX_TURN_CHARGE_S, now - last))


def _touch(sid: str, now: float) -> None:
    _LAST_SEEN[sid] = now
    _LAST_SEEN.move_to_end(sid)
    while len(_LAST_SEEN) > 5000:
        _LAST_SEEN.popitem(last=False)


# --------------------------------------------------------------------------------------------- public: the page

def _text_page(title: str, message: str, status: int, headers: Optional[dict] = None) -> Response:
    body = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">'
            f'<meta name="color-scheme" content="light dark"><title>{html.escape(title)} | BonClay</title>'
            f'<link rel="icon" href="data:,"><link rel="stylesheet" href="/call/assets/call.css"></head>'
            f'<body class="plain"><main class="card"><p class="eyebrow">BonClay</p><h1>{html.escape(title)}</h1>'
            f'<p>{html.escape(message)}</p></main></body></html>')
    return Response(body, status_code=status, media_type="text/html; charset=utf-8", headers={
        "Cache-Control": "no-store", **(headers or {})})


def render_page(name: str, link_id: str, session_minutes: int) -> str:
    """The page template with its three values filled in in one pass (a value is never itself searched for markers)."""
    values = {"NAME": html.escape(name, quote=True), "LINK": html.escape(link_id, quote=True),
              "MINUTES": str(int(session_minutes)), "VERIFY": FOOTER_VERIFY}
    template = (ASSET_DIR / "index.html").read_text(encoding="utf-8")
    return re.sub(r"\{\{(NAME|LINK|MINUTES|VERIFY)\}\}", lambda m: values[m.group(1)], template)


@public_router.api_route("/call/assets/call.js", methods=["GET", "HEAD"], include_in_schema=False)
async def asset_js():
    return _asset("call.js", "application/javascript; charset=utf-8")


@public_router.api_route("/call/assets/call.css", methods=["GET", "HEAD"], include_in_schema=False)
async def asset_css():
    return _asset("call.css", "text/css; charset=utf-8")


def _asset(name: str, media: str):
    path = ASSET_DIR / name
    if not path.is_file():
        raise AppError("NOT_FOUND", "Not found", 404)
    return FileResponse(path, media_type=media, headers={"Cache-Control": "public, max-age=300"})


@public_router.get("/call/{link_id}", include_in_schema=False)
async def call_page(link_id: str, request: Request):
    wait = _hit(request)
    if wait:
        return _text_page("Too many visits", "Please try again in a minute.", 429, {"Retry-After": str(wait)})
    try:
        link = await _live_link(link_id)
        agent = await _agent_for(link)
    except AppError as e:
        if e.status == 410:
            return _text_page("This link is no longer active", "The person who shared it switched it off or it has "
                              "expired. Ask them for a new link.", 410)
        return _text_page("Page not found", "Check the link you were given.", 404)
    name = str(agent.get("name") or "Assistant")[:80]
    page = render_page(name, link_id, _link_limits(link)["max_session_minutes"])
    return Response(page, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------------------------------- public: the session

@public_router.post("/call/api/{link_id}/session")
async def call_session(link_id: str, request: Request):
    """A signed session token for this page. The Origin must be the page's own."""
    ip = client_ip(request)
    wait = limiter.consume([(f"hit:{ip}", HITS_PER_IP_MINUTE, 60),
                            (f"sess:{link_id[:12]}:{ip}", SESSIONS_PER_IP_LINK_HOUR, 3600),
                            (f"sess:{ip}", SESSIONS_PER_IP_HOUR, 3600)])
    if wait:
        return _too_many(wait, "Too many people are calling right now. Please try again in a few minutes.")
    origin = _page_origin(request)
    if origin is None:
        raise _forbidden()
    link = await _live_link(link_id)
    agent = await _agent_for(link)
    wait = limiter.consume([(f"sess-link:{link_id}", SESSIONS_PER_LINK_DAY, 86400)])
    if wait:
        return _too_many(wait, "This line is busy today. Please try again later.")
    _check_budgets(link, ip)
    await asyncio.to_thread(store.record_session, link_id)
    limits = _link_limits(link)
    now = int(_now())
    session_s = limits["max_session_minutes"] * 60
    sid = secrets.token_hex(8)
    token = widget.sign_session({"k": link_id, "a": agent["id"], "m": link.get("mode") or "local", "o": origin,
                                 "v": secrets.token_hex(12), "iat": now, "exp": now + session_s, "sid": sid,
                                 "kind": "call"})
    _touch(sid, float(now))
    return JSONResponse({"token": token, "expires_in": session_s, "max_session_s": session_s,
                         "agent": {"name": str(agent.get("name") or "Assistant")[:80]}},
                        headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------------------------- public: the turn

@public_router.post("/call/api/{link_id}/turn")
async def call_turn(link_id: str, request: Request, audio: UploadFile = File(...)):
    """One spoken turn. The pipeline is the owner's streaming turn (`_stream_turn_response`), as for the widget."""
    claims = _claims(request, link_id)
    ip = client_ip(request)
    link = await _live_link(link_id)
    agent = await _agent_for(link)
    if agent["id"] != claims["a"]:
        raise _inactive()
    wait = limiter.consume([(f"turn-v:{claims['v']}", TURNS_PER_VISITOR_MINUTE, 60),
                            (f"turn-ip:{link_id[:12]}:{ip}", TURNS_PER_IP_HOUR, 3600),
                            (f"hit:{ip}", HITS_PER_IP_MINUTE * 4, 60)])
    if wait:
        return _too_many(wait, "You are talking a bit fast. Wait a moment and try again.")
    _check_budgets(link, ip)
    if audio.size is not None and audio.size > MAX_UPLOAD_BYTES:
        raise AppError("FILE_TOO_LARGE", "That recording is too long.", 413)
    now = _now()
    seconds = _charge_seconds(claims["sid"], now, float(claims["iat"]), minimum=MIN_TURN_CHARGE_S)
    _touch(claims["sid"], now)
    _add_visitor_seconds(link_id, ip, seconds)
    await asyncio.to_thread(store.charge, link_id, seconds, 1)
    b = _b()
    run = await b._turn_begin(request, agent["id"], agent["user_id"], storage_mode=claims["m"], agent=agent,
                              source="widget")
    response = await b._stream_turn_response(run, audio, widget._VISITOR_CONV.get(claims["v"]))
    response.body_iterator = widget._visitor_events(response.body_iterator, claims["v"],
                                                    f"/call/api/{link_id}/clip")
    response.headers["Cache-Control"] = "no-store"
    return response


@public_router.get("/call/api/{link_id}/clip/{turn_id}/{index}")
async def call_clip(link_id: str, turn_id: str, index: int, request: Request):
    """One sentence clip of a turn that the caller's own session started (the token is the Authorization header)."""
    token = widget._bearer(request)
    try:
        claims = widget.read_session(token)
    except AppError:
        raise AppError("CALL_SESSION_INVALID", "This session is not valid. Start again.", 401) from None
    if claims.get("kind") != "call" or claims.get("k") != link_id:
        raise AppError("CALL_SESSION_INVALID", "This session is not valid. Start again.", 401)
    await _live_link(link_id)
    b = _b()
    b._check_id(turn_id, "turn id")
    if claims["m"] != "local" or index < 0 or index > 999 or widget._TURN_OWNERS.get(turn_id) != claims["v"]:
        raise AppError("NOT_FOUND", "Clip not found", 404)
    path = os.path.join(b.LOCAL_AUDIO_DIR, f"turn_{turn_id}_{index}.wav")
    if not os.path.isfile(path):
        raise AppError("NOT_FOUND", "Clip not found", 404)
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@public_router.post("/call/api/{link_id}/end")
async def call_end(link_id: str, request: Request):
    """The visitor hung up: the time since the last turn is counted (capped), and the session stops being tracked."""
    claims = _claims(request, link_id, grace=3600)
    if limiter.consume([(f"hit:{client_ip(request)}", HITS_PER_IP_MINUTE, 60)]):
        return JSONResponse({"ok": True})
    sid = claims["sid"]
    if sid in _LAST_SEEN:
        now = min(_now(), float(claims["exp"]))
        seconds = max(0.0, min(MAX_TURN_CHARGE_S, now - _LAST_SEEN.pop(sid)))
        if seconds and await asyncio.to_thread(store.get, link_id):
            _add_visitor_seconds(link_id, client_ip(request), seconds)
            await asyncio.to_thread(store.charge, link_id, seconds, 0)
    return JSONResponse({"ok": True}, headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------------------------------- public app wiring

def public_paths() -> List[str]:
    return sorted({r.path for r in public_router.routes})


class PublicCallMiddleware:
    """For the /call/* routes of the phone-only app: strict security headers on every response (also on errors), and an
    oversized upload is refused before it is read. Responses never carry an Access-Control-* header (no CORS)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/call/"):
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        length = headers.get("content-length", "")
        if scope["method"] == "POST" and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 65536:
            res = error_response("FILE_TOO_LARGE", "That recording is too long.", 413, headers=dict(SECURITY_HEADERS))
            return await res(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                raw = list(message.get("headers", []))
                present = {k.decode("latin-1").lower() for k, _ in raw}
                raw = [(k, v) for k, v in raw if not k.decode("latin-1").lower().startswith("access-control-")]
                for name, value in SECURITY_HEADERS.items():
                    if name.lower() not in present:
                        raw.append((name.lower().encode("latin-1"), value.encode("latin-1")))
                message = {**message, "headers": raw}
            await send(message)

        await self.app(scope, receive, send_wrapper)


def install_public(app) -> None:
    """Called by phone_public.py: adds the visitor routes and their middleware to the phone-only public app."""
    app.include_router(public_router)
    app.add_middleware(PublicCallMiddleware)
