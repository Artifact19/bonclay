"""Request authentication: the single source of truth for resolving the caller's user id.

`get_current_user_id` is used by every route, in backend.py and in the routers alike, and handles all three
credentials: personal API keys (`aur_...`), Supabase JWTs, and the unauthenticated local/demo shortcuts. Those
shortcuts (local storage mode, demo tokens) are only honoured when Supabase auth is not configured, or when
explicitly enabled via env flags, so a production deployment cannot be bypassed by sending a header.
"""
import asyncio
import os
import secrets

from fastapi import Request

from .common import api_error

LOCAL_USER = "local-user"
DEMO_USER = "demo-user"
DEMO_TOKENS = {"demo-token", "test"}

# Local mode trusts a request as the local user only when it arrives on this computer's loopback interface. The
# TestClient reports "testclient". Headers that tunnels and proxies add (ngrok, cloudflared, nginx) mean the request
# came from outside even if the last hop was 127.0.0.1.
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
FORWARDED_HEADERS = ("x-forwarded-for", "x-forwarded-host", "cf-connecting-ip", "x-real-ip", "forwarded")


def is_loopback(request: Request) -> bool:
    """True when the request came from this computer and not through a tunnel or proxy."""
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client else None
    if host is None:
        return True  # no transport (a Request built by hand in a unit test)
    if host not in LOOPBACK_HOSTS:
        return False
    return not any(h in request.headers for h in FORWARDED_HEADERS)


def local_token() -> str:
    """BONCLAY_LOCAL_TOKEN: the one secret that lets a request from another device use local mode. Empty = none."""
    return os.getenv("BONCLAY_LOCAL_TOKEN", "").strip()


def _local_user(request: Request) -> str:
    """The local user, or 401 when the request did not come from this computer and carries no BONCLAY_LOCAL_TOKEN.
    Without this, a tunnel or a LAN address would hand the whole unauthenticated API to anyone who found it."""
    if is_loopback(request):
        return LOCAL_USER
    token = local_token()
    presented = bearer_token(request) or request.headers.get("x-bonclay-token", "").strip()
    if token and presented and secrets.compare_digest(presented, token):
        return LOCAL_USER
    raise api_error(401, "This request did not come from this computer. Local mode answers only on localhost; from "
                         "another device set BONCLAY_LOCAL_TOKEN in backend/.env and send it as the bearer token.",
                    "LOCAL_TOKEN_REQUIRED", None)


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def auth_configured() -> bool:
    return bool((os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL"))
                and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_JWT_SECRET")))


def local_mode_allowed() -> bool:
    """Local (on-disk, single-user) mode: always allowed without Supabase, opt-in otherwise."""
    return not auth_configured() or _flag("BONCLAY_ALLOW_LOCAL_MODE")


def demo_mode_allowed() -> bool:
    return not auth_configured() or _flag("BONCLAY_ALLOW_DEMO_TOKENS")


def _apikeys():
    try:
        from core import apikeys
    except ImportError:  # running from a different cwd
        from backend.core import apikeys
    return apikeys


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer ") and header[7:].strip():
        return header[7:].strip()
    return None


async def get_current_user_id(request: Request) -> str:
    """Return the authenticated user's id, or raise 401 with the structured error shape.

    The id is also recorded on `request.state.user_id` so the request log attributes the call."""
    uid = await _resolve_user_id(request)
    request.state.user_id = uid
    return uid


async def _resolve_user_id(request: Request) -> str:
    token = bearer_token(request)

    apikeys = _apikeys()
    if token and token.startswith(apikeys.KEY_PREFIX):  # a personal API key: checked before anything else
        uid = await asyncio.to_thread(apikeys.authenticate_key, token)
        if not uid:
            raise api_error(401, "Invalid API key", "INVALID_API_KEY", None)
        return uid

    if request.headers.get("storage-mode") == "local" and local_mode_allowed():
        return _local_user(request)
    if token in DEMO_TOKENS and demo_mode_allowed():
        return DEMO_USER
    if not auth_configured():
        return _local_user(request)

    if not token:
        raise api_error(401, "Missing bearer token", "UNAUTHORIZED", None)
    secret = os.getenv("SUPABASE_JWT_SECRET")
    try:
        if secret:
            import jwt
            payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
            uid = payload.get("sub")
        else:
            from supabase import create_client
            url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
            client = create_client(url, os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
            uid = (await asyncio.to_thread(client.auth.get_user, token)).user.id
    except Exception as e:
        raise api_error(401, "Invalid or expired token", "UNAUTHORIZED", str(e))
    if not uid:
        raise api_error(401, "Invalid token", "UNAUTHORIZED", None)
    return str(uid)
