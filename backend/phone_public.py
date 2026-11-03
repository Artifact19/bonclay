"""The phone-only public app: the routes Exotel must reach and the web call link page, and nothing else, on a second port.

Exotel connects to this computer over a public tunnel (ngrok or cloudflared, started by `./bonclay start`). Until now
the tunnel pointed at the whole API, so every unauthenticated local-mode route (voices, cloning, keys, the call
endpoint) was reachable from the internet. Now the tunnel points at PHONE_PUBLIC_PORT (default 8010), where this app
serves only:

  * the call stream   WS   /api/phone/exotel/stream/{token}          (signed token in the URL)
  * the status callback POST /api/phone/exotel/status-callback/{token} (signed token in the URL)
  * for every other provider in use (Twilio, Plivo, Vobiz, Telnyx; services/telephony) the same two plus an answer document:
                       WS   /api/phone/{provider}/stream/{token}, POST .../status-callback/{token}, GET|POST .../answer/{token}
                       (signed token in the URL; inert unless PHONE_PROVIDERS or PHONE_ROUTES names the provider)
  * a health probe    GET  /api/phone/public/health
  * the web call link (routers/web_call.py): GET /call/{link} (a page where anyone with the link talks to one agent in
    their browser), its two static files under /call/assets/, and POST /call/api/{link}/session|turn|end plus
    GET /call/api/{link}/clip/{turn}/{n}. A link id is an unguessable, revocable capability for one agent only.

It runs inside the main backend process, on the same event loop, so live calls share their state (call sessions,
monitoring, greeting cache) with the owner's routes on the main port. Anything else through the tunnel is a 404.
PHONE_PUBLIC_PORT=0 turns the second port off (the stream stays reachable on the main, loopback-only port)."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Optional

from fastapi import FastAPI

try:
    from core.errors import register_exception_handlers
    from routers import phone_exotel, web_call
except ImportError:  # pragma: no cover - imported as backend.phone_public
    from backend.core.errors import register_exception_handlers
    from backend.routers import phone_exotel, web_call

log = logging.getLogger("bonclay.phone.public")

DEFAULT_PORT = 8010

public_app = FastAPI(title="BonClay phone (public)", docs_url=None, redoc_url=None, openapi_url=None)
register_exception_handlers(public_app)
public_app.include_router(phone_exotel.public_router)
web_call.install_public(public_app)   # the web call link page, its two static files and four API routes, all under /call/ (routers/web_call.py)


@public_app.get("/api/phone/public/health")
async def public_health():
    return {"ok": True, "routes": phone_exotel.active_public_paths()}


def port() -> Optional[int]:
    """PHONE_PUBLIC_PORT (default 8010); None when it is 0, empty or not a number."""
    raw = os.getenv("PHONE_PUBLIC_PORT", str(DEFAULT_PORT)).strip()
    try:
        value = int(raw)
    except ValueError:
        log.warning("PHONE_PUBLIC_PORT=%r is not a number; the phone-only port is off", raw)
        return None
    return value if 0 < value < 65536 else None


class PublicServer:
    """A second uvicorn server for `public_app` on the running event loop. start() returns at once; stop() waits."""

    def __init__(self) -> None:
        self._server = None
        self._task: Optional[asyncio.Task] = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, host: str = "127.0.0.1") -> Optional[int]:
        p = port()
        if p is None or self.running:
            return p if self.running else None
        import uvicorn
        config = uvicorn.Config(public_app, host=host, port=p, log_level="warning", lifespan="off")
        server = uvicorn.Server(config)
        # uvicorn's serve() captures SIGINT/SIGTERM for itself; the main server already owns them in this process.
        if hasattr(server, "capture_signals"):
            server.capture_signals = contextlib.nullcontext  # type: ignore[method-assign]
        self._server = server
        self._task = asyncio.get_running_loop().create_task(server.serve(), name="phone-public-server")
        log.info("phone-only public app listening on http://%s:%d (stream + status callback only)", host, p)
        return p

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, timeout=5.0)
        self._server, self._task = None, None


server = PublicServer()
