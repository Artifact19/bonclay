"""Heartbeat: one small request a minute to an uptime checker, so a sleeping laptop, a dead Wi-Fi link or a stopped
backend is noticed within two minutes by someone who is not looking at the screen.

BONCLAY_HEARTBEAT_URL is the address to ping (a healthchecks.io, Uptime Kuma or Grafana OnCall "ping" URL: the service
alerts when the pings stop). Every BONCLAY_HEARTBEAT_S seconds (default 60) the backend POSTs a short JSON body:
{"sent_at", "uptime_s", "active_calls", "version"}. Nothing personal leaves the machine. Unset: nothing is sent. In
offline mode (BONCLAY_OFFLINE=1) the heartbeat is refused like every other outbound call.

The loop also notices its own gaps: when a tick arrives much later than scheduled (the laptop slept, the process was
suspended) it logs the gap and keeps it in `status()`, so `GET /api/health` shows "slept for 540 s" after the fact.

`./bonclay start` keeps the Mac awake while the API runs (caffeinate -is); this is the belt to that brace."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

try:
    from core import offline
except ImportError:  # pragma: no cover
    from backend.core import offline

log = logging.getLogger("bonclay.heartbeat")

DEFAULT_INTERVAL_S = 60.0
MIN_INTERVAL_S = 10.0
GAP_FACTOR = 2.0            # a tick this many intervals late counts as a gap (sleep, suspend)
REQUEST_TIMEOUT_S = 10.0

Sender = Callable[[str, bytes], int]   # (url, body) -> HTTP status; raises on failure


def url() -> str:
    return os.getenv("BONCLAY_HEARTBEAT_URL", "").strip()


def interval_s() -> float:
    try:
        return max(MIN_INTERVAL_S, float(os.getenv("BONCLAY_HEARTBEAT_S", str(DEFAULT_INTERVAL_S))))
    except ValueError:
        return DEFAULT_INTERVAL_S


def disabled_reason() -> Optional[str]:
    """Why no heartbeat is sent, or None when it is."""
    target = url()
    if not target:
        return "BONCLAY_HEARTBEAT_URL is not set"
    if not urlparse(target).scheme in ("http", "https"):
        return "BONCLAY_HEARTBEAT_URL is not an http(s) address"
    if offline.enabled() and not offline.is_local_url(target):
        return "offline mode (BONCLAY_OFFLINE=1) refuses outbound calls"
    return None


def _default_sender(target: str, body: bytes) -> int:
    req = urllib.request.Request(target, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "bonclay-heartbeat"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
        return int(r.status)


class Heartbeat:
    """The loop. `active_calls` is a callable giving the live call count (the phone router's counter)."""

    def __init__(self, sender: Sender = _default_sender, active_calls: Callable[[], int] = lambda: 0,
                 version: str = "", clock: Callable[[], float] = time.monotonic) -> None:
        self.sender, self.active_calls, self.version, self.clock = sender, active_calls, version, clock
        self.started_at: Optional[float] = None
        self.last_sent_at: Optional[str] = None
        self.last_status: Optional[int] = None
        self.last_error: Optional[str] = None
        self.sent = 0
        self.failures = 0
        self.last_gap_s: Optional[float] = None
        self.gaps = 0
        self._task: Optional[asyncio.Task] = None
        self._last_tick: Optional[float] = None

    # ---- one beat ------------------------------------------------------------------------------------------------

    def payload(self) -> Dict[str, Any]:
        uptime = (self.clock() - self.started_at) if self.started_at is not None else 0.0
        return {"sent_at": datetime.now(timezone.utc).isoformat(), "uptime_s": round(uptime),
                "active_calls": int(self.active_calls()), "version": self.version}

    def beat(self) -> bool:
        """Send one heartbeat (blocking). True when the checker answered 2xx. Never raises."""
        reason = disabled_reason()
        if reason:
            self.last_error = reason
            return False
        try:
            status = self.sender(url(), json.dumps(self.payload()).encode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {str(e)[:120]}"
            log.warning("heartbeat not delivered (%d failures): %s", self.failures, self.last_error)
            return False
        self.last_status, self.last_error = status, None
        self.last_sent_at = datetime.now(timezone.utc).isoformat()
        self.sent += 1
        return 200 <= status < 300

    def note_tick(self, now: Optional[float] = None) -> Optional[float]:
        """Record a tick; returns the gap in seconds when this tick came far later than scheduled (the machine slept)."""
        now = self.clock() if now is None else now
        gap = None
        if self._last_tick is not None:
            late = now - self._last_tick
            if late > GAP_FACTOR * interval_s():
                gap = round(late, 1)
                self.gaps += 1
                self.last_gap_s = gap
                log.warning("heartbeat gap: no tick for %.0f s (the computer slept or the process was suspended)", late)
        self._last_tick = now
        return gap

    # ---- the loop ------------------------------------------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            self.note_tick()
            await asyncio.to_thread(self.beat)
            await asyncio.sleep(interval_s())

    def start(self) -> Optional[str]:
        """Start the loop on the running event loop. Returns the reason it is not running, or None."""
        reason = disabled_reason()
        self.started_at = self.clock()
        if reason:
            return reason
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run(), name="heartbeat")
            log.info("heartbeat every %.0f s to %s", interval_s(), urlparse(url()).netloc)
        return None

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    def status(self) -> Dict[str, Any]:
        """What /api/health shows."""
        reason = disabled_reason()
        return {"enabled": reason is None, "reason": reason, "target": urlparse(url()).netloc or None,
                "interval_s": interval_s(), "running": self._task is not None and not self._task.done(),
                "sent": self.sent, "failures": self.failures, "last_sent_at": self.last_sent_at,
                "last_status": self.last_status, "last_error": self.last_error,
                "last_gap_s": self.last_gap_s, "gaps": self.gaps}


def _live_calls() -> int:
    try:
        from routers import phone_exotel
    except ImportError:  # pragma: no cover
        from backend.routers import phone_exotel
    return int(getattr(phone_exotel, "_active_calls", 0))


heartbeat = Heartbeat(active_calls=_live_calls)
