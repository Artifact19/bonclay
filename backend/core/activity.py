"""When did a conversation last use this machine? Background jobs (services/background_llm.py) ask before they load a
model, so they never compete with a caller.

`mark_turn()` is called at the start of every conversation turn (app, widget, WhatsApp, phone: backend._turn_begin).
`is_idle(quiet_s)` is true when no turn started in the last `quiet_s` seconds and no phone call is live. It is the same
rule the answer audit uses (services/answer_audit.live_traffic: "a conversation had a turn in the last N seconds"), applied
to the turns of this process instead of the stored rows, so it costs no database read.

Listeners registered with `on_turn()` are told about every turn (the background model unloads itself); they run on the
caller's thread, so they must return at once.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, List, Optional

log = logging.getLogger("bonclay.activity")

DEFAULT_IDLE_S = 60.0

_lock = threading.Lock()
_last_turn: Optional[float] = None       # time.monotonic() of the latest turn start
_listeners: List[Callable[[], None]] = []


def idle_s() -> float:
    """BACKGROUND_LLM_IDLE_S: seconds without a turn before background work may use the machine (default 60)."""
    try:
        return max(0.0, float(os.getenv("BACKGROUND_LLM_IDLE_S", "") or DEFAULT_IDLE_S))
    except ValueError:
        return DEFAULT_IDLE_S


def mark_turn(now: Optional[float] = None) -> None:
    global _last_turn
    with _lock:
        _last_turn = now if now is not None else time.monotonic()
        listeners = list(_listeners)
    for fn in listeners:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - a listener must never break a live turn
            log.warning("activity listener failed: %s", e)


def on_turn(fn: Callable[[], None]) -> None:
    with _lock:
        if fn not in _listeners:
            _listeners.append(fn)


def seconds_since_turn(now: Optional[float] = None) -> Optional[float]:
    with _lock:
        last = _last_turn
    if last is None:
        return None
    return max(0.0, (now if now is not None else time.monotonic()) - last)


def call_live() -> bool:
    """True while a phone call is registered (services/call_control)."""
    try:
        try:
            from services import call_control  # noqa: PLC0415
        except ImportError:  # pragma: no cover - imported as backend.core.activity
            from backend.services import call_control  # noqa: PLC0415
        return bool(call_control.live_sessions())
    except Exception:  # noqa: BLE001
        return False


def is_idle(quiet_s: Optional[float] = None, now: Optional[float] = None) -> bool:
    """No phone call is live and no conversation turn started in the last `quiet_s` seconds (default BACKGROUND_LLM_IDLE_S)."""
    if call_live():
        return False
    since = seconds_since_turn(now)
    return since is None or since >= (idle_s() if quiet_s is None else quiet_s)


def reset() -> None:
    """Forget the last turn (tests)."""
    global _last_turn
    with _lock:
        _last_turn = None
