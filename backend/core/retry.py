"""Exponential-backoff retry helpers (sync + async) for flaky Supabase calls."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

log = logging.getLogger("bonclay.retry")

_PERMANENT_MARKERS = ("not found", "404", "not_found", "no such", "does not exist", "unauthorized", "403")


def is_transient(exc: BaseException) -> bool:
    """Missing objects / auth failures will not fix themselves; everything else may."""
    text = str(exc).lower()
    return not any(m in text for m in _PERMANENT_MARKERS)


def retry_sync(fn: Callable[..., Any], *args, attempts: int = 3, base_delay: float = 0.4,
               factor: float = 2.0, should_retry: Optional[Callable[[BaseException], bool]] = None,
               label: str = "", **kwargs):
    should_retry = should_retry or is_transient
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt >= attempts or not should_retry(exc):
                raise
            log.warning("retry %s attempt %d/%d failed: %s", label or getattr(fn, "__name__", "call"),
                        attempt, attempts, exc)
            time.sleep(delay)
            delay *= factor


async def retry_async(fn: Callable[..., Any], *args, attempts: int = 3, base_delay: float = 0.4,
                      factor: float = 2.0, should_retry: Optional[Callable[[BaseException], bool]] = None,
                      label: str = "", **kwargs):
    """Retry a coroutine function, or a sync callable (run in a worker thread)."""
    should_retry = should_retry or is_transient
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as exc:
            if attempt >= attempts or not should_retry(exc):
                raise
            log.warning("retry %s attempt %d/%d failed: %s", label or getattr(fn, "__name__", "call"),
                        attempt, attempts, exc)
            await asyncio.sleep(delay)
            delay *= factor
