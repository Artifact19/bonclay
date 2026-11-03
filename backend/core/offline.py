"""Offline mode: BONCLAY_OFFLINE=1 refuses every network path that could carry audio, text or numbers off this machine.

Blocked: Groq (language model and speech recognition), a language-model server that is not on this machine,
Supabase, outbound webhooks and Google Sheets, Google Calendar, Exotel phone calls, WhatsApp, and Hugging Face downloads
(HF_HUB_OFFLINE, so models must already be cached: run `./bonclay warm` once with the network on).

Importing this module in offline mode sets HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE for the whole process; import it
before any model library. `block()` is the one call every outbound path makes."""
from __future__ import annotations

import ipaddress
import os
from typing import Optional
from urllib.parse import urlsplit

from core.errors import AppError


def enabled() -> bool:
    return os.getenv("BONCLAY_OFFLINE", "").strip().lower() in ("1", "true", "on", "yes")


try:  # BONCLAY_OFFLINE may be set in backend/.env, which backend.py only reads after the model libraries load
    from dotenv import load_dotenv
    if not os.getenv("BONCLAY_SKIP_DOTENV"):  # the test suite (conftest.py) never reads the real .env
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:  # pragma: no cover
    pass

if enabled():  # before any model library is imported
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def is_local_url(url: Optional[str]) -> bool:
    """True when the host of `url` is this machine (localhost, 127.x, ::1) or a private LAN address."""
    try:
        host = urlsplit(url or "").hostname or ""
    except ValueError:
        return False
    if host in ("localhost", "host.docker.internal") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def reason(what: str) -> str:
    return f"Offline mode is on (BONCLAY_OFFLINE=1): {what} is switched off so nothing leaves this computer."


def block(what: str) -> None:
    """Raise a clear 503 when offline mode is on; a no-op otherwise."""
    if enabled():
        raise AppError("OFFLINE_MODE", reason(what), 503, {"blocked": what})
