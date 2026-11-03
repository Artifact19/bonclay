"""Which provider and which account carry a call: the settings and the routing.

PHONE_PROVIDERS (comma list, default `exotel`) names the providers in order of preference. PHONE_ROUTES optionally pins numbers:
`+919876543210:exotel#2,+919812345678:twilio` means that number goes to Exotel account 2 first, the other to Twilio (any
account, in its own order) first; whatever fails then falls through to the providers in PHONE_PROVIDERS order. For one
call `attempts(to)` is the ordered list of (provider, account) to try, each pair at most once: routes first, then each
provider in PHONE_PROVIDERS order with its accounts active-first-then-marked (state.AccountBook.dial_order).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import base
from .base import Provider, env
from .numbers import mask, normalise_number
from .state import AccountBook, hashed_key

log = logging.getLogger("bonclay.phone")

NAMES = ("exotel", "twilio", "plivo", "vobiz", "telnyx")
DEFAULT_PROVIDERS = ("exotel",)
_warned: set = set()
_providers: Dict[str, Provider] = {}
_books: Dict[str, AccountBook] = {}


def provider(name: str) -> Provider:
    """The provider with this name (imported on first use). KeyError for an unknown name."""
    name = name.lower()
    if name not in NAMES:
        raise KeyError(name)
    if name not in _providers:
        if name == "exotel":
            from . import exotel as mod  # noqa: PLC0415
            _providers[name] = mod.PROVIDER
        elif name == "twilio":
            from . import twilio as mod  # noqa: PLC0415
            _providers[name] = mod.PROVIDER
        elif name == "plivo":
            from . import plivo as mod  # noqa: PLC0415
            _providers[name] = mod.PLIVO
        elif name == "vobiz":
            from . import plivo as mod  # noqa: PLC0415
            _providers[name] = mod.VOBIZ
        else:
            from . import telnyx as mod  # noqa: PLC0415
            _providers[name] = mod.PROVIDER
    return _providers[name]


def book(name: str) -> AccountBook:
    """The account state of one provider. Exotel's is the one routers/phone_exotel.py binds its globals to; the others keep
    theirs in phone_accounts_<provider>.json next to Exotel's state file."""
    name = name.lower()
    if name == "exotel":
        from . import exotel as mod  # noqa: PLC0415
        return mod.BOOK
    if name not in _books:
        p = provider(name)

        def path() -> str:
            from . import exotel as mod  # noqa: PLC0415
            return os.path.join(os.path.dirname(mod.state_path()), f"phone_accounts_{name}.json")

        _books[name] = AccountBook(name, p.display, p.env_prefix, path, lambda a, _n=name, _p=p: hashed_key(_n, _p.account_id(a)))
    return _books[name]


def _warn_once(key: tuple, message: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(message, *args)


def enabled() -> List[str]:
    """PHONE_PROVIDERS as a list of known provider names, in order, without repeats (default: exotel)."""
    out: List[str] = []
    for raw in env("PHONE_PROVIDERS", "exotel").split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in NAMES:
            _warn_once(("provider", name), "PHONE_PROVIDERS names an unknown provider %r (known: %s); it is ignored", name,
                       ", ".join(NAMES))
            continue
        if name not in out:
            out.append(name)
    return out or list(DEFAULT_PROVIDERS)


def routes() -> Dict[str, List[Tuple[str, Optional[str]]]]:
    """PHONE_ROUTES: {E.164 number: [(provider, account label or None), ...]}. A bad entry is skipped with one warning."""
    out: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    for raw in env("PHONE_ROUTES").split(","):
        entry = raw.strip()
        if not entry:
            continue
        number_raw, _, target = entry.partition(":")
        number = normalise_number(number_raw)
        name, _, label = target.strip().partition("#")
        name = name.strip().lower()
        label = label.strip() or None
        if number is None or name not in NAMES or (label is not None and not label.isdigit()):
            _warn_once(("route", entry), "PHONE_ROUTES entry %r is ignored: use +919876543210:provider or "
                       "+919876543210:provider#2", mask(number_raw) + ":" + target.strip()[:24])
            continue
        out.setdefault(number, []).append((name, label))
    return out


@dataclass
class Attempt:
    provider: Provider
    account: Dict[str, str]

    @property
    def label(self) -> str:
        return f"{self.provider.name} {self.account['label']}"


def attempts(to: Optional[str] = None) -> List[Attempt]:
    """The (provider, account) pairs to try for a call to `to`, in order, each pair once. Empty when nothing is configured."""
    orders: Dict[str, List[Dict[str, str]]] = {}

    def order_of(name: str) -> List[Dict[str, str]]:
        if name not in orders:
            orders[name] = book(name).dial_order(provider(name).accounts())
        return orders[name]

    plan: List[Attempt] = []
    seen = set()

    def add(name: str, account: Dict[str, str]) -> None:
        key = (name, account["label"])
        if key not in seen:
            seen.add(key)
            plan.append(Attempt(provider(name), account))

    for name, label in (routes().get(to or "") or []):
        for a in order_of(name):
            if label is None or a["label"] == label:
                add(name, a)
    # The providers in PHONE_PROVIDERS order, each with its accounts, active accounts of every provider before any marked one:
    # an account known to be out of credit is a last resort, not something every call tries before the next provider.
    marked: List[Tuple[str, Dict[str, str]]] = []
    for name in enabled():
        for a in order_of(name):
            if book(name).state(a)["state"] == "active":
                add(name, a)
            else:
                marked.append((name, a))
    for name, a in marked:
        add(name, a)
    return plan


def routed_providers() -> List[str]:
    seen: List[str] = []
    for targets in routes().values():
        for name, _ in targets:
            if name not in seen:
                seen.append(name)
    return seen


def configured_providers() -> List[str]:
    """Enabled and routed providers that have at least one complete account."""
    return [n for n in dict.fromkeys(enabled() + routed_providers()) if provider(n).configured()]


def missing_settings() -> List[str]:
    """[] when at least one complete account exists and PUBLIC_BASE_URL is set; otherwise the names still to be set (those
    of the first provider in PHONE_PROVIDERS when nothing is configured)."""
    public = bool(env("PUBLIC_BASE_URL"))
    if configured_providers():
        return [] if public else ["PUBLIC_BASE_URL"]
    return provider(enabled()[0]).missing_vars() + ([] if public else ["PUBLIC_BASE_URL"])


def account_by_label(name: str, label: Optional[str]) -> Optional[Dict[str, str]]:
    """The configured account of this provider with this label; else its first one (the call was placed before a restart
    that removed its account); else None."""
    accounts = provider(name).accounts()
    for a in accounts:
        if a["label"] == label:
            return a
    return accounts[0] if accounts else None


def public_status() -> List[dict]:
    """For GET /api/phone/exotel/status: every provider, whether it has an account and is in use, and its accounts with
    their state. Never a key, token or SID."""
    from .numbers import mask_caller_id  # noqa: PLC0415
    on = enabled()
    out = []
    for name in NAMES:
        p = provider(name)
        accounts = p.accounts()
        out.append({"name": name, "display": p.display, "configured": bool(accounts), "enabled": name in on,
                    "verified": p.verified,
                    "accounts": book(name).public(accounts, mask_caller_id)})
    return out
