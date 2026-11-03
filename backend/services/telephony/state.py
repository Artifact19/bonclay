"""Which account of a provider is usable: `active`, `exhausted` (out of credit) or `auth_failed` (credentials refused).

The state is kept in memory and in a small JSON file (mode 0600), keyed by a hash of the account's identity, so a restart does not
send the next call to an account that is known to be empty. A refused state is forgotten after
<PREFIX>_EXHAUSTED_RESET_HOURS (default 24), and a successful call makes the account active again. `dial_order` lists
the accounts to try for one call: the active ones first (in order, or rotated by one each call with
<PREFIX>_ROTATION=round_robin), then the marked ones, each at most once: a marked account is a last resort, never left out,
because a recharge or a fixed key cannot be seen from here and a call must not stop while some account might still work.

Exotel's book is bound to routers/phone_exotel.py's module globals (`_now_ts`, `_rr_counter`, `_state_loaded_from`), which is
where it always lived and where the tests set them."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .base import env

log = logging.getLogger("bonclay.phone")


class AccountBook:
    def __init__(self, provider: str, display: str, env_prefix: str, path_fn: Callable[[], str],
                 key_fn: Callable[[Dict[str, str]], str]) -> None:
        self.provider = provider
        self.display = display
        self.env_prefix = env_prefix
        self.path_fn = path_fn
        self.key_fn = key_fn
        self.lock = threading.RLock()
        self.states: Dict[str, Dict[str, Any]] = {}       # key -> {"state", "since" (epoch seconds)}
        self._ns: Optional[dict] = None
        self._loaded_from: Optional[str] = None
        self._rr = 0

    # ---- scalars that live in a module's globals for Exotel (so tests can set them there) ---------------------
    def bind_namespace(self, ns: dict) -> None:
        self._ns = ns

    def _now(self) -> float:
        if self._ns is not None and "_now_ts" in self._ns:
            return self._ns["_now_ts"]()
        return time.time()

    @property
    def loaded_from(self) -> Optional[str]:
        return self._ns["_state_loaded_from"] if self._ns is not None and "_state_loaded_from" in self._ns else self._loaded_from

    @loaded_from.setter
    def loaded_from(self, value: Optional[str]) -> None:
        if self._ns is not None and "_state_loaded_from" in self._ns:
            self._ns["_state_loaded_from"] = value
        else:
            self._loaded_from = value

    @property
    def rr(self) -> int:
        return self._ns["_rr_counter"] if self._ns is not None and "_rr_counter" in self._ns else self._rr

    @rr.setter
    def rr(self, value: int) -> None:
        if self._ns is not None and "_rr_counter" in self._ns:
            self._ns["_rr_counter"] = value
        else:
            self._rr = value

    # ---- file ---------------------------------------------------------------------------------------------------
    def _load(self) -> None:
        """(Re)read the state file when its path changed since the last read. Caller holds the lock."""
        path = self.path_fn()
        if self.loaded_from == path:
            return
        self.loaded_from = path
        self.states.clear()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and v.get("state") in ("exhausted", "auth_failed"):
                    try:
                        self.states[str(k)] = {"state": v["state"], "since": float(v.get("since") or 0)}
                    except (TypeError, ValueError):
                        continue

    def _save(self) -> None:
        path = self.path_fn()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{uuid.uuid4().hex}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.states, f, indent=2)
            os.replace(tmp, path)
        except OSError as e:  # the state is a convenience: a call must never fail because it cannot be saved
            logging.getLogger(f"bonclay.phone.{self.provider}").warning(
                "could not save the %s account state: %s", self.display, e.__class__.__name__)

    def reset_seconds(self) -> float:
        try:
            return max(0.0, float(env(f"{self.env_prefix}_EXHAUSTED_RESET_HOURS", "24") or 24)) * 3600
        except ValueError:
            return 24 * 3600.0

    # ---- state --------------------------------------------------------------------------------------------------
    def state(self, account: Dict[str, str]) -> Dict[str, Any]:
        """{"state": "active" | "exhausted" | "auth_failed", "since": epoch seconds or None}. A refused account counts
        as active again once the reset window has passed."""
        with self.lock:
            self._load()
            row = self.states.get(self.key_fn(account))
            if row and self._now() - row["since"] < self.reset_seconds():
                return {"state": row["state"], "since": row["since"]}
        return {"state": "active", "since": None}

    def set(self, account: Dict[str, str], state: str) -> None:
        with self.lock:
            self._load()
            key = self.key_fn(account)
            if state == "active":
                if self.states.pop(key, None) is None:
                    return
            else:
                self.states[key] = {"state": state, "since": self._now()}
            self._save()

    def dial_order(self, accounts: List[Dict[str, str]]) -> List[Dict[str, str]]:
        usable = [a for a in accounts if self.state(a)["state"] == "active"]
        refused = [a for a in accounts if a not in usable]
        if env(f"{self.env_prefix}_ROTATION", "failover").lower().replace("-", "_") == "round_robin" and len(usable) > 1:
            with self.lock:
                start = self.rr % len(usable)
                self.rr = self.rr + 1
            usable = usable[start:] + usable[:start]
        return usable + refused

    def public(self, accounts: List[Dict[str, str]], mask_caller: Callable[[str], Optional[str]]) -> List[Dict[str, Any]]:
        """Per account, for the status route: label, masked caller id, state and since. Never a key, token or SID."""
        out = []
        for a in accounts:
            st = self.state(a)
            since = datetime.fromtimestamp(st["since"], timezone.utc).isoformat(timespec="seconds") if st["since"] else None
            out.append({"label": a["label"], "caller_id_masked": mask_caller(a.get("caller_id", "")),
                        "state": st["state"], "since": since})
        return out


def hashed_key(provider: str, identity: str) -> str:
    return hashlib.sha256(f"{provider}:{identity}".encode("utf-8")).hexdigest()[:16]
