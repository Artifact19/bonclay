"""Personal API keys. Keys look like `aur_<id>_<secret>`; only a salted hash is stored."""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

try:
    import bcrypt  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    bcrypt = None

log = logging.getLogger("bonclay.apikeys")
KEY_PREFIX = "aur_"
_PBKDF2_ITERS = 200_000

_get_supabase: Callable[[], Any] = lambda: None
_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_lock = threading.Lock()
_verify_cache: Dict[str, tuple] = {}  # sha256(key) -> (user_id, expiry)


def configure(get_supabase: Callable[[], Any], data_dir: Optional[str] = None) -> None:
    global _get_supabase, _data_dir
    _get_supabase = get_supabase
    if data_dir:
        _data_dir = data_dir


# ---- hashing ---------------------------------------------------------------

def hash_key(key: str) -> str:
    if bcrypt is not None:
        return bcrypt.hashpw(key.encode(), bcrypt.gensalt()).decode()
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", key.encode(), salt, _PBKDF2_ITERS)
    return "pbkdf2$%d$%s$%s" % (_PBKDF2_ITERS, base64.b64encode(salt).decode(), base64.b64encode(dk).decode())


def verify_key(key: str, stored: str) -> bool:
    try:
        if stored.startswith("pbkdf2$"):
            _, iters, salt, dk = stored.split("$")
            calc = hashlib.pbkdf2_hmac("sha256", key.encode(), base64.b64decode(salt), int(iters))
            return hmac.compare_digest(calc, base64.b64decode(dk))
        if bcrypt is None:
            return False
        return bcrypt.checkpw(key.encode(), stored.encode())
    except Exception:  # noqa: BLE001
        return False


# ---- storage ---------------------------------------------------------------

def _file() -> str:
    os.makedirs(_data_dir, exist_ok=True)
    return os.path.join(_data_dir, "api_keys.json")


def _read_local() -> List[dict]:
    if not os.path.exists(_file()):
        return []
    try:
        with open(_file(), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _write_local(rows: List[dict]) -> None:
    tmp = _file() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, _file())


def _sb():
    return _get_supabase()


def _public(row: dict) -> dict:
    return {k: row.get(k) for k in ("id", "name", "prefix", "created_at", "last_used_at")}


# ---- operations (sync; call via to_thread) ---------------------------------

def create_key(user_id: str, name: str) -> dict:
    key_id = secrets.token_hex(4)
    key = f"{KEY_PREFIX}{key_id}_{secrets.token_urlsafe(32)}"
    row = {
        "id": str(uuid.uuid4()), "user_id": user_id, "name": name,
        "prefix": f"{KEY_PREFIX}{key_id}", "key_hash": hash_key(key),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "last_used_at": None,
    }
    sb = _sb()
    stored = False
    if sb is not None:
        try:
            from .retry import retry_sync
            retry_sync(lambda: sb.table("api_keys").insert(row).execute(), label="api_keys.insert")
            stored = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Supabase api_keys insert failed, using local file: %s", exc)
    if not stored:
        with _lock:
            rows = _read_local()
            rows.append(row)
            _write_local(rows)
    return {**_public(row), "key": key}


def list_keys(user_id: str) -> List[dict]:
    rows: List[dict] = []
    sb = _sb()
    if sb is not None:
        try:
            res = sb.table("api_keys").select("*").eq("user_id", user_id).execute()
            rows.extend(res.data or [])
        except Exception as exc:  # noqa: BLE001
            log.warning("Supabase api_keys select failed: %s", exc)
    rows.extend(r for r in _read_local() if r.get("user_id") == user_id)
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return [_public(r) for r in rows]


def delete_key(user_id: str, key_id: str) -> bool:
    removed = False
    sb = _sb()
    if sb is not None:
        try:
            res = sb.table("api_keys").delete().eq("id", key_id).eq("user_id", user_id).execute()
            removed = bool(res.data)
        except Exception as exc:  # noqa: BLE001
            log.warning("Supabase api_keys delete failed: %s", exc)
    with _lock:
        rows = _read_local()
        kept = [r for r in rows if not (r.get("id") == key_id and r.get("user_id") == user_id)]
        if len(kept) != len(rows):
            _write_local(kept)
            removed = True
    for k in [k for k, v in _verify_cache.items() if v[2] == key_id]:
        _verify_cache.pop(k, None)
    return removed


def authenticate_key(key: str) -> Optional[str]:
    """Return the owning user_id for a valid key, else None."""
    if not key.startswith(KEY_PREFIX) or key.count("_") < 2:
        return None
    digest = hashlib.sha256(key.encode()).hexdigest()
    hit = _verify_cache.get(digest)
    if hit and hit[1] > time.time():
        return hit[0]
    prefix = "_".join(key.split("_", 2)[:2])
    candidates: List[dict] = []
    sb = _sb()
    if sb is not None:
        try:
            res = sb.table("api_keys").select("*").eq("prefix", prefix).execute()
            candidates.extend(res.data or [])
        except Exception as exc:  # noqa: BLE001
            log.warning("Supabase api_keys lookup failed: %s", exc)
    candidates.extend(r for r in _read_local() if r.get("prefix") == prefix)
    for row in candidates:
        if verify_key(key, row.get("key_hash", "")):
            _verify_cache[digest] = (row["user_id"], time.time() + 60, row["id"])
            _touch(row)
            return row["user_id"]
    return None


def _touch(row: dict) -> None:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    sb = _sb()
    try:
        if sb is not None:
            sb.table("api_keys").update({"last_used_at": now}).eq("id", row["id"]).execute()
        with _lock:
            rows = _read_local()
            changed = False
            for r in rows:
                if r.get("id") == row["id"]:
                    r["last_used_at"] = now
                    changed = True
            if changed:
                _write_local(rows)
    except Exception:  # noqa: BLE001
        pass
