"""Several Groq accounts, one list of keys.

Groq's limits are per account (organisation) and per model: requests and tokens per minute (RPM, TPM) and per day
(RPD, TPD), audio seconds per hour and per day for Whisper. A deployment on the free tier hits them fast, so the
owner can add more accounts and a request that one account cannot take goes out on the next account's key instead of
failing or dropping to a weaker model.

    GROQ_API_KEY       account 1 (as before)
    GROQ_API_KEY_2 .. GROQ_API_KEY_9   accounts 2 to 9 (gaps are fine, an empty value is ignored)

A key is known by its label, "1" to "9", never by its value: the value is never logged, returned by status() or kept in
any state below (an auth failure remembers a short hash of the value, so that a key that was replaced is tried again).

State, all in this process:
- daily pause   (label, model) -> the moment that account's daily quota of that model resets
- cool-down     (label, model) -> a per-minute limit, usually a few seconds
- auth failed   label          -> the key was refused (401 / 403); skipped for an hour

An exclusion by cool-down or by auth failure is soft: when it would leave nothing to try, the excluded keys are tried
anyway (a limit may have lifted, a key may have been fixed), exactly as a single-key setup always did. A daily pause is
hard, as it always was for a model.

The environment is read at call time, so a test can set a key with monkeypatch.setenv."""
import hashlib
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("bonclay.llm")

MAX_KEYS = 9
AUTH_FAILED_S = 3600.0
MINUTE_COOLDOWN_S = 20.0           # when a per-minute 429 gives no hint how long to wait
MINUTE_COOLDOWN_MAX_S = 300.0

_lock = threading.RLock()
# (label, model) -> (monotonic time the daily quota resets, the same moment as a UTC datetime for messages)
_daily_paused: Dict[Tuple[str, str], tuple] = {}
_cooldown: Dict[Tuple[str, str], float] = {}          # (label, model) -> monotonic time the per-minute limit lifts
_auth_failed: Dict[str, Tuple[float, str]] = {}       # label -> (monotonic time to try it again, hash of the key)
_sdk_clients: Dict[str, Tuple[str, Any]] = {}         # label -> (hash of the key, groq.Groq client)


def _env_name(n: int) -> str:
    return "GROQ_API_KEY" if n == 1 else f"GROQ_API_KEY_{n}"


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def keys() -> List[Dict[str, str]]:
    """The configured keys in account order: [{"label": "1", "key": "..."}, ...]. A key that repeats an earlier one is
    left out (the same account twice would only double the failures)."""
    out: List[Dict[str, str]] = []
    seen = set()
    for n in range(1, MAX_KEYS + 1):
        value = (os.environ.get(_env_name(n)) or "").strip()
        if value and value not in seen:
            seen.add(value)
            out.append({"label": str(n), "key": value})
    return out


def has_any() -> bool:
    return bool(keys())


def key_for(label: str) -> Optional[str]:
    return next((k["key"] for k in keys() if k["label"] == label), None)


def reset() -> None:
    """Forget every pause, cool-down and auth failure (tests)."""
    with _lock:
        _daily_paused.clear()
        _cooldown.clear()
        _auth_failed.clear()
        _sdk_clients.clear()


# ---------------------------------------------------------------------------------------------- state

def daily_reset(label: str, model: str, now: Optional[float] = None) -> Optional[datetime]:
    """When the daily quota of `model` on key `label` resets (UTC), None when it is not paused."""
    entry = _daily_paused.get((label, model))
    if entry and entry[0] > (now if now is not None else time.monotonic()):
        return entry[1]
    return None


def is_daily_paused(label: str, model: str, now: Optional[float] = None) -> bool:
    return daily_reset(label, model, now) is not None


def is_cooling(label: str, model: str, now: Optional[float] = None) -> bool:
    return _cooldown.get((label, model), 0.0) > (now if now is not None else time.monotonic())


def is_auth_failed(label: str, now: Optional[float] = None) -> bool:
    entry = _auth_failed.get(label)
    if not entry or entry[0] <= (now if now is not None else time.monotonic()):
        return False
    current = key_for(label)
    return current is not None and _fingerprint(current) == entry[1]


def pause_daily(label: str, model: str, seconds: float) -> Tuple[Optional[datetime], bool]:
    """Pause `model` on key `label` until its daily quota resets. Returns (reset time, True when this call set it, False
    when another thread had already)."""
    with _lock:
        if is_daily_paused(label, model):
            return daily_reset(label, model), False
        wall = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        _daily_paused[(label, model)] = (time.monotonic() + seconds, wall)
        return wall, True


def cool_down(label: str, model: str, seconds: Optional[float]) -> None:
    """Keep `model` off key `label` for a per-minute limit: `seconds` (Groq's own hint) or MINUTE_COOLDOWN_S."""
    wait = MINUTE_COOLDOWN_S if seconds is None else min(MINUTE_COOLDOWN_MAX_S, max(0.5, float(seconds)))
    with _lock:
        _cooldown[(label, model)] = time.monotonic() + wait


def mark_auth_failed(label: str, seconds: float = AUTH_FAILED_S) -> bool:
    """Skip key `label` for an hour (it was refused). True when this call set it (log once)."""
    key = key_for(label)
    with _lock:
        newly = not is_auth_failed(label)
        _auth_failed[label] = (time.monotonic() + seconds, _fingerprint(key or ""))
    if newly:
        log.warning("Groq key %s was refused (401/403); not using it for %.0f min", label, seconds / 60)
    return newly


# ---------------------------------------------------------------------------------------------- ordering

def ordered_keys(model: str, *, soft: bool = True) -> List[Dict[str, str]]:
    """The keys to try for `model`, in account order. A key whose daily quota of this model is used up is never in it.
    With soft=True a key that is cooling down for this model or was refused is left out too; soft=False keeps them (the
    caller uses it when soft=True leaves nothing)."""
    now = time.monotonic()
    out = []
    for k in keys():
        label = k["label"]
        if is_daily_paused(label, model, now):
            continue
        if soft and (is_cooling(label, model, now) or is_auth_failed(label, now)):
            continue
        out.append(k)
    return out


def plan(models: List[str]) -> List[Tuple[str, Dict[str, str]]]:
    """The attempts for one request, best model first, every usable key on a model before the next model:
    [(model, {"label", "key"}), ...]. Empty when there is nothing to try."""
    for soft in (True, False):
        found = [(m, k) for m in models for k in ordered_keys(m, soft=soft)]
        if found:
            return found
    return []


def model_resets(model: str) -> Optional[datetime]:
    """When `model` can be used again on some key, if every configured key has used its daily quota of it; else None."""
    ks = keys()
    if not ks:
        return None
    now = time.monotonic()
    resets = [daily_reset(k["label"], model, now) for k in ks]
    if any(r is None for r in resets):
        return None
    return min(resets)


def paused_model_names() -> List[str]:
    now = time.monotonic()
    return sorted({m for (_, m), (until, _) in list(_daily_paused.items()) if until > now})


# ---------------------------------------------------------------------------------------------- status

def status() -> List[Dict[str, Any]]:
    """For the dashboard: [{"label", "state": "active" | "paused" | "auth_failed", "paused_models": [{"model",
    "until"}]}]. No key material."""
    now = time.monotonic()
    out = []
    for k in keys():
        label = k["label"]
        paused = []
        for (lab, model), (until, wall) in sorted(_daily_paused.items()):
            if lab == label and until > now:
                paused.append({"model": model, "until": wall.isoformat(timespec="seconds")})
        state = "auth_failed" if is_auth_failed(label, now) else ("paused" if paused else "active")
        out.append({"label": label, "state": state, "paused_models": paused})
    return out


# ---------------------------------------------------------------------------------------------- Groq SDK clients

def sdk_client(label: str) -> Any:
    """One groq.Groq client per key (speech recognition), cached; None when the key or the SDK is missing."""
    key = key_for(label)
    if not key:
        return None
    fp = _fingerprint(key)
    with _lock:
        cached = _sdk_clients.get(label)
        if cached and cached[0] == fp:
            return cached[1]
        try:
            import groq  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return None
        client = groq.Groq(api_key=key)
        _sdk_clients[label] = (fp, client)
        return client
