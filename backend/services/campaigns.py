"""Outbound call campaigns: a list of people who agreed to be called, a calling window, retries and a do-not-call list.

Pure logic and storage; routers/campaigns.py holds the API and the scheduler loop that calls `tick`. Everything is in
local files under data/ (campaigns.json, do_not_call.json), written atomically and readable only by this user (0600).
Contact numbers must be kept to dial them, so unlike call records they are stored in full: delete a campaign (or let
`purge_finished` do it after CAMPAIGN_KEEP_DAYS) to remove them. The owner attests, per campaign, that every person on
the list agreed to be called; BonClay does not check that, and does not check national do-not-disturb registers.

Contact states: pending (waiting for its turn), calling (a call is in flight), done (the call connected), failed (out of
attempts), dnc (on the do-not-call list: never dialled). A busy, no-answer or failed call goes back to pending after
`retry_minutes` until `max_attempts` calls were made."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

_lock = threading.RLock()

MAX_CONTACTS = 500
MAX_CONCURRENCY = 3
STALE_CALL_S = 15 * 60          # a call "in flight" this long without a final status counts as a failed attempt
BREAKER_WINDOW = 10             # the last N finished attempts ...
BREAKER_FAILURE_RATE = 0.5      # ... pause the campaign when this share ended in "failed" (not busy / no-answer)
RETRY_STATUSES = ("busy", "no-answer", "failed", "canceled")
STATUSES = ("draft", "running", "paused", "done", "stopped")


def _dir() -> str:
    env = os.getenv("BONCLAY_CAMPAIGNS_DIR", "").strip()
    return env or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _campaigns_path() -> str:
    return os.path.join(_dir(), "campaigns.json")


def _dnc_path() -> str:
    return os.path.join(_dir(), "do_not_call.json")


def _read(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------------------------- do-not-call list

def dnc_all(owner: str) -> List[str]:
    with _lock:
        return sorted((_read(_dnc_path(), {}) or {}).get(owner, []))


def dnc_add(owner: str, numbers: List[str]) -> int:
    """Add E.164 numbers to the owner's list; returns how many were new. Contacts already in campaigns are skipped
    from then on (checked again at dial time)."""
    with _lock:
        data = _read(_dnc_path(), {}) or {}
        have = set(data.get(owner, []))
        new = [n for n in numbers if n not in have]
        data[owner] = sorted(have | set(numbers))
        _write(_dnc_path(), data)
        return len(new)


def dnc_remove(owner: str, number: str) -> bool:
    with _lock:
        data = _read(_dnc_path(), {}) or {}
        have = set(data.get(owner, []))
        if number not in have:
            return False
        have.discard(number)
        data[owner] = sorted(have)
        _write(_dnc_path(), data)
        return True


# ---------------------------------------------------------------------------------------------- CSV

def parse_csv(text: str, normalise: Callable[[str], Optional[str]]) -> Dict[str, Any]:
    """Contacts from CSV text. The first row may be a header (number, phone, mobile, ...; optional name). A file of bare
    numbers also works. Returns {"contacts": [{number, name}], "invalid": n, "duplicates": n, "over_limit": n}."""
    rows = list(csv.reader(io.StringIO(text or "")))
    rows = [r for r in rows if any(c.strip() for c in r)]
    number_col, name_col, start = 0, None, 0
    if rows:
        head = [c.strip().lower() for c in rows[0]]
        names = ("number", "phone", "mobile", "msisdn", "contact", "phone_number", "phone number")
        if any(h in names for h in head):
            number_col = next(i for i, h in enumerate(head) if h in names)
            name_col = next((i for i, h in enumerate(head) if h in ("name", "full_name", "contact_name")), None)
            start = 1
        elif len(head) > 1 and normalise(rows[0][0]) is None and normalise(rows[0][1]) is not None:
            number_col, name_col = 1, 0
    out: List[Dict[str, str]] = []
    seen, invalid, dup, over = set(), 0, 0, 0
    for r in rows[start:]:
        if number_col >= len(r):
            invalid += 1
            continue
        n = normalise(r[number_col])
        if n is None:
            invalid += 1
        elif n in seen:
            dup += 1
        elif len(out) >= MAX_CONTACTS:
            over += 1
        else:
            seen.add(n)
            name = r[name_col].strip()[:60] if name_col is not None and name_col < len(r) else ""
            out.append({"number": n, "name": name})
    return {"contacts": out, "invalid": invalid, "duplicates": dup, "over_limit": over}


# ---------------------------------------------------------------------------------------------- calling windows

DEFAULT_WINDOW = {"days": [0, 1, 2, 3, 4, 5], "start": "10:00", "end": "18:00", "timezone": "Asia/Kolkata"}


def clean_window(raw: Any) -> Dict[str, Any]:
    """A valid calling window: days (0=Monday .. 6=Sunday), local "HH:MM" start and end, an IANA time zone."""
    w = dict(DEFAULT_WINDOW)
    if isinstance(raw, dict):
        days = raw.get("days")
        if isinstance(days, list) and days and all(isinstance(d, int) and 0 <= d <= 6 for d in days):
            w["days"] = sorted(set(days))
        for k in ("start", "end"):
            if isinstance(raw.get(k), str) and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", raw[k]):
                w[k] = raw[k]
        if isinstance(raw.get("timezone"), str):
            try:
                ZoneInfo(raw["timezone"])
                w["timezone"] = raw["timezone"]
            except Exception:  # noqa: BLE001
                pass
    if w["start"] >= w["end"]:
        w["start"], w["end"] = DEFAULT_WINDOW["start"], DEFAULT_WINDOW["end"]
    return w


def in_window(now: float, window: Dict[str, Any]) -> bool:
    local = datetime.fromtimestamp(now, tz=timezone.utc).astimezone(ZoneInfo(window["timezone"]))
    hm = local.strftime("%H:%M")
    return local.weekday() in window["days"] and window["start"] <= hm < window["end"]


# ---------------------------------------------------------------------------------------------- campaigns

def create(owner: str, *, name: str, agent_id: str, mode: str, contacts: List[Dict[str, str]], window: Any = None,
           max_attempts: int = 2, retry_minutes: int = 30, concurrency: int = 1, max_per_hour: int = 30) -> Dict[str, Any]:
    dnc = set(dnc_all(owner))
    rows = []
    for c in contacts:
        rows.append({"id": uuid.uuid4().hex[:10], "number": c["number"], "name": c.get("name", ""),
                     "status": "dnc" if c["number"] in dnc else "pending", "attempts": 0, "next_at": 0.0,
                     "last_sid": None, "last_status": None, "history": []})
    camp = {"id": uuid.uuid4().hex[:12], "owner": owner, "name": (name or "Campaign").strip()[:80],
            "agent_id": agent_id, "mode": mode, "status": "draft", "created_at": time.time(), "finished_at": None,
            "window": clean_window(window), "max_attempts": max(1, min(int(max_attempts), 5)),
            "retry_minutes": max(1, min(int(retry_minutes), 24 * 60)),
            "concurrency": max(1, min(int(concurrency), MAX_CONCURRENCY)),
            "max_per_hour": max(1, min(int(max_per_hour), 120)), "dial_times": [], "paused_reason": None,
            "caller_agreed": True, "contacts": rows}
    with _lock:
        data = _read(_campaigns_path(), {}) or {}
        data[camp["id"]] = camp
        _write(_campaigns_path(), data)
    return camp


def get(campaign_id: str) -> Optional[Dict[str, Any]]:
    return (_read(_campaigns_path(), {}) or {}).get(campaign_id)


def list_for(owner: str) -> List[Dict[str, Any]]:
    data = _read(_campaigns_path(), {}) or {}
    return sorted((c for c in data.values() if c.get("owner") == owner), key=lambda c: -c.get("created_at", 0))


def running() -> List[Dict[str, Any]]:
    return [c for c in (_read(_campaigns_path(), {}) or {}).values() if c.get("status") == "running"]


def save(camp: Dict[str, Any]) -> None:
    with _lock:
        data = _read(_campaigns_path(), {}) or {}
        data[camp["id"]] = camp
        _write(_campaigns_path(), data)


def delete(campaign_id: str, owner: str) -> bool:
    with _lock:
        data = _read(_campaigns_path(), {}) or {}
        if campaign_id not in data or data[campaign_id].get("owner") != owner:
            return False
        del data[campaign_id]
        _write(_campaigns_path(), data)
        return True


def set_status(campaign_id: str, owner: str, status: str, reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with _lock:
        camp = get(campaign_id)
        if not camp or camp.get("owner") != owner or status not in STATUSES:
            return None
        if camp["status"] in ("done", "stopped") and status != camp["status"]:
            return camp                                   # a finished campaign stays finished
        camp["status"] = status
        camp["paused_reason"] = reason if status == "paused" else None
        if status in ("done", "stopped"):
            camp["finished_at"] = time.time()
        save(camp)
        return camp


def counts(camp: Dict[str, Any]) -> Dict[str, int]:
    out = {"pending": 0, "calling": 0, "done": 0, "failed": 0, "dnc": 0}
    for c in camp["contacts"]:
        out[c["status"]] = out.get(c["status"], 0) + 1
    return out


def purge_finished(now: Optional[float] = None) -> int:
    """Delete campaigns finished more than CAMPAIGN_KEEP_DAYS (default 30) ago, with their numbers."""
    days = float(os.getenv("CAMPAIGN_KEEP_DAYS", "30") or 30)
    now = now if now is not None else time.time()
    with _lock:
        data = _read(_campaigns_path(), {}) or {}
        old = [k for k, c in data.items() if c.get("finished_at") and c["finished_at"] < now - days * 86400]
        for k in old:
            del data[k]
        if old:
            _write(_campaigns_path(), data)
        return len(old)


# ---------------------------------------------------------------------------------------------- the scheduler step

def _breaker_tripped(camp: Dict[str, Any]) -> bool:
    finished = sorted((h for c in camp["contacts"] for h in c["history"]), key=lambda h: h["at"])[-BREAKER_WINDOW:]
    if len(finished) < BREAKER_WINDOW:
        return False
    return sum(1 for h in finished if h["status"] == "failed") / len(finished) >= BREAKER_FAILURE_RATE


def _settle(camp: Dict[str, Any], now: float, record_of: Callable[[str], Optional[Dict[str, Any]]]) -> None:
    """Move contacts whose call ended out of "calling". The call record decides: the call connected (it completed, or its
    audio stream ran and ended) = done; busy, no-answer, failed or canceled = retry later or fail; a call with no
    answer from the phone network for STALE_CALL_S counts as a failed attempt."""
    for c in camp["contacts"]:
        if c["status"] != "calling":
            continue
        rec = (record_of(c["last_sid"]) if c.get("last_sid") else None) or {}
        st = rec.get("status")
        voicemail = rec.get("outcome") == "voicemail"
        connected = not voicemail and (st == "completed" or bool(rec.get("hangup_reason") and rec.get("started_at")))
        if connected:
            c["status"], c["last_status"] = "done", "completed"
            c["history"].append({"at": now, "status": "completed"})
            continue
        stale = now - c.get("called_at", now) > STALE_CALL_S
        if voicemail or st in RETRY_STATUSES or (stale and st != "in-progress"):
            st = "voicemail" if voicemail else (st if st in RETRY_STATUSES else "failed")
            c["last_status"] = st
            c["history"].append({"at": now, "status": st})
            if c["attempts"] >= camp["max_attempts"]:
                c["status"] = "failed"
            else:
                c["status"], c["next_at"] = "pending", now + camp["retry_minutes"] * 60
        elif st == "in-progress":
            c["last_status"] = st


async def tick(camp_id: str, now: float, dial: Callable[[Dict[str, Any], Dict[str, Any]], Any],
               record_of: Callable[[str], Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """One scheduler step for one running campaign: settle finished calls, trip the circuit breaker, then start as many
    calls as the window, concurrency cap and hourly cap allow. `dial(camp, contact)` places a call and returns its
    call_sid (it raises when the call cannot be placed); `record_of(call_sid)` returns the call record (services/call_records) or None.
    Returns {"dialled": n, "state": "..."}."""
    camp = get(camp_id)
    if not camp or camp["status"] != "running":
        return {"dialled": 0, "state": "not_running"}
    _settle(camp, now, record_of)
    dnc = set(dnc_all(camp["owner"]))
    for c in camp["contacts"]:
        if c["status"] == "pending" and c["number"] in dnc:
            c["status"] = "dnc"
    if _breaker_tripped(camp):
        camp["status"], camp["paused_reason"] = "paused", "Too many calls failed to connect. Check the phone setup, then resume."
        save(camp)
        return {"dialled": 0, "state": "paused_breaker"}
    if all(c["status"] in ("done", "failed", "dnc") for c in camp["contacts"]):
        camp["status"], camp["finished_at"] = "done", now
        save(camp)
        return {"dialled": 0, "state": "done"}
    if not in_window(now, camp["window"]):
        save(camp)
        return {"dialled": 0, "state": "outside_window"}
    camp["dial_times"] = [t for t in camp.get("dial_times", []) if t > now - 3600]
    busy = sum(1 for c in camp["contacts"] if c["status"] == "calling")
    dialled = 0
    for c in camp["contacts"]:
        if busy >= camp["concurrency"] or len(camp["dial_times"]) >= camp["max_per_hour"]:
            break
        if c["status"] != "pending" or c["next_at"] > now:
            continue
        try:
            sid = await dial(camp, c)
        except Exception as e:  # noqa: BLE001 - the call could not be placed: count the attempt, try again later
            c["attempts"] += 1
            c["last_status"] = "failed"
            c["history"].append({"at": now, "status": "failed", "error": str(e)[:120]})
            c["status"] = "failed" if c["attempts"] >= camp["max_attempts"] else "pending"
            c["next_at"] = now + camp["retry_minutes"] * 60
            continue
        c["attempts"] += 1
        c["status"], c["last_sid"], c["called_at"] = "calling", sid, now
        camp["dial_times"].append(now)
        busy += 1
        dialled += 1
    save(camp)
    return {"dialled": dialled, "state": "running"}
