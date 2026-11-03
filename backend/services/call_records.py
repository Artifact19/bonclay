"""Phone call records: one row per call, kept across restarts (data/call_records.jsonl, newest row of a call wins).

A row says what happened to a call (queued, ringing, in-progress, completed, failed, busy, no-answer, canceled), which
agent and owner it belongs to, its direction, when it started and ended, how it ended and which conversation holds its
transcript. The callee's number is stored masked only (the same privacy default as the Calls page); campaigns that
need real numbers will keep them in their own table, not here.

The log is append-only and tolerant: a torn last line is skipped, and a call that appears only in the callback (a
restart lost the in-memory register) is still recorded."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

_lock = threading.RLock()
FIELDS = ("call_sid", "owner", "agent_id", "direction", "status", "to_masked", "created_at", "updated_at", "started_at",
          "ended_at", "duration_s", "hangup_reason", "conversation_id", "message", "outcome", "dtmf_presses", "transfer_to", "exotel_account",
          "provider", "account")
TERMINAL = ("completed", "failed", "busy", "no-answer", "canceled")


def _path() -> str:
    env = os.getenv("BONCLAY_CALL_RECORDS_FILE", "").strip()
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "call_records.jsonl")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _load() -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    try:
        with open(_path(), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("call_sid"):
                    rows[str(row["call_sid"])] = row
    except OSError:
        pass
    return rows


def upsert(call_sid: Optional[str], **fields: Any) -> Optional[Dict[str, Any]]:
    """Merge `fields` into the call's row and append it. Unknown keys are dropped; None values do not overwrite. A
    terminal status is never replaced by a non-terminal one (a late 'in-progress' must not undo 'completed').
    Never raises: a record that cannot be written must not break a call."""
    if not call_sid:
        return None
    try:
        with _lock:
            rows = _load()
            row = dict(rows.get(str(call_sid)) or {"call_sid": str(call_sid), "created_at": _now()})
            for k, v in fields.items():
                if k in FIELDS and v is not None:
                    row[k] = v
            old = (rows.get(str(call_sid)) or {}).get("status")
            if old in TERMINAL and row.get("status") not in TERMINAL:
                row["status"] = old
            row["updated_at"] = _now()
            os.makedirs(os.path.dirname(_path()), exist_ok=True)
            with open(_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps({k: row.get(k) for k in ("call_sid",) + FIELDS[1:] if row.get(k) is not None},
                                   ensure_ascii=False) + "\n")
            return row
    except Exception:  # noqa: BLE001
        return None


def get(call_sid: str) -> Optional[Dict[str, Any]]:
    with _lock:
        row = _load().get(str(call_sid))
    return dict(row) if row else None


def list_for(owner: str, limit: int = 50) -> List[Dict[str, Any]]:
    """The owner's calls, newest first."""
    with _lock:
        rows = [r for r in _load().values() if r.get("owner") == owner]
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return [dict(r) for r in rows[:max(1, min(limit, 200))]]
