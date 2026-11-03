"""Generation history: Supabase `generations` table when configured, else backend/data/generations.json."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("bonclay.generations")

_get_supabase: Callable[[], Any] = lambda: None
_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_lock = threading.Lock()


def configure(get_supabase: Callable[[], Any], data_dir: Optional[str] = None) -> None:
    global _get_supabase, _data_dir
    _get_supabase = get_supabase
    if data_dir:
        _data_dir = data_dir


def _file() -> str:
    os.makedirs(_data_dir, exist_ok=True)
    return os.path.join(_data_dir, "generations.json")


def _read_local() -> List[dict]:
    path = _file()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _append_local(row: dict) -> None:
    with _lock:
        rows = _read_local()
        rows.append(row)
        tmp = _file() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows[-5000:], f, indent=2)
        os.replace(tmp, _file())


def record_generation(user_id: str, voice_id: Optional[str], voice_name: Optional[str], text: str,
                      engine: str, language: str, duration_s: float, audio_url: Optional[str],
                      use_supabase: bool = True, gen_id: Optional[str] = None) -> dict:
    """Synchronous (call via to_thread from async code). Never raises."""
    row: Dict[str, Any] = {
        "id": gen_id or str(uuid.uuid4()), "user_id": user_id, "voice_id": voice_id or None,
        "voice_name": voice_name, "text": text, "engine": engine, "language": language,
        "duration_s": round(float(duration_s), 3), "audio_url": audio_url,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    sb = _get_supabase() if use_supabase else None
    if sb is not None:
        try:
            from .retry import retry_sync
            retry_sync(lambda: sb.table("generations").insert(row).execute(), label="generations.insert")
            return row
        except Exception as exc:  # noqa: BLE001
            log.warning("Supabase generations insert failed, using local file: %s", exc)
    try:
        _append_local(row)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not record generation locally: %s", exc)
    return row


def _summarise(rows: List[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r.get("created_at") or "", reverse=True)
    total_s = sum(float(r.get("duration_s") or 0) for r in rows)
    keys = ("id", "voice_id", "voice_name", "text", "engine", "language", "duration_s", "audio_url", "created_at")
    return {
        "generations_count": len(rows),
        "minutes_synthesized": round(total_s / 60.0, 3),
        "recent": [{k: r.get(k) for k in keys} for r in rows[:10]],
    }


def get_generation_stats(user_id: str, use_supabase: bool = True) -> dict:
    sb = _get_supabase() if use_supabase else None
    if sb is not None:
        try:
            from .retry import retry_sync
            res = retry_sync(lambda: sb.table("generations").select("*").eq("user_id", user_id)
                             .order("created_at", desc=True).limit(10000).execute(),
                             label="generations.select")
            return _summarise(res.data or [])
        except Exception as exc:  # noqa: BLE001
            log.warning("Supabase generations query failed, using local file: %s", exc)
    return _summarise([r for r in _read_local() if r.get("user_id") == user_id])
