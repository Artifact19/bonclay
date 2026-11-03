"""Reply-time numbers for the "reply under 3 seconds" claim: per-turn timings stored by the conversation turn
endpoints, plus p50 / p95 over the user's most recent spoken turns. Streamed turns (POST .../turn/stream) also record
`first_audio_ms`, the time until the caller could hear the first sentence; the summary reports it separately."""
import asyncio
import math
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request

try:
    from services.auth import get_current_user_id
except ImportError:  # imported as backend.routers.perf
    from backend.services.auth import get_current_user_id

router = APIRouter(prefix="/api/perf", tags=["perf"])

TARGET_MS = 3000
STAGES = ("asr_ms", "retrieval_ms", "llm_ms", "tts_ms")


def _percentile(values: List[int], pct: float) -> Optional[int]:
    """Nearest-rank percentile: the ceil(pct/100 * n)-th smallest value."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return int(ordered[min(rank, len(ordered)) - 1])


def _num(v) -> Optional[int]:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _turn_json(row: dict, agent_names: Dict[str, str]) -> Optional[dict]:
    timings = row.get("timings")
    if not isinstance(timings, dict) or not timings:
        return None
    agent_id = row.get("agent_id")
    total = _num(row.get("total_ms"))
    if total is None:
        total = _num(row.get("latency_ms"))
    if total is None:
        total = sum(_num(timings.get(s)) or 0 for s in STAGES)
    first_audio = _num(timings.get("first_audio_ms"))
    return {
        "id": row.get("id"),
        "created_at": row.get("created_at"),
        "agent_id": agent_id,
        "agent_name": agent_names.get(agent_id),
        "total_ms": total,
        "timings": {s: _num(timings.get(s)) or 0 for s in STAGES},
        "first_audio_ms": first_audio,  # streamed turns only
        "llm_first_token_ms": _num(timings.get("llm_first_token_ms")),
        "engine": row.get("engine"),
        "llm_provider": row.get("llm_provider"),
        "language": row.get("language"),
        "fallback": bool(row.get("fallback")),
    }


def _summary(turns: List[dict]) -> dict:
    totals = [t["total_ms"] for t in turns]
    firsts = [t["first_audio_ms"] for t in turns if t.get("first_audio_ms") is not None]
    return {
        "turns": turns,
        "count": len(turns),
        "p50_ms": _percentile(totals, 50),
        "p95_ms": _percentile(totals, 95),
        "stage_p50": {s: _percentile([t["timings"][s] for t in turns], 50) for s in STAGES},
        "target_ms": TARGET_MS,
        "within_target_pct": round(100 * sum(1 for x in totals if x <= TARGET_MS) / len(totals), 1)
        if totals else None,
        # time to first audio, over the streamed turns among them (null when none were streamed)
        "first_audio_count": len(firsts),
        "first_audio_p50_ms": _percentile(firsts, 50),
        "first_audio_p95_ms": _percentile(firsts, 95),
        "first_audio_within_target_pct": round(100 * sum(1 for x in firsts if x <= TARGET_MS) / len(firsts), 1)
        if firsts else None,
    }


@router.get("/turns")
async def perf_turns(request: Request, limit: int = Query(20, ge=1, le=200), agent_id: Optional[str] = Query(None),
                     user_id: str = Depends(get_current_user_id)):
    import backend as B  # lazy: backend.py loads this router while it is still being imported

    if B._storage_mode(request) == "local":
        def load():
            convs = {c["id"]: c["agent_id"] for c in B.local_db.select(
                "conversations", lambda r: r.get("user_id") == user_id
                and (agent_id is None or r.get("agent_id") == agent_id))}
            names = {a["id"]: a.get("name") for a in B.local_db.select("agents", lambda r: r.get("user_id") == user_id)}
            rows = B.local_db.select(
                "conversation_turns", lambda r: r.get("conversation_id") in convs and r.get("role") == "agent"
                and isinstance(r.get("timings"), dict), order="created_at", desc=True, limit=limit)
            for r in rows:
                r["agent_id"] = convs[r["conversation_id"]]
            return rows, names
        rows, names = await asyncio.to_thread(load)
    else:
        def query():
            q = (B.supabase.table("conversation_turns")
                 .select("id, created_at, timings, total_ms, latency_ms, engine, llm_provider, language, fallback, "
                         "conversations!inner(agent_id, user_id)")
                 .eq("role", "agent").eq("conversations.user_id", user_id).not_.is_("timings", "null"))
            if agent_id:
                q = q.eq("conversations.agent_id", agent_id)
            data = q.order("created_at", desc=True).limit(limit).execute().data or []
            for r in data:
                r["agent_id"] = (r.pop("conversations", None) or {}).get("agent_id")
            ids = sorted({r["agent_id"] for r in data if r.get("agent_id")})
            agents = (B.supabase.table("agents").select("id, name").in_("id", ids).execute().data or []) if ids else []
            return data, {a["id"]: a.get("name") for a in agents}
        rows, names = await B._cloud_run(query, "list turn metrics")

    turns = [t for t in (_turn_json(r, names) for r in rows) if t]
    return _summary(turns)
