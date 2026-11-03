# Extra pip deps: supabase (optional; falls back to backend/data/generations.json)
"""Usage analytics over the `generations` table."""
import builtins
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Literal

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

try:
    from services.generations import load_generations
    from services.auth import get_current_user_id
    from services import knowledge_gaps as kg
    from routers import perf
    from mcp_servers import meetings
except ImportError:
    from backend.services.generations import load_generations
    from backend.services.auth import get_current_user_id
    from backend.services import knowledge_gaps as kg
    from backend.routers import perf
    from backend.mcp_servers import meetings

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary")
async def summary(range: Literal["7d", "30d"] = Query("7d"), user_id: str = Depends(get_current_user_id)):
    days = 7 if range == "7d" else 30
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days - 1)
    since = since.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = load_generations(since.isoformat(), user_id)

    by_engine, by_lang, daily = defaultdict(int), defaultdict(int), defaultdict(int)
    lat = defaultdict(list)
    for r in rows:
        eng = r.get("engine") or "unknown"
        by_engine[eng] += 1
        by_lang[r.get("language") or "unknown"] += 1
        day = str(r.get("created_at", ""))[:10]
        if day:
            daily[day] += 1
        if isinstance(r.get("duration_ms"), (int, float)):
            lat[eng].append(r["duration_ms"])

    def ranked(d, key):
        return [{key: k, "count": v} for k, v in sorted(d.items(), key=lambda x: -x[1])]

    return {
        "range": range,
        "by_engine": ranked(by_engine, "engine"),
        "by_language": ranked(by_lang, "language"),
        "daily": [{"date": (since + timedelta(days=i)).date().isoformat(),
                   "count": daily.get((since + timedelta(days=i)).date().isoformat(), 0)} for i in builtins.range(days)],
        "latency": [{"engine": e, "avg_ms": round(sum(v) / len(v), 1)} for e, v in sorted(lat.items())],
    }


def _load_turns(B, mode: str, user_id: str, agent_id: Optional[str], since_iso: str, tests: Optional[set] = None):
    """(conversation id -> agent id, agent id -> name, turn rows) for the owner's conversations since `since_iso`.
    Owner Test calls stay out, unless `tests` is a set: then they are included and their conversation ids are added
    to it (the reply-time figures count them, like the Overview page does)."""
    keep_tests = tests is not None
    if mode == "local":
        rows_c = B.local_db.select(
            "conversations", lambda r: r.get("user_id") == user_id and (keep_tests or not r.get("is_test"))
            and (agent_id is None or r.get("agent_id") == agent_id))
        convs = {c["id"]: c["agent_id"] for c in rows_c}
        if keep_tests:
            tests.update(c["id"] for c in rows_c if c.get("is_test"))
        names = {a["id"]: a.get("name") for a in B.local_db.select("agents", lambda r: r.get("user_id") == user_id)}
        rows = B.local_db.select("conversation_turns", lambda r: r.get("conversation_id") in convs
                                 and str(r.get("created_at") or "") >= since_iso)
        return convs, names, rows
    def conversations(skip_tests: bool):
        q = B.supabase.table("conversations").select("id, agent_id, is_test" if keep_tests else "id, agent_id") \
            .eq("user_id", user_id)
        if skip_tests and not keep_tests:
            q = q.eq("is_test", False)  # owner Test calls stay out (migration 010_conversation_is_test.sql)
        if agent_id:
            q = q.eq("agent_id", agent_id)
        data = q.execute().data or []
        if keep_tests:
            tests.update(c["id"] for c in data if c.get("is_test"))
        return {c["id"]: c["agent_id"] for c in data}
    try:
        convs = conversations(True)
    except Exception:  # noqa: BLE001 - migration 010 may not be applied yet: no test flag to filter on
        if keep_tests:
            tests.clear()
            q = B.supabase.table("conversations").select("id, agent_id").eq("user_id", user_id)
            convs = {c["id"]: c["agent_id"] for c in ((q.eq("agent_id", agent_id) if agent_id else q).execute().data or [])}
        else:
            convs = conversations(False)
    agents = B.supabase.table("agents").select("id, name").eq("user_id", user_id).execute().data or []
    rows: list = []
    ids = list(convs)
    for i in builtins.range(0, len(ids), 100):  # bounded batches keep the request URL short
        def fetch(cols, chunk=ids[i:i + 100]):
            return (B.supabase.table("conversation_turns").select(cols).in_("conversation_id", chunk)
                    .gte("created_at", since_iso).order("created_at", desc=True).limit(2 * kg.MAX_TURNS)
                    .execute().data or [])
        tiers = ("conversation_id, role, text_content, created_at, fallback, language, retrieval, latency_ms, total_ms",  # + migration 012
                 "conversation_id, role, text_content, created_at, fallback, language, latency_ms, total_ms",  # + migration 005
                 "conversation_id, role, text_content, created_at, latency_ms",
                 "conversation_id, role, text_content, created_at")
        for n, cols in enumerate(tiers):
            try:
                rows += fetch(cols)
                break
            except Exception:  # noqa: BLE001 - that migration may not be applied yet: read fewer columns
                if n == len(tiers) - 1:
                    raise
    return convs, {a["id"]: a.get("name") for a in agents}, rows


@router.get("/knowledge-gaps")
async def knowledge_gaps(request: Request, agent_id: Optional[str] = Query(None),
                         days: int = Query(30, ge=1, le=90), user_id: str = Depends(get_current_user_id)):
    """Questions the agents could not answer well, grouped by similarity. Owner-scoped; text is redacted first.
    Cached for a few minutes per (storage mode, user, agent, window)."""
    import backend as B  # lazy: backend.py loads this router while it is still being imported

    mode = B._storage_mode(request)
    key = (mode, user_id, agent_id or "", days)
    hit = kg.cache_get(key)
    if hit is not None:
        return hit
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    def load():
        convs, names, rows = _load_turns(B, mode, user_id, agent_id, since)
        return names, kg.pair_turns(rows, convs)

    names, pairs = await (asyncio.to_thread(load) if mode == "local" else B._cloud_run(load, "load knowledge gaps"))

    def embed(texts):  # the retrieval model; if it cannot load, build_clusters falls back to exact-text grouping
        return B.embedding_model.encode(texts, normalize_embeddings=True)

    result = await asyncio.to_thread(kg.build_report, pairs, names, embed, days)
    kg.cache_put(key, result)
    return result


def _bookings(since_day: str, agent_id: Optional[str]) -> Optional[dict]:
    """Meeting requests the HOD office agent put in. The meeting store is one office-wide file, not split by agent or
    owner, so the figure is only offered for "all agents" (None for one agent: not derivable, never faked)."""
    if agent_id:
        return None
    try:
        every = meetings.list_requests(None)
    except Exception:  # noqa: BLE001 - a missing or unreadable meetings file: report "not tracked", not a 500
        return None
    return {
        "requested": sum(1 for r in every if str(r.get("created_at") or "")[:10] >= since_day),
        "booked": sum(1 for r in every if r.get("status") == "approved"
                      and str(r.get("decided_at") or r.get("created_at") or "")[:10] >= since_day),
        "waiting": sum(1 for r in every if r.get("status") == "pending"),
    }


def build_results(convs: dict, tests: set, rows: list, days: int, now: datetime) -> dict:
    """The numbers behind the Results page, from one agent's (or every agent's) turns. Pure: no storage access.
    `convs` maps conversation id to agent id, `tests` holds the ids of owner Test calls: they are left out of the
    conversation and question counts, but their reply times are kept (Overview counts them too)."""
    since_day = (now - timedelta(days=days - 1)).date()
    real = {c: a for c, a in convs.items() if c not in tests}
    real_rows = [r for r in rows if r.get("conversation_id") in real]

    pairs = kg.pair_turns(real_rows, real)
    gaps, asked = kg.find_gaps(pairs)
    answerable = [p for p in pairs if not p["fallback"] and (p["question"] or "").strip()]  # what find_gaps counts

    day_asked: dict = defaultdict(int)
    day_missed: dict = defaultdict(int)
    langs: dict = defaultdict(int)
    for p in answerable:
        day_asked[p["created_at"][:10]] += 1
        if p.get("language"):
            langs[p["language"]] += 1
    for g in gaps:
        day_missed[g["created_at"][:10]] += 1

    totals = []
    for r in rows:
        if r.get("role") != "agent":
            continue
        v = perf._num(r.get("total_ms"))
        v = v if v is not None else perf._num(r.get("latency_ms"))
        if v is not None and v > 0:
            totals.append(v)

    daily = []
    for i in builtins.range(days):
        d = (since_day + timedelta(days=i)).isoformat()
        daily.append({"date": d, "asked": day_asked.get(d, 0), "unanswered": day_missed.get(d, 0)})
    conv_ids = {r["conversation_id"] for r in real_rows}
    return {
        "days": days,
        "conversations": len(conv_ids),
        "questions_asked": asked,
        "questions_answered": asked - len(gaps),
        "questions_unanswered": len(gaps),
        "reply_time": {
            "count": len(totals),
            "p50_ms": perf._percentile(totals, 50),
            "p95_ms": perf._percentile(totals, 95),
            "target_ms": perf.TARGET_MS,
            "within_target_pct": round(100 * sum(1 for x in totals if x <= perf.TARGET_MS) / len(totals), 1)
            if totals else None,
        },
        "daily": daily,
        "by_language": [{"language": k, "count": v} for k, v in sorted(langs.items(), key=lambda x: -x[1])],
    }


@router.get("/results")
async def results(request: Request, range: Literal["7d", "30d"] = Query("7d"), agent_id: Optional[str] = Query(None),
                  user_id: str = Depends(get_current_user_id)):
    """Is the agent working? Conversations, answered / unanswered questions, usual and slowest reply times and
    meeting requests, for the owner's agents (or one agent) over the last 7 or 30 days. Owner-scoped. Test calls are
    left out of the counts; reply times use the same conversation turns as GET /api/perf/turns (Overview).
    A figure that cannot be derived is null, never a made-up zero."""
    import backend as B  # lazy: backend.py loads this router while it is still being imported

    days = 7 if range == "7d" else 30
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    mode = B._storage_mode(request)
    tests: set = set()

    def load():
        return _load_turns(B, mode, user_id, agent_id, since.isoformat(), tests)

    convs, names, rows = await (asyncio.to_thread(load) if mode == "local" else B._cloud_run(load, "load results"))
    out = build_results(convs, tests, rows, days, now)
    out["range"] = range
    out["agent_id"] = agent_id
    out["agent_name"] = names.get(agent_id) if agent_id else None
    out["bookings"] = await asyncio.to_thread(_bookings, since.date().isoformat(), agent_id)
    return out
