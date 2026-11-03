"""Answer audit: the replies a conversation gave from the owner's documents, checked after the conversation (never live)
against the passages the agent was given (services/answer_audit.py). The Results page lists the flagged ones.

    GET   /api/answer-audit?agent_id=&days=30   flagged answers (question, reply, what the document says, score), counts,
                                                whether the checking model is on this machine, and the latest run
    POST  /api/answer-audit/run                 "Check recent answers": owner-triggered, rate limited, runs in the
                                                background (poll the GET; ?wait=true waits). Body: {agent_id?, days?}
    PATCH /api/answer-audit/{turn_id}           {"feedback": "fine" | "wrong" | null}: the owner's verdict (PUT does the same)

Every route needs the owner. A background job (started with the app) checks each conversation after it ends: a phone call
45 seconds after its last turn, anything else after 2 minutes. It waits until the owner has run the check once, because
that first run downloads the checking model (see the service docstring)."""
import asyncio
import logging
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Body, Depends, Query, Request
from pydantic import BaseModel, Field

try:
    from core.errors import AppError
    from services import answer_audit as audit
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover - imported as backend.routers.answer_audit
    from backend.core.errors import AppError
    from backend.services import answer_audit as audit
    from backend.services.auth import get_current_user_id

log = logging.getLogger("bonclay.answer_audit")
router = APIRouter(tags=["answer-audit"])

NOTE = "Checked automatically in English; Hindi and Hinglish answers are not checked yet."
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_tasks: set = set()                     # strong references so background runs are not garbage collected
_runs: Dict[str, dict] = {}             # user id -> the latest owner-triggered run
_loop_task: Optional[asyncio.Task] = None


def _b():
    """The backend module (imported lazily; also works when backend.py runs as __main__)."""
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "limiter"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "limiter"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


def _limit(rule: str):
    try:
        return _b().limiter.limit(rule)
    except Exception:  # noqa: BLE001
        return lambda fn: fn


def _mode(request: Request) -> str:
    return _b()._storage_mode(request)


def _store(mode: str) -> Any:
    B = _b()
    audit.configure(supabase=lambda: getattr(B, "supabase", None))
    return audit.get_store(mode)


async def _run(mode: str, fn, what: str):
    """A blocking store or database call off the event loop; hosted mode turns a failure into a 500 DB_ERROR."""
    if mode == "local":
        return await asyncio.to_thread(fn)
    return await _b()._cloud_run(fn, what)


def _since(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _backends(mode: str):
    """(load(user_id, agent_id, since_iso), chunk source) for a storage mode."""
    B = _b()
    if mode == "local" or getattr(B, "supabase", None) is None:
        db = B.local_db
        return (lambda uid, aid, since: audit.load_local(db, uid, aid, since)), audit.LocalChunks(db)
    sb = B.supabase
    return (lambda uid, aid, since: audit.load_cloud(sb, uid, aid, since)), audit.CloudChunks(sb)


def _run_view(run: Optional[dict]) -> Optional[dict]:
    if run is None:
        return None
    return {k: run.get(k) for k in ("id", "state", "total", "done", "checked", "flagged", "skipped", "language", "error",
                                    "started_at", "finished_at")}


# ------------------------------------------------------------------------------------------------------ read

@router.get("/api/answer-audit")
async def list_flagged(request: Request, agent_id: Optional[str] = Query(None, max_length=64),
                       days: int = Query(30, ge=1, le=365), user_id: str = Depends(get_current_user_id)):
    mode = _mode(request)
    agent_names: Dict[str, str] = {}
    if agent_id:
        agent_id = _b()._check_id(agent_id, "agent_id")
        agent = await _b()._load_agent(request, agent_id, user_id)   # 404 unless it is the owner's agent
        agent_names[agent_id] = agent.get("name") or ""
    store = _store(mode)
    rows = await _run(mode, lambda: store.list(user_id, agent_id, _since(days)), "list answer audit")
    shown = [audit.public_view(r) for r in audit.flagged_rows(rows)]
    if shown and not agent_id:
        names = await _agent_names(request, mode, user_id)
        agent_names.update(names)
    for v in shown:
        v["agent_name"] = agent_names.get(v.get("agent_id") or "")
    return {
        "flagged": shown, "counts": audit.summarize(rows), "threshold": audit.threshold(), "days": days,
        "enabled": audit.enabled(), "auto": audit.auto_ready(), "model_ready": audit.model_ready(),
        "model": audit.model_label(), "note": NOTE, "run": _run_view(_runs.get(user_id)),
    }


async def _agent_names(request: Request, mode: str, user_id: str) -> Dict[str, str]:
    B = _b()
    try:
        if mode == "local" or getattr(B, "supabase", None) is None:
            rows = await asyncio.to_thread(B.local_db.select, "agents", lambda r: r.get("user_id") == user_id)
        else:
            rows = await B._cloud_run(lambda: B.supabase.table("agents").select("id, name").eq("user_id", user_id)
                                      .execute().data or [], "list agents")
        return {r["id"]: r.get("name") or "" for r in rows}
    except Exception:  # noqa: BLE001 - names are decoration
        return {}


# ------------------------------------------------------------------------------------------------------ run

class RunBody(BaseModel):
    agent_id: Optional[str] = Field(None, max_length=64)
    days: int = Field(7, ge=1, le=90)


async def _do_run(run: dict, mode: str, user_id: str, agent_id: Optional[str], days: int) -> None:
    try:
        load, chunks = _backends(mode)
        store = _store(mode)
        convs = await _run(mode, lambda: load(user_id, agent_id, _since(days)), "load conversations")
        known = await _run(mode, lambda: store.known(user_id), "list answer audit")
        cands = audit.collect(convs, known, limit=audit.run_max(), agent_id=agent_id)
        run["total"] = len(cands)

        def progress(s: Dict[str, int]) -> None:
            run.update({k: s[k] for k in ("total", "done", "checked", "flagged", "skipped", "language")})

        await asyncio.to_thread(audit.run_candidates, cands, lambda _c: chunks, audit.get_verifier(),
                                lambda _c: store, progress=progress)
        run["state"] = "done"
    except AppError as e:   # e.g. not enough free memory for the model right now
        run.update(state="error", error=e.message if hasattr(e, "message") else str(e))
        log.warning("answer audit run stopped: %s", e)
    except Exception as e:  # noqa: BLE001 - a failed run never takes the app down
        run.update(state="error", error="The check could not finish. Try again in a moment.")
        log.warning("answer audit run failed: %s", e, exc_info=True)
    finally:
        run["finished_at"] = audit.now_iso()


@router.post("/api/answer-audit/run", status_code=202)
@_limit("6/hour")
async def run_audit(request: Request, body: Optional[RunBody] = Body(None), wait: bool = Query(False),
                    user_id: str = Depends(get_current_user_id)):
    body = body or RunBody()
    if not audit.enabled():
        raise AppError("ANSWER_AUDIT_OFF", "The answer check is turned off on this server.", 409)
    mode = _mode(request)
    agent_id = body.agent_id
    if agent_id:
        agent_id = _b()._check_id(agent_id, "agent_id")
        await _b()._load_agent(request, agent_id, user_id)
    current = _runs.get(user_id)
    if current is not None and current.get("state") == "running":
        raise AppError("AUDIT_RUNNING", "A check is already running. Wait for it to finish.", 409)
    run = {"id": uuid.uuid4().hex[:12], "state": "running", "total": 0, "done": 0, "checked": 0, "flagged": 0,
           "skipped": 0, "language": 0, "error": None, "started_at": audit.now_iso(), "finished_at": None}
    _runs[user_id] = run
    task = asyncio.create_task(_do_run(run, mode, user_id, agent_id, body.days))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    if wait:
        await task
    return {"run": _run_view(run), "first_run_downloads_model": not audit.model_ready()}


# ------------------------------------------------------------------------------------------------------ feedback

class FeedbackBody(BaseModel):
    feedback: Optional[Literal["fine", "wrong"]] = None


async def _feedback(turn_id: str, body: FeedbackBody, request: Request, user_id: str) -> dict:
    if not _ID.match(turn_id or ""):
        raise AppError("INVALID_ID", "Invalid id", 422, {"field": "turn_id"})
    mode = _mode(request)
    store = _store(mode)
    row = await _run(mode, lambda: store.set_feedback(user_id, turn_id, body.feedback), "save answer feedback")
    if row is None:
        raise AppError("AUDIT_NOT_FOUND", "That checked answer was not found.", 404)
    return audit.public_view(row)


@router.patch("/api/answer-audit/{turn_id}")
async def patch_feedback(turn_id: str, body: FeedbackBody, request: Request, user_id: str = Depends(get_current_user_id)):
    return await _feedback(turn_id, body, request, user_id)


@router.put("/api/answer-audit/{turn_id}")
async def put_feedback(turn_id: str, body: FeedbackBody, request: Request, user_id: str = Depends(get_current_user_id)):
    return await _feedback(turn_id, body, request, user_id)


# ------------------------------------------------------------------------------------------------------ background job

async def sweep_once(now: Optional[datetime] = None) -> Dict[str, Any]:
    """Check the answers of conversations that have ended and were not checked yet (every store this server has). Skips the
    sweep while a live conversation is using the machine, when the job is off, or when the model is not on this machine
    yet. Returns what it did (for logs and tests)."""
    if not audit.auto_ready():
        return {"skipped": "off or model not downloaded"}
    B = _b()
    sources = [("local", B.local_db)] if getattr(B, "local_db", None) is not None else []
    if getattr(B, "supabase", None) is not None:
        sources.append(("cloud", B.supabase))
    due_now = audit.take_due_now()
    since = _since(audit.sweep_days())
    total = {"checked": 0, "flagged": 0, "skipped": 0, "language": 0, "done": 0}
    for mode, _handle in sources:
        try:
            load, chunks = _backends(mode)
            store = _store(mode)
            convs = await asyncio.to_thread(load, None, None, since)
            if audit.live_traffic(convs, now):
                return {**total, "skipped": "a live conversation is in progress"}
            ended = [(c, t) for c, t in convs if audit.conversation_over(c, t, now, due_now)]
            known = await asyncio.to_thread(store.known, None)
            cands = audit.collect(ended, known, limit=audit.run_max())
            if not cands:
                continue
            stats = await asyncio.to_thread(audit.run_candidates, cands, lambda _c: chunks, audit.get_verifier(),
                                            lambda _c: store)
            for k in total:
                total[k] += stats.get(k, 0)
            audit._due_now.difference_update({str(c["id"]) for c, _ in ended})  # noqa: SLF001
        except AppError as e:
            log.warning("answer audit sweep (%s) stopped: %s", mode, e)
        except Exception as e:  # noqa: BLE001 - e.g. the hosted table is missing: logged, never fatal
            log.warning("answer audit sweep (%s) failed: %s", mode, e)
    return total


async def _loop() -> None:
    await asyncio.sleep(60)   # let the app finish starting before any checking
    while True:
        try:
            await sweep_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("answer audit job failed: %s", e)
        await asyncio.sleep(audit.poll_s())


async def _start_job() -> None:
    global _loop_task
    import os  # noqa: PLC0415
    if os.getenv("PYTEST_CURRENT_TEST") or not audit.auto_enabled() or _loop_task is not None:
        return
    _loop_task = asyncio.get_running_loop().create_task(_loop())


async def _stop_job() -> None:
    global _loop_task
    if _loop_task is not None:
        _loop_task.cancel()
        _loop_task = None
    verifier = audit._verifier  # noqa: SLF001
    if verifier is not None and hasattr(verifier, "unload"):
        verifier.unload()


def install(app) -> None:
    """Called by backend.py after the router is included: starts the background job with the app (the same
    `app.on_event` API backend.py's own startup work uses)."""
    app.on_event("startup")(_start_job)
    app.on_event("shutdown")(_stop_job)
