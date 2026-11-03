"""Instant answers of an agent: reviewed, pre-written answers to the questions callers ask most (services/instant_answers.py).

    GET    /api/agents/{id}/instant-answers          the answers, how the last build went, and whether one is running
    POST   /api/agents/{id}/instant-answers/build    draft answers from the documents (owner-triggered, rate limited, runs
                                                     in the background; poll the GET for progress; ?wait=true waits)
    POST   /api/agents/{id}/instant-answers          add a question (with an answer it is approved at once)
    PATCH  /api/agents/{id}/instant-answers/{aid}    edit the question or answer, approve, turn off (PUT does the same)
    DELETE /api/agents/{id}/instant-answers/{aid}    delete

Every route needs the agent's owner. Only "approved" answers are used on calls, and an approved answer whose documents
changed becomes "needs_review" (seen by the next GET, and by `match` right away). Nothing here touches the turn pipeline:
the lead calls `services.instant_answers.match` (see its docstring)."""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, Body, Depends, Query, Request
from pydantic import BaseModel, Field

try:
    from core.errors import AppError
    from services import instant_answers as ia
    from services import knowledge_gaps as kg
    from services import language_policy, llm
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover - imported as backend.routers.instant_answers
    from backend.core.errors import AppError
    from backend.services import instant_answers as ia
    from backend.services import knowledge_gaps as kg
    from backend.services import language_policy, llm
    from backend.services.auth import get_current_user_id

log = logging.getLogger("bonclay.instant_answers")
router = APIRouter(tags=["instant-answers"])

CALL_WINDOW_DAYS = 90
_tasks: set = set()  # strong references so background builds are not garbage collected
_jobs: Dict[Tuple[str, str], dict] = {}  # (user id, agent id) -> the latest build


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
    """The app's rate limiter (same buckets as every other endpoint); a no-op if it cannot be found."""
    try:
        return _b().limiter.limit(rule)
    except Exception:  # noqa: BLE001
        return lambda fn: fn


def _mode(request: Request) -> str:
    return _b()._storage_mode(request)


def _store(request: Request) -> Any:
    B = _b()
    ia.configure(supabase=lambda: getattr(B, "supabase", None))
    return ia.get_store(_mode(request))


async def _run(request: Request, fn, what: str):
    """Run a blocking store call off the event loop; in hosted mode a failure is a 500 DB_ERROR, never a crash."""
    if _mode(request) == "local":
        return await asyncio.to_thread(fn)
    return await _b()._cloud_run(fn, what)


async def _agent(request: Request, agent_id: str, user_id: str) -> dict:
    return await _b()._load_agent(request, agent_id, user_id)


def _row_id(value: str) -> str:
    return _b()._check_id(value, "id")


def _fingerprint(agent_id: str, user_id: str, mode: str):
    return asyncio.to_thread(ia.current_fingerprint, agent_id, user_id, mode)


# ------------------------------------------------------------------------------------------------------ list

def _job_view(job: Optional[dict]) -> Optional[dict]:
    if job is None:
        return None
    return {k: job.get(k) for k in ("state", "total", "done", "current", "added", "updated", "skipped", "error",
                                    "started_at", "finished_at")}


async def _refresh_review(request: Request, store: Any, agent_id: str, user_id: str, rows: List[dict]) -> List[dict]:
    """Approved answers whose documents changed since the owner approved them become "needs_review" (saved)."""
    fp = await _fingerprint(agent_id, user_id, _mode(request))
    ups = ia.review_updates(rows, fp)
    if ups:
        await _run(request, lambda: store.update_many(user_id, agent_id, ups), "mark instant answers for review")
        rows = [{**r, **ups.get(r["id"], {})} for r in rows]
    return rows


@router.get("/api/agents/{agent_id}/instant-answers")
async def list_instant_answers(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    agent = await _agent(request, agent_id, user_id)
    store = _store(request)
    rows = await _run(request, lambda: store.list(user_id, agent_id), "list instant answers")
    rows = await _refresh_review(request, store, agent_id, user_id, rows)
    shown = sorted(ia.visible(rows), key=lambda r: (r.get("created_at") or ""), reverse=True)
    counts = {s: sum(1 for r in shown if r.get("status") == s) for s in ia.STATUSES}
    B = _b()
    return {
        "answers": [ia.public_view(r) for r in shown],
        "counts": counts,
        "build": _job_view(_jobs.get((user_id, agent_id))),
        "document_count": await B._document_count(request, agent_id),
        "has_tools": bool(agent.get("tools")),
        "threshold": ia.threshold(),
        "enabled": ia.enabled(),
    }


# ------------------------------------------------------------------------------------------------------ add / edit

class AddBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=ia.MAX_QUESTION_CHARS + 200)
    answer: Optional[str] = Field(None, max_length=ia.MAX_ANSWER_CHARS + 500)
    language: Optional[str] = Field(None, max_length=12)


class PatchBody(BaseModel):
    question: Optional[str] = Field(None, max_length=ia.MAX_QUESTION_CHARS + 200)
    answer: Optional[str] = Field(None, max_length=ia.MAX_ANSWER_CHARS + 500)
    status: Optional[Literal["approved", "disabled", "suggested"]] = None
    paraphrases: Optional[List[str]] = Field(None, max_length=10)


def _check_question(text: str, agent: dict) -> str:
    q = " ".join((text or "").split())
    if not q:
        raise AppError("INVALID_QUESTION", "Write the question callers ask.", 422, {"field": "question"})
    problem = ia.question_problem(q, has_tools=bool(agent.get("tools")))
    if problem:
        raise AppError("INVALID_QUESTION", f"This cannot be an instant answer: {problem}.", 422, {"field": "question"})
    return q


def _check_answer(text: str) -> str:
    a = ia.clean_answer(text)
    if len(a) > ia.MAX_ANSWER_CHARS or ia.word_count(a) > ia.MAX_ANSWER_WORDS:
        raise AppError("INVALID_ANSWER", f"Keep the answer short enough to say aloud (at most {ia.MAX_ANSWER_WORDS} words).",
                       422, {"field": "answer"})
    if a and kg.is_gap_reply(a):
        raise AppError("INVALID_ANSWER", "An answer that says the agent does not know cannot be an instant answer.", 422,
                       {"field": "answer"})
    return a


def _clean_lang(code: Optional[str], question: str, agent: dict) -> str:
    if code:
        c = ia.lang_code(code)
        if c.isalpha() and 2 <= len(c) <= 8:
            return c
        raise AppError("INVALID_LANGUAGE", "Unknown language code.", 422, {"field": "language"})
    return ia.lang_code(language_policy.detect_language(question, agent.get("supported_languages") or ["en"]))


@router.post("/api/agents/{agent_id}/instant-answers")
async def add_instant_answer(agent_id: str, body: AddBody, request: Request, user_id: str = Depends(get_current_user_id)):
    agent = await _agent(request, agent_id, user_id)
    store = _store(request)
    mode = _mode(request)
    question = _check_question(body.question, agent)
    answer = _check_answer(body.answer or "")
    lang = _clean_lang(body.language, question, agent)
    rows = await _run(request, lambda: store.list(user_id, agent_id), "list instant answers")
    same = next((r for r in rows if ia.qkey(r["question"]) == ia.qkey(question)
                 and ia.lang_code(r.get("language")) == lang), None)
    if same is not None and same.get("status") != ia.STATUS_DISMISSED:
        raise AppError("DUPLICATE_QUESTION", "You already have this question.", 409, {"id": same["id"]})
    if len([r for r in rows if r.get("status") != ia.STATUS_DISMISSED]) >= ia.MAX_ROWS_PER_AGENT:
        raise AppError("TOO_MANY_ANSWERS", f"An agent can have at most {ia.MAX_ROWS_PER_AGENT} instant answers.", 409)
    fp = await _fingerprint(agent_id, user_id, mode)
    row = await asyncio.to_thread(
        lambda: ia.new_row(agent_id=agent_id, user_id=user_id, question=question, answer=answer, language=lang,
                           source="owner", status=ia.STATUS_APPROVED if answer else ia.STATUS_SUGGESTED, fingerprint=fp))
    if same is not None:  # a deleted question the owner now wants back
        saved = await _run(request, lambda: store.update(user_id, agent_id, same["id"], {**row, "created_at": ia.now_iso()}),
                           "add instant answer")
    else:
        saved = await _run(request, lambda: store.insert(row), "add instant answer")
    return ia.public_view(saved)


async def _patch(agent_id: str, aid: str, body: PatchBody, request: Request, user_id: str) -> dict:
    agent = await _agent(request, agent_id, user_id)
    aid = _row_id(aid)
    store = _store(request)
    row = await _run(request, lambda: store.get(user_id, agent_id, aid), "load instant answer")
    if row is None or row.get("status") == ia.STATUS_DISMISSED:
        raise AppError("INSTANT_ANSWER_NOT_FOUND", "Instant answer not found", 404)
    fields: Dict[str, Any] = {}
    reembed = False

    if body.question is not None and ia.qkey(body.question) != ia.qkey(row["question"]):
        question = _check_question(body.question, agent)
        rows = await _run(request, lambda: store.list(user_id, agent_id), "list instant answers")
        lang = ia.lang_code(row.get("language"))
        if any(r["id"] != aid and r.get("status") != ia.STATUS_DISMISSED and ia.qkey(r["question"]) == ia.qkey(question)
               and ia.lang_code(r.get("language")) == lang for r in rows):
            raise AppError("DUPLICATE_QUESTION", "You already have this question.", 409)
        fields["question"] = question
        fields["paraphrases"] = []  # the other wordings belonged to the old question
        if row.get("status") == ia.STATUS_APPROVED and body.status is None:
            fields["status"] = ia.STATUS_SUGGESTED  # the answer was approved for another question: look at it again
        reembed = True
    if body.paraphrases is not None:
        fields["paraphrases"] = ia.clean_paraphrases(fields.get("question") or row["question"], body.paraphrases)
        reembed = True
    if body.answer is not None:
        answer = _check_answer(body.answer)
        if answer != (row.get("answer") or ""):
            fields["answer"] = answer
            reembed = True

    merged = {**row, **fields}
    if body.status is not None:
        if body.status == ia.STATUS_APPROVED:
            if not (merged.get("answer") or "").strip():
                raise AppError("ANSWER_REQUIRED", "Write an answer before approving this question.", 422, {"field": "answer"})
            fields["docs_fingerprint"] = await _fingerprint(agent_id, user_id, _mode(request))
            fields["approved_at"] = ia.now_iso()
            reembed = reembed or not row.get("vectors")
        fields["status"] = body.status
    elif row.get("status") in (ia.STATUS_APPROVED,) and "answer" in fields:
        # an owner's edit of a live answer: they have just read it against the documents as they are now
        fields["docs_fingerprint"] = await _fingerprint(agent_id, user_id, _mode(request))

    if reembed and (merged.get("answer") or "").strip():
        fields["vectors"], fields["embed_model"] = None, None  # never keep the old question's vectors
        fields.update(await asyncio.to_thread(ia.embed_row, {**merged, **fields}))
    elif reembed:
        fields["vectors"], fields["embed_model"] = None, None
    if not fields:
        return ia.public_view(row)
    saved = await _run(request, lambda: store.update(user_id, agent_id, aid, fields), "update instant answer")
    if saved is None:
        raise AppError("INSTANT_ANSWER_NOT_FOUND", "Instant answer not found", 404)
    return ia.public_view(saved)


@router.patch("/api/agents/{agent_id}/instant-answers/{aid}")
async def patch_instant_answer(agent_id: str, aid: str, body: PatchBody, request: Request,
                               user_id: str = Depends(get_current_user_id)):
    return await _patch(agent_id, aid, body, request, user_id)


@router.put("/api/agents/{agent_id}/instant-answers/{aid}")
async def put_instant_answer(agent_id: str, aid: str, body: PatchBody, request: Request,
                             user_id: str = Depends(get_current_user_id)):
    """Same as PATCH (the web client has no PATCH helper)."""
    return await _patch(agent_id, aid, body, request, user_id)


@router.delete("/api/agents/{agent_id}/instant-answers/{aid}")
async def delete_instant_answer(agent_id: str, aid: str, request: Request, user_id: str = Depends(get_current_user_id)):
    await _agent(request, agent_id, user_id)
    aid = _row_id(aid)
    store = _store(request)
    row = await _run(request, lambda: store.get(user_id, agent_id, aid), "load instant answer")
    if row is None or row.get("status") == ia.STATUS_DISMISSED:
        raise AppError("INSTANT_ANSWER_NOT_FOUND", "Instant answer not found", 404)
    if row.get("source") == "owner":
        await _run(request, lambda: store.delete(user_id, agent_id, aid), "delete instant answer")
    else:
        # Kept hidden, so the next build does not suggest the same template or caller question again.
        await _run(request, lambda: store.update(user_id, agent_id, aid, {
            "status": ia.STATUS_DISMISSED, "answer": "", "paraphrases": [], "vectors": None, "embed_model": None}),
            "delete instant answer")
    return {"deleted": True}


# ------------------------------------------------------------------------------------------------------ build

class BuildBody(BaseModel):
    languages: Optional[List[str]] = Field(None, max_length=6, description="Also write each answer in these languages")
    include_template: bool = True
    include_calls: bool = True


def _llm_kwargs(B, **kw) -> dict:
    return {"timeout": B._llm_timeout(30.0), **kw}


def _make_deps(B, agent: dict, user_id: str, mode: str, loop: asyncio.AbstractEventLoop) -> "ia.DraftDeps":
    """The model side of a build, in the agent's normal grounded path: its retrieval, its system prompt (persona, rules,
    language policy) and its language model. Everything blocks, and runs in a worker thread."""
    agent_id = agent["id"]

    def retrieve(question: str):
        signals: dict = {}
        chunks = asyncio.run_coroutine_threadsafe(
            B.retrieve_context(agent_id, question, user_id=user_id, storage_mode=mode, signals=signals), loop).result(180)
        if signals.get("error"):
            raise RuntimeError("the documents could not be searched")
        return chunks, signals.get("top_score")

    def messages_for(question: str, language: str, chunks: List[str]) -> List[dict]:
        return [{"role": "system", "content": B._agent_system_prompt(agent, None, language)},
                {"role": "user", "content": B._turn_user_content(question, chunks)}]

    def answer(question: str, language: str, chunks: List[str]) -> str:
        msgs = messages_for(question, language, chunks)
        text = (llm.chat(msgs, **_llm_kwargs(B, temperature=0.2, max_tokens=160)).text or "").strip()

        def regen() -> str:
            m = language_policy.regen_messages(msgs, B.LANGUAGE_NAMES.get(language, language), text)
            return (llm.chat(m, **_llm_kwargs(B, temperature=0.2, max_tokens=160)).text or "").strip()

        fixed, _info = language_policy.enforce(language, text, regen)
        return fixed

    def paraphrase(question: str, language: str) -> List[str]:
        name = B.LANGUAGE_NAMES.get(language, language)
        prompt = (f"A caller phones a business and asks: \"{question}\"\n"
                  f"Write {ia.MAX_PARAPHRASES} different ways another caller could ask exactly the same thing on the phone, "
                  f"in {name}. Keep every day, number, name and the meaning exactly the same, and never add or remove a "
                  "\"not\". One short question each, no lists. Reply with JSON only: {\"paraphrases\": [\"...\", \"...\", \"...\"]}")
        out = llm.chat([{"role": "system", "content": "You rewrite questions. Reply with one JSON object."},
                        {"role": "user", "content": prompt}], json_mode=True,
                       **_llm_kwargs(B, temperature=0.5, max_tokens=220))
        data = llm.extract_json(out.text)
        items = data.get("paraphrases")
        return [str(x) for x in items] if isinstance(items, list) else []

    return ia.DraftDeps(retrieve=retrieve, answer=answer, paraphrase=paraphrase, embed=ia.default_embed)


def _call_pairs(B, mode: str, user_id: str, agent_id: str) -> List[dict]:
    """Blocking: this agent's real (non-test) caller questions of the last 90 days, paired with the agent's replies."""
    from routers import analytics  # noqa: PLC0415 - the same loader the knowledge-gap report uses
    since = (datetime.now(timezone.utc) - timedelta(days=CALL_WINDOW_DAYS)).isoformat()
    convs, _names, rows = analytics._load_turns(B, mode, user_id, agent_id, since)  # noqa: SLF001
    return kg.pair_turns(rows, convs)


def _templates() -> List[dict]:
    try:
        from routers import templates  # noqa: PLC0415
        return templates._load_templates()  # noqa: SLF001
    except Exception as e:  # noqa: BLE001 - suggestions from templates are optional
        log.warning("could not load the agent templates: %s", e)
        return []


def _collect(B, mode: str, agent: dict, user_id: str, body: BuildBody, store: Any) -> List["ia.Candidate"]:
    """Blocking: what to draft, in order: template questions, frequent real questions, then the owner's own questions and
    answers that need a new draft."""
    agent_id = agent["id"]
    existing = store.list(user_id, agent_id)
    groups: List[List[ia.Candidate]] = []
    if body.include_template:
        groups.append(ia.template_candidates(agent, _templates()))
    if body.include_calls:
        try:
            pairs = _call_pairs(B, mode, user_id, agent_id)
            popular = ia.popular_questions(pairs, ia.default_embed)
        except Exception as e:  # noqa: BLE001 - no call history (or an unreachable store) must not stop a build
            log.warning("could not read caller questions for instant answers: %s", e)
            popular = []
        groups.append([ia.Candidate(question=q, source="calls", count=n) for q, n in popular])
    redo = [ia.Candidate(question=r["question"], source=r.get("source") or "owner", count=int(r.get("times_asked") or 0),
                         row_id=r["id"], language=r.get("language"))
            for r in existing
            if r.get("status") == ia.STATUS_REVIEW or (r.get("status") == ia.STATUS_SUGGESTED and not (r.get("answer") or "").strip())]
    groups.insert(0, redo)
    return ia.merge_candidates(groups, existing, ia.default_embed, limit=ia.build_limit(),
                               supported=agent.get("supported_languages") or ["en"])


async def _run_build(request_mode: str, agent: dict, user_id: str, body: BuildBody, store: Any, job: dict) -> None:
    B = _b()
    agent_id = agent["id"]
    loop = asyncio.get_running_loop()
    try:
        job["state"] = "collecting"
        cands = await asyncio.to_thread(_collect, B, request_mode, agent, user_id, body, store)
        languages = [ia.lang_code(x) for x in (body.languages or [])]
        job["total"] = sum(len({ia.lang_code(c.language), *languages}) for c in cands)
        if not cands:
            job.update(state="done", total=0, done=0)
            return
        fp = await asyncio.to_thread(ia.current_fingerprint, agent_id, user_id, request_mode)
        deps = _make_deps(B, agent, user_id, request_mode, loop)
        job["state"] = "running"

        def progress(done: int, total: int, current: str) -> None:
            job.update(done=done, total=total, current=current)

        report = await asyncio.to_thread(
            lambda: ia.build(agent=agent, user_id=user_id, storage_mode=request_mode, candidates=cands,
                             languages=languages or None, deps=deps, fingerprint=fp, store=store, progress=progress))
        job.update(state="done", added=report.added, updated=report.updated, skipped=report.skipped[:50])
    except llm.LLMUnavailable:
        job.update(state="failed", error=getattr(B, "LLM_UNAVAILABLE_MSG", "The language model is not available right now."))
    except Exception as e:  # noqa: BLE001 - the owner sees a plain message; the details go to the log
        log.exception("instant answers build failed")
        job.update(state="failed", error=f"The build stopped: {str(e)[:160] or 'unexpected error'}")
    finally:
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        job["current"] = ""


@router.post("/api/agents/{agent_id}/instant-answers/build", status_code=202)
@_limit("10/hour")
async def build_instant_answers(agent_id: str, request: Request, body: Optional[BuildBody] = Body(None),
                                wait: bool = Query(False), user_id: str = Depends(get_current_user_id)):
    body = body or BuildBody()
    agent = await _agent(request, agent_id, user_id)
    B = _b()
    if await B._document_count(request, agent_id) == 0:
        raise AppError("NO_DOCUMENTS", "Upload a document first: instant answers are written from your documents.", 409)
    key = (user_id, agent_id)
    current = _jobs.get(key)
    if current and current.get("state") in ("collecting", "running"):
        raise AppError("BUILD_RUNNING", "Suggestions are already being built for this agent.", 409)
    store = _store(request)
    job = {"state": "collecting", "total": 0, "done": 0, "current": "", "added": 0, "updated": 0, "skipped": [],
           "error": None, "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None}
    _jobs[key] = job
    mode = _mode(request)
    if wait:
        await _run_build(mode, agent, user_id, body, store, job)
    else:
        task = asyncio.create_task(_run_build(mode, agent, user_id, body, store, job))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    return {"build": _job_view(job)}
