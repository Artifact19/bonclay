"""Ready-made agent templates and the one-click demo.

Templates live in backend/templates/agents/ (a JSON file plus a sample knowledge document each).
Creating an agent from one needs no language model: the persona is stored as written and the document is
chunked and embedded with the normal knowledge pipeline (`ingest_document`).

The backend module is imported lazily inside handlers so this router never creates an import cycle."""
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel

log = logging.getLogger("bonclay.templates")

router = APIRouter(tags=["templates"])

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "agents"
DEMO_TEMPLATES = ("clinic_receptionist", "college_admissions")
DEMO_EVAL_TEMPLATE = "clinic_receptionist"

# agent_id -> "running" | "done" | "failed" for the evaluation started by /api/demo/load.
_eval_state: Dict[str, str] = {}
_tasks: set = set()  # strong references so background tasks are not garbage collected


def _b():
    """The backend module (imported lazily; also works when backend.py runs as __main__)."""
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "ingest_document"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "ingest_document"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


async def current_user(request: Request) -> str:
    return await _b().get_current_user(request, None)


def _load_templates() -> List[dict]:
    out = []
    for path in sorted(TEMPLATES_DIR.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            log.error("Bad agent template %s: %s", path.name, e)
    return out


def _find_template(template_id: str) -> Optional[dict]:
    return next((t for t in _load_templates() if t.get("id") == template_id), None)


def _public(t: dict) -> dict:
    out = {k: t.get(k) for k in ("id", "name", "role", "languages", "greeting", "document", "suggested_questions")}
    out["sample_pdf"] = f"/api/agent-templates/{t['id']}/sample.pdf" if t.get("sample_pdf") else None
    # What the "Use template" form needs: who the template is for, how to ask for the organisation name, and the
    # sample names to swap out (the page previews the greeting with the same rules the server applies).
    p = t.get("personalise") or {}
    out["audience"] = t.get("audience") or p.get("audience") or t.get("role")
    out["org_label"] = p.get("org_label") or "Organisation name"
    out["org_placeholder"] = p.get("org_placeholder") or ""
    out["personalise"] = {
        "name_pattern": p.get("name_pattern") or "{org}",
        "sample_orgs": list(p.get("sample_orgs") or []),
        "sample_city": p.get("sample_city"),
        "strip": list(p.get("strip") or []),
    }
    return out


def _not_found(template_id: str):
    return _b().AppError("TEMPLATE_NOT_FOUND", "Agent template not found", 404, {"template_id": template_id})


SAMPLE_PREFIX = "Sample - "
SAMPLE_BANNER = ("[SAMPLE DOCUMENT: the names, people, prices and dates below are made up so you can try the agent. "
                 "Replace this document with your own details before real callers use the agent.]\n\n")
MAX_ORG, MAX_CITY, MAX_DETAILS = 80, 60, 1000
_LANG_CODE = re.compile(r"^[a-z]{2,3}$")


class FromTemplateRequest(BaseModel):
    template_id: str
    voice_id: Optional[str] = None
    # Optional personalisation. With `org_name` the sample organisation is replaced everywhere (agent name, greeting,
    # instructions) and the sample document is labelled as a sample; `include_sample=false` leaves it out.
    org_name: Optional[str] = None
    city: Optional[str] = None
    details: Optional[str] = None  # opening hours, or what callers usually ask
    languages: Optional[List[str]] = None  # ISO 639-1 codes
    include_sample: Optional[bool] = None


def _clean(value: Optional[str], limit: int, field: str) -> str:
    v = re.sub(r"[ \t]+", " ", (value or "").replace("\r", "")).strip()
    if len(v) > limit:
        raise _b().AppError("INVALID_TEMPLATE_INPUT", f"{field} is too long (at most {limit} characters).", 422,
                            {"field": field})
    return v


def _swap(text: str, names: List[str], new: str) -> str:
    """Replace every sample name in one pass, longest first, so a replacement is never scanned again."""
    names = sorted({n for n in names if n}, key=len, reverse=True)
    if not names:
        return text
    return re.sub("|".join(re.escape(n) for n in names), lambda _m: new, text)


def personalise(tpl: dict, org: str, city: str = "") -> Dict[str, str]:
    """The agent's name, greeting and instructions with the template's sample organisation replaced by `org`."""
    p = tpl.get("personalise") or {}
    names = list(p.get("sample_orgs") or [])
    out = {}
    for key in ("greeting", "system_instructions"):
        text = tpl[key]
        for phrase in p.get("strip") or []:
            text = text.replace(phrase, "")
        sample_city = p.get("sample_city")
        if sample_city and key == "system_instructions":
            text = text.replace(f" in {sample_city}", f" in {city}" if city else "")
        out[key] = _swap(text, names, org)
    if not names:  # e.g. a tutor whose persona is not an organisation: say who it speaks for instead
        out["system_instructions"] += f" You speak on behalf of {org}."
    if city and not (p.get("sample_city")):
        out["system_instructions"] += f" The organisation is in {city}."
    out["name"] = (p.get("name_pattern") or "{org}").replace("{org}", org)[:120]
    return out


def _about_document(org: str, city: str, details: str) -> str:
    lines = [f"# About {org}", ""]
    if city:
        lines.append(f"Location: {city}")
    if details:
        lines.append(f"Opening hours and what callers usually ask: {details}")
    return "\n".join(lines) + "\n"


async def _create_from_template(tpl: dict, request: Request, user_id: str, voice_id: Optional[str],
                                extra: Optional[Dict[str, Any]] = None,
                                body: Optional[FromTemplateRequest] = None) -> dict:
    B = _b()
    org = _clean(body.org_name, MAX_ORG, "org_name") if body else ""
    city = _clean(body.city, MAX_CITY, "city") if body else ""
    details = _clean(body.details, MAX_DETAILS, "details") if body else ""
    langs = None
    if body and body.languages is not None:
        langs = list(dict.fromkeys(str(x).strip().lower() for x in body.languages if str(x).strip()))
        if not langs or len(langs) > 8 or not all(_LANG_CODE.match(x) for x in langs):
            raise B.AppError("INVALID_TEMPLATE_INPUT", "Pick at least one language.", 422, {"field": "languages"})
    if body and (body.org_name is not None or city or details) and not org:
        raise B.AppError("INVALID_TEMPLATE_INPUT", "Give the organisation name.", 422, {"field": "org_name"})
    include_sample = True if body is None or body.include_sample is None else bool(body.include_sample)

    mine = personalise(tpl, org, city) if org else {}
    local = B._storage_mode(request) == "local"
    voice_id = await B._check_voice_for_agent(voice_id, request, user_id)
    project_id = B.LOCAL_PROJECT_ID if local else await B._get_or_create_project(user_id)
    row = {
        "project_id": project_id, "user_id": user_id, "voice_id": voice_id, "disclose_ai": True,
        "name": mine.get("name", tpl["name"]), "greeting": mine.get("greeting", tpl["greeting"]),
        "system_instructions": mine.get("system_instructions", tpl["system_instructions"]),
        "supported_languages": langs or list(tpl.get("languages") or ["en"]),
        "tools": list(tpl.get("tools") or []),  # MCP tool servers the agent may use (services/tools.py)
    }
    if local:
        row.update(extra or {})
        saved = await asyncio.to_thread(B.local_db.insert, "agents", row)
    else:
        row["id"] = str(B.uuid.uuid4())
        res = await B._insert_cloud_agent(row)
        saved = (res.data or [row])[0]

    mode = "local" if local else "cloud"
    try:
        if org and (city or details):  # the owner's own facts, so the agent can answer from them
            safe = re.sub(r"[^\w .,'()&-]", "", org)[:60].strip() or "organisation"
            await B.ingest_document(saved["id"], user_id, mode, f"About {safe}.md", _about_document(org, city, details))
        if include_sample:
            # Always labelled, in the file name and in the text, so nobody mistakes the made-up details for real ones.
            text = (TEMPLATES_DIR / tpl["document"]).read_text(encoding="utf-8")
            await B.ingest_document(saved["id"], user_id, mode, SAMPLE_PREFIX + tpl["document"], SAMPLE_BANNER + text)
    except Exception:
        # Do not leave a half-built agent behind.
        try:
            if local:
                await asyncio.to_thread(B.local_db.delete_agent, saved["id"], user_id)
            else:
                await B._cloud_run(lambda: B.supabase.table("agents").delete().eq("id", saved["id"])
                                   .eq("user_id", user_id).execute(), "delete agent")
        except Exception:  # noqa: BLE001
            log.exception("Could not clean up agent %s", saved.get("id"))
        raise
    return B._agent_json(saved, await B._document_count(request, saved["id"]))


@router.get("/api/agent-templates")
async def list_templates(user_id: str = Depends(current_user)):
    return {"templates": [_public(t) for t in _load_templates()]}


@router.get("/api/agent-templates/{template_id}/sample.pdf")
async def template_sample_pdf(template_id: str, user_id: str = Depends(current_user)):
    """The template's sample PDF, to read or to upload as the agent's document."""
    tpl = _find_template(template_id)
    name = (tpl or {}).get("sample_pdf")
    if not name:
        raise _not_found(template_id)
    return Response((TEMPLATES_DIR / name).read_bytes(), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ----------------------------------------------------------------------------------- questions to ask an agent

_questions_cache: Dict[tuple, List[str]] = {}
QUESTIONS_CACHE_MAX = 128   # (agent, document set) keys: one per edit of every agent's documents, so it is capped


def _cache_questions(key: tuple, questions: List[str]) -> None:
    _questions_cache.pop(key, None)
    _questions_cache[key] = questions
    while len(_questions_cache) > QUESTIONS_CACHE_MAX:
        _questions_cache.pop(next(iter(_questions_cache)))
QUESTIONS_MAX = 8


def _heading_questions(texts: List[str]) -> List[str]:
    """Without a language model: one question per heading. The chunker keeps a section's heading as the first line of
    its chunk (without the #), so a short first line with no full stop is taken as the heading."""
    out: List[str] = []
    for t in texts:
        lines = t.strip().splitlines()
        first = lines[0].strip().lstrip("#").strip() if lines else ""
        if len(lines) > 1 and 3 <= len(first) <= 60 and not first.endswith((".", "?", "!", ",")):
            q = f"What does the document say about {first[0].lower() + first[1:]}?"
            if q not in out:
                out.append(q)
    return out[:QUESTIONS_MAX]


def _parse_questions(text: str) -> List[str]:
    m = re.search(r"\[.*\]", text or "", re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except ValueError:
        items = [re.sub(r"^[\s\-*\d.)]+", "", ln).strip() for ln in (text or "").splitlines()]
    out = []
    for q in items:
        q = str(q).strip().strip('"')
        if 8 <= len(q) <= 160 and q.endswith("?") and q not in out:
            out.append(q)
    return out[:QUESTIONS_MAX]


def _with_header(row: dict) -> str:
    """A passage's text with its section heading first: newer documents keep the heading in its own column."""
    h = (row.get("header") or "").strip()
    return f"{h}\n{row['chunk_text']}" if h else row["chunk_text"]


async def _agent_chunk_texts(request: Request, agent_id: str):
    """(document ids and names, the agent's chunk texts in document order)."""
    B = _b()
    if B._storage_mode(request) == "local":
        docs = await asyncio.to_thread(B.local_db.select, "documents", lambda r: r.get("agent_id") == agent_id,
                                       order="created_at")
        ids = {d["id"] for d in docs}
        chunks = await asyncio.to_thread(B.local_db.select, "document_chunks",
                                         lambda r: r.get("document_id") in ids)
        chunks.sort(key=lambda c: c.get("chunk_index") or 0)
        by_doc = {d["id"]: [] for d in docs}
        for c in chunks:
            by_doc[c["document_id"]].append(_with_header(c))
        return [(d["id"], d.get("filename", "")) for d in docs], [t for d in docs for t in by_doc[d["id"]]]
    res = await B._cloud_run(lambda: B.supabase.table("documents").select("id, filename").eq("agent_id", agent_id)
                             .execute(), "list documents")
    docs = res.data or []
    texts: List[str] = []
    for d in docs:
        try:
            cr = await B._cloud_run(lambda: B.supabase.table("document_chunks").select("chunk_text, header")
                                    .eq("document_id", d["id"]).order("chunk_index").limit(200).execute(), "chunks")
        except Exception:  # noqa: BLE001 - a database from before the header column
            cr = await B._cloud_run(lambda: B.supabase.table("document_chunks").select("chunk_text")
                                    .eq("document_id", d["id"]).order("chunk_index").limit(200).execute(), "chunks")
        texts += [_with_header(r) for r in cr.data or []]
    return [(d["id"], d.get("filename", "")) for d in docs], texts


@router.get("/api/agents/{agent_id}/example-questions")
async def example_questions(agent_id: str, request: Request, refresh: bool = False,
                            user_id: str = Depends(current_user)):
    """Questions the user can ask this agent, written from its documents. A template agent that still has only its
    sample document gets the template's own list; otherwise the language model reads excerpts spread over the documents
    (headings are the fallback when no model answers). Cached until the documents change."""
    B = _b()
    agent = await B._load_agent(request, agent_id, user_id, "id, name")
    docs, texts = await _agent_chunk_texts(request, agent_id)
    if not texts:
        return {"questions": [], "source": "none", "documents": 0}
    key = (agent_id, tuple(sorted(i for i, _ in docs)))
    if not refresh and key in _questions_cache:
        return {"questions": _questions_cache[key], "source": "cache", "documents": len(docs)}
    names = [n for _, n in docs]
    if names and all(n.startswith(SAMPLE_PREFIX) for n in names):
        for tpl in _load_templates():
            if SAMPLE_PREFIX + tpl.get("document", "") in names and tpl.get("suggested_questions"):
                qs = list(tpl["suggested_questions"])
                _cache_questions(key, qs)
                return {"questions": qs, "source": "template", "documents": len(docs)}
    step = max(1, len(texts) // 10)
    excerpts = "\n\n".join(f"[{i + 1}] {t[:600]}" for i, t in enumerate(texts[::step][:10]))
    questions: List[str] = []
    source = "headings"
    try:
        prompt = ("Below are excerpts from a document. Write 8 different questions a reader might ask that the excerpts "
                  "answer. Each is one short sentence ending with a question mark, in the language of the document, "
                  "and answerable from the excerpts alone. Reply with only a JSON array of strings.\n\n" + excerpts)
        res = await asyncio.wait_for(asyncio.to_thread(
            B.llm.chat, [{"role": "user", "content": prompt}], temperature=0.4, max_tokens=500,
            timeout=B._llm_timeout(20.0)), timeout=B._llm_timeout(25.0))
        questions = _parse_questions(res.text or "")
        source = "model"
    except Exception as e:  # noqa: BLE001 - no model available or a bad reply: use the headings
        log.info("example questions: no model answer (%r), using headings", e)
    if len(questions) < 3:
        questions, source = _heading_questions(texts), "headings"
    if questions:
        _cache_questions(key, questions)
    return {"questions": questions, "source": source, "documents": len(docs)}


@router.post("/api/agents/from-template")
async def agent_from_template(body: FromTemplateRequest, request: Request, user_id: str = Depends(current_user)):
    tpl = _find_template(body.template_id)
    if tpl is None:
        raise _not_found(body.template_id)
    return await _create_from_template(tpl, request, user_id, body.voice_id, body=body)


# ----------------------------------------------------------------------------------------------- demo mode

def _require_local(request: Request) -> None:
    B = _b()
    if B._storage_mode(request) != "local":
        raise B.AppError("DEMO_LOCAL_ONLY", "Demo mode is only available with local storage.", 409)


async def _demo_rows(user_id: str) -> List[dict]:
    B = _b()
    rows = await asyncio.to_thread(B.local_db.select, "agents",
                                   lambda r: r.get("user_id") == user_id and bool(r.get("demo")),
                                   order="created_at")
    return rows


def _row_for(rows: List[dict], template_id: str) -> Optional[dict]:
    return next((r for r in rows if r.get("template_id") == template_id), None)


async def _run_eval(request: Request, agent_id: str, user_id: str) -> None:
    """The same code path as POST /api/evaluations/run (called directly, so the stored results are identical)."""
    B = _b()
    await B.run_evaluation(B.RunEvalRequest(agent_id=agent_id), request, user_id)


async def _eval_task(request: Request, agent_id: str, user_id: str) -> None:
    _eval_state[agent_id] = "running"
    try:
        await _run_eval(request, agent_id, user_id)
        _eval_state[agent_id] = "done"
    except Exception as e:  # noqa: BLE001 - e.g. no language model available; the page just shows no results
        log.warning("Demo evaluation for %s did not finish: %s", agent_id, e)
        _eval_state[agent_id] = "failed"


async def _eval_ready(agent_id: Optional[str]) -> bool:
    if not agent_id:
        return False
    B = _b()
    rows = await asyncio.to_thread(B.local_db.select, "evaluation_runs", lambda r: r.get("agent_id") == agent_id,
                                   limit=1)
    return bool(rows)


@router.post("/api/demo/load")
async def demo_load(request: Request, user_id: str = Depends(current_user)):
    _require_local(request)
    rows = await _demo_rows(user_id)
    created_clinic = False
    for tid in DEMO_TEMPLATES:
        if _row_for(rows, tid) is not None:
            continue
        tpl = _find_template(tid)
        if tpl is None:
            raise _not_found(tid)
        agent = await _create_from_template(tpl, request, user_id, None, {"demo": True, "template_id": tid})
        created_clinic = created_clinic or tid == DEMO_EVAL_TEMPLATE
        rows.append({"id": agent["id"], "template_id": tid})
    clinic = _row_for(rows, DEMO_EVAL_TEMPLATE)
    clinic_id = clinic["id"] if clinic else None

    if clinic_id and created_clinic:
        task = asyncio.create_task(_eval_task(request, clinic_id, user_id))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
        evaluation = "running"
    elif clinic_id and _eval_state.get(clinic_id) == "running":
        evaluation = "running"
    else:
        evaluation = "ready" if await _eval_ready(clinic_id) else "not_started"
    ids = [_row_for(rows, t)["id"] for t in DEMO_TEMPLATES if _row_for(rows, t)]
    return {"loaded": True, "agents": ids, "evaluation": evaluation}


@router.get("/api/demo/status")
async def demo_status(request: Request, user_id: str = Depends(current_user)):
    _require_local(request)
    rows = await _demo_rows(user_id)
    clinic = _row_for(rows, DEMO_EVAL_TEMPLATE)
    clinic_id = clinic["id"] if clinic else None
    ids = [_row_for(rows, t)["id"] for t in DEMO_TEMPLATES if _row_for(rows, t)]
    return {
        "loaded": bool(ids),
        "agents": ids,
        "evaluation_ready": await _eval_ready(clinic_id),
        "evaluation_running": _eval_state.get(clinic_id) == "running" if clinic_id else False,
    }


@router.post("/api/demo/reset")
async def demo_reset(request: Request, user_id: str = Depends(current_user)):
    _require_local(request)
    B = _b()
    rows = await _demo_rows(user_id)
    removed = 0
    for r in rows:
        await B.delete_agent(r["id"], request, user_id)  # purges documents, conversations, runs and audio
        _eval_state.pop(r["id"], None)
        removed += 1
    return {"reset": True, "removed": removed}
