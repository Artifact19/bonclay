"""Calls and bookings: what the agents have talked about, read from what is already stored. Nothing here writes.

    GET /api/calls?agent_id=&days=30&limit=50      the owner's recent conversations, newest first
    GET /api/calls/{conversation_id}               one conversation with its transcript (and what the tools did) and
                                                   `details`: what the caller told the agent (services/caller_memory.py),
                                                   the number and e-mail masked, and `sheet` (sent / waiting /
                                                   failed) when the conversation goes to a Google Sheet
                                                   (services/caller_sheet.py)

Where the facts come from (written by backend._turn_store, routers/phone_exotel.py, routers/widget.py and
routers/whatsapp.py; local JSON store or Supabase, chosen like every other route by the `Storage-Mode` header):
  * channel: `conversations.source` is "phone", "widget" (a website visitor) or "whatsapp"; no source means the owner
    used the app (Talk to your agent, or a Test call, which is also flagged `is_test`);
  * caller: only WhatsApp keeps one (`external_id`); a phone call does not store the caller's number, so it is None.
    Numbers are always masked ("••••3210"), also inside the transcript text;
  * duration: phone calls only, from the first turn's conversation row to the end of the last reply (a close
    estimate: the call itself is not timed anywhere);
  * outcome, derived and never guessed:
      booked          a booking tool call succeeded (appointments), or the meeting request it made was approved;
      needs_approval  a meeting request it made is still pending (see Approvals);
      declined        the meeting request it made was declined;
      handed_off      WhatsApp: the caller was handed to a person (`needs_human`);
      ended_early     a phone call where the caller never got past the greeting;
      couldnt_answer  the agent said it did not know (services/knowledge_gaps.is_gap_reply) or its model failed;
      answered        everything else.
"""
import asyncio
import builtins
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request

try:
    from core.errors import AppError
    from services import caller_memory, caller_sheet
    from services import knowledge_gaps as kg
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover - imported as backend.routers.conversations
    from backend.core.errors import AppError
    from backend.services import caller_memory, caller_sheet
    from backend.services import knowledge_gaps as kg
    from backend.services.auth import get_current_user_id

log = logging.getLogger("bonclay.calls")
router = APIRouter(prefix="/api/calls", tags=["calls"])

CHANNELS = {"phone": "phone", "widget": "website", "whatsapp": "whatsapp"}
LABELS = {
    "answered": "Answered", "booked": "Booked", "needs_approval": "Needs approval", "declined": "Request declined",
    "handed_off": "Needs a person", "couldnt_answer": "Couldn't answer", "ended_early": "Ended early",
}
BOOK, CANCEL_BOOKING, REQUEST, CANCEL_REQUEST = "book_appointment", "cancel_appointment", "request_meeting", "cancel_meeting"
MAX_CANDIDATES = 500          # conversations looked at per request
MAX_CALL_SECONDS = 4 * 3600   # a longer span is not a call (the row was reused or the clock is off)
PREVIEW_CHARS = 140

# ------------------------------------------------------------------------------------------------------ masking

_PHONE = re.compile(r"(?<![\d-])(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?![\d-])")
_DIGITS = re.compile(r"(?<![\d-])\d{7,}(?![\d-])")


def mask_number(value: Any) -> Optional[str]:
    """`••••3210` for a stored phone number; None when there is no number."""
    digits = re.sub(r"\D", "", str(value or ""))
    if not digits:
        return None
    return f"••••{digits[-4:]}" if len(digits) >= 5 else "••••"


def mask_text(text: Any) -> str:
    """The text with every phone number or long digit run reduced to its last four digits."""
    def one(m: "re.Match[str]") -> str:
        return mask_number(m.group(0)) or "••••"
    return _DIGITS.sub(one, _PHONE.sub(one, str(text or "")))


def caller_details(B, request: Request, conversation_id: str, user_id: str) -> Dict[str, str]:
    """What the caller told the agent in this conversation, for the owner: the number and the e-mail address masked like
    the transcript's. {} when nothing was kept (or CALLER_MEMORY was off)."""
    db = B.local_db if B._storage_mode(request) == "local" else None
    raw = caller_memory.load(conversation_id, db, user_id)
    form = caller_memory.public_view(raw)
    if raw.get("phone"):
        form["phone"] = mask_number(raw["phone"]) or ""
    return {k: v for k, v in form.items() if v}


# ------------------------------------------------------------------------------------------------------ derivation

def _ts(value: Any) -> Optional[datetime]:
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


_REQUEST_ID = re.compile(r"\b(M\d+)\b")


def _tool_calls(turn: dict) -> List[dict]:
    calls = turn.get("tool_calls")
    return [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []


def derive_outcome(source: Optional[str], needs_human: bool, turns: List[dict],
                   meeting_status: Callable[[str], Optional[str]]) -> Dict[str, Any]:
    """{key, label, detail, request_id} for one conversation, from its stored turns (oldest first)."""
    booked: Optional[str] = None
    open_requests: Dict[str, str] = {}
    for t in turns:
        for c in _tool_calls(t):
            if not c.get("ok"):
                continue
            name, summary = c.get("name"), str(c.get("result") or "")
            if name == BOOK:
                booked = mask_text(summary).capitalize()
            elif name == CANCEL_BOOKING:
                booked = None
            elif name == REQUEST:
                m = _REQUEST_ID.search(summary)
                if m:
                    open_requests[m.group(1)] = summary
            elif name == CANCEL_REQUEST:
                m = _REQUEST_ID.search(summary)
                if m:
                    open_requests.pop(m.group(1), None)
    statuses = {rid: (meeting_status(rid) or "pending") for rid in open_requests}
    pick = lambda s: next((rid for rid, st in statuses.items() if st == s), None)  # noqa: E731
    if booked:
        return {"key": "booked", "label": LABELS["booked"], "detail": booked, "request_id": None}
    if pick("approved"):
        rid = pick("approved")
        return {"key": "booked", "label": LABELS["booked"], "detail": f"Meeting request {rid} was approved.", "request_id": rid}
    if pick("pending"):
        rid = pick("pending")
        return {"key": "needs_approval", "label": LABELS["needs_approval"],
                "detail": f"Meeting request {rid} is waiting for your decision.", "request_id": rid}
    if pick("declined"):
        rid = pick("declined")
        return {"key": "declined", "label": LABELS["declined"], "detail": f"Meeting request {rid} was declined.", "request_id": rid}
    replies = [t for t in turns if t.get("role") == "agent"]
    if needs_human:
        return {"key": "handed_off", "label": LABELS["handed_off"], "detail": "The caller asked for a person.", "request_id": None}
    if source == "phone" and len(replies) <= 1:
        return {"key": "ended_early", "label": LABELS["ended_early"],
                "detail": "The caller hung up after the greeting." if replies else "The call connected but nothing was said.",
                "request_id": None}
    question: Optional[str] = None
    for t in turns:
        if t.get("role") == "user":
            question = t.get("text_content")
        elif t.get("role") == "agent" and (t.get("fallback") or kg.is_gap_reply(t.get("text_content") or "")):
            what = "The agent had a technical problem" if t.get("fallback") else "The agent said it did not know"
            ask = kg.redact(question or "")
            return {"key": "couldnt_answer", "label": LABELS["couldnt_answer"],
                    "detail": f"{what}{': ' + mask_text(ask) if ask else '.'}", "request_id": None}
    return {"key": "answered", "label": LABELS["answered"], "detail": None, "request_id": None}


def _duration_s(conv: dict, turns: List[dict]) -> Optional[int]:
    if conv.get("source") != "phone" or not turns:
        return None
    start, last = _ts(conv.get("created_at")), turns[-1]
    end = _ts(last.get("created_at"))
    if start is None or end is None:
        return None
    seconds = (end - start).total_seconds() + (last.get("total_ms") or last.get("latency_ms") or 0) / 1000.0
    return int(seconds) if 0 < seconds <= MAX_CALL_SECONDS else None


def _preview(turns: List[dict]) -> str:
    users = [str(t.get("text_content") or "").strip() for t in turns if t.get("role") == "user"]
    pick = next((u for u in users if len(u.split()) >= 3), users[0] if users else "")
    text = mask_text(pick)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS - 1].rstrip() + "…"


def build_call(conv: dict, turns: List[dict], agent_name: Optional[str], meeting_status: Callable[[str], Optional[str]]) -> dict:
    """One row of the list."""
    source = conv.get("source")
    channel = CHANNELS.get(source or "", "playground")
    when = conv.get("created_at") or (turns[0].get("created_at") if turns else None)
    return {
        "id": conv["id"], "agent_id": conv.get("agent_id"), "agent_name": agent_name or "Deleted agent",
        "channel": channel, "practice": bool(conv.get("is_test")), "when": when,
        "caller": mask_number(conv.get("external_id")) if source == "whatsapp" else None,
        "duration_s": _duration_s(conv, turns),
        "outcome": derive_outcome(source, bool(conv.get("needs_human")), turns, meeting_status),
        "replies": sum(1 for t in turns if t.get("role") == "agent"),
        "preview": _preview(turns),
    }


def build_transcript(turns: List[dict]) -> List[dict]:
    out = []
    for t in turns:
        item = {"role": "agent" if t.get("role") == "agent" else "caller", "text": mask_text(t.get("text_content")),
                "at": t.get("created_at"), "actions": []}
        for c in _tool_calls(t):  # what the agent did, never the arguments (they hold the caller's number)
            item["actions"].append({"name": c.get("name"), "ok": bool(c.get("ok")), "summary": mask_text(c.get("result"))[:200]})
        out.append(item)
    return out


# ------------------------------------------------------------------------------------------------------ loading

def _meeting_lookup() -> Callable[[str], Optional[str]]:
    """request id -> its status in the Approvals store, read once and only when a conversation made a request."""
    cache: Dict[str, Optional[Dict[str, str]]] = {"rows": None}

    def status(request_id: str) -> Optional[str]:
        if cache["rows"] is None:
            try:
                try:
                    from mcp_servers import meetings
                except ImportError:  # pragma: no cover
                    from backend.mcp_servers import meetings
                cache["rows"] = {r["request_id"]: r["status"] for r in meetings.list_requests(None)}
            except Exception as e:  # noqa: BLE001 - the Approvals store is optional here: the request is shown as pending
                log.warning("could not read the meeting requests: %s", e)
                cache["rows"] = {}
        return cache["rows"].get(request_id)

    return status


def _load_local(B, user_id: str, agent_id: Optional[str], since_iso: str, conv_id: Optional[str] = None):
    def keep(r: dict) -> bool:
        if r.get("user_id") != user_id:
            return False
        if conv_id is not None:
            return r.get("id") == conv_id
        return (agent_id is None or r.get("agent_id") == agent_id) and str(r.get("created_at") or "") >= since_iso

    convs = B.local_db.select("conversations", keep, order="created_at", desc=True, limit=MAX_CANDIDATES)
    ids = {c["id"] for c in convs}
    turns: Dict[str, List[dict]] = {}
    for t in B.local_db.select("conversation_turns", lambda r: r.get("conversation_id") in ids, order="created_at"):
        turns.setdefault(t["conversation_id"], []).append(t)
    names = {a["id"]: a.get("name") for a in B.local_db.select("agents", lambda r: r.get("user_id") == user_id)}
    return convs, turns, names


def _load_cloud(B, user_id: str, agent_id: Optional[str], since_iso: str, conv_id: Optional[str] = None):
    """Supabase. Columns added by later migrations (010 is_test, 013 source, 016 external_id and needs_human) are asked
    for first and dropped one tier at a time when the database does not have them yet."""
    sb = B.supabase
    tiers = ("id, agent_id, started_at, source, is_test, external_id, needs_human",
             "id, agent_id, started_at, source, is_test", "id, agent_id, started_at")
    rows: List[dict] = []
    for n, cols in enumerate(tiers):
        try:
            q = sb.table("conversations").select(cols).eq("user_id", user_id)
            if conv_id is not None:
                q = q.eq("id", conv_id)
            else:
                q = q.gte("started_at", since_iso)
                if agent_id:
                    q = q.eq("agent_id", agent_id)
            rows = q.order("started_at", desc=True).limit(MAX_CANDIDATES).execute().data or []
            break
        except Exception:  # noqa: BLE001
            if n == len(tiers) - 1:
                raise
    convs = [{**r, "created_at": r.get("started_at")} for r in rows]
    turns: Dict[str, List[dict]] = {}
    ids = [c["id"] for c in convs]
    tcols = ("conversation_id, role, text_content, created_at, fallback, total_ms, tool_calls",
             "conversation_id, role, text_content, created_at, fallback, total_ms",
             "conversation_id, role, text_content, created_at")
    for i in builtins.range(0, len(ids), 100):
        for n, cols in enumerate(tcols):
            try:
                got = (sb.table("conversation_turns").select(cols).in_("conversation_id", ids[i:i + 100])
                       .order("created_at").execute().data or [])
                break
            except Exception:  # noqa: BLE001
                if n == len(tcols) - 1:
                    raise
        for t in got:
            turns.setdefault(t["conversation_id"], []).append(t)
    agents = sb.table("agents").select("id, name").eq("user_id", user_id).execute().data or []
    return convs, turns, {a["id"]: a.get("name") for a in agents}


async def _load(request: Request, user_id: str, agent_id: Optional[str], since_iso: str, conv_id: Optional[str] = None):
    import backend as B  # lazy: backend.py loads this router while it is still being imported

    if B._storage_mode(request) == "local":
        return await asyncio.to_thread(_load_local, B, user_id, agent_id, since_iso, conv_id)
    return await B._cloud_run(lambda: _load_cloud(B, user_id, agent_id, since_iso, conv_id), "load calls")


# ------------------------------------------------------------------------------------------------------ routes

@router.get("")
async def list_calls(request: Request, agent_id: Optional[str] = Query(None, max_length=64),
                     days: int = Query(30, ge=1, le=365), limit: int = Query(50, ge=1, le=200),
                     user_id: str = Depends(get_current_user_id)):
    """The owner's recent conversations, newest first. An empty conversation is left out, except a phone call that
    connected and was never answered (that is an "Ended early" call)."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    convs, turns, names = await _load(request, user_id, agent_id, since)
    status = _meeting_lookup()

    def assemble() -> List[dict]:
        out = []
        for c in convs:
            t = turns.get(c["id"], [])
            if not t and c.get("source") != "phone":
                continue
            call = build_call(c, t, names.get(c.get("agent_id")), status)
            sheet = caller_sheet.sheet_status(c["id"])   # a dict lookup: only when the conversation went to a sheet
            if sheet:
                call["sheet"] = sheet
            out.append(call)
            if len(out) >= limit:
                break
        return out

    calls = await asyncio.to_thread(assemble)
    return {"calls": calls, "days": days}


@router.get("/{conversation_id}")
async def get_call(conversation_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    """One conversation (the owner's only) with its transcript."""
    convs, turns, names = await _load(request, user_id, None, "", conversation_id)
    conv = next((c for c in convs if c["id"] == conversation_id), None)
    if conv is None:
        raise AppError("NOT_FOUND", "Conversation not found", 404)
    t = turns.get(conversation_id, [])
    call = await asyncio.to_thread(build_call, conv, t, names.get(conv.get("agent_id")), _meeting_lookup())
    import backend as B  # lazy: backend.py loads this router while it is still being imported
    details = await asyncio.to_thread(caller_details, B, request, conversation_id, user_id)
    sheet = caller_sheet.sheet_status(conversation_id)
    return {**call, "transcript": build_transcript(t), "details": details, **({"sheet": sheet} if sheet else {})}
