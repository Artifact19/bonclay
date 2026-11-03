"""WhatsApp voice notes: a caller messages the business's WhatsApp number (text or a voice note) and the BonClay agent
answers with a voice note plus the same text, in the caller's language, from the agent's documents and guardrails.

Meta's WhatsApp Cloud API calls us (public side, no owner login; every call is authenticated by Meta's signature):
    GET  /api/whatsapp/webhook   Meta's one-time verification: hub.mode=subscribe, hub.verify_token, hub.challenge
    POST /api/whatsapp/webhook   messages. The body is verified with X-Hub-Signature-256 (the Meta app secret), the
                                 answer is a fast 200, and the work runs in the background

Owner side (normal auth; the agent must be the caller's):
    GET    /api/agents/{id}/whatsapp                 settings (flags and a masked hint, never a secret) + the Webhook URL
    PUT    /api/agents/{id}/whatsapp                 phone number id, access token and app secret (write-only), handoff
                                                     text, enable
    DELETE /api/agents/{id}/whatsapp                 remove the settings and the stored secrets
    POST   /api/agents/{id}/whatsapp/verify-token    a new verify token
    POST   /api/agents/{id}/whatsapp/check           ask Meta which number the token and phone number id belong to
    POST   /api/agents/{id}/whatsapp/simulate        run a fake inbound message through the flow without calling Meta
    GET    /api/agents/{id}/whatsapp/handoffs        callers waiting for a person
    POST   /api/agents/{id}/whatsapp/handoffs/{conversation_id}/resolve

How a message is answered: the turn is the shared blocking turn of the owner's playground (`_turn_blocking` in
backend.py): the same retrieval, guardrails, language policy, tools, AI disclosure on the first message of a
conversation, watermark and signed receipt. A voice note is transcribed first (so a request for a person can be
recognised) and the transcript is handed to the turn as text (`run.text_input`). The reply WAV is converted to an
OGG/Opus voice note, uploaded to Meta and sent, then the text follows as a second message. Replies only ever go to
people who wrote first (Meta's 24 hour window); nothing is sent unprompted.

Security: the signature is checked against the app secret of the agent that owns the receiving phone number id (or the
server-wide WHATSAPP_APP_SECRET), constant time, and an unsigned or wrongly signed call is a 403 that does nothing.
There is no way to switch the check off. The access token and the app secret are write-only (see services/whatsapp.py).
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import sys
import time
import uuid
import weakref
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

try:
    from core.errors import AppError, error_response
    from services import language_policy
    from services import whatsapp as wa
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover - imported as backend.routers.whatsapp
    from backend.core.errors import AppError, error_response
    from backend.services import language_policy
    from backend.services import whatsapp as wa
    from backend.services.auth import get_current_user_id

log = logging.getLogger("bonclay.whatsapp.api")

router = APIRouter(tags=["whatsapp"])

WEBHOOK_PATH = "/api/whatsapp/webhook"
CHALLENGE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
SENDER_PER_HOUR = 40
MAX_PARALLEL_TURNS = 3
INBOUND_TEXT_CHARS = 2000

seen = wa.SeenMessages()
sender_limiter = wa.SenderLimiter(SENDER_PER_HOUR)
_CHATS: "OrderedDict[Tuple[str, str], Tuple[str, float]]" = OrderedDict()  # (agent, wa id) -> (conversation id, last at)
_BACKGROUND: set = set()


def _b():
    """The backend module (imported lazily so this router never creates an import cycle)."""
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "local_db"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "local_db"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(value: Any) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


# ----------------------------------------------------------------------------------------------------- agents

def _raw(row: Optional[dict]) -> dict:
    raw = (row or {}).get("whatsapp")
    return dict(raw) if isinstance(raw, dict) else {}


async def _agents_with_whatsapp() -> List[Tuple[str, dict]]:
    """(storage mode, agent row) of every agent that has WhatsApp settings: the local store and, when configured, Supabase."""
    b = _b()
    out: List[Tuple[str, dict]] = [("local", r) for r in await asyncio.to_thread(
        b.local_db.select, "agents", lambda r: bool(r.get("whatsapp")))]
    if b.supabase is not None:
        try:
            res = await asyncio.to_thread(lambda: b.supabase.table("agents").select("*")
                                          .not_.is_("whatsapp", "null").execute())
            out += [("cloud", r) for r in (res.data or [])]
        except Exception as e:  # noqa: BLE001 - migration 016 not applied, or the database is down
            log.warning("WhatsApp agent lookup in Supabase failed (apply migration 016_whatsapp.sql?): %s", e)
    return out


async def _find_agent_by_phone(phone_number_id: str) -> Optional[Tuple[str, dict]]:
    if not phone_number_id:
        return None
    for mode, row in await _agents_with_whatsapp():
        if _raw(row).get("phone_number_id") == phone_number_id:
            return mode, row
    return None


def _secrets_of(mode: str, row: dict) -> Dict[str, str]:
    return wa.load_secrets(row["id"], local=mode == "local", enc=_raw(row).get("secrets_enc"))


def _app_secret(mode: str, row: dict) -> str:
    return _secrets_of(mode, row).get("app_secret") or wa.global_app_secret()


def _access_token(mode: str, row: dict) -> str:
    return _secrets_of(mode, row).get("access_token") or wa.global_access_token()


async def _save(request: Request, agent_id: str, user_id: str, raw: Optional[dict]) -> dict:
    """Write the WhatsApp settings of an owned agent (local file or Supabase); returns the updated row."""
    b = _b()
    if b._storage_mode(request) == "local":
        row = await asyncio.to_thread(b.local_db.update, "agents", agent_id, {"whatsapp": raw}, user_id)
        if row is None:
            raise AppError("AGENT_NOT_FOUND", "Agent not found", 404)
        return row
    try:
        await asyncio.to_thread(lambda: b.supabase.table("agents").update({"whatsapp": raw}).eq("id", agent_id)
                                .eq("user_id", user_id).execute())
    except Exception as e:  # noqa: BLE001
        log.warning("saving WhatsApp settings failed: %s", e)
        raise AppError("DB_ERROR", "Could not save the WhatsApp settings. Apply "
                       "backend/supabase/migrations/016_whatsapp.sql and try again.", 500) from e
    return await b._owned_agent(agent_id, user_id)


# ----------------------------------------------------------------------------------------------------- owner API

def _public_base(request: Request) -> str:
    import os
    base = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    return base or str(request.base_url).rstrip("/")


def _view(request: Request, row: dict) -> dict:
    cfg = wa.clean_config(_raw(row))
    local = _b()._storage_mode(request) == "local"
    has_token = cfg["access_token_set"] or bool(wa.global_access_token())
    has_secret = cfg["app_secret_set"] or bool(wa.global_app_secret())
    ok, reason = wa.secrets_storage_ok(local)
    return {
        "enabled": cfg["enabled"],
        "phone_number_id": cfg["phone_number_id"],
        "handoff_text": cfg["handoff_text"],
        "voice_for_text": cfg["voice_for_text"],
        "verify_token": cfg["verify_token"],
        "access_token_set": has_token,
        "access_token_hint": cfg["access_token_hint"] if cfg["access_token_set"] else ("environment" if has_token else ""),
        "app_secret_set": has_secret,
        "webhook_url": _public_base(request) + WEBHOOK_PATH,
        "webhook_path": WEBHOOK_PATH,
        "ready": bool(cfg["phone_number_id"] and has_token and has_secret),
        "secrets_storage": {"available": ok, "reason": reason},
        "default_handoff_text": wa.DEFAULT_HANDOFF["en"],
        "graph_url": wa.graph_base(),
        "created_at": cfg["created_at"],
    }


class WhatsAppUpdate(BaseModel):
    enabled: Optional[bool] = None
    phone_number_id: Optional[str] = Field(None, max_length=40)
    handoff_text: Optional[str] = Field(None, max_length=600)
    voice_for_text: Optional[bool] = None
    access_token: Optional[str] = Field(None, max_length=2000)  # write-only
    app_secret: Optional[str] = Field(None, max_length=200)     # write-only


def _clean_secret(value: str, what: str) -> str:
    v = (value or "").strip()
    if len(v) < 8 or re.search(r"\s", v):
        raise AppError("INVALID_" + what.upper().replace(" ", "_"), f"That {what} does not look right. Paste it exactly "
                       "as Meta shows it (no spaces).", 422)
    return v


@router.get("/api/agents/{agent_id}/whatsapp")
async def get_whatsapp(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    row = await _b()._load_agent(request, agent_id, user_id)
    return _view(request, row)


@router.put("/api/agents/{agent_id}/whatsapp")
async def put_whatsapp(agent_id: str, body: WhatsAppUpdate, request: Request,
                       user_id: str = Depends(get_current_user_id)):
    b = _b()
    row = await b._load_agent(request, agent_id, user_id)
    local = b._storage_mode(request) == "local"
    raw = _raw(row)
    cfg = wa.clean_config(raw)
    sent = body.model_fields_set
    if "phone_number_id" in sent and body.phone_number_id is not None:
        try:
            cfg["phone_number_id"] = wa.normalize_phone_number_id(body.phone_number_id)
        except ValueError as e:
            raise AppError("INVALID_PHONE_NUMBER_ID", str(e), 422, {"field": "phone_number_id"}) from e
        if cfg["phone_number_id"]:
            other = await _find_agent_by_phone(cfg["phone_number_id"])
            if other and other[1]["id"] != agent_id:
                raise AppError("PHONE_NUMBER_IN_USE", "That WhatsApp number is already connected to another agent.",
                               409)
    if "handoff_text" in sent and body.handoff_text is not None:
        cfg["handoff_text"] = re.sub(r"\s+", " ", body.handoff_text).strip()
    if "voice_for_text" in sent and body.voice_for_text is not None:
        cfg["voice_for_text"] = bool(body.voice_for_text)

    secrets = wa.load_secrets(agent_id, local=local, enc=raw.get("secrets_enc"))
    changed = False
    if body.access_token:
        secrets["access_token"] = _clean_secret(body.access_token, "access token")
        cfg["access_token_hint"] = wa.token_hint(secrets["access_token"])
        changed = True
    if body.app_secret:
        secrets["app_secret"] = _clean_secret(body.app_secret, "app secret")
        changed = True
    cfg["access_token_set"] = bool(secrets.get("access_token"))
    cfg["app_secret_set"] = bool(secrets.get("app_secret"))
    if not cfg["verify_token"]:
        cfg["verify_token"] = wa.new_verify_token()
        cfg["created_at"] = _now_iso()

    if "enabled" in sent and body.enabled is not None:
        if body.enabled:
            if not cfg["phone_number_id"]:
                raise AppError("PHONE_NUMBER_ID_REQUIRED", "Add the phone number ID from Meta's dashboard first.", 422,
                               {"field": "phone_number_id"})
            if not (cfg["access_token_set"] or wa.global_access_token()):
                raise AppError("ACCESS_TOKEN_REQUIRED", "Add the access token first.", 422, {"field": "access_token"})
            if not (cfg["app_secret_set"] or wa.global_app_secret()):
                raise AppError("APP_SECRET_REQUIRED", "Add the app secret first: it is how this server knows a "
                               "message really came from Meta.", 422, {"field": "app_secret"})
        cfg["enabled"] = bool(body.enabled)
    elif cfg["enabled"] and not cfg["phone_number_id"]:
        cfg["enabled"] = False

    new_raw: Dict[str, Any] = {k: v for k, v in cfg.items()}
    if changed:
        try:
            enc = wa.save_secrets(agent_id, secrets, local=local)
        except wa.WhatsAppError as e:
            raise AppError(e.code, e.message, 409) from e
        if enc:
            new_raw["secrets_enc"] = enc
    elif raw.get("secrets_enc"):
        new_raw["secrets_enc"] = raw["secrets_enc"]
    saved = await _save(request, agent_id, user_id, new_raw)
    return _view(request, saved)


@router.delete("/api/agents/{agent_id}/whatsapp")
async def delete_whatsapp(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    await _b()._load_agent(request, agent_id, user_id)
    saved = await _save(request, agent_id, user_id, None)
    await asyncio.to_thread(wa.forget_agent, agent_id)
    return _view(request, saved)


@router.post("/api/agents/{agent_id}/whatsapp/verify-token")
async def rotate_verify_token(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    """A new verify token. Meta only asks for it when the webhook is (re)registered, so paste the new one there."""
    row = await _b()._load_agent(request, agent_id, user_id)
    raw = _raw(row)
    raw["verify_token"] = wa.new_verify_token()
    raw.setdefault("created_at", _now_iso())
    return _view(request, await _save(request, agent_id, user_id, raw))


@router.post("/api/agents/{agent_id}/whatsapp/check")
async def check_whatsapp(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    """Ask Meta (GET /{phone-number-id}) which business number the stored token and phone number id belong to."""
    b = _b()
    row = await b._load_agent(request, agent_id, user_id)
    mode = "local" if b._storage_mode(request) == "local" else "cloud"
    pid, token = _raw(row).get("phone_number_id"), _access_token(mode, row)
    if not pid or not token:
        raise AppError("NOT_CONFIGURED", "Add the phone number ID and the access token first.", 409)
    try:
        info = await wa.GraphClient(token, pid).phone_info()
    except wa.WhatsAppError as e:
        return {"ok": False, "error": e.message}
    return {"ok": True, "display_phone_number": str(info.get("display_phone_number") or ""),
            "verified_name": str(info.get("verified_name") or "")}


# ----------------------------------------------------------------------------------------------------- webhook

@router.get(WEBHOOK_PATH)
async def webhook_verify(request: Request):
    """Meta's verification request (sent when the webhook URL is saved in the app dashboard): echo `hub.challenge` when
    `hub.verify_token` is the verify token of any agent with WhatsApp settings or WHATSAPP_VERIFY_TOKEN."""
    q = request.query_params
    mode, token, challenge = q.get("hub.mode", ""), q.get("hub.verify_token", ""), q.get("hub.challenge", "")
    known = [_raw(r).get("verify_token", "") for _m, r in await _agents_with_whatsapp()]
    known.append(wa.global_verify_token())
    if mode != "subscribe" or not wa.token_matches(token, known):
        return error_response("WHATSAPP_VERIFY_FAILED", "Verification failed.", 403)
    if not CHALLENGE_RE.match(challenge):
        return error_response("BAD_REQUEST", "Bad challenge.", 400)
    return PlainTextResponse(challenge)


@router.post(WEBHOOK_PATH)
async def webhook_receive(request: Request, background: BackgroundTasks):
    """Messages from Meta. Verified first; answered with 200 at once; processed in the background (Meta retries a slow
    or failed delivery, and a turn takes seconds)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > wa.MAX_BODY_BYTES:
        return error_response("PAYLOAD_TOO_LARGE", "Too large.", 413)
    body = await request.body()
    if len(body) > wa.MAX_BODY_BYTES:
        return error_response("PAYLOAD_TOO_LARGE", "Too large.", 413)
    signature = request.headers.get("x-hub-signature-256")
    try:
        payload = json.loads(body)
    except ValueError:
        payload = None
    ids = wa.phone_number_ids(payload)

    # Which phone numbers of this payload carry a valid signature: the app secret of the agent that owns the number
    # (or the server-wide one). Nothing is trusted, not even which agent is meant, before that.
    accepted: Dict[str, Optional[Tuple[str, dict]]] = {}
    for pid in ids:
        found = await _find_agent_by_phone(pid)
        secret = _app_secret(*found) if found else wa.global_app_secret()
        if wa.signature_valid(secret, body, signature):
            accepted[pid] = found
    if not ids and wa.signature_valid(wa.global_app_secret(), body, signature):
        return {"status": "ok"}
    if not accepted:
        log.warning("WhatsApp webhook rejected: missing or invalid signature (or no app secret configured)")
        return error_response("WHATSAPP_SIGNATURE_INVALID", "Invalid signature.", 403)

    messages, statuses = wa.parse_messages(payload)
    work = [(accepted[m.phone_number_id], m) for m in messages
            if m.phone_number_id in accepted and accepted[m.phone_number_id] is not None]
    log.info("WhatsApp webhook: %d message(s), %d status update(s) ignored", len(work), statuses)
    if work:
        background.add_task(_process_webhook, work)
    return {"status": "ok"}


async def _process_webhook(work: List[Tuple[Tuple[str, dict], wa.Inbound]]) -> None:
    """Background: handle each message of an accepted webhook call. Never raises."""
    for (mode, row), msg in work:
        try:
            cfg = wa.clean_config(_raw(row))
            if not cfg["enabled"]:
                continue
            if msg.timestamp and time.time() - msg.timestamp > wa.WINDOW_S:
                log.info("WhatsApp: skipped message %s older than the 24 hour window", msg.message_id)
                continue
            if not seen.first_time(msg.message_id):
                log.info("WhatsApp: duplicate message %s ignored", msg.message_id)
                continue
            if not sender_limiter.allow(f"{row['id']}:{msg.wa_id}"):
                log.warning("WhatsApp: %s is over %d messages an hour for agent %s; ignored",
                            wa.mask_phone(msg.wa_id), SENDER_PER_HOUR, row["id"])
                continue
            token = _access_token(mode, row)
            if not token:
                log.warning("WhatsApp: agent %s has no access token; cannot reply", row["id"])
                continue
            client = wa.GraphClient(token, cfg["phone_number_id"])
            await process_message(mode, row, msg, client)
        except Exception:  # noqa: BLE001 - one bad message must not stop the others
            log.exception("WhatsApp message %s failed", getattr(msg, "message_id", "?"))


# ----------------------------------------------------------------------------------------------------- conversations

_CHAT_LOCKS: Dict[Tuple[str, str], List[Any]] = {}
_SLOTS: "weakref.WeakKeyDictionary[Any, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


@contextlib.asynccontextmanager
async def _serialized(key: Tuple[str, str]):
    """One message of a chat at a time, in arrival order; at most MAX_PARALLEL_TURNS turns at once overall."""
    entry = _CHAT_LOCKS.setdefault(key, [asyncio.Lock(), 0])
    entry[1] += 1
    loop = asyncio.get_running_loop()
    slots = _SLOTS.get(loop)
    if slots is None:
        slots = _SLOTS[loop] = asyncio.Semaphore(MAX_PARALLEL_TURNS)
    try:
        async with entry[0]:
            async with slots:
                yield
    finally:
        entry[1] -= 1
        if entry[1] <= 0:
            _CHAT_LOCKS.pop(key, None)


def _remember_chat(agent_id: str, wa_id: str, conv_id: str) -> None:
    _CHATS[(agent_id, wa_id)] = (conv_id, time.time())
    _CHATS.move_to_end((agent_id, wa_id))
    while len(_CHATS) > 4000:
        _CHATS.popitem(last=False)


async def _find_conversation(mode: str, agent: dict, wa_id: str) -> Optional[str]:
    """The caller's current conversation (their last message was inside the 24 hour window), else None."""
    cached = _CHATS.get((agent["id"], wa_id))
    if cached and time.time() - cached[1] < wa.WINDOW_S:
        return cached[0]
    b = _b()
    try:
        if mode == "local":
            rows = await asyncio.to_thread(
                b.local_db.select, "conversations",
                lambda r: r.get("agent_id") == agent["id"] and r.get("external_id") == wa_id
                and r.get("source") == "whatsapp" and not r.get("is_test"), desc=True, limit=1)
            row = rows[0] if rows else None
        else:
            res = await asyncio.to_thread(lambda: b.supabase.table("conversations").select("id, last_inbound_at")
                                          .eq("agent_id", agent["id"]).eq("external_id", wa_id)
                                          .order("started_at", desc=True).limit(1).execute())
            row = res.data[0] if res.data else None
    except Exception as e:  # noqa: BLE001 - migration 016 not applied
        log.warning("WhatsApp conversation lookup failed (apply migration 016_whatsapp.sql?): %s", e)
        return None
    if row and time.time() - _parse_iso(row.get("last_inbound_at") or row.get("created_at")) < wa.WINDOW_S:
        return row["id"]
    return None


async def _new_conversation(mode: str, agent: dict, wa_id: str, *, is_test: bool) -> Optional[str]:
    b = _b()
    fields = {"agent_id": agent["id"], "user_id": agent["user_id"], "source": "whatsapp", "external_id": wa_id,
              "needs_human": False, "last_inbound_at": _now_iso(), **({"is_test": True} if is_test else {})}
    try:
        if mode == "local":
            return (await asyncio.to_thread(b.local_db.insert, "conversations", fields))["id"]
        res = await asyncio.to_thread(lambda: b.supabase.table("conversations").insert(fields).execute())
        return res.data[0]["id"]
    except Exception as e:  # noqa: BLE001 - migration 016 not applied: the pipeline makes a plain conversation
        log.warning("could not create a WhatsApp conversation (apply migration 016_whatsapp.sql?): %s", e)
        return None


async def _touch_conversation(mode: str, agent: dict, conv_id: str, fields: Dict[str, Any]) -> None:
    b = _b()
    try:
        if mode == "local":
            await asyncio.to_thread(b.local_db.update, "conversations", conv_id, fields, agent["user_id"])
        else:
            await asyncio.to_thread(lambda: b.supabase.table("conversations").update(fields).eq("id", conv_id)
                                    .eq("agent_id", agent["id"]).execute())
    except Exception as e:  # noqa: BLE001
        log.warning("could not update a WhatsApp conversation (apply migration 016_whatsapp.sql?): %s", e)


async def _add_turns(mode: str, conv_id: str, rows: List[Tuple[str, str]]) -> None:
    """Text-only turns (the handoff exchange): `rows` are (role, text)."""
    b = _b()
    try:
        if mode == "local":
            await asyncio.to_thread(lambda: [b.local_db.insert("conversation_turns", {
                "conversation_id": conv_id, "role": role, "text_content": text}) for role, text in rows])
        else:
            await asyncio.to_thread(lambda: b.supabase.table("conversation_turns").insert(
                [{"conversation_id": conv_id, "role": role, "text_content": text} for role, text in rows]).execute())
    except Exception as e:  # noqa: BLE001
        log.warning("could not store the handoff turns: %s", e)


async def _conversation_for(mode: str, agent: dict, wa_id: str, *, is_test: bool) -> Optional[str]:
    conv = await _find_conversation(mode, agent, wa_id) if not is_test else (
        _CHATS.get((agent["id"], wa_id)) or (None,))[0]
    if conv is None:
        conv = await _new_conversation(mode, agent, wa_id, is_test=is_test)
    if conv:
        _remember_chat(agent["id"], wa_id, conv)
    return conv


# ----------------------------------------------------------------------------------------------------- the flow

async def _safe(coro, what: str) -> bool:
    try:
        await coro
        return True
    except wa.WhatsAppError as e:
        log.warning("WhatsApp %s failed: %s", what, e.message)
    except Exception:  # noqa: BLE001
        log.exception("WhatsApp %s failed", what)
    return False


async def _reply_text(client, msg: wa.Inbound, text: str) -> bool:
    return await _safe(client.send_text(msg.wa_id, text), "text reply")


async def _transcribe(client, msg: wa.Inbound) -> Tuple[Optional[str], int, Optional[str]]:
    """(transcript, asr ms, canned reply key) of a voice note. The key is None when all went well."""
    b = _b()
    t0 = time.time()
    try:
        data, _mime = await client.fetch_media(msg.media_id)
        wav, _seconds = await asyncio.to_thread(wa.voice_note_to_wav, data)
    except wa.WhatsAppError as e:
        log.warning("WhatsApp voice note %s could not be read: %s", msg.message_id, e.message)
        return None, 0, "too_long" if e.code in ("TOO_LONG", "MEDIA_TOO_LARGE") else "failed"
    text = await b._speech_to_text(wav, "whatsapp", None)
    if text is None:
        return None, 0, "failed"  # no speech recognition on this server
    return text.strip(), int((time.time() - t0) * 1000), ("unheard" if not text.strip() else None)


async def process_message(mode: str, agent: dict, msg: wa.Inbound, client, *, dry_run: bool = False) -> dict:
    """One inbound message, start to finish: transcribe, (hand off | run the turn), send the reply. Returns what
    happened (the simulator shows it). `client` is a wa.GraphClient or, for the simulator, a wa.DryRunClient."""
    async with _serialized((agent["id"], msg.wa_id)):
        if not dry_run:
            task = asyncio.ensure_future(client.mark_read(msg.message_id))
            _BACKGROUND.add(task)
            task.add_done_callback(_BACKGROUND.discard)
        return await _process_locked(mode, agent, msg, client, dry_run)


def _lang(text: str, agent: dict) -> str:
    try:
        return language_policy.detect_language(text, agent.get("supported_languages") or ["en"])
    except Exception:  # noqa: BLE001
        return "en"


async def _process_locked(mode: str, agent: dict, msg: wa.Inbound, client, dry_run: bool) -> dict:
    b = _b()
    cfg = wa.clean_config(_raw(agent))
    out: Dict[str, Any] = {"handoff": False, "voice_sent": False, "user_text": "", "agent_text": "",
                           "kind": msg.kind, "dry_run": dry_run}
    asr_ms = 0
    if msg.kind == "text":
        text = msg.text.strip()[:INBOUND_TEXT_CHARS]
    elif msg.kind == "audio":
        text, asr_ms, problem = await _transcribe(client, msg)
        if problem:
            table = {"too_long": wa.TOO_LONG_REPLY, "unheard": wa.UNHEARD_REPLY, "failed": wa.FAILED_REPLY}[problem]
            out["agent_text"] = wa.canned(table, "en")
            await _reply_text(client, msg, out["agent_text"])
            return out
    else:
        out["agent_text"] = wa.canned(wa.UNSUPPORTED_REPLY, "en")
        await _reply_text(client, msg, out["agent_text"])
        return out
    out["user_text"] = text
    lang = _lang(text, agent)

    if wa.wants_human(text):
        return await _handoff(mode, agent, cfg, msg, client, text, lang, out, dry_run)

    conv_id = await _conversation_for(mode, agent, msg.wa_id, is_test=dry_run)
    try:
        run = await b._turn_begin(None, agent["id"], agent["user_id"], storage_mode=mode, agent=agent,
                                  source="whatsapp")
        run.text_input = text
        run.timings["asr_ms"] = asr_ms
        run.is_test = dry_run
        result = await b._turn_blocking(run, b"", conv_id)
    except Exception:  # noqa: BLE001 - AppError (agent gone, ...) or a pipeline failure
        log.exception("WhatsApp turn failed for agent %s", agent["id"])
        out["agent_text"] = wa.canned(wa.FAILED_REPLY, lang)
        await _reply_text(client, msg, out["agent_text"])
        return out

    conv_id = result.get("conversation_id") or conv_id
    if conv_id:
        _remember_chat(agent["id"], msg.wa_id, conv_id)
        await _touch_conversation(mode, agent, conv_id, {"last_inbound_at": _now_iso()})
    agent_text = result.get("agent_text") or ""
    out.update({"agent_text": agent_text, "language": result.get("language"), "conversation_id": conv_id,
                "latency_ms": result.get("latency_ms"), "receipt_id": result.get("receipt_id"),
                "fallback": result.get("fallback"), "grounded": result.get("grounded")})

    if run.reply_audio and (msg.kind == "audio" or cfg["voice_for_text"]):
        async def send_voice() -> None:
            ogg = await asyncio.to_thread(wa.wav_to_voice_note, run.reply_audio)
            media_id = await client.upload_media(ogg)
            await client.send_audio(msg.wa_id, media_id)
            out["voice_note_bytes"] = len(ogg)
        out["voice_sent"] = await _safe(send_voice(), "voice note")
        if dry_run:
            out["reply_wav"] = run.reply_audio
    await _reply_text(client, msg, agent_text)  # the text follows the voice note (or replaces it when that failed)
    return out


async def _handoff(mode: str, agent: dict, cfg: dict, msg: wa.Inbound, client, text: str, lang: str, out: dict,
                   dry_run: bool) -> dict:
    """The caller asked for a person: send the configured handoff message and flag the conversation for follow-up."""
    reply = cfg["handoff_text"] or wa.canned(wa.DEFAULT_HANDOFF, lang)
    conv_id = await _conversation_for(mode, agent, msg.wa_id, is_test=dry_run)
    if conv_id:
        await _add_turns(mode, conv_id, [("user", text), ("agent", reply)])
        await _touch_conversation(mode, agent, conv_id, {"needs_human": True, "handoff_at": _now_iso(),
                                                         "last_inbound_at": _now_iso()})
    log.info("WhatsApp handoff requested by %s for agent %s", wa.mask_phone(msg.wa_id), agent["id"])
    out.update({"handoff": True, "agent_text": reply, "conversation_id": conv_id, "language": lang})
    await _reply_text(client, msg, reply)
    return out


# ----------------------------------------------------------------------------------------------------- simulator

class SimulateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=INBOUND_TEXT_CHARS)


@router.post("/api/agents/{agent_id}/whatsapp/simulate")
async def simulate(agent_id: str, body: SimulateRequest, request: Request,
                   user_id: str = Depends(get_current_user_id)):
    """Dry run: a fake inbound text goes through the real flow (handoff check, turn, voice note encoding) and the
    replies are returned instead of being sent. Nothing is sent to Meta and no Meta settings are needed. The
    conversation is an owner test (kept out of Analytics and of the handoff list)."""
    b = _b()
    row = await b._load_agent(request, agent_id, user_id)
    mode = "local" if b._storage_mode(request) == "local" else "cloud"
    client = wa.DryRunClient()
    msg = wa.Inbound(message_id="sim-" + uuid.uuid4().hex, wa_id="simulator", phone_number_id="dry-run", kind="text",
                     text=body.text.strip(), name="Simulator", timestamp=time.time())
    out = await process_message(mode, row, msg, client, dry_run=True)
    messages: List[dict] = []
    wav = out.pop("reply_wav", None)
    for item in client.sent:
        if item["call"] == "message" and item["type"] == "audio":
            m = {"type": "audio", "mime": wa.OGG_MIME, "bytes": item.get("bytes", 0)}
            if wav and len(wav) < 4 * 1024 * 1024:  # playable in every browser (a real voice note is OGG/Opus)
                m["audio_url"] = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
            messages.append(m)
        elif item["call"] == "message":
            messages.append({"type": "text", "body": item["body"]})
    out["messages"] = messages
    out["dry_run"] = True
    return out


# ----------------------------------------------------------------------------------------------------- handoffs

@router.get("/api/agents/{agent_id}/whatsapp/handoffs")
async def list_handoffs(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    """Callers who asked for a person and are not marked as followed up yet (newest first, at most 50)."""
    b = _b()
    await b._load_agent(request, agent_id, user_id)
    rows: List[dict] = []
    try:
        if b._storage_mode(request) == "local":
            rows = await asyncio.to_thread(
                b.local_db.select, "conversations",
                lambda r: r.get("agent_id") == agent_id and r.get("user_id") == user_id and r.get("needs_human")
                and not r.get("is_test"), order="handoff_at", desc=True, limit=50)
        else:
            res = await asyncio.to_thread(lambda: b.supabase.table("conversations")
                                          .select("id, external_id, handoff_at").eq("agent_id", agent_id)
                                          .eq("user_id", user_id).eq("needs_human", True)
                                          .or_("is_test.is.null,is_test.eq.false")
                                          .order("handoff_at", desc=True).limit(50).execute())
            rows = res.data or []
    except Exception as e:  # noqa: BLE001 - migration 016 not applied
        log.warning("WhatsApp handoff list failed (apply migration 016_whatsapp.sql?): %s", e)
    return {"handoffs": [{"conversation_id": r["id"], "contact": str(r.get("external_id") or ""),
                          "requested_at": r.get("handoff_at")} for r in rows]}


@router.post("/api/agents/{agent_id}/whatsapp/handoffs/{conversation_id}/resolve")
async def resolve_handoff(agent_id: str, conversation_id: str, request: Request,
                          user_id: str = Depends(get_current_user_id)):
    b = _b()
    row = await b._load_agent(request, agent_id, user_id)
    if not re.match(r"^[A-Za-z0-9-]{8,64}$", conversation_id):
        raise AppError("NOT_FOUND", "Conversation not found", 404)
    mode = "local" if b._storage_mode(request) == "local" else "cloud"
    if mode == "local":
        conv = await asyncio.to_thread(b.local_db.get, "conversations", conversation_id, user_id)
        found = bool(conv and conv.get("agent_id") == agent_id)
    else:
        res = await asyncio.to_thread(lambda: b.supabase.table("conversations").select("id")
                                      .eq("id", conversation_id).eq("agent_id", agent_id).execute())
        found = bool(res.data)
    if not found:
        raise AppError("NOT_FOUND", "Conversation not found", 404)
    await _touch_conversation(mode, row, conversation_id, {"needs_human": False})
    return {"conversation_id": conversation_id, "needs_human": False}
