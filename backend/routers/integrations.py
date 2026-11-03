"""Owner-side API for an agent's outbound integrations (signed webhook, Google Sheets). See services/integrations.py.

    GET    /api/agents/{id}/integrations                  settings of both integrations (flags only, never a secret)
    PUT    /api/agents/{id}/integrations/webhook          URL, on/off, events, include test calls, redact phones
    POST   /api/agents/{id}/integrations/webhook/rotate-secret   a new signing secret (shown once, in this response)
    POST   /api/agents/{id}/integrations/webhook/test    send one signed test event now
    DELETE /api/agents/{id}/integrations/webhook          remove the webhook and its secret
    PUT    /api/agents/{id}/integrations/sheets           spreadsheet, tab, on/off, key (write-only), toggles
    PUT    /api/agents/{id}/integrations/caller-sheet     send caller details to a Google Sheet: link, tab, on/off, toggles
    POST   /api/agents/{id}/integrations/caller-sheet/test  write (or update) a marked test row; Google's exact answer
    POST   /api/agents/{id}/integrations/sheets/test      append a test row; reports Google's exact answer
    DELETE /api/agents/{id}/integrations/sheets           remove the Sheets settings and the stored key
    GET    /api/agents/{id}/integrations/deliveries       the delivery log (newest 50)

The signing secret is returned once (when created or rotated). The service-account key is write-only: the API only
says whether one is stored and which e-mail address to share the spreadsheet with.
"""
import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

try:
    from core.errors import AppError
    from services import caller_sheet, integrations as svc
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover - imported as backend.routers.integrations
    from backend.core.errors import AppError
    from backend.services import caller_sheet, integrations as svc
    from backend.services.auth import get_current_user_id

log = logging.getLogger("bonclay.integrations.api")

router = APIRouter(tags=["integrations"])

PRIVACY_NOTE = ("These integrations send what callers say, and what the agent answers, to systems you choose "
                "(your webhook, your Google spreadsheet). Tell callers if your policy requires it. Turn on "
                "\"Redact phone numbers\" to replace phone numbers with [phone] before anything is sent.")


def _b():
    """The backend module (imported lazily so this router never creates an import cycle)."""
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "local_db"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "local_db"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


def _fail(e: "svc.IntegrationError") -> AppError:
    status = 409 if e.code == "SECRETS_STORAGE_UNAVAILABLE" else 422
    return AppError(e.code, e.message, status)


def _raw(row: dict) -> dict:
    raw = row.get("integrations")
    return dict(raw) if isinstance(raw, dict) else {}


def _view(row: dict, local: bool, extra: Optional[dict] = None) -> dict:
    out = svc.public_cfg(row.get("integrations"))
    out.update({"privacy_note": PRIVACY_NOTE, "secrets_storage": svc.secrets_storage_status(local),
                "local_webhooks_allowed": svc.local_webhooks_allowed(),
                "sheet_columns": list(svc.SHEETS_COLUMNS)})
    out["caller_sheet"] = {**out["caller_sheet"], **caller_sheet.describe(
        str(row.get("id") or ""), svc.normalize_cfg(row.get("integrations")), local)}
    if extra:
        out.update(extra)
    return out


async def _save(request: Request, agent_id: str, user_id: str, raw: dict) -> dict:
    """Write the integrations settings of an owned agent (local file or Supabase); returns the updated row."""
    b = _b()
    if b._storage_mode(request) == "local":
        row = await asyncio.to_thread(b.local_db.update, "agents", agent_id, {"integrations": raw}, user_id)
        if row is None:
            raise AppError("AGENT_NOT_FOUND", "Agent not found", 404)
        return row
    try:
        await asyncio.to_thread(lambda: b.supabase.table("agents").update({"integrations": raw}).eq("id", agent_id)
                                .eq("user_id", user_id).execute())
    except Exception as e:  # noqa: BLE001
        log.warning("saving integrations failed: %s", e)
        raise AppError("DB_ERROR", "Could not save the integration settings. Apply "
                       "backend/supabase/migrations/015_agent_integrations.sql and try again.", 500) from e
    return await b._owned_agent(agent_id, user_id)


def _is_local(request: Request) -> bool:
    return _b()._storage_mode(request) == "local"


def _stored_secrets(agent_id: str, raw: dict, local: bool) -> dict:
    return svc.load_secrets(agent_id, local=local, enc=raw.get("secrets_enc"))


def _put_secrets(agent_id: str, raw: dict, local: bool, secrets: dict) -> None:
    """Store `secrets`; cloud mode puts the encrypted blob into `raw`. Raises AppError when storing is not possible."""
    try:
        enc = svc.save_secrets(agent_id, secrets, local=local)
    except svc.IntegrationError as e:
        raise _fail(e) from e
    if not local:
        if enc:
            raw["secrets_enc"] = enc
        else:
            raw.pop("secrets_enc", None)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------------- read

@router.get("/api/agents/{agent_id}/integrations")
async def get_integrations(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    row = await _b()._load_agent(request, agent_id, user_id)
    return _view(row, _is_local(request))


@router.get("/api/agents/{agent_id}/integrations/deliveries")
async def get_deliveries(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    await _b()._load_agent(request, agent_id, user_id)
    rows = await asyncio.to_thread(svc.get_log, agent_id, local=_is_local(request))
    return {"deliveries": rows[: svc.LOG_KEEP]}


# --------------------------------------------------------------------------------------------------- webhook

class WebhookUpdate(BaseModel):
    enabled: Optional[bool] = None
    url: Optional[str] = Field(None, max_length=600)
    events: Optional[List[str]] = Field(None, max_length=10)
    include_test: Optional[bool] = None
    redact_phones: Optional[bool] = None


@router.put("/api/agents/{agent_id}/integrations/webhook")
async def put_webhook(agent_id: str, body: WebhookUpdate, request: Request,
                      user_id: str = Depends(get_current_user_id)):
    """Set the URL (the first time one is set, a signing secret is created and returned once)."""
    row = await _b()._load_agent(request, agent_id, user_id)
    local = _is_local(request)
    raw = _raw(row)
    cfg = svc.normalize_cfg(raw)
    wh = cfg["webhook"]
    sent = body.model_fields_set
    new_secret: Optional[str] = None
    try:
        if "url" in sent and body.url is not None:
            if body.url.strip():
                wh["url"] = await asyncio.to_thread(svc.validate_webhook_url, body.url)
            else:
                wh["url"], wh["enabled"] = "", False
        if "events" in sent and body.events is not None:
            bad = [e for e in body.events if e not in svc.EVENTS]
            if bad:
                raise svc.IntegrationError("INVALID_EVENTS", f"Unknown event(s): {', '.join(bad)[:100]}. "
                                           f"Choose from {', '.join(svc.EVENTS)}.")
            wh["events"] = list(dict.fromkeys(body.events))
        for flag in ("include_test", "redact_phones"):
            if flag in sent and getattr(body, flag) is not None:
                wh[flag] = bool(getattr(body, flag))
        if "enabled" in sent and body.enabled is not None:
            if body.enabled and not wh["url"]:
                raise svc.IntegrationError("WEBHOOK_URL_REQUIRED", "Enter the webhook URL before turning it on.")
            if body.enabled and not wh["events"]:
                raise svc.IntegrationError("INVALID_EVENTS", "Choose at least one event to send.")
            wh["enabled"] = bool(body.enabled)
        if wh["url"] and not wh["secret_set"]:
            new_secret = svc.new_webhook_secret()
    except svc.IntegrationError as e:
        raise _fail(e) from e
    if not wh["url"]:  # no URL: nothing to sign for
        if wh["secret_set"]:
            secrets = _stored_secrets(agent_id, raw, local)
            secrets.pop("webhook_secret", None)
            _put_secrets(agent_id, raw, local, secrets)
        wh.update({"secret_set": False, "secret_created_at": None, "enabled": False})
    if new_secret:
        secrets = _stored_secrets(agent_id, raw, local)
        secrets["webhook_secret"] = new_secret
        _put_secrets(agent_id, raw, local, secrets)
        wh.update({"secret_set": True, "secret_created_at": _now_iso()})
    raw["webhook"] = wh
    raw["sheets"] = cfg["sheets"]
    saved = await _save(request, agent_id, user_id, raw)
    return _view(saved, local, {"secret": new_secret} if new_secret else None)


@router.post("/api/agents/{agent_id}/integrations/webhook/rotate-secret")
async def rotate_webhook_secret(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    """A new signing secret; the old one stops being used at once. The secret is in this response only."""
    row = await _b()._load_agent(request, agent_id, user_id)
    local = _is_local(request)
    raw = _raw(row)
    cfg = svc.normalize_cfg(raw)
    if not cfg["webhook"]["url"]:
        raise AppError("WEBHOOK_URL_REQUIRED", "Enter the webhook URL first.", 409)
    secrets = _stored_secrets(agent_id, raw, local)
    secret = svc.new_webhook_secret()
    secrets["webhook_secret"] = secret
    _put_secrets(agent_id, raw, local, secrets)
    cfg["webhook"].update({"secret_set": True, "secret_created_at": _now_iso()})
    raw["webhook"], raw["sheets"] = cfg["webhook"], cfg["sheets"]
    saved = await _save(request, agent_id, user_id, raw)
    return _view(saved, local, {"secret": secret})


@router.post("/api/agents/{agent_id}/integrations/webhook/test")
async def test_webhook(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    """Send one signed `test.ping` event and report the receiver's answer."""
    row = await _b()._load_agent(request, agent_id, user_id)
    local = _is_local(request)
    raw = _raw(row)
    cfg = svc.normalize_cfg(raw)
    if not cfg["webhook"]["url"] or not cfg["webhook"]["secret_set"]:
        raise AppError("WEBHOOK_URL_REQUIRED", "Save a webhook URL first.", 409)
    entry = await asyncio.to_thread(svc.send_test_event, agent_id, user_id, local, raw.get("secrets_enc"), cfg,
                                    str(row.get("name") or ""))
    return {"delivery": entry, "ok": entry["status"] == "delivered"}


@router.delete("/api/agents/{agent_id}/integrations/webhook")
async def delete_webhook(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    row = await _b()._load_agent(request, agent_id, user_id)
    local = _is_local(request)
    raw = _raw(row)
    cfg = svc.normalize_cfg(raw)
    secrets = _stored_secrets(agent_id, raw, local)
    secrets.pop("webhook_secret", None)
    _put_secrets(agent_id, raw, local, secrets)
    raw["webhook"] = svc.default_cfg()["webhook"]
    raw["sheets"] = cfg["sheets"]
    saved = await _save(request, agent_id, user_id, raw)
    return _view(saved, local)


# ---------------------------------------------------------------------------------------------------- sheets

class SheetsUpdate(BaseModel):
    enabled: Optional[bool] = None
    spreadsheet: Optional[str] = Field(None, max_length=600)           # link or id
    sheet_name: Optional[str] = Field(None, max_length=200)            # tab; empty: the first tab
    service_account_json: Optional[str] = Field(None, max_length=20000)  # write-only
    include_test: Optional[bool] = None
    redact_phones: Optional[bool] = None


@router.put("/api/agents/{agent_id}/integrations/sheets")
async def put_sheets(agent_id: str, body: SheetsUpdate, request: Request,
                     user_id: str = Depends(get_current_user_id)):
    row = await _b()._load_agent(request, agent_id, user_id)
    local = _is_local(request)
    raw = _raw(row)
    cfg = svc.normalize_cfg(raw)
    sh = cfg["sheets"]
    sent = body.model_fields_set
    try:
        if "service_account_json" in sent and body.service_account_json and body.service_account_json.strip():
            creds = svc.parse_service_account(body.service_account_json)
            secrets = _stored_secrets(agent_id, raw, local)
            secrets["service_account"] = creds
            _put_secrets(agent_id, raw, local, secrets)
            sh.update({"credentials_set": True, "service_account_email": creds["client_email"]})
        if "spreadsheet" in sent and body.spreadsheet is not None:
            sh["spreadsheet_id"] = svc.spreadsheet_id_from(body.spreadsheet) if body.spreadsheet.strip() else ""
            if not sh["spreadsheet_id"]:
                sh["enabled"] = False
        if "sheet_name" in sent and body.sheet_name is not None:
            sh["sheet_name"] = svc.clean_sheet_name(body.sheet_name)
        for flag in ("include_test", "redact_phones"):
            if flag in sent and getattr(body, flag) is not None:
                sh[flag] = bool(getattr(body, flag))
        if "enabled" in sent and body.enabled is not None:
            if body.enabled and not (sh["credentials_set"] and sh["spreadsheet_id"]):
                raise svc.IntegrationError("SHEETS_INCOMPLETE", "Add the service-account key and the spreadsheet "
                                           "before turning Google Sheets logging on.")
            sh["enabled"] = bool(body.enabled)
    except svc.IntegrationError as e:
        raise _fail(e) from e
    raw["webhook"], raw["sheets"] = cfg["webhook"], sh
    saved = await _save(request, agent_id, user_id, raw)
    return _view(saved, local)


@router.post("/api/agents/{agent_id}/integrations/sheets/test")
async def test_sheets(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    """Append a test row to the spreadsheet and say exactly what Google answered (for example: not shared)."""
    row = await _b()._load_agent(request, agent_id, user_id)
    raw = _raw(row)
    cfg = svc.normalize_cfg(raw)
    result = await asyncio.to_thread(svc.check_sheets_connection, agent_id, user_id, _is_local(request),
                                     raw.get("secrets_enc"), cfg, str(row.get("name") or ""))
    return result


@router.delete("/api/agents/{agent_id}/integrations/sheets")
async def delete_sheets(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    row = await _b()._load_agent(request, agent_id, user_id)
    local = _is_local(request)
    raw = _raw(row)
    cfg = svc.normalize_cfg(raw)
    secrets = _stored_secrets(agent_id, raw, local)
    secrets.pop("service_account", None)
    _put_secrets(agent_id, raw, local, secrets)
    raw["webhook"] = cfg["webhook"]
    raw["sheets"] = svc.default_cfg()["sheets"]
    saved = await _save(request, agent_id, user_id, raw)
    return _view(saved, local)


# ---------------------------------------------------------------------------------------------- caller sheet

class CallerSheetUpdate(BaseModel):
    enabled: Optional[bool] = None
    spreadsheet: Optional[str] = Field(None, max_length=600)    # link or id
    sheet_name: Optional[str] = Field(None, max_length=200)     # tab; empty: "BonClay"
    include_phone: Optional[bool] = None
    include_test: Optional[bool] = None


@router.put("/api/agents/{agent_id}/integrations/caller-sheet")
async def put_caller_sheet(agent_id: str, body: CallerSheetUpdate, request: Request,
                           user_id: str = Depends(get_current_user_id)):
    """The caller-details sheet of an agent: one row per conversation (services/caller_sheet.py). It signs in with the key
    pasted under Google Sheets for this agent, else with the server's GOOGLE_SERVICE_ACCOUNT_FILE."""
    row = await _b()._load_agent(request, agent_id, user_id)
    local = _is_local(request)
    raw = _raw(row)
    cfg = svc.normalize_cfg(raw)
    cs = cfg["caller_sheet"]
    sent = body.model_fields_set
    try:
        if "spreadsheet" in sent and body.spreadsheet is not None:
            cs["spreadsheet_id"] = svc.spreadsheet_id_from(body.spreadsheet) if body.spreadsheet.strip() else ""
            if not cs["spreadsheet_id"]:
                cs["enabled"] = False
        if "sheet_name" in sent and body.sheet_name is not None:
            cs["sheet_name"] = caller_sheet.clean_tab(body.sheet_name)
        for flag in ("include_phone", "include_test"):
            if flag in sent and getattr(body, flag) is not None:
                cs[flag] = bool(getattr(body, flag))
        if "enabled" in sent and body.enabled is not None:
            if body.enabled:
                if not cs["spreadsheet_id"]:
                    raise svc.IntegrationError("CALLER_SHEET_INCOMPLETE", "Paste the spreadsheet link before turning "
                                               "this on.")
                creds, _src = await asyncio.to_thread(caller_sheet.credentials_for, agent_id, local,
                                                      raw.get("secrets_enc"))
                if not creds:
                    raise svc.IntegrationError("CALLER_SHEET_INCOMPLETE", "There is no Google service-account key yet. "
                                               "Set GOOGLE_SERVICE_ACCOUNT_FILE on the server, or paste a key under "
                                               "Google Sheets above.")
            cs["enabled"] = bool(body.enabled)
    except svc.IntegrationError as e:
        raise _fail(e) from e
    raw["webhook"], raw["sheets"], raw["caller_sheet"] = cfg["webhook"], cfg["sheets"], cs
    saved = await _save(request, agent_id, user_id, raw)
    return _view(saved, local)


@router.post("/api/agents/{agent_id}/integrations/caller-sheet/test")
async def test_caller_sheet(agent_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    """Write (or update) one clearly marked test row and say exactly what Google answered (for example: not shared)."""
    row = await _b()._load_agent(request, agent_id, user_id)
    raw = _raw(row)
    cfg = svc.normalize_cfg(raw)
    return await asyncio.to_thread(caller_sheet.check_connection, agent_id, user_id, _is_local(request),
                                   raw.get("secrets_enc"), cfg, str(row.get("name") or ""))
