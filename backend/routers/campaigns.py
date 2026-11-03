"""Outbound call campaigns (services/campaigns.py): create from CSV, start, pause, stop, delete, and a do-not-call list.

A background loop (every CAMPAIGN_TICK_S seconds, default 10) places calls for running campaigns through the same
Exotel path as a single call (routers/phone_exotel.dial). The owner attests with `caller_agreed` that everyone on the
list agreed to be called; the AI disclosure still plays at the start of every call."""
import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

try:
    from core.errors import AppError
    from routers import phone_exotel as px
    from services import call_records, campaigns
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover - imported as backend.routers.campaigns
    from backend.core.errors import AppError
    from backend.routers import phone_exotel as px
    from backend.services import call_records, campaigns
    from backend.services.auth import get_current_user_id

log = logging.getLogger("bonclay.campaigns")
router = APIRouter(tags=["campaigns"])


class WindowIn(BaseModel):
    days: Optional[List[int]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    timezone: Optional[str] = None


class CampaignIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    agent_id: str = Field(..., min_length=1, max_length=64)
    csv: str = Field(..., min_length=1, max_length=200_000, description="CSV text: a number column, optional name")
    caller_agreed: bool = Field(False, description="Everyone on this list agreed to receive this call")
    window: Optional[WindowIn] = None
    max_attempts: int = Field(2, ge=1, le=5)
    retry_minutes: int = Field(30, ge=1, le=1440)
    concurrency: int = Field(1, ge=1, le=campaigns.MAX_CONCURRENCY)
    max_per_hour: int = Field(30, ge=1, le=120)


class NumbersIn(BaseModel):
    numbers: List[str] = Field(..., min_length=1, max_length=500)


def _public(camp: Dict[str, Any], contacts: bool = False) -> Dict[str, Any]:
    out = {k: camp.get(k) for k in ("id", "name", "agent_id", "status", "created_at", "finished_at", "window",
                                    "max_attempts", "retry_minutes", "concurrency", "max_per_hour", "paused_reason")}
    out["counts"] = campaigns.counts(camp)
    out["total"] = len(camp["contacts"])
    if contacts:   # numbers are masked even for their owner: the full list is the owner's own upload
        out["contacts"] = [{"name": c.get("name", ""), "number": px.mask(c["number"]), "status": c["status"],
                            "attempts": c["attempts"], "last_status": c.get("last_status")} for c in camp["contacts"]]
    return out


def _owned(campaign_id: str, user_id: str) -> Dict[str, Any]:
    camp = campaigns.get(campaign_id)
    if not camp or camp.get("owner") != user_id:
        raise AppError("NOT_FOUND", "Campaign not found", 404)
    return camp


@router.post("/api/campaigns")
async def create_campaign(body: CampaignIn, request: Request, user_id: str = Depends(get_current_user_id)):
    if not body.caller_agreed:
        raise AppError("CONSENT_REQUIRED", "Only list people who agreed to receive this call: set caller_agreed.", 400)
    agent = await px._owned_agent(request, body.agent_id, user_id)
    parsed = campaigns.parse_csv(body.csv, px.normalise_number)
    if not parsed["contacts"]:
        raise AppError("NO_CONTACTS", "No valid phone numbers found in the list.", 422,
                       {"invalid": parsed["invalid"]})
    mode = px._b()._storage_mode(request)
    camp = campaigns.create(user_id, name=body.name, agent_id=agent["id"], mode=mode, contacts=parsed["contacts"],
                            window=body.window.model_dump() if body.window else None, max_attempts=body.max_attempts,
                            retry_minutes=body.retry_minutes, concurrency=body.concurrency,
                            max_per_hour=body.max_per_hour)
    return {**_public(camp), "import": {k: parsed[k] for k in ("invalid", "duplicates", "over_limit")}}


@router.get("/api/campaigns")
async def list_campaigns(user_id: str = Depends(get_current_user_id)):
    return {"campaigns": [_public(c) for c in campaigns.list_for(user_id)]}


@router.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str, user_id: str = Depends(get_current_user_id)):
    return _public(_owned(campaign_id, user_id), contacts=True)


@router.post("/api/campaigns/{campaign_id}/{action}")
async def campaign_action(campaign_id: str, action: str, user_id: str = Depends(get_current_user_id)):
    """start, pause or stop. Starting needs Exotel to be set up; stopping is final."""
    _owned(campaign_id, user_id)
    status = {"start": "running", "pause": "paused", "stop": "stopped"}.get(action)
    if status is None:
        raise AppError("BAD_ACTION", "Use start, pause or stop.", 422)
    if status == "running":
        miss = px.missing_settings()
        if miss:
            raise AppError("PHONE_NOT_CONFIGURED", px._not_set_up(miss), 503)
    camp = campaigns.set_status(campaign_id, user_id, status, "Paused by you." if status == "paused" else None)
    return _public(camp)


@router.delete("/api/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: str, user_id: str = Depends(get_current_user_id)):
    """Delete a campaign and the phone numbers in it."""
    _owned(campaign_id, user_id)
    campaigns.delete(campaign_id, user_id)
    return {"status": "deleted"}


@router.get("/api/do-not-call")
async def get_dnc(user_id: str = Depends(get_current_user_id)):
    return {"numbers": campaigns.dnc_all(user_id)}


@router.post("/api/do-not-call")
async def add_dnc(body: NumbersIn, user_id: str = Depends(get_current_user_id)):
    good = [n for n in (px.normalise_number(x) for x in body.numbers) if n]
    if not good:
        raise AppError("BAD_NUMBER", "Give numbers in international format, for example +919876543210.", 422)
    return {"added": campaigns.dnc_add(user_id, good), "total": len(campaigns.dnc_all(user_id))}


@router.delete("/api/do-not-call/{number}")
async def remove_dnc(number: str, user_id: str = Depends(get_current_user_id)):
    n = px.normalise_number(number)
    if not n or not campaigns.dnc_remove(user_id, n):
        raise AppError("NOT_FOUND", "That number is not on the list.", 404)
    return {"status": "removed"}


# -------------------------------------------------------------------------------------------------- the scheduler

async def _dial(camp: Dict[str, Any], contact: Dict[str, Any]) -> str:
    b = px._b()
    agent = await px._widget()._load_agent_row(camp["mode"], camp["agent_id"])
    if agent is None or agent.get("user_id") != camp["owner"]:
        raise RuntimeError("the agent no longer exists")
    if contact["number"] in set(campaigns.dnc_all(camp["owner"])):
        raise RuntimeError("number is on the do-not-call list")
    result = await px.dial(b, agent, camp["owner"], camp["mode"], contact["number"])
    if not result.get("call_sid"):
        raise RuntimeError("Exotel gave no call id")
    return result["call_sid"]


async def run_once(now: Optional[float] = None) -> int:
    """One scheduler pass over every running campaign. Returns how many calls it started. Never raises."""
    started = 0
    for camp in campaigns.running():
        try:
            out = await campaigns.tick(camp["id"], now if now is not None else time.time(), _dial, call_records.get)
            started += out["dialled"]
        except Exception as e:  # noqa: BLE001 - one bad campaign must not stop the others
            log.warning("campaign %s: scheduler step failed: %s", camp.get("id"), e)
    return started


async def _loop() -> None:
    interval = max(2.0, float(os.getenv("CAMPAIGN_TICK_S", "10") or 10))
    while True:
        await asyncio.sleep(interval)
        try:
            await run_once()
            await asyncio.to_thread(campaigns.purge_finished)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("campaign loop: %s", e)


def install(app) -> None:
    """Start the scheduler loop with the app (CAMPAIGN_SCHEDULER=off disables it)."""
    async def _start() -> None:
        if os.getenv("CAMPAIGN_SCHEDULER", "on").strip().lower() in ("off", "0", "false", "no"):
            return
        app.state.campaign_task = asyncio.create_task(_loop())

    app.on_event("startup")(_start)
