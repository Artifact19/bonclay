"""Meeting approvals for the HOD office agent (mcp_servers/meetings.py): what the agent has put in as pending requests,
and ma'am's decisions.

    GET  /api/meetings?status=pending|approved|declined|cancelled|all   the requests, soonest first, with counts
    GET  /api/meetings/settings                                         her office rules and the Google Calendar status
    GET  /api/meetings/week?start=YYYY-MM-DD                            one week as a grid: days, slot times, what each cell is
    PUT  /api/meetings/settings   {availability_check?, sample_schedule?} the two switches (the agent checks the schedule on
                                                                         calls; a made-up sample week fills the grid)
    PUT  /api/meetings/slots      {date, time, state: busy|free|clear}   the owner's mark on one slot
    POST /api/meetings/{request_id}/approve   {note?, date?, time?}      approve, optionally at another time; the meeting
                                                                         goes on her Google Calendar when it is connected
    POST /api/meetings/{request_id}/decline   {note?, suggest_date?, suggest_time?}

The agent never confirms a meeting; only these two calls change a pending request. A refused decision (the slot is no longer free,
her calendar is busy then, the request was already decided) is a 409 with a message that can be shown as it is."""
import asyncio
import logging
from datetime import date, datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, StrictBool

try:
    from core.errors import AppError
    from mcp_servers import meetings
    from services import gcal
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover - imported as backend.routers.approvals
    from backend.core.errors import AppError
    from backend.mcp_servers import meetings
    from backend.services import gcal
    from backend.services.auth import get_current_user_id

log = logging.getLogger("bonclay.approvals")
router = APIRouter(tags=["meetings"])

STATUSES = ("pending", "approved", "declined", "cancelled")


class ApproveBody(BaseModel):
    note: str = Field("", max_length=300)
    date: Optional[str] = Field(None, description="Another day, YYYY-MM-DD, to approve the meeting at a different time")
    time: Optional[str] = Field(None, description="Another start time, HH:MM")


class DeclineBody(BaseModel):
    note: str = Field("", max_length=300)
    suggest_date: Optional[str] = None
    suggest_time: Optional[str] = None


class SettingsBody(BaseModel):
    availability_check: Optional[StrictBool] = Field(None, description="The agent checks the schedule during calls")
    sample_schedule: Optional[StrictBool] = Field(None, description="Fill each week with a made-up set of busy slots")


class SlotBody(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")
    time: str = Field(..., description="HH:MM, a slot start")
    state: Literal["busy", "free", "clear"]


async def _run(fn, *args, **kwargs):
    """Run a blocking meetings call off the event loop; its refusals become a 409 whose message is fit to show."""
    try:
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))
    except meetings.MeetingError as e:
        raise AppError("MEETING_REFUSED", str(e), 409) from e
    except gcal.CalendarError as e:
        raise AppError("CALENDAR_ERROR", str(e), 502) from e


def _view(r: dict) -> dict:
    out = {k: r.get(k) for k in ("request_id", "name", "role", "phone", "class_or_ref", "reason", "date", "weekday", "time",
                                 "minutes", "status", "decision_note", "suggested", "created_at", "decided_at")}
    out["on_calendar"] = bool(r.get("event_id"))
    return out


@router.get("/api/meetings")
async def list_meetings(status: str = "pending", user_id: str = Depends(get_current_user_id)):
    if status != "all" and status not in STATUSES:
        raise AppError("BAD_STATUS", f"status must be one of {', '.join(STATUSES)} or all", 422)
    every = await _run(meetings.list_requests, None)
    counts = {s: sum(1 for r in every if r["status"] == s) for s in STATUSES}
    rows = [r for r in every if status == "all" or r["status"] == status]
    return {"requests": [_view(r) for r in rows], "counts": counts}


@router.get("/api/meetings/settings")
async def meeting_settings(user_id: str = Depends(get_current_user_id)):
    return {"settings": await _run(meetings.settings), "calendar": gcal.status()}


@router.get("/api/meetings/week")
async def meeting_week(start: Optional[str] = None, user_id: str = Depends(get_current_user_id)):
    """The week (Monday to Sunday) that holds `start`, default this week:
    `days` [{date, weekday, open, full, calendar_error?}], `times` (every slot start in the week's office windows) and
    `cells` {date: {HH:MM: {state, reason?, source?, request?, too_soon?}}}. `state` is closed, past, busy (source blocked,
    calendar, owner or sample), asked_for (a pending request), booked (an approved one) or free. A calendar that cannot be read
    marks the day `calendar_error` instead of looking empty. Also the two switches, the slot length and the calendar status."""
    if start:
        try:
            day = datetime.strptime(start.strip(), "%Y-%m-%d").date()
        except ValueError:
            raise AppError("BAD_DATE", "start must look like 2026-10-19 (year-month-day)", 422) from None
        if not date(2000, 1, 1) <= day <= date(2100, 12, 31):
            raise AppError("BAD_DATE", "start is outside the supported years", 422)
    else:
        day = meetings.now().date()
    view = await _run(meetings.week_view, day)
    return {**view, "calendar": gcal.status()}


@router.put("/api/meetings/settings")
async def update_meeting_settings(body: SettingsBody, user_id: str = Depends(get_current_user_id)):
    if body.availability_check is None and body.sample_schedule is None:
        raise AppError("BAD_SETTINGS", "Send availability_check, sample_schedule or both (true or false).", 422)
    out = await _run(meetings.update_settings, body.availability_check, body.sample_schedule)
    log.info("meeting switches: %s", out)
    return out


@router.put("/api/meetings/slots")
async def mark_meeting_slot(body: SlotBody, user_id: str = Depends(get_current_user_id)):
    return await _run(meetings.set_slot, body.date, body.time, body.state)


@router.post("/api/meetings/{request_id}/approve")
async def approve_meeting(request_id: str, body: ApproveBody, user_id: str = Depends(get_current_user_id)):
    r = await _run(meetings.approve, request_id, body.note, body.date, body.time)
    log.info("meeting %s approved", r["request_id"])
    return _view(r)


@router.post("/api/meetings/{request_id}/decline")
async def decline_meeting(request_id: str, body: DeclineBody, user_id: str = Depends(get_current_user_id)):
    r = await _run(meetings.decline, request_id, body.note, body.suggest_date, body.suggest_time)
    log.info("meeting %s declined", r["request_id"])
    return _view(r)
