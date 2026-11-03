"""A local MCP server for the HOD's office: meeting requests with Prof. Jayashri Prasad, with her approval.

Runs over stdio like mcp_servers/appointments.py (see its docstring), so the agent and any other MCP client can use it:

    cd backend && .venv/bin/python -m mcp_servers.meetings

What the agent can do (tools): check when ma'am is free (and get the nearest alternatives when she is not), put in a
request for a slot, ask what happened to a request, and withdraw a request. What the agent cannot do: confirm a meeting.
A request is only *pending* and holds its slot until ma'am approves or declines it (routers/approvals.py, the Approvals
page). Approving puts the meeting on her Google Calendar when one is connected (services/gcal.py).

Free time is her office hours (settings below) minus: other pending or approved requests, her blocked times (lectures,
meetings), when Google Calendar is connected whatever is already on that calendar, and, with the two switches below, a made-up
sample week and the owner's own busy marks. Times are the machine's local time.

Two switches (Approvals page, routers/approvals.py), kept in the data file so this process sees them on every call:
`availability_check` (default on): on, the agent offers only free times; off, check_availability says the schedule is not
checked on calls and request_meeting takes any time inside office hours as a pending request the office will confirm.
`sample_schedule` (default on, for demos): a believable busy pattern per ISO week; `overrides` hold the owner's edits to it.

Settings live in the data file (default backend/data/mcp/meetings.json, override with BONCLAY_MEETINGS_FILE) under
"settings" and are created with the defaults below on first use; edit them to change the rules."""
import contextlib
import json
import os
import random
import re
import sys
import tempfile
import threading
from datetime import date, datetime, time as dtime, timedelta
from typing import Annotated, Any, Dict, List, Optional, Tuple

from pydantic import Field

try:  # the official MCP SDK: v2 calls the server class MCPServer, v1 called it FastMCP
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # pragma: no cover - depends on the installed SDK major version
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    from services import gcal
except ImportError:  # pragma: no cover - imported as backend.mcp_servers.meetings
    from backend.services import gcal

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ROLES = ("student", "parent", "faculty", "other")
_ID_FIRST = 10
_thread_lock = threading.RLock()

DEFAULT_SETTINGS: Dict[str, Any] = {
    "owner": "Prof. Jayashri Prasad",
    "title": "Head of the Department of Artificial Intelligence",
    "location": "the HOD cabin, AI Department",
    "slot_minutes": 20,
    "min_notice_minutes": 60,          # nobody can book a slot that starts sooner than this
    "max_days_ahead": 30,
    "max_per_day": 10,                 # approved + pending meetings in one day
    "max_open_per_person": 2,          # pending or approved upcoming requests for one phone number
    # Weekday number (Monday = 0) -> the windows she sees people in. Saturday and Sunday: none.
    "hours": {str(d): [["10:30", "13:00"], ["14:00", "16:30"]] for d in range(5)},
    # Times she is never available: [weekday or "*", "HH:MM", "HH:MM", "why"]
    "blocked": [["*", "11:30", "12:00", "Department meeting"]],
    "roles": list(ROLES),
    # The agent checks the schedule during calls. Off: it takes the caller's preferred time and the office confirms it.
    "availability_check": True,
    # Fill each week with a made-up set of busy slots (deterministic per ISO week) so a demo looks real.
    "sample_schedule": True,
    # The owner's edits per slot: {"YYYY-MM-DD": {"HH:MM": "busy" | "free"}}; "free" clears a sample busy slot.
    "overrides": {},
}
SAMPLE_EVENTS = (("Lecture", 3), ("Department meeting", 2), ("Exam duty", 3), ("Visitors", 1), ("Visitors", 2), ("Lecture", 2))
SAMPLE_CHANCE = 0.21                   # chance that an event starts at a free slot: about a third of the slots end up busy


class MeetingError(ToolError):
    """A request that cannot be done. Worded to be said out loud (the SDK sends the message to the caller as the tool's error)."""


def data_path() -> str:
    return os.getenv("BONCLAY_MEETINGS_FILE") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mcp", "meetings.json")


def now() -> datetime:
    """The current local time (a function so tests can move the clock)."""
    return datetime.now()


# ---------------------------------------------------------------------------------------------- storage

def _fresh() -> dict:
    return {"settings": json.loads(json.dumps(DEFAULT_SETTINGS)), "requests": [], "next_id": _ID_FIRST}


def _read(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("requests"), list):
            data.setdefault("next_id", _ID_FIRST)
            merged = json.loads(json.dumps(DEFAULT_SETTINGS))
            merged.update(data.get("settings") or {})
            data["settings"] = merged
            return data
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        print(f"meetings: {path} is unreadable ({e}); starting a new file", file=sys.stderr)
        with contextlib.suppress(OSError):
            os.replace(path, path + ".corrupt")
    return _fresh()


def _write(path: str, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@contextlib.contextmanager
def transaction():
    """Load the data under a lock shared by threads and processes; it is saved when the block ends without error."""
    path = data_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _thread_lock:
        lock_file = open(path + ".lock", "a")
        try:
            if fcntl:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            existed = os.path.exists(path)
            data = _read(path)
            before = json.dumps(data, sort_keys=True)
            yield data
            if not existed or json.dumps(data, sort_keys=True) != before:
                _write(path, data)
        finally:
            lock_file.close()


# ---------------------------------------------------------------------------------------------- parsing

def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except ValueError:
        raise MeetingError("The date must look like 2026-10-16 (year-month-day).") from None


def _parse_time(value: str) -> dtime:
    text = re.sub(r"\s+", " ", str(value).strip().upper().replace(".", ""))
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise MeetingError("The time must look like 14:30 (24-hour clock).")


def _hhmm(t: dtime) -> str:
    return t.strftime("%H:%M")


def _say(day: date, at: Optional[str] = None) -> str:
    """'Monday 19 October at 10:30' for the model to read out."""
    text = f"{WEEKDAYS[day.weekday()]} {day.day} {day.strftime('%B')}"
    return f"{text} at {at}" if at else text


def _clean_name(value: str) -> str:
    name = re.sub(r"\s+", " ", str(value)).strip()
    if not 2 <= len(name) <= 80 or not re.fullmatch(r"(?:[^\W\d_]|[ .'\-])+", name):
        raise MeetingError("Please give the person's name (2 to 80 letters).")
    return name


def _normalize_phone(value: str) -> str:
    digits = re.sub(r"[\s\-().,]", "", str(value))
    if digits.startswith("+"):
        digits = digits[1:]
    if not digits.isdigit():
        raise MeetingError("The phone number may contain only digits (spaces, dashes and a leading + are fine).")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) != 10:
        raise MeetingError("The phone number must have 10 digits.")
    return digits


def _clean_role(value: str, settings: dict) -> str:
    role = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    aliases = {"teacher": "faculty", "professor": "faculty", "staff": "faculty", "lecturer": "faculty", "guardian": "parent",
               "mother": "parent", "father": "parent", "mom": "parent", "dad": "parent", "pupil": "student"}
    role = aliases.get(role, role)
    if role not in settings["roles"]:
        raise MeetingError("Please say who is asking: a student, a parent, a faculty member, or someone else (other).")
    return role


# ---------------------------------------------------------------------------------------------- availability

def _windows(settings: dict, day: date) -> List[Tuple[datetime, datetime]]:
    out = []
    for start, end in settings["hours"].get(str(day.weekday()), []):
        out.append((datetime.combine(day, _parse_time(start)), datetime.combine(day, _parse_time(end))))
    return out


def _blocked(settings: dict, day: date) -> List[Tuple[datetime, datetime, str]]:
    out = []
    for entry in settings.get("blocked", []):
        wd, start, end = entry[0], entry[1], entry[2]
        if str(wd) in ("*", str(day.weekday())):
            out.append((datetime.combine(day, _parse_time(start)), datetime.combine(day, _parse_time(end)),
                        entry[3] if len(entry) > 3 else "blocked"))
    return out


def _held(data: dict, day: date, ignore: Optional[str] = None) -> List[Tuple[datetime, datetime]]:
    minutes = data["settings"]["slot_minutes"]
    out = []
    for r in data["requests"]:
        if r["status"] in ("pending", "approved") and r["date"] == day.isoformat() and r["request_id"] != ignore:
            start = datetime.combine(day, _parse_time(r["time"]))
            out.append((start, start + timedelta(minutes=minutes)))
    return out


def _calendar_busy(day: date) -> List[Tuple[datetime, datetime]]:
    """What is already on ma'am's Google Calendar that day. A calendar that cannot be read must not look empty: the
    caller is told ma'am's calendar could not be checked, and the request can still go in for her to approve."""
    if not gcal.configured():
        return []
    start = datetime.combine(day, dtime(0, 0))
    return gcal.busy(start, start + timedelta(days=1))


def _overlaps(a: Tuple[datetime, datetime], spans: List[Tuple[datetime, datetime]]) -> bool:
    return any(a[0] < s[1] and s[0] < a[1] for s in spans)


def checks_availability(settings: dict) -> bool:
    """Whether the agent checks the schedule during calls (the `availability_check` switch; on unless turned off)."""
    return settings.get("availability_check", True) is not False


def _slot_starts(settings: dict, day: date) -> List[datetime]:
    """Start of every slot inside the day's office windows (the grid the Approvals page shows)."""
    step = timedelta(minutes=settings["slot_minutes"])
    out = []
    for w_start, w_end in _windows(settings, day):
        cursor = w_start
        while cursor + step <= w_end:
            out.append(cursor)
            cursor += step
    return out


def _overrides(settings: dict, day: date) -> Dict[str, str]:
    """The owner's marks for one day: {"HH:MM": "busy" | "free"}; anything malformed in a hand-edited file is ignored."""
    marks = settings.get("overrides")
    day_marks = marks.get(day.isoformat()) if isinstance(marks, dict) else None
    if not isinstance(day_marks, dict):
        return {}
    return {k: v for k, v in day_marks.items() if v in ("busy", "free")}


def _sample_cells(settings: dict, day: date) -> Dict[str, str]:
    """The made-up busy slots of a day, {"HH:MM": why}. The same ISO week always gives the same pattern (each weekday has
    its own seeded generator, so a change of hours on one day does not move the others). Events sit inside one window."""
    if not settings.get("sample_schedule") or day.weekday() > 4:
        return {}
    iso = day.isocalendar()
    rng = random.Random(f"bonclay-sample-{iso[0]}-W{iso[1]}-{day.weekday()}")
    step = timedelta(minutes=settings["slot_minutes"])
    out: Dict[str, str] = {}
    for w_start, w_end in _windows(settings, day):
        slots, cursor = [], w_start
        while cursor + step <= w_end:
            slots.append(cursor)
            cursor += step
        i = 0
        while i < len(slots):
            if rng.random() < SAMPLE_CHANCE:
                why, length = rng.choice(SAMPLE_EVENTS)
                for slot in slots[i:i + length]:
                    out[_hhmm(slot.time())] = why
                i += length
            else:
                i += 1
    return out


def _soft_busy(settings: dict, day: date) -> List[Tuple[datetime, datetime, str, str]]:
    """Busy slots that come from the sample week and the owner's marks: (start, end, why, source), source "sample" or "owner".
    A "free" mark clears a sample slot; a "busy" mark makes any slot busy."""
    marks = _overrides(settings, day)
    step = timedelta(minutes=settings["slot_minutes"])
    out = []
    for at, why in _sample_cells(settings, day).items():
        if marks.get(at) != "free":
            start = datetime.combine(day, _parse_time(at))
            out.append((start, start + step, why, "sample"))
    for at, mark in marks.items():
        if mark == "busy":
            start = datetime.combine(day, _parse_time(at))
            out.append((start, start + step, "Marked busy by you", "owner"))
    return out


def free_times(data: dict, day: date, ignore: Optional[str] = None, calendar: Optional[list] = None,
               soft: Optional[bool] = None) -> List[str]:
    """Start times (HH:MM) of the free slots of a day, after the minimum notice. The sample week and the owner's busy marks
    count as busy when `soft` is true; by default that follows the availability_check switch (approvals pass True: she sees
    the same grid whichever way the agent is set)."""
    s = data["settings"]
    step = timedelta(minutes=s["slot_minutes"])
    held = _held(data, day, ignore)
    if len([r for r in data["requests"] if r["status"] in ("pending", "approved") and r["date"] == day.isoformat()
            and r["request_id"] != ignore]) >= s["max_per_day"]:
        return []
    if soft is None:
        soft = checks_availability(s)
    blocked = [(a, b) for a, b, _ in _blocked(s, day)] + ([(a, b) for a, b, _, _ in _soft_busy(s, day)] if soft else [])
    busy = calendar if calendar is not None else _calendar_busy(day)
    earliest = now() + timedelta(minutes=s["min_notice_minutes"])
    out = []
    for w_start, w_end in _windows(s, day):
        cursor = w_start
        while cursor + step <= w_end:
            slot = (cursor, cursor + step)
            if cursor >= earliest and not _overlaps(slot, held + blocked + busy):
                out.append(_hhmm(cursor.time()))
            cursor += step
    return out


def _next_days(start: date, s: dict):
    today = now().date()
    last = today + timedelta(days=s["max_days_ahead"])
    d = start
    while d <= last:
        yield d
        d += timedelta(days=1)


def alternatives(data: dict, day: date, at: Optional[str] = None, count: int = 3) -> List[dict]:
    """Up to `count` other options: the free slots nearest to the wanted time on the same day (one before and one after
    first), then the first free slots of the following days."""
    s = data["settings"]
    found: List[dict] = []
    same = free_times(data, day)
    if same:
        if at:
            want = _parse_time(at)
            ref = datetime.combine(day, want)
            ordered = sorted(same, key=lambda t: abs((datetime.combine(day, _parse_time(t)) - ref).total_seconds()))
            pick = ordered[:2]  # the closest two (they are often one on each side)
        else:
            pick = same[:2]
        found += [{"date": day.isoformat(), "weekday": WEEKDAYS[day.weekday()], "time": t} for t in sorted(pick)]
    for d in _next_days(day + timedelta(days=1), s):
        if len(found) >= count:
            break
        times = free_times(data, d)
        if times:
            # a slot at about the same time of day first, so "Monday at 11" becomes "Tuesday at 11" when possible
            near = min(times, key=lambda t: abs((datetime.combine(d, _parse_time(t)) - datetime.combine(d, _parse_time(at))).total_seconds())) if at else times[0]
            found.append({"date": d.isoformat(), "weekday": WEEKDAYS[d.weekday()], "time": near})
    return found[:count]


def _describe_alternatives(alts: List[dict]) -> str:
    return "; ".join(_say(date.fromisoformat(a["date"]), a["time"]) for a in alts)


def _check_day(s: dict, day: date) -> None:
    today = now().date()
    if day < today:
        raise MeetingError(f"{day.isoformat()} is in the past. Please pick today or a later day.")
    if day > today + timedelta(days=s["max_days_ahead"]):
        raise MeetingError(f"Meetings can be booked up to {s['max_days_ahead']} days ahead.")


def _why_closed(s: dict, day: date) -> str:
    if not _windows(s, day):
        return f"{s['owner']} does not see visitors on {WEEKDAYS[day.weekday()]}s."
    return ""


def _hours_text(s: dict, day: date) -> str:
    return " and ".join(f"{a} to {b}" for a, b in s["hours"].get(str(day.weekday()), []))


def _check_office_hours(s: dict, day: date, at: str) -> None:
    """For a request taken without checking the schedule: the day must be open, the time inside a window with room for a whole
    slot, and not already past. Nothing else is looked at."""
    closed = _why_closed(s, day)
    if closed:
        raise MeetingError(f"{closed} Please choose another day.")
    start = datetime.combine(day, _parse_time(at))
    if not any(w_start <= start and start + timedelta(minutes=s["slot_minutes"]) <= w_end for w_start, w_end in _windows(s, day)):
        raise MeetingError(f"{at} is outside office hours. On {WEEKDAYS[day.weekday()]}s that is {_hours_text(s, day)}. "
                           "Please choose a time inside those hours.")
    if start <= now():
        raise MeetingError("That time has already passed. Please choose a later time.")


def _not_checked(s: dict, day: date, at: Optional[str]) -> dict:
    """What check_availability returns when the owner has turned the schedule check off. It says nothing about free or busy."""
    out = {"date": day.isoformat(), "weekday": WEEKDAYS[day.weekday()], "slot_minutes": s["slot_minutes"], "schedule_checked": False}
    if at:
        out["time"] = at
    closed = _why_closed(s, day)
    if closed:
        out.update(office_open=False, reason=closed, say=f"{closed} Could you pick another day?")
        return out
    out["office_hours"] = _hours_text(s, day)
    out["say"] = ("I can't check the schedule on this call, so I can't say whether that time is free. I'll take your preferred "
                  "day and time, and the office will confirm it.")
    out["next"] = ("Do not say any time is free, busy or booked. Collect the caller's name, role, reason and phone number, then "
                   "call request_meeting with their preferred day and time (inside the office hours above). Tell them the "
                   "office will confirm it.")
    return out


# ---------------------------------------------------------------------------------------------- views

def _public(r: dict) -> dict:
    keys = ("request_id", "name", "role", "reason", "date", "weekday", "time", "minutes", "status", "decision_note", "phone")
    return {k: r.get(k) for k in keys if k in r}


# ---------------------------------------------------------------------------------------------- tools

mcp = _Server("HOD office meetings")

# Short on purpose: these words go into every prompt of an agent with tools (docs/DOC_QA_EVAL.md, "Fixes after").
DATE = Field(description="YYYY-MM-DD")
TIME = Field(description="HH:MM, 24-hour")


@mcp.tool()
def check_availability(date: Annotated[str, DATE],
                       time: Annotated[Optional[str], Field(description="HH:MM, 24-hour; leave out for all the day's "
                                                            "free slots")] = None) -> dict:
    """Check when the HOD is free. With a time: whether that slot is free and, if not, the nearest other options. Without:
    the day's free slots, or the next days with room. If it says the schedule is not checked, never call a time free or
    busy; take the caller's preferred time."""
    day = _parse_date(date)
    with transaction() as data:
        s = data["settings"]
        _check_day(s, day)
        if not checks_availability(s):
            return _not_checked(s, day, _hhmm(_parse_time(time)) if time else None)
        try:
            free = free_times(data, day)
            calendar_note = None
        except gcal.CalendarError:
            free, calendar_note = free_times(data, day, calendar=[]), "Her Google Calendar could not be checked just now."
        closed = _why_closed(s, day)
        if time:
            at = _hhmm(_parse_time(time))
            ok = at in free
            out = {"date": day.isoformat(), "weekday": WEEKDAYS[day.weekday()], "time": at, "available": ok,
                   "slot_minutes": s["slot_minutes"]}
            if not ok:
                alts = alternatives(data, day, at)
                reason = closed or ("That day is fully booked." if not free else "That time is not free.")
                out.update(reason=reason, alternatives=alts, say=f"{reason} Nearest options: {_describe_alternatives(alts)}." if alts
                           else f"{reason} There is no free slot in the next {s['max_days_ahead']} days.")
            else:
                out["say"] = f"{_say(day, at)} is free."
        else:
            out = {"date": day.isoformat(), "weekday": WEEKDAYS[day.weekday()], "slot_minutes": s["slot_minutes"], "free": free}
            if not free:
                alts = alternatives(data, day)
                out.update(reason=closed or "That day is fully booked.", alternatives=alts)
        if calendar_note:
            out["note"] = calendar_note
        return out


@mcp.tool()
def request_meeting(name: Annotated[str, Field(description="Visitor's full name")],
                    role: Annotated[str, Field(description="student, parent, faculty or other")],
                    reason: Annotated[str, Field(description="Why, in a few words")],
                    date: Annotated[str, DATE], time: Annotated[str, TIME],
                    phone: Annotated[str, Field(description="10 digits")],
                    class_or_ref: Annotated[Optional[str], Field(description="Student's class and division, or a parent's "
                                                                 "child's name and class; ask once, leave out if not given")] = None,
                    ) -> dict:
    """Request a meeting with the HOD. NOT confirmed: the HOD approves or declines it later; the slot is held until then.
    Check the slot with check_availability first. Returns a request_id to tell the caller; relay what the result says
    about confirmation."""
    with transaction() as data:
        s = data["settings"]
        person = _clean_name(name)
        who = _clean_role(role, s)
        digits = _normalize_phone(phone)
        why = re.sub(r"\s+", " ", str(reason or "")).strip()[:240]
        if len(why) < 4:
            raise MeetingError("Please tell me briefly what the meeting is about, so that ma'am can decide.")
        ref = re.sub(r"\s+", " ", str(class_or_ref or "")).strip()[:120]
        day = _parse_date(date)
        at = _hhmm(_parse_time(time))
        _check_day(s, day)
        open_now = [r for r in data["requests"] if r["phone"] == digits and r["status"] in ("pending", "approved")
                    and datetime.combine(date_cls(r["date"]), _parse_time(r["time"])) > now()]
        # The same caller asking again for the same slot (often a "yes, go ahead" after the request was already made):
        # give back the request they already have instead of calling their own held slot "not free".
        same = next((r for r in open_now if r["date"] == day.isoformat() and r["time"] == at), None)
        if same is not None:
            state = "already approved" if same["status"] == "approved" else "already with " + s["owner"] + " for approval"
            return {**_public(same), "already_requested": True,
                    "message": f"Request {same['request_id']} for {_say(day, at)} is {state}; no new request was made."}
        if len(open_now) >= s["max_open_per_person"]:
            raise MeetingError(f"There are already {len(open_now)} meeting requests under this number "
                               f"({', '.join(r['request_id'] for r in open_now)}). Please ask about those, or cancel one first.")
        checked = checks_availability(s)
        if checked:
            try:
                free = free_times(data, day)
            except gcal.CalendarError:
                free = free_times(data, day, calendar=[])
            if at not in free:
                alts = alternatives(data, day, at)
                reason_text = _why_closed(s, day) or ("That day is fully booked." if not free else "That time is not free.")
                more = f" Nearest options: {_describe_alternatives(alts)}." if alts else ""
                raise MeetingError(f"{reason_text}{more} Please choose one of these.")
        else:  # the schedule is not checked: any time inside office hours goes in as a request for the office to confirm
            _check_office_hours(s, day, at)
        request = {"request_id": f"M{data['next_id']}", "name": person, "role": who, "phone": digits, "class_or_ref": ref,
                   "reason": why, "date": day.isoformat(), "weekday": WEEKDAYS[day.weekday()], "time": at,
                   "minutes": s["slot_minutes"], "status": "pending", "created_at": now().isoformat(timespec="seconds")}
        data["next_id"] += 1
        data["requests"].append(request)
    if not checked:
        return {**_public(request), "schedule_checked": False,
                "message": f"Request {request['request_id']} for {_say(day, at)} is with {s['owner']}'s office. The schedule was "
                           "not checked on this call, so nothing is confirmed: the office will confirm the time or suggest "
                           "another. Tell the caller exactly that, and do not say the time is available or booked."}
    return {**_public(request), "message": f"Request {request['request_id']} is with {s['owner']} for approval. It is not "
            f"confirmed yet. The slot is held for {_say(day, at)}."}


def date_cls(text: str) -> date:
    return date.fromisoformat(text)


@mcp.tool()
def meeting_status(request_id: Annotated[Optional[str], Field(description="e.g. M12")] = None,
                   phone: Annotated[Optional[str], Field(description="Phone number the request was made with")] = None) -> dict:
    """What happened to a meeting request: pending, approved (with the time) or declined (with the HOD's note and any
    suggested time). Give the request_id, or the phone number for the person's recent requests."""
    if not request_id and not phone:
        raise MeetingError("Please give the request number or the phone number.")
    with transaction() as data:
        if request_id:
            wanted = str(request_id).strip().upper().replace(" ", "")
            found = [r for r in data["requests"] if r["request_id"] == wanted]
            if not found:
                raise MeetingError(f"There is no request {wanted}.")
        else:
            digits = _normalize_phone(phone)
            found = [r for r in data["requests"] if r["phone"] == digits and r["status"] in ("pending", "approved", "declined")]
            found = sorted(found, key=lambda r: (r["date"], r["time"]))[-3:]
        return {"requests": [{**_public(r), "suggested": r.get("suggested")} for r in found]}


@mcp.tool()
def cancel_meeting(request_id: Annotated[str, Field(description="e.g. M12")]) -> dict:
    """Withdraw a pending or approved request and free the slot. Ask the caller to confirm first."""
    wanted = str(request_id).strip().upper().replace(" ", "")
    with transaction() as data:
        r = next((x for x in data["requests"] if x["request_id"] == wanted), None)
        if r is None:
            raise MeetingError(f"There is no request {wanted}.")
        if r["status"] not in ("pending", "approved"):
            raise MeetingError(f"Request {wanted} is already {r['status']}.")
        if datetime.combine(date_cls(r["date"]), _parse_time(r["time"])) <= now():
            raise MeetingError(f"Request {wanted} is in the past.")
        was_approved = r["status"] == "approved"
        r["status"], r["cancelled_at"] = "cancelled", now().isoformat(timespec="seconds")
        event_id = r.get("event_id")
    if was_approved and event_id and gcal.configured():
        with contextlib.suppress(gcal.CalendarError):
            gcal.delete_event(event_id)
    return _public(r)


# ---------------------------------------------------------------------------------------------- approvals (not tools)

def list_requests(status: Optional[str] = None) -> List[dict]:
    with transaction() as data:
        rows = [r for r in data["requests"] if status in (None, "", "all") or r["status"] == status]
    return sorted(rows, key=lambda r: (r["date"], r["time"]))


def settings() -> dict:
    with transaction() as data:
        return json.loads(json.dumps(data["settings"]))


def update_settings(availability_check: Optional[bool] = None, sample_schedule: Optional[bool] = None) -> dict:
    """Set the two switches (None leaves one as it is) and return both."""
    with transaction() as data:
        s = data["settings"]
        if availability_check is not None:
            s["availability_check"] = bool(availability_check)
        if sample_schedule is not None:
            s["sample_schedule"] = bool(sample_schedule)
        return {"availability_check": checks_availability(s), "sample_schedule": bool(s.get("sample_schedule"))}


def _request_cell(s: dict, day: date, starts: List[datetime], r: dict) -> Optional[str]:
    """The HH:MM of the grid slot a request sits in: its own start, or the first slot it overlaps."""
    start = datetime.combine(day, _parse_time(r["time"]))
    end = start + timedelta(minutes=r.get("minutes") or s["slot_minutes"])
    step = timedelta(minutes=s["slot_minutes"])
    if start in starts:
        return _hhmm(start.time())
    hit = next((x for x in starts if x < end and start < x + step), None)
    return _hhmm(hit.time()) if hit else None


def set_slot(day_text: str, time_text: str, state: str) -> dict:
    """The owner marks one slot busy or free, or clears her mark ("clear"). Past slots, times outside office hours, slots that
    hold a request and slots blocked by the office rules are refused with a MeetingError. A "free" mark is also how a sample
    busy slot is cleared."""
    if state not in ("busy", "free", "clear"):
        raise MeetingError("The state must be busy, free or clear.")
    day, at = _parse_date(day_text), _hhmm(_parse_time(time_text))
    with transaction() as data:
        s = data["settings"]
        today = now().date()
        if day < today or day > today + timedelta(days=s["max_days_ahead"]):
            raise MeetingError("That day is outside the days that can be booked.")
        starts = _slot_starts(s, day)
        slot = datetime.combine(day, _parse_time(at))
        if slot not in starts:
            raise MeetingError("That time is outside office hours.")
        if slot <= now():
            raise MeetingError("That time has already passed.")
        if any(_request_cell(s, day, starts, r) == at for r in data["requests"]
               if r["status"] in ("pending", "approved") and r["date"] == day.isoformat()):
            raise MeetingError("A meeting request holds this time. Approve or decline it first.")
        step = timedelta(minutes=s["slot_minutes"])
        clash = next((why for a, b, why in _blocked(s, day) if a < slot + step and slot < b), None)
        if clash and state != "clear":
            raise MeetingError(f"This time is blocked in the office rules ({clash}).")
        marks = s["overrides"] = {d: m for d, m in (s.get("overrides") or {}).items()
                                  if isinstance(m, dict) and d >= today.isoformat()}  # marks for days gone by are dropped
        day_marks = marks.setdefault(day.isoformat(), {})
        if state == "clear":
            day_marks.pop(at, None)
        else:
            day_marks[at] = state
        if not day_marks:
            marks.pop(day.isoformat(), None)
        return {"date": day.isoformat(), "time": at, "state": state}


def week_view(start: date) -> dict:
    """The week (Monday to Sunday) that holds `start` as the Approvals page draws it. See routers/approvals.py for the shape.
    The grid is built from the same rules free_times uses, so the agent and the page agree about which slots are free."""
    monday = start - timedelta(days=start.weekday())
    with transaction() as data:
        snap = json.loads(json.dumps(data))
    s = snap["settings"]
    step = timedelta(minutes=s["slot_minutes"])
    current = now()
    today = current.date()
    last_day = today + timedelta(days=s["max_days_ahead"])
    dates = [monday + timedelta(days=i) for i in range(7)]
    grid = {d: _slot_starts(s, d) for d in dates}
    times = sorted({_hhmm(x.time()) for starts in grid.values() for x in starts})
    days, cells = [], {}
    for d in dates:
        starts = set(grid[d])
        held = [r for r in snap["requests"] if r["date"] == d.isoformat() and r["status"] in ("pending", "approved")]
        info = {"date": d.isoformat(), "weekday": WEEKDAYS[d.weekday()], "open": bool(starts), "full": len(held) >= s["max_per_day"]}
        calendar: List[Tuple[datetime, datetime]] = []
        if starts and d >= today and d <= last_day and gcal.configured():
            try:
                calendar = _calendar_busy(d)
            except gcal.CalendarError:
                info["calendar_error"] = True
        blocked = _blocked(s, d)
        marks = _overrides(s, d)
        sample = _sample_cells(s, d)
        by_cell: Dict[str, List[dict]] = {}
        for r in held:
            at = _request_cell(s, d, sorted(starts), r)
            if at:
                by_cell.setdefault(at, []).append(r)
        day_cells = {}
        for t in times:
            slot = datetime.combine(d, _parse_time(t))
            span = (slot, slot + step)
            if slot not in starts:
                day_cells[t] = {"state": "closed"}
                continue
            if d > last_day:
                day_cells[t] = {"state": "closed", "reason": "Too far ahead to book"}
                continue
            asked = by_cell.get(t)
            if asked:
                r = next((x for x in asked if x["status"] == "approved"), asked[0])
                day_cells[t] = {"state": "booked" if r["status"] == "approved" else "asked_for",
                                "request": {"request_id": r["request_id"], "name": r["name"], "role": r["role"],
                                            "on_calendar": bool(r.get("event_id"))}}
                if len(asked) > 1:
                    day_cells[t]["more"] = len(asked) - 1
            elif slot < current:
                day_cells[t] = {"state": "past"}
            elif (why := next((w for a, b, w in blocked if a < span[1] and span[0] < b), None)) is not None:
                day_cells[t] = {"state": "busy", "reason": why, "source": "blocked"}
            elif _overlaps(span, calendar):
                day_cells[t] = {"state": "busy", "reason": "Busy on Google Calendar", "source": "calendar"}
            elif marks.get(t) == "busy":
                day_cells[t] = {"state": "busy", "reason": "Marked busy by you", "source": "owner"}
            elif t in sample and marks.get(t) != "free":
                day_cells[t] = {"state": "busy", "reason": sample[t], "source": "sample"}
            else:
                day_cells[t] = {"state": "free"}
                if slot < current + timedelta(minutes=s["min_notice_minutes"]):
                    day_cells[t]["too_soon"] = True   # free, but too close for the agent to offer it
        days.append(info)
        cells[d.isoformat()] = day_cells
    return {"start": monday.isoformat(), "today": today.isoformat(), "max_days_ahead": s["max_days_ahead"],
            "slot_minutes": s["slot_minutes"], "owner": s.get("owner"), "label": s.get("label") or None,
            "days": days, "times": times, "cells": cells,
            "settings": {"availability_check": checks_availability(s), "sample_schedule": bool(s.get("sample_schedule"))}}


def approve(request_id: str, note: str = "", new_date: Optional[str] = None, new_time: Optional[str] = None) -> dict:
    """Ma'am approves a pending request, optionally at another time. The meeting goes on her calendar when one is connected;
    if the calendar says she is busy then, nothing changes and a MeetingError says so."""
    with transaction() as data:
        s = data["settings"]
        r = next((x for x in data["requests"] if x["request_id"] == str(request_id).strip().upper()), None)
        if r is None:
            raise MeetingError(f"There is no request {request_id}.")
        if r["status"] != "pending":
            raise MeetingError(f"Request {r['request_id']} is already {r['status']}.")
        day = _parse_date(new_date) if new_date else date_cls(r["date"])
        at = _hhmm(_parse_time(new_time)) if new_time else r["time"]
        if (day.isoformat(), at) != (r["date"], r["time"]):
            if at not in free_times(data, day, ignore=r["request_id"], calendar=_safe_busy(day), soft=True):
                raise MeetingError(f"{_say(day, at)} is not free.")
        start = datetime.combine(day, _parse_time(at))
        event_id = None
        if gcal.configured():
            clash = [b for b in gcal.busy(start, start + timedelta(minutes=r["minutes"]))]
            if clash:
                raise MeetingError("Her Google Calendar is busy at that time. Pick another time or decline.")
            event_id = gcal.create_event(f"Meeting: {r['name']} ({r['role']})",
                                         f"{r['reason']}\nRef: {r.get('class_or_ref') or '-'}\nPhone: {r['phone']}\nBooked by the BonClay agent, request {r['request_id']}.",
                                         start, r["minutes"], s["location"], request_id=r["request_id"])
        r.update(date=day.isoformat(), weekday=WEEKDAYS[day.weekday()], time=at, status="approved",
                 decision_note=(note or "").strip()[:300], decided_at=now().isoformat(timespec="seconds"))
        if event_id:
            r["event_id"] = event_id
        return {**_public(r), **{k: r[k] for k in ("event_id", "created_at", "decided_at", "class_or_ref") if k in r}}


def _safe_busy(day: date) -> list:
    try:
        return _calendar_busy(day)
    except gcal.CalendarError:
        return []


def decline(request_id: str, note: str = "", suggest_date: Optional[str] = None, suggest_time: Optional[str] = None) -> dict:
    """Ma'am declines a pending request, with an optional note and a suggested other time the agent will offer."""
    with transaction() as data:
        r = next((x for x in data["requests"] if x["request_id"] == str(request_id).strip().upper()), None)
        if r is None:
            raise MeetingError(f"There is no request {request_id}.")
        if r["status"] != "pending":
            raise MeetingError(f"Request {r['request_id']} is already {r['status']}.")
        r.update(status="declined", decision_note=(note or "").strip()[:300], decided_at=now().isoformat(timespec="seconds"))
        if suggest_date and suggest_time:
            d = _parse_date(suggest_date)
            r["suggested"] = {"date": d.isoformat(), "weekday": WEEKDAYS[d.weekday()], "time": _hhmm(_parse_time(suggest_time))}
        return {**_public(r), "suggested": r.get("suggested")}


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from services.parent_watch import watch_parent
        watch_parent()
    except ImportError:  # pragma: no cover
        pass
    mcp.run()


if __name__ == "__main__":
    main()
