"""A local MCP server for the Sunrise Family Clinic: free slots, booking, finding and cancelling appointments.

Runs over stdio, so BonClay (services/tools.py) and any other MCP client (Claude Desktop, MCP Inspector, ...) can use it:

    cd backend && .venv/bin/python -m mcp_servers.appointments

The data is one JSON file (default backend/data/mcp/appointments.json, override with BONCLAY_APPOINTMENTS_FILE). It is
created on first use with the fictional doctors and OPD hours of backend/templates/agents/clinic_receptionist.md, so
the tools and the receptionist's knowledge agree. The clinic takes no appointments on Sundays (emergency walk-ins only).

Every tool validates its input and raises AppointmentError with a message the caller can say out loud; MCP reports it
as a tool error. Times are the machine's local time. Slots are 30 minutes long."""
import contextlib
import json
import os
import re
import sys
import tempfile
import threading
from datetime import date, datetime, time as dtime, timedelta
from typing import Annotated, List, Optional

from pydantic import Field

try:  # the official MCP SDK: v2 calls the server class MCPServer, v1 called it FastMCP
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # pragma: no cover - depends on the installed SDK major version
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError

try:
    import fcntl  # one booking file shared by several server processes (BonClay and Claude Desktop at once)
except ImportError:  # pragma: no cover - Windows
    fcntl = None

SLOT_MINUTES = 30
MAX_DAYS_AHEAD = 60
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Weekday numbers (Monday = 0) to the OPD windows of that day, from the clinic template's sample document.
DEFAULT_DOCTORS: List[dict] = [
    {"name": "Dr. Meera Kulkarni", "specialty": "General Physician",
     "hours": {"0": [["09:00", "13:30"]], "2": [["09:00", "13:30"]], "4": [["09:00", "13:30"]]}},
    {"name": "Dr. Rohan Deshmukh", "specialty": "Paediatrician",
     "hours": {"1": [["16:00", "20:00"]], "3": [["16:00", "20:00"]], "5": [["16:00", "20:00"]]}},
    {"name": "Dr. Anjali Rao", "specialty": "Gynaecologist",
     "hours": {"0": [["16:00", "19:00"]], "3": [["16:00", "19:00"]]}},
    {"name": "Dr. Imran Shaikh", "specialty": "Dentist",
     "hours": {"1": [["10:00", "13:00"], ["16:00", "19:00"]], "4": [["10:00", "13:00"], ["16:00", "19:00"]]}},
    {"name": "Dr. Sunita Patil", "specialty": "Dermatologist", "hours": {"5": [["10:00", "13:00"]]}},
]

DEFAULT_SPECIALTY = "General Physician"  # who a booking without a doctor is for
_ID_FIRST = 10  # booking ids are A10, A11, ...: short enough to say on a call
_thread_lock = threading.RLock()


class AppointmentError(ToolError):
    """A request that cannot be done (bad input, closed day, slot taken). The SDK sends its message to the caller as the
    tool's error (a plain exception would be reported without its text), so it is worded to be said out loud."""


def data_path() -> str:
    return os.getenv("BONCLAY_APPOINTMENTS_FILE") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mcp", "appointments.json")


def now() -> datetime:
    """The current local time (a function so tests can move the clock)."""
    return datetime.now()


# ---------------------------------------------------------------------------------------------- storage

def _fresh() -> dict:
    return {"doctors": DEFAULT_DOCTORS, "bookings": [], "next_id": _ID_FIRST}


def _read(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("doctors"), list) and isinstance(data.get("bookings"), list):
            data.setdefault("next_id", _ID_FIRST)
            return data
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        print(f"appointments: {path} is unreadable ({e}); starting a new file", file=sys.stderr)
        with contextlib.suppress(OSError):
            os.replace(path, path + ".corrupt")
    return _fresh()


def _write(path: str, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@contextlib.contextmanager
def _transaction():
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
            if not existed or json.dumps(data, sort_keys=True) != before:  # a read-only tool leaves the file alone
                _write(path, data)
        finally:
            lock_file.close()  # releases the flock


# ---------------------------------------------------------------------------------------------- validation

def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except ValueError:
        raise AppointmentError("The date must look like 2026-10-16 (year-month-day).") from None


def _parse_time(value: str) -> dtime:
    text = re.sub(r"\s+", " ", str(value).strip().upper().replace(".", ""))
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise AppointmentError("The time must look like 14:30 (24-hour clock).")


def _hhmm(t: dtime) -> str:
    return t.strftime("%H:%M")


def _check_day(day: date) -> None:
    today = now().date()
    if day < today:
        raise AppointmentError(f"{day.isoformat()} is in the past. Appointments are for today or later.")
    if day > today + timedelta(days=MAX_DAYS_AHEAD):
        raise AppointmentError(f"Appointments can be booked up to {MAX_DAYS_AHEAD} days ahead.")
    if day.weekday() == 6:
        raise AppointmentError(f"{day.isoformat()} is a Sunday. The clinic takes no appointments on Sundays "
                         "(emergency walk-ins only). Please pick Monday to Saturday.")


def _normalize_phone(value: str) -> str:
    digits = re.sub(r"[\s\-().,]", "", str(value))  # speech recognition may write 9,876,543,210
    if digits.startswith("+"):
        digits = digits[1:]
    if not digits.isdigit():
        raise AppointmentError("The phone number may contain only digits (spaces, dashes and a leading + are fine).")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) != 10:
        raise AppointmentError("The phone number must have 10 digits.")
    return digits


def _clean_name(value: str) -> str:
    name = re.sub(r"\s+", " ", str(value)).strip()
    if not 2 <= len(name) <= 80 or not re.fullmatch(r"(?:[^\W\d_]|[ .'\-])+", name):
        raise AppointmentError("Please give the patient's name (2 to 80 letters).")
    return name


def _doctor_query(doctor: Optional[str]) -> str:
    """What to look for in a doctor's name or specialty: "Dr. Meera" -> "meera"; "" for no preference ("any doctor")."""
    query = re.sub(r"^dr\.?\s+", "", (doctor or "").strip().lower())
    return "" if query in ("doctor", "dr", "any", "anyone", "any doctor", "a doctor", "none", "null") else query


def _match_doctors(data: dict, doctor: Optional[str]) -> List[dict]:
    """Doctors whose name or specialty contains the text ("meera", "Dr. Imran Shaikh", "dentist"); all when empty."""
    doctors = data["doctors"]
    query = _doctor_query(doctor)
    if not query:
        return list(doctors)
    found = [d for d in doctors if query in d["name"].lower() or query in d["specialty"].lower()]
    if not found:
        known = ", ".join(f"{d['name']} ({d['specialty']})" for d in doctors)
        raise AppointmentError(f"No doctor matches '{doctor}'. The doctors are: {known}.")
    return found


def _day_slots(doctor: dict, day: date) -> List[dtime]:
    """All 30-minute slot starts of a doctor's OPD on a day."""
    out: List[dtime] = []
    for start, end in doctor.get("hours", {}).get(str(day.weekday()), []):
        cursor = datetime.combine(day, _parse_time(start))
        stop = datetime.combine(day, _parse_time(end))
        while cursor + timedelta(minutes=SLOT_MINUTES) <= stop:
            out.append(cursor.time())
            cursor += timedelta(minutes=SLOT_MINUTES)
    return out


def _taken(data: dict, doctor: dict, day: date) -> set:
    return {b["time"] for b in data["bookings"]
            if b["status"] == "booked" and b["date"] == day.isoformat() and b["doctor"] == doctor["name"]}


def _free_times(data: dict, doctor: dict, day: date) -> List[str]:
    taken = _taken(data, doctor, day)
    cutoff = now()
    out = []
    for slot in _day_slots(doctor, day):
        if _hhmm(slot) in taken or datetime.combine(day, slot) <= cutoff:
            continue
        out.append(_hhmm(slot))
    return out


def _public(b: dict) -> dict:
    return {k: b[k] for k in ("booking_id", "name", "phone", "doctor", "specialty", "date", "weekday", "time",
                              "reason", "status")}


# ---------------------------------------------------------------------------------------------- tools

mcp = _Server("Sunrise Family Clinic appointments")

# Short on purpose: these words go into every prompt of an agent with tools (docs/DOC_QA_EVAL.md, "Fixes after").
DATE = Field(description="YYYY-MM-DD")
TIME = Field(description="HH:MM, 24-hour, on the hour or half hour")
DOCTOR_ANY = Field(description="Doctor's name or specialty; leave out for all")
DOCTOR_DEFAULT = Field(description="Doctor's name or specialty; leave out when the caller names none (books the General "
                                    "Physician)")


@mcp.tool()
def list_free_slots(date: Annotated[str, DATE], doctor: Annotated[Optional[str], DOCTOR_ANY] = None) -> dict:
    """List the free 30-minute slots on a date, for one doctor or all."""
    day = _parse_date(date)
    _check_day(day)
    with _transaction() as data:
        free = [{"doctor": d["name"], "specialty": d["specialty"], "times": _free_times(data, d, day)}
                for d in _match_doctors(data, doctor)]
    free = [row for row in free if row["times"]]
    return {"date": day.isoformat(), "weekday": WEEKDAYS[day.weekday()], "slot_minutes": SLOT_MINUTES, "free": free}


@mcp.tool()
def book_appointment(name: Annotated[str, Field(description="Patient's name")],
                     phone: Annotated[str, Field(description="10 digits")],
                     date: Annotated[str, DATE], time: Annotated[str, TIME],
                     doctor: Annotated[Optional[str], DOCTOR_DEFAULT] = None,
                     reason: Annotated[Optional[str], Field(description="Why they are coming, a few words")] = None
                     ) -> dict:
    """Book a 30-minute appointment. Returns the booking with its booking_id."""
    patient = _clean_name(name)
    digits = _normalize_phone(phone)
    day = _parse_date(date)
    at = _parse_time(time)
    reason_text = re.sub(r"\s+", " ", str(reason or "")).strip()[:200]
    _check_day(day)
    if at.minute % SLOT_MINUTES or at.second:
        raise AppointmentError("Appointments start on the hour or the half hour (for example 10:00 or 10:30).")
    if datetime.combine(day, at) <= now():
        raise AppointmentError("That time has already passed. Please pick a later time.")
    with _transaction() as data:
        candidates = _match_doctors(data, _doctor_query(doctor) or DEFAULT_SPECIALTY)
        on_duty = [d for d in candidates if at in _day_slots(d, day)]
        if not on_duty:
            names = ", ".join(d["name"] for d in candidates)
            raise AppointmentError(f"{names} does not see patients at {_hhmm(at)} on {WEEKDAYS[day.weekday()]}. "
                             "Use list_free_slots to see the available times.")
        free = [d for d in on_duty if _hhmm(at) not in _taken(data, d, day)]
        if not free:
            raise AppointmentError(f"The {_hhmm(at)} slot on {day.isoformat()} is already booked. "
                             "Use list_free_slots to see the free times.")
        chosen = free[0]
        booking = {"booking_id": f"A{data['next_id']}", "name": patient, "phone": digits, "doctor": chosen["name"],
                   "specialty": chosen["specialty"], "date": day.isoformat(), "weekday": WEEKDAYS[day.weekday()],
                   "time": _hhmm(at), "reason": reason_text, "status": "booked",
                   "created_at": now().isoformat(timespec="seconds")}
        data["next_id"] += 1
        data["bookings"].append(booking)
    return _public(booking)


@mcp.tool()
def find_booking(phone: Annotated[str, Field(description="Phone number the booking was made with")]) -> dict:
    """Find the upcoming bookings made with a phone number."""
    digits = _normalize_phone(phone)
    cutoff = now()
    with _transaction() as data:
        found = [_public(b) for b in data["bookings"]
                 if b["status"] == "booked" and b["phone"] == digits
                 and datetime.combine(date.fromisoformat(b["date"]), _parse_time(b["time"])) > cutoff]
    found.sort(key=lambda b: (b["date"], b["time"]))
    return {"bookings": found}


@mcp.tool()
def cancel_appointment(booking_id: Annotated[str, Field(description="e.g. A12")]) -> dict:
    """Cancel an upcoming booking and free its slot. Ask the caller to confirm first."""
    wanted = str(booking_id).strip().upper().replace(" ", "")
    with _transaction() as data:
        booking = next((b for b in data["bookings"] if b["booking_id"] == wanted), None)
        if booking is None:
            raise AppointmentError(f"There is no booking {wanted}.")
        if booking["status"] != "booked":
            raise AppointmentError(f"Booking {wanted} is already cancelled.")
        if datetime.combine(date.fromisoformat(booking["date"]), _parse_time(booking["time"])) <= now():
            raise AppointmentError(f"Booking {wanted} is in the past and cannot be cancelled.")
        booking["status"] = "cancelled"
        booking["cancelled_at"] = now().isoformat(timespec="seconds")
    return _public(booking)


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from services.parent_watch import watch_parent
        watch_parent()  # exit when the process that started this server (BonClay, Claude Desktop) goes away
    except ImportError:  # pragma: no cover
        pass
    mcp.run()


if __name__ == "__main__":
    main()
