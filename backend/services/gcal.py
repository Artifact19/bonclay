"""A small Google Calendar client for the HOD meeting booking (mcp_servers/meetings.py, routers/approvals.py).

It uses a Google *service account*: a robot identity that has a JSON key file. Nobody signs in with a browser, so it works on a
laptop and in a server. The calendar's owner shares the calendar with the service account's e-mail address ("Make changes
to events"), and this module then reads when the calendar is busy and creates or deletes events on it.

Settings (backend/.env):
    GOOGLE_CALENDAR_ID            the calendar's id (Calendar settings -> Integrate calendar), e.g. abc123@group.calendar.google.com
    GOOGLE_SERVICE_ACCOUNT_FILE   the key file, default backend/data/google_calendar.json (never committed)
    BONCLAY_TIMEZONE              default Asia/Kolkata

Nothing here talks to Google unless both the id and the key file exist, so without them the booking tool simply works from
its own list. `_http` is the one place a request leaves the program; tests replace it.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Tuple
from zoneinfo import ZoneInfo

try:
    from core import offline
except ImportError:  # pragma: no cover - running from a different cwd
    from backend.core import offline

log = logging.getLogger("bonclay.gcal")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCOPE = "https://www.googleapis.com/auth/calendar"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/calendar/v3"
TIMEOUT_S = 8.0


class CalendarError(RuntimeError):
    """The calendar could not be read or written (not configured, refused, unreachable)."""


def calendar_id() -> str:
    return os.getenv("GOOGLE_CALENDAR_ID", "").strip()


def key_path() -> str:
    return os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip() or os.path.join(BACKEND_DIR, "data", "google_calendar.json")


def timezone_name() -> str:
    return os.getenv("BONCLAY_TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata"


def tz() -> ZoneInfo:
    return ZoneInfo(timezone_name())


def configured() -> bool:
    if offline.enabled():
        return False
    return bool(calendar_id()) and os.path.isfile(key_path())


def status() -> dict:
    """What the settings page may say: never the key itself."""
    email = None
    try:
        with open(key_path(), "r", encoding="utf-8") as f:
            email = json.load(f).get("client_email")
    except (OSError, ValueError):
        pass
    return {"configured": configured(), "calendar_id_set": bool(calendar_id()), "key_file_found": os.path.isfile(key_path()),
            "service_account_email": email, "timezone": timezone_name()}


# -------------------------------------------------------------------------------------------------- transport

def _http(method: str, url: str, *, headers: Optional[dict] = None, params: Optional[dict] = None,
          json_body: Optional[dict] = None, data: Optional[dict] = None) -> Tuple[int, dict]:
    import httpx  # noqa: PLC0415
    try:
        r = httpx.request(method, url, headers=headers, params=params, json=json_body, data=data, timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        raise CalendarError(f"Could not reach Google Calendar ({e.__class__.__name__}).") from e
    try:
        body = r.json() if r.content else {}
    except ValueError:
        body = {"raw": r.text[:200]}
    return r.status_code, body


_token_lock = threading.Lock()
_token: dict = {"value": None, "expires": 0.0}


def _access_token() -> str:
    with _token_lock:
        if _token["value"] and _token["expires"] - 60 > time.time():
            return _token["value"]
        try:
            with open(key_path(), "r", encoding="utf-8") as f:
                key = json.load(f)
        except (OSError, ValueError) as e:
            raise CalendarError("The Google service account key file is missing or unreadable.") from e
        import jwt  # noqa: PLC0415  (PyJWT with the cryptography package)
        now = int(time.time())
        assertion = jwt.encode({"iss": key["client_email"], "scope": SCOPE, "aud": TOKEN_URL, "iat": now, "exp": now + 3000},
                               key["private_key"], algorithm="RS256")
        status_code, body = _http("POST", TOKEN_URL, data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                                                           "assertion": assertion})
        if status_code != 200 or "access_token" not in body:
            raise CalendarError(f"Google refused the service account key ({status_code}): {body.get('error_description') or body.get('error')}")
        _token.update(value=body["access_token"], expires=time.time() + int(body.get("expires_in", 3000)))
        return _token["value"]


def _call(method: str, path: str, *, params: Optional[dict] = None, json_body: Optional[dict] = None) -> dict:
    if not configured():
        raise CalendarError("Google Calendar is not set up.")
    code, body = _http(method, f"{API}{path}", headers={"Authorization": f"Bearer {_access_token()}"}, params=params,
                       json_body=json_body)
    if code >= 300:
        message = (body.get("error") or {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
        if code in (403, 404):
            message = f"{message}. Share the calendar with the service account and check GOOGLE_CALENDAR_ID."
        raise CalendarError(f"Google Calendar said {code}: {message}")
    return body


# ---------------------------------------------------------------------------------------------------- calls

def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=tz())


def busy(start: datetime, end: datetime) -> List[Tuple[datetime, datetime]]:
    """When the calendar is busy between two local times, as (start, end) pairs in local time."""
    start, end = _aware(start), _aware(end)
    body = _call("POST", "/freeBusy", json_body={"timeMin": _rfc3339(start), "timeMax": _rfc3339(end),
                                                 "timeZone": timezone_name(), "items": [{"id": calendar_id()}]})
    cal = (body.get("calendars") or {}).get(calendar_id()) or {}
    if cal.get("errors"):
        raise CalendarError(f"Google Calendar could not read this calendar: {cal['errors'][0].get('reason')}")
    out = []
    for b in cal.get("busy") or []:
        s = datetime.fromisoformat(b["start"].replace("Z", "+00:00")).astimezone(tz())
        e = datetime.fromisoformat(b["end"].replace("Z", "+00:00")).astimezone(tz())
        out.append((s.replace(tzinfo=None), e.replace(tzinfo=None)))
    return out


def create_event(summary: str, description: str, start: datetime, minutes: int, location: str = "",
                 attendees: Optional[List[str]] = None, request_id: str = "") -> str:
    """Create an event and return its id. `request_id` is stored with the event so it can be traced back."""
    start = _aware(start)
    body = {"summary": summary[:200], "description": description[:2000], "location": location[:200],
            "start": {"dateTime": start.isoformat(), "timeZone": timezone_name()},
            "end": {"dateTime": (start + timedelta(minutes=minutes)).isoformat(), "timeZone": timezone_name()},
            "extendedProperties": {"private": {"bonclayRequest": request_id}}}
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]
    return _call("POST", f"/calendars/{_quote(calendar_id())}/events", json_body=body)["id"]


def delete_event(event_id: str) -> None:
    code, _ = _http("DELETE", f"{API}/calendars/{_quote(calendar_id())}/events/{_quote(event_id)}",
                    headers={"Authorization": f"Bearer {_access_token()}"})
    if code not in (200, 204, 404, 410):
        raise CalendarError(f"Google Calendar said {code} when removing the event.")


def _quote(text: str) -> str:
    from urllib.parse import quote  # noqa: PLC0415
    return quote(text, safe="")
