"""The HOD meeting server (mcp_servers/meetings.py): free slots from office hours, held requests that need ma'am's approval,
alternatives when a slot is not available, approvals and declines, and the Google Calendar hooks (with a fake calendar)."""
from datetime import datetime

import pytest

from mcp_servers import meetings as m
from services import gcal

NOW = datetime(2026, 10, 15, 9, 0)  # a Thursday
FRIDAY, SATURDAY, SUNDAY, MONDAY, TUESDAY = "2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19", "2026-10-20"


@pytest.fixture(autouse=True)
def meetings_file(monkeypatch, tmp_path):
    monkeypatch.setenv("BONCLAY_MEETINGS_FILE", str(tmp_path / "mcp" / "meetings.json"))
    monkeypatch.setattr(m, "now", lambda: NOW)
    monkeypatch.setitem(m.DEFAULT_SETTINGS, "sample_schedule", False)   # the made-up busy slots have their own tests
    monkeypatch.delenv("GOOGLE_CALENDAR_ID", raising=False)
    return tmp_path / "mcp" / "meetings.json"


def ask(**kw):
    args = {"name": "Riya Patil", "role": "student", "reason": "project guide approval", "date": MONDAY, "time": "10:30",
            "phone": "98765 43210", "class_or_ref": "TY AI-A"}
    args.update(kw)
    return m.request_meeting(**args)


def err(fn, *a, **kw):
    with pytest.raises(m.MeetingError) as e:
        fn(*a, **kw)
    return str(e.value)


# ------------------------------------------------------------------------------------------------ availability

def test_free_slots_follow_office_hours_and_leave_out_blocked_times():
    out = m.check_availability(MONDAY)
    assert out["weekday"] == "Monday" and out["slot_minutes"] == 20
    assert out["free"][:3] == ["10:30", "10:50", "11:10"]
    assert not {"11:30", "11:50"} & set(out["free"])          # the department meeting (11:30 to 12:00)
    assert out["free"][3] == "12:10" and "12:50" not in out["free"]   # the morning window ends at 13:00
    assert "13:00" not in out["free"] and "14:00" in out["free"] and "16:00" in out["free"] and "16:20" not in out["free"]


def test_a_closed_day_says_why_and_offers_the_next_days():
    out = m.check_availability(SATURDAY)
    assert out["free"] == [] and "does not see visitors on Saturdays" in out["reason"]
    assert [a["date"] for a in out["alternatives"]][0] == MONDAY


def test_asking_for_a_taken_time_offers_the_nearest_other_times():
    ask()                                                      # Monday 10:30 is now held
    out = m.check_availability(MONDAY, "10:30")
    assert out["available"] is False and out["reason"] == "That time is not free."
    alts = out["alternatives"]
    assert alts[0] == {"date": MONDAY, "weekday": "Monday", "time": "10:50"}        # the closest free one first
    assert len(alts) == 3 and alts[-1]["date"] == TUESDAY                           # then the next day, at about that time
    assert alts[-1]["time"] == "10:30"
    assert "Nearest options" in out["say"] and "Monday 19 October at 10:50" in out["say"]


def test_a_free_time_is_confirmed_in_words():
    out = m.check_availability(MONDAY, "2 pm")
    assert out["available"] is True and out["say"] == "Monday 19 October at 14:00 is free."


def test_slots_closer_than_the_minimum_notice_are_not_offered(monkeypatch):
    monkeypatch.setattr(m, "now", lambda: datetime(2026, 10, 16, 10, 0))   # Friday 10:00: one hour of notice -> from 11:00
    out = m.check_availability(FRIDAY)
    assert out["free"][0] == "11:10"


def test_a_full_day_offers_the_next_one():
    with m.transaction() as data:
        data["settings"]["max_per_day"] = 1
    ask()
    out = m.check_availability(MONDAY)
    assert out["free"] == [] and out["reason"] == "That day is fully booked." and out["alternatives"][0]["date"] == TUESDAY


# --------------------------------------------------------------------------------------------------- requests

def test_a_request_is_pending_and_holds_the_slot():
    r = ask()
    assert r["status"] == "pending" and r["request_id"] == "M10" and "not confirmed" in r["message"]
    assert "10:30" not in m.check_availability(MONDAY)["free"]
    msg = err(ask, name="Another Person", phone="9123456789")
    assert "That time is not free" in msg and "Monday 19 October at 10:50" in msg


def test_a_request_needs_a_real_name_role_and_reason_but_not_the_class():
    assert "name" in err(ask, name="1234")
    assert "student, a parent" in err(ask, role="alien")
    assert "what the meeting is about" in err(ask, reason="hi")
    assert ask(class_or_ref="")["status"] == "pending"                           # asked once, never required
    assert ask(role="faculty", class_or_ref="", time="10:50", phone="9000000009")["status"] == "pending"
    assert ask(role="Teacher", date=TUESDAY, phone="9000000001", class_or_ref="")["role"] == "faculty"   # said another way


def test_phone_numbers_are_checked_and_normalised():
    assert "10 digits" in err(ask, phone="12345")
    assert ask(phone="+91 98765-43210")["phone"] == "9876543210"


def test_one_person_cannot_hold_many_slots():
    ask(time="10:30")
    ask(time="10:50")
    assert "already 2 meeting requests" in err(ask, time="11:10")


def test_past_and_far_dates_are_refused():
    assert "in the past" in err(ask, date="2026-10-14")
    assert "30 days" in err(ask, date="2026-12-31")


def test_status_cancel_and_free_the_slot():
    ask()
    assert m.meeting_status(request_id="m10")["requests"][0]["status"] == "pending"
    assert m.meeting_status(phone="9876543210")["requests"][0]["request_id"] == "M10"
    assert m.cancel_meeting("M10")["status"] == "cancelled"
    assert "10:30" in m.check_availability(MONDAY)["free"]
    assert "already cancelled" in err(m.cancel_meeting, "M10")
    assert "no request M99" in err(m.meeting_status, request_id="M99")


# ----------------------------------------------------------------------------------------------- approvals

def test_ma_am_approves_declines_and_the_agent_can_report_it():
    ask()
    ask(name="Sanjay Rao", role="parent", phone="9000000002", time="14:00", class_or_ref="Neha Rao, SY AI-B")
    ok = m.approve("M10", "Please bring your report.")
    assert ok["status"] == "approved" and ok["decision_note"] == "Please bring your report."
    no = m.decline("M11", "I am in an exam meeting then.", MONDAY, "15:00")
    assert no["status"] == "declined" and no["suggested"]["time"] == "15:00"
    said = m.meeting_status(phone="9000000002")["requests"][0]
    assert said["status"] == "declined" and said["decision_note"].startswith("I am in an exam") and said["suggested"]["date"] == MONDAY
    assert "14:00" in m.check_availability(MONDAY)["free"]            # a declined request frees its slot
    assert "already approved" in err(m.approve, "M10")


def test_approving_at_another_time_moves_the_meeting_when_that_time_is_free():
    ask()
    assert m.approve("M10", new_date=TUESDAY, new_time="15:00")["time"] == "15:00"
    ask(name="Sanjay Rao", role="faculty", phone="9000000002", class_or_ref="")
    assert "is not free" in err(m.approve, "M11", new_date=TUESDAY, new_time="15:00")


# --------------------------------------------------------------------------------------------- Google Calendar

class FakeCalendar:
    def __init__(self, monkeypatch, busy_spans=(), fail=False):
        self.created, self.deleted, self.busy_spans, self.fail = [], [], list(busy_spans), fail
        monkeypatch.setattr(gcal, "configured", lambda: True)
        monkeypatch.setattr(gcal, "busy", self.busy)
        monkeypatch.setattr(gcal, "create_event", self.create_event)
        monkeypatch.setattr(gcal, "delete_event", lambda event_id: self.deleted.append(event_id))

    def busy(self, start, end):
        if self.fail:
            raise gcal.CalendarError("down")
        return [(a, b) for a, b in self.busy_spans if a < end and start < b]

    def create_event(self, summary, description, start, minutes, location="", attendees=None, request_id=""):
        self.created.append({"summary": summary, "start": start, "minutes": minutes, "request": request_id})
        return "evt1"


def test_what_is_already_on_her_calendar_is_not_offered(monkeypatch):
    FakeCalendar(monkeypatch, [(datetime(2026, 10, 19, 10, 30), datetime(2026, 10, 19, 11, 30))])
    free = m.check_availability(MONDAY)["free"]
    assert not {"10:30", "10:50", "11:10"} & set(free) and free[0] == "12:10"
    assert "that time is not free" in err(ask, time="10:30").lower()


def test_approving_puts_the_meeting_on_the_calendar_and_cancelling_removes_it(monkeypatch):
    cal = FakeCalendar(monkeypatch)
    ask()
    assert m.approve("M10")["status"] == "approved"
    assert cal.created[0]["summary"] == "Meeting: Riya Patil (student)" and cal.created[0]["start"] == datetime(2026, 10, 19, 10, 30)
    assert cal.created[0]["request"] == "M10"
    m.cancel_meeting("M10")
    assert cal.deleted == ["evt1"]


def test_a_clash_found_at_approval_time_is_reported_and_nothing_is_created(monkeypatch):
    cal = FakeCalendar(monkeypatch)
    ask()
    cal.busy_spans.append((datetime(2026, 10, 19, 10, 0), datetime(2026, 10, 19, 11, 0)))   # ma'am added a meeting meanwhile
    assert "Calendar is busy" in err(m.approve, "M10")
    assert cal.created == [] and m.list_requests("pending")[0]["request_id"] == "M10"


def test_an_unreachable_calendar_does_not_stop_requests_but_is_said(monkeypatch):
    FakeCalendar(monkeypatch, fail=True)
    out = m.check_availability(MONDAY)
    assert out["free"] and "could not be checked" in out["note"]
    assert ask()["status"] == "pending"


def test_the_tool_server_gets_the_data_file_and_the_calendar_settings(monkeypatch):
    """Regression: the meeting server read another file than the Approvals page and never saw the Google Calendar settings,
    because only the appointments file was passed to a server process."""
    from services import tools
    monkeypatch.setenv("BONCLAY_MEETINGS_FILE", "/data/meetings.json")
    monkeypatch.setenv("GOOGLE_CALENDAR_ID", "cal@group.calendar.google.com")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_FILE", "/keys/sa.json")
    monkeypatch.setenv("BONCLAY_TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("SECRET_ELSEWHERE", "no")
    env = tools.server_env({"env": {"EXTRA": 1}})
    assert env["BONCLAY_MEETINGS_FILE"] == "/data/meetings.json" and env["GOOGLE_CALENDAR_ID"] == "cal@group.calendar.google.com"
    assert env["GOOGLE_SERVICE_ACCOUNT_FILE"] == "/keys/sa.json" and env["BONCLAY_TIMEZONE"] == "Asia/Kolkata"
    assert env["EXTRA"] == "1" and "SECRET_ELSEWHERE" not in env      # nothing else leaks into the child


def test_asking_again_for_the_same_slot_returns_the_existing_request():
    first = ask(time="14:20")
    again = ask(time="14:20")
    assert again["request_id"] == first["request_id"] and again["already_requested"] is True
    assert "no new request" in again["message"]
    assert len([r for r in m.check_availability(MONDAY)["free"] if r == "14:20"]) == 0   # still held, once


def test_approve_reports_the_calendar_event(monkeypatch):
    monkeypatch.setenv("GOOGLE_CALENDAR_ID", "test@example.com")
    monkeypatch.setattr(gcal, "configured", lambda: True)
    monkeypatch.setattr(gcal, "busy", lambda *a, **k: [])
    monkeypatch.setattr(gcal, "create_event", lambda *a, **k: "evt123")
    r = ask(time="14:20")
    out = m.approve(r["request_id"])
    assert out["status"] == "approved" and out["event_id"] == "evt123" and out["decided_at"]
