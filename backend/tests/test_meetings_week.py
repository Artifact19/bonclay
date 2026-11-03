"""The weekly schedule of the HOD meeting feature: GET /api/meetings/week, the sample week, the owner's marks
(PUT /api/meetings/slots), the two switches (PUT /api/meetings/settings) and what the agent's tools do when the schedule
check is off. The agent and the grid must agree: with the check on, the grid's free cells are the times free_times offers."""
import json
from datetime import datetime

import pytest

from mcp_servers import meetings as m
from services import gcal
from tests.test_offline_agents import LOCAL, env  # noqa: F401  (env is a fixture)

NOW = datetime(2026, 10, 15, 9, 0)  # a Thursday
THIS_WEEK, NEXT_WEEK = "2026-10-12", "2026-10-19"
MONDAY, TUESDAY, SATURDAY = "2026-10-19", "2026-10-20", "2026-10-24"


@pytest.fixture(autouse=True)
def office(monkeypatch, tmp_path):
    monkeypatch.setenv("BONCLAY_MEETINGS_FILE", str(tmp_path / "mcp" / "meetings.json"))
    monkeypatch.setattr(m, "now", lambda: NOW)
    monkeypatch.delenv("GOOGLE_CALENDAR_ID", raising=False)
    return tmp_path / "mcp" / "meetings.json"


def set_switches(**kw):
    return m.update_settings(**kw)


def week(client, start=NEXT_WEEK):
    r = client.get(f"/api/meetings/week?start={start}", headers=LOCAL)
    assert r.status_code == 200, r.text
    return r.json()


def ask(**kw):
    args = {"name": "Riya Patil", "role": "student", "reason": "project guide approval", "date": MONDAY, "time": "10:30",
            "phone": "9876543210", "class_or_ref": "TY AI-A"}
    args.update(kw)
    return m.request_meeting(**args)


def put_slot(client, date, time, state):
    return client.put("/api/meetings/slots", headers=LOCAL, json={"date": date, "time": time, "state": state})


def err(fn, *a, **kw):
    with pytest.raises(m.MeetingError) as e:
        fn(*a, **kw)
    return str(e.value)


# ------------------------------------------------------------------------------------------------ the grid

def test_the_week_has_the_days_the_times_and_a_cell_for_each(app_client):
    set_switches(sample_schedule=False)
    body = week(app_client)
    assert body["start"] == NEXT_WEEK and body["slot_minutes"] == 20
    assert [d["weekday"] for d in body["days"]] == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert [d["open"] for d in body["days"]] == [True] * 5 + [False] * 2
    assert body["times"][:3] == ["10:30", "10:50", "11:10"] and body["times"][-1] == "16:00" and "13:00" not in body["times"]
    assert body["settings"] == {"availability_check": True, "sample_schedule": False}
    assert body["calendar"]["configured"] is False
    for day in body["days"]:
        assert set(body["cells"][day["date"]]) == set(body["times"])


def test_cells_are_closed_past_busy_asked_for_booked_or_free(app_client):
    set_switches(sample_schedule=False)
    ask()                                                                   # Monday 10:30, pending
    ask(name="Sanjay Rao", role="faculty", phone="9000000002", time="14:00")
    m.approve("M11")                                                        # Monday 14:00, approved
    cells = week(app_client)["cells"]
    mon, sat = cells[MONDAY], cells[SATURDAY]
    assert mon["10:30"]["state"] == "asked_for"
    assert mon["10:30"]["request"] == {"request_id": "M10", "name": "Riya Patil", "role": "student", "on_calendar": False}
    assert mon["14:00"]["state"] == "booked" and mon["14:00"]["request"]["request_id"] == "M11"
    assert mon["11:30"] == {"state": "busy", "reason": "Department meeting", "source": "blocked"}
    assert mon["11:50"]["source"] == "blocked"
    assert mon["10:50"] == {"state": "free"}
    assert mon["12:30"]["state"] == "free" and mon["14:20"]["state"] == "free"
    assert all(c["state"] == "closed" for c in sat.values())
    # the week that holds today: earlier days are past, today's slots from 9:00 on are not
    this = week(app_client, THIS_WEEK)["cells"]
    assert this["2026-10-12"]["10:30"]["state"] == "past" and this["2026-10-14"]["16:00"]["state"] == "past"
    assert this["2026-10-15"]["10:30"]["state"] == "free"


def test_a_slot_too_close_for_the_agent_is_free_but_marked_too_soon(app_client, monkeypatch):
    set_switches(sample_schedule=False)
    assert week(app_client, THIS_WEEK)["cells"]["2026-10-15"]["10:30"] == {"state": "free"}    # 90 minutes ahead: offered
    monkeypatch.setattr(m, "now", lambda: datetime(2026, 10, 15, 10, 0))
    assert week(app_client, THIS_WEEK)["cells"]["2026-10-15"]["10:30"] == {"state": "free", "too_soon": True}
    assert "10:30" not in m.check_availability("2026-10-15")["free"]


def test_a_start_in_the_middle_of_the_week_gives_that_weeks_monday(app_client):
    assert week(app_client, "2026-10-22")["start"] == NEXT_WEEK
    default = app_client.get("/api/meetings/week", headers=LOCAL).json()
    assert default["start"] == THIS_WEEK and default["today"] == "2026-10-15"
    for bad in ("tomorrow", "2026-13-40", "1066-10-14"):
        r = app_client.get(f"/api/meetings/week?start={bad}", headers=LOCAL)
        assert r.status_code == 422 and r.json()["code"] == "BAD_DATE", bad


def test_a_full_day_is_flagged_but_its_cells_stay_free(app_client):
    set_switches(sample_schedule=False)
    with m.transaction() as data:
        data["settings"]["max_per_day"] = 2
    ask()
    ask(name="Sanjay Rao", role="faculty", phone="9000000002", time="14:00")
    body = week(app_client)
    day = body["days"][0]
    assert day["full"] is True and body["days"][1]["full"] is False
    assert body["cells"][MONDAY]["10:50"] == {"state": "free"}
    assert m.check_availability(MONDAY)["free"] == []                      # the agent still sees the day as full


def test_two_requests_for_one_cell_show_the_approved_one_first(app_client):
    set_switches(availability_check=False, sample_schedule=False)
    ask()
    ask(name="Sanjay Rao", role="faculty", phone="9000000002")             # the same time, taken without a check
    m.approve("M11")
    cell = week(app_client)["cells"][MONDAY]["10:30"]
    assert cell["state"] == "booked" and cell["request"]["request_id"] == "M11" and cell["more"] == 1


# ------------------------------------------------------------------------------------------------ the sample week

def test_the_sample_week_is_the_same_every_time_and_differs_between_weeks(app_client):
    first = week(app_client)["cells"]
    assert week(app_client)["cells"] == first
    assert week(app_client, "2026-10-26")["cells"] != {**first}            # another ISO week, another pattern
    sample = [c for day in first.values() for c in day.values() if c.get("source") == "sample"]
    assert sample and {c["reason"] for c in sample} <= {"Lecture", "Department meeting", "Exam duty", "Visitors"}
    assert all(c["state"] == "busy" for c in sample)


def test_about_a_third_of_the_slots_are_busy_over_several_weeks(app_client):
    busy = total = 0
    for start in ("2026-10-19", "2026-10-26", "2026-11-02", "2026-11-09", "2026-11-16", "2026-11-23"):
        for day, cells in week(app_client, start)["cells"].items():
            for c in cells.values():
                if c["state"] != "closed":
                    total += 1
                    busy += c["state"] == "busy"
    assert 0.25 <= busy / total <= 0.5, busy / total


def test_the_sample_week_is_absent_when_it_is_off(app_client):
    set_switches(sample_schedule=False)
    cells = week(app_client)["cells"]
    assert not [c for day in cells.values() for c in day.values() if c.get("source") == "sample"]
    assert all(c["state"] in ("free", "closed") or c.get("source") == "blocked" for day in cells.values() for c in day.values())


def test_sample_busy_slots_are_not_offered_to_the_agent_and_the_grid_agrees(app_client):
    cells = week(app_client)["cells"]
    for day in (MONDAY, TUESDAY, "2026-10-21", "2026-10-22", "2026-10-23"):
        grid_free = sorted(t for t, c in cells[day].items() if c["state"] == "free")
        assert grid_free == m.check_availability(day)["free"], day


def test_sample_slots_do_not_stop_a_request_when_the_check_is_off(app_client):
    cells = week(app_client)["cells"]
    taken = next(t for t, c in cells[MONDAY].items() if c.get("source") == "sample")
    assert "not free" in err(ask, time=taken)                              # on: the made-up busy slot is refused
    set_switches(availability_check=False)
    assert ask(time=taken)["status"] == "pending"


# ------------------------------------------------------------------------------------------------ the owner's marks

def test_a_busy_mark_makes_a_free_slot_busy_for_the_agent_too(app_client):
    set_switches(sample_schedule=False)
    r = put_slot(app_client, MONDAY, "10:50", "busy")
    assert r.status_code == 200 and r.json() == {"date": MONDAY, "time": "10:50", "state": "busy"}
    assert week(app_client)["cells"][MONDAY]["10:50"] == {"state": "busy", "reason": "Marked busy by you", "source": "owner"}
    assert "10:50" not in m.check_availability(MONDAY)["free"]
    assert "not free" in err(ask, time="10:50")
    assert put_slot(app_client, MONDAY, "10:50", "clear").status_code == 200
    assert week(app_client)["cells"][MONDAY]["10:50"] == {"state": "free"}
    assert "10:50" in m.check_availability(MONDAY)["free"]


def test_a_free_mark_clears_a_sample_busy_slot(app_client):
    cells = week(app_client)["cells"][MONDAY]
    slot = next(t for t, c in cells.items() if c.get("source") == "sample")
    assert slot not in m.check_availability(MONDAY)["free"]
    assert put_slot(app_client, MONDAY, slot, "free").status_code == 200
    assert week(app_client)["cells"][MONDAY][slot] == {"state": "free"}
    assert slot in m.check_availability(MONDAY)["free"]
    put_slot(app_client, MONDAY, slot, "clear")                            # her mark removed: the sample is back
    assert week(app_client)["cells"][MONDAY][slot]["source"] == "sample"


def test_marks_are_kept_in_the_data_file_and_old_days_are_dropped(office, app_client):
    put_slot(app_client, MONDAY, "10:50", "busy")
    assert json.loads(office.read_text())["settings"]["overrides"] == {MONDAY: {"10:50": "busy"}}
    with m.transaction() as data:
        data["settings"]["overrides"]["2026-09-01"] = {"10:30": "busy"}
        data["settings"]["overrides"]["junk"] = "not a dict"
    put_slot(app_client, TUESDAY, "10:50", "busy")
    assert set(json.loads(office.read_text())["settings"]["overrides"]) == {MONDAY, TUESDAY}


@pytest.mark.parametrize("date,time,state,fragment", [
    ("2026-10-14", "10:30", "busy", "outside the days"),           # a day gone by
    ("2026-10-15", "08:30", "busy", "outside office hours"),       # before the office opens today
    ("2026-10-24", "10:30", "busy", "outside office hours"),       # Saturday
    (MONDAY, "13:00", "busy", "outside office hours"),             # between the windows
    (MONDAY, "10:40", "busy", "outside office hours"),             # not on the slot grid
    (MONDAY, "11:30", "busy", "blocked in the office rules (Department meeting)"),
    (MONDAY, "11:30", "free", "blocked in the office rules"),
    ("2027-03-01", "10:30", "busy", "outside the days"),           # beyond the days that can be booked
])
def test_marks_that_cannot_be_made_are_refused_with_a_plain_message(app_client, date, time, state, fragment):
    r = put_slot(app_client, date, time, state)
    assert r.status_code == 409 and r.json()["code"] == "MEETING_REFUSED" and fragment in r.json()["error"], r.text


def test_a_slot_already_started_or_holding_a_request_cannot_be_marked(app_client, monkeypatch):
    set_switches(sample_schedule=False)
    monkeypatch.setattr(m, "now", lambda: datetime(2026, 10, 15, 10, 45))
    r = put_slot(app_client, "2026-10-15", "10:30", "busy")
    assert r.status_code == 409 and "already passed" in r.json()["error"]
    monkeypatch.setattr(m, "now", lambda: NOW)
    ask()
    ask(name="Sanjay Rao", role="faculty", phone="9000000002", time="14:00")
    m.approve("M11")
    for at in ("10:30", "14:00"):
        r = put_slot(app_client, MONDAY, at, "busy")
        assert r.status_code == 409 and "holds this time" in r.json()["error"]
    assert put_slot(app_client, MONDAY, "10:30", "clear").status_code == 409      # even "clear" leaves the request alone
    assert put_slot(app_client, MONDAY, "10:50", "busy").status_code == 200


def test_a_slot_mark_with_an_unknown_state_or_date_is_refused(app_client):
    assert put_slot(app_client, MONDAY, "10:50", "maybe").status_code == 422
    assert put_slot(app_client, "next week", "10:50", "busy").status_code == 409
    assert put_slot(app_client, MONDAY, "soon", "busy").status_code == 409
    assert app_client.put("/api/meetings/slots", headers=LOCAL, json={"date": MONDAY}).status_code == 422


# ------------------------------------------------------------------------------------------------ Google Calendar

def test_calendar_busy_times_show_in_the_grid_once_per_day(app_client, monkeypatch):
    set_switches(sample_schedule=False)
    calls = []

    def busy(a, b):
        calls.append(a.date().isoformat())
        return [(datetime(2026, 10, 20, 14, 0), datetime(2026, 10, 20, 14, 40))] if a.date().isoformat() == TUESDAY else []
    monkeypatch.setattr(gcal, "configured", lambda: True)
    monkeypatch.setattr(gcal, "busy", busy)
    cells = week(app_client)["cells"]
    assert cells[TUESDAY]["14:00"] == {"state": "busy", "reason": "Busy on Google Calendar", "source": "calendar"}
    assert cells[TUESDAY]["14:20"]["source"] == "calendar" and cells[TUESDAY]["14:40"] == {"state": "free"}
    assert sorted(calls) == ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"]   # open days only, once each


def test_an_unreadable_calendar_flags_the_day_instead_of_looking_empty(app_client, monkeypatch):
    set_switches(sample_schedule=False)

    def busy(a, b):
        if a.date().isoformat() == TUESDAY:
            raise gcal.CalendarError("down")
        return []
    monkeypatch.setattr(gcal, "configured", lambda: True)
    monkeypatch.setattr(gcal, "busy", busy)
    days = {d["date"]: d for d in week(app_client)["days"]}
    assert days[TUESDAY]["calendar_error"] is True and "calendar_error" not in days[MONDAY]


# ------------------------------------------------------------------------------------------------ the two switches

def test_the_switches_default_on_and_are_changed_and_kept_by_put(app_client, office):
    got = app_client.get("/api/meetings/settings", headers=LOCAL).json()["settings"]
    assert got["availability_check"] is True and got["sample_schedule"] is True and got["overrides"] == {}
    r = app_client.put("/api/meetings/settings", headers=LOCAL, json={"availability_check": False})
    assert r.status_code == 200 and r.json() == {"availability_check": False, "sample_schedule": True}
    r = app_client.put("/api/meetings/settings", headers=LOCAL, json={"sample_schedule": False})
    assert r.json() == {"availability_check": False, "sample_schedule": False}
    saved = json.loads(office.read_text())["settings"]
    assert saved["availability_check"] is False and saved["sample_schedule"] is False
    assert week(app_client)["settings"] == {"availability_check": False, "sample_schedule": False}


@pytest.mark.parametrize("body", [{}, {"availability_check": "yes"}, {"sample_schedule": 1}, {"availability_check": None},
                                  {"availability_check": [True]}])
def test_the_switches_only_take_true_or_false(app_client, body):
    assert app_client.put("/api/meetings/settings", headers=LOCAL, json=body).status_code == 422


def test_a_data_file_from_before_the_switches_still_loads(office, app_client):
    office.parent.mkdir(parents=True)
    old = {"settings": {"owner": "Prof. Old", "slot_minutes": 30, "hours": {"0": [["10:00", "12:00"]]}, "blocked": []},
           "requests": [], "next_id": 10}
    office.write_text(json.dumps(old))
    body = week(app_client)
    assert body["settings"] == {"availability_check": True, "sample_schedule": True} and body["slot_minutes"] == 30
    assert body["times"] == ["10:00", "10:30", "11:00", "11:30"] and body["owner"] == "Prof. Old"
    assert m.check_availability(MONDAY)["free"] is not None


def test_a_week_with_no_office_hours_has_no_times(app_client):
    with m.transaction() as data:
        data["settings"]["hours"] = {}
    body = week(app_client)
    assert body["times"] == [] and not any(d["open"] for d in body["days"])


# ------------------------------------------------------------------------------------------------ the agent, check off

SAYS_FREE = ("available", "is busy", "not free", "fully booked", "nearest options", "is booked")   # any claim about the slot


def test_with_the_check_off_the_agent_is_never_told_a_time_is_free_or_busy():
    set_switches(availability_check=False, sample_schedule=True)
    ask()                                                                  # a held slot, a blocked slot and sample slots
    for at in ("10:30", "11:30", "14:00", None):
        out = m.check_availability(MONDAY, at)
        assert out["schedule_checked"] is False
        assert "available" not in out and "free" not in out and "alternatives" not in out
        text = json.dumps(out).lower()
        assert not [w for w in SAYS_FREE if w in text], text
        assert "i can't say whether that time is free" in text                 # the only mention is that it is not known
        assert "can't check the schedule" in out["say"] and "office will confirm" in out["say"]
        assert out["office_hours"] == "10:30 to 13:00 and 14:00 to 16:30" and "request_meeting" in out["next"]
    assert m.check_availability(MONDAY, "2 pm")["time"] == "14:00"


def test_with_the_check_off_a_closed_day_is_still_said_to_be_closed():
    set_switches(availability_check=False)
    out = m.check_availability(SATURDAY)
    assert out["schedule_checked"] is False and out["office_open"] is False
    assert "does not see visitors on Saturdays" in out["reason"] and "free" not in out
    assert "in the past" in err(m.check_availability, "2026-10-14")


def test_with_the_check_off_an_otherwise_busy_time_goes_in_as_pending_to_be_confirmed(app_client):
    set_switches(availability_check=False)
    ask(time="14:00")
    out = ask(name="Sanjay Rao", role="faculty", phone="9000000002", time="14:00")   # held by M10, a second request is fine
    assert out["status"] == "pending" and out["schedule_checked"] is False and out["request_id"] == "M11"
    assert "office will confirm" in out["message"] and "nothing is confirmed" in out["message"]
    assert "do not say the time is available or booked" in out["message"]
    blocked = ask(name="Asha Nair", role="parent", phone="9000000003", time="11:30")   # inside the Department meeting
    assert blocked["status"] == "pending"
    listed = {r["request_id"]: r for r in m.list_requests("pending")}
    assert set(listed) == {"M10", "M11", "M12"}


def test_with_the_check_off_closed_days_hours_and_the_past_are_still_refused(monkeypatch):
    set_switches(availability_check=False)
    assert "does not see visitors on Saturdays" in err(ask, date=SATURDAY)
    assert "outside office hours" in err(ask, time="13:00") and "10:30 to 13:00 and 14:00 to 16:30" in err(ask, time="09:00")
    assert "outside office hours" in err(ask, time="16:20")                # a whole slot must fit inside the window
    monkeypatch.setattr(m, "now", lambda: datetime(2026, 10, 15, 11, 0))
    assert "has already passed" in err(ask, date="2026-10-15", time="10:30")
    assert "in the past" in err(ask, date="2026-10-14")
    assert ask(date="2026-10-15", time="14:00")["status"] == "pending"


def test_with_the_check_off_one_person_still_cannot_flood_the_office_and_repeats_are_merged():
    set_switches(availability_check=False)
    first = ask()
    again = ask()
    assert again["already_requested"] is True and again["request_id"] == first["request_id"]
    ask(time="10:50")
    assert "already 2 meeting requests" in err(ask, time="11:10")


def test_with_the_check_on_nothing_changes_for_the_agent():
    set_switches(availability_check=True, sample_schedule=False)
    ok = m.check_availability(MONDAY, "2 pm")
    assert ok["available"] is True and ok["say"] == "Monday 19 October at 14:00 is free." and "schedule_checked" not in ok
    first = ask(time="14:00")
    assert "not confirmed yet" in first["message"] and "schedule_checked" not in first
    assert "That time is not free." in err(ask, name="Sanjay Rao", role="faculty", phone="9000000002", time="14:00")
    assert m.check_availability(MONDAY, "14:00")["available"] is False


def test_the_tool_descriptions_tell_the_model_what_to_do_when_the_check_is_off():
    descriptions = {t.name: t.description for t in m.mcp._tool_manager.list_tools()} if hasattr(m.mcp, "_tool_manager") \
        else {"check_availability": m.check_availability.__doc__, "request_meeting": m.request_meeting.__doc__}
    check = " ".join(descriptions["check_availability"].split())
    assert "schedule is not checked" in check and "never call a time free or busy" in check
    assert "confirmation" in descriptions["request_meeting"]


def test_approving_at_another_time_respects_the_owners_own_marks_whichever_way_the_agent_is_set(app_client):
    set_switches(availability_check=False, sample_schedule=False)
    ask()
    put_slot(app_client, TUESDAY, "15:00", "busy")
    r = app_client.post("/api/meetings/M10/approve", headers=LOCAL, json={"date": TUESDAY, "time": "15:00"})
    assert r.status_code == 409 and "not free" in r.json()["error"]
    assert app_client.post("/api/meetings/M10/approve", headers=LOCAL, json={"date": TUESDAY, "time": "15:20"}).status_code == 200
    assert week(app_client)["cells"][TUESDAY]["15:20"]["state"] == "booked"
