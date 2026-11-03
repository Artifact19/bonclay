"""The local appointment MCP server (mcp_servers/appointments.py): input validation, booking, double booking, Sunday,
cancelling, finding a booking, and that its doctors and hours agree with the clinic template's sample document."""
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from mcp_servers import appointments as ap

TEMPLATE_DOC = Path(__file__).resolve().parent.parent / "templates" / "agents" / "clinic_receptionist.md"
NOW = datetime(2026, 10, 15, 9, 0)  # a Thursday
FRIDAY, SATURDAY, SUNDAY, MONDAY = "2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19"


@pytest.fixture(autouse=True)
def appointments_file(monkeypatch, tmp_path):
    monkeypatch.setenv("BONCLAY_APPOINTMENTS_FILE", str(tmp_path / "mcp" / "appointments.json"))
    monkeypatch.setattr(ap, "now", lambda: NOW)
    return tmp_path / "mcp" / "appointments.json"


def book(**kw):
    args = {"name": "Priya Sharma", "phone": "98765 43210", "date": MONDAY, "time": "10:30"}
    args.update(kw)
    return ap.book_appointment(**args)


def err(fn, *a, **kw):
    with pytest.raises(ap.AppointmentError) as e:
        fn(*a, **kw)
    return str(e.value)


def test_free_slots_follow_the_doctors_opd_hours():
    out = ap.list_free_slots(MONDAY, "meera")
    assert out["weekday"] == "Monday" and out["slot_minutes"] == 30
    [row] = out["free"]
    assert row["doctor"] == "Dr. Meera Kulkarni" and row["times"][0] == "09:00" and row["times"][-1] == "13:00"
    assert len(row["times"]) == 9  # 9:00 to 13:30 in half hours
    both = ap.list_free_slots(MONDAY)  # no doctor: everyone who sees patients on Monday
    assert {r["doctor"] for r in both["free"]} == {"Dr. Meera Kulkarni", "Dr. Anjali Rao"}
    assert [r["doctor"] for r in ap.list_free_slots(SATURDAY, "dentist")["free"]] == []  # the dentist is off on Saturday


def test_hours_match_the_template_document():
    text = TEMPLATE_DOC.read_text(encoding="utf-8")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    for doc in ap.DEFAULT_DOCTORS:
        line = next(l for l in text.splitlines() if l.startswith(f"- {doc['name']}"))
        in_line = {d for d in days if d in line}
        assert in_line == {days[int(k)] for k in doc["hours"]}, doc["name"]
        for windows in doc["hours"].values():
            for start, end in windows:
                for t in (start, end):
                    h, m = map(int, t.split(":"))
                    assert re.search(rf"\b{h % 12 or 12}:{m:02d} (AM|PM)", line), (doc["name"], t)


def test_booking_returns_an_id_and_is_stored(appointments_file):
    b = book(reason="fever")
    assert b["booking_id"] == "A10" and b["status"] == "booked"
    assert (b["doctor"], b["weekday"], b["date"], b["time"]) == ("Dr. Meera Kulkarni", "Monday", MONDAY, "10:30")
    assert b["phone"] == "9876543210" and b["reason"] == "fever"
    assert book(time="11:00")["booking_id"] == "A11"
    assert appointments_file.is_file()
    assert "10:30" not in ap.list_free_slots(MONDAY, "meera")["free"][0]["times"]


def test_a_slot_cannot_be_booked_twice():
    book()
    assert "already booked" in err(book, name="Someone Else", phone="9123456780")
    assert book(doctor="Anjali", time="16:00")["doctor"] == "Dr. Anjali Rao"  # another doctor is unaffected


def test_sunday_is_closed():
    assert "Sunday" in err(book, date=SUNDAY)
    assert "Sunday" in err(ap.list_free_slots, SUNDAY)


def test_dates_and_times_are_validated():
    assert "past" in err(book, date="2026-10-14")
    assert "year-month-day" in err(book, date="tomorrow")
    assert "year-month-day" in err(book, date="2026-13-40")
    assert "24-hour" in err(book, time="late morning")
    assert "half hour" in err(book, time="10:10")
    assert "does not see patients" in err(book, time="17:00")  # Dr. Meera's OPD ends at 13:30
    assert "does not see patients" in err(book, date="2026-10-20", time="10:30")  # Tuesday: not her day
    assert "days ahead" in err(book, date=(NOW.date() + timedelta(days=90)).isoformat())
    assert book(time="10:30 AM")["time"] == "10:30"  # 12-hour input is understood too


def test_a_time_that_has_passed_today_is_refused(monkeypatch):
    monkeypatch.setattr(ap, "now", lambda: datetime(2026, 10, 16, 10, 0))  # Friday 10:00
    assert "passed" in err(book, date=FRIDAY, time="09:30")
    assert book(date=FRIDAY, time="10:30")["weekday"] == "Friday"
    times = ap.list_free_slots(FRIDAY, "meera")["free"][0]["times"]
    assert times[0] == "11:00"  # nothing at or before now is offered, and 10:30 was just booked


@pytest.mark.parametrize("phone", ["12345", "98765abcde", "", "1234567890123456"])
def test_phone_must_be_digits(phone):
    assert "phone" in err(book, phone=phone).lower()


def test_phone_formats_are_normalised():
    assert book(phone="+91 98765-43210")["phone"] == "9876543210"
    assert book(phone="(987) 654 3211", time="11:00")["phone"] == "9876543211"


@pytest.mark.parametrize("name", ["", "A", "Priya123", "<script>", "x" * 90])
def test_patient_name_is_validated(name):
    assert "name" in err(book, name=name).lower()


def test_unknown_doctor_is_named_in_the_error():
    msg = err(book, doctor="Dr. Nobody")
    assert "No doctor matches" in msg and "Dr. Meera Kulkarni" in msg
    assert book(doctor="any doctor")["doctor"] == "Dr. Meera Kulkarni"  # no preference: the General Physician


def test_cancel_frees_the_slot_and_cannot_repeat():
    b = book()
    assert ap.cancel_appointment(b["booking_id"].lower())["status"] == "cancelled"
    assert "already cancelled" in err(ap.cancel_appointment, b["booking_id"])
    assert "no booking" in err(ap.cancel_appointment, "A999")
    assert book(name="Next Person", phone="9111111111")["time"] == "10:30"  # the slot is free again


def test_find_booking_by_phone_lists_upcoming_only():
    a = book()
    b = book(time="11:00")
    ap.cancel_appointment(b["booking_id"])
    book(phone="9000000001", time="11:30")
    found = ap.find_booking("98765 43210")["bookings"]
    assert [x["booking_id"] for x in found] == [a["booking_id"]]
    assert ap.find_booking("9999999999") == {"bookings": []}


def test_read_only_calls_do_not_rewrite_the_file(appointments_file):
    ap.list_free_slots(MONDAY)
    first = appointments_file.stat().st_mtime_ns
    ap.list_free_slots(MONDAY)
    ap.find_booking("9876543210")
    assert appointments_file.stat().st_mtime_ns == first


def test_a_corrupt_file_is_set_aside_not_crashed_on(appointments_file):
    appointments_file.parent.mkdir(parents=True)
    appointments_file.write_text("{not json")
    assert book()["booking_id"] == "A10"
    assert appointments_file.with_name(appointments_file.name + ".corrupt").exists()
