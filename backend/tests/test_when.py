"""services/when.py: dates, times and slots from the caller's words, in code. The clock is Thursday 22 October 2026,
14:00 local time, so "kal" is Friday the 23rd, "parso" Saturday the 24th, "somvaar" Monday the 26th."""
from datetime import date, datetime, time

import pytest

from services import when

NOW = datetime(2026, 10, 22, 14, 0)  # a Thursday


DATE_CASES = [
    # English relative days
    ("today", date(2026, 10, 22)),
    ("tomorrow", date(2026, 10, 23)),
    ("day after tomorrow", date(2026, 10, 24)),
    ("in 3 days", date(2026, 10, 25)),
    ("next week", date(2026, 10, 29)),
    # Hinglish relative days
    ("aaj", date(2026, 10, 22)),
    ("kal", date(2026, 10, 23)),
    ("parso", date(2026, 10, 24)),
    ("parson", date(2026, 10, 24)),
    ("narso", date(2026, 10, 25)),
    ("3 din baad", date(2026, 10, 25)),
    ("agle hafte", date(2026, 10, 29)),
    # Devanagari relative days
    ("आज", date(2026, 10, 22)),
    ("कल", date(2026, 10, 23)),
    ("परसों", date(2026, 10, 24)),
    # Weekdays, English
    ("monday", date(2026, 10, 26)),
    ("next monday", date(2026, 10, 26)),
    ("friday", date(2026, 10, 23)),
    ("next friday", date(2026, 10, 23)),
    ("thursday", date(2026, 10, 29)),          # today is Thursday and no time was named: next week
    ("sunday", date(2026, 10, 25)),
    ("on wed", date(2026, 10, 28)),
    # Weekdays, Hinglish
    ("agle somvaar", date(2026, 10, 26)),
    ("somwar ko", date(2026, 10, 26)),
    ("mangalvaar", date(2026, 10, 27)),
    ("budhwar", date(2026, 10, 28)),
    ("guruvaar", date(2026, 10, 29)),
    ("shukravaar", date(2026, 10, 23)),
    ("shanivaar", date(2026, 10, 24)),
    ("itvaar", date(2026, 10, 25)),
    ("ravivaar ko", date(2026, 10, 25)),
    # Weekdays, Devanagari
    ("अगले सोमवार", date(2026, 10, 26)),
    ("शुक्रवार", date(2026, 10, 23)),
    # Explicit dates
    ("27 october", date(2026, 10, 27)),
    ("27th october", date(2026, 10, 27)),
    ("october 27", date(2026, 10, 27)),
    ("27/10", date(2026, 10, 27)),
    ("27-10-2026", date(2026, 10, 27)),
    ("2026-10-27", date(2026, 10, 27)),
    ("5 january", date(2027, 1, 5)),            # already passed this year: next year
    ("27 अक्टूबर", date(2026, 10, 27)),
]


@pytest.mark.parametrize("phrase, expected", DATE_CASES)
def test_dates(phrase, expected):
    assert when.resolve(phrase, NOW).date == expected, phrase


TIME_CASES = [
    ("5 baje", time(17, 0), True),
    ("paanch baje", time(17, 0), True),
    ("9 baje", time(9, 0), True),
    ("12 baje", time(12, 0), False),
    ("7 baje", time(19, 0), True),
    ("subah 10 baje", time(10, 0), False),
    ("sham ko 6 baje", time(18, 0), False),
    ("raat 9 baje", time(21, 0), False),
    ("dopahar 2 baje", time(14, 0), False),
    ("11 am", time(11, 0), False),
    ("11am", time(11, 0), False),
    ("11:30 pm", time(23, 30), False),
    ("11.30 am", time(11, 30), False),
    ("14:30", time(14, 30), False),
    ("saade paanch baje", time(17, 30), True),
    ("dhai baje", time(14, 30), True),
    ("dedh baje", time(13, 30), True),
    ("sawa das baje", time(10, 15), True),
    ("paune paanch", time(16, 45), True),
    ("शाम 5 बजे", time(17, 0), False),
    ("सुबह ९ बजे", time(9, 0), False),
    ("half past ten", time(10, 30), True),
    ("quarter to five", time(16, 45), True),
    ("ten thirty", time(10, 30), True),
    ("noon", time(12, 0), False),
    ("at 3 in the afternoon", time(15, 0), False),
]


@pytest.mark.parametrize("phrase, expected, assumed", TIME_CASES)
def test_times(phrase, expected, assumed):
    r = when.resolve_time(phrase)
    assert r.time == expected, phrase
    assert r.assumed_meridiem is assumed, phrase


COMBINED_CASES = [
    ("kal subah 10 baje", date(2026, 10, 23), time(10, 0)),
    ("agle somvaar sham 5 baje", date(2026, 10, 26), time(17, 0)),
    ("tomorrow at 11", date(2026, 10, 23), time(11, 0)),
    ("parso 4 baje", date(2026, 10, 24), time(16, 0)),
    ("thursday 5 pm", date(2026, 10, 22), time(17, 0)),      # later today: today
    ("thursday 10 am", date(2026, 10, 29), time(10, 0)),     # already passed today: next week
    ("27 october 11:30", date(2026, 10, 27), time(11, 30)),
]


@pytest.mark.parametrize("phrase, d, t", COMBINED_CASES)
def test_date_and_time_together(phrase, d, t):
    r = when.resolve(phrase, NOW)
    assert (r.date, r.time) == (d, t), phrase


def test_nothing_date_like_gives_none():
    r = when.resolve("I would like to see the doctor", NOW)
    assert r.date is None and r.time is None and not r.matched


def test_next_weekday_within_two_days_is_flagged():
    r = when.resolve("next friday", NOW)
    assert r.date == date(2026, 10, 23)
    assert any("week after" in n for n in r.notes)


def test_kal_is_noted_as_tomorrow():
    assert any("tomorrow" in n for n in when.resolve("kal", NOW).notes)


def test_tool_args_in_the_callers_words_become_iso():
    args, notes = when.normalise_tool_args({"name": "Priya", "phone": "9876543210", "date": "kal", "time": "5 baje"}, NOW)
    assert args["date"] == "2026-10-23" and args["time"] == "17:00"
    assert args["name"] == "Priya" and args["phone"] == "9876543210"
    assert any("am/pm assumed" in n for n in notes)


def test_tool_args_already_iso_are_untouched():
    original = {"date": "2026-10-27", "time": "11:00"}
    args, notes = when.normalise_tool_args(original, NOW)
    assert args == original and notes == []


def test_tool_args_date_phrase_may_carry_the_time():
    args, _ = when.normalise_tool_args({"date": "agle somvaar 11 baje", "time": None}, NOW)
    assert args["date"] == "2026-10-26"


def test_tool_args_not_understood_are_left_for_the_tool_to_reject():
    args, notes = when.normalise_tool_args({"date": "sometime soon", "time": "whenever"}, NOW)
    assert args["date"] == "sometime soon" and args["time"] == "whenever"
    assert any("not understood" in n for n in notes)


def test_at_least_fifty_cases_are_covered():
    assert len(DATE_CASES) + len(TIME_CASES) + len(COMBINED_CASES) >= 50
