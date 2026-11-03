"""How long the phone agent waits for the caller after asking something (services/turn_taking.py), whether the caller's
answer looks unfinished, and the digits a caller says (services/spoken_digits.py). Pure functions: no call, no model."""
import numpy as np
import pytest

from services import phone_audio as pa
from services import spoken_digits as sd
from services import turn_taking as tt


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("PHONE_END_SILENCE_MS", "PHONE_DETAIL_END_SILENCE_MS", "PHONE_DETAILS_END_SILENCE_MS",
                 "PHONE_INCOMPLETE_EXTRA_MS", "PHONE_ADAPTIVE_END", "PHONE_CARRY_UNANSWERED"):
        monkeypatch.delenv(name, raising=False)


# ----------------------------------------------------------------------------------------- the end silence

@pytest.mark.parametrize("agent_text, ms, kind", [
    # several details at once
    ("What are the patient details?", 1500, "details"),
    ("Please share the patient details.", 1500, "details"),
    ("Please tell me your name and phone number.", 1500, "details"),
    ("Sure. What is the patient's name? And a phone number?", 1500, "details"),
    ("Kripya apna naam aur phone number bataiye.", 1500, "details"),
    ("मरीज़ की जानकारी बताइए।", 1500, "details"),
    ("कृपया अपना नाम और फोन नंबर बताइए।", 1500, "details"),
    ("Please tell me the date and time you prefer.", 1500, "details"),
    # one detail
    ("Please tell me your phone number.", 1100, "detail"),
    ("What is the patient's name?", 1100, "detail"),
    ("What date works for you?", 1100, "detail"),
    ("What time would you like?", 1100, "detail"),
    ("When would you like to come?", 1100, "detail"),
    ("Aapka naam kya hai?", 1100, "detail"),
    ("Aap kis tareekh ko aana chahenge?", 1100, "detail"),
    ("आप किस तारीख को आना चाहेंगे?", 1100, "detail"),
    ("Could you share your email address?", 1100, "detail"),
    # nothing asked, or not a detail the caller dictates
    ("How can I help you today?", 500, "normal"),
    ("The clinic opens at nine. Is there anything else?", 500, "normal"),
    ("Your appointment is booked for Monday at 5 pm.", 500, "normal"),
    ("Shall I book it?", 500, "normal"),
    ("Which doctor would you like to see?", 500, "normal"),
    ("", 500, "normal"),
    (None, 500, "normal"),
])
def test_the_end_silence_follows_what_the_agent_asked(agent_text, ms, kind):
    w = tt.end_wait(agent_text)
    assert (w.ms, w.kind) == (ms, kind)
    assert w.label == f"{ms}({kind})"


def test_the_waits_are_settings_and_never_below_the_base(monkeypatch):
    monkeypatch.setenv("PHONE_END_SILENCE_MS", "700")
    monkeypatch.setenv("PHONE_DETAIL_END_SILENCE_MS", "900")
    monkeypatch.setenv("PHONE_DETAILS_END_SILENCE_MS", "2000")
    assert tt.end_wait("How can I help?").ms == 700
    assert tt.end_wait("What is your name?").ms == 900
    assert tt.end_wait("What are the details?").ms == 2000
    monkeypatch.setenv("PHONE_DETAIL_END_SILENCE_MS", "300")           # a detail never waits less than a normal turn
    assert tt.end_wait("What is your name?").ms == 700
    monkeypatch.setenv("PHONE_DETAILS_END_SILENCE_MS", "999999")      # and never more than the cap
    assert tt.end_wait("What are the details?").ms == tt.MAX_END_MS


def test_adaptive_end_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("PHONE_ADAPTIVE_END", "off")
    assert tt.end_wait("What are the patient details?").ms == 500
    assert tt.end_wait("What are the patient details?").kind == "normal"


def test_the_detector_takes_a_new_end_silence_for_the_next_utterance():
    d = pa.UtteranceDetector(8000, vad=pa.EnergyVad(8000), config=pa.DetectorConfig(end_silence_ms=400, early_silence_ms=250))
    t = np.arange(8000) / 8000
    tone = (np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype("<i2").tobytes()          # one second
    silence = b"\x00\x00" * 8000
    kinds = [e.kind for e in d.feed(tone + silence[:int(0.7 * 16000)])]
    assert kinds[-1] == "utterance"                                     # 700 ms of silence ended it at 400 ms
    d.set_end_silence(1100)
    assert d.end_silence_ms == 1100
    kinds = [e.kind for e in d.feed(tone + silence[:int(0.7 * 16000)])]
    assert "utterance" not in kinds                                     # ...now it is still waiting
    assert "utterance" in [e.kind for e in d.feed(silence[:int(0.5 * 16000)])]


# -------------------------------------------------------------------------------- an unfinished answer

@pytest.mark.parametrize("text", [
    "Priya Sharma aur", "Priya Sharma, aur.", "naam Priya hai toh", "uske baad", "matlab", "um", "uh...", "Haan wo",
    "Priya Sharma,", "नाम प्रिया है और", "मतलब", "haan wo", "and then", "name is Priya or", "Priya phir",
])
def test_a_trailing_conjunction_filler_or_comma_means_the_caller_has_not_finished(text):
    assert tt.is_incomplete(text) == "trailing"


@pytest.mark.parametrize("text", [
    "Priya Sharma", "Yes, that is right.", "kal shaam paanch baje", "I want to book an appointment", "haan ji", "",
    "Rahul and Priya", "Is it open on Monday?",
])
def test_a_finished_answer_is_not_extended(text):
    assert tt.is_incomplete(text, "How can I help?") is None


def test_part_of_a_phone_number_is_unfinished_only_when_a_number_was_asked_for():
    ask = "Please tell me your phone number."
    assert tt.is_incomplete("98765", ask) == "phone"
    assert tt.is_incomplete("9876543", ask) == "phone"                      # 7 digits
    assert tt.is_incomplete("nine eight seven six five four three", ask) == "phone"
    assert tt.is_incomplete("नौ आठ सात छह पाँच", ask) == "phone"
    assert tt.is_incomplete("9876543210", ask) is None                       # all ten
    assert tt.is_incomplete("98765 43210", ask) is None
    assert tt.is_incomplete("+91 98765 43210", ask) is None
    assert tt.is_incomplete("it is Priya", ask) is None                      # no digits yet: not a number in progress
    assert tt.is_incomplete("98765", "What date works for you?") is None     # a number was not asked for
    assert tt.is_incomplete("98765", None) is None


def test_a_request_for_the_details_counts_a_number_started_from_three_digits():
    ask = "Please tell me the patient details."
    assert tt.is_incomplete("Priya Sharma 9876543", ask) == "phone"
    assert tt.is_incomplete("Priya Sharma 9876543210", ask) is None
    assert tt.is_incomplete("Priya Sharma, 25 October", ask) is None         # a day of the month is no number started
    assert tt.is_incomplete("Priya Sharma, 98", ask) is None


def test_a_day_of_the_month_is_not_a_phone_number_in_progress():
    both = "Please tell me your phone number and the date."
    assert tt.is_incomplete("25 October", both) is None                      # 2 digits, and a date was asked as well
    assert tt.is_incomplete("5 baje", both) is None
    assert tt.is_incomplete("98765 4", both) == "phone"                      # but 6 digits are a number started


def test_the_extra_wait_is_a_setting_and_zero_turns_it_off(monkeypatch):
    assert tt.incomplete_extra_ms() == 1200
    monkeypatch.setenv("PHONE_INCOMPLETE_EXTRA_MS", "800")
    assert tt.incomplete_extra_ms() == 800
    monkeypatch.setenv("PHONE_INCOMPLETE_EXTRA_MS", "0")
    assert tt.incomplete_extra_ms() == 0


# --------------------------------------------------------------------------------------- spoken digits

@pytest.mark.parametrize("said", [
    "98765 43210",
    "9876543210",
    "+91 98765 43210",
    "+919876543210",
    "091 98765 43210".replace("091 ", "0"),
    "nine eight seven six five four three two one zero",
    "nau aath saat chhe paanch char teen do ek shunya",
    "९८७६५ ४३२१०",
    "नौ आठ सात छह पाँच चार तीन दो एक शून्य",
    "98765, 43210.",
    "my number is 98765 43210 please",
    "98765 43210 25 October",
    "9 8 7 6 5 4 3 2 1 0",
])
def test_a_mobile_number_is_found_however_it_is_said(said):
    assert sd.find_mobile(said) == "9876543210"


def test_double_and_triple_stand_for_repeated_digits():
    assert sd.find_mobile("double nine eight double seven six five four three two") == "9987765432"
    assert sd.find_mobile("nine triple eight seven six five four three two") == "9888765432"
    assert sd.find_mobile("डबल नौ आठ डबल सात छह पाँच चार तीन दो") == "9987765432"


@pytest.mark.parametrize("said", [
    "98765 4321",                         # nine digits
    "12345 67890",                        # does not start 6-9
    "I did it two or three times",       # "two" "three" are words, not a number
    "I sat down and did one thing",      # ambiguous words only
    "",
])
def test_things_that_are_not_a_mobile_number(said):
    assert sd.find_mobile(said) is None


def test_the_longest_run_counts_what_has_been_said_so_far():
    assert sd.longest_run_length("98765 43") == 7
    assert sd.longest_run_length("98765. my name is Priya. 43") == 5      # a sentence end ends a run
    assert sd.longest_run_length("hello there") == 0
    assert sd.longest_run_length("I did it") == 0


def test_a_phone_number_is_taken_out_of_the_words_a_time_reader_sees():
    assert sd.strip_digit_runs("my number is 98765 43210 and 5 baje", 4) == "my number is and 5 baje"
    assert sd.strip_digit_runs("98765 43210 25 October 5 baje", 4) == "25 october 5 baje"
