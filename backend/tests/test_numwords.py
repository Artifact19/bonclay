import json
from pathlib import Path

import pytest

from services import numwords as nw

DATA = Path(__file__).resolve().parent.parent / "bench" / "data" / "indic_benchmark"


@pytest.mark.parametrize("n,hi,mr", [
    (0, "शून्य", "शून्य"), (5, "पाँच", "पाच"), (10, "दस", "दहा"), (15, "पंद्रह", "पंधरा"),
    (21, "इक्कीस", "एकवीस"), (99, "निन्यानवे", "नव्व्याण्णव"), (100, "एक सौ", "शंभर"),
    (400, "चार सौ", "चारशे"), (450, "चार सौ पचास", "चारशे पन्नास"), (1000, "एक हज़ार", "एक हजार"),
    (2026, "दो हज़ार छब्बीस", "दोन हजार सव्वीस"),
    (123456, "एक लाख तेईस हज़ार चार सौ छप्पन", "एक लाख तेवीस हजार चारशे छप्पन्न"),
    (10000000, "एक करोड़", "एक कोटी"),
])
def test_whole_numbers(n, hi, mr):
    assert nw.number_to_words(n, "hi") == hi
    assert nw.number_to_words(n, "mr") == mr


def test_benchmark_sentences_round_trip():
    """The digit form of each Hindi and Marathi benchmark sentence with numbers turns back into the written sentence."""
    digit = {"hi": {"hi-02": "आपकी मुलाकात 15 नवंबर को सुबह 10 बजे तय हुई है।", "hi-03": "पहली जांच की फीस 450 रुपये है।"},
             "mr": {"mr-03": "तुमची भेट 15 नोव्हेंबरला सकाळी 10 वाजता ठरली आहे.", "mr-04": "पहिल्या तपासणीची फी 450 रुपये आहे."}}
    for lang, cases in digit.items():
        rows = {r["id"]: r for r in json.loads((DATA / f"{lang}.json").read_text(encoding="utf-8"))}
        for rid, text in cases.items():
            assert nw.normalize(text, lang) == rows[rid]["text"], rid


def test_amounts_percent_decimal_commas():
    assert nw.normalize("फीस ₹450 है", "hi") == "फीस चार सौ पचास रुपये है"
    assert nw.normalize("Rs. 1,23,456", "hi") == "एक लाख तेईस हज़ार चार सौ छप्पन रुपये"
    assert nw.normalize("रु 99 मात्र", "mr") == "नव्व्याण्णव रुपये मात्र"
    assert nw.normalize("50% छूट", "hi") == "पचास प्रतिशत छूट"
    assert nw.normalize("2.5 घंटे", "hi") == "दो दशमलव पाँच घंटे"
    assert nw.normalize("५० टक्के", "mr") == "पन्नास टक्के"  # native digits


def test_times_and_dates():
    assert nw.normalize("शाम 7:30 बजे", "hi") == "शाम सात बजकर तीस मिनट बजे"
    assert nw.normalize("7:00", "hi") == "सात बजे"
    assert nw.normalize("7:00", "mr") == "सात वाजता"
    assert nw.normalize("15/11/2026", "hi") == "पंद्रह नवंबर दो हज़ार छब्बीस"
    assert nw.normalize("15-11-2026", "mr") == "पंधरा नोव्हेंबर दोन हजार सव्वीस"
    assert nw.normalize("32/13/2026", "hi") == "बत्तीस/तेरह/दो हज़ार छब्बीस"  # not a date: read as numbers


def test_phone_and_pin_are_read_digit_by_digit():
    assert nw.normalize("9876543210", "hi") == "नौ आठ सात छह पाँच चार तीन दो एक शून्य"
    assert nw.normalize("+91 98765 43210", "hi") == "नौ आठ सात छह पाँच चार तीन दो एक शून्य"
    assert nw.normalize("पिन 411001", "hi") == "पिन चार एक एक शून्य शून्य एक"


def test_unsupported_language_and_no_digits_are_untouched():
    for lang in ("ta", "te", "bn", "kn", "gu", "ml", "pa", "en", ""):
        assert nw.normalize("450 ரூபாய்", lang) == "450 ரூபாய்"
    assert nw.normalize("कोई अंक नहीं", "hi") == "कोई अंक नहीं"
    assert nw.normalize("", "hi") == ""


def test_apply_only_for_dhvaani_and_off_switch(monkeypatch):
    monkeypatch.delenv("NUMWORDS", raising=False)
    assert nw.apply("10 बजे", "hi", "dhvaani") == "दस बजे"
    assert nw.apply("10 बजे", "hi-IN", "DhVaani") == "दस बजे"
    assert nw.apply("10 बजे", "hi", "kokoro") == "10 बजे"
    monkeypatch.setenv("NUMWORDS", "off")
    assert nw.apply("10 बजे", "hi", "dhvaani") == "10 बजे"


def test_huge_number_is_read_digit_by_digit():
    assert nw.number_to_words(10 ** 11, "hi").split()[0] == "एक"
    assert len(nw.number_to_words(10 ** 11, "hi").split()) == 12
