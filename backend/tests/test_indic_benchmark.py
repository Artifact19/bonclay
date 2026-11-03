"""Benchmark plumbing that needs no model: the sentence files, the single CER definition, the report."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import indic_benchmark as ib  # noqa: E402
import scoring  # noqa: E402


def test_cer_basic_values():
    assert scoring.cer("abcd", "abcd") == 0.0
    assert scoring.cer("abcd", "abed") == pytest.approx(0.25)
    assert scoring.cer("abcd", "") == 1.0
    assert scoring.cer("", "abc") is None


def test_cer_matches_jiwer():
    jiwer = pytest.importorskip("jiwer")
    for ref, hyp in [("वेळेवर कळवू", "वेलेवर कलवू"), ("நாளை காலை", "நாளை காளை"), ("hello", "help")]:
        r, h = scoring.letters(ref), scoring.letters(hyp)
        assert scoring.cer(ref, hyp) == pytest.approx(jiwer.cer(r, h))


def test_letters_drops_punctuation_digits_spaces_zero_width():
    assert scoring.letters("डॉक्टर, कल 10 बजे।") == "डॉक्टरकलबजे"
    assert scoring.letters("क‍ष") == "कष"  # zero-width joiner is not a letter
    assert scoring.letters("நாளை, 5!") == "நாளை"


def test_devanagari_only_matches_the_old_readback_rule():
    mixed = "डॉक्टर clinic ठीक है"
    assert scoring.letters(mixed, script="devanagari") == "डॉक्टरठीकहै"
    assert "clinic" in scoring.letters(mixed)
    # readback.py is not imported: its `from common import ...` clashes with services/common.py when the whole suite runs
    src = (Path(__file__).resolve().parent.parent / "bench" / "readback.py").read_text(encoding="utf-8")
    assert 'return letters(s, script="devanagari")' in src


def test_normalise_hyp_maps_digits_in_order_in_every_script():
    assert scoring.normalise_hyp("10 बजे और 450 रुपये", ["दस", "चार सौ"]) == "दस बजे और चार सौ रुपये"
    assert scoring.normalise_hyp("மணி ௫", ["ஐந்து"]) == "மணி ஐந்து"
    assert scoring.normalise_hyp("3 4", ["a"]) == "a "  # extra runs are dropped


def test_old_trial_helpers_are_the_same_functions():
    import indic_clone_trial as t
    assert t.cer is scoring.cer and t.normalise_hyp is scoring.normalise_hyp and t._letters is scoring.letters


def test_all_languages_load_and_are_valid():
    items = ib.load_sentences()
    assert {i["language"] for i in items} == set(ib.LANGS)
    for lang in ib.LANGS:
        assert len([i for i in items if i["language"] == lang]) >= 6
    for i in items:
        assert i["kind"] and isinstance(i["number_words"], list)
        # a sentence with number words must not be written with digits: the digits are what the ASR returns
        assert not scoring._DIGITS.search(i["text"])


def test_each_sentence_script_matches_its_language():
    ranges = {"hi": (0x900, 0x97F), "mr": (0x900, 0x97F), "ta": (0xB80, 0xBFF), "bn": (0x980, 0x9FF),
              "te": (0xC00, 0xC7F), "kn": (0xC80, 0xCFF), "gu": (0xA80, 0xAFF), "ml": (0xD00, 0xD7F), "pa": (0xA00, 0xA7F)}
    for i in ib.load_sentences():
        if i["language"] == "hinglish":
            assert i["text"].isascii() and i["spoken"]
            continue
        lo, hi = ranges[i["language"]]
        letters = [c for c in scoring.letters(i["text"])]
        assert letters and all(lo <= ord(c) <= hi for c in letters), i["id"]


def test_validation_rejects_bad_files(tmp_path):
    (tmp_path / "hi.json").write_text('[{"language":"hi","id":"hi-01","text":"x"}]', encoding="utf-8")
    with pytest.raises(ValueError):
        ib.load_sentences(["hi"], data_dir=tmp_path)


def test_marathi_leak_scan_is_applied_to_marathi_only():
    mr = next(i for i in ib.load_sentences(["mr"]) if i["kind"] == "la")
    out = ib.score_item(mr, mr["text"].replace("ळ", "ल"))
    assert out["cer"] > 0 and any("ळ lost" in f for f in out["leak_flags"])
    hi = ib.load_sentences(["hi"])[0]
    assert "leak_flags" not in ib.score_item(hi, hi["text"])
    assert ib.score_item(hi, hi["text"])["cer"] == 0.0


def test_report_states_the_confound_and_listening_column(tmp_path):
    csvp = tmp_path / "r.csv"
    csvp.write_text("engine,id,language,wav,text,rating\ndhvaani,mr-01,mr,x,y,ok\ndhvaani,mr-02,mr,x,y,bad\ndhvaani,mr-03,mr,x,y,\n")
    ratings = ib.load_ratings(csvp)
    assert ratings == {("dhvaani", "mr-01"): "ok", ("dhvaani", "mr-02"): "bad"}
    items = [{"engine": "dhvaani", "language": "mr", "id": f"mr-0{n}", "cer": c, "rtf": 1.0, "error": None,
              "leak_flags": []} for n, c in ((1, 0.1), (2, 0.3), (3, 0.2))]
    result = {"timestamp": "t", "git": {"commit": "abc"}, "machine": {"cpu": "M4", "ram_gb": 16}, "quick": False,
              "items": items, "settings": {"engines": ["dhvaani"], "asr_model": "whisper"}}
    md = ib.render_markdown(result, ratings)
    assert "toward Hindi-like spelling" in md and "Listened" in md
    assert "| Marathi | 3 | 0.200 (0.200) | 1 of 2 |" in md
    assert "0 of 0" in ib.render_markdown(result)


def test_summarise_skips_errors():
    s = ib.summarise([{"engine": "e", "language": "hi", "id": "a", "cer": 0.5, "rtf": 2.0, "error": None},
                      {"engine": "e", "language": "hi", "id": "b", "error": "boom"}])
    assert s[("e", "hi")]["n"] == 1
