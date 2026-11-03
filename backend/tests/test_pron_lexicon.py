import io

import numpy as np
import pytest
import soundfile as sf

from services import pron_lexicon as pl


def test_off_by_default_and_never_changes_text(monkeypatch):
    monkeypatch.delenv(pl.ENV_FLAG, raising=False)
    monkeypatch.setitem(pl.RULES, ("mr", "dhvaani"), [pl.rule("t", "ळ", "X")])
    assert pl.apply("वेळ", "mr", "dhvaani") == "वेळ"


def test_on_applies_only_to_the_matching_language_and_engine(monkeypatch):
    monkeypatch.setenv(pl.ENV_FLAG, "on")
    monkeypatch.setitem(pl.RULES, ("mr", "dhvaani"), [pl.rule("t", "ळ", "X")])
    assert pl.apply("वेळ", "mr", "dhvaani") == "वेX"
    assert pl.apply("वेळ", "mr-IN", "DhVaani") == "वेX"
    assert pl.apply("वेळ", "hi", "dhvaani") == "वेळ"
    assert pl.apply("वेळ", "mr", "kokoro") == "वेळ"
    assert pl.apply("", "mr", "dhvaani") == ""


def test_hits_report_without_the_flag(monkeypatch):
    monkeypatch.delenv(pl.ENV_FLAG, raising=False)
    monkeypatch.setitem(pl.RULES, ("mr", "dhvaani"), [pl.rule("la", "ळ", "X"), pl.rule("rha", "ऱ्ह", "Y")])
    assert pl.hits("वेळ", "mr", "dhvaani") == ["la"]
    assert pl.has_rules("mr", "dhvaani") and not pl.has_rules("ta", "dhvaani")


def test_shipped_rules_never_touch_unmarked_text(monkeypatch):
    monkeypatch.setenv(pl.ENV_FLAG, "on")
    for key, rules in pl.RULES.items():
        for text in ("नमस्कार, तुमचे स्वागत आहे.", "hello world", "123"):
            assert pl.apply(text, *key) == text or any(r.pattern.search(text) for r in rules)


# ------------------------------------------------------------------------------------------ Hindi, Kokoro (hf_alpha)
HI_KOKORO = ("hi", "kokoro")
# (rule name, word as written, what Kokoro is handed, a longer word that contains the written word and must stay)
HINDI_CASES = [
    ("mumbai", "मुंबई", "मुम्बई", "मुंबईकर"),
    ("chhah", "छह", "छे", "छहों"),
    ("shaant", "शांत", "शान्त", "शांति"),
]


def test_hindi_kokoro_rules_are_the_measured_ones():
    assert [r.name for r in pl.RULES[HI_KOKORO]] == [c[0] for c in HINDI_CASES]
    assert all(r.note for r in pl.RULES[HI_KOKORO])


@pytest.mark.parametrize("name,word,spoken,longer", HINDI_CASES)
def test_each_hindi_rule_rewrites_its_word_and_nothing_else(monkeypatch, name, word, spoken, longer):
    monkeypatch.setenv(pl.ENV_FLAG, "on")
    the_rule = next(r for r in pl.RULES[HI_KOKORO] if r.name == name)
    sentence = f"वह कल {word} जा रहा है।"
    assert pl.apply(sentence, "hi", "kokoro") == pl.apply_rules(sentence, [the_rule]) == f"वह कल {spoken} जा रहा है।"
    assert pl.hits(sentence, "hi", "kokoro") == [name]
    # a sentence without the word is returned as it came
    plain = "आज मौसम अच्छा है, कृपया समय पर आइए।"
    assert pl.apply(plain, "hi", "kokoro") == plain and pl.hits(plain, "hi", "kokoro") == []
    # the rule is for the whole word: the same letters inside a longer word stay as written
    inside = f"यह {longer} है।"
    assert pl.apply(inside, "hi", "kokoro") == inside and pl.hits(inside, "hi", "kokoro") == []
    # a danda right after the word still ends the word
    assert pl.apply(f"{word}।", "hi", "kokoro") == f"{spoken}।"


def test_hindi_rules_reach_only_kokoro_in_hindi(monkeypatch):
    monkeypatch.setenv(pl.ENV_FLAG, "on")
    assert pl.apply("शांत", "hi-IN", "Kokoro") == "शान्त"
    assert pl.apply("शांत", "mr", "kokoro") == "शांत"       # Marathi through Kokoro's Hindi voice is not covered
    assert pl.apply("शांत", "hi", "dhvaani") == "शांत"      # engines with their own text front end are left alone
    assert pl.apply("शांत", "hi", "indic_speak") == "शांत"
    assert pl.apply("शांत", "en", "kokoro") == "शांत"


@pytest.mark.parametrize("value", [None, "", "off", "0"])
def test_hindi_apply_is_a_noop_with_the_flag_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(pl.ENV_FLAG, raising=False)
    else:
        monkeypatch.setenv(pl.ENV_FLAG, value)
    for _name, word, _spoken, _longer in HINDI_CASES:
        assert pl.apply(f"वह {word} है", "hi", "kokoro") == f"वह {word} है"


def _router_harness(monkeypatch):
    """/api/kokoro/generate with synthesis and provenance recorded: the text handed to the engine, the text in the receipt."""
    from routers import kokoro as router
    seen = {"engine": [], "receipt": []}
    buf = io.BytesIO()
    sf.write(buf, np.zeros(2400, dtype=np.float32), 24000, format="WAV")
    wav = buf.getvalue()

    async def fake_mlx(voice, text, speed):
        seen["engine"].append(text)
        return wav

    def fake_provenance(**kw):
        seen["receipt"].append(kw["text"])
        return kw["wav_bytes"], {"receipt_id": "r1"}

    monkeypatch.setattr(router, "_synthesize_mlx", fake_mlx)
    monkeypatch.setattr(router.provenance, "apply_provenance", fake_provenance)
    return seen


@pytest.mark.parametrize("flag,expected", [("", "वह कल मुंबई में शांत है।"), ("on", "वह कल मुम्बई में शान्त है।")])
def test_kokoro_route_speaks_the_hint_and_records_what_the_caller_wrote(app_client, monkeypatch, flag, expected):
    monkeypatch.setenv(pl.ENV_FLAG, flag)
    seen = _router_harness(monkeypatch)
    written = "वह कल मुंबई में शांत है।"
    r = app_client.post("/api/kokoro/generate", headers={"storage-mode": "local"}, json={"voice": "hf_alpha", "text": written})
    assert r.status_code == 200, r.text
    assert seen["engine"] == [expected]
    assert seen["receipt"] == [written]  # what is recorded keeps the spelling the caller wrote


@pytest.mark.parametrize("flag,expected", [("", "सब शांत है"), ("on", "सब शान्त है")])
def test_turn_path_hands_kokoro_the_hint(app_client, monkeypatch, flag, expected):
    import asyncio
    import backend
    monkeypatch.setenv(pl.ENV_FLAG, flag)
    got = []
    monkeypatch.setattr(backend, "_kokoro_generate",
                        lambda model, text, lang: got.append((text, lang)) or np.zeros(10, dtype=np.float32))
    plan = backend._TtsPlan("kokoro", "hi", False, model=object())
    asyncio.run(backend._turn_synth(plan, "सब शांत है"))
    assert got == [(expected, "hi")]


def test_kokoro_route_leaves_non_hindi_voices_alone(app_client, monkeypatch):
    monkeypatch.setenv(pl.ENV_FLAG, "on")
    seen = _router_harness(monkeypatch)
    r = app_client.post("/api/kokoro/generate", headers={"storage-mode": "local"},
                        json={"voice": "af_heart", "text": "मुंबई शांत"})
    assert r.status_code == 200, r.text
    assert seen["engine"] == ["मुंबई शांत"]
