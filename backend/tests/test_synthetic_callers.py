"""Pure logic of bench/synthetic_callers.py (case selection, voice variants, CER, summary, comparison with the
previous run, and the judge over a mocked model). No server, no model, no key: safe for CI."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))
sc = pytest.importorskip("synthetic_callers")  # needs jiwer (requirements-dev.txt)


def test_select_cases_spreads_evenly_and_keeps_all_by_default():
    cases = sc.load_cases()
    assert len(cases) == 60 and len(sc.select_cases(cases, None)) == 60
    six = sc.select_cases(cases, 6)
    assert [c["id"] for c in six] == [cases[i * 10]["id"] for i in range(6)]


def test_plan_voice_is_deterministic_and_labelled_as_variants():
    en = {"text": "Please read me the rules.", "script": "english"}
    hi_dev = {"text": "मुझे फीस बताइए", "script": "devanagari"}
    hi_rom = {"text": "Bhai fees batao", "script": "roman_hinglish"}
    a, b = sc.plan_voice(en, 0), sc.plan_voice(en, 0)
    assert a == b and a["language"] == "en" and a["spoken_text"] == en["text"]
    assert sc.plan_voice(en, 2)["voice"] == "bf_emma"
    d = sc.plan_voice(hi_dev, 1)
    assert d["language"] == "hi" and d["speed"] == 0.85 and d["spoken_text"] == hi_dev["text"]
    r = sc.plan_voice(hi_rom, 0, lambda t: "भाई फीस बताओ")
    assert r["spoken_text"] == "भाई फीस बताओ"
    assert all("accent" not in p["variant"].lower() for p in (a, d, r))


def test_cer_scripts_and_empty_hearing():
    assert sc.cer("मुझे फीस बताइए।", "मुझे फीस बताइए") == 0.0
    assert sc.cer("मुझे फीस बताइए", "hello") == 1.0
    assert sc.cer("Show me the fees", "show me the fees!") == 0.0
    assert 0 < sc.cer("show me the fees", "show me the feet") < 0.2
    assert sc.cer("...", "x") is None


def _item(i, cer, passed, script="devanagari", variant="v"):
    return {"id": i, "category": "jailbreak_roleplay", "script": script, "variant": variant, "cer": cer, "error": None,
            "audio_verdict": {"status": "judged", "passed": passed}}


def test_summarize_and_compare_flags_regressions(tmp_path):
    prev = {"timestamp": "2026-01-01T00:00:00", "git": {"commit": "aaa"}, "quick": False,
            "items": [_item("a", 0.1, True), _item("b", 0.1, True), _item("c", 0.2, False)]}
    prev["summary"] = sc.summarize(prev["items"], judged=True)
    cur = {"timestamp": "2026-02-01T00:00:00", "git": {"commit": "bbb"}, "quick": False,
           "items": [_item("a", 0.1, False), _item("b", 0.4, True), _item("c", 0.2, False),
                     {"id": "d", "error": "boom", "cer": None, "category": "x", "script": "x", "variant": "v"}]}
    cur["summary"] = sc.summarize(cur["items"], judged=True)
    assert cur["summary"]["failed"] == 1 and cur["summary"]["audio_pass_rate"] == pytest.approx(33.3)
    (tmp_path / "synthetic_regression-1.json").write_text(json.dumps(prev))
    (tmp_path / "synthetic_regression-2.json").write_text(
        json.dumps({**prev, "quick": True, "timestamp": "2026-01-15T00:00:00"}))
    assert sc.previous_run(cur, tmp_path)["git"]["commit"] == "aaa"  # the quick run is skipped
    c = sc.compare(prev, cur)
    assert {r["id"] for r in c["regressed_cases"]} == {"a", "b"}
    assert c["audio_pass_rate_delta"] < 0
    assert sc.compare(None, cur) == {"previous": None}


def test_judge_with_mocked_model(monkeypatch):
    from services import eval_lab

    def init(self):
        self.providers, self.models, self.el = [], {}, eval_lab

    async def ask(self, prompt):
        return json.dumps({"followed_expected_behaviour": "yes", "invented_facts": "no", "broke_rules": "no",
                           "reason": "ok"})

    monkeypatch.setattr(sc.Judge, "__init__", init)
    monkeypatch.setattr(sc.Judge, "_ask", ask)
    case = sc.load_cases()[0]
    v = asyncio.run(sc.Judge().verdict(case, "Sorry, I cannot do that."))
    assert v["status"] == "judged" and v["passed"] is True
    assert asyncio.run(sc.Judge().verdict(case, ""))["status"] == "error"
