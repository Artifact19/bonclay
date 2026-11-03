#!/usr/bin/env python3
"""Audio-level regression run with synthetic callers (run before a release; local only, CI has no models).

The 60 adversarial cases of backend/eval/attacks_hinglish.json are spoken by Kokoro, posted as audio to the real turn
endpoint of a RUNNING API (speech recognition, retrieval, language model, language policy, text to speech), and scored:

  * speech-recognition error: character error rate (CER, jiwer) of what the API heard against the text that was spoken
    (Devanagari letters only for Hindi speech; lowercase letters and digits for English speech);
  * the eval-lab verdict on the agent's reply (the same judge prompts, two-pass rule and pass definition as the
    Evaluation Lab, services/eval_lab.py), unless --no-judge. The judge needs a cloud key (GROQ_API_KEY or
    OPENAI_API_KEY; EVAL_JUDGE_PROVIDERS overrides the order, e.g. "mlx" for an offline comparison of a model with itself).

Voice variants, NOT accents. No engine here has accent control and the repo has no accented reference audio, so the
callers differ only in which synthetic Kokoro voice (and, for Hindi, which speaking speed) says the line. Kokoro's
English presets are American or British voices; the one Hindi preset is hf_alpha at three speeds. Results label these
"voice variants". They say nothing about how real regional accents are recognised. Roman-script Hinglish is spoken by
the Hindi voice after the dictionary transliteration to Devanagari that the app's own test-call caller uses, so the CER
of those cases partly reflects that transliteration. Only Kokoro is used (one engine at a time keeps a 16 GB machine
safe); DhVaani would need a reference clip and is not part of this run.

--with-text also sends each case as typed text (no speech recognition) and judges it the same way, so the audio verdict
can be read against the text verdict of the same build. The judge here sees no retrieved context (the turn response
does not return it), identically for both arms, so compare the two arms and runs with each other, not with the Lab's
own score.

    backend/.venv/bin/python backend/bench/synthetic_callers.py --no-judge            # speech recognition only
    backend/.venv/bin/python backend/bench/synthetic_callers.py --with-text           # full run, judge key needed
    backend/.venv/bin/python backend/bench/synthetic_callers.py --quick               # 6 cases, smoke test
    ./bonclay regression [options]                                                    # the same
Results: backend/bench/results/synthetic_regression-*.json, compared with the previous run of the same kind.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCH_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCH_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import jiwer  # noqa: E402

from common import (RESULTS_DIR, Api, ApiError, add_common_args, delete_agent, ensure_agent, err_text,  # noqa: E402
                    new_result, print_header, require_api, spoken_turn, stats, write_result)
from readback import norm_en, norm_hi  # noqa: E402

CASES_PATH = BACKEND_DIR / "eval" / "attacks_hinglish.json"
NAME = "synthetic_regression"
# Labelled "voice variants": different synthetic voices or speeds, never claimed to be regional accents.
EN_VOICES = [("af_heart", "American female"), ("am_adam", "American male"), ("bf_emma", "British female"),
             ("bm_george", "British male")]
HI_VARIANTS = [("hf_alpha", 1.0, "Hindi female, normal speed"), ("hf_alpha", 0.85, "Hindi female, slow"),
               ("hf_alpha", 1.15, "Hindi female, fast")]
DEFAULT_TEMPLATE = "clinic_receptionist"


def has_devanagari(text: str) -> bool:
    return any(0x0900 <= ord(c) <= 0x097F for c in text or "")


# ------------------------------------------------------------------------------------- pure helpers (CI-tested)

def load_cases(path: Path = CASES_PATH) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def select_cases(cases: List[dict], limit: Optional[int]) -> List[dict]:
    """At most `limit` cases spread evenly over the file (so every category and script is hit), in file order."""
    if not limit or limit >= len(cases):
        return list(cases)
    step = len(cases) / limit
    return [cases[int(i * step)] for i in range(limit)]


def plan_voice(case: dict, index: int, to_devanagari=None) -> Dict[str, Any]:
    """The voice variant, speed and exact text to speak for a case. Variants rotate with the case position, so a run is
    reproducible. Hindi speech needs Devanagari: Roman text goes through `to_devanagari` (the app's dictionary)."""
    text = case["text"]
    if case.get("script") == "english" and not has_devanagari(text):
        voice, label = EN_VOICES[index % len(EN_VOICES)]
        return {"engine": "kokoro", "voice": voice, "speed": 1.0, "variant": label, "language": "en", "spoken_text": text}
    voice, speed, label = HI_VARIANTS[index % len(HI_VARIANTS)]
    spoken = text if has_devanagari(text) or to_devanagari is None else to_devanagari(text)
    return {"engine": "kokoro", "voice": voice, "speed": speed, "variant": label, "language": "hi", "spoken_text": spoken}


def cer(reference: str, heard: str) -> Optional[float]:
    """Character error rate of `heard` against the spoken text, 0..1+ (1.0 when nothing comparable was heard).
    Devanagari reference: Devanagari letters and marks only. Otherwise lowercase letters and digits only."""
    if has_devanagari(reference):
        ref, hyp = norm_hi(reference), norm_hi(heard or "")
    else:
        strip = lambda s: "".join(norm_en(s).split())  # noqa: E731
        ref, hyp = strip(reference), strip(heard or "")
    if not ref:
        return None
    return 1.0 if not hyp else float(jiwer.cer(ref, hyp))


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def summarize(items: List[dict], judged: bool) -> Dict[str, Any]:
    ok = [i for i in items if not i.get("error")]

    def cer_block(rows):
        s = stats([r["cer"] for r in rows])
        return {"n": s["n"], "mean": s["mean"], "p50": s["p50"], "p95": s["p95"]}

    def group(key):
        out: Dict[str, Any] = {}
        for k in sorted({str(i[key]) for i in ok}):
            rows = [i for i in ok if str(i[key]) == k]
            out[k] = {"cer": cer_block(rows), **_verdict_block(rows, judged)}
        return out

    return {"cases": len(items), "failed": len(items) - len(ok), "cer": cer_block(ok),
            **_verdict_block(ok, judged), "by_script": group("script"), "by_variant": group("variant"),
            "by_category": {k: v for k, v in group("category").items()}}


def _verdict_block(rows: List[dict], judged: bool) -> Dict[str, Any]:
    if not judged:
        return {}
    j = [r for r in rows if r.get("audio_verdict") and r["audio_verdict"].get("status") == "judged"]
    block: Dict[str, Any] = {"audio_judged": len(j), "audio_pass_rate": _rate([r["audio_verdict"]["passed"] for r in j])}
    t = [r for r in rows if r.get("text_verdict") and r["text_verdict"].get("status") == "judged"]
    if t:
        block.update({"text_judged": len(t), "text_pass_rate": _rate([r["text_verdict"]["passed"] for r in t])})
    return block


def _rate(flags: List[bool]) -> Optional[float]:
    return round(100 * sum(1 for f in flags if f) / len(flags), 1) if flags else None


def previous_run(current: dict, results_dir: Path = RESULTS_DIR) -> Optional[dict]:
    """The newest earlier result of this kind that is not a quick run (None when there is none)."""
    runs = []
    for f in sorted(results_dir.glob(f"{NAME}-*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not r.get("quick") and r.get("timestamp", "") < current.get("timestamp", "9999"):
            runs.append(r)
    return runs[-1] if runs else None


def compare(prev: Optional[dict], cur: dict) -> Dict[str, Any]:
    """Deltas of the headline numbers against the previous run, plus the cases whose verdict or CER got clearly worse."""
    if not prev:
        return {"previous": None}
    ps, cs = prev.get("summary") or {}, cur.get("summary") or {}
    delta = lambda a, b: None if a is None or b is None else round(b - a, 4)  # noqa: E731
    out: Dict[str, Any] = {
        "previous": {"timestamp": prev.get("timestamp"), "commit": (prev.get("git") or {}).get("commit")},
        "cer_mean_delta": delta((ps.get("cer") or {}).get("mean"), (cs.get("cer") or {}).get("mean")),
        "audio_pass_rate_delta": delta(ps.get("audio_pass_rate"), cs.get("audio_pass_rate")),
        "script_cer_mean_delta": {k: delta(((ps.get("by_script") or {}).get(k) or {}).get("cer", {}).get("mean"),
                                           v.get("cer", {}).get("mean")) for k, v in (cs.get("by_script") or {}).items()},
    }
    before = {i["id"]: i for i in prev.get("items", [])}
    regress = []
    for i in cur.get("items", []):
        b = before.get(i["id"])
        if not b or i.get("error") or b.get("error"):
            continue
        was = (b.get("audio_verdict") or {}).get("passed")
        now = (i.get("audio_verdict") or {}).get("passed")
        worse_cer = i["cer"] is not None and b["cer"] is not None and i["cer"] - b["cer"] >= 0.15
        if (was and now is False) or worse_cer:
            regress.append({"id": i["id"], "was_pass": was, "now_pass": now, "cer_before": b["cer"], "cer_now": i["cer"]})
    out["regressed_cases"] = regress
    return out


# ---------------------------------------------------------------------------------------------------- judging

def judge_providers() -> List[str]:
    raw = os.getenv("EVAL_JUDGE_PROVIDERS", "")
    return [p.strip() for p in raw.split(",") if p.strip()] or ["groq", "openai"]


class Judge:
    """The Evaluation Lab's own judge (services/eval_lab.py) over services/llm.py, with no retrieved context."""

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(BACKEND_DIR / ".env")
        from services import eval_lab as el
        from services import llm
        self.el, self.llm = el, llm
        self.providers = judge_providers()
        self.models: Dict[str, int] = {}

    async def _ask(self, prompt: str) -> str:
        res = await asyncio.to_thread(self.llm.chat, [{"role": "user", "content": prompt}], json_mode=True,
                                      temperature=0.0, max_tokens=200, timeout=60, providers=self.providers)
        key = f"{getattr(res, 'provider', '?')}/{getattr(res, 'model', '?')}"
        self.models[key] = self.models.get(key, 0) + 1
        return res.text

    def check(self) -> None:
        asyncio.run(self._ask('Reply with only {"ok": "yes"}'))

    async def verdict(self, case: dict, reply: str) -> Dict[str, Any]:
        if not (reply or "").strip():
            return {"status": "error", "passed": False, "reason": "no reply"}
        try:
            judged = await self.el.judge_case(case, "", reply, self._ask, retry={"retries": 1, "delays": (3.0,)})
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "passed": False, "reason": f"{type(e).__name__}: {str(e)[:120]}"}
        first = judged["first"]
        return {"status": "judged", "passed": bool(judged["pass"]), "disputed": judged["disputed"],
                "followed_expected_behaviour": first["followed_expected_behaviour"],
                "invented_facts": first["invented_facts"], "broke_rules": first["broke_rules"], "reason": first["reason"]}


# ------------------------------------------------------------------------------------------------------ run

def render(api: Api, plan: dict) -> bytes:
    r, _, _ = api.post("/api/kokoro/generate", json={"voice": plan["voice"], "text": plan["spoken_text"],
                                                     "speed": plan["speed"]}, timeout=120)
    if r.status_code != 200:
        raise ApiError(r.status_code, err_text(r))
    return r.content


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--agent", help="agent id to call (default: a fresh clinic_receptionist agent, deleted afterwards)")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--max-cases", type=int, default=None, help="use an evenly spread subset (quick: 6)")
    ap.add_argument("--no-judge", action="store_true", help="score speech recognition only; no judge key needed")
    ap.add_argument("--with-text", action="store_true", help="also send each case as typed text and judge it (control arm)")
    ap.add_argument("--keep-audio", metavar="DIR", help="save the rendered WAVs (synthetic voices) into this folder")
    a = ap.parse_args(argv)

    cases = select_cases(load_cases(), a.max_cases or (6 if a.quick else None))
    judge = None
    if not a.no_judge:
        judge = Judge()
        try:
            judge.check()
        except Exception as e:  # noqa: BLE001
            sys.exit(f"The judge is not available ({type(e).__name__}: {str(e)[:150]}). Set GROQ_API_KEY or OPENAI_API_KEY "
                     "in backend/.env (or EVAL_JUDGE_PROVIDERS), or use --no-judge to score speech recognition only.")
    api = Api(a.api)
    health = require_api(api)
    from services import hinglish
    settings = {"quick": a.quick, "cases": len(cases), "template": a.template, "judge": not a.no_judge,
                "judge_providers": judge.providers if judge else None, "with_text": a.with_text,
                "engine": "kokoro", "en_voices": [v for v, _ in EN_VOICES],
                "hi_variants": [{"voice": v, "speed": s} for v, s, _ in HI_VARIANTS],
                "voice_variants_note": "synthetic voices and speeds only; not regional accents",
                "judge_context": "none (the turn response carries no retrieved context); same for both arms",
                "cer": "jiwer; Devanagari letters only for Hindi speech, lowercase letters and digits for English"}
    res = new_result(NAME, api, health, settings)
    print_header("Synthetic-caller audio regression", res)
    agent_id, created = ensure_agent(api, a.agent, a.template)
    audio_dir = Path(a.keep_audio) if a.keep_audio else None
    if audio_dir:
        audio_dir.mkdir(parents=True, exist_ok=True)
    try:
        for idx, case in enumerate(cases):
            plan = plan_voice(case, idx, hinglish.dictionary_to_devanagari)
            item: Dict[str, Any] = {"id": case["id"], "category": case["category"], "script": case["script"],
                                    "variant": plan["variant"], "voice": plan["voice"], "speed": plan["speed"],
                                    "spoken_text": plan["spoken_text"], "error": None, "cer": None}
            try:
                wav = render(api, plan)
                if audio_dir:
                    (audio_dir / f"{case['id']}.wav").write_bytes(wav)
                r, secs, = spoken_turn(api, agent_id, wav)
                if r.status_code != 200:
                    raise ApiError(r.status_code, err_text(r))
                j = r.json()
                item.update({"heard": j.get("user_text"), "reply": j.get("agent_text"), "cer": cer(plan["spoken_text"], j.get("user_text")),
                             "turn_seconds": round(secs, 2), "fallback": j.get("fallback")})
                if judge:
                    item["audio_verdict"] = asyncio.run(judge.verdict(case, j.get("agent_text") or ""))
                if a.with_text:
                    rt, _, _ = api.post(f"/api/conversations/{agent_id}/turn", data={"text": case["text"]}, timeout=240)
                    if rt.status_code != 200:
                        raise ApiError(rt.status_code, err_text(rt))
                    item["text_reply"] = rt.json().get("agent_text")
                    if judge:
                        item["text_verdict"] = asyncio.run(judge.verdict(case, item["text_reply"] or ""))
            except Exception as e:  # noqa: BLE001 - one bad case must not end the run
                item["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            res["items"].append(item)
            v = (item.get("audio_verdict") or {}).get("passed")
            print(f"[{idx + 1}/{len(cases)}] {case['id']} {plan['variant']:<28} "
                  f"CER {'n/a' if item['cer'] is None else format(item['cer'], '.2f')}"
                  f"{'' if v is None else '  verdict ' + ('PASS' if v else 'FAIL')}"
                  f"{'  ERROR ' + item['error'] if item['error'] else ''}", flush=True)
    finally:
        if created:
            delete_agent(api, agent_id)

    res["summary"] = summarize(res["items"], judged=bool(judge))
    if judge:
        res["summary"]["judge_models_used"] = judge.models
    res["comparison"] = compare(previous_run(res), res)
    path = write_result(res)
    s = res["summary"]
    print(f"\nspeech-recognition CER: mean {s['cer']['mean'] if s['cer']['mean'] is None else round(s['cer']['mean'], 3)}, "
          f"p95 {s['cer']['p95'] if s['cer']['p95'] is None else round(s['cer']['p95'], 3)} over {s['cer']['n']} cases; "
          f"{s['failed']} case(s) failed to run")
    if judge:
        print(f"audio verdict pass rate: {s.get('audio_pass_rate')}%"
              + (f"  text control arm: {s.get('text_pass_rate')}%" if a.with_text else ""))
    c = res["comparison"]
    if c.get("previous"):
        print(f"vs previous run {c['previous']['timestamp']} ({c['previous']['commit']}): CER mean delta {c['cer_mean_delta']}, "
              f"pass-rate delta {c['audio_pass_rate_delta']}, {len(c['regressed_cases'])} regressed case(s)")
    else:
        print("no previous full run to compare with: this one is the baseline")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
