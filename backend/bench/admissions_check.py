#!/usr/bin/env python3
"""Ask an admissions agent the questions of docs/samples/MIT_ADT_Admissions_QA_SAMPLE.md and score the replies.

Creates (or reuses with --agent) an agent from the document_qa template with the sample document uploaded, sends typed
questions through the real turn endpoint (language policy, retrieval, the language model, the grounding checks; no
phone call, no Exotel) and checks every reply for the facts it must contain (`need`: any one of the alternatives in
each group) and for facts it must not invent (`never`). Unanswerable questions must make the agent decline.

    backend/.venv/bin/python backend/bench/admissions_check.py                 # 24 questions, new agent, deleted at the end
    backend/.venv/bin/python backend/bench/admissions_check.py --keep          # leave the agent in Agent Studio to try
    backend/.venv/bin/python backend/bench/admissions_check.py --agent <id>    # an agent you already made
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from common import Api, delete_agent, err_text, fmt, new_result, require_api, stats, write_result

DOC = Path(__file__).resolve().parents[2] / "docs" / "samples" / "MIT_ADT_Admissions_QA_SAMPLE.md"

# (question, [[alternatives that must appear, any one of them], ...], [facts that must not appear])
CASES = [
    ("What is the application fee for design programmes?", [["1,500", "1500"]], []),
    ("Where is the campus?", [["loni kalbhor"]], []),
    ("What is the eligibility for B.Tech?", [["50%", "50 percent", "50 per cent"], ["physics"]], []),
    ("Which entrance exams are accepted for B.Tech?", [["cet"], ["jee"]], []),
    ("When do applications close?", [["30 june"]], []),
    ("When is the first merit list?", [["10 june"]], []),
    ("What are the hostel fees for a single room?", [["1,25,000", "125000", "1.25 lakh", "125,000"]], []),
    ("What is the annual tuition for B.Tech?", [["2,40,000", "240000", "2.4 lakh", "240,000"]], []),
    ("What is the refund if I withdraw on 20 July?", [["50%", "50 percent", "half"]], []),
    ("Can I pay the fees in instalments?", [["two", "2"], ["1 january", "january"]], []),
    ("Which payment methods do you accept?", [["upi"]], []),
    ("Is there a merit scholarship, and what does the top scorer get?", [["100%", "full"]], []),
    ("What CGPA do I need to keep my merit scholarship?", [["8.0", "8"]], []),
    ("I have a three year diploma. Can I join B.Tech?", [["second year", "lateral"]], []),
    ("How many B.Tech seats are there in Computer Science?", [["240"]], []),
    ("How much reservation is there for EWS?", [["10%", "10 percent"]], []),
    ("What facilities does the campus have?", [["library"]], []),
    ("When do hostel gates close for first year students?", [["10"]], []),
    ("What was the highest placement package?", [["32"]], []),
    ("What is the total first year cost for B.Tech with a double sharing hostel room?", [["2,40,000", "240000", "2.4 lakh", "240,000"], ["95,000", "95000"]], []),
    ("What is the fee for the PhD in Physics?", [["not", "don't", "do not", "no information", "cannot", "can't", "isn't", "unable", "sorry", "couldn't", "find"]], ["lakh"]),
    ("Who is the vice chancellor?", [["not", "don't", "do not", "no information", "cannot", "can't", "isn't", "unable", "sorry", "couldn't", "find"]], []),
    ("Is there a bus from Nashik?", [["not", "don't", "do not", "no information", "cannot", "can't", "isn't", "unable", "sorry", "pune station", "hadapsar"]], []),
    ("बी.टेकसाठी पात्रता काय आहे?", [["५०", "50", "भौतिक", "physics"]], []),
]


def reply_text(body: dict) -> str:
    for k in ("reply_text", "reply", "response", "text", "answer", "agent_text"):
        v = body.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return json.dumps(body, ensure_ascii=False)[:600]


def passes(reply: str, need, never) -> bool:
    low = re.sub(r"[\u202f\u00a0\u2011\u2019]", lambda m: {"\u2011": "-", "\u2019": "'"}.get(m.group(0), " "), reply).lower()
    low = re.sub(r"\s+%", "%", low)
    ok = all(any(a.lower() in low for a in group) for group in need)
    return ok and not any(n.lower() in low for n in never)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--limit", type=int, default=len(CASES))
    args = ap.parse_args()
    api = Api()
    health = require_api(api)
    created = False
    agent = args.agent
    if not agent:
        r, _, _ = api.post("/api/agents/from-template", json={"template_id": "document_qa", "include_sample": False},
                           timeout=300)
        if r.status_code != 200:
            sys.exit(f"could not create the agent: {r.status_code} {err_text(r)}")
        agent, created = r.json()["id"], True
        with open(DOC, "rb") as f:
            r, _, _ = api.post("/api/knowledge/upload", data={"agent_id": agent},
                               files={"file": (DOC.name, f, "text/markdown")}, timeout=600)
        if r.status_code != 200:
            sys.exit(f"upload failed: {r.status_code} {err_text(r)}")
        print(f"agent {agent}: {r.json().get('chunks_created')} passages from {DOC.name}")
    rows, times, convo = [], [], None
    for q, need, never in CASES[: args.limit]:
        t0 = time.perf_counter()
        r, _, _ = api.post(f"/api/conversations/{agent}/turn", data={"text": q, "test_call": "1"}, timeout=240)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            rows.append({"q": q, "ok": False, "ms": dt * 1000, "reply": f"HTTP {r.status_code} {err_text(r)[:120]}"})
            continue
        text = reply_text(r.json())
        ok = passes(text, need, never)
        times.append(dt * 1000)
        rows.append({"q": q, "ok": ok, "ms": dt * 1000, "reply": text})
        print(f"{'PASS' if ok else 'FAIL'}  {dt * 1000:6.0f} ms  {q}\n        -> {text[:170]!r}")
    good = sum(r["ok"] for r in rows)
    print(f"\n{good}/{len(rows)} correct; turn time (text in, reply and audio out) {stats(times)}")
    res = new_result("admissions_check", api, health, {"questions": len(rows)})
    res["rows"], res["correct"] = rows, good
    print("saved", write_result(res))
    if created and not args.keep:
        delete_agent(api, agent)
    elif created:
        print("kept agent", agent)


if __name__ == "__main__":
    main()
