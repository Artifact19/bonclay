"""Instant-answer threshold study: BGE-M3 cosine between an owner-approved question and what callers actually say, for
paraphrases that SHOULD get the approved answer and near misses that must NOT (another day, number, negation, an
opposite such as open/close, another course or topic, a booking request). Picks INSTANT_ANSWER_THRESHOLD
(services/instant_answers.py): the value with zero false matches and the most true matches, with the guards `match`
applies after the cosine (days/numbers/negations must agree, no opposites such as open/closed, one short question, not
sensitive, not too long).

Offline, the same shared model and precision as the API (services/retrieval.encode_texts: the float16 copy in
backend/data/models on Apple Silicon). About 1-2 GB of RAM while it runs. Pairs: bench/data/instant_answer_pairs.json.

    backend/.venv/bin/python -I backend/bench/instant_answer_threshold.py
    backend/.venv/bin/python -I backend/bench/instant_answer_threshold.py --tools    # an agent with tools (booking guard on)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCH_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from services import instant_answers as ia  # noqa: E402
from services import retrieval  # noqa: E402

DATA = BENCH_DIR / "data" / "instant_answer_pairs.json"


def guard_reason(approved: str, asked: str, has_tools: bool):
    """Why `match` would refuse this pair whatever the cosine (None when only the cosine decides)."""
    problem = ia.question_problem(asked, has_tools=has_tools)
    if problem:
        return problem
    if ia.guard_tokens(asked) != ia.guard_tokens(approved):
        return "days/numbers/negation differ"
    if ia.opposite_conflict(asked, approved):
        return "opposites (open/closed...)"
    if ia.word_count(asked) > ia.word_count(approved) * 1.8 + 3:
        return "much longer"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", action="store_true", help="the agent has tools (booking requests are refused by a guard)")
    ap.add_argument("--json", type=Path, help="also write the scores here")
    args = ap.parse_args()
    pairs = json.loads(DATA.read_text(encoding="utf-8"))["pairs"]
    name = retrieval.embedding_model_name()
    texts = sorted({t for p in pairs for t in (p["approved"], p["asked"])})
    vecs = dict(zip(texts, retrieval.encode_texts(name, texts, normalize_embeddings=True)))
    rows = []
    for p in pairs:
        cos = float(np.dot(vecs[p["approved"]], vecs[p["asked"]]))
        rows.append({**p, "cos": round(cos, 4), "guard": guard_reason(p["approved"], p["asked"], args.tools)})

    print(f"model {name} ({retrieval.embedding_dtype()}), {len(rows)} pairs, agent with tools: {args.tools}\n")
    print(f"{'cos':>6}  {'want':<5} {'lang':<8} {'guard':<28} asked  |  approved")
    for r in sorted(rows, key=lambda r: -r["cos"]):
        print(f"{r['cos']:6.3f}  {'yes' if r['should_match'] else 'NO':<5} {r['lang']:<8} {(r['guard'] or '-')[:28]:<28} "
              f"{r['asked']}  |  {r['approved']}")

    pos = [r for r in rows if r["should_match"]]
    neg = [r for r in rows if not r["should_match"]]
    print("\nthreshold  true matches  false matches (after guards)  false matches (cosine only)")
    sweep = []
    for th in np.round(np.arange(0.70, 0.99, 0.01), 2):
        tp = sum(1 for r in pos if r["cos"] >= th and not r["guard"])
        fp = sum(1 for r in neg if r["cos"] >= th and not r["guard"])
        fp_raw = sum(1 for r in neg if r["cos"] >= th)
        sweep.append((float(th), tp, fp, fp_raw))
        print(f"{th:9.2f}  {tp:5d}/{len(pos):<6d} {fp:5d}/{len(neg):<22d} {fp_raw:5d}/{len(neg)}")
    safe = [s for s in sweep if s[2] == 0]
    best_tp = max(s[1] for s in safe)
    lowest = min(s[0] for s in safe if s[1] == best_tp)
    worst_neg = max((r["cos"] for r in neg if not r["guard"]), default=0.0)
    print(f"\nhighest near miss that only the cosine stops: {worst_neg:.3f}")
    print(f"zero false matches with the most true matches ({best_tp}/{len(pos)}) from threshold {lowest:.2f}; "
          f"current default {ia.DEFAULT_THRESHOLD}")
    if args.json:
        args.json.write_text(json.dumps({"model": name, "rows": rows, "sweep": sweep}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
