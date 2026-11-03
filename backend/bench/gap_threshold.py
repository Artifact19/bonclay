"""Knowledge-gap threshold study: the distribution of `top_score` (the best dense cosine among the chunks retrieval
hands the model, see backend._retrieval_signals) for answerable vs unanswerable questions, to pick KNOWLEDGE_GAP_MIN_SCORE.

Offline, same model and fusion as the API. Answerable = the 45 questions of bench/data/retrieval_questions.json asked of
their own document. Unanswerable = the same questions asked of another template's document (a farmer's question put to
the clinic's knowledge base, ...) plus a few off-topic ones. Run once per chunk size:

    backend/.venv/bin/python backend/bench/gap_threshold.py --chunk-words 40     # small chunks
    backend/.venv/bin/python backend/bench/gap_threshold.py --chunk-words 500    # the production chunker's size
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCH_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BENCH_DIR))

from services import retrieval  # noqa: E402
from retrieval_eval import MODELS, chunk_markdown, DATA  # noqa: E402

OFF_TOPIC = [
    ("en", "What is the capital of France?"), ("en", "Who won the cricket world cup in 2011?"),
    ("en", "Can you recommend a good laptop under fifty thousand rupees?"),
    ("en", "How do I reset my Instagram password?"), ("hi", "फ्रांस की राजधानी क्या है?"),
    ("hi", "मेरे फोन की बैटरी जल्दी खत्म क्यों हो जाती है?"), ("hi", "आज मौसम कैसा रहेगा?"),
    ("hinglish", "Bitcoin ka rate kya chal raha hai?"), ("hinglish", "Mujhe ek acchi movie suggest karo"),
    ("hinglish", "Flight ticket kaise book karte hain?"),
]


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-words", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()
    spec = json.loads(DATA.read_text(encoding="utf-8"))
    docs = {n: chunk_markdown((BACKEND_DIR / rel).read_text(encoding="utf-8"), args.chunk_words)
            for n, rel in spec["docs"].items()}
    name = MODELS["bge-m3"]
    vecs = {d: retrieval.encode_texts(name, c, normalize_embeddings=True) for d, c in docs.items()}

    def top_score(question, doc):
        qv = retrieval.encode_texts(name, [question], normalize_embeddings=True)[0]
        sims = [float(s) for s in vecs[doc] @ qv]
        order = retrieval.hybrid_order(question, docs[doc], sims)[:args.top_k]
        return max(sims[i] for i in order)

    answered, other_doc, off = [], [], []
    for q in spec["questions"]:
        answered.append((top_score(q["q"], q["doc"]), q["lang"]))
        for d in docs:
            if d != q["doc"]:
                other_doc.append((top_score(q["q"], d), q["lang"]))
        for lang, text in OFF_TOPIC:
            pass
    for lang, text in OFF_TOPIC:
        for d in docs:
            off.append((top_score(text, d), lang))
    print(f"chunk_words={args.chunk_words} chunks={ {d: len(c) for d, c in docs.items()} } top_k={args.top_k}")
    for label, data in (("answerable", answered), ("other-doc", other_doc), ("off-topic", off)):
        a = [s for s, _ in data]
        print(f"{label:11s} n={len(a):3d} min={min(a):.3f} p5={pct(a,5):.3f} p25={pct(a,25):.3f} median={pct(a,50):.3f} "
              f"p75={pct(a,75):.3f} p95={pct(a,95):.3f} max={max(a):.3f}")
        for lang in ("en", "hi", "hinglish"):
            b = [s for s, l in data if l == lang]
            if b:
                print(f"    {lang:9s} n={len(b):3d} min={min(b):.3f} median={pct(b,50):.3f} max={max(b):.3f}")
    gap = [s for s, _ in other_doc + off]
    ans = [s for s, _ in answered]
    print("\nthreshold  answerable-flagged  unanswerable-caught")
    for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        print(f"  {t:.2f}     {sum(s < t for s in ans):3d}/{len(ans)} ({sum(s < t for s in ans)/len(ans):5.1%})"
              f"      {sum(s < t for s in gap):3d}/{len(gap)} ({sum(s < t for s in gap)/len(gap):5.1%})")


if __name__ == "__main__":
    main()
