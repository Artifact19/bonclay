"""Knowledge-retrieval evaluation: MiniLM dense vs BGE-M3 dense vs BGE-M3 + BM25 hybrid (+ cross-encoder reranker).

Offline: it imports services/retrieval.py directly (the same tokenizer, BM25 and Reciprocal Rank Fusion the API uses),
loads the embedding models itself and never calls the running API.

Data: bench/data/retrieval_questions.json (45 questions about three template documents, in English, Hindi in
Devanagari and Hinglish in Roman letters). A question is a HIT at k when one of the top-k chunks contains one of its
`gold` strings. The documents are split into small chunks (about 40 words, never across a heading), because the
production chunker (500 words) turns each of these short templates into a single chunk and every system would score 100%.

    backend/.venv/bin/python backend/bench/retrieval_eval.py                       # MiniLM, BGE-M3, hybrid
    backend/.venv/bin/python backend/bench/retrieval_eval.py --reranker            # also BGE-reranker-v2-m3
    backend/.venv/bin/python backend/bench/retrieval_eval.py --models bge-m3 --quick
    backend/.venv/bin/python backend/bench/retrieval_eval.py --models bge-m3 --chunker production   # the app's 120-word chunks

Writes bench/results/retrieval_eval-<timestamp>.json (git-ignored) with every ranking and the timings.
"""
import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCH_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from services import chunking, retrieval  # noqa: E402

DATA = BENCH_DIR / "data" / "retrieval_questions.json"
RESULTS_DIR = BENCH_DIR / "results"
MODELS = {"minilm": retrieval.LEGACY_EMBEDDING_MODEL, "bge-m3": retrieval.DEFAULT_EMBEDDING_MODEL}
KS = (1, 3)


def chunk_markdown(text: str, max_words: int = 40) -> List[str]:
    """Split by `#` headings, then pack lines into chunks of up to `max_words` words (a line is never split); every chunk
    starts with its heading, which is how a section title reaches the embedding."""
    chunks: List[str] = []
    heading, buf, words = "", [], 0

    def flush():
        nonlocal buf, words
        if buf:
            chunks.append((heading + "\n" if heading else "") + "\n".join(buf))
        buf, words = [], 0

    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            flush()
            heading = line.lstrip("# ").strip()
            continue
        n = len(line.split())
        if buf and words + n > max_words:
            flush()
        buf.append(line)
        words += n
    flush()
    return chunks


def load_corpus(max_words: int, chunker: str = "markdown", overlap: int = chunking.DEFAULT_OVERLAP_WORDS):
    """`chunker` markdown: the 40-word packer above (no overlap). production: services/chunking.py, the splitter an
    uploaded document goes through (about 120 words, 20 words of overlap, never across a heading)."""
    spec = json.loads(DATA.read_text(encoding="utf-8"))
    cut = (lambda text: chunking.chunk_text(text, max_words, overlap)) if chunker == "production" \
        else (lambda text: chunk_markdown(text, max_words))
    docs = {name: cut((BACKEND_DIR / rel).read_text(encoding="utf-8")) for name, rel in spec["docs"].items()}
    problems = []
    for i, q in enumerate(spec["questions"]):
        for g in q["gold"]:
            if not any(g in c for c in docs[q["doc"]]):
                problems.append(f"question {i} ({q['q']!r}): gold {g!r} is in no chunk")
    if problems:
        raise SystemExit("gold strings missing from the chunks:\n" + "\n".join(problems))
    return spec["questions"], docs


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        return float("nan")


def is_hit(chunks: List[str], order: List[int], gold: List[str], k: int) -> bool:
    return any(g in chunks[i] for i in order[:k] for g in gold)


def evaluate(questions, docs, model_key: str, use_reranker: bool, timings: Dict[str, dict]):
    """Returns {system name: {"orders": [...per question...]}} for one embedding model."""
    name = MODELS[model_key]
    t = time.perf_counter()
    model = retrieval.load_sentence_transformer(name)
    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    timings[model_key] = {"model": name, "load_s": round(time.perf_counter() - t, 2), "dim": int(dim_fn())}
    doc_vecs = {d: retrieval.encode_texts(name, chunks) for d, chunks in docs.items()}
    # RSS of the whole process after the first real encode (run one model per process for a clean figure)
    timings[model_key]["rss_after_first_encode_mb"] = round(rss_mb())
    systems: Dict[str, List[List[int]]] = {f"{model_key} dense": [], f"{model_key} + BM25 (RRF)": []}
    if use_reranker:
        systems[f"{model_key} + BM25 + reranker"] = []
    qtimes, rr_times = [], []
    for q in questions:
        chunks = docs[q["doc"]]
        t = time.perf_counter()
        qv = retrieval.encode_texts(name, [q["q"]])[0]
        qtimes.append((time.perf_counter() - t) * 1000)
        sims = [float(s) for s in doc_vecs[q["doc"]] @ qv]
        dense = retrieval.dense_ranking(sims)
        fused = retrieval.rrf_fuse([dense, retrieval.bm25_ranking(q["q"], chunks)])
        systems[f"{model_key} dense"].append(dense)
        systems[f"{model_key} + BM25 (RRF)"].append(fused)
        if use_reranker:
            cand = fused[:retrieval.RERANK_CANDIDATES]
            t = time.perf_counter()
            scores = retrieval.rerank_scores(q["q"], [chunks[i] for i in cand])
            rr_times.append((time.perf_counter() - t) * 1000)
            systems[f"{model_key} + BM25 + reranker"].append([cand[j] for j in sorted(range(len(cand)), key=lambda j: -scores[j])])
    timings[model_key]["query_encode_ms_median"] = round(statistics.median(qtimes[1:] or qtimes), 1)  # first one is warm-up
    timings[model_key]["query_encode_ms_p95"] = round(float(np.percentile(qtimes[1:] or qtimes, 95)), 1)
    if rr_times:
        timings[model_key]["rerank_ms_median"] = round(statistics.median(rr_times[1:] or rr_times), 1)
    return systems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="minilm,bge-m3", help="comma list of: minilm, bge-m3")
    ap.add_argument("--reranker", action="store_true", help="also score the fused top 20 with the cross-encoder")
    ap.add_argument("--chunk-words", type=int, default=None, help="words per chunk (default 40; 120 with --chunker production)")
    ap.add_argument("--chunker", choices=["markdown", "production"], default="markdown",
                    help="markdown: 40-word line packer; production: the app's splitter (services/chunking.py)")
    ap.add_argument("--quick", action="store_true", help="first 9 questions only (smoke test)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if args.chunk_words is None:
        args.chunk_words = chunking.DEFAULT_CHUNK_WORDS if args.chunker == "production" else 40
    questions, docs = load_corpus(args.chunk_words, args.chunker)
    if args.quick:
        questions = questions[:9]
    print(f"{len(questions)} questions, {sum(len(c) for c in docs.values())} chunks "
          f"({', '.join(f'{d}: {len(c)}' for d, c in docs.items())}), device={retrieval.embedding_device()}")

    timings: Dict[str, dict] = {}
    results: Dict[str, List[List[int]]] = {}
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        results.update(evaluate(questions, docs, key, args.reranker and key == "bge-m3", timings))
    # BM25 alone, for reference (no model)
    results["BM25 only"] = [retrieval.bm25_ranking(q["q"], docs[q["doc"]]) for q in questions]

    langs = ["en", "hi", "hinglish"]
    header = f"{'system':32s}" + "".join(f"{'hit@' + str(k):>8s}" for k in KS) + "   " + \
             "  ".join(f"{lang + ' @1/@3':>13s}" for lang in langs)
    print("\n" + header)
    rows_out = {}
    for name, orders in results.items():
        total = {k: sum(is_hit(docs[q["doc"]], o, q["gold"], k) for q, o in zip(questions, orders)) for k in KS}
        by_lang = {}
        for lang in langs:
            idx = [i for i, q in enumerate(questions) if q["lang"] == lang]
            by_lang[lang] = {k: sum(is_hit(docs[questions[i]["doc"]], orders[i], questions[i]["gold"], k) for i in idx)
                             for k in KS} | {"n": len(idx)}
        n = len(questions)
        rows_out[name] = {"hit@1": total[1] / n, "hit@3": total[3] / n, "by_lang": by_lang}
        print(f"{name:32s}" + "".join(f"{total[k] / n:8.1%}" for k in KS) + "   " + "  ".join(
            f"{by_lang[l][1]:>4d}/{by_lang[l][3]:>2d} of {by_lang[l]['n']:<2d}" for l in langs))

    print("\ntimings:")
    for key, t in timings.items():
        print(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in t.items()))

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"retrieval_eval-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        misses = {name: [questions[i]["q"] for i, o in enumerate(orders)
                         if not is_hit(docs[questions[i]["doc"]], o, questions[i]["gold"], 3)]
                  for name, orders in results.items()}
        out.write_text(json.dumps({"settings": vars(args) | {"device": retrieval.embedding_device()},
                                   "questions": len(questions), "results": rows_out, "timings": timings,
                                   "missed_at_3": misses}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
