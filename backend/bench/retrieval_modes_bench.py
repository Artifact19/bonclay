"""Retrieval fusion modes: RRF vs weighted score fusion, with and without BGE-M3's sparse and multi-vector scores, the
Hinglish/Devanagari normaliser, and the retrieval confidence rating.

Offline and self-contained: it never calls the running API and never touches backend/data. It loads BGE-M3 in THIS
process (after checking free memory), splits the test documents the way the app does, and ranks with
services/retrieval.rank_rows (the code LocalStore.match_chunks and the hosted path call).

Data:
  bench/data/retrieval_questions.json   45 questions about three template documents (en / hi / Hinglish); a hit is a
                                        passage containing one of the `gold` strings.
  bench/data/doc_qa_questions.json      (when present) hand-written questions about four documents; a hit at k is ALL
                                        `evidence` strings inside the top k (as in doc_qa_eval.py); MRR uses the first
                                        passage with any of them. Its unanswerable questions calibrate the confidence.

Systems (each on the same passages and query vectors):
  head rrf          git HEAD's tokenizer + Reciprocal Rank Fusion (the previous code, loaded from `git show HEAD:`)
  rrf               the new normaliser + RRF (RETRIEVAL_FUSION=rrf)
  weighted (default) the new normaliser + weighted fusion (DEFAULT_FUSION_WEIGHTS), theoretical min-max
  weighted-minmax   the same with observed min-max normalisation
  dense / bm25      one signal only
  +sparse, +colbert, +m3 (all four), rrf+m3   only when BGE-M3's head files are present (services/m3_modes.py)
  bm25 weight grid  weighted dense+BM25 with BM25 at 0.1 .. 0.6

    backend/.venv/bin/python -I backend/bench/retrieval_modes_bench.py
    backend/.venv/bin/python -I backend/bench/retrieval_modes_bench.py --fake-heads     # M3 LATENCY only (random heads)
    backend/.venv/bin/python -I backend/bench/retrieval_modes_bench.py --latency-only   # synthetic 50-page corpus, no model

Writes bench/results/retrieval_modes-<stamp>.json (git-ignored).
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCH_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from services import chunking, m3_modes, retrieval  # noqa: E402

RESULTS_DIR = BENCH_DIR / "results"
RQ = BENCH_DIR / "data" / "retrieval_questions.json"
DQ = BENCH_DIR / "data" / "doc_qa_questions.json"
LANGS = ("en", "hi", "hinglish")


def ws(s: str) -> str:
    return " ".join(s.split())


def available_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def pct(xs: List[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


# ----------------------------------------------------------------------------------------------- corpora

def passages_of(text: str, filename: str, chunker: str) -> List[dict]:
    if chunker == "structured":
        ps = chunking.chunk_document(text, chunking.guess_title(text, filename))
        return [{"chunk_text": p["content"], "header": p.get("header") or "", "kind": "passage"} for p in ps]
    if chunker == "small40":
        sys.path.insert(0, str(BENCH_DIR))
        from retrieval_eval import chunk_markdown  # noqa: PLC0415
        return [{"chunk_text": c, "header": "", "kind": "passage"} for c in chunk_markdown(text, 40)]
    return [{"chunk_text": c, "header": "", "kind": "passage"} for c in chunking.chunk_text(text)]


def load_sets(chunker_rq: str) -> List[dict]:
    """[{name, docs: {key: [rows]}, questions: [...], hit: "any"|"all"}]"""
    sets = []
    spec = json.loads(RQ.read_text(encoding="utf-8"))
    docs = {k: passages_of((BACKEND_DIR / rel).read_text(encoding="utf-8"), rel, chunker_rq)
            for k, rel in spec["docs"].items()}
    qs = [{"id": f"rq-{i}", "doc": q["doc"], "lang": q["lang"], "q": q["q"], "gold": q["gold"], "answerable": True}
          for i, q in enumerate(spec["questions"])]
    sets.append({"name": f"retrieval_questions ({chunker_rq})", "docs": docs, "questions": qs, "hit": "any"})
    if DQ.exists():
        sys.path.insert(0, str(BENCH_DIR))
        import doc_qa_eval  # noqa: PLC0415 - read-only use of its document loader
        dspec = doc_qa_eval.load_spec()
        texts = doc_qa_eval.doc_texts(dspec)
        ddocs = {k: passages_of(texts[k], dspec["docs"][k]["source"], "structured") for k in texts}
        dqs = [{"id": q["id"], "doc": q["doc"], "lang": q["lang"], "q": q["q"], "type": q["type"],
                "gold": q.get("evidence") or [], "answerable": q["type"] != "unanswerable"} for q in dspec["questions"]]
        sets.append({"name": "doc_qa_questions (structured)", "docs": ddocs, "questions": dqs, "hit": "all"})
    for s in sets:
        for key, rows in s["docs"].items():
            for i, r in enumerate(rows):
                r["id"] = f"{key}-{i}"
    return sets


def ranks(rows: List[dict], order: List[int], gold: List[str], mode: str):
    """(rank where the hit condition is met, rank of the first passage with any gold string), 1-based or None."""
    golds = [ws(g) for g in gold]
    seen, first_any = set(), None
    for r, i in enumerate(order, start=1):
        text = ws(retrieval.passage_text(rows[i]))
        found = {g for g in golds if g in text}
        if found and first_any is None:
            first_any = r
            if mode == "any":
                return r, r
        seen |= found
        if mode == "all" and len(seen) == len(golds):
            return r, first_any
    return None, first_any


def metrics(results: List[dict]) -> dict:
    n = len(results)
    if not n:
        return {"n": 0}
    out = {"n": n}
    for k in (1, 3):
        out[f"hit@{k}"] = sum(1 for r in results if r["rank"] is not None and r["rank"] <= k) / n
    out["mrr"] = sum(1.0 / r["first"] for r in results if r["first"]) / n
    return out


# ----------------------------------------------------------------------------------------------- encoding

def make_fake_heads(dim: int = 1024) -> str:
    """Random sparse/colbert heads in a temp dir (latency measurements only: the scores mean nothing)."""
    import torch
    d = tempfile.mkdtemp(prefix="fake_m3_heads_")
    g = torch.Generator().manual_seed(0)
    torch.save({"weight": torch.randn(1, dim, generator=g) * 0.05, "bias": torch.zeros(1)},
               os.path.join(d, "sparse_linear.pt"))
    torch.save({"weight": torch.randn(dim, dim, generator=g) / dim ** 0.5, "bias": torch.zeros(dim)},
               os.path.join(d, "colbert_linear.pt"))
    return d


def encode_passages(model_name: str, texts: List[str], with_m3: bool):
    """Dense vectors (and PassageM3 when the heads are present) from ONE forward pass of the resident model, the same
    float16 model the app answers questions with (passages are embedded in fp32 by the app; hits are identical, see
    retrieval_eval.py)."""
    model = retrieval.load_sentence_transformer(model_name)
    heads = m3_modes.load_heads(model_name) if with_m3 else None
    dense, reps = [], []
    for i in range(0, len(texts), 8):
        part = texts[i:i + 8]
        outs = m3_modes.forward_outputs(model, part)
        for o, t in zip(outs, part):
            if heads is not None:
                d, rep = m3_modes.reps_from_output(o, t, heads, model.tokenizer, int(model.max_seq_length or 512))
                reps.append(rep)
            else:
                d = o["sentence_embedding"].float().cpu().numpy()
                d = d / max(float(np.linalg.norm(d)), 1e-12)
                reps.append(None)
            dense.append(np.asarray(d, dtype=np.float32))
    return dense, reps


# ----------------------------------------------------------------------------------------------- systems

def load_head_retrieval():
    """services/retrieval.py as committed at git HEAD (the previous tokenizer and RRF), as a separate module."""
    try:
        src = subprocess.run(["git", "-C", str(REPO_ROOT), "show", "HEAD:backend/services/retrieval.py"],
                             capture_output=True, text=True, check=True).stdout
    except Exception as e:  # noqa: BLE001
        print(f"  (no git HEAD baseline: {e})")
        return None
    path = Path(tempfile.mkdtemp(prefix="head_retrieval_")) / "retrieval_head.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("retrieval_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class Env:
    """Temporarily set environment variables."""

    def __init__(self, **kv):
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def systems(has_m3: bool) -> Dict[str, dict]:
    """name -> {env, m3}"""
    out: Dict[str, dict] = {
        "rrf": {"env": {"RETRIEVAL_FUSION": "rrf"}, "m3": False},
        "weighted (default)": {"env": {"RETRIEVAL_FUSION": "weighted", "RETRIEVAL_WEIGHTS": None}, "m3": False},
        "weighted-minmax": {"env": {"RETRIEVAL_FUSION": "weighted", "RETRIEVAL_WEIGHTS": None,
                                    "RETRIEVAL_NORM": "minmax"}, "m3": False},
        "dense": {"env": {"HYBRID_SEARCH": "off"}, "m3": False},
        "bm25": {"env": {"RETRIEVAL_FUSION": "weighted", "RETRIEVAL_WEIGHTS": "dense:0,sparse:0,colbert:0,bm25:1"},
                 "m3": False},
    }
    for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        out[f"grid bm25={w}"] = {"env": {"RETRIEVAL_FUSION": "weighted",
                                         "RETRIEVAL_WEIGHTS": f"dense:{1 - w:.2f},sparse:0,colbert:0,bm25:{w}"},
                                 "m3": False}
    if has_m3:
        out.update({
            "weighted+sparse": {"env": {"RETRIEVAL_WEIGHTS": "dense:0.4,sparse:0.1,colbert:0,bm25:0.1"}, "m3": True},
            "weighted+colbert": {"env": {"RETRIEVAL_WEIGHTS": "dense:0.4,sparse:0,colbert:0.4,bm25:0.1"}, "m3": True},
            "weighted+m3 (default)": {"env": {"RETRIEVAL_WEIGHTS": None}, "m3": True},
            "m3 only (paper 1:0.3:1)": {"env": {"RETRIEVAL_WEIGHTS": "dense:1,sparse:0.3,colbert:1,bm25:0"}, "m3": True},
            "rrf+m3": {"env": {"RETRIEVAL_FUSION": "rrf"}, "m3": True},
        })
    return out


# ----------------------------------------------------------------------------------------------- synthetic latency

def synthetic_latency(pages: int = 50, words_per_page: int = 500, tokens: int = 200, runs: int = 60) -> dict:
    """rank_rows on a synthetic `pages`-page corpus (120-word passages, random vectors), with and without the M3
    signals (fake passage reps in memory). Measures the fusion/scoring work a live turn adds; no model."""
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(3000)] + "fee doctor clinic sunday monday time open closed rupee hospital".split()
    n_pass = pages * words_per_page // 120
    rows = [{"id": f"p{i}", "chunk_text": " ".join(rng.choice(vocab) for _ in range(120)), "header": f"Doc > S{i % 12}",
             "kind": "passage"} for i in range(n_pass)]
    nrng = np.random.default_rng(0)
    emb = nrng.normal(size=(n_pass, 1024)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    store = m3_modes.M3Store(tempfile.mkdtemp(prefix="m3_synth_"), "synthetic", encoder=lambda t: iter(()),
                             mem_mb=512)
    texts = [retrieval.passage_text(r) for r in rows]
    for r, t in zip(rows, texts):
        vecs = nrng.normal(size=(tokens, 1024)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store._remember(store.key(r["id"], t), m3_modes.PassageM3(
            sparse={k: float(nrng.random()) for k in retrieval.content_tokens(t)[:60]}, colbert=vecs.astype(np.float16)))
    qv = nrng.normal(size=(16, 1024)).astype(np.float32)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    qm3 = m3_modes.QueryM3(sparse={"fe": 0.3, "doctor": 0.25, "sunday": 0.2}, colbert=qv, store=store)
    out = {"passages": n_pass, "tokens_per_passage": tokens}
    for label, m3 in (("dense+bm25", None), ("dense+bm25+sparse+colbert", qm3)):
        times = []
        for i in range(runs):
            q = emb[i % n_pass] + nrng.normal(size=1024).astype(np.float32) * 0.5
            q /= np.linalg.norm(q)
            sims = (emb @ q).tolist()
            query = f"doctor fee kitni hai sunday {rows[i % n_pass]['chunk_text'].split()[3]}"
            t = time.perf_counter()
            info: dict = {}
            ranked = retrieval.rank_rows(query, rows, sims, query_m3=m3, info=info)
            retrieval.assess_confidence([{**rows[e["index"]], **e} for e in ranked[:3]], query, info)
            times.append((time.perf_counter() - t) * 1000)
        times = times[3:]
        out[label] = {"ms_median": round(statistics.median(times), 2), "ms_p95": round(pct(times, 95), 2),
                      "signals": info.get("signals")}
    out["memory_mb_passage_vectors"] = round(store.memory_bytes / 2 ** 20, 1)
    return out


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake-heads", action="store_true", help="random M3 heads: latency only, accuracy meaningless")
    ap.add_argument("--latency-only", action="store_true", help="only the synthetic 50-page latency test (no model)")
    ap.add_argument("--rq-chunker", choices=["structured", "small40", "legacy"], default="small40",
                    help="passages for retrieval_questions.json (default small40: the app's structured chunker makes "
                         "only 7-8 passages per template, which every system ranks perfectly at 3)")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    out: dict = {"settings": vars(args), "sets": {}}

    print("synthetic 50-page corpus (fusion + scoring work per query, no model):")
    out["synthetic_latency"] = synthetic_latency()
    print("  " + json.dumps(out["synthetic_latency"]))
    if args.latency_only:
        return save(out, args)

    model = retrieval.embedding_model_name()
    free = available_gb()
    print(f"\nmodel {model}, {free:.1f} GB available")
    if free < args.min_free_gb:
        raise SystemExit(f"only {free:.1f} GB available; need {args.min_free_gb} GB to load BGE-M3 in this process")
    if args.fake_heads:
        os.environ["M3_HEADS_DIR"] = make_fake_heads()
        m3_modes._heads_paths_cache.clear()
    has_m3 = m3_modes.active(model)
    print(f"BGE-M3 heads: {'present' if has_m3 else 'MISSING (sparse/colbert systems skipped)'}"
          + (" [FAKE, latency only]" if args.fake_heads else ""))
    t = time.perf_counter()
    retrieval.load_sentence_transformer(model)
    out["model_load_s"] = round(time.perf_counter() - t, 2)
    head = load_head_retrieval()

    # query encoding: plain dense vs one pass with every output
    sets = load_sets(args.rq_chunker)
    all_q = [q for s in sets for q in s["questions"]]
    qvec, qm3s, t_plain, t_m3 = {}, {}, [], []
    for q in all_q:
        t = time.perf_counter()
        v = retrieval.encode_texts(model, [q["q"]])[0]
        t_plain.append((time.perf_counter() - t) * 1000)
        qvec[q["id"]] = np.asarray(v, dtype=np.float32)
        if has_m3:
            t = time.perf_counter()
            v2, qm = m3_modes.encode_query(model, q["q"])
            t_m3.append((time.perf_counter() - t) * 1000)
            qm3s[q["id"]] = qm
            assert float(np.dot(v2, qvec[q["id"]])) > 0.999, "one-pass dense vector differs from encode_texts"
    out["query_encode_ms"] = {"plain_median": round(statistics.median(t_plain[2:]), 1),
                              "plain_p95": round(pct(t_plain[2:], 95), 1)}
    if t_m3:
        out["query_encode_ms"].update({"one_pass_m3_median": round(statistics.median(t_m3[2:]), 1),
                                       "one_pass_m3_p95": round(pct(t_m3[2:], 95), 1)})
    print("query encoding ms: " + json.dumps(out["query_encode_ms"]))

    for s in sets:
        name = s["name"]
        all_rows = [r for rows in s["docs"].values() for r in rows]
        texts = [chunking.embedding_text(r["header"], r["chunk_text"]) for r in all_rows]
        t = time.perf_counter()
        dense, reps = encode_passages(model, texts, has_m3)
        enc_s = time.perf_counter() - t
        store = m3_modes.M3Store(tempfile.mkdtemp(prefix="m3_bench_"), model, encoder=lambda x: iter(()), mem_mb=1024)
        for r, d, rep in zip(all_rows, dense, reps):
            r["_vec"] = d
            if rep is not None:
                store._remember(store.key(r["id"], retrieval.passage_text(r)), rep)
        for qm in qm3s.values():
            if qm is not None:
                qm.store = store
        print(f"\n== {name}: {len(s['questions'])} questions, {len(all_rows)} passages "
              f"({', '.join(f'{k}: {len(v)}' for k, v in s['docs'].items())}); passages encoded in {enc_s:.1f} s")
        answerable = [q for q in s["questions"] if q["answerable"]]
        res_set: dict = {"passages": len(all_rows), "systems": {}}
        sysdefs = systems(has_m3)
        names = (["head rrf"] if head else []) + list(sysdefs)
        for sname in names:
            results, times = [], []
            for q in answerable:
                rows = s["docs"][q["doc"]]
                sims = [float(np.dot(r["_vec"], qvec[q["id"]])) for r in rows]
                t = time.perf_counter()
                if sname == "head rrf":
                    order = head.hybrid_order(q["q"], [retrieval.passage_text(r) for r in rows], sims)
                else:
                    d = sysdefs[sname]
                    with Env(**d["env"]):
                        ranked = retrieval.rank_rows(q["q"], rows, sims, query_m3=qm3s.get(q["id"]) if d["m3"] else None)
                    order = [e["index"] for e in ranked]
                times.append((time.perf_counter() - t) * 1000)
                rank, first = ranks(rows, order, q["gold"], s["hit"])
                results.append({"id": q["id"], "lang": q["lang"], "rank": rank, "first": first})
            m = metrics(results)
            m["by_lang"] = {lang: metrics([r for r in results if r["lang"] == lang]) for lang in LANGS}
            m["rank_ms_median"] = round(statistics.median(times), 2)
            m["rank_ms_p95"] = round(pct(times, 95), 2)
            m["misses_at_3"] = [r["id"] for r in results if r["rank"] is None or r["rank"] > 3]
            res_set["systems"][sname] = m
            bl = "  ".join(f"{lang} {m['by_lang'][lang].get('hit@1', 0):.0%}/{m['by_lang'][lang].get('hit@3', 0):.0%}"
                           for lang in LANGS if m["by_lang"][lang]["n"])
            print(f"  {sname:26s} hit@1 {m['hit@1']:6.1%}  hit@3 {m['hit@3']:6.1%}  MRR {m['mrr']:.3f}  "
                  f"rank {m['rank_ms_median']:5.2f} ms   [{bl}]")

        # confidence calibration on the default system: answerable (hit / miss at 3) vs unanswerable
        table: Dict[str, Dict[str, int]] = {}
        feats = []
        for q in s["questions"]:
            rows = s["docs"][q["doc"]]
            sims = [float(np.dot(r["_vec"], qvec[q["id"]])) for r in rows]
            info: dict = {}
            ranked = retrieval.rank_rows(q["q"], rows, sims, query_m3=qm3s.get(q["id"]), info=info)
            top = [{**rows[e["index"]], **e} for e in ranked[:3]]
            conf = retrieval.assess_confidence(top, q["q"], info)
            rank, _ = ranks(rows, [e["index"] for e in ranked], q["gold"], s["hit"]) if q["answerable"] else (None, None)
            bucket = "unanswerable" if not q["answerable"] else ("hit@3" if rank and rank <= 3 else "miss@3")
            table.setdefault(bucket, {}).setdefault(conf["level"], 0)
            table[bucket][conf["level"]] += 1
            feats.append({"id": q["id"], "bucket": bucket, "lang": q["lang"], **{k: conf.get(k) for k in
                                                                              ("level", "top", "gap", "dense", "overlap", "agree")}})
        res_set["confidence"] = table
        res_set["confidence_features"] = feats
        print("  confidence level by bucket: " + json.dumps(table))
        for b in ("hit@3", "miss@3", "unanswerable"):
            ds = [f["dense"] for f in feats if f["bucket"] == b and f["dense"] is not None]
            if ds:
                print(f"    {b:12s} top dense: min {min(ds):.3f}  p10 {pct(ds, 10):.3f}  median {statistics.median(ds):.3f}"
                      f"  p90 {pct(ds, 90):.3f}  max {max(ds):.3f}  (n={len(ds)})")
        out["sets"][name] = res_set
    save(out, args)


def save(out: dict, args) -> None:
    if args.no_save:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"retrieval_modes-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
