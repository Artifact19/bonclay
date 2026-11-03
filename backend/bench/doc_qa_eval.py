"""Document question answering evaluation: retrieval quality and end-to-end answer accuracy on four documents.

Test set: bench/data/doc_qa_questions.json (133 hand-written questions about the Sahyadri Care Hospital PDF in docs/demo
and the clinic receptionist, college admissions and HOD office template documents; English, Hindi in Devanagari and
Hinglish in Roman letters; 20 of them unanswerable). Each answerable question lists `evidence`: exact substrings of the
document text, all of which a grounded answer needs.

It never talks to the running API and never writes to backend/data: everything runs in this process, on a temporary
copy of the local store (a LocalStore in a temp dir), with the app's own code.

Phase 1, retrieval (--retrieval-only). Loads BGE-M3 the way the app does (services/retrieval.py: float16 copy for
questions, float32 for passages), splits each document with services/chunking.py, stores the passages in a temporary
LocalStore and ranks them with LocalStore.match_chunks (dense cosine + BM25 fused with RRF, exactly what
backend.retrieve_context calls in local mode). A question is a hit at k when ALL its evidence strings are inside the
top k passages; MRR uses the first passage holding ANY evidence string. Variants:
    prod      120-word passages, 20-word overlap, hybrid (what the app does; the app hands the model the top 3)
    dense     the same passages, HYBRID_SEARCH=off (dense cosine only)
    bm25      the same passages, BM25 only (retrieval.bm25_ranking)
    chunk80 / chunk200   80- or 200-word passages, hybrid
    pooled    all four documents in ONE agent (a harder, multi-document setting), hybrid
    rerank    prod + bge-reranker-v2-m3 over the fused candidates (RERANKER=on path, retrieval.rerank); skipped with a
              note when the model is not downloaded or less than --min-free-gb of memory is available

Phase 2, answers (--answers-only). Imports backend.py with its data paths pointed at a temp dir, creates the agents
through the real template code (routers.templates._create_from_template; the hospital agent is the clinic template
personalised for Sahyadri Care Hospital with the PDF uploaded through backend._extract_knowledge_text and
backend.ingest_document) and answers every question with backend._turn_blocking, the typed-text turn path of
POST /api/agents/{id}/turn: language policy, retrieval, tools, the system prompt (_turn_messages), Groq through
services/llm.py with its fallbacks, and _enforce_language. Only speech synthesis, the watermark and the audio file are
stubbed (_turn_spoken, _turn_voice, _turn_synth, provenance.apply_provenance, _turn_store). Instant answers are off
(the temp store has none anyway), Google Calendar is off and the MCP tool servers write to temp files.
Groq calls are paced to stay under the free tier (8,000 tokens a minute per model); a reply that a fallback model wrote
is recorded and, with --retry-fallback (default), the turn is asked again once after a pause.

Phase 3, judge (runs after phase 2 unless --no-judge; --judge-only FILE re-judges saved answers). A different model
(openai/gpt-oss-120b by default; gpt-oss-20b for a reply gpt-oss-120b wrote) grades each reply with a strict rubric,
next to deterministic checks: the reply language (language_policy.reply_matches, the app's own check), the key numbers
of the gold answer, and numbers in the reply that appear nowhere in the agent's documents, prompt or the question.

    backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --retrieval-only
    backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --retrieval-only --variants prod,rerank
    backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --answers-only --limit 5          # smoke test
    backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --answers-only                     # full run, ~1 h
    backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --answers-only --answer-top-k 5    # top-5 variant
    backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --judge-only bench/results/doc_qa_answers-....json
    backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --summary bench/results/doc_qa_answers-....json
    # "after" numbers for newer code: drop the baseline pins, set what is under test
    backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --no-baseline-pins --set FACT_SHEET=on --label -after

Every result file records the code it ran (git commit and `git status --short`, or --code-label), the pinned and
--set values, and the retrieval/answer settings in the environment (never secrets).

Results: bench/results/doc_qa_retrieval-<stamp>.json and doc_qa_answers-<stamp>-k<k>.json (git-ignored), written as the
run goes so an interrupted run can be resumed with --resume FILE.
"""
import argparse
import asyncio
import collections
import datetime as dt
import gc
import io
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BENCH_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCH_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DATA = BENCH_DIR / "data" / "doc_qa_questions.json"
RESULTS_DIR = BENCH_DIR / "results"
LANGS = ("en", "hi", "hinglish")
TYPES = ("direct", "paraphrase", "number", "yesno", "multi", "distractor", "unanswerable")
DOC_ORDER = ("hospital", "clinic", "college", "hod")
KS = (1, 3, 5)
PRIMARY_MODEL = "qwen/qwen3.8-27b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
USER_AGENT = "BonClay/1.0 (+https://github.com/Artifact19/bonclay)"
EVAL_USER = "doc-qa-eval"
GIB = 1024 ** 3


# The switches other work adds to retrieval and answering, pinned to the ORIGINAL behaviour (commit a9f150c) by default
# so a run measures that baseline; --no-baseline-pins (plus --set KEY=VALUE) measures the code as configured instead.
# Switches the code under test does not read are harmless. LLM_ANSWER_TEMPERATURE=0.3 is the temperature the original
# turn path hard-codes.
BASELINE_PINS = {
    "QUESTION_INDEX": "off", "STRUCTURED_CHUNKING": "off", "GROUNDING_PROMPT": "v1", "CONFIDENCE_GATE": "off",
    "ANSWER_GUARD": "off", "FACT_SHEET": "off", "QUERY_CONDENSE": "off", "ASR_VOCAB_PROMPT": "off", "RERANKER": "off",
    "RETRIEVAL_FUSION": "rrf", "HYBRID_SEARCH": "on", "LLM_THINKING": "off", "LLM_ANSWER_TEMPERATURE": "0.3",
}
RECORDED_ENV = tuple(BASELINE_PINS) + (
    "EMBEDDING_MODEL", "EMBEDDING_DTYPE", "EMBEDDING_DEVICE", "EMBEDDING_FP16_DIR", "GROQ_CHAT_MODEL",
    "GROQ_FALLBACK_MODELS", "LLM_PROVIDERS", "INSTANT_ANSWERS", "BONCLAY_TOOL_PROTOCOL", "RETRIEVAL_WEIGHTS",
    "RETRIEVAL_NORM")
RUN_INFO: Dict[str, Any] = {}


def apply_settings(args) -> None:
    """Pins and --set values go into the environment before anything reads it (load_dotenv never overrides them)."""
    applied = dict(BASELINE_PINS) if args.baseline_pins else {}
    for item in args.set or []:
        if "=" not in item:
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        applied[k.strip()] = v
    os.environ.update(applied)
    import platform
    import subprocess
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(BACKEND_DIR), *a], capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    RUN_INFO.update({
        "code_dir": str(BACKEND_DIR), "code_label": args.code_label or None,
        "git_commit": git("rev-parse", "HEAD") or None,
        "git_status": (git("status", "--short") or "").splitlines() if git("rev-parse", "HEAD") else None,
        "baseline_pins": bool(args.baseline_pins), "set": applied, "python": sys.version.split()[0],
        "platform": platform.platform(), "started": dt.datetime.now().isoformat(timespec="seconds"),
    })


def record_env() -> dict:
    """The settings that shape retrieval and answering, as this process sees them (never keys or secrets)."""
    return {k: os.environ.get(k) for k in RECORDED_ENV}


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def load_env() -> None:
    """backend/.env (and frontend/.env.local) the way backend.py loads them: never overriding what is already set.
    Values are never printed."""
    from dotenv import load_dotenv
    env_file = Path(os.environ.get("DOC_QA_ENV_FILE") or BACKEND_DIR / ".env")
    load_dotenv(env_file.parent.parent / "frontend" / ".env.local")
    load_dotenv(env_file)


def available_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / GIB


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1e6


# ----------------------------------------------------------------------------------------------- documents

def extract_pdf_text(path: Path) -> str:
    """The text an uploaded PDF becomes (the pypdf loop of backend._extract_knowledge_text, copied for the retrieval
    phase so it does not need to import backend.py; the answers phase uses the backend function itself and checks
    that both agree)."""
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(path.read_bytes()))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()


def sample_banner() -> str:
    from routers.templates import SAMPLE_BANNER
    return SAMPLE_BANNER


def load_spec() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def doc_texts(spec: dict) -> Dict[str, str]:
    """Each document's text as the app stores it: the PDF's extracted text, a template document with the sample
    banner the template code puts in front of it."""
    out = {}
    banner = sample_banner()
    for key, d in spec["docs"].items():
        path = REPO_ROOT / d["source"]
        out[key] = extract_pdf_text(path) if d["kind"] == "pdf" else banner + path.read_text(encoding="utf-8")
    return out


def check_gold(spec: dict, texts: Dict[str, str]) -> None:
    problems = []
    for q in spec["questions"]:
        for e in q.get("evidence") or []:
            if e not in texts[q["doc"]]:
                problems.append(f"{q['id']}: evidence {e!r} is not in the {q['doc']} text")
        if q["type"] == "unanswerable" and q.get("evidence"):
            problems.append(f"{q['id']}: unanswerable but has evidence")
        if q["type"] != "unanswerable" and not q.get("evidence"):
            problems.append(f"{q['id']}: answerable without evidence")
    if problems:
        raise SystemExit("gold evidence problems:\n" + "\n".join(problems))


def select(questions: List[dict], args) -> List[dict]:
    qs = questions
    if args.ids:
        want = set(args.ids.split(","))
        qs = [q for q in qs if q["id"] in want]
    if args.docs:
        want = set(args.docs.split(","))
        qs = [q for q in qs if q["doc"] in want]
    if args.langs:
        want = set(args.langs.split(","))
        qs = [q for q in qs if q["lang"] in want]
    if args.limit and args.limit < len(qs):
        step = len(qs) / args.limit          # evenly spaced: a spread over documents, languages and types
        qs = [qs[int(i * step)] for i in range(args.limit)]
    return qs


# ----------------------------------------------------------------------------------------------- metrics

def ranks(chunks: List[str], order: List[int], evidence: List[str], doc_ok=None) -> Tuple[Optional[int], Optional[int]]:
    """(rank at which ALL evidence strings are covered, rank of the first passage with ANY of them), 1-based."""
    seen = set()
    first_any = None
    for r, i in enumerate(order, start=1):
        if doc_ok is not None and not doc_ok(i):
            continue
        found = {e for e in evidence if e in chunks[i]}
        if found and first_any is None:
            first_any = r
        seen |= found
        if len(seen) == len(evidence):
            return r, first_any
    return None, first_any


def retrieval_metrics(rows: List[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    out = {"n": n}
    for k in KS:
        out[f"hit@{k}"] = sum(1 for r in rows if r["rank_all"] is not None and r["rank_all"] <= k) / n
    out["any@3"] = sum(1 for r in rows if r["rank_any"] is not None and r["rank_any"] <= 3) / n
    out["mrr"] = sum(1.0 / r["rank_any"] for r in rows if r["rank_any"]) / n
    return out


def breakdown(rows: List[dict], fn) -> dict:
    out = {"all": fn(rows)}
    for key, values in (("lang", LANGS), ("type", TYPES), ("doc", DOC_ORDER)):
        for v in values:
            sub = [r for r in rows if r[key] == v]
            if sub:
                out[f"{key}={v}"] = fn(sub)
    return out


# ----------------------------------------------------------------------------------------------- phase 1

def run_retrieval(args, spec: dict, questions: List[dict]) -> dict:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from services import chunking, retrieval
    from services.local_store import LocalStore

    texts = doc_texts(spec)
    check_gold(spec, texts)
    questions = [q for q in questions if q["type"] != "unanswerable"]
    model = retrieval.embedding_model_name()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"retrieval: {len(questions)} answerable questions, model {model} ({retrieval.embedding_dtype()} queries, "
          f"device {retrieval.embedding_device()}), variants {variants}; {available_gb():.1f} GB available")
    if available_gb() < args.min_free_gb:
        raise SystemExit(f"only {available_gb():.1f} GB of memory available; need {args.min_free_gb} GB to load BGE-M3 "
                         "and its float32 passage encoder. Close something and run again.")

    tmp = tempfile.mkdtemp(prefix="doc_qa_retrieval_")
    out: Dict[str, Any] = {"run": {**RUN_INFO, "env": record_env()}, "settings": {"model": model, "dtype": retrieval.embedding_dtype(),
                                        "device": retrieval.embedding_device(), "variants": variants,
                                        "questions": len(questions)},
                           "variants": {}, "timings": {}, "notes": []}
    t0 = time.perf_counter()
    retrieval.load_sentence_transformer(model)
    out["timings"]["model_load_s"] = round(time.perf_counter() - t0, 2)

    def build(chunk_words: int, overlap: int, pooled: bool = False):
        """A temp LocalStore with one agent per document (or one pooled agent) and the passages embedded the way
        backend.ingest_document embeds them (one encode call per document)."""
        store = LocalStore(tempfile.mkdtemp(dir=tmp))
        agents, chunk_doc = {}, {}
        pooled_id = store.insert("agents", {"user_id": EVAL_USER, "name": "pooled"})["id"] if pooled else None
        for key in DOC_ORDER:
            chunks = chunking.chunk_text(texts[key], chunk_words, overlap)
            vecs = [passage_vec[c] for c in chunks]
            agent_id = pooled_id or store.insert("agents", {"user_id": EVAL_USER, "name": key})["id"]
            doc = store.insert("documents", {"agent_id": agent_id, "user_id": EVAL_USER, "filename": key,
                                             "chunk_count": len(chunks)})
            store.insert_many("document_chunks", [
                {"document_id": doc["id"], "chunk_text": c, "chunk_index": i, "embedding": [float(x) for x in v],
                 "embed_model": model, "embed_dim": len(v)} for i, (c, v) in enumerate(zip(chunks, vecs))])
            agents[key] = agent_id
            chunk_doc[doc["id"]] = key
        return store, agents, chunk_doc, enc_ms

    # Every passage of every variant, embedded in ONE call: encode_texts takes the same float32 path an uploaded
    # document takes (_encode_bulk_fp32), but the 2.2 GB float32 copy of the weights is loaded once, not 16 times.
    all_chunks: List[str] = []
    for v in variants:
        cw, ov = {"chunk80": (80, 20), "chunk200": (200, 20)}.get(
            v, (chunking.DEFAULT_CHUNK_WORDS, chunking.DEFAULT_OVERLAP_WORDS))
        for key in DOC_ORDER:
            all_chunks += [c for c in chunking.chunk_text(texts[key], cw, ov) if c not in all_chunks]
    t = time.perf_counter()
    passage_vec = dict(zip(all_chunks, retrieval.encode_texts(model, all_chunks)))
    out["timings"]["passage_encode_s_all"] = round(time.perf_counter() - t, 1)
    out["timings"]["passages_encoded"] = len(all_chunks)
    print(f"  embedded {len(all_chunks)} passages in {out['timings']['passage_encode_s_all']} s; "
          f"RSS {rss_mb():.0f} MB, {available_gb():.1f} GB available")
    enc_ms = 0.0

    qvecs: Dict[str, Any] = {}
    q_ms: List[float] = []

    def qvec(q: dict):
        if q["id"] not in qvecs:
            t = time.perf_counter()
            qvecs[q["id"]] = retrieval.encode_texts(model, [q["q"]])[0]
            q_ms.append((time.perf_counter() - t) * 1000)
        return qvecs[q["id"]]

    def evaluate(name: str, store, agents, chunk_doc, mode: str = "hybrid", keep_rows: bool = False):
        rows, rank_ms, saved = [], [], {}
        prev = os.environ.get("HYBRID_SEARCH")
        if mode == "dense":
            os.environ["HYBRID_SEARCH"] = "off"
        try:
            for q in questions:
                agent_id = agents[q["doc"]]
                n_all = len(store.select("document_chunks"))
                t = time.perf_counter()
                if mode == "bm25":
                    doc_ids = {d["id"] for d in store.select("documents", lambda r: r.get("agent_id") == agent_id)}
                    cand = store.select("document_chunks", lambda r: r.get("document_id") in doc_ids)
                    order = retrieval.bm25_ranking(q["q"], [c["chunk_text"] for c in cand])
                    hits = [cand[i] for i in order]
                else:
                    hits = store.match_chunks(agent_id, qvec(q), n_all, q["q"])
                rank_ms.append((time.perf_counter() - t) * 1000)
                chunks = [h["chunk_text"] for h in hits]
                docs = [chunk_doc.get(h["document_id"]) for h in hits]
                r_all, r_any = ranks(chunks, list(range(len(chunks))), q["evidence"],
                                     doc_ok=lambda i, d=q["doc"]: docs[i] == d)
                rows.append({"id": q["id"], "lang": q["lang"], "type": q["type"], "doc": q["doc"],
                             "rank_all": r_all, "rank_any": r_any, "n_chunks": len(chunks),
                             "top3": [c[:90] for c in chunks[:3]]})
                if keep_rows:
                    saved[q["id"]] = hits[:retrieval.RERANK_CANDIDATES]
        finally:
            if prev is None:
                os.environ.pop("HYBRID_SEARCH", None)
            else:
                os.environ["HYBRID_SEARCH"] = prev
        res = {"metrics": breakdown(rows, retrieval_metrics), "rows": rows,
               "rank_ms_median": round(statistics.median(rank_ms), 2)}
        out["variants"][name] = res
        m = res["metrics"]["all"]
        print(f"  {name:10s} hit@1 {m['hit@1']:.1%}  hit@3 {m['hit@3']:.1%}  hit@5 {m['hit@5']:.1%}  MRR {m['mrr']:.3f}")
        return saved

    prod_rows: Dict[str, List[dict]] = {}
    sizes = {"prod": (chunking.DEFAULT_CHUNK_WORDS, chunking.DEFAULT_OVERLAP_WORDS),
             "chunk80": (80, 20), "chunk200": (200, 20)}
    if any(v in variants for v in ("prod", "dense", "bm25", "rerank")):
        store, agents, chunk_doc, _ = build(*sizes["prod"])
        prod_rows = evaluate("prod", store, agents, chunk_doc, keep_rows=True)
        if "dense" in variants:
            evaluate("dense", store, agents, chunk_doc, mode="dense")
        if "bm25" in variants:
            evaluate("bm25", store, agents, chunk_doc, mode="bm25")
    for v in variants:
        out["settings"][f"chunks_{v}"] = {k: len(chunking.chunk_text(texts[k], *sizes.get(v, sizes["prod"])))
                                          for k in DOC_ORDER}
        if v in ("chunk80", "chunk200"):
            store, agents, chunk_doc, _ = build(*sizes[v])
            evaluate(v, store, agents, chunk_doc)
        elif v == "pooled":
            store, agents, chunk_doc, _ = build(*sizes["prod"], pooled=True)
            evaluate("pooled", store, agents, chunk_doc)
        elif v not in ("prod", "dense", "bm25", "rerank"):
            raise SystemExit(f"unknown variant {v}")
    out["timings"]["query_encode_ms_median"] = round(statistics.median(q_ms[1:] or q_ms), 1) if q_ms else None
    out["timings"]["query_encode_ms_p95"] = round(sorted(q_ms)[int(0.95 * (len(q_ms) - 1))], 1) if q_ms else None
    out["timings"]["rss_mb_with_embedder"] = round(rss_mb())

    if "rerank" in variants:
        run_rerank(args, out, questions, prod_rows)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def run_rerank(args, out: dict, questions: List[dict], cand_rows: Dict[str, List[dict]]) -> None:
    """The fused candidates of the prod variant, re-scored by the cross-encoder (retrieval.rerank with RERANKER=on).
    The embedding model is freed first: one model at a time."""
    from services import retrieval
    name = retrieval.reranker_model_name()
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = isinstance(try_to_load_from_cache(name, "config.json"), str)
    except Exception:  # noqa: BLE001
        cached = False
    if not cached:
        note = (f"rerank skipped: {name} is not downloaded on this machine (about 2.3 GB; this offline run does not "
                "download models). Earlier figure (retrieval_eval, 40-word chunks): about 945 ms per question for 20 "
                "candidates on this Mac's CPU.")
        print("  " + note)
        out["notes"].append(note)
        return
    retrieval._embedders.clear()
    retrieval._half_models.clear()
    gc.collect()
    free = available_gb()
    if free < args.min_free_gb:
        note = f"rerank skipped: only {free:.1f} GB available (needs {args.min_free_gb} GB)"
        print("  " + note)
        out["notes"].append(note)
        return
    os.environ["RERANKER"] = "on"
    t = time.perf_counter()
    retrieval._load_reranker()
    out["timings"]["reranker_load_s"] = round(time.perf_counter() - t, 2)
    rows, ms = [], []
    for q in questions:
        cands = cand_rows[q["id"]]
        t = time.perf_counter()
        reranked = retrieval.rerank(q["q"], cands, len(cands))
        ms.append((time.perf_counter() - t) * 1000)
        chunks = [r["chunk_text"] for r in reranked]
        r_all, r_any = ranks(chunks, list(range(len(chunks))), q["evidence"])
        rows.append({"id": q["id"], "lang": q["lang"], "type": q["type"], "doc": q["doc"], "rank_all": r_all,
                     "rank_any": r_any, "n_chunks": len(chunks), "top3": [c[:90] for c in chunks[:3]]})
    os.environ.pop("RERANKER", None)
    out["variants"]["rerank"] = {"metrics": breakdown(rows, retrieval_metrics), "rows": rows,
                                 "rerank_ms_median": round(statistics.median(ms[1:] or ms), 1),
                                 "rerank_ms_p95": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 1),
                                 "rss_mb": round(rss_mb())}
    m = out["variants"]["rerank"]["metrics"]["all"]
    print(f"  {'rerank':10s} hit@1 {m['hit@1']:.1%}  hit@3 {m['hit@3']:.1%}  hit@5 {m['hit@5']:.1%}  MRR {m['mrr']:.3f}"
          f"  ({out['variants']['rerank']['rerank_ms_median']} ms median)")


# ----------------------------------------------------------------------------------------------- pacing

class Pacer:
    """Keeps each Groq model under its tokens-per-minute budget: a sliding 60 s window of the tokens actually used."""

    def __init__(self, budget: int):
        self.budget = budget
        self.calls: Dict[str, List[Tuple[float, int]]] = collections.defaultdict(list)
        self.lock = threading.Lock()
        self.waited_s = 0.0

    def used(self, model: str) -> int:
        now = time.time()
        with self.lock:
            self.calls[model] = [(t, n) for t, n in self.calls[model] if now - t < 60]
            return sum(n for _, n in self.calls[model])

    def wait(self, model: str, estimate: int) -> float:
        started = time.time()
        while 0 < self.used(model) and self.used(model) + estimate > self.budget:   # a big call alone may go
            time.sleep(1.0)
        waited = time.time() - started
        self.waited_s += waited
        return waited

    def record(self, model: str, tokens: int) -> None:
        with self.lock:
            self.calls[model].append((time.time(), int(tokens or 0)))


# ----------------------------------------------------------------------------------------------- phase 2

def _isolate_env(tmp: str) -> None:
    """Environment for importing backend.py in this process without touching backend/data or the user's calendar.
    Set BEFORE the import: backend.py's load_dotenv never overrides a variable that already exists."""
    os.environ.update({
        "INSTANT_ANSWERS": "off",                    # nothing to match in a fresh store; also never reads the user's file
        "BONCLAY_INSTANT_ANSWERS_FILE": os.path.join(tmp, "instant_answers.json"),
        "LLM_PROVIDERS": "groq",                     # never start the local MLX model on a 16 GB machine
        "BONCLAY_APPOINTMENTS_FILE": os.path.join(tmp, "appointments.json"),
        "BONCLAY_MEETINGS_FILE": os.path.join(tmp, "meetings.json"),
        "BONCLAY_MCP_SERVERS": os.path.join(tmp, "servers.json"),
        "GOOGLE_CALENDAR_ID": "",                    # the meeting server must not read the real calendar
        "GOOGLE_SERVICE_ACCOUNT_FILE": os.path.join(tmp, "no_calendar.json"),
        "WARM_ON_START": "off",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MEMORY_GUARD": os.environ.get("MEMORY_GUARD", "off"),
    })


def import_backend(tmp: str):
    """backend.py imported with every data path it writes to moved under `tmp`."""
    _isolate_env(tmp)
    cwd = os.getcwd()
    os.chdir(BACKEND_DIR)
    try:
        import backend as B  # noqa: PLC0415
    finally:
        os.chdir(cwd)
    from services.local_store import LocalStore
    from services import provenance, tools as tools_mod
    from core import generations as generations_store, apikeys as apikeys_store
    B.LOCAL_DATA_DIR = tmp
    B.LOCAL_AUDIO_DIR = os.path.join(tmp, "audio")
    os.makedirs(B.LOCAL_AUDIO_DIR, exist_ok=True)
    B.local_db = LocalStore(tmp)
    provenance.configure(data_dir=tmp)
    generations_store.configure(lambda: None, tmp)
    apikeys_store.configure(lambda: None, tmp)
    errlog = os.path.join(tmp, "mcp_servers.log")
    tools_mod.hub._open_errlog = lambda: open(errlog, "a", encoding="utf-8")  # instead of backend/data/logs
    return B


def fake_request():
    from starlette.requests import Request
    return Request({"type": "http", "method": "POST", "path": "/doc-qa-eval", "query_string": b"",
                    "headers": [(b"storage-mode", b"local")], "client": ("127.0.0.1", 0)})


async def create_agents(B, spec: dict) -> Dict[str, dict]:
    """The four agents, made with the app's own template code in the temp store."""
    from fastapi import UploadFile
    from starlette.datastructures import Headers
    from routers import templates as T
    req = fake_request()
    agents = {}
    for key, d in spec["docs"].items():
        if d["kind"] == "template":
            tpl = T._find_template(d["template"])
            agents[key] = await T._create_from_template(tpl, req, EVAL_USER, None)
    tpl = T._find_template("clinic_receptionist")
    body = T.FromTemplateRequest(template_id="clinic_receptionist", org_name="Sahyadri Care Hospital", city="Pune",
                                 include_sample=False)
    agent = await T._create_from_template(tpl, req, EVAL_USER, None, body=body)
    pdf = REPO_ROOT / spec["docs"]["hospital"]["source"]
    upload = UploadFile(file=io.BytesIO(pdf.read_bytes()), filename=pdf.name,
                        headers=Headers({"content-type": "application/pdf"}))
    text = await B._extract_knowledge_text(upload)               # what POST /api/knowledge/upload does
    if text != extract_pdf_text(pdf):
        raise SystemExit("backend._extract_knowledge_text and this script's PDF extraction disagree")
    await B.ingest_document(agent["id"], EVAL_USER, "local", pdf.name, text)
    agents["hospital"] = agent
    return agents


class LlmMeter:
    """Wraps services.llm.chat (the function every reply, tool turn and language retry goes through) to record which
    model answered, the tokens and the latency of each call. It does not change what is sent."""

    def __init__(self, llm, pacer: Pacer):
        self.llm, self.pacer = llm, pacer
        self.original = llm.chat
        self.calls: List[dict] = []
        llm.chat = self.chat

    def chat(self, messages, **kw):
        t = time.time()
        try:
            res = self.original(messages, **kw)
        except Exception as e:
            self.calls.append({"model": None, "error": f"{type(e).__name__}: {str(e)[:200]}",
                               "ms": int((time.time() - t) * 1000), "tokens": 0})
            raise
        tokens = int((res.usage or {}).get("total_tokens") or 0)
        self.pacer.record(res.model, tokens)
        self.calls.append({"model": res.model, "provider": res.provider, "ms": res.latency_ms, "tokens": tokens,
                           "prompt_tokens": int((res.usage or {}).get("prompt_tokens") or 0),
                           "max_tokens": kw.get("max_tokens")})
        return res


def stub_speech(B) -> None:
    """Speech synthesis, the watermark and the audio file are not part of answering: replaced by no-ops."""
    import numpy as np
    from services import provenance

    async def spoken(language, text, fallback, *, use_llm=True, converted=None):
        return text, ("hi" if language == "hinglish" else language)

    class Plan:
        engine_key, used_voice, clone_required, voice_id = "kokoro", None, False, None

        def cleanup(self):
            pass

    async def voice(run, tts_language, fallback, text_len=0):
        return Plan()

    async def synth(plan, spoken_text):
        return np.zeros(2400, dtype=np.float32), 24000

    async def store(run, turn_id, audio_bytes, agent_text, **kw):
        return None, int((time.time() - run.start_time) * 1000)

    B._turn_spoken, B._turn_voice, B._turn_synth, B._turn_store = spoken, voice, synth, store
    provenance.apply_provenance = lambda **kw: (kw["wav_bytes"], {"receipt_id": None})


async def answer_one(B, meter: LlmMeter, agent: dict, q: dict) -> dict:
    """One typed turn through backend._turn_blocking (a new conversation each time, as a first caller turn)."""
    req = fake_request()
    run = await B._turn_begin(req, agent["id"], EVAL_USER)
    run.text_input = q["q"]
    n0 = len(meter.calls)
    t = time.time()
    res = await B._turn_blocking(run, b"", None)
    wall_ms = int((time.time() - t) * 1000)
    calls = meter.calls[n0:]
    disclosure = B.ai_disclosure(res["language"])
    reply = res["agent_text"]
    body = reply[len(disclosure):].strip() if reply.startswith(disclosure) else reply
    return {"agent_text": reply, "reply": body, "language_target": res["language"],
            "language_check": res.get("language_check"), "fallback_reply": res["fallback"],
            "llm_model": res.get("llm_model"), "llm_calls": calls, "timings": res["timings"], "wall_ms": wall_ms,
            "chunks": list(run.chunks), "retrieval": res.get("retrieval"),
            "tool_calls": res.get("tool_calls") or [],
            "messages": B._turn_messages(run)}


def run_answers(args, spec: dict, questions: List[dict], out_path: Path) -> Path:
    tmp = tempfile.mkdtemp(prefix="doc_qa_answers_")
    print(f"answers: temp store {tmp}; {available_gb():.1f} GB available")
    if available_gb() < args.min_free_gb:
        raise SystemExit(f"only {available_gb():.1f} GB of memory available; need {args.min_free_gb} GB")
    B = import_backend(tmp)
    from services import llm, language_policy  # noqa: F401
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set (backend/.env)")
    stub_speech(B)
    if args.answer_top_k != B.RETRIEVAL_TOP_K:
        B.retrieve_context.__defaults__ = (args.answer_top_k,)   # the default _turn_context relies on
    pacer = Pacer(args.tpm_budget)
    meter = LlmMeter(llm, pacer)
    texts = doc_texts(spec)
    check_gold(spec, texts)

    data = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "settings": {"answer_top_k": args.answer_top_k, "primary_model": os.getenv("GROQ_CHAT_MODEL", PRIMARY_MODEL),
                     "fallbacks": llm._groq_models()[1:], "tpm_budget": args.tpm_budget,
                     "retry_fallback": args.retry_fallback, "started": stamp(),
                     "code_path": "backend._turn_blocking (typed turn) with _turn_spoken/_turn_voice/_turn_synth/"
                                  "_turn_store/provenance.apply_provenance stubbed; agents from "
                                  "routers.templates._create_from_template; PDF via backend._extract_knowledge_text + "
                                  "backend.ingest_document; INSTANT_ANSWERS=off; LLM_PROVIDERS=groq"},
        "answers": {}}
    data.setdefault("runs", []).append({**RUN_INFO, "env": record_env()})
    loop = asyncio.new_event_loop()
    agents = loop.run_until_complete(create_agents(B, spec))
    data["agents"] = {k: {"name": a["name"], "tools": a["tools"], "languages": a["supported_languages"],
                          "system_instructions": a["system_instructions"]} for k, a in agents.items()}
    doc_counts = {k: len(B.local_db.select("documents", lambda r, a=a: r.get("agent_id") == a["id"]))
                  for k, a in agents.items()}
    print("agents:", {k: f"{a['name']} (tools {a['tools']}, {doc_counts[k]} docs)" for k, a in agents.items()})
    # Tokens a turn may need: the agent's prompt size seen so far plus the completion allowance (Groq counts
    # max_tokens against the per-minute budget when the request arrives), more for an agent with tools (a tool turn
    # can make a second call).
    est_doc: Dict[str, int] = {}
    primary = data["settings"]["primary_model"]
    pace_model = primary
    todo = [q for q in questions if q["id"] not in data["answers"]]
    for n, q in enumerate(todo, start=1):
        attempts = []
        for attempt in range(2 if args.retry_fallback else 1):
            est = est_doc.get(q["doc"], 3000)
            waited = pacer.wait(pace_model, est)     # the model that answered last (a fallback when the primary is out)
            try:
                a = loop.run_until_complete(answer_one(B, meter, agents[q["doc"]], q))
            except Exception as e:  # noqa: BLE001 - recorded, never stops the run
                a = {"error": f"{type(e).__name__}: {str(e)[:300]}", "llm_calls": [], "reply": "", "agent_text": ""}
            a["pacing_wait_s"] = round(waited, 1)
            attempts.append(a)
            models = {c.get("model") for c in a.get("llm_calls", [])}
            answered = [c.get("model") for c in a.get("llm_calls", []) if c.get("model")]
            if answered:
                pace_model = answered[-1]
            prompts = [c.get("prompt_tokens") or 0 for c in a.get("llm_calls", [])]
            if prompts:
                per_call = max(prompts) + 320
                est_doc[q["doc"]] = max(est_doc.get(q["doc"], 0), int(per_call * (1.5 if agents[q["doc"]]["tools"] else 1.0)))
            if a.get("llm_calls") and models <= {primary} and not a.get("fallback_reply") and not a.get("error"):
                break
            if attempt == 0 and args.retry_fallback:
                print(f"    {q['id']}: answered by {sorted(m or 'none' for m in models)}"
                      f"{' (fallback reply)' if a.get('fallback_reply') else ''}; pausing 65 s and asking again")
                time.sleep(65)
        final = attempts[-1]
        final["attempts"] = len(attempts)
        if len(attempts) > 1:
            final["first_attempt"] = {k: attempts[0].get(k) for k in ("reply", "llm_calls", "fallback_reply", "error")}
        data["answers"][q["id"]] = final
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        models = ",".join(sorted({c.get("model") or "-" for c in final.get("llm_calls", [])}))
        print(f"  [{n}/{len(todo)}] {q['id']:11s} {q['lang']:8s} {models:28s} "
              f"{final.get('timings', {}).get('llm_ms', '-')} ms  {final.get('reply', '')[:90]!r}")
    data["settings"]["finished"] = stamp()
    data["settings"]["pacing_wait_s"] = round(pacer.waited_s)
    data["settings"]["doc_counts"] = doc_counts
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        from services import tools as tools_mod
        tools_mod.hub.close()
    except Exception:  # noqa: BLE001
        pass
    loop.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return out_path


# ----------------------------------------------------------------------------------------------- phase 3

_DEVA_DIGITS = {0x0966 + i: ord("0") + i for i in range(10)}


def norm_numbers(text: str) -> str:
    """Devanagari digits to ASCII; separators inside a number dropped ("1,45,000" -> "145000", "020 5550 1299" ->
    "02055501299")."""
    t = (text or "").translate(_DEVA_DIGITS)
    return re.sub(r"(?<=\d)[,\s\-–](?=\d)", "", t)


def numbers_in(text: str) -> List[str]:
    return re.findall(r"\d+", norm_numbers(text))


def deterministic_checks(q: dict, a: dict, allowed_text: str) -> dict:
    from services import language_policy
    reply = a.get("reply") or ""
    nums = norm_numbers(reply)
    key = q.get("key_numbers") or []
    allowed = set(numbers_in(allowed_text)) | set(numbers_in(q["q"])) | set(q.get("derived_numbers") or [])
    # a long piece of an allowed number ("5550" of a phone number said in groups) also counts as allowed
    long_allowed = " ".join(n for n in allowed if len(n) >= 6)
    unknown = [n for n in numbers_in(reply) if n not in allowed and not (len(n) >= 4 and n in long_allowed)]
    tokens = numbers_in(reply)
    digits = re.sub(r"\D", "", nums)
    key_ok = all(k in tokens or (len(k) >= 6 and k in digits) for k in key) if key else None
    chunks = a.get("chunks") or []
    return {
        "language_ok": language_policy.reply_matches(q["lang"], reply),
        "app_target": a.get("language_target"),
        "app_target_ok": a.get("language_target") == q["lang"],
        "key_numbers_ok": key_ok,
        "ungrounded_numbers": unknown,
        "evidence_in_context": all(any(e in c for c in chunks) for e in q.get("evidence") or []) if q.get("evidence") else None,
        "evidence_any_in_context": any(e in c for c in chunks for e in q.get("evidence") or []) if q.get("evidence") else None,
    }


JUDGE_SYSTEM = """You grade replies of a phone receptionist AI that must answer ONLY from document passages it was given.
Be strict and literal. Output a single JSON object only."""

JUDGE_RUBRIC = """Grade EACH item with these rules.
- "correct":
  * ANSWERABLE question: "yes" if the reply states the key facts of the reference answer correctly. Numbers, prices,
    phone numbers, days, dates and times must match exactly (another format such as "4 PM"/"4:00 pm"/"चार बजे" or a
    number in words is fine if the VALUE is the same; a different value is not). For a distractor question the reply
    must correct the caller's wrong assumption. For a question needing two facts, both must be there. Extra correct
    details are fine. "partial" if it gives part of the key facts with nothing wrong; "no" if any key fact is wrong or
    missing, if it says it does not know, or if it only asks a question back instead of answering.
  * UNANSWERABLE question: "yes" if the reply says it does not have that information (or cannot help with it) and
    states no invented fact about it; offering to connect to staff is good but not required. "no" if it answers with
    anything not in the passages (an invented fact, price, name, yes/no).
- "unsupported_claims": every factual statement in the reply that the item's passages, tool results and the agent
  instructions do not support. Courtesy, an offer to connect to staff, a question back to the caller and saying the
  information is unavailable are NOT claims. Empty list if none.
- "grounded": true if "unsupported_claims" is empty.
- "declined": true if the reply says it does not have / cannot give the requested information.
- "offered_handoff": true if the reply offers to connect the caller with staff or says whom to contact.
- "language_ok": true if the reply is in the caller's language and script: English for English; Hindi in Devanagari
  script for Hindi; Hinglish = Hindi written in Roman (Latin) letters for Hinglish (English words mixed in are fine;
  a reply fully in standard English or in Devanagari is wrong). Names, numbers and technical terms are fine anywhere.
- "reason": one short sentence for the "correct" grade.

Return JSON: {"grades": [{"id": "<item id>", "correct": "yes"|"partial"|"no", "unsupported_claims": [..],
"grounded": true|false, "declined": true|false, "offered_handoff": true|false, "language_ok": true|false,
"reason": ".."}, ...]} with one entry per item, in order."""

LANG_NAMES = {"en": "English", "hi": "Hindi (Devanagari script)", "hinglish": "Hinglish (Hindi in Roman letters)"}
INSTRUCTIONS_CHARS = 700   # the persona part the judge sees (keeps a judge call small; facts live in the passages)


def judge_item(q: dict, a: dict) -> str:
    if q["type"] == "unanswerable":
        block = ("ANSWERABLE: no. The answer is NOT in the documents; expected: say it is not available (offering staff "
                 f"is good) and invent nothing. Test writer's note: {q['gold_answer']}")
    else:
        block = (f"ANSWERABLE: yes. REFERENCE ANSWER: {q['gold_answer']}\nDOCUMENT EVIDENCE: " + " | ".join(q["evidence"]))
    passages = "\n".join(f"[{i + 1}] {' '.join(c.split())}" for i, c in enumerate(a.get("chunks") or [])) or "(none)"
    tools = a.get("tool_calls") or []
    tools_txt = ("\nTOOL CALLS (results are an allowed source): " +
                 " ; ".join(json.dumps(t, ensure_ascii=False)[:400] for t in tools)) if tools else ""
    return (f"### ITEM {q['id']}\nCALLER LANGUAGE: {LANG_NAMES[q['lang']]} | QUESTION TYPE: {q['type']}\n"
            f"CALLER SAID: {q['q']}\n{block}\nPASSAGES THE AGENT WAS GIVEN:\n{passages}{tools_txt}\n"
            f"AGENT REPLY: {a.get('reply') or '(empty)'}\n")


def judge_prompt(items: List[Tuple[dict, dict]], agent: dict) -> str:
    instr = agent.get("system_instructions", "")
    if len(instr) > INSTRUCTIONS_CHARS:
        instr = instr[:INSTRUCTIONS_CHARS] + " [...]"
    return ("AGENT INSTRUCTIONS (persona; facts stated here, such as the organisation name or an emergency number, are "
            f"also an allowed source):\n{instr}\n\n" + "\n".join(judge_item(q, a) for q, a in items) + "\n" + JUDGE_RUBRIC)


def groq_json(model: str, system: str, user: str, pacer: Pacer, max_tokens: int = 1800) -> Tuple[dict, dict]:
    """One Groq call in JSON mode with polite 429 handling. Returns (parsed JSON, usage)."""
    import httpx
    key = os.environ["GROQ_API_KEY"]
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0, "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
        body["include_reasoning"] = False
    est = len(user) // 3 + max_tokens
    tpm_waits = 0
    for attempt in range(60):           # a shared key: other traffic can fill the per-minute budget for a while
        if attempt - tpm_waits >= 6:
            break
        pacer.wait(model, est)
        r = httpx.post(GROQ_URL, json=body, timeout=90,
                       headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT})
        if r.status_code == 429:
            msg = (r.json().get("error") or {}).get("message", "") if r.headers.get("content-type", "").startswith(
                "application/json") else ""
            if "per day" in msg:
                raise RuntimeError("daily token limit reached: " + msg[:200])
            wait = float(r.headers.get("retry-after") or 20)
            tpm_waits += 1
            time.sleep(min(90, wait + 3 + (attempt % 5)))
            continue
        if r.status_code >= 500:
            time.sleep(10)
            continue
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        pacer.record(model, usage.get("total_tokens") or est)
        text = data["choices"][0]["message"].get("content") or ""
        try:
            return json.loads(text), usage
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                return json.loads(m.group(0)), usage
            if attempt < 5:
                continue
            raise
    raise RuntimeError(f"judge {model} kept failing")


def run_judge(args, spec: dict, path: Path) -> None:
    load_env()
    data = json.loads(path.read_text(encoding="utf-8"))
    texts = doc_texts(spec)
    qmap = {q["id"]: q for q in spec["questions"]}
    pacer = Pacer(args.tpm_budget)
    data.setdefault("judge", {"model": args.judge_model, "usage_tokens": 0, "calls": 0, "batch": args.judge_batch})
    todo = [i for i, a in data["answers"].items() if "judge" not in a or args.rejudge]
    for qid in todo:   # deterministic checks first (no model)
        q, a = qmap[qid], data["answers"][qid]
        agent = data["agents"][q["doc"]]
        allowed = texts[q["doc"]] + "\n" + agent.get("system_instructions", "") + "\n" + \
            json.dumps(a.get("tool_calls") or [], ensure_ascii=False) + "\n" + (" ".join(a.get("chunks") or []))
        a["checks"] = deterministic_checks(q, a, allowed)
    batches: List[List[str]] = []
    for doc in DOC_ORDER:
        ids = [i for i in todo if qmap[i]["doc"] == doc]
        batches += [ids[j:j + args.judge_batch] for j in range(0, len(ids), args.judge_batch)]
    print(f"judging {len(todo)} answers with {args.judge_model} in {len(batches)} calls")
    for n, ids in enumerate(batches, start=1):
        items = []
        for qid in ids:
            a = data["answers"][qid]
            if a.get("error") or not a.get("reply"):
                a["judge"] = {"correct": "no", "grounded": True, "unsupported_claims": [], "declined": False,
                              "offered_handoff": False, "language_ok": False, "reason": "no reply: " + str(a.get("error"))}
            else:
                items.append((qmap[qid], a))
        if not items:
            continue
        answered_by = {c.get("model") for _, a in items for c in a.get("llm_calls", [])}
        model = args.judge_model if args.judge_model not in answered_by else "openai/gpt-oss-20b"
        agent = data["agents"][items[0][0]["doc"]]
        try:
            verdict, usage = groq_json(model, JUDGE_SYSTEM, judge_prompt(items, agent), pacer)
        except Exception as e:  # noqa: BLE001
            print(f"    judge failed for {ids}: {e}")
            if "daily token limit" in str(e):
                break
            continue
        grades = {g.get("id"): g for g in verdict.get("grades") or [] if isinstance(g, dict)}
        if len(grades) != len(items):   # ids not echoed exactly: fall back to the order
            grades = {q["id"]: g for (q, _), g in zip(items, verdict.get("grades") or [])}
        for q, a in items:
            g = grades.get(q["id"])
            if g:
                g["judge_model"] = model
                a["judge"] = g
        data["judge"]["usage_tokens"] += int(usage.get("total_tokens") or 0)
        data["judge"]["calls"] += 1
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        for q, a in items:
            j, c = a.get("judge") or {}, a["checks"]
            print(f"  [{n}/{len(batches)}] {q['id']:11s} correct={str(j.get('correct')):7s} grounded={j.get('grounded')} "
                  f"lang={j.get('language_ok')}/{c['language_ok']}  {str(j.get('reason', ''))[:80]}")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- summary

def answer_rows(spec: dict, data: dict) -> List[dict]:
    qmap = {q["id"]: q for q in spec["questions"]}
    rows = []
    for qid, a in data["answers"].items():
        q = qmap[qid]
        j, c = a.get("judge") or {}, a.get("checks") or {}
        calls = a.get("llm_calls") or []
        hand = (data.get("hand_check") or {}).get(qid) or {}
        correct = hand.get("correct", j.get("correct"))
        grounded = hand.get("grounded", j.get("grounded"))
        rows.append({
            "id": qid, "lang": q["lang"], "type": q["type"], "doc": q["doc"], "q": q["q"],
            "reply": a.get("reply"), "correct": correct, "judge_correct": j.get("correct"),
            "grounded": grounded, "ungrounded_numbers": c.get("ungrounded_numbers") or [],
            "hallucinated": (grounded is False) or bool(c.get("ungrounded_numbers")),
            "language_ok": c.get("language_ok"), "judge_language_ok": j.get("language_ok"),
            "app_target": c.get("app_target"), "declined": j.get("declined"), "handoff": j.get("offered_handoff"),
            "evidence_in_context": c.get("evidence_in_context"), "key_numbers_ok": c.get("key_numbers_ok"),
            "fallback_model": any((x.get("model") or "") != data["settings"]["primary_model"] for x in calls),
            "fallback_reply": a.get("fallback_reply"), "attempts": a.get("attempts", 1),
            "first_attempt_fallback": bool(a.get("first_attempt")),
            "llm_ms": sum(x.get("ms") or 0 for x in calls), "retrieval_ms": (a.get("timings") or {}).get("retrieval_ms"),
            "turn_ms": ((a.get("timings") or {}).get("retrieval_ms") or 0) + sum(x.get("ms") or 0 for x in calls),
            "n_llm_calls": len(calls), "tokens": sum(x.get("tokens") or 0 for x in calls),
            "tool_calls": [t.get("name") for t in a.get("tool_calls") or []],
            "lang_retried": bool((a.get("language_check") or {}).get("retried")),
            "judge_reason": j.get("reason"), "unsupported_claims": j.get("unsupported_claims"),
        })
    return rows


def answer_metrics(rows: List[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    lat = sorted(r["turn_ms"] for r in rows)
    return {
        "n": n,
        "correct": sum(r["correct"] == "yes" for r in rows) / n,
        "correct_or_partial": sum(r["correct"] in ("yes", "partial") for r in rows) / n,
        "hallucination": sum(bool(r["hallucinated"]) for r in rows) / n,
        "language_ok": sum(bool(r["language_ok"]) for r in rows) / n,
        "answerable_correct": (sum(r["correct"] == "yes" for r in ans) / len(ans)) if ans else None,
        "unanswerable_handled": (sum(r["correct"] == "yes" for r in una) / len(una)) if una else None,
        "turn_ms_median": statistics.median(lat),
        "turn_ms_p90": lat[int(0.9 * (n - 1))],
    }


def summarise(spec: dict, path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = answer_rows(spec, data)
    out = breakdown(rows, answer_metrics)
    calls = [c for a in data["answers"].values() for c in a.get("llm_calls") or []]
    calls += [c for a in data["answers"].values() for c in ((a.get("first_attempt") or {}).get("llm_calls") or [])]
    models = collections.Counter(c.get("model") for c in calls)
    print(f"\n{path.name}: {len(rows)} answers (top-k {data['settings']['answer_top_k']})")
    print(f"{'slice':22s} {'n':>4s} {'correct':>8s} {'c+part':>7s} {'halluc':>7s} {'lang ok':>8s} {'unans ok':>9s} {'p50 ms':>7s}")
    for k, m in out.items():
        if not m.get("n"):
            continue
        ua = "" if m["unanswerable_handled"] is None else f"{m['unanswerable_handled']:.0%}"
        print(f"{k:22s} {m['n']:4d} {m['correct']:8.1%} {m['correct_or_partial']:7.1%} {m['hallucination']:7.1%} "
              f"{m['language_ok']:8.1%} {ua:>9s} {m['turn_ms_median']:7.0f}")
    print("LLM calls by model:", dict(models), " tokens:", sum(c.get("tokens") or 0 for c in calls),
          " judge:", data.get("judge"))
    hand = data.get("hand_check") or {}
    if hand:
        pairs = [(data["answers"][i].get("judge") or {}, h) for i, h in hand.items()
                 if i in data["answers"] and (data["answers"][i].get("judge") or {}).get("correct")]
        if pairs:
            same3 = sum(j.get("correct") == h.get("correct") for j, h in pairs)
            same2 = sum((j.get("correct") == "yes") == (h.get("correct") == "yes") for j, h in pairs)
            sameg = sum(bool(j.get("grounded")) == bool(h.get("grounded")) for j, h in pairs)
            print(f"judge vs hand on {len(pairs)} answers: correct (yes/partial/no) {same3}/{len(pairs)}, "
                  f"correct yes vs not {same2}/{len(pairs)}, grounded {sameg}/{len(pairs)}")
    print("final answers with a fallback model:", sum(r["fallback_model"] for r in rows),
          " first attempts that fell back:", sum(r["first_attempt_fallback"] for r in rows),
          " fallback replies:", sum(bool(r["fallback_reply"]) for r in rows))
    return {"metrics": out, "rows": rows, "models": dict(models)}


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--answers-only", action="store_true", help="phase 2 (and the judge, unless --no-judge)")
    ap.add_argument("--judge-only", metavar="FILE", help="judge (or re-judge with --rejudge) a saved answers file")
    ap.add_argument("--summary", metavar="FILE", help="print the tables of a judged answers file")
    ap.add_argument("--resume", metavar="FILE", help="continue an interrupted answers run")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--rejudge", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only N questions, spread over the documents")
    ap.add_argument("--ids", help="comma list of question ids")
    ap.add_argument("--docs", help="comma list of: hospital, clinic, college, hod")
    ap.add_argument("--langs", help="comma list of: en, hi, hinglish")
    ap.add_argument("--variants", default="prod,dense,bm25,chunk80,chunk200,pooled,rerank",
                    help="retrieval variants (see the docstring)")
    ap.add_argument("--answer-top-k", type=int, default=3, help="passages per turn in phase 2 (production: 3)")
    ap.add_argument("--tpm-budget", type=int, default=6000, help="tokens per minute per Groq model (free tier: 8000)")
    ap.add_argument("--retry-fallback", action=argparse.BooleanOptionalAction, default=True,
                    help="ask again once (after a pause) when a fallback model wrote the reply")
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    ap.add_argument("--judge-batch", type=int, default=3, help="answers of one document graded per judge call")
    ap.add_argument("--min-free-gb", type=float, default=2.5, help="refuse to load a model with less memory available")
    ap.add_argument("--baseline-pins", action=argparse.BooleanOptionalAction, default=True,
                    help="pin the newer retrieval/answer switches to the original behaviour (see BASELINE_PINS)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="extra environment setting (repeatable)")
    ap.add_argument("--env-file", help="the backend .env to read GROQ_API_KEY etc. from (default backend/.env)")
    ap.add_argument("--results-dir", help="where results go (default backend/bench/results)")
    ap.add_argument("--code-label", help="a note saved with the results about which code ran (e.g. a commit export)")
    ap.add_argument("--label", default="", help="added to the result file names")
    args = ap.parse_args()
    if args.env_file:
        os.environ["DOC_QA_ENV_FILE"] = args.env_file
    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
    apply_settings(args)

    spec = load_spec()
    RESULTS_DIR.mkdir(exist_ok=True)
    if args.summary:
        summarise(spec, Path(args.summary))
        return
    if args.judge_only:
        run_judge(args, spec, Path(args.judge_only))
        summarise(spec, Path(args.judge_only))
        return
    load_env()
    questions = select(spec["questions"], args)
    do_retrieval = not args.answers_only
    do_answers = not args.retrieval_only
    if do_retrieval:
        res = run_retrieval(args, spec, questions)
        path = RESULTS_DIR / f"doc_qa_retrieval-{stamp()}{args.label}.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {path}")
        if do_answers:
            from services import retrieval
            retrieval._embedders.clear()
            gc.collect()
    if do_answers:
        path = Path(args.resume) if args.resume else RESULTS_DIR / f"doc_qa_answers-{stamp()}-k{args.answer_top_k}{args.label}.json"
        run_answers(args, spec, questions, path)
        print(f"wrote {path}")
        if not args.no_judge:
            run_judge(args, spec, path)
            summarise(spec, path)


if __name__ == "__main__":
    main()
