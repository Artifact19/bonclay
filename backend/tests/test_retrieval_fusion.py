"""Retrieval ranking (services/retrieval.py): weighted score fusion maths and the RRF switch, question-index rows that
stand for their parent passage (contract 3), passage headers (contract 4), the confidence rating (contract 2), the
gated reranker (RERANKER=auto) and the backend hooks (retrieve_context, the hosted path with a fake Supabase client,
the per-turn retrieval metadata). No model is loaded."""
import asyncio

import numpy as np
import pytest

from services import retrieval
from services.local_store import LocalStore
from tests.test_offline_agents import LOCAL, env, make_agent, upload_txt  # noqa: F401 - `env` is a fixture
from tests.test_retrieval import Dim, FakeSupabase, _cloud_env


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------------------------------------ fusion maths

def test_minmax_and_tmm():
    assert retrieval.minmax([2.0, 4.0, None, 3.0]) == [0.0, 1.0, 0.0, 0.5]
    assert retrieval.minmax([0.5, 0.5]) == [1.0, 1.0] and retrieval.minmax([None, None]) == [0.0, 0.0]
    assert retrieval.tmm([0.6, 0.3, None]) == [1.0, 0.5, 0.0]
    assert retrieval.tmm([-0.2, 0.4]) == [0.0, 1.0]                     # below the floor clips to 0
    assert retrieval.tmm([0.0, 0.0]) == [0.0, 0.0]


def test_weighted_fuse_is_a_convex_combination(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_NORM", "minmax")
    sig = {"dense": [0.9, 0.5, 0.7], "bm25": [0.0, 6.0, 3.0]}
    out = retrieval.weighted_fuse(sig, {"dense": 0.75, "bm25": 0.25})
    assert out == pytest.approx([0.75, 0.25, 0.75 * 0.5 + 0.25 * 0.5])
    assert all(0.0 <= v <= 1.0 for v in out)
    # weights are renormalised over the signals that are present
    assert retrieval.weighted_fuse({"dense": [0.9, 0.5], "colbert": [None, None]}, {"dense": 0.4, "colbert": 0.4}) \
        == pytest.approx([1.0, 0.0])
    assert retrieval.weighted_fuse({"dense": [0.1]}, {"dense": 0.0}) == [0.0]


def test_weighted_fuse_default_norm_is_tmm(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_NORM", raising=False)
    out = retrieval.weighted_fuse({"dense": [0.6, 0.57], "bm25": [0.0, 2.0]}, {"dense": 0.8, "bm25": 0.2})
    # tmm: dense 1.0 / 0.95, bm25 0 / 1 -> 0.8 vs 0.76 + 0.2 = 0.96: a close passage with the exact word wins
    assert out == pytest.approx([0.8, 0.8 * 0.95 + 0.2])


def test_fusion_settings_from_env(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_FUSION", raising=False)
    monkeypatch.delenv("RETRIEVAL_WEIGHTS", raising=False)
    assert retrieval.fusion_mode() == "weighted" and retrieval.fusion_weights() == retrieval.DEFAULT_FUSION_WEIGHTS
    d = retrieval.DEFAULT_FUSION_WEIGHTS
    # bench/retrieval_modes_bench.py grid, 19 Oct 2026: dense 0.4, multi-vector 0.3, BM25 0.1, sparse off
    assert d == {"dense": 0.4, "sparse": 0.0, "colbert": 0.3, "bm25": 0.1}
    assert d["dense"] / (d["dense"] + d["bm25"]) == pytest.approx(0.8)                      # without the M3 heads
    monkeypatch.setenv("RETRIEVAL_FUSION", "RRF")
    monkeypatch.setenv("RETRIEVAL_WEIGHTS", "dense:0.5, bm25:0.5, colbert:x, bogus:3")
    w = retrieval.fusion_weights()
    assert retrieval.fusion_mode() == "rrf" and w["dense"] == 0.5 and w["bm25"] == 0.5 and w["colbert"] == d["colbert"]
    monkeypatch.setenv("RETRIEVAL_FUSION", "nonsense")
    assert retrieval.fusion_mode() == "weighted"


def _rows(*items):
    """(text, sim) or dicts -> rows, sims"""
    rows, sims = [], []
    for i, it in enumerate(items):
        if isinstance(it, dict):
            rows.append({"id": f"r{i}", **{k: v for k, v in it.items() if k != "sim"}})
            sims.append(it.get("sim"))
        else:
            rows.append({"id": f"r{i}", "chunk_text": it[0]})
            sims.append(it[1])
    return rows, sims


def test_rank_rows_candidates_are_the_union_of_dense_and_bm25_tops(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "2")
    rows, sims = _rows(("alpha", 0.9), ("beta", 0.8), ("gamma", 0.7), ("delta", 0.6), ("the zebra", 0.1))
    info: dict = {}
    ranked = retrieval.rank_rows("zebra", rows, sims, info=info)
    assert info["n_candidates"] == 3 and info["fusion"] == "weighted" and info["signals"] == ["dense", "bm25"]
    assert [e["index"] for e in ranked][:2] == [0, 1]
    assert {e["index"] for e in ranked} == {0, 1, 2, 3, 4}                    # the rest follows (full ranking)
    assert all(e["score"] == 0.0 for e in ranked if e["index"] in (2, 3))
    assert info["dense_top"] == 0 and info["bm25_top"] == 4 and info["agree"] is False


def test_rrf_mode_reproduces_the_previous_order(monkeypatch):
    rows, sims = _rows(("alpha beta", 0.9), ("gamma delta", 0.8), ("the unique zebra word", 0.1), ("epsilon", 0.5))
    monkeypatch.setenv("RETRIEVAL_FUSION", "rrf")
    order = [e["index"] for e in retrieval.rank_rows("zebra", rows, sims)]
    old = retrieval.rrf_fuse([retrieval.dense_ranking(sims), retrieval.bm25_ranking("zebra", [r["chunk_text"] for r in rows])])
    assert order == old
    scores = [e["score"] for e in retrieval.rank_rows("zebra", rows, sims)]
    assert all(0 <= s <= 1 for s in scores) and scores == sorted(scores, reverse=True)


def test_hybrid_off_is_dense_only(monkeypatch):
    monkeypatch.setenv("HYBRID_SEARCH", "off")
    rows, sims = _rows(("refund", 0.2), ("other", 0.9), ("no vector refund", None))
    ranked = retrieval.rank_rows("refund", rows, sims)
    assert [e["index"] for e in ranked] == [1, 0] and ranked[0]["score"] == pytest.approx(0.9)


def test_rank_rows_empty_and_no_query():
    assert retrieval.rank_rows("q", [], []) == []
    rows, sims = _rows(("a", 0.2), ("b", 0.7))
    assert [e["index"] for e in retrieval.rank_rows("", rows, sims)] == [1, 0]


# ------------------------------------------------------------------------------------------------ question rows + headers

def test_question_row_hit_returns_its_parent_passage_once():
    rows, sims = _rows(
        {"chunk_text": "Orthopaedics OPD: Monday, Wednesday, Friday, 11 am to 3 pm.", "kind": "passage", "sim": 0.40},
        {"chunk_text": "Cardiology OPD: Tuesday and Thursday.", "kind": "passage", "sim": 0.45},
        {"chunk_text": "When does the bone doctor sit?", "kind": "question", "parent_chunk_id": "r0", "sim": 0.92},
        {"chunk_text": "haddi ke doctor kab baithte hain?", "kind": "question", "parent_chunk_id": "r0", "sim": 0.80},
        {"chunk_text": "orphan question about bones", "kind": "question", "parent_chunk_id": "gone", "sim": 0.99},
    )
    ranked = retrieval.rank_rows("bone doctor kab baithte", rows, sims)
    assert [e["index"] for e in ranked] == [0, 1]                         # parents only, deduplicated, orphan dropped
    assert ranked[0]["via"] == "question" and ranked[0]["similarity"] == pytest.approx(0.92)
    assert ranked[1]["via"] == "passage"


def test_question_rows_never_returned_by_the_local_store(tmp_path):
    db = LocalStore(str(tmp_path))
    doc = db.insert("documents", {"agent_id": "a1", "user_id": "u", "filename": "f"})
    p = db.insert("document_chunks", {"document_id": doc["id"], "chunk_text": "Mon-Sat 9 am to 5 pm", "kind": "passage",
                                      "header": "Sahyadri Hospital > Cardiology OPD", "chunk_index": 0,
                                      "embedding": [0.0, 1.0]})
    db.insert("document_chunks", {"document_id": doc["id"], "chunk_text": "Pharmacy open 24 hours", "kind": "passage",
                                  "chunk_index": 1, "embedding": [0.6, 0.8]})
    db.insert("document_chunks", {"document_id": doc["id"], "chunk_text": "When is the heart OPD?", "kind": "question",
                                  "parent_chunk_id": p["id"], "chunk_index": 0, "embedding": [1.0, 0.0]})
    info: dict = {}
    out = db.match_chunks("a1", [1.0, 0.0], 5, "heart OPD timings", info=info)
    assert [r["id"] for r in out][0] == p["id"] and len(out) == 2
    assert all(r.get("kind") is None for r in out) and {r["chunk_text"] for r in out} == {"Mon-Sat 9 am to 5 pm",
                                                                                         "Pharmacy open 24 hours"}
    assert out[0]["header"] == "Sahyadri Hospital > Cardiology OPD" and out[0]["via"] == "question"
    assert out[0]["similarity"] == pytest.approx(1.0) and 0 <= out[0]["score"] <= 1
    assert db.match_chunks("a1", [1.0, 0.0], 0, "x") == []
    dense_only = db.match_chunks("a1", [1.0, 0.0], 5)
    assert [r["id"] for r in dense_only][0] == p["id"] and len(dense_only) == 2


def test_header_words_are_searched():
    rows, sims = _rows({"chunk_text": "Mon-Sat 9 am to 5 pm", "header": "Hospital > Cardiology OPD", "sim": 0.5},
                       {"chunk_text": "Mon-Sat 10 am to 2 pm", "header": "Hospital > Dental OPD", "sim": 0.5})
    assert retrieval.rank_rows("cardiology timings", rows, sims)[0]["index"] == 0
    assert retrieval.rank_rows("dentist kab", rows, sims)[0]["index"] == 1     # "dental" and "dentist" share a key
    assert retrieval.passage_text(rows[0]) == "Hospital > Cardiology OPD\nMon-Sat 9 am to 5 pm"
    assert retrieval.passage_text({"chunk_text": "x", "header": "  "}) == "x"
    assert retrieval.passage_text({"content": "y"}) == "y"


# ------------------------------------------------------------------------------------------------ confidence

def _ranked(*entries):
    return [{"chunk_text": t, "score": s, "similarity": d} for t, s, d in entries]


def test_confidence_high_when_dense_is_strong_and_signals_agree():
    rows = _ranked(("Sunday: 10:00 AM to 1:00 PM. Clinic hours.", 1.0, 0.71), ("Fees", 0.6, 0.5))
    c = retrieval.assess_confidence(rows, "clinic hours on Sunday", {"agree": True})
    assert c["level"] == "high" and c["top"] == 1.0 and c["gap"] == pytest.approx(0.4)
    assert c["overlap"] == 1.0 and c["agree"] is True and set(c) >= {"level", "top", "gap", "reason"}


def test_confidence_medium_for_an_ordinary_match():
    rows = _ranked(("Dentist: ₹300 consultation", 0.8, 0.5), ("General physician ₹400", 0.75, 0.48))
    assert retrieval.assess_confidence(rows, "teeth doctor fees", {"agree": False})["level"] == "medium"


def test_confidence_low_when_far_in_meaning_and_no_shared_words():
    rows = _ranked(("Parking is free for patients", 0.9, 0.36), ("Visiting hours 4 to 7", 0.85, 0.35))
    c = retrieval.assess_confidence(rows, "do you sell gold jewellery", {"agree": False})
    assert c["level"] == "low" and "dense" in c["reason"]


def test_confidence_none_without_passages_or_any_match():
    assert retrieval.assess_confidence([], "anything")["level"] == "none"
    rows = _ranked(("Parking is free", 0.5, 0.2))
    assert retrieval.assess_confidence(rows, "bitcoin price", {"agree": None})["level"] == "none"
    assert retrieval.assess_confidence(rows, "bitcoin price", {"agree": False})["level"] == "low"


def test_confidence_uses_the_reranker_when_it_ran(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_CONF_RERANK_LOW", raising=False)
    rows = [{"chunk_text": "Dentist ₹300", "score": 0.9, "similarity": 0.62, "rerank_score": 0.01}]
    assert retrieval.assess_confidence(rows, "dentist fee")["level"] != "low"   # a low reranker score alone: off
    monkeypatch.setenv("RETRIEVAL_CONF_RERANK_LOW", "0.05")                     # opt in
    assert retrieval.assess_confidence(rows, "dentist fee")["level"] == "low"
    monkeypatch.delenv("RETRIEVAL_CONF_RERANK_LOW")
    rows[0]["rerank_score"] = 0.95
    c = retrieval.assess_confidence(rows, "teeth price")
    assert c["level"] == "high" and c["rerank"] == 0.95


def test_confidence_accepts_strings_and_env_overrides(monkeypatch):
    assert retrieval.assess_confidence(["The clinic opens at 9"], "clinic opens")["level"] == "high"
    assert retrieval.assess_confidence(["Unrelated"], "clinic opens")["level"] == "low"
    rows = _ranked(("Sunday hours", 1.0, 0.59), ("x", 0.5, 0.4))
    assert retrieval.assess_confidence(rows, "sunday hours", {"agree": True})["level"] == "high"
    monkeypatch.setenv("RETRIEVAL_CONF_DENSE_HIGH", "0.7")
    assert retrieval.assess_confidence(rows, "sunday hours", {"agree": True})["level"] == "medium"
    assert retrieval.confidence_thresholds()["conf_dense_high"] == 0.7


def test_confidence_never_raises():
    assert retrieval.assess_confidence([{"score": "bad", "similarity": object()}], None)["level"] in (
        "high", "medium", "low", "none")


# ------------------------------------------------------------------------------------------------ gated reranker

def _cand(*scores):
    return [{"chunk_text": f"passage {i}", "score": s, "similarity": 0.5} for i, s in enumerate(scores)]


@pytest.fixture()
def reranker_state(monkeypatch):
    monkeypatch.setattr(retrieval, "_reranker_failed", None)
    monkeypatch.setattr(retrieval, "_reranker", None)
    monkeypatch.setattr(retrieval, "_reranker_loading", False)
    on_disk = retrieval._reranker_on_disk
    on_disk.cache_clear()
    yield
    on_disk.cache_clear()


def test_reranker_mode_parsing(monkeypatch):
    monkeypatch.delenv("RERANKER", raising=False)
    assert retrieval.reranker_mode() == "auto" and not retrieval.reranker_enabled()
    for v, m in (("on", "on"), ("1", "on"), ("off", "off"), ("no", "off"), ("auto", "auto"), ("??", "auto")):
        monkeypatch.setenv("RERANKER", v)
        assert retrieval.reranker_mode() == m


def test_rerank_gate():
    assert retrieval.rerank_gate(_cand(1.0), {"level": "medium"})[0] is False             # one passage
    assert retrieval.rerank_gate(_cand(1.0, 0.95), {"level": "high"})[0] is False
    assert retrieval.rerank_gate(_cand(1.0, 0.5), {"level": "none"})[0] is False
    assert retrieval.rerank_gate(_cand(1.0, 0.5), {"level": "medium"})[0] is True
    assert retrieval.rerank_gate(_cand(1.0, 0.95), {"level": "low"})[0] is True             # small gap
    assert retrieval.rerank_gate(_cand(1.0, 0.5), {"level": "low"})[0] is False             # clear winner


def test_auto_skips_silently_when_the_model_is_not_downloaded(monkeypatch, reranker_state):
    monkeypatch.setenv("RERANKER", "auto")
    monkeypatch.setattr(retrieval, "_reranker_on_disk", lambda name: False)
    monkeypatch.setattr(retrieval, "rerank_scores", lambda *a: pytest.fail("must not run"))
    rows, info = retrieval.maybe_rerank("q", _cand(1.0, 0.98, 0.5), 2, {"level": "medium"})
    assert [r["chunk_text"] for r in rows] == ["passage 0", "passage 1"]
    assert info == {"mode": "auto", "ran": False, "why": "model not downloaded"}
    assert retrieval.candidate_count(3) == 3


def test_auto_loads_in_the_background_first_and_respects_memory(monkeypatch, reranker_state):
    monkeypatch.setenv("RERANKER", "auto")
    monkeypatch.setattr(retrieval, "_reranker_on_disk", lambda name: True)
    started = []
    monkeypatch.setattr(retrieval.threading, "Thread", lambda target, **k: type(
        "T", (), {"start": lambda self: started.append(target)})())
    monkeypatch.setattr(retrieval, "_memory_allows_reranker", lambda: False)
    rows, info = retrieval.maybe_rerank("q", _cand(1.0, 0.98), 2, {"level": "medium"})
    assert info["ran"] is False and info["why"] == "not loaded (memory)" and not started
    monkeypatch.setattr(retrieval, "_memory_allows_reranker", lambda: True)
    rows, info = retrieval.maybe_rerank("q", _cand(1.0, 0.98), 2, {"level": "medium"})
    assert info["why"] == "loading in the background" and len(started) == 1
    assert retrieval.candidate_count(3) == retrieval.RERANK_AUTO_CANDIDATES


def test_auto_reranks_at_most_rerank_auto_candidates_truncated_passages(monkeypatch, reranker_state):
    monkeypatch.setenv("RERANKER", "auto")
    monkeypatch.setattr(retrieval, "_reranker_on_disk", lambda name: True)
    monkeypatch.setattr(retrieval, "_reranker", object())
    seen = {}

    def scores(q, texts):
        seen["texts"] = texts
        return [0.1 * i for i in range(len(texts))]                  # the last of the head wins

    monkeypatch.setattr(retrieval, "rerank_scores", scores)
    rows = [{"chunk_text": " ".join(["word"] * 400), "score": 1.0 - 0.01 * i, "similarity": 0.5, "id": i}
            for i in range(12)]
    assert retrieval.RERANK_AUTO_CANDIDATES == 3
    out, info = retrieval.maybe_rerank("q", rows, 3, {"level": "medium"})
    assert len(seen["texts"]) == 3 and all(len(t.split()) == retrieval.RERANK_AUTO_WORDS for t in seen["texts"])
    assert [r["id"] for r in out] == [2, 1, 0] and out[0]["rerank_score"] == pytest.approx(0.2)
    assert info["ran"] is True and info["n"] == 3 and info["ms"] >= 0
    monkeypatch.setenv("RERANK_AUTO_CANDIDATES", "8")
    out, info = retrieval.maybe_rerank("q", rows, 3, {"level": "medium"})
    assert len(seen["texts"]) == 8 and [r["id"] for r in out] == [7, 6, 5] and info["n"] == 8
    monkeypatch.delenv("RERANK_AUTO_CANDIDATES")
    out, info = retrieval.maybe_rerank("q", rows, 3, {"level": "high"})        # a sure search is left alone
    assert info["ran"] is False and [r["id"] for r in out] == [0, 1, 2]


def test_auto_failure_falls_back(monkeypatch, reranker_state):
    monkeypatch.setenv("RERANKER", "auto")
    monkeypatch.setattr(retrieval, "_reranker_on_disk", lambda name: True)
    monkeypatch.setattr(retrieval, "_reranker", object())

    def boom(q, texts):
        raise RuntimeError("no")

    monkeypatch.setattr(retrieval, "rerank_scores", boom)
    out, info = retrieval.maybe_rerank("q", _cand(1.0, 0.99, 0.5), 2, {"level": "medium"})
    assert info["why"] == "failed" and [r["chunk_text"] for r in out] == ["passage 0", "passage 1"]
    assert retrieval.reranker_available() is False


def test_off_and_on_modes(monkeypatch, reranker_state):
    monkeypatch.setenv("RERANKER", "off")
    out, info = retrieval.maybe_rerank("q", _cand(1.0, 0.99, 0.5), 2, {"level": "medium"})
    assert info == {"mode": "off", "ran": False} and len(out) == 2
    monkeypatch.setenv("RERANKER", "on")
    monkeypatch.setattr(retrieval, "rerank_scores", lambda q, texts: [0.1, 0.9, 0.5])
    out, info = retrieval.maybe_rerank("q", _cand(1.0, 0.99, 0.5), 2, None)
    assert [r["chunk_text"] for r in out] == ["passage 1", "passage 2"] and info["ran"] is True


def test_reranker_on_disk_check_never_downloads(monkeypatch):
    retrieval._reranker_on_disk.cache_clear()
    calls = []
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, f: calls.append((repo, f)) or None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("must not download"))
    assert retrieval._reranker_on_disk("someone/not-there") is False and calls
    retrieval._reranker_on_disk.cache_clear()


# ------------------------------------------------------------------------------------------------ backend hooks

def test_retrieve_context_returns_header_and_text_and_fills_confidence(app_client, env, monkeypatch):
    monkeypatch.setattr(env, "embedding_model", Dim(16))
    a = make_agent(app_client)
    doc = env.local_db.insert("documents", {"agent_id": a["id"], "user_id": "local-user", "filename": "f.txt"})
    p = env.local_db.insert("document_chunks", {
        "document_id": doc["id"], "chunk_text": "The refund window is thirty days.", "header": "Policy > Refunds",
        "kind": "passage", "chunk_index": 0})
    env.local_db.insert("document_chunks", {"document_id": doc["id"], "chunk_text": "How long do I have for a refund?",
                                            "kind": "question", "parent_chunk_id": p["id"], "chunk_index": 0})
    env.local_db.insert("document_chunks", {"document_id": doc["id"], "chunk_text": "Shipping takes a week.",
                                            "kind": "passage", "chunk_index": 1})
    sig: dict = {}
    out = run(env.retrieve_context(a["id"], "refund window", user_id="local-user", storage_mode="local", signals=sig))
    assert out[0] == "Policy > Refunds\nThe refund window is thirty days."
    assert "How long do I have for a refund?" not in out and len(out) == 2
    conf = sig["confidence"]
    assert conf["level"] in ("high", "medium", "low", "none") and {"top", "gap", "reason"} <= set(conf)
    assert sig["fusion"] == "weighted" and "dense" in sig["signals_used"] and sig["rerank"]["mode"] in ("auto", "off", "on")
    assert sig["n_chunks"] == 2 and len(sig["scores"]) == 2
    # the passage was re-embedded from header + text (contract 4), the question from its own text
    rows = {r["chunk_text"]: r for r in env.local_db.select("document_chunks")}
    assert rows["The refund window is thirty days."]["embedding"] == Dim(16).encode(
        ["Policy > Refunds\nThe refund window is thirty days."])[0].tolist()


def test_turn_sets_run_retrieval_confidence(app_client, env, monkeypatch, ref_wav):
    from tests.test_perf import fake_speech, speak
    from services import provenance
    monkeypatch.setattr(provenance, "unavailable_reason", lambda: "stubbed out")
    fake_speech(env, monkeypatch)
    seen = {}
    original = env._turn_context

    async def spy(run_, *a, **k):
        out = await original(run_, *a, **k)
        seen["conf"] = getattr(run_, "retrieval_confidence", "missing")
        return out

    monkeypatch.setattr(env, "_turn_context", spy)
    a = make_agent(app_client)
    upload_txt(app_client, a["id"])
    body = speak(app_client, a["id"], ref_wav.read_bytes())
    assert isinstance(seen["conf"], dict) and seen["conf"]["level"] in ("high", "medium", "low", "none")
    assert body["retrieval"]["confidence"] == seen["conf"]


def test_cloud_path_maps_question_rows_to_parents_and_tolerates_old_rpcs(env, monkeypatch):
    dim = 1024
    vecs = np.eye(dim, dtype=np.float32)
    chunks = [
        {"id": "p1", "document_id": "d1", "chunk_text": "Orthopaedics OPD Monday Wednesday Friday", "embedding": None,
         "embedding_1024": (vecs[0] * 0.3 + vecs[3]).tolist()},
        {"id": "p2", "document_id": "d1", "chunk_text": "Cardiology OPD Tuesday", "embedding": None,
         "embedding_1024": (vecs[1]).tolist()},
        {"id": "q1", "document_id": "d1", "chunk_text": "When does the bone doctor sit?", "embedding": None,
         "embedding_1024": vecs[0].tolist(), "kind": "question", "parent_chunk_id": "p1"},
    ]
    fake = _cloud_env(env, monkeypatch, dim, chunks)
    original_rpc = fake.rpc

    def rpc_020(name, params):          # migration 020: kind / parent_chunk_id / header come back too
        res = original_rpc(name, params).execute()
        by_id = {c["id"]: c for c in chunks}
        data = [{**r, "kind": by_id[r["id"]].get("kind", "passage"), "parent_chunk_id": by_id[r["id"]].get("parent_chunk_id"),
                 "header": "Hospital > OPD" if r["id"] == "p1" else None} for r in res.data]
        return type("Q", (), {"execute": lambda self_: type("R", (), {"data": data})()})()

    monkeypatch.setattr(fake, "rpc", rpc_020)
    info: dict = {}
    q = vecs[0].tolist()
    rows = run(env._cloud_match("a1", "bone doctor", q, 3, info=info))
    assert [r["id"] for r in rows] == ["p1", "p2"] and rows[0]["via"] == "question"
    assert rows[0]["similarity"] == pytest.approx(1.0) and info["fusion"] == "weighted"
    assert retrieval.passage_text(rows[0]) == "Hospital > OPD\nOrthopaedics OPD Monday Wednesday Friday"
    # an old match function (no kind column): every row is a passage, the question text may then come back
    monkeypatch.setattr(fake, "rpc", original_rpc)
    rows = run(env._cloud_match("a1", "bone doctor", q, 3))
    assert {r["id"] for r in rows} == {"p1", "p2", "q1"} and all("score" in r for r in rows)


def test_cloud_retrieve_context_end_to_end(env, monkeypatch):
    chunks = [{"id": "c0", "document_id": "d1", "chunk_text": "refund thirty days", "embedding": [1.0, 0.0, 0.0] + [0.0] * 381,
               "embedding_1024": None}]
    _cloud_env(env, monkeypatch, 384, chunks)

    async def owned(*a, **k):
        return {"id": "a1"}

    monkeypatch.setattr(env, "_owned_agent", owned)
    sig: dict = {}
    out = run(env.retrieve_context("a1", "refund", user_id="u1", storage_mode="cloud", signals=sig))
    assert out == ["refund thirty days"] and sig["confidence"]["level"] in ("high", "medium", "low", "none")


def test_migration_020_returns_structure_columns_and_parents():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "supabase"
    sql = (root / "migrations" / "020_match_question_rows.sql").read_text().lower()
    for fn in ("match_document_chunks(", "match_document_chunks_1024("):
        assert f"drop function if exists public.{fn}" in sql and f"create function public.{fn}" in sql
    assert sql.count("coalesce(c.kind, 'passage')") == 2 and sql.count("parent_chunk_id uuid") == 2
    assert sql.count("header text") == 2 and sql.count("union all") == 2
    assert "018_chunk_structure.sql" in sql
    assert "020_match_question_rows.sql" in (root / "README.md").read_text()


def test_rerank_auto_gap_env(monkeypatch):
    rows = _cand(1.0, 0.85)
    assert retrieval.rerank_gate(rows, {"level": "low"})[0] is False                    # gap 0.15 >= 0.10
    monkeypatch.setenv("RERANK_AUTO_GAP", "0.2")
    assert retrieval.rerank_gate(rows, {"level": "low"})[0] is True
