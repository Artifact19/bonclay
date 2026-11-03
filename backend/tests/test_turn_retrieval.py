"""Retrieval signals stored with each agent turn ({n_chunks, top_score, top_dense, doc_ids, model, ...}): the signal
builder, local persistence through the real turn endpoint, hosted persistence with a fake Supabase client (including
the migration 012 column missing), and the migration SQL."""
import asyncio
from pathlib import Path

import numpy as np
import pytest

from services import provenance
from tests.test_offline_agents import LOCAL, env, make_agent, upload_txt  # noqa: F401  (env is a fixture)
from tests.test_perf import fake_speech, speak

BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def no_watermark_model(monkeypatch):
    monkeypatch.setattr(provenance, "unavailable_reason", lambda: "stubbed out")


# -- the signal builder ------------------------------------------------------------------------------------------

def test_signals_use_dense_cosine_and_cap_doc_ids(env):
    rows = [{"similarity": 0.41, "document_id": "d1"}, {"similarity": 0.62, "document_id": "d2"},
            {"similarity": 0.5, "document_id": "d1"}, {"similarity": None, "document_id": "d3"},
            {"similarity": 0.2, "document_id": "d4"}]
    s = env._retrieval_signals(rows)
    assert s["n_chunks"] == 5 and s["top_score"] == 0.62      # the best dense cosine of the chunks the model got
    assert s["top_dense"] == 0.41                             # the chunk fused ranking put first (BM25 promoted it)
    assert s["doc_ids"] == ["d1", "d2", "d3"]                 # unique, at most three
    assert s["model"] == env.EMBEDDING_MODEL and "error" not in s and "top_rerank" not in s


def test_signals_with_nothing_found_reranker_and_error(env):
    assert env._retrieval_signals([]) == {"n_chunks": 0, "top_score": None, "top_dense": None, "doc_ids": [],
                                          "model": env.EMBEDDING_MODEL, "hybrid": env.retrieval.hybrid_enabled()}
    s = env._retrieval_signals([{"similarity": 0.5, "rerank_score": 3.2, "document_id": "d"}])
    assert s["top_rerank"] == 3.2
    assert env._retrieval_signals([], error=True)["error"] is True


def test_retrieve_context_fills_signals_and_keeps_its_return_type(app_client, env):
    a = make_agent(app_client)
    upload_txt(app_client, a["id"])
    sig: dict = {}
    out = asyncio.run(env.retrieve_context(a["id"], "refund window", user_id="local-user", storage_mode="local",
                                           signals=sig))
    assert out and all(isinstance(c, str) for c in out)
    assert sig["n_chunks"] == len(out) and isinstance(sig["top_score"], float) and sig["doc_ids"]


def test_retrieval_failure_is_marked_as_an_error_not_a_gap(app_client, env, monkeypatch):
    a = make_agent(app_client)

    def boom(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(env.local_db, "match_chunks", boom)
    sig: dict = {}
    assert asyncio.run(env.retrieve_context(a["id"], "refund", user_id="local-user", storage_mode="local",
                                            signals=sig)) == []
    assert sig["n_chunks"] == 0 and sig["error"] is True


# -- local persistence -------------------------------------------------------------------------------------------

def test_agent_turn_row_and_done_payload_carry_the_signals_locally(app_client, env, monkeypatch, ref_wav):
    fake_speech(env, monkeypatch)
    a = make_agent(app_client)
    upload_txt(app_client, a["id"])
    body = speak(app_client, a["id"], ref_wav.read_bytes())
    row = env.local_db.select("conversation_turns", lambda r: r.get("role") == "agent")[0]
    r = row["retrieval"]
    assert set(r) >= {"n_chunks", "top_score", "top_dense", "doc_ids", "model"}
    assert r["n_chunks"] >= 1 and 0 < len(r["doc_ids"]) <= 3 and r["model"] == env.EMBEDDING_MODEL
    assert body["retrieval"] == r
    user = env.local_db.select("conversation_turns", lambda r: r.get("role") == "user")[0]
    assert "retrieval" not in user


def test_agent_without_documents_stores_zero_chunks(app_client, env, monkeypatch, ref_wav):
    fake_speech(env, monkeypatch)
    a = make_agent(app_client)
    speak(app_client, a["id"], ref_wav.read_bytes())
    r = env.local_db.select("conversation_turns", lambda r: r.get("role") == "agent")[0]["retrieval"]
    assert r["n_chunks"] == 0 and r["top_score"] is None and r["doc_ids"] == []


# -- hosted persistence (fake Supabase) --------------------------------------------------------------------------

class FakeSb:
    """Records inserts into conversation_turns; fails the ones that carry `bad_column`."""

    def __init__(self, bad_columns=()):
        self.bad_columns, self.inserted, self.attempts = tuple(bad_columns), [], 0
        self.storage = self
        self.signed = []

    def from_(self, bucket):
        return self

    def upload(self, *a, **k):
        return None

    def create_signed_url(self, path, expires):
        return {"signedURL": f"https://x/{path}?token=t"}

    def table(self, name):
        sb = self

        class T:
            def insert(self, rows):
                self.rows = rows
                return self

            def execute(self):
                sb.attempts += 1
                if any(c in row for row in self.rows for c in sb.bad_columns):
                    raise RuntimeError('column "retrieval" does not exist')
                sb.inserted.append(self.rows)
        return T()


class Run:
    local, user_id, conversation_id, user_text, language = False, "u1", "c1", "q", "en"
    start_time, tool_objs = 0.0, []

    def __init__(self, sb, retrieval):
        self.sb, self.retrieval, self.timings = sb, retrieval, {"asr_ms": 5}


def store(env, monkeypatch, sb, retrieval):
    async def upload(*a, **k):
        return None
    monkeypatch.setattr(env, "sb_upload", upload)
    monkeypatch.setattr(env, "_signed_audio_url", lambda sb_, path: "https://x/a.wav")
    return asyncio.run(env._turn_store(Run(sb, retrieval), "t1", b"RIFF", "hello", engine_key="kokoro",
                                       llm_provider=None, llm_model=None, fallback=False))


def test_cloud_turn_insert_carries_retrieval(env, monkeypatch):
    sb = FakeSb()
    sig = {"n_chunks": 2, "top_score": 0.61, "top_dense": 0.6, "doc_ids": ["d1"], "model": "m"}
    store(env, monkeypatch, sb, sig)
    user_row, agent_row = sb.inserted[0]
    assert agent_row["retrieval"] == sig and "retrieval" not in user_row and agent_row["timings"]["asr_ms"] == 5


def test_cloud_turn_without_signals_has_no_retrieval_field(env, monkeypatch):
    sb = FakeSb()
    store(env, monkeypatch, sb, None)
    assert "retrieval" not in sb.inserted[0][1]


def test_cloud_missing_retrieval_column_keeps_the_turn_metrics(env, monkeypatch):
    sb = FakeSb(bad_columns=["retrieval"])               # migration 012 not applied
    store(env, monkeypatch, sb, {"n_chunks": 1, "top_score": 0.5})
    assert sb.attempts == 2
    agent_row = sb.inserted[0][1]
    assert "retrieval" not in agent_row and agent_row["timings"] == {"asr_ms": 5} and agent_row["engine"] == "kokoro"


def test_cloud_older_schema_still_saves_the_turn(env, monkeypatch):
    sb = FakeSb(bad_columns=["retrieval", "timings"])    # neither migration applied
    store(env, monkeypatch, sb, {"n_chunks": 1, "top_score": 0.5})
    agent_row = sb.inserted[0][1]
    assert agent_row["text_content"] == "hello" and "timings" not in agent_row and "retrieval" not in agent_row


def test_migration_012_adds_an_idempotent_jsonb_column():
    sql = (BACKEND / "supabase" / "migrations" / "012_turn_retrieval.sql").read_text().lower()
    assert "conversation_turns" in sql and "add column if not exists retrieval jsonb" in sql
