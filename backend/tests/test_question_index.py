"""The question index (doc2query): generation with a mocked model, batching, pacing, 429 retry, offline skip; ingestion
stores passages with headers and question rows, never blocks, deletes with the document; the backfill script."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from services import llm, question_index as qi, restructure
from services.local_store import LocalStore

LOCAL = {"storage-mode": "local"}
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rechunk_knowledge.py"


# ------------------------------------------------------------------------------------------------- fakes

def fake_embed(texts):
    out = []
    for t in texts:
        v = np.zeros(64, dtype=np.float32)
        for w in t.lower().split():
            v[int(hashlib.md5(w.strip(".,?!|").encode()).hexdigest(), 16) % 64] += 1.0
        out.append(v)
    return np.stack(out)


class FakeEmbedding:
    loaded = True

    def __init__(self):
        self.calls = []

    def encode(self, texts, *a, **k):
        self.calls.append(list(texts))
        return fake_embed(list(texts))


def reply_for(messages, per_language=2):
    """A model reply for the passages in the user message: questions in three languages naming the passage's first words."""
    payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
    items = []
    for p in payload:
        subject = " ".join(p["text"].split()[-3:]).strip(".,")
        items.append({"id": p["id"],
                      "en": [f"What is {subject} {k}?" for k in range(per_language)],
                      "hi": [f"क्या {k} समय खुला है?" for k in range(per_language)],
                      "hinglish": [f"{subject} ke baare mein batao {k}?" for k in range(per_language)]})
    return json.dumps({"items": items})


class FakeChat:
    def __init__(self, replies=None):
        self.calls = []
        self.replies = replies

    def __call__(self, messages, **kw):
        if messages[0]["content"] != qi.SYSTEM_PROMPT:      # another feature's call (the agent's profile)
            persona = {"name": "Ria", "greeting": "Hi.", "system_instructions": "helper", "suggested_languages": ["en"]}
            return llm.ChatResult(json.dumps(persona), "fake", "m", 1, {})
        self.calls.append({"messages": messages, **kw})
        if self.replies:
            r = self.replies.pop(0)
            if isinstance(r, BaseException):
                raise r
            return llm.ChatResult(r, "groq", "m", 5, {"total_tokens": 700})
        return llm.ChatResult(reply_for(messages), "groq", "m", 5, {"total_tokens": 700})


def passages(n, words=12):
    return [{"id": f"p{i}", "header": f"Doc › Section {i}", "content": " ".join(f"word{i}x{j}" for j in range(words)) + "."}
            for i in range(n)]


class NoWait:
    def __init__(self):
        self.slept = []

    def __call__(self, s):
        self.slept.append(s)


@pytest.fixture(autouse=True)
def fast_pacing(monkeypatch):
    monkeypatch.setenv("QUESTION_INDEX_MIN_INTERVAL_S", "0")
    monkeypatch.setenv("QUESTION_INDEX_TPM", "1000000")
    monkeypatch.setenv("QUESTION_INDEX_BACKOFF_S", "10")


# ------------------------------------------------------------------------------------------------ parsing

def test_parse_reply_classifies_by_script_dedupes_and_caps():
    text = json.dumps({"items": [{
        "id": 0,
        "en": ["What are the OPD timings?", "what are the opd timings", "Is parking free?", "Is the pharmacy open?",
               "Where is the canteen?", "x?"],
        "hi": ["ओपीडी का समय क्या है?", "Cardiology kab khulti hai?", "पार्किंग मुफ़्त है क्या?"],
        "hinglish": ["OPD ka time kya hai?", "Parking free hai kya?", "ओपीडी कब खुलती है"]}]})
    q = qi.parse_reply(text, [0])[0]
    assert q[:3] == ["What are the OPD timings?", "Is parking free?", "Is the pharmacy open?"]   # en: duplicate, 4th and 1-word dropped
    hi = [x for x in q if qi._script_of(x) == "hi"]
    assert hi == ["ओपीडी का समय क्या है?", "पार्किंग मुफ़्त है क्या?", "ओपीडी कब खुलती है?"]
    roman = [x for x in q if qi._script_of(x) == "latin"][3:]
    assert roman == ["Cardiology kab khulti hai?", "OPD ka time kya hai?", "Parking free hai kya?"]
    assert len(q) <= qi.PER_PASSAGE_MAX == 9


def test_parse_reply_ignores_unknown_ids_and_tolerates_shapes():
    assert qi.parse_reply(json.dumps({"items": [{"id": 7, "en": ["Is it open today?"]}]}), [0]) == {}
    assert qi.parse_reply(json.dumps({"items": {"0": {"en": "Is it open today?"}}}), [0]) == {0: ["Is it open today?"]}
    assert qi.parse_reply(json.dumps([{"id": "0", "questions": ["Is it open today"]}]), [0]) == {0: ["Is it open today?"]}
    with pytest.raises(ValueError):
        qi.parse_reply(json.dumps({"nope": 1}), [0])


# ------------------------------------------------------------------------------------------------ batching

def test_batches_hold_a_few_passages_and_skip_tiny_ones(monkeypatch):
    ps = passages(10) + [{"id": "t", "header": "x", "content": "Title"}]
    batches = qi.make_batches(ps)
    assert [len(b) for b in batches] == [4, 4, 2] and all(p["id"] != "t" for b in batches for _, p in b)
    monkeypatch.setenv("QUESTION_INDEX_BATCH", "3")
    assert [len(b) for b in qi.make_batches(ps)] == [3, 3, 3, 1]


def test_generate_batch_prompt_and_json_call():
    chat = FakeChat()
    batch = qi.make_batches(passages(3))[0]
    out = qi.generate_batch(batch, chat=chat, pacer=qi.Pacer(), sleep=NoWait())
    assert set(out) == {0, 1, 2} and all(len(v) == 6 for v in out.values())
    call = chat.calls[0]
    assert call["json_mode"] is True and call["temperature"] == 0.3 and call["providers"] == ["groq", "openai"]
    system = call["messages"][0]["content"]
    assert "ANSWERS" in system and "Never use a fact that is not in the passage" in system and "Devanagari" in system
    assert "Hinglish" in system or "hinglish" in system
    user = json.loads(call["messages"][1]["content"].split("\n", 1)[1])
    assert [u["id"] for u in user] == [0, 1, 2] and user[0]["text"].startswith("Doc › Section 0\n")


def test_language_mix_per_passage():
    out = qi.generate_batch(qi.make_batches(passages(1))[0], chat=FakeChat(), pacer=qi.Pacer(), sleep=NoWait())[0]
    scripts = [qi._script_of(q) for q in out]
    assert scripts.count("hi") == 2 and scripts.count("latin") == 4 and 6 <= len(out) <= 9


def test_429_is_retried_with_a_growing_wait():
    rate = llm.LLMUnavailable({"groq": "failed: RuntimeError: HTTP 429 rate limit"})
    chat = FakeChat([rate, rate, reply_for([{"content": 'x\n[{"id": 0, "text": "alpha beta gamma"}]'}])])
    sleep = NoWait()
    out = qi.generate_batch(qi.make_batches(passages(1))[0], chat=chat, pacer=qi.Pacer(sleep=sleep), sleep=sleep)
    assert len(out[0]) == 6 and len(chat.calls) == 3
    assert sleep.slept == [10, 20]


def test_retries_run_out_cleanly(monkeypatch):
    monkeypatch.setenv("QUESTION_INDEX_RETRIES", "3")
    rate = llm.LLMUnavailable({"groq": "failed: RuntimeError: HTTP 429"})
    chat = FakeChat([rate, rate, rate])
    with pytest.raises(RuntimeError, match="failed"):
        qi.generate_batch(qi.make_batches(passages(1))[0], chat=chat, pacer=qi.Pacer(), sleep=NoWait())
    assert len(chat.calls) == 3


def test_offline_is_a_skip_not_a_retry():
    chat = FakeChat([llm.LLMUnavailable({"groq": "GROQ_API_KEY is not set", "openai": "LLM_BASE_URL is not set"})])
    with pytest.raises(qi.Skipped):
        qi.generate_batch(qi.make_batches(passages(1))[0], chat=chat, pacer=qi.Pacer(), sleep=NoWait())
    assert len(chat.calls) == 1


def test_a_bad_reply_is_retried_once_then_gives_up():
    chat = FakeChat([json.dumps({"oops": 1}), json.dumps({"oops": 2})])
    with pytest.raises(RuntimeError):
        qi.generate_batch(qi.make_batches(passages(1))[0], chat=chat, pacer=qi.Pacer(), sleep=NoWait())
    assert len(chat.calls) == 2


def test_pacer_keeps_to_the_token_budget_and_the_gap():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(round(s, 3))
        now[0] += s

    p = qi.Pacer(tpm=3000, min_interval=2.0, clock=lambda: now[0], sleep=sleep)
    p.wait(1500)
    p.record(1500)
    p.wait(1500)                      # fits the budget but not the 2 s gap
    p.record(1500)
    assert slept == [2.0]
    p.wait(1500)                      # the minute is full: wait until the first call ages out (at t=60)
    assert now[0] == pytest.approx(60.0)
    p.record(1500)
    p.wait(100000)                    # bigger than the budget: capped, still possible
    assert now[0] < 125


# ------------------------------------------------------------------------------ the job (no app, no store)

def test_index_passages_saves_batch_by_batch_and_stops_when_the_document_is_gone():
    saved, progress = [], []
    alive = iter([True, True, False])
    res = qi.index_passages(passages(10), embed=fake_embed, chat=FakeChat(), pacer=qi.Pacer(), sleep=NoWait(),
                            save=lambda rows: saved.append(rows) or len(rows), alive=lambda: next(alive),
                            progress=progress.append)
    assert res["status"] == "cancelled" and len(saved) == 2 and res["questions"] == sum(len(r) for r in saved)
    assert all(len(r["embedding"]) == 64 and r["parent_chunk_id"].startswith("p") for rows in saved for r in rows)
    assert progress[0]["status"] == "indexing"


def test_index_passages_keeps_what_it_has_when_the_quota_runs_out(monkeypatch):
    monkeypatch.setenv("QUESTION_INDEX_RETRIES", "1")
    rate = llm.LLMUnavailable({"groq": "failed: HTTP 429"})
    good = reply_for([{"content": 'x\n' + json.dumps([{"id": i, "text": "alpha beta gamma"} for i in range(4)])}])
    chat = FakeChat([good, rate])
    saved = []
    res = qi.index_passages(passages(8), embed=fake_embed, chat=chat, pacer=qi.Pacer(), sleep=NoWait(),
                            save=lambda rows: saved.append(rows) or len(rows))
    assert res["status"] == "failed" and res["questions"] == len(saved[0]) > 0


def test_index_passages_offline_and_no_text():
    chat = FakeChat([llm.LLMUnavailable({"groq": "GROQ_API_KEY is not set"})])
    res = qi.index_passages(passages(2), embed=fake_embed, save=lambda r: len(r), chat=chat, pacer=qi.Pacer(),
                            sleep=NoWait())
    assert res["status"] == "skipped" and res["questions"] == 0
    assert qi.index_passages([{"id": "a", "header": "", "content": "Hi"}], embed=fake_embed, save=lambda r: len(r))["status"] == "done"


def test_the_switch(monkeypatch):
    assert qi.enabled()
    monkeypatch.setenv("QUESTION_INDEX", "off")
    assert not qi.enabled()


# ------------------------------------------------------------------------------------------ in the app

@pytest.fixture()
def env(app_client, monkeypatch, tmp_path):
    import backend

    monkeypatch.setattr(backend, "local_db", LocalStore(str(tmp_path)))
    monkeypatch.setattr(backend, "LOCAL_AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(backend, "LOCAL_VOICES_FILE", str(tmp_path / "voices.json"))
    emb = FakeEmbedding()
    monkeypatch.setattr(backend, "embedding_model", emb)
    if hasattr(backend.limiter, "reset"):
        backend.limiter.reset()
    backend.fake_embedding = emb
    return backend


@pytest.fixture()
def hosted(monkeypatch):
    """A hosted model that answers with the fake questions."""
    chat = FakeChat()
    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(qi, "hosted_available", lambda: True)
    return chat


def make_agent(client):
    r = client.post("/api/agents", headers=LOCAL, json={"name": "Support", "role_description": "helper"})
    assert r.status_code == 200, r.text
    return r.json()


DOC = ("# Hospital Guide\n\n## Timings\n\nThe OPD is open Monday to Saturday from nine to five. Emergency care is open every "
       "day of the year.\n\n## FAQ\n\nIs parking free?\nYes, parking is free for patients in the basement.\n\n"
       "| Department | Days |\n|---|---|\n| Cardiology | Tuesday, Thursday |\n| Eye | Monday to Friday |\n")


def upload(client, agent_id, text=DOC, name="guide.md"):
    return client.post("/api/knowledge/upload", headers=LOCAL, data={"agent_id": agent_id},
                       files={"file": (name, text.encode(), "text/markdown")})


def rows_of(env, kind=None):
    rows = env.local_db.select("document_chunks")
    return [r for r in rows if (kind is None) or (r.get("kind") == kind)]


def test_upload_stores_passages_with_headers_and_returns_before_the_questions(app_client, env, hosted, monkeypatch):
    import threading
    a = make_agent(app_client)
    started, release = threading.Event(), threading.Event()
    real = hosted.__call__

    def slow(messages, **kw):
        started.set()
        release.wait(10)
        return real(messages, **kw)

    monkeypatch.setattr(llm, "chat", slow)
    try:
        r = upload(app_client, a["id"])
        assert r.status_code == 200, r.text            # returned while the model call is still blocked
        doc = r.json()["document"]
        assert doc["question_index"] == "pending" and doc["question_count"] == 0
        passages = rows_of(env, "passage")
        assert len(passages) == doc["chunk_count"] == r.json()["chunks_created"] >= 4
        assert all("header" in p and p["header"].startswith("Hospital Guide") for p in passages)
        assert any(p["header"].endswith("Is parking free?") for p in passages)
        assert {p["chunk_text"] for p in passages if "Cardiology" in p["chunk_text"]} == {
            "Department | Days\nCardiology | Tuesday, Thursday"}
        # what was embedded is header + newline + content
        embedded = [t for call in env.fake_embedding.calls for t in call]
        assert ("Hospital Guide \u203a Timings\nThe OPD is open Monday to Saturday from nine to five. "
                "Emergency care is open every day of the year.") in embedded
        assert not rows_of(env, "question")
        assert started.wait(5)
    finally:
        release.set()
    qi.wait_idle(20)
    questions = rows_of(env, "question")
    ids = {p["id"] for p in rows_of(env, "passage")}
    assert len(questions) >= 6 * 4 and all(q["parent_chunk_id"] in ids for q in questions)
    q = questions[0]
    assert q["kind"] == "question" and q["chunk_text"] == q["content"] and q["chunk_text"].endswith("?")
    assert len(q["embedding"]) == 64 and q["embed_model"] == env.EMBEDDING_MODEL and q["document_id"] == doc["id"]
    assert {qi._script_of(x["chunk_text"]) for x in questions} == {"hi", "latin"}
    listed = app_client.get(f"/api/agents/{a['id']}/documents", headers=LOCAL).json()["documents"]
    assert listed[0]["question_count"] == len(questions) and listed[0]["question_index"] == "done"
    assert listed[0]["chunk_count"] == len(ids)               # questions are not passages


def test_delete_removes_passages_and_question_rows(app_client, env, hosted):
    a = make_agent(app_client)
    doc = upload(app_client, a["id"]).json()["document"]
    qi.wait_idle(20)
    other = upload(app_client, a["id"], "# Other\n\nThe canteen opens at seven in the morning every day.", "o.md").json()["document"]
    qi.wait_idle(20)
    assert rows_of(env, "question")
    assert app_client.delete(f"/api/agents/{a['id']}/documents/{doc['id']}", headers=LOCAL).json() == {"deleted": True}
    left = env.local_db.select("document_chunks")
    assert left and {r["document_id"] for r in left} == {other["id"]}
    assert app_client.delete(f"/api/agents/{a['id']}/documents/{other['id']}", headers=LOCAL).status_code == 200
    assert env.local_db.select("document_chunks") == []


def test_a_document_deleted_while_indexing_leaves_no_question_rows(app_client, env, hosted, monkeypatch):
    import threading
    a = make_agent(app_client)
    started, release = threading.Event(), threading.Event()
    real = hosted.__call__

    def slow(messages, **kw):
        started.set()
        release.wait(10)
        return real(messages, **kw)

    monkeypatch.setattr(llm, "chat", slow)
    doc = upload(app_client, a["id"]).json()["document"]
    assert started.wait(5)
    app_client.delete(f"/api/agents/{a['id']}/documents/{doc['id']}", headers=LOCAL)
    release.set()
    qi.wait_idle(20)
    assert env.local_db.select("document_chunks") == []


def test_no_hosted_model_keeps_the_agent_working_without_questions(app_client, env, monkeypatch):
    monkeypatch.setattr(qi, "hosted_available", lambda: False)
    chat = FakeChat()
    monkeypatch.setattr(llm, "chat", chat)
    a = make_agent(app_client)
    doc = upload(app_client, a["id"]).json()["document"]
    qi.wait_idle(20)
    assert doc["question_index"] == "skipped" and doc["question_count"] == 0 and not chat.calls
    assert rows_of(env, "passage") and not rows_of(env, "question")
    import asyncio
    got = asyncio.run(env.retrieve_context(a["id"], "parking basement", user_id="local-user", storage_mode="local"))
    assert got and any("parking" in g.lower() for g in got)


def test_the_switch_turns_the_index_off(app_client, env, hosted, monkeypatch):
    monkeypatch.setenv("QUESTION_INDEX", "off")
    a = make_agent(app_client)
    doc = upload(app_client, a["id"]).json()["document"]
    qi.wait_idle(20)
    assert doc["question_index"] == "off" and not hosted.calls and not rows_of(env, "question")


def test_structured_chunking_switch_gives_the_old_passages(app_client, env, monkeypatch):
    monkeypatch.setenv("STRUCTURED_CHUNKING", "off")
    monkeypatch.setattr(qi, "hosted_available", lambda: False)
    monkeypatch.setattr(llm, "chat", FakeChat())
    a = make_agent(app_client)
    upload(app_client, a["id"])
    rows = rows_of(env, "passage")
    assert rows and all(r["header"] == "" for r in rows) and rows[0]["chunk_text"].startswith("Timings\n")


def test_local_store_drops_orphans_and_ignores_gone_documents(tmp_path):
    store = LocalStore(str(tmp_path))
    doc = store.insert("documents", {"agent_id": "a", "user_id": "u", "filename": "f", "chunk_count": 1})
    (p,) = store.insert_many("document_chunks", [{"document_id": doc["id"], "chunk_text": "x y z a b c", "kind": "passage"}])
    q = {"chunk_text": "Q?", "kind": "question", "embedding": [1.0]}
    assert store.add_question_rows(doc["id"], [{**q, "parent_chunk_id": p["id"]}, {**q, "parent_chunk_id": "gone"}]) == 1
    assert store.add_question_rows("no-such-doc", [{**q, "parent_chunk_id": p["id"]}]) == 0
    store.delete_document(doc["id"])
    assert store.select("document_chunks") == []


# ------------------------------------------------------------------------------------------- the backfill

@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("rechunk_knowledge", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def old_style_doc(store, name="guide.md", user="u1", agent="a1"):
    """A document as stored before this change: passages with the heading inside the text, no header, no kind."""
    from services import chunking
    chunks = chunking.chunk_text(DOC.replace("| Department | Days |\n|---|---|\n", "").replace("| ", "").replace(" |", ""))
    doc = store.insert("documents", {"agent_id": agent, "user_id": user, "filename": name, "chunk_count": len(chunks)})
    store.insert_many("document_chunks", [{"document_id": doc["id"], "chunk_text": c, "chunk_index": i,
                                           "embedding": [9.0], "embed_model": "old", "embed_dim": 1}
                                          for i, c in enumerate(chunks)])
    return doc


def enc(texts):
    return [[float(len(t.split())), 1.0, 0.5] for t in texts]


def test_script_dry_run_changes_nothing_and_loads_no_model(tmp_path, script):
    store = LocalStore(str(tmp_path))
    old_style_doc(store)
    before = (tmp_path / "local_db" / "document_chunks.json").read_bytes()
    lines = []

    def no_model(_):
        raise AssertionError("a dry run must not embed")

    r = script.run(str(tmp_path), dry_run=True, encode=no_model, model_name="m", out=lines.append,
                   with_structure=True, questions=True)
    assert r["failed"] == 0 and r["structure"]["documents"] == 1 and r["questions"]["documents"] == 1
    assert r["questions"]["calls"] >= 1 and r["questions"]["passages"] >= 1
    assert (tmp_path / "local_db" / "document_chunks.json").read_bytes() == before
    assert not list(tmp_path.glob("rechunk-backup-*"))
    assert any("would be re-cut" in l for l in lines) and any("would be indexed" in l for l in lines)


def test_script_questions_need_a_hosted_model(tmp_path, script, monkeypatch):
    store = LocalStore(str(tmp_path))
    old_style_doc(store)
    monkeypatch.setattr(qi, "hosted_available", lambda: False)
    out = []
    r = script.run(str(tmp_path), encode=enc, model_name="m", out=out.append, questions=True)
    assert "no hosted language model" in r["questions"]["status"] and r["questions"]["questions"] == 0
    assert not [c for c in store.select("document_chunks") if c.get("kind") == "question"]


def test_script_real_run_restructures_then_indexes_and_is_idempotent(tmp_path, script):
    store = LocalStore(str(tmp_path))
    doc = old_style_doc(store)
    chat = FakeChat()
    r = script.run(str(tmp_path), encode=enc, model_name="m", out=lambda *_: None, with_structure=True, questions=True,
                   chat=chat, pacer=qi.Pacer(), sleep=NoWait())
    assert r["failed"] == 0 and len(list(tmp_path.glob("rechunk-backup-*"))) == 1
    rows = store.select("document_chunks")
    passages = [x for x in rows if x.get("kind") == "passage"]
    questions = [x for x in rows if x.get("kind") == "question"]
    assert passages and questions and all("header" in p and p["embed_model"] == "m" for p in passages)
    assert {q["parent_chunk_id"] for q in questions} <= {p["id"] for p in passages}
    assert all(q["embed_model"] == "m" and q["document_id"] == doc["id"] for q in questions)
    d = store.get("documents", doc["id"])
    assert d["chunk_count"] == len(passages) and d["question_count"] == len(questions) and d["question_index"] == "done"
    # again: nothing to do
    calls_before = len(chat.calls)
    r2 = script.run(str(tmp_path), encode=enc, model_name="m", out=lambda *_: None, with_structure=True, questions=True,
                    chat=chat, pacer=qi.Pacer(), sleep=NoWait(), backup=False)
    assert r2["structure"]["documents"] == 0 and r2["questions"]["documents"] == 0 and len(chat.calls) == calls_before
    assert len(store.select("document_chunks")) == len(rows)
    # --redo-questions replaces them
    script.run(str(tmp_path), encode=enc, model_name="m", out=lambda *_: None, questions=True, redo_questions=True,
               chat=chat, pacer=qi.Pacer(), sleep=NoWait(), backup=False)
    assert len([x for x in store.select("document_chunks") if x.get("kind") == "question"]) == len(questions)


def test_script_reads_the_original_file_when_given_a_source_dir(tmp_path, script):
    store = LocalStore(str(tmp_path / "data"))
    (tmp_path / "data" / "local_db").mkdir(exist_ok=True)
    doc = old_style_doc(store)
    src = tmp_path / "originals" / "nested"
    src.mkdir(parents=True)
    (src / "guide.md").write_text(DOC, encoding="utf-8")
    out = []
    r = script.run(str(tmp_path / "data"), encode=enc, model_name="m", out=out.append, with_structure=True,
                   source_dir=str(tmp_path / "originals"))
    assert r["structure"]["from_source"] == 1 and any("original file" in l for l in out)
    texts = [x["chunk_text"] for x in store.select("document_chunks")]
    assert "Department | Days\nCardiology | Tuesday, Thursday" in texts
    assert store.get("documents", doc["id"])["question_index"] == "none"


def test_rebuild_text_drops_the_overlap_and_keeps_lines():
    from services import chunking
    original = "\n".join(f"Line {i}: the quick brown fox number {i} jumps over the lazy dog today." for i in range(60))
    chunks = chunking.chunk_text(original)
    assert len(chunks) > 3
    rebuilt = restructure.rebuild_text(chunks)
    assert rebuilt.split() == original.split()


def test_legacy_rechunk_ignores_question_rows(tmp_path):
    from services.rechunk import rechunk_documents
    store = LocalStore(str(tmp_path))
    words = " ".join(f"w{i}" for i in range(1300))
    old = [" ".join(words.split()[i:i + 500]) for i in range(0, 1300, 450)]
    doc = store.insert("documents", {"agent_id": "a", "user_id": "u", "filename": "f", "chunk_count": len(old)})
    rows = store.insert_many("document_chunks", [{"document_id": doc["id"], "chunk_text": c, "chunk_index": i,
                                                  "embedding": [1.0]} for i, c in enumerate(old)])
    store.add_question_rows(doc["id"], [{"chunk_text": "Is it open?", "kind": "question", "parent_chunk_id": rows[0]["id"]}])
    r = rechunk_documents(store, enc, "m")
    assert r["documents"] == 1
    new_text = " ".join(c["chunk_text"] for c in store.select("document_chunks", lambda c: c["document_id"] == doc["id"]))
    assert "Is it open?" not in new_text


def test_new_modes_are_dry_runs_unless_apply_is_given(tmp_path, script, capsys):
    store = LocalStore(str(tmp_path))
    old_style_doc(store)
    before = (tmp_path / "local_db" / "document_chunks.json").read_bytes()
    assert script.main(["--data-dir", str(tmp_path), "--with-structure", "--questions"]) == 0
    out = capsys.readouterr().out
    assert "add --apply" in out and "would be re-cut" in out
    assert (tmp_path / "local_db" / "document_chunks.json").read_bytes() == before
    assert not list(tmp_path.glob("rechunk-backup-*"))
