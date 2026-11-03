"""The one-shot re-chunk of every stored document (backend/scripts/rechunk_knowledge.py, services/rechunk.py)."""
import importlib.util
import json
from pathlib import Path

import pytest

from services import chunking
from services.local_store import LocalStore
from services.rechunk import rechunk_documents

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rechunk_knowledge.py"
ORIGINAL = " ".join(f"w{i}" for i in range(1300))


def legacy_chunks(text, size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size - overlap)]


def fake_encode(texts):
    return [[float(len(t.split())), 1.0, 0.5] for t in texts]


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("rechunk_knowledge", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_doc(store, agent, name, chunks, user="u1"):
    doc = store.insert("documents", {"agent_id": agent, "user_id": user, "filename": name, "chunk_count": len(chunks)})
    store.insert_many("document_chunks", [{"document_id": doc["id"], "chunk_text": c, "chunk_index": i,
                                           "embedding": [9.0], "embed_model": "old", "embed_dim": 1}
                                          for i, c in enumerate(chunks)])
    return doc


@pytest.fixture
def data(tmp_path):
    store = LocalStore(str(tmp_path))
    old1 = add_doc(store, "a1", "old1.txt", legacy_chunks(ORIGINAL))
    old2 = add_doc(store, "a2", "old2.txt", legacy_chunks(ORIGINAL + " tail"), user="u2")
    new = add_doc(store, "a1", "new.txt", chunking.chunk_text(ORIGINAL))
    return tmp_path, store, old1, old2, new


def chunks_of(store, doc):
    return sorted(store.select("document_chunks", lambda c: c["document_id"] == doc["id"]), key=lambda c: c["chunk_index"])


def test_dry_run_reports_and_writes_nothing(data, script):
    tmp, store, old1, old2, new = data
    before = (tmp / "local_db" / "document_chunks.json").read_bytes()
    lines = []

    def no_embed(_):
        raise AssertionError("a dry run must not embed")

    r = script.run(str(tmp), dry_run=True, encode=no_embed, model_name="m", out=lines.append)
    assert r["documents"] == 2 and r["skipped"] == 1 and r["failed"] == 0
    assert r["chunks_before"] == 6 and r["chunks_after"] > 6
    assert (tmp / "local_db" / "document_chunks.json").read_bytes() == before
    assert not list(tmp.glob("rechunk-backup-*"))
    assert any("would be re-chunked" in l for l in lines)


def test_real_run_converts_all_agents_keeps_text_and_is_idempotent(data, script):
    tmp, store, old1, old2, new = data
    new_before = chunks_of(store, new)
    r = script.run(str(tmp), encode=fake_encode, model_name="m", out=lambda *_: None)
    assert r["documents"] == 2 and r["skipped"] == 1 and r["failed"] == 0
    assert len(list(tmp.glob("rechunk-backup-*"))) == 1
    for doc in (old1, old2):
        rows = chunks_of(store, doc)
        assert [c["chunk_index"] for c in rows] == list(range(len(rows))) and len(rows) > 3
        assert all(len(c["chunk_text"].split()) <= 120 for c in rows)
        assert all(c["embed_model"] == "m" and c["embed_dim"] == 3 and len(c["embedding"]) == 3 for c in rows)
        assert store.get("documents", doc["id"])["chunk_count"] == len(rows)
        assert {w for c in rows for w in c["chunk_text"].split()} >= {f"w{i}" for i in range(1300)}
    assert chunks_of(store, new) == new_before                     # a document already at the new size is untouched
    again = script.run(str(tmp), encode=fake_encode, model_name="m", backup=False, out=lambda *_: None)
    assert again["documents"] == 0 and again["skipped"] == 3 and again["chunks_after"] == 0


def test_agent_filter(data, script):
    tmp, store, old1, old2, new = data
    r = script.run(str(tmp), agent_id="a2", encode=fake_encode, model_name="m", backup=False, out=lambda *_: None)
    assert r["documents"] == 1
    assert len(chunks_of(store, old1)) == 3                        # agent a1 not touched


def test_a_failing_document_keeps_its_old_passages_and_the_rest_go_on(data):
    tmp, store, old1, old2, new = data
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("model fell over")
        return fake_encode(texts)

    r = rechunk_documents(store, flaky, "m")
    assert r["failed"] == 1 and r["documents"] == 1 and "model fell over" in r["errors"][0]["error"]
    bad = old1 if r["errors"][0]["document_id"] == old1["id"] else old2
    assert len(chunks_of(store, bad)) == 3 and chunks_of(store, bad)[0]["embed_model"] == "old"


def test_document_without_passages_is_skipped(tmp_path):
    store = LocalStore(str(tmp_path))
    store.insert("documents", {"agent_id": "a", "user_id": "u", "filename": "empty.txt", "chunk_count": 0})
    r = rechunk_documents(store, fake_encode, "m")
    assert r["skipped"] == 1 and r["documents"] == 0


def test_replace_is_a_single_write_of_the_chunk_table(tmp_path):
    store = LocalStore(str(tmp_path))
    doc = add_doc(store, "a", "x.txt", ["one two", "three four"])
    other = add_doc(store, "a", "y.txt", ["keep me"])
    store.replace_document_chunks(doc["id"], [{"chunk_text": "new", "chunk_index": 0}], {"chunk_count": 1})
    assert [c["chunk_text"] for c in chunks_of(store, doc)] == ["new"]
    assert [c["chunk_text"] for c in chunks_of(store, other)] == ["keep me"]
    assert store.get("documents", doc["id"])["chunk_count"] == 1


def test_missing_data_dir_exits(tmp_path, script):
    assert script.main(["--data-dir", str(tmp_path / "nope"), "--dry-run"]) == 2
    (tmp_path / "local_db").mkdir()
    assert script.main(["--data-dir", str(tmp_path), "--dry-run"]) == 0


def test_a_flattened_markdown_document_is_really_split():
    """Old passages were one line of words, and a markdown file starts with '# Title': that line must not be read as a
    heading (it used to come back as a single 480-word passage, so the document was 'converted' and still too big)."""
    flat = "# Reception Guide Sample data. ## About the clinic " + " ".join(f"w{i}" for i in range(480))
    old = legacy_chunks(flat)
    assert chunking.rechunk_needed(old)
    new = chunking.chunk_text(chunking.text_from_legacy_chunks(old))
    assert len(new) >= 4 and not chunking.rechunk_needed(new)
    assert all(len(c.split()) <= 120 for c in new)
    assert "w479" in new[-1] and "w0" in new[0]


def test_a_real_short_heading_is_still_a_heading():
    out = chunking.chunk_text("# Fees\nConsultation costs 500 rupees.\n# Hours\nOpen nine to five.")
    assert out == ["Fees\nConsultation costs 500 rupees.", "Hours\nOpen nine to five."]
