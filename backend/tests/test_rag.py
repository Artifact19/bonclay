"""RAG router against a fixture TXT, with embeddings disabled (lexical path)."""
import pytest

pytest.importorskip("httpx")
pytest.importorskip("multipart", reason="python-multipart needed for uploads")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

rag = pytest.importorskip("routers.rag")
rag_store = pytest.importorskip("services.rag_store")
errors = pytest.importorskip("core.errors")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_store, "RAG_DIR", str(tmp_path))
    monkeypatch.setattr(rag_store, "INDEX_FILE", str(tmp_path / "index.json"))
    monkeypatch.setattr(rag_store, "embed", lambda texts: None)  # never download a model
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app = FastAPI()
    errors.register_exception_handlers(app)
    app.include_router(rag.router)
    return TestClient(app)


def upload(client, path):
    return client.post("/api/rag/upload", files={"file": (path.name, path.read_bytes(), "text/plain")})


def test_upload_list_delete(client, rag_txt):
    r = upload(client, rag_txt)
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] >= 1
    docs = client.get("/api/rag/documents").json()["documents"]
    assert any(d.get("id") == body["doc_id"] for d in docs)
    assert client.delete(f"/api/rag/documents/{body['doc_id']}").status_code == 200
    assert client.delete(f"/api/rag/documents/{body['doc_id']}").status_code == 404


def test_ask_streams_sources_and_done(client, rag_txt):
    doc_id = upload(client, rag_txt).json()["doc_id"]
    r = client.post("/api/rag/ask", json={"doc_id": doc_id, "question": "What is the refund window?"})
    assert r.status_code == 200
    text = r.text
    assert "event: token" in text and "event: sources" in text and "event: done" in text
    assert "refund" in text.lower()


def test_ask_unknown_doc_and_empty_question(client, rag_txt):
    r = client.post("/api/rag/ask", json={"doc_id": "nope", "question": "hi"})
    assert r.status_code == 404 and r.json()["code"] == "DOC_NOT_FOUND"
    doc_id = upload(client, rag_txt).json()["doc_id"]
    r = client.post("/api/rag/ask", json={"doc_id": doc_id, "question": "   "})
    assert r.status_code == 400 and r.json()["code"] == "EMPTY_QUESTION"


def test_rejects_bad_type_and_empty_text(client):
    r = client.post("/api/rag/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400 and r.json()["code"] == "INVALID_FILE_TYPE"
    r = client.post("/api/rag/upload", files={"file": ("empty.txt", b"   ", "text/plain")})
    assert r.status_code == 400
