"""RAG store: chunking, embeddings (sentence-transformers or TF-IDF fallback), persistence."""
import json
import os
import re
import threading
from typing import List, Optional

from .common import DATA_DIR
from . import retrieval

RAG_DIR = os.path.join(DATA_DIR, "rag")
INDEX_FILE = os.path.join(RAG_DIR, "index.json")
_lock = threading.Lock()
_st_model = None
_st_failed = False


def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """RecursiveCharacterTextSplitter(512, 64); pure-python equivalent if langchain is missing."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap).split_text(text)
    except Exception:
        pass
    seps = ["\n\n", "\n", " ", ""]

    def split(t: str, level: int) -> List[str]:
        if len(t) <= chunk_size:
            return [t]
        sep = seps[level]
        parts = list(t) if sep == "" else t.split(sep)
        out: List[str] = []
        cur = ""
        for p in parts:
            piece = p if not cur else cur + sep + p
            if len(piece) <= chunk_size:
                cur = piece
                continue
            if cur:
                out.append(cur)
                cur = ""
            if len(p) > chunk_size and level + 1 < len(seps):
                sub = split(p, level + 1)
                out.extend(sub[:-1])
                cur = sub[-1]
            else:
                cur = p
        if cur:
            out.append(cur)
        return out

    pieces = [p for p in split(text, 0) if p.strip()]
    chunks: List[str] = []
    for p in pieces:
        if chunks and overlap:
            tail = chunks[-1][-overlap:]
            sp = tail.find(" ")
            tail = tail[sp + 1:] if sp != -1 else tail
            cand = (tail + " " + p).strip()
            chunks.append(cand if len(cand) <= chunk_size + overlap else p)
        else:
            chunks.append(p)
    return [c.strip() for c in chunks if c.strip()]


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join((pg.extract_text() or "") for pg in reader.pages).strip()
    return data.decode("utf-8", errors="replace").strip()


def _get_st():
    """The process-wide embedding model (shared with the agent knowledge search; see services/retrieval.py)."""
    global _st_model, _st_failed
    if _st_model is None and not _st_failed:
        try:
            _st_model = retrieval.load_sentence_transformer(retrieval.embedding_model_name())
        except Exception:
            _st_failed = True
    return _st_model


def embed(texts: List[str]) -> Optional[List[List[float]]]:
    if _get_st() is None:
        return None
    return [[float(x) for x in v] for v in retrieval.encode_texts(retrieval.embedding_model_name(), texts)]


_tok = re.compile(r"\w+", re.UNICODE)


def _tokens(t: str) -> List[str]:
    return [w for w in _tok.findall(t.lower()) if len(w) > 1]


def _load() -> dict:
    os.makedirs(RAG_DIR, exist_ok=True)
    if not os.path.exists(INDEX_FILE):
        return {"documents": {}}
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"documents": {}}


def _save(idx: dict) -> None:
    os.makedirs(RAG_DIR, exist_ok=True)
    tmp = INDEX_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f)
    os.replace(tmp, INDEX_FILE)


def add_document(doc_id: str, filename: str, chunks: List[str], user_id: str) -> dict:
    vecs = embed(chunks)
    doc = {"id": doc_id, "user_id": user_id, "filename": filename, "chunks": chunks, "embeddings": vecs,
           "embedder": "sentence-transformers" if vecs else "tfidf",
           "embed_model": retrieval.embedding_model_name() if vecs else None, "embed_dim": len(vecs[0]) if vecs else None}
    with _lock:
        idx = _load()
        idx["documents"][doc_id] = doc
        _save(idx)
    return doc


def list_documents(user_id: str) -> List[dict]:
    with _lock:
        docs = [d for d in _load()["documents"].values() if d.get("user_id") == user_id]
    return [{"id": d["id"], "filename": d["filename"], "chunks": len(d["chunks"]), "embedder": d["embedder"]}
            for d in docs]


def get_document(doc_id: str, user_id: str) -> Optional[dict]:
    with _lock:
        d = _load()["documents"].get(doc_id)
    return d if d and d.get("user_id") == user_id else None


def delete_document(doc_id: str, user_id: str) -> bool:
    with _lock:
        idx = _load()
        d = idx["documents"].get(doc_id)
        if not d or d.get("user_id") != user_id:
            return False
        del idx["documents"][doc_id]
        _save(idx)
    return True


def search(doc: dict, question: str, top_k: int = 4) -> List[dict]:
    """Top chunks of one document: dense cosine fused with BM25 (Reciprocal Rank Fusion). A document embedded by another
    model, or with another vector size, is searched with BM25 alone: vectors of different sizes are never compared."""
    chunks = doc["chunks"]
    if not chunks:
        return []
    sims: List[Optional[float]] = [None] * len(chunks)
    stored = doc.get("embeddings")
    usable = bool(stored) and len(stored) == len(chunks) and doc.get("embed_model", retrieval.embedding_model_name()) == retrieval.embedding_model_name()
    if usable:
        q = embed([question])
        if q and all(len(ev) == len(q[0]) for ev in stored):
            sims = [sum(a * b for a, b in zip(q[0], ev)) for ev in stored]
    order = retrieval.hybrid_order(question, chunks, sims)
    if not any(s is not None for s in sims):          # lexical only: matches first, then the rest in document order
        seen = set(order)
        order = order + [i for i in range(len(chunks)) if i not in seen]
    order = order[:top_k]
    return [{"index": i, "text": chunks[i], "score": round(float(sims[i] if sims[i] is not None else 0.0), 4)}
            for i in order]
