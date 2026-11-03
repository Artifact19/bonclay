"""Re-split stored knowledge documents whose passages are still the old 500 words (local storage only).

Used by `POST /api/agents/{id}/knowledge/rechunk` (one agent) and `backend/scripts/rechunk_knowledge.py` (every agent).
It takes the store and the embedding function as arguments, so it needs no running app and a test can pass fakes.

A document is skipped when none of its passages is bigger than twice the current chunk size (`chunking.rechunk_needed`),
so running it again does nothing. Each document is replaced in one write of the chunk table. The original line breaks
and headings of an old document were lost when it was first split, so the new passages are word windows without
headings; uploading the file again gives better passages.
"""
import logging
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    from services import chunking
except ImportError:  # pragma: no cover
    from backend.services import chunking

log = logging.getLogger("bonclay.rechunk")


def rechunk_documents(store, encode: Callable[[List[str]], Sequence], embed_model: str, *,
                      agent_id: Optional[str] = None, user_id: Optional[str] = None, dry_run: bool = False,
                      on_document: Optional[Callable[[dict], None]] = None) -> Dict[str, object]:
    """Re-chunk the old documents of one agent (`agent_id`) or of all agents. `user_id` limits it to that user's
    documents (and unowned ones). With `dry_run` nothing is embedded or written. Returns
    {"documents": converted (or that would be), "skipped": already fine, "failed": n, "chunks_before", "chunks_after",
    "errors": [...]}; `chunks_after` is the real count, and in a dry run the count the new split would give."""
    def mine(r: dict) -> bool:
        return ((agent_id is None or r.get("agent_id") == agent_id)
                and (user_id is None or r.get("user_id") in (user_id, None)))

    report = {"documents": 0, "skipped": 0, "failed": 0, "chunks_before": 0, "chunks_after": 0, "errors": []}
    all_chunks: Dict[str, List[dict]] = {}
    for c in store.select("document_chunks"):
        if c.get("kind") == "question":      # question rows (services/question_index.py) are not passages of the text
            continue
        all_chunks.setdefault(c.get("document_id"), []).append(c)
    for doc in store.select("documents", mine):
        old = sorted(all_chunks.get(doc["id"], []), key=lambda c: c.get("chunk_index") or 0)
        texts = [c.get("chunk_text") or "" for c in old]
        if not texts or not chunking.rechunk_needed(texts):
            report["skipped"] += 1
            continue
        try:
            new = chunking.chunk_text(chunking.text_from_legacy_chunks(texts))
            if not new:
                raise ValueError("no text could be rebuilt from the stored passages")
            if not dry_run:
                vectors = [np.asarray(v, dtype=np.float32).tolist() for v in encode(new)]
                if len(vectors) != len(new):
                    raise ValueError("the embedder returned the wrong number of vectors")
                store.replace_document_chunks(doc["id"], [
                    {"chunk_text": c, "chunk_index": i, "embedding": v, "embed_model": embed_model, "embed_dim": len(v)}
                    for i, (c, v) in enumerate(zip(new, vectors))], {"chunk_count": len(new)})
        except Exception as e:  # noqa: BLE001 - one bad document must not stop the rest; its old passages stay as they were
            log.warning("re-chunk failed for document %s: %s", doc.get("id"), e)
            report["failed"] += 1
            report["errors"].append({"document_id": doc.get("id"), "filename": doc.get("filename"), "error": str(e)[:200]})
            continue
        report["documents"] += 1
        report["chunks_before"] += len(texts)
        report["chunks_after"] += len(new)
        if on_document:
            on_document({"id": doc["id"], "agent_id": doc.get("agent_id"), "filename": doc.get("filename"),
                         "chunks_before": len(texts), "chunks_after": len(new), "dry_run": dry_run})
    return report
