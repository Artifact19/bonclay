"""Backfill for stored documents (local storage): structure-aware passages with headers, and the question index.

Used by `backend/scripts/rechunk_knowledge.py --with-structure` and `--questions`. Like services/rechunk.py it takes the
store and the embedding function as arguments, so it needs no running app and a test can pass fakes.

`restructure_documents` re-cuts each document with `chunking.chunk_document` and replaces its rows in one write (the
document's question rows go with them: their parents are gone). The original text is read from `source_dir` when a file with
the same name is there (best: tables and headings survive); otherwise it is rebuilt from the stored passages, whose line
breaks and overlaps are only partly recoverable (tables of an old PDF come out as plain lines).

`index_questions_for_documents` writes the question rows (services/question_index.py) for documents that have none.
"""
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    from services import chunking, question_index
except ImportError:  # pragma: no cover
    from backend.services import chunking, question_index

log = logging.getLogger("bonclay.restructure")

SOURCE_SUFFIXES = (".txt", ".md", ".pdf")


def is_question(row: dict) -> bool:
    return row.get("kind") == "question"


def is_structured(passages: List[dict]) -> bool:
    """True when the document's passages already carry a header field (it was ingested, or re-cut, by chunk_document)."""
    return bool(passages) and all("header" in p for p in passages)


def rebuild_text(chunk_texts: Sequence[str], max_overlap: int = 60, min_overlap: int = 6) -> str:
    """The document text rebuilt from its stored passages, in order. Passages overlap by a few words (20 now, 50 for the
    oldest ones): when the start of a passage repeats the end of the previous one (after an optional heading line) the
    repeated words are dropped. Line breaks inside passages are kept."""
    parts: List[str] = []
    prev_words: List[str] = []
    for c in chunk_texts:
        text = c or ""
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        words = [text[a:b] for a, b in spans]
        if prev_words and words:
            first_line_words = len(text.split("\n", 1)[0].split()) if "\n" in text else 0
            done = False
            for offset in dict.fromkeys((0, first_line_words)):
                for k in range(min(max_overlap, len(prev_words), len(words) - offset), min_overlap - 1, -1):
                    if prev_words[-k:] == words[offset:offset + k]:
                        a, b = spans[offset][0], spans[offset + k - 1][1]
                        text = text[:a] + text[b:].lstrip(" ")
                        done = True
                        break
                if done:
                    break
        parts.append(text.strip())
        prev_words = words
    return "\n".join(p for p in parts if p)


def find_source_text(source_dir: Optional[str], filename: str) -> Optional[str]:
    """Text of the original file named `filename` somewhere under `source_dir`, or None."""
    if not source_dir or not filename:
        return None
    want = os.path.basename(filename).lower()
    if not want.endswith(SOURCE_SUFFIXES):
        return None
    for root, _dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower() != want:
                continue
            path = os.path.join(root, f)
            try:
                if want.endswith(".pdf"):
                    import pypdf  # noqa: PLC0415
                    text = chunking.pdf_text_from_reader(pypdf.PdfReader(path))
                else:
                    with open(path, "r", encoding="utf-8-sig") as fh:
                        text = fh.read().replace("\x00", "")
            except Exception as e:  # noqa: BLE001
                log.warning("could not read %s: %s", path, e)
                continue
            if text.strip():
                return text.strip()
    return None


def _group_rows(store) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for c in store.select("document_chunks"):
        grouped.setdefault(c.get("document_id"), []).append(c)
    return grouped


def _mine(agent_id: Optional[str], user_id: Optional[str]):
    def mine(r: dict) -> bool:
        return ((agent_id is None or r.get("agent_id") == agent_id)
                and (user_id is None or r.get("user_id") in (user_id, None)))
    return mine


def restructure_documents(store, encode: Callable[[List[str]], Sequence], embed_model: str, *,
                          agent_id: Optional[str] = None, user_id: Optional[str] = None, dry_run: bool = False,
                          source_dir: Optional[str] = None, force: bool = False,
                          on_document: Optional[Callable[[dict], None]] = None) -> Dict[str, object]:
    """Re-cut documents with `chunking.chunk_document` (headers, FAQ pairs, table rows, sentence passages). A document whose
    passages already have a header is skipped unless `force`. With `dry_run` nothing is embedded or written. Returns
    {"documents", "skipped", "failed", "chunks_before", "chunks_after", "from_source", "errors"}."""
    report = {"documents": 0, "skipped": 0, "failed": 0, "chunks_before": 0, "chunks_after": 0, "from_source": 0,
              "errors": []}
    grouped = _group_rows(store)
    for doc in store.select("documents", _mine(agent_id, user_id)):
        rows = sorted(grouped.get(doc["id"], []), key=lambda c: c.get("chunk_index") or 0)
        passages = [r for r in rows if not is_question(r)]
        if not passages or (is_structured(passages) and not force):
            report["skipped"] += 1
            continue
        try:
            source = find_source_text(source_dir, doc.get("filename") or "")
            text = source if source is not None else rebuild_text([p.get("chunk_text") or "" for p in passages])
            title = chunking.guess_title(text, doc.get("filename") or "")
            new = chunking.chunk_document(text, title)
            if not new:
                raise ValueError("no text could be rebuilt from the stored passages")
            if not dry_run:
                vectors = [np.asarray(v, dtype=np.float32).tolist()
                           for v in encode([chunking.embedding_text(p["header"], p["content"]) for p in new])]
                if len(vectors) != len(new):
                    raise ValueError("the embedder returned the wrong number of vectors")
                store.replace_document_chunks(doc["id"], [
                    {"chunk_text": p["content"], "header": p["header"], "kind": "passage", "chunk_index": i,
                     "embedding": v, "embed_model": embed_model, "embed_dim": len(v)}
                    for i, (p, v) in enumerate(zip(new, vectors))],
                    {"chunk_count": len(new), "question_count": 0, "question_index": "none"})
        except Exception as e:  # noqa: BLE001 - one bad document must not stop the rest; its old rows stay as they were
            log.warning("restructure failed for document %s: %s", doc.get("id"), e)
            report["failed"] += 1
            report["errors"].append({"document_id": doc.get("id"), "filename": doc.get("filename"), "error": str(e)[:200]})
            continue
        report["documents"] += 1
        report["from_source"] += 1 if source is not None else 0
        report["chunks_before"] += len(passages)
        report["chunks_after"] += len(new)
        if on_document:
            on_document({"id": doc["id"], "agent_id": doc.get("agent_id"), "filename": doc.get("filename"),
                         "chunks_before": len(passages), "chunks_after": len(new), "dry_run": dry_run,
                         "from_source": source is not None})
    return report


def index_questions_for_documents(store, encode: Callable[[List[str]], Sequence], embed_model: Optional[str] = None, *,
                                  agent_id: Optional[str] = None,
                                  user_id: Optional[str] = None, dry_run: bool = False, redo: bool = False,
                                  chat: Optional[Callable] = None, pacer: Optional[question_index.Pacer] = None,
                                  sleep: Callable[[float], None] = question_index._sleep,
                                  local: Optional[bool] = None, local_chat: Optional[Callable] = None,
                                  on_document: Optional[Callable[[dict], None]] = None) -> Dict[str, object]:
    """Write question rows for documents that have none (all with `redo`, which first removes the old ones). In a dry run
    nothing is generated or written: the report gives how many passages and model calls it would take. Needs a hosted
    or the local background model for a real run (`question_index.available()`), unless `chat` is given; `local=True`
    writes every question with the local model. Returns {"documents", "skipped",
    "failed", "passages", "questions", "calls", "status", "errors"}; `status` is "ok", or "no hosted language model"."""
    report = {"documents": 0, "skipped": 0, "failed": 0, "passages": 0, "questions": 0, "calls": 0, "status": "ok",
              "errors": []}
    if not dry_run and chat is None and local_chat is None:
        if local:
            reason = question_index.background_llm.unavailable_reason()
            if reason:
                report["status"] = "no local language model (" + reason + ")"
                return report
        elif not question_index.available():
            report["status"] = "no hosted language model (set GROQ_API_KEY)"
            return report
    grouped = _group_rows(store)
    for doc in store.select("documents", _mine(agent_id, user_id)):
        rows = sorted(grouped.get(doc["id"], []), key=lambda c: c.get("chunk_index") or 0)
        passages = [r for r in rows if not is_question(r)]
        have = [r for r in rows if is_question(r)]
        if not passages or (have and not redo):
            report["skipped"] += 1
            continue
        items = [{"id": p["id"], "header": p.get("header") or "", "content": p.get("chunk_text") or ""} for p in passages]
        calls = len(question_index.make_batches(items))
        report["passages"] += len(items)
        report["calls"] += calls
        if dry_run:
            report["documents"] += 1
            if on_document:
                on_document({"id": doc["id"], "filename": doc.get("filename"), "passages": len(items), "calls": calls,
                             "questions": 0, "dry_run": True, "status": "would index"})
            continue
        if have:
            store.delete("document_chunks", lambda r, d=doc["id"]: r.get("document_id") == d and is_question(r))
        state = {"n": 0}

        def save(qrows, doc_id=doc["id"], state=state):
            stored = store.add_question_rows(doc_id, [
                {"chunk_text": q["question"], "content": q["question"], "kind": "question",
                 "parent_chunk_id": q["parent_chunk_id"], "chunk_index": 0, "embedding": q["embedding"],
                 "embed_model": embed_model or _embed_model_of(passages), "embed_dim": len(q["embedding"])} for q in qrows],
                {"question_count": state["n"] + len(qrows), "question_index": "indexing"})
            state["n"] += stored
            return stored

        result = question_index.index_passages(items, embed=encode, save=save, chat=chat, pacer=pacer, sleep=sleep,
                                               local=local, local_chat=local_chat)
        store.update("documents", doc["id"], {"question_count": state["n"], "question_index": result["status"]})
        if result["status"] in ("done",):
            report["documents"] += 1
        elif result["status"] == "skipped":
            report["status"] = result["reason"] or "skipped"
            report["skipped"] += 1
        else:
            report["failed"] += 1
            report["errors"].append({"document_id": doc.get("id"), "filename": doc.get("filename"),
                                     "error": result["reason"]})
        report["questions"] += state["n"]
        if on_document:
            on_document({"id": doc["id"], "filename": doc.get("filename"), "passages": len(items), "calls": calls,
                         "questions": state["n"], "dry_run": False, "status": result["status"]})
        if result["status"] == "skipped":
            break                      # no hosted model: every other document would be skipped the same way
    return report


def _embed_model_of(passages: List[dict]) -> Optional[str]:
    """The model that embedded the document's passages (question rows must use the same one)."""
    for p in passages:
        if p.get("embed_model"):
            return p["embed_model"]
    return None
