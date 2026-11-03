"""A small JSON-file store so agents, conversations, knowledge and evaluations work without Supabase.

Mirrors the Supabase tables the API uses (agents, documents, document_chunks, conversations, conversation_turns,
evaluation_runs) as one JSON file per table under backend/data/local_db/. Every row carries `user_id`, and every
read is scoped by it. Writes are atomic (temp file + rename) and serialised by a process-wide lock. Embeddings are
stored as plain float lists and searched with a numpy cosine similarity, which is fast enough for the few thousand
chunks a local project holds.
"""
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

TABLES = ("agents", "documents", "document_chunks", "conversations", "conversation_turns", "evaluation_runs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalStore:
    def __init__(self, root: str):
        self.root = os.path.join(root, "local_db")
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.RLock()

    # -- file I/O ---------------------------------------------------------------------------------

    def _path(self, table: str) -> str:
        if table not in TABLES:
            raise ValueError(f"unknown table {table}")
        return os.path.join(self.root, f"{table}.json")

    def _load(self, table: str) -> List[dict]:
        path = self._path(table)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def signature(self, *tables: str) -> tuple:
        """A cheap stand-in for "did these tables change?": (modification time, size, inode) of each table's file (a
        write replaces the file, so its inode and time change). Lets a caller skip re-reading a table it has already
        checked."""
        out = []
        for table in tables:
            try:
                st = os.stat(self._path(table))
                out.append((st.st_mtime_ns, st.st_size, st.st_ino))
            except OSError:
                out.append(None)
        return tuple(out)

    def _save(self, table: str, rows: List[dict]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        os.replace(tmp, self._path(table))

    # -- generic CRUD -----------------------------------------------------------------------------

    def insert(self, table: str, row: dict) -> dict:
        row = {"id": str(uuid.uuid4()), "created_at": _now(), **row}
        with self._lock:
            rows = self._load(table)
            rows.append(row)
            self._save(table, rows)
        return row

    def insert_many(self, table: str, new_rows: List[dict]) -> List[dict]:
        stamped = [{"id": str(uuid.uuid4()), "created_at": _now(), **r} for r in new_rows]
        with self._lock:
            rows = self._load(table)
            rows.extend(stamped)
            self._save(table, rows)
        return stamped

    def select(self, table: str, where: Optional[Callable[[dict], bool]] = None, *, order: str = "created_at",
               desc: bool = False, limit: Optional[int] = None) -> List[dict]:
        with self._lock:
            rows = [r for r in self._load(table) if where is None or where(r)]
        rows.sort(key=lambda r: r.get(order) or "", reverse=desc)
        return rows[:limit] if limit else rows

    def select_last(self, table: str, where: Callable[[dict], bool], n: int, *, order: str = "created_at") -> List[dict]:
        """The newest `n` matching rows, oldest first (the local twin of order desc + limit n, then reversed)."""
        if n <= 0:
            return []
        return self.select(table, where, order=order)[-n:]

    def get(self, table: str, row_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        for r in self.select(table, lambda r: r.get("id") == row_id):
            if user_id is None or r.get("user_id") == user_id:
                return r
        return None

    def update(self, table: str, row_id: str, fields: Dict[str, Any], user_id: Optional[str] = None) -> Optional[dict]:
        with self._lock:
            rows = self._load(table)
            for r in rows:
                if r.get("id") == row_id and (user_id is None or r.get("user_id") == user_id):
                    r.update(fields)
                    self._save(table, rows)
                    return r
        return None

    def delete(self, table: str, where: Callable[[dict], bool]) -> int:
        with self._lock:
            rows = self._load(table)
            keep = [r for r in rows if not where(r)]
            self._save(table, keep)
        return len(rows) - len(keep)

    # -- domain helpers ---------------------------------------------------------------------------

    def delete_agent(self, agent_id: str, user_id: str) -> bool:
        """Delete an agent and everything that hangs off it (like ON DELETE CASCADE)."""
        if not self.get("agents", agent_id, user_id):
            return False
        with self._lock:
            doc_ids = {d["id"] for d in self._load("documents") if d.get("agent_id") == agent_id}
            conv_ids = {c["id"] for c in self._load("conversations") if c.get("agent_id") == agent_id}
            self.delete("document_chunks", lambda r: r.get("document_id") in doc_ids)
            self.delete("documents", lambda r: r.get("agent_id") == agent_id)
            self.delete("conversation_turns", lambda r: r.get("conversation_id") in conv_ids)
            self.delete("conversations", lambda r: r.get("agent_id") == agent_id)
            self.delete("evaluation_runs", lambda r: r.get("agent_id") == agent_id)
            self.delete("agents", lambda r: r.get("id") == agent_id)
        return True

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self.delete("document_chunks", lambda r: r.get("document_id") == document_id)
            self.delete("documents", lambda r: r.get("id") == document_id)

    def replace_document_chunks(self, document_id: str, new_rows: List[dict], doc_fields: Optional[dict] = None) -> List[dict]:
        """Swap all rows of one document (its passages and any question rows) for `new_rows` in a single write of the chunk
        table (a crash can never leave the document with no passages), then apply `doc_fields` to the document row.
        Returns the stored rows (with their new ids)."""
        stamped = [{"id": str(uuid.uuid4()), "created_at": _now(), **r, "document_id": document_id} for r in new_rows]
        with self._lock:
            rows = [r for r in self._load("document_chunks") if r.get("document_id") != document_id]
            rows.extend(stamped)
            self._save("document_chunks", rows)
            if doc_fields:
                self.update("documents", document_id, doc_fields)
        return stamped

    def add_question_rows(self, document_id: str, new_rows: List[dict], doc_fields: Optional[dict] = None) -> int:
        """Append question rows (`kind="question"`, `parent_chunk_id`) to a document that still exists, in one write. Rows
        whose parent passage is gone (the document was re-chunked meanwhile) are dropped. Returns how many were stored, 0
        when the document no longer exists. `doc_fields` (e.g. the question count) are applied to the document row."""
        with self._lock:
            if not any(d.get("id") == document_id for d in self._load("documents")):
                return 0
            rows = self._load("document_chunks")
            parents = {r["id"] for r in rows if r.get("document_id") == document_id and r.get("kind") != "question"}
            stamped = [{"id": str(uuid.uuid4()), "created_at": _now(), **r, "document_id": document_id}
                       for r in new_rows if r.get("parent_chunk_id") in parents]
            if stamped:
                rows.extend(stamped)
                self._save("document_chunks", rows)
            if doc_fields:
                self.update("documents", document_id, doc_fields)
            return len(stamped)

    def update_many(self, table: str, updates: Dict[str, Dict[str, Any]]) -> int:
        """Apply {row_id: fields} in one read and one write (re-embedding thousands of chunks must not rewrite the file
        once per chunk). Returns how many rows changed."""
        if not updates:
            return 0
        with self._lock:
            rows = self._load(table)
            n = 0
            for r in rows:
                fields = updates.get(r.get("id"))
                if fields is not None:
                    r.update(fields)
                    n += 1
            if n:
                self._save(table, rows)
        return n

    def match_chunks(self, agent_id: str, query_embedding, k: int = 3, query_text: Optional[str] = None, *,
                     query_m3=None, info: Optional[dict] = None) -> List[dict]:
        """Top-k PASSAGES of an agent: the match_document_chunks RPC row shape (id, document_id, chunk_text,
        chunk_index, similarity) plus `header`, `score` (fused, 0..1), `signals` and `via` (see retrieval.rank_rows).

        Dense: cosine over the rows whose embedding has the same dimension as the query (a row from another embedding
        model is never compared, whatever its length). With `query_text` and HYBRID_SEARCH on, BM25 over header + text
        (and, with `query_m3`, BGE-M3's sparse and multi-vector scores) are fused in (RETRIEVAL_FUSION, weighted by
        default); rows without a usable embedding can still be found through BM25. Question rows (kind "question",
        the question index) are scored too but stand for their parent passage; they are never returned.
        `similarity` is the dense cosine (the best of the passage and its questions; None when not comparable).
        `info`, when given, receives what the fusion did (for retrieval.assess_confidence)."""
        try:
            from services import retrieval  # noqa: PLC0415
        except ImportError:  # imported as backend.services.local_store
            from backend.services import retrieval  # noqa: PLC0415
        doc_ids = {d["id"] for d in self.select("documents", lambda r: r.get("agent_id") == agent_id)}
        chunks = self.select("document_chunks", lambda r: r.get("document_id") in doc_ids)
        if not chunks or k <= 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        sims: List[Optional[float]] = [None] * len(chunks)
        usable = [i for i, c in enumerate(chunks)
                  if c.get("embedding") is not None and len(c["embedding"]) == q.shape[0]]
        if usable:
            m = np.asarray([chunks[i]["embedding"] for i in usable], dtype=np.float32)
            cos = (m @ q) / (np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-9)
            for i, s in zip(usable, cos):
                sims[i] = float(s)
        ranked = retrieval.rank_rows(query_text or "", chunks, sims, query_m3=query_m3, info=info)
        out = []
        for e in ranked[:k]:
            c = chunks[e["index"]]
            out.append({"id": c["id"], "document_id": c["document_id"], "chunk_text": c["chunk_text"],
                        "chunk_index": c.get("chunk_index", 0), "header": c.get("header") or None,
                        "similarity": e["similarity"], "score": e["score"], "signals": e["signals"], "via": e["via"]})
        return out
