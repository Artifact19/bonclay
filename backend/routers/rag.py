# Extra pip deps (all optional): langchain-text-splitters, sentence-transformers, scikit-learn, pypdf, httpx, python-multipart
"""RAG endpoints: upload PDF/TXT, list/delete documents, streamed Q&A over SSE."""
import asyncio
import json
import logging
import os
import uuid

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

try:
    from services import rag_store
    from services import groq_keys
    from services.llm import DEFAULT_GROQ_MODEL, ProviderHTTPError, _mark_gone, key_failure
    from services.auth import get_current_user_id
    from services.common import api_error
    from core.validation import read_limited, validate_pdf_upload
except ImportError:  # imported as backend.routers.rag
    from backend.services import rag_store
    from backend.services import groq_keys
    from backend.services.llm import DEFAULT_GROQ_MODEL, ProviderHTTPError, _mark_gone, key_failure
    from backend.services.auth import get_current_user_id
    from backend.services.common import api_error
    from backend.core.validation import read_limited, validate_pdf_upload

log = logging.getLogger("bonclay.rag")
router = APIRouter(prefix="/api/rag", tags=["rag"])

MAX_UPLOAD = 20 * 1024 * 1024
GROQ_MODEL = os.getenv("GROQ_MODEL") or os.getenv("GROQ_CHAT_MODEL") or DEFAULT_GROQ_MODEL


class AskRequest(BaseModel):
    doc_id: str
    question: str
    voice_id: Optional[str] = None


@router.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    name = file.filename or ""
    if not name.lower().endswith((".pdf", ".txt")):
        raise api_error(400, "Only PDF and TXT files are supported", "INVALID_FILE_TYPE", {"filename": name})
    if name.lower().endswith(".pdf"):
        data = await validate_pdf_upload(file, MAX_UPLOAD)  # extension, size cap (streamed), %PDF magic
    else:
        data = await read_limited(file, MAX_UPLOAD)
        if b"\x00" in data[:4096]:
            raise api_error(400, "File does not look like plain text", "INVALID_FILE_CONTENT", None)
    try:
        text = await asyncio.to_thread(rag_store.extract_text, name, data)
    except Exception as e:
        raise api_error(400, "Failed to parse file", "PARSE_FAILED", str(e))
    if not text:
        raise api_error(400, "No extractable text found", "NO_TEXT", None)
    chunks = rag_store.split_text(text, 512, 64)
    doc_id = str(uuid.uuid4())
    await asyncio.to_thread(rag_store.add_document, doc_id, name, chunks, user_id)
    return {"doc_id": doc_id, "chunks": len(chunks)}


@router.get("/documents")
async def documents(user_id: str = Depends(get_current_user_id)):
    return {"documents": await asyncio.to_thread(rag_store.list_documents, user_id)}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user_id: str = Depends(get_current_user_id)):
    if not await asyncio.to_thread(rag_store.delete_document, doc_id, user_id):
        raise api_error(404, "Document not found", "DOC_NOT_FOUND", {"doc_id": doc_id})
    return {"deleted": doc_id}


def _sse(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    lines = "\n".join(f"data: {ln}" for ln in payload.split("\n"))
    return f"event: {event}\n{lines}\n\n"


async def _groq_stream(question: str, context: str):
    """Stream the answer from Groq. With several Groq accounts (services/groq_keys.py) a 429 or a 401/403 before the first
    token repeats the request on the next account's key; with one key it is the single request it always was."""
    import httpx
    candidates = groq_keys.ordered_keys(GROQ_MODEL) or groq_keys.keys()
    if not candidates:
        raise RuntimeError("GROQ_API_KEY is not set")
    for n, k in enumerate(candidates):
        last = n + 1 == len(candidates)
        try:
            async for tok in _groq_stream_with(httpx, k["key"], question, context):
                yield tok
            return
        except _KeyRefused as refused:
            fail = key_failure(ProviderHTTPError(refused.code, refused.body, refused.headers))
            if fail is None:
                raise RuntimeError(f"Groq HTTP {refused.code}: {refused.body[:200]!r}")
            if fail[0] == "daily":
                _mark_gone(GROQ_MODEL, ProviderHTTPError(refused.code, refused.body, refused.headers), k["label"])
            elif fail[0] == "auth":
                groq_keys.mark_auth_failed(k["label"])
            elif len(candidates) > 1:
                groq_keys.cool_down(k["label"], GROQ_MODEL, fail[1])
            if last:
                raise RuntimeError(f"Groq HTTP {refused.code}: {refused.body[:200]!r}")
            log.warning("Groq key %s hit the %s limit for %s; trying key %s", k["label"],
                        {"daily": "daily", "minute": "per-minute", "auth": "auth"}[fail[0]], GROQ_MODEL,
                        candidates[n + 1]["label"])


class _KeyRefused(Exception):
    """Groq answered the stream request with an error status before any token."""

    def __init__(self, code: int, body: str, headers: dict):
        super().__init__(code)
        self.code, self.body, self.headers = code, body, headers


async def _groq_stream_with(httpx, api_key: str, question: str, context: str):
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "BonClay/1.0 (+https://github.com/Artifact19/bonclay)"}
    body = {
        "model": GROQ_MODEL, "stream": True, "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Answer using only the provided context. If the answer is not in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", "https://api.groq.com/openai/v1/chat/completions",
                                 headers=headers, json=body) as r:
            if r.status_code >= 400:
                raw = await r.aread()
                raise _KeyRefused(r.status_code, raw.decode(errors="replace"), dict(r.headers))
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                except Exception:
                    continue
                if delta:
                    yield delta


async def _extractive_stream(question: str, hits):
    import re
    qwords = set(re.findall(r"\w+", question.lower()))
    sents = []
    for h in hits:
        for s in re.split(r"(?<=[.!?])\s+", h["text"].replace("\n", " ")):
            if s.strip():
                sents.append((len(qwords & set(re.findall(r"\w+", s.lower()))), s.strip()))
    best = sorted(sents, key=lambda x: x[0], reverse=True)[:3]
    answer = " ".join(s for _, s in best) or hits[0]["text"][:400]
    for w in answer.split():
        yield w + " "
        await asyncio.sleep(0.02)


@router.post("/ask")
async def ask(req: AskRequest, user_id: str = Depends(get_current_user_id)):
    doc = await asyncio.to_thread(rag_store.get_document, req.doc_id, user_id)
    if doc is None:
        raise api_error(404, "Document not found", "DOC_NOT_FOUND", {"doc_id": req.doc_id})
    if not req.question.strip():
        raise api_error(400, "Question is empty", "EMPTY_QUESTION", None)

    async def gen():
        try:
            hits = await asyncio.to_thread(rag_store.search, doc, req.question, 4)
            if not hits:
                yield _sse("error", {"error": "No content to search", "code": "NO_CONTEXT", "details": None})
                return
            parts = []
            if groq_keys.has_any():
                try:
                    context = "\n\n".join(h["text"] for h in hits)
                    async for tok in _groq_stream(req.question, context):
                        parts.append(tok)
                        yield _sse("token", tok)
                except Exception as e:
                    if parts:
                        yield _sse("error", {"error": "LLM stream failed", "code": "LLM_ERROR", "details": str(e)})
                        return
                    async for tok in _extractive_stream(req.question, hits):
                        parts.append(tok)
                        yield _sse("token", tok)
            else:
                async for tok in _extractive_stream(req.question, hits):
                    parts.append(tok)
                    yield _sse("token", tok)
            yield _sse("sources", [{"index": h["index"], "score": h["score"], "text": h["text"][:300]} for h in hits])
            yield _sse("done", {"answer": "".join(parts).strip(), "doc_id": req.doc_id, "voice_id": req.voice_id})
        except Exception as e:
            yield _sse("error", {"error": "Ask failed", "code": "ASK_FAILED", "details": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
