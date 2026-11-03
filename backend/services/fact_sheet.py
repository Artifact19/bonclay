"""Per-agent fact sheet: the hard facts of an agent's documents (timings per department and day, fees, phone numbers,
addresses, doctor names and days) as a short key-value list, put in front of the model as "Key facts".

Built once per document set by the hosted language model (JSON, "copy verbatim"), then every value is checked against
the document text: a value that does not appear there word for word is dropped. The sheet is cached in memory and on
disk (FACT_SHEET_DIR, default backend/data/fact_sheets/, git-ignored) under the documents' fingerprint, so it is built
again only when a document is added or deleted.

The live turn never waits for it: `turn_facts` answers from memory in well under a millisecond and, when the sheet is
missing or due for a check, starts a background refresh (disk first, the model only for a new document set). Without a
hosted model (offline) nothing is built and the turn simply has no key facts.

FACT_SHEET=on|off (default on).

When the hosted model is rate limited, out of quota or not configured, the local background model (services/background_llm.py:
BACKGROUND_LOCAL_MODEL, default Gemma 4 E4B) extracts the facts instead, only while the app is idle and memory allows it
(otherwise the build waits and is tried again later). Its JSON is repaired when it has slips, and every value is still
checked against the document text, so a wrong or invented value is dropped whichever model wrote it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from . import background_llm, grounding

log = logging.getLogger("bonclay.fact_sheet")

EXTRACT_VERSION = "1"        # bump when the extraction prompt or validation changes: old sheets are rebuilt
MAX_FACTS = 80               # kept per agent
MAX_PROMPT_FACTS = 10        # put in one turn's prompt
BATCH_CHARS = 12000          # document text per extraction request
MAX_BATCHES = 6              # requests per document set (about 70k characters); the rest is left out
RECHECK_S = 30.0             # how long a sheet is trusted before the documents' fingerprint is read again
RETRY_AFTER_FAIL_S = 600.0   # a failed build (model down, bad JSON) is tried again after this long

_AGENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def enabled() -> bool:
    return (os.getenv("FACT_SHEET", "on") or "on").strip().lower() not in ("off", "0", "false", "no")


def cache_dir() -> Path:
    override = os.getenv("FACT_SHEET_DIR")
    return Path(override) if override else Path(__file__).resolve().parents[1] / "data" / "fact_sheets"


# --------------------------------------------------------------------------------------------- the documents

Loader = Callable[[str, str, str], Tuple[Optional[str], Callable[[], List[str]]]]
_loader_override: Optional[Loader] = None


def set_loader(fn: Optional[Loader]) -> None:
    """Replace how an agent's documents are read (tests). `fn(agent_id, user_id, storage_mode)` returns
    (fingerprint or None, a function returning the documents' texts)."""
    global _loader_override
    _loader_override = fn


def _backend():
    import sys  # noqa: PLC0415
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "local_db"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "local_db"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


def _fingerprint(docs: Sequence[dict]) -> str:
    parts = sorted(f"{d.get('id')}:{d.get('filename') or ''}" for d in docs)
    return hashlib.sha1(("v" + EXTRACT_VERSION + "\n" + "\n".join(parts)).encode("utf-8")).hexdigest()[:16]


def _passage(row: dict) -> str:
    body = str(row.get("chunk_text") or row.get("content") or "")
    header = str(row.get("header") or "").strip()
    return f"{header}\n{body}" if header and not body.startswith(header) else body


def _join_doc(rows: List[dict]) -> str:
    rows = [r for r in rows if (r.get("kind") or "passage") == "passage"]  # contract 3: question rows are not text
    rows.sort(key=lambda r: int(r.get("chunk_index") or 0))
    return "\n".join(_passage(r) for r in rows)


def _default_loader(agent_id: str, user_id: str, storage_mode: str):
    B = _backend()
    if storage_mode == "local" or getattr(B, "supabase", None) is None:
        docs = B.local_db.select("documents", lambda r: r.get("agent_id") == agent_id)
        ids = [d["id"] for d in docs]

        def texts() -> List[str]:
            wanted = set(ids)
            rows = B.local_db.select("document_chunks", lambda r: r.get("document_id") in wanted)
            return [_join_doc([r for r in rows if r.get("document_id") == i]) for i in ids]
        return (_fingerprint(docs) if docs else None), texts
    sb = B.supabase
    res = sb.table("documents").select("id, filename").eq("agent_id", agent_id).eq("user_id", user_id).execute()
    docs = res.data or []
    ids = [d["id"] for d in docs]

    def cloud_texts() -> List[str]:
        out = []
        for i in ids:
            try:
                rows = (sb.table("document_chunks").select("document_id, chunk_index, chunk_text, kind")
                        .eq("document_id", i).execute().data or [])
            except Exception:  # noqa: BLE001 - the `kind` column (question rows) may not exist
                rows = (sb.table("document_chunks").select("document_id, chunk_index, chunk_text")
                        .eq("document_id", i).execute().data or [])
            out.append(_join_doc(rows))
        return out
    return (_fingerprint(docs) if docs else None), cloud_texts


# --------------------------------------------------------------------------------------------- extraction

EXTRACT_PROMPT = """You extract the hard facts from a business's documents for its phone assistant.
Return JSON only: {"facts": [{"key": "...", "value": "..."}]}.
Take only facts with a concrete value: opening hours and timings (per department and per day), fees and prices, phone
numbers, e-mail addresses, postal addresses, names of doctors or staff with their department and the days they are in,
dates and deadlines.
- "key": a short English label of what the value is about, naming the department, day or item (for example
  "Cardiology OPD timings, Monday to Friday" or "Consultation fee, general physician"). Use only words from the text.
- "value": COPIED VERBATIM from the text, character for character: do not reformat, translate, convert, shorten or
  combine values from different places. If you cannot copy it exactly, leave the fact out.
- At most 60 facts. No facts: {"facts": []}."""


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def validate(facts: Any, source: str) -> List[dict]:
    """Keep only well-formed facts whose value appears verbatim in `source` (whitespace and letter case aside) and whose
    key names no number the source does not have. Duplicates are dropped."""
    if not isinstance(facts, list):
        return []
    hay = _norm_ws(source).lower()
    index = grounding.SourceIndex([source])
    out: List[dict] = []
    seen: Set[Tuple[str, str]] = set()
    for f in facts:
        if not isinstance(f, dict):
            continue
        key, value = _norm_ws(str(f.get("key") or "")), _norm_ws(str(f.get("value") or ""))
        if not key or not value or len(key) > 100 or len(value) > 200:
            continue
        if value.lower() not in hay:
            continue
        if grounding.ungrounded_values(key, index):
            continue
        sig = (key.lower(), value.lower())
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"key": key, "value": value})
    return out


def _batches(texts: Sequence[str]) -> List[str]:
    out: List[str] = []
    cur = ""
    for text in texts:
        text = text or ""
        while text:
            room = BATCH_CHARS - len(cur)
            piece, text = text[:room], text[room:]
            cur = (cur + "\n\n" + piece) if cur else piece
            if len(cur) >= BATCH_CHARS:
                out.append(cur)
                cur = ""
    if cur.strip():
        out.append(cur)
    return out[:MAX_BATCHES]


def hosted_llm_available() -> bool:
    try:
        from . import llm  # noqa: PLC0415
        return any(name in ("groq", "openai") and llm.PROVIDERS[name][0]() is None for name in llm._order())
    except Exception:  # noqa: BLE001
        return False


def local_fallback_ready() -> bool:
    try:
        return background_llm.local_ready()
    except Exception:  # noqa: BLE001
        return False


def _hosted_then_local(messages, *, local: bool = False):
    """The hosted model first; when it cannot answer (429, daily quota, failure, not configured) the local background model,
    which waits for an idle app and enough memory (background_llm.run_local). `local=True` skips the hosted call."""
    from . import llm  # noqa: PLC0415
    if not local:
        try:
            return llm.chat(messages, json_mode=True, temperature=0.0, max_tokens=2000, timeout=60.0,
                            providers=["groq", "openai"])
        except llm.LLMUnavailable:
            if not local_fallback_ready():
                raise
            log.info("fact sheet: the hosted model is not available; using the local model")
    return background_llm.run_local(messages, max_tokens=2400, temperature=0.0, sleep=time.sleep)


def build_facts(texts: Sequence[str], chat: Optional[Callable[..., Any]] = None, *,
                local: bool = False) -> Tuple[List[dict], Optional[str]]:
    """The validated facts of `texts` (blocking: one model request per ~12k characters). Returns (facts, model).
    Without `chat` the hosted model is asked, then the local background model (always the local one with `local=True`);
    the local model's JSON is repaired (background_llm.parse_json) before it is read."""
    if chat is None:
        def chat(messages):
            return _hosted_then_local(messages, local=local)
    facts: List[dict] = []
    model = None
    source = "\n\n".join(t for t in texts if t)
    for batch in _batches(texts):
        res = chat([{"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": "Documents:\n" + batch}])
        model = getattr(res, "model", None) or model
        try:
            from . import llm  # noqa: PLC0415
            reply = getattr(res, "text", res) if not isinstance(res, str) else res
            try:
                data = llm.extract_json(reply)
            except Exception:  # noqa: BLE001 - a small model's slips: fences, trailing commas, a cut-off reply
                data = background_llm.parse_json(reply)
            if isinstance(data, list):
                data = {"facts": data}
        except Exception as e:  # noqa: BLE001
            log.warning("fact sheet: the model's reply was not JSON: %s", e)
            continue
        facts += validate(data.get("facts"), source)
    return validate(facts, source)[:MAX_FACTS], model


# --------------------------------------------------------------------------------------------- cache

@dataclass
class _Entry:
    fingerprint: Optional[str]
    facts: List[dict]
    checked: float
    tokens: List[Tuple[Set[str], Set[str]]] = field(default_factory=list)  # (key tokens, value tokens) per fact


_mem: Dict[Tuple[str, str, str], _Entry] = {}
_failed: Dict[Tuple[str, str, str, str], float] = {}
_running: Set[Tuple[str, str, str]] = set()
_lock = threading.Lock()
_build_slot = threading.Semaphore(1)  # one model build at a time, whatever the number of agents


def _path(agent_id: str) -> Optional[Path]:
    return cache_dir() / f"{agent_id}.json" if _AGENT_ID_RE.match(agent_id or "") else None


def _read_disk(agent_id: str, fingerprint: str) -> Optional[List[dict]]:
    p = _path(agent_id)
    if p is None or not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("fingerprint") != fingerprint or not isinstance(data.get("facts"), list):
        return None
    return [f for f in data["facts"] if isinstance(f, dict) and f.get("key") and f.get("value")]


def _write_disk(agent_id: str, fingerprint: str, facts: List[dict], model: Optional[str]) -> None:
    p = _path(agent_id)
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerprint": fingerprint, "version": EXTRACT_VERSION, "model": model,
                                   "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "facts": facts},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        log.warning("fact sheet: could not save %s: %s", p, e)


def _tokens(text: str) -> Set[str]:
    return grounding._tokens(text)


def _entry(fingerprint: Optional[str], facts: List[dict]) -> _Entry:
    return _Entry(fingerprint, facts, time.monotonic(),
                  [(_tokens(f["key"]), _tokens(f["value"])) for f in facts])


def refresh(agent_id: str, user_id: str, storage_mode: str = "local", *, chat: Optional[Callable[..., Any]] = None) -> List[dict]:
    """Blocking: bring the agent's sheet up to date (disk cache, else a model build) and return its facts."""
    key = (agent_id, user_id, storage_mode)
    loader = _loader_override or _default_loader
    fingerprint, texts = loader(agent_id, user_id, storage_mode)
    with _lock:
        cur = _mem.get(key)
    if fingerprint is None:  # no documents (or they cannot be read): no facts
        with _lock:
            _mem[key] = _entry(None, [])
        return []
    if cur is not None and cur.fingerprint == fingerprint:
        cur.checked = time.monotonic()
        return cur.facts
    facts = _read_disk(agent_id, fingerprint)
    if facts is None:
        failed_at = _failed.get((*key, fingerprint))
        if failed_at is not None and time.monotonic() - failed_at < RETRY_AFTER_FAIL_S:
            facts = None
        elif chat is None and not hosted_llm_available() and not local_fallback_ready():
            facts = None  # offline: no sheet (tried again at the next check)
        else:
            with _build_slot:
                try:
                    facts, model = build_facts(texts(), chat)
                    _write_disk(agent_id, fingerprint, facts, model)
                    log.info("fact sheet for agent %s: %d facts", agent_id, len(facts))
                except Exception as e:  # noqa: BLE001
                    log.warning("fact sheet for agent %s not built: %s", agent_id, e)
                    _failed[(*key, fingerprint)] = time.monotonic()
                    facts = None
                finally:
                    if chat is None:
                        background_llm.release()   # a local model used for this build does not stay in memory
    with _lock:
        if facts is None:
            # Nothing from an older document set is kept (a deleted document's facts must not linger); the next check
            # (RECHECK_S later) tries again.
            _mem[key] = _entry(None, [])
        else:
            _mem[key] = _entry(fingerprint, facts)
    return facts or []


def _refresh_bg(key: Tuple[str, str, str]) -> None:
    try:
        refresh(*key)
    except Exception as e:  # noqa: BLE001 - a fact sheet is an extra, never worth a failure
        log.warning("fact sheet refresh failed: %s", e)
    finally:
        with _lock:
            _running.discard(key)


def prefetch(agent_id: str, user_id: str, storage_mode: str = "local") -> None:
    """Start a background refresh of the agent's sheet unless one is running or it was checked recently."""
    if not enabled() or not agent_id:
        return
    key = (agent_id, user_id, storage_mode)
    with _lock:
        cur = _mem.get(key)
        if key in _running or (cur is not None and time.monotonic() - cur.checked < RECHECK_S):
            return
        _running.add(key)
    threading.Thread(target=_refresh_bg, args=(key,), name="fact-sheet", daemon=True).start()


def facts_for(agent_id: str, user_id: str, storage_mode: str = "local") -> _Entry:
    """Non-blocking: the sheet in memory now (maybe empty) and a background refresh when it is due."""
    prefetch(agent_id, user_id, storage_mode)
    with _lock:
        return _mem.get((agent_id, user_id, storage_mode)) or _Entry(None, [], 0.0)


# --------------------------------------------------------------------------------------------- per turn

# Words a caller uses for a kind of fact, mapped to the words a fact's key uses (both normalised by the retrieval
# tokenizer, so Devanagari, Hinglish and English spellings meet).
_TOPICS = {
    "timing": "time timing timings hour hours open opens close closes khulta khulti khulega band samay kab baje "
              "समय कब खुलता खुलती बंद बजे",
    "fee": "fee fees charge charges price cost kitna paisa paise rupay rupees shulk फीस शुल्क कितना पैसे रुपये",
    "phone": "phone number contact call mobile helpline nambar फोन नंबर संपर्क",
    "address": "address location where kahan pata situated directions पता कहाँ कहां",
    "doctor": "doctor dr consultant specialist available daktar डॉक्टर",
    "day": "monday tuesday wednesday thursday friday saturday sunday somvar mangalvar budhvar guruvar shukravar "
           "shanivar ravivar सोमवार मंगलवार बुधवार गुरुवार शुक्रवार शनिवार रविवार",
}
_topic_tokens: Optional[Dict[str, Set[str]]] = None


def _expand(tokens: Set[str]) -> Set[str]:
    global _topic_tokens
    if _topic_tokens is None:
        _topic_tokens = {k: _tokens(v) for k, v in _TOPICS.items()}
    out = set(tokens)
    for words in _topic_tokens.values():
        if tokens & words:
            out |= words
    return out


def relevant(entry: _Entry, question: str, passages: Sequence[str], limit: int = MAX_PROMPT_FACTS) -> List[str]:
    """The facts that bear on this question, as "key: value" lines (at most `limit`): those sharing words with the
    question (after topic expansion: "kab khulta hai" meets "timings"), then those whose key is mostly in the passages."""
    if not entry.facts:
        return []
    q = _expand(_tokens(question))
    p = _tokens(" ".join(passages))
    scored = []
    for i, (fact, (ktoks, vtoks)) in enumerate(zip(entry.facts, entry.tokens)):
        sq = len(q & ktoks) * 2 + len(q & vtoks)
        sp = (len(p & ktoks) / len(ktoks)) if ktoks else 0.0
        if sq >= 2 or sp >= 0.6:
            scored.append((sq, sp, -i, fact))
    scored.sort(reverse=True)
    return [f"{f['key']}: {f['value']}" for _, _, _, f in scored[:limit]]


def turn_facts(run: Any) -> List[str]:
    """The "Key facts" lines for a turn (empty with FACT_SHEET=off, without documents, or before the sheet exists).
    Never blocks and never raises."""
    if not enabled():
        return []
    try:
        entry = facts_for(run.agent_id, run.user_id, getattr(run, "storage_mode", "local") or "local")
        passages = [grounding.passage_text(c) for c in (getattr(run, "chunks", None) or [])]
        return relevant(entry, getattr(run, "user_text", "") or "", passages)
    except Exception as e:  # noqa: BLE001
        log.warning("key facts skipped: %s", e)
        return []


def clear_memory() -> None:
    """Forget every sheet held in memory (tests)."""
    with _lock:
        _mem.clear()
        _failed.clear()
        _running.clear()
