"""Instant answers: the questions callers ask most get a reviewed, pre-written answer that is spoken at once, instead of
running retrieval + the language model (about a second) on every call.

The owner builds suggestions (a button, never automatic), reviews them and approves the ones to use. Only APPROVED
answers are ever used live. This module is pure logic plus two small stores; `routers/instant_answers.py` is the web
layer and the model calls (retrieval, language model, embeddings) are passed in as callables, so everything here is
unit-testable without a model.

Live use (one function): `match(agent_id, user_id, question, language, tools=..., storage_mode=...)` returns an
`InstantAnswer` or None. It is conservative on purpose: a wrong instant answer is worse than a slow right one. A question
is answered instantly only when ALL of these hold:
  - the agent has an approved answer in the caller's reply language, and the agent's documents have not changed since the
    owner approved it (otherwise the answer is "needs review" and is not used);
  - the caller's turn is ONE question (not two, not a long speech), is not a request the agent's tools must handle
    (booking, cancelling, checking a request) and is not sensitive (emergencies, OTPs and passwords always go to the full
    path);
  - BGE-M3 cosine similarity to the approved question or one of its paraphrases is at least INSTANT_ANSWER_THRESHOLD
    (default 0.86), no other approved answer scores within 0.02 with a different text, the days, months, numbers and
    "not" of the two questions are the same (dense embeddings cannot tell "Sunday" from "Monday"), and they do not ask
    opposite things (open/closed, arrive/leave: see opposite_conflict).

Storage: one JSON file (`backend/data/instant_answers.json`, BONCLAY_INSTANT_ANSWERS_FILE overrides) in local mode; the
`instant_answers` table (migration 017) in hosted mode. Every row carries `user_id` and every read is scoped by it.
Switch everything off with INSTANT_ANSWERS=off."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from services import knowledge_gaps as kg
    from services import language_policy
    from services import retrieval
except ImportError:  # pragma: no cover - imported as backend.services.instant_answers
    from backend.services import knowledge_gaps as kg
    from backend.services import language_policy
    from backend.services import retrieval

log = logging.getLogger("bonclay.instant_answers")

STATUS_SUGGESTED = "suggested"      # drafted (or added without an answer); never used live
STATUS_APPROVED = "approved"        # the owner approved it: used live
STATUS_DISABLED = "disabled"        # the owner turned it off
STATUS_REVIEW = "needs_review"      # approved once, but the agent's documents changed since: not used until re-approved
STATUS_DISMISSED = "dismissed"      # deleted by the owner; kept (hidden) so a rebuild does not suggest it again
STATUSES = (STATUS_SUGGESTED, STATUS_APPROVED, STATUS_DISABLED, STATUS_REVIEW)
SOURCES = ("template", "calls", "owner")

# Chosen with bench/instant_answer_threshold.py (real BGE-M3 fp16, 38 clinic/college pairs in English, Hindi and Hinglish,
# with the guards below): 0.81 to 0.92 gives no false match. 0.86 keeps 9 of 18 paraphrases with a margin of about 0.06
# over the highest near miss the guards leave to the cosine (0.802, "lab" vs "pharmacy"); 0.82 would keep one more
# ("Are you open on Sundays", 0.834) with only 0.018 of margin. A wrong instant answer costs more than a normal reply,
# so the safer value is the default. Hinglish paraphrases score 0.66 to 0.80 and mostly need a stored paraphrase.
DEFAULT_THRESHOLD = 0.86
AMBIGUITY_MARGIN = 0.02             # another answer this close (with a different text) makes the match ambiguous
DEDUP_SIMILARITY = 0.93             # a new candidate this close to an existing question is the same question
DEFAULT_MIN_SUPPORT = 0.40          # best retrieved dense cosine needed before a draft is trusted (see build_drafts)
MAX_QUESTION_CHARS = 300
MAX_ANSWER_CHARS = 700
MAX_ANSWER_WORDS = 80               # a spoken quick answer; the agent's own limit is 40 words
MAX_QUESTION_WORDS = 40
MAX_ROWS_PER_AGENT = 200
MAX_PARAPHRASES = 3
DEFAULT_BUILD_LIMIT = 12
DEFAULT_MIN_ASKS = 2

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# -- settings (read on every call so they can change without a restart) ---------------------------------------------

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def enabled() -> bool:
    """INSTANT_ANSWERS=off turns the feature off everywhere (match always returns None)."""
    return (os.getenv("INSTANT_ANSWERS", "on") or "on").strip().lower() not in ("off", "0", "false", "no")


def threshold() -> float:
    """INSTANT_ANSWER_THRESHOLD (default 0.86): the cosine a caller's question needs to count as the approved one."""
    v = _float_env("INSTANT_ANSWER_THRESHOLD", DEFAULT_THRESHOLD)
    return v if 0.0 < v <= 1.0 else DEFAULT_THRESHOLD


def min_support() -> float:
    """INSTANT_ANSWER_MIN_SUPPORT (default 0.40): a draft needs a document passage at least this close to the question."""
    return _float_env("INSTANT_ANSWER_MIN_SUPPORT", DEFAULT_MIN_SUPPORT)


def build_limit() -> int:
    return max(1, min(40, _int_env("INSTANT_ANSWER_BUILD_MAX", DEFAULT_BUILD_LIMIT)))


def min_asks() -> int:
    """How many times callers must have asked a question before it is suggested from real calls."""
    return max(1, _int_env("INSTANT_ANSWER_MIN_ASKS", DEFAULT_MIN_ASKS))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


now_iso = _now


# -- text helpers -----------------------------------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return kg._norm(text)  # noqa: SLF001 - the same normalisation the knowledge-gap report groups questions with


_PUNCT = re.compile(r"[^\w\u0900-\u0DFF]+", re.UNICODE)


def qkey(text: str) -> str:
    """What makes two questions "the same question": lower case, no punctuation, one space between words."""
    return _PUNCT.sub(" ", norm(text)).strip()


def clean_answer(text: str) -> str:
    """The reply as it is spoken: no markdown, bullets, numbering or surrounding quotes, one line."""
    t = (text or "").replace("\r", "\n")
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"(?m)^\s*(?:[-•]|\d+[.)])\s+", "", t)
    t = _WS.sub(" ", t).strip().strip('"“”').strip()
    return t


def lang_code(language: Optional[str]) -> str:
    """"en-IN" -> "en"; "hinglish" stays; empty -> "en"."""
    return language_policy.base_code(language or "") or "en"


def word_count(text: str) -> int:
    return len((text or "").split())


# -- guards: when an instant answer must not be used -----------------------------------------------------------------

_INTERROGATIVES = {
    # English
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    # Hindi
    "क्या", "कब", "कहाँ", "कहां", "कौन", "कैसे", "कैसा", "कैसी", "कितना", "कितनी", "कितने", "क्यों", "क्यूँ", "किस",
    "किसे", "कौनसा", "कौनसी", "कोनसा", "किसको",
    # Hinglish
    "kya", "kab", "kahan", "kahaan", "kaun", "kaise", "kitna", "kitni", "kitne", "kyun", "kyu", "konsa", "kaunsa",
}
_QMARKS = re.compile(r"[?？]")
_SEGMENT_SPLIT = re.compile(r"[?？!.।\n]+")
_WORD = language_policy.WORD


def _tokens(text: str) -> List[str]:
    t = (text or "").translate(kg._APOS).lower()  # noqa: SLF001
    t = re.sub(r"n't\b", " not", t)
    return _WORD.findall(t)


def looks_multi_question(text: str) -> bool:
    """True when the turn holds more than one question (two question marks, two question words, or two sentences that
    each ask something). One answer cannot serve two questions, so these always take the full path."""
    if len(_QMARKS.findall(text or "")) >= 2:
        return True
    toks = _tokens(text)
    if sum(1 for t in toks if t in _INTERROGATIVES) >= 2:
        return True
    asking = 0
    for seg in _SEGMENT_SPLIT.split(text or ""):
        words = _tokens(seg)
        if len(words) >= 3 and any(w in _INTERROGATIVES for w in words):
            asking += 1
    return asking >= 2


_SENSITIVE = re.compile(
    r"\b(?:otp|one[- ]time (?:password|pin|code)|password|passcode|cvv|cvc|pin (?:number|code)|aadhaar|aadhar|pan card|"
    r"card number|account number|bank details|net ?banking|upi pin)\b"
    r"|\b(?:emergency|ambulance|chest pain|heart attack|can'?t breathe|cannot breathe|unconscious|bleeding|overdose|"
    r"poison|suicide|suicidal|kill myself|self[- ]harm|seizure|stroke|accident|urgent(?:ly)?|dying)\b"
    r"|ओ\s?टी\s?पी|पासवर्ड|पिन नंबर|आधार|इमरजेंसी|आपातकाल|एम्बुलेंस|एंबुलेंस|सीने में दर्द|दिल का दौरा|बेहोश|खून|"
    r"आत्महत्या|जहर|ज़हर|दुर्घटना",
    re.IGNORECASE)


def is_sensitive(text: str) -> bool:
    """Emergencies and anything that asks for or involves a secret (OTP, password, PIN, card): never an instant answer."""
    return bool(_SENSITIVE.search((text or "").translate(kg._APOS)))  # noqa: SLF001


_TOOL_INTENT = re.compile(
    r"\b(?:book(?:ing|ed)?|appointments?|apointments?|schedul\w*|reschedul\w*|cancel\w*|slots?|meetings?|meet|"
    r"requests?|free (?:on|at|today|tomorrow)|available (?:on|at|today|tomorrow)|status of|availability|"
    r"milna|milne|mulakat|mulaqat|bukking)\b"
    r"|बुक|अपॉइंटमेंट|अपाइंटमेंट|अपोइंटमेंट|मुलाक़ात|मुलाकात|मीटिंग|मिलना|मिलने|रद्द|कैंसल|स्लॉट|रिक्वेस्ट|अनुरोध|फ्री",
    re.IGNORECASE)


def needs_tool(text: str) -> bool:
    """True when the caller's words look like a request that an agent's tools handle (booking, changing, cancelling or
    checking an appointment or meeting request). Applied only to agents that have tools."""
    return bool(_TOOL_INTENT.search((text or "").translate(kg._APOS)))  # noqa: SLF001


_DISCRIMINATORS = {
    # days, months and relative days: embeddings score "Sunday" and "Monday" questions as near-identical
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "weekend", "weekday", "weekdays",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november",
    "december", "today", "tomorrow", "yesterday", "tonight", "morning", "evening", "night", "afternoon",
    "सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार", "आज", "कल", "परसों", "सुबह", "शाम", "रात",
    "somvar", "mangalvar", "budhvar", "guruvar", "shukravar", "shanivar", "ravivar", "itwar", "aaj", "kal",
    # negation
    "not", "no", "never", "without", "cannot", "नहीं", "नही", "मत", "बिना", "nahi", "nahin", "bina", "mat",
    # number words
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "first", "second", "third", "half",
}


_PLURAL_DAYS = {d + "s": d for d in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                                     "weekend", "weekday")}   # "open on Sundays" asks the same as "open on Sunday"


def guard_tokens(text: str) -> frozenset:
    """The words that change the answer without changing the embedding much: days, months, numbers, negations."""
    out = set()
    for t in _tokens(text):
        t = _PLURAL_DAYS.get(t, t)
        if t in _DISCRIMINATORS or any(ch.isdigit() for ch in t):
            out.add(t)
    return frozenset(out)


# Opposites BGE-M3 scores as near-identical questions (bench/instant_answer_threshold.py: "Is the clinic closed on Sunday?"
# vs "...open on Sunday?" 0.958, "क्या रविवार को क्लिनिक बंद रहता है?" vs "...खुला रहता है?" 0.961, "Doctor kitne baje jaate
# hain?" vs "...aate hain?" 0.911), far above any usable threshold. Two questions that name opposite sides of one of these
# pairs are never the same question; a question that names neither side ("What are the clinic timings?") is not affected.
_OPPOSITES: Tuple[Tuple[frozenset, frozenset], ...] = (
    (frozenset({"open", "opens", "opened", "opening", "khula", "khuli", "khule", "khulta", "khulti", "khulte", "khulega",
                "khulegi", "khulenge", "khulne", "खुला", "खुली", "खुले", "खुलता", "खुलती", "खुलते", "खुलेगा", "खुलेगी",
                "खुलेंगे", "खुलने"}),
     frozenset({"close", "closes", "closed", "closing", "shut", "shuts", "band", "bandh", "बंद", "बन्द"})),
    (frozenset({"arrive", "arrives", "arrival", "aate", "aata", "aati", "aayenge", "aaenge", "aayega", "aayegi", "आते",
                "आता", "आती", "आएंगे", "आएँगे", "आएगा", "आएगी"}),
     frozenset({"leave", "leaves", "departure", "jaate", "jaata", "jaati", "jayenge", "jaayenge", "jaenge", "jayega",
                "jaayega", "जाते", "जाता", "जाती", "जाएंगे", "जाएँगे", "जाएगा", "जाएगी"})),
)


def _sides(text: str) -> List[Tuple[bool, bool]]:
    words = set(_tokens(text))
    return [(bool(words & a), bool(words & b)) for a, b in _OPPOSITES]


def opposite_conflict(a: str, b: str) -> bool:
    """True when `a` and `b` take different sides of an opposite pair (open/closed, arrive/leave): one names one side,
    the other names the other side and not the first."""
    for (a_one, a_two), (b_one, b_two) in zip(_sides(a), _sides(b)):
        if (a_one or a_two) and (b_one or b_two) and (a_one, a_two) != (b_one, b_two):
            return True
    return False


def question_problem(text: str, *, has_tools: bool) -> Optional[str]:
    """Why this question can never be an instant answer (plain language for the owner), or None when it can."""
    t = (text or "").strip()
    if word_count(t) < 2:
        return "too short to be a question"
    if word_count(t) > MAX_QUESTION_WORDS or len(t) > MAX_QUESTION_CHARS:
        return "too long: instant answers are for one short question"
    if is_sensitive(t):
        return "emergencies and secrets (OTP, password, PIN) are always handled by the full agent"
    if looks_multi_question(t):
        return "it asks more than one thing at once"
    if has_tools and needs_tool(t):
        return "it is a request the agent's tools handle (booking, meetings, requests)"
    return None


# -- stores ----------------------------------------------------------------------------------------------------------

def default_path() -> str:
    return os.getenv("BONCLAY_INSTANT_ANSWERS_FILE") or os.path.join(_DATA_DIR, "instant_answers.json")


class LocalFileStore:
    """All agents' instant answers in one JSON file. Writes are atomic (temp file + rename) and serialised by a lock;
    reads are cached until the file changes. Every method takes the owner's `user_id`: rows of other users are
    invisible."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_path()
        self._lock = threading.RLock()
        self._writes = 0
        self._cache: Optional[Tuple[Hashable, List[dict]]] = None

    def _stat(self) -> Tuple[Any, ...]:
        try:
            st = os.stat(self.path)
            return (st.st_mtime_ns, st.st_size, self._writes)
        except OSError:
            return (None, None, self._writes)

    def _read(self) -> List[dict]:
        with self._lock:
            rev = self._stat()
            if self._cache is not None and self._cache[0] == rev:
                return self._cache[1]
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
                rows = rows if isinstance(rows, list) else []
            except FileNotFoundError:
                rows = []
            except (OSError, ValueError) as e:
                log.error("instant answers file %s is unreadable: %s", self.path, e)
                rows = []
            self._cache = (rev, rows)
            return rows

    def _write(self, rows: List[dict]) -> None:
        folder = os.path.dirname(self.path) or "."
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        os.replace(tmp, self.path)
        self._writes += 1
        self._cache = None

    def revision(self, user_id: str, agent_id: str) -> Hashable:
        return (self.path, *self._stat())

    def list(self, user_id: str, agent_id: str) -> List[dict]:
        return [dict(r) for r in self._read() if r.get("user_id") == user_id and r.get("agent_id") == agent_id]

    def get(self, user_id: str, agent_id: str, row_id: str) -> Optional[dict]:
        return next((r for r in self.list(user_id, agent_id) if r.get("id") == row_id), None)

    def insert(self, row: dict) -> dict:
        row = {"id": str(uuid.uuid4()), "created_at": _now(), "updated_at": _now(), **row}
        with self._lock:
            rows = list(self._read())
            rows.append(row)
            self._write(rows)
        return dict(row)

    def update(self, user_id: str, agent_id: str, row_id: str, fields: Dict[str, Any]) -> Optional[dict]:
        with self._lock:
            rows = [dict(r) for r in self._read()]
            for r in rows:
                if r.get("id") == row_id and r.get("user_id") == user_id and r.get("agent_id") == agent_id:
                    r.update(fields)
                    r["updated_at"] = _now()
                    self._write(rows)
                    return dict(r)
        return None

    def update_many(self, user_id: str, agent_id: str, updates: Dict[str, Dict[str, Any]]) -> int:
        """Apply {row_id: fields} in one write. Returns how many rows changed."""
        if not updates:
            return 0
        with self._lock:
            rows = [dict(r) for r in self._read()]
            n = 0
            for r in rows:
                f = updates.get(r.get("id"))
                if f is not None and r.get("user_id") == user_id and r.get("agent_id") == agent_id:
                    r.update(f)
                    r["updated_at"] = _now()
                    n += 1
            if n:
                self._write(rows)
        return n

    def delete(self, user_id: str, agent_id: str, row_id: str) -> bool:
        with self._lock:
            rows = list(self._read())
            keep = [r for r in rows if not (r.get("id") == row_id and r.get("user_id") == user_id
                                            and r.get("agent_id") == agent_id)]
            if len(keep) == len(rows):
                return False
            self._write(keep)
            return True

    def purge_agent(self, user_id: str, agent_id: str) -> int:
        with self._lock:
            rows = list(self._read())
            keep = [r for r in rows if not (r.get("user_id") == user_id and r.get("agent_id") == agent_id)]
            if len(keep) != len(rows):
                self._write(keep)
            return len(rows) - len(keep)


class SupabaseStore:
    """The hosted twin (table `instant_answers`, migration 017). Same methods as LocalFileStore. Not exercised by the
    test suite: every call is wrapped by the callers so a failure never breaks a call (a miss is the safe result)."""

    TABLE = "instant_answers"
    TTL_S = 30

    def __init__(self, client: Any):
        self.client = client
        self._writes = 0

    def _t(self) -> Any:
        return self.client.table(self.TABLE)

    def revision(self, user_id: str, agent_id: str) -> Hashable:
        return ("supabase", int(time.time() // self.TTL_S), self._writes)

    def list(self, user_id: str, agent_id: str) -> List[dict]:
        res = self._t().select("*").eq("user_id", user_id).eq("agent_id", agent_id).limit(MAX_ROWS_PER_AGENT * 2).execute()
        return [dict(r) for r in (res.data or [])]

    def get(self, user_id: str, agent_id: str, row_id: str) -> Optional[dict]:
        res = self._t().select("*").eq("id", row_id).eq("user_id", user_id).eq("agent_id", agent_id).limit(1).execute()
        return dict(res.data[0]) if res.data else None

    def insert(self, row: dict) -> dict:
        row = {"id": str(uuid.uuid4()), "created_at": _now(), "updated_at": _now(), **row}
        self._t().insert(row).execute()
        self._writes += 1
        return dict(row)

    def update(self, user_id: str, agent_id: str, row_id: str, fields: Dict[str, Any]) -> Optional[dict]:
        fields = {**fields, "updated_at": _now()}
        res = (self._t().update(fields).eq("id", row_id).eq("user_id", user_id).eq("agent_id", agent_id).execute())
        self._writes += 1
        return dict(res.data[0]) if res.data else None

    def update_many(self, user_id: str, agent_id: str, updates: Dict[str, Dict[str, Any]]) -> int:
        n = 0
        for row_id, fields in updates.items():
            if self.update(user_id, agent_id, row_id, fields) is not None:
                n += 1
        return n

    def delete(self, user_id: str, agent_id: str, row_id: str) -> bool:
        res = self._t().delete().eq("id", row_id).eq("user_id", user_id).eq("agent_id", agent_id).execute()
        self._writes += 1
        return bool(res.data)

    def purge_agent(self, user_id: str, agent_id: str) -> int:
        res = self._t().delete().eq("user_id", user_id).eq("agent_id", agent_id).execute()
        self._writes += 1
        return len(res.data or [])


_local_store: Optional[LocalFileStore] = None
_supabase_client_getter: Optional[Callable[[], Any]] = None
_store_lock = threading.Lock()


def configure(*, supabase: Optional[Callable[[], Any]] = None) -> None:
    """Tell the module where the Supabase client is (optional; backend.py need not call it, the router does)."""
    global _supabase_client_getter
    _supabase_client_getter = supabase


def get_store(storage_mode: str = "local") -> Any:
    """The store for a storage mode. Hosted mode without a Supabase client falls back to the local file."""
    global _local_store
    if storage_mode != "local" and _supabase_client_getter is not None:
        try:
            client = _supabase_client_getter()
        except Exception:  # noqa: BLE001
            client = None
        if client is not None:
            return SupabaseStore(client)
    with _store_lock:
        path = default_path()
        if _local_store is None or _local_store.path != path:
            _local_store = LocalFileStore(path)
        return _local_store


# -- embeddings ------------------------------------------------------------------------------------------------------

Embedder = Callable[[List[str]], Any]
_embedder_override: Optional[Embedder] = None


def set_embedder(fn: Optional[Embedder]) -> None:
    """Replace the embedding function (tests); None restores BGE-M3 through the backend's shared model."""
    global _embedder_override
    _embedder_override = fn


def _backend():
    import sys  # noqa: PLC0415
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "embedding_model"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "embedding_model"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


def default_embed(texts: List[str]) -> np.ndarray:
    if _embedder_override is not None:
        return np.asarray(_embedder_override(list(texts)), dtype=np.float32)
    return np.asarray(_backend().embedding_model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def model_name() -> str:
    return "test-embedder" if _embedder_override is not None else retrieval.embedding_model_name()


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def embed_row(row: dict, embed: Optional[Embedder] = None) -> dict:
    """Fields to store with a row so a caller's question can be compared to it: the vectors of the question and each
    paraphrase. Empty dict when the model is unavailable (the live index then embeds the row itself)."""
    variants = [row["question"], *(row.get("paraphrases") or [])]
    try:
        vecs = _unit((embed or default_embed)(variants))
        if vecs.shape[0] != len(variants):
            return {}
    except Exception as e:  # noqa: BLE001 - the embedding model may be unavailable; the row still saves
        log.warning("could not embed instant answer %r: %s", row.get("question"), e)
        return {}
    return {"vectors": [[round(float(x), 5) for x in v] for v in vecs], "embed_model": model_name(),
            "variants": variants}


# -- the live index and match() ---------------------------------------------------------------------------------------

@dataclass
class InstantAnswer:
    """A hit of `match`. Speak `text` through the normal TTS path. `cache_key` identifies this exact answer text in this
    language (it changes when the owner edits the answer): combine it with the engine and voice for an audio cache."""
    id: str
    agent_id: str
    text: str
    language: str
    question: str            # the approved question that matched
    matched_text: str        # the question or paraphrase the caller's words were closest to
    score: float
    cache_key: str


@dataclass
class _Index:
    rows: List[dict]
    revision: Hashable
    mat: Optional[np.ndarray] = None            # (N, D) unit vectors of every variant of every row
    owner: Optional[np.ndarray] = None          # row position of each vector
    variants: List[str] = field(default_factory=list)
    ready: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


_indexes: Dict[Tuple[Any, ...], _Index] = {}
_index_lock = threading.Lock()


def clear_cache() -> None:
    with _index_lock:
        _indexes.clear()


def _ensure_vectors(ix: _Index, embed: Embedder) -> None:
    """Stack the vectors of every variant of every row (stored ones when they come from the current model, else embedded now)."""
    with ix.lock:
        if ix.ready:
            return
        model = model_name()
        per_row: List[Optional[np.ndarray]] = []
        for r in ix.rows:
            vs = [r["question"], *(r.get("paraphrases") or [])]
            vecs = r.get("vectors")
            if r.get("embed_model") == model and isinstance(vecs, list) and len(vecs) == len(vs):
                per_row.append(_unit(np.asarray(vecs, dtype=np.float32)))
            else:
                per_row.append(None)
        dims = {m.shape[1] for m in per_row if m is not None}
        if len(dims) > 1:
            per_row = [None] * len(per_row)  # vectors of two sizes cannot be stacked: embed everything again
        todo = [i for i, m in enumerate(per_row) if m is None]
        if todo:
            flat = [t for i in todo for t in [ix.rows[i]["question"], *(ix.rows[i].get("paraphrases") or [])]]
            out = _unit(embed(flat))
            pos = 0
            for i in todo:
                n = 1 + len(ix.rows[i].get("paraphrases") or [])
                per_row[i] = out[pos:pos + n]
                pos += n
        mats = [m for m in per_row if m is not None]
        if mats and len({m.shape[1] for m in mats}) == 1:
            ix.mat = np.vstack(mats)
            ix.owner = np.asarray([i for i, m in enumerate(per_row) for _ in range(m.shape[0])], dtype=np.int64)
            ix.variants = [t for r in ix.rows for t in [r["question"], *(r.get("paraphrases") or [])]]
        ix.ready = True


def _index_for(user_id: str, agent_id: str, storage_mode: str) -> Optional[_Index]:
    store = get_store(storage_mode)
    rev = store.revision(user_id, agent_id)
    key = (store.path if isinstance(store, LocalFileStore) else "cloud", user_id, agent_id)
    with _index_lock:
        ix = _indexes.get(key)
        if ix is not None and ix.revision == rev:
            return ix
    rows = [r for r in store.list(user_id, agent_id) if r.get("status") == STATUS_APPROVED and (r.get("answer") or "").strip()]
    ix = _Index(rows=rows, revision=rev)
    with _index_lock:
        if len(_indexes) > 256:
            _indexes.clear()
        _indexes[key] = ix
    return ix


def cache_key_for(row: dict) -> str:
    raw = f"{row.get('id')}|{lang_code(row.get('language'))}|{row.get('answer') or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _agent_tools(agent_id: str, user_id: str) -> Optional[List[str]]:
    """The agent's tool names from the local store, or None when they cannot be looked up (then be conservative)."""
    try:
        row = _backend().local_db.get("agents", agent_id, user_id)
    except Exception:  # noqa: BLE001
        return None
    if row is None:
        return None
    return [str(t) for t in (row.get("tools") or [])]


def docs_fingerprint(documents: Iterable[dict]) -> str:
    """A short id of an agent's set of documents. It changes when a document is added or deleted (the only ways an agent's
    knowledge changes) and not when a document is re-split into passages."""
    parts = sorted(f"{d.get('id')}:{d.get('filename') or ''}" for d in documents)
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


Fingerprinter = Callable[[str, str, str], Optional[str]]
_fingerprint_override: Optional[Fingerprinter] = None


def set_fingerprinter(fn: Optional[Fingerprinter]) -> None:
    """Replace how the agent's current document fingerprint is found (tests)."""
    global _fingerprint_override
    _fingerprint_override = fn


_fp_cache: Dict[Tuple[str, str, str], Tuple[float, Optional[str]]] = {}
FP_TTL_S = 30


def current_fingerprint(agent_id: str, user_id: str, storage_mode: str = "local") -> Optional[str]:
    """Blocking: the fingerprint of the agent's documents now, or None when it cannot be read."""
    if _fingerprint_override is not None:
        return _fingerprint_override(agent_id, user_id, storage_mode)
    try:
        B = _backend()
        if storage_mode == "local" or getattr(B, "supabase", None) is None:
            docs = B.local_db.select("documents", lambda r: r.get("agent_id") == agent_id)
            return docs_fingerprint(docs)
        key = (agent_id, user_id, storage_mode)
        hit = _fp_cache.get(key)
        if hit and time.monotonic() - hit[0] < FP_TTL_S:
            return hit[1]
        res = B.supabase.table("documents").select("id, filename").eq("agent_id", agent_id).eq("user_id", user_id).execute()
        fp = docs_fingerprint(res.data or [])
        _fp_cache[key] = (time.monotonic(), fp)
        return fp
    except Exception as e:  # noqa: BLE001
        log.warning("could not read the agent's documents for instant answers: %s", e)
        return None


def match(agent_id: str, user_id: str, question: str, language: str, *, tools: Optional[Sequence[str]] = None,
          storage_mode: str = "local", embed: Optional[Embedder] = None,
          fingerprint: Optional[str] = None) -> Optional[InstantAnswer]:
    """The approved instant answer for a caller's transcribed question, or None (use the normal pipeline).

    `language` is the turn's reply language (run.language). `tools` is the agent's tool list (`run.agent.get("tools")`);
    leave it out only if unknown (the booking guard is then applied, to be safe). `storage_mode` is `run.storage_mode`.
    Blocking (about one embedding of the question, 20 to 80 ms on CPU with BGE-M3; nothing is encoded when the agent has
    no approved answer in this language): call it with `asyncio.to_thread`. Never raises: any failure is a miss."""
    try:
        return _match(agent_id, user_id, question, language, tools, storage_mode, embed, fingerprint)
    except Exception as e:  # noqa: BLE001 - an instant answer is an optimisation: the full path is always the fallback
        log.warning("instant answer match failed, using the normal answer: %s", e)
        return None


def _match(agent_id, user_id, question, language, tools, storage_mode, embed, fingerprint) -> Optional[InstantAnswer]:
    if not enabled():
        return None
    q = (question or "").strip()
    if not q or not agent_id or not user_id:
        return None
    known = tools if tools is not None else _agent_tools(agent_id, user_id)
    has_tools = True if known is None else bool(known)  # unknown: assume it has tools (the safe side)
    if question_problem(q, has_tools=has_tools):
        return None
    lang = lang_code(language)
    ix = _index_for(user_id, agent_id, storage_mode)
    if ix is None or not ix.rows:
        return None
    if not any(lang_code(r.get("language")) == lang for r in ix.rows):
        return None
    fp = fingerprint if fingerprint is not None else current_fingerprint(agent_id, user_id, storage_mode)
    if fp is None:
        return None  # cannot tell whether the documents changed: not worth the risk
    ok_rows = [i for i, r in enumerate(ix.rows) if lang_code(r.get("language")) == lang
               and (r.get("docs_fingerprint") or None) == fp]
    if not ok_rows:
        return None
    emb = embed or default_embed
    _ensure_vectors(ix, emb)
    if ix.mat is None or ix.owner is None:
        return None
    qv = _unit(emb([q]))[0]
    if qv.shape[0] != ix.mat.shape[1]:
        return None
    sims = ix.mat @ qv
    allowed = np.isin(ix.owner, np.asarray(ok_rows, dtype=np.int64))
    sims = np.where(allowed, sims, -1.0)
    # best score per row
    best_per_row: Dict[int, Tuple[float, int]] = {}
    for vi in np.flatnonzero(allowed):
        r_i = int(ix.owner[vi])
        s = float(sims[vi])
        if r_i not in best_per_row or s > best_per_row[r_i][0]:
            best_per_row[r_i] = (s, int(vi))
    ranked = sorted(best_per_row.items(), key=lambda kv: kv[1][0], reverse=True)
    top_row, (top_score, top_var) = ranked[0]
    if top_score < threshold():
        return None
    for other_row, (s, _v) in ranked[1:]:
        if s >= top_score - AMBIGUITY_MARGIN and norm(ix.rows[other_row]["answer"]) != norm(ix.rows[top_row]["answer"]):
            return None  # two different answers fit equally well: let the full path decide
    row = ix.rows[top_row]
    matched_text = ix.variants[top_var]
    wanted = guard_tokens(q)
    if wanted != guard_tokens(matched_text) and wanted != guard_tokens(row["question"]):
        return None  # a different day, number or a "not": not the same question
    if opposite_conflict(q, matched_text) or opposite_conflict(q, row["question"]):
        return None  # "closed" against "open", "leave" against "arrive": the opposite question
    if word_count(q) > word_count(matched_text) * 1.8 + 3:
        return None  # much longer than the question it resembles: probably more than that question
    return InstantAnswer(id=row["id"], agent_id=agent_id, text=row["answer"].strip(), language=lang,
                         question=row["question"], matched_text=matched_text, score=round(top_score, 4),
                         cache_key=cache_key_for(row))


# -- audio cache (optional) -------------------------------------------------------------------------------------------

class AudioCache:
    """A small in-memory cache of rendered audio, keyed by a string. `get_or_render(key, render)` returns
    (value, hit); `render` is called at most once at a time per key (a second caller waits for the first), and its result
    is kept. Bounded by item count and an estimated size so a long-running server cannot grow without limit.
    The cache holds whatever `render` returns (for example the UNWATERMARKED (samples, sample_rate) of the answer): watermark
    a copy for every turn, as the greeting and filler clips are, so every reply still carries its own receipt."""

    def __init__(self, max_items: int = 64, max_bytes: int = 256 * 1024 * 1024):
        self.max_items, self.max_bytes = max_items, max_bytes
        self._items: "OrderedDict[str, Tuple[Any, int]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._inflight: Dict[str, threading.Event] = {}

    @staticmethod
    def _size(value: Any) -> int:
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, (bytes, bytearray)):
            return len(value)
        if isinstance(value, tuple):
            return sum(AudioCache._size(v) for v in value)
        return 1024

    def peek(self, key: str) -> Any:
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit[0]
        return None

    def put(self, key: str, value: Any) -> None:
        size = self._size(value)
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._items[key] = (value, size)
            self._bytes += size
            while self._items and (len(self._items) > self.max_items or self._bytes > self.max_bytes):
                _k, (_v, s) = self._items.popitem(last=False)
                self._bytes -= s

    def get_or_render(self, key: str, render: Callable[[], Any]) -> Tuple[Any, bool]:
        while True:
            value = self.peek(key)
            if value is not None:
                return value, True
            with self._lock:
                wait = self._inflight.get(key)
                if wait is None:
                    self._inflight[key] = threading.Event()
            if wait is None:
                break
            wait.wait(timeout=60)
        try:
            value = render()
            if value is not None:
                self.put(key, value)
            return value, False
        finally:
            with self._lock:
                ev = self._inflight.pop(key, None)
            if ev is not None:
                ev.set()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def __len__(self) -> int:
        return len(self._items)


audio_cache = AudioCache()


def audio_key(answer: InstantAnswer, *parts: Any) -> str:
    """`answer.cache_key` plus whatever else changes the audio (engine, voice id, tts language, speed)."""
    return "|".join([answer.cache_key, *[str(p) for p in parts]])


def cached_audio(answer: InstantAnswer, render: Callable[[], Any], *parts: Any) -> Tuple[Any, bool]:
    """(audio, was_cached). `render()` runs the normal TTS for `answer.text` and returns the audio to keep."""
    return audio_cache.get_or_render(audio_key(answer, *parts), render)


# -- review state: documents changed -----------------------------------------------------------------------------------

def review_updates(rows: Sequence[dict], fingerprint: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Approved answers whose documents changed since they were approved become "needs_review". Returns {row id: fields}.
    With no fingerprint (documents unreadable) nothing changes."""
    if fingerprint is None:
        return {}
    return {r["id"]: {"status": STATUS_REVIEW} for r in rows
            if r.get("status") == STATUS_APPROVED and (r.get("docs_fingerprint") or None) != fingerprint}


def invalidate_agent(agent_id: str, user_id: str, storage_mode: str = "local") -> int:
    """Mark every approved answer of the agent "needs review" (call it when a document is uploaded or deleted). Returns how
    many changed. The GET route and `match` also notice a changed document set on their own, so calling this is optional."""
    store = get_store(storage_mode)
    rows = store.list(user_id, agent_id)
    ups = {r["id"]: {"status": STATUS_REVIEW} for r in rows if r.get("status") == STATUS_APPROVED}
    return store.update_many(user_id, agent_id, ups)


# -- candidates --------------------------------------------------------------------------------------------------------

@dataclass
class Candidate:
    question: str
    source: str                                  # template | calls | owner
    count: int = 0                               # how often callers asked it (calls)
    row_id: Optional[str] = None                 # an existing row to (re)draft instead of a new one
    language: Optional[str] = None


def _tokens_set(text: str) -> set:
    return set(_tokens(text))


def find_template(agent: dict, templates: Sequence[dict]) -> Optional[dict]:
    """The template an agent was made from: its stored `template_id`, else the template whose persona is nearly the same
    text (personalising only swaps organisation names)."""
    tid = agent.get("template_id")
    if tid:
        found = next((t for t in templates if t.get("id") == tid), None)
        if found:
            return found
    mine = _tokens_set(agent.get("system_instructions") or "")
    if len(mine) < 20:
        return None
    best, best_j = None, 0.0
    for t in templates:
        theirs = _tokens_set(t.get("system_instructions") or "")
        if not theirs:
            continue
        j = len(mine & theirs) / len(mine | theirs)
        if j > best_j:
            best, best_j = t, j
    return best if best_j >= 0.6 else None


def template_candidates(agent: dict, templates: Sequence[dict]) -> List[Candidate]:
    t = find_template(agent, templates)
    if not t:
        return []
    return [Candidate(question=q.strip(), source="template") for q in (t.get("suggested_questions") or [])
            if isinstance(q, str) and q.strip()]


def popular_questions(pairs: Sequence[dict], embed: Optional[Embedder], *, min_count: Optional[int] = None,
                      limit: int = 20) -> List[Tuple[str, int]]:
    """The most frequent real questions: (representative question, times asked). `pairs` are `kg.pair_turns` rows of ONE
    agent's real (non-test) calls. Questions that contain personal data (the redaction changes them), turns where the agent
    said it did not know, outages and one-word greetings are left out; similar wordings are grouped like the
    knowledge-gap report does."""
    need = min_count if min_count is not None else min_asks()
    items: List[dict] = []
    for p in pairs:
        qtext = _WS.sub(" ", (p.get("question") or "")).strip()
        if not qtext or p.get("fallback") or word_count(qtext) < 3 or word_count(qtext) > MAX_QUESTION_WORDS:
            continue
        if kg.is_gap_reply(p.get("reply") or ""):
            continue
        if kg.redact(qtext) != qtext:   # an email, phone number or id was in it: never store it
            continue
        items.append({**p, "question": qtext})
    if not items:
        return []
    groups, vecs = kg.cluster_agent_gaps(items, embed)
    out: List[Tuple[str, int]] = []
    for g in groups:
        if len(g) < need:
            continue
        rep = kg._representative(g, vecs)  # noqa: SLF001
        out.append((rep, len(g)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:limit]


def merge_candidates(groups: Sequence[Sequence[Candidate]], existing: Sequence[dict], embed: Optional[Embedder],
                     *, limit: int, supported: Optional[Sequence[str]] = None) -> List[Candidate]:
    """One list of questions to draft, in the order of `groups` (template questions, then real ones by frequency, then the
    owner's questions awaiting a draft), minus questions the agent already has in that language (same words or a
    near-identical meaning, dismissed ones included) and repeats among the candidates themselves. A candidate with a
    `row_id` is an existing row that needs a new draft and is never dropped as a duplicate of itself. Each candidate gets its
    language (the script of the question)."""
    supported = list(supported or ["en"])
    cands: List[Candidate] = []
    seen: set = set()
    taken = {(qkey(r["question"]), lang_code(r.get("language"))) for r in existing}
    for grp in groups:
        for c in grp:
            c.language = lang_code(c.language or language_policy.detect_language(c.question, supported))
            key = (qkey(c.question), c.language)
            if key in seen or (c.row_id is None and key in taken):
                continue
            seen.add(key)
            cands.append(c)
    if embed is not None and cands:
        try:
            vecs = _unit(embed([c.question for c in cands]))
            ex = _unit(embed([r["question"] for r in existing])) if existing else None
            ex_lang = [lang_code(r.get("language")) for r in existing]
            kept: List[Candidate] = []
            kept_vecs: List[np.ndarray] = []
            for c, v in zip(cands, vecs):
                if c.row_id is None and ex is not None:
                    same = [i for i, lg in enumerate(ex_lang) if lg == c.language]
                    if same and float(np.max(ex[same] @ v)) >= DEDUP_SIMILARITY:
                        continue
                if kept_vecs and float(np.max(np.vstack(kept_vecs) @ v)) >= DEDUP_SIMILARITY:
                    continue
                kept.append(c)
                kept_vecs.append(v)
            cands = kept
        except Exception as e:  # noqa: BLE001 - exact-text de-duplication already happened
            log.warning("could not compare candidate questions: %s", e)
    return cands[:limit]


# -- drafting answers ------------------------------------------------------------------------------------------------

@dataclass
class DraftDeps:
    """What a build needs from the model side. All blocking; the router supplies them from the agent's normal path.
      retrieve(question) -> (passages, best dense cosine or None)   the agent's retrieval over its documents
      answer(question, language, passages) -> text                  the agent's grounded answer (its prompt and language rule)
      paraphrase(question, language) -> list of other wordings      optional
      embed(texts) -> vectors                                       optional (rows are embedded when approved otherwise)
    """
    retrieve: Callable[[str], Tuple[List[str], Optional[float]]]
    answer: Callable[[str, str, List[str]], str]
    paraphrase: Optional[Callable[[str, str], List[str]]] = None
    embed: Optional[Embedder] = None


@dataclass
class DraftResult:
    ok: bool
    reason: str = ""
    answer: str = ""
    paraphrases: List[str] = field(default_factory=list)


def draft(question: str, language: str, deps: DraftDeps, *, has_tools: bool) -> DraftResult:
    """One grounded draft. Skipped (ok False, with a plain-language reason) when the question can never be an instant answer,
    when the documents do not support an answer, or when the model says it does not know: "I don't know" is never stored."""
    problem = question_problem(question, has_tools=has_tools)
    if problem:
        return DraftResult(False, problem)
    chunks, score = deps.retrieve(question)
    if not chunks:
        return DraftResult(False, "nothing in the documents is about this")
    if score is not None and score < min_support():
        return DraftResult(False, "the documents do not cover this well enough")
    text = clean_answer(deps.answer(question, language, chunks))
    if not text:
        return DraftResult(False, "the agent gave no answer")
    if kg.is_gap_reply(text):
        return DraftResult(False, "the agent says it does not have this information")
    if word_count(text) > MAX_ANSWER_WORDS or len(text) > MAX_ANSWER_CHARS:
        return DraftResult(False, "the answer is too long to say as a quick reply")
    if not language_policy.reply_matches(language, text):
        return DraftResult(False, "the answer was not in the right language")
    paras: List[str] = []
    if deps.paraphrase is not None:
        try:
            paras = clean_paraphrases(question, deps.paraphrase(question, language))
        except Exception as e:  # noqa: BLE001 - paraphrases only improve matching; the answer is still good
            log.warning("paraphrases failed for %r: %s", question, e)
    return DraftResult(True, "", text, paras)


def clean_paraphrases(question: str, raw: Sequence[Any]) -> List[str]:
    """At most three other wordings. Drops duplicates and any wording that changes a day, number or "not", or that asks
    several things, since matching one of those to the answer could be wrong."""
    want = guard_tokens(question)
    seen = {qkey(question)}
    out: List[str] = []
    for p in raw or []:
        if not isinstance(p, str):
            continue
        t = _WS.sub(" ", p).strip()
        if not t or qkey(t) in seen or word_count(t) < 2 or len(t) > MAX_QUESTION_CHARS:
            continue
        if guard_tokens(t) != want or opposite_conflict(t, question) or looks_multi_question(t) or is_sensitive(t):
            continue
        seen.add(qkey(t))
        out.append(t)
        if len(out) >= MAX_PARAPHRASES:
            break
    return out


def new_row(*, agent_id: str, user_id: str, question: str, answer: str, language: str, source: str, status: str,
            paraphrases: Optional[List[str]] = None, fingerprint: Optional[str] = None, count: int = 0,
            embed: Optional[Embedder] = None) -> dict:
    row = {"agent_id": agent_id, "user_id": user_id, "question": question.strip(), "answer": answer.strip(),
           "language": lang_code(language), "source": source, "status": status, "paraphrases": list(paraphrases or []),
           "docs_fingerprint": fingerprint, "times_asked": int(count)}
    if answer.strip():
        row.update(embed_row(row, embed))
    return row


def public_view(row: dict) -> dict:
    """What the API returns: no vectors."""
    return {"id": row["id"], "question": row.get("question") or "", "answer": row.get("answer") or "",
            "language": row.get("language") or "en", "status": row.get("status") or STATUS_SUGGESTED,
            "source": row.get("source") or "owner", "paraphrases": list(row.get("paraphrases") or []),
            "times_asked": int(row.get("times_asked") or 0), "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at")}


def visible(rows: Iterable[dict]) -> List[dict]:
    return [r for r in rows if r.get("status") != STATUS_DISMISSED]


@dataclass
class BuildReport:
    added: int = 0
    updated: int = 0
    skipped: List[dict] = field(default_factory=list)


def build(*, agent: dict, user_id: str, storage_mode: str, candidates: Sequence[Candidate], languages: Optional[Sequence[str]],
          deps: DraftDeps, fingerprint: Optional[str], store: Any = None,
          progress: Optional[Callable[[int, int, str], None]] = None,
          should_stop: Optional[Callable[[], bool]] = None) -> BuildReport:
    """Draft an answer for every candidate (blocking). New drafts are stored as "suggested"; a row that needed a draft (an
    owner question without an answer, or an answer whose documents changed) is updated in place and becomes "suggested".
    Questions the documents do not support are skipped and listed in the report with the reason. Stops early if
    `should_stop()` is true. `progress(done, total, question)` is called before each draft."""
    store = store or get_store(storage_mode)
    agent_id = agent["id"]
    supported = [str(x) for x in (agent.get("supported_languages") or ["en"])]
    has_tools = bool(agent.get("tools"))
    report = BuildReport()
    jobs: List[Tuple[Candidate, str]] = []
    for c in candidates:
        own = lang_code(c.language or language_policy.detect_language(c.question, supported))
        for lg in dict.fromkeys([own, *[lang_code(x) for x in (languages or [])]]):
            jobs.append((c, lg))
    have = {(qkey(r["question"]), lang_code(r.get("language"))): r for r in store.list(user_id, agent_id)}
    total = len(jobs)
    for n, (c, lg) in enumerate(jobs):
        if should_stop is not None and should_stop():
            break
        if progress is not None:
            progress(n, total, c.question)
        redraft = bool(c.row_id) and lg == lang_code(c.language or lg)
        if not redraft and (qkey(c.question), lg) in have:
            report.skipped.append({"question": c.question, "reason": "you already have this question"})
            continue
        res = draft(c.question, lg, deps, has_tools=has_tools)
        if not res.ok:
            report.skipped.append({"question": c.question, "reason": res.reason})
            continue
        if redraft:
            fields = {"answer": res.answer, "paraphrases": res.paraphrases, "status": STATUS_SUGGESTED,
                      "docs_fingerprint": fingerprint, "language": lg}
            fields.update(embed_row({"question": c.question, "paraphrases": res.paraphrases}, deps.embed))
            if store.update(user_id, agent_id, c.row_id, fields) is not None:
                report.updated += 1
            continue
        row = new_row(agent_id=agent_id, user_id=user_id, question=c.question, answer=res.answer, language=lg,
                      source=c.source, status=STATUS_SUGGESTED, paraphrases=res.paraphrases, fingerprint=fingerprint,
                      count=c.count, embed=deps.embed)
        have[(qkey(c.question), lg)] = store.insert(row)
        report.added += 1
    if progress is not None:
        progress(total, total, "")
    return report
