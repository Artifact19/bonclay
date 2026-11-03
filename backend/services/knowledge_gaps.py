"""Knowledge-gap report: the questions callers asked that the agent could not answer from its documents.

Pure logic, no web framework: `routers/analytics.py` loads the turns (local JSON store or Supabase) and calls
`build_report`. What a stored turn tells us (see `backend._turn_store`): the agent row has `text_content`, `fallback`
(True only when the language model call itself failed, which is an outage, not a gap), `language`, `timings` and
`retrieval` ({n_chunks, top_score, ...}, see `backend._retrieval_signals`); the caller's words are the user row just
before it. Three signals mark a turn as a gap, in this order:
  1. "said it didn't know": the agent's guardrail prompt makes it say "I don't have that information" when the context
     lacks the answer, and `is_gap_reply` recognises that phrase in several languages;
  2. "no matching document": retrieval found no chunk at all (only for agents that have found chunks on other turns, so
     an agent without a knowledge base does not flag every call);
  3. "weak match": the best dense cosine among the retrieved chunks is below KNOWLEDGE_GAP_MIN_SCORE (default 0.35).
     Chosen with bench/gap_threshold.py (BGE-M3, 45 answerable questions vs 120 unanswerable): at 0.35 it flags 2% to 4%
     of answerable questions and catches 24% to 27% of unanswerable ones, so it only fires on clearly unrelated
     questions; 0.40 would catch 55% to 59% but flag 7% of answerable ones (Hinglish questions score lowest).
Signals 2 and 3 need at least MIN_SIGNAL_WORDS words in the question, so "hello" or "thanks" is never a gap, and ignore
turns whose retrieval failed (an outage, not a gap). Older turns without `retrieval` are judged by the reply alone.

Privacy: every question is passed through `redact` before it is embedded, cached or returned; audio never enters here.
"""
import os
import re
import threading
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SIMILARITY_THRESHOLD = 0.75
MAX_TURNS = 2000
MAX_CLUSTERS = 50
MAX_EXAMPLES = 3
MAX_TEXT_CHARS = 300
CACHE_TTL_S = 300

DEFAULT_MIN_SCORE = 0.35   # KNOWLEDGE_GAP_MIN_SCORE: best retrieved dense cosine below this is a "weak match"
MIN_SIGNAL_WORDS = 3       # shorter questions (greetings, thanks) are never flagged by the retrieval signals
REASON_SAID = "said it didn't know"
REASON_NONE = "no matching document"
REASON_WEAK = "weak match"


def min_score() -> float:
    """The weak-match threshold, read from the environment on every call (so it can change without a restart)."""
    try:
        return float(os.getenv("KNOWLEDGE_GAP_MIN_SCORE") or DEFAULT_MIN_SCORE)
    except ValueError:
        return DEFAULT_MIN_SCORE

# -- gap detection ---------------------------------------------------------------------------------------------------

# Lower-case, curly apostrophes already straightened. Substring match; kept specific enough not to fire on answers.
GAP_PHRASES: Tuple[str, ...] = (
    # English (the guardrail prompt's own wording first)
    "don't have that information", "do not have that information", "don't have any information",
    "do not have any information", "don't have information", "i don't have the details", "not in my documents",
    "not in the documents", "not in my knowledge", "couldn't find", "could not find", "can't find", "cannot find",
    "unable to find", "i'm not sure about that", "i am not sure about that", "i don't know", "i do not know",
    "no information about", "not something i have", "outside what i know", "connect you with a member of staff",
    "member of our staff", "member of staff can",
    # Hindi (Devanagari)
    "मेरे पास यह जानकारी नहीं", "मेरे पास इसकी जानकारी नहीं", "मेरे पास जानकारी नहीं", "जानकारी उपलब्ध नहीं",
    "जानकारी नहीं है", "मुझे नहीं पता", "मुझे पता नहीं", "पता नहीं है", "मुझे यह नहीं पता", "मुझे जानकारी नहीं",
    "मुझे इसकी जानकारी नहीं", "स्टाफ के सदस्य", "कर्मचारी से जोड़",
    # Hinglish / romanised Hindi
    "mere paas yeh jaankari nahi", "mere paas ye jaankari nahi", "mere paas jaankari nahi",
    "mere paas is baare mein jaankari nahi", "mere paas information nahi", "jaankari nahi hai", "jankari nahi hai",
    "jaankari uplabdh nahi", "mujhe nahi pata", "mujhe pata nahi", "pata nahi hai", "mujhe iski jaankari nahi",
    "staff se baat", "staff member se",
    # Marathi / Gujarati / Bengali / Tamil / Telugu (short, common refusals)
    "माझ्याकडे ही माहिती नाही", "मला माहित नाही", "મારી પાસે આ માહિતી નથી", "আমার কাছে এই তথ্য নেই",
    "என்னிடம் அந்த தகவல் இல்லை", "నా దగ్గర ఆ సమాచారం లేదు",
    # Spanish / French / German / Portuguese
    "no tengo esa información", "no tengo información", "je n'ai pas cette information", "je ne dispose pas",
    "ich habe keine informationen", "ich habe diese informationen nicht", "não tenho essa informação",
)

_APOS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").translate(_APOS).lower()).strip()


def is_gap_reply(reply: str) -> bool:
    """True when the agent's reply says it does not have the answer (or hands the caller to a human for lack of one)."""
    t = _norm(reply)
    return bool(t) and any(p in t for p in GAP_PHRASES)


# -- redaction -------------------------------------------------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_AADHAAR = re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}(?!\d)")
_PHONE_IN = re.compile(r"(?<![\w])(?:\+\s?91[\s-]?|91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_LONG_DIGITS = re.compile(r"(?<!\d)(?:\d[ -]?){5,}\d(?!\d)")


def redact(text: str) -> str:
    """Strip emails, Indian phone numbers (10 digits, +91), Aadhaar-like 12-digit groups and long digit runs."""
    t = (text or "")[:2000]
    t = _EMAIL.sub("[email]", t)
    t = _AADHAAR.sub("[id]", t)
    t = _PHONE_IN.sub("[phone]", t)
    t = _LONG_DIGITS.sub("[number]", t)
    t = _WS.sub(" ", t).strip()
    return t[:MAX_TEXT_CHARS]


# -- turn pairing ----------------------------------------------------------------------------------------------------

def pair_turns(rows: Sequence[dict], conv_agent: Dict[str, str]) -> List[dict]:
    """(question, reply) pairs: each agent turn with the nearest earlier user turn of the same conversation.
    `rows` are conversation_turns (any order); `conv_agent` maps conversation id to agent id and defines ownership."""
    by_conv: Dict[str, List[dict]] = {}
    for r in rows:
        cid = r.get("conversation_id")
        if cid in conv_agent:
            by_conv.setdefault(cid, []).append(r)
    pairs: List[dict] = []
    for cid, turns in by_conv.items():
        turns.sort(key=lambda r: str(r.get("created_at") or ""))  # stable: insertion order breaks ties
        question: Optional[dict] = None
        for t in turns:
            if t.get("role") == "user":
                question = t
            elif t.get("role") == "agent" and question is not None:
                pairs.append({"agent_id": conv_agent[cid], "created_at": str(t.get("created_at") or ""),
                              "question": question.get("text_content") or "", "reply": t.get("text_content") or "",
                              "fallback": bool(t.get("fallback")), "language": t.get("language"),
                              "retrieval": t.get("retrieval") if isinstance(t.get("retrieval"), dict) else None})
                question = None
    pairs.sort(key=lambda p: p["created_at"], reverse=True)
    return pairs[:MAX_TURNS]


def gap_reason(p: dict, has_kb: bool, threshold: float) -> Optional[str]:
    """Why a turn is a knowledge gap (REASON_*), or None when it is not."""
    if is_gap_reply(p["reply"]):
        return REASON_SAID
    r = p.get("retrieval")
    if not r or r.get("error") or len((p["question"] or "").split()) < MIN_SIGNAL_WORDS:
        return None
    n = r.get("n_chunks")
    if n == 0 and has_kb:
        return REASON_NONE
    score = r.get("top_score")
    if n and isinstance(score, (int, float)) and score < threshold:
        return REASON_WEAK
    return None


def find_gaps(pairs: Sequence[dict]) -> Tuple[List[dict], int]:
    """(gap turns with redacted questions and a `reason`, total answerable turns). Turns where the model call failed
    (`fallback`) are outages, not missing knowledge, and count toward neither number."""
    threshold = min_score()
    pairs = [p if isinstance(p.get("retrieval"), dict) or p.get("retrieval") is None else {**p, "retrieval": None}
             for p in pairs]
    has_kb = {p["agent_id"] for p in pairs if (p.get("retrieval") or {}).get("n_chunks")}
    gaps: List[dict] = []
    total = 0
    for p in pairs:
        if p["fallback"] or not (p["question"] or "").strip():
            continue
        total += 1
        reason = gap_reason(p, p["agent_id"] in has_kb, threshold)
        if reason:
            q = redact(p["question"])
            if q:
                gaps.append({**p, "question": q, "reason": reason})
    return gaps, total


# -- clustering ------------------------------------------------------------------------------------------------------

Embedder = Callable[[List[str]], Optional[np.ndarray]]


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def greedy_clusters(vectors: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> List[List[int]]:
    """Greedy single pass: each vector joins the cluster whose running centroid it is most similar to (when that is at
    least `threshold`), else starts one. Returns member index lists."""
    v = _unit(vectors)
    centroids: List[np.ndarray] = []  # unnormalised sums
    members: List[List[int]] = []
    for i in range(len(v)):
        best, best_sim = -1, -1.0
        for c, s in enumerate(centroids):
            sim = float(v[i] @ s) / (float(np.linalg.norm(s)) + 1e-9)
            if sim > best_sim:
                best, best_sim = c, sim
        if best >= 0 and best_sim >= threshold:
            members[best].append(i)
            centroids[best] = centroids[best] + v[i]
        else:
            members.append([i])
            centroids.append(v[i].copy())
    return members


def cluster_agent_gaps(items: List[dict], embed: Optional[Embedder]) -> Tuple[List[List[dict]], Optional[Dict[str, np.ndarray]]]:
    """Group one agent's gap turns by question similarity. Identical questions share one vector; with no embedder
    (model unavailable) only identical questions group together. Also returns the vector of each distinct question."""
    by_text: Dict[str, List[dict]] = {}
    for it in items:
        by_text.setdefault(_norm(it["question"]), []).append(it)
    texts = list(by_text)
    vecs = None
    if embed is not None and len(texts) > 1:
        try:
            out = embed([by_text[t][0]["question"] for t in texts])
            vecs = np.asarray(out, dtype=np.float32) if out is not None else None
        except Exception:  # noqa: BLE001 - embeddings are a nicety; exact-text grouping still answers the question
            vecs = None
        if vecs is not None and (vecs.ndim != 2 or len(vecs) != len(texts)):
            vecs = None
    if vecs is None:
        return [by_text[t] for t in texts], None
    groups = [[it for i in g for it in by_text[texts[i]]] for g in greedy_clusters(vecs)]
    return groups, {t: vecs[i] for i, t in enumerate(texts)}


def _representative(group: List[dict], vecs: Optional[Dict[str, np.ndarray]]) -> str:
    """The question closest to the group's centre; with no vectors, the most repeated one."""
    texts = Counter(_norm(g["question"]) for g in group)
    if vecs and len(texts) > 2:
        keys = list(texts)
        m = _unit(np.stack([vecs[k] for k in keys]))
        pick = keys[int(np.argmax(m @ m.mean(axis=0)))]
    else:
        pick = texts.most_common(1)[0][0]
    return next(g["question"] for g in group if _norm(g["question"]) == pick)


def build_clusters(gaps: Sequence[dict], agent_names: Dict[str, str], embed: Optional[Embedder]) -> List[dict]:
    by_agent: Dict[str, List[dict]] = {}
    for g in gaps:
        by_agent.setdefault(g["agent_id"], []).append(g)
    clusters: List[dict] = []
    for agent_id, items in by_agent.items():
        groups, vecs = cluster_agent_gaps(items, embed)
        for group in groups:
            rep = _representative(group, vecs)
            seen, examples = {_norm(rep)}, [rep]
            for it in sorted(group, key=lambda x: x["created_at"], reverse=True):
                k = _norm(it["question"])
                if k not in seen and len(examples) < MAX_EXAMPLES:
                    seen.add(k)
                    examples.append(it["question"])
            langs = Counter(it["language"] for it in group if it.get("language"))
            reasons = Counter(it.get("reason") or REASON_SAID for it in group)
            order = (REASON_SAID, REASON_NONE, REASON_WEAK)  # the strongest signal wins a tie
            reason = max(reasons, key=lambda r: (reasons[r], -order.index(r) if r in order else -9))
            clusters.append({
                "question": rep, "examples": examples, "count": len(group),
                "last_asked": max(it["created_at"] for it in group), "agent_id": agent_id,
                "agent_name": agent_names.get(agent_id), "language": langs.most_common(1)[0][0] if langs else None,
                "reason": reason, "reasons": dict(reasons),
            })
    clusters.sort(key=lambda c: (c["count"], c["last_asked"]), reverse=True)
    return clusters[:MAX_CLUSTERS]


# -- report + cache --------------------------------------------------------------------------------------------------

_cache: Dict[tuple, Tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def cache_get(key: tuple) -> Optional[dict]:
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
            return hit[1]
    return None


def cache_put(key: tuple, value: dict) -> None:
    with _cache_lock:
        if len(_cache) > 64:
            _cache.clear()
        _cache[key] = (time.monotonic(), value)


def build_report(pairs: Sequence[dict], agent_names: Dict[str, str], embed: Optional[Embedder], days: int) -> dict:
    gaps, total = find_gaps(pairs)
    return {"clusters": build_clusters(gaps, agent_names, embed), "total_gap_turns": len(gaps),
            "total_turns": total, "window_days": days}
