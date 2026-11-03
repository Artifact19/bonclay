"""Question index (doc2query): the questions a caller might ask, written once per passage when a document is stored.

A caller's question lands closer to another question than to a paragraph, so each passage gets 2-3 questions in each of
three forms (English, Hindi in Devanagari, Roman Hinglish) written by the hosted language model. Each question is stored
as its own row (`kind="question"`, `parent_chunk_id` = the passage, `chunk_text`/`content` = the question, its own
embedding); retrieval scores them like passages and hands the parent passage to the model, never the question.

Nothing here runs on a call and nothing blocks an upload: `submit()` queues the work on one background thread after the
document is stored and usable. The thread paces itself for Groq's free tier (a token budget per minute plus a minimum gap
between calls), batches several passages per call, retries politely when the provider says 429, and stops quietly when
no hosted model is configured (the agent keeps working without question rows).

Switches (backend/.env.example):
    QUESTION_INDEX=on|off              default on
    QUESTION_INDEX_BATCH=4             passages per model call
    QUESTION_INDEX_TPM=4000            model tokens this job may use per minute (live replies share the same quota)
    QUESTION_INDEX_MIN_INTERVAL_S=2    smallest gap between two model calls
    QUESTION_INDEX_RETRIES=4           tries per call when the provider is rate limited or fails
    QUESTION_INDEX_BACKOFF_S=15        first wait after a 429 (doubles each time, at most 120)

Local fallback (services/background_llm.py): when the hosted model answers 429, has used its daily quota, fails, or is not
configured, a batch is written by the local background model (BACKGROUND_LOCAL_MODEL, default Gemma 4 E4B) instead, but only
while the app is idle, memory allows it and no call is live; otherwise the job waits and asks again. A reply from the local
model is checked harder (strict JSON repair, every item must be a question in its language, numbers must come from the
passage). BACKGROUND_LLM_FALLBACK=off keeps the hosted-only behaviour.
"""
import atexit
import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

try:
    from services import background_llm, llm
except ImportError:  # pragma: no cover - imported as backend.services.question_index
    from backend.services import background_llm, llm

log = logging.getLogger("bonclay.question_index")

HOSTED_PROVIDERS = ["groq", "openai"]      # the live chain's local model (a 3B Llama) writes poor Hindi questions; the
                                           # background model (Gemma 4 E4B) is the only local one used, see background_llm
MIN_PASSAGE_WORDS = 5                      # a shorter passage (a bare title) answers nothing a caller would ask
MAX_QUESTION_WORDS = 25
MIN_QUESTION_WORDS = 2
PER_LANGUAGE_MAX = 3
PER_PASSAGE_MAX = 9
PROMPT_PASSAGE_CHARS = 1600
BATCH_CHAR_BUDGET = 6000
OUTPUT_TOKENS_PER_PASSAGE = 260
EMBED_BATCH = 32


# ----------------------------------------------------------------------------------------------------- settings

def _env_float(name: str, default: float, low: float = 0.0) -> float:
    try:
        return max(low, float(os.getenv(name, "") or default))
    except ValueError:
        return default


def enabled() -> bool:
    return os.getenv("QUESTION_INDEX", "on").strip().lower() not in ("off", "0", "false", "no")


def batch_size() -> int:
    return max(1, int(_env_float("QUESTION_INDEX_BATCH", 4, 1)))


def hosted_available() -> bool:
    """True when a hosted (Groq or OpenAI-compatible) model is configured."""
    try:
        order = llm._order()
        return any(name in HOSTED_PROVIDERS and llm.PROVIDERS[name][0]() is None for name in order)
    except Exception:  # noqa: BLE001
        return False


def available() -> bool:
    """A model can write the questions: a hosted one, or the local background model (downloaded, fallback not off)."""
    return hosted_available() or background_llm.local_ready()


class Skipped(RuntimeError):
    """The index cannot be built now (no hosted model, switched off, document gone)."""


# --------------------------------------------------------------------------------------------- stopping / sleeping

_stop = threading.Event()


def _sleep(seconds: float) -> None:
    if _stop.wait(seconds):
        raise Skipped("shutting down")


atexit.register(_stop.set)


class Pacer:
    """Keeps this job inside a token-per-minute budget and a minimum gap between calls (Groq free tier)."""

    def __init__(self, tpm: Optional[float] = None, min_interval: Optional[float] = None, *,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = _sleep):
        self.tpm = tpm if tpm is not None else _env_float("QUESTION_INDEX_TPM", 4000, 500)
        self.min_interval = min_interval if min_interval is not None else _env_float("QUESTION_INDEX_MIN_INTERVAL_S", 2.0)
        self.clock, self.sleep = clock, sleep
        self.spent: deque = deque()      # (time, tokens)
        self.last_call = -1e9

    def wait(self, tokens: int) -> float:
        """Sleep until a call of about `tokens` fits; returns the seconds waited."""
        waited = 0.0
        tokens = min(tokens, int(self.tpm))    # one call bigger than the whole budget still has to be possible
        while True:
            now = self.clock()
            while self.spent and now - self.spent[0][0] >= 60:
                self.spent.popleft()
            used = sum(t for _, t in self.spent)
            gap = self.min_interval - (now - self.last_call)
            need = 0.0
            if used + tokens > self.tpm and self.spent:
                # wait until enough of the oldest calls have aged out of the minute
                running, free_at = used, now
                for ts, t in self.spent:
                    running -= t
                    free_at = ts + 60
                    if running + tokens <= self.tpm:
                        break
                need = max(need, free_at - now)
            need = max(need, gap)
            if need <= 0:
                return waited
            self.sleep(need)
            waited += need

    def record(self, tokens: int) -> None:
        now = self.clock()
        self.spent.append((now, int(tokens)))
        self.last_call = now


# ------------------------------------------------------------------------------------------------ the prompt

SYSTEM_PROMPT = (
    "You write search questions for a phone assistant's knowledge base. You receive numbered passages from a business "
    "document. For each passage write the questions a caller might ask that the passage ANSWERS.\n"
    "Rules:\n"
    "- Only questions the passage text itself answers. Never use a fact that is not in the passage. If the passage "
    "answers nothing a caller would ask (for example it is only a title), return empty lists.\n"
    "- Every question must name its subject (the department, service, item or place) so it makes sense on its own, and "
    "must be short, at most 14 words, the way a person speaks on the phone.\n"
    "- Give 2 or 3 questions in each of three forms: \"en\" = English; \"hi\" = Hindi in Devanagari script; "
    "\"hinglish\" = Hindi written in Roman letters, the way people text it (for example \"Cardiology ki OPD kab hoti hai?\").\n"
    "- Do not repeat a question. Keep numbers and names exactly as the passage writes them.\n"
    "Reply with JSON only: {\"items\": [{\"id\": <number>, \"en\": [...], \"hi\": [...], \"hinglish\": [...]}]} with one "
    "item per passage id."
)


def _passage_text(p: dict) -> str:
    text = f"{p.get('header') or ''}\n{p.get('content') or ''}".strip()
    return text if len(text) <= PROMPT_PASSAGE_CHARS else text[:PROMPT_PASSAGE_CHARS].rsplit(" ", 1)[0]


def build_messages(batch: Sequence[tuple]) -> List[dict]:
    """`batch` is [(number, passage dict)]."""
    payload = [{"id": n, "text": _passage_text(p)} for n, p in batch]
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Passages (JSON):\n" + json.dumps(payload, ensure_ascii=False)}]


def estimate_tokens(batch: Sequence[tuple]) -> int:
    chars = len(SYSTEM_PROMPT) + sum(len(_passage_text(p)) for _, p in batch)
    return int(chars / 3.2) + OUTPUT_TOKENS_PER_PASSAGE * len(batch)


# --------------------------------------------------------------------------------------------- reading the reply

def _script_of(q: str) -> str:
    letters = [c for c in q if c.isalpha()]
    if not letters:
        return "other"
    dev = sum(1 for c in letters if "ऀ" <= c <= "ॿ")
    return "hi" if dev / len(letters) > 0.5 else "latin"


def _key(q: str) -> str:
    kept = [c if (unicodedata.category(c)[0] in "LMN") else " " for c in q.casefold()]
    return " ".join("".join(kept).split())


def clean_question(q) -> str:
    if not isinstance(q, str):
        return ""
    q = " ".join(q.replace("​", "").split())
    q = re.sub(r"^(?:[-*•]|\d{1,2}[.)])\s+", "", q).strip().strip('"“”')
    n = len(q.split())
    if n < MIN_QUESTION_WORDS or n > MAX_QUESTION_WORDS:
        return ""
    if not q.endswith(("?", "？")):
        q = q.rstrip(".!। ") + "?"
    return q


_QUESTION_WORDS = {
    "hinglish": {"kya", "kab", "kahan", "kahaan", "kaun", "kaunsa", "kaunsi", "kitna", "kitni", "kitne", "kaise", "kyun",
                 "kyon", "kis", "kisse", "kisko", "kidhar", "kon", "konsa", "konsi"},
    "hi": {"क्या", "कब", "कहाँ", "कहां", "कौन", "कौनसा", "कौनसी", "कितना", "कितनी", "कितने", "कैसे", "क्यों", "किस", "किसे",
           "किसको", "किधर"},
}
# An English question starts with one of these (a statement such as "The OPD is open" does not).
_ENGLISH_STARTERS = {"what", "when", "where", "who", "whom", "whose", "which", "why", "how", "is", "are", "do", "does", "did",
                     "can", "could", "will", "would", "should", "may", "has", "have", "was", "were", "tell", "give"}
# Words that mark Roman Hindi. An "en" question with two of them is Hinglish, a "hinglish" one with none is English.
_HINGLISH_MARKERS = {"kya", "kab", "kahan", "kahaan", "kaun", "kitna", "kitni", "kitne", "kaise", "kyun", "kyon", "hai", "hain",
                     "ka", "ki", "ke", "ko", "mein", "me", "se", "par", "pe", "hota", "hoti", "hote", "milta", "milti",
                     "chahiye", "kare", "karen", "karna", "batao", "bataiye", "bataye", "aap", "hum", "nahi", "nahin", "wala",
                     "wali", "ho", "tak", "liye", "lie", "kitna", "kidhar", "sakta", "sakte", "sakti", "milega", "milegi"}
_DIGITS = re.compile(r"\d+")


def _words_of(q: str) -> List[str]:
    return [w for w in re.split(r"[^\w\u0900-\u097F]+", q.casefold()) if w]


def looks_like_question(q: str, lang: str) -> bool:
    """As the model wrote it (before clean_question adds a "?"): a question mark at the end, an English question word first,
    or an interrogative of the language anywhere ("Cardiology OPD kab hoti hai")."""
    q = q.strip().strip('"\u201c\u201d ')
    if q.endswith(("?", "\uff1f")):
        return True
    words = _words_of(q)
    if lang == "en":
        return bool(words) and words[0] in _ENGLISH_STARTERS
    return bool(set(words) & _QUESTION_WORDS.get(lang, set()))


def _hinglish_markers(q: str) -> int:
    return sum(1 for w in _words_of(q) if w in _HINGLISH_MARKERS)


def _numbers_from_passage(q: str, passage: Optional[str]) -> bool:
    """Every number in the question appears in the passage (the question may not invent one)."""
    if passage is None:
        return True
    have = set(_DIGITS.findall(passage))
    return all(n in have for n in _DIGITS.findall(q))


def check_strict(q: str, lang: str, passage: Optional[str] = None, raw: Optional[str] = None) -> bool:
    """The extra checks for a question written by the small local model: it is a question (`raw` is what the model wrote,
    since clean_question adds the "?"), it is in its language's script (Devanagari for "hi"; Roman letters for "en" and
    "hinglish"; Roman Hindi only in "hinglish"), and its numbers come from the passage."""
    script = _script_of(q)
    if lang == "hi" and script != "hi":
        return False
    if lang in ("en", "hinglish") and script != "latin":
        return False
    if lang == "en" and _hinglish_markers(q) >= 2:
        return False
    if lang == "hinglish" and _hinglish_markers(q) == 0:
        return False
    return looks_like_question(q if raw is None else raw, lang) and _numbers_from_passage(q, passage)


_QUOTED = re.compile(r'"((?:[^"\\\n]|\\.)*)"')


def salvage_items(text: str, wanted: Sequence[int]) -> dict:
    """{"items": [...]} from a reply whose JSON is broken beyond repair (the small model sometimes writes a stray token in
    the middle of a list): each `"id": N` starts an item, each `"en"|"hi"|"hinglish": [` starts a list whose quoted strings are
    taken up to the next key. What this recovers is judged by parse_reply's strict checks like everything else.
    Raises ValueError when no item is found."""
    marks = [(m.start(), int(m.group(1))) for m in re.finditer(r'"id"\s*:\s*(\d+)', text)]
    items = []
    for k, (start, pid) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        chunk = text[start:end]
        item: dict = {"id": pid}
        keys = [(m.start(), m.group(1)) for m in re.finditer(r'"(en|hi|hinglish)"\s*:\s*\[', chunk)]
        for j, (kstart, lang) in enumerate(keys):
            kend = keys[j + 1][0] if j + 1 < len(keys) else len(chunk)
            body = chunk[kstart:kend].split("[", 1)[1]
            body = body.split("]", 1)[0]
            item[lang] = [q for q in _QUOTED.findall(body) if len(q.split()) >= MIN_QUESTION_WORDS]
        if len(item) > 1:
            items.append(item)
    if not items:
        raise ValueError("no items could be recovered from the reply")
    return {"items": items}


def parse_reply(text: str, wanted: Sequence[int], *, strict: bool = False,
                passages: Optional[Dict[int, str]] = None) -> Dict[int, List[str]]:
    """{passage number: [questions]} from the model's JSON. Each question is checked by script: a Devanagari one belongs to
    "hi", a Latin-script one in "hi" to "hinglish" and one in "hinglish" or "en" stays where the model put it. Duplicates
    (ignoring case and punctuation) are dropped; at most 3 per language and 9 per passage; the order is en, hi, hinglish.

    `strict` (the local model): the JSON is repaired when it has slips (background_llm.parse_json), and a question that
    fails `check_strict` is dropped (a statement, the wrong script, a number the passage does not have). `passages` maps
    a passage number to its text for that number check."""
    try:
        data = json.loads(text)
    except ValueError:
        if not strict:
            raise
        try:
            data = background_llm.parse_json(text)
        except ValueError:
            data = salvage_items(text, wanted)      # a glitch in the middle: keep the items before and after it
    items = data.get("items") if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = [{"id": k, **v} for k, v in items.items() if isinstance(v, dict)]
    if not isinstance(items, list):
        raise ValueError("no items in the reply")
    out: Dict[int, List[str]] = {}
    wanted_set = set(wanted)
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if pid not in wanted_set or pid in out:
            continue
        buckets: Dict[str, List[str]] = {"en": [], "hi": [], "hinglish": []}
        listed = []
        for lang in ("en", "hi", "hinglish"):
            vals = item.get(lang) or []
            if isinstance(vals, str):
                vals = [vals]
            listed += [(lang, v) for v in vals if isinstance(v, str)]
        for v in item.get("questions") or []:      # tolerate one flat list
            listed.append(("en", v))
        seen = set()
        for lang, raw in listed:
            q = clean_question(raw)
            if not q or _key(q) in seen:
                continue
            script = _script_of(q)
            if script == "hi":
                lang = "hi"
            elif lang == "hi":
                lang = "hinglish"
            if strict and not check_strict(q, lang, (passages or {}).get(pid), raw=raw):
                continue
            if len(buckets[lang]) >= PER_LANGUAGE_MAX:
                continue
            seen.add(_key(q))
            buckets[lang].append(q)
        out[pid] = (buckets["en"] + buckets["hi"] + buckets["hinglish"])[:PER_PASSAGE_MAX]
    return out


# ---------------------------------------------------------------------------------------------------- the calls

def _is_offline(e: "llm.LLMUnavailable") -> bool:
    """No provider was even tried (no key, not configured), as opposed to a provider that failed."""
    reasons = getattr(e, "reasons", {}) or {}
    return not any(str(r).startswith("failed") for r in reasons.values())


def _is_rate_limited(e: BaseException) -> bool:
    text = " ".join(str(v) for v in (getattr(e, "reasons", None) or {}).values()) + " " + str(e)
    return "429" in text or "rate limit" in text.lower() or "rate_limit" in text.lower()


def _fallback_allowed() -> bool:
    try:
        return background_llm.local_ready()
    except Exception:  # noqa: BLE001
        return False


def local_batch_size() -> int:
    """Passages per request to the local model (BACKGROUND_LLM_BATCH, default 2): a small model slips more on long lists."""
    return max(1, int(_env_float("BACKGROUND_LLM_BATCH", 2, 1)))


def _local_batch(batch: Sequence[tuple], *, sleep: Callable[[float], None], local_chat: Optional[Callable] = None) -> Dict[int, List[str]]:
    """The questions of `batch` from the local background model, a few passages per request, each checked strictly. A request
    whose reply cannot be read is made once more; then the job fails (RuntimeError) and what was saved stays."""
    out: Dict[int, List[str]] = {}
    size = local_batch_size()
    for k in range(0, len(batch), size):
        sub = list(batch[k:k + size])
        messages = build_messages(sub)
        texts = {n: _passage_text(p) for n, p in sub}
        last: Optional[BaseException] = None
        for attempt in range(2):
            try:
                res = background_llm.run_local(
                    messages, max_tokens=int(OUTPUT_TOKENS_PER_PASSAGE * 1.6) * len(sub) + 300,
                    temperature=0.2 if attempt == 0 else 0.5, sleep=sleep, chat=local_chat)
            except background_llm.Busy as e:
                raise RuntimeError(f"question generation waited for the local model: {str(e)[:160]}") from e
            try:
                out.update(parse_reply(res.text, [n for n, _ in sub], strict=True, passages=texts))
                break
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                last = e
                log.warning("question index: unreadable local reply (%s); try %d of 2", str(e)[:100], attempt + 1)
        else:
            raise RuntimeError(f"question generation failed: {str(last)[:200]}")
    return out


def generate_batch(batch: Sequence[tuple], *, chat: Optional[Callable] = None, pacer: Optional[Pacer] = None,
                   sleep: Callable[[float], None] = _sleep, local_chat: Optional[Callable] = None,
                   local: Optional[bool] = None) -> Dict[int, List[str]]:
    """Questions for one batch [(number, passage)]. The hosted model is tried first; when it is rate limited, out of quota,
    failing or not configured, the local background model writes the batch (background_llm.run_local waits for an idle
    app and enough memory). Without the local fallback: retries on 429 and other provider failures with a growing wait,
    Skipped when no hosted model is configured. RuntimeError when the retries run out or the local model stayed busy.

    `local=True` goes straight to the local model (BACKGROUND_LLM_FORCE=on does the same). `local_chat` replaces the local
    model call (tests)."""
    chat = chat or llm.chat
    pacer = pacer or Pacer()
    messages = build_messages(batch)
    est = estimate_tokens(batch)
    wanted = [n for n, _ in batch]
    texts = {n: _passage_text(p) for n, p in batch}
    retries = int(_env_float("QUESTION_INDEX_RETRIES", 4, 1))
    backoff = _env_float("QUESTION_INDEX_BACKOFF_S", 15.0)
    to_local = background_llm.forced() if local is None else local
    last: Optional[BaseException] = None
    bad = 0
    for attempt in range(retries):
        text: Optional[str] = None
        strict = False
        if not to_local:
            pacer.wait(est)
            try:
                res = chat(messages, json_mode=True, temperature=0.3,
                           max_tokens=OUTPUT_TOKENS_PER_PASSAGE * len(batch) + 200, timeout=60.0,
                           providers=HOSTED_PROVIDERS)
            except llm.LLMUnavailable as e:
                fallback = _fallback_allowed()
                if _is_offline(e) and not fallback:
                    raise Skipped("no hosted language model: " + "; ".join(f"{k}: {v}" for k, v in e.reasons.items())) from e
                last = e
                pacer.record(est)         # a rejected call still counts against the minute on the provider's side
                if not fallback:
                    wait = min(120.0, backoff * (2 ** attempt))
                    log.warning("question index: model call failed (%s); waiting %.0fs (try %d of %d)",
                                "rate limited" if _is_rate_limited(e) else str(e)[:120], wait, attempt + 1, retries)
                    if attempt + 1 < retries:
                        sleep(wait)
                    continue
                log.info("question index: the hosted model is not available (%s); using the local model",
                         "rate limited" if _is_rate_limited(e) else "offline" if _is_offline(e) else str(e)[:80])
            else:
                usage = getattr(res, "usage", None) or {}
                pacer.record(int(usage.get("total_tokens") or est))
                text = res.text
        if text is None:                                  # the local background model
            return _local_batch(batch, sleep=sleep, local_chat=local_chat)
        try:
            return parse_reply(text, wanted, strict=strict, passages=texts if strict else None)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            last = e
            bad += 1
            log.warning("question index: unreadable reply (%s); try %d of %d", str(e)[:100], attempt + 1, retries)
            if bad >= 2:     # a model that cannot produce the JSON twice will not on the third try
                break
    raise RuntimeError(f"question generation failed: {str(last)[:200]}")


def make_batches(passages: Sequence[dict], size: Optional[int] = None) -> List[List[tuple]]:
    """[(index, passage)] groups of at most `size` passages and BATCH_CHAR_BUDGET characters; passages too short to be
    asked about are left out."""
    size = size or batch_size()
    batches: List[List[tuple]] = []
    cur: List[tuple] = []
    chars = 0
    for i, p in enumerate(passages):
        if len((p.get("content") or "").split()) < MIN_PASSAGE_WORDS:
            continue
        n = len(_passage_text(p))
        if cur and (len(cur) >= size or chars + n > BATCH_CHAR_BUDGET):
            batches.append(cur)
            cur, chars = [], 0
        cur.append((i, p))
        chars += n
    if cur:
        batches.append(cur)
    return batches


def _index_passages(passages: Sequence[dict], *, embed: Callable[[List[str]], Sequence], save: Callable[[List[dict]], int],
                   alive: Callable[[], bool] = lambda: True, progress: Optional[Callable[[dict], None]] = None,
                   chat: Optional[Callable] = None, pacer: Optional[Pacer] = None,
                   sleep: Callable[[float], None] = _sleep, local_chat: Optional[Callable] = None,
                   local: Optional[bool] = None) -> dict:
    """Generate, embed and save questions for passages [{"id": passage row id, "header", "content"}].

    `embed(texts)` returns one vector per text; `save(rows)` stores rows [{"parent_chunk_id", "question", "embedding"}] and
    returns how many it stored (0 when the document is gone); `alive()` is checked before every call so a deleted
    document stops the job; `progress({"status", "questions"})` is told about every step. Questions are saved batch by
    batch, so a job that stops half way (quota, restart) keeps what it has. Returns {"status": "done"|"skipped"|"failed"|
    "cancelled", "questions": n, "reason": str}."""
    pacer = pacer or Pacer(sleep=sleep)
    total = 0
    batches = make_batches(passages)
    report = lambda status: progress and progress({"status": status, "questions": total})  # noqa: E731
    if not batches:
        report("done")
        return {"status": "done", "questions": 0, "reason": "no passage long enough"}
    report("indexing")
    try:
        for batch in batches:
            if not alive():
                return {"status": "cancelled", "questions": total, "reason": "the document was removed"}
            found = generate_batch(batch, chat=chat, pacer=pacer, sleep=sleep, local_chat=local_chat, local=local)
            rows: List[dict] = []
            for number, passage in batch:
                for q in found.get(number, []):
                    rows.append({"parent_chunk_id": passage["id"], "question": q})
            if not rows:
                continue
            vectors: List = []
            for k in range(0, len(rows), EMBED_BATCH):
                vectors += list(embed([r["question"] for r in rows[k:k + EMBED_BATCH]]))
            if len(vectors) != len(rows):
                raise RuntimeError("the embedder returned the wrong number of vectors")
            for r, v in zip(rows, vectors):
                r["embedding"] = [float(x) for x in v]
            stored = save(rows)
            if not stored:
                return {"status": "cancelled", "questions": total, "reason": "the document was removed"}
            total += stored
            report("indexing")
    except Skipped as e:
        log.info("question index skipped: %s", e)
        status = "skipped" if total == 0 else "failed"
        report(status)
        return {"status": status, "questions": total, "reason": str(e)}
    except Exception as e:  # noqa: BLE001 - the document stays usable; what was saved stays
        log.warning("question index stopped after %d questions: %s", total, e)
        report("failed")
        return {"status": "failed", "questions": total, "reason": str(e)[:200]}
    report("done")
    return {"status": "done", "questions": total, "reason": ""}


def index_passages(passages: Sequence[dict], **kw) -> dict:
    """See _index_passages. The local background model, when this job used it, is unloaded at the end unless more jobs are
    queued behind this one (they would only load it again)."""
    try:
        return _index_passages(passages, **kw)
    finally:
        if _pending <= 1:
            background_llm.release()


# ------------------------------------------------------------------------------------------ background worker

_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


_pending = 0     # jobs queued or running on the background thread


def submit(fn: Callable, *args, **kwargs) -> Future:
    """Queue a job on the single background thread (one job at a time, so the pacing is shared by every document)."""
    global _executor, _pending
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="question-index")
        ex = _executor
        _pending += 1

    def run():
        global _pending
        try:
            return fn(*args, **kwargs)
        finally:
            with _executor_lock:
                _pending -= 1
    return ex.submit(run)


def wait_idle(timeout: float = 30.0) -> None:
    """Block until the jobs queued so far are done (tests, scripts)."""
    submit(lambda: None).result(timeout)
