"""Query understanding for a live turn: what the caller most likely said, and what to search the documents for.

Four cheap steps around speech recognition (research: scratchpad rag_voice_indic.md, sections 1, 4 and 6):

1. Vocabulary prompt (ASR_VOCAB_PROMPT). Whisper's `prompt` (Groq) / `initial_prompt` (local Whisper) steers spelling
   toward words it has seen in the prompt. Each agent gets a short prompt built from its name, greeting and the key
   entities of its documents (doctor names, departments, places, fee words). Plain terms in sentence form, no
   instructions, the rarest names last (Whisper reads the end of the prompt most). Built once per set of documents in a
   background thread and cached, so a turn pays a dictionary lookup. With ASR_LANGUAGE_HINT the agent's language is
   passed to Whisper too, but only for a single-language agent (a multilingual agent must keep Hinglish working).
2. Entity snapping (QUERY_ENTITY_SNAP). The same extraction gives a lexicon; near-miss spellings in the transcript
   ("Kulkarany", "pediatrician", "Shivaji nagar") are snapped to the lexicon's spelling with a phonetic key plus a
   RapidFuzz ratio. Only for the search query, never for the text the language model sees. Common words, numbers and
   Devanagari are never changed (Devanagari only through an exact alias).
3. ASR confidence and clarification (ASR_VERBOSE_JSON, ASR_CLARIFY). Groq's `verbose_json` carries avg_logprob,
   no_speech_prob and compression_ratio per segment: a per-utterance level high / medium / low. On a phone call a low
   level on a short utterance, or on one that carries a number or a name, makes the agent ask one short question in the
   caller's language instead of answering ("Sorry, I didn't catch that. Which doctor did you mean?"). Never twice in a
   row.
4. Follow-up condensation (QUERY_CONDENSE). A short follow-up ("aur Monday ko?", "what about her fees?", "और उनकी
   फीस?") is rewritten into a standalone search query from the previous question: rules first (carry the previous
   question's entity and frame over), a tiny Groq call only when the rules cannot (timeout 400 ms, then the raw text).

Everything here is pure Python and fast (a turn adds well under a millisecond, see tests/test_query_understanding.py),
except the optional condensation LLM call, which only runs on gated follow-ups.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import math
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("bonclay.query")

try:  # RapidFuzz is a backend dependency (requirements.txt); a plain ratio is the fallback
    from rapidfuzz import fuzz as _fuzz

    def _ratio(a: str, b: str) -> float:
        return float(_fuzz.ratio(a, b))
except ImportError:  # pragma: no cover
    import difflib

    def _ratio(a: str, b: str) -> float:
        return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------------------------- switches

def _flag(name: str, default: str = "on") -> bool:
    return os.getenv(name, default).strip().lower() not in ("off", "0", "false", "no")


def vocab_prompt_enabled() -> bool:
    return _flag("ASR_VOCAB_PROMPT")


def language_hint_enabled() -> bool:
    return _flag("ASR_LANGUAGE_HINT")


def verbose_json_enabled() -> bool:
    return _flag("ASR_VERBOSE_JSON")


def snap_enabled() -> bool:
    return _flag("QUERY_ENTITY_SNAP")


def condense_mode() -> str:
    """QUERY_CONDENSE: "on" (rules, then the LLM), "rules" (rules only), "off"."""
    v = os.getenv("QUERY_CONDENSE", "on").strip().lower()
    if v in ("off", "0", "false", "no"):
        return "off"
    return "rules" if v == "rules" else "on"


def clarify_enabled(source: Optional[str]) -> bool:
    """ASR_CLARIFY: "phone" (default: phone calls only), "on" (every spoken turn), "off"."""
    v = os.getenv("ASR_CLARIFY", "phone").strip().lower()
    if v in ("off", "0", "false", "no"):
        return False
    if v in ("on", "1", "true", "yes", "all"):
        return True
    return source == "phone"


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------------------------- word lists

# Everyday English words: never a snap target and never snapped. Kept short: the lexicon only holds capitalised names
# and terms of the documents, so this mainly guards words that happen to be capitalised there ("Call", "Payment").
_COMMON_EN = frozenset("""
a about above after again against all also am an and any are as at back be because been before being below between
both but by call called can cannot could day days did do does doing done down during each early either else even
every few first for from further get give go going good got had has have having he her here hers him his how i if in
into is it its just know last late less like make many may me more most much must my near never new next no nor not
now of off on once one only or other our out over own please same see she should so some such take than that the
their them then there these they this those through time to today tomorrow too under until up very was we well were
what when where which while who whom why will with within would yes yet you your yours
about after before open closed close book booking help need want ask tell said says say thank thanks hello hi
morning evening night noon week weeks month months year years hour hours minute minutes number phone name names
please sorry okay ok sir madam
monday tuesday wednesday thursday friday saturday sunday mondays tuesdays wednesdays thursdays fridays saturdays
sundays professor never always important note warning january february march april june july august september
october november december
general family small big main free full other others another sample data guide about things note notes only
patient patients doctor doctors clinic hospital road street office department room reception receptionist
emergency emergencies timing timings fee fees charge charges cost price payment pay paid cash card online
appointment appointments consultation consultations service services test tests report reports visit visits
walk walking available availability nearest nearby student students course courses class classes school college
how where what why who which first second third
""".split())

# Hinglish function words (also in services/hinglish.HINGLISH_MARKERS): "mera" must never become "Meera".
_COMMON_HINGLISH = frozenset("""
hai hain hoon hun ho hoga hogi tha thi raha rahi rahe gaya gayi kiya kiye diya liya hua hui
kya kyun kyon kaise kaisa kaisi kab kahan kaun kitna kitni kitne kaunsa kaunsi konsa
nahi nahin nhi haan han accha acha achha theek thik sahi bilkul zaroor
mujhe mera meri mere hum humein hamara hamari tum tumhara aap aapko aapka aapki aapke
yeh ye woh wo vo yahan wahan uska uski uske unka unki unke iska iski iske
kal aaj abhi jaldi subah shaam sham raat baad pehle phir fir bahut thoda zyada
ko ka ki ke se tak mein me par pe
batao bataiye bolo boliye karo karna karne kijiye chahiye
bhai ji yaar wala wali wale bhi toh to aur lekin agar kuch koi sab sirf bas
milega milegi sakta sakti sakte baje baithte baithti baithta milte milti
""".split())

_DAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow", "tonight",
    "weekend", "weekday", "weekdays", "morning", "afternoon", "evening", "night", "noon",
    "somvar", "mangalvar", "budhvar", "guruvar", "shukravar", "shanivar", "ravivar", "itvaar", "itwar",
    "kal", "aaj", "parso", "parson", "subah", "shaam", "sham", "dopahar", "raat",
    "सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार", "इतवार", "कल", "आज", "परसों", "सुबह",
    "शाम", "दोपहर", "रात",
}

# Follow-up markers at the start of a turn (longest first when matched), per register.
_FOLLOW_MARKERS = [
    "what about", "how about", "and what about", "same for", "also for",
    "and", "also", "then", "so",
    "aur kya", "kya wo", "kya woh", "kya vo", "aur", "toh", "to phir", "phir", "fir", "accha", "acha", "achha",
    "और क्या", "क्या वो", "क्या वह", "और", "फिर", "तो", "अच्छा",
]
# Pronouns that point back to something said before.
_PRONOUNS = frozenset("""
his her hers its their them him it that those this these there
uska uski uske unka unki unke iska iski iske wo woh vo wahan unhe usse unse
उसका उसकी उसके उनका उनकी उनके इसका इसकी इसके वो वह वहाँ वहां उन्हें उससे उनसे
""".split())
_ACKS = frozenset("""
ok okay thanks thank yes no yeah yep nope bye goodbye hello hi hey sure fine great alright right
haan han ha ji nahi nahin theek thik accha acha achha shukriya dhanyavaad dhanyawad namaste
you very much so sir madam hai ok. bas
हाँ हां जी नहीं ठीक अच्छा शुक्रिया धन्यवाद नमस्ते
""".split())

_TITLE_WORDS = {"dr", "dr.", "doctor", "doc", "daktar", "prof", "prof.", "professor", "mr", "mr.", "mrs", "mrs.",
                "ms", "ms.", "shri", "smt", "sri", "डॉ", "डॉ.", "डॉक्टर", "डा.", "प्रो.", "श्री", "श्रीमती"}
_DOCTOR_WORDS = {"dr", "doctor", "doc", "daktar", "doktar", "डॉ", "डॉक्टर", "डा", "physician"}

# Domain words worth spelling right when the documents use them (only added when present).
_FEE_WORDS = ["fees", "consultation", "OPD", "appointment", "admission", "hostel", "scholarship", "EMI",
              "installment", "UPI", "refund", "deposit", "prescription", "semester", "फीस", "शुल्क", "अपॉइंटमेंट"]

_ACRONYM_SKIP = {"AM", "PM", "NOT", "AND", "OR", "THE", "IS", "NO", "OK", "A", "I", "IT", "IN", "ON", "TO", "OF",
                 "AT", "BY", "IF", "AN", "BE", "DO", "MY", "WE", "US", "SO", "UP"}

_DEVA = "ऀ-ॣ॰-ॿ"
_DEVA_RE = re.compile(f"[{_DEVA}]")
_WORD_RE = re.compile(rf"[A-Za-z][A-Za-z'\-]*|[{_DEVA}]+\.?|\d[\d,.:/\-]*")
_CAP = r"[A-Z][a-zA-Z'\-]+"
_TITLE_RE = re.compile(rf"\b(Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Shri|Smt|Sri)\.?[ \t]+({_CAP}(?:[ \t]+{_CAP}){{0,2}})")
_DEVA_TITLE_RE = re.compile(rf"(डॉ\.?|डॉक्टर|डा\.|प्रो\.|श्रीमती|श्री)[ \t]+([{_DEVA}]+)(?:[ \t]+([{_DEVA}]+))?")
_CAP_SEQ_RE = re.compile(rf"\b{_CAP}(?:[ \t]+(?:of|and|&)[ \t]+{_CAP}|[ \t]+{_CAP}){{1,4}}")
_SINGLE_CAP_RE = re.compile(rf"(?<![.!?:\n#\-*•]\s)(?<![.!?:\n#])(?<!^)\b({_CAP})\b", re.MULTILINE)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_ORG_TAIL = {"clinic", "hospital", "institute", "college", "school", "university", "bank", "centre", "center",
             "academy", "office", "ltd", "pvt", "foundation", "trust", "society", "labs", "lab", "pharmacy"}
_PLACE_TAIL = {"road", "nagar", "marg", "street", "lane", "chowk", "colony", "sector", "park", "market", "station",
               "society", "path", "peth", "wadi", "gaon", "pur", "abad"}


def _is_common(word: str) -> bool:
    w = word.lower().strip(".'")
    return w in _COMMON_EN or w in _COMMON_HINGLISH or len(w) < 3


# ---------------------------------------------------------------------------------------------- phonetic key

def phonetic_key(word: str) -> str:
    """A small Indian-English phonetic key: aspirates folded (kh->k, bh->b, sh->s, ph->f), ae/ee/oo folded, then the
    first letter and the consonant skeleton with repeats collapsed. "Kulkarni" / "Kulkarny" / "kul karni" -> "klkrn";
    "Shaikh" / "Sheikh" -> "sk"; "Paediatrician" / "pediatrician" -> "pdtrkn". Roman letters only ("" otherwise)."""
    w = re.sub(r"[^a-z]", "", (word or "").lower())
    if not w:
        return ""
    for a, b in (("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("sh", "s"),
                 ("ch", "C"), ("zh", "j"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "j"),
                 ("ae", "e"), ("ee", "i"), ("oo", "u")):
        w = w.replace(a, b)
    w = w.replace("c", "k").replace("C", "c")
    first = "A" if w[0] in "aeiouy" else w[0]
    rest = [ch for ch in w[1:] if ch not in "aeiouyh"]
    out: List[str] = [first]
    for ch in rest:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------------------------- entities / lexicon

@dataclass
class Entity:
    text: str                 # as written in the documents ("Dr. Meera Kulkarni")
    kind: str                 # person | org | place | term
    count: int = 1


@dataclass
class Lexicon:
    entities: List[Entity] = field(default_factory=list)
    words: Dict[str, str] = field(default_factory=dict)          # lower -> canonical spelling (snap targets)
    by_key: Dict[str, List[str]] = field(default_factory=dict)   # phonetic key -> lower words
    entity_words: set = field(default_factory=set)               # every word of every entity, lower (both scripts)
    person_words: set = field(default_factory=set)               # words of person names, lower
    aliases: Dict[str, str] = field(default_factory=dict)        # exact variant (lower) -> canonical

    def add_alias(self, variant: str, canonical: str) -> None:
        self.aliases[variant.lower()] = canonical


def _sentence_first(text: str, start: int) -> bool:
    """True when position `start` begins a sentence, a line or a list item."""
    before = text[:start].rstrip(" \t")
    return not before or before[-1] in ".!?:\n#-*•|>" or before.endswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.",
                                                                            "8.", "9."))


def extract_entities(texts: Sequence[str]) -> List[Entity]:
    """Deterministic entities of the agent's texts: titled names ("Dr. Meera Kulkarni", "डॉ. मीरा"), capitalised
    multi-word names ("Sunrise Family Clinic", "Lake View Road"), capitalised single words that do not start a
    sentence ("Shivajinagar", "Paediatrician"), acronyms ("OPD", "UPI") and fee/service words found in the text.
    Ordered by kind and frequency; duplicates merged."""
    found: Dict[str, Entity] = {}

    def add(text: str, kind: str) -> None:
        text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
        if not text:
            return
        key = text.lower()
        if key in found:
            found[key].count += 1
            if kind == "person":
                found[key].kind = "person"
        else:
            found[key] = Entity(text, kind)

    for raw in texts:
        text = unicodedata.normalize("NFC", raw or "")
        covered: List[Tuple[int, int]] = []
        for m in _TITLE_RE.finditer(text):
            words = [w for w in m.group(2).split() if not _is_common(w) or w[0].isupper()]
            # stop at the first common word ("Dr. Meera Kulkarni Monday" -> "Dr. Meera Kulkarni")
            name: List[str] = []
            for w in m.group(2).split():
                if w.lower() in _COMMON_EN:
                    break
                name.append(w)
            if name and words:
                title = {"Doctor": "Dr", "Professor": "Prof"}.get(m.group(1), m.group(1))
                add(f"{title}. {' '.join(name)}", "person")
                covered.append(m.span())
        for m in _DEVA_TITLE_RE.finditer(text):
            title = m.group(1) if m.group(1).endswith(".") or m.group(1) in ("डॉक्टर", "श्री", "श्रीमती") else m.group(1) + "."
            parts = [m.group(2)]
            if m.group(3) and m.group(3) not in _DEVA_STOP:
                parts.append(m.group(3))
            add(f"{title} {' '.join(parts)}", "person")
        for m in _CAP_SEQ_RE.finditer(text):
            if any(s <= m.start() < e for s, e in covered):
                continue
            words = m.group(0).split()
            # an all-caps shout ("NEVER", "XXXX") ends the name
            for k, w in enumerate(words):
                if w.isupper() and (len(w) > 5 or w.lower() in _COMMON_EN or len(set(w)) == 1):
                    words = words[:k]
                    break
            # drop a sentence-initial common word ("The Clinic Road" -> "Clinic Road")
            while words and words[0].lower() in _COMMON_EN and words[0].lower() not in _ORG_TAIL:
                words = words[1:]
            if len(words) < 2 or all(_is_common(w) for w in words):
                continue
            tail = words[-1].lower()
            kind = "org" if tail in _ORG_TAIL else "place" if tail in _PLACE_TAIL else "term"
            add(" ".join(words), kind)
        for m in re.finditer(rf"\b{_CAP}\b", text):
            w = m.group(0)
            if _is_common(w) or _sentence_first(text, m.start()):
                continue
            low = w.lower()
            kind = "place" if any(low.endswith(t) for t in ("nagar", "pur", "abad", "gaon", "wadi", "peth")) else "term"
            add(w, kind)
        for m in _ACRONYM_RE.finditer(text):
            if m.group(0) not in _ACRONYM_SKIP and len(set(m.group(0))) > 1 and m.group(0).lower() not in _COMMON_EN:
                add(m.group(0), "term")
        low_text = text.lower()
        for fw in _FEE_WORDS:
            if (fw.lower() in low_text) if fw.isascii() else (fw in text):
                add(fw, "fee")
    order = {"person": 0, "org": 1, "place": 2, "term": 3, "fee": 4}
    return sorted(found.values(), key=lambda e: (order.get(e.kind, 9), -e.count, e.text))


_DEVA_STOP = frozenset("""
है हैं था थे थी का की के को में से पर ने भी और या तो यह ये वह वो क्या कब कैसे कितना कितने कितनी क्यों कोई एक
जी साहब मैडम सर
""".split())


def build_lexicon(entities: Sequence[Entity]) -> Lexicon:
    lex = Lexicon(entities=list(entities))
    for e in entities:
        for w in _WORD_RE.findall(e.text):
            w = w.strip(".'-")
            if not w:
                continue
            low = w.lower()
            lex.entity_words.add(low)
            if e.kind == "person" and low not in _TITLE_WORDS:
                lex.person_words.add(low)
            if (w.isascii() and w.isalpha() and len(w) >= 4 and not _is_common(w) and low not in _TITLE_WORDS
                    and not w.isupper()):
                lex.words.setdefault(low, w)
    for low in lex.words:
        lex.by_key.setdefault(phonetic_key(low), []).append(low)
    return lex


# ---------------------------------------------------------------------------------------------- vocab prompt

PROMPT_TOKEN_LIMIT = 200   # hard cap: Groq / OpenAI read at most 224 prompt tokens; the estimate below over-counts


def prompt_token_budget() -> int:
    """ASR_VOCAB_PROMPT_TOKENS (default 100 estimated, about 60-70 real Whisper tokens; at most 200). The research
    behind this suggests short prompts (20-60 tokens): long name lists hurt more segments than they help."""
    try:
        return max(20, min(PROMPT_TOKEN_LIMIT, int(os.getenv("ASR_VOCAB_PROMPT_TOKENS", "100"))))
    except ValueError:
        return 100


def estimate_tokens(text: str) -> int:
    """A conservative Whisper token count: about 2 ASCII characters per token and 1.5 tokens per other character
    (Devanagari). Over-counts Whisper's own tokenizer on English, Hindi and names (checked in the tests)."""
    ascii_n = sum(1 for c in text if c.isascii())
    return int(math.ceil(ascii_n / 2.0 + (len(text) - ascii_n) * 1.5))


def _first_sentence(text: str, max_words: int = 18) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    m = re.match(r"(.+?[.!?।])(\s|$)", text)
    s = m.group(1) if m else text
    words = s.split()
    return " ".join(words[:max_words])


def build_vocab_prompt(agent: dict, entities: Sequence[Entity], limit: Optional[int] = None) -> str:
    """The Whisper prompt of one agent: its greeting's first sentence, then places, terms and fee words, then the
    person names LAST (Whisper weighs the end of the prompt most). Natural sentences of plain terms, no instructions.
    Items are added in value order (people, then terms, places, organisations, greeting) while the estimate fits."""
    limit = min(PROMPT_TOKEN_LIMIT, limit if limit is not None else prompt_token_budget())
    greeting = _first_sentence(agent.get("greeting") or "")
    name = re.sub(r"\s+", " ", (agent.get("name") or "").strip())
    people = [e.text for e in entities if e.kind == "person"]
    orgs = [e.text for e in entities if e.kind == "org"]
    places = [e.text for e in entities if e.kind == "place"]
    terms = [e.text for e in entities if e.kind == "term"]
    fees = [e.text for e in entities if e.kind == "fee"]

    groups: Dict[str, List[str]] = {"greeting": [], "org": [], "place": [], "term": [], "person": []}
    order_of_value = ([("person", p) for p in people] + [("term", t) for t in terms] + [("place", p) for p in places]
                      + [("org", o) for o in orgs] + [("term", f) for f in fees])
    if name:
        order_of_value.append(("org", name))
    if greeting:
        order_of_value.append(("greeting", greeting))

    def render() -> str:
        parts = []
        if groups["greeting"]:
            parts.append(groups["greeting"][0].rstrip(".!?।") + ".")
        for g in ("org", "place", "term", "person"):
            if groups[g]:
                parts.append(", ".join(groups[g]) + ".")
        return " ".join(parts)

    # a single word that is already part of a longer name ("Kulkarni" of "Dr. Meera Kulkarni") adds nothing
    multi_words = {w.lower().strip(".") for e in entities if " " in e.text for w in e.text.split()}
    seen = set()
    for g, item in order_of_value:
        k = item.lower()
        if k in seen or (" " not in item and g != "greeting" and k in multi_words):
            continue
        if g == "org" and any(k in o.lower() or o.lower() in k for o in groups["org"]):
            continue
        groups[g].append(item)
        if estimate_tokens(render()) > limit:
            groups[g].pop()
            continue
        seen.add(k)
    return render()


def with_format_example(prompt: Optional[str], example: str, limit: Optional[int] = None) -> Optional[str]:
    """`prompt` (the agent's vocabulary) with a short format example after it ("Name: Priya Sharma. Phone: 98765 43210."):
    Whisper reads the end of a prompt most, so the example goes last. The whole stays within the same token limit as the
    vocabulary alone (PROMPT_TOKEN_LIMIT): when it does not fit, the vocabulary loses words from its start, never the
    example. `prompt` unchanged when there is no example."""
    example = (example or "").strip()
    if not example:
        return prompt
    limit = min(PROMPT_TOKEN_LIMIT, limit if limit is not None else PROMPT_TOKEN_LIMIT)
    base = (prompt or "").strip()
    while base and estimate_tokens(f"{base} {example}") > limit:
        words = base.split(" ", 1)
        base = words[1] if len(words) > 1 else ""
    return f"{base} {example}".strip()


def asr_language(agent: Optional[dict]) -> Optional[str]:
    """The `language` to give Whisper: the agent's only language when it has exactly one real one (not Hinglish, which
    Whisper has no code for). None for a multilingual agent: forcing a language there would break Hinglish."""
    if not language_hint_enabled() or agent is None:
        return None
    codes = {str(c).split("-")[0].lower() for c in (agent.get("supported_languages") or ["en"]) if c}
    if len(codes) != 1:
        return None
    code = next(iter(codes))
    return None if code in ("hinglish", "auto", "") else code


# ---------------------------------------------------------------------------------------------- per-agent cache

@dataclass
class AgentProfile:
    fingerprint: str
    lexicon: Lexicon
    prompt: str
    built_ms: float


DocLoader = Callable[[str, str, str], Tuple[Optional[str], List[str]]]
_loader_override: Optional[DocLoader] = None
_profiles: Dict[Tuple[str, str], AgentProfile] = {}
_checked: Dict[Tuple[str, str], Tuple[float, Any]] = {}     # key -> (monotonic time, cheap signature) of the last check
_building: set = set()
_lock = threading.Lock()
CLOUD_CHECK_S = 60.0


def set_document_loader(fn: Optional[DocLoader]) -> None:
    """Replace how an agent's documents are read: fn(agent_id, user_id, storage_mode) -> (fingerprint, texts) (tests)."""
    global _loader_override
    _loader_override = fn
    clear_cache()


def clear_cache() -> None:
    with _lock:
        _profiles.clear()
        _checked.clear()


def _backend():
    import sys  # noqa: PLC0415
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "local_db"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "local_db"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


def _load_documents(agent_id: str, user_id: str, storage_mode: str) -> Tuple[Optional[str], List[str]]:
    """Blocking: (fingerprint, passage texts) of the agent's documents; (None, []) when they cannot be read."""
    if _loader_override is not None:
        return _loader_override(agent_id, user_id, storage_mode)
    try:
        B = _backend()
        if storage_mode == "local" or getattr(B, "supabase", None) is None:
            docs = B.local_db.select("documents", lambda r: r.get("agent_id") == agent_id)
            ids = {d["id"] for d in docs}
            rows = B.local_db.select("document_chunks", lambda r: r.get("document_id") in ids) if ids else []
        else:
            docs = (B.supabase.table("documents").select("id, filename").eq("agent_id", agent_id)
                    .eq("user_id", user_id).execute().data or [])
            ids = [d["id"] for d in docs]
            def read(columns: str) -> List[dict]:
                return (B.supabase.table("document_chunks").select(columns).in_("document_id", ids).limit(5000)
                        .execute().data or []) if ids else []
            try:
                rows = read("chunk_text, header, kind")
            except Exception:  # noqa: BLE001 - the header / kind columns may not be migrated yet
                rows = read("chunk_text")
        parts = sorted(f"{d.get('id')}:{d.get('filename') or ''}" for d in docs)
        fp = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]
        texts = []
        for r in rows:
            t = r.get("chunk_text") or ""
            if r.get("header"):
                t = f"{r['header']}\n{t}"
            if (r.get("kind") or "passage") == "passage":   # generated caller questions are not document text
                texts.append(t)
        return fp, texts
    except Exception as e:  # noqa: BLE001
        log.warning("query understanding: could not read the documents of agent %s: %s", agent_id, e)
        return None, []


def _agent_sig(agent: dict) -> str:
    raw = "\x1f".join(str(agent.get(k) or "") for k in ("name", "greeting", "system_instructions",
                                                         "supported_languages"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def build_profile(agent: dict, texts: Sequence[str], fingerprint: str) -> AgentProfile:
    started = time.perf_counter()
    agent_texts = [agent.get("name") or "", agent.get("greeting") or "", agent.get("system_instructions") or ""]
    entities = extract_entities([*agent_texts, *texts])
    lex = build_lexicon(entities)
    prompt = build_vocab_prompt(agent, entities)
    return AgentProfile(f"{fingerprint}|{_agent_sig(agent)}", lex, prompt, (time.perf_counter() - started) * 1000)


def _cheap_signature(storage_mode: str) -> Any:
    """Local mode: the documents and chunk tables' file stats (changes when a document is added or removed)."""
    if _loader_override is not None or storage_mode != "local":
        return None
    try:
        return _backend().local_db.signature("documents", "document_chunks")
    except Exception:  # noqa: BLE001
        return None


def _refresh(agent: dict, user_id: str, storage_mode: str, key: Tuple[str, str]) -> None:
    try:
        fp, texts = _load_documents(str(agent.get("id")), user_id, storage_mode)
        full = f"{fp}|{_agent_sig(agent)}"
        with _lock:
            current = _profiles.get(key)
        if current is not None and current.fingerprint == full:
            return
        profile = build_profile(agent, texts, fp or "none")
        with _lock:
            _profiles[key] = profile
            while len(_profiles) > 256:
                _profiles.pop(next(iter(_profiles)))
        log.info("query understanding: agent %s lexicon %d words, prompt %d chars (~%d tokens), built in %.1f ms",
                 agent.get("id"), len(profile.lexicon.words), len(profile.prompt), estimate_tokens(profile.prompt),
                 profile.built_ms)
    except Exception as e:  # noqa: BLE001 - never fails a turn
        log.warning("query understanding: profile build failed for agent %s: %r", agent.get("id"), e)
    finally:
        with _lock:
            _building.discard(key)


def get_profile(agent: Optional[dict], user_id: str, storage_mode: str, *, wait: bool = False) -> Optional[AgentProfile]:
    """The agent's cached profile (vocab prompt + lexicon), starting a background rebuild when its documents may have
    changed. Never blocks unless `wait` (tests, warm-up): a turn uses what is ready, the next turn gets the new one."""
    if not agent or not agent.get("id"):
        return None
    key = (str(agent["id"]), storage_mode or "local")
    now = time.monotonic()
    sig = _cheap_signature(storage_mode)
    with _lock:
        profile = _profiles.get(key)
        last = _checked.get(key)
        agent_changed = profile is not None and not profile.fingerprint.endswith("|" + _agent_sig(agent))
        stale = (profile is None or agent_changed or last is None
                 or (sig is not None and last[1] != sig)
                 or (sig is None and now - last[0] >= CLOUD_CHECK_S))
        start = stale and key not in _building
        if start:
            _building.add(key)
            _checked[key] = (now, sig)
    if start:
        th = threading.Thread(target=_refresh, args=(dict(agent), user_id, storage_mode, key), daemon=True,
                              name="query-profile")
        th.start()
        if wait:
            th.join(timeout=10)
            with _lock:
                profile = _profiles.get(key)
    return profile


# ---------------------------------------------------------------------------------------------- ASR plumbing

# Set by the turn around its speech-to-text call: {"prompt": str|None, "verbose": bool}; the ASR helper fills
# "segments" (verbose_json) and "prompt_used". A context variable, so the helper's signature (and every test fake of
# it) stays as it is, and non-turn callers (consent, WhatsApp voice notes) see None.
ASR_CTX: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("bonclay_asr_ctx", default=None)


def asr_hints() -> Optional[dict]:
    return ASR_CTX.get()


def turn_asr_hints(agent: Optional[dict], user_id: str, storage_mode: str) -> dict:
    """What the turn's speech recognition is given: the agent's vocab prompt (or None) and whether to ask Groq for
    verbose_json. Also starts the agent's profile build when needed (the snap after ASR then usually finds it)."""
    profile = get_profile(agent, user_id, storage_mode)
    prompt = profile.prompt if (profile is not None and vocab_prompt_enabled() and profile.prompt) else None
    return {"prompt": prompt, "verbose": verbose_json_enabled()}


def segments_of(result: Any) -> Optional[List[dict]]:
    """The verbose_json segments of a Groq / OpenAI transcription result (pydantic extras or a dict), or None."""
    segs = getattr(result, "segments", None)
    if segs is None and isinstance(result, dict):
        segs = result.get("segments")
    if segs is None:
        extra = getattr(result, "model_extra", None) or {}
        segs = extra.get("segments")
    if not segs:
        return None
    out = []
    for s in segs:
        d = s if isinstance(s, dict) else (s.model_dump() if hasattr(s, "model_dump") else vars(s))
        out.append(d)
    return out


# ---------------------------------------------------------------------------------------------- confidence

def asr_confidence(segments: Optional[Sequence[dict]], text: str = "") -> dict:
    """Per-utterance confidence from verbose_json segments: duration-weighted avg_logprob, the worst no_speech_prob and
    compression_ratio. Level "low" when avg_logprob < ASR_LOW_LOGPROB (default -0.9), or a likely-silence segment with a
    weak logprob, or a repetitive one (compression > 2.4); "high" when avg_logprob >= ASR_HIGH_LOGPROB (default -0.4)
    and no_speech_prob < 0.3; else "medium". "unknown" without segments (local Whisper, typed text)."""
    if not segments:
        return {"level": "unknown"}
    low_lp = _float_env("ASR_LOW_LOGPROB", -0.9)
    high_lp = _float_env("ASR_HIGH_LOGPROB", -0.4)
    total_w = 0.0
    acc = 0.0
    min_lp = 0.0
    no_speech = 0.0
    compression = 0.0
    weak: List[str] = []
    for s in segments:
        lp = float(s.get("avg_logprob") or 0.0)
        dur = float(s.get("end") or 0.0) - float(s.get("start") or 0.0)
        w = dur if dur > 0 else max(1.0, len(str(s.get("text") or "")) / 10.0)
        total_w += w
        acc += lp * w
        min_lp = min(min_lp, lp)
        no_speech = max(no_speech, float(s.get("no_speech_prob") or 0.0))
        compression = max(compression, float(s.get("compression_ratio") or 0.0))
        if lp < low_lp:
            weak.append(str(s.get("text") or "").strip())
    avg = acc / total_w if total_w else 0.0
    if avg < low_lp or (no_speech > 0.6 and avg < -0.5) or compression > 2.4:
        level = "low"
    elif avg >= high_lp and no_speech < 0.3:
        level = "high"
    else:
        level = "medium"
    return {"level": level, "avg_logprob": round(avg, 3), "min_logprob": round(min_lp, 3),
            "no_speech_prob": round(no_speech, 3), "compression_ratio": round(compression, 3),
            "segments": len(segments), "weak_segments": weak}


# ---------------------------------------------------------------------------------------------- clarification

CLARIFY_TEXTS: Dict[str, Dict[str, str]] = {
    "en": {
        "generic": "Sorry, I didn't catch that. Could you say it again?",
        "doctor": "Sorry, I didn't catch that. Which doctor did you mean?",
        "number": "Sorry, I didn't catch the number. Could you say it again slowly?",
        "name": "Sorry, I didn't catch the name. Could you say it again?",
    },
    "hi": {
        "generic": "माफ़ कीजिए, मैं ठीक से सुन नहीं पाई। क्या आप दोबारा बता सकते हैं?",
        "doctor": "माफ़ कीजिए, मैं ठीक से सुन नहीं पाई। आप किस डॉक्टर के बारे में पूछ रहे हैं?",
        "number": "माफ़ कीजिए, नंबर ठीक से सुनाई नहीं दिया। कृपया धीरे-धीरे दोबारा बताइए।",
        "name": "माफ़ कीजिए, नाम ठीक से सुनाई नहीं दिया। कृपया दोबारा बताइए।",
    },
    "hinglish": {
        "generic": "Sorry, main theek se sun nahi paayi. Kya aap dobara bata sakte hain?",
        "doctor": "Sorry, main theek se sun nahi paayi. Aap kis doctor ke baare mein pooch rahe hain?",
        "number": "Sorry, number theek se sunai nahi diya. Please dheere se dobara bataiye.",
        "name": "Sorry, naam theek se sunai nahi diya. Please dobara bataiye.",
    },
}


def clarification_text(kind: str, language: str) -> str:
    lang = (language or "en").split("-")[0].lower()
    table = CLARIFY_TEXTS.get(lang) or (CLARIFY_TEXTS["hi"] if lang == "mr" else CLARIFY_TEXTS["en"])
    return table.get(kind) or table["generic"]


def is_clarification(text: Optional[str]) -> bool:
    """True when an agent reply is (or ends with, after the AI disclosure) one of the clarification questions."""
    t = (text or "").strip()
    return bool(t) and any(q in t for table in CLARIFY_TEXTS.values() for q in table.values())


_NUMBER_RE = re.compile(r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|lakh)\b|"
                        r"\b(?:ek|do|teen|char|paanch|chhe|saat|aath|nau|das|sau|hazaar)\b|"
                        r"एक|दो|तीन|चार|पांच|पाँच|छह|सात|आठ|नौ|दस|सौ|हज़ार|हजार", re.IGNORECASE)


def clarify_kind(text: str, lexicon: Optional[Lexicon], snap_changes: Sequence[dict] = ()) -> Optional[str]:
    """What a doubtful utterance hinges on: "doctor", "name", "number", or None when it names nothing in particular."""
    words = [w.lower().strip(".") for w in _WORD_RE.findall(text or "")]
    if any(w in _DOCTOR_WORDS for w in words):
        return "doctor"
    if lexicon is not None:
        if any(w in lexicon.person_words for w in words) or any(
                c.get("to", "").lower() in lexicon.person_words for c in snap_changes):
            return "name"
    if snap_changes:
        return "name"
    if _NUMBER_RE.search(text or ""):
        return "number"
    return None


def should_clarify(confidence: Optional[dict], text: str, lexicon: Optional[Lexicon] = None,
                   snap_changes: Sequence[dict] = (), *, source: Optional[str] = None,
                   last_agent_text: Optional[str] = None) -> Optional[str]:
    """The kind of clarification to ask ("generic", "doctor", "number", "name"), or None to answer normally.

    Asks only when ASR confidence is low AND the utterance is short (<= 4 words) or a weak segment carries a number,
    a name or a doctor; never twice in a row (`last_agent_text` is the agent's previous reply); ASR_CLARIFY decides
    the channels (phone only by default)."""
    if not clarify_enabled(source) or not confidence or confidence.get("level") != "low":
        return None
    if is_clarification(last_agent_text):
        return None
    words = _WORD_RE.findall(text or "")
    kind = clarify_kind(text, lexicon, snap_changes)
    weak_text = " ".join(confidence.get("weak_segments") or []) or text
    weak_kind = clarify_kind(weak_text, lexicon, snap_changes)
    if len(words) <= 4:
        return kind or "generic"
    if weak_kind is not None:
        return weak_kind
    return None


# ---------------------------------------------------------------------------------------------- entity snapping

def _snap_candidate(token: str, lex: Lexicon, *, titled: bool) -> Optional[Tuple[str, float]]:
    """(canonical word, score) for one Roman token, or None. Same phonetic key and ratio >= 70 (>= 80 when the key is
    short), or a different key with ratio >= 88 on a long word. Ambiguous ties are not snapped."""
    low = token.lower()
    if low in lex.words or not token.isalpha() or len(low) < 4:
        return None
    if _is_common(low) and not titled:
        return None
    key = phonetic_key(low)
    scored: List[Tuple[float, str]] = []
    for cand in lex.by_key.get(key, []):
        r = _ratio(low, cand)
        if r >= (60 if titled else 80 if len(key) <= 2 else 70):   # "Dr. Mira" -> "Dr. Meera"
            scored.append((r + 5, cand))   # the phonetic agreement is worth a little
    if len(low) >= 6:
        for cand in lex.words:
            if phonetic_key(cand) == key:
                continue
            r = _ratio(low, cand)
            if r >= 88 and _ratio(key, phonetic_key(cand)) >= 75:
                scored.append((r, cand))
    if not scored:
        return None
    scored.sort(reverse=True)
    best, cand = scored[0]
    if len(scored) > 1 and scored[1][1] != cand and best - scored[1][0] < 3:
        return None
    return lex.words[cand], best


def snap_entities(text: str, lexicon: Optional[Lexicon]) -> Tuple[str, List[dict]]:
    """Fix near-miss spellings of the agent's names and terms in a transcript: ("dr meera kulkarany ki fees",
    lexicon) -> ("dr meera Kulkarni ki fees", [{"from": "kulkarany", "to": "Kulkarni", "score": ...}]).

    Roman words only (>= 4 letters, not a common English or Hinglish word unless right after a title like "Dr");
    two adjacent words may join ("kul karni" -> "Kulkarni"). Numbers are never touched; Devanagari only through an
    exact alias in the lexicon. Returns the new text and the list of changes."""
    if not text or lexicon is None or (not lexicon.words and not lexicon.aliases):
        return text, []
    tokens = list(re.finditer(rf"[A-Za-z]+|[{_DEVA}]+|\d[\d,.:]*|\S", text))
    out: List[str] = []
    changes: List[dict] = []
    pos = 0
    i = 0
    prev_word = ""
    while i < len(tokens):
        m = tokens[i]
        tok = m.group(0)
        replacement: Optional[str] = None
        consumed = 1
        is_deva = bool(_DEVA_RE.match(tok))
        if is_deva or (tok.isalpha() and tok.isascii()):
            alias = lexicon.aliases.get(tok.lower())
            if alias and alias != tok:
                replacement, score = alias, 100.0
            elif not is_deva:
                titled = prev_word.lower().rstrip(".") in _TITLE_WORDS
                nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                if nxt is not None:   # two words that make one lexicon word ("kul karni", "shivaji nagar")
                    gap = text[m.end():nxt.start()]
                    nt = nxt.group(0)
                    if (nt.isalpha() and nt.isascii() and gap == " " and len(tok + nt) >= 6
                            and (titled or not (_is_common(tok) and _is_common(nt)))):
                        joined = (tok + nt).lower()
                        hit = (lexicon.words[joined], 100.0) if joined in lexicon.words else \
                            _snap_candidate(tok + nt, lexicon, titled=True)
                        if hit:
                            replacement, score, consumed = hit[0], hit[1], 2
                if replacement is None:
                    hit = _snap_candidate(tok, lexicon, titled=titled)
                    if hit and hit[0].lower() != tok.lower():
                        replacement, score = hit
        end = tokens[i + consumed - 1].end()
        out.append(text[pos:m.start()])
        if replacement is not None:
            changes.append({"from": text[m.start():end], "to": replacement, "score": round(score, 1)})
            out.append(replacement)
        else:
            out.append(text[m.start():end])
        pos = end
        if tok.isalpha() or is_deva:
            prev_word = tok
        i += consumed
    out.append(text[pos:])
    return "".join(out), changes


# ---------------------------------------------------------------------------------------------- condensation

def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def _strip_markers(text: str) -> Tuple[str, bool]:
    """The turn without its leading follow-up markers ("aur Monday ko?" -> "Monday ko?"), and whether it had any."""
    t = (text or "").strip()
    had = False
    changed = True
    while changed and t:
        changed = False
        low = t.lower()
        for mk in sorted(_FOLLOW_MARKERS, key=len, reverse=True):
            if low.startswith(mk) and (len(low) == len(mk) or not low[len(mk)].isalnum()):
                t = t[len(mk):].lstrip(" ,.-…")
                had = changed = True
                break
    return t, had


_STOP = frozenset("""
a an the of in on at to for and or is are was were be it this that what when where which who how why do does did can
could will would should please tell me my i you your about is there any much many
hai hain ka ki ke ko mein me se par kya kab kaise kitna kitni kitne kaun konsa kaunsa bhi to toh ye yeh wo woh
hota hoti hote milta milti milte baithte baithti baithta lagta lagti sakta sakti sakte ji
है हैं का की के को में से पर क्या कब कैसे कितना कितनी कितने कौन भी तो ये वो होता होती होते मिलता मिलती
बैठते बैठती बैठता लगता लगती सकता सकती सकते जी
""".split())


def _content_words(text: str) -> List[str]:
    out = []
    for w in _words(text):
        low = w.lower().strip(".'?")
        if low and low not in _STOP and low not in _PRONOUNS and low not in _ACKS:
            out.append(low)
    return out


def _has_slot(words: Sequence[str]) -> bool:
    return any(w.lower().strip(".?") in _DAYS or any(c.isdigit() for c in w) for w in words)


def find_entity_spans(text: str, lexicon: Optional[Lexicon]) -> List[Tuple[int, int, str]]:
    """(start, end, text) of named things in `text`: a title plus the following name words ("Dr. Meera", "doctor
    Rohan Deshmukh", "डॉ. मीरा"), and runs of the lexicon's person/place/org words."""
    spans: List[Tuple[int, int, str]] = []
    toks = list(re.finditer(rf"[A-Za-z][A-Za-z'\-]*\.?|[{_DEVA}]+\.?", text or ""))
    ent_words = lexicon.entity_words if lexicon is not None else set()
    name_words = (lexicon.person_words | {w for w in lexicon.words} | {w for w in ent_words if _DEVA_RE.search(w)}
                  if lexicon is not None else set())
    i = 0
    while i < len(toks):
        w = toks[i].group(0)
        low = w.lower().rstrip(".")
        if low in _TITLE_WORDS or w.lower() in _TITLE_WORDS:
            j = i + 1
            while j < len(toks) and j - i <= 3:
                nw = toks[j].group(0).rstrip(".")
                nlow = nw.lower()
                is_name = (nlow in name_words or (nw[:1].isupper() and nlow not in _COMMON_EN)
                           or (_DEVA_RE.match(nw) and j == i + 1 and nw not in _DEVA_STOP and nlow not in _STOP))
                if not is_name or nlow in _STOP or nlow in _COMMON_HINGLISH:
                    break
                j += 1
            if j > i + 1:
                spans.append((toks[i].start(), toks[j - 1].end(), text[toks[i].start():toks[j - 1].end()].rstrip(".")))
                i = j
                continue
        elif lexicon is not None and low in (lexicon.person_words | lexicon.words.keys()) and not _is_common(low):
            j = i + 1
            while j < len(toks) and toks[j].group(0).lower().rstrip(".") in (lexicon.person_words | lexicon.words.keys()):
                j += 1
            spans.append((toks[i].start(), toks[j - 1].end(), text[toks[i].start():toks[j - 1].end()].rstrip(".")))
            i = j
            continue
        i += 1
    return spans


def _is_sov(text: str) -> bool:
    """Hindi word order (the slot goes before the verb): Devanagari, or Roman text with Hindi function words."""
    if _DEVA_RE.search(text or ""):
        return True
    low = {w.lower().strip("?.") for w in _words(text)}
    return len(low & {"hai", "hain", "kya", "kab", "ko", "ka", "ki", "ke", "kitna", "kitni", "kaise", "milte",
                      "baithte", "baithti", "hota", "hoti", "mein", "kahan", "kaun"}) >= 1


def is_follow_up(text: str, history: Sequence[dict], lexicon: Optional[Lexicon] = None) -> bool:
    """A turn that only makes sense with the previous one: there is an earlier caller turn, and the turn starts with a
    follow-up marker ("and", "what about", "aur", "फिर", "kya wo"), points back with a pronoun ("uska", "her"), or is
    short (<= 6 words) with fewer than two content words of its own and no name. Acknowledgements are not follow-ups."""
    if not text or not any(h.get("role") == "user" for h in history or []):
        return False
    words = _words(text)
    if not words:
        return False
    if all(w.lower().strip(".?!") in _ACKS for w in words):
        return False
    rest, had_marker = _strip_markers(text)
    if had_marker:
        return True
    if any(w.lower().strip("?.") in _PRONOUNS for w in words) and len(words) <= 8:
        return True
    if len(words) > 6:
        return False
    if find_entity_spans(text, lexicon):
        return False
    return len(_content_words(text)) < 2


def _last(history: Sequence[dict], role: str) -> Optional[str]:
    return next((h.get("text_content") for h in reversed(history or []) if h.get("role") == role), None)


def condense_rules(text: str, history: Sequence[dict], lexicon: Optional[Lexicon] = None) -> Optional[str]:
    """The standalone search query of a follow-up by rules, or None when the rules cannot tell.

    - a new name in place of the old ("aur Dr. Rohan?") -> the previous question with the name swapped;
    - a day/time/number slot ("aur Monday ko?") -> the previous question with the slot (before the verb in Hindi and
      Hinglish: "Dr. Meera Monday ko kab baithti hain"; at the end in English);
    - an attribute ("uski fees?", "what about her fees?") -> the entity last talked about + the attribute."""
    prev_q = _last(history, "user")
    if not prev_q:
        return None
    prev_a = _last(history, "agent") or ""
    rest, _had = _strip_markers(text)
    rest = rest.strip().rstrip("?.!। ").strip()
    rest_words = _words(rest)
    prev_clean = prev_q.strip().rstrip("?.!। ").strip()
    prev_spans = find_entity_spans(prev_clean, lexicon)
    cur_spans = find_entity_spans(rest, lexicon)

    # 1. a new name replaces the previous one
    if cur_spans:
        leftover = rest
        for s, e, _t in reversed(cur_spans):
            leftover = leftover[:s] + leftover[e:]
        if _content_words(leftover) and not _has_slot(_words(leftover)):
            return None if not _had else rest          # "aur Dr. Rohan ki fees" stands on its own
        if prev_spans:
            s, e, _t = prev_spans[0]
            swapped = f"{prev_clean[:s]}{cur_spans[0][2]}{prev_clean[e:]}"
            if _has_slot(_words(leftover)):
                return _insert_slot(swapped, cur_spans[0][2], leftover.strip())
            return re.sub(r"\s+", " ", swapped).strip()
        return None

    # pronoun words are dropped from the slot ("uski fees" -> "fees")
    slot_words = [w for w in rest_words if w.lower().strip("?.") not in _PRONOUNS]
    slot = " ".join(slot_words).strip()
    entity = prev_spans[0][2] if prev_spans else None
    if entity is None:
        # the one person the agent's last reply named ("Dr. Rohan Deshmukh hamare Paediatrician hain")
        people = [t for _s, _e, t in find_entity_spans(prev_a, lexicon)
                  if t.split()[0].lower().rstrip(".") in _TITLE_WORDS
                  or (lexicon is not None and t.split()[0].lower() in lexicon.person_words)]
        if len({p.lower() for p in people}) == 1:
            entity = people[0]
    if not slot:
        return None
    # 2. a day / time / number slot: the previous question with the slot added ("aur Monday ko?"); a longer turn
    #    ("kya wo Sunday ko bhi baithti hain?") already has its own frame and only needs the person it points to
    if _has_slot(slot_words) and len(rest_words) >= 3 and entity:
        return f"{entity} {slot}"
    if _has_slot(slot_words):
        return _insert_slot(prev_clean, prev_spans[0][2] if prev_spans else None, slot)
    # 3. an attribute of the entity talked about
    if entity:
        return f"{entity} {slot}"
    points_back = any(w.lower().strip("?.") in _PRONOUNS for w in rest_words)
    if points_back:
        return None                                    # "uska number?" with nobody to refer to: the LLM decides
    return None


def _insert_slot(frame: str, entity: Optional[str], slot: str) -> str:
    frame = frame.strip()
    if _is_sov(frame) or _is_sov(slot):
        if entity and entity in frame:
            i = frame.index(entity) + len(entity)
            out = f"{frame[:i]} {slot} {frame[i:]}"
        else:
            out = f"{slot} {frame}"
    else:
        out = f"{frame} {slot}"
    return re.sub(r"\s+", " ", out).strip()


CONDENSE_SYSTEM = ("Rewrite the caller's last message as one short standalone search query, using the previous "
                   "question and answer to fill in what it refers to. Keep the caller's language and script. "
                   "Reply with the query only.")


def _llm_condense(text: str, history: Sequence[dict], timeout: float) -> Optional[str]:
    """Blocking: a tiny Groq call; None on any failure."""
    try:
        from services import llm  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from backend.services import llm  # noqa: PLC0415
    prev_q = _last(history, "user") or ""
    prev_a = _last(history, "agent") or ""
    msgs = [{"role": "system", "content": CONDENSE_SYSTEM},
            {"role": "user", "content": f"Previous question: {prev_q[:300]}\nAnswer: {prev_a[:300]}\n"
                                        f"Last message: {text[:200]}\nStandalone query:"}]
    res = llm.chat(msgs, temperature=0.0, max_tokens=40, timeout=timeout, providers=["groq"])
    out = (res.text or "").strip().splitlines()[0].strip().strip('"\'“”') if (res.text or "").strip() else ""
    out = re.sub(r"^(standalone query|query)\s*:\s*", "", out, flags=re.IGNORECASE).strip()
    if not out or len(out) > 240:
        return None
    return out


async def condense(text: str, history: Sequence[dict], lexicon: Optional[Lexicon] = None, *,
                   llm_call: Optional[Callable[[str, Sequence[dict], float], Optional[str]]] = None,
                   timeout_s: Optional[float] = None) -> Tuple[str, dict]:
    """(search query, info) for one turn. info = {"path": "none"|"rules"|"llm"|"llm_timeout"|"llm_failed", "ms"}.
    Non-follow-ups come back unchanged at once."""
    started = time.perf_counter()
    info: dict = {"path": "none"}
    mode = condense_mode()
    if mode == "off" or not is_follow_up(text, history, lexicon):
        info["ms"] = round((time.perf_counter() - started) * 1000, 2)
        return text, info
    q = condense_rules(text, history, lexicon)
    if q:
        info.update(path="rules", ms=round((time.perf_counter() - started) * 1000, 2))
        return q, info
    rest, had_marker = _strip_markers(text)
    points_back = had_marker or any(w.lower().strip("?.") in _PRONOUNS for w in _words(text))
    if mode != "on" or not points_back and _content_words(rest):
        info.update(path="none", ms=round((time.perf_counter() - started) * 1000, 2))
        return text, info
    timeout = timeout_s if timeout_s is not None else _float_env("QUERY_CONDENSE_TIMEOUT_MS", 400) / 1000.0
    call = llm_call or _llm_condense
    try:
        out = await asyncio.wait_for(asyncio.to_thread(call, text, history, timeout), timeout=timeout)
        if out:
            info.update(path="llm", ms=round((time.perf_counter() - started) * 1000, 1))
            return out, info
        info["path"] = "llm_failed"
    except asyncio.TimeoutError:
        info["path"] = "llm_timeout"
    except Exception as e:  # noqa: BLE001 - LLMUnavailable, HTTP errors: the raw text is the query
        info.update(path="llm_failed", error=type(e).__name__)
    info["ms"] = round((time.perf_counter() - started) * 1000, 1)
    return text, info


async def search_query(user_text: str, history: Sequence[dict], lexicon: Optional[Lexicon], *,
                       answering_agent: bool = False, llm_call=None) -> Tuple[str, dict]:
    """The text to search the documents with: the transcript with entities snapped, then condensed when it is a
    follow-up. `answering_agent` (the agent's last reply asked for a detail): the caller's words are an answer, not a
    question, so nothing is condensed. Returns (query, info with "snap" changes and "condense" path/ms)."""
    started = time.perf_counter()
    info: dict = {"snap": [], "condense": {"path": "none"}}
    q = user_text or ""
    if snap_enabled() and lexicon is not None:
        q, info["snap"] = snap_entities(q, lexicon)
    if not answering_agent:
        q, info["condense"] = await condense(q, history, lexicon, llm_call=llm_call)
    info["ms"] = round((time.perf_counter() - started) * 1000, 2)
    return q, info


# ---------------------------------------------------------------------------------------------- turn hooks

_CLARIFIED: "Dict[str, float]" = {}   # conversation id -> when its last turn was a clarification (history unknown)


def clarification_answer(agent_id: str, kind: str, language: str, last_agent_text: Optional[str],
                         conversation_id: Optional[str]) -> Any:
    """The clarification as an instant answer (services/instant_answers.InstantAnswer, id "asr_clarify"), so both turn
    paths speak it like an approved answer: no retrieval, no language model, its audio cached per voice. None when the
    previous agent turn was already a clarification (never twice in a row)."""
    if is_clarification(last_agent_text) or (conversation_id and conversation_id in _CLARIFIED):
        return None
    try:
        from services import instant_answers as ia  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from backend.services import instant_answers as ia  # noqa: PLC0415
    text = clarification_text(kind or "generic", language)
    if conversation_id:
        _CLARIFIED[conversation_id] = time.time()
        while len(_CLARIFIED) > 1024:
            _CLARIFIED.pop(next(iter(_CLARIFIED)))
    key = hashlib.sha1(f"asr_clarify|{language}|{text}".encode("utf-8")).hexdigest()[:16]
    return ia.InstantAnswer(id="asr_clarify", agent_id=agent_id, text=text, language=language, question="",
                            matched_text="", score=0.0, cache_key=f"asr_clarify:{key}")


def note_answered(conversation_id: Optional[str]) -> None:
    """A turn of this conversation was answered normally: a later doubtful turn may ask again."""
    if conversation_id:
        _CLARIFIED.pop(conversation_id, None)


async def turn_search_query(agent: Optional[dict], user_id: str, storage_mode: str, user_text: str,
                            history: Sequence[dict], *, answering_agent: bool = False) -> Tuple[str, dict]:
    """`search_query` with the agent's cached lexicon (none yet: no snapping, rules without names). Never raises."""
    try:
        profile = get_profile(agent, user_id, storage_mode)
        lexicon = profile.lexicon if profile is not None else None
        q, info = await search_query(user_text, history, lexicon, answering_agent=answering_agent)
        if q != user_text:
            log.info("search query %r -> %r (snap %s, condense %s, %.1f ms)", user_text, q,
                     [f"{c['from']}->{c['to']}" for c in info.get("snap") or []],
                     (info.get("condense") or {}).get("path"), info.get("ms", 0.0))
        return (q or user_text), info
    except Exception as e:  # noqa: BLE001
        log.warning("search query step failed, searching the transcript: %r", e)
        return user_text, {"error": type(e).__name__}
