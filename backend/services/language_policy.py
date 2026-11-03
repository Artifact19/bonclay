"""Which language the agent answers in, and a check that it did.

The policy (one rule per line; every one of them is covered by tests/test_language_policy.py):

1. Mirror the caller. The reply is in the language AND script of the caller's latest turn: Devanagari gets Hindi
   (Marathi when the agent only supports Marathi), another Indian script gets that language, Hindi typed in Roman
   letters ("Hinglish") gets Hinglish, anything else Latin gets English.
2. Ultra-short turns ("ok", "9876543210") carry no language: the agent keeps the language of its previous reply.
3. An explicit request ("Tamil mein bolo", "please speak English", "हिंदी में बताइए") for a language the agent supports
   (agent.supported_languages) switches the reply and is remembered for the rest of the conversation: the choice is
   stored on the conversation and beats rule 1 until the caller asks for another language. A request for "Hindi"
   keeps the caller's script: Hinglish when they type Roman Hindi, Devanagari otherwise.
4. A request for a language the agent does not support is not followed: the agent says so in one short sentence, in
   the language it was already using, and offers the languages it supports. Nothing is stored.
5. After the model answers, `reply_matches` checks the script and language of the reply against the target. A mismatch
   triggers ONE regeneration with a stronger instruction; if that is still wrong the first answer is kept.

Pure functions only (no model, no I/O) so the whole policy is unit-testable."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    from services import hinglish
except ImportError:  # pragma: no cover - package-style imports
    from backend.services import hinglish

# ------------------------------------------------------------------------------------------------ scripts

# First and last code point of each script block -> language it implies. Devanagari defaults to Hindi.
INDIC_SCRIPTS: List[Tuple[int, int, str]] = [
    (0x0900, 0x097F, "hi"), (0x0980, 0x09FF, "bn"), (0x0A00, 0x0A7F, "pa"), (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"), (0x0B80, 0x0BFF, "ta"), (0x0C00, 0x0C7F, "te"), (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml")]
# Scripts of languages outside India. They only count when the agent supports that language, so a stray character
# never moves an English agent into a language its voices cannot speak.
OTHER_SCRIPTS: List[Tuple[int, int, str]] = [
    (0x0600, 0x06FF, "ur"), (0x0400, 0x04FF, "ru"), (0xAC00, 0xD7AF, "ko"), (0x3040, 0x30FF, "ja"),
    (0x4E00, 0x9FFF, "zh")]
INDIC_CODES = {code for _, _, code in INDIC_SCRIPTS}
# Languages written in Latin letters that the reply check cannot tell apart from each other.
LATIN_CODES = {"en", "es", "fr", "de", "it", "pt", "hinglish"}

LANGUAGE_DISPLAY: Dict[str, str] = {
    "en": "English", "hi": "Hindi", "hinglish": "Hinglish", "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "or": "Odia", "ur": "Urdu",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese", "ja": "Japanese",
    "zh": "Chinese", "ko": "Korean", "ru": "Russian", "ar": "Arabic", "sa": "Sanskrit", "ne": "Nepali",
    "as": "Assamese"}

WORD = re.compile(r"[\w\u0900-\u0963\u0966-\u0DFF\u200c\u200d]+", re.UNICODE)  # no danda: punctuation


def base_code(code: str) -> str:
    return (code or "").split("-")[0].split("_")[0].lower()


def display_name(code: str) -> str:
    return LANGUAGE_DISPLAY.get(base_code(code), code)


def _script_of(ch: str) -> Optional[str]:
    cp = ord(ch)
    for lo, hi, code in INDIC_SCRIPTS:
        if lo <= cp <= hi:
            return code
    for lo, hi, code in OTHER_SCRIPTS:
        if lo <= cp <= hi:
            return code
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isalpha() and ord(ch) < 0x250:
        return "latin"  # accented Latin letters (Spanish, French, ...)
    return None


def letter_counts(text: str) -> Dict[str, int]:
    """Letters per script: Indian scripts and the other tracked ones by language code, Latin as "latin"."""
    counts: Dict[str, int] = {}
    for ch in text or "":
        s = _script_of(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    return counts


def _letter_words(text: str) -> List[str]:
    return [w for w in WORD.findall(text or "") if any(c.isalpha() for c in w)]


# ------------------------------------------------------------------------------------------------ detection

def supported_codes(supported: Optional[Sequence[str]]) -> List[str]:
    codes = [base_code(c) for c in (supported or ["en"]) if base_code(c)]
    return codes or ["en"]


def detect_language(user_text: str, supported: Optional[Sequence[str]] = None) -> str:
    """The language of one caller turn: "hi" (Devanagari), another Indian script's code, "hinglish" (Roman Hindi), "en".
    A turn with no letters falls back to the agent's first language."""
    codes = supported_codes(supported)
    counts = letter_counts(user_text)
    indic = {c: n for c, n in counts.items() if c in INDIC_CODES}
    if indic:
        lang = max(indic, key=indic.get)
        if lang == "hi" and "mr" in codes and "hi" not in codes:
            lang = "mr"
        return lang
    other = {c: n for c, n in counts.items() if c not in INDIC_CODES and c != "latin" and c in codes}
    if other:
        if "ja" in other and "zh" in other:
            return "ja"  # kanji with kana is Japanese
        return max(other, key=other.get)
    if hinglish.looks_hinglish(user_text):
        return "hinglish"
    if (user_text or "").strip() and "en" in codes:
        return "en"
    return codes[0]


# ------------------------------------------------------------------------------------------------ explicit requests

_ALIASES: Dict[str, List[str]] = {
    "en": ["english", "angrezi", "angreji", "angrejee", "inglish", "अंग्रेज़ी", "अंग्रेजी", "अँग्रेजी", "इंग्लिश",
           "इंग्लिश", "ஆங்கிலம்", "ஆங்கிலத்தில்", "ఇంగ్లీష్"],
    "hi": ["hindi", "हिंदी", "हिन्दी", "ஹிந்தி", "హిందీ"],
    "hinglish": ["hinglish", "हिंग्लिश"],
    "mr": ["marathi", "मराठी"],
    "ta": ["tamil", "tamizh", "तमिल", "तमिळ", "தமிழ்", "தமிழில்", "தமிழ"],
    "te": ["telugu", "तेलुगु", "तेलुगू", "तेलगु", "తెలుగు"],
    "kn": ["kannada", "kannad", "कन्नड़", "कन्नड", "ಕನ್ನಡ"],
    "ml": ["malayalam", "मलयालम", "മലയാളം"],
    "bn": ["bengali", "bangla", "bangali", "बंगाली", "बांग्ला", "বাংলা"],
    "gu": ["gujarati", "gujrati", "गुजराती", "ગુજરાતી"],
    "pa": ["punjabi", "panjabi", "पंजाबी", "ਪੰਜਾਬੀ"],
    "or": ["odia", "oriya", "उड़िया", "ओड़िया"],
    "ur": ["urdu", "उर्दू", "اردو"],
    "es": ["spanish"], "fr": ["french"], "de": ["german"], "it": ["italian"], "pt": ["portuguese"],
    "ja": ["japanese"], "zh": ["chinese", "mandarin"], "ko": ["korean"], "ru": ["russian"],
    "ar": ["arabic"], "sa": ["sanskrit", "संस्कृत"], "ne": ["nepali", "नेपाली"], "as": ["assamese"],
}
_ALIAS_TO_CODE: Dict[str, str] = {unicodedata.normalize("NFC", alias.lower()): code for code, names in _ALIASES.items() for alias in names}

_POSTPOSITIONS = {"mein", "me", "main", "men", "में", "मे"}
# Verbs a caller uses to ask for a language, by stem (Roman and Devanagari): bol-, baat, bata-, samjha-, jawab, likh-,
# sun-, kar-/kij-, chal-, shuru, dij-
_HI_VERB_STEMS = ("bol", "baat", "bat", "samjh", "jawab", "jawaab", "uttar", "likh", "sun", "kar", "kij", "chal",
                  "shuru", "dij", "reply", "speak", "talk", "answer", "continue", "बोल", "बात", "बता", "समझ", "जवाब",
                  "उत्तर", "लिख", "सुन", "कर", "कीजि", "चल", "शुरू", "दीजि", "दें", "बोलिए")
_HI_BOL_STEMS = ("bol", "बोल")
_EN_VERBS = {"speak", "talk", "reply", "respond", "answer", "continue", "switch", "change", "explain", "tell",
             "write", "converse", "chat", "go", "repeat", "speaking", "talking", "replying", "answering"}
_EN_PREPS = {"in", "to", "into", "with"}
_NEG = {"mat", "nahi", "nahin", "nhi", "नहीं", "नही", "मत", "band", "बंद", "not", "dont", "don", "never", "stop",
        "no", "without", "kabhi", "nope"}
_FILLER = {"please", "pls", "plz", "ok", "okay", "now", "wait", "hello", "hi", "sir", "madam", "ji", "yes", "and", "then",
           "but", "so", "just", "sorry", "bhai", "hey", "uh", "um", "sure", "thanks"}
# "does the doctor speak Marathi?" asks about someone else, it is not a request to the agent
_THIRD_PERSON = {"doctor", "doctors", "dr", "staff", "nurse", "anyone", "anybody", "someone", "somebody", "he", "she",
                 "they", "people", "who", "any", "receptionist", "dentist", "teacher", "teachers", "member"}


@dataclass(frozen=True)
class SwitchRequest:
    code: str          # "en", "hi", "hinglish", "ta", ...
    name: str          # display name of the language that was asked for
    supported: bool    # the agent can speak it
    matched: str       # the word the caller used for the language


def _is_request_at(tokens: List[str], i: int) -> bool:
    """Whether the language named at tokens[i] is being asked for (not just mentioned)."""
    n = len(tokens)

    def verbish(t: str) -> bool:
        return t.startswith(_HI_VERB_STEMS)

    # 1. "<lang> mein <verb>": Tamil mein bolo, हिंदी में बताइए, English me baat karo
    if i + 1 < n and tokens[i + 1] in _POSTPOSITIONS:
        window = tokens[i + 2:i + 7]
        for k, t in enumerate(window):
            if t in _NEG:
                return False  # "Hindi mein nahi, English mein baat karo": the first language is refused
            if verbish(t):
                return True
        return False
    # 2. "<lang> bolo", "<lang> bol sakte ho": the verb follows directly
    if i + 1 < n and tokens[i + 1].startswith(_HI_BOL_STEMS):
        return True
    # 3. English: "speak English", "switch to Hindi", "reply in Tamil", "can you speak Marathi", "in English please"
    j = i - 1
    prep = False
    if j >= 0 and tokens[j] in _EN_PREPS:
        prep, j = True, j - 1
    while j >= 0 and tokens[j] in {"back", "over", "only", "just", "fully", "completely", "all"}:
        j -= 1
    verb_at = None
    for k in range(j, max(-1, j - 5), -1):
        if tokens[k] in _EN_VERBS:
            verb_at = k
            break
    if verb_at is not None:
        before = tokens[max(0, verb_at - 3):verb_at]
        if any(t in _NEG for t in before) or any(t in _NEG for t in tokens[verb_at + 1:i]):
            return False
        if any(t in _THIRD_PERSON for t in before):
            return False
        # "tell me the fees in Hindi" is about the answer's language; so is "explain in English": both are requests
        return True
    if prep and (i + 1 >= n or tokens[i + 1] in {"please", "pls", "only", "now", "instead", "thanks", "plz"}):
        # "in English please" / "In Hindi.": nothing but filler may come before the preposition
        return all(t in _FILLER for t in tokens[:i - 1])
    return False


def find_switch_request(user_text: str, supported: Optional[Sequence[str]] = None) -> Optional[SwitchRequest]:
    """The language the caller explicitly asked the agent to speak in this turn (the last such request when they ask
    for several), or None. `supported` is the agent's language list: it only sets `.supported` on the result."""
    tokens = [t.lower() for t in WORD.findall(unicodedata.normalize("NFC", user_text or ""))]
    codes = supported_codes(supported)
    found: Optional[SwitchRequest] = None
    for i, tok in enumerate(tokens):
        code = _ALIAS_TO_CODE.get(tok)
        if code is None or not _is_request_at(tokens, i):
            continue
        if code == "hinglish":
            ok = "hi" in codes or "hinglish" in codes
        else:
            ok = code in codes or (code == "hi" and "hinglish" in codes)
        found = SwitchRequest(code, display_name(code), ok, tok)
    return found


def mentions_language(text: str, codes: Sequence[str]) -> bool:
    """Whether the text names one of the languages ("I can only speak English and Hindi", "हिंदी या अंग्रेज़ी")."""
    wanted = {base_code(c) for c in codes}
    if "hi" in wanted:
        wanted.add("hinglish")
    tokens = [unicodedata.normalize("NFC", t.lower()) for t in WORD.findall(text or "")]
    return any(_ALIAS_TO_CODE.get(t) in wanted for t in tokens)


# Languages whose script tells them apart from Hindi and English: a reply written in them can be spotted.
_DISTINCT_SCRIPT = INDIC_CODES - {"hi"} | {"ur", "ru", "ko", "ja", "zh"}


def declined_unsupported(request: SwitchRequest, reply: str, supported: Sequence[str]) -> bool:
    """For a request the agent cannot follow: the reply is not written in the language that was asked for (when its script
    can tell) and it names the languages the agent does speak."""
    spoke = request.code in _DISTINCT_SCRIPT and reply_matches(request.code, reply)
    return not spoke and mentions_language(reply, supported)


# ------------------------------------------------------------------------------------------------ the decision

@dataclass
class LanguageDecision:
    target: str                          # the language the reply must be in
    choice: Optional[str] = None         # the explicit choice to remember on the conversation (None: none made)
    choice_changed: bool = False         # the choice differs from the stored one: store it
    request: Optional[SwitchRequest] = None   # an explicit request made in this turn
    unsupported: Optional[str] = None    # display name of an unsupported language the caller asked for
    reason: str = "script"               # script | explicit | sticky | inherit | default

    @property
    def switched(self) -> bool:
        return bool(self.request and self.request.supported)


def _apply_choice(choice: str, user_text: str) -> str:
    """The target language for a remembered choice. Hindi keeps the caller's script."""
    if choice == "hi":
        return "hinglish" if hinglish.looks_hinglish(user_text) else "hi"
    return choice


def _is_ultra_short(user_text: str) -> bool:
    return len(_letter_words(user_text)) <= 2


def decide(user_text: str, supported: Optional[Sequence[str]] = None, sticky: Optional[str] = None,
           previous_agent_text: Optional[str] = None) -> LanguageDecision:
    """The whole policy for one caller turn. `sticky` is the language the caller asked for earlier in this conversation
    (stored on the conversation); `previous_agent_text` the agent's last reply (to carry the language through an
    ultra-short turn)."""
    codes = supported_codes(supported)
    sticky = base_code(sticky or "") or None
    req = find_switch_request(user_text, codes)
    choice = sticky
    unsupported = None
    if req is not None:
        if req.supported:
            choice = "hi" if req.code == "hi" else req.code
        else:
            unsupported = req.name
    base = detect_language(user_text, codes)
    if choice:
        reason = "explicit" if (req is not None and req.supported) else "sticky"
        return LanguageDecision(_apply_choice(choice, user_text), choice, choice != sticky, req, unsupported, reason)
    if _is_ultra_short(user_text) and base in ("en", codes[0]) and previous_agent_text:
        # nothing to read a language from ("ok", "thanks", a phone number): stay in the language of the last reply
        prev = detect_language(previous_agent_text, codes)
        if prev != base:
            return LanguageDecision(prev, None, False, req, unsupported, "inherit")
    return LanguageDecision(base, None, False, req, unsupported, "script" if (user_text or "").strip() else "default")


# ------------------------------------------------------------------------------------------------ the reply check

def _hinglish_reply(reply: str) -> bool:
    """Roman Hindi: enough Hindi function words. Looser than looks_hinglish for callers, since replies carry names."""
    tokens = hinglish._latin_tokens(reply)
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in hinglish.HINGLISH_MARKERS)
    return (hits >= 2 and hits / len(tokens) >= 0.12) or (hits >= 1 and len(tokens) <= 6)


_LATIN_WORD = re.compile(r"[A-Za-z\u00C0-\u024F][A-Za-z0-9\u00C0-\u024F.&'’/-]*")


def _latin_term(word: str, known: frozenset) -> bool:
    """A Latin word a reply in an Indian script may carry as it is: a name or a term ("Dr.", "Meera", "OPD", "MRI",
    "B.Tech", "Mechanical", "COVID-19"), or a word the caller's message or the passages contain."""
    w = word.strip(".'’-/&")
    if not w:
        return True
    return w[0].isupper() or "." in word or any(c.isdigit() for c in w) or w.lower() in known


def _native_share(reply: str, key: str, known_text: str = "") -> float:
    """The target script's share of the letters once names and terms in Latin letters are left out (see _latin_term).
    0 when the reply has no letter of the target script."""
    known = frozenset(w.lower().strip(".'’-/&") for w in _LATIN_WORD.findall(known_text or ""))
    rest = "".join(w for w in _LATIN_WORD.findall(reply) if not _latin_term(w, known))
    native = letter_counts(reply).get(key, 0)
    other = sum(n for k, n in letter_counts(_LATIN_WORD.sub(" ", reply)).items() if k != key)  # other scripts
    latin = letter_counts(rest).get("latin", 0)
    return native / (native + latin + other) if native else 0.0


def reply_matches(target: str, reply: str, known_text: str = "") -> bool:
    """Whether the reply is in the target language's script (and, for English and Hinglish, the right Latin language).
    True when it cannot tell (a language whose Latin text looks like another's, a reply without letters).

    For an Indian-script target, names and terms written in Latin letters ("Dr. Meera Kulkarni", "OPD", "MRI",
    "B.Tech Mechanical Engineering") do not count against the reply: a correct Hindi answer that names an English
    programme was flagged and regenerated before (docs/DOC_QA_EVAL.md, college-18). A capitalised word, an acronym, a
    word with a dot or a digit counts as a term, and so does any Latin word in `known_text` (what the caller said). The
    reply still needs a fifth of its letters in its own script: "Dr. Meera Monday ko milengi" is not Hindi."""
    counts = letter_counts(reply)
    total = sum(counts.values())
    if total == 0:
        return True
    target = base_code(target)
    frac = lambda key: counts.get(key, 0) / total  # noqa: E731
    if target in INDIC_CODES or target == "mr":
        key = "hi" if target == "mr" else target
        # at least a fifth of the letters in its own script, so "जी, Dr. Meera is available on Monday" stays English
        return frac(key) >= 0.5 or (frac(key) >= 0.2 and _native_share(reply, key, known_text) >= 0.5)
    if target in ("ur", "ru", "ko", "ja", "zh"):
        if target == "ja":
            return frac("ja") + frac("zh") >= 0.5
        return frac(target) >= 0.5
    if target == "hinglish":
        return frac("latin") >= 0.85 and _hinglish_reply(reply)
    if target == "en":
        return frac("latin") >= 0.85 and not hinglish.looks_hinglish(reply)
    if target in LATIN_CODES:
        return frac("latin") >= 0.85 and not hinglish.looks_hinglish(reply)
    return True


def enforce(target: str, reply: str, regenerate: Callable[[], Optional[str]],
            known_text: str = "") -> Tuple[str, dict]:
    """Keep `reply` when it is in the target language, else call `regenerate()` once and use its answer if THAT matches;
    otherwise keep the first reply. Returns (text, info) with info {"target", "matched" (of the first reply), "retried",
    "retry_matched"}. `regenerate` may raise or return None/empty: the first reply is kept. `known_text`: see
    reply_matches."""
    ok = reply_matches(target, reply, known_text)
    info = {"target": target, "matched": ok, "retried": False, "retry_matched": None}
    if ok:
        return reply, info
    info["retried"] = True
    try:
        again = (regenerate() or "").strip()
    except Exception:  # noqa: BLE001 - the retry is a nicety: never lose the reply we already have
        again = ""
    info["retry_matched"] = bool(again) and reply_matches(target, again, known_text)
    return (again, info) if info["retry_matched"] else (reply, info)


# ------------------------------------------------------------------------------------------------ prompt text

def supported_names(supported: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    for c in supported_codes(supported):
        name = display_name(c)
        if name not in out:
            out.append(name)
    if "hi" in supported_codes(supported) and "Hinglish" not in out:
        out.append("Hinglish")
    return out


def _join(names: Sequence[str]) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _standing(langs: str, target_name: Optional[str] = None) -> str:
    """The standing policy. `target_name` (this reply's language) is named only in the older layout: the turn code keeps
    it out of the system prompt, so the system prompt is the same on every turn of an agent."""
    where = f" ({target_name} for this reply)" if target_name else ""
    return (f"Language policy: answer in the language and script the caller uses in this message{where}; never switch "
            "language or script on your own. You can speak " + langs + ". If the caller asks you to speak one of "
            "those, switch to it and keep it until they ask for another. If they ask for any other language, say in one "
            "short sentence that you can only speak " + langs + ", offer to continue in one of them, and do not answer "
            "in the language they asked for. Keep names, numbers and times as they are.")


def _decision_note(decision: Optional[LanguageDecision], target_name: str, langs: str) -> str:
    """This turn's request, if the caller asked for a switch or for a language the agent cannot speak ("" otherwise)."""
    if decision is not None and decision.switched and decision.reason == "explicit":
        return (f" The caller has just asked you to speak {decision.request.name}: start with a few words "
                f"confirming the change, then answer in {target_name} only.")
    if decision is not None and decision.unsupported:
        return (f" The caller has just asked for {decision.unsupported}, which you cannot speak: say so briefly in "
                f"{target_name} and offer {langs}.")
    return ""


def policy_prompt(decision: Optional[LanguageDecision], supported: Optional[Sequence[str]],
                  target_name: str) -> str:
    """The language section of the system prompt (the older layout, the Evaluation Lab's): the standing policy naming
    this reply's language, plus a note about this turn's request."""
    langs = _join(supported_names(supported))
    return _standing(langs, target_name) + _decision_note(decision, target_name, langs)


def policy_rules(supported: Optional[Sequence[str]]) -> str:
    """The standing language policy alone: the same text on every turn of an agent, for the system prompt."""
    return _standing(_join(supported_names(supported)))


def turn_note(decision: Optional[LanguageDecision], supported: Optional[Sequence[str]], target_name: str) -> str:
    """This turn's language lines, for the caller's message: the reply language, then the note about this turn's
    request (if any). The per-turn half of policy_prompt."""
    note = _decision_note(decision, target_name, _join(supported_names(supported)))
    return f"Reply in {target_name}, even if the context is in another language.{note}"


def strong_instruction(target_name: str) -> str:
    """Appended to the prompt for the one regeneration after a reply in the wrong language."""
    return (f"IMPORTANT: your previous answer was not in the required language. Write the whole reply again in "
            f"{target_name}, and only in {target_name}: no other language or script, except names and numbers.")


def regen_messages(messages: List[dict], target_name: str, draft: Optional[str] = None) -> List[dict]:
    """The chat messages of the one regeneration: the prompt plus the stronger instruction, and, when there is a
    `draft`, the draft followed by a request to say it again in the target language."""
    msgs = [dict(m) for m in messages]
    msgs[0]["content"] += "\n\n" + strong_instruction(target_name)
    if draft:
        msgs += [{"role": "assistant", "content": draft},
                 {"role": "user", "content": f"Say exactly that again, in {target_name} only."}]
    return msgs
