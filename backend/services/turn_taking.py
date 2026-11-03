"""Turn taking on a phone call: how long to wait for the caller after the agent has asked something, and whether what the
caller has said so far is a finished answer.

Why. A fixed half second of silence ends the caller's turn. People dictating details ("the name is ..., the number is
..." [pause] "... tomorrow at five") pause 0.7 to 1.5 s between the items, so the agent cut in during the pause and asked
again for what it had already been told. The agent knows what it just asked for; this module turns that into a wait:

    several details asked at once ("details", two kinds of detail)   PHONE_DETAILS_END_SILENCE_MS  (default 1500)
    one name, number, date, time, e-mail or address asked            PHONE_DETAIL_END_SILENCE_MS   (default 1100)
    anything else                                                    PHONE_END_SILENCE_MS          (default 500)

and, on the transcript the speculative turn already has, `is_incomplete` says the caller has probably not finished
("... and", a trailing comma, 7 digits of a phone number): the end of turn is then not confirmed once, for
PHONE_INCOMPLETE_EXTRA_MS (default 1200) more.

The detail-word lists (`DETAIL_WORDS`, `DETAIL_REQUEST`, `asks_for_detail`) used to live in backend.py, where the
instant-answer guard still uses them; they are here so the phone router and the turn share one definition.

Settings: PHONE_ADAPTIVE_END (on|off, default on), PHONE_DETAILS_END_SILENCE_MS, PHONE_DETAIL_END_SILENCE_MS,
PHONE_INCOMPLETE_EXTRA_MS (0 turns the incomplete-answer wait off), PHONE_CARRY_UNANSWERED (on|off, default on: see
routers/phone_exotel.py).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

try:
    from services import spoken_digits
except ImportError:  # pragma: no cover - imported as backend.services.turn_taking
    from backend.services import spoken_digits

# ------------------------------------------------------------------------------------------------ settings

DEFAULT_END_MS = 500
DEFAULT_DETAIL_END_MS = 1100
DEFAULT_DETAILS_END_MS = 1500
DEFAULT_INCOMPLETE_EXTRA_MS = 1200
MAX_END_MS = 4000          # whatever the settings say, a caller is never kept waiting longer than this for silence


def _flag(name: str, default: str = "on") -> bool:
    return os.getenv(name, default).strip().lower() not in ("off", "0", "false", "no")


def _int_env(name: str, default: int, low: int = 0, high: int = MAX_END_MS) -> int:
    try:
        value = int(float(os.getenv(name, "").strip() or default))
    except ValueError:
        value = default
    return max(low, min(high, value))


def adaptive_enabled() -> bool:
    return _flag("PHONE_ADAPTIVE_END")


def carry_enabled() -> bool:
    return _flag("PHONE_CARRY_UNANSWERED")


def base_end_ms() -> int:
    return _int_env("PHONE_END_SILENCE_MS", DEFAULT_END_MS, low=100)


def incomplete_extra_ms() -> int:
    return _int_env("PHONE_INCOMPLETE_EXTRA_MS", DEFAULT_INCOMPLETE_EXTRA_MS)


# --------------------------------------------------------------------------------- what did the agent ask for

# The agent's previous reply asked the caller for a detail (a name, a number, a date...): the caller's next words answer
# it. Simple and on the careful side: a detail word in the last sentence of that reply, and the sentence is a question
# or a request ("please tell me", "बताइए").
DETAIL_WORDS = re.compile(
    r"\b(?:name|number|phone|mobile|contact|e-?mail|address|date|day|time|slot|age|doctor|department|course|details?|"
    r"confirm|spell|when|which|pin ?code|naam|tareekh|tarikh|samay|kab|kaunsa|kaun sa|kaunsi|kaun si)\b"
    r"|नाम|नंबर|नम्बर|मोबाइल|फ़ोन|फोन|तारीख|तारीख़|दिन|समय|पता|ईमेल|उम्र|डॉक्टर|कब|कौन\s?सा|कौन\s?सी|पुष्टि|कन्फर्म",
    re.IGNORECASE)
DETAIL_REQUEST = re.compile(
    r"\b(?:please (?:tell|share|give|confirm|provide|spell|send|say)|could you (?:tell|share|give|confirm|provide)|"
    r"can you (?:tell|share|give|confirm|provide)|may i (?:have|know|get)|can i (?:have|get|know)|"
    r"bataiye|batayein|bataye|bata dijiye|bata do|dijiye)\b"
    r"|बताइए|बताइये|बताएं|बताएँ|बताओ|बता दीजिए|बता दें|दीजिए|दीजिये",
    re.IGNORECASE)
LAST_SENTENCE_SPLIT = re.compile(r"(?<=[.!।])\s+")
_ANY_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।？])\s+")
# "tell me your name" / "share your number" without a "please": a request too, for the wait (not for instant answers)
_ASK_VERB = re.compile(r"\b(?:tell me|share|give me|let me know|spell|say|bataiye|batao|bolo|boliye|bata)\b|बताइए|बताओ|बोलिए",
                       re.IGNORECASE)


def asks_for_detail(text: Optional[str]) -> bool:
    """True when the agent's reply ends by asking the caller for a detail (see DETAIL_WORDS)."""
    t = (text or "").strip()
    if not t:
        return False
    last = [p for p in LAST_SENTENCE_SPLIT.split(t) if p.strip()][-1]
    if not DETAIL_WORDS.search(last):
        return False
    return last.rstrip().endswith(("?", "？")) or bool(DETAIL_REQUEST.search(last))


# The kinds of detail, by what the agent's words say. "when" asks for a date; "which", "confirm" and the like are
# detail words but not a kind of detail of their own.
_KINDS = {
    "name": re.compile(r"\bnames?\b|\bnaam\b|नाम", re.IGNORECASE),
    "phone": re.compile(r"\b(?:phone|mobile|contact|number|nambar|no\.)\b|नंबर|नम्बर|मोबाइल|फ़ोन|फोन", re.IGNORECASE),
    "email": re.compile(r"\be-?mail\b|ईमेल", re.IGNORECASE),
    "address": re.compile(r"\baddress\b|\bpin ?code\b|\bpata\b|पता|पिन ?कोड", re.IGNORECASE),
    "date": re.compile(r"\b(?:date|day|tareekh|tarikh|kab|when)\b|तारीख़?|दिन|कब", re.IGNORECASE),
    "time": re.compile(r"\b(?:time|slot|samay|baje)\b|समय|बजे", re.IGNORECASE),
    "age": re.compile(r"\bage\b|\bumar\b|उम्र", re.IGNORECASE),
    "doctor": re.compile(r"\b(?:doctor|department|course)\b|डॉक्टर", re.IGNORECASE),
}
_DETAILS_WORD = re.compile(r"\bdetails?\b|\bjaankari\b|\bjankari\b|जानकारी", re.IGNORECASE)
# kinds a caller dictates slowly, with pauses: one of these asked gets the longer wait
_DICTATED = ("name", "phone", "email", "address", "date", "time")


@dataclass(frozen=True)
class Asked:
    kinds: tuple = ()          # the kinds of detail the last reply asked for, in the order of _KINDS
    several: bool = False      # two or more kinds, or the word "details"

    @property
    def phone(self) -> bool:
        return "phone" in self.kinds

    @property
    def any(self) -> bool:
        return bool(self.kinds) or self.several


def asked_for(agent_text: Optional[str]) -> Asked:
    """What the agent's last reply asked the caller for. Nothing unless its last sentence is a question or a request for
    a detail; then the kinds are read from the last two sentences ("What is the patient's name? And a phone number?")."""
    t = (agent_text or "").strip()
    if not t:
        return Asked()
    parts = [p for p in _ANY_SENTENCE_SPLIT.split(t) if p.strip()]
    last = parts[-1]
    if not (DETAIL_WORDS.search(last) or _DETAILS_WORD.search(last)):
        return Asked()
    asks = last.rstrip().endswith(("?", "？")) or bool(DETAIL_REQUEST.search(last)) or bool(_ASK_VERB.search(last))
    if not asks:
        return Asked()
    window = re.sub(r"e-?mail\s+address", "email", " ".join(parts[-2:]), flags=re.IGNORECASE)
    kinds = tuple(k for k, rx in _KINDS.items() if rx.search(window))
    several = len(kinds) >= 2 or bool(_DETAILS_WORD.search(window))
    return Asked(kinds=kinds, several=several)


@dataclass(frozen=True)
class EndWait:
    ms: int
    kind: str                  # "details" | "detail" | "normal"
    asked: Asked = Asked()

    @property
    def label(self) -> str:
        return f"{self.ms}({self.kind})"


def end_wait(agent_text: Optional[str]) -> EndWait:
    """The end-of-turn silence to use for the caller's next utterance, from what the agent's last reply asked."""
    base = base_end_ms()
    if not adaptive_enabled():
        return EndWait(base, "normal")
    asked = asked_for(agent_text)
    if asked.several:
        return EndWait(max(base, _int_env("PHONE_DETAILS_END_SILENCE_MS", DEFAULT_DETAILS_END_MS)), "details", asked)
    if any(k in _DICTATED for k in asked.kinds):
        return EndWait(max(base, _int_env("PHONE_DETAIL_END_SILENCE_MS", DEFAULT_DETAIL_END_MS)), "detail", asked)
    return EndWait(base, "normal", asked)


# ------------------------------------------------------------------------------------ an unfinished answer

# What a caller says while still thinking of the next item, compared with the end of the transcript.
_INCOMPLETE_END = re.compile(
    r"(?:^|[\s,])(?:and|or|but|then|also|um+|uh+|er+|erm|hmm+|aur|ya|toh|phir|fir|uske baad|iske baad|aur phir|matlab|"
    r"haan wo|haan woh|और|या|तो|फिर|उसके बाद|इसके बाद|मतलब|हाँ वो|हां वो)\s*$", re.IGNORECASE)


def is_incomplete(transcript: Optional[str], agent_text: Optional[str] = None) -> Optional[str]:
    """Why the caller has probably not finished (a short reason), or None when the answer looks complete.

      "trailing"  the transcript ends in a conjunction or filler ("... and", "... aur", "um", "uh", "haan wo") or a
                  comma or ellipsis: the next item is on its way;
      "phone"     the agent asked for a phone number (or for "the details") and the longest run of digits said so far has
                  1 to 9 digits (3 or more when it also asked for a date or a time, or only for "the details": a day of
                  the month is not a number in progress)."""
    text = (transcript or "").strip()
    if not text:
        return None
    if text.endswith((",", "…", "...", "،")):
        return "trailing"
    bare = re.sub(r"[.।!]+$", "", text).strip()
    if bare and _INCOMPLETE_END.search(bare):
        return "trailing"
    if bare and bare.endswith(("-", "–")):
        return "trailing"
    asked = asked_for(agent_text)
    # "the patient details" may well include a number, but no kind was named: only a run of 3 or more digits counts
    if asked.phone or (asked.several and not asked.kinds):
        n = spoken_digits.longest_run_length(text)
        floor = 3 if (not asked.phone or "date" in asked.kinds or "time" in asked.kinds) else 1
        if floor <= n <= 9:
            return "phone"
    return None
