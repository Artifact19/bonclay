"""Digits as a caller says them: "98765 43210", "nine eight seven...", "nau aath saat ...", "९८७६५...", "double five".

Used by two things that must agree on what a caller said:
  * services/caller_memory.py reads a 10-digit Indian mobile number out of the caller's words;
  * services/turn_taking.py counts how many digits of a number have been said so far (a phone number dictated in two
    pieces is not finished at 5 digits).

Pure functions, no I/O. A "run" is digits said one after another, written either as numerals or as words, with
nothing else in between; a sentence end (. ! ? । ;) always ends a run. Commas, hyphens and spaces do not (people
read a number in groups).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Words that name a digit. English, romanised Hindi (and the usual spellings) and Devanagari.
DIGIT_WORDS = {
    "zero": 0, "shunya": 0, "shoonya": 0, "sunya": 0, "jeero": 0, "ziro": 0, "शून्य": 0, "जीरो": 0, "ज़ीरो": 0,
    "one": 1, "ek": 1, "एक": 1,
    "two": 2, "do": 2, "दो": 2,
    "three": 3, "teen": 3, "tin": 3, "तीन": 3,
    "four": 4, "char": 4, "chaar": 4, "चार": 4,
    "five": 5, "paanch": 5, "panch": 5, "pach": 5, "paach": 5, "पाँच": 5, "पांच": 5,
    "six": 6, "chhe": 6, "chhah": 6, "chheh": 6, "chah": 6, "che": 6, "chhay": 6, "छह": 6, "छः": 6, "छे": 6, "छ": 6,
    "seven": 7, "saat": 7, "sat": 7, "सात": 7,
    "eight": 8, "aath": 8, "ath": 8, "आठ": 8,
    "nine": 9, "nau": 9, "nao": 9, "नौ": 9,
    "oh": 0, "o": 0,
}
# "oh" is a zero ("nine eight oh") but also a sigh; these are real words in other sentences. A run made only of them
# ("do" = to do, "sat" = sat down, "one") is not a number.
AMBIGUOUS = {"oh", "o", "do", "sat", "ath", "che", "one", "tin", "ek", "एक", "दो", "छ"}

MULTIPLIERS = {"double": 2, "dubal": 2, "dabal": 2, "डबल": 2, "triple": 3, "treble": 3, "tripal": 3, "ट्रिपल": 3}

_HARD_BREAK = re.compile(r"[.!?।;]+(?=\s|$)")
_STRIP = ".,;:!?-–—()[]{}\"'“”‘’`|/\\"
_BREAK = "|"


def _tokens(text: str) -> List[str]:
    """Words and numerals of `text`, lower case, Devanagari digits as ASCII. A sentence end becomes the token "|"."""
    text = str(text or "").translate(_DEVANAGARI_DIGITS).lower().replace("+", " ")
    text = _HARD_BREAK.sub(" | ", text)
    out: List[str] = []
    for raw in text.split():
        if raw == _BREAK:
            out.append(_BREAK)
            continue
        tok = raw.strip(_STRIP)
        if tok:
            out.append(tok)
    return out


@dataclass
class _Entry:
    digits: str
    ambiguous: bool
    start: int        # index of its first token ("double" counts) in the token list
    end: int          # one past its last token


def _scan(tokens: List[str]) -> List[List[_Entry]]:
    """The runs of digit tokens: each run is a list of entries. A run with only ambiguous words in it is dropped."""
    runs: List[List[_Entry]] = []
    current: List[_Entry] = []
    pending, pending_at = 1, -1

    def flush() -> None:
        nonlocal current
        if current and not all(e.ambiguous for e in current):
            runs.append(current)
        current = []

    for i, tok in enumerate(tokens):
        if tok in MULTIPLIERS:
            pending, pending_at = MULTIPLIERS[tok], i
            continue
        start = pending_at if pending > 1 else i
        if tok.isdigit() and tok.isascii():
            digits = tok * pending if (pending > 1 and len(tok) == 1) else tok
            current.append(_Entry(digits, False, start, i + 1))
        elif tok in DIGIT_WORDS:
            current.append(_Entry(str(DIGIT_WORDS[tok]) * pending, tok in AMBIGUOUS, start, i + 1))
        else:
            flush()
        pending, pending_at = 1, -1
    flush()
    return runs


def digit_token_runs(text: str) -> List[List[Tuple[str, bool]]]:
    """The runs of digit tokens in `text`: each run is a list of (digits, ambiguous) in the order said. "double five"
    gives ("55", False)."""
    return [[(e.digits, e.ambiguous) for e in run] for run in _scan(_tokens(text))]


def longest_run_length(text: str) -> int:
    """How many digits the longest run of spoken digits in `text` has (0 when none)."""
    return max((sum(len(e.digits) for e in run) for run in _scan(_tokens(text))), default=0)


def _valid_mobile(digits: str) -> Optional[str]:
    """The 10-digit Indian mobile number in `digits` (6-9 first), taking off a leading 91 or 0 when the length says so."""
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    elif len(digits) == 13 and digits.startswith("091"):
        digits = digits[3:]
    return digits if len(digits) == 10 and digits[0] in "6789" else None


def _mobile_span(run: List[_Entry]) -> Optional[Tuple[int, int, str]]:
    """(first entry, last entry + 1, number) of a mobile number made of whole entries of `run`: the whole run, or a
    stretch of it (the number may be followed by a day of the month: "98765 43210 25")."""
    for i in range(len(run)):
        total = ""
        for j in range(i, len(run)):
            total += run[j].digits
            if len(total) > 13:
                break
            found = _valid_mobile(total)
            if found:
                return i, j + 1, found
    return None


def find_mobile(text: str) -> Optional[str]:
    """A 10-digit Indian mobile number said in `text`, as 10 digits, or None."""
    runs = _scan(_tokens(text))
    for run in runs:
        span = _mobile_span(run)
        if span:
            return span[2]
    # a number read with something in between ("98765, uh, 43210" or "98765. 43210"): join the runs when that is one
    if 1 < len(runs) <= 3:
        return _valid_mobile("".join(e.digits for run in runs for e in run))
    return None


def strip_digit_runs(text: str, min_digits: int = 7) -> str:
    """`text` without a phone number's digits (the 10 of a mobile number, or any run of `min_digits` or more), so a time
    or date reader does not take "9" or "five" of a number for an hour. Other words stay in order."""
    tokens = _tokens(text)
    drop = set()
    for run in _scan(tokens):
        span = _mobile_span(run)
        if span:
            drop.update(range(run[span[0]].start, run[span[1] - 1].end))
        elif sum(len(e.digits) for e in run) >= min_digits:
            drop.update(range(run[0].start, run[-1].end))
    return " ".join(t for i, t in enumerate(tokens) if i not in drop and t != _BREAK)
