"""One definition of read-back character error rate (CER) for every Indian-language script.

Before this module two definitions existed: `indic_clone_trial._letters` (all letters and vowel signs, any script) and
`readback.norm_hi` (Devanagari only). They agree on clean Hindi/Marathi and differ as soon as the transcript holds
Latin words. The rule now:

  letters(s)   NFC, keep Unicode letters and combining marks (categories L*, M*) of any script; drop spaces,
               punctuation (danda included), digits, and zero-width joiners. `script="devanagari"` additionally drops
               every letter outside that block (the old `norm_hi`; `readback.py` keeps using it so its numbers do not move).
  cer(ref, hyp) Levenshtein distance over those characters / reference length. None for an empty reference, 1.0
               for an empty hypothesis. Same value as jiwer.cer on the same strings, without needing jiwer.
  normalise_hyp Whisper writes numbers as digits; replace the digit runs with the sentence's number words, in order.

Limits: CER on a script treats a dependent vowel sign as one character, so numbers are not comparable across
scripts, only across engines in the same language. Reading back through an ASR also measures the ASR (see
docs/BENCHMARK_INDIC.md).
"""
import re
import unicodedata
from typing import Iterable, Optional

_DIGITS = re.compile(r"[0-9०-९০-৯௦-௯౦-౯೦-೯૦-૯൦-൯੦-੯]+(?:[.,][0-9०-९০-৯௦-௯౦-౯೦-೯૦-૯൦-൯੦-੯]+)?")
_BLOCKS = {"devanagari": (0x0900, 0x097F)}


def letters(s: str, script: Optional[str] = None) -> str:
    s = unicodedata.normalize("NFC", s or "")
    lo_hi = _BLOCKS[script] if script else None
    return "".join(c for c in s if unicodedata.category(c)[0] in "LM" and (lo_hi is None or lo_hi[0] <= ord(c) <= lo_hi[1]))


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, script: Optional[str] = None) -> Optional[float]:
    r, h = letters(ref, script), letters(hyp, script)
    if not r:
        return None
    if not h:
        return 1.0
    return edit_distance(r, h) / len(r)


def normalise_hyp(hyp: str, nums: Iterable[str]) -> str:
    """Replace digit runs in the transcript with the sentence's number words, in order (extra runs are dropped)."""
    it = iter(nums)
    return _DIGITS.sub(lambda m: next(it, ""), hyp or "")
