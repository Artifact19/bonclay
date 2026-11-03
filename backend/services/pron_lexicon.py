"""Pronunciation lexicon: spelling hints for engines that read characters, keyed by (language, engine).

DhVaani is character-level (ZipVoice): it has no pronunciation dictionary, so a letter it renders badly can only be
fixed by writing the word differently. Each rule rewrites one spelling into another that the engine reads closer to
the intended sound. The text shown and stored is never touched; only the string handed to the engine changes.

OFF by default. `PRON_LEXICON=on` enables it. The shipped rules are hypotheses scored with a Whisper read-back, and
Whisper writes Marathi toward Hindi spelling, so a read-back cannot confirm them: the owner has to listen first
(listening files: see docs/BENCHMARK_INDIC.md, "Marathi lexicon"). Until then, leave it off.

The ("hi", "kokoro") rules are for Kokoro's Hindi voice hf_alpha, which has a phoneme front end but mispronounces a few
words; they were kept only when a read-back proved them (docs/BENCHMARK_INDIC.md, "Hindi Kokoro lexicon"). They are
reached from /api/kokoro/generate (voice hf_*), the turn path, Studio generation and Say it once.

    from services import pron_lexicon
    pron_lexicon.apply("वेळेवर कळवू", "mr", "dhvaani")        # unchanged unless PRON_LEXICON=on
    pron_lexicon.apply_rules(text, pron_lexicon.RULES[("mr", "dhvaani")])   # always applies (used by the bench)
"""
import os
import re
import unicodedata
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

ENV_FLAG = "PRON_LEXICON"


class Rule(NamedTuple):
    name: str
    pattern: "re.Pattern[str]"
    repl: str
    note: str


def rule(name: str, pattern: str, repl: str, note: str = "") -> Rule:
    return Rule(name, re.compile(pattern), repl, note)


# A Devanagari letter, mark or digit: everything in the block except the danda and double danda (U+0964, U+0965).
_DEV = "\u0900-\u0963\u0966-\u097f"


def whole_word(word: str) -> str:
    """Regex for `word` as a whole Devanagari word: not preceded or followed by another Devanagari character, so the
    rule for शांत never fires inside शांति and the rule for सात never fires inside सातवां. NFC, like apply_rules."""
    return f"(?<![{_DEV}]){re.escape(unicodedata.normalize('NFC', word))}(?![{_DEV}])"


# Engines that read characters and so can be steered by spelling. Engines with their own text frontend are not listed
# and are returned unchanged.
RULES: Dict[Tuple[str, str], List[Rule]] = {
    ("mr", "dhvaani"): [
        # Candidates measured by bench/pron_lexicon_check.py (docs/BENCHMARK_INDIC.md, "Marathi lexicon"). Both are
        # unconfirmed by ear: the read-back changes are small and Whisper is biased, so neither is proven.
        rule("rha->raha", "ऱ्ह", "रह", "read-back 'ऱ्ह lost' 3 -> 1 over 4 sentences; the only candidate with a visible change"),
        rule("la->la-nukta", "ळ", "ल\u093c", "read-back 'ळ lost' 6 -> 5: within noise; kept as the least invasive ळ candidate"),
    ],
    ("hi", "kokoro"): [
        # Measured against Kokoro's Hindi voice hf_alpha by bench/hindi_lexicon_check.py (docs/BENCHMARK_INDIC.md, "Hindi
        # Kokoro lexicon"). Each word is replaced only as a whole word. A rule is here when the word was transcribed
        # right in at least 2 of 3 carrier sentences, with no more errors on the word in any sentence, fewer in one,
        # and no rise in mean sentence CER. Whisper proves what it hears, not what a listener prefers: listen to the
        # before/after pairs named in the doc before turning PRON_LEXICON on.
        rule("mumbai", whole_word("मुंबई"), "मुम्बई",
             "anusvara spelled as the nasal conjunct (espeak: mˈũbˌi -> mˈʊmbəˌi); word right 0/3 -> 2/3, Whisper writes मुंबई"),
        rule("chhah", whole_word("छह"), "छे",
             "छह is spoken 'chhe'; the plain spelling came back as छाहा/चाह. Word right 1/3 -> 3/3 (Whisper writes छे)"),
        rule("shaant", whole_word("शांत"), "शान्त",
             "anusvara spelled as न् + conjunct (espeak: ʃˈãt -> ʃˈaːnt); word right 2/3 -> 3/3 (शाथ before)"),
    ],
}


def enabled() -> bool:
    return os.getenv(ENV_FLAG, "").strip().lower() in ("on", "1", "true", "yes")


def _key(language: Optional[str], engine: Optional[str]) -> Tuple[str, str]:
    return ((language or "").strip().lower().split("-")[0], (engine or "").strip().lower())


def has_rules(language: Optional[str], engine: Optional[str]) -> bool:
    return bool(RULES.get(_key(language, engine)))


def apply_rules(text: str, rules: Sequence[Rule]) -> str:
    text = unicodedata.normalize("NFC", text or "")
    for r in rules:
        text = r.pattern.sub(r.repl, text)
    return text


def apply(text: str, language: Optional[str], engine: Optional[str]) -> str:
    """The text to hand to `engine`. A no-op unless PRON_LEXICON=on and the (language, engine) pair has rules."""
    if not text or not enabled():
        return text
    rules = RULES.get(_key(language, engine))
    return apply_rules(text, rules) if rules else text


def hits(text: str, language: Optional[str], engine: Optional[str]) -> List[str]:
    """Names of the rules that would change `text` (for logs and the benchmark), regardless of the env flag."""
    return [r.name for r in RULES.get(_key(language, engine), []) if r.pattern.search(unicodedata.normalize("NFC", text or ""))]
