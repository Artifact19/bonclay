"""Scoring speech recognition on Hinglish honestly: the same words in two scripts must not count as errors.

The 15-clip Hinglish bench of 20 Oct 2026 scored WER 0.799, with plain `jiwer.wer` against Roman references while the
recogniser often answered in Devanagari: a perfect Devanagari transcript of a Roman reference scores about 1.0. Here both
sides are brought to one Roman spelling before scoring:

  1. Devanagari is transliterated to Roman (indic_transliteration, MIT; the ITRANS scheme).
  2. Roman spellings are folded: lower case, diacritics removed, long vowels collapsed (aa->a, ee->i, oo->u, ai->e,
     au->o), aspirates dropped (kh->k, bh->b, sh->s), w->v, z->j, c->k except in "ch", every non-initial "a" dropped
     (the schwa: Roman Hinglish writes "kitna", "naam", "subah"; ITRANS writes "kitanA", "nAma", "subaha"), doubled
     letters collapsed, punctuation removed. English loanwords keep a residue ("doctor" against "DOkTara" folds to
     "doktor" against "doktr"), which the CER reports as a fraction of a word rather than a whole wrong word.

Folding loses distinctions (it is for scoring, never for display), so a WER on folded text is a lower bound on the
real error and a CER on it is the more forgiving, spelling-tolerant view. Report raw WER, folded WER and folded CER
side by side; the folded numbers are the ones to compare recognisers by for Hinglish.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))


def transliterate(text: str) -> str:
    """Devanagari runs of `text` in Roman letters (ITRANS); other characters unchanged. Without the library, the
    Devanagari is left as it is (the fold then still scores it, just badly)."""
    if not has_devanagari(text):
        return text
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate as _tr
    except ImportError:  # pragma: no cover - the bench venv has it
        return text
    # ITRANS has no sign for the candra vowels of English loanwords (डॉक्टर, बैंक): read them as plain o / e first.
    text = text.replace("ॉ", "ो").replace("ऑ", "ओ").replace("ॅ", "े")
    out = []
    for run in re.split(r"([ऀ-ॿ]+)", text):
        if has_devanagari(run):
            run = _tr(run, sanscript.DEVANAGARI, sanscript.ITRANS).replace("M", "n")   # anusvara -> n
        out.append(run)
    return "".join(out)


_FOLD_STEPS = (
    (re.compile(r"[.~^'`\"]"), ""),              # ITRANS marks (anusvara ".n", visarga ".h", chandra "~")
    (re.compile(r"[^a-z0-9 ]+"), " "),            # punctuation (Devanagari danda included) to spaces
    (re.compile(r"aa"), "a"),                     # long vowels, before the schwa rule below
    (re.compile(r"ee|ii"), "i"),
    (re.compile(r"oo|uu"), "u"),
    (re.compile(r"ai"), "e"),
    (re.compile(r"au"), "o"),
    (re.compile(r"c(?!h)"), "k"),                 # c -> k, but "ch" stays a sound of its own
    (re.compile(r"(?<=[kgjtdpbs])h"), ""),       # aspirates: kh gh th dh ph bh -> k g t d p b; sh -> s
    (re.compile(r"ch"), "c"),
    (re.compile(r"w"), "v"),
    (re.compile(r"z"), "j"),
    (re.compile(r"q"), "k"),
    (re.compile(r"x"), "ks"),
    (re.compile(r"aya?\b"), "e"),                 # Roman "chai" (-> che) against ITRANS "chAya" (-> chaya -> che)
    (re.compile(r"(?<=[a-z])a"), ""),             # schwa: Roman Hinglish writes "kitna", ITRANS "kitanA"; both -> kitn
    (re.compile(r"iy"), "i"),                     # "chahiye" against "chAhie"
    (re.compile(r"([a-z])\1+"), r"\1"),           # doubled letters
    (re.compile(r"\s+"), " "),
)


def fold(text: str) -> str:
    """One Roman spelling for scoring: transliterate, strip accents, lower-case, then the folds above."""
    s = transliterate(text or "").translate(_DEVANAGARI_DIGITS)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    for pattern, repl in _FOLD_STEPS:
        s = pattern.sub(repl, s)
    return s.strip()


def wer(ref: str, hyp: str) -> float:
    import jiwer
    r, h = ref.strip(), hyp.strip()
    if not r:
        return 0.0 if not h else 1.0
    return 1.0 if not h else float(jiwer.wer(r, h))


def cer(ref: str, hyp: str) -> float:
    import jiwer
    r, h = ref.strip(), hyp.strip()
    if not r:
        return 0.0 if not h else 1.0
    return 1.0 if not h else float(jiwer.cer(r, h))


def hinglish_scores(ref: str, hyp: str) -> Dict[str, Optional[float]]:
    """Raw WER (the old number), folded WER and folded CER for one Hinglish clip."""
    raw_ref = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", unicodedata.normalize("NFKC", ref).lower().replace("’", "'")).replace("'", "")).strip()
    raw_hyp = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", unicodedata.normalize("NFKC", hyp).lower().replace("’", "'")).replace("'", "")).strip()
    fr, fh = fold(ref), fold(hyp)
    return {"wer_raw": wer(raw_ref, raw_hyp), "wer_folded": wer(fr, fh), "cer_folded": cer(fr, fh),
            "hyp_devanagari": has_devanagari(hyp)}
