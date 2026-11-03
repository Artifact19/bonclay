"""Numbers in a Hindi (Devanagari) reply, rewritten so the Hindi voice reads them the way a person would say them.

Kokoro speaks Hindi through espeak-ng (misaki's EspeakG2P). Checked on 2026-10-19 with that exact path:
- "1,60,000" (Indian digit grouping) came out as "ek, saath, shunya shunya shunya": the commas split the number.
  Without them espeak-ng reads "160000" as "ek lakh saath hazaar", and "25000000" as "do karod pachaas lakh".
- "₹" was silent and "Rs." was spelt out in English ("R S").
- "9876543210" was read as one amount ("nau arab ..."), "020 5550 1210" as "shunya, bees, paanch hazaar ...".
- "1:00 PM" came out as "ek shunya shunya P M".

The agent writes numbers in digits (the grounding prompt asks for it, so the number guard can check them); this
module turns them into what the voice should say, for the TTS only: the text shown and stored is unchanged.
Deterministic and fast (a few regular expressions).
"""
from __future__ import annotations

import re
import unicodedata

_DEVA_DIGITS = {0x0966 + i: ord("0") + i for i in range(10)}

# 1,60,000 / 2,50,00,000 (Indian grouping) and 160,000 (Western grouping), with an optional decimal part
_GROUPED = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{2})*,\d{3}(?:\.\d+)?(?![\d,])|(?<![\d,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])")
# a currency mark before an amount: ₹1500, Rs. 1500, Rs 1500, INR 1500
_CURRENCY = re.compile(r"(?:₹|\bRs\.?|\bINR\b)\s*(\d+(?:\.\d+)?)(?:\s*/-)?(?:\s*(?:रुपये|रुपए|रुपया|रूपये)(?![\u0900-\u0963\u0971-\u097F]))?",
                       re.IGNORECASE)
_PERIOD_BEFORE = re.compile(r"(?:सुबह|दोपहर|शाम|रात)\s*$")
# a phone number: 7+ digits, in one block or in groups joined by ONE space or hyphen ("020 5550 1210", "98765-43210"),
# optionally with +91 ("500 - 1000", a range, has spaces around the hyphen and is left alone)
_PHONE = re.compile(r"(?<![\d.:,])(?:\+\s?91[\s-]?)?\d(?:[ -]?\d){6,}(?![\d:,])")
_TIME = re.compile(r"(?<![\d.:])(\d{1,2}):(\d{2})(?!\d)(?:\s*(am|pm|a\.m\.|p\.m\.)(?![a-z]))?(\s*बजे)?", re.IGNORECASE)
_HOUR_AMPM = re.compile(r"(?<![\d.:])(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)(?![a-z])(\s*बजे)?", re.IGNORECASE)
_HALF_MULT = re.compile(r"(?<![\d.])(\d+)\.5\s*(लाख|करोड़|करोड|हज़ार|हजार)")
_DECIMAL = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])")
_PERCENT = re.compile(r"(\d)\s*%")


def _period(h24: int) -> str:
    """The part of the day a Hindi speaker puts before a clock time."""
    if h24 < 4 or h24 >= 20:
        return "रात"
    if h24 < 12:
        return "सुबह"
    if h24 < 16:
        return "दोपहर"
    return "शाम"


def _clock(h: int, m: int, ampm: str, period_said: bool = False) -> str:
    """A clock time as said in Hindi: "सुबह साढ़े 9 बजे", "दोपहर 1 बजे", "शाम 4 बजकर 10 मिनट"."""
    ampm = ampm.lower().replace(".", "")
    if ampm == "pm" and h < 12:
        h24 = h + 12
    elif ampm == "am" and h == 12:
        h24 = 0
    else:
        h24 = h
    h12 = (h24 % 12) or 12
    # a period word only when the time says which half of the day it is (AM/PM or the 24-hour clock)
    period = _period(h24) + " " if (ampm or h24 > 12 or h24 == 0) and not period_said else ""
    nxt = (h12 % 12) + 1
    if m == 0:
        said = f"{h12} बजे"
    elif m == 30:
        said = "डेढ़ बजे" if h12 == 1 else "ढाई बजे" if h12 == 2 else f"साढ़े {h12} बजे"
    elif m == 15:
        said = f"सवा {h12} बजे"
    elif m == 45:
        said = f"पौने {nxt} बजे"
    else:
        said = f"{h12} बजकर {m} मिनट"
    return period + said


def _time(match: re.Match) -> str:
    h, m = int(match.group(1)), int(match.group(2))
    if h > 24 or m > 59:
        return match.group(0)
    return _clock(h, m, match.group(3) or "", bool(_PERIOD_BEFORE.search(match.string[:match.start()])))


def _hour_ampm(match: re.Match) -> str:
    h = int(match.group(1))
    if h < 1 or h > 12:
        return match.group(0)
    return _clock(h, 0, match.group(2), bool(_PERIOD_BEFORE.search(match.string[:match.start()])))


_AMOUNT_BEFORE = re.compile(r"(?:₹|\bRs\.?|\bINR)\s*$", re.IGNORECASE)
_AMOUNT_AFTER = re.compile(r"^\s*(?:/-|रुपये|रुपए|रुपया|रूपये|लाख|करोड़|हज़ार|हजार)")


def _phone(match: re.Match) -> str:
    raw = match.group(0)
    before, after = match.string[:match.start()], match.string[match.end():]
    if (re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw) or _AMOUNT_BEFORE.search(before[-6:])
            or _AMOUNT_AFTER.match(after)):
        return raw  # a date (2026-10-20) or an amount written without commas (₹2500000): not a phone number
    digits = re.sub(r"\D", "", raw)
    if raw.lstrip().startswith("+") and digits.startswith("91") and len(digits) > 10:
        digits = digits[2:]
    return " ".join(digits)


def _half(match: re.Match) -> str:
    n = int(match.group(1))
    word = "डेढ़" if n == 1 else "ढाई" if n == 2 else f"साढ़े {n}"
    return f"{word} {match.group(2)}"


def hindi_numbers_for_speech(text: str) -> str:
    """`text` with its digits written for the Hindi voice: grouped amounts without their commas, a currency mark as
    "रुपये" after the amount, phone numbers digit by digit, clock times as they are said, "1.5 लाख" as "डेढ़ लाख",
    other decimals with "दशमलव", "%" as "प्रतिशत". Text without digits comes back unchanged."""
    if not text or not any(ch.isdigit() for ch in text):
        return text
    t = unicodedata.normalize("NFC", text).translate(_DEVA_DIGITS)
    # Phone numbers first, so their groups are not read as amounts or times. A grouped amount (1,60,000) never
    # matches: the phone pattern stops at commas.
    t = _PHONE.sub(_phone, t)
    t = _GROUPED.sub(lambda m: m.group(0).replace(",", ""), t)

    def currency(m: re.Match) -> str:
        return f"{m.group(1)} रुपये"

    t = _CURRENCY.sub(currency, t)
    t = _TIME.sub(_time, t)
    t = _HOUR_AMPM.sub(_hour_ampm, t)
    t = _HALF_MULT.sub(_half, t)
    t = _DECIMAL.sub(lambda m: f"{m.group(1)} दशमलव {' '.join(m.group(2))}", t)
    t = _PERCENT.sub(r"\1 प्रतिशत", t)
    return re.sub(r"[ \t]{2,}", " ", t)
