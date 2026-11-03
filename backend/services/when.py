"""Dates, times and slots from what a caller says, worked out in code, never by the language model.

Why: the model was given a clock line and asked to compute "next Monday" or "parso" itself, and on 21 Oct 2026 the
fallback model turned a Monday into a Saturday. Here the model passes the caller's words through to the tool as they
were said ("kal", "agle somvaar", "5 baje", "next Monday 11 am") and `normalise_tool_args` turns them into the
YYYY-MM-DD and HH:MM the appointment and meeting tools expect. Deterministic rules, microseconds, no model call.

Covers English, romanised Hindi/Hinglish and Devanagari:
  * relative days: today/aaj/आज, tomorrow/kal/कल (in a booking "kal" is tomorrow), day after tomorrow/parso/परसों,
    narso (three days on), "in 3 days" / "3 din baad", next week / agle hafte;
  * weekdays in all three scripts, with or without "next"/"agle"/"अगले": the next such day after today; a bare weekday
    may be today when the time asked for has not passed yet;
  * explicit dates: 27 October, 27th Oct, October 27, 27/10, 27-10-2026, 27 अक्टूबर, YYYY-MM-DD (passed through);
    a month-day that has already passed this year means next year;
  * times: 11 am, 11:30 pm, 11.30, 14:30 (passed through), 5 baje, paanch baje, saade paanch (5:30), dhai (2:30),
    dedh (1:30), sawa (quarter past), paune (quarter to), half past ten, quarter to five, ten thirty, noon, ५ बजे.
    Hours without am/pm or a time of day: 8-11 are morning, 12 is noon, 1-7 are afternoon or evening (a clinic or an
    office), and `assumed_meridiem` is set so a read-back can ask. subah/savere/सुबह/morning make it AM; dopahar/sham/
    raat/दोपहर/शाम/रात/afternoon/evening/night make it PM.

Everything is in the machine's local time, like the MCP servers (mcp_servers/appointments.py, meetings.py)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HHMM_RE = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")

# A token boundary that also works for Devanagari (Python's \b does not treat vowel signs as word characters).
_B = r"(?<![\wऀ-ॿ])"
_E = r"(?![\wऀ-ॿ])"

WEEKDAY_WORDS: Dict[str, int] = {
    # English
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3,
    "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
    # Romanised Hindi / Marathi
    "somvaar": 0, "somvar": 0, "somwar": 0, "somwaar": 0,
    "mangalvaar": 1, "mangalvar": 1, "mangalwar": 1, "mangalwaar": 1,
    "budhvaar": 2, "budhvar": 2, "budhwar": 2, "budhwaar": 2,
    "guruvaar": 3, "guruvar": 3, "guruwar": 3, "guruwaar": 3, "brihaspativaar": 3, "brihaspativar": 3, "veervaar": 3,
    "shukravaar": 4, "shukravar": 4, "shukrawar": 4, "shukrawaar": 4,
    "shanivaar": 5, "shanivar": 5, "shaniwar": 5, "shaniwaar": 5, "shanichar": 5,
    "ravivaar": 6, "ravivar": 6, "raviwar": 6, "raviwaar": 6, "itvaar": 6, "itvar": 6, "itwar": 6, "itwaar": 6,
    "etvaar": 6, "etwar": 6,
    # Devanagari
    "सोमवार": 0, "मंगलवार": 1, "बुधवार": 2, "गुरुवार": 3, "बृहस्पतिवार": 3, "शुक्रवार": 4, "शनिवार": 5, "रविवार": 6,
    "इतवार": 6,
}

MONTH_WORDS: Dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5, "june": 6,
    "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    "जनवरी": 1, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9,
    "अक्टूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
}

RELATIVE_DAYS: Dict[str, int] = {
    "today": 0, "tonight": 0, "aaj": 0, "aj": 0, "आज": 0,
    "tomorrow": 1, "tmrw": 1, "kal": 1, "कल": 1,
    "day after tomorrow": 2, "overmorrow": 2,
    "parso": 2, "parson": 2, "parsoon": 2, "parsu": 2, "परसों": 2, "परसो": 2,
    "narso": 3, "narson": 3, "नरसों": 3,
}

NUMBER_WORDS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
    "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "pach": 5, "chhe": 6,
    "che": 6, "chah": 6, "chhah": 6, "saat": 7, "sat": 7, "aath": 8, "ath": 8, "nau": 9, "das": 10, "dus": 10,
    "gyarah": 11, "gyaarah": 11, "baarah": 12, "barah": 12, "bara": 12,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "छे": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10, "ग्यारह": 11, "बारह": 12,
}
MINUTE_WORDS: Dict[str, int] = {"thirty": 30, "fifteen": 15, "fortyfive": 45, "forty-five": 45, "oclock": 0,
                                "o'clock": 0}

AM_WORDS = ("am", "a.m.", "a.m", "subah", "subha", "savere", "sawere", "morning", "सुबह", "सवेरे", "प्रातः")
PM_WORDS = ("pm", "p.m.", "p.m", "dopahar", "dopaher", "dopahr", "sham", "shaam", "shyam", "raat", "rat", "afternoon",
            "evening", "night", "दोपहर", "शाम", "रात", "संध्या")
NEXT_WORDS = ("next", "coming", "agle", "agla", "agli", "aglay", "aane wale", "aane waale", "अगले", "अगला", "अगली", "आने वाले")

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


@dataclass
class Resolved:
    """What was understood. `notes` names assumptions a read-back may want to confirm."""
    date: Optional[date] = None
    time: Optional[time] = None
    assumed_meridiem: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.date is not None or self.time is not None


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).translate(_DEVANAGARI_DIGITS).lower()
    text = text.replace("’", "'").replace("o'clock", "oclock").replace("a.m.", "am").replace("p.m.", "pm")
    text = re.sub(r"(\d)\s*(am|pm)\b", r"\1 \2", text)              # "11am" -> "11 am"
    text = re.sub(r"(\d)\.(\d{2})\b", r"\1:\2", text)              # "11.30" -> "11:30"
    text = re.sub(r"[,;!?।]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_in(words, text: str) -> bool:
    return any(re.search(_B + re.escape(w) + _E, text) for w in words)


# ------------------------------------------------------------------------------------------------------- dates

def _days_until(target_wd: int, today: date, allow_today: bool) -> int:
    d = (target_wd - today.weekday()) % 7
    return 7 if d == 0 and not allow_today else d


def _explicit_date(text: str, today: date) -> Optional[date]:
    """27 october | 27th oct | october 27 | 27/10 | 27-10-2026 | 2026-10-27 | 27 अक्टूबर."""
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", text)
    if m and not re.search(r"\b\d{1,2}:\d{2}", m.group(0)):
        day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
        y = int(year) + (2000 if year and len(year) == 2 else 0) if year else None
        return _month_day(day, month, y, today)
    month_alt = "|".join(sorted(map(re.escape, MONTH_WORDS), key=len, reverse=True))
    m = re.search(_B + r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + month_alt + r")" + _E + r"(?:\s+(\d{4}))?", text)
    if m:
        return _month_day(int(m.group(1)), MONTH_WORDS[m.group(2)], int(m.group(3)) if m.group(3) else None, today)
    m = re.search(_B + r"(" + month_alt + r")\s+(\d{1,2})(?:st|nd|rd|th)?" + _E + r"(?:\s+(\d{4}))?", text)
    if m:
        return _month_day(int(m.group(2)), MONTH_WORDS[m.group(1)], int(m.group(3)) if m.group(3) else None, today)
    return None


def _month_day(day: int, month: int, year: Optional[int], today: date) -> Optional[date]:
    try:
        if year:
            return date(year, month, day)
        d = date(today.year, month, day)
    except ValueError:
        return None
    return d if d >= today else d.replace(year=today.year + 1)


def resolve_date(text: str, now: Optional[datetime] = None, *, allow_today_weekday: bool = False) -> Resolved:
    """The date a phrase names, relative to `now` (default: the current local time). Resolved.date is None when
    nothing date-like was found."""
    now = now or datetime.now()
    today = now.date()
    text = _clean(text)
    out = Resolved()
    if not text:
        return out
    explicit = _explicit_date(text, today)
    if explicit:
        out.date = explicit
        return out
    m = re.search(_B + r"(?:in\s+)?(\d{1,2})\s*(?:days?|din|दिन)(?:\s+(?:baad|bad|later|mein|me|में|बाद))?" + _E, text)
    if m:
        out.date = today + timedelta(days=int(m.group(1)))
        return out
    if re.search(_B + r"(?:next week|agle hafte|agle saptah|agle week|अगले हफ्ते|अगले सप्ताह)" + _E, text):
        out.date = today + timedelta(days=7)
        return out
    for word in sorted(RELATIVE_DAYS, key=len, reverse=True):
        if re.search(_B + re.escape(word) + _E, text):
            out.date = today + timedelta(days=RELATIVE_DAYS[word])
            if word in ("kal", "कल"):
                out.notes.append("kal read as tomorrow")
            return out
    for word in sorted(WEEKDAY_WORDS, key=len, reverse=True):
        if re.search(_B + re.escape(word) + _E, text):
            is_next = _word_in(NEXT_WORDS, text)
            days = _days_until(WEEKDAY_WORDS[word], today, allow_today=allow_today_weekday and not is_next)
            out.date = today + timedelta(days=days)
            if is_next and days <= 2:
                out.notes.append("next <weekday> read as the coming one; the caller may mean the week after")
            return out
    return out


# ------------------------------------------------------------------------------------------------------- times

def _hour_from_words(text: str) -> Optional[int]:
    for w in sorted(NUMBER_WORDS, key=len, reverse=True):
        if re.search(_B + re.escape(w) + _E, text):
            return NUMBER_WORDS[w]
    return None


def resolve_time(text: str) -> Resolved:
    """The time of day a phrase names. Resolved.time is None when nothing time-like was found."""
    text = _clean(text)
    out = Resolved()
    if not text:
        return out
    if re.search(_B + r"(?:noon|midday|dopahar 12|दोपहर 12)" + _E, text) and not re.search(r"\d\s*(?:baje|बजे|:)", text):
        out.time = time(12, 0)
        return out
    if re.search(_B + r"midnight" + _E, text):
        out.time = time(0, 0)
        return out

    hour: Optional[int] = None
    minute = 0
    explicit_24h = False

    m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        explicit_24h = hour > 12 or hour == 0
    else:
        # Hindi fractions first: saade/dhai/dedh/sawa/paune change the hour or the minutes.
        if re.search(_B + r"(?:dedh|डेढ़|डेढ)" + _E, text):
            hour, minute = 1, 30
        elif re.search(_B + r"(?:dhai|dhaai|ढाई)" + _E, text):
            hour, minute = 2, 30
        else:
            m = re.search(r"\b(\d{1,2})\b", text)
            hour = int(m.group(1)) if m else _hour_from_words(text)
            if hour is None:
                return out
            if re.search(_B + r"(?:saade|sadhe|saadhe|sade|half past|साढ़े|साढे)" + _E, text):
                minute = 30
            elif re.search(_B + r"(?:sawa|sava|quarter past|सवा)" + _E, text):
                minute = 15
            elif re.search(_B + r"(?:paune|pone|quarter to|पौने)" + _E, text):
                hour, minute = (hour - 1) if hour > 1 else 12, 45
            else:
                for w, mins in MINUTE_WORDS.items():
                    if re.search(_B + re.escape(w) + _E, text):
                        minute = mins
                        break
                m2 = re.search(r"\b(\d{1,2})\s+(\d{2})\b", text)   # "ten 30", "10 30"
                if m2 and minute == 0 and int(m2.group(1)) == hour and int(m2.group(2)) < 60:
                    minute = int(m2.group(2))
    if hour is None or not (0 <= hour <= 23) or not (0 <= minute <= 59):
        return out

    if explicit_24h or hour > 12:
        out.time = time(hour, minute)
        return out
    if _word_in(AM_WORDS, text):
        hour = 0 if hour == 12 else hour
    elif _word_in(PM_WORDS, text):
        hour = hour if hour == 12 else hour + 12
    else:
        if 1 <= hour <= 7:          # "5 baje" at a clinic or office is the evening
            hour += 12
            out.assumed_meridiem = True
        elif 8 <= hour <= 11:       # "9 baje" is the morning
            out.assumed_meridiem = True
        # 12 stays noon, 0 stays midnight
    out.time = time(hour % 24, minute)
    return out


# ------------------------------------------------------------------------------------------------------- both

def resolve(text: str, now: Optional[datetime] = None) -> Resolved:
    """Date and time from one phrase ("kal subah 10 baje", "next Monday 11 am"). A bare weekday that is today counts
    as today when the time named has not passed yet."""
    now = now or datetime.now()
    t = resolve_time(text)
    allow_today = bool(t.time and t.time > now.time())
    d = resolve_date(text, now, allow_today_weekday=allow_today)
    return Resolved(d.date, t.time, t.assumed_meridiem, d.notes + t.notes)


def normalise_tool_args(args: dict, now: Optional[datetime] = None) -> Tuple[dict, List[str]]:
    """Rewrite `date` and `time` arguments a model passed in the caller's words into YYYY-MM-DD and HH:MM. Values
    already in those shapes are left alone; values nothing can be made of are left alone too (the tool then says what
    shape it wants). Returns the new arguments and the notes (assumptions) made."""
    if not isinstance(args, dict):
        return args, []
    raw_date = args.get("date")
    raw_time = args.get("time")
    date_ok = isinstance(raw_date, str) and bool(ISO_DATE_RE.match(raw_date.strip()))
    time_ok = raw_time is None or (isinstance(raw_time, str) and bool(HHMM_RE.match(raw_time.strip())))
    if (raw_date is None or date_ok) and time_ok:
        return args, []
    now = now or datetime.now()
    phrase = " ".join(str(v) for v in (raw_date, raw_time) if isinstance(v, str) and v.strip())
    res = resolve(phrase, now)
    out = dict(args)
    notes = list(res.notes)
    if raw_date is not None and not date_ok:
        if res.date:
            out["date"] = res.date.isoformat()
            notes.append(f"date '{raw_date}' read as {out['date']}")
        else:
            notes.append(f"date '{raw_date}' not understood")
    if raw_time is not None and not time_ok:
        if res.time:
            out["time"] = res.time.strftime("%H:%M")
            notes.append(f"time '{raw_time}' read as {out['time']}" + (" (am/pm assumed)" if res.assumed_meridiem else ""))
        else:
            notes.append(f"time '{raw_time}' not understood")
    return out, notes
