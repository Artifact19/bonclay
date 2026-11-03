"""Numbers to spoken words (Hindi, Marathi) for engines that read characters.

Measured 2026-10-20 (docs/INDIC_EXPERIMENTS.md, experiment A): DhVaani spoke the right number in 0 of 25 sentences when the
text held digits ("फीस 450 रुपये" came out as syllable noise) and in 19 of 25 with number words. So digits, amounts, times,
dates, percentages and phone numbers are rewritten as words before the text reaches the engine. Deterministic rules, no
model call, microseconds. The text shown and stored is never touched.

Covered: ASCII and native (Devanagari) digits, Indian and Western thousands separators (1,23,456), whole numbers up to
99,99,99,999 (lakh, crore), decimals, rupee amounts (₹450, Rs. 450, रु. 450), percentages, h:mm times, dd/mm/yyyy dates,
10-digit phone numbers and PIN codes (digit by digit). Not covered: other languages (Tamil, Telugu, Bengali, Kannada,
Gujarati, Malayalam, Punjabi return the text unchanged), ordinals, fractions, ranges. The word tables below were written
without a native-speaker review; Marathi has spelling variants (see tests for the forms used).

    from services import numwords
    numwords.apply("फीस ₹450 है", "hi", "dhvaani")   # 'फीस चार सौ पचास रुपये है'
    NUMWORDS=off disables it.
"""
import os
import re
from typing import Callable, Dict, List, Optional

ENGINES = frozenset({"dhvaani"})  # engines measured to garble digits; others are not touched

_HI_0_99 = """शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस
बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस
चालीस इकतालीस बयालीस तैंतालीस चौंतालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ
साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी
अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे""".split()

_MR_0_99 = """शून्य एक दोन तीन चार पाच सहा सात आठ नऊ दहा अकरा बारा तेरा चौदा पंधरा सोळा सतरा अठरा एकोणीस
वीस एकवीस बावीस तेवीस चोवीस पंचवीस सव्वीस सत्तावीस अठ्ठावीस एकोणतीस तीस एकतीस बत्तीस तेहेतीस चौतीस पस्तीस छत्तीस सदतीस अडतीस एकोणचाळीस
चाळीस एकेचाळीस बेचाळीस त्रेचाळीस चव्वेचाळीस पंचेचाळीस सेहेचाळीस सत्तेचाळीस अठ्ठेचाळीस एकोणपन्नास पन्नास एकावन्न बावन्न त्रेपन्न चौपन्न पंचावन्न छप्पन्न सत्तावन्न अठ्ठावन्न एकोणसाठ
साठ एकसष्ट बासष्ट त्रेसष्ट चौसष्ट पासष्ट सहासष्ट सदुसष्ट अडुसष्ट एकोणसत्तर सत्तर एक्काहत्तर बाहत्तर त्र्याहत्तर चौऱ्याहत्तर पंचाहत्तर शहात्तर सत्त्याहत्तर अठ्ठ्याहत्तर एकोणऐंशी
ऐंशी एक्याऐंशी ब्याऐंशी त्र्याऐंशी चौऱ्याऐंशी पंच्याऐंशी शहाऐंशी सत्त्याऐंशी अठ्ठ्याऐंशी एकोणनव्वद नव्वद एक्याण्णव ब्याण्णव त्र्याण्णव चौऱ्याण्णव पंच्याण्णव शहाण्णव सत्त्याण्णव अठ्ठ्याण्णव नव्व्याण्णव""".split()

assert len(_HI_0_99) == 100 and len(_MR_0_99) == 100

_LANG = {
    "hi": {
        "n": _HI_0_99, "hundred": lambda d: f"{_HI_0_99[d]} सौ", "thousand": "हज़ार", "lakh": "लाख", "crore": "करोड़",
        "rupees": "रुपये", "percent": "प्रतिशत", "point": "दशमलव",
        "months": "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split(),
        "time": lambda h, m: f"{{h}} बजे" if m == 0 else f"{{h}} बजकर {{m}} मिनट",
    },
    "mr": {
        "n": _MR_0_99, "hundred": lambda d: "शंभर" if d == 1 else f"{_MR_0_99[d]}शे", "thousand": "हजार", "lakh": "लाख",
        "crore": "कोटी", "rupees": "रुपये", "percent": "टक्के", "point": "पूर्णांक",
        "months": "जानेवारी फेब्रुवारी मार्च एप्रिल मे जून जुलै ऑगस्ट सप्टेंबर ऑक्टोबर नोव्हेंबर डिसेंबर".split(),
        "time": lambda h, m: f"{{h}} वाजता" if m == 0 else f"{{h}} वाजून {{m}} मिनिटे",
    },
}

SUPPORTED = frozenset(_LANG)
_NATIVE = str.maketrans("०१२३४५६७८९", "0123456789")


def _under_100(lang: dict, n: int) -> str:
    return lang["n"][n]


def _under_1000(lang: dict, n: int) -> str:
    h, rest = divmod(n, 100)
    parts = []
    if h:
        parts.append(lang["hundred"](h))
    if rest or not h:
        parts.append(_under_100(lang, rest))
    return " ".join(parts)


def number_to_words(n: int, language: str) -> str:
    """0 to 99,99,99,999 in the Indian grouping. Larger numbers are read digit by digit."""
    lang = _LANG[language]
    if n < 0:
        raise ValueError("negative")
    if n >= 10 ** 11:
        return digits_to_words(str(n), language)
    if n == 0:
        return lang["n"][0]
    parts: List[str] = []
    crore, n = divmod(n, 10 ** 7)
    lakh, n = divmod(n, 10 ** 5)
    thousand, n = divmod(n, 1000)
    if crore:
        parts.append(f"{number_to_words(crore, language)} {lang['crore']}")
    if lakh:
        parts.append(f"{_under_100(lang, lakh)} {lang['lakh']}")
    if thousand:
        parts.append(f"{_under_100(lang, thousand)} {lang['thousand']}")
    if n:
        parts.append(_under_1000(lang, n))
    return " ".join(parts)


def digits_to_words(digits: str, language: str) -> str:
    lang = _LANG[language]
    return " ".join(lang["n"][int(c)] for c in digits if c.isdigit())


def _decimal(whole: str, frac: str, language: str) -> str:
    lang = _LANG[language]
    return f"{number_to_words(int(whole), language)} {lang['point']} {digits_to_words(frac, language)}"


_NUM = r"\d{1,3}(?:,\d{2,3})+|\d+"
_RUPEE = re.compile(rf"(?:₹|Rs\.?|INR|रु\.?)\s*({_NUM})(?:\.(\d+))?", re.I)
_DATE = re.compile(r"(?<![\d/.-])(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?![\d/.-])")
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")
_PERCENT = re.compile(rf"({_NUM})(?:\.(\d+))?\s*%")
_PHONE = re.compile(r"(?<![\d])(?:\+?91[\s-]?)?([6-9]\d{4}[\s-]?\d{5})(?![\d])")
_PIN = re.compile(r"((?:पिन|पिनकोड|pin(?:\s*code)?)\s*(?:कोड|क्रमांक|:)?\s*)(\d{6})(?!\d)", re.I)
_PLAIN = re.compile(rf"(?<![\d.])({_NUM})(?:\.(\d+))?(?![\d])")


def _clean_int(s: str) -> int:
    return int(s.replace(",", ""))


def normalize(text: str, language: str) -> str:
    """Rewrite numbers in `text` as words. Returns the text unchanged for languages that are not supported."""
    code = (language or "").strip().lower().split("-")[0]
    if code not in SUPPORTED or not text:
        return text
    lang = _LANG[code]
    text = text.translate(_NATIVE)
    if not re.search(r"\d", text):
        return text

    def rupees(m):
        amt = _decimal(m.group(1).replace(",", ""), m.group(2), code) if m.group(2) else number_to_words(_clean_int(m.group(1)), code)
        return f"{amt} {lang['rupees']}"

    def date(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        return f"{number_to_words(d, code)} {lang['months'][mo - 1]} {number_to_words(y, code)}"

    def time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 23 or mi > 59:
            return m.group(0)
        h12 = h % 12 or 12
        return lang["time"](h12, mi).format(h=number_to_words(h12, code), m=number_to_words(mi, code))

    def percent(m):
        n = _decimal(m.group(1).replace(",", ""), m.group(2), code) if m.group(2) else number_to_words(_clean_int(m.group(1)), code)
        return f"{n} {lang['percent']}"

    def phone(m):
        return digits_to_words(re.sub(r"\D", "", m.group(1)), code)

    def pin(m):
        return m.group(1) + digits_to_words(m.group(2), code)

    def plain(m):
        if m.group(2):
            return _decimal(m.group(1).replace(",", ""), m.group(2), code)
        return number_to_words(_clean_int(m.group(1)), code)

    # order matters: the most specific patterns first, plain numbers last
    for pat, fn in ((_PIN, pin), (_RUPEE, rupees), (_DATE, date), (_TIME, time), (_PERCENT, percent), (_PHONE, phone),
                    (_PLAIN, plain)):
        text = pat.sub(fn, text)
    return text


def enabled() -> bool:
    return os.getenv("NUMWORDS", "").strip().lower() not in ("off", "0", "false", "no")


def apply(text: str, language: Optional[str], engine: Optional[str]) -> str:
    """The text to hand to `engine`: numbers as words for the engines in ENGINES, otherwise unchanged."""
    if not text or not enabled() or (engine or "").strip().lower() not in ENGINES:
        return text
    return normalize(text, language or "")
