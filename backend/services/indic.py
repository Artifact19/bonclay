"""Indic language helpers: detection, transliteration, resampling."""
from typing import Dict

INDIC_LANGS = ["hi", "mr", "bn", "gu", "kn", "ml", "or", "pa", "ta", "te", "as"]

# Unicode blocks. Devanagari (hi/mr) and Bengali (bn/as) are shared scripts.
SCRIPT_RANGES = [
    (0x0900, 0x097F, "deva"), (0x0980, 0x09FF, "beng"), (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"), (0x0B00, 0x0B7F, "or"), (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"), (0x0C80, 0x0CFF, "kn"), (0x0D00, 0x0D7F, "ml"),
]

SAMPLES: Dict[str, str] = {
    "hi": "नमस्ते, आप कैसे हैं?", "mr": "नमस्कार, तुम्ही कसे आहात?",
    "bn": "নমস্কার, আপনি কেমন আছেন?", "gu": "નમસ્તે, તમે કેમ છો?",
    "kn": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?", "ml": "നമസ്കാരം, സുഖമാണോ?",
    "or": "ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?", "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
    "ta": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "te": "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    "as": "নমস্কাৰ, আপুনি কেনে আছে?",
}


def script_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ch in text:
        o = ord(ch)
        for lo, hi, name in SCRIPT_RANGES:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    return counts


def detect_language(text: str) -> dict:
    """Return {language, confidence, method}. Script first; langdetect disambiguates/falls back."""
    text = (text or "").strip()
    if not text:
        return {"language": None, "confidence": 0.0, "method": "empty"}
    counts = script_counts(text)
    ld_code, ld_conf = None, 0.0
    try:
        from langdetect import detect_langs
        res = detect_langs(text)
        if res:
            ld_code, ld_conf = res[0].lang, float(res[0].prob)
    except Exception:
        pass
    if counts:
        script = max(counts, key=counts.get)
        share = counts[script] / max(1, sum(1 for c in text if not c.isspace()))
        if script == "deva":
            lang = "mr" if ld_code == "mr" else "hi"
        elif script == "beng":
            lang = "as" if (ld_code == "as" or any(c in text for c in "ৰৱ")) else "bn"
        else:
            lang = script
        return {"language": lang, "confidence": round(max(share, ld_conf if ld_code == lang else 0.0), 3),
                "method": "script+langdetect" if ld_code else "script"}
    if ld_code:
        return {"language": ld_code, "confidence": round(ld_conf, 3), "method": "langdetect"}
    return {"language": "en" if text.isascii() else None, "confidence": 0.3, "method": "heuristic"}


def resample_to_24k(wav, sr: int):
    """Resample a mono waveform to 24 kHz using librosa. Returns (wav, 24000)."""
    if sr == 24000:
        return wav, 24000
    import librosa
    import numpy as np
    arr = np.asarray(wav, dtype="float32")
    return librosa.resample(arr, orig_sr=sr, target_sr=24000), 24000


def transliterate_to_devanagari(text: str) -> dict:
    """ITRANS/HK -> Devanagari via indic_transliteration; basic ITRANS fallback otherwise."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        scheme = sanscript.HK if _looks_hk(text) else sanscript.ITRANS
        return {"text": transliterate(text, scheme, sanscript.DEVANAGARI),
                "scheme": scheme, "engine": "indic_transliteration"}
    except Exception:
        return {"text": _fallback_itrans(text), "scheme": "itrans", "engine": "fallback"}


def _looks_hk(text: str) -> bool:
    if "~" in text or "^" in text:
        return False
    return any(t in text for t in ("lR", "jJ", "RR")) or any(c in text for c in "zS")


_VOW = {"aa": "आ", "ii": "ई", "uu": "ऊ", "ai": "ऐ", "au": "औ", "a": "अ", "i": "इ", "u": "उ", "e": "ए", "o": "ओ"}
_VOW_M = {"aa": "ा", "ii": "ी", "uu": "ू", "ai": "ै", "au": "ौ", "a": "", "i": "ि", "u": "ु", "e": "े", "o": "ो"}
_CON = {"chh": "छ", "kh": "ख", "gh": "घ", "ch": "च", "jh": "झ", "Th": "ठ", "Dh": "ढ", "th": "थ", "dh": "ध",
        "ph": "फ", "bh": "भ", "sh": "श", "k": "क", "g": "ग", "j": "ज", "T": "ट", "D": "ड", "N": "ण",
        "t": "त", "d": "द", "n": "न", "p": "प", "b": "ब", "m": "म", "y": "य", "r": "र", "l": "ल",
        "v": "व", "w": "व", "s": "स", "h": "ह", "f": "फ", "z": "ज", "q": "क", "x": "क्स"}
_VKEYS = sorted(_VOW, key=len, reverse=True)
_CKEYS = sorted(_CON, key=len, reverse=True)


def _fallback_itrans(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        c = next((k for k in _CKEYS if text.startswith(k, i)), None)
        if c:
            i += len(c)
            v = next((k for k in _VKEYS if text.startswith(k, i)), None)
            if v:
                i += len(v)
                out.append(_CON[c] + _VOW_M[v])
            else:
                out.append(_CON[c] + ("्" if text[i:i + 1].isalpha() else ""))
            continue
        v = next((k for k in _VKEYS if text.startswith(k, i)), None)
        if v:
            out.append(_VOW[v])
            i += len(v)
            continue
        out.append(text[i])
        i += 1
    return "".join(out)
