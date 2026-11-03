"""Voicemail / answering-machine detection for outbound calls, from what the first thing the callee says sounds like.

Conservative on purpose: hanging up on a person is worse than talking to a machine, so a phrase that a person could also
say ("not available") only counts together with a long first utterance, and very long speech alone is only trusted at
VOICEMAIL_LONG_MS (default 15 s). Needs the transcript of the first utterance (the call already transcribes it) and its
length. It looks for English, Hindi and Hinglish greetings. Machines that play only a beep or a tone are not detected."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

# A person would almost never say these to a caller: enough on their own.
STRONG = (
    "leave a message", "leave your message", "leave me a message", "after the beep", "after the tone", "voicemail",
    "voice mail", "mailbox", "record your message", "unable to take your call", "cannot take your call",
    "can't take your call", "not able to take your call", "the person you are calling", "the number you have dialed",
    "the number you have dialled", "the number you are trying", "currently switched off", "is switched off",
    "out of coverage area", "not reachable", "सन्देश छोड़", "संदेश छोड़", "मैसेज छोड़", "बीप", "स्विच ऑफ", "पहुंच से बाहर",
    "पहुँच से बाहर", "डायल किया गया नंबर", "डायल किया गया नम्बर", "आपके द्वारा डायल",
)
# A person could say these too: they count only when the callee spoke for a while.
WEAK = ("not available", "currently unavailable", "unavailable right now", "उपलब्ध नहीं", "अभी उपलब्ध नहीं",
        "please leave", "कृपया प्रतीक्षा", "please hold")
WEAK_MIN_MS = 4000


def _norm(text: str) -> str:
    return re.sub(r"[\s]+", " ", re.sub(r"[.,!?;:\"'()\-–—…]", " ", str(text or "").lower())).strip()


def long_ms() -> int:
    try:
        return int(os.getenv("VOICEMAIL_LONG_MS", "15000"))
    except ValueError:
        return 15000


def enabled() -> bool:
    return os.getenv("VOICEMAIL_DETECT", "on").strip().lower() not in ("off", "0", "false", "no")


def classify(text: Optional[str], speech_ms: int = 0) -> Dict[str, Any]:
    """{"voicemail": bool, "reason": "phrase" | "long_speech" | None, "phrase": str | None}."""
    t = _norm(text or "")
    for p in STRONG:
        if _norm(p) in t:
            return {"voicemail": True, "reason": "phrase", "phrase": p}
    if speech_ms >= WEAK_MIN_MS:
        for p in WEAK:
            if _norm(p) in t:
                return {"voicemail": True, "reason": "phrase", "phrase": p}
    if speech_ms >= long_ms():
        return {"voicemail": True, "reason": "long_speech", "phrase": None}
    return {"voicemail": False, "reason": None, "phrase": None}
