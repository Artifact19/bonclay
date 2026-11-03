"""Phone number helpers shared by every provider: E.164 normalisation and masking."""
from __future__ import annotations

import re
from typing import Optional

E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")


def mask(number: str) -> str:
    s = re.sub(r"\D", "", str(number or ""))
    return s if len(s) <= 6 else f"{s[:2]}{'*' * (len(s) - 6)}{s[-4:]}"


def normalise_number(raw: str) -> Optional[str]:
    """A phone number as E.164 (+919876543210), or None when it is not a valid number. Spaces, dashes, dots and
    brackets are ignored. Indian numbers are accepted as 9876543210, 09876543210, 919876543210 or +919876543210."""
    s = re.sub(r"[\s\-().]", "", str(raw or ""))
    if not s:
        return None
    if s.startswith("+"):
        out = "+" + s[1:]
    else:
        digits = s[2:] if s.startswith("00") else s
        if not digits.isdigit():
            return None
        if s.startswith("00"):
            out = "+" + digits
        elif len(digits) == 10:
            out = "+91" + digits
        elif len(digits) == 11 and digits.startswith("0"):
            out = "+91" + digits[1:]
        elif len(digits) == 12 and digits.startswith("91"):
            out = "+" + digits
        else:
            return None
    return out if E164_RE.match(out) else None


def mask_e164(number: str) -> str:
    """+919876543210 -> +91•••••43210 (the country code and the last five digits stay)."""
    n = str(number or "")
    return n if len(n) <= 8 else f"{n[:3]}{'•' * (len(n) - 8)}{n[-5:]}"


def mask_caller_id(raw: str) -> Optional[str]:
    """A provider's caller id (a number) for the status route: E.164 masked when it is a valid number, else digits masked."""
    if not raw:
        return None
    n = normalise_number(raw)
    return mask_e164(n) if n else (mask(raw) or None)
