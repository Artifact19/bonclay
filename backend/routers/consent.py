"""Fresh consent challenges.

`GET /api/consent/challenge?name=...` issues a sentence for the person to read aloud when they clone a voice. It
contains their name, today's date and a random code word and number, so an old recording of a previous consent
sentence cannot be replayed. `POST /api/voices/clone` accepts the returned `challenge_id` and verifies the
recording against the stored text instead of text sent by the client.

Challenges live in memory (one process), expire after 15 minutes and can be used once. They belong to the user
that requested them."""
import re
import secrets
import threading
import time
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Query

try:
    from services.auth import get_current_user_id
except ImportError:  # pragma: no cover
    from backend.services.auth import get_current_user_id
try:
    from core.errors import AppError
    from services import consent_record
except ImportError:  # pragma: no cover
    from backend.core.errors import AppError
    from backend.services import consent_record

router = APIRouter(tags=["consent"])

CHALLENGE_TTL_SECONDS = 15 * 60
MAX_CHALLENGES = 2000  # hard cap so a client asking in a loop cannot grow memory without bound
# Easy to say and easy for speech recognition to spell.
CODE_WORDS = ("mango", "river", "tiger", "blue", "seven", "orange", "garden", "silver", "window", "pencil",
              "yellow", "candle", "ocean", "maple", "planet", "button", "lemon", "rabbit", "cloud", "green")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")

_lock = threading.Lock()
_challenges: Dict[str, dict] = {}  # id -> {text, user_id, expires_at, state: "open" | "in_use" | "used"}


def _spoken_name(name: str) -> str:
    """The name as it will be read aloud: letters, digits, spaces and . ' - only, at most 60 characters."""
    cleaned = re.sub(r"[^\w .'\-]", " ", name or "", flags=re.UNICODE).replace("_", " ")
    cleaned = " ".join(cleaned.split())[:60].strip()
    return cleaned or "the speaker"


def _date_text(now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    return f"{now.day} {MONTHS[now.month - 1]} {now.year}"


def _purge(now: float) -> None:
    for cid in [c for c, v in _challenges.items() if v["expires_at"] <= now]:
        del _challenges[cid]
    while len(_challenges) >= MAX_CHALLENGES:
        del _challenges[next(iter(_challenges))]  # oldest first (dicts keep insertion order)


def issue(user_id: str, name: str) -> dict:
    text = (f"I, {_spoken_name(name)}, consent to creating an AI version of my voice with BonClay on "
            f"{_date_text()}. My code is {secrets.choice(CODE_WORDS)} {secrets.randbelow(90) + 10}.")
    now = time.time()
    challenge_id = secrets.token_urlsafe(16)
    expires_at = now + CHALLENGE_TTL_SECONDS
    with _lock:
        _purge(now)
        _challenges[challenge_id] = {"text": text, "user_id": user_id, "expires_at": expires_at, "state": "open"}
    return {"challenge_id": challenge_id, "text": text,
            "expires_at": datetime.fromtimestamp(expires_at).astimezone().isoformat(timespec="seconds")}


def _invalid(reason: str) -> AppError:
    return AppError("CONSENT_CHALLENGE_INVALID",
                    f"This consent sentence {reason}. Request a new one and read that instead.", 422,
                    {"reason": reason})


def claim(challenge_id: str, user_id: str) -> str:
    """Reserve an open challenge for one clone attempt and return its text.

    Raises CONSENT_CHALLENGE_INVALID (422) if the id is unknown, belongs to someone else, has expired or has been
    used. A second request for the same challenge while the first is running is refused too. Call `finish` when
    the attempt ends."""
    with _lock:
        item = _challenges.get(challenge_id)
        if item is None or item["user_id"] != user_id:
            raise _invalid("is not recognised")
        if item["expires_at"] <= time.time():
            del _challenges[challenge_id]
            raise _invalid("has expired")
        if item["state"] != "open":
            raise _invalid("has already been used")
        item["state"] = "in_use"
        return item["text"]


def finish(challenge_id: str, success: bool) -> None:
    """End a clone attempt: a successful one uses the challenge up, a failed one lets it be tried again."""
    with _lock:
        item = _challenges.get(challenge_id)
        if item is not None:
            item["state"] = "used" if success else "open"


@router.get("/api/consent/challenge")
async def consent_challenge(name: str = Query("", max_length=200), user_id: str = Depends(get_current_user_id)):
    """A fresh sentence to read aloud as consent: {"challenge_id", "text", "expires_at"} (valid 15 minutes, one use).

    Also carries what the plain-language notice needs: `notice_version` (stored in the consent record), how long the
    voice is kept (`retention_days`) and who to contact (`contact`)."""
    return {**issue(user_id, name), "notice_version": consent_record.NOTICE_VERSION,
            "retention_days": consent_record.retention_days(), "contact": consent_record.contact()}
