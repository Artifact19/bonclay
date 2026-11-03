"""Licences of the speech engines, and the commercial mode that hides the ones not cleared for paid use.

The live call path (Kokoro, Qwen3-TTS, Chatterbox) is licence-clean. Two Studio engines are not: DhVaani-0.5 is
CC-BY-NC-4.0, and Indic-Speak's Indic Open Model Licence needs Bodhan AI's written approval before the model is
offered as a hosted service and bans robocalls. IndicF5 carries an MIT label on a CC-BY-NC F5-TTS base.

BONCLAY_COMMERCIAL=1 (commercial mode) removes those engines from the engine list, refuses to load them, routes
Indian-language speech to Kokoro and Hindi cloning to Chatterbox, and every receipt names its engine's licence
(`engine_licence`), so a paid deployment cannot ship a non-commercial clip by accident. Licences read on the model
cards on 22 Oct 2026; this is a product safeguard, not legal advice."""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

# engine key -> (licence, cleared for paid use, caveat)
LICENCES: Dict[str, tuple] = {
    "kokoro": ("Apache-2.0", True, "StyleTTS2 asks that listeners be told the speech is synthetic (BonClay does)"),
    "qwen3_0_6b": ("Apache-2.0", True, None),
    "chatterbox_ml": ("MIT", True, "keep Resemble's Perth watermark"),
    "dhvaani": ("CC-BY-NC-4.0", False,
                "non-commercial licence; paid use needs written permission from ARTPARK-IISc"),
    "indic_speak": ("Indic Open Model Licence v1.0", False,
                    "offering it as a hosted service needs Bodhan AI's written approval; robocalls are banned"),
    "indic_f5": ("MIT label on a CC-BY-NC-4.0 F5-TTS base (unclear)", False, "lineage unclear; treat as non-commercial"),
    "audioseal_16bits": ("MIT", True, None),
}

ERROR_CODE = "ENGINE_LICENCE"


def commercial_mode() -> bool:
    """BONCLAY_COMMERCIAL=1|on|true|yes: only engines cleared for paid use are offered."""
    return os.getenv("BONCLAY_COMMERCIAL", "").strip().lower() in ("1", "on", "true", "yes")


def licence_of(engine: Optional[str]) -> Optional[str]:
    row = LICENCES.get(engine or "")
    return row[0] if row else None


def paid_use_ok(engine: Optional[str]) -> bool:
    """True for engines cleared for paid use; unknown engines count as cleared (they are not voice models we ship)."""
    row = LICENCES.get(engine or "")
    return True if row is None else bool(row[1])


def caveat_of(engine: Optional[str]) -> Optional[str]:
    row = LICENCES.get(engine or "")
    return row[2] if row else None


def blocked_reason(engine: Optional[str]) -> Optional[str]:
    """Why `engine` may not be used right now, or None. Only commercial mode blocks anything."""
    if not commercial_mode() or paid_use_ok(engine):
        return None
    return (f"{engine} is not available in commercial mode (BONCLAY_COMMERCIAL=1): its licence is "
            f"{licence_of(engine)}; {caveat_of(engine)}.")


def allowed(engines: Iterable[str]) -> List[str]:
    return [e for e in engines if blocked_reason(e) is None]


def describe(engine: str) -> dict:
    """{"licence", "paid_use", "caveat"} for an engine list or a status page."""
    return {"licence": licence_of(engine), "paid_use": paid_use_ok(engine), "caveat": caveat_of(engine)}
