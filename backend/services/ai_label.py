"""Optional spoken label: a short "AI generated voice" clip put in front of Studio downloads.

Off by default. Set AUDIBLE_AI_LABEL=on to prepend it. The clip is made once with a Kokoro preset voice (English
only), cached in memory and resampled to the clip's sample rate. It is added before the watermark and the receipt,
so both cover the whole file. If Kokoro is not available the audio is left as it is and a warning is logged: a
label that cannot be made never blocks generation.

This is a convenience for people who want an audible disclosure; the inaudible watermark, the signed receipt and the
metadata written into the file are always on."""
from __future__ import annotations

import io
import logging
import os
import threading
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("bonclay.ai_label")

LABEL_TEXT = "AI generated voice."
LABEL_VOICE = "af_heart"
GAP_SECONDS = 0.35  # silence between the label and the speech

_lock = threading.Lock()
_clip: Optional[Tuple[np.ndarray, int]] = None  # (mono float32, sample rate)


def enabled() -> bool:
    return os.getenv("AUDIBLE_AI_LABEL", "off").strip().lower() in ("on", "1", "true", "yes")


def _synthesize() -> Tuple[np.ndarray, int]:
    """Speak LABEL_TEXT with Kokoro. Patched in tests."""
    import soundfile as sf
    try:
        from routers import kokoro
    except ImportError:  # pragma: no cover
        from backend.routers import kokoro
    wav = kokoro._synthesize(kokoro._VOICE_MAP[LABEL_VOICE], LABEL_TEXT, 1.0)
    audio, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)


def _label_clip() -> Optional[Tuple[np.ndarray, int]]:
    global _clip
    if _clip is not None:
        return _clip
    with _lock:
        if _clip is None:
            try:
                _clip = _synthesize()
            except Exception as e:  # noqa: BLE001 - Kokoro or espeak missing, offline model download ...
                log.warning("AUDIBLE_AI_LABEL is on but the spoken label could not be made: %s", e)
                return None
        return _clip


def prepend_label(wav_bytes: bytes) -> Tuple[bytes, float]:
    """(WAV with the spoken label in front, seconds added). Unchanged audio and 0.0 when the label is off or cannot
    be made. Blocking (the first call loads Kokoro): call it with asyncio.to_thread."""
    if not enabled():
        return wav_bytes, 0.0
    clip = _label_clip()
    if clip is None:
        return wav_bytes, 0.0
    try:
        import soundfile as sf
        try:
            from services.provenance import _resample
        except ImportError:  # pragma: no cover
            from backend.services.provenance import _resample
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim != 1:
            return wav_bytes, 0.0  # only mono clips (everything BonClay makes)
        label = _resample(clip[0], clip[1], int(sr))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0 and float(np.max(np.abs(label))) > 0:
            label = label * min(1.0, peak / float(np.max(np.abs(label))))  # no louder than the speech
        gap = np.zeros(int(GAP_SECONDS * int(sr)), dtype=np.float32)
        joined = np.concatenate([label.astype(np.float32), gap, audio.astype(np.float32)])
        buf = io.BytesIO()
        sf.write(buf, joined, int(sr), format="WAV")
        return buf.getvalue(), (label.size + gap.size) / float(sr)
    except Exception as e:  # noqa: BLE001
        log.warning("could not prepend the spoken AI label: %s", e)
        return wav_bytes, 0.0
