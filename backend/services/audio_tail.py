"""Cut the noise some voice-cloning engines add after the last word.

DhVaani and Chatterbox pick a clip length before they speak. When the real speech is shorter, the spare seconds are
filled with babble. Speech in these languages runs at about 8-12 characters a second, so a clip that is much longer
than its text allows, and that has a pause before its last stretch, is cut at that pause. It does nothing when the clip
is not suspiciously long, when it has no usable pause, or when the cut would remove more than a third of it."""
from typing import Optional

import numpy as np

SLOWEST_CPS = 8.0      # slowest plausible speech (characters without spaces per second); slower means a tail
FASTEST_CPS = 12.5     # fastest plausible speech: the cut never lands before the text could have been spoken
SLACK_S = 0.6          # allowance on top of the slowest estimate
MIN_GAP_S = 0.25       # a pause this long, below SILENCE_FRACTION of the loudest frame, ends the speech
SILENCE_FRACTION = 0.06
MAX_CUT_FRACTION = 0.34
KEEP_S = 0.12          # a little of the pause is kept so the last word does not end abruptly
FADE_S = 0.06
FRAME_S = 0.02


def _chars(text: str) -> int:
    return sum(1 for ch in (text or "") if not ch.isspace() and ch not in ".,!?;:।'\"-–—…")


def trim_tail(audio: np.ndarray, sr: int, text: str) -> np.ndarray:
    """`audio` without a babble tail, or `audio` itself when there is nothing to cut."""
    n = _chars(text)
    if n < 12 or audio is None or audio.size == 0 or sr <= 0:
        return audio
    total = audio.size / sr
    if total <= n / SLOWEST_CPS + SLACK_S:
        return audio
    cut = _cut_point(np.asarray(audio, dtype=np.float32), sr, earliest=n / FASTEST_CPS)
    if cut is None or (total - cut) / total > MAX_CUT_FRACTION:
        return audio
    end = min(audio.size, int((cut + KEEP_S) * sr))
    out = np.array(audio[:end], dtype=audio.dtype, copy=True)
    fade = min(out.size, int(FADE_S * sr))
    if fade > 1:
        out[-fade:] = out[-fade:] * np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out


def _cut_point(audio: np.ndarray, sr: int, earliest: float) -> Optional[float]:
    """Start (seconds) of the last pause that begins after `earliest` and is followed by more sound."""
    fr = max(1, int(FRAME_S * sr))
    frames = audio[: audio.size // fr * fr].reshape(-1, fr)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    if rms.size == 0 or rms.max() <= 0:
        return None
    quiet = rms < rms.max() * SILENCE_FRACTION
    need = int(round(MIN_GAP_S / FRAME_S))
    best = None
    i = 0
    while i < quiet.size:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < quiet.size and quiet[j]:
            j += 1
        start = i * FRAME_S
        if j - i >= need and j < quiet.size and start >= earliest:  # followed by more sound: a tail, not the end
            best = start
        i = j
    return best
