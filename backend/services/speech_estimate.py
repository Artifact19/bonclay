"""How long speaking a sentence will take on this machine, and how far along a running clip is.

Nothing here runs a model. It turns two measured quantities into a time:

* the real-time factor (RTF) of each engine: seconds of generation per second of audio. The starting values are what
  was measured on this laptop (docs/RESULTS.md and docs/decisions/indian-language-cloning.md); every finished clip
  then updates a running average, kept in a tiny JSON file, so the estimate gets better the more it is used;
* how long the audio will be: the characters of the sentence divided by a speaking rate for its script (also learned
  from finished clips).

`Estimator.voice_progress` turns "seconds spent so far" into a 0..1 fraction that rises smoothly, slows down if the
estimate was too low, and never reaches 1: the caller shows it as at most 95 % until the clip is really finished.
"""
import json
import logging
import math
import os
import threading
from typing import Dict, Optional, Tuple

log = logging.getLogger("bonclay.speech_estimate")

# Generation seconds per audio second, measured on this Apple M4 (see the module docstring for the sources).
DEFAULT_RTF: Dict[str, float] = {
    "kokoro": 0.18,        # Kokoro MLX, 0.18 over 20 read-back sentences
    "qwen3_0_6b": 1.7,     # cloned voice, 1.69
    "chatterbox_ml": 5.0,  # CPU, 3.9 to 5.6
    "dhvaani": 2.0,        # Apple GPU, 16 steps: 1.2 to 1.5 on a quiet M4 (bench/dhvaani_live_check.py), 1.3 to 3.1 under load
    "indic_f5": 7.0,
    "indic_speak": 7.0,    # 6.96 over 20 sentences
}
# Never estimate less than this for the voice stage: worker start-up and the watermark/save step cost this much.
MIN_VOICE_S: Dict[str, float] = {"kokoro": 1.0, "qwen3_0_6b": 3.0, "dhvaani": 3.0, "chatterbox_ml": 6.0, "indic_f5": 8.0,
                                 "indic_speak": 8.0}
# Extra seconds the first clip of an engine costs when its model is not loaded yet (worker start-up, weights into
# memory). Indic-Speak: 15.9 s until its worker is ready, and its first request took 62.9 s for two short sentences.
# DhVaani: 4.1 s until its worker is ready and about 3 s more for the first request (warm-up), 8.1 s in all on a quiet M4.
DEFAULT_LOAD_S: Dict[str, float] = {"kokoro": 3.0, "qwen3_0_6b": 6.0, "dhvaani": 10.0, "chatterbox_ml": 12.0,
                                    "indic_f5": 15.0, "indic_speak": 25.0}
WARM_WINDOW_S = 300.0  # the model manager unloads a model that has been idle for 300 s (backend.py, AdvancedModelManager)
OVERRUN_FACTOR = 1.25  # a voice stage this far past its estimate is "taking longer than expected"
DEFAULT_TRANSLATE_S = 4.0  # one short sentence through the local model (Groq answers in about a second)
FALLBACK_RTF = 5.0

# Characters spoken per second of audio, by language. Indic scripts measured about 12 (docs/RESULTS.md); Latin text
# about 14; ideographs carry more per character.
DEFAULT_CPS: Dict[str, float] = {"ja": 6.0, "zh": 4.5, "ko": 8.0}
_INDIC = {"hi", "mr", "ta", "bn", "te", "gu", "kn", "ml", "pa", "or", "ur", "as", "ne"}
LATIN_CPS = 14.0
INDIC_CPS = 12.0

# The progress bands of one language (percent of that language's bar).
TRANSLATE_END = 15.0
VOICE_END = 95.0   # the voice stage never shows more than this until the clip is really saved
EMA_ALPHA = 0.3
MAX_SAMPLE_WEIGHT = 4  # the first samples are averaged; after that a sample moves the average by EMA_ALPHA


def _default_path() -> str:
    try:
        from services.common import DATA_DIR
    except ImportError:  # pragma: no cover
        from backend.services.common import DATA_DIR
    return os.path.join(DATA_DIR, "say_it_once_rtf.json")


class Estimator:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._lock = threading.Lock()
        self._rtf: Dict[str, Tuple[float, int]] = {}    # engine -> (average RTF, samples)
        self._cps: Dict[str, Tuple[float, int]] = {}    # language -> (average characters per second, samples)
        self._load: Dict[str, Tuple[float, int]] = {}   # engine -> (average cold-start seconds, samples)
        self._translate: Tuple[float, int] = (DEFAULT_TRANSLATE_S, 0)
        self._last_used: Dict[str, float] = {}          # engine -> when it last finished a clip (this process only)
        self._loaded = False

    # ------------------------------------------------------------------ persistence (best effort, never raises)
    def _file(self) -> str:
        return self.path or _default_path()

    def _load_state(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(self._file(), "r", encoding="utf-8") as f:
                raw = json.load(f)
            for key, store in (("rtf", self._rtf), ("cps", self._cps), ("load", self._load)):
                for name, pair in (raw.get(key) or {}).items():
                    store[name] = (float(pair[0]), int(pair[1]))
            if raw.get("translate"):
                self._translate = (float(raw["translate"][0]), int(raw["translate"][1]))
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001 - a damaged file only costs the learned numbers
            log.warning("ignoring unreadable estimate file %s: %s", self._file(), e)

    def _save(self) -> None:
        try:
            path = self._file()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"rtf": self._rtf, "cps": self._cps, "load": self._load, "translate": self._translate}, f)
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001
            log.warning("could not save estimates: %s", e)

    @staticmethod
    def _update(old: Tuple[float, int], sample: float) -> Tuple[float, int]:
        """Fold one measurement into (average, samples). The built-in starting value counts as half a sample, the real
        ones are averaged equally for the first few, then each new one moves the average by EMA_ALPHA."""
        avg, n = old
        if n < MAX_SAMPLE_WEIGHT:
            return ((avg * (n + 0.5) + sample) / (n + 1.5), n + 1)
        return (avg + EMA_ALPHA * (sample - avg), n + 1)

    # ------------------------------------------------------------------ the numbers
    def rtf(self, engine: Optional[str]) -> float:
        with self._lock:
            self._load_state()
            if engine in self._rtf:
                return self._rtf[engine][0]
        return DEFAULT_RTF.get(engine or "", FALLBACK_RTF)

    def samples(self, engine: str) -> int:
        with self._lock:
            self._load_state()
            return self._rtf.get(engine, (0.0, 0))[1]

    def translate_seconds(self) -> float:
        with self._lock:
            self._load_state()
            return self._translate[0]

    def chars_per_second(self, language: str) -> float:
        with self._lock:
            self._load_state()
            if language in self._cps:
                return self._cps[language][0]
        if language in DEFAULT_CPS:
            return DEFAULT_CPS[language]
        return INDIC_CPS if language in _INDIC else LATIN_CPS

    def audio_seconds(self, text: str, language: str) -> float:
        return max(1.0, len(text.strip()) / self.chars_per_second(language))

    # ------------------------------------------------------------------ warm or cold
    def is_warm(self, engine: Optional[str], now: float) -> bool:
        """True when this engine finished a clip within the model manager's idle window, so its model is still loaded."""
        with self._lock:
            last = self._last_used.get(engine or "")
        return last is not None and 0 <= now - last <= WARM_WINDOW_S

    def touch(self, engine: Optional[str], now: float) -> None:
        if engine:
            with self._lock:
                self._last_used[engine] = now

    def load_seconds(self, engine: Optional[str]) -> float:
        with self._lock:
            self._load_state()
            if engine in self._load:
                return self._load[engine][0]
        return DEFAULT_LOAD_S.get(engine or "", 5.0)

    def voice_seconds(self, engine: Optional[str], text: str, language: str, warm: bool = True) -> float:
        """Expected seconds of the voice stage: RTF x expected audio length (never below the engine's floor), plus the
        cold-start cost when the engine's model is not loaded (`warm=False`)."""
        if not engine:
            return 0.0
        base = max(MIN_VOICE_S.get(engine, 3.0), self.rtf(engine) * self.audio_seconds(text, language))
        return base + (0.0 if warm else self.load_seconds(engine))

    def total_seconds(self, engine: Optional[str], text: str, language: str, translate: bool,
                      warm: bool = True) -> float:
        """Translation (when the text is not ready yet) plus the voice stage."""
        if not engine:
            return 0.0
        return (self.translate_seconds() if translate else 0.0) + self.voice_seconds(engine, text, language, warm)

    # ------------------------------------------------------------------ learning
    def record_voice(self, engine: str, language: str, text: str, voice_s: float, audio_s: Optional[float],
                     cold: bool = False) -> None:
        """A finished clip: `voice_s` seconds of generation for `audio_s` seconds of speech. A clip that had to load
        its model (`cold`) teaches the cold-start cost, never the speed: the loading would be counted as slowness."""
        if not engine or not audio_s or audio_s <= 0.2 or voice_s <= 0:
            return
        chars = len(text.strip())
        with self._lock:
            self._load_state()
            current = self._rtf.get(engine, (DEFAULT_RTF.get(engine, FALLBACK_RTF), 0))[0]
            if cold:
                extra = max(0.0, voice_s - max(MIN_VOICE_S.get(engine, 3.0), current * audio_s))
                prior = DEFAULT_LOAD_S.get(engine, 5.0)
                self._load[engine] = self._update(self._load.get(engine, (prior, 0)), min(extra, 300.0))
            else:
                rtf = max(0.02, min(voice_s / audio_s, 60.0))
                self._rtf[engine] = self._update(self._rtf.get(engine, (current, 0)), rtf)
            if chars >= 8:
                prior = DEFAULT_CPS.get(language, INDIC_CPS if language in _INDIC else LATIN_CPS)
                self._cps[language] = self._update(self._cps.get(language, (prior, 0)), chars / audio_s)
            self._save()

    def record_translate(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            self._load_state()
            self._translate = self._update(self._translate, min(seconds, 120.0))
            self._save()

    # ------------------------------------------------------------------ progress of one running stage
    @staticmethod
    def stage_fraction(elapsed: float, expected: float) -> float:
        """0..1 progress of a stage that is expected to take `expected` seconds and has run for `elapsed`.

        Linear up to 90 % of the expected time; after that it slows down and approaches 1 without ever reaching it,
        so a stage that overruns keeps creeping instead of freezing, and never claims to be finished."""
        if expected <= 0 or elapsed <= 0:
            return 0.0
        x = elapsed / expected
        if x <= 0.9:
            return x
        return min(0.999, 0.9 + 0.1 * (1.0 - math.exp(-(x - 0.9) * 2.0)))

    @staticmethod
    def remaining(elapsed: float, expected: float) -> float:
        """Honest seconds left for a stage: the plan when on schedule, a growing allowance once it overruns."""
        if elapsed < expected:
            return expected - elapsed
        return max(2.0, 0.25 * elapsed)


ESTIMATOR = Estimator()
