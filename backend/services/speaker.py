"""Speaker verification for the consent check: do two recordings sound like the same person?

Uses SpeechBrain's ECAPA-TDNN model (`speechbrain/spkrec-ecapa-voxceleb`, 192-dimensional embeddings) on CPU.
The model is loaded lazily on first use and downloaded once into `data/models/`. When SpeechBrain or the model
is unavailable every function returns None so the caller records the check as "unchecked" instead of failing.

Set CONSENT_SPEAKER_CHECK=off to disable the check; CONSENT_SPEAKER_THRESHOLD sets the minimum cosine similarity."""
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("bonclay.speaker")

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
MIN_SECONDS = 0.5  # shorter audio gives a meaningless embedding
DEFAULT_THRESHOLD = 0.45

_lock = threading.Lock()
_classifier: Any = None
_failed: Optional[str] = None  # why loading failed (not retried until restart)


def enabled() -> bool:
    return os.getenv("CONSENT_SPEAKER_CHECK", "on").strip().lower() not in ("off", "0", "false", "no")


def threshold() -> float:
    try:
        return float(os.getenv("CONSENT_SPEAKER_THRESHOLD", DEFAULT_THRESHOLD))
    except ValueError:
        return DEFAULT_THRESHOLD


def _load() -> Any:
    """The speaker-embedding model, or None when SpeechBrain or the model weights are unavailable."""
    global _classifier, _failed
    if _classifier is not None or _failed is not None:
        return _classifier
    with _lock:
        if _classifier is not None or _failed is not None:
            return _classifier
        try:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # SpeechBrain < 1.0
                from speechbrain.pretrained import EncoderClassifier  # type: ignore
            MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            _classifier = EncoderClassifier.from_hparams(
                source=MODEL_SOURCE, savedir=str(MODEL_DIR), run_opts={"device": "cpu"})
        except Exception as e:  # noqa: BLE001 - missing package, no network on first run, corrupt download...
            _failed = f"{type(e).__name__}: {e}"
            log.warning("speaker verification unavailable (%s)", _failed)
    return _classifier


def available() -> bool:
    return enabled() and _load() is not None


def _decode(audio_bytes: bytes) -> np.ndarray:
    """Mono 16 kHz float32, decoded the same way as the backend's upload loader (ffmpeg handles webm/ogg/m4a)."""
    try:
        from services.audio_decode import decode_file
    except ImportError:  # pragma: no cover
        from backend.services.audio_decode import decode_file

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        return decode_file(path, SAMPLE_RATE)
    finally:
        if os.path.exists(path):
            os.remove(path)


def embed(audio: np.ndarray) -> Optional[np.ndarray]:
    """192-d speaker embedding of 16 kHz mono audio (None if the model is unavailable)."""
    clf = _load()
    if clf is None:
        return None
    import torch  # noqa: PLC0415

    with torch.no_grad():
        emb = clf.encode_batch(torch.from_numpy(audio).unsqueeze(0))
    return emb.squeeze().cpu().numpy().astype(np.float32)


def similarity(wav_a_bytes: bytes, wav_b_bytes: bytes) -> Optional[float]:
    """Cosine similarity of the two speakers' embeddings (-1..1; same speaker is typically above 0.5).

    None when the check cannot run: SpeechBrain or the model is missing, a clip is too short or cannot be
    decoded, or CONSENT_SPEAKER_CHECK=off. Blocking: call it from a worker thread inside async code."""
    if not enabled():
        return None
    try:
        a, b = _decode(wav_a_bytes), _decode(wav_b_bytes)
        if min(len(a), len(b)) < MIN_SECONDS * SAMPLE_RATE:
            log.warning("speaker check skipped: audio shorter than %.1fs", MIN_SECONDS)
            return None
        ea, eb = embed(a), embed(b)
        if ea is None or eb is None:
            return None
        denom = float(np.linalg.norm(ea) * np.linalg.norm(eb))
        return float(np.dot(ea, eb) / denom) if denom else None
    except Exception as e:  # noqa: BLE001
        log.warning("speaker check failed: %s", e)
        return None
