"""Provenance for synthetic audio: an inaudible watermark and a signed receipt for every clip BonClay makes.

Watermark: Meta's AudioSeal (16-bit payload, MIT licence for code and weights). The payload is the first 16 bits of
sha256(generation id), so a detected message narrows a clip down to the receipts that carry it (collisions are
possible; the receipt resolves them). Models load lazily on CPU. If AudioSeal is missing, the weights cannot be
fetched, or WATERMARK=off, watermark() returns the audio unchanged and detect() reports detected=False with a
reason; generation is never broken by this module.

Level: AudioSeal's generator and detector are both level-dependent, so watermark_ex() rescales each clip's watermark to
a fixed speech-to-watermark ratio (see TARGET_WM_SNR_DB) and detect() also scores a quieter copy; the 16-bit payload and
the detect() result shape are unchanged, and clips marked before the scaling still verify.

Receipt: a small JSON record signed twice: with Ed25519 (anyone can check it against the public key published at
/.well-known/bonclay-receipt-key, no secret needed) and with HMAC-SHA256 (BONCLAY_RECEIPT_SECRET, else a secret generated on first run
in data/receipt_secret). It states which engine, voice and consent produced a clip and carries the sha256 of the
watermarked file, so an uploaded copy can be matched byte for byte. Receipts are kept in data/receipts.jsonl and, for
cloud generations, in the Supabase `receipts` table (migration 008_receipts.sql).

Evidence, not proof: the detector flags audio that probably carries the watermark. It can miss a clip that has been
heavily edited, and no detector is infallible."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import struct
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from cryptography.hazmat.primitives import serialization
except ImportError:  # receipts then carry only the HMAC signature
    serialization = None  # type: ignore[assignment]

log = logging.getLogger("bonclay.provenance")

MODEL_SR = 16000
NBITS = 16
CHUNK_SECONDS = 60          # long clips go through the models in pieces to bound memory
MAX_DETECT_SECONDS = 600    # analyse at most the first ten minutes of an upload
RECEIPT_VERSION = 1
# AudioSeal's watermark does not track the speech level: its RMS grows only about level^0.6 (measured: speech 12 dB
# louder gives a watermark 8 dB louder), so loud speech is marked relatively weaker. Kokoro Hindi (hf_alpha) renders
# ~6 dB louder than English (af_heart), which put its watermark ~33 dB under speech instead of ~27 and made it
# collapse under 20 dB SNR noise. watermark() therefore rescales the generator's output per chunk so the
# watermark-to-speech ratio is the same for every clip. 27 dB is the English reference level, so English output
# barely changes. Detection is unchanged; clips marked before this change still verify.
TARGET_WM_SNR_DB = 27.0      # speech (active frames) over watermark, dB; WATERMARK_SNR_DB overrides it
# The detector is not level-invariant either (Kokoro clips in pink noise at 20 dB SNR: loud Hindi scored 0.30-0.46 mean
# frame fraction, the same audio 12 dB quieter 0.52). But no single level is best for every kind of damage: a fixed
# normalisation made clips marked by the old code worse under Opus. So detect() scores a piece as it is and, when that
# is not already convincing, again at this active-speech level, and keeps the better one. A genuinely clean clip scored
# at most 0.09 either way, so taking the maximum of two looks adds no false positives in the bench (0 of 420 + 0 of 420).
DETECT_SPEECH_RMS = 0.03    # about -30 dBFS
CONFIDENT_PROB = 0.9        # a piece scoring at least this on the raw audio is not rescored
MIN_DETECT_GAIN, MAX_DETECT_GAIN = 0.05, 20.0
MIN_SCALE, MAX_SCALE = 0.1, 8.0  # cap on the correction, so a degenerate clip cannot get an absurd mark
RETRY_AFTER_S = 120.0       # after a failed model load, wait this long before trying again

# Fields covered by the signature, in no particular order (the canonical form sorts keys).
SIGNED_FIELDS = ("version", "receipt_id", "kind", "generation_id", "turn_id", "created_at", "engine", "voice_id",
                 "voice_name", "consent_id", "consent_verified", "text_sha256", "audio_sha256", "watermark_message")

_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_get_supabase: Callable[[], Any] = lambda: None


def configure(get_supabase: Optional[Callable[[], Any]] = None, data_dir: Optional[str] = None) -> None:
    """Wire in the Supabase client and/or move the data directory (tests). Resets cached secret and receipts."""
    global _get_supabase, _data_dir, _secret_cache, _signing_cache
    if get_supabase is not None:
        _get_supabase = get_supabase
    if data_dir is not None:
        _data_dir = data_dir
    _secret_cache = None
    _signing_cache = None
    _store.reset()
    _withdrawals.reset()


# ---------------------------------------------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------------------------------------------

def enabled() -> bool:
    return os.getenv("WATERMARK", "on").strip().lower() not in ("off", "0", "false", "no")


def detect_threshold() -> float:
    try:
        return float(os.getenv("WATERMARK_DETECT_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


_load_lock = threading.Lock()
_infer_lock = threading.Lock()  # one CPU inference at a time: the models are small, parallel runs only fight
_models: Optional[Tuple[Any, Any]] = None
_load_error: Optional[Tuple[str, float]] = None


def _load_models() -> Tuple[Any, Any]:
    """(generator, detector), loaded once on CPU. Raises on failure (message kept for _engine_reason)."""
    global _models, _load_error
    if _models is not None:
        return _models
    with _load_lock:
        if _models is not None:
            return _models
        if _load_error is not None and time.time() - _load_error[1] < RETRY_AFTER_S:
            raise RuntimeError(_load_error[0])
        try:
            import torch  # noqa: F401
            # AudioSeal wraps its SEANet blocks in torch.compile, which compiles again for every new clip length until
            # Dynamo reaches its recompile limit (8) and runs eagerly from then on. Measured: the first ten clips took
            # 0.8 s each instead of 0.1 s, it holds ~140 MB, and warm clips are no faster (about 0.1 s either way).
            # So run eagerly from the start. audioseal reads this when it is imported, so set it before that.
            os.environ.setdefault("NO_TORCH_COMPILE", "1")
            try:
                import torch._dynamo
                torch._dynamo.config.disable = True
                torch._dynamo.config.suppress_errors = True
            except Exception:
                pass
            from audioseal import AudioSeal
            gen = AudioSeal.load_generator("audioseal_wm_16bits", device="cpu")
            det = AudioSeal.load_detector("audioseal_detector_16bits", device="cpu")
            gen.eval()
            det.eval()
            _models = (gen, det)
            _load_error = None
            log.info("AudioSeal watermark models loaded (CPU)")
        except Exception as e:  # noqa: BLE001 - never let the watermark break generation
            _load_error = (f"AudioSeal unavailable: {type(e).__name__}: {str(e)[:160]}", time.time())
            log.warning(_load_error[0])
            raise RuntimeError(_load_error[0]) from e
        return _models


def _engine_reason() -> Optional[str]:
    """Why AudioSeal cannot run here (not installed, weights missing), or None. Loads the models on first call."""
    try:
        _load_models()
        return None
    except Exception as e:  # noqa: BLE001
        return str(e) or "AudioSeal unavailable"


def unavailable_reason() -> Optional[str]:
    if not enabled():
        return "Watermarking is turned off (WATERMARK=off)"
    return _engine_reason()


def status() -> dict:
    reason = unavailable_reason()
    return {"enabled": enabled(), "available": reason is None, "reason": reason, "engine": "audioseal_16bits"}


def message_for(generation_id: str) -> int:
    """16-bit watermark payload for an id: the first two bytes of sha256(id)."""
    return int.from_bytes(hashlib.sha256(str(generation_id).encode("utf-8")).digest()[:2], "big")


def _bits(message: int) -> List[int]:
    return [(int(message) >> (NBITS - 1 - i)) & 1 for i in range(NBITS)]


def _from_bits(bits) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def _resample(x: np.ndarray, orig: int, new: int) -> np.ndarray:
    if orig == new or x.size == 0:
        return x.astype(np.float32, copy=False)
    try:
        import torch
        import torchaudio.functional as taf
        return taf.resample(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), orig, new).numpy()
    except Exception:  # noqa: BLE001
        pass
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(orig), int(new))
        return resample_poly(x, int(new) // g, int(orig) // g).astype(np.float32)
    except Exception:  # noqa: BLE001
        n = max(1, int(round(x.size * new / orig)))
        return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


def target_snr_db() -> float:
    try:
        return float(os.getenv("WATERMARK_SNR_DB", TARGET_WM_SNR_DB))
    except ValueError:
        return TARGET_WM_SNR_DB


def _speech_rms(x: np.ndarray) -> float:
    """RMS over the active 20 ms frames (those within 20 dB of the 95th-percentile frame), so pauses and silence do not
    drag the level down. 0.0 for silence."""
    n = (x.size // 320) * 320
    if n == 0:
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    e = np.square(x[:n].astype(np.float64)).reshape(-1, 320).mean(axis=1)
    top = float(np.percentile(e, 95))
    if top <= 0.0:
        return 0.0
    return float(np.sqrt(e[e >= top * 0.01].mean()))


def _scaled_watermark(chunk: np.ndarray, wm: np.ndarray, whole: np.ndarray) -> np.ndarray:
    """Scale a chunk's raw watermark so speech/watermark = target_snr_db(). The speech level comes from the whole clip
    for a chunk under 2 s (a short tail would otherwise be judged on its own, possibly silent, level). A silent clip
    or a zero watermark is returned as is."""
    speech = _speech_rms(chunk if chunk.size >= 2 * MODEL_SR else whole)
    wm_rms = float(np.sqrt(np.mean(np.square(wm, dtype=np.float64)))) if wm.size else 0.0
    if speech < 1e-4 or wm_rms < 1e-9:
        return wm
    scale = speech * 10 ** (-target_snr_db() / 20) / wm_rms
    return (wm * np.float32(min(MAX_SCALE, max(MIN_SCALE, scale)))).astype(np.float32)


def _detect_piece(x16: np.ndarray) -> Tuple[float, Optional[int]]:
    """_detect16 on the piece as it is and, unless that is already convincing, at DETECT_SPEECH_RMS; the higher score
    wins (with its message). Silent pieces and pieces already near that level are scored once."""
    prob, msg = _detect16(x16)
    if prob >= CONFIDENT_PROB:
        return prob, msg
    level = _speech_rms(x16)
    if level < 1e-4:
        return prob, msg
    gain = min(MAX_DETECT_GAIN, max(MIN_DETECT_GAIN, DETECT_SPEECH_RMS / level))
    if 0.8 < gain < 1.25:
        return prob, msg
    prob2, msg2 = _detect16((x16 * np.float32(gain)).astype(np.float32))
    return (prob2, msg2) if prob2 > prob else (prob, msg)


def _embed16(x16: np.ndarray, message: int) -> np.ndarray:
    """The watermark signal (same length as x16, 16 kHz mono float32) for a message. Patched in tests."""
    import torch
    gen, _ = _load_models()
    msg = torch.tensor([_bits(message)], dtype=torch.int32)
    with _infer_lock, torch.inference_mode():
        wm = gen.get_watermark(torch.from_numpy(x16)[None, None, :], MODEL_SR, message=msg)
    return wm[0, 0].cpu().numpy().astype(np.float32)


def _detect16(x16: np.ndarray) -> Tuple[float, Optional[int]]:
    """(fraction of 16 kHz frames flagged as watermarked, decoded message or None). Patched in tests."""
    import torch
    _, det = _load_models()
    with _infer_lock, torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(x16, dtype=np.float32))[None, None, :]
        if getattr(det, "normalizer", None) is not None:
            x = det.normalizer.loudness_normalization(x)
        raw = det.detector(x)                                   # 1 x (2 + 16) x frames
        p_wm = torch.softmax(raw[:, :2, :], dim=1)[0, 1]       # per-frame watermark probability
        flagged = p_wm > 0.5
        prob = float(flagged.float().mean())
        if not bool(flagged.any()):
            return prob, None
        logits = raw[0, 2:, :][:, flagged].mean(dim=-1)         # message read only from watermarked frames
        return prob, _from_bits((torch.sigmoid(logits) > 0.5).int().tolist())


def watermark_ex(wav: np.ndarray, sr: int, message_bits: int) -> Tuple[np.ndarray, dict]:
    """watermark() plus what happened: {"applied": bool, "message": int|None, "reason": str|None}."""
    info: Dict[str, Any] = {"applied": False, "message": None, "reason": None}
    original = wav
    reason = unavailable_reason()
    if reason:
        info["reason"] = reason
        return original, info
    try:
        x = np.asarray(wav, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1) if x.ndim == 2 and 1 in x.shape else None
            if x is None:
                info["reason"] = "Only mono audio can be watermarked"
                return original, info
        if x.size == 0:
            info["reason"] = "Empty audio"
            return original, info
        x16 = _resample(x, int(sr), MODEL_SR)
        step = CHUNK_SECONDS * MODEL_SR
        msg = int(message_bits) & 0xFFFF
        wm16 = np.concatenate([_scaled_watermark(x16[i:i + step], _embed16(x16[i:i + step], msg), x16)
                               for i in range(0, x16.size, step)])
        wm = _resample(wm16, MODEL_SR, int(sr))
        if wm.size < x.size:
            wm = np.pad(wm, (0, x.size - wm.size))
        out = np.clip(x + wm[:x.size], -1.0, 1.0).astype(np.float32)
        info.update(applied=True, message=int(message_bits) & 0xFFFF)
        return out, info
    except Exception as e:  # noqa: BLE001
        log.warning("watermarking failed, audio left unmarked: %s", e)
        info["reason"] = f"Watermarking failed: {type(e).__name__}: {str(e)[:120]}"
        return original, info


def watermark(wav: np.ndarray, sr: int, message_bits: int) -> np.ndarray:
    """Embed a 16-bit message in mono float audio at any sample rate. Unavailable or failed -> returned unchanged."""
    return watermark_ex(wav, sr, message_bits)[0]


def detect(wav: np.ndarray, sr: int) -> dict:
    """{"detected": bool, "probability": float, "message": int|None}; adds "reason" when detection is unavailable."""
    reason = unavailable_reason()
    if reason:
        return {"detected": False, "probability": 0.0, "message": None, "reason": reason}
    try:
        x = np.asarray(wav, dtype=np.float32).reshape(-1)
        x16 = _resample(x, int(sr), MODEL_SR)[:MAX_DETECT_SECONDS * MODEL_SR]
        if x16.size < MODEL_SR // 4:
            return {"detected": False, "probability": 0.0, "message": None, "reason": "Clip is too short to check"}
        step = CHUNK_SECONDS * MODEL_SR
        pieces = [x16[i:i + step] for i in range(0, x16.size, step)]
        if len(pieces) > 1 and pieces[-1].size < MODEL_SR:  # a sliver is folded into the previous piece
            pieces[-2] = np.concatenate([pieces[-2], pieces.pop()])
        results = [(p.size,) + tuple(_detect_piece(p)) for p in pieces]
        total = sum(n for n, _, _ in results)
        prob = sum(n * pr for n, pr, _ in results) / total
        votes = [msg for _, pr, msg in results if msg is not None and pr > 0.5]
        detected = prob >= detect_threshold()
        message = None
        if detected and votes:
            message = max(set(votes), key=votes.count)
        return {"detected": bool(detected), "probability": round(float(prob), 4), "message": message}
    except Exception as e:  # noqa: BLE001
        log.warning("watermark detection failed: %s", e)
        return {"detected": False, "probability": 0.0, "message": None,
                "reason": f"Detection failed: {type(e).__name__}: {str(e)[:120]}"}


# ---------------------------------------------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------------------------------------------

_secret_cache: Optional[bytes] = None
_secret_lock = threading.Lock()


def _secret() -> bytes:
    """BONCLAY_RECEIPT_SECRET, else a random secret created on first use and kept in data/receipt_secret."""
    global _secret_cache
    env = os.getenv("BONCLAY_RECEIPT_SECRET", "").strip()
    if env:
        return env.encode("utf-8")
    with _secret_lock:
        if _secret_cache is not None:
            return _secret_cache
        path = os.path.join(_data_dir, "receipt_secret")
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = f.read().strip()
        except OSError:
            value = ""
        if not value:
            value = secrets.token_hex(32)
            os.makedirs(_data_dir, exist_ok=True)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(value)
            except FileExistsError:  # another worker won the race; use its secret
                with open(path, "r", encoding="utf-8") as f:
                    value = f.read().strip()
            except OSError as e:
                log.warning("could not persist the receipt secret (%s); receipts signed with it will not verify "
                            "after a restart. Set BONCLAY_RECEIPT_SECRET.", e)
        _secret_cache = value.encode("utf-8")
        return _secret_cache


def _canonical(receipt: dict) -> bytes:
    payload = {k: receipt.get(k) for k in SIGNED_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign(receipt: dict) -> str:
    return hmac.new(_secret(), _canonical(receipt), hashlib.sha256).hexdigest()


def signature_valid(receipt: Optional[dict]) -> bool:
    if not receipt or not isinstance(receipt.get("signature"), str):
        return False
    return hmac.compare_digest(sign(receipt), receipt["signature"])


# ---- Ed25519: a signature a third party can verify without the operator's secret ----------------------------------

_signing_cache: Optional[Any] = None
_signing_lock = threading.Lock()
LOG_GENESIS = "0" * 64   # `prev` of the first signed row of the withdrawal log


def _signing_key() -> Optional[Any]:
    """The Ed25519 private key: BONCLAY_RECEIPT_SIGNING_KEY (base64 of the 32 raw bytes), else a key generated on first
    use and kept in data/receipt_ed25519.key (0600). None when `cryptography` is missing or the key cannot be made."""
    global _signing_cache
    if serialization is None:
        return None
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    env = os.getenv("BONCLAY_RECEIPT_SIGNING_KEY", "").strip()
    if env:
        try:
            return Ed25519PrivateKey.from_private_bytes(base64.b64decode(env))
        except Exception as e:  # noqa: BLE001
            log.warning("BONCLAY_RECEIPT_SIGNING_KEY is not a base64 Ed25519 key (%s); not signing", e)
            return None
    with _signing_lock:
        if _signing_cache is not None:
            return _signing_cache
        path = os.path.join(_data_dir, "receipt_ed25519.key")
        raw = b""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = base64.b64decode(f.read().strip())
        except (OSError, ValueError):
            raw = b""
        if len(raw) != 32:
            key = Ed25519PrivateKey.generate()
            raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                    serialization.NoEncryption())
            os.makedirs(_data_dir, exist_ok=True)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(base64.b64encode(raw).decode("ascii"))
            except FileExistsError:  # another worker won the race; use its key
                with open(path, "r", encoding="utf-8") as f:
                    raw = base64.b64decode(f.read().strip())
            except OSError as e:
                log.warning("could not persist the receipt signing key (%s); receipts signed with it will not verify "
                            "after a restart. Set BONCLAY_RECEIPT_SIGNING_KEY.", e)
        _signing_cache = Ed25519PrivateKey.from_private_bytes(raw)
        return _signing_cache


def _raw_public(key: Any) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def public_key_b64() -> Optional[str]:
    """The public half, base64 of the 32 raw bytes (what /.well-known/bonclay-receipt-key publishes)."""
    key = _signing_key()
    return base64.b64encode(_raw_public(key)).decode("ascii") if key is not None else None


def key_id(public_b64: Optional[str] = None) -> Optional[str]:
    """A short id of a public key (first 16 hex of its sha256), so a verifier can tell keys apart."""
    pub = public_b64 or public_key_b64()
    return hashlib.sha256(base64.b64decode(pub)).hexdigest()[:16] if pub else None


# ---- key rotation: retired public keys stay known, so old receipts and log rows keep verifying ----------------------
# Every receipt and every signed withdrawal row carries the `key_id` of the key that signed it. Rotation
# (rotate_signing_key, scripts/rotate_receipt_key.py) retires the current public key into data/receipt_ed25519.retired
# and starts a new private key; BONCLAY_RECEIPT_RETIRED_KEYS (comma-separated base64 public keys) adds more, for an
# operator whose key lives in BONCLAY_RECEIPT_SIGNING_KEY. Verification picks the key by `key_id` first and falls back
# to every known key, so receipts made before `key_id` existed still verify. /.well-known/bonclay-receipt-key lists all.

def _retired_path() -> str:
    return os.path.join(_data_dir, "receipt_ed25519.retired")


def retired_public_keys() -> List[str]:
    """Base64 public keys that signed receipts in the past (env plus the retired file), without the current one."""
    out: List[str] = []
    for part in os.getenv("BONCLAY_RECEIPT_RETIRED_KEYS", "").split(","):
        part = part.strip()
        if part and part not in out:
            out.append(part)
    try:
        with open(_retired_path(), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and line not in out:
                    out.append(line)
    except OSError:
        pass
    current = public_key_b64()
    valid: List[str] = []
    for pub in out:
        try:
            if len(base64.b64decode(pub)) == 32 and pub != current:
                valid.append(pub)
        except (ValueError, TypeError):
            continue
    return valid


def all_public_keys() -> List[dict]:
    """[{"key_id", "public_key", "status"}]: the current key first, then the retired ones (what the verify endpoint
    publishes, so a verifier can match a receipt's `key_id`)."""
    current = public_key_b64()
    keys = [{"key_id": key_id(current), "public_key": current, "status": "current"}] if current else []
    keys += [{"key_id": key_id(pub), "public_key": pub, "status": "retired"} for pub in retired_public_keys()]
    return keys


def _candidate_keys(kid: Optional[str]) -> List[str]:
    """Public keys to try for a signature: the one whose id matches `kid` first, then the current key, then the retired
    ones (a signature made before `key_id` was recorded has to be tried against each)."""
    known = [pub for pub in [public_key_b64(), *retired_public_keys()] if pub]
    ordered = [pub for pub in known if kid and key_id(pub) == kid]
    ordered += [pub for pub in known if pub not in ordered]
    return ordered


def rotate_signing_key() -> dict:
    """Retire the current Ed25519 key (its public half goes to data/receipt_ed25519.retired) and start signing with a
    new one. Run with the API stopped (scripts/rotate_receipt_key.py): a running server keeps the old key in memory.
    Refused when the key comes from BONCLAY_RECEIPT_SIGNING_KEY: rotate that by hand (new value there, the old public
    key in BONCLAY_RECEIPT_RETIRED_KEYS). Returns {"retired_key_id", "key_id", "retired_count"}."""
    global _signing_cache
    if serialization is None:
        raise RuntimeError("the cryptography package is not installed")
    if os.getenv("BONCLAY_RECEIPT_SIGNING_KEY", "").strip():
        raise RuntimeError("the signing key comes from BONCLAY_RECEIPT_SIGNING_KEY: generate a new key, set it there, "
                           "and add the old public key to BONCLAY_RECEIPT_RETIRED_KEYS")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    old = _signing_key()
    if old is None:
        raise RuntimeError("no signing key could be loaded or made")
    old_pub = base64.b64encode(_raw_public(old)).decode("ascii")
    already_retired = old_pub in retired_public_keys()   # outside the lock: these helpers take it themselves
    with _signing_lock:
        os.makedirs(_data_dir, exist_ok=True)
        if not already_retired:
            with open(_retired_path(), "a", encoding="utf-8") as f:
                f.write(f"# retired {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n{old_pub}\n")
        new = Ed25519PrivateKey.generate()
        raw = new.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                serialization.NoEncryption())
        path = os.path.join(_data_dir, "receipt_ed25519.key")
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(base64.b64encode(raw).decode("ascii"))
        os.replace(tmp, path)
        _signing_cache = new
    new_pub = base64.b64encode(_raw_public(new)).decode("ascii")
    log.info("receipt signing key rotated: %s retired, %s current", key_id(old_pub), key_id(new_pub))
    return {"retired_key_id": key_id(old_pub), "key_id": key_id(new_pub), "retired_count": len(retired_public_keys())}


def _sign_bytes(data: bytes) -> Optional[str]:
    key = _signing_key()
    return base64.b64encode(key.sign(data)).decode("ascii") if key is not None else None


def _verify_bytes(data: bytes, signature_b64: Any, public_b64: Optional[str] = None, kid: Optional[str] = None) -> bool:
    """True when `signature_b64` is a valid Ed25519 signature of `data` by `public_b64`; without one, by the key whose
    id is `kid`, else by any key this server has ever signed with (current, then retired)."""
    if not isinstance(signature_b64, str) or serialization is None:
        return False
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        signature = base64.b64decode(signature_b64)
    except (ValueError, TypeError):
        return False
    for pub in ([public_b64] if public_b64 else _candidate_keys(kid)):
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(pub)).verify(signature, data)
            return True
        except (InvalidSignature, ValueError, TypeError):
            continue
    return False


def sign_ed25519(receipt: dict) -> Optional[str]:
    return _sign_bytes(_canonical(receipt))


def signature_ed25519_valid(receipt: Optional[dict], public_b64: Optional[str] = None) -> Optional[bool]:
    """Does the receipt's Ed25519 signature check out (against `public_b64`, default the key named by the receipt's
    `key_id`, then any key this server has signed with)? None for a receipt that has none (made before Ed25519 receipts
    existed): it still has its HMAC signature."""
    if not receipt or not receipt.get("signature_ed25519"):
        return None
    return _verify_bytes(_canonical(receipt), receipt["signature_ed25519"], public_b64, receipt.get("key_id"))


def public_receipt(receipt: dict) -> dict:
    """The receipt as shown to anyone: signed fields + signature + whether the signature checks out (no owner id).

    Also says whether the voice has since been removed: `voice_withdrawn_at` (when the voice owner withdrew consent or
    the voice was deleted), `voice_removed_reason` ("withdrawn" or "retention_expired"). They are looked up from the
    withdrawal log, whose rows are hash-chained and Ed25519-signed (`voice_withdrawal_signed`; None when the voice was
    not withdrawn): the receipt itself is never changed (it is an audit trail)."""
    out = {k: receipt.get(k) for k in SIGNED_FIELDS}
    out["signature"] = receipt.get("signature")
    out["signature_valid"] = signature_valid(receipt)
    out["signature_ed25519"] = receipt.get("signature_ed25519")
    out["key_id"] = receipt.get("key_id")
    out["signature_ed25519_valid"] = signature_ed25519_valid(receipt)
    gone = withdrawal_for(receipt.get("voice_id"))
    out["voice_withdrawn_at"] = gone["withdrawn_at"] if gone else None
    out["voice_removed_reason"] = gone["reason"] if gone else None
    out["voice_withdrawal_signed"] = withdrawal_row_valid(gone) if gone else None
    out["voice_withdrawal_row_hash"] = gone.get("row_hash") if gone else None
    out["voice_consent_status"] = consent_status(receipt, gone)
    return out


def consent_status(receipt: dict, withdrawal: Optional[dict] = None) -> str:
    """One word on the consent behind the voice in a receipt, as of now:
    "preset_voice" (a built-in voice: nobody's voice was cloned, so no consent applies), "withdrawn" (the voice owner
    withdrew consent), "retention_ended" (the voice was deleted when its retention period ended), "active" (a cloned
    voice whose consent was verified and not withdrawn), or "unverified" (a cloned voice without a verified consent)."""
    if withdrawal is None:
        withdrawal = withdrawal_for(receipt.get("voice_id")) if receipt.get("consent_id") else None
    if withdrawal:
        return "retention_ended" if withdrawal.get("reason") == "retention_expired" else "withdrawn"
    if not receipt.get("consent_id"):
        return "preset_voice"
    return "active" if receipt.get("consent_verified") else "unverified"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _ReceiptStore:
    """data/receipts.jsonl: one receipt per line, append-only, so adding a receipt costs the same at 100 or 100,000
    receipts. The reader keeps an in-memory list and only reads lines appended since its last look (another worker
    may append too). A legacy data/receipts.json list is migrated on first use."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rows: List[dict] = []
        self._offset = 0

    def reset(self) -> None:
        with self._lock:
            self._rows, self._offset = [], 0

    @staticmethod
    def _path() -> str:
        return os.path.join(_data_dir, "receipts.jsonl")

    def _migrate_legacy(self) -> None:
        legacy = os.path.join(_data_dir, "receipts.json")
        if not os.path.exists(legacy) or os.path.exists(self._path()):
            return
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except (OSError, ValueError):
            rows = []
        with open(self._path(), "a", encoding="utf-8") as f:
            for r in rows if isinstance(rows, list) else []:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(legacy, legacy + ".migrated")

    def _refresh(self) -> None:
        self._migrate_legacy()
        try:
            size = os.path.getsize(self._path())
        except OSError:
            self._rows, self._offset = [], 0
            return
        if size < self._offset:  # file was replaced or truncated: read it again from the start
            self._rows, self._offset = [], 0
        if size == self._offset:
            return
        with open(self._path(), "rb") as f:
            f.seek(self._offset)
            chunk = f.read(size - self._offset)
        complete = chunk[: chunk.rfind(b"\n") + 1]  # ignore a line another writer is still appending
        for line in complete.splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    self._rows.append(row)
            except ValueError:
                continue
        self._offset += len(complete)

    def add(self, row: dict) -> None:
        with self._lock:
            os.makedirs(_data_dir, exist_ok=True)
            self._refresh()
            with open(self._path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._refresh()

    def find(self, **match) -> List[dict]:
        with self._lock:
            self._refresh()
            return [dict(r) for r in self._rows if all(r.get(k) == v for k, v in match.items())]


_store = _ReceiptStore()


class _WithdrawalLog:
    """data/voice_withdrawals.jsonl: one line per voice that was withdrawn (or deleted at the end of its retention
    period): {"voice_id", "withdrawn_at", "reason"}. Receipts are never edited; /verify looks a voice up here.
    The log holds no personal data (ids and a date). Re-read when the file changes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rows: Dict[str, dict] = {}
        self._stamp: Optional[Tuple[str, float, int]] = None

    def reset(self) -> None:
        with self._lock:
            self._rows, self._stamp = {}, None

    @staticmethod
    def _path() -> str:
        return os.path.join(_data_dir, "voice_withdrawals.jsonl")

    def _refresh(self) -> None:
        try:
            st = os.stat(self._path())
            stamp: Optional[Tuple[str, float, int]] = (self._path(), st.st_mtime, st.st_size)
        except OSError:
            self._rows, self._stamp = {}, None
            return
        if stamp == self._stamp:
            return
        rows: Dict[str, dict] = {}
        try:
            with open(self._path(), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict) and row.get("voice_id"):
                        rows.setdefault(str(row["voice_id"]), row)  # the first withdrawal date stands
            self._rows, self._stamp = rows, stamp
        except OSError:
            pass

    def add(self, voice_id: str, reason: str) -> dict:
        with self._lock:
            self._refresh()
            if voice_id in self._rows:
                return dict(self._rows[voice_id])
            row = {"voice_id": voice_id, "withdrawn_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "reason": reason}
            os.makedirs(_data_dir, exist_ok=True)
            row.update(_sign_row(row, self._last_hash()))
            with open(self._path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self._refresh()
            return row

    def _last_hash(self) -> str:
        """`row_hash` of the newest signed row on disk (the chain continues from it), else the genesis value."""
        last = LOG_GENESIS
        for row in self.all_rows():
            if row.get("row_hash"):
                last = row["row_hash"]
        return last

    def get(self, voice_id: str) -> Optional[dict]:
        with self._lock:
            self._refresh()
            row = self._rows.get(str(voice_id))
            return dict(row) if row else None

    def all_rows(self) -> List[dict]:
        """Every line of the log, in file order."""
        out: List[dict] = []
        with self._lock:
            try:
                with open(self._path(), "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(row, dict):
                            out.append(row)
            except OSError:
                pass
        return out


_withdrawals = _WithdrawalLog()


def _row_hash(row: dict, prev: str) -> str:
    body = {"voice_id": row.get("voice_id"), "withdrawn_at": row.get("withdrawn_at"), "reason": row.get("reason"),
            "prev": prev}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sign_row(row: dict, prev: str) -> dict:
    """The chain fields of a new log row: `prev` (hash of the row before it), `row_hash` and an Ed25519 `sig` of that
    hash. Empty when the signing key is unavailable, which leaves a legacy unsigned row."""
    digest = _row_hash(row, prev)
    sig = _sign_bytes(digest.encode("ascii"))
    if not sig:
        return {}
    return {"prev": prev, "row_hash": digest, "key_id": key_id(), "sig": sig}


def withdrawal_row_valid(row: Optional[dict], public_b64: Optional[str] = None) -> bool:
    """A log row is valid when its hash matches its fields and its signature matches the hash."""
    if not row or not row.get("row_hash") or not row.get("sig"):
        return False
    if _row_hash(row, str(row.get("prev"))) != row["row_hash"]:
        return False
    return _verify_bytes(str(row["row_hash"]).encode("ascii"), row["sig"], public_b64, row.get("key_id"))


def withdrawal_rows() -> List[dict]:
    """The whole withdrawal log as published (ids, dates and signatures; no personal data)."""
    return _withdrawals.all_rows()


def verify_withdrawal_log(rows: Optional[List[dict]] = None, public_b64: Optional[str] = None) -> dict:
    """Walk the log: every signed row must hash and verify, and each `prev` must be the hash of the signed row before it
    (a deleted, edited or reordered row breaks the chain). Legacy unsigned rows (written before the log was signed) are
    counted and skipped. Returns {"ok", "rows", "signed_rows", "legacy_rows", "broken_at", "head"}; `broken_at` is the
    index of the first bad row, or None."""
    rows = withdrawal_rows() if rows is None else rows
    prev, signed, legacy, broken = LOG_GENESIS, 0, 0, None
    for i, row in enumerate(rows):
        if not row.get("row_hash"):
            legacy += 1
            continue
        if row.get("prev") != prev or not withdrawal_row_valid(row, public_b64):
            broken = i
            break
        prev = row["row_hash"]
        signed += 1
    return {"ok": broken is None, "rows": len(rows), "signed_rows": signed, "legacy_rows": legacy,
            "broken_at": broken, "head": prev}


def record_withdrawal(voice_id: str, reason: str = "withdrawn") -> Optional[dict]:
    """Mark a voice as withdrawn (call once its data is deleted). Receipts that name the voice then read as
    'voice withdrawn' on /verify. Never raises; returns the log row, or None when it could not be written."""
    if not voice_id:
        return None
    try:
        return _withdrawals.add(str(voice_id), reason if reason in ("withdrawn", "retention_expired") else "withdrawn")
    except Exception as e:  # noqa: BLE001
        log.warning("could not record the withdrawal of voice %s: %s", voice_id, e)
        return None


def withdrawal_for(voice_id: Optional[str]) -> Optional[dict]:
    """{"voice_id", "withdrawn_at", "reason"} when the voice was withdrawn, else None."""
    if not voice_id:
        return None
    try:
        return _withdrawals.get(str(voice_id))
    except Exception as e:  # noqa: BLE001
        log.warning("withdrawal lookup failed: %s", e)
        return None


def _cloud_find(column: str, value: Any) -> List[dict]:
    sb = _get_supabase()
    if sb is None:
        return []
    try:
        res = sb.table("receipts").select("*").eq(column, value).order("created_at", desc=True).limit(20).execute()
        return list(res.data or [])
    except Exception as e:  # noqa: BLE001 - table missing (migration 008 not applied) or Supabase down
        log.warning("receipt lookup in Supabase failed: %s", e)
        return []


def _merge(local: List[dict], cloud: List[dict]) -> List[dict]:
    seen, out = set(), []
    for r in local + cloud:
        rid = r.get("receipt_id") or r.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        if "receipt_id" not in r:
            r = {**r, "receipt_id": rid}
        out.append(r)
    out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return out


def get_receipt(receipt_id: str) -> Optional[dict]:
    found = _store.find(receipt_id=receipt_id)
    if found:
        return found[0]
    cloud = _cloud_find("receipt_id", receipt_id)
    return cloud[0] if cloud else None


def find_by_audio_sha(sha: str) -> List[dict]:
    return _merge(_store.find(audio_sha256=sha), _cloud_find("audio_sha256", sha))


def find_by_message(message: int) -> List[dict]:
    """Receipts whose watermark carries this 16-bit message (several when generation ids collide)."""
    return _merge(_store.find(watermark_message=int(message)), _cloud_find("watermark_message", int(message)))


def create_receipt(*, kind: str, subject_id: str, engine: Optional[str], voice_id: Optional[str],
                   voice_name: Optional[str], consent_id: Optional[str], consent_verified: Optional[bool],
                   text: str, audio_bytes: bytes, watermark_message: Optional[int], user_id: Optional[str],
                   cloud: bool = False, receipt_id: Optional[str] = None) -> dict:
    """Build, sign and store a receipt (synchronous; call via to_thread). Never raises: returns {} on failure.

    `receipt_id` lets the caller choose the id up front (apply_provenance writes it into the file's metadata before the
    file is hashed)."""
    try:
        receipt = {
            "version": RECEIPT_VERSION,
            "receipt_id": receipt_id or str(uuid.uuid4()),
            "kind": kind,
            "generation_id": subject_id if kind == "generation" else None,
            "turn_id": subject_id if kind == "agent_reply" else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine": engine,
            "voice_id": voice_id or None,
            "voice_name": voice_name,
            "consent_id": consent_id or None,
            "consent_verified": None if consent_verified is None else bool(consent_verified),
            "text_sha256": sha256_hex((text or "").encode("utf-8")),
            "audio_sha256": sha256_hex(audio_bytes),
            "watermark_message": None if watermark_message is None else int(watermark_message),
        }
        receipt["signature"] = sign(receipt)
        ed = sign_ed25519(receipt)
        if ed:
            receipt["signature_ed25519"] = ed
            receipt["key_id"] = key_id()
        # The engine's licence, so a receipt shows whether its clip is cleared for paid use (services/licences.py).
        # Informational: not among SIGNED_FIELDS, so receipts made before 22 Oct 2026 still verify.
        try:
            from services import licences as _licences
        except ImportError:  # pragma: no cover
            from backend.services import licences as _licences
        receipt["engine_licence"] = _licences.licence_of(engine)
        stored = {**receipt, "user_id": user_id}
        _store.add(stored)
        sb = _get_supabase() if cloud else None
        if sb is not None:
            try:
                sb.table("receipts").insert({k: v for k, v in stored.items()}).execute()
            except Exception as e:  # noqa: BLE001 - the local copy still exists
                log.warning("could not store receipt in Supabase (apply migration 008_receipts.sql): %s", e)
        return receipt
    except Exception as e:  # noqa: BLE001
        log.warning("could not create a receipt: %s", e)
        return {}


# ---------------------------------------------------------------------------------------------------------------
# Labels in the file (metadata)
# ---------------------------------------------------------------------------------------------------------------

LABEL_SOFTWARE = "BonClay"


def label_text(receipt_id: str) -> str:
    """The text written into an exported file: states that the voice is AI-generated and names the receipt."""
    return f"AI-generated voice \u00b7 BonClay \u00b7 receipt {receipt_id}"


def _info_chunk(tag: bytes, text: str) -> bytes:
    data = text.encode("utf-8") + b"\x00"  # INFO strings are NUL-terminated
    chunk = tag + struct.pack("<I", len(data)) + data
    return chunk + (b"\x00" if len(data) % 2 else b"")  # chunks are word aligned


def label_wav(wav_bytes: bytes, receipt_id: str) -> bytes:
    """Return the WAV with a RIFF LIST/INFO chunk saying it is AI-generated: ICMT = "AI-generated voice · BonClay ·
    receipt <id>", ISFT = "BonClay". The chunk is appended after the audio data, so the samples (and the watermark in
    them) are byte for byte unchanged. Anything that is not a plain RIFF/WAVE file, or that already has an INFO
    list, is returned untouched. Never raises."""
    try:
        b = bytes(wav_bytes)
        if len(b) < 12 or b[:4] != b"RIFF" or b[8:12] != b"WAVE" or b[12:16] != b"fmt ":
            return wav_bytes
        riff_end = min(len(b), 8 + struct.unpack("<I", b[4:8])[0])
        pos = 12
        while pos + 8 <= riff_end:  # walk the chunks: refuse a file that already carries a LIST INFO block
            cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
            if cid == b"LIST" and b[pos + 8:pos + 12] == b"INFO":
                return wav_bytes
            pos += 8 + size + (size & 1)
        if pos != riff_end and pos != len(b):  # trailing bytes we cannot account for
            return wav_bytes
        body = b"INFO" + _info_chunk(b"ICMT", label_text(receipt_id)) + _info_chunk(b"ISFT", LABEL_SOFTWARE)
        chunk = b"LIST" + struct.pack("<I", len(body)) + body
        base = b[:riff_end]
        if len(base) % 2:
            base += b"\x00"
        out = bytearray(base + chunk)
        out[4:8] = struct.pack("<I", len(out) - 8)
        return bytes(out)
    except Exception as e:  # noqa: BLE001 - labelling must never break generation
        log.warning("could not write the AI label into the WAV: %s", e)
        return wav_bytes


def read_wav_label(wav_bytes: bytes) -> Dict[str, str]:
    """The INFO tags of a WAV as {"ICMT": ..., "ISFT": ...} (empty when there are none). Used by tests and tools."""
    out: Dict[str, str] = {}
    b = bytes(wav_bytes)
    if len(b) < 12 or b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        return out
    pos = 12
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        if cid == b"LIST" and b[pos + 8:pos + 12] == b"INFO":
            i, end = pos + 12, pos + 8 + size
            while i + 8 <= end:
                tag, n = b[i:i + 4].decode("ascii", "replace"), struct.unpack("<I", b[i + 4:i + 8])[0]
                out[tag] = b[i + 8:i + 8 + n].rstrip(b"\x00").decode("utf-8", "replace")
                i += 8 + n + (n & 1)
        pos += 8 + size + (size & 1)
    return out


def _with_c2pa_manifest(wav_bytes: bytes, receipt_id: str, engine: Optional[str], voice_id: Optional[str],
                        message: Optional[int]) -> bytes:
    """The WAV with a C2PA manifest embedded when C2PA_MANIFEST=on and c2pa-python is installed
    (services/c2pa_manifest.py); the bytes as they were otherwise. Never raises: a manifest is a bonus layer."""
    try:
        from services import c2pa_manifest
    except ImportError:  # pragma: no cover
        from backend.services import c2pa_manifest
    if not c2pa_manifest.active():
        return wav_bytes
    try:
        return c2pa_manifest.embed(wav_bytes, receipt_id, engine=engine, voice_id=voice_id, key_id=key_id(),
                                   watermark="audioseal_16bits" if message is not None else None, mime="audio/wav")
    except Exception as e:  # noqa: BLE001
        log.warning("could not embed a C2PA manifest (the clip is still watermarked and receipted): %s", e)
        return wav_bytes


def apply_provenance(*, kind: str, subject_id: str, wav_bytes: bytes, text: str, engine: Optional[str],
                     voice_id: Optional[str], voice_name: Optional[str], consent_id: Optional[str],
                     consent_verified: Optional[bool], user_id: Optional[str], cloud: bool = False,
                     ) -> Tuple[bytes, dict]:
    """Watermark a WAV (bytes), label it (RIFF INFO metadata) and write its receipt. Returns (wav bytes to store and
    serve, receipt or {}).

    Falls back to the original bytes when watermarking is unavailable; the receipt then has watermark_message=None."""
    out_bytes = wav_bytes
    message: Optional[int] = None
    receipt_id = str(uuid.uuid4())
    if unavailable_reason() is None:
        try:
            import io
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
            marked, info = watermark_ex(audio, int(sr), message_for(subject_id))
            if info["applied"]:
                buf = io.BytesIO()
                sf.write(buf, marked, int(sr), format="WAV")
                out_bytes, message = buf.getvalue(), info["message"]
        except Exception as e:  # noqa: BLE001
            log.warning("could not watermark generated audio: %s", e)
    unlabelled = out_bytes
    out_bytes = label_wav(out_bytes, receipt_id)  # before hashing: the receipt matches the file as delivered
    out_bytes = _with_c2pa_manifest(out_bytes, receipt_id, engine, voice_id, message)  # also before hashing
    receipt = create_receipt(kind=kind, subject_id=subject_id, engine=engine, voice_id=voice_id,
                             voice_name=voice_name, consent_id=consent_id, consent_verified=consent_verified,
                             text=text, audio_bytes=out_bytes, watermark_message=message, user_id=user_id,
                             cloud=cloud, receipt_id=receipt_id)
    if not receipt:  # no receipt was written, so the label would name one that does not exist
        return unlabelled, receipt
    return out_bytes, receipt


# ---------------------------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------------------------

def verify_clip(raw_bytes: bytes, wav: np.ndarray, sr: int) -> dict:
    """The /api/verify result for an uploaded clip (raw_bytes: the file as uploaded; wav, sr: its decoded audio)."""
    exact = find_by_audio_sha(sha256_hex(raw_bytes))
    wm = detect(wav, sr)
    receipt: Optional[dict] = None
    candidates: List[dict] = []
    if exact:
        receipt = exact[0]
    elif wm["detected"] and wm["message"] is not None:
        candidates = find_by_message(wm["message"])
        if len(candidates) == 1:
            receipt = candidates[0]
    valid = signature_valid(receipt)
    if receipt is not None and valid:
        verdict = "made_with_bonclay"
    elif candidates and len(candidates) > 1 and any(signature_valid(c) for c in candidates):
        verdict = "made_with_bonclay"
    elif wm["detected"]:
        verdict = "watermark_only"
    else:
        verdict = "not_detected"
    watermark_out = {"detected": wm["detected"], "probability": wm["probability"]}
    if wm.get("reason"):
        watermark_out["reason"] = wm["reason"]
    try:
        from services import c2pa_manifest
    except ImportError:  # pragma: no cover
        from backend.services import c2pa_manifest
    manifest_out = c2pa_manifest.read_summary(raw_bytes)   # None without the library; the second layer when present
    return {
        "watermark": watermark_out,
        "manifest": manifest_out,
        "match": {
            "exact": bool(exact),
            "receipt": public_receipt(receipt) if receipt else None,
            "signature_valid": valid,
            "candidates": [public_receipt(c) for c in candidates[:5]] if len(candidates) > 1 else [],
        },
        "verdict": verdict,
    }
