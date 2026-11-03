"""MOS stub: deterministic quality proxy so evaluation plumbing is testable without a MOS model.

Uses a real evaluator if one exposes `mos_score(samples, sr)` (services.evaluator /
routers.evaluator); otherwise the SNR-based stub below.
"""
import importlib
import wave

import numpy as np


def _stub_mos(samples: np.ndarray, sr: int) -> float:
    """Map a frame-level dynamic-range SNR proxy to the 1..5 MOS range."""
    x = samples.astype(np.float64)
    frame = max(sr // 50, 1)
    n = x.size // frame
    if n == 0 or not np.any(x):
        return 1.0
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    snr = 20 * np.log10((np.percentile(rms, 90) + 1e-9) / (np.percentile(rms, 10) + 1e-9))
    return float(np.clip(1 + snr / 12, 1.0, 5.0))


def _mos():
    for name in ("services.evaluator", "routers.evaluator"):
        try:
            return importlib.import_module(name).mos_score
        except Exception:
            continue
    return _stub_mos


def load(path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768, sr


def test_fixture_is_5s_24k(ref_wav):
    x, sr = load(ref_wav)
    assert sr == 24000 and abs(len(x) / sr - 5.0) < 0.01


def test_mos_in_range(ref_wav):
    x, sr = load(ref_wav)
    assert 1.0 <= _mos()(x, sr) <= 5.0


def test_silence_scores_minimum():
    assert _mos()(np.zeros(24000, dtype=np.float32), 24000) == 1.0


def test_noise_scores_lower_than_clean(ref_wav):
    x, sr = load(ref_wav)
    clean = np.concatenate([x, np.zeros(sr, dtype=np.float32)])
    noisy = x + np.random.default_rng(0).normal(0, 0.5, x.shape).astype(np.float32)
    assert _mos()(clean, sr) >= _mos()(noisy, sr)


def test_deterministic(ref_wav):
    x, sr = load(ref_wav)
    assert _mos()(x, sr) == _mos()(x, sr)
