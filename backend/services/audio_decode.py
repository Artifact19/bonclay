"""Decode any uploaded audio file to mono float32 at a chosen sample rate.

librosa 1.0 reads files through soundfile only (WAV, FLAC, OGG/Vorbis, MP3), so formats that phones and browsers
produce, such as M4A/AAC and WebM/Opus, fail there. Those fall back to ffmpeg, which reads almost anything.
"""
import shutil
import subprocess

import numpy as np


def decode_file(path: str, sr: int = 16000) -> np.ndarray:
    """Mono float32 samples at `sr`. Raises ValueError when the file cannot be decoded."""
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True)
        if audio.size:
            return audio.astype(np.float32)
    except Exception:  # noqa: BLE001  try ffmpeg next
        pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("cannot decode this audio format without ffmpeg (install it: brew install ffmpeg)")
    proc = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                          capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(f"ffmpeg could not decode the audio: {proc.stderr.decode(errors='replace')[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
