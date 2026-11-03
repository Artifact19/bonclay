"""One Whisper pass on a shorter encoder window (docs/OPTIMIZATION_RESEARCH.md, idea 3; measured in docs/RESULTS.md).

whisper-large-v3-turbo's encoder always sees 30 s (1,500 frames), so a 5 s question pays for 30 s. Here the mel is padded
only to the smallest of 10/15/20/30 s that holds the clip plus a margin, and the positional embedding is sliced to
match. The language is detected from the encoder output the decoder already has, so no separate whisper-small
language-id pass is needed. If the result looks wrong (repetition, low confidence, empty text) the clip is decoded again
on the full 30 s window.

Needs mlx-whisper, so it runs in backend/.venv-mlx (imported by services/mlx_asr_worker.py). The pure helpers
(pick_bucket, looks_bad) import nothing heavy and are unit-tested without MLX."""
from typing import Optional, Sequence

BUCKETS = (10, 15, 20, 30)      # seconds; fixed sizes keep MLX's shape-specialised kernels warm. No 8 s window: it
                                # raised Hindi CER by 2.2 points in the 40-clip check, 10 s by 0.4 (docs/RESULTS.md)
MARGIN_S = 1.5                  # silence kept after the speech inside the window
SAMPLE_RATE = 16000
MAX_COMPRESSION_RATIO = 2.4     # above this the text is repetitive (Whisper's own hallucination check)
MIN_AVG_LOGPROB = -1.0


def pick_bucket(seconds: float, buckets: Sequence[int] = BUCKETS, margin: float = MARGIN_S) -> int:
    """Smallest window (s) that holds `seconds` of speech plus the margin; the largest one when nothing does."""
    for b in buckets:
        if seconds + margin <= b:
            return b
    return buckets[-1]


def looks_bad(text: str, compression_ratio: float, avg_logprob: float, seconds: float) -> bool:
    """True when a short-window result should be re-run on the full window."""
    if not (text or "").strip():
        return seconds >= 0.5       # nothing heard in a clip long enough to hold speech
    return compression_ratio > MAX_COMPRESSION_RATIO or avg_logprob < MIN_AVG_LOGPROB


_patched = False


def patch_encoder() -> None:
    """Let AudioEncoder accept fewer than 1,500 frames: add only the first T positional embeddings. For a full 30 s
    input this is the original behaviour. Idempotent."""
    global _patched
    if _patched:
        return
    import mlx.nn as nn
    from mlx_whisper.whisper import AudioEncoder

    def call(self, x):
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        x = x + self._positional_embedding[: x.shape[1]]
        for block in self.blocks:
            x, _, _ = block(x)
        return self.ln_post(x)

    AudioEncoder.__call__ = call

    # Whisper.detect_language only recognises already-encoded audio when it has exactly 1,500 frames and otherwise runs the
    # encoder again on it. Recognise encoded features of any length (last dimension = model width, not mel bins).
    import mlx.core as mx
    import numpy as np
    from mlx_whisper.tokenizer import get_tokenizer
    from mlx_whisper.whisper import Whisper

    original = Whisper.detect_language

    def detect_language(self, mel, tokenizer=None):
        if mel.ndim != 3 or mel.shape[-1] != self.dims.n_audio_state:
            return original(self, mel, tokenizer)
        tokenizer = tokenizer or get_tokenizer(self.is_multilingual, num_languages=self.num_languages)
        x = mx.array([[tokenizer.sot]] * mel.shape[0])
        logits = self.logits(x, mel)[:, 0]
        mask = mx.full(logits.shape[-1], -mx.inf, dtype=mx.float32)
        mask[list(tokenizer.all_language_tokens)] = 0.0
        logits = logits + mask
        probs = np.array(mx.softmax(logits, axis=-1))
        codes = list(zip(tokenizer.all_language_tokens, tokenizer.all_language_codes))
        return mx.argmax(logits, axis=-1), [{c: probs[i, j].item() for j, c in codes} for i in range(mel.shape[0])]

    Whisper.detect_language = detect_language
    _patched = True


def _decode(model, audio, window: int, language: Optional[str], prompt: Optional[str] = None):
    import mlx.core as mx
    from mlx_whisper.audio import SAMPLE_RATE as _SR, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions, decode

    frames = window * 100  # the mel has 100 frames per second
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=window * _SR)
    mel = pad_or_trim(mel, frames, axis=-2).astype(mx.float16)
    return decode(model, mel, DecodingOptions(language=language, without_timestamps=True, fp16=True,
                                              **({"prompt": prompt} if prompt else {})))


def transcribe_window(repo: str, audio, language: Optional[str] = None, buckets: Sequence[int] = BUCKETS,
                      fallback: bool = True, prompt: Optional[str] = None) -> dict:
    """Transcribe `audio` (16 kHz float32 waveform) with one decode on the shortest suitable window.

    Returns {"text", "language", "window", "fallback"}; `window` is the window in seconds that produced the text and
    `fallback` is True when the short window was rejected and the 30 s window was used instead. Audio longer than the
    largest bucket is not handled here: the caller should use the normal transcribe path for it."""
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    patch_encoder()
    model = ModelHolder.get_model(repo, mx.float16)
    seconds = len(audio) / SAMPLE_RATE
    window = pick_bucket(seconds, buckets)
    res = _decode(model, audio, window, language, prompt)
    fell_back = False
    if fallback and window < 30 and looks_bad(res.text, res.compression_ratio, res.avg_logprob, seconds):
        window, fell_back = 30, True
        res = _decode(model, audio, 30, language, prompt)
    return {"text": res.text.strip(), "language": res.language, "window": window, "fallback": fell_back,
            "segments": [{"start": 0.0, "end": float(seconds), "text": res.text.strip(),
                          "avg_logprob": float(res.avg_logprob), "no_speech_prob": float(res.no_speech_prob),
                          "compression_ratio": float(res.compression_ratio)}]}
