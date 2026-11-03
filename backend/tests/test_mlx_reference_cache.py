"""The MLX clone worker's reference cache (services/mlx_worker.install_reference_cache): the speech-tokenizer codes
and the speaker embedding of a reference clip are computed once per clip, not once per generation."""
import numpy as np

from services import mlx_worker


class _Tokenizer:
    def __init__(self):
        self.calls = 0

    def encode(self, audio):
        self.calls += 1
        return ("codes", float(np.asarray(audio).sum()))


class _Model:
    def __init__(self):
        self.speech_tokenizer = _Tokenizer()
        self.spk_calls = 0

    def extract_speaker_embedding(self, audio, sr=24000):
        self.spk_calls += 1
        return ("spk", float(np.asarray(audio).mean()), sr)


def test_the_same_clip_is_encoded_once():
    m = _Model()
    evaluated = []
    cache = mlx_worker.install_reference_cache(m, size=4, evaluate=evaluated.append)
    clip = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)
    first = m.speech_tokenizer.encode(clip)
    again = m.speech_tokenizer.encode(clip.copy())            # same samples, a different array object
    assert first == again and m.speech_tokenizer.calls == 1
    assert m.extract_speaker_embedding(clip) == m.extract_speaker_embedding(clip) and m.spk_calls == 1
    assert len(cache) == 2 and len(evaluated) == 2          # each fresh result was forced once


def test_a_different_clip_or_argument_is_not_confused():
    m = _Model()
    mlx_worker.install_reference_cache(m, size=4, evaluate=lambda *_: None)
    a = np.ones(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32) * 0.5
    assert m.speech_tokenizer.encode(a) != m.speech_tokenizer.encode(b) and m.speech_tokenizer.calls == 2
    assert m.extract_speaker_embedding(a, sr=24000)[2] == 24000
    assert m.extract_speaker_embedding(a, sr=16000)[2] == 16000 and m.spk_calls == 2


def test_the_cache_is_bounded_least_recently_used():
    m = _Model()
    cache = mlx_worker.install_reference_cache(m, size=2, evaluate=lambda *_: None)
    clips = [np.full(10, i, dtype=np.float32) for i in range(3)]
    for c in clips:
        m.speech_tokenizer.encode(c)
    assert len(cache) == 2 and m.speech_tokenizer.calls == 3
    m.speech_tokenizer.encode(clips[0])                       # evicted earliest: computed again
    assert m.speech_tokenizer.calls == 4


def test_a_model_without_those_methods_is_left_alone():
    class Bare:
        pass
    cache = mlx_worker.install_reference_cache(Bare(), evaluate=lambda *_: None)
    assert cache == {}
