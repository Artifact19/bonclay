"""One Whisper pass on a short encoder window (services/asr_window.py, the `single` strategy). The model is never loaded:
only the window choice, the re-run rule and the strategy plumbing are tested."""
import pytest

from services import asr_window, mlx_asr, mlx_asr_worker
from services.mlx_asr_worker import transcribe_plan

FAST, ACCURATE = "small", "turbo"


@pytest.mark.parametrize("seconds,expected", [(0.5, 10), (3.0, 10), (8.5, 10), (8.6, 15), (13.5, 15), (13.6, 20),
                                              (18.5, 20), (18.6, 30), (28.5, 30), (60, 30)])
def test_pick_bucket_keeps_a_margin_after_the_speech(seconds, expected):
    assert asr_window.pick_bucket(seconds) == expected


def test_pick_bucket_honours_a_custom_bucket_list():
    assert asr_window.pick_bucket(5.0, (8, 30)) == 8
    assert asr_window.pick_bucket(5.0, (30,)) == 30


@pytest.mark.parametrize("text,ratio,logprob,seconds,bad", [
    ("hello there", 1.2, -0.3, 4.0, False),
    ("", 0.0, -0.1, 4.0, True),             # silence in a clip long enough to hold speech
    ("", 0.0, -0.1, 0.2, False),            # a blip: nothing to recover
    ("   ", 0.0, -0.1, 4.0, True),
    ("ha ha ha ha ha ha", 3.1, -0.3, 4.0, True),     # repetition
    ("something", 1.5, -1.4, 4.0, True),             # low confidence
    ("something", 2.4, -1.0, 4.0, False),            # exactly on the limits is fine
])
def test_looks_bad(text, ratio, logprob, seconds, bad):
    assert asr_window.looks_bad(text, ratio, logprob, seconds) is bad


def test_single_is_a_known_strategy_on_both_sides_and_the_default():
    assert "single" in mlx_asr.STRATEGIES and "single" in mlx_asr_worker.STRATEGIES
    assert mlx_asr.DEFAULT_STRATEGY == "single"


def test_env_can_still_choose_the_older_strategies(monkeypatch):
    monkeypatch.delenv("ASR_STRATEGY", raising=False)
    monkeypatch.delenv("MLX_ASR_STRATEGY", raising=False)
    assert mlx_asr.resolve_strategy() == "single"
    monkeypatch.setenv("MLX_ASR_STRATEGY", "auto")
    assert mlx_asr.resolve_strategy() == "auto"


class Fakes:
    def __init__(self, window_result):
        self.window_result, self.runs, self.detects, self.windows = window_result, [], [], []

    def run(self, path, repo, language):
        self.runs.append((repo, language))
        return {"text": "full", "language": language or "en"}

    def detect(self, path, repo):
        self.detects.append(repo)
        return "en"

    def window(self, path, repo, language):
        self.windows.append((repo, language))
        return self.window_result


@pytest.mark.parametrize("hint", [None, "en", "hi"])
def test_single_is_one_window_pass_with_no_small_model_and_no_language_detection(hint):
    f = Fakes({"text": "short", "language": "hi", "window": 10, "fallback": False})
    out, model = transcribe_plan("single", hint, "wave", f.run, f.detect, FAST, ACCURATE, f.window)
    assert out["text"] == "short" and model == ACCURATE
    assert f.windows == [(ACCURATE, hint)] and f.runs == [] and f.detects == []


def test_single_uses_the_normal_large_pass_when_the_clip_does_not_fit_a_window():
    f = Fakes(None)
    out, model = transcribe_plan("single", None, "wave", f.run, f.detect, FAST, ACCURATE, f.window)
    assert out["text"] == "full" and model == ACCURATE
    assert f.runs == [(ACCURATE, None)] and f.detects == []


def test_single_without_a_window_function_is_the_accurate_strategy():
    f = Fakes(None)
    _, model = transcribe_plan("single", None, "wave", f.run, f.detect, FAST, ACCURATE)
    assert model == ACCURATE and f.runs == [(ACCURATE, None)]


def test_other_strategies_ignore_the_window_function():
    f = Fakes({"text": "short"})
    transcribe_plan("auto", None, "wave", f.run, f.detect, FAST, ACCURATE, f.window)
    assert f.windows == []
