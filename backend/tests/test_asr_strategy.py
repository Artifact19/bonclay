"""Local speech recognition strategy selection: which MLX Whisper model(s) a request uses. The model calls are faked, so
nothing is loaded and no worker is started."""
import pytest

from services import mlx_asr, mlx_asr_worker
from services.mlx_asr_worker import INDIAN, transcribe_plan

FAST, ACCURATE = "small", "turbo"


class Fakes:
    """What each fake model 'hears' is set per test: `heard` is the language the small and large model detect."""

    def __init__(self, heard="en"):
        self.heard = heard
        self.runs = []      # (repo, language) per transcription
        self.detects = []   # repo per language-id-only call

    def run(self, path, repo, language):
        self.runs.append((repo, language))
        return {"text": f"{repo}:{path}", "language": language or self.heard}

    def detect(self, path, repo):
        self.detects.append(repo)
        return self.heard


def plan(strategy, language, heard="en"):
    f = Fakes(heard)
    out, model = transcribe_plan(strategy, language, "clip.wav", f.run, f.detect, FAST, ACCURATE)
    return f, out, model


# --- hybrid: small first, large again only for an Indian language ------------------------------------------------------

def test_hybrid_english_is_one_small_pass():
    f, out, model = plan("hybrid", None, heard="en")
    assert f.runs == [(FAST, None)] and f.detects == [] and model == FAST


def test_hybrid_indian_language_runs_small_then_large_with_the_detected_language():
    f, out, model = plan("hybrid", None, heard="hi")
    assert f.runs == [(FAST, None), (ACCURATE, "hi")] and model == ACCURATE
    assert out["text"].startswith(ACCURATE)


# --- accurate: one large pass, language left to the model ------------------------------------------------------------

@pytest.mark.parametrize("heard", ["en", "hi"])
def test_accurate_is_one_large_pass_without_a_hint(heard):
    f, _, model = plan("accurate", None, heard=heard)
    assert f.runs == [(ACCURATE, None)] and f.detects == [] and model == ACCURATE


# --- auto: the small model only detects; one transcription pass after that -------------------------------------------

def test_auto_hindi_is_detection_plus_one_large_pass_and_never_decodes_with_the_small_model():
    f, _, model = plan("auto", None, heard="hi")
    assert f.detects == [FAST]
    assert f.runs == [(ACCURATE, "hi")] and model == ACCURATE


def test_auto_english_is_detection_plus_one_small_pass():
    f, _, model = plan("auto", None, heard="en")
    assert f.detects == [FAST]
    assert f.runs == [(FAST, "en")] and model == FAST


@pytest.mark.parametrize("lang", sorted(INDIAN))
def test_auto_sends_every_indian_language_to_the_large_model(lang):
    f, _, model = plan("auto", None, heard=lang)
    assert f.runs == [(ACCURATE, lang)] and model == ACCURATE


@pytest.mark.parametrize("lang", ["fr", "es", "de", "ja"])
def test_auto_keeps_other_languages_on_the_small_model(lang):
    f, _, model = plan("auto", None, heard=lang)
    assert f.runs == [(FAST, lang)] and model == FAST


# --- a caller's language hint ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["hybrid", "accurate", "auto", "fast"])
@pytest.mark.parametrize("hint", ["hi", "ta", "bn"])
def test_indian_hint_is_always_one_large_pass_with_no_detection(strategy, hint):
    f, _, model = plan(strategy, hint, heard="en")
    assert f.runs == [(ACCURATE, hint)] and f.detects == [] and model == ACCURATE


@pytest.mark.parametrize("strategy", ["hybrid", "auto", "fast"])
def test_english_hint_is_one_small_pass_with_no_detection(strategy):
    f, _, model = plan(strategy, "en", heard="hi")
    assert f.runs == [(FAST, "en")] and f.detects == [] and model == FAST


def test_english_hint_with_accurate_uses_the_large_model():
    f, _, model = plan("accurate", "en")
    assert f.runs == [(ACCURATE, "en")] and model == ACCURATE


# --- fast and unknown ------------------------------------------------------------------------------------------------

def test_fast_never_uses_the_large_model_for_unhinted_audio_even_if_it_hears_hindi():
    f, _, model = plan("fast", None, heard="hi")
    assert f.runs == [(FAST, None)] and model == FAST


@pytest.mark.parametrize("strategy", ["", None, "turbo", "HYBRID"])
def test_unknown_strategy_behaves_like_hybrid(strategy):
    f, _, model = plan(strategy, None, heard="hi")
    assert f.runs == [(FAST, None), (ACCURATE, "hi")] and model == ACCURATE


def test_the_path_is_passed_through_untouched():
    """auto hands the plan an already-decoded waveform instead of a path; the plan must not look at it."""
    seen = []
    wave = object()
    transcribe_plan("auto", None, wave, lambda p, r, l: seen.append(p) or {"text": "", "language": l},
                    lambda p, r: seen.append(p) or "hi", FAST, ACCURATE)
    assert seen == [wave, wave]


def test_worker_and_client_agree_on_the_strategy_names():
    assert tuple(mlx_asr_worker.STRATEGIES) == tuple(mlx_asr.STRATEGIES)
    assert mlx_asr.DEFAULT_STRATEGY in mlx_asr.STRATEGIES


# --- the environment variable ----------------------------------------------------------------------------------------

@pytest.fixture()
def clean_env(monkeypatch):
    monkeypatch.delenv("MLX_ASR_STRATEGY", raising=False)
    monkeypatch.delenv("ASR_STRATEGY", raising=False)
    return monkeypatch


def test_default_strategy(clean_env):
    assert mlx_asr.resolve_strategy() == mlx_asr.DEFAULT_STRATEGY


@pytest.mark.parametrize("value,expected", [("hybrid", "hybrid"), ("accurate", "accurate"), ("auto", "auto"),
                                            ("FAST", "fast"), ("  Auto ", "auto")])
def test_env_selects_a_strategy(clean_env, value, expected):
    clean_env.setenv("MLX_ASR_STRATEGY", value)
    assert mlx_asr.resolve_strategy() == expected


def test_older_env_name_still_works_and_the_new_one_wins(clean_env):
    clean_env.setenv("ASR_STRATEGY", "accurate")
    assert mlx_asr.resolve_strategy() == "accurate"
    clean_env.setenv("MLX_ASR_STRATEGY", "auto")
    assert mlx_asr.resolve_strategy() == "auto"


def test_unknown_env_value_falls_back_to_the_default(clean_env):
    clean_env.setenv("MLX_ASR_STRATEGY", "turbo-max")
    assert mlx_asr.resolve_strategy() == mlx_asr.DEFAULT_STRATEGY


def test_client_sends_the_resolved_strategy_to_the_worker(clean_env, monkeypatch):
    """mlx_asr.transcribe puts the strategy and language in the request body; no worker is started."""
    import io
    import json

    sent = {}

    class FakeWorker:
        def ensure(self):
            return 1

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        sent.update(json.loads(req.data))
        return Resp(b'{"text": "hi", "language": "hi", "model": "turbo", "seconds": 1}')

    monkeypatch.setattr(mlx_asr, "_worker", FakeWorker())
    monkeypatch.setattr(mlx_asr.urllib.request, "urlopen", fake_urlopen)
    clean_env.setenv("MLX_ASR_STRATEGY", "auto")
    assert mlx_asr.transcribe("a.wav", "hi")["text"] == "hi"
    assert sent == {"path": "a.wav", "language": "hi", "strategy": "auto"}
