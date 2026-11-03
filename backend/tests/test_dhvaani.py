"""DhVaani-0.5 (ARTPARK-IISc, voice cloning in Indian languages): availability, the routing table (language x voice type x
availability x INDIC_CLONE_ENGINE), choosing the 3-10 s reference clip and its transcript, the cached reference pair, the
worker client against a fake HTTP worker, generation, receipts, voice deletion, the guarantee that a live agent turn never
uses it, and the speed estimate. No model is loaded and no worker process is started."""
import importlib.util
import io
import json
import os
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from test_offline_agents import LOCAL, PERSONA, env, make_agent, FakeEmbedding  # noqa: F401  (env is a fixture)

dh = pytest.importorskip("services.dhvaani_clone")
from services import llm  # noqa: E402
from services import chatterbox_clone as cbs  # noqa: E402

SR = 24000


@pytest.fixture()
def backend_mod(app_client):
    import backend

    backend.limiter.reset()
    return backend


def avail(monkeypatch, dhvaani=None, chatterbox=None, f5=None):
    """Switch engines on (True) or off (False); None leaves the suite's default (everything off)."""
    if dhvaani is not None:
        monkeypatch.setattr(dh, "unavailable_reason", lambda: None if dhvaani else "DhVaani is not set up")
    if chatterbox is not None:
        monkeypatch.setattr(cbs, "unavailable_reason", lambda: None if chatterbox else "not set up")
    if f5 is not None:
        import backend

        monkeypatch.setattr(backend, "indic_f5_unavailable_reason", lambda: None if f5 else "no f5_tts")


# ---- synthetic speech: tone bursts with silences ---------------------------------------------------------------

def burst(seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    wobble = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)  # a syllable rhythm that never reaches silence
    return (amp * (np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 540 * t)) * wobble / 1.5).astype(np.float32)


def gap(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def wav_bytes(audio, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


# ---- availability ----------------------------------------------------------------------------------------------

def test_off_switch_wins(monkeypatch):
    monkeypatch.setenv("DHVAANI", "off")
    assert "DHVAANI=off" in dh.unavailable_reason()
    monkeypatch.setenv("DHVAANI", "False")
    assert dh.unavailable_reason() is not None


def test_missing_venv_python_says_how_to_set_up(monkeypatch, tmp_path):
    monkeypatch.setenv("DHVAANI", "")
    monkeypatch.setenv("DHVAANI_PYTHON", str(tmp_path / "nope" / "python"))
    monkeypatch.setattr(dh, "BACKEND_DIR", str(tmp_path))
    assert dh.dhvaani_python() is None
    assert "./bonclay setup --dhvaani" in dh.unavailable_reason()


def test_weights_not_cached_is_reported(monkeypatch, tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    monkeypatch.setenv("DHVAANI", "")
    monkeypatch.setenv("DHVAANI_PYTHON", str(py))
    monkeypatch.setattr(dh, "weights_cached", lambda model=None: False)
    assert "weights are not downloaded" in dh.unavailable_reason()
    monkeypatch.setattr(dh, "weights_cached", lambda model=None: True)
    assert dh.unavailable_reason() is None and dh.available() is True


def test_weights_cached_looks_for_the_model_files(monkeypatch):
    import huggingface_hub

    seen = []

    def fake(repo, filename):
        seen.append((repo, filename))
        return None if filename == "tokens.txt" else "/cache/" + filename

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake)
    assert dh.weights_cached() is False
    assert {f for _, f in seen} <= set(dh._WEIGHT_FILES) and seen[0][0] == "ARTPARK-IISc/DhVaani-0.5"


def test_language_support():
    for code in ("hi", "mr", "ta", "bn", "te", "kn", "ml", "gu", "pa", "or", "ur", "as", "mr-IN", "TA"):
        assert dh.supports(code), code
    for code in ("en", "fr", "ja", "zh", "sa", "ks", "doi", "", None):  # Qwen3's languages and the weak ones
        assert not dh.supports(code), code


def test_free_memory_gate():
    msg = dh.memory_problem(free_mb=600, minimum_mb=1500)
    assert msg and "0.6 GB" in msg and "1.5 GB" in msg and "Close other apps" in msg
    assert dh.memory_problem(free_mb=4000, minimum_mb=1500) is None
    assert dh.memory_problem(free_mb=None, minimum_mb=1500) is None or dh.free_memory_mb() is not None
    assert dh.memory_problem(free_mb=10, minimum_mb=0) is None  # DHVAANI_MIN_FREE_MB=0 turns the gate off


# ---- routing table: language x voice type x availability x INDIC_CLONE_ENGINE --------------------------------------

ERR = "ENGINE_UNAVAILABLE"


@pytest.mark.parametrize("lang,dhv,cb,f5,pref,expected", [
    # Hindi: DhVaani first, Chatterbox the fallback, IndicF5 the last resort
    ("hi", True, True, False, "auto", "dhvaani"),
    ("hi", True, True, True, "auto", "dhvaani"),
    ("hi", False, True, False, "auto", "chatterbox_ml"),
    ("hi", False, False, True, "auto", "indic_f5"),
    ("hi", False, False, False, "auto", ERR),
    # INDIC_CLONE_ENGINE=chatterbox: Hindi on Chatterbox only, never DhVaani (an error when Chatterbox cannot run)
    ("hi", True, True, False, "chatterbox", "chatterbox_ml"),
    ("hi", True, False, False, "chatterbox", ERR),
    ("hi", True, False, True, "chatterbox", ERR),
    # INDIC_CLONE_ENGINE=dhvaani: DhVaani only, no silent substitute
    ("hi", True, True, False, "dhvaani", "dhvaani"),
    ("hi", False, True, True, "dhvaani", ERR),
    # the other Indian languages: only DhVaani clones them (Chatterbox cannot; IndicF5 is the old optional engine)
    ("mr", True, True, False, "auto", "dhvaani"),
    ("ta", True, False, False, "auto", "dhvaani"),
    ("bn", True, True, False, "chatterbox", "dhvaani"),
    ("te", True, True, False, "dhvaani", "dhvaani"),
    ("kn", True, False, False, "auto", "dhvaani"),
    ("ml", True, False, False, "auto", "dhvaani"),
    ("gu", True, False, False, "auto", "dhvaani"),
    ("pa", True, False, False, "auto", "dhvaani"),
    ("or", True, False, False, "auto", "dhvaani"),
    ("as", True, False, False, "auto", "dhvaani"),
    ("ur", True, False, False, "auto", "dhvaani"),
    ("mr", False, True, False, "auto", ERR),
    ("mr", False, False, True, "auto", "indic_f5"),
    ("mr", False, False, True, "dhvaani", ERR),
    ("ur", False, True, True, "auto", ERR),
    # Qwen3 keeps its ten languages whatever is installed
    ("en", True, True, True, "auto", "qwen3_0_6b"),
    ("fr", True, True, True, "dhvaani", "qwen3_0_6b"),
    ("ja", True, True, True, "chatterbox", "qwen3_0_6b"),
    ("de", False, False, False, "auto", "qwen3_0_6b"),
])
def test_cloned_voice_routing_table(backend_mod, monkeypatch, lang, dhv, cb, f5, pref, expected):
    from core.errors import AppError

    avail(monkeypatch, dhvaani=dhv, chatterbox=cb, f5=f5)
    monkeypatch.setenv("INDIC_CLONE_ENGINE", pref)
    for mode in ("premium", "fast"):
        if expected == ERR:
            with pytest.raises(AppError) as e:
                backend_mod.route_engine(lang, mode, True)
            assert e.value.code == "ENGINE_UNAVAILABLE" and e.value.status == 503
        else:
            assert backend_mod.route_engine(lang, mode, True) == expected


def test_unavailable_dhvaani_error_names_the_setup_command(backend_mod, monkeypatch):
    from core.errors import AppError

    avail(monkeypatch, dhvaani=False, chatterbox=False, f5=False)
    with pytest.raises(AppError) as e:
        backend_mod.route_engine("mr", "premium", True)
    assert "./bonclay setup --dhvaani" in e.value.message or "./bonclay setup --dhvaani" in str(e.value.details)
    assert "Preset voices" in e.value.message
    with pytest.raises(AppError) as e:
        backend_mod.route_engine("hi", "premium", True)
    assert "Chatterbox" in e.value.message  # Hindi can also be cloned with Chatterbox
    monkeypatch.setenv("INDIC_CLONE_ENGINE", "dhvaani")
    with pytest.raises(AppError) as e:
        backend_mod.route_engine("hi", "premium", True)
    assert "Chatterbox" not in e.value.message  # the owner chose DhVaani only: no pointer to another engine


def test_an_unknown_preference_means_auto(backend_mod, monkeypatch):
    avail(monkeypatch, dhvaani=True, chatterbox=True, f5=False)
    monkeypatch.setenv("INDIC_CLONE_ENGINE", "whatever")
    assert backend_mod.indic_clone_preference() == "auto"
    assert backend_mod.route_engine("hi", "premium", True) == "dhvaani"
    monkeypatch.setenv("INDIC_CLONE_ENGINE", " Chatterbox ")
    assert backend_mod.route_engine("hi", "premium", True) == "chatterbox_ml"


@pytest.mark.parametrize("lang", ["hi", "mr", "ta", "bn", "te", "gu", "kn", "ml", "pa", "or", "ur", "as", "en", "fr"])
def test_preset_voices_never_route_to_dhvaani(backend_mod, monkeypatch, lang):
    avail(monkeypatch, dhvaani=True, chatterbox=True, f5=False)
    for mode in ("premium", "fast"):
        assert backend_mod.route_engine(lang, mode, False) != "dhvaani"
    assert backend_mod.route_engine("en", "fast", False) == "kokoro"


# ---- registry, status, hints -----------------------------------------------------------------------------------

def test_registry_entry_matches_the_client(backend_mod):
    entry = backend_mod.ENGINE_REGISTRY["dhvaani"]
    assert entry["type"] == "clone" and entry["hf_id"] == "ARTPARK-IISc/DhVaani-0.5"
    assert entry["languages"] == set(dh.LANGUAGES)
    assert "dhvaani" in backend_mod.CLONE_ENGINES and "dhvaani" in backend_mod.LAZY_PROMPT_ENGINES
    assert "dhvaani" not in backend_mod.PLAYGROUND_CLONE_ENGINES  # the Playground stays on Qwen3


def test_models_status_lists_it_with_availability(app_client, monkeypatch):
    avail(monkeypatch, dhvaani=False)
    engines = {e["id"]: e for e in app_client.get("/api/models/status", headers=LOCAL).json()["engines"]}
    e = engines["dhvaani"]
    assert e["name"] == "DhVaani-0.5 (ARTPARK-IISc)" and e["loaded"] is False
    assert e["available"] is False and e["unavailable_reason"] == "DhVaani is not set up"
    assert e["setup"] == "./bonclay setup --dhvaani"
    avail(monkeypatch, dhvaani=True)
    engines = {e["id"]: e for e in app_client.get("/api/models/status", headers=LOCAL).json()["engines"]}
    assert engines["dhvaani"]["available"] is True and engines["dhvaani"]["unavailable_reason"] is None


def test_setup_hint_when_not_installed(app_client, monkeypatch):
    monkeypatch.setenv("DHVAANI", "")  # the suite's default is "off", which (rightly) hides the hint
    avail(monkeypatch, dhvaani=False)
    hints = {h["id"]: h for h in app_client.get("/api/setup-hints", headers=LOCAL).json()["hints"]}
    assert hints["no_dhvaani"]["severity"] == "info"
    assert hints["no_dhvaani"]["command"] == "./bonclay setup --dhvaani"
    avail(monkeypatch, dhvaani=True)
    assert "no_dhvaani" not in {h["id"] for h in app_client.get("/api/setup-hints", headers=LOCAL).json()["hints"]}


# ---- choosing the reference clip --------------------------------------------------------------------------------

def test_a_short_sample_is_used_whole_with_its_stored_transcript():
    audio = np.concatenate([gap(0.5), burst(6.0), gap(0.5)])
    sel = dh.select_reference(audio, SR, "good morning, you have reached the front desk")
    assert sel.choice.whole and sel.choice.clean_cut
    assert not sel.needs_transcription and sel.text == "good morning, you have reached the front desk"
    assert 6.0 <= sel.choice.duration <= 6.5  # the half-second silences are trimmed to a short margin
    assert abs(len(sel.audio) / SR - sel.choice.duration) < 0.05


def test_a_short_sample_without_a_transcript_needs_one():
    sel = dh.select_reference(np.concatenate([burst(5.0), gap(0.3)]), SR, None)
    assert sel.choice.whole and sel.needs_transcription and sel.text is None and sel.approximate_text is None


def test_a_long_sample_is_cut_between_pauses_into_a_4_to_10_second_clip():
    audio = np.concatenate([burst(5.0), gap(0.45), burst(6.0), gap(0.45), burst(7.0), gap(0.45), burst(5.0)])
    words = " ".join(f"w{i}" for i in range(24))
    sel = dh.select_reference(audio, SR, words)
    c = sel.choice
    assert not c.whole and c.clean_cut
    assert 4.0 <= c.duration <= 10.0
    assert sel.needs_transcription and sel.text is None  # a cut clip has no known transcript...
    assert sel.approximate_text and set(sel.approximate_text.split()) < set(words.split())  # ...only a proportional slice
    edge = int(0.05 * SR)
    assert rms(sel.audio[:edge]) < 0.01 and rms(sel.audio[-edge:]) < 0.01  # both ends sit in silence, not mid-word


def test_the_cut_prefers_a_clip_close_to_eight_seconds():
    audio = np.concatenate([burst(3.0), gap(0.4), burst(8.0), gap(0.4), burst(3.0), gap(0.4), burst(9.0)])
    sel = dh.select_reference(audio, SR, "a b c")
    assert 7.0 <= sel.choice.duration <= 9.0


def test_continuous_speech_without_pauses_is_cut_at_the_quietest_moment():
    audio = burst(25.0)
    sel = dh.select_reference(audio, SR, None)
    assert not sel.choice.whole and not sel.choice.clean_cut
    assert 5.0 <= sel.choice.duration <= 10.0


def test_a_silent_or_too_short_sample_is_refused():
    with pytest.raises(dh.ReferenceError, match="silent"):
        dh.select_reference(np.zeros(SR * 5, dtype=np.float32), SR, "x")
    with pytest.raises(dh.ReferenceError, match="too short"):
        dh.select_reference(burst(0.8), SR, "x")


def test_stereo_samples_are_mixed_down():
    stereo = np.stack([burst(5.0), burst(5.0)], axis=1)
    assert dh.select_reference(stereo, SR, "hello").audio.ndim == 1


def test_proportional_words_follow_the_clip_position():
    choice = dh.ReferenceChoice(start=5.0, end=15.0, whole=False, clean_cut=True)
    text = " ".join(str(i) for i in range(20))
    assert dh.proportional_words(text, choice, 0.0, 20.0).split() == [str(i) for i in range(5, 15)]
    assert dh.proportional_words("", choice, 0.0, 20.0) == ""
    one = dh.proportional_words("only", dh.ReferenceChoice(0.0, 1.0, False, True), 0.0, 30.0)
    assert one == "only"  # never an empty slice for a non-empty transcript


def test_the_prompt_is_the_clip_and_its_transcript():
    p = dh.prompt_from_clip(burst(4.0), SR, "  नमस्ते  ")
    assert p["backend"] == "dhvaani" and p["ref_text"] == "नमस्ते" and p["ref_wav"][:4] == b"RIFF"
    assert dh.is_dhvaani_prompt(p) and not dh.is_dhvaani_prompt({"backend": "chatterbox"}) and not dh.is_dhvaani_prompt(None)
    with pytest.raises(dh.ReferenceError):
        dh.prompt_from_clip(burst(4.0), SR, "   ")
    client = dh.DhVaaniClone.__new__(dh.DhVaaniClone)  # no worker: only the pure method is used
    assert client.create_voice_clone_prompt([(burst(4.0), SR)], ref_text="x")["ref_text"] == "x"
    with pytest.raises(dh.ReferenceError):
        client.create_voice_clone_prompt([(burst(4.0), SR)], x_vector_only_mode=True)  # no transcript-free mode


def test_the_fingerprint_changes_with_the_recording_or_its_transcript():
    a = dh.source_fingerprint(b"audio", "hello")
    assert a == dh.source_fingerprint(b"audio", " hello ")
    assert a != dh.source_fingerprint(b"audio2", "hello") and a != dh.source_fingerprint(b"audio", "hello there")


# ---- the voice's reference pair (stored recording -> cached clip + transcript) ---------------------------------------

class FakeDhvaani:
    """Stands in for DhVaaniClone: same prompt surface, no worker."""
    prompt_backend = dh.PROMPT_BACKEND
    prompt_sample_rate = dh.PROMPT_SAMPLE_RATE
    wants_language_code = True
    engine_key = "dhvaani"

    def __init__(self):
        self.calls = []

    def generate_voice_clone(self, text, language="Auto", voice_clone_prompt=None, **kw):
        self.calls.append((text, language, voice_clone_prompt))
        return [np.zeros(2400, dtype=np.float32)], 24000

    def eval(self):
        return self


def stored_voice(tmp_path, audio, transcript="good morning", language="en", voice_id="v-dh"):
    (tmp_path / f"{voice_id}.wav").write_bytes(wav_bytes(audio))
    return {"id": voice_id, "engine": "qwen3_0_6b", "language": language, "display_name": "Ayush",
            "reference_audio_path": f"local:{voice_id}", "transcript": transcript, "user_id": "local-user"}


def no_asr(env, monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("speech recognition must not run when the transcript is known")

    monkeypatch.setattr(env, "_speech_to_text", boom)


@pytest.mark.asyncio
async def test_short_sample_uses_the_stored_transcript_and_is_cached(env, tmp_path, monkeypatch):
    no_asr(env, monkeypatch)
    row = stored_voice(tmp_path, np.concatenate([gap(0.3), burst(6.0), gap(0.3)]), "good morning, front desk")
    prompt = await env.load_clone_prompt("v-dh", "local-user", "local", row, FakeDhvaani())
    assert prompt["backend"] == "dhvaani" and prompt["ref_text"] == "good morning, front desk"
    audio, sr = sf.read(io.BytesIO(prompt["ref_wav"]))
    assert sr == 24000 and 6.0 <= len(audio) / sr <= 6.7
    assert (tmp_path / "v-dh.dhvaani.wav").is_file() and (tmp_path / "v-dh.dhvaani.json").is_file()
    assert (tmp_path / "v-dh.wav").is_file() and not (tmp_path / "v-dh.pt").exists()  # the voice itself is untouched

    def boom(*a, **k):
        raise AssertionError("the second use must come from the cache")

    monkeypatch.setattr(dh, "select_reference", boom)
    again = await env.load_clone_prompt("v-dh", "local-user", "local", row, FakeDhvaani())
    assert again == prompt


@pytest.mark.asyncio
async def test_cache_is_rebuilt_when_the_transcript_changes(env, tmp_path, monkeypatch):
    no_asr(env, monkeypatch)
    row = stored_voice(tmp_path, burst(5.0), "first words")
    first = await env.load_clone_prompt("v-dh", "local-user", "local", row, FakeDhvaani())
    row["transcript"] = "second words"
    second = await env.load_clone_prompt("v-dh", "local-user", "local", row, FakeDhvaani())
    assert first["ref_text"] == "first words" and second["ref_text"] == "second words"


@pytest.mark.asyncio
async def test_long_sample_is_cut_and_the_cut_is_transcribed(env, tmp_path, monkeypatch):
    audio = np.concatenate([burst(5.0), gap(0.45), burst(6.0), gap(0.45), burst(7.0), gap(0.45), burst(5.0)])
    row = stored_voice(tmp_path, audio, "one two three four five six seven eight nine ten", language="en")
    seen = []

    async def asr(audio_bytes, name, language=None):
        clip, sr = sf.read(io.BytesIO(audio_bytes))
        seen.append((name, language, len(clip) / sr))
        return "what the cut clip says"

    monkeypatch.setattr(env, "_speech_to_text", asr)
    prompt = await env.load_clone_prompt("v-dh", "local-user", "local", row, FakeDhvaani())
    assert prompt["ref_text"] == "what the cut clip says"  # the recognised words, not a slice of the stored text
    clip, sr = sf.read(io.BytesIO(prompt["ref_wav"]))
    assert 4.0 <= len(clip) / sr <= 10.0
    assert seen == [("reference", "en", pytest.approx(len(clip) / sr, abs=0.01))]  # the cut itself was transcribed
    assert (tmp_path / "v-dh.dhvaani.json").is_file()
    record = json.loads((tmp_path / "v-dh.dhvaani.json").read_text(encoding="utf-8"))
    assert record["text"] == "what the cut clip says" and record["whole"] is False


@pytest.mark.asyncio
async def test_without_speech_recognition_a_slice_of_the_stored_transcript_is_used_but_not_cached(env, tmp_path, monkeypatch):
    audio = np.concatenate([burst(5.0), gap(0.45), burst(6.0), gap(0.45), burst(7.0), gap(0.45), burst(5.0)])
    row = stored_voice(tmp_path, audio, " ".join(f"w{i}" for i in range(24)))

    async def asr(*a, **k):
        return None

    monkeypatch.setattr(env, "_speech_to_text", asr)
    prompt = await env.load_clone_prompt("v-dh", "local-user", "local", row, FakeDhvaani())
    assert prompt["ref_text"] and set(prompt["ref_text"].split()) < {f"w{i}" for i in range(24)}
    assert not (tmp_path / "v-dh.dhvaani.json").exists()  # an approximate pair is never kept


@pytest.mark.asyncio
async def test_no_transcript_and_no_recognition_is_a_clear_error(env, tmp_path, monkeypatch):
    from core.errors import AppError

    row = stored_voice(tmp_path, burst(6.0), transcript=None)

    async def asr(*a, **k):
        return ""

    monkeypatch.setattr(env, "_speech_to_text", asr)
    with pytest.raises(AppError) as e:
        await env.load_clone_prompt("v-dh", "local-user", "local", row, FakeDhvaani())
    assert e.value.code == "REFERENCE_TRANSCRIPT_MISSING" and e.value.status == 422


@pytest.mark.asyncio
async def test_a_silent_recording_is_refused_with_a_clear_error(env, tmp_path, monkeypatch):
    from core.errors import AppError

    no_asr(env, monkeypatch)
    row = stored_voice(tmp_path, gap(6.0), "x")
    with pytest.raises(AppError) as e:
        await env.load_clone_prompt("v-dh", "local-user", "local", row, FakeDhvaani())
    assert e.value.code == "REFERENCE_UNUSABLE" and e.value.status == 422


@pytest.mark.asyncio
async def test_missing_recording_is_a_404(env, tmp_path):
    from core.errors import AppError

    row = {"id": "ghost", "engine": "qwen3_0_6b", "reference_audio_path": "local:ghost", "transcript": "x"}
    with pytest.raises(AppError) as e:
        await env.load_clone_prompt("ghost", "local-user", "local", row, FakeDhvaani())
    assert e.value.code == "VOICE_NOT_FOUND"


def test_a_dhvaani_prompt_is_never_handed_to_qwen3_or_chatterbox(env):
    p = {"backend": "dhvaani", "ref_wav": b"x", "ref_text": "t"}
    assert env._is_clone_prompt(p, FakeDhvaani()) is True
    assert env._is_clone_prompt(p, object()) is False  # a Qwen3 model
    assert env._is_clone_prompt({"backend": "chatterbox", "ref_wav": b"x"}, FakeDhvaani()) is False
    assert env._is_clone_prompt({"backend": "qwen-fake"}, FakeDhvaani()) is False


@pytest.mark.asyncio
async def test_clone_chunks_passes_the_language_and_maps_worker_errors(env):
    from core.errors import AppError

    m = FakeDhvaani()
    await env._clone_chunks(m, "नमस्ते। आप कैसे हैं?", {"backend": "dhvaani"}, "mr")
    assert [c[1] for c in m.calls] == ["mr"]

    class Broken(FakeDhvaani):
        def generate_voice_clone(self, *a, **k):
            raise dh.DhVaaniError("worker said no", "GENERATION_TIMEOUT", 504)

    with pytest.raises(AppError) as e:
        await env._clone_chunks(Broken(), "text", {"backend": "dhvaani"}, "mr")
    assert (e.value.code, e.value.status) == ("GENERATION_TIMEOUT", 504)


# ---- the worker client against a fake HTTP worker ------------------------------------------------------------------

class FakeProc:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code

    def terminate(self):
        self.code = -15

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


@pytest.fixture()
def fake_worker():
    """Starts a tiny HTTP server that plays the DhVaani worker; yields (start, requests_seen)."""
    servers, seen = [], []

    def start(status=200, body=None, delay=0.0, audio_seconds=0.5):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                seen.append(json.loads(self.rfile.read(n)))
                if delay:
                    import time
                    time.sleep(delay)
                if status == 200:
                    payload, ctype = wav_bytes(burst(audio_seconds)), "audio/wav"
                else:
                    payload, ctype = json.dumps(body or {"error": "boom"}).encode(), "application/json"
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return srv.server_address[1]

    yield start, seen
    for s in servers:
        s.shutdown()
        s.server_close()


def make_client(port, proc=None, timeout=5.0):
    c = dh.DhVaaniClone.__new__(dh.DhVaaniClone)
    c.proc = proc or FakeProc()
    c.port = port
    c.request_timeout = timeout
    c._active = 0
    c._active_lock = threading.Lock()
    c._ref_dir = tempfile.mkdtemp(prefix="bonclay_test_refs_")
    return c


PROMPT = {"backend": "dhvaani", "ref_wav": b"RIFF-ref", "ref_text": "reference words"}


def test_client_sends_the_reference_pair_and_returns_the_audio(fake_worker):
    start, seen = fake_worker
    c = make_client(start())
    try:
        wavs, sr = c.generate_voice_clone("मराठी वाक्य", "mr", PROMPT, num_step=8)
        assert sr == 24000 or sr > 0
        assert len(wavs) == 1 and wavs[0].dtype == np.float32 and len(wavs[0]) > 0
        body = seen[0]
        assert body["text"] == "मराठी वाक्य" and body["language"] == "mr" and body["ref_text"] == "reference words"
        assert body["num_step"] == 8
        assert open(body["ref_audio_path"], "rb").read() == b"RIFF-ref"  # the reference is a file the worker can read
        assert c.busy is False
        # languages DhVaani does not clone, and Qwen3's "Auto", are not sent as a language
        c.generate_voice_clone("hello", "Auto", PROMPT)
        c.generate_voice_clone("hello", "en", PROMPT)
        assert "language" not in seen[1] and "language" not in seen[2]
    finally:
        c.close()
    assert not os.path.isdir(c._ref_dir)  # the owner's reference clips are removed when the worker stops


def test_client_rejects_a_prompt_built_for_another_engine(fake_worker):
    start, _ = fake_worker
    c = make_client(start())
    try:
        with pytest.raises(ValueError, match="DhVaani"):
            c.generate_voice_clone("x", "mr", {"backend": "chatterbox", "ref_wav": b"x"})
        with pytest.raises(ValueError):
            c.generate_voice_clone("x", "mr", None)
    finally:
        c.close()


def test_a_rejected_request_is_a_422_with_the_workers_message(fake_worker):
    start, _ = fake_worker
    c = make_client(start(status=400, body={"error": "ref_text is required"}))
    try:
        with pytest.raises(dh.DhVaaniError) as e:
            c.generate_voice_clone("x", "mr", PROMPT)
        assert e.value.status == 422 and e.value.code == "GENERATION_FAILED" and "ref_text is required" in e.value.message
        assert c.busy is False  # the busy flag is released on errors too
    finally:
        c.close()


def test_a_worker_crash_during_generation_is_a_500(fake_worker):
    start, _ = fake_worker
    c = make_client(start(status=500, body={"error": "RuntimeError: Metal blew up"}))
    try:
        with pytest.raises(dh.DhVaaniError) as e:
            c.generate_voice_clone("x", "mr", PROMPT)
        assert e.value.status == 500 and "Metal blew up" in e.value.message
    finally:
        c.close()


def test_a_dead_worker_is_a_503_the_manager_can_recover_from(fake_worker):
    import socket

    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    c = make_client(port, proc=FakeProc(code=1))
    try:
        assert c.alive() is False
        with pytest.raises(dh.DhVaaniError) as e:
            c.generate_voice_clone("x", "mr", PROMPT)
        assert e.value.code == "ENGINE_UNAVAILABLE" and e.value.status == 503
    finally:
        c.close()


def test_a_slow_worker_is_a_504(fake_worker):
    start, _ = fake_worker
    c = make_client(start(delay=1.0), timeout=0.2)
    try:
        with pytest.raises(dh.DhVaaniError) as e:
            c.generate_voice_clone("x", "mr", PROMPT)
        assert e.value.code == "GENERATION_TIMEOUT" and e.value.status == 504
    finally:
        c.close()


def test_starting_with_too_little_memory_is_refused_before_any_process(monkeypatch):
    import subprocess

    monkeypatch.setattr(dh, "memory_problem", lambda *a, **k: "Only 0.4 GB of memory is free")

    def no_process(*a, **k):
        raise AssertionError("no worker may be started when memory is too low")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    with pytest.raises(dh.DhVaaniError) as e:
        dh.DhVaaniClone("/nonexistent/python")
    assert (e.value.code, e.value.status) == ("INSUFFICIENT_MEMORY", 503) and "0.4 GB" in e.value.message


def test_the_model_manager_turns_low_memory_into_a_structured_error(backend_mod, monkeypatch):
    from core.errors import AppError

    avail(monkeypatch, dhvaani=True)
    monkeypatch.setattr(dh, "dhvaani_python", lambda: "/nonexistent/python")
    monkeypatch.setattr(dh, "memory_problem", lambda *a, **k: "Only 0.4 GB of memory is free")
    with pytest.raises(AppError) as e:
        backend_mod.advanced_manager._instantiate_model_sync("dhvaani")
    assert e.value.code == "INSUFFICIENT_MEMORY" and e.value.status == 503


def test_the_model_manager_refuses_when_dhvaani_is_not_installed(backend_mod, monkeypatch):
    from core.errors import AppError

    avail(monkeypatch, dhvaani=False)
    with pytest.raises(AppError) as e:
        backend_mod.advanced_manager._instantiate_model_sync("dhvaani")
    assert e.value.code == "ENGINE_UNAVAILABLE" and e.value.details["setup"] == "./bonclay setup --dhvaani"


# ---- generation endpoint, receipts, deletion --------------------------------------------------------------------------

def test_voices_generate_in_marathi_uses_dhvaani_and_the_receipt_says_so(app_client, env, monkeypatch, tmp_path, ref_wav):
    avail(monkeypatch, dhvaani=True, chatterbox=True, f5=False)
    row = stored_voice(tmp_path, burst(6.0), "good morning front desk", voice_id="v-mr")
    env.save_local_voice(row)
    fake, loaded, persisted = FakeDhvaani(), [], []

    async def fake_load(key):
        loaded.append(key)
        return fake

    real_persist = env.persist_generation

    async def spy(*args, **kw):
        persisted.append(kw.get("engine"))
        return await real_persist(*args, **kw)

    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
    monkeypatch.setattr(env, "persist_generation", spy)
    no_asr(env, monkeypatch)
    r = app_client.post("/api/voices/generate", headers=LOCAL,
                        data={"text": "नमस्कार, तुमची भेट उद्या आहे.", "voice_id": "v-mr", "language": "mr"})
    assert r.status_code == 200, r.text
    assert set(loaded) == {"dhvaani"} and persisted == ["dhvaani"]  # the receipt's engine field
    assert fake.calls and fake.calls[0][1] == "mr" and fake.calls[0][2]["ref_text"] == "good morning front desk"
    assert r.headers["content-type"] == "audio/wav" and r.headers.get("x-receipt-id")


@pytest.mark.parametrize("flag,expected", [("", "वऱ्हाडी वेळ"), ("on", "वरहाडी वेल\u093c")])
def test_pron_lexicon_changes_only_what_dhvaani_reads_and_only_when_on(app_client, env, monkeypatch, tmp_path, flag,
                                                                       expected):
    monkeypatch.setenv("PRON_LEXICON", flag)
    avail(monkeypatch, dhvaani=True, chatterbox=True, f5=False)
    env.save_local_voice(stored_voice(tmp_path, burst(6.0), "good morning front desk", voice_id="v-mr"))
    fake, stored = FakeDhvaani(), []

    async def fake_load(key):
        return fake

    real_persist = env.persist_generation

    async def spy(*args, **kw):
        stored.append(kw.get("text"))
        return await real_persist(*args, **kw)

    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
    monkeypatch.setattr(env, "persist_generation", spy)
    no_asr(env, monkeypatch)
    r = app_client.post("/api/voices/generate", headers=LOCAL,
                        data={"text": "वऱ्हाडी वेळ", "voice_id": "v-mr", "language": "mr"})
    assert r.status_code == 200, r.text
    assert fake.calls[0][0] == expected
    assert stored == ["वऱ्हाडी वेळ"]  # the stored text is what the user wrote


def test_digits_reach_dhvaani_as_words_and_stored_text_is_unchanged(app_client, env, monkeypatch, tmp_path):
    monkeypatch.delenv("NUMWORDS", raising=False)
    avail(monkeypatch, dhvaani=True, chatterbox=True, f5=False)
    env.save_local_voice(stored_voice(tmp_path, burst(6.0), "good morning front desk", voice_id="v-mr"))
    fake, stored = FakeDhvaani(), []

    async def fake_load(key):
        return fake

    real_persist = env.persist_generation

    async def spy(*args, **kw):
        stored.append(kw.get("text"))
        return await real_persist(*args, **kw)

    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
    monkeypatch.setattr(env, "persist_generation", spy)
    no_asr(env, monkeypatch)
    text = "फी ₹450 आहे, सकाळी 10 वाजता या."
    r = app_client.post("/api/voices/generate", headers=LOCAL, data={"text": text, "voice_id": "v-mr", "language": "mr"})
    assert r.status_code == 200, r.text
    assert fake.calls[0][0] == "फी चारशे पन्नास रुपये आहे, सकाळी दहा वाजता या."
    assert stored == [text]


def test_hindi_generation_goes_to_dhvaani_when_it_is_set_up(app_client, env, monkeypatch, tmp_path):
    env.save_local_voice(stored_voice(tmp_path, burst(6.0), "good morning front desk", voice_id="v-hi"))
    loaded = []

    async def fake_load(key):
        loaded.append(key)
        return FakeDhvaani()

    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
    no_asr(env, monkeypatch)
    avail(monkeypatch, dhvaani=True, chatterbox=True, f5=False)
    r = app_client.post("/api/voices/generate", headers=LOCAL,
                        data={"text": "नमस्ते, यह एक परीक्षण है।", "voice_id": "v-hi", "language": "hi"})
    assert r.status_code == 200, r.text
    assert set(loaded) == {"dhvaani"}  # Chatterbox, though ready, is only the fallback


def test_over_long_text_is_refused_before_the_worker_starts(app_client, env, monkeypatch, tmp_path):
    avail(monkeypatch, dhvaani=True)
    env.save_local_voice(stored_voice(tmp_path, burst(6.0), "x", voice_id="v-long"))

    async def boom(key):
        raise AssertionError("the worker must not be started for a request that is refused")

    monkeypatch.setattr(env.advanced_manager, "load_model", boom)
    r = app_client.post("/api/voices/generate", headers=LOCAL,
                        data={"text": "नमस्ते। " * 400, "voice_id": "v-long", "language": "mr"})
    assert r.status_code == 422 and r.json()["code"] == "TEXT_TOO_LONG"


def test_unavailable_dhvaani_is_a_503_with_the_setup_command(app_client, env, monkeypatch, tmp_path):
    avail(monkeypatch, dhvaani=False, chatterbox=False, f5=False)
    env.save_local_voice(stored_voice(tmp_path, burst(6.0), "x", voice_id="v-no"))
    r = app_client.post("/api/voices/generate", headers=LOCAL,
                        data={"text": "नमस्कार", "voice_id": "v-no", "language": "mr"})
    assert r.status_code == 503 and r.json()["code"] == "ENGINE_UNAVAILABLE"
    assert "./bonclay setup --dhvaani" in r.text


def test_deleting_a_voice_removes_its_dhvaani_reference_and_other_derived_files(env, tmp_path):
    import asyncio

    row = stored_voice(tmp_path, burst(6.0), "x", voice_id="00000000-0000-4000-8000-000000000001")
    vid = row["id"]
    row.update(user_id="local-user", consent_id=None)
    env.save_local_voice(row)
    for ext in (".dhvaani.wav", ".dhvaani.json", ".chatterbox.pt", ".pt"):
        (tmp_path / f"{vid}{ext}").write_bytes(b"x")
    out = asyncio.run(env._delete_voice_core(vid, "local-user", "local"))
    assert out["deleted"] is True
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(vid)) == []
    assert out["removed"]["files"] == 5  # wav, pt, chatterbox prompt, dhvaani clip + record


# ---- a live agent or phone turn never uses it --------------------------------------------------------------------------

def test_agent_turn_in_hindi_never_uses_dhvaani(app_client, env, monkeypatch, tmp_path, ref_wav):
    avail(monkeypatch, dhvaani=True, chatterbox=True, f5=False)
    spoken, loaded = [], []

    def chat(messages, *, json_mode=False, **kw):
        text = json.dumps(PERSONA) if json_mode else "आपका रिफंड तीस दिन में आएगा।"
        return llm.ChatResult(text, "fake", "m", 1, {})

    monkeypatch.setattr(llm, "chat", chat)

    async def fake_asr(audio_bytes, name, language=None):
        return "मेरा रिफंड कब मिलेगा"

    async def fake_load(key):
        loaded.append(key)
        return object()

    monkeypatch.setattr(env, "_speech_to_text", fake_asr)
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": spoken.append(lang) or np.zeros(2400, dtype=np.float32))
    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
    monkeypatch.setattr(env, "indic_speak_service",
                        lambda: types.SimpleNamespace(indic_speak_unavailable_reason=lambda: "off", MAX_TEXT_CHARS=400))
    row = stored_voice(tmp_path, burst(6.0), "x", language="en", voice_id="v-turn")
    env.save_local_voice(row)
    a = make_agent(app_client, voice_id="v-turn", disclose_ai=False)
    routed = []
    real_route = env.route_engine
    monkeypatch.setattr(env, "route_engine", lambda *args: routed.append(real_route(*args)) or routed[-1])

    r = app_client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL,
                        files={"audio": ("rec.wav", ref_wav.read_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert set(routed) == {"dhvaani"}  # the router did pick it...
    assert r.json()["engine"] == "kokoro" and spoken == ["hi"]  # ...and the turn spoke with the preset voice instead
    assert "dhvaani" not in loaded


# ---- the speed estimate -----------------------------------------------------------------------------------------------

def test_estimator_has_a_dhvaani_prior(tmp_path):
    from services.speech_estimate import DEFAULT_LOAD_S, DEFAULT_RTF, Estimator, MIN_VOICE_S

    est = Estimator(path=str(tmp_path / "rtf.json"))
    assert DEFAULT_RTF["dhvaani"] == 2.0 and est.rtf("dhvaani") == 2.0
    text = "तुमची अपॉइंटमेंट गुरुवारी चार वाजता निश्चित आहे."  # 49 characters: 4.1 s of speech at 12 characters a second
    warm = est.voice_seconds("dhvaani", text, "mr", warm=True)
    assert warm == pytest.approx(2.0 * len(text) / 12.0, rel=0.01) and warm >= MIN_VOICE_S["dhvaani"]
    assert est.voice_seconds("dhvaani", text, "mr", warm=False) == pytest.approx(warm + DEFAULT_LOAD_S["dhvaani"])
    assert 2.0 < DEFAULT_RTF["chatterbox_ml"] and DEFAULT_RTF["dhvaani"] > DEFAULT_RTF["qwen3_0_6b"] > DEFAULT_RTF["kokoro"]
    assert est.chars_per_second("ur") == 12.0  # Urdu and Assamese are spoken at the Indic rate too


def test_estimator_learns_dhvaani_speed_and_cold_start_separately(tmp_path):
    from services.speech_estimate import Estimator

    est = Estimator(path=str(tmp_path / "rtf.json"))
    est.record_voice("dhvaani", "mr", "x" * 60, voice_s=40.0, audio_s=5.0, cold=True)  # slow because it had to load
    assert est.rtf("dhvaani") == 2.0 and est.load_seconds("dhvaani") > 12.0
    est.record_voice("dhvaani", "mr", "x" * 60, voice_s=7.5, audio_s=5.0)  # warm: RTF 1.5
    assert est.rtf("dhvaani") < 2.0


# ---- the worker's pure helpers (imported without torch) ----------------------------------------------------------------

def load_worker():
    path = os.path.join(dh.BACKEND_DIR, "services", "dhvaani_worker.py")
    spec = importlib.util.spec_from_file_location("dhvaani_worker_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worker_text_cleaning_and_clamping():
    w = load_worker()
    assert w.clean_text("  नमस्ते \t  दुनिया\x00\x07 ") == "नमस्ते दुनिया"
    assert w.clean_text(None) == ""
    assert w.clamp("99", 4, 32, 16) == 32 and w.clamp("2", 4, 32, 16) == 4 and w.clamp("x", 4, 32, 16) == 16
    assert w.clamp(None, 0.5, 1.5, 1.0) == 1.0


def test_worker_picks_cuda_then_mps_then_cpu():
    w = load_worker()

    def fake_torch(cuda, mps):
        return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda),
                                     backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)))

    assert w.pick_device(fake_torch(True, True)) == "cuda"
    assert w.pick_device(fake_torch(False, True)) == "mps"
    assert w.pick_device(fake_torch(False, False)) == "cpu"
    assert w.pick_device(fake_torch(True, True), "cpu") == "cpu"  # an explicit --device wins


def test_worker_languages_cover_the_backend_list_and_the_download_keeps_the_selftest_sample():
    w = load_worker()
    assert set(dh.LANGUAGES) <= set(w.LANGUAGES)
    assert "samples/hindi.wav" not in w.DOWNLOAD_IGNORE and "samples/tamil.wav" in w.DOWNLOAD_IGNORE


def test_worker_model_default_matches_the_client():
    assert load_worker().DEFAULT_MODEL == dh.DEFAULT_MODEL


def test_worker_pins_hf_home_to_the_real_cache(monkeypatch, tmp_path):
    w = load_worker()
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert w.default_hf_home() == os.path.join(str(tmp_path), "huggingface")
    monkeypatch.setenv("HF_HOME", "/somewhere/else")
    assert w.default_hf_home() == "/somewhere/else"


def test_chatterbox_for_hindi_names_the_setting_when_chatterbox_is_missing(backend_mod, monkeypatch):
    from core.errors import AppError

    avail(monkeypatch, dhvaani=True, chatterbox=False, f5=False)
    monkeypatch.setenv("INDIC_CLONE_ENGINE", "chatterbox")
    with pytest.raises(AppError) as e:
        backend_mod.route_engine("hi", "premium", True)
    assert "INDIC_CLONE_ENGINE" in e.value.message and "./bonclay setup --chatterbox" in e.value.message
    assert backend_mod.route_engine("mr", "premium", True) == "dhvaani"  # the other languages keep DhVaani


def test_models_status_says_which_engine_clones_hindi(backend_mod, monkeypatch):
    import asyncio
    avail(monkeypatch, dhvaani=True, chatterbox=True, f5=False)
    monkeypatch.setenv("INDIC_CLONE_ENGINE", "chatterbox")
    out = asyncio.run(backend_mod.models_status())
    assert out["indic_clone"] == {"preference": "chatterbox", "hindi_engine": "chatterbox_ml"}
    monkeypatch.setenv("INDIC_CLONE_ENGINE", "auto")
    assert asyncio.run(backend_mod.models_status())["indic_clone"]["hindi_engine"] == "dhvaani"
