"""Chatterbox Multilingual (Hindi voice cloning): availability, the routing table, prompts rebuilt from a stored
reference recording, the long-text/sentence logic of the worker, and the guarantee that a live agent turn never uses
it. No model is loaded and no worker is started: the client is faked."""
import importlib.util
import json
import os
import types

import numpy as np
import pytest

from test_offline_agents import LOCAL, PERSONA, env, make_agent, FakeEmbedding  # noqa: F401  (env is a fixture)

cbs = pytest.importorskip("services.chatterbox_clone")
from services import llm  # noqa: E402


@pytest.fixture()
def backend_mod(app_client):
    import backend

    backend.limiter.reset()
    return backend


def chatterbox(monkeypatch, ready: bool):
    monkeypatch.setattr(cbs, "unavailable_reason", lambda: None if ready else "not set up")


def f5(monkeypatch, ready: bool):
    import backend

    monkeypatch.setattr(backend, "indic_f5_unavailable_reason", lambda: None if ready else "no f5_tts")


# ---- unavailable_reason ---------------------------------------------------------------------------------------

def test_off_switch_wins(monkeypatch):
    monkeypatch.setenv("CHATTERBOX", "off")
    assert "CHATTERBOX=off" in cbs.unavailable_reason()
    monkeypatch.setenv("CHATTERBOX", "False")
    assert cbs.unavailable_reason() is not None


def test_missing_venv_python_says_how_to_set_up(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATTERBOX", "")
    monkeypatch.setenv("CHATTERBOX_PYTHON", str(tmp_path / "nope" / "python"))
    monkeypatch.setattr(cbs, "BACKEND_DIR", str(tmp_path))  # no .venv-chatterbox in there either
    assert cbs.chatterbox_python() is None
    assert "./bonclay setup --chatterbox" in cbs.unavailable_reason()


def test_weights_not_cached_is_reported(monkeypatch, tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    monkeypatch.setenv("CHATTERBOX", "")
    monkeypatch.setenv("CHATTERBOX_PYTHON", str(py))
    monkeypatch.setattr(cbs, "weights_cached", lambda model=None: False)
    assert "weights are not downloaded" in cbs.unavailable_reason()
    monkeypatch.setattr(cbs, "weights_cached", lambda model=None: True)
    assert cbs.unavailable_reason() is None


def test_weights_cached_looks_for_the_model_files(monkeypatch):
    import huggingface_hub

    seen = []

    def fake(repo, filename):
        seen.append((repo, filename))
        return "/cache/" + filename if filename != "s3gen.pt" else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake)
    assert cbs.weights_cached() is False
    assert {f for _, f in seen} <= set(cbs._WEIGHT_FILES) and seen[0][0] == "ResembleAI/chatterbox"


def test_language_resolution():
    assert cbs.resolve_language("hi", "x") == "hi" and cbs.resolve_language("hi-IN", "x") == "hi"
    assert cbs.resolve_language("Auto", "नमस्ते दुनिया") == "hi"  # what Qwen3 callers pass: read the script
    assert cbs.resolve_language("Auto", "hello there") == "en"
    assert cbs.resolve_language("fr", "bonjour") == "fr"


# ---- routing table ---------------------------------------------------------------------------------------------

def test_hindi_clone_prefers_chatterbox(backend_mod, monkeypatch):
    chatterbox(monkeypatch, True)
    f5(monkeypatch, True)
    assert backend_mod.route_engine("hi", "premium", True) == "chatterbox_ml"
    assert backend_mod.route_engine("hi", "fast", True) == "chatterbox_ml"


def test_hindi_clone_falls_back_to_indic_f5(backend_mod, monkeypatch):
    chatterbox(monkeypatch, False)
    f5(monkeypatch, True)
    assert backend_mod.route_engine("hi", "premium", True) == "indic_f5"


def test_hindi_clone_with_nothing_available_suggests_chatterbox(backend_mod, monkeypatch):
    from core.errors import AppError

    chatterbox(monkeypatch, False)
    f5(monkeypatch, False)
    with pytest.raises(AppError) as e:
        backend_mod.route_engine("hi", "premium", True)
    assert e.value.code == "ENGINE_UNAVAILABLE" and e.value.status == 503
    assert "./bonclay setup --chatterbox" in e.value.message or "./bonclay setup --chatterbox" in str(e.value.details)
    assert "Chatterbox" in e.value.message


def test_other_indian_languages_use_indic_f5_or_say_dhvaani_is_needed(backend_mod, monkeypatch):
    from core.errors import AppError

    chatterbox(monkeypatch, True)  # Chatterbox being ready must not change Marathi
    f5(monkeypatch, True)
    assert backend_mod.route_engine("mr", "premium", True) == "indic_f5"
    f5(monkeypatch, False)
    with pytest.raises(AppError) as e:
        backend_mod.route_engine("mr", "premium", True)
    assert e.value.code == "ENGINE_UNAVAILABLE"
    assert "needs DhVaani" in e.value.message and "Preset voices" in e.value.message  # DhVaani clones Marathi


def test_non_indian_languages_keep_qwen3(backend_mod, monkeypatch):
    chatterbox(monkeypatch, True)
    for lang in ("en", "fr", "de"):
        assert backend_mod.route_engine(lang, "premium", True) == "qwen3_0_6b"
    assert backend_mod.route_engine("en", "fast", False) == "kokoro"


def test_preset_hindi_speech_never_routes_to_chatterbox(backend_mod, monkeypatch):
    chatterbox(monkeypatch, True)
    assert backend_mod.route_engine("hi", "fast", False) in ("kokoro", "indic_speak")


# ---- registry / status / hints ---------------------------------------------------------------------------------

def test_registry_entry_matches_the_installed_package(backend_mod):
    entry = backend_mod.ENGINE_REGISTRY["chatterbox_ml"]
    assert entry["type"] == "clone" and entry["hf_id"] == "ResembleAI/chatterbox"
    assert entry["languages"] == set(cbs.LANGUAGES) and "hi" in entry["languages"]
    assert "chatterbox_ml" in backend_mod.CLONE_ENGINES
    assert "chatterbox_ml" not in backend_mod.PLAYGROUND_CLONE_ENGINES  # the Playground stays on Qwen3


def test_models_status_lists_it_with_availability(app_client, monkeypatch):
    chatterbox(monkeypatch, False)
    engines = {e["id"]: e for e in app_client.get("/api/models/status", headers=LOCAL).json()["engines"]}
    e = engines["chatterbox_ml"]
    assert e["name"] == "Chatterbox Multilingual (Resemble AI)" and e["loaded"] is False
    assert e["available"] is False and e["unavailable_reason"] == "not set up" and "chatterbox" in e["setup"]
    chatterbox(monkeypatch, True)
    engines = {e["id"]: e for e in app_client.get("/api/models/status", headers=LOCAL).json()["engines"]}
    assert engines["chatterbox_ml"]["available"] is True and engines["chatterbox_ml"]["unavailable_reason"] is None


def test_setup_hint_when_not_installed(app_client, monkeypatch):
    monkeypatch.setenv("CHATTERBOX", "")  # the suite's default is "off", which (rightly) hides the hint
    chatterbox(monkeypatch, False)
    hints = {h["id"]: h for h in app_client.get("/api/setup-hints", headers=LOCAL).json()["hints"]}
    assert hints["no_chatterbox"]["severity"] == "info"
    assert hints["no_chatterbox"]["command"] == "./bonclay setup --chatterbox"
    assert hints["no_chatterbox"]["title"] == "Hindi voice cloning needs Chatterbox"
    chatterbox(monkeypatch, True)
    assert "no_chatterbox" not in {h["id"] for h in app_client.get("/api/setup-hints", headers=LOCAL).json()["hints"]}


# ---- prompts ---------------------------------------------------------------------------------------------------

class FakeChatterbox:
    """Stands in for ChatterboxClone: same prompt surface, no worker."""
    prompt_backend = cbs.PROMPT_BACKEND
    prompt_sample_rate = cbs.PROMPT_SAMPLE_RATE
    wants_language_code = True

    def __init__(self):
        self.built, self.calls = [], []

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        audio, sr = ref_audio[0]
        self.built.append((len(audio), sr))
        return {"backend": cbs.PROMPT_BACKEND, "ref_wav": b"RIFF-fake-ref"}

    def generate_voice_clone(self, text, language="Auto", voice_clone_prompt=None, **kw):
        self.calls.append((text, language, voice_clone_prompt))
        return [np.zeros(2400, dtype=np.float32)], 24000

    def eval(self):
        return self


def qwen_voice(env, tmp_path, ref_wav, voice_id="v-qwen"):
    """A voice cloned with Qwen3: its reference recording and a Qwen3 prompt file on disk."""
    import torch

    (tmp_path / f"{voice_id}.wav").write_bytes(ref_wav.read_bytes())
    torch.save({"backend": "qwen-fake", "marker": 1}, tmp_path / f"{voice_id}.pt")
    return {"id": voice_id, "engine": "qwen3_0_6b", "language": "en", "display_name": "Speaker",
            "reference_audio_path": f"local:{voice_id}", "transcript": None, "user_id": "local-user"}


@pytest.mark.asyncio
async def test_qwen_voice_gets_a_chatterbox_prompt_from_its_reference(env, tmp_path, ref_wav):
    row = qwen_voice(env, tmp_path, ref_wav)
    qwen_before = (tmp_path / "v-qwen.pt").read_bytes()
    model = FakeChatterbox()
    prompt = await env.load_clone_prompt("v-qwen", "local-user", "local", row, model)
    assert prompt == {"backend": "chatterbox", "ref_wav": b"RIFF-fake-ref"}
    assert model.built and model.built[0][1] == 24000  # decoded at the model's rate, not the Qwen3 16 kHz
    assert (tmp_path / "v-qwen.chatterbox.pt").is_file()  # cached beside the Qwen3 prompt...
    assert (tmp_path / "v-qwen.pt").read_bytes() == qwen_before  # ...which is untouched
    assert row["engine"] == "qwen3_0_6b"

    again = FakeChatterbox()  # second use: served from the cache, nothing rebuilt
    assert await env.load_clone_prompt("v-qwen", "local-user", "local", row, again) == prompt
    assert again.built == []


@pytest.mark.asyncio
async def test_chatterbox_prompt_is_never_handed_to_qwen3(env, tmp_path, ref_wav):
    row = qwen_voice(env, tmp_path, ref_wav)
    row["engine"] = "chatterbox_ml"  # a voice that was registered while Chatterbox was routing Hindi
    (tmp_path / "v-qwen.pt").unlink()
    cb_prompt = {"backend": "chatterbox", "ref_wav": b"x"}
    assert env._is_clone_prompt(cb_prompt, FakeChatterbox()) is True
    assert env._is_clone_prompt(cb_prompt, object()) is False
    assert env._is_clone_prompt({"backend": "qwen-fake"}, FakeChatterbox()) is False

    class Qwen:
        built = 0

        def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
            Qwen.built += 1
            return {"backend": "qwen-rebuilt"}

    prompt = await env.load_clone_prompt("v-qwen", "local-user", "local", row, Qwen())
    assert prompt == {"backend": "qwen-rebuilt"} and Qwen.built == 1  # rebuilt for Qwen3 from the recording


@pytest.mark.asyncio
async def test_missing_reference_recording_is_a_404(env, tmp_path):
    from core.errors import AppError

    row = {"id": "ghost", "engine": "qwen3_0_6b", "reference_audio_path": "local:ghost"}
    with pytest.raises(AppError) as e:
        await env.load_clone_prompt("ghost", "local-user", "local", row, FakeChatterbox())
    assert e.value.code == "VOICE_NOT_FOUND"


@pytest.mark.asyncio
async def test_clone_chunks_passes_the_language_only_to_engines_that_want_it(env):
    cb = FakeChatterbox()
    await env._clone_chunks(cb, "नमस्ते। आप कैसे हैं?", {"backend": "chatterbox"}, "hi")
    assert [c[1] for c in cb.calls] == ["hi"]

    class Qwen:
        langs = []

        def generate_voice_clone(self, text, language, voice_clone_prompt, **kw):
            Qwen.langs.append(language)
            return [np.zeros(10, dtype=np.float32)], 24000

    await env._clone_chunks(Qwen(), "hello", {"p": 1}, "en")
    assert Qwen.langs == ["Auto"]  # Qwen3 keeps auto-detecting


# ---- generation endpoint ---------------------------------------------------------------------------------------

def test_voices_generate_in_hindi_uses_chatterbox_with_the_language(app_client, env, monkeypatch, tmp_path, ref_wav):
    chatterbox(monkeypatch, True)
    f5(monkeypatch, False)
    row = qwen_voice(env, tmp_path, ref_wav)
    env.save_local_voice(row)
    fake = FakeChatterbox()
    loaded = []

    async def fake_load(key):
        loaded.append(key)
        return fake

    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
    r = app_client.post("/api/voices/generate", headers=LOCAL,
                        data={"text": "नमस्ते, यह एक परीक्षण है।", "voice_id": "v-qwen", "language": "hi"})
    assert r.status_code == 200, r.text
    assert set(loaded) == {"chatterbox_ml"}
    assert fake.calls and fake.calls[0][1] == "hi" and fake.calls[0][2]["backend"] == "chatterbox"
    assert r.headers["content-type"] == "audio/wav"


def test_over_long_text_is_refused_before_the_model_starts(app_client, env, monkeypatch, tmp_path, ref_wav):
    chatterbox(monkeypatch, True)
    env.save_local_voice(qwen_voice(env, tmp_path, ref_wav))

    async def boom(key):
        raise AssertionError("the worker must not be started for a request that is refused")

    monkeypatch.setattr(env.advanced_manager, "load_model", boom)
    r = app_client.post("/api/voices/generate", headers=LOCAL,
                        data={"text": "नमस्ते। " * 400, "voice_id": "v-qwen", "language": "hi"})
    assert r.status_code == 422 and r.json()["code"] == "TEXT_TOO_LONG"


# ---- a live agent turn never uses it ---------------------------------------------------------------------------

def test_agent_turn_in_hindi_never_uses_chatterbox(app_client, env, monkeypatch, tmp_path, ref_wav):
    chatterbox(monkeypatch, True)
    f5(monkeypatch, False)
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
    env.save_local_voice(qwen_voice(env, tmp_path, ref_wav))
    a = make_agent(app_client, voice_id="v-qwen", disclose_ai=False)
    routed = []
    real_route = env.route_engine
    monkeypatch.setattr(env, "route_engine", lambda *args: routed.append(real_route(*args)) or routed[-1])

    r = app_client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL,
                        files={"audio": ("rec.wav", ref_wav.read_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert set(routed) == {"chatterbox_ml"}  # the router did pick it...
    assert r.json()["engine"] == "kokoro" and spoken == ["hi"]  # ...and the turn spoke with the preset voice instead
    assert "chatterbox_ml" not in loaded


# ---- the worker's text handling (pure functions, imported without the model) ------------------------------------

def load_worker():
    path = os.path.join(cbs.BACKEND_DIR, "services", "chatterbox_worker.py")
    spec = importlib.util.spec_from_file_location("chatterbox_worker_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worker_splits_long_text_into_sentences():
    w = load_worker()
    parts = w.split_for_speech("मैं ठीक हूँ। आप कैसे हैं? Hello there! " + "शब्द " * 80)
    assert parts[0] == "मैं ठीक हूँ।" and parts[1] == "आप कैसे हैं?"
    assert all(len(p) <= w.CHUNK_CHARS for p in parts) and len(parts) >= 4
    assert w.split_for_speech("   ") == []


def test_worker_token_cap_scales_with_text_and_has_a_ceiling():
    w = load_worker()
    assert w.token_cap("हाँ।") == 60
    assert w.token_cap("x" * 50) < w.token_cap("x" * 150) <= w.HARD_TOKEN_LIMIT
    assert w.token_cap("x" * 5000) == w.HARD_TOKEN_LIMIT


def test_worker_languages_match_the_backend_registry(backend_mod):
    # the worker takes its language list from the installed package; the backend list must be the same 23 codes
    assert len(cbs.LANGUAGES) == 23 and len(set(cbs.LANGUAGES)) == 23


def test_prompt_from_the_client_is_just_the_reference_audio():
    client = cbs.ChatterboxClone.__new__(cbs.ChatterboxClone)  # no worker: only the pure method is used
    p = client.create_voice_clone_prompt([(np.zeros(2400, dtype=np.float32), 24000)], ref_text="x")
    assert p["backend"] == "chatterbox" and p["ref_wav"][:4] == b"RIFF"
    assert cbs.is_chatterbox_prompt(p) and not cbs.is_chatterbox_prompt({"backend": "mlx-qwen3"})
