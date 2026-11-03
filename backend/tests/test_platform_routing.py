"""Platform detection, engine routing and the engine registry. Pure logic plus a few API checks through
the TestClient. No models are loaded and nothing touches the network (Supabase health is stubbed)."""
import platform

import pytest

LOCAL = {"storage-mode": "local"}


@pytest.fixture()
def backend_mod(app_client):
    import backend

    backend.limiter.reset()  # the generate endpoints are rate limited; start each test with a fresh window
    return backend


@pytest.fixture(autouse=True)
def indic_speak_not_installed(monkeypatch):
    """Routing must not depend on whether this machine happens to have the Indic-Speak worker set up."""
    from services import indic_speak

    monkeypatch.setattr(indic_speak, "indic_speak_unavailable_reason", lambda: "not set up")


@pytest.fixture()
def no_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return monkeypatch


# ---- platform_info ---------------------------------------------------------------------------------

def test_platform_info_has_expected_shape(backend_mod):
    info = backend_mod.platform_info()
    expected_keys = {"os", "os_version", "arch", "device", "accelerator", "accelerator_name",
                     "vram_gb", "gpu_budget_gb"}
    assert expected_keys <= set(info)
    assert info["os"] in {"macOS", "Windows", "Linux", platform.system() or "Unknown"}
    assert info["accelerator"] in {"apple", "nvidia", "cpu"}


def test_platform_info_on_cpu_reports_no_accelerator(backend_mod, monkeypatch):
    monkeypatch.setattr(backend_mod, "DEVICE", "cpu")
    info = backend_mod.platform_info()
    assert info["device"] == "cpu"
    assert info["accelerator"] == "cpu"
    assert info["accelerator_name"] == "CPU"
    assert info["vram_gb"] is None


@pytest.mark.parametrize("raw, expected", [
    ("Darwin", "macOS"),
    ("Windows", "Windows"),
    ("Linux", "Linux"),
    ("FreeBSD", "FreeBSD"),  # unknown systems fall back to the raw name
])
def test_platform_info_maps_system_name(backend_mod, monkeypatch, raw, expected):
    monkeypatch.setattr(platform, "system", lambda: raw)
    assert backend_mod.platform_info()["os"] == expected


# ---- clone_engine_device ---------------------------------------------------------------------------

def test_clone_device_honours_env_override(backend_mod, monkeypatch):
    monkeypatch.setenv("CLONE_DEVICE", "mps")
    monkeypatch.setattr(backend_mod, "DEVICE", "cuda:0")
    assert backend_mod.clone_engine_device() == "mps"


def test_clone_device_is_cpu_on_mps_without_override(backend_mod, monkeypatch):
    monkeypatch.delenv("CLONE_DEVICE", raising=False)
    monkeypatch.setattr(backend_mod, "DEVICE", "mps")
    assert backend_mod.clone_engine_device() == "cpu"


def test_clone_device_follows_cuda_without_override(backend_mod, monkeypatch):
    monkeypatch.delenv("CLONE_DEVICE", raising=False)
    monkeypatch.setattr(backend_mod, "DEVICE", "cuda:0")
    assert backend_mod.clone_engine_device() == "cuda:0"


# ---- route_engine ----------------------------------------------------------------------------------

def test_english_fast_preset_routes_to_kokoro(backend_mod):
    assert backend_mod.route_engine("en", "fast", False) == "kokoro"


def test_english_premium_clone_routes_to_qwen(backend_mod):
    assert backend_mod.route_engine("en", "premium", True) == backend_mod.CLONE_ENGINE == "qwen3_0_6b"


def test_indic_clone_without_hf_token_is_unavailable(backend_mod, no_hf_token):
    from core.errors import AppError

    with pytest.raises(AppError) as exc:
        backend_mod.route_engine("hi", "premium", True)
    assert exc.value.code == "ENGINE_UNAVAILABLE"
    assert exc.value.status == 503


def test_indic_preset_without_hf_token_falls_back_to_kokoro(backend_mod, no_hf_token):
    assert backend_mod.route_engine("hi", "fast", False) == "kokoro"


def test_indic_clone_with_hf_token_routes_to_indic_f5(backend_mod, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token-not-real")
    monkeypatch.setattr(backend_mod, "indic_f5_unavailable_reason", lambda: None)  # f5_tts present
    assert backend_mod.route_engine("hi", "premium", True) == "indic_f5"
    # IndicF5 only clones (it needs a reference clip), so a preset never goes to it
    assert backend_mod.route_engine("mr", "fast", False) == "kokoro"


@pytest.fixture()
def token_but_no_f5_tts(monkeypatch):
    """HF_TOKEN is set (the weights are reachable) but the f5_tts package is not installed."""
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setenv("HF_TOKEN", "test-token-not-real")
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "f5_tts" else real_find_spec(name, *a, **k))
    return monkeypatch


def test_indic_preset_with_token_but_no_f5_tts_falls_back_to_kokoro(backend_mod, token_but_no_f5_tts):
    # Setting HF_TOKEN must not send Hindi/Marathi speech to an engine that cannot load.
    assert backend_mod.route_engine("hi", "fast", False) == "kokoro"
    assert backend_mod.route_engine("mr", "premium", False) == "kokoro"


def test_indic_clone_with_token_but_no_f5_tts_says_why(backend_mod, token_but_no_f5_tts):
    from core.errors import AppError

    with pytest.raises(AppError) as exc:
        backend_mod.route_engine("mr", "premium", True)
    assert exc.value.code == "ENGINE_UNAVAILABLE"
    assert exc.value.status == 503
    assert "DhVaani" in exc.value.message  # DhVaani is the engine that clones Marathi; IndicF5 is the old optional one


def test_indic_f5_unavailable_reason(backend_mod, no_hf_token, monkeypatch):
    import importlib.util

    assert "HF_TOKEN" in backend_mod.indic_f5_unavailable_reason()
    monkeypatch.setenv("HF_TOKEN", "test-token-not-real")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: object() if name == "f5_tts" else real_find_spec(name, *a, **k))
    assert backend_mod.indic_f5_unavailable_reason() is None


def test_non_indic_languages_ignore_indic_f5(backend_mod, token_but_no_f5_tts):
    assert backend_mod.route_engine("en", "fast", False) == "kokoro"
    assert backend_mod.route_engine("en", "premium", True) == "qwen3_0_6b"


# ---- engine registry -------------------------------------------------------------------------------

def test_audio8_is_retired_from_registry(backend_mod):
    assert "audio8_0_6b" not in backend_mod.ENGINE_REGISTRY
    assert backend_mod.CLONE_ENGINES == ("qwen3_0_6b", "chatterbox_ml", "dhvaani")
    assert backend_mod.PLAYGROUND_CLONE_ENGINES == ("qwen3_0_6b",)


# ---- API -------------------------------------------------------------------------------------------

def test_health_reports_platform(app_client, backend_mod, monkeypatch):
    async def unreachable():  # never hit Supabase from a test
        return False

    monkeypatch.setattr(backend_mod, "_supabase_reachable", unreachable)
    r = app_client.get("/api/health", headers=LOCAL)
    assert r.status_code == 200, r.text
    plat = r.json()["platform"]
    assert "os" in plat
    assert plat["accelerator"] in {"apple", "nvidia", "cpu"}


@pytest.mark.parametrize("path, form", [
    ("/api/generate-preset", {"text": "hello there", "speaker": "Vivian"}),
    ("/api/generate-design", {"text": "hello there", "instruct": "a calm narrator"}),
])
def test_preset_and_design_generation_are_not_available(app_client, backend_mod, path, form):
    r = app_client.post(path, headers=LOCAL, data=form)
    assert r.status_code == 501, r.text
    assert r.json()["code"] == "ENGINE_FEATURE_UNAVAILABLE"


# ---- _is_clone_prompt ------------------------------------------------------------------------------

def test_audio8_style_prompt_is_not_a_clone_prompt(backend_mod):
    assert backend_mod._is_clone_prompt({"audio_array": [0.0], "sample_rate": 16000}) is False


def test_arbitrary_object_is_a_clone_prompt(backend_mod):
    assert backend_mod._is_clone_prompt(object()) is True
