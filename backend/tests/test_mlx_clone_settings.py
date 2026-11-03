"""The MLX clone path's measured settings (23 Oct trial): streaming decode, temperature 0.5, a 24 kHz reference and the
8-bit model. The model is faked; nothing here loads MLX or starts a worker."""
import asyncio
import io
import json
import tempfile
import types

import numpy as np
import pytest
import soundfile as sf

from services import mlx_clone, mlx_worker


# ---------------------------------------------------------------------------------------------- worker settings

class _FakeModel:
    """Stands in for the mlx_audio Qwen3-TTS model: records the call and yields the audio in chunks."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        for c in self.chunks:
            yield types.SimpleNamespace(audio=c, sample_rate=24000)


@pytest.fixture()
def clean_env(monkeypatch):
    for name in ("MLX_CLONE_STREAM", "MLX_CLONE_STREAM_INTERVAL", "MLX_CLONE_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_generate_gets_streaming_and_temperature_by_default(clean_env):
    m = _FakeModel([np.zeros(10, dtype=np.float32)])
    mlx_worker.synthesize(m, "Hello there.", "/tmp/ref.wav", "reference words", "auto")
    (kw,) = m.calls
    assert kw["stream"] is True and kw["streaming_interval"] == 0.32 and kw["temperature"] == 0.5
    assert kw["text"] == "Hello there." and kw["ref_audio"] == "/tmp/ref.wav"
    assert kw["ref_text"] == "reference words" and kw["lang_code"] == "auto"
    assert set(kw) == {"text", "ref_audio", "ref_text", "lang_code", "stream", "streaming_interval", "temperature"}


def test_environment_overrides_the_settings(clean_env):
    clean_env.setenv("MLX_CLONE_STREAM_INTERVAL", "1.5")
    clean_env.setenv("MLX_CLONE_TEMPERATURE", "0.7")
    assert mlx_worker.generation_settings() == {"temperature": 0.7, "stream": True, "streaming_interval": 1.5}


def test_an_unreadable_number_falls_back_to_the_default(clean_env):
    clean_env.setenv("MLX_CLONE_STREAM_INTERVAL", "fast")
    clean_env.setenv("MLX_CLONE_TEMPERATURE", "")
    kw = mlx_worker.generation_settings()
    assert kw["streaming_interval"] == 0.32 and kw["temperature"] == 0.5


@pytest.mark.parametrize("value", ["off", "OFF", "0", "false", " no "])
def test_stream_off_restores_the_plain_call(clean_env, value):
    clean_env.setenv("MLX_CLONE_STREAM", value)
    clean_env.setenv("MLX_CLONE_STREAM_INTERVAL", "0.5")   # ignored when streaming is off
    m = _FakeModel([np.zeros(10, dtype=np.float32)])
    mlx_worker.synthesize(m, "Hi.", "/tmp/ref.wav", None, "auto")
    (kw,) = m.calls
    assert "stream" not in kw and "streaming_interval" not in kw
    assert kw["temperature"] == 0.5 and kw["ref_text"] is None


def test_temperature_zero_is_passed_on_not_replaced(clean_env):
    clean_env.setenv("MLX_CLONE_TEMPERATURE", "0")
    assert mlx_worker.generation_settings()["temperature"] == 0.0


def test_chunks_are_joined_in_order_into_one_array_and_one_wav(clean_env):
    chunks = [np.full(5, 0.1, dtype=np.float32), np.full(3, 0.2, dtype=np.float32), np.full(4, 0.3, dtype=np.float32)]
    audio, sr = mlx_worker.synthesize(_FakeModel(chunks), "Hi.", "/tmp/ref.wav", "x", "auto")
    assert sr == 24000 and audio.dtype == np.float32 and audio.shape == (12,)
    assert np.allclose(audio, np.concatenate(chunks))
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="FLOAT")    # what the HTTP handler returns: one whole WAV
    back, back_sr = sf.read(io.BytesIO(buf.getvalue()), dtype="float32")
    assert back_sr == 24000 and np.allclose(back, audio)


def test_generate_voice_clone_ignores_the_callers_sampling_settings(monkeypatch):
    """The API sends temperature 0.8 on /api/generate-stream; the worker's environment rules, so it is not forwarded."""
    sent = []
    wav = io.BytesIO()
    sf.write(wav, np.zeros(100, dtype=np.float32), 24000, format="WAV")

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        sent.append(req.data)
        return _Resp(wav.getvalue())

    monkeypatch.setattr(mlx_clone.urllib.request, "urlopen", fake_urlopen)
    clone = object.__new__(mlx_clone.MlxQwen3Clone)
    clone.port, clone._ref_dir, clone.proc = 1, tempfile.mkdtemp(), None
    prompt = {"backend": mlx_clone.PROMPT_BACKEND, "ref_wav": b"RIFF", "ref_text": "hi"}
    wavs, sr = clone.generate_voice_clone("Hello", "Auto", prompt, temperature=0.8, top_p=0.5, top_k=3)
    assert sr == 24000 and len(wavs) == 1
    assert set(json.loads(sent[0])) == {"text", "ref_audio", "ref_text", "language"}


# ---------------------------------------------------------------------------------------------- the model name

def _start_without_a_process(monkeypatch):
    started = {}

    class _Proc:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    class _Health(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def popen(cmd, *a, **k):
        started["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(mlx_clone.subprocess, "Popen", popen)
    monkeypatch.setattr(mlx_clone.urllib.request, "urlopen", lambda *a, **k: _Health(b'{"ok": true}'))
    return started


def test_default_model_is_the_published_8bit_one():
    assert mlx_clone.DEFAULT_MODEL == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"


def test_the_worker_is_started_with_the_default_model(monkeypatch):
    monkeypatch.delenv("MLX_CLONE_MODEL", raising=False)
    started = _start_without_a_process(monkeypatch)
    c = mlx_clone.MlxQwen3Clone("/fake/python")
    assert started["cmd"][started["cmd"].index("--model") + 1] == mlx_clone.DEFAULT_MODEL
    assert c.model_name == mlx_clone.DEFAULT_MODEL


def test_mlx_clone_model_overrides_the_default(monkeypatch):
    monkeypatch.setenv("MLX_CLONE_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
    started = _start_without_a_process(monkeypatch)
    c = mlx_clone.MlxQwen3Clone("/fake/python")
    assert started["cmd"][started["cmd"].index("--model") + 1] == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    assert c.model_name.endswith("bf16")


# ---------------------------------------------------------------------------------------------- 24 kHz reference

def test_the_clone_engine_asks_for_a_24k_reference():
    assert mlx_clone.MlxQwen3Clone.prompt_sample_rate == 24000


def test_create_voice_profile_decodes_at_the_engines_rate(monkeypatch):
    import backend

    rates, prompts = [], []

    async def fake_load(audio_bytes, sample_rate=16000):
        rates.append(sample_rate)
        return np.zeros(sample_rate, dtype=np.float32), sample_rate

    class _Mlx(mlx_clone.MlxQwen3Clone):
        proc = None

        def __init__(self):  # no worker process
            pass

        def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
            prompts.append((ref_audio[0][1], ref_text, x_vector_only_mode))
            return {"backend": mlx_clone.PROMPT_BACKEND}

    class _Other:  # an engine without the attribute keeps the 16 kHz default
        def create_voice_clone_prompt(self, **kw):
            return {}

    monkeypatch.setattr(backend, "_load_audio_bytes", fake_load)
    asyncio.run(backend._create_voice_profile(_Mlx(), b"x", "some words"))
    assert rates == [24000] and prompts == [(24000, "some words", False)]
    rates.clear()
    asyncio.run(backend._create_voice_profile(_Other(), b"x", None))
    assert rates == [16000]


def test_a_16k_prompt_made_before_this_change_still_drives_the_clone_engine():
    """A stored prompt embeds its reference WAV at whatever rate it was decoded at; the worker resamples, so a 16 kHz one
    is neither refused by _is_clone_prompt nor rebuilt: the decision keys on the prompt's backend tag, not its rate."""
    import backend

    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    legacy = {"backend": mlx_clone.PROMPT_BACKEND, "ref_wav": buf.getvalue(), "ref_text": "hello"}
    engine = object.__new__(mlx_clone.MlxQwen3Clone)
    engine.proc = None
    assert backend._is_clone_prompt(legacy, engine) is True
    new = io.BytesIO()
    sf.write(new, np.zeros(24000, dtype=np.float32), 24000, format="WAV")
    assert backend._is_clone_prompt({**legacy, "ref_wav": new.getvalue()}, engine) is True
    assert backend._is_clone_prompt({"ref_wav": b"x", "ref_text": "hello"}, engine) is False   # another engine's prompt


# ---------------------------------------------------------------------------------------------- upload storage

class _Request:
    pass


@pytest.fixture()
def upload(monkeypatch, tmp_path):
    """_clone_voice_core with every dependency faked; run() returns the sample rate of the stored reference WAV and
    state['decoded'] lists the rates the upload was decoded at."""
    import backend

    state = {"decoded": [], "stored": [], "kind": "webm", "mlx": True}

    async def fake_load(audio_bytes, sample_rate=16000):
        state["decoded"].append(sample_rate)
        return np.zeros(sample_rate, dtype=np.float32), sample_rate

    async def ok(*a, **k):
        return {"verified": True}

    async def transcribe(*a, **k):
        return "words"

    async def consent(*a, **k):
        return "consent-1"

    async def load_model(key):
        return object()

    async def profile(model, audio, transcript=None):
        return {"backend": mlx_clone.PROMPT_BACKEND}

    monkeypatch.setattr(backend, "sniff_audio_kind", lambda b: state["kind"])
    monkeypatch.setattr(backend, "_load_audio_bytes", fake_load)
    monkeypatch.setattr(backend, "_verify_consent", ok)
    monkeypatch.setattr(backend, "_verify_speaker", ok)
    monkeypatch.setattr(backend, "_auto_transcribe", transcribe)
    monkeypatch.setattr(backend, "_record_consent", consent)
    monkeypatch.setattr(backend.consent_records, "build_record", lambda **k: {})
    monkeypatch.setattr(backend, "_storage_mode", lambda request: "local")
    monkeypatch.setattr(backend, "mlx_python", lambda: "/fake/python" if state["mlx"] else None)
    monkeypatch.setattr(backend.advanced_manager, "load_model", load_model)
    monkeypatch.setattr(backend, "_create_voice_profile", profile)
    monkeypatch.setattr(backend, "_serialize_prompt", lambda p: b"pt")
    monkeypatch.setattr(backend, "LOCAL_AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(backend, "_voice_file_write", lambda path, vid, label, data: state["stored"].append((path, data)))
    monkeypatch.setattr(backend, "save_local_voice", lambda row: None)

    def run(language="en", kind="webm", mlx=True):
        state.update(kind=kind, mlx=mlx)
        state["decoded"].clear()
        state["stored"].clear()
        asyncio.run(backend._clone_voice_core(_Request(), "u1", b"upload", b"consent", "say this", "", language, "Me", ""))
        return next(d for p, d in state["stored"] if p.endswith(".wav"))

    return run, state


def test_a_non_wav_upload_is_stored_at_24k_for_the_mlx_clone_engine(upload):
    run, state = upload
    assert sf.info(io.BytesIO(run("en", kind="webm", mlx=True))).samplerate == 24000
    assert state["decoded"] == [24000]


def test_a_non_wav_upload_stays_16k_when_the_clone_engine_is_not_mlx(upload):
    run, state = upload
    assert sf.info(io.BytesIO(run("en", kind="webm", mlx=False))).samplerate == 16000
    assert state["decoded"] == [16000]


def test_a_wav_upload_is_stored_untouched(upload):
    run, state = upload
    assert run("en", kind="wav", mlx=True) == b"upload"
    assert state["decoded"] == []
