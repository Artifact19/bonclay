"""Kokoro on MLX with PyTorch fallback: KOKORO_ENGINE routing, the worker client, fallback for one request, no PyTorch
model loaded while MLX speaks, and the audio format (24 kHz mono float32) being the same on both engines.

The worker is faked: a tiny stdlib HTTP server that speaks the worker's protocol, or an object with the handle's
surface. Nothing here starts a process or loads a model; the one live test is opt in (KOKORO_MLX_LIVE=1)."""
import asyncio
import io
import json
import logging
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

mk = pytest.importorskip("services.mlx_kokoro")
worker = pytest.importorskip("services.mlx_kokoro_worker")


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """No remembered failures or 'already logged' keys from another test."""
    monkeypatch.setattr(mk, "_failed_at", None)
    monkeypatch.setattr(mk, "_logged", set())


@pytest.fixture()
def apple(monkeypatch):
    """Pretend an MLX environment exists (the worker is faked anyway)."""
    monkeypatch.setattr(mk, "mlx_python", lambda: "/fake/.venv-mlx/bin/python")


# ------------------------------------------------------------------------------------------ a fake worker
class FakeWorker:
    """Speaks the worker's HTTP protocol. `mode`: ok | unsupported | error | (closed: server not running)."""

    def __init__(self, mode="ok", samples=24000):
        self.mode, self.samples, self.requests = mode, samples, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                outer.requests.append(body)
                if outer.mode == "unsupported":
                    self._send(422, "application/json", json.dumps({"unsupported": True, "error": "ImportError: misaki.ja"}).encode())
                elif outer.mode == "error":
                    self._send(500, "application/json", json.dumps({"error": "RuntimeError: boom"}).encode())
                else:
                    buf = io.BytesIO()
                    sf.write(buf, 0.1 * np.sin(np.arange(outer.samples) / 20.0).astype(np.float32), 24000,
                             format="WAV", subtype="FLOAT")
                    self._send(200, "audio/wav", buf.getvalue())

            def _send(self, code, ctype, data):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def handle_for(port, torch_pipeline=None):
    """An MlxKokoro that talks to `port` without starting a process."""
    h = mk.MlxKokoro.__new__(mk.MlxKokoro)
    h.model_name, h.port = "fake", port
    h.last_engine = h.last_error = None
    h._inflight, h._lock, h._torch_lock = 0, threading.Lock(), threading.Lock()
    h._torch = torch_pipeline
    h.proc = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda timeout=None: 0)
    return h


@pytest.fixture()
def fake_worker():
    made = []

    def make(mode="ok", **kw):
        w = FakeWorker(mode, **kw)
        made.append(w)
        return w

    yield make
    for w in made:
        try:
            w.stop()
        except Exception:  # noqa: BLE001
            pass


class FakeTorchPipeline:
    """Stands in for kokoro.KPipeline: yields (graphemes, phonemes, torch tensor) like the real one."""

    def __init__(self, lang_code="a", model=None):
        self.lang_code, self.model, self.calls = lang_code, model or object(), []

    def __call__(self, text, voice=None, speed=1.0):
        import torch
        self.calls.append((text, voice))
        yield "g", "p", torch.full((12000,), 0.05, dtype=torch.float32)
        yield "g", "p", torch.full((12000,), 0.05, dtype=torch.float32)


# ------------------------------------------------------------------------------------------ KOKORO_ENGINE routing
def test_default_mode_is_auto_and_bad_values_fall_back_to_auto(monkeypatch, caplog):
    monkeypatch.delenv("KOKORO_ENGINE", raising=False)
    assert mk.engine_mode() == "auto"
    for good in ("mlx", "torch", " MLX "):
        monkeypatch.setenv("KOKORO_ENGINE", good)
        assert mk.engine_mode() == good.strip().lower()
    monkeypatch.setenv("KOKORO_ENGINE", "gpu")
    with caplog.at_level(logging.WARNING, logger="bonclay.mlx_kokoro"):
        assert mk.engine_mode() == "auto"
        assert mk.engine_mode() == "auto"
    assert sum("KOKORO_ENGINE" in r.message for r in caplog.records) == 1  # logged once


@pytest.mark.parametrize("mode,expect", [("auto", True), ("mlx", True), ("torch", False)])
def test_mlx_is_used_in_auto_and_mlx_modes_but_never_in_torch_mode(monkeypatch, apple, mode, expect):
    monkeypatch.setenv("KOKORO_ENGINE", mode)
    assert mk.wants_mlx() is expect
    assert (mk.unavailable_reason() is None) is expect


def test_auto_uses_pytorch_where_there_is_no_mlx_environment(monkeypatch):
    monkeypatch.setenv("KOKORO_ENGINE", "auto")
    monkeypatch.setattr(mk, "mlx_python", lambda: None)  # Linux, Windows, Intel Mac, or setup_mlx.sh not run
    assert mk.wants_mlx() is False
    assert "Apple Silicon" in mk.unavailable_reason()


def test_mlx_python_needs_apple_silicon(monkeypatch, tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("MLX_PYTHON", str(fake))
    monkeypatch.setattr(mk.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mk.platform, "machine", lambda: "arm64")
    assert mk.mlx_python() == str(fake)
    monkeypatch.setattr(mk.platform, "machine", lambda: "x86_64")
    assert mk.mlx_python() is None
    monkeypatch.setattr(mk.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mk.platform, "machine", lambda: "arm64")
    assert mk.mlx_python() is None


def test_a_failed_start_means_pytorch_for_a_minute_then_the_worker_is_tried_again(monkeypatch, apple, caplog):
    monkeypatch.setenv("KOKORO_ENGINE", "auto")
    with caplog.at_level(logging.WARNING, logger="bonclay.mlx_kokoro"):
        mk.note_start_failure(RuntimeError("no misaki"))
        mk.note_start_failure(RuntimeError("no misaki"))
    assert mk.wants_mlx() is False
    assert sum("could not start" in r.message for r in caplog.records) == 1
    monkeypatch.setattr(mk, "_failed_at", mk.time.time() - mk.RETRY_AFTER_S - 1)
    assert mk.wants_mlx() is True


# ------------------------------------------------------------------------------------------ the worker client
def test_worker_audio_is_24k_mono_float32(fake_worker):
    w = fake_worker("ok", samples=12000)
    audio = handle_for(w.port).generate("Hello there.", "af_heart", "a", 1.0)
    assert audio.dtype == np.float32 and audio.ndim == 1 and audio.shape == (12000,)
    assert w.requests == [{"text": "Hello there.", "voice": "af_heart", "lang_code": "a", "speed": 1.0}]


def test_unsupported_voice_returns_none_for_that_request_and_logs_once(fake_worker, caplog):
    w = fake_worker("unsupported")
    h = handle_for(w.port)
    with caplog.at_level(logging.WARNING, logger="bonclay.mlx_kokoro"):
        assert h.generate("こんにちは", "jf_alpha", "j") is None
        assert h.generate("こんにちは", "jf_alpha", "j") is None
    assert h.last_engine == "torch" and "jf_alpha" in h.last_error
    assert sum("cannot speak language 'j'" in r.message for r in caplog.records) == 1
    with pytest.raises(mk.Unsupported):
        h.synthesize("こんにちは", "jf_alpha", "j")


def test_worker_error_and_dead_worker_both_mean_fallback_not_an_exception(fake_worker, caplog):
    w = fake_worker("error")
    h = handle_for(w.port)
    with caplog.at_level(logging.WARNING, logger="bonclay.mlx_kokoro"):
        assert h.generate("hi", "af_heart", "a") is None
        w.stop()  # now nothing listens on the port
        assert h.generate("hi", "af_heart", "a") is None
        assert h.generate("hi", "af_heart", "a") is None
    assert h.last_engine == "torch" and "MLX worker failed" in h.last_error
    assert sum("worker failed" in r.message for r in caplog.records) == 1  # logged once, not per sentence


def test_empty_audio_counts_as_a_failure_so_pytorch_speaks(fake_worker):
    w = fake_worker("ok", samples=0)
    assert handle_for(w.port).generate("...", "af_heart", "a") is None


def test_handle_reports_busy_while_a_request_runs_and_alive_follows_the_process(fake_worker):
    h = handle_for(fake_worker("ok").port)
    assert h.alive() and h.busy is False
    h._inflight = 1
    assert h.busy is True
    h.proc = types.SimpleNamespace(poll=lambda: 1, terminate=lambda: None)
    assert h.alive() is False


# ------------------------------------------------------------------------------------------ backend glue
@pytest.fixture()
def be(monkeypatch):
    try:
        import backend
    except BaseException as exc:  # noqa: BLE001
        pytest.skip(f"backend not importable here: {exc}")
    monkeypatch.setattr(backend, "_kokoro_lang_pipelines", {})
    return backend


class FakeHandle:
    """The surface of MlxKokoro that backend uses; records what happened."""
    def __init__(self, audio=None):
        self.audio, self.calls, self.torch = audio, [], FakeTorchPipeline()
        self.last_engine = self.last_error = None
        self.torch_built = 0

    def generate(self, text, voice, code, speed=1.0):
        self.calls.append((text, voice, code))
        self.last_engine = "mlx" if self.audio is not None else "torch"
        return self.audio

    def torch_pipeline(self):
        self.torch_built += 1
        return self.torch


@pytest.fixture()
def as_handle(monkeypatch):
    """Make FakeHandle count as an MlxKokoro for backend's isinstance-based check."""
    monkeypatch.setattr(mk, "is_handle", lambda o: isinstance(o, FakeHandle))


def test_english_goes_to_mlx_and_pytorch_is_never_built(be, as_handle):
    audio = np.full(5000, 0.1, dtype=np.float32)
    h = FakeHandle(audio)
    out = be._kokoro_generate(h, "Hello.", "en")
    assert out is audio and h.calls == [("Hello.", "af_heart", "a")]
    assert h.torch_built == 0 and h.torch.calls == []


def test_hindi_uses_the_hindi_voice_on_mlx(be, as_handle):
    h = FakeHandle(np.zeros(100, dtype=np.float32))
    be._kokoro_generate(h, "नमस्ते", "hi")
    be._kokoro_generate(h, "नमस्ते", "mr")  # an Indian language with no engine of its own speaks Hindi as before
    assert [c[1:] for c in h.calls] == [("hf_alpha", "h"), ("hf_alpha", "h")]


def test_a_request_mlx_cannot_serve_falls_back_to_pytorch_for_that_request_only(be, as_handle):
    h = FakeHandle(None)
    out = be._kokoro_generate(h, "Hello.", "en")
    assert h.torch_built == 1 and h.torch.calls == [("Hello.", "af_heart")]
    assert out.dtype == np.float32 and out.ndim == 1 and out.shape == (24000,)
    h.audio = np.ones(10, dtype=np.float32)      # the next request is MLX again
    assert be._kokoro_generate(h, "Again.", "en").shape == (10,)
    assert h.torch_built == 1


def test_with_voice_helper_routes_by_voice_prefix(be, as_handle):
    h = FakeHandle(np.zeros(10, dtype=np.float32))
    be._kokoro_with_voice(h, "Hola", "ef_dora")
    assert h.calls == [("Hola", "ef_dora", "e")]
    h2 = FakeHandle(None)
    out = be._kokoro_with_voice(h2, "Hello", "am_adam")
    assert h2.torch.calls == [("Hello", "am_adam")] and out.dtype == np.float32


def test_a_plain_pytorch_pipeline_still_takes_the_old_path(be):
    pipe = FakeTorchPipeline()
    out = be._kokoro_generate(pipe, "Hello.", "en")
    assert pipe.calls == [("Hello.", "af_heart")] and out.dtype == np.float32 and out.ndim == 1


def test_audio_format_is_identical_on_both_engines_and_through_the_wav_writer(be, as_handle):
    mlx_audio = be._kokoro_generate(FakeHandle(np.full(24000, 0.05, dtype=np.float32)), "Hi.", "en")
    torch_audio = be._kokoro_generate(FakeTorchPipeline(), "Hi.", "en")
    for a in (mlx_audio, torch_audio):
        assert a.dtype == np.float32 and a.ndim == 1 and a.shape == (24000,)
        data, sr = sf.read(io.BytesIO(be._wav_bytes(a, 24000)), dtype="float32")
        assert sr == 24000 and data.ndim == 1 and data.shape == (24000,)
    np.testing.assert_allclose(mlx_audio, torch_audio, atol=1e-6)  # the fakes use the same sample value


def _fake_kokoro_package(monkeypatch):
    built = []

    class KPipeline(FakeTorchPipeline):
        def __init__(self, lang_code="a", model=None):
            super().__init__(lang_code, model)
            built.append(lang_code)

    pkg = types.ModuleType("kokoro")
    pkg.KPipeline = KPipeline
    monkeypatch.setitem(sys.modules, "kokoro", pkg)
    return built


def test_loading_kokoro_starts_the_mlx_worker_and_does_not_build_pytorch_kokoro(be, monkeypatch, apple):
    built = _fake_kokoro_package(monkeypatch)
    monkeypatch.setenv("KOKORO_ENGINE", "auto")
    sentinel = object()
    monkeypatch.setattr(mk, "MlxKokoro", lambda *a, **k: sentinel)
    model = be.AdvancedModelManager()._instantiate_model_sync("kokoro")
    assert model is sentinel and built == []


def test_loading_kokoro_falls_back_to_pytorch_when_the_worker_cannot_start(be, monkeypatch, apple, caplog):
    built = _fake_kokoro_package(monkeypatch)
    monkeypatch.setenv("KOKORO_ENGINE", "auto")

    def boom(*a, **k):
        raise RuntimeError("MLX Kokoro worker exited with code 1")

    monkeypatch.setattr(mk, "MlxKokoro", boom)
    with caplog.at_level(logging.WARNING, logger="bonclay.mlx_kokoro"):
        model = be.AdvancedModelManager()._instantiate_model_sync("kokoro")
        be.AdvancedModelManager()._instantiate_model_sync("kokoro")
    assert isinstance(model, FakeTorchPipeline) and built == ["a", "a"]
    assert sum("could not start" in r.message for r in caplog.records) == 1
    assert mk.wants_mlx() is False  # and it does not retry on every turn


def test_torch_mode_never_touches_the_worker(be, monkeypatch, apple):
    built = _fake_kokoro_package(monkeypatch)
    monkeypatch.setenv("KOKORO_ENGINE", "torch")
    monkeypatch.setattr(mk, "MlxKokoro", lambda *a, **k: pytest.fail("worker started in torch mode"))
    assert isinstance(be.AdvancedModelManager()._instantiate_model_sync("kokoro"), FakeTorchPipeline)
    assert built == ["a"]


# ------------------------------------------------------------------------------------------ /api/kokoro/generate
LOCAL = {"storage-mode": "local"}


@pytest.fixture()
def router_mlx(app_client, monkeypatch, apple):
    """/api/kokoro/generate with the manager returning a FakeHandle and PyTorch synthesis recorded."""
    import backend
    from routers import kokoro as router
    monkeypatch.setenv("KOKORO_ENGINE", "mlx")
    monkeypatch.setattr(mk, "is_handle", lambda o: isinstance(o, FakeHandle))
    state = types.SimpleNamespace(handle=FakeHandle(np.full(12000, 0.05, dtype=np.float32)), torch_calls=[])

    async def load_model(key):
        assert key == "kokoro"
        return state.handle

    monkeypatch.setattr(backend.advanced_manager, "load_model", load_model)
    monkeypatch.setattr(router, "_synthesize", lambda *a: state.torch_calls.append(a) or _wav(0.2))
    return state


def _wav(seconds):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(24000 * seconds), dtype=np.float32), 24000, format="WAV")
    return buf.getvalue()


def test_kokoro_generate_endpoint_uses_mlx_and_skips_pytorch(app_client, router_mlx):
    r = app_client.post("/api/kokoro/generate", headers=LOCAL, json={"voice": "hf_alpha", "text": "नमस्ते"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 24000 and data.ndim == 1
    assert router_mlx.handle.calls == [("नमस्ते", "hf_alpha", "h")]
    assert router_mlx.torch_calls == []
    assert r.headers.get("X-Receipt-Id")  # still watermarked and receipted


def test_kokoro_generate_endpoint_falls_back_when_mlx_cannot_serve(app_client, router_mlx):
    router_mlx.handle.audio = None
    r = app_client.post("/api/kokoro/generate", headers=LOCAL, json={"voice": "af_heart", "text": "hello"})
    assert r.status_code == 200
    assert len(router_mlx.torch_calls) == 1


# ------------------------------------------------------------------------------------------ warm-up
def test_warmup_reports_mlx_or_the_fallback(be, monkeypatch, apple):
    from routers import warmup
    monkeypatch.setenv("KOKORO_ENGINE", "mlx")
    monkeypatch.setattr(mk, "is_handle", lambda o: isinstance(o, FakeHandle))
    monkeypatch.setattr(warmup, "_has", lambda m: False)  # PyTorch Kokoro need not even be installed for MLX
    h = FakeHandle(np.zeros(24000, dtype=np.float32))

    async def load_model(key):
        return h

    monkeypatch.setattr(be.advanced_manager, "load_model", load_model)
    detail = asyncio.run(warmup._kokoro("en", "Hello"))
    assert "MLX" in detail and "1.0 s" in detail
    assert h.torch_built == 0

    h.audio = None
    h.last_error = "MLX cannot speak hf_alpha: x"
    detail = asyncio.run(warmup._kokoro("en", "Hello"))
    assert "PyTorch fallback" in detail


# ------------------------------------------------------------------------------------------ the worker's synthesis
class FakeMlxPipeline:
    """KokoroPipeline stand-in: yields results whose audio is a (1, N) array (MLX shape) per newline-split segment."""
    def __init__(self):
        self.calls = []

    def __call__(self, text, voice=None, speed=1.0, split_pattern=r"\n+"):
        self.calls.append((text, voice, speed, split_pattern))
        for _ in text.split("\n"):
            yield types.SimpleNamespace(audio=np.full((1, 100), 0.1, dtype=np.float64))
        yield types.SimpleNamespace(audio=None)


def test_worker_synthesis_returns_a_flat_float32_array_and_caches_one_pipeline_per_language():
    pipe = FakeMlxPipeline()
    model = types.SimpleNamespace(_get_pipeline=lambda code: pipe)
    pipelines = {}
    audio = worker.synthesize(model, pipelines, "one\ntwo", "af_heart", "a", 1.2)
    assert audio.dtype == np.float32 and audio.shape == (200,)
    assert pipe.calls == [("one\ntwo", "af_heart", 1.2, r"\n+")]
    worker.synthesize(model, pipelines, "x", "af_heart", "a", 1.0)
    assert list(pipelines) == ["a"]


def test_worker_synthesis_with_nothing_spoken_is_an_error():
    empty = types.SimpleNamespace(_get_pipeline=lambda code: (lambda *a, **k: iter([])))
    with pytest.raises(RuntimeError):
        worker.synthesize(empty, {}, "...", "af_heart", "a", 1.0)


# ------------------------------------------------------------------------------------------ live (opt in)
@pytest.mark.skipif(os.getenv("KOKORO_MLX_LIVE") != "1", reason="starts the real MLX worker; set KOKORO_MLX_LIVE=1")
def test_live_worker_speaks_english_and_hindi_in_the_pytorch_format():
    h = mk.MlxKokoro()
    try:
        for voice, code, text in (("af_heart", "a", "Hello, how can I help you today?"),
                                  ("hf_alpha", "h", "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?")):
            audio = h.generate(text, voice, code)
            assert audio is not None and audio.dtype == np.float32 and audio.ndim == 1
            assert 1.0 < len(audio) / 24000 < 8.0 and float(np.abs(audio).max()) > 0.05
    finally:
        h.close()
