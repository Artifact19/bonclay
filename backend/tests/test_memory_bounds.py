"""The memory fixes from the 45-minute soak: what grew, and the bounds that keep it from growing again.

These run without MLX or AudioSeal: the pieces that need them are replaced by small fakes, and what is checked is
the behaviour this project added (launcher defaults, resident ASR models, eager watermark models)."""
import importlib
import os
import sys
import types

import pytest

SERVICES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "services")
if SERVICES not in sys.path:
    sys.path.insert(0, SERVICES)

import parent_watch  # noqa: E402  (services/parent_watch.py is a script folder, not a package module path)


# ------------------------------------------------------------------------------------- mlx_lm server launcher

def test_the_llm_server_gets_a_bounded_prompt_cache_by_default(monkeypatch):
    for name in ("MLX_LM_PROMPT_CACHE_SIZE", "MLX_LM_PROMPT_CACHE_BYTES"):
        monkeypatch.delenv(name, raising=False)
    args = parent_watch.default_mlx_lm_args(["server", "--model", "m", "--port", "1"])
    assert args[:5] == ["server", "--model", "m", "--port", "1"]
    assert args[args.index("--prompt-cache-size") + 1] == "4"
    assert args[args.index("--prompt-cache-bytes") + 1] == "512M"


def test_explicit_prompt_cache_flags_and_env_overrides_win(monkeypatch):
    monkeypatch.setenv("MLX_LM_PROMPT_CACHE_SIZE", "2")
    monkeypatch.setenv("MLX_LM_PROMPT_CACHE_BYTES", "")  # empty: leave the server's own default
    args = parent_watch.default_mlx_lm_args(["server", "--prompt-cache-bytes=1G"])
    assert args == ["server", "--prompt-cache-bytes=1G", "--prompt-cache-size", "2"]
    assert parent_watch.default_mlx_lm_args(["server", "--prompt-cache-size", "9"]).count("--prompt-cache-size") == 1


def test_other_mlx_lm_commands_are_left_alone():
    assert parent_watch.default_mlx_lm_args(["generate", "--model", "m"]) == ["generate", "--model", "m"]
    assert parent_watch.default_mlx_lm_args([]) == []


def test_the_mlx_buffer_pool_is_capped(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(set_cache_limit=lambda n: calls.append(n))
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake))
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    monkeypatch.delenv("MLX_CACHE_LIMIT_MB", raising=False)
    parent_watch.limit_mlx_cache()
    monkeypatch.setenv("MLX_CACHE_LIMIT_MB", "64")
    parent_watch.limit_mlx_cache()
    monkeypatch.setenv("MLX_CACHE_LIMIT_MB", "not a number")
    parent_watch.limit_mlx_cache()  # a bad value must not stop the worker from starting
    assert calls == [parent_watch.DEFAULT_MLX_CACHE_LIMIT_MB * 2 ** 20, 64 * 2 ** 20]


def test_limit_mlx_cache_is_a_no_op_without_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)  # makes `import mlx.core` raise ImportError
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    parent_watch.limit_mlx_cache()


# --------------------------------------------------------------------------------------- speech recognition worker

def test_the_asr_worker_keeps_both_whisper_models_in_memory(monkeypatch):
    """mlx_whisper holds one model, so the hybrid small/large switch reloaded weights on every Hindi clip."""
    loads = []

    class ModelHolder:  # what mlx_whisper.transcribe has: one model, replaced whenever another path is asked for
        model = path = None

        @classmethod
        def get_model(cls, model_path, dtype):
            raise AssertionError("the original single-model loader must not be used")

    load_models = types.ModuleType("mlx_whisper.load_models")
    load_models.load_model = lambda path, dtype=None: loads.append((path, dtype)) or object()
    transcribe = types.ModuleType("mlx_whisper.transcribe")
    transcribe.ModelHolder = ModelHolder
    package = types.ModuleType("mlx_whisper")
    package.__path__ = []
    for name, mod in (("mlx_whisper", package), ("mlx_whisper.load_models", load_models),
                      ("mlx_whisper.transcribe", transcribe)):
        monkeypatch.setitem(sys.modules, name, mod)

    worker = importlib.import_module("mlx_asr_worker")
    worker.keep_models_resident()
    small, large = "small", "large"
    first = [ModelHolder.get_model(small, "f16"), ModelHolder.get_model(large, "f16")]
    again = [ModelHolder.get_model(small, "f16"), ModelHolder.get_model(large, "f16"), ModelHolder.get_model(small, "f16")]
    assert loads == [(small, "f16"), (large, "f16")]  # each model is read from disk once, however often it is asked for
    assert again[0] is first[0] and again[1] is first[1] and again[2] is first[0]
    ModelHolder.get_model(small, "f32")  # a different precision is a different model
    assert len(loads) == 3


# ------------------------------------------------------------------------------------------- watermark models

def test_the_watermark_models_are_loaded_without_torch_compile(monkeypatch):
    from services import provenance

    seen = {}

    class Net:
        def eval(self):
            return self

    class AudioSeal:
        load_generator = staticmethod(lambda *a, **k: Net())
        load_detector = staticmethod(lambda *a, **k: Net())

    fake = types.ModuleType("audioseal")
    fake.AudioSeal = AudioSeal
    previous = os.environ.pop("NO_TORCH_COMPILE", None)
    monkeypatch.setattr(provenance, "_models", None)
    monkeypatch.setattr(provenance, "_load_error", None)
    real_import = __import__

    def spying_import(name, *a, **k):
        if name == "audioseal":  # audioseal reads the variable at import, so it must already be set here
            seen["env_at_import"] = os.environ.get("NO_TORCH_COMPILE")
            return fake
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", spying_import)
    try:
        provenance._load_models()
    finally:
        monkeypatch.setattr("builtins.__import__", real_import)
        if previous is None:
            os.environ.pop("NO_TORCH_COMPILE", None)
        else:
            os.environ["NO_TORCH_COMPILE"] = previous
    assert seen["env_at_import"] == "1"


def test_an_explicit_torch_compile_choice_is_respected(monkeypatch):
    import inspect
    from services import provenance
    # setdefault, not assignment: someone who exports NO_TORCH_COMPILE=0 keeps what they asked for
    assert 'os.environ.setdefault("NO_TORCH_COMPILE", "1")' in inspect.getsource(provenance._load_models)


def test_gemma4_thinking_is_turned_off_for_voice(monkeypatch):
    monkeypatch.delenv("MLX_LM_THINKING", raising=False)
    args = parent_watch.default_mlx_lm_args(["server", "--model", "mlx-community/gemma-4-e4b-it-4bit", "--port", "1"])
    i = args.index("--chat-template-args")
    assert args[i + 1] == '{"enable_thinking": false}'
    assert "--chat-template-args" not in parent_watch.default_mlx_lm_args(["server", "--model", "mlx-community/Llama-3.2-3B-Instruct-4bit"])
    monkeypatch.setenv("MLX_LM_THINKING", "on")
    assert "--chat-template-args" not in parent_watch.default_mlx_lm_args(["server", "--model", "mlx-community/gemma-4-e4b-it-4bit"])
