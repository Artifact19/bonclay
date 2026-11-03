"""The local model server's prompt cache limits (services/llm.prompt_cache_args)."""
import pytest

from services import llm


def test_defaults_cap_the_cache(monkeypatch):
    monkeypatch.delenv("MLX_PROMPT_CACHE_MB", raising=False)
    monkeypatch.delenv("MLX_PROMPT_CACHE_SIZE", raising=False)
    assert llm.prompt_cache_args() == ["--prompt-cache-bytes", "600000000", "--prompt-cache-size", "4"]


@pytest.mark.parametrize("mb,size,expected", [
    ("250", "2", ["--prompt-cache-bytes", "250000000", "--prompt-cache-size", "2"]),
    ("0", "0", []),                                              # 0 removes a limit
    ("0", "6", ["--prompt-cache-size", "6"]),
    ("lots", "", ["--prompt-cache-bytes", "600000000", "--prompt-cache-size", "4"]),   # junk falls back to the defaults
])
def test_env_overrides(monkeypatch, mb, size, expected):
    monkeypatch.setenv("MLX_PROMPT_CACHE_MB", mb)
    monkeypatch.setenv("MLX_PROMPT_CACHE_SIZE", size)
    assert llm.prompt_cache_args() == expected
