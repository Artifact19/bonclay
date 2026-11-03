"""Security regressions: auth enforced when configured, local header no bypass, GPU mode guard."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import backend  # noqa: E402


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "x" * 40)
    monkeypatch.delenv("BONCLAY_ALLOW_LOCAL_MODE", raising=False)
    monkeypatch.delenv("BONCLAY_ALLOW_DEMO_TOKENS", raising=False)


@pytest.fixture
def client():
    return TestClient(backend.app, raise_server_exceptions=False)


def _audio():
    return {"audio_file": ("a.wav", b"RIFF0000WAVEfmt ", "audio/wav")}


def test_transcribe_requires_auth(auth_env, client):
    assert client.post("/api/transcribe", files=_audio()).status_code == 401


def test_local_storage_header_does_not_bypass(auth_env, client):
    r = client.post("/api/transcribe", files=_audio(), headers={"storage-mode": "local"})
    assert r.status_code == 401
    r = client.post("/api/hardware/mode", json={"mode": "cpu"}, headers={"storage-mode": "local"})
    assert r.status_code == 401


def test_hardware_nvidia_without_gpu_is_409(monkeypatch, client):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("NEXT_PUBLIC_SUPABASE_URL", raising=False)
    monkeypatch.setattr(backend.torch.cuda, "is_available", lambda: False)
    r = client.post("/api/hardware/mode", json={"mode": "nvidia"})
    assert r.status_code == 409
    assert "GPU_UNAVAILABLE" in r.text
