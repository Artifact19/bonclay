"""Offline mode: every outbound path refuses, and the attestation says what the configuration allows."""
import base64
import json

import pytest

from core import offline
from services import integrations, llm, provenance


@pytest.fixture
def offline_on(monkeypatch):
    monkeypatch.setenv("BONCLAY_OFFLINE", "1")


def test_is_local_url():
    assert offline.is_local_url("http://localhost:11434/v1")
    assert offline.is_local_url("http://127.0.0.1:1234")
    assert offline.is_local_url("http://192.168.1.20:8000")
    assert not offline.is_local_url("https://api.openai.com/v1")
    assert not offline.is_local_url("https://8.8.8.8/")
    assert not offline.is_local_url("")


def test_block_is_a_noop_online_and_a_503_offline(monkeypatch):
    monkeypatch.delenv("BONCLAY_OFFLINE", raising=False)
    offline.block("anything")
    monkeypatch.setenv("BONCLAY_OFFLINE", "1")
    with pytest.raises(Exception) as e:
        offline.block("WhatsApp")
    assert getattr(e.value, "code", "") == "OFFLINE_MODE" and getattr(e.value, "status_code", 503) == 503


def test_groq_and_remote_llm_servers_are_unavailable_offline(offline_on, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    assert "Offline mode" in llm._groq_reason()
    assert "Offline mode" in llm._openai_reason()
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    assert llm._openai_reason() is None     # a local server (Ollama, LM Studio) is still fine


def test_webhooks_and_sheets_refuse_offline(offline_on):
    status, error, retryable = integrations._post_pinned("https://example.com/hook", b"{}", {}, 5.0)
    assert status is None and "Offline mode" in error and retryable is False
    with pytest.raises(Exception) as e:
        integrations._google_http("GET", "https://sheets.googleapis.com/v4/spreadsheets/x")
    assert getattr(e.value, "code", "") == "OFFLINE_MODE"


def test_calendar_is_not_configured_offline(offline_on, monkeypatch):
    from services import gcal
    monkeypatch.setattr(gcal, "calendar_id", lambda: "cal", raising=False)
    monkeypatch.setattr(gcal, "key_path", lambda: __file__, raising=False)
    assert gcal.configured() is False


def test_attestation_is_signed_and_says_not_offline_by_default(app_client, monkeypatch):
    monkeypatch.delenv("BONCLAY_OFFLINE", raising=False)
    body = app_client.get("/api/privacy/attestation").json()
    st = body["statement"]
    assert st["claim"] == "not_offline" and st["settings"]["offline_mode"] is False
    raw = json.dumps(st, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert provenance._verify_bytes(raw, body["signature_ed25519"], body["public_key"]) is True
    tampered = json.dumps({**st, "claim": "configured_to_keep_data_on_this_machine"}, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert provenance._verify_bytes(tampered, body["signature_ed25519"], body["public_key"]) is False


def test_attestation_claims_local_only_when_every_remote_path_is_off(app_client, monkeypatch):
    import sys
    b = sys.modules["backend"]
    monkeypatch.setenv("BONCLAY_OFFLINE", "1")
    monkeypatch.setattr(b, "groq_client", None, raising=False)
    monkeypatch.setattr(b, "supabase", None, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "x")           # set, but offline mode makes Groq unavailable
    st = app_client.get("/api/privacy/attestation").json()["statement"]
    assert st["settings"]["offline_mode"] is True
    assert st["settings"]["language_model_remote_possible"] is False
    assert st["claim"] == "configured_to_keep_data_on_this_machine"
    assert "configuration only" in st["scope"]
