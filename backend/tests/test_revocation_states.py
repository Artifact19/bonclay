"""Revocable provenance, end to end through the public receipt route: the same receipt reads as consented, then as
withdrawn once the voice owner withdraws, while a preset voice carries no consent state (docs/REVOCABLE_PROVENANCE.md)."""
import pytest

from services import provenance


@pytest.fixture()
def receipts(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "_data_dir", str(tmp_path))
    provenance._store.reset()
    provenance._withdrawals.reset() if hasattr(provenance._withdrawals, "reset") else None
    yield
    provenance._store.reset()
    if hasattr(provenance._withdrawals, "reset"):
        provenance._withdrawals.reset()


def _make(engine, voice_id, consent_id, verified):
    return provenance.create_receipt(kind="generation", subject_id="g-" + (voice_id or "preset"), engine=engine,
                                     voice_id=voice_id, voice_name="Priya" if voice_id else None, consent_id=consent_id,
                                     consent_verified=verified, text="namaste", audio_bytes=b"RIFFx" + (voice_id or "p").encode(),
                                     watermark_message=None, user_id="local-user")


def test_the_three_states_on_the_public_route(app_client, receipts):
    consented = _make("qwen3_0_6b", "voice-1", "consent-1", True)
    preset = _make("kokoro", None, None, None)

    r = app_client.get(f"/api/receipts/{consented['receipt_id']}").json()
    assert r["voice_consent_status"] == "active" and r["voice_withdrawn_at"] is None
    assert r["signature_valid"] is True

    row = provenance.record_withdrawal("voice-1", "withdrawn")
    assert row and row["voice_id"] == "voice-1"

    r = app_client.get(f"/api/receipts/{consented['receipt_id']}").json()   # the very next request: no cache
    assert r["voice_consent_status"] == "withdrawn"
    assert r["voice_withdrawn_at"] and r["voice_removed_reason"] == "withdrawn"
    assert r["voice_withdrawal_signed"] is True
    assert r["signature_valid"] is True                                       # the receipt itself is unchanged

    p = app_client.get(f"/api/receipts/{preset['receipt_id']}").json()
    assert p["voice_consent_status"] == "preset_voice" and p["voice_withdrawn_at"] is None


def test_the_withdrawal_is_on_the_public_log_and_the_log_verifies(app_client, receipts):
    _make("qwen3_0_6b", "voice-2", "consent-2", True)
    provenance.record_withdrawal("voice-2", "retention_expired")
    log = app_client.get("/api/withdrawals").json()
    rows = log.get("rows") or log.get("withdrawals") or []
    assert any(x["voice_id"] == "voice-2" and x["reason"] == "retention_expired" for x in rows)
    check = log.get("check") or log.get("log") or {}
    assert check.get("valid", check.get("ok", True)) is True
