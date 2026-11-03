"""BONCLAY_COMMERCIAL=1 hides and refuses the engines not cleared for paid use (services/licences.py)."""
import pytest

import backend
from core.errors import AppError
from services import licences

NON_COMMERCIAL = {"dhvaani", "indic_speak", "indic_f5"}


@pytest.fixture()
def commercial(monkeypatch):
    monkeypatch.setenv("BONCLAY_COMMERCIAL", "1")


def test_every_registered_engine_has_a_licence_row():
    for key in backend.ENGINE_REGISTRY:
        assert licences.licence_of(key), key


def test_off_by_default_nothing_is_blocked(monkeypatch):
    monkeypatch.delenv("BONCLAY_COMMERCIAL", raising=False)
    assert licences.commercial_mode() is False
    assert all(licences.blocked_reason(k) is None for k in backend.ENGINE_REGISTRY)


def test_engine_list_drops_the_non_commercial_engines(app_client, commercial):
    ids = {e["id"] for e in app_client.get("/api/engines").json()}
    assert ids.isdisjoint(NON_COMMERCIAL)
    assert {"kokoro", "qwen3_0_6b", "chatterbox_ml"} <= ids
    for e in app_client.get("/api/engines").json():
        assert e["licence"] and e["paid_use"] is True


def test_engine_list_names_licences_when_off(app_client, monkeypatch):
    monkeypatch.delenv("BONCLAY_COMMERCIAL", raising=False)
    rows = {e["id"]: e for e in app_client.get("/api/engines").json()}
    assert rows["dhvaani"]["licence"] == "CC-BY-NC-4.0" and rows["dhvaani"]["paid_use"] is False
    assert rows["kokoro"]["paid_use"] is True


def test_health_lists_only_allowed_engines(app_client, commercial):
    body = app_client.get("/health").json()
    assert body["commercial_mode"] is True
    assert set(body["engines"]).isdisjoint(NON_COMMERCIAL)


def test_indian_preset_speech_goes_to_kokoro(commercial, monkeypatch):
    monkeypatch.setattr(backend.indic_speak_service(), "indic_speak_unavailable_reason", lambda: None)
    assert backend.route_engine("hi", "premium", False) == "kokoro"
    assert backend.route_engine("mr", "fast", False) == "kokoro"


def test_hindi_cloning_goes_to_chatterbox_never_dhvaani(commercial, monkeypatch):
    monkeypatch.setattr(backend.chatterbox_service(), "unavailable_reason", lambda: None)
    monkeypatch.setattr(backend.dhvaani_service(), "unavailable_reason", lambda: None)
    monkeypatch.setenv("INDIC_CLONE_ENGINE", "dhvaani")  # even when the owner prefers DhVaani
    assert backend.route_indic_clone("hi") == backend.CHATTERBOX_ENGINE


def test_other_indian_cloning_is_refused_with_the_licence_code(commercial, monkeypatch):
    monkeypatch.setattr(backend.dhvaani_service(), "unavailable_reason", lambda: None)
    with pytest.raises(AppError) as info:
        backend.route_indic_clone("mr")
    assert info.value.code == licences.ERROR_CODE and info.value.status == 403
    assert "CC-BY-NC" in info.value.message


def test_loading_a_blocked_engine_is_refused(commercial):
    for key in NON_COMMERCIAL:
        with pytest.raises(AppError) as info:
            backend._refuse_if_licence_blocked(key)
        assert info.value.code == licences.ERROR_CODE and info.value.status == 403
    backend._refuse_if_licence_blocked("kokoro")  # no error


def test_receipts_name_the_engine_licence(tmp_path, monkeypatch):
    from services import provenance

    monkeypatch.setattr(provenance, "_data_dir", str(tmp_path), raising=False)
    receipt = provenance.create_receipt(kind="generation", subject_id="g1", engine="kokoro", voice_id=None,
                                        voice_name=None, consent_id=None, consent_verified=None, text="hello",
                                        audio_bytes=b"RIFF....", watermark_message=None, user_id="local-user")
    assert receipt["engine_licence"] == "Apache-2.0"
    assert "engine_licence" not in provenance.SIGNED_FIELDS
