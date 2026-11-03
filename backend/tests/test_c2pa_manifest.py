"""C2PA manifests on exported clips (services/c2pa_manifest.py). The first tests need no library: the flag is off by
default, a missing library is reported and never breaks a generation. The rest run only where c2pa-python is installed:
the manifest embeds, leaves the samples untouched, reads back with the receipt id, and the receipt's hash still matches
the delivered file."""
import hashlib
import importlib.util
import io
import os

import numpy as np
import pytest
import soundfile as sf

from services import c2pa_manifest as cm
from services import provenance


def _wav(seconds: float = 0.5, sr: int = 24000) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.delenv("C2PA_MANIFEST", raising=False)
    monkeypatch.delenv("C2PA_CERT_FILE", raising=False)
    monkeypatch.delenv("C2PA_KEY_FILE", raising=False)
    monkeypatch.delenv("BONCLAY_PUBLIC_URL", raising=False)
    monkeypatch.setattr(cm, "_data_dir", str(tmp_path))
    monkeypatch.setattr(provenance, "_data_dir", str(tmp_path))
    provenance._store.reset()
    yield
    provenance._store.reset()


# ---------------------------------------------------------------------------------------------- without the library

def test_off_by_default_and_a_missing_library_is_reported(monkeypatch):
    assert cm.enabled() is False and cm.active() is False
    monkeypatch.setattr(cm, "_import_c2pa", lambda: (_ for _ in ()).throw(ImportError("no c2pa")))
    assert "pip install c2pa-python" in cm.unavailable_reason()
    monkeypatch.setenv("C2PA_MANIFEST", "on")
    assert cm.enabled() is True and cm.active() is False
    assert cm.read_summary(_wav()) is None
    assert cm.status()["available"] is False and cm.status()["enabled"] is True


def test_the_flag_off_leaves_apply_provenance_bytes_unchanged(monkeypatch):
    monkeypatch.setattr(cm, "_import_c2pa", lambda: (_ for _ in ()).throw(ImportError("no c2pa")))
    wav = _wav()
    out, receipt = provenance.apply_provenance(kind="generation", subject_id="g1", wav_bytes=wav, text="hi",
                                               engine="kokoro", voice_id=None, voice_name=None, consent_id=None,
                                               consent_verified=None, user_id="local-user")
    assert receipt and receipt["audio_sha256"] == hashlib.sha256(out).hexdigest()
    assert cm.read_summary(out) is None                       # no library: nothing to read


def test_the_flag_on_without_the_library_still_delivers_the_clip(monkeypatch):
    monkeypatch.setenv("C2PA_MANIFEST", "on")
    monkeypatch.setattr(cm, "_import_c2pa", lambda: (_ for _ in ()).throw(ImportError("no c2pa")))
    out, receipt = provenance.apply_provenance(kind="generation", subject_id="g2", wav_bytes=_wav(), text="hi",
                                               engine="kokoro", voice_id=None, voice_name=None, consent_id=None,
                                               consent_verified=None, user_id="local-user")
    assert receipt and receipt["audio_sha256"] == hashlib.sha256(out).hexdigest()


def test_sniffing_and_the_manifest_body():
    assert cm.sniff_mime(_wav()) == "audio/wav"
    assert cm.sniff_mime(b"ID3\x04\x00" + b"\x00" * 20) == "audio/mpeg"
    assert cm.sniff_mime(b"\xff\xfb\x90\x00" + b"\x00" * 20) == "audio/mpeg"
    assert cm.sniff_mime(b"OggS" + b"\x00" * 20) is None
    body = cm.manifest("r-1", engine="kokoro", voice_id=None, key_id="abcd", watermark="audioseal_16bits",
                       verify_base="https://verify.example")
    labels = [a["label"] for a in body["assertions"]]
    assert labels == ["c2pa.actions", cm.RECEIPT_LABEL]
    action = body["assertions"][0]["data"]["actions"][0]
    assert action["action"] == "c2pa.created" and action["digitalSourceType"].endswith("trainedAlgorithmicMedia")
    receipt = body["assertions"][1]["data"]
    assert receipt["receipt_id"] == "r-1" and receipt["receipt_key_id"] == "abcd"
    assert receipt["verify_url"] == "https://verify.example/api/receipts/r-1"
    assert "voice_id" not in receipt                            # None fields are left out


def test_a_self_signed_test_chain_is_made_once_with_a_private_key(tmp_path):
    chain, key = cm.signer_material()
    assert b"BEGIN CERTIFICATE" in chain and chain.count(b"BEGIN CERTIFICATE") == 2
    assert b"BEGIN PRIVATE KEY" in key
    if os.name != "nt":
        assert oct((tmp_path / "c2pa" / "signer.key").stat().st_mode & 0o777) == "0o600"
    again, _ = cm.signer_material()
    assert again == chain                                       # reused, not regenerated


# ---------------------------------------------------------------------------------------------- with the library

needs_c2pa = pytest.mark.skipif(importlib.util.find_spec("c2pa") is None,
                                reason="c2pa-python is not installed in this environment")


@needs_c2pa
def test_embed_keeps_the_samples_and_reads_back(monkeypatch):
    monkeypatch.setenv("C2PA_MANIFEST", "on")
    wav = _wav()
    signed = cm.embed(wav, "r-42", engine="kokoro", key_id="k1", watermark="audioseal_16bits",
                      verify_base="https://verify.example")
    assert signed != wav and len(signed) > len(wav)
    a, sr_a = sf.read(io.BytesIO(wav), dtype="float32")
    b, sr_b = sf.read(io.BytesIO(signed), dtype="float32")
    assert sr_a == sr_b and np.array_equal(a, b)
    summary = cm.read_summary(signed)
    assert summary["present"] is True and summary["generator"] == "BonClay"
    assert summary["receipt"]["receipt_id"] == "r-42" and summary["receipt"]["receipt_key_id"] == "k1"
    assert cm.RECEIPT_LABEL in summary["assertions"]
    assert cm.read_summary(wav) == {"present": False}


@needs_c2pa
def test_a_tampered_sample_breaks_the_manifest(monkeypatch):
    monkeypatch.setenv("C2PA_MANIFEST", "on")
    signed = bytearray(cm.embed(_wav(), "r-43", engine="kokoro"))
    signed[len(signed) // 2] ^= 0x01
    summary = cm.read_summary(bytes(signed))
    assert summary["present"] is True
    assert "valid" not in summary["validation_state"].lower() or summary["failures"]


@needs_c2pa
def test_verify_reports_the_manifest_of_an_uploaded_clip(monkeypatch, app_client):
    monkeypatch.setenv("C2PA_MANIFEST", "on")
    out, receipt = provenance.apply_provenance(kind="generation", subject_id="g4", wav_bytes=_wav(1.0), text="hello",
                                               engine="kokoro", voice_id=None, voice_name=None, consent_id=None,
                                               consent_verified=None, user_id="local-user")
    r = app_client.post("/api/verify", files={"file": ("clip.wav", out, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["manifest"]["present"] is True
    assert body["manifest"]["receipt"]["receipt_id"] == receipt["receipt_id"]
    assert body["match"]["exact"] is True and body["match"]["receipt"]["receipt_id"] == receipt["receipt_id"]
    plain = app_client.post("/api/verify", files={"file": ("plain.wav", _wav(1.0), "audio/wav")}).json()
    assert plain["manifest"] == {"present": False}


@needs_c2pa
def test_apply_provenance_embeds_and_the_receipt_hash_matches_the_delivered_file(monkeypatch):
    monkeypatch.setenv("C2PA_MANIFEST", "on")
    out, receipt = provenance.apply_provenance(kind="generation", subject_id="g3", wav_bytes=_wav(), text="namaste",
                                               engine="kokoro", voice_id=None, voice_name=None, consent_id=None,
                                               consent_verified=None, user_id="local-user")
    assert receipt and receipt["audio_sha256"] == hashlib.sha256(out).hexdigest()
    summary = cm.read_summary(out)
    assert summary["present"] is True and summary["receipt"]["receipt_id"] == receipt["receipt_id"]
    assert summary["receipt"]["receipt_key_id"] == receipt["key_id"]
    # the exact match on /verify still finds it
    assert provenance.find_by_audio_sha(hashlib.sha256(out).hexdigest())[0]["receipt_id"] == receipt["receipt_id"]
