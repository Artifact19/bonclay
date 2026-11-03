"""Receipt signing-key rotation (services/provenance.py): a rotated key is retired, not forgotten. Receipts and
withdrawal-log rows signed before the rotation still verify by their key_id, new ones carry the new id, and the public
key endpoint lists every key."""
import base64

import pytest

from services import provenance


@pytest.fixture()
def fresh_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "_data_dir", str(tmp_path))
    monkeypatch.setattr(provenance, "_signing_cache", None)
    monkeypatch.delenv("BONCLAY_RECEIPT_SIGNING_KEY", raising=False)
    monkeypatch.delenv("BONCLAY_RECEIPT_RETIRED_KEYS", raising=False)
    provenance._store.reset()
    provenance._withdrawals.reset()
    yield tmp_path
    provenance._store.reset()
    provenance._withdrawals.reset()


def _receipt(tag: str) -> dict:
    return provenance.create_receipt(kind="generation", subject_id=f"g-{tag}", engine="kokoro", voice_id=None,
                                     voice_name=None, consent_id=None, consent_verified=None, text=tag,
                                     audio_bytes=b"RIFF" + tag.encode(), watermark_message=None, user_id="local-user")


def test_old_receipts_verify_after_rotation_and_new_ones_carry_the_new_key(fresh_keys):
    before = _receipt("before")
    kid1 = provenance.key_id()
    assert before["key_id"] == kid1 and provenance.signature_ed25519_valid(before) is True

    out = provenance.rotate_signing_key()
    kid2 = provenance.key_id()
    assert out == {"retired_key_id": kid1, "key_id": kid2, "retired_count": 1}
    assert kid2 != kid1

    assert provenance.signature_ed25519_valid(before) is True          # picked by its key_id
    after = _receipt("after")
    assert after["key_id"] == kid2 and provenance.signature_ed25519_valid(after) is True
    # the public view says both check out
    assert provenance.public_receipt(before)["signature_ed25519_valid"] is True
    assert provenance.public_receipt(after)["signature_ed25519_valid"] is True


def test_a_receipt_without_a_key_id_is_tried_against_every_known_key(fresh_keys):
    legacy = dict(_receipt("legacy"))
    legacy.pop("key_id")
    provenance.rotate_signing_key()
    assert provenance.signature_ed25519_valid(legacy) is True
    # and a forged signature still fails against all of them
    bad = {**legacy, "signature_ed25519": base64.b64encode(b"\x00" * 64).decode("ascii")}
    assert provenance.signature_ed25519_valid(bad) is False


def test_the_withdrawal_log_verifies_across_a_rotation(fresh_keys):
    provenance.record_withdrawal("voice-old", "withdrawn")
    provenance.rotate_signing_key()
    provenance.record_withdrawal("voice-new", "retention_expired")
    check = provenance.verify_withdrawal_log()
    assert check["ok"] is True and check["signed_rows"] == 2 and check["broken_at"] is None
    rows = provenance.withdrawal_rows()
    assert rows[0]["key_id"] != rows[1]["key_id"]


def test_the_endpoint_lists_current_and_retired_keys(fresh_keys, app_client):
    kid1 = provenance.key_id()
    provenance.rotate_signing_key()
    body = app_client.get("/.well-known/bonclay-receipt-key").json()
    assert body["key_id"] == provenance.key_id() and body["key_id"] != kid1
    assert [k["status"] for k in body["keys"]] == ["current", "retired"]
    assert body["keys"][1]["key_id"] == kid1
    assert all(len(base64.b64decode(k["public_key"])) == 32 for k in body["keys"])


def test_rotating_twice_keeps_both_old_keys(fresh_keys):
    kids = [provenance.key_id()]
    provenance.rotate_signing_key()
    kids.append(provenance.key_id())
    provenance.rotate_signing_key()
    retired = {k["key_id"] for k in provenance.all_public_keys() if k["status"] == "retired"}
    assert retired == set(kids)


def test_retired_keys_from_the_environment_count_too(fresh_keys, monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    other = Ed25519PrivateKey.generate()
    pub = base64.b64encode(other.public_key().public_bytes(serialization.Encoding.Raw,
                                                           serialization.PublicFormat.Raw)).decode("ascii")
    monkeypatch.setenv("BONCLAY_RECEIPT_RETIRED_KEYS", f" {pub} , not-a-key ")
    assert pub in provenance.retired_public_keys()
    assert "not-a-key" not in provenance.retired_public_keys()
    # a receipt signed by that other key verifies through its key_id
    r = dict(_receipt("env"))
    r["key_id"] = provenance.key_id(pub)
    r["signature_ed25519"] = base64.b64encode(other.sign(provenance._canonical(r))).decode("ascii")
    assert provenance.signature_ed25519_valid(r) is True


def test_an_environment_key_is_not_rotated_by_the_script(fresh_keys, monkeypatch):
    monkeypatch.setenv("BONCLAY_RECEIPT_SIGNING_KEY", base64.b64encode(b"\x07" * 32).decode("ascii"))
    with pytest.raises(RuntimeError, match="BONCLAY_RECEIPT_RETIRED_KEYS"):
        provenance.rotate_signing_key()
