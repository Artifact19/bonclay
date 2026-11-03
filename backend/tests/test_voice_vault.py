"""Voice vault: a cloned voice's files are ciphertext, and withdrawing consent shreds the key.

The first half tests services/voice_vault.py on its own; the second half clones a voice through the API with the vault
on and checks the files on disk, that generation still works, and that withdrawal makes even a copy unreadable."""
import base64
import glob
import json
import os
from pathlib import Path

import pytest

from services import provenance, voice_vault
from tests.test_compliance import cloned, env, clone, fake_seal, isolate  # noqa: F401  (fixtures)
from tests.test_consent import LOCAL, backend_mod, fake_groq  # noqa: F401


@pytest.fixture(autouse=True)
def vault_on(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICE_VAULT", "on")
    monkeypatch.setattr(provenance, "_data_dir", str(tmp_path))
    provenance.configure(data_dir=str(tmp_path))
    voice_vault.reset()


# ---- the vault itself -------------------------------------------------------------------------------------------------

def test_sealed_data_is_ciphertext_and_opens_to_the_original():
    secret = b"RIFF....the owner's voice, sixteen bits at a time" * 20
    sealed = voice_vault.seal("v1", secret, "wav")
    assert voice_vault.is_sealed(sealed) and secret[:20] not in sealed and len(sealed) > len(secret)
    assert voice_vault.unseal("v1", sealed, "wav") == secret
    assert voice_vault.seal("v1", secret, "wav") != sealed          # a fresh nonce every time
    assert voice_vault.seal("v1", sealed, "wav") == sealed          # already sealed: unchanged


def test_a_file_from_before_the_vault_is_returned_as_it_is():
    assert voice_vault.unseal("v1", b"plain old bytes", "wav") == b"plain old bytes"


def test_the_vault_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("VOICE_VAULT", "off")
    assert voice_vault.seal("v1", b"abc", "wav") == b"abc"
    assert not voice_vault.has_key("v1")


def test_sealed_data_is_bound_to_its_voice_and_kind():
    sealed = voice_vault.seal("v1", b"voice one", "wav")
    voice_vault.seal("v2", b"warm up the second key", "wav")
    for other_voice, other_label in (("v2", "wav"), ("v1", "pt")):
        with pytest.raises(voice_vault.VaultLocked) as e:
            voice_vault.unseal(other_voice, sealed, other_label)
        assert e.value.reason in ("corrupt", "no_key")


def test_a_changed_byte_does_not_open():
    sealed = bytearray(voice_vault.seal("v1", b"voice one" * 10, "wav"))
    sealed[-1] ^= 1
    with pytest.raises(voice_vault.VaultLocked) as e:
        voice_vault.unseal("v1", bytes(sealed), "wav")
    assert e.value.reason == "corrupt"


def test_destroying_the_key_makes_every_copy_unreadable():
    sealed = voice_vault.seal("v1", b"voice one", "wav")
    backup = bytes(sealed)                                  # a copy taken before the withdrawal
    assert voice_vault.has_key("v1") and voice_vault.destroy("v1") is True
    assert not voice_vault.has_key("v1")
    for blob in (sealed, backup):
        with pytest.raises(voice_vault.VaultLocked) as e:
            voice_vault.unseal("v1", blob, "wav")
        assert e.value.reason == "no_key"
    voice_vault.reset()                                      # a fresh process cannot reopen it either
    with pytest.raises(voice_vault.VaultLocked):
        voice_vault.unseal("v1", backup, "wav")
    assert voice_vault.destroy("v1") is False                # nothing left to destroy


def test_a_stale_key_file_does_not_unlock_a_withdrawn_voice():
    sealed = voice_vault.seal("v1", b"voice one", "wav")
    provenance.record_withdrawal("v1", "withdrawn")
    assert voice_vault.has_key("v1")                         # the delete failed to remove it
    with pytest.raises(voice_vault.VaultLocked) as e:
        voice_vault.unseal("v1", sealed, "wav")
    assert e.value.reason == "withdrawn"
    with pytest.raises(voice_vault.VaultLocked):
        voice_vault.seal("v1", b"new data for a withdrawn voice", "wav")


def test_copying_the_data_directory_is_not_enough_without_the_kek(monkeypatch):
    """The wrapped key sits in data/; the key that wraps it does not. Another machine's KEK opens nothing."""
    sealed = voice_vault.seal("v1", b"voice one", "wav")
    monkeypatch.setenv("BONCLAY_VAULT_KEY", base64.b64encode(b"z" * 32).decode("ascii"))
    voice_vault.reset()
    with pytest.raises(voice_vault.VaultLocked) as e:
        voice_vault.unseal("v1", sealed, "wav")
    assert e.value.reason == "no_key"


def test_a_zeroed_key_file_means_locked_not_a_crash():
    sealed = voice_vault.seal("v1", b"voice one", "wav")
    path = Path(voice_vault._key_path("v1"))
    path.write_bytes(b"\0" * path.stat().st_size)
    voice_vault.reset()
    with pytest.raises(voice_vault.VaultLocked):
        voice_vault.unseal("v1", sealed, "wav")


def test_the_key_file_holds_only_a_wrapped_key():
    voice_vault.seal("v1", b"voice one", "wav")
    row = json.loads(Path(voice_vault._key_path("v1")).read_text())
    assert set(row) == {"v", "voice_id", "wrapped", "created_at"} and row["voice_id"] == "v1"
    if os.name != "nt":
        assert oct(Path(voice_vault._key_path("v1")).stat().st_mode & 0o777) == "0o600"
    assert len(base64.b64decode(row["wrapped"])) == 12 + 32 + 16   # nonce + key + tag, never the bare key


def test_the_home_kek_file_is_made_once_with_tight_permissions(monkeypatch, tmp_path):
    monkeypatch.delenv("BONCLAY_VAULT_KEY", raising=False)
    monkeypatch.setenv("VOICE_VAULT_KEK_FILE", str(tmp_path / "home" / "voice_kek"))
    voice_vault.reset()
    first = voice_vault._kek()
    voice_vault.reset()
    assert voice_vault._kek() == first and len(first) == 32
    if os.name != "nt":
        assert oct((tmp_path / "home" / "voice_kek").stat().st_mode & 0o777) == "0o600"


# ---- through the API ---------------------------------------------------------------------------------------------------

def test_a_cloned_voice_is_sealed_on_disk_and_still_speaks(app_client, env, cloned):
    vid = cloned["id"]
    wav_path = Path(env.LOCAL_AUDIO_DIR) / f"{vid}.wav"
    raw = wav_path.read_bytes()
    assert raw.startswith(voice_vault.MAGIC) and b"RIFF" not in raw[:64]
    assert voice_vault.has_key(vid)
    pt = Path(env.LOCAL_AUDIO_DIR) / f"{vid}.pt"
    if pt.exists():
        assert pt.read_bytes().startswith(voice_vault.MAGIC)
    # the app can still read it
    ref = env._voice_file_read(str(wav_path), vid, "wav")
    assert ref[:4] == b"RIFF"


def test_a_plain_file_from_before_the_vault_is_sealed_when_it_is_read(env, tmp_path):
    path = tmp_path / "v-old.wav"
    path.write_bytes(b"RIFF old plain recording")
    assert env._voice_file_read(str(path), "v-old", "wav") == b"RIFF old plain recording"
    assert path.read_bytes().startswith(voice_vault.MAGIC) and voice_vault.has_key("v-old")
    assert env._voice_file_read(str(path), "v-old", "wav") == b"RIFF old plain recording"


def test_withdrawing_a_voice_destroys_its_key_and_a_copy_stays_locked(app_client, env, cloned):
    vid = cloned["id"]
    wav_path = Path(env.LOCAL_AUDIO_DIR) / f"{vid}.wav"
    backup = wav_path.read_bytes()                           # an off-machine copy made before the withdrawal
    assert voice_vault.has_key(vid)

    r = app_client.delete(f"/api/voices/{vid}", headers=LOCAL)
    assert r.status_code == 200
    assert not voice_vault.has_key(vid)
    assert not glob.glob(str(Path(env.LOCAL_AUDIO_DIR) / f"{vid}.*"))
    with pytest.raises(voice_vault.VaultLocked):
        voice_vault.unseal(vid, backup, "wav")


def test_a_withdrawn_voice_cannot_be_read_by_the_app(env, tmp_path):
    path = tmp_path / "v-gone.wav"
    env._voice_file_write(str(path), "v-gone", "wav", b"RIFF recording")
    provenance.record_withdrawal("v-gone", "withdrawn")
    with pytest.raises(env.AppError) as e:
        env._voice_file_read(str(path), "v-gone", "wav")
    assert e.value.code == "CONSENT_WITHDRAWN" and e.value.status == 403
