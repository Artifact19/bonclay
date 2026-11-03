"""Voice vault: a cloned voice's files are ciphertext on disk, and the key lives only as long as the owner's consent.

What is protected: the reference recording (`{voice}.wav`), the voice prompt (`{voice}.pt`) and the prompts built from
them (`.chatterbox.pt`, `.dhvaani.wav`). Each is sealed with AES-256-GCM under a key of its own voice (the DEK, a random
256-bit key per voice). The DEK is stored only wrapped (AES-GCM under the KEK) in `data/voice_keys/{voice_id}.json`.
The KEK comes from, in order: BONCLAY_VAULT_KEY (base64 of 32 bytes), the macOS Keychain, a 0600 file in the user's home
directory (`~/.bonclay/voice_kek`, or VOICE_VAULT_KEK_FILE). It never sits beside the voice files, so copying
`backend/data` (a backup, a bucket, a stolen laptop folder) yields ciphertext.

Withdrawal is crypto-shredding: `destroy(voice_id)` deletes the wrapped DEK, and every copy of the files, backups
included, becomes unreadable. `unseal` also refuses once the voice is on the withdrawal log, even if a stale key file
survived a failed delete.

What this does NOT do (see docs/research/B1-consent-bound-voice.md): once a voice is unsealed to generate, the prompt is
plaintext in this process's memory, so code running as the same user, or a malicious operator, still wins. It makes
copied files useless and withdrawal real; it does not stop someone who holds the machine and the model. Cloud
(Supabase) objects are not sealed yet: the key service would have to be hosted.

Files written before the vault existed are plain; `unseal` returns them as they are (the backend re-seals them when it
reads them). VOICE_VAULT=off turns the whole thing off."""
from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Dict, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

log = logging.getLogger("bonclay.vault")

MAGIC = b"AURV1"
NONCE_BYTES = 12
KEYCHAIN_SERVICE = "bonclay-voice-vault"
KEYCHAIN_ACCOUNT = "kek"

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_lock = threading.RLock()
_kek_cache: Optional[bytes] = None
_dek_cache: Dict[str, bytes] = {}


class VaultLocked(Exception):
    """A sealed file cannot be opened: its voice was withdrawn, or its key is gone. `reason` says which."""

    def __init__(self, reason: str, voice_id: str = ""):
        super().__init__(f"voice {voice_id or '?'} is locked: {reason}")
        self.reason = reason
        self.voice_id = voice_id


def enabled() -> bool:
    return os.getenv("VOICE_VAULT", "on").strip().lower() not in ("off", "0", "false", "no")


def is_sealed(data: bytes) -> bool:
    return bool(data) and data.startswith(MAGIC)


def reset() -> None:
    """Forget cached keys (tests, and after the key directory moves)."""
    global _kek_cache
    with _lock:
        _kek_cache = None
        _dek_cache.clear()


def _keys_dir() -> str:
    return os.getenv("VOICE_VAULT_DIR") or os.path.join(_DATA_DIR, "voice_keys")


def _key_path(voice_id: str) -> str:
    return os.path.join(_keys_dir(), f"{voice_id}.json")


# ---------------------------------------------------------------------------------------------------------------
# The key-encrypting key
# ---------------------------------------------------------------------------------------------------------------

def _kek_file() -> str:
    return os.getenv("VOICE_VAULT_KEK_FILE") or os.path.join(os.path.expanduser("~"), ".bonclay", "voice_kek")


def _keychain() -> Optional[bytes]:
    """The KEK from the macOS Keychain (created on first use), or None when the Keychain is not there or not usable.
    `security add-generic-password -w <value>` puts the value on the command line for a moment, visible to this user's
    other processes; a process running as this user can reach the Keychain item anyway, so the exposure adds nothing
    against the threat this vault covers (copied files)."""
    if sys.platform != "darwin" or not shutil.which("security") or os.getenv("VOICE_VAULT_KEYCHAIN", "on") == "off":
        return None
    try:
        got = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT, "-w"],
                             capture_output=True, text=True, timeout=10)
        if got.returncode == 0 and got.stdout.strip():
            raw = base64.b64decode(got.stdout.strip())
            if len(raw) == 32:
                return raw
        fresh = secrets.token_bytes(32)
        made = subprocess.run(["security", "add-generic-password", "-U", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT,
                               "-w", base64.b64encode(fresh).decode("ascii")], capture_output=True, text=True, timeout=10)
        return fresh if made.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        log.info("macOS Keychain not usable for the voice vault: %s", e)
        return None


def _dpapi_protect(raw: bytes) -> bytes:
    import ctypes
    import ctypes.wintypes
    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", ctypes.wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
    in_blob = DATA_BLOB(len(raw), ctypes.cast(ctypes.create_string_buffer(raw), ctypes.POINTER(ctypes.c_char)))
    out_blob = DATA_BLOB()
    if ctypes.windll.crypt32.CryptProtectData(ctypes.byref(in_blob), "bonclay_voice_kek", None, None, None, 0, ctypes.byref(out_blob)):
        res = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)
        return res
    raise OSError("DPAPI encryption failed")


def _dpapi_unprotect(cipher: bytes) -> bytes:
    import ctypes
    import ctypes.wintypes
    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", ctypes.wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
    in_blob = DATA_BLOB(len(cipher), ctypes.cast(ctypes.create_string_buffer(cipher), ctypes.POINTER(ctypes.c_char)))
    out_blob = DATA_BLOB()
    if ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)):
        res = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)
        return res
    raise OSError("DPAPI decryption failed")


def _home_file() -> bytes:
    path = _kek_file()
    try:
        with open(path, "r", encoding="utf-8") as f:
            stored = base64.b64decode(f.read().strip())
        if sys.platform == "win32":
            try:
                dec = _dpapi_unprotect(stored)
                if len(dec) == 32:
                    return dec
            except Exception:
                if len(stored) == 32:
                    return stored
        elif len(stored) == 32:
            return stored
    except (OSError, ValueError):
        pass
    raw = secrets.token_bytes(32)
    to_write = raw
    if sys.platform == "win32":
        try:
            to_write = _dpapi_protect(raw)
        except Exception as e:
            log.warning("DPAPI encryption unavailable for voice vault: %s", e)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(base64.b64encode(to_write).decode("ascii"))
    except FileExistsError:  # another worker won the race; use its key
        with open(path, "r", encoding="utf-8") as f:
            stored = base64.b64decode(f.read().strip())
            if sys.platform == "win32":
                try:
                    dec = _dpapi_unprotect(stored)
                    if len(dec) == 32:
                        return dec
                except Exception:
                    if len(stored) == 32:
                        return stored
            return stored
    return raw


def _kek() -> bytes:
    global _kek_cache
    with _lock:
        if _kek_cache is not None:
            return _kek_cache
        env = os.getenv("BONCLAY_VAULT_KEY", "").strip()
        if env:
            raw = base64.b64decode(env)
            if len(raw) != 32:
                raise ValueError("BONCLAY_VAULT_KEY must be the base64 of 32 bytes")
        else:
            raw = _keychain() or _home_file()
        _kek_cache = raw
        return raw


# ---------------------------------------------------------------------------------------------------------------
# One data key per voice
# ---------------------------------------------------------------------------------------------------------------

def _wrap(voice_id: str, dek: bytes) -> str:
    nonce = secrets.token_bytes(NONCE_BYTES)
    return base64.b64encode(nonce + AESGCM(_kek()).encrypt(nonce, dek, voice_id.encode("utf-8"))).decode("ascii")


def _unwrap(voice_id: str, wrapped: str) -> bytes:
    blob = base64.b64decode(wrapped)
    return AESGCM(_kek()).decrypt(blob[:NONCE_BYTES], blob[NONCE_BYTES:], voice_id.encode("utf-8"))


def has_key(voice_id: str) -> bool:
    return os.path.isfile(_key_path(voice_id))


def _dek(voice_id: str, create: bool) -> Optional[bytes]:
    with _lock:
        cached = _dek_cache.get(voice_id)
        if cached is not None and has_key(voice_id):  # a deleted key file locks the voice even if a copy is cached
            return cached
        _dek_cache.pop(voice_id, None)
        path = _key_path(voice_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                dek = _unwrap(voice_id, json.load(f)["wrapped"])
            _dek_cache[voice_id] = dek
            return dek
        except FileNotFoundError:
            if not create:
                return None
        except (ValueError, KeyError, TypeError, InvalidTag):  # a zeroed, half-deleted or foreign key file: locked
            return None
        dek = secrets.token_bytes(32)
        os.makedirs(_keys_dir(), exist_ok=True)
        row = {"v": 1, "voice_id": voice_id, "wrapped": _wrap(voice_id, dek),
               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(row, f)
        _dek_cache[voice_id] = dek
        return dek


def _aad(voice_id: str, label: str) -> bytes:
    return f"{voice_id}|{label}".encode("utf-8")


def seal(voice_id: str, data: bytes, label: str) -> bytes:
    """`data` as ciphertext: MAGIC + nonce + AES-256-GCM. Data that is already sealed, or any data when the vault is
    off, comes back unchanged. `label` names the file kind and is authenticated, so a sealed file cannot be passed off
    as another voice's or another kind."""
    if not enabled() or is_sealed(data):
        return data
    if _withdrawn(voice_id):
        raise VaultLocked("withdrawn", voice_id)
    dek = _dek(voice_id, create=True)
    if dek is None:
        raise VaultLocked("no_key", voice_id)
    nonce = secrets.token_bytes(NONCE_BYTES)
    return MAGIC + nonce + AESGCM(dek).encrypt(nonce, data, _aad(voice_id, label))


def _withdrawn(voice_id: str) -> bool:
    try:
        from services import provenance  # noqa: PLC0415
        return provenance.withdrawal_for(voice_id) is not None
    except Exception:  # noqa: BLE001 - a broken log lookup must not unlock or lock anything by itself
        return False


def unseal(voice_id: str, data: bytes, label: str) -> bytes:
    """The plaintext of a sealed file; a file that is not sealed (made before the vault) is returned as it is.
    Raises VaultLocked("withdrawn") when the voice is on the withdrawal log, VaultLocked("no_key") when its key is
    gone, and VaultLocked("corrupt") when the data does not authenticate (wrong voice, wrong kind, or changed)."""
    if not is_sealed(data):
        return data
    if _withdrawn(voice_id):
        raise VaultLocked("withdrawn", voice_id)
    dek = _dek(voice_id, create=False)
    if dek is None:
        raise VaultLocked("no_key", voice_id)
    nonce, body = data[len(MAGIC):len(MAGIC) + NONCE_BYTES], data[len(MAGIC) + NONCE_BYTES:]
    try:
        return AESGCM(dek).decrypt(nonce, body, _aad(voice_id, label))
    except Exception as e:  # noqa: BLE001 - InvalidTag
        raise VaultLocked("corrupt", voice_id) from e


def destroy(voice_id: str) -> bool:
    """Crypto-shred a voice: overwrite and delete its wrapped key. Every sealed copy of its files is unreadable from
    here on. True when a key existed. Never raises."""
    with _lock:
        _dek_cache.pop(voice_id, None)
        path = _key_path(voice_id)
        try:
            if not os.path.isfile(path):
                return False
            size = os.path.getsize(path)
            with open(path, "r+b") as f:  # overwrite first: a deleted file's blocks may linger
                f.write(b"\0" * size)
                f.flush()
                os.fsync(f.fileno())
            os.remove(path)
            return True
        except OSError as e:
            log.warning("could not destroy the vault key of voice %s: %s", voice_id, e)
            return False
