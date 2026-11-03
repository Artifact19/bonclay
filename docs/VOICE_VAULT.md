# Voice vault

A cloned voice is the most sensitive thing BonClay stores. The vault makes its files ciphertext on disk, and ties the
key to the owner's consent: withdraw consent and the key is destroyed, so every copy of the files (backups included)
becomes unreadable.

## What is sealed

For each cloned voice: the reference recording (`{voice}.wav`), the voice prompt (`{voice}.pt`) and the prompts built
from them (`.chatterbox.pt`, `.dhvaani.wav`). AES-256-GCM, one random key per voice, authenticated with the voice id and
the file kind (a sealed file cannot be passed off as another voice's).

## Where the keys are

- Per-voice key: `backend/data/voice_keys/{voice_id}.json`, stored **wrapped** (encrypted) only.
- The key that wraps it (the KEK) is **not** in `backend/data`. Order: `BONCLAY_VAULT_KEY` (base64 of 32 bytes), then
  the macOS Keychain (service `bonclay-voice-vault`, created on first use), then `~/.bonclay/voice_kek` (0600).
  Copying `backend/data` therefore yields ciphertext and wrapped keys that open nothing.

## Withdrawal

Deleting a voice (or retention expiry) first destroys its wrapped key (overwritten, then removed), then deletes the
files. A voice on the withdrawal log is also refused when read, even if a stale key file survived. Reading a withdrawn
voice answers `403 CONSENT_WITHDRAWN`.

## Existing voices

A voice made before the vault is plain on disk. The first time the app reads one it is sealed in place.

## Settings

| Setting | Effect |
|---|---|
| `VOICE_VAULT=off` | Do not seal; plain files as before |
| `BONCLAY_VAULT_KEY` | Supply the KEK yourself (base64, 32 bytes) |
| `VOICE_VAULT_KEYCHAIN=off` | Skip the macOS Keychain, use the home file |
| `VOICE_VAULT_KEK_FILE` | Where the home-file KEK lives |
| `VOICE_VAULT_DIR` | Where the wrapped per-voice keys live |

Back up the KEK (Keychain item or `~/.bonclay/voice_kek`) with your voices: without it, sealed voices cannot be opened.

## Limits (honest)

- Once a voice is opened to generate speech, its prompt is plaintext in this process's memory. Code running as the
  same user, or a malicious operator, still wins. This protects copied files and makes withdrawal real; it does not
  stop someone who holds the machine and the model.
- Cloud (Supabase) objects are **not** sealed yet: that needs a hosted key service.
- The consent recording and the DhVaani transcript record are not sealed (they are deleted on withdrawal).
- Tested on the file KEK and an injected key. The macOS Keychain path has not been exercised on a real login keychain.
- Moving to another machine needs the KEK moved with the voices.

See `docs/research/B1-consent-bound-voice.md` for the design and what it can and cannot claim.
