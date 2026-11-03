# BonClay receipt format (version 1)

Every clip BonClay makes carries an inaudible watermark and has a **signed receipt**. This page says exactly what a
receipt is and how anyone can check it without trusting or even contacting an BonClay server. A reference
implementation is in [`sdk/verify/`](../sdk/verify/) (Python and Node, about 100 lines each).

## What a receipt says

| Field | Meaning |
|---|---|
| `version` | Receipt format version. Always `1` today. A verifier must refuse versions it does not know. |
| `receipt_id`, `kind` | Id, and `generation` (Studio) or `agent_reply` (a voice agent's turn) |
| `generation_id`, `turn_id` | The thing that made the audio |
| `created_at` | ISO 8601 time the clip was made |
| `engine` | The speech engine (`kokoro`, `qwen3_0_6b`, `dhvaani`, ...) |
| `voice_id`, `voice_name` | The voice used. Built-in voices have no `consent_id`. |
| `consent_id`, `consent_verified` | For a cloned voice: the consent record and whether the spoken-consent check passed when the voice was made |
| `text_sha256` | sha256 (hex) of the text that was spoken |
| `audio_sha256` | sha256 (hex) of the exact audio file that was issued |
| `watermark_message` | The 16-bit message embedded in the audio (AudioSeal), as a string |

Unsigned fields that also appear on a published receipt: `signature_ed25519` (base64), `key_id`, `signature` (an HMAC
that only the operator can check, ignore it as a third party), and the live fields below.

## How it is signed

1. Take the **signed fields**: `version, receipt_id, kind, generation_id, turn_id, created_at, engine, voice_id,
   voice_name, consent_id, consent_verified, text_sha256, audio_sha256, watermark_message`. The list is also published
   at `/.well-known/bonclay-receipt-key` as `signed_fields`.
2. Build a JSON object with exactly those keys (a missing one is `null`), serialise with **keys sorted**, separators
   `,` and `:` (no spaces), non-ASCII left as is, encode as **UTF-8**.
3. The signature is **Ed25519** over those bytes, base64 in `signature_ed25519`.
4. The public key is base64 of the 32 raw bytes, from `GET /.well-known/bonclay-receipt-key`:
   `{"alg":"Ed25519","key_id":"...","public_key":"...","signed_fields":[...],"receipt_version":1,"keys":[...]}`.
   `key_id` is the first 16 hex characters of the sha256 of the raw public key.

To check a clip: verify the signature, then check that the sha256 of the file equals `audio_sha256`. A re-encoded copy
(MP3, resampled) no longer matches the hash; the server can still find it by its watermark (`POST /api/verify`).

## Consent and withdrawal

A receipt is never edited. When a voice owner withdraws consent (or the voice is deleted when its retention period
ends), a row is added to the public **withdrawal log**, `GET /api/withdrawals`:

```
{"voice_id", "withdrawn_at", "reason": "withdrawn" | "retention_expired", "prev", "row_hash", "key_id", "sig"}
```

* `row_hash` = sha256 of the JSON `{voice_id, withdrawn_at, reason, prev}` (sorted keys, no spaces).
* `sig` = Ed25519 signature (same key) over the ASCII bytes of the hex `row_hash`.
* `prev` = the previous signed row's `row_hash` (the first one uses 64 zeros). Deleting, editing or reordering a row
  breaks the chain at that row. Rows written before the log was signed have no `row_hash`; they are listed as legacy.

The server adds live fields when it shows a receipt: `voice_withdrawn_at`, `voice_removed_reason`,
`voice_withdrawal_signed`, `voice_withdrawal_row_hash` and `voice_consent_status`
(`preset_voice`, `active`, `unverified`, `withdrawn`, `retention_ended`). A verifier that wants the same answer offline
looks the receipt's `voice_id` up in a log snapshot it fetched.

## What this does and does not prove

* **Proves:** the holder of the key issued this receipt, none of the signed fields were changed, and these exact bytes
  are the audio it describes.
* **Does not prove:** who holds the key (get the public key over a channel you trust, and note its `key_id`); that a
  re-encoded copy came from BonClay (that needs the watermark detector, which needs the server's models); that a voice
  has *not* been withdrawn since the log snapshot you hold.
* The watermark carries only 16 bits, so it identifies a receipt by lookup, and collisions are possible; the hash and
  signature are the strong evidence.
* There is one signing key today. `keys` in the key endpoint is a list so future rotation can publish old keys with
  `"status": "retired"`; verifiers should accept any listed key whose `key_id` matches the receipt's.

## Test vectors

`backend/tests/test_provenance.py` generates real receipts and checks them with both SDKs
(`test_the_sdk_verifies_a_receipt_made_by_the_backend_without_the_server`,
`test_the_javascript_sdk_agrees_with_python`).
