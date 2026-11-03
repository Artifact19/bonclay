# Revocable provenance: threat model (22 October 2026)

One page on what an BonClay receipt proves, what withdrawing consent changes, and what it cannot do. It is the
basis for the `/verify` page, the paper claim and the sales line. Not legal advice.

## The claim

Every clip BonClay makes carries an AudioSeal watermark (16-bit payload, `services/provenance.py`) and a receipt
signed with HMAC and Ed25519 that names the engine, the voice and the consent record behind it. Verification is
**revocable**: when the voice owner withdraws consent, the receipt is not changed (it is an audit trail), but the
withdrawal is appended to a hash-chained, Ed25519-signed log, and `/verify` reads the clip differently from then on.
No competitor in the 22 Oct review links withdrawal to verification.

## What a verified clip reads as (the states)

`voice_consent_status` on every public receipt (`provenance.consent_status`):

| State | Meaning | How `/verify` shows it |
| --- | --- | --- |
| `active` | A cloned voice; consent was verified when the clip was made and has not been withdrawn | "Consent verified" |
| `withdrawn` | The voice owner has since withdrawn consent (`withdrawn_at` on the receipt) | "Consent verified when made, now withdrawn"; "Voice withdrawn" |
| `retention_ended` | The voice was deleted when its retention period ended | as withdrawn, with the reason |
| `unverified` | A cloned voice whose consent record exists but was not verified | "Consent recorded, not verified" |
| `preset_voice` | A built-in voice: nobody's voice was cloned, so no consent applies | no consent row |

Clip verdicts: `made_with_bonclay` (watermark or exact hash leads to a receipt whose signature checks), `watermark_only`
(AudioSeal detected, no receipt found: AI-made, origin unknown), `not_detected`.

## Assets and adversaries

| Asset | Who wants to break it | How |
| --- | --- | --- |
| "This clip is AI-made" | A user passing a clip off as human | Remove the watermark: speed or pitch change, EnCodec re-encoding (the repo's own bench, TRIAL_REPORT §4) |
| "This clip was consented" | Someone cloning a voice without consent | Pass the consent gate with a zero-shot clone reading the code (plan Part 3.3) |
| "This clip verifies as withdrawn" | A user re-using a clip after withdrawal | Keep the clip; the receipt still reads as made-while-consented, and the withdrawal flag is what changes |
| The withdrawal log | An operator hiding a withdrawal | Delete or edit a row; the hash chain and signatures make this detectable, not impossible |
| The 16-bit payload | Chance | 65,536 ids: by about 300 clips a collision is ~50% likely (birthday bound); the receipt store disambiguates with the audio hash and `candidates` on `/verify` |
| The signing key | Theft, or rotation without `kid` | One key today; rotation is a week-7 item (plan 3.3) |

## What revocation does and does not do

Does: delete the voice's files (sealed by the vault; the key is destroyed), its embedding and prompt caches; append a
signed row to the public log; make every existing receipt for that voice read as withdrawn on `/verify`, within one
request of the write (no caching layer in between).

Does not: recall clips already copied elsewhere; remove the watermark from them (so they still verify as AI-made,
which is the point); change the receipt's signed fields; stop an operator with the machine and the model from
generating more (see `docs/VOICE_VAULT.md`).

## Evaluation the claim still needs

1. Revocation latency: withdraw, then `GET /api/receipts/{id}`; target under 60 s, measured today as one request.
2. Payload collisions at 300, 3,000 and 30,000 receipts, and how often `candidates` is non-empty.
3. Watermark detection versus receipt recovery after ±5% speed change (the paper reports detection surviving 1.25×;
   the repo loses the payload at ±5%): report both numbers separately.
4. C2PA manifest beside the watermark (`bench/c2pa_spike.py`) and whether a reader shows the withdrawal.
5. The consent gate's false accept against BonClay's own engines (plan 3.3), since "consented" is only as strong as the gate.
