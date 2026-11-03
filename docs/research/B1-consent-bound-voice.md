# B1. Consent cryptographically bound to the cloned voice (deep re-research)

Date of research: 2026-10-17. Method: fetched primary pages with WebFetch and read the BonClay backend. Tags: [V] = read on the primary page this session; [S] = secondary source only (law-firm blog, news, search snippet); [U] = unverified or from memory. Several pages returned 403 or truncated text, listed in section 9.

## 0. Bottom line (verdict)

1. No design can bind consent to a voice against someone who holds the files and the model. A voice prompt or x-vector is just a tensor; any copy runs on any Qwen3-TTS/Chatterbox install. What you can bind is (a) the keys, on a machine the attacker does not control, and (b) the outputs, via watermark + signed receipt + revocation list. Say this on the page; it is the honest claim and is still publishable.
2. Recommended build, in order: Design 1 (consent-gated key release, local, cheap), Design 2 (consent ID in the watermark/receipt chain and a public, signed revocation list), Design 3 (W3C VC consent credential with did:web or did:key, status via Bitstring Status List). Design 3 is only worth doing as an export format; it adds no security beyond Design 2 unless a third party must verify offline.
3. Genuinely novel: the combination (voice-prompt encryption keyed to a consent state + per-clip consent ID recoverable from the watermark + third-party-verifiable VC), applied to an open local-first studio with Indic speech. Each piece alone has prior art. I found no prior art for the combination, but my search is not exhaustive [U].
4. Biggest practical gap is not crypto: your receipt is symmetric HMAC (`provenance.py`), so a third party cannot verify it without asking your server, and the watermark payload carries 16 bits of sha256(generation_id), not a consent ID. Fixing those two is the highest-value work.

## 1. What the backend does today (read from code)

- `routers/consent.py`: single-use, 15-minute, in-memory challenge sentence with name, date, code word and number, bound to the requesting user. Defeats replay of an old consent recording. Does not defeat a live voice clone reading the sentence (see section 5).
- `backend.py` `_verify_speaker` (line ~2375) compares consent recording and sample (ECAPA); `_verify_consent` checks the transcript against the stored sentence.
- `services/consent_record.py`: itemised JSON record (consent_id, challenge, scores, notice_version, retention). It is a plain dict stored with the voice row; not signed and not hash-chained. Its own text says Qwen3-TTS prompt is stored as `.pt` embedding (`backend.py` `load_voice_embedding`, line ~967) plus reference audio.
- `services/provenance.py`:
  - Receipt signed with HMAC-SHA256 over `SIGNED_FIELDS` (includes `voice_id`, `consent_id`, `consent_verified`, `audio_sha256`, `watermark_message`). Secret from `BONCLAY_RECEIPT_SECRET` or `data/receipt_secret` (0600). Symmetric: only the operator can verify.
  - Watermark payload `message_for(generation_id)` = first 16 bits of sha256(generation id). AudioSeal 16-bit. Consent is not in the payload; it is reachable only by receipt lookup.
  - `_withdrawals` log (`data/voice_withdrawals.jsonl`): `{voice_id, withdrawn_at, reason}`; `/verify` joins it onto receipts. The log is unsigned ("looked up from the withdrawal log, not signed").
  - Withdrawal deletes sample, embedding and consent recording (`backend.py` ~1776-1867). Past clips stay.
- `bench/c2pa_spike.py`: an earlier spike that signs WAV and MP3 with a c2pa-python manifest including a custom receipt assertion, using a self-signed test CA (untrusted by any real trust list, per its own comment).
- Gap: generation does not check that the consent is still valid at generation time beyond the voice row existing. Copying `data/` leaves embedding and reference audio in the clear on disk (no encryption at rest in what I read) [U: I did not audit every storage path; Supabase mode stores `.pt` in a bucket].

## 2. Prior art (what exists)

| Area | Finding | Status |
|---|---|---|
| OpenAI Voice Engine (Mar 2024) | Partner contracts require "explicit and informed consent of the original speaker"; no self-service voice creation for individuals; watermarking and proactive monitoring; OpenAI argues for voice authentication of the original speaker and a no-go voice list. Primary page returned 403; content from news/search summaries. It is contractual and operator-side, not cryptographic. | [S] |
| ElevenLabs PVC | Primary help page: "All Professional Voice Clones require a verification process to confirm that the voice belongs to you"; "For now, we only allow you to clone your own voice"; "Even with their consent, you cannot clone someone else's voice"; verification is reading lines, ideally on the same equipment. Docs page: "Voice-captcha technology is used to verify that Professional Voice Clones are created from your own voice samples." Instant cloning: no consent mechanism stated. Enrolment-time check only; the clone is not cryptographically gated afterwards. | [V] |
| Resemble | Docs front page now says voice products are no longer offered commercially (Chatterbox stays open under "Voice AI Research"); the voice-archive page lists no consent text. I could not read its consent-statement mechanism. | [V] for the notice, [U] for the mechanism |
| C2PA 2.2 spec | Entity-specific assertion labels are allowed ("Entity-specific namespaces shall begin with the Internet domain name for the entity"); `trainedAlgorithmicData` digital source type exists; the "Training & Data Mining" assertion was removed in 2.0; the word "consent" did not appear in the text I could read (about 100k of 579k characters read twice, so incomplete). ID3 and RIFF are normative references (audio containers plausible); I could not read the supported-formats appendix. So: there is no standard voice-consent assertion; a custom `org.bonclay.voice-consent` assertion is the path. | [V] partly, [U] for the unread parts |
| AntiFake (CCS 2023) | Adversarial perturbation of the reference audio to block synthesis; abstract claims over 95% protection rate including unseen models, five synthesizers, 24-person usability study; covers short clips only. Via university summaries, not the paper. | [S] |
| AttackVC, VoiceBlock, SafeSpeech, "unlearnable" embeddings | Not fetched. | [U] |
| Voice NFTs / tokens | Not researched to a primary source. | [U] |

Takeaway: industry consent is enrolment-time verification plus contracts. Nobody I found gates generation cryptographically on a revocable consent state, or embeds a consent handle in per-clip watermarks. That is the white space, with the caveat in section 0.

## 3. W3C VC / DIDs for consent proofs, without a registry

Primary facts:
- VC Data Model 2.0: issuer creates and transmits to holder; verifier receives credentials. Verification "is an authentic and current statement of the issuer or presenter", checks conformance, securing mechanism and, if present, status; it does not judge whether the claims are true. A verifiable data registry is a role "a system might perform", not mandatory. "Decentralized Identifiers are optional." [V] (https://www.w3.org/TR/vc-data-model-2.0/)
- DID Core: "DIDs have been designed so that they may be decoupled from centralized registries"; controller proves control "without requiring permission from any other party". [V] (https://www.w3.org/TR/did-core/)
- did:web resolves to `https://<domain>/.well-known/did.json`; the spec says it "does not specify any authentication or authorization mechanism for writing to, removing or creating the DID Document" and warns about DNS interception (recommends DNSSEC/DoH); with a path, signatures only prove control of that path's key, not domain-operator key custody. [V] (https://w3c-ccg.github.io/did-method-web/)
- Bitstring Status List: statusPurpose `revocation` (irreversible) and `suspension` (reversible); minimum list length 131,072 entries (16 KB uncompressed) for herd privacy; issuers SHOULD publish cacheable lists, verifiers SHOULD cache and use proxies. [V] (https://www.w3.org/TR/vc-bitstring-status-list/)
- did:key: appears only in examples in what I read. Its property (key encoded in the DID, no network) is from memory [U].

Practicality without a registry:
- A registry is not needed. Use did:web on the studio's domain (the `/verify` host) for the issuer, or did:key for an offline studio. Subject: the voice owner, identified by a did:key generated on their device (the owner holds a key; the consent VC is the holder's claim, countersigned by the studio).
- Credential content (data minimisation): `voiceCommitment` = SHA-256 of the embedding (not the voice), `consentSentenceHash`, `challengeId`, `scope`/purpose, `notice_version`, `validFrom`, `validUntil` (retention), `credentialStatus` (Bitstring list URL + index). No audio, no name unless the owner wants it.
- Verification by a third party: check the Data Integrity or JOSE signature against the did:web document, fetch the status list, compare the VC's voice commitment with the receipt on a clip. No access to the voice needed.
- Limits: a VC proves the issuer said consent happened; it does not prove the speaker was the owner. did:web trust equals domain trust, which for a student laptop studio is weak; a student team running `localhost` cannot publish a did:web document, so a hosted `/verify` domain is a prerequisite. Status lists with a few hundred voices give little herd privacy (spec minimum list is 131,072 entries; pad it). Selective-disclosure / ZK (BBS, SD-JWT) is feasible but heavy; I recommend not doing ZK in v1 [U on library maturity].
- India angle: a consent VC is not a DPDP "consent manager" artefact; see section 7.

## 4. Candidate designs, ranked

### Design 1 (rank 1). Consent-gated key release for the voice prompt (local-first)

Goal: copying `data/` or the Supabase bucket yields ciphertext.

- Data flow: at clone time generate a 256-bit voice-data key (DEK). Encrypt the `.pt` embedding and the reference WAV with AES-256-GCM (AAD = voice_id || consent_id || notice_version). Wrap the DEK with a key-encryption key (KEK) held in the OS keystore: macOS Keychain, optionally a Secure Enclave P-256 key used for key agreement (ECIES-style: ephemeral ECDH with the enclave key, HKDF, AES-GCM wrap); Windows DPAPI/CNG (TPM-backed where available) [U details]. Store the wrapped DEK and the consent state in a small `consent_state` record `{voice_id, status: active|suspended|revoked, seq}`.
- Release rule: the generator process calls `unwrap(voice_id)`. It succeeds only if `consent_state.status == active`, retention not expired, and the signed state (below) verifies. On revoke: delete the wrapped DEK (crypto-shredding) and write `status: revoked` to a signed, append-only log. Ciphertext becomes unrecoverable, a stronger version of today's file deletion because it covers backups and bucket copies.
- Primitives: AES-256-GCM, HKDF-SHA256, P-256 ECDH (Secure Enclave supports P-256 only [U from memory; the Apple page fetched gave only the UID/key-hierarchy text, "keys rooted in entropy stored in the Secure Enclave nonvolatile storage" and "UID ... not available through debugging interfaces" [V] (https://support.apple.com/guide/security/secure-enclave-sec59b0b31ff/web)]).
- Effort: about 3 to 5 days for a Python `cryptography`+`keyring` implementation on macOS and Windows, plus tests; Secure Enclave path needs a small Swift helper (+3 to 5 days) [estimate, U].
- Real limits (honest):
  - Once unwrapped, the embedding is plaintext in process memory; any process that can call the unwrap API or dump memory wins. The OS keychain protects against file theft, not against code running as the same user. Keychain items for an unsigned Python process are reachable by any same-user process that can pass the ACL prompt [U].
  - The Secure Enclave cannot hold the TTS model or run it; it can only guard the key, so the plaintext still exists during generation.
  - A malicious operator who controls the machine can keep the prompt after one legitimate generation. Not defeated by any local design.
  - Windows RTX laptops: TPM-backed keys vary by device; fall back to DPAPI [U].
  - Backups and device migration: keys bound to a device do not move, so "export voice" needs an explicit re-wrap flow.
  - Revocation relies on the local state being honest; offline machines cannot learn of a remote revocation. A time-boxed lease (see Design 2) bounds this.

### Design 2 (rank 2). Consent handle in every clip plus signed revocation list

Goal: any clip, anywhere, resolves to a consent state a third party can check.

- Change A, receipt crypto: switch receipts from HMAC to an Ed25519 signature (public key published at `/.well-known/` and in the did:web document); keep HMAC for back-compat. This alone lets a third party verify a receipt offline. Today they cannot (symmetric secret in `provenance._secret`).
- Change B, consent in the receipt: already present (`consent_id`, `consent_verified` are signed). Add `consent_record_sha256` (hash of the canonical consent record) and `voice_commitment` (SHA-256 of embedding), so the receipt binds to what the owner actually agreed to.
- Change C, watermark: the 16-bit payload cannot hold a consent ID (it is sha256(generation_id)[:2], collisions possible). Options: (i) keep 16 bits as a lookup hint and rely on the receipt (current), or (ii) move to a payload of a consent-bucket hash and rely on the receipt chain. A 16-bit payload carries about 65k distinct values, so it works as an index into a signed Merkle log, not as an ID. Do not claim "the consent ID is in the watermark" unless you move to a larger payload model (AudioSeal 16-bit is what you ship; larger-capacity models are in B6 scope).
- Change D, revocation: publish a signed, append-only revocation log (Ed25519, hash chain or Merkle tree, signed tree head). `/verify` shows "made with a voice whose owner withdrew consent on <date>" backed by a signature, not an unsigned jsonl line. Add a lease: a voice may be generated for at most N hours without refreshing a signed "active" status (bounds the offline-revocation gap of Design 1).
- Effort: about 4 to 6 days (receipt signature, log, `/verify` changes, tests). The Merkle log is optional; a hash chain plus a daily signed head is enough.
- Limits: it flags clips, it does not stop someone running a stolen prompt elsewhere; stripping or attacking the watermark (speed/pitch/neural codec per your own B6 prompt) removes the handle, though the receipt still matches by `audio_sha256` only for byte-identical files.

### Design 3 (rank 3). Verifiable-credential consent proof

Goal: a third party (a clinic, a platform, a regulator) checks "this voice has an unrevoked consent" without seeing the voice.

- Architecture: studio issues a VC (VC 2.0, Data Integrity `eddsa-jcs-2022` or JOSE/SD-JWT [U on suite names]) with fields from section 3; issuer = `did:web:<verify-domain>`; status via Bitstring Status List on the same domain; the clip's receipt references the VC id.
- Effort: about 5 to 8 days including a status-list generator, a verifier endpoint and a downloadable VC from `/api/voices/{id}/consent-record` [estimate, U].
- Value: interoperability and a clean story for regulators and platforms; no added security over Design 2 because trust still ends at the issuer's domain.
- Optional: wrap the same data in a C2PA manifest with a custom assertion `org.<domain>.voice-consent` (section 2) so that Content Credentials-aware tools show it. Your own spike already signs WAV/MP3 with a custom assertion; the open question is trust-list acceptance (a self-signed chain is untrusted by design).

## 5. Liveness and anti-spoofing for the consent recording

Facts:
- ASVspoof 5 database: about 2,000 speakers (about 100 in earlier editions), 32 attack algorithms including TTS, voice conversion and adversarial attacks, seven speaker-disjoint partitions, crowdsourced diverse acoustic conditions. [V] (https://arxiv.org/abs/2502.08857)
- Challenge results (secondary summary of arXiv 2601.03944): closed condition top five EER below 15%, baselines above 29% EER; open condition better, with most strong systems using SSL features such as wav2vec 2.0; one team (XMUspeech) reports EER 9.36% open vs 20.45% closed on the evaluation set; performance degrades under adversarial attacks and neural codecs; calibration was poor (actDCF near 1.0). [S] (https://arxiv.org/html/2601.03944v1, https://arxiv.org/pdf/2509.18102); I did not open the results tables.
- Indian-language evidence is thin and mixed: a Bengali BanglaFake study reports zero-shot Wav2Vec2-XLSR-53 EER 46.2% (near chance) and 24.35% EER for the best fine-tuned model [S, search snippet]. IndicFake (4.2M samples, 7,350 h, English + 17 Indian languages), IndicSynth (4,000+ h, 12 languages) and SEA-Spoof (Tamil, Hindi among six languages) exist as datasets; I did not read their per-language EER [U].
- Interpretation: a general SSL detector trained on English is roughly chance on Indian-language speech and modern zero-shot TTS; even in-distribution EERs of about 9 to 20% are too high to be a gate. Use detectors as a risk signal, never as the sole control.

Design for the consent recording (better than a detector):
1. Challenge-response with unpredictable content: you already have name, date, code word and number. Add a second, per-session random phrase in the speaker's language and a timing bound (record must start within T seconds of challenge issue, total duration within a window). A real-time voice clone must run ASR to LLM to TTS inside the window; latency is the discriminator. Measured numbers for modern streaming cloning latency on Indian languages: none found [U].
2. Challenge content that is hard for TTS: spoken digits mixed with a code-mixed Hinglish word, and one instruction such as "say the second word again slower". Raises attacker cost; not a proof.
3. Acoustic consistency check: compare room response / noise floor / bandwidth between the enrolment sample and the consent clip, and require the same device where possible (ElevenLabs advises the same equipment for verification [V]). Detects a spliced or separately generated clip; fooled by an attacker who clones both in one session. No measured accuracy found [U].
4. SSL detector score as a soft flag shown to the operator ("review before activating"), with the caveats above.
5. Human-in-the-loop for high-risk voices (public figures; no-go list as OpenAI proposes [S]).
6. Bind the owner to a device key: the consent sentence is also signed by an owner-held key (did:key or WebAuthn passkey). This does not prove voice ownership but proves the same enrolled device approved it.

## 6. Threat-model table

| Threat | D1 key release | D2 receipt + revocation log | D3 VC | Residual |
|---|---|---|---|---|
| Stolen `data/` or bucket files | Defeats (ciphertext only, key in OS keystore) | No effect | No effect | Plaintext during generation; same-user malware |
| Malicious operator (holds machine, keys, secret) | Does not defeat | Does not defeat (they hold the signing key); partially mitigated by publishing the public key and a transparency log with external witnesses [U] | Does not defeat | Inherent to local-first; say so |
| Replayed old consent recording | Gate is enrolment; fresh single-use challenge already defeats | n/a | VC carries challengeId | Replay of a live session is not covered |
| Clone reads the consent code live | Not defeated by crypto | Not defeated | Not defeated | Timing window, device binding, detector flag, human review (section 5) |
| Consent withdrawn but model copied | Crypto-shred defeats the copies you control; copies made while unwrapped survive | Flags new clips only if they carry the watermark and are checked; the copy can drop the watermark | Status list shows revoked | Cannot recall a copied prompt |
| Receipt forged by third party | n/a | HMAC: only the operator can forge; Ed25519: only the holder of the signing key | Same | Key custody of the studio |
| Receipt/clip separation (watermark stripped, re-recorded, neural codec) | n/a | Loses link, byte-hash fails | Same | Watermark robustness is B6 |
| Offline machine misses revocation | Lease expiry bounds it | Lease expiry bounds it | Verifier must fetch list; list caching makes it stale | Choose lease length (e.g., 24 h) |
| Impersonation: attacker enrols someone else's voice | Not defeated by crypto | Same | Same | Challenge-response + same-device + spoof flag + no-go list |

## 7. Regulatory fit (what each text requires, with source quality)

Primary texts were not obtainable: the MeitY PDF returned 403, and I did not reach the Gazette or the DPDP Rules PDF. Treat every Indian statement below as [S].

- India IT Rules amendment on synthetically generated information (secondary: notified about 10 Feb 2026 as G.S.R. 120(E), effective 20 Feb 2026, per law-firm commentary [S]). Commentary says SGI (audio, visual or audio-visual content created or altered so that it appears real or indistinguishable from a real person or event) must be prominently labelled, with audio disclosure "where relevant", embedded permanent metadata/unique identifiers that cannot be removed, and platforms enabling SGI must put technical safeguards against deceptive impersonation and non-consensual content. One commentator says "where technically feasible" qualifies the metadata duty; another does not; I could not resolve it. [S] (https://ksandk.com/information-technology/it-rules-2026-regulating-synthetic-media/, https://blog.ipleaders.in/it-rules-2026/). Fit: your watermark + label + receipt id (`label_wav` writes a label into the WAV) matches the metadata/identifier theme; Design 2 strengthens "unique identifier" with a signed, third-party-verifiable record. A consent gate maps to "technical safeguards against non-consensual impersonation".
- DPDP Rules 2025 (secondary): notified 13 Nov 2025; Rule 4 (consent manager registration) in force one year after publication, i.e. 13 Nov 2026; most other operative provisions about 13 May 2027 (one SCC Online post disagrees and says a date to be notified) [S] (https://www.mondaq.com/india/data-protection/1708164/digital-personal-data-protection-rules-2025-notified, https://www.scconline.com/blog/post/2025/12/26/digital-personal-data-protection-rules-2025-key-highlights/). Consent managers must run an interoperable platform, keep records of consents "given, denied or withdrawn", ensure personal data content is "not readable by the Consent Manager", and keep records at least seven years [S] (https://www.mondaq.com/building-the-consent-layer-consent-managers-under-the-dpdp-rules-2025/1832140). Applicant must be an Indian company with minimum net worth Rs 2 crore [S]. Fit: BonClay is a data fiduciary/processor, not a consent manager; do not call yourself one. Design 1 + 3 are a good structural match to "records of consent given and withdrawn" and "not readable". A tension: seven-year consent-record retention for consent managers vs BonClay's 365-day voice retention. They apply to different actors; the consent record (without voice) can outlive the voice. Treat as a design question for counsel [U].
- EU AI Act Art. 50 [V] (https://artificialintelligenceact.eu/article/50/): providers of generative AI must ensure outputs are "marked in a machine-readable format and detectable as artificially generated or manipulated"; deployers of deepfake audio must "disclose that the content has been artificially generated or manipulated" (narrower for evidently artistic work); information must be presented "in a clear and distinguishable manner" and "at the latest at the time of the first interaction or exposure". Art. 50 does not require consent for cloning; consent comes from GDPR/biometric law [U]. Fit: watermark + receipt + disclosure satisfies marking in spirit; the consent binding is extra, not required.
- UK/US guidance: not researched [U].

This is not legal advice.

## 8. Effort, regulatory fit and novelty summary

| Design | Effort (est.) | Stops | Regulatory fit | Novelty |
|---|---|---|---|---|
| 1 Consent-gated key release | 3 to 5 d (+3 to 5 d Secure Enclave helper) | File theft, backups; crypto-shred on withdrawal | Supports DPDP withdrawal and erasure story; no mandate | Moderate: standard envelope encryption, but gating a TTS voice prompt on a revocable consent state in a local-first studio has no prior art I found |
| 2 Signed receipts + revocation log + consent hash in receipt | 4 to 6 d | Misattribution, silent withdrawal edits; third-party verification | Strong for IT Rules identifier/label, EU Art. 50 marking | Moderate-high in combination with per-clip consent hash |
| 3 VC + status list (+ C2PA custom assertion) | 5 to 8 d | Interoperable checking | Good narrative with DPDP consent records | Low alone; useful as export |

Publishable angle (my judgement): (a) a threat model that separates what crypto can bind (keys, outputs) from what it cannot (copied weights), with the measured limits; (b) "consent-state-gated generation + per-clip consent commitment + public signed revocation log" for open voice-cloning tools; (c) an evaluation of challenge-response timing against live cloning on Indian languages, which does not exist in the sources I found. Item (c) is the strongest empirical contribution and needs experiments, not just design.

## 9. What I could not verify

- OpenAI Voice Engine primary page: 403; used news/search summaries.
- Resemble consent mechanism: archived docs; nothing read.
- C2PA supported-format list and the full spec for any consent assertion: only about 200k of 579k characters read.
- IT Rules and DPDP Rules primary text; "where technically feasible" qualifier; exact Rule 4 date (one source disagrees).
- ASVspoof 5 results table numbers (only a summary; XMUspeech figures are self-reported by one team); no Indian-language per-language EER read for IndicFake/IndicSynth/SEA-Spoof; BanglaFake numbers from a snippet.
- AntiFake numbers from university summaries; AttackVC, VoiceBlock, SafeSpeech, voice NFTs not researched.
- Secure Enclave key types, keychain same-user access, DPAPI/TPM behaviour on RTX laptops: from memory.
- Effort estimates are my own.
- BonClay side: I read `consent_record.py`, `routers/consent.py`, and the receipt/withdrawal parts of `provenance.py`; I did not audit every storage path or the speaker-match thresholds.
