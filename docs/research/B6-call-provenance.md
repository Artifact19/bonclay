# B6. Provenance that survives a phone call (deep re-research, 2026-10-17)

Status legend: [P] primary source opened and read; [M] measured locally by us; [S] secondary source only; [U] unverified.

## 1. Bottom line

- A watermark is a poor primary signal for a LIVE call listener: the listener cannot run a detector. It is useful for after-the-fact forensics (recordings, re-uploads). For the listener the primary layers are (a) a spoken disclosure and (b) a verification path that does not depend on audio at all (callback to a registered number / SMS-WhatsApp link with the receipt).
- AudioSeal as shipped survives G.711 and 8 kHz telephone band-limiting in our own tests (5/5 clips), fails under speed changes (+5%, -10%) and under very-low-bitrate speech codecs (G.723.1, Opus 6 kbps). AMR-NB itself could not be tested (no encoder in the local ffmpeg).
- India: the TRAI Third Amendment (18 Sep 2026) regulates A2P calls (pre-declaration to the operator, termination charge) but, in the press release I read, contains NO duty to tell the recipient the voice is AI. The IT Rules 2026 amendment (G.S.R. 120(E)) puts labelling duties on intermediaries; I could not read the Gazette text, and whether it reaches a voice-agent operator is unresolved.

## 2. Watermark robustness: numbers from primary sources

### 2.1 AudioSeal paper (arXiv 2401.17264) [P, full PDF text read]
Setup: 16 kHz, 10k ten-second VoxPopuli clips, threshold chosen for best accuracy. Training range vs evaluation range are given in App. D.2.

| Edit (eval parameter) | AudioSeal acc (TPR/FPR), AUC | WavMark acc, AUC |
|---|---|---|
| Bandpass 500-5000 Hz | 1.00 (1.00/0.00), 1.00 | 1.00, 1.00 |
| Highpass 1500 Hz | 0.61 (0.82/0.60), 0.61 | 1.00, 1.00 |
| Lowpass 500 Hz | 0.99 (0.99/0.00), 0.99 | 0.50, 0.50 |
| Speed fast 1.25x | 0.99 (0.99/0.00), 1.00 | 0.50 (0.01/0.00), 0.15 |
| Resample via 32 kHz | 1.00, 1.00 | 1.00, 1.00 |
| White noise sd 0.05 | 0.91, 0.95 | 0.50, 0.50 |
| Pink noise sd 0.1 | 1.00, 1.00 | 0.88, 0.93 |
| MP3 32 kbps | 1.00, 1.00 | 1.00 (0.99/0.00), 0.99 |
| AAC 64 kbps | 1.00, 1.00 | 1.00, 1.00 |
| EnCodec (nq=16, 24 kHz) | 0.98 (0.98/0.01), 1.00 | 0.51, 0.50 |
| Average over all edits | 0.96, AUC 0.97 | 0.85, AUC 0.84 |

Notes: training used speed 0.9-1.1 and a bandpass 300-8000 Hz; evaluation speed was 1.25x. The paper states AudioSeal puts energy at lower frequencies (TF-loudness loss), which is why it is lowpass-robust, and that highpass above 500 Hz (training range) hurts. Telephone band (300-3400 Hz) removes only energy above 3.4 kHz, which the lowpass-500 Hz result suggests AudioSeal does not rely on [inference]. The paper does NOT test G.711, AMR, GSM, 8 kHz sampling, or pitch shift; it says desynchronisation attacks (time and pitch scaling) are a known weakness (Appendix text on synchronisation). URL: https://arxiv.org/pdf/2401.17264

### 2.2 AudioMarkBench (arXiv 2406.06979) [P, full PDF read]
- Twelve no-box perturbations (Table 2): time stretch speed factor 0.7-1.5, Gaussian noise SNR 5-40 dB, background noise SNR 5-40, SoundStream 4-16 quantizers, Opus 16-256 kbps, EnCodec 1.5-24 kbps, quantization 4-64 bits, highpass and lowpass cutoff ratio 0.1-0.5, smooth window 6-22, and others. No G.711, AMR, GSM, EVS, pitch shift, or 8 kHz resampling.
- Results are in FIGURES (13-20), not numeric tables. Text findings, quoted in substance: watermarks are robust to time stretch, lowpass, highpass, echo; EnCodec, SoundStream and Opus "achieve very high FNRs" at ViSQOL >= 3 (watermark removed with quality kept); AudioSeal "most robust", WavMark "least robust" (removed even by Gaussian noise and MP3); threshold tau=0.15 gave FPR and FNR near 0 for AudioSeal on clean data; FNR differs by sex and language (Tamil is among the 20 languages in AudioMarkData; per-language numbers are only in figures, I did not extract them) [U for any Hindi/Tamil number].
- Implication: Opus-class codecs at low bitrates (WhatsApp uses Opus [U, not checked]) can remove AudioSeal. Our own Opus test (below) agrees.
URL: https://arxiv.org/pdf/2406.06979

### 2.3 RAW-Bench (arXiv 2505.19663, Sony, Interspeech 2025) [P, Table 5 read via HTML; cell values read from a flattened table, verify before citing]
Pre-trained models, strict setting, bitwise accuracy ("tick" = >= 0.99-ish): AudioSeal: time stretch .97, reverb .87, EnCodec .96, DAC .52, MP3/OGG/AAC pass, polarity inversion .18, phase shift .62. SilentCipher: time stretch .41, EnCodec .33, DAC .32, MP3 .54, OGG .61, AAC .93. Timbre: reverb .96, EnCodec .64, DAC .60, MP3/OGG/AAC pass. WavMark: time stretch .82, EnCodec .00, DAC .00, MP3 .42, reverb .89. Clean bit/full-message accuracy: AudioSeal .997/.962, SilentCipher .999/.993, Timbre 1.000/1.000, WavMark .998/.993. Paper: full-message accuracy is "unacceptable for all models and settings, except for AS with EN compression". NOT evaluated: pitch shift, standalone resampling, Opus, any telephony codec. URL: https://arxiv.org/abs/2505.19663

### 2.4 Newer work that is relevant but not telephony
- Latent audio watermarking for neural codec resynthesis (arXiv 2609.25830) [P abstract read]: AudioSeal and WavMark degrade under repeated or low-bitrate EnCodec; a latent watermark degrades more gracefully. No AMR/G.711 tests found in the text I searched.
- Codec augmentation (Juvela and Wang, arXiv 2409.13382) [abstract only]: black-box codecs can be put in the training loop; DAC 8 kbps cited; no telephony codec confirmed. Titles seen in search but NOT read: CRAW (2609.03107), VoxWatermark (2606.15187, text did not decode), AWARE (2510.17512), Deep Audio Watermarks are Shallow (2504.10782). XAttnMark: no telephony data found [U].
- Deepfake DETECTION under communication codecs (arXiv 2504.12423) [P]: AMR-WB 6.6-23.85 kbps, EVS 5.9-128 kbps, IVAS, Opus 6-510 kbps, Speex, SILK, packet loss 0/1/5/10/20%; no AMR-NB or G.711. Useful as a codec test matrix, not watermark data.

### 2.5 Work on real phone/VoLTE/WhatsApp channels
- Searched twice (extended mode). Found NO paper measuring a speech watermark over a real cellular, VoLTE or WhatsApp call. This is a genuine gap, and filling it with a 20-call field test would itself be publishable.
- Closest: CallShield (arXiv 2601.09327, Qatar/QCRI) [P, full text read]: real-time 8 kHz neural watermark, 1 bit per 40 ms frame = 25 bps, 16-bit message, BCH + stop-and-wait ARQ, symmetric-key challenge-response between two endpoints. Reports above 91% bit accuracy at alpha 0.6 under 80% ducking/echo/pink noise/lowpass, 92% on clean audio, PESQ 4.44 (alpha 0.6) and 4.25 (alpha 1.0) on clean speech, perfect 16-bit recovery only 0.67 on clean and below 0.2 under white noise or bandpass, and authentication success above 99.2% clean / above 95% under "common distortions" with retries, taking about 63 s on average. Distortions are SIMULATED, not real calls, no codec is run (AMR-NB and EVS only appear in a sentence saying they preserve midband). Lesson: bit accuracy 92% yet full-message recovery 0.67 means you need error correction and repeated payloads, and a 63 s authentication is too slow for a listener check.
- A 1990s patent (US 6892175) used 8 kHz + IMA ADPCM + GSM 6.10 as a telephone-channel proxy [S via search snippet].

## 3. Our own measurement [M]

Method: `backend/services/provenance.py` (AudioSeal 16-bit message, speech-RMS-scaled to 27 dB SNR, threshold 0.5), 5 clips (3 English, 2 Hindi) from macOS `say` voices (Rishi en_IN, Lekha hi_IN), NOT Kokoro, so treat as indicative, n=5. Marked at 24 kHz, then attacked with ffmpeg. Unmarked clean clips scored 0.00-0.16 (below threshold). Script is in the session scratchpad, not committed. "msgOK" = the 16-bit payload decoded exactly.

| Channel | mean P | detected | payload OK |
|---|---|---|---|
| none | 1.000 | 5/5 | 5/5 |
| G.711 mu-law 8 kHz | 1.000 | 5/5 | 5/5 |
| G.711 A-law 8 kHz | 1.000 | 5/5 | 5/5 |
| G.726 32 kbps 8 kHz | 1.000 | 5/5 | 5/5 |
| G.723.1 (CELP, 8 kHz) | 0.034 | 0/5 | 0/5 |
| Opus NB 12 kbps voip | 0.737 | 5/5 | 0/5 |
| Opus NB 6 kbps voip | 0.007 | 0/5 | 0/5 |
| Opus WB 16 kHz 16 kbps | 0.958 | 5/5 | 5/5 |
| 300-3400 Hz bandpass, 8 kHz | 1.000 | 5/5 | 5/5 |
| mu-law + bandpass + noise at -30 dB | 0.867 | 5/5 | 5/5 |
| G.711 then G.726 (two hops) | 0.998 | 5/5 | 5/5 |
| speed +5% | 0.307 | 0/5 | 0/5 |
| speed -10% | 0.282 | 0/5 | 0/5 |
| varispeed +5% (pitch+speed) | 0.302 | 0/5 | 0/5 |

Same test with the watermark applied directly to the 8 kHz stream (what a live call would do): same pattern; G.711 5/5; Opus NB 12 kbps detected 5/5 but payload 0/5; noisy chain payload 3/5; speed and varispeed 0/5.
Caveats: no AMR-NB / AMR-WB / EVS encoder available locally, so real Indian mobile networks (VoLTE uses AMR-WB or EVS) are UNTESTED; G.723.1 failing suggests low-rate CELP modes (AMR-NB 4.75-7.4 kbps) may fail too [inference, U]. A 12.2 kbps AMR-NB mode may pass [U]. Test with the 3GPP reference codec (opencore-amr or the 3GPP TS 26.073 C code) before claiming anything. Speed result is consistent with the paper's "speed 1.25x passes" only partly: our attack parameters were within +-10% but failed, so the paper's 0.99 at 1.25x likely reflects a different attack implementation (resampling vs atempo) [U, worth a separate check].

## 4. Regulation

### 4.1 TRAI Third Amendment to TCCCPR 2018 [P: press release read; the Regulation PDF (…/Regulation_18092026.pdf) is a >10 MB scan that I could not fetch]
Press release No. 119/2026 dated 18 Sep 2026, https://www.trai.gov.in/sites/default/files/2026-09/PR_No119of2026.pdf. Quoted (short): A2P calls are defined as "voice calls initiated by an application, software system or automated platform without direct human dialing, including using autodialing, robo-calls and pre-recorded/artificial voice technologies"; "every entity using A2P calls must pre-declare such use to its TSP along with the details of CLIs"; "A2P calls made without the required prior declaration will be treated as UCC"; termination charge "of up to 0.05 per minute"; 140xx, 1600xx and 1601xx series "shall not be flagged as suspected spam"; inquiry-based commercial calls only for seven days and in verifiable written/digital form; complaint action at 3 or more unique complaints in ten days if the CLI is also AI-flagged; legacy consents valid only if verifiable and registered on the TSP digital platform.
- What it does NOT say (in the press release): any duty to tell the recipient the voice is AI/artificial. A vendor blog (caller.digital) claims the first 3-5 seconds must disclose; it quotes no regulation text, so treat that as [U] and do not rely on it. Medianama and IndiaLaw summaries [S] also show no recipient-disclosure duty, only the declaration to the operator.
- Implementation dates [S, IndiaLaw]: general provisions after 30 days, mandatory A2P declarations after 60 days, appellate/other after 90 days from Gazette publication.
- Practical reading: a voice agent placing automated calls from an ExoPhone is an A2P caller: declare to the TSP (Exotel), use the right series (140 for promotional, 1600 for service/transactional regulated entities [S: vendor blogs, e.g. Bolna docs, not TRAI text]), keep consent in verifiable form. Calling hours 9am-9pm for promotional calls [S, vendor blog, U]. Not legal advice.

### 4.2 IT (Intermediary Guidelines and Digital Media Ethics Code) Amendment Rules, 2026 [S only; Gazette text NOT read]
Searches repeatedly failed to surface the egazette/meity PDF; three attempts to fetch law-firm PDFs returned unreadable data or 403. What multiple secondary sources agree on: G.S.R. 120(E) notified 10 Feb 2026, in force 20 Feb 2026 (Shardul Amarchand Mangaldas legislative update, PDF read [S]; DSCI; iPleaders). Definition of "synthetically generated information": audio, visual or audio-visual content "artificially or algorithmically created, generated, modified or altered using a computer resource" so as to appear real and be "indistinguishable from a natural person or real-world occurrence" (SAM update [S]); good-faith routine editing, accessibility, and research/educational drafts excluded. Labelling (Hogan Lovells summary [S], IFF article [S] quoting the rules): content to be "prominently" labelled, audio disclosure "prominently fixed" in the audio, with "permanent metadata or technical provenance mechanisms ... including a unique identifier", "to the extent technically feasible", not to be "modified, suppressed, or removed"; Rule 3(3) places this on intermediaries that offer tools enabling creation of such content; significant social media intermediaries must obtain user declarations. Takedown 3 hours (2 hours for non-consensual intimate imagery) [S].
- Open question [U]: whether an operator running a voice agent on its own telephony is an "intermediary" offering a generation tool. BonClay itself, as a studio that lets users generate speech, plausibly IS in scope for the tool-provider duty (label plus provenance with unique identifier), which the existing receipt ID and AudioSeal payload map to well. Calls made by customers are a separate question. Needs counsel and the actual Gazette text.
- Not found: any rule text on what a "live-call disclosure" must contain. No primary text prescribes wording or timing of a spoken disclosure in India. (US comparison: FCC Feb 2024 ruling treats AI voices as "artificial" under the TCPA [S]; its Aug 2024 NPRM proposed disclosing AI use at the start of each call [S]; finalisation not checked [U].)

## 5. Caller attestation (STIR/SHAKEN equivalents)
- India has no STIR/SHAKEN-style signed attestation that I could confirm. What exists is CNAP (Calling Name Presentation): TRAI response to DoT 28 Oct 2025, DoT target of nationwide rollout by 31 Mar 2026 (not confirmed that it happened) [S]. It is a database lookup (CNAM linked to KYC via LNPD), not a cryptographic signature, and initially 5G/compatible 4G handsets only. Sources: scconline, Business Standard, thecybersecguru [S]. If Exotel numbers on 140/1600 series show registered entity names via CNAP, that is the closest "this call is from clinic X" signal; unverified for businesses [U].
- TRAI AI/ML spam flagging (Regulation 21A) shares suspect-CLI intelligence among TSPs; the 140xx/1600xx/1601xx series are exempted from being flagged (press release [P]).

## 6. Listener verification flows
- Real products: I found NO shipping outbound AI-voice product where the called party keys a DTMF code or speaks a code to verify the caller. Only patents (US 10397398 "Authentication using DTMF tones" and family) and vendor/practitioner guidance (JustCall; NoAICalls marketing page) [S]. Everything below is therefore a design, not an existing practice.
- Design options for BonClay, ranked by listener effort:
  1. Spoken disclosure in the first sentence (already implemented: `AI_DISCLOSURES` in backend.py, en/hi/mr...; phone_exotel starts the call with the agent speaking first). Add the business name, the purpose, and a short reference code.
  2. Short spoken reference, for example three words or four digits derived from the receipt (HMAC of call_sid truncated), repeated at the end. Listener checks it at /verify/<code> or by SMS reply. Low capacity, high robustness because it is plain speech.
  3. Out-of-band: send an SMS/WhatsApp message to the called number with the call receipt link right after the call starts or ends (the system already knows the called number on outbound calls; on inbound calls it knows the CLI). Verification time under 10 s needs the link to be pre-sent.
  4. Callback verification: caller dials the clinic's published number, the platform confirms "we called you at HH:MM about X". No audio fragility at all.
  5. DTMF challenge: caller presses a key sequence shown on the SMS/clinic site so the agent proves possession of the call context. Exotel streams DTMF events [A1 prompt scope; not re-verified here].
- Watermark-in-call-as-authentication (CallShield style) needs a client app that decodes audio; impractical for ordinary phone listeners, so deprioritise.

## 7. Re-upload detection (fingerprinting)
- No source found for fingerprinting of speech or AI speech for provenance. Neural fingerprinting results are for MUSIC (arXiv 2511.05399, Araz et al. ISMIR 2025), and Chromaprint/Dejavu behaviour on telephone-band speech is unverified [U].
- Practical stack that does not need a research result: (a) AudioSeal detect (survives MP3 32 kbps, AAC, 8 kHz G.711 per sections 2-3); (b) receipt lookup by payload message (provenance.find_by_message) and by audio SHA (exact copies only); (c) transcript hash plus ASR of the clip matched against stored turn text (survives speed changes and re-recordings, unlike the watermark); (d) store a landmark/chroma fingerprint of each served clip and test it on 20 re-encoded variants before trusting it. (c) is cheap and covers exactly the cases where the watermark fails (speed/pitch, low-bitrate codecs, neural codecs) [design inference].

## 8. Layered design grounded in the code

Code facts: phone path uses `pa.to_phone_pcm` to 8 kHz PCM (phone_audio.py, SUPPORTED_RATES 8/16/24 kHz); the Exotel router says the call reuses `_turn_event_stream` which includes AI disclosure, watermark and receipt (docstring, line 9); I did not find a call-level receipt object or a spoken reference code. provenance.py already has HMAC receipts, a 16-bit AudioSeal message per generation (`message_for(generation_id)`), `find_by_message`, `find_by_audio_sha`, voice withdrawal, and WAV label metadata.

| Layer | What it gives | Survives | Fails at |
|---|---|---|---|
| L1 spoken disclosure + business name + 4-digit code (per call) | Listener knows in under 10 s | Everything a call can do | Nothing technical; only compliance wording unknown |
| L2 SMS/WhatsApp link with call receipt, or callback number | Listener verification | Independent of audio | Needs number on file; consent to message |
| L3 signed call receipt: call_sid, agent, voice, per-turn receipts, transcript hash | Evidence for the operator and /verify | Cryptographic, exact copy | Any edit of the audio |
| L4 AudioSeal 16-bit payload per turn, applied at 8 kHz send time | Forensics on recordings | G.711, 300-3400 Hz band, two codec hops, noise -30 dB [M] | speed/pitch +-5-10% [M], Opus below ~12 kbps [M], neural codecs [P], unknown AMR/EVS |
| L5 transcript-hash and clip fingerprint lookup | Re-upload matching after L4 fails | speed/pitch, codec changes (design inference) | paraphrase, translation, voice conversion |

Listener flow target under 10 s: hear disclosure and code, open the SMS link (or type /verify and the code), see "AI call from clinic X, receipt valid, purpose Y".

Repeated-payload advice from CallShield: embed the same 16-bit message in every sentence and majority-vote across turns (provenance.detect already votes across 60 s pieces), because per-message exact recovery is much lower than bit accuracy.

## 9. Effort (one student developer)
- L1 code + wording, per-call reference code and /verify lookup: 1-2 days.
- L2 SMS/WhatsApp link: 1 day plus DLT/template registration lead time (days to weeks [S]).
- L3 call-level receipt: 1-2 days.
- Robustness harness on real codecs (3GPP AMR-NB/WB reference encoder, then a 20-call real Exotel/VoLTE/WhatsApp field test with the trial account): 2-3 days. This closes the largest evidence gap.
- L5 transcript-hash + fingerprint test: 2 days. Fine-tuning AudioSeal with codec/speed augmentation: weeks and a GPU, not recommended now.

## 10. Verdict
Build L1 to L3 and L5 now; treat the watermark (L4) as forensic only and describe it as such. Claim in marketing: "survives G.711 and telephone-band audio" (measured, n=5, synthetic voices), not "survives phone calls" (AMR/EVS/VoLTE/WhatsApp untested). Do not claim compliance with a TRAI disclosure duty (none found in primary text) or the IT Rules (Gazette not read; applicability unclear); get the Regulation PDF and the Gazette text read by someone with OCR or counsel. Novelty: a field measurement of watermark survival over real Indian mobile and WhatsApp calls plus an SMS/receipt listener flow is not found in the literature I searched; CallShield is the nearest and is simulation-only.

## 11. Unverified or not done
- Gazette (G.S.R. 120(E)) rule text; TRAI Regulation PDF text; Exotel DTMF event shape; WhatsApp = Opus bitrate; AMR/EVS tests; Hindi/Tamil-specific AudioMarkBench numbers; RAW-Bench cell values (flattened table); VoxWatermark, XAttnMark, CRAW, AWARE, "Deep Audio Watermarks are Shallow" contents; real-product DTMF flows; speech fingerprint performance; calling-hour rules from primary text.
