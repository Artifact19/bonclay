# Research prompts for BonClay

Copy any prompt below into a deep-research tool. Each one is self-contained: it carries the project context it needs.

## Shared context (already included in every prompt)

> BonClay is an open, local-first voice-AI studio for India. It runs on an Apple M4 MacBook
> (16 GB unified memory) with no keys, and should also run on Windows laptops with an RTX 3050 Ti (4 GB VRAM). Pipeline
> per spoken turn: browser VAD (Silero) → ASR (Groq Whisper, or MLX Whisper small + large-v3-turbo) → retrieval (BGE-M3
> + BM25, RRF) → LLM (Groq qwen/qwen3.8-27b → gpt-oss fallbacks → local Llama-3.2-3B 4-bit via mlx_lm) streamed
> sentence by sentence → TTS (Kokoro-82M on MLX; Qwen3-TTS 0.6B for cloning in 10 languages; Chatterbox for Hindi
> cloning; Indic-Speak for 22 Indian-language presets) → AudioSeal watermark + HMAC-signed receipt → public /verify page.
> Voice cloning requires a spoken one-time consent sentence + ECAPA speaker match. Agents use MCP tools (appointment
> booking). Measured locally: first audio 0.84 s, first answer 2.23 s (p95 3.58 s), Hindi first answer 3.07 s. Languages
> that matter: English, Hindi, Hinglish (code-mixed, Roman and Devanagari), Marathi, Tamil, Bengali, Telugu. Priorities:
> open/commercial-friendly licences, laptop-friendly, consent and provenance as the differentiator.

---

## Part A: priority research (unblocks the next build)

### A1. Exotel bidirectional streaming for a voice bot

```
Context: [paste the shared context above]

I need to connect BonClay agents to real phone calls through Exotel (India). I have an Exotel trial account with an
ExoPhone, a verified trial number, 500 credits and a custom app flow. I want Exotel's Voicebot / AgentStream
(bidirectional WebSocket media streaming) to stream the caller's audio to my server and play my agent's audio back.

Research, using Exotel's official developer docs, support articles, GitHub samples and community posts (2024-2026):
1. The exact WebSocket protocol: every event Exotel sends (connected, start, media, dtmf, mark, stop, clear…), all
   field names (stream_sid, call_sid, from, to, custom_parameters, sequence_number, chunk, timestamp, payload), and
   every event my server must send back (media, mark, clear) with exact JSON shapes. Include real example messages.
2. Audio format: encoding (raw/slin PCM16 vs μ-law), sample rate (8 kHz? 16 kHz option?), channels, endianness,
   required chunk size (multiples of 320 bytes?), pacing rules, maximum message size, and what happens if I send audio
   faster than real time.
3. How the WebSocket URL is configured in the flow builder (Voicebot vs Stream applet), whether query parameters or
   custom parameters can carry an auth token, whether wss is mandatory, timeouts, and how the call continues after
   the stream ends (next applet, hangup).
4. Barge-in: how to stop playback ("clear"), how marks confirm playback finished, typical latency.
5. Trial-account limits: verified numbers only, call duration, concurrent calls, credits per minute, inbound vs
   outbound, KYC needed for production, DLT/TRAI requirements for automated/AI calls in India (2026).
6. Any open-source reference implementations (Python/Node) for Exotel voicebots, Pipecat/LiveKit Exotel transports.
7. Pitfalls others hit: audio choppy, silence, one-way audio, 8 kHz resampling quality, tunnel (cloudflared/ngrok)
   issues, latency budget.

Output: a precise integration spec (tables of events/fields with examples), an audio-format spec, a step-by-step
setup checklist for the flow builder, a list of trial limits and compliance requirements, and a list of open
questions you could not verify - each claim with a source URL and date.
```

### A2. DhVaani-0.5 licence lineage

```
Context: [paste the shared context above]

I want to use ARTPARK-IISc's DhVaani-0.5 (Hugging Face: ARTPARK-IISc/DhVaani-0.5) for voice cloning in Marathi,
Tamil, Bengali, Telugu and other Indian languages, in a product that may be commercial. In my tests it works well
(0 failures in 50 renders; read-back error 4-18%). The model card says Apache-2.0, but it is fine-tuned from
k2-fsa/ZipVoice, whose pre-trained weights are released under CC-BY-NC due to the Emilia training data, and it was
trained on IndicTTS (IIT Madras licence), Rasa and IISc SYSPIN corpora.

Research:
1. The exact licence of every component: ZipVoice code vs weights (quote the text), the Emilia dataset licence, the
   IndicTTS corpus licence terms (find the IIT Madras licence PDF or text), Rasa and SYSPIN (CC-BY-4.0?), the Vocos
   vocoder. Quote the relevant clauses.
2. Whether a model fine-tuned from CC-BY-NC weights can be released under Apache-2.0 - what legal commentators,
   Hugging Face policy, and similar cases (e.g. derivatives of NC models) say. Does re-initialising the text embedding
   change anything?
3. Any public statement by ARTPARK/IISc about commercial use, a contact for licensing, and whether they offer
   commercial licences.
4. Equivalent alternatives with a clean commercial licence for Indian-language voice cloning (2025-2026), with their
   licence lineage verified the same way.
5. A draft, polite email to ARTPARK-IISc asking for written clarification (what to ask precisely).

Output: a licence lineage table (component → licence → commercial use allowed? → source URL), a risk assessment (not
legal advice), alternatives, and the draft email.
```

### A3. Real-speech ASR accuracy and speed for Hindi and Hinglish

```
Context: [paste the shared context above]

All my speech-recognition tests used synthetic voices. I need to know how ASR options perform on REAL Indian speech,
especially Hindi and Hinglish (code-mixed), on an Apple M4 (16 GB) and on an RTX 3050 Ti (4 GB VRAM).

Research:
1. Public benchmark results (WER/CER) on real Indian speech - IndicVoices, Kathbath, Vistaar, Svarah, MUCS
   Hindi-English code-switching, Lahaja, Voice of India (2026) - for: Whisper small, large-v3, large-v3-turbo,
   distil-whisper, NVIDIA Nemotron 3.5 / Canary / Parakeet multilingual, AI4Bharat IndicConformer 600M, Sarvam
   Saarika/Saaras, Meta MMS / Omnilingual ASR, and any newer 2026 open models. Note the dataset, split and whether
   Hinglish is in Roman or Devanagari.
2. Which models output Hinglish in Roman vs Devanagari, and how code-mixed speech is scored.
3. Latency/RTF on Apple Silicon (MLX, whisper.cpp, WhisperKit) and on small NVIDIA GPUs (faster-whisper int8,
   NeMo/TensorRT), for 3-8 s utterances.
4. Effects of trimming Whisper's 30 s input window to ~8-10 s on hallucinations and accuracy (any papers, issues,
   or reports), and safer ways to cut encoder cost (VAD trimming, padding strategies).
5. Streaming ASR options with partial hypotheses that work for Hindi (Nemotron streaming, Sarvam streaming,
   IndicConformer streaming?), their licences and hardware needs.
6. Free/open real-speech test sets I can legally download (≤ 1 hour) to build a 40-100 clip evaluation for Hindi and
   Hinglish, with licences.

Output: a comparison table (model × dataset × WER/CER × latency × licence × hardware), a recommended ASR setup for each
machine, a recommended evaluation set, and the risks of window trimming - with source URLs.
```

### A4. Reliable offline tool calling under 5 GB

```
Context: [paste the shared context above]

Offline (no cloud) my agents must call tools reliably: e.g. book_appointment(name, phone, date, time, doctor) from a
Hindi/Hinglish/English request like "Kal subah 11 baje Dr. Rao ke saath appointment book kar do, naam Priya, number
98…". With Llama-3.2-3B-4bit and Gemma-4-E4B-4bit on MLX, offline tool-call accuracy was only 25% and 12% (Gemma
needs thinking on to get 7/8, but then takes ~30 s).

Research (2025-2026):
1. Small models (≤ ~5 GB at 4-bit) with the best measured function-calling accuracy (BFCL v3/v4, tau-bench,
   NexusRaven, multilingual tool-use benchmarks): Qwen3 / Qwen3.5 small, Gemma 4 variants, Llama 3.x/4 small,
   Phi-4-mini, Hammer, xLAM-2, ToolACE, Granite, Sarvam-M-small - with Hindi/Hinglish ability.
2. Constrained decoding on Apple Silicon: grammar/JSON-schema constrained generation in mlx_lm or mlx-structured /
   outlines / xgrammar / llguidance on MLX, and how much it improves small-model tool calls and latency.
3. Small fine-tunes: datasets and recipes (LoRA on MLX) for multilingual tool calling, how many examples are needed,
   expected gains; synthetic data generation from a bigger model (Groq) for Indian-language booking intents.
4. Prompting tricks proven to help small models with tools (few-shot, slot-filling first, two-step plan→call).
5. Date/time resolution ("kal", "parso", "agle Monday", "tomorrow at four") - known failure modes and deterministic
   pre-parsing libraries for Hindi/Hinglish dates.

Output: a ranked shortlist (model/technique → measured accuracy → memory → latency on M4 → licence), a recommended
approach, a fine-tuning plan with data size and time, and source URLs.
```

### A5. Why macOS shows ~11 GB "wired" memory with MLX workers

```
Context: [paste the shared context above]

On a 16 GB M4 MacBook running several MLX processes (mlx_lm server with a 3B 4-bit model, mlx-whisper with small +
large-v3-turbo, mlx-audio Kokoro, sometimes a Qwen3-TTS worker) plus PyTorch CPU models, Activity Monitor showed
Wired Memory 11.4 GB, swap 11.6 GB and high memory pressure; one Python process used 7.5 GB.

Research:
1. How MLX/Metal allocates memory on Apple Silicon: wired vs active vs compressed; why Metal buffers count as wired;
   the role of iogpu.wired_limit_mb; mx.set_memory_limit / set_wired_limit / set_cache_limit; residency sets
   (macOS 15+); how mlx_lm and mlx-whisper hold weights (mmap vs copies); whether memory is released on idle.
2. How to measure exactly which process holds wired memory (footprint, vmmap, Instruments, `sudo memory_pressure`,
   MLX get_active_memory/get_peak_memory/get_cache_memory).
3. Proven ways to reduce wired memory for multi-model MLX stacks: one process hosting several models vs one per
   model, lazy loading, unloading, weight sharing, quantised Whisper, limits that don't hurt speed.
4. Interactions with PyTorch (CPU) models and the macOS compressor/swap - when swap hurts latency and when it's stale.
5. Any known MLX issues/bugs (GitHub) about memory not being freed, leaks in long-running servers, and fixes in 2026
   versions.

Output: an explanation in plain language, a measurement recipe, a ranked list of fixes with expected savings, and
links to sources and issues.
```

---

## Part B: deep research on novel ideas (innovation and optimisation)

### B1. Consent bound to the voice itself

```
Context: [paste the shared context above]

Today, BonClay checks consent at enrolment (one-time sentence + speaker match) and watermarks every output with a
receipt. I want to explore making consent cryptographically and technically BOUND to the cloned voice, so that a
voice prompt/embedding cannot be used without a valid, unrevoked consent - even if someone copies the files.

Research deeply and propose novel designs:
1. Prior art: consent management for biometric/voice data, "voice NFTs"/tokens, ElevenLabs/Resemble voice-verification
   mechanisms, Adobe/C2PA ingredient assertions, OpenAI Voice Engine safeguards, academic work on "unlearnable" or
   protected speaker embeddings, anti-cloning perturbations (AntiFake, AttackVC, VoiceBlock, SafeSpeech).
2. Designs to evaluate: encrypting speaker embeddings with a key released only by a consent service; signing voice
   prompts with the consent receipt and refusing generation without verification; embedding the consent ID in the
   watermark payload of every clip; revocation that invalidates future generation and flags past clips on /verify;
   zero-knowledge or verifiable-credential proofs of consent (W3C VC, DIDs) that a third party can check without
   seeing the voice.
3. Threat model: what each design defends against (stolen files, malicious operator, replayed consent, a clone reading
   the consent code) and what it cannot.
4. Liveness/anti-spoofing for the consent recording on real Indian speech: challenge-response timing, acoustic
   environment/room-response consistency between sample and consent, SSL-based synthetic speech detectors - measured
   accuracy on Indian languages.
5. Regulatory fit: India DPDP Rules 2025 (consent managers from Nov 2026), IT Rules 2026 on synthetic media, EU AI Act
   Article 50, UK/US guidance.

Output: a ranked set of 3 concrete designs (architecture, crypto primitives, data flow, effort, risks), a threat-model
table, evidence for each claim, and what would be genuinely novel/publishable.
```

### B2. Compile the knowledge base into instant speech

```
Context: [paste the shared context above]

Idea: instead of running ASR → retrieval → LLM → TTS for every question, "compile" each agent's documents ahead of time
into a graph of likely questions and approved, pre-synthesised, watermarked answers (in each supported language and
script), then serve a near-instant spoken answer when a caller's question matches - falling back to the live pipeline
otherwise. This could cut answer latency from ~2-3 s to ~0.3 s for common questions.

Research deeply:
1. Prior art: FAQ retrieval for voice bots, semantic caching for LLMs (GPTCache, semantic-router, vCache), answer
   precomputation in IVR/contact-centre systems, "speculative" responses, query auto-generation from documents
   (doc2query, synthetic Q generation), cache safety literature (false hits).
2. Matching for code-mixed speech: transliteration-invariant embeddings (Roman vs Devanagari Hinglish), phonetic
   matching on ASR output, matching on partial transcripts before the caller finishes, calibrated thresholds and
   abstention, cross-encoder verification; measured false-hit rates.
3. Freshness and correctness: invalidating answers when documents change, owner approval workflows, time-dependent
   answers (today's slots), personalisation.
4. Audio side: pre-synthesising answers per voice/language, stitching cached phrases with live slots ("your appointment
   is at {time}"), prosody continuity, watermark/receipt per served clip.
5. Evaluation design: hit rate, latency, false-hit harm on a realistic Indian SMB question set.

Output: an architecture, matching approach with expected false-hit rates, a safe rollout plan, an evaluation protocol,
and which parts are novel - with sources.
```

### B3. Talking before the caller finishes (speculative pipeline for Hindi and Hinglish)

```
Context: [paste the shared context above]

I want near-human turn-taking on a laptop: begin ASR, retrieval and even LLM drafting while the caller is still
speaking, decide end-of-turn early and correctly in Hindi/Hinglish, and start speaking within ~300-500 ms of the caller
finishing - without cutting people off.

Research deeply:
1. Streaming ASR with stable partial hypotheses (and stability scoring) for Hindi/Hinglish; incremental retrieval on
   partials; speculative LLM generation on partial transcripts with rollback (papers on incremental/simultaneous
   dialogue, "listen while speaking", duplex models such as Moshi as reference).
2. End-of-turn prediction using prosody + semantics for Indian languages: Smart Turn, LiveKit turn detector,
   VAP (voice activity projection) models, and what is known about Hindi turn-taking cues; how to build a small Hindi
   turn-taking dataset ethically.
3. Backchannels and fillers ("haan", "ji", "ek second") - when they help perceived latency without being annoying;
   culturally appropriate patterns.
4. Scheduling on one laptop: running streaming ASR, embeddings and the LLM concurrently on Apple GPU/ANE/CPU without
   contention; on a 4 GB NVIDIA GPU.
5. Evaluation: metrics for perceived responsiveness and interruption errors; a test protocol with real callers.

Output: a staged design (what to build first), expected latency gains with arithmetic, risks for code-mixed speech,
and what would be novel - with sources.
```

### B4. One scheduler for the Apple Neural Engine, GPU and CPU

```
Context: [paste the shared context above]

On Apple Silicon, all my models compete for the same GPU and memory. Idea: a scheduler that places each model on the
best unit (ANE via CoreML, GPU via MLX/Metal, CPU via Accelerate/ONNX) and orders work per turn to minimise latency and
memory - e.g. ASR encoder on ANE, LLM on GPU, embeddings on ANE/CPU, TTS on GPU only when needed.

Research deeply:
1. Measured performance of ANE vs GPU vs CPU on M3/M4 for Whisper encoders (WhisperKit, whisper.cpp CoreML), BERT-
   class embedders (CoreML, ANEMLL), small TTS (Kokoro CoreML), and small LLMs (ANEMLL, CoreML LLMs) - and the cost of
   moving data between units.
2. Concurrency: can ANE and GPU run truly in parallel for different processes? Memory bandwidth contention (120 GB/s on
   M4) and how it limits overlap. Metal/MLX streams, priorities, QoS.
3. Practical Python integration paths (coremltools, CoreML via PyObjC, Swift helper processes, WhisperKit server).
4. Memory: unified-memory accounting for CoreML vs MLX models; quantisation on ANE (palettisation, int8).
5. Prior systems: any research or products doing heterogeneous scheduling for on-device voice assistants (Apple's own
   Siri pipeline papers, Qualcomm AI Hub, academic work).

Output: a placement plan with expected per-stage latency and memory, a feasibility rating per component, the main
risks, and what would be novel - with sources and any benchmark numbers.
```

### B5. Teacher in the cloud, student on the laptop

```
Context: [paste the shared context above]

Idea: when online, BonClay uses a strong cloud model (Groq qwen 27B / gpt-oss-120b); offline it falls back to a weak 3B
local model (49% judged quality on Indian-language tasks vs 84% for Gemma 4 E4B). Could each agent continuously distil
the cloud model's behaviour (its own documents, tools, Hindi/Hinglish style) into a small per-agent LoRA adapter for the
local model, trained overnight on the laptop with MLX - so offline quality approaches online quality for that agent?

Research deeply:
1. Prior art on task-specific distillation from large to small models, on-device LoRA fine-tuning with MLX (speed and
   memory on M-series), multi-LoRA serving (hot-swapping adapters per agent in mlx_lm or llama.cpp).
2. Data: generating synthetic conversations per agent from its documents (question generation, multilingual,
   code-mixed), using cloud replies as targets, filtering with a judge; privacy constraints (DPDP) when using real
   call transcripts.
3. Expected gains: published results for distilling tool use, grounded QA and Hindi/Hinglish style into 1-4B models;
   how many examples and minutes of training are needed.
4. Safety: preserving guardrails (refusals, AI disclosure) through distillation; evaluating with the 60-case
   Hindi/Hinglish attack suite.
5. Same idea on Windows with a 4 GB NVIDIA GPU (QLoRA feasibility).

Output: a concrete pipeline (data → training → evaluation → serving), time/memory budget on an M4 16 GB, expected
quality gains with evidence, risks, and the novel angle - with sources.
```

### B6. Provenance that survives a phone call

```
Context: [paste the shared context above]

BonClay watermarks every clip (AudioSeal) and issues signed receipts verifiable on a public page. Watermarks survive
MP3, Opus and 8 kHz μ-law, but not ±5% speed, pitch shift, heavy noise or neural codecs, and they say nothing to the
person on the other end of a live call. I want provenance that works for the LISTENER of a phone/WhatsApp call in India.

Research deeply:
1. Watermark robustness research 2024-2026 (AudioSeal, WavMark, Timbre, SilentCipher, XAttnMark, latent watermarks,
   AudioMarkBench, RAW-Bench) - which survive speed/pitch changes, telephony codecs (AMR-NB/WB, G.711, EVS), VoLTE,
   acoustic re-recording, and neural codecs; capacity vs robustness trade-offs; psychoacoustic masking per language.
2. Live-call provenance UX: spoken disclosure plus a verifiable token (DTMF code, short spoken code, SMS/WhatsApp link
   with receipt), STIR/SHAKEN-like caller attestation in India, C2PA for live streams, how a caller could verify "this
   AI call is from clinic X" in under 10 s.
3. Detecting BonClay audio in the wild (social media re-uploads) - fingerprinting + watermark + receipt lookup.
4. India's 2026 IT Rules on labelling synthetic audio and TRAI rules - what a compliant live-call disclosure must
   contain.

Output: a recommended multi-layer provenance design for calls (what each layer survives), a listener-verification
flow, expected robustness numbers, regulatory fit, and what is novel - with sources.
```

---

### Quick prompts for the remaining questions (short form)

- **Pronunciation:** "Which open tools let me control pronunciation of Indian names and medicine names in Kokoro and
  multilingual TTS (lexicons, phoneme input, G2P for Hindi/English)? Give methods, quality evidence, effort."
- **Cheap reliable judge:** "Which open or cheap models agree best with human labels when judging Hindi/Hinglish voice-
  agent replies (grounding, safety, language)? Give agreement studies and calibration methods."
- **Groq alternatives:** "Compare Groq, Sarvam, Cerebras, OpenRouter, Together for Hindi/Hinglish chat + Whisper:
  quality, latency from India, free tiers, paid price per 1,000 three-minute calls (2026)."
- **WhatsApp 2026:** "WhatsApp Cloud API in India after the 1 Oct 2026 change: pricing, free service window, AI-
  assistant policy for business bots, voice notes; cite Meta pages."
- **Telecom compliance:** "Exact Indian rules (2026) for AI/automated voice calls: DLT, 140/160 series, consent, calling
  hours, disclosure, penalties; what a student pilot may do."
- **RTX 3050 Ti plan:** "Measured performance of faster-whisper int8, Kokoro (PyTorch, ONNX, TensorRT) and llama.cpp 3B
  Q4 on 4 GB laptop GPUs; VRAM budget for running them together."
- **Pricing:** "What Indian clinics, coaching centres and small shops pay in 2026 for receptionists, IVR and voice bots;
  price points that convert; interview script for 10 owners."
- **Licence strategy:** "Apache-2.0 + paid hosting vs AGPL vs open-core for an Indian voice-AI startup; examples
  (Bolna, Pipecat, LiveKit) and outcomes."
