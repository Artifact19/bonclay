# What is worth adding to BonClay: research, October 2026

Twenty-one research passes (15 October 2026) looked at models, channels, integrations, trust, regulation, cost and
market. Each recommendation below was checked against BonClay's constraints: open and commercial-friendly licences,
runs on an Apple Silicon laptop and on NVIDIA, Indian languages first, consent and provenance as the differentiator,
and a student team's time. Facts come from vendor pages, model cards, papers and law-firm summaries; several searches
were rate-limited, so anything marked *unverified* must be checked before it is relied on. Nothing here is legal advice.

**Verdicts:** **Adopt** = do it, evidence is good. **Try** = time-boxed spike, keep only if it wins on our own data.
**Skip** = not worth it now. **Effort:** S = days, M = 1-2 weeks, L = more.

## Where BonClay stands

| Area | Today | Main gap |
|---|---|---|
| Consent and cloning | One-time sentence with a code, speaker match (0.88 same / 0.14 different), delete-my-voice | A clone of the victim could read the sentence (no liveness); no itemised consent record |
| Provenance | AudioSeal + HMAC receipts + public /verify | No file metadata label (India IT Rules, 20 Feb 2026); not tested against codec attacks |
| Live agent | Streaming, first audio p50 1.6 s, full reply p50 4.4 s, hands-free, Hinglish | Silence-based turn end; Hindi speech recognition runs two passes (~2.1 s) |
| Voices | Kokoro live; Qwen3 cloning in 10 languages; Hindi clone via Chatterbox; 22 Indic presets | Indian-language cloning stops at Hindi; Indic voices are slow |
| Knowledge | MiniLM embeddings, dense search | Weak on names, numbers and Hinglish spellings; no reranking, no scanned-PDF OCR |
| Evaluation | 6 text attack personas, one yes/no judge (83/100) | Single-turn, text only, no audio callers, no Hinglish injection set |
| Analytics | Per-turn timings, performance panel | Nothing a business owner acts on (unanswered questions, success rate) |
| Channels | Browser only | No phone, WhatsApp or website widget - every competitor has phone numbers |
| Integrations | MCP appointment tool (being built), Groq, Supabase | No calendar, sheet/CRM or webhook out |

## The plan, ranked by value for effort

### Tier 1 - do next (small effort, clear value)

1. **Faster-feeling replies** - *Adopt, S.* Play a pre-rendered filler ("ek second…", "one moment") while the model
   writes, and use one Whisper large-v3-turbo pass for Hindi instead of small-then-turbo (measure on 20 clips first).
   Targets the 4.4 s full-reply gap without a rewrite.
2. **Silero VAD in the browser for hands-free** - *Adopt, M.* `@ricky0123/vad-web` (ONNX, runs in Safari/Chrome);
   a 2026 study puts Silero well ahead of energy and WebRTC VAD on real audio. Keep `echoCancellation: true`, require
   ~100 ms of speech before barge-in, flush audio within 200 ms. Check the model's LICENSE file (MIT per NVIDIA
   catalog; a GitHub badge says otherwise).
3. **Better knowledge search** - *Adopt, S.* Swap MiniLM for **BGE-M3** (MIT, 100+ languages, 8k context, dense +
   sparse) and add **BM25 + vector hybrid with reciprocal-rank fusion** for Hinglish, names and numbers. *Try*
   **bge-reranker-v2-m3** (Apache-2.0) only on knowledge turns, if it fits the latency budget.
4. **Knowledge-gap report** - *Adopt, S.* Cluster fallback answers and low-retrieval questions (with the embeddings
   already loaded) and show "top questions your agent could not answer" on Analytics. The single analytics feature
   owners value most; no new vendor. Redact phone and Aadhaar-like numbers first.
5. **Compliance pack** - *Adopt, S.* Itemised consent record per voice (what, why, retention, how to withdraw), a
   plain-language notice, a retention period, one-click withdrawal (deletion exists), and an "AI-generated" label plus
   the receipt ID written into exported audio metadata. Maps to the DPDP Rules (core duties from 13 May 2027) and the
   IT Rules on synthetic media (in force 20 Feb 2026).
6. **Owner-facing "test call" and a real attack set** - *Adopt, S-M.* A simulated caller that talks to the agent
   through the existing turn endpoint, so an owner hears it before going live; hand-write 50-100 Hinglish
   prompt-injection cases (no public dataset exists) and run text regressions with **promptfoo** (MIT) in CI.
7. **Groq model choice** - *Adopt, S.* `qwen/qwen3.8-27b` is a **Preview** model on Groq (no SLA, $0.80 in /
   $4.00 out per M tokens). Production models `openai/gpt-oss-20b`/`120b` cost about a fifth. Benchmark Hindi,
   Marathi and tool calls on both and pick the default deliberately.
8. **Apply for credits now** - *Adopt, S, no code.* Startup India DPIIT recognition (free, 7-14 days, unlocks IndiaAI
   compute at ~40% subsidy), Sarvam Startup Program, AWS Activate Founders, Microsoft Founders Hub.

### Tier 2 - open real channels (medium effort, the biggest competitive gap)

9. **WhatsApp voice notes** - *Adopt, M (1-2 weeks).* Meta Cloud API directly: webhook → Whisper → agent → TTS →
   OGG/Opus voice note back. Start on Meta's test number. In-window replies are free per Meta's page (secondary
   sources disagree after the 1 Oct 2026 pricing change - check the rate card). Needs bot disclosure and a
   human-escalation path; Meta bars general-purpose AI assistants, a business agent is likely fine (verify).
   *Skip* the WhatsApp Calling API for now (strict permission limits).
10. **Phone prototype with Exotel** - *Adopt as prototype, M.* Indian-native, bidirectional WebSocket (8 kHz PCM),
    7-day trial, one test number, up to 10 whitelisted numbers, KYC only for production. *Try* Twilio Media Streams
    (~$0.074/min India) as a second option; Plivo needs an India-incorporated company. Call only whitelisted numbers
    until DLT registration; AI disclosure at call start; TRAI's Sep 2026 amendment requires declaring automated
    calls to the telecom provider.
11. **Business integrations** - *Adopt, S-M.* A signed outbound webhook (same HMAC as receipts) so customers connect
    their own n8n/Make/Zapier (don't bundle n8n - Sustainable Use License); **Google Sheets** via a service account
    as a lightweight CRM/booking log; **Google Calendar** with shared calendars to avoid the sensitive-scope review.
    *Try* the official **Razorpay MCP** (MIT) for payment links, test mode first, with confirmation outside the model.
12. **Website widget** - *Try, M.* One script tag (`<bonclay-agent>`) with a short-lived signed session, domain
    allowlist and rate limit - and the consent challenge inside it, which no competitor offers.

### Tier 3 - time-boxed spikes (keep only if they win on our data)

13. **Indian-language cloning beyond Hindi** - *Try, 1 week:* **DhVaani-0.5** (ARTPARK-IISc, Apache-2.0, gated,
    0.1B, 10 Indian languages + English, 3-10 s reference). No published quality numbers: listening test in Hindi and
    Tamil, plus read-back CER. IndicF5 (MLX port exists) only as a non-commercial prototype (F5 weights CC-BY-NC).
14. **Better local LLM** - *Try:* **Gemma 4 E4B** 4-bit on MLX (~5 GB, native function calling). Adopt only if it
    beats Llama 3.2 3B on our Hinglish and tool-call set.
15. **Faster live voices** - *Try:* Supertonic 3 (99M ONNX, very fast on CPU; licence OpenRAIL-M; Hindi unconfirmed),
    OmniVoice via mlx-audio (offline cloning), Orpheus Hindi checkpoint (CUDA-first, 3B - memory risk on 16 GB).
16. **Smarter end of turn** - *Try:* Pipecat **Smart Turn v3** (BSD-2, ONNX on CPU) after VAD; verify on Hinglish.
17. **Provenance hardening** - *Try:* C2PA manifests on WAV/MP3 with `c2pa-python` (a trusted signing certificate
    costs money); run **AudioMarkBench** attacks in the eval lab and publish the measured detection rate; WavMark as
    a second watermark.
18. **Consent liveness** - *Adopt* a tighter challenge (shorter answer window, random digits, timing check); *Try*
    an XLS-R-300M + AASIST-style synthetic-speech score, logged but not used to refuse until we have an Indic test
    set (open detectors are near chance on real-world fakes).
19. **Hindi/Marathi interface** - *Try, M:* next-intl + Noto Sans Devanagari; the work is the translation, not the
    font.
20. **More templates** - *Try, S:* school-fee reminder (a copy of the EMI template), mandi prices for the farmer line
    (data.gov.in key, a day's lag), government-scheme helpline grounded in PDFs.

### Hosting when there are customers

One AWS Mumbai **g6.xlarge** (L4, 24 GB, ~$0.97/h; GPU quota can take weeks) plus Supabase in Mumbai for a pilot;
E2E Networks L4 (~$0.57/h + GST, INR billing) as the cheaper option; Groq Whisper (~$0.04 per audio hour) + Kokoro on
CPU as the budget path. The MLX paths need a CUDA alternative (faster-whisper, PyTorch Qwen3) in the Docker image.

### Pricing reference (India, per minute)

Sarvam ₹3.50 end to end; Ringg ₹6 per connected minute; white-label offers ₹3-4/min plus a commitment; Retell
$0.07-0.31; Vapi $0.05 platform + pass-through; ElevenLabs Agents +$0.08; Bland $0.12-0.14. A prepaid per-minute
wallet topped up through Razorpay (UPI Autopay) fits clinics and coaching centres better than seats.

## Skip for now

- **End-to-end speech-to-speech models** (Moshi, Qwen3-Omni, Step-Audio): no open model speaks Hindi, and they bypass
  grounding, tools, consent and watermarking. Keep the ASR → LLM → TTS cascade.
- **Rewriting on Pipecat or LiveKit Agents, WebRTC, Opus** - large rewrite; on a laptop the delay is model time, not
  network time.
- **Closed eval platforms** (Coval, Hamming, Cekura), **Arize Phoenix** (Elastic licence), **Helicone** (maintenance
  mode), **Langfuse on the laptop** (needs ClickHouse; fine later on a server).
- **jina-embeddings-v3** (CC BY-NC), **PyMuPDF** (AGPL), **AASIST/RawNet2 as a consent gate** (~43-49% error in the
  wild), **Resemble Detect/Pindrop** (cloud, closed), **Sarvam-1** (non-commercial), **sarvam-translate** (GPL-3).
- **AI voice for EMI recovery calls in v1** - RBI recovery rules from 1 Jan 2027 (8am-7pm, recorded, six-month
  retention) and pending litigation; keep the EMI template as a reminder demo.
