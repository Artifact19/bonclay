# B3: speculative pipeline and early end-of-turn for Hindi/Hinglish phone calls

Date: 2026-10-17. Sources opened and read unless marked **[unverified]** (I only saw a search snippet, or the page did not load).

## Baseline (measured on real Exotel calls)

| Stage | ms |
|---|---|
| Silence wait (`DetectorConfig.end_silence_ms`, env `PHONE_END_SILENCE_MS`) | 500 |
| ASR (Groq Whisper, whole utterance) | ~300 |
| Retrieval | ~70 |
| LLM first token | ~270 |
| TTS first sentence | 500-900 |
| **Caller stops speaking to first audio** | **~1400 (0.5 s wait plus ~0.9 s)** |

Code today: `backend/services/phone_audio.py` `UtteranceDetector` (Silero at 8 kHz, `end_silence_ms=500`, `min_speech_ms=400`, `barge_ms=300`) emits one utterance. `backend/routers/phone_exotel.py` `_turn()` then runs the whole `_turn_event_stream` (ASR, retrieval, LLM, TTS) serially and sends PCM per `sentence` event. Nothing starts before the silence timer fires.

## Findings

### 1. Pipecat Smart Turn v3.x
- Licence **BSD-2-Clause**; 8M params; 8 MB int8 ONNX / 32 MB fp32. https://huggingface.co/pipecat-ai/smart-turn-v3
- Weights, training data and script are open; trained on `pipecat-ai/smart-turn-data-v3.1-train` (270k) with a 31.5k test set (model card). The GitHub README asks for more human data, and community cards say the corpus is "mostly synthetic TTS" (https://huggingface.co/codewithmoin/smart-turn-hinglish). The share of real human speech is **[unverified]**.
- 23 languages. Accuracy: **Hindi 93.44% (n=1,295)**, Marathi 87.60%, Bengali 84.10%, English 94.31%. https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/
- CPU latency (preprocessing plus inference): 12.6 ms on c7a.2xlarge, 33.8 ms on t3.2xlarge, 94.8 ms on t3.medium (same blog). README: "under 100ms on most cloud instances". M4 latency is **[unverified]**; expect around 10-20 ms.
- Input is **16 kHz mono PCM**, up to 8 s of audio (https://github.com/pipecat-ai/smart-turn). Neither page mentions 8 kHz or telephony. Our 8 kHz Exotel audio would have to be upsampled to 16 kHz, so the 4-8 kHz band would be empty. Accuracy on narrowband audio is **unknown and must be measured**. This is the biggest risk.
- The README says v3.2 is current; the community cards report v3.2 Hindi at 92.8% on the v3.2 test set.

### Community Hinglish fine-tunes (all self-reported, test sets differ)
| Model | Licence | Claimed result |
|---|---|---|
| codewithmoin/smart-turn-hinglish | BSD-2 ("check upstream terms" for commercial use) | Hindi 93.9% vs v3.2 92.8%; English 93.7% vs 94.7% |
| deveshu/hinglish-turn-detector | MIT (training corpora keep their own licences; edge-tts audio is under Microsoft terms) | Hindi 0.932, synthetic Hinglish 0.951; **real-voice test 0.800 (n=30, 1 speaker)**; 59-91 ms on laptop CPU, 14 ms for the distilled variant |
| Shrey160/hinglish-turn-v2 | **[not checked]** | 0.882 vs baseline 0.927 on the main test set |
| abhishek-040010/smart-turn-hinglish | **[not checked]** | on a 518-clip human Hinglish set: ROC-AUC 0.60 vs stock v3.2 0.75 (worse) |

Verdict: none of them clearly beats stock v3.2 on real speech. Use stock v3.2 and fine-tune later on our own call data.

### 2. LiveKit turn detector
- Multilingual model, 14 languages including Hindi: **Hindi TPR 99.4%, TNR 96.3%**, the best TNR of any language. About 0.1B params, int8 ONNX, under 500 MB RAM, CPU only. https://huggingface.co/livekit/turn-detector
- **Text-only**: it reads the transcript, so it needs streaming partial ASR. Licence is the **LiveKit Model License** (not OSI). Whether it may be used outside LiveKit Agents is **[unverified]**, so read it before shipping. No latency numbers are published; tens of ms on CPU is an **[unverified]** estimate.

### 3. Flipkart, ACL 2023 (arXiv 2305.18596)
https://arxiv.org/html/2305.18596
- Streaming CTC (LSTM) Hinglish voice search; a `</s>` token is added to the vocabulary. Endpoints come from forced alignment of a character-level model. Training adds `</s>` to the transcripts with early/late penalties, then fine-tunes for ~48k iterations (~12 h).
- Inference: the endpoint fires when at least one word has been emitted, `</s>` is the argmax, and P(`</s>`) ≥ α^(1+n_t/β), with α=0.8 and β=2.0. A VAD plus a maximum-time limit is the fallback.
- Mean EOS latency **2858 → 1525 ms (−1333 ms, 46.64%)**; WER 3.69% → 4.78% with EOS; early cut-offs in 2.24% of utterances, of which ~1.19% of all samples lost content. EOS coverage was 64.13%.
- What transfers: (a) the decaying-threshold rule, which can be reused on any end-of-turn probability, including Smart Turn's; (b) the idea that a ~1% harmful cut-off rate is acceptable; (c) a VAD fallback. Their latencies come from short voice-search queries; conversation is harder.

### 4. Local streaming ASR with partials
- **NVIDIA Nemotron 3.5 ASR Streaming Multilingual 0.6B**: 40 locales, chunk latency 80/160/560/1120 ms. Hindi readiness comes only from a third-party review **[unverified]**. The HF model card returned 401; licence and Hindi WER are **[unverified]**. An ONNX int4 community port exists (https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4). It may fit a 4 GB GPU at int4 **[unverified]**.
- **IndicConformer streaming**: no Hindi streaming latency or WER found **[gap]**.
- **WhisperLiveKit** (SimulStreaming/AlignAtt): licence conflicts between Apache-2.0 (README badge) and MIT (third-party catalogue), and SimulStreaming may carry a separate dual licence **[unverified]**. The README claims ~0.07 s per step for base.en on M4 with MLX **[unverified, English only]**. Whisper-family Hindi partials are known to be unstable (code-mixed script flips between Roman and Devanagari) **[unverified, expert judgement]**.
- Practical answer for now: re-run Groq Whisper on the growing buffer every ~300-500 ms of speech to get "pseudo-partials". This costs API calls, needs no new model, and is enough to feed the LLM turn detector and retrieval prefetch.

### 5. VAP (voice activity projection)
- Multilingual VAP (EN/ZH/JA) is arXiv 2403.06487 (CC BY 4.0). A model trained on one language transfers poorly; a jointly trained model matches monolingual models. It predicts 2 s ahead in bins of 0-200/200-600/600-1200/1200-2000 ms. A real-time CPU demo exists (Inoue, Ekstedt et al.) **[snippet only]**.
- It needs two-channel stereo dialogue data. **No Hindi VAP model or Hindi data found.** Phone calls do give clean separate channels (caller vs bot), which is a strong future fit. Not a near-term option.
- Hindi turn-taking cues (sentence-final verbs and auxiliaries "hai/tha/hoga", particles "na?", "haina", trailing "toh…", "matlab…", "aur…" as holds): **[unverified, linguistic expectation; no paper found]**. Verb-final SOV order actually helps semantic end-of-turn detection compared with English.

### 6. Speculative LLM generation
- **PredGen** (arXiv 2506.15556, Li and Grover): drafts a response on partial input; simulated **~2× lower latency**. Qwen2.5-7B Greedy reaches 1.9× on LMSYS and 1.6× on MT-Bench (lossless); Top-3 and Reflection go higher at a small quality cost **[numbers from snippet plus abstract; tables not opened]**. Code: github.com/jacklishufan/PredGen. Simulation only, English only.
- "Voice-Light": **not found [unverified that it exists under that name]**. Related: "Endpoint Anticipation for Low-Latency Spoken Dialogue" (arXiv 2606.13450) and "Speculative Interaction Agents" (arXiv 2605.13360), **[titles only, not read]**.
- For us: with Groq the LLM first token is only 270 ms, so speculation saves at most ~270 ms + 70 ms of retrieval, and only when the final transcript matches the draft's prefix.

### 7. Backchannels and fillers
- Duplex Conversation (arXiv 2205.15060) inserts backchannels before answers to cut perceived latency **[snippet]**. A 1999 Eurospeech study found immediate backchannels during speech gave some users a negative impression **[snippet]**. A 360-participant study found contextual fillers improved perceived response time **[snippet, source not opened]**. PACE (arXiv 2609.10372) co-designs fillers with routing **[title only]**.
- Takeaway: play a short, pre-rendered filler ("ji", "haan ji, ek second") only when the predicted delay is above ~700 ms (for example tool calls or slow TTS), never on every turn. The threshold is **[my judgement]**.

## Staged plan (savings against ~1400 ms)

| Stage | Change | Saving | Effort | Risk |
|---|---|---|---|---|
| **S0** | Cache TTS for frequent first sentences and fillers; send the first TTS chunk on the first clause (comma) rather than the full sentence | 200-400 ms on TTS (500-900 → ~300-500) **[estimate]** | 1-2 d | low |
| **S1** | Add Smart Turn v3.2 ONNX to `UtteranceDetector`: at 200 ms of silence, run it on the last ≤8 s (upsampled to 16 kHz). p≥0.7 ends the turn at 200 ms; p≥0.5 waits 400 ms; otherwise wait up to 1000-1200 ms (also fixes cut-offs on "aur…"). Use the Flipkart decaying threshold. Make it a new `DetectorConfig` field (`early_silence_ms`, `turn_model`). | ~300 ms on confident turns (500 → 200 wait), plus ~15-30 ms inference | 2-3 d | 8 kHz accuracy unknown → measure on 50-100 recorded call turns first |
| **S2** | Speculative start: at 200 ms of silence, begin ASR, retrieval and the LLM immediately (before end-of-turn is confirmed). If speech resumes, cancel (`stop_turn` already exists) and discard. Do not send audio until end-of-turn is confirmed. | Hides ASR + retrieval + LLM TTFT (~640 ms) inside the remaining wait whenever the wait is ≥200 ms | 2-3 d | extra Groq calls on false starts (~10-30%) **[estimate]** |
| **S3** | Pseudo-partials (Groq re-transcribe every 400 ms of speech) → prefetch retrieval, plus LiveKit text turn detector as a second vote (Hindi TNR 96.3%) | ~70 ms + better decisions | 3-5 d | LiveKit licence; Groq cost/rate limits |
| **S4** | PredGen-style draft on partials | ≤270 ms, only on prefix match | 1 w | low gain with a fast Groq TTFT; skip unless using local LLM |
| **S5** | Own Hindi/Hinglish turn data from consented calls (stereo, labelled by actual turn shifts) → fine-tune Smart Turn at 8 kHz; later VAP | quality | weeks | consent and anonymisation required; recordings only with explicit opt-in |

Arithmetic (confident turn after S0+S1+S2): wait 200 ms; ASR, retrieval and LLM start at that point, so the first token arrives ~640 ms after the caller stops; TTS clause ~350 ms → **first audio ≈ 990 ms, down from ~1400 ms (about −400 ms)**. Reaching the target of 300-500 ms needs local streaming ASR (removing the 300 ms) and sub-200 ms TTS first chunk. That is not achievable on the current stack without S3 and faster TTS **[estimate]**.

Hardware: Smart Turn (8M) and Silero run comfortably on CPU on both M4 and RTX 3050 Ti laptops. Nemotron 0.6B with local LLM and TTS on a 4 GB GPU is likely too much at once **[unverified]**, so keep ASR on Groq for phone calls.

## Risks
- Narrowband (8 kHz) audio and phone noise lower turn-model accuracy; there is no published number for this.
- Code-mixed trailing conjunctions ("toh", "aur", "matlab") lead to premature cut-offs; Flipkart saw ~1-2% harmful early cut-offs on simpler queries.
- Community Hinglish models are trained mainly on synthetic TTS speech and degrade on real voices (0.800 on the one real-voice test).
- LiveKit licence terms; Nemotron licence **[unverified]**.
- Speculative runs cost money and risk side effects: never execute MCP tools (bookings) speculatively, only read-only retrieval and LLM text.

## Verdict
Build S0, S1 and S2 now (about one week): Smart Turn v3.2 (BSD-2, 8 MB, ~15 ms) as an early end-of-turn gate on top of Silero, plus speculative start with cancellation. Expected first audio is ~1.0 s instead of ~1.4 s, with fewer cut-offs on hesitations. Gate the rollout on an offline test against recorded 8 kHz call turns: false-cut rate under 3% and median wait under 300 ms. Defer local streaming ASR, PredGen and VAP. The novel part worth claiming is a measured Hindi/Hinglish 8 kHz turn-taking benchmark and a fine-tune built from consented calls; no public one exists.
