# Optimisation research: latency, memory and quality for the BonClay voice pipeline (15 October 2026)

Scope: what could make a spoken turn faster, lighter or better on (A) an Apple M4 MacBook with 16 GB and
(B) a Windows 11 laptop with an RTX 3050 Ti Laptop GPU (4 GB) and 16 GB RAM, without cloud costs. The report checks the
ideas already on the list (stable-prefix prompt cache, 4-bit LLM, faster-whisper on CUDA, kokoro-onnx and so on) and
does not repeat them.

Method: I read the turn path in `backend/backend.py` (`_turn_event_stream`, `_turn_messages`, `_agent_system_prompt`,
`retrieve_context`, `chunk_text`) and the services it calls (`services/llm.py`, `tools.py`, `sentences.py`,
`mlx_asr_worker.py`, `mlx_kokoro_worker.py`, `provenance.py`, `retrieval.py`, `hinglish.py`). I ran 8 small
experiments on this Mac while the app was running (section 3). Six parallel literature searches were then spot-checked
against the primary pages.

Tags: **[measured]** = I measured it here. **[verified]** = I read it on the primary page. **[secondary]** = from a
search summary or a third party, not confirmed. **[estimate]** = my arithmetic, not measured.

Measurement conditions: the machine was under heavy memory pressure during every experiment. Swap use was 4.3 GB at the
start and 5.4 to 6.3 GB during the runs, from the app plus other desktop apps. Absolute timings are therefore
pessimistic. The ratios between variants are the reliable part.

---

## 0. Six findings from reading the code (none were on the existing list)

1. **Agents with tools do not stream their reply.** When an agent has MCP tools (the clinic template does), `llm_worker`
   runs `_tool_turn` with `llm.chat()`, which is blocking, and only splits the finished text into sentences. That is
   why `llm_first_token_ms == llm_ms` in every row of `bench/results/latency_stream-20261015-183553.json`. Text-to-speech
   waits for the whole reply. A Hindi reply of about 90 tokens at the measured ~45 tok/s is 2 s of decoding before the
   first sound.
2. **Production chunks are 500 words, and the prompt puts them where the cache cannot reuse them.**
   `backend.chunk_text(chunk_size=500)` counts words, and `retrieve_context(top_k=4)` passes 4 chunks, so up to 2,000
   words (2,700+ tokens in English, far more in Hindi). `_agent_system_prompt` places that CONTEXT *inside the system
   message*. History, the tools section and the question all follow it, so every turn re-prefills them. The tools
   section (`tools.tools_prompt`) also contains the time to the minute, which breaks the prefix once a minute.
3. **The mlx_lm prompt cache has no byte limit.** `llm.py` starts `mlx_lm server` (0.32.0) without `--prompt-cache-bytes`.
   The server keeps up to 10 KV caches (`--prompt-cache-size` default). For Llama-3.2-3B, KV costs 28 layers × 8 KV heads ×
   128 × 2 (K, V) × 2 bytes = **114,688 B per token** [verified config]. A 3,800-token RAG prompt is therefore about 436 MB,
   and the worst case for 10 entries is about 4.3 GB of hidden memory.
4. **Whisper encodes 30 s of audio for a 3–6 s question.** The Hindi path runs the whisper-small encoder for language
   detection and then the large-v3-turbo encoder, both on audio padded to 30 s.
5. **Kokoro's memory peak grows with sentence length:** about 190 MB per second of audio [measured]. `sentences.MAX_CHARS
   = 220` allows sentences of about 15 s, which puts the transient peak around 3.3 GB on a 16 GB machine that is already
   swapping.
6. **Hinglish replies make an extra LLM call per sentence before speech.** `_turn_spoken` calls
   `hinglish.to_devanagari`, which calls the LLM, whenever the provider is not holding the local lock: always on Groq,
   and after the local reply has finished. That adds a network round trip per sentence on the critical path and uses
   more of Groq's free-tier quota. [estimate: +0.3–0.7 s per Hinglish sentence on Groq; not measured]

---

## 1. Executive summary: top ideas ranked by (gain × confidence) / effort

Baseline (bench, 30 turns, local models): first answer audio p50 **1.94 s en / 3.79 s hi**. Stage p50 en / hi:
ASR 0.44 / 1.26 s, retrieval 0.09 / 0.11 s, LLM (whole reply, not streamed: finding 0.1) 0.71 / 1.42 s,
TTS 0.69 / 0.75 s. These numbers do not include the **700 ms Silero `redemptionMs`** of silence the caller waits before
the upload even starts (`frontend/src/lib/vad.ts`).

### A. Apple M4, 16 GB

| # | Idea | Expected gain | Conf. | Effort |
|---|---|---|---|---|
| 1 | **Prompt layout and small chunks**: CONTEXT moves into the last user message, static rules and tools sit first, the minute-resolution clock leaves the system prompt; ~120-word chunks with top-3; `--prompt-cache-bytes` cap | Measured: TTFT **7.5 s → 0.75–0.95 s** on a 3-document agent. Up to ~4 GB of KV memory bounded | High | S, 5 h |
| 2 | **BGE-M3 in fp16** (it currently loads fp32) | **−1.1 GB RAM**; query encode 35 ms vs 94 ms; same hit@1/@3 on the 45-question set | High | S, 1.5 h |
| 3 | **One Whisper (turbo) with a trimmed encoder context (8–10 s buckets) and one-pass language ID**; whisper-small dropped | Measured: hi ASR **1.38 s → ~0.55 s**, en **0.43 s → ~0.31 s**, about −0.5 GB | Medium (2 clips; needs a WER check) | M, 10 h |
| 4 | **Stream the reply on tool agents** (sniff for `{`), **first phrase first**, cap chunks at ~100 chars | First answer: hi −0.5 to −0.9 s, en −0.2 to −0.4 s; p95 −2 to −4 s; Kokoro peak −0.6 to −1.3 GB | High | M, 8 h |
| 5 | **Pre-warm at speech start**: the browser's `onSpeechStart` pings the server, which prefills the static prefix and history on the idle LLM (max_tokens=1) and touches the workers | −0.1 to −0.3 s TTFT; removes swap page-in stalls on the first token | Medium | S, 4 h |
| 6 | **Speculative endpointing with Smart Turn v3.2** (8 MB, BSD-2, Hindi supported): upload at ~250 ms of silence, start ASR and retrieval, commit when Smart Turn says the turn is complete, cancel if speech resumes | −0.35 to −0.45 s on every turn (outside today's measurement window) | Medium | M–L, 16 h |
| 7 | **Hinglish transliteration off the critical path**: dictionary first for the sentence being spoken, the model call pipelined one sentence ahead | −0.3 to −0.7 s per Hinglish sentence on Groq [estimate] | Medium | S, 4 h |
| 8 | **Owner-approved answer cache** with pre-synthesised audio, plus static intents (hello, thanks, bye, "phir se boliye" replays the last clip) | On a hit, first answer = ASR + ~0.15 s (en ~0.45 s, hi ~0.7 s after idea 3) | Medium-low (unknown hit rate; Hinglish precision risk measured) | M–L, 18 h |
| 9 | **Whisper turbo 4-bit** (already in the HF cache, but its file layout does not load in mlx-whisper 0.4.3) | about −1 GB [estimate] | Low-medium | S, 3 h + validation |
| 10 | **Nemotron 3.5 ASR Streaming 0.6B** spike (Hindi in the top tier, cache-aware streaming, OpenMDW) | Transcript about 0.2–0.6 s after end of speech, with partials for early retrieval | Low (no Mac runtime confirmed) | L, 20 h |

Together, ideas 1–5 should bring first answer p50 to about **1.5 s en / 2.3 s hi**. Idea 6 then saves about another
0.4 s of perceived wait. The memory items (1, 2, 3, 4, 9) add up to roughly 3–5 GB, which is about the size of today's
swap.

### B. RTX 3050 Ti Laptop (4 GB) + 16 GB RAM, Windows 11

| # | Idea | Expected gain | Conf. | Effort |
|---|---|---|---|---|
| 1 | **VRAM budget discipline**: llama.cpp `-np 1 -c 4096 -fa on -ctk q8_0 -ctv q8_0`; as few CUDA processes as possible; NVIDIA "CUDA – Sysmem Fallback Policy = Prefer No Sysmem Fallback" for `python.exe` and `llama-server.exe` | KV 0.46 → 0.23 GB; avoids the silent driver fallback to shared memory | High (mechanism) / Medium (size) | S, 4 h |
| 2 | **Page Whisper into VRAM on demand**: keep turbo int8 in pinned RAM, upload at speech start, release after ASR (the LLM stays resident) | Lets turbo and a 3B LLM share 4 GB; upload ~60–130 ms hidden behind speech [estimate] | Medium-low | M, 10 h |
| 3 | Prompt layout and small chunks (shared code with A1) | Smaller than on the Mac (CUDA prefill is faster), but less KV VRAM and Groq TPM headroom | High | S (shared) |
| 4 | Stream tool-agent replies and first phrase first (shared with A4) | Same as on the Mac | High | shared |
| 5 | Speculative endpointing and Smart Turn (shared with A6; Smart Turn has a CPU int8 build) | −0.35 to −0.45 s | Medium | shared |
| 6 | **Kokoro ONNX fp16 on CUDA EP in fixed-length buckets, then try the TensorRT EP** | Third-party GTX 1650 run: CUDA EP 482 ms → TRT 101 ms [secondary] | Low-medium | M, 8 h |
| 7 | Owner-approved answer cache (shared with A8) | Same | Medium-low | shared |
| 8 | **Nemotron 3.5 streaming ASR on CUDA** (NeMo or NeMo-Speech.cpp GGUF): first-class Hindi, streaming | Removes post-speech ASR time; FLEURS hi WER 7.1–8.1 % with a language ID [verified] | Low-medium | L, 20 h |
| 9 | Embeddings stay on the CPU in **int8 ONNX/OpenVINO**, not fp16 (most laptop x86 CPUs lack fast fp16, so idea A2 does not port) | Keeps the 4 GB GPU free; ~570 MB int8 [secondary] | Medium | S, 4 h |
| 10 | Power profile: plugged in, Windows "Best performance", NVIDIA "Prefer maximum performance" for the app's executables | Avoids clock drops in power-saving mode [unverified size] | Low | S, 1 h |

---

## 2. Deep dives (top 6)

### 2.1 Prompt layout, chunk size and a bounded prompt cache (A1, B3)

**How it works.** In a transformer, prefill cost is proportional to the uncached prompt tokens. `mlx_lm server` 0.32 keeps
an LRU trie of KV caches and reuses the longest common token prefix (`LRUPromptCache.fetch_nearest_cache`;
`usage.prompt_tokens_details.cached_tokens` reports the hit). Anything that changes early in the prompt invalidates
everything after it.

**Evidence [measured, section 3.1].** On the live server (Llama-3.2-3B-4bit, M4, under swap) prefill runs at about
**470–540 tok/s**, so each 1,000 uncached prompt tokens costs about **2 s**.

- Today's layout with a 3-document agent and four 500-word chunks: 3,700–4,000-token prompts, 324 cached, TTFT median **7.56 s**.
- Same prompt, cached system prefix and history: identical system + new user message gave 893 cached, **0.11 s**. With
  history appended, 904 cached, **0.13 s**.
- Small chunks (2 × 120 words, static system first): 1,250–1,330 tokens, 893–895 cached, TTFT **0.74–0.95 s**.

The retrieval eval already uses ~40-word chunks and gets hit@3 = 0.978 hybrid
(`bench/results/retrieval_eval-20261015-130600.json`), so small chunks do not hurt finding the right passage on this set.

**Gain arithmetic.**
- 3-document agent: 7.5 s → ~0.9 s TTFT (−6.6 s). This is the case real customers hit with multi-page PDFs.
- Clinic template: the whole document is one 2,827-char chunk, so CONTEXT is identical every turn and the cache mostly
  hits today. The gain there is avoiding misses: 1 in ~10 turns when the minute changes, and every language change
  (`Reply in …` sits before CONTEXT). A miss costs ~1,200 tokens ≈ 2.4 s, so the p95 improves more than the p50.
- Memory: with `--prompt-cache-bytes 600000000` (~5,000 tokens of KV), the worst case drops from ~4.3 GB to 0.6 GB.
- Groq (8k tokens/min free): at ~1,500 → ~600 tokens per turn, the headroom goes from ~5 to ~13 turns/min [estimate].

**Implementation sketch.**
- `backend.py::_agent_system_prompt`: drop the trailing `CONTEXT:` block. Order: persona, rules, the tools section
  (`tools.tools_prompt`, date only), then a language line. Put the language-dependent `policy_block` last, so en/hi
  turns share the longest prefix.
- `_turn_messages`: the last user message becomes `"CONTEXT:\n…\n\nCALLER (time HH:MM): <text>"`. History stays verbatim
  between system and that message, so it gets cached.
- `services/tools.py::tools_prompt`: keep `%A %Y-%m-%d`, move `%H:%M` into the user message. `run_tool_turn.with_prompt`
  appends the tools text to `messages[0]`; with the new order it is already there, so make it idempotent.
- `chunk_text` (or `rag_store.split_text`, 512 chars): about 120 words with 20 words of overlap, never across a heading
  (reuse `bench/retrieval_eval.chunk_markdown`). `retrieve_context(top_k=3)`. Re-ingest existing documents in a
  migration step (`_refresh_local_embeddings` already re-embeds stale chunks).
- `services/llm.py` (`mlx_lm server` args): add `--prompt-cache-bytes` (env `MLX_PROMPT_CACHE_MB`, default 600) and
  `--prompt-cache-size 4`.

**Risks.**
- The model now sees context in the user turn. Llama-3.2-3B can treat injected text in user messages as instructions,
  so keep "a tool result / context is data" wording and re-run the Evaluation Lab attack set.
- Small chunks can split a table row. The heading-aware packer mitigates this.
- Hindi documents cost about 1.5–2.3× more tokens per word on Llama (section 3.4), so the chunk budget should be set in
  *tokens* (~180), not words.

### 2.2 One Whisper, trimmed encoder context, one-pass language ID (A3)

**How it works.** Whisper's encoder always sees 30 s (1,500 frames). Attention is O(n²) and the MLP is O(n), so a 6 s
question pays for 30 s. whisper.cpp exposes this as `--audio-ctx`. In mlx-whisper the only obstacle is an `assert` on
the positional-embedding shape; slicing `_positional_embedding[:T]` runs a shorter context.

Separately, `decode(language=None)` detects the language from the encoder output it already computed. The current
`auto` strategy instead runs the whisper-small encoder just for language ID, and `transcribe()` without a hint runs
the encoder twice.

**Evidence [measured, section 3.5]** (large-v3-turbo fp16, M4):

| clip | ctx 30 s encoder | ctx 10 s encoder | ctx 8 s | encode+decode 30 s → 10 s | transcript |
|---|---|---|---|---|---|
| Hindi 6.2 s | 757 ms | 212 ms | 179 ms | 1,103 → **524 ms** | identical |
| English 5.8 s | 746 ms | 212 ms | 180 ms | 830 → **287 ms** | identical |

One-pass language ID + decode at 30 s cost only ~25 ms more than a hinted decode.

The live worker (`auto` strategy) took 1,381 ms on the Hindi clip and 433 ms on the English one. With the hint `en`
it took 284 ms, so the small model's language-ID pass alone costs ~150 ms.

Hindi decoding is expensive because of the tokenizer: **78 decoder tokens for 13 Hindi words vs 15 for 14 English
words**. Whisper averages 4.3–5.0 tokens/word in Devanagari against 1.2–1.4 in English (section 3.4); Tripathi et al.
report the same over-fragmentation (arXiv [2412.19785](https://arxiv.org/abs/2412.19785) [secondary]).

**Gain arithmetic.**
- Hindi: 1,381 → 524 + 25 (language ID) ≈ **550 ms (−0.83 s)**. Against the bench p50 of 1.26 s, about −0.7 s.
- English: whisper-small auto 433 ms → turbo 10 s context 287 + 25 ≈ **310 ms (−0.12 s)**, and turbo is more accurate
  than small.
- Memory: whisper-small (~0.5 GB fp16) is no longer needed.

**Implementation sketch** (`services/mlx_asr_worker.py`).
- Patch `AudioEncoder.__call__` once at start-up to add `pos[:x.shape[1]]`.
- Add `run_trimmed(audio, language)`:
  - `bucket = next(b for b in (8, 10, 15, 20, 30) if dur + 1.5 <= b)`. Fixed buckets keep MLX shape caches warm.
  - mel = `pad_or_trim(log_mel_spectrogram(audio, padding=N_SAMPLES), bucket*100)`.
  - `decode(model, mel, DecodingOptions(language=hint_or_None, without_timestamps=True, fp16=True))`.
- Guard: if `res.compression_ratio > 2.4`, `avg_logprob < -1.0`, or the text is empty with speech present, re-run at the
  30 s context.
- New strategy `turbo_trim`; keep `auto` as the fallback.
- Language prior: pass the conversation's last language as a hint only when the agent is monolingual. Otherwise let the
  one-pass language ID decide (it costs ~25 ms).
- Keep the warm-up and `keep_models_resident`, with only the turbo model.

**Risks.**
- Turbo is not fine-tuned for short contexts. FUTO's audio-context-fine-tuned models
  ([github.com/futo-org/whisper-acft](https://github.com/futo-org/whisper-acft), tiny/base/small only) exist because
  default Whisper can repeat or hallucinate when the context is cut to the exact clip length: small.en scored WER 79.7
  at a dynamic context in FUTO's table [verified]. Keeping ≥1.5 s of padding inside the bucket is milder, and both of
  my clips were exact, but **n = 2 synthetic Kokoro clips is not validation**. Run 40 real clips (en, hi, Hinglish,
  noisy, 8 kHz phone) comparing WER and hallucination counts at 30 s vs bucket.
- Hinglish: Whisper's script choice for code-mixed speech varies with context. Check that the output script is
  unchanged, because `language_policy` keys off the script.

### 2.3 Stream tool-agent replies, first phrase first, cap chunk length (A4, B4)

**How it works.**
- **Stream sniffing.** In the JSON tool protocol the model answers either a JSON object (starts with `{`) or plain
  speech. Stream it, buffer until the first non-space character arrives, and if it is not `{`, treat the stream as the
  answer and feed `SentenceSplitter` immediately.
- **First phrase first.** For the first chunk only, cut at the first `, ; :` or `।`-like clause boundary after at
  least 4 words (Hindi: also after `है`/`हैं` followed by a comma). Later chunks stay sentence-sized for prosody.
- **Chunk cap.** `MAX_CHARS` 220 → ~110.

**Evidence [measured, section 3.2].** Kokoro-MLX time is linear in audio length: en ≈ 22 ms + 69.6 ms per audio-second,
hi ≈ 14 ms + 70 ms/s (RTF 0.071–0.083), so a short first chunk is not penalised by a fixed overhead. Peak MLX memory ≈
441 MB + 188 MB per audio-second: 761 MB at 1.7 s, 2,024 MB at 8.4 s. The LLM decodes at ~45 tok/s with 1.62
tokens/word (en) and 2.4 (hi) on Llama-3.2 [measured, section 3.1/3.4]. Kokoro-FastAPI warns that small chunks cause
intonation artifacts ([remsky/Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) [secondary]); that is why only
the first chunk is cut short.

**Gain arithmetic (local LLM, tool agent).**
- Hindi, first answer sentence of 15 words:
  - today: the full reply (~40 words × 2.3 = 92 tokens ≈ 2.0 s) must finish, then the 15-word TTS takes ~455 ms.
  - after: a 5-word phrase (≈12 tokens ≈ 0.27 s), then TTS ≈ 14 + 70 × 2.0 s ≈ 155 ms.
  - Saving ≈ (2.0 − 0.27) + (0.455 − 0.155) ≈ **2.0 s for long replies, ~0.6–0.9 s at the p50** (the p50 Hindi reply is
    only ~20 words).
- English p50: the reply is ~1 sentence of 13 words (≈21 tokens, 0.47 s). The phrase arrives ~0.25 s earlier and its
  TTS is ~0.18 s shorter: **≈ −0.3 s**.
- Bench p95 rows of 5–9 s (multi-sentence or tool replies) drop by 2–4 s.
- Memory: a 110-char cap keeps Kokoro's peak at about 1.8 GB or less (vs up to ~3.3 GB); first phrases at ~1.1 GB.

**Implementation sketch.**
- New `tools.run_tool_turn_stream(...)` mirrors `run_tool_turn`'s JSON branch:
  - `llm.chat_stream(msgs)`; peek at the first non-space character.
  - `{`: accumulate, then `parse_tool_request`, execute, loop as today.
  - Otherwise: yield deltas.
- In `_turn_event_stream.llm_worker`, call it on the tools path and reuse the existing `route()` (first-sentence
  language check).
- The booking-ID guard (`unconfirmed_booking_ids`) must run **per sentence before it is spoken**, against
  `calls` + `known_text`. Replace a failing sentence with `UNCONFIRMED_LINE` and stop.
- `services/sentences.py`: `SentenceSplitter(first_min_words=4, first_max_chars=60)`; `MAX_CHARS=110`, `SOFT_MIN=40`.
- The native-tools path (Groq `tools=`) can keep the blocking call at first.

**Risks.**
- The model may emit prose and then JSON ("Let me check… {"tool":…}"). Detect `{"tool"` inside the stream and stop
  speaking at that point (it already fails `parse_tool_request` today).
- A wrong-language first phrase is caught by the existing `route()` check, but with fewer words (≥3 needed).
- In Hindi a comma split can separate a verb from its object; keep `first_min_words` ≥ 4 and listen to 20 replies.

### 2.4 Speculative endpointing with a semantic turn detector (A6, B5)

**How it works.**
- Today the browser waits for 700 ms of silence (`DEFAULT_SILERO.redemptionMs`), then uploads.
- Instead, at ~250 ms of silence the browser uploads the utterance as **tentative**. The server starts ASR, retrieval
  and the LLM prefill, and runs **Smart Turn v3.2** on the last ≤8 s of audio.
- If Smart Turn says "complete" (or the browser reaches 700 ms of silence), the turn is committed and its SSE stream
  goes ahead. If the caller resumes speaking, the browser sends `cancel` and later uploads the whole utterance.

**Evidence.**
- Smart Turn v3.2: BSD-2-Clause, ~8M parameters, 8 MB int8 CPU ONNX
  ([pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn), files on
  [HF](https://huggingface.co/pipecat-ai/smart-turn-v3/tree/main) [verified]); 23 languages including Hindi, Marathi
  and Bengali [verified]. Hindi accuracy 93.4 % on 1,295 samples and 12.6–95 ms CPU latency (Daily blog [secondary]).
- LiveKit's turn detector also covers Hindi, but it is a 396 MB Qwen2.5-0.5B under its own licence
  ([docs](https://docs.livekit.io/agents/logic/turns/turn-detector) [secondary]).
- Deepgram's Flux "eager end of turn" reports ~150 ms median savings (Telnyx [secondary]). PredGen (arXiv 2506.15556)
  reports roughly 2× lower audio latency by speculating before the user finishes [secondary].
- Not found anywhere: a validated 200–300 ms silence floor for Hindi or Hinglish.

**Gain arithmetic.**
- Commit at 250 ms + 30 ms (Smart Turn) instead of 700 ms: **−0.42 s** on turns judged complete.
- Even when the turn only commits at 700 ms, the ASR and retrieval (~0.4–0.65 s after idea 3) started 450 ms earlier,
  so **≈ −0.45 s** either way.
- Wasted work on false starts costs one ASR run (~0.3–0.55 s of GPU) each.

**Implementation sketch.**
- `frontend/src/hooks/useHandsFree.ts`: add a `tentativeMs = 250` timer in the Silero frame callback; on expiry, POST
  `/turn/stream?tentative=1&turn_key=…`.
- `backend.py::process_turn_stream`: keep a `tentative[turn_key]` task that runs `_turn_asr` and `_turn_context` but
  holds the SSE `sentence` events. A new `POST /turn/{key}/commit|cancel` releases or cancels it. ASR runs under the
  worker lock, so a cancel must not interrupt a decode midway; just drop the result.
- Run Smart Turn (onnxruntime, already a dependency through Silero on the web side; on the server add the
  `onnxruntime` CPU wheel) inside the tentative task. Commit server-side when P(complete) > 0.7.
- Telephony (Exotel 8 kHz) gets the same logic server-side; Smart Turn expects 16 kHz, so upsample.

**Risks.**
- Cutting callers off mid-thought, which is worse than slowness. Hindi speakers often pause after `तो`, `और`, `कि`.
  Build a 100-clip Hindi/Hinglish false-cutoff set first and ship behind a flag.
- Downloading the 8.7 MB model needs the user's approval (I did not download it).

### 2.5 Owner-approved answer cache with pre-synthesised audio, plus static intents (A8, B7)

**How it works.**
- At ingest, the LLM proposes 20–60 likely questions per agent from the documents (doc2query style), each with a short
  answer. They are generated **in English, Hindi (Devanagari) and Hinglish (Roman) as separate keys** and stored in
  `services/answer_cache.py`.
- The owner approves or edits each answer in the Knowledge tab; `services/knowledge_gaps.py` already clusters real
  questions, which can feed candidates later.
- Approved answers are pre-synthesised per voice. At turn time, the query embedding (computed once, reused by
  `retrieve_context`) is matched against keys of the same language.
- Serve the cached clip only when **all** of these hold:
  - sim ≥ τ for that language;
  - margin (top-1 − top-2) ≥ 0.05;
  - numbers, weekdays and doctor names in the query appear in the key;
  - the turn has no anaphora ("and on Sunday?") and no active tool flow.
- The clip is still watermarked per turn (~100 ms for 5 s of audio, section 3.3) and gets its own receipt.
- Static intents (greeting, thanks, bye, "repeat") work the same way. "Repeat" replays the previous turn's audio file
  with zero synthesis.

**Evidence.**
- vCache (arXiv [2502.03771](https://arxiv.org/abs/2502.03771), ICLR 2026): static thresholds at 0.97–0.99 gave
  37–67 % hits with 2.5–5.2 % wrong answers on open-domain traffic; per-prompt learned thresholds reach 0.5 % error at
  41 % hits [secondary].
- Sierra says pre-computed phrases bring playback latency "to zero"
  ([sierra.ai/blog/voice-latency](https://sierra.ai/blog/voice-latency) [secondary]).
- **My cross-lingual test [measured, section 3.6]** (BGE-M3, English keys, Hindi/Hinglish queries): correct matches
  scored 0.57–0.90, wrong top-1 matches up to **0.685**. Devanagari Hindi matched at a median of 0.86; **Hinglish
  matched at a median of 0.64**, which sits *below* the worst wrong match. One global threshold cannot be safe for
  Hinglish, which is why keys must be stored per script.

**Gain arithmetic.**
- Hit path = ASR + embed 35 ms + lookup 0.03 ms (2,000 keys) + watermark ~100 ms + file write. That gives
  **en ≈ 0.44 + 0.15 ≈ 0.6 s** (0.45 s with idea 3) vs 1.94 s, and **hi ≈ 0.7 s** (after idea 3) vs 3.79 s.
- Expected value = hit rate × saving: at an assumed 25 % hit rate on clinic-style FAQ traffic, the mean first answer
  drops by about 0.35 s en / 0.6 s hi. The hit rate is unknown; measure it on stored `conversation_turns` before
  building the full feature.

**Risks.**
- Wrong answers delivered with confidence. Mitigations: owner approval, per-language keys, the entity check, logging
  every hit for Analytics review, and an automatic expiry when a document changes.
- Answers that depend on time ("are you open now?") must never be cached; mark them by a `now|today|abhi|अभी` check.

### 2.6 Memory: bring the Mac out of swap (A2, A1-cap, A3, A4-cap, A9)

The **swap of 3–5 GB** comes from the components below. The items are cheap and independent.

| Component | Today | After | How | Evidence |
|---|---|---|---|---|
| BGE-M3 (CPU PyTorch) | fp32, RSS ~1.95 GB | fp16 ~1.0 GB | `SentenceTransformer(..., model_kwargs={"dtype": torch.float16})` in `retrieval.load_sentence_transformer`, env `EMBEDDING_DTYPE` | [measured] identical hit@1/@3 (0.689/0.933 dense, 0.756/0.978 hybrid); encode 35 ms vs 94 ms |
| mlx_lm prompt cache | ≤10 entries, no byte cap | ≤600 MB | `--prompt-cache-bytes` | arithmetic in finding 3 |
| Whisper | small + turbo resident (~2.1 GB peak measured in my scratch run of turbo alone) | turbo only | idea A3 | [measured] |
| Kokoro transient peak | up to ~3.3 GB for a 220-char sentence | ≤1.8 GB | 110-char cap | [measured] 188 MB per audio-second |
| Whisper turbo | fp16 ~1.6 GB | 4-bit ~0.5 GB | re-save `mlx-community/whisper-large-v3-turbo-4bit` as `weights.safetensors` + config for mlx-whisper 0.4.3, or `mlx_whisper.convert -q` | [estimate]; the 4-bit snapshot failed to load as-is (section 3.5) |

Worst-case total: about −1 GB (BGE) −0.5 GB (small) −1 to −1.5 GB (Kokoro peak) − up to 3.7 GB (prompt cache) −1 GB
(turbo 4-bit). The 4-bit turbo needs a Hindi WER check, because quantising Whisper's encoder hurts low-resource
languages more [unverified].

---

## 3. Experiments I ran

All scripts are in the session scratchpad (`opus_research/`); none touched the repo or restarted a service. Requests to
the live `mlx_lm server` (port 49356) and ASR worker (port 49365) were sequential and few.

### 3.1 Prompt layout and prefill on the live mlx_lm server (`llm_cache_exp.py`, `cache_probe.py`)

Setup: the clinic agent's real rules text and `tools.tools_prompt()` with the 4 appointment tools. Corpus = the three
template documents chunked like `backend.chunk_text` (500 words, 450 step). Six questions, history growing,
`max_tokens=1`.

| Layout | prompt tokens | cached | TTFT (median of turns 2–6) |
|---|---|---|---|
| A: today (CONTEXT + tools in system, 4 × 500 words) | 3,679–4,019 | 0 then 324 | **7.56 s** |
| B: static system, CONTEXT in user msg, 4 × 500 words | same | 324* | 7.51 s |
| Probe: identical system (rules + tools), new question | 904 | 893 | **0.11 s** |
| Probe: + history turn | 923 | 904 | 0.13 s |
| C: static system + 2 × 120-word chunks in user msg | 1,248–1,329 | 893–895 | **0.74–0.95 s** |

\*In run B the trie matched the interleaved A entries instead of the previous B entry: A and B were sent alternately to
one server, so this row is an artefact of my test, not of layout B. The probes confirm the cache works when the system
prefix is stable.

Derived: prefill ≈ 583 new tokens in 1.07 s → **~500 tok/s**. Decode ≈ **45 tok/s** (Hindi 120 tokens in 2.65 s).
Generated words per token: en 1.62, hi 2.4 tokens/word.

### 3.2 Kokoro-82M (MLX, bf16) latency and memory vs length (`kokoro_len.py`, `kokoro_peak.py`)

Separate scratch process, same `synthesize()` as the worker, warm, 3 repeats (p50).

| text | audio | synth | RTF | MLX peak |
|---|---|---|---|---|
| en 3 words | 1.68 s | 139 ms | 0.083 | 761 MB |
| en 5 words | 1.95 s | 161 ms | 0.082 | – |
| en 13 words | 4.62 s | 350 ms | 0.076 | 1,393 MB |
| en 23 words | 8.43 s | 609 ms | 0.072 | 2,024 MB |
| hi 3 words | 1.98 s | 153 ms | 0.078 | – |
| hi 5 words | 2.67 s | 203 ms | 0.076 | – |
| hi 15 words | 6.28 s | 455 ms | 0.073 | – |
| hi 25 words | 9.90 s | 707 ms | 0.071 | – |

Fit: en 22 ms + 69.6 ms per audio-second; hi 14 ms + 70.0 ms/s. Memory: 441 MB + 188 MB per audio-second (weights
328 MB). Conclusion: no meaningful fixed overhead, so first-phrase chunking helps almost in proportion, and short chunks
also cap the memory spike.

### 3.3 AudioSeal watermark cost (`wm_time.py`, `wm_mps.py`)

`provenance.watermark_ex` on CPU (4 threads): **43 ms (2 s audio), 103 ms (5 s), 213 ms (10 s)**, about 21 ms per
audio-second; resampling is negligible. The same generator on PyTorch MPS took 28.6 / 69.5 ms for 2 / 5 s, only
**1.5× faster**, and it would compete with MLX for the GPU, so I recommend against moving it to MPS.

The first answer sentence currently pays ~100 ms of watermarking; first-phrase chunking reduces this to ~40 ms. The
stitched-reply re-watermark (~170 ms for 8 s) happens before `done`, not before first audio.

### 3.4 Tokenizer fertility (`fertility.py`; tokenizers only, from the HF cache)

Tokens per whitespace word on the bench's real questions and replies:

| set | Llama-3.2 | Gemma-4 (E4B) | Qwen2/3 vocab (from Qwen3-TTS) | XLM-R (BGE-M3) | Whisper v3 |
|---|---|---|---|---|---|
| English questions | 1.18 | 1.18 | 1.19 | 1.29 | 1.20 |
| Hindi questions | **2.68** | 1.36 | **4.47** | 1.46 | **4.97** |
| English replies | 1.39 | 1.50 | 1.49 | 1.34 | 1.39 |
| Hindi replies | **2.31** | 1.59 | **4.13** | 1.45 | **4.34** |
| Hinglish (Roman) | 1.79 | 1.50 | 1.82 | 1.44 | 1.82 |

Implications:
- Hindi costs Llama-3.2 **1.7–2.3×** the tokens of English, in both prefill and decode. This explains much of the
  hi/en LLM gap.
- Whisper decodes **3.5×** more tokens for Hindi. This is the case for CTC/RNNT Hindi ASR (Nemotron, IndicConformer).
- Gemma-4's tokenizer is 1.5–1.7× more efficient on Hindi than Llama's. The model is larger, so the net speed is
  unknown; test it with the Gemma 4 E4B spike already proposed in `IMPROVEMENTS_RESEARCH.md`.
- If Groq's `qwen/qwen3.8-27b` uses the 151k Qwen vocabulary (unverified), Hindi replies cost ~3× the tokens against
  the free-tier TPM.

### 3.5 Whisper: live worker strategies and trimmed encoder context (`asr_probe.py`, `whisper_ctx.py`)

Clips: Kokoro-synthesised questions (en 5.78 s, hi 6.20 s, resampled to 16 kHz).

Live worker (p50 of 3):

| clip | auto | accurate | auto + language hint | fast (small) |
|---|---|---|---|---|
| hi | 1,381 ms | 1,932 ms | 1,184 ms | 787 ms (misspellings) |
| en | 433 ms | 1,633 ms | 284 ms | 433 ms |

Readings:
- The turbo encoder costs ~750 ms. `accurate` without a hint encodes twice (+750 ms).
- The small language-ID pass costs ~150 ms.

Scratch process (turbo fp16), positional embedding sliced: see the table in 2.2. Transcripts were identical at 30, 10
and 8 s contexts for both clips. Peak MLX memory 2.15 GB.

The 4-bit turbo in the cache did not load in mlx-whisper 0.4.3 (it ships `model.safetensors`; the loader expects
`weights.*`).

### 3.6 BGE-M3: precision, truncation, quantisation and a semantic-cache test (`emb_exp.py`)

Setup: BGE-M3 **fp16** on CPU, the bench's 45 questions and 40-word chunks, `services.retrieval.hybrid_order` for RRF.

| vectors | dense hit@1 / @3 | hybrid hit@1 / @3 |
|---|---|---|
| fp 1024 (fp16 model) | 0.689 / 0.933 | 0.756 / 0.978 |
| fp32 model (earlier bench) | 0.689 / 0.933 | 0.756 / 0.978 |
| truncated 512 | 0.711 / 0.956 | 0.756 / 0.978 |
| truncated 256 | 0.667 / 0.956 | 0.778 / 0.978 |
| truncated 128 | 0.600 / 0.844 | 0.733 / 0.911 |
| int8 docs | 0.689 / 0.933 | 0.756 / 0.978 |
| binary (sign) | 0.756 / 0.978 | 0.800 / 0.978 |

On 45 questions the differences of ±1–2 questions are noise. The conclusions: **fp16 weights lose nothing**; 512/256-d
and binary vectors are safe; 128-d is not.

Semantic cache (English questions as keys, 32 Hindi/Hinglish queries, only 5 of which have an English twin with the
same gold):

| threshold | served | correct | wrong |
|---|---|---|---|
| 0.60 | 10 | 4 | **6** |
| 0.65 | 6 | 4 | **2** |
| 0.70 | 4 | 4 | 0 |
| 0.75 | 3 | 3 | 0 |

Correct-match similarities: 0.574, 0.713, 0.78, 0.859, 0.895. Highest wrong: 0.685. Lookup over 2,000 keys: 0.026 ms.

### 3.7 What I could not run

- Smart Turn (it needs a model download, which requires your approval).
- Nemotron 3.5 (needs a download and NeMo).
- Anything on the RTX 3050 Ti (no Windows machine available).
- A WER evaluation on real speech (no real Hindi corpus in the repo; `trial-audio/` holds only Indic cloning
  references).

---

## 4. Ideas rejected (or parked) and why

| Idea | Verdict | Reason |
|---|---|---|
| Matryoshka/truncated or binary BGE-M3 vectors | Parked | Safe to 256-d (3.6), but the index is KB-scale (15–200 chunks per agent). The memory is in the 2.2 GB of *weights*, not the vectors. Revisit above ~100k chunks. |
| Zero-copy shared memory between workers | Reject | A 5 s 24 kHz clip is ~240 KB over loopback HTTP, about 1–3 ms, which is noise against 350 ms of synthesis. ASR already passes a file path. |
| AudioSeal on GPU or fused | Reject | MPS is only 1.5× faster (3.3) and would compete with MLX; shorter first chunks give the same saving. |
| Speculative decoding (1B draft) | Agree it is low value | At ~45 tok/s decode with 20–90-token replies, prefill and endpointing dominate. |
| `--kv-bits` on the Mac | Agree it is low value | Short prompts. **But** on the 4 GB GPU, q8_0 KV is a VRAM win (B1). |
| KV precompute per chunk (CacheBlend, TurboRAG, APE, Block-Attention) | Reject for now | Needs vLLM/LMCache or fine-tuning; no MLX implementation found [secondary]. With small chunks the remaining prefill is ~0.5 s. |
| Grammar-constrained JSON for tools (xgrammar/llguidance/outlines) | Parked | No native mlx-lm support [secondary]. Small models show a "constraint tax" in recent work (arXiv 2605.26128, 2609.23742 [secondary]). Stream sniffing (2.3) removes most of the latency cost. |
| Kokoro on CoreML/ANE | Reject for now | The public port is Swift, ships English G2P only, and runs most stages on CPU+GPU, so there is little relief from GPU contention ([card](https://huggingface.co/mattmireles/kokoro-coreml) [verified]). FluidAudio measures CoreML ≈ MLX speed on M4 Pro (23.2× vs 23.8× RTFx, [docs](https://docs.fluidinference.com/tts/kokoro) [secondary]). |
| Retrieval on partial transcripts | Parked | No partials exist until streaming ASR (idea A10/B8). No paper measures the gain [secondary]. |
| CUDA MPS / multi-stream overlap on the 3050 Ti | Reject | MPS is documented for Linux/QNX (NVIDIA [MPS docs](https://docs.nvidia.com/deploy/mps/); Windows not listed). In one turn ASR, LLM and TTS are mostly sequential, so there is little to overlap. |
| fp8 / TensorRT-LLM on the 3050 Ti | Reject | No fp8 on Ampere. TensorRT-LLM's Windows support and 4 GB footprint are a poor fit next to llama.cpp Q4_K_M. |
| ExLlamaV3 on 4 GB | Reject for now | Described as an early preview with Ampere performance issues ([repo](https://github.com/turboderp-org/exllamav3) [secondary]). |
| Full-duplex models (Moshi, Human-1 Hindi Moshi) | Reject | 7.6–8B parameters; Moshi is English-only; Human-1 has no measured latency and does not fit 4 GB [secondary]. |
| Distilling the 3B into a per-agent model | Reject | Days of work per template, and it needs eval data that does not exist yet. The answer cache gets most of the gain for FAQ traffic. |
| Per-chunk TTS prefix caching ("prefix tree of sentence starts") | Reject | Kokoro is non-autoregressive with sentence-level prosody, so concatenating a cached prefix with a new suffix gives audible seams. Cache whole approved answers instead (2.5). |
| IndicXlit for Hinglish → Devanagari | Parked | MIT ([PyPI](https://pypi.org/project/ai4bharat-transliteration/) [secondary]), but it depends on fairseq, which is hard to install on Python 3.12. Pipelining the existing LLM call (A7) is simpler. |
| Groq prompt caching | Not available for our model | Only `openai/gpt-oss-20b`/`120b`/`safeguard-20b` ([docs](https://console.groq.com/docs/prompt-caching) [secondary]). The layout fix (2.1) still cuts Groq TPM use. |
| mlx-lm upgrade | Caution | 0.31.0 was **yanked** for batched KV cross-contamination ([PyPI](https://pypi.org/project/mlx-lm/) [secondary]). Stay on 0.32.0 (installed) or newer. |

---

## 5. Suggested 2-week plan (one developer, ~6 productive hours a day)

**Week 1: cheap, high-confidence wins (Mac first, the shared code helps Windows too)**

| Day | Work | Exit check |
|---|---|---|
| 1 | A2 BGE-M3 fp16 behind `EMBEDDING_DTYPE`; A1 `--prompt-cache-bytes` / `--prompt-cache-size` | `retrieval_eval.py` unchanged; RSS −1 GB; swap after a 30-turn soak lower than today |
| 1–2 | A1 prompt layout + token-budgeted ~120-word chunks + top-3; re-ingest | `latency.py --stream` on the clinic template **and** a new 3-document agent; `cached_tokens` > 80 % of the prompt from turn 2; Evaluation Lab attack score not worse |
| 3–4 | A4 streaming of tool-agent replies, booking-ID guard per sentence, first-phrase chunking, `MAX_CHARS` 110 | bench first answer p50 en ≤ 1.7 s, hi ≤ 3.0 s; `tests/` for the sniffing and splitter edge cases; 20-reply Hindi listening check |
| 5 | A7 Hinglish transliteration pipelining; A5 speech-start pre-warm endpoint | Hinglish turn bench (add 10 Hinglish questions to `bench/latency.py`) |

**Week 2: ASR, endpointing, the cache spike and the Windows bring-up**

| Day | Work | Exit check |
|---|---|---|
| 6–7 | A3 turbo-only trimmed-context ASR with fallback guard; record 40 real clips (en/hi/Hinglish, quiet/noisy, one 8 kHz phone set) | WER (bucket) ≤ WER (30 s) + 0.5 pt per language; zero new hallucinations; hi ASR p50 ≤ 0.7 s |
| 8–9 | A6 speculative endpointing (flagged), Smart Turn on the server (after you approve the model download); 100-clip Hindi/Hinglish false-cutoff set | False cutoffs ≤ 2 % at the chosen threshold; median commit time ≤ 350 ms after end of speech |
| 9 | B1 Windows bring-up: llama.cpp `-np 1 -c 4096 -fa on -ctk q8_0 -ctv q8_0`; sysmem fallback off for the app executables; record `nvidia-smi` VRAM per component | Full turn without OOM or shared-memory use; per-stage timings recorded |
| 10 | A8 spike only: measure the would-be hit rate on stored `conversation_turns` with per-script keys and the entity check (no serving yet) | If the projected hit rate is ≥ 20 % with zero wrong hits in the review, schedule the full feature (approval UI and pre-synthesis) |
| Buffer | A9 turbo 4-bit conversion + Hindi WER; B6 Kokoro ONNX bucketed export on CUDA | Keep only if both quality and speed gates pass |

Defer to after week 2: Nemotron 3.5 streaming ASR (A10/B8, the right long-term answer for the phone channel and Hindi,
but it needs a runtime decision), Whisper paging on the 3050 Ti (B2) once B1's VRAM numbers are known, and
TensorRT-EP Kokoro.

### Projected effect on the bench (local models, p50, before speculative endpointing) [estimate]

| | today | after week 1 | after week 2 |
|---|---|---|---|
| First answer audio, en | 1.94 s | ~1.6 s | ~1.5 s |
| First answer audio, hi | 3.79 s | ~2.9 s | ~2.2 s |
| Caller's perceived wait after speaking, en (adds the 700 ms endpoint) | ~2.6 s | ~2.3 s | ~1.8 s |
| RAM pressure (swap during the 30-turn bench) | 3–5 GB | ~1–2 GB | ~0–1 GB |

These are arithmetic projections from the stage measurements above, not measured end to end. Re-run
`bench/latency.py --stream` after each step and update `docs/RESULTS.md`.
