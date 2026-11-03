# B4 (deep re-research): ANE / GPU / CPU scheduling on Apple Silicon

Date of research: 2026-10-17. Target: M4 base, 16 GB, macOS 15.8.1. Tags: [V] = number read on the cited page; [S] = secondary/blog source, treat as rough; [U] = unverified or my own estimate.

## Verdict (read this first)

Do NOT build a heterogeneous ANE/GPU/CPU scheduler now. Evidence for it is thin and mostly for other chips. Cheaper, proven wins come first:
1. Trim the Whisper encoder window (or use short-window-tolerant weights) and stream/partial-decode ASR. whisper.cpp's own commit message claims `audio_ctx=512` is ~3x faster for ~10 s of audio, at some quality cost [V, source 9]. This is a 3x on the biggest local-ASR cost with no new runtime.
2. Only if ASR is still the bottleneck after (1): try a WhisperKit/Core ML (ANE) path as a separate helper process, benchmarked on your own M4. No published M4 number exists for it (see gaps).
3. Keep LLM, TTS and embeddings on MLX (GPU). ANE LLM decode is 2-5x slower than GPU on tokens/s [S]. The ANE's selling point is power and freeing the GPU, not speed.

Novelty: a per-turn scheduler across ANE+GPU for a voice agent is not documented in any source I found [U: absence of evidence]. It may be publishable, but only after you measure real overlap gains, which no one has published for M4.

## Findings per gap, with sources

### Gap 1: WhisperKit vs mlx-whisper vs whisper.cpp on M4

No source gives all three on M4 or M4 base for 3-8 s clips. Pieces that exist:

- mlx-whisper vs whisper.cpp, large-v3-turbo, long MLK speech (duration and chip NOT stated): whisper.cpp 26.704 s +/- 0.625, mlx_whisper 13.135 s +/- 0.280, ratio 2.03x. [V] https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html (whisper.cpp ran its default build; unclear whether Core ML was enabled. Chip unknown, so M4 relevance is [U].)
- MLX Whisper on M4 Max 128 GB, 46.7 s of synthetic speech: small 0.47 s (98x), large-v3 1.96 s (24x), large-v3-turbo 0.86 s (55x). [V] https://vllm-mlx.is-a.dev/benchmarks/audio/ (third-party project, synthetic audio, M4 Max has 40 GPU cores and 546 GB/s, so base M4 will be much slower; my guess is 3-5x slower, [U]).
- WhisperKit (ANE) published by Argmax: leaderboard rows exist only for M3 Max and M2 Pro, not M4. M2 Pro: whisper-large-v3-v20240930 turbo "speed" 12.85, whisper-small 18.73 (unit not defined on the page, likely x real-time on 10-min files). [V] https://huggingface.co/spaces/argmaxinc/whisperkit-benchmarks/commit/255a6321aaad7ed5f02cb98dfcdf3458c6e5806a . A search snippet put large-v3-turbo on M3 Max at speed 4.4 / 14.94 tok/s in one row [S, not independently re-read]; rows for the same model vary 3.8 to 12.9 by compression variant, so do not quote one number.
- WhisperKit paper (ICML 2025 workshop), M3 Max ANE: large-v3-turbo encoder 612 ms original (30 s window) vs 218 ms (d750), 125 ms (d500), 50 ms (d250) with a reduced-window variant; hypothesis latency ~0.45 s, confirmed ~1.7 s; model 1.6 GB FP16 -> 0.6 GB compressed; target under 2 GB RAM. [V] https://arxiv.org/html/2507.10860v1 . This is the strongest hard evidence for "trim the window": it is the same lever, applied inside Argmax's own stack.
- whisper.cpp Core ML encoder: README claims "more than x3 faster compared with CPU-only" (no chip) [V] https://github.com/ggml-org/whisper.cpp . Only encode timings in issue #89 are M1 Ultra (small 114.99 ms default, 148.01 ms --no-gpu; large-v3-turbo Q8_0 ~618-635 ms) [V, relayed by fetch tool] https://github.com/ggml-org/whisper.cpp/issues/89 . No M4 rows there.
- Rough marketing-style M4 figure: ~14-18x real time for large-v3-turbo whisper.cpp Metal [S, weak] https://www.promptquorum.com/local-llms/apple-silicon-whisper-metal-benchmark .
- No Hindi/Indic benchmark of ANE vs GPU Whisper found [U].

Takeaway: the 30 s window costs the same regardless of clip length. For a 3-8 s clip you pay for 30 s of encoder. Window trimming attacks that directly. Whisper decoders were trained on 30 s windows; stock models can loop at short windows (reported: base model at 5 s window CER 10.4 on one clip; FUTO "ACFT" fine-tunes claim ~8x encoder speedup at 5 s) [S, one author's eval] https://huggingface.co/litert-community/whisper-acft . whisper.cpp commit notes quality drops with audio_ctx [V, via search snippet] https://app.semanticdiff.com/gh/ggerganov/whisper.cpp/commit/1b7a7df793b76a78b3c11a309b1edfd195864087 . Hindi/Hinglish effect of trimming is unmeasured [U]; test on your own clips.

### Gap 2: BGE-M3 (568M) on Core ML/ANE vs MLX/MPS

No benchmark found. Nothing published on BGE-M3 on ANE [V: searched, absent]. Related:
- Generic claim Core ML ~1.5-2x MPS for supported models [S, unsourced] https://dev.to/shaktitiwari/apple-silicon-ml-execution-paths-explained-mps-coreml-mlx-and-the-neural-engine-486o
- ANE needs fixed shapes, FP16, limited ops; XLM-R 8192-token, 250k-vocab embedding (about 256M params of embeddings) makes conversion heavy [U, my reasoning]. FP16 overflow is a known ANE problem (ANEMLL README documents Gemma 3 NaN/Inf from the 65,504 limit) [V] https://github.com/Anemll/Anemll
- Estimate for a retrieval query (about 20 tokens): single-digit to ~20 ms on MLX/GPU on M4 [U]. Embedding is not your bottleneck; ANE gain cannot exceed ~20 ms per turn.

### Gap 3: ANEMLL / Core ML LLM tokens/s on M4

- ANEMLL repo: no tokens/s table; supports Llama 3.1/3.2 1B/8B, Qwen 3 0.6B/1.7B/8B, Qwen 2.5, Gemma 3 270M/1B/4B; context 512-4096; LUT4/LUT6 quality "fairly low"; first load slow. [V] https://github.com/Anemll/Anemll . Not on the list: Llama-3.2-3B, Qwen3.8-27B-class, Gemma-4 E4B [V: absent from the table].
- anemll-bench measures only ANE memory bandwidth: M4 ANE 64 GB/s of 120 GB/s system (53%); M4 Pro 126/273; M3 63/100. [V] https://github.com/Anemll/anemll-bench
- Secondary: Llama 3.2 1B on ANE 47-62 tok/s vs ~204 tok/s on MLX (M4 Max); DeepSeek-R1-8B ~9.3 vs ~50; ANE power ~2 W vs GPU ~20 W; ANE memory ~500 MB vs ~8 GB for the 8B case [S] https://insiderllm.com/guides/apple-neural-engine-llm-inference/ . Another post: Llama 3.2 1B 10.0 -> 24.3 tok/s on base M3 after a DMA fix; Qwen3-8B 2.97 tok/s [S] https://runaihome.com/blog/apple-neural-engine-llm-inference-2026/ . These two blogs disagree (47-62 vs 24 tok/s); treat both as unverified. Community MLX M4 (10c) Llama-3.2-1B ~96-111 tok/s [S, omlx.ai].
- Conclusion: for the 3B local fallback, ANE is slower and unsupported. No benefit.

### Gap 4: Core ML + MLX concurrency; "macOS 26.3 fix"

- MLX has no ANE path (maintainer: closed-source API) [V] https://upd.dev/ml-explore/mlx/issues/18 (mirror of ml-explore/mlx #18; 2023 thread, may be dated). Core ML is the only supported way to reach the ANE; the ANE paper says "Core ML remains the path for shipped software" [V] https://arxiv.org/abs/2606.22283 .
- Two separate processes (one MLX, one Core ML) is the standard architecture and works in principle (separate frameworks, separate address spaces). I found no primary-source test of overlap or slowdown [U].
- ANE concurrency: an Apple Developer Forums post (iOS 27.0.1, iPhone 17 Pro Max) says concurrent Core ML ANE predictions are now serialized: latency grows ~N x single call (~2.8 ms); same code still overlaps on macOS 26.4.1 M3 Pro. No Apple reply. [V] https://developer.apple.com/forums/thread/850110 . ANE DMA engines also serialize internally (blog, base M3) [S] runaihome link above.
- "macOS 26.3 fix": I could NOT verify any fix. Searches found no Apple, MLX or Core ML statement tying 26.3 to MLX/Core ML concurrency. Reports show GPU/IOGPU panics persisting on 26.4 and 26.5.2 [S, via search snippets]; Core ML GPU-path output corruption after 26.1, with CPU/ANE unaffected; Ultralytics moved Core ML to ANE-only to dodge a GPU-path crash [S]. Mark the 26.3 claim FALSE/UNVERIFIED. Irrelevant anyway: your Mac runs 15.8.1 and cannot take 26.x guidance as given. Core ML GPU regressions on 26.x are another reason not to depend on them.
- WhisperKit needs macOS 14+; TTSKit needs macOS 15+ [V] https://github.com/argmaxinc/WhisperKit (README fetched as argmax-oss-swift).

### Gap 5: Memory accounting

- No primary source says whether a compiled Core ML model's ANE weights count in wired memory the way MLX Metal buffers do [U: searched, absent]. All units share one LPDDR pool, so the total is additive in practice [S].
- WhisperKit paper: large-v3-turbo 1.6 GB FP16, 0.6 GB compressed; peak target under 2 GB. [V] arXiv 2507.10860. MLX turbo in 4-bit or 8-bit is of similar size [U].
- ANEMLL 8B ~500 MB vs ~8 GB on GPU [S] (strange for 8B; likely a transient-footprint measure; low confidence).
- Practical: swapping MLX Whisper (small + turbo) for one compressed WhisperKit turbo could save roughly 1-2 GB wired [U estimate]. A separate process adds Swift runtime overhead (small).

### Gap 6: Does the ANE reduce end-to-end voice-agent latency?

No evidence found, for any system [V: searched, absent]. WhisperKit's 0.45 s hypothesis latency is a streaming ASR number on M3 Max, not an agent number. FluidAudio (Core ML) reports ~190x RTFx for Parakeet batch ASR on M4 Pro and Kokoro ANE v3 36-54 ms median for 3-3.7 s audio on M5 Pro [V] https://github.com/FluidInference/FluidAudio . Caveats: Parakeet TDT v3 covers 25 European languages, no Hindi; Kokoro there lists Hindi. M5 Pro, not M4. Not BonClay's TTS stack (Qwen3-TTS, Chatterbox, Indic-Speak).

## Placement plan (M4 base 16 GB); ms are estimates [U] unless noted

| Stage | Today | Recommended | Est. gain | Effort | Feasibility |
|---|---|---|---|---|---|
| Silero VAD | browser | keep | 0 | 0 | done |
| ASR (3-8 s clip) | MLX Whisper, 30 s window | Step 1: shorten window / trim, streaming partials. Step 2 (optional): WhisperKit turbo via ANE helper | Step 1: up to ~3x encoder (source 9 claim); Step 2: unknown on M4, M3 Max encoder 612 -> 50-218 ms with reduced window [V paper] | Step 1: 1-3 days incl. Hindi eval; Step 2: 1-2 weeks | Step 1 high; Step 2 medium |
| Retrieval embedding BGE-M3 | PyTorch | MLX or ONNX; skip ANE | under 20 ms | 1-2 days (MLX), ANE conversion unproven | ANE: low |
| LLM local 3B | mlx_lm | keep on GPU | none; ANE slower | 0 | ANE: not feasible |
| LLM cloud | Groq | keep | n/a | 0 | n/a |
| TTS Kokoro | MLX | keep (Core ML Kokoro exists via FluidAudio on newer chips, [V]) | possibly 0.1-0.3 s if first-audio gets faster; unmeasured on M4 | 1 week | medium |
| Qwen3-TTS, Chatterbox | PyTorch/MLX | keep | none | 0 | low |

Realistic budget: first answer 2.23 s today. A 0.3-0.8 s saving in the ASR stage is plausible from window trimming alone [U]; an ANE scheduler adds a further unknown, probably under 0.3 s, and only if GPU is the bottleneck during overlapping turns.

## Risks

- ANE is a different process and runtime; concurrent ANE calls may serialize on newer OS (forum post, iOS 27).
- Core ML GPU path regressions on macOS 26.1+; stay on CPU+ANE compute units.
- First-run Core ML compile/load is slow (whisper.cpp README; ANEMLL README); model cold start must be hidden at app launch.
- Hindi quality unmeasured for compressed WhisperKit variants (OD-MBP) and for reduced windows.
- Memory pressure on 16 GB remains dominated by LLM + TTS, not ASR.
- Two-process IPC complexity (Swift helper, packaging, Windows has no equivalent: the RTX 3050 Ti path cannot share it).
- Many sources above are blogs of unknown rigor; some numbers conflict.

## Gaps still open (do not trust without measuring)

1. Any M4 base WhisperKit turbo/small timings on 3-8 s clips. Run `whisperkit-cli` and `mlx_whisper` and `whisper-cli` (Core ML on and off) on the same 40 Hindi/Hinglish clips; 1-day experiment.
2. BGE-M3 on ANE: no data at all.
3. Overlap test: run MLX LLM decode while a Core ML ANE Whisper encode runs; measure each side's slowdown. No published numbers; this single test decides whether a scheduler is worth anything.
4. Whether the 26.3 claim has a real source (none found).

## Effort summary

- Window trim + eval: 1-3 days.
- Streaming ASR (LocalAgreement-style partials, as in WhisperKit paper): 1-2 weeks.
- WhisperKit helper process: 1-2 weeks.
- Full scheduler: 4+ weeks, evidence not there.

## Source list

1. https://arxiv.org/html/2507.10860v1 (WhisperKit paper)
2. https://github.com/argmaxinc/WhisperKit
3. https://huggingface.co/spaces/argmaxinc/whisperkit-benchmarks/commit/255a6321aaad7ed5f02cb98dfcdf3458c6e5806a
4. https://github.com/ggml-org/whisper.cpp/issues/89 and https://github.com/ggml-org/whisper.cpp
5. https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html
6. https://vllm-mlx.is-a.dev/benchmarks/audio/
7. https://github.com/Anemll/Anemll and https://github.com/Anemll/anemll-bench
8. https://insiderllm.com/guides/apple-neural-engine-llm-inference/ ; https://runaihome.com/blog/apple-neural-engine-llm-inference-2026/
9. https://app.semanticdiff.com/gh/ggerganov/whisper.cpp/commit/1b7a7df793b76a78b3c11a309b1edfd195864087 (audio_ctx commit)
10. https://huggingface.co/litert-community/whisper-acft
11. https://developer.apple.com/forums/thread/850110
12. https://arxiv.org/abs/2606.22283
13. https://github.com/FluidInference/FluidAudio
14. https://upd.dev/ml-explore/mlx/issues/18
15. https://dev.to/shaktitiwari/apple-silicon-ml-execution-paths-explained-mps-coreml-mlx-and-the-neural-engine-486o
