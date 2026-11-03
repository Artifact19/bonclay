# B5. Cloud teacher, per-agent LoRA student on the laptop (deep re-research)

Date: 2026-10-17. Method: fetched primary pages and quoted numbers; anything I could not open or that is my own arithmetic is marked
**[UNVERIFIED]** or **[ESTIMATE]**. Note: WebFetch summarises pages with a small model, so quotes below are as returned by that tool, not
read raw. Items marked **[SEARCH-ONLY]** came from a search snippet, not an opened page.

## 0. Verdict (read this first)

**Conditional go, but narrow.** Build a "style + grounding + tool-format" LoRA for ONE agent (the HOD office agent) as a 2-week experiment,
not a platform feature. Reasons:

- The mechanism is real: small LoRAs on 3B models trained on a few hundred to a few thousand teacher-style examples give large gains on
  format-like skills (Hinglish fluency, tool-call syntax). Evidence in section 3.
- The pain point is real: the current server (`backend/services/llm.py`) cannot swap adapters per request without a full model reload
  (section 2). So "per-agent adapter" means either a server restart per agent switch, or a fused model per agent, or one server per agent.
- It does NOT fix knowledge. Offline quality on hard Indian-language tasks will be limited by the 3B base; the better lever for the
  49% vs 84% gap may be swapping the base model (Gemma 4 E4B scored 84%) before any LoRA. Do that first (cheap, no training).
- Safety erodes measurably after LoRA (section 4); you must re-run the 60-case attack suite on every adapter and mix refusal data in.
- No published result shows "approaches teacher quality" for a 3B on Indian-language open-domain chat. Expect to close part of the gap on
  narrow in-domain tasks only.

## 1. How the local model is loaded today (read from `backend/services/llm.py`)

- Provider order `groq,openai,mlx`. `_MlxServer.ensure()` spawns `python parent_watch.py --run-module mlx_lm server --model <LOCAL_LLM_MODEL>
  --host 127.0.0.1 --port <free> --log-level WARNING` and keeps it warm. Default model `mlx-community/Llama-3.2-3B-Instruct-4bit`.
- No `--adapter-path` is passed. Requests go through `_openai_chat(base, _mlx.model, ...)` with the model name fixed to the base repo id.
- A `request_lock` serialises requests ("the local server answers one request at a time").
- Implication: to serve an adapter you can add `--adapter-path` at spawn, or send a per-request `"adapters"` field (section 2).

## 2. Gap 2: can mlx_lm serve different adapters per request without reloading the base?

Source: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/server.py (main branch, fetched today).

Findings (from the fetch):
- The request body accepts `"adapters"` (a string path): `self.adapter = self.body.get("adapters", None)`, validated as string-or-null.
- Cache key: `model_key = (model_path, adapter_path, draft_model_path); if self.model_key != model_key: self._load(*model_key)`.
- `_load` calls `self.reset()`, `gc.collect()`, `mx.clear_cache()` and then a full `load(...)`. **So a different adapter triggers a full
  reload of base + adapter. There is no hot-swap.**
- `--adapter-path` applies only to the default model (`_adapter_map["default_model"]`).
- Prompt caches are keyed by `model_key`, so a changed adapter also loses the prompt cache. Batching requires the same
  model/adapter; a different adapter must wait for the batch to drain.
- Flags seen: `--model`, `--adapter-path`, `--draft-model`, `--num-draft-tokens` (3), `--prompt-cache-size` (10), `--decode-concurrency` (32),
  `--prompt-concurrency` (8), `--kv-bits` (disables batching).

Lower level, https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/tuner/utils.py: `load_adapters` reads `adapter_config.json`, calls
`linear_to_lora_layers` to wrap the last N layers, validates tensor names (`lora_a`, `lora_b`, `m`) and shapes, then
`model.load_weights(list(weights.items()), strict=False)`. Whether you can re-run `load_weights` with a second adapter's weights on an
already-wrapped model to swap in-process is **[UNVERIFIED]** (the tool said the page does not show it; it is plausible because LoRA params
are ordinary module parameters and shapes match for the same rank/layers, but I did not run it). That would be a ~20-line custom patch or a
small custom server instead of `mlx_lm server`.

Options, cheapest first:
1. **Fuse per agent** (`mlx_lm.fuse`, documented in https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/LORA.md) and set
   `LOCAL_LLM_MODEL` to the fused dir when that agent is the active one. Costs ~2 GB disk per agent for a 3B 4-bit **[ESTIMATE]**
   (the D-Harshith repo reports a 28 MB adapter and 2.5 GB 4-bit peak). Caveat: fusing LoRA into a 4-bit model needs de-quantise/re-quantise
   behaviour; mlx_lm.fuse has `--dequantize`; quality effect **[UNVERIFIED]**.
2. **One `mlx_lm server` per agent with `--adapter-path`**, started lazily (the code already supports a single process; make `_MlxServer`
   keyed by agent). Each holds ~2-2.5 GB **[ESTIMATE from reported 2.5 GB peak]**, fine for 1-2 agents on 16 GB, not for many.
3. **Accept reload on agent switch.** Reload time for a 3B 4-bit **[UNVERIFIED, I measured nothing; likely a few seconds from page cache]**.
   Fine if one agent is active per session.
4. **llama.cpp route** has true per-request adapters. Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md :
   `--lora`, `--lora-scaled`, `--lora-init-without-apply`, POST `/lora-adapters`, and a per-request `lora` field: "A list of LoRA adapters to be
   applied to this specific request"; caveat quoted: "requests with different LoRA configurations will not be batched together, which may
   result in performance degradation." Requires GGUF base + GGUF LoRA (conversion script, **[UNVERIFIED]** for Llama-3.2 + MLX-trained adapters;
   MLX adapters are safetensors in MLX layout and would need conversion to PEFT then to GGUF). The Windows/CUDA path already suits llama.cpp.

Recommendation: option 2 for the demo (one adapter), option 4 if you ever need many agents.

## 3. Gaps 1, 3, 6: training cost on Apple Silicon and expected gains

### 3a. mlx_lm LoRA speed and memory (real numbers are scarce; I found NO M4 16 GB number)

| Source | Hardware | Model | Result |
|---|---|---|---|
| https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/LORA.md | M1 Max 32 GB | not stated in extract | "about 250 tokens-per-second"; defaults batch 4, 16 layers; QLoRA automatic if model is quantized |
| https://dev.to/rainytechlab/fine-tune-an-llm-on-your-mac-with-mlx-qlora-in-16-seconds-real-run-no-cloud-39ni | M2 Pro 16 GB | Qwen2.5-3B-Instruct 4-bit | 20 iters, batch 1, 16 train examples: ~16 s, peak ~2.5 GB training vs 1.9 GB chat. It/s not reported (20/16 = ~1.25 it/s is my arithmetic; tiny sequences). Smoke test only. |
| https://github.com/D-Harshith/mlx-quantization-lora-eval | Apple M5 16 GB, mlx-lm 0.32.0 | Llama 3.2 3B 4-bit | LoRA rank 8 on last 16 layers, 6.9 M trainable (0.2%), 28 MB adapter, 1,350 examples, ~23 min training, peak 2.5 GB (4-bit) / 2.9 GB (4-bit+LoRA). The reported 61.7 tok/s is not clearly training throughput (page says 4-bit+LoRA "not measured"). Treat speed as ambiguous. |
| https://insiderllm.com/guides/fine-tuning-mac-lora-mlx/ | various (blog, "rough numbers", no methodology) | LoRA on 8B | M2 Pro 16 GB ~200 tok/s; M3 Max ~320; M4 Max ~380. 500 iters on 300 examples "15-45 minutes". Gradient checkpointing adds ~30% time. |

Gaps: no M4 base 16 GB number from a primary source **[UNVERIFIED]**. Quoted GitHub issues #738, #1025, #481 in mlx-examples are about memory leaks / OOM on larger models, not usable as a
benchmark (**[SEARCH-ONLY]**). mlx-lm issue #644 is inference on M4 Mac mini 16 GB (gpt-oss-20b), not training (**[SEARCH-ONLY]**).

**Budget for the M4 16 GB [ESTIMATE, derived, please measure]:**
- Memory: 3B 4-bit QLoRA, batch 1-2, seq <=1024: ~3-4 GB peak (reported 2.5-2.9 GB at short sequences; grows with sequence length and batch). 16 GB is comfortable. Even
  the 8B QLoRA claim (~7 GB, insiderllm) is plausible but untested by me. Keep the backend server stopped while training (it holds ~2 GB and the ASR/TTS models add more).
- Time: if a 3B trains at ~300-500 tok/s on M4 (8B at ~200 on M2 Pro scaled ~2x for 3B; unverified), 1,000 examples x ~500 tokens = 0.5 M tokens = ~15-30 min per epoch, ~1 hour
  for 2-3 epochs. The M5 run above (1,350 examples, 23 min) is the closest real datum. "Overnight" is more than needed; "while the laptop is on the charger for an hour" is realistic.
- Do: `--mask-prompt`, `--num-layers 16`, `--batch-size 2`, `--grad-checkpoint` only if memory pressure goes yellow/red (per LORA.md tips).
- Known pitfall: mlx issue #2616 (**[SEARCH-ONLY]**) says Qwen3 LoRA with default target modules trained only ~0.28% of parameters; check trainable-param count for whichever base you pick.

### 3b. Published gains from distillation / small LoRA

| Skill | Evidence | Numbers | Examples | Source |
|---|---|---|---|---|
| Tool/function calling | AWS Bedrock distillation, Llama 405B teacher -> Llama 3B student, BFCL v2 | Simple 0.478 -> 0.783 (vanilla) -> 0.826 (+augmentation); Multiple 0.586 -> 0.742 -> 0.828 | "about 200" train/test examples per the extract **[UNVERIFIED, suspiciously small; check the post]** | https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-model-distillation-boost-function-calling-accuracy-while-reducing-cost-and-latency/ |
| Tool calling (data recipe) | APIGen-MT xLAM-2 multi-turn accuracy | 3B 56.00%, 1B 43.12%, 8B 69.25% **[SEARCH-ONLY]** | large curated sets | https://arxiv.org/pdf/2504.03601 |
| Structured extraction, 0.6B student | QLoRA r=32, Qwen3-0.6B | summary 75.1 vs 58.2 constrained decoding (+16.8, CI 11.4-22.5); +4.9 over few-shot; closes 58% of gap to teacher; teacher-choice matters: reasoning teacher +18.7, same-size non-reasoning teacher +0.6 (n.s.); faithfulness fell 64.0 -> 55.0 on 22 short articles; classification 55.3 < few-shot 58.5 | 401 | https://arxiv.org/html/2607.08268 |
| Hinglish style | Qwen2.5-3B LoRA on synthetic dialogues, human ratings 1-5 | fluency 2.90 -> 4.10, coherence 3.30 -> 4.70, Hindi usage 3.00 -> 3.60, persona 3.20 -> 4.10. 7B: fluency 3.20 -> 3.90, coherence 4.30 -> 4.60. GPT-4o judge on 100 prompts: tuned 7B 4.1 and tuned 3B 3.86 vs LLaMA 70B 4.6. Human A/B only 10 prompts. | >3,000 dialogues (~5 M tokens) | https://arxiv.org/html/2504.19070 |
| General instruction/format | LIMA | 1,000 curated examples; 43% equal or better than GPT-4 in human eval | 1,000 | https://arxiv.org/abs/2305.11206 |
| Toy classification | Llama 3.2 3B 4-bit LoRA (M5 16 GB) | 67.0% -> 81.7% issue classification; gain almost all from one label (28% -> 81%); bug accuracy fell 92% -> 80% | 1,350 | https://github.com/D-Harshith/mlx-quantization-lora-eval |

Take-aways with honest weighting:
- Format and style are learned with hundreds to low-thousands of examples. Tool-call syntax gains are large (BFCL simple +0.35 absolute) but the teacher was 405B and the base was 3B; your base is already instruct-tuned and
  the task is booking tools, i.e. narrower, so expect gains mainly in "calls the right tool with the right args, doesn't hallucinate a call".
- Teacher choice matters: reasoning-style teacher traces transferred, plain ones did not (2607.08268). Groq qwen/gpt-oss-120b with reasoning hidden may behave like the weak teacher; consider keeping a short rationale in the target.
- Faithfulness can DROP after distillation (55.0 vs 64.0). Grounded QA is the skill most at risk of hallucinating, so measure groundedness explicitly.
- The Hinglish paper is weak evidence for you: same-API synthetic train and eval, GPT-4o judge, 10-prompt human test, no hardware or LoRA config reported. No paper I found compares human vs synthetic Hinglish training data under matched conditions on a real test set **[UNVERIFIED gap]**.
- Nothing I found measures "grounded QA over documents" or Marathi/Tamil/Bengali/Telugu distillation into 1-4B with numbers **[UNVERIFIED / not found]**.

### 3c. Synthetic Hinglish data and judge filtering (gap 6)

- Real code-mixed data tended to beat synthetic (statement in a literature review, **[SEARCH-ONLY]** HinGE context; https://www.researchgate.net/publication/357383651_HinGE_A_Dataset_for_Generation_and_Evaluation_of_Code-Mixed_Hinglish_Text).
- LLMs generate longer but less diverse text than humans: https://arxiv.org/html/2511.01490v1 **[SEARCH-ONLY]**. Enforce diversity: vary persona, script (Roman vs Devanagari), topic, and ask for typos and filler, then dedupe by embedding (you already have BGE-M3).
- Synthetic code-mixed translation rated by 3 bilinguals: adequacy 4.76, fluency 4.44 on 5-point scale (https://arxiv.org/pdf/2403.16771, **[SEARCH-ONLY]**, translation not dialogue).
- Judge reliability on Indic languages, https://arxiv.org/pdf/2606.03043 **[SEARCH-ONLY]**: inter-LLM agreement r~0.35 exceeds LLM-human r~0.27-0.32 across 8 Indic languages; judges use under half the human range on subjective rubrics; a verifiable-answer rubric reaches r=0.519; Pariksha (https://arxiv.org/html/2406.15053v2, **[SEARCH-ONLY]**) found pairwise judging agrees with humans better than direct scoring, worse for Bengali and Odia, with GPT self-bias.
  Practical rule: judge on checkable criteria (answer present in the retrieved doc, tool name and args valid JSON and correct, language/script matches the user), use pairwise vs the teacher answer, and hand-check ~100 samples per language. Use a judge from a different family than the teacher to limit self-bias.
- Privacy (DPDP): train only on synthetic questions generated from the agent's own documents; do not train on real call transcripts without consent and redaction (legal read not done; **[UNVERIFIED, not a legal opinion]**).

## 4. Gap 4: safety erosion after LoRA and cheap mitigations

- Qi et al.: https://arxiv.org/abs/2310.03693 : 10 adversarial examples jailbreak GPT-3.5 Turbo for under $0.20; benign datasets also degrade alignment "to a lesser extent".
- LoRA on Llama-3.2-1B-Instruct, https://arxiv.org/html/2506.18931 (Safe Pruning LoRA): fine-tuning on Dialog Summary (1,000 benign) + PureBad (100 harmful): attack success 6.15% (base) -> **22.52%** (LoRA) -> 9.37% (SafeLoRA) -> 5.73% (SPLoRA); harmfulness 1.15 -> 2.62 -> 1.76 -> 1.58. Note this includes 100 harmful examples; purely benign data is milder (Qi), but still a drift.
- Mixing safety data: Bianchi et al., https://arxiv.org/abs/2309.07875 : adding "just 3% safety examples (a few hundred demonstrations)" substantially improves safety without large capability loss, but too much causes exaggerated refusals.
- Lisa vs LoRA (https://arxiv.org/pdf/2506.00676, **[SEARCH-ONLY]**): AdvBench ASR 17% -> 49% for Lisa vs 13% -> 62% for plain LoRA as poison grows. SafeLoRA works on Llama-3.1-8B but failed on Gemma-2-9B (https://arxiv.org/html/2603.07445v2, **[SEARCH-ONLY]**): mitigations are model-specific.
- Another report: a Llama-3B study with refusal 82% baseline -> 95% LoRA benign (https://arxiv.org/pdf/2605.25073, **[SEARCH-ONLY]**) shows benign LoRA can even raise refusals; direction varies, so measure.
- Alignment is "shallow" (first tokens), so refusals in Hindi/Hinglish (low-resource for safety training) are the likely first thing to go. No Hindi-specific post-LoRA data found **[UNVERIFIED gap]**.

Cheap mitigations for BonClay (ordered by effort):
1. Add 3-5% refusal and AI-disclosure examples to every training set, in Hindi, Hinglish (both scripts) and English, with the teacher's refusals, and format-matched to the task (Eiras et al. idea, **[SEARCH-ONLY]**).
2. Low learning rate, 1-2 epochs, small rank (8), fewer layers: Huang et al. found defences fail at large LR/many epochs (**[SEARCH-ONLY]**).
3. Do not train on user attack content; filter teacher outputs through the judge for policy compliance.
4. Gate promotion: run the 60-case Hindi/Hinglish attack suite on base vs adapter; reject if refusal rate drops more than a set margin (e.g. 5 points; threshold is my suggestion). Add a runtime guard (disclosure/refusal keyword/classifier check) that does not depend on the adapter.
5. LoRA scaling (merge at scale <1) as a dial, since llama.cpp supports `--lora-scaled`; effect on safety **[UNVERIFIED]**.

## 5. Gap 5: QLoRA on a 4 GB NVIDIA GPU (RTX 3050 Ti)

- Unsloth minimum VRAM table, https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements : 3B QLoRA 3.5 GB (lowest row), 7B 5 GB, 8B 6 GB; "these numbers are the absolute minimum"; no 4 GB or 1B row; needs CUDA capability 7.0+ (RTX 30xx is 8.6 so fine); Python 3.11-3.13; Windows without WSL supported via Unsloth Studio; needs VS Build Tools 2022.
- So: **1B (Llama-3.2-1B) and 1.5B are very likely to fit with batch 1, seq 512-1024, gradient checkpointing [ESTIMATE]; 3B at 3.5 GB minimum on a 4 GB card shared with the Windows display is marginal and unlikely to be stable [ESTIMATE]**. A Japanese blog's claim that a 4 GB 3050 handles 1-3B QLoRA at batch 2-4, seq 2048 (https://jisaku.com/posts/batch92-lora-4gb-vram-finetuning, **[SEARCH-ONLY, low reliability]**) is not credible for 3B at that setting.
- I found no tested report of QLoRA on a 3050 Ti 4 GB (a Hugging Face thread asked, unanswered in snippet) **[UNVERIFIED]**. Unsloth blog page returned HTTP 403, so its Llama 3.2 1B/3B numbers are **[UNVERIFIED]**.
- Mismatch: the Windows path would train a 1B CUDA model while Mac serves a 3B MLX model. Adapters are base-specific, so Windows needs its own base (e.g. Llama-3.2-1B GGUF via llama.cpp) and its own adapters; there is no cross-platform adapter reuse. Cleaner: only train on Mac, or train in the cloud/Colab for the Windows users.

## 6. Proposed pipeline (Mac, one agent)

1. **Seed**: the agent's documents + system prompt + tool schemas (appointment booking, meetings). ~200-500 doc chunks.
2. **Question synth** (teacher on Groq): per chunk, generate user turns in English, Hindi (Devanagari), Hinglish (Roman), plus a Marathi slice; include tool-needing turns, off-topic turns, and 3-5% attack/refusal turns. Vary persona/noise. Target 1,000-2,000 candidates.
3. **Targets**: teacher answers via the real production path (retrieval + tools), so the student learns grounded answers and valid tool calls. Keep brief rationale only if it helps (see 2607.08268 teacher finding).
4. **Filter**: rule checks (JSON/tool schema validity, answer-in-context span, script/language match, length), then a pairwise LLM judge from another family; dedupe with BGE-M3; hand-audit ~100. Keep ~600-1,000.
5. **Train**: `mlx_lm.lora --train` on Llama-3.2-3B-Instruct-4bit, rank 8, 16 layers, batch 2, `--mask-prompt`, lr ~1e-5 to 5e-5 (typical, **[UNVERIFIED for this task]**), 1-2 epochs, with holdout. Stop the backend during training.
6. **Eval**: (a) held-out in-domain set vs base and vs teacher (groundedness, tool accuracy, language match) with judge plus human spot check; (b) the 60-case attack suite; (c) the existing Indian-language judged-quality benchmark (base 49%) for regressions; (d) latency (adapter adds no runtime cost once fused).
7. **Serve**: per-agent server with `--adapter-path` (option 2) or fused dir; extend `_MlxServer` to key on agent. Promote only if gates pass.

Budget (M4 16 GB, **[ESTIMATE]**): data gen ~30-60 min of Groq calls (rate limits apply); training 15-60 min; eval 20-30 min; peak training RAM 3-4 GB; adapter 28 MB (rank 8, reported by the M5 run) or ~2 GB fused per agent.

## 7. Effort and risks

Effort: ~1-1.5 person-weeks for one agent end to end (data gen scripts 2 d, eval harness 2 d, train/serve wiring 1-2 d, safety gate 1 d); "continuous per-agent distillation for all agents" is multiples of that plus ongoing eval cost.

Risks:
- Gains may be small versus simply swapping to a stronger base (Gemma 4 E4B 84% vs 49%); try that first.
- Safety erosion, worst in Hindi/Hinglish (section 4).
- Faithfulness can drop after distillation (2607.08268: 64.0 -> 55.0 on a subset).
- Judge unreliability in Indic languages; judge-teacher self-bias; synthetic-data homogeneity.
- Server cannot hot-swap adapters (section 2); concurrency lock already serialises requests.
- Teacher ToS: check Groq/Qwen/gpt-oss licences for training on outputs **[UNVERIFIED]** (Qwen and gpt-oss are open-weight; licence terms not read).
- Staleness: documents change; retraining cadence and knowledge in weights vs retrieval. Keep retrieval as the source of facts and use the LoRA for style/tool/format only.
- Windows 4 GB users get at most a 1B model; separate base, separate adapters.

## 8. Novel angle

Provenance-aware distillation: every adapter ships with a signed manifest (teacher model, data hash, judge version, safety-suite score) using the existing HMAC receipt machinery, shown on the /verify page, plus a consent gate for any real-transcript data. Differentiator rather than a measured quality claim.

## 9. Unresolved list

No primary M4 16 GB LoRA timing; no reload-time measurement; in-process adapter swap untested; AWS example count; Hindi post-LoRA safety data; Marathi/Tamil/Bengali/Telugu distillation numbers; 3050 Ti 4 GB real run; Unsloth Llama-3.2 blog (403); several safety numbers are search-snippet only.
