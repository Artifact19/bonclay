# Which Indian-language voice model for what (measured 2026-10-19, one Apple M4 16 GB)

Source numbers: `docs/BENCHMARK_INDIC.md` (raw JSON in `backend/bench/results/`, not committed). All scores are read-back character error rate (CER) through Whisper large-v3-turbo, lower is better, **score = 100 x (1 - mean CER)**. Voices are synthetic (Kokoro-made references); nobody has listened to any audio yet. Whisper biases Marathi toward Hindi spelling, so every Marathi number is uncertain. 6 to 12 sentences per language, one run: gaps under about 0.02 CER are noise.

## Mean CER by language (successes / sentences in brackets when not all succeeded)

| Language | DhVaani | Indic-Speak | Chatterbox | Pocket TTS | Kokoro Hindi |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hindi | **0.094** | 0.129 | 0.130 | 0.117 | 0.150 |
| Hinglish | 0.104 | 0.131 | 0.152 | 0.066 | **0.035** |
| Marathi | **0.121** | 0.147 (11/12) | n/a | 0.182 | n/a |
| Tamil | **0.039** | 0.056 | n/a | 0.081 | n/a |
| Bengali | **0.127** | 0.187 | n/a | n/a | n/a |
| Telugu | **0.075** | 0.114 | n/a | 0.150 | n/a |
| Kannada | **0.033** | 0.074 | n/a | 0.110 | n/a |
| Gujarati | **0.084** | 0.106 | n/a | n/a | n/a |
| Malayalam | 0.676 | 0.562 | n/a | **0.555** | n/a |
| Punjabi | 0.343 | **0.270** | n/a | n/a | n/a |

n/a = the engine does not support the language (Chatterbox: Hindi only here; Pocket: no Bengali, Gujarati, Punjabi; Kokoro: Hindi only).

## Overall scores

| Engine | Languages | Sentences (failed) | Mean CER | Score | Speed (s of work per s of audio) | Clones a voice | Licence note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| DhVaani 0.5 | 10 | 82 (0) | 0.169 | **83.1** | 1.5 | yes | approval specific to this project |
| Indic-Speak | 10 | 82 (1) | 0.178 | 82.2 | 6.1 | no (presets) | attribution string required |
| Pocket TTS multilingual | 7 | 58 (0) | 0.184 (0.162 for DhVaani on the same 7) | 81.6 | **0.14** | yes | CC-BY-4.0 per card; community, training data undisclosed |
| Chatterbox Multilingual | 1 (+Hinglish) | 14 (0) | 0.139 | 86.1 | 3.4 | yes | MIT |
| Kokoro Hindi | 1 (+Hinglish) | 14 (0) | 0.101 | 89.9 | 0.14 | no | Apache-2.0 |

On the 14 Hindi and Hinglish sentences all five can be compared: Pocket 90.5, DhVaani 90.2, Kokoro 89.9, Indic-Speak 87.0, Chatterbox 86.1. The top three are within noise of each other.

## Other tests run in this work

| Test | Result |
| --- | --- |
| Marathi spelling hints (4 sentences, `docs/RESULTS.md`) | only ऱ्ह -> रह moved a count beyond one ("ऱ्ह lost" 3 -> 1); ळ hints within noise. Not confirmed by ear, so `PRON_LEXICON` stays off. |
| Backend tests, relevant paths | 247 passed, plus the new benchmark, lexicon and flag tests (all passing) |
| Pocket TTS card's own numbers | Self-reported word error 10.5% Hindi, 11.0% Tamil, 13.8% Telugu, 14.7% Kannada, 23.7% Malayalam, 5.8% Marathi, using a different ASR (Indic Transcribe) and a different eval set; not comparable with our CER. Our run gave higher errors for Marathi (0.182) than the card's 5.8%, which suggests the ASR and test set matter more than the model. |
| Voxtral TTS (Mistral) | **Not run.** 4B parameters, bf16; the official card serves it through vLLM on a 16 GB or larger GPU and lists no Apple-Silicon path. Hindi is the only Indian language. Licence CC-BY-NC-4.0 (non-commercial). A community MLX port exists but was not verified. |
| IndicF5 (AI4Bharat) | **Not run.** Gated weights, needs the `f5_tts` package in its own environment. A large listener study ranked it last of 7. |

## Inference (what the numbers support, and what they do not)

1. **DhVaani is the best all-round open choice we measured**: lowest CER in 7 of 10 languages, clones voices, no failures, and fast enough for studio renders (1.5x real time). Its weak spots are Malayalam and Punjabi, where every engine scored badly.
2. **Indic-Speak is not worth its cost for most languages**: slightly worse than DhVaani almost everywhere, 4x slower, 7 GB of memory, one refused sentence. It is better on Malayalam and Punjabi and has 22 languages, so keep it as a fallback and for presets.
3. **Pocket TTS multilingual is the surprise**: about 10x faster than DhVaani, 0.4 GB, CPU-friendly, level with DhVaani on Hindi and best on Hinglish among cloning engines. It is clearly worse on Marathi, Tamil, Telugu and Kannada, and weak on Malayalam like the rest. It is a one-person community fine-tune with undisclosed training data: treat it as a candidate for live calls to evaluate, not a release dependency.
4. **Chatterbox gives no accuracy advantage** here (86.1 on Hindi and Hinglish, last of five) and is 2x slower than DhVaani. Its value is licence (MIT) and being a known base.
5. **Kokoro Hindi is best for Hinglish text read in Devanagari** (0.035) and is instant, but it is a preset voice, not a clone.
6. **Malayalam and Punjabi results (0.27 to 0.68 for all engines) point at the measurement** (Whisper's accuracy or my unreviewed sentences) more than at any engine. Do not publish those numbers before a native speaker has listened.
7. **Limits that apply to everything above:** Whisper read-back is not listening; 6 to 12 sentences per language; sentences are AI-drafted and not native-reviewed; one run; Indic-Speak ran with a reduced memory reserve so its speed is slightly pessimistic.

## What suits what

| Need | Pick | Why / caveat |
| --- | --- | --- |
| Clone a voice in an Indian language (Studio renders) | **DhVaani** | best measured accuracy across 10 languages; keep the approval and the attribution requirements |
| Live phone calls (latency matters) | **Kokoro Hindi** for Hindi/Hinglish presets; evaluate **Pocket TTS** if cloning is needed | both at 0.14x real time; DhVaani is 1.5x, not live |
| Hinglish | **Kokoro** (0.035), Pocket second (0.066) | read the Devanagari rendering of the text |
| Hindi with cloning and a clean licence | **DhVaani** (0.094) or **Pocket** (0.117); Chatterbox (MIT, 0.130) if licence is the deciding factor | all within a few points |
| Marathi, Tamil, Bengali, Telugu, Kannada, Gujarati | **DhVaani** | clear lead; for Tamil and Telugu try Praxy next (not run) |
| Malayalam, Punjabi | **Indic-Speak** (Punjabi), **Pocket/Indic-Speak** (Malayalam) | nobody is good; verify by ear first |
| 22 languages with preset voices, no cloning | **Indic-Speak** | slow; attribution required |
| Lowest memory (0.4 GB) / CPU only | **Pocket TTS** | 7 languages |
| Non-commercial research only | Voxtral TTS (Hindi) | not measured here |
| Best quality regardless of cost | paid APIs (Gemini TTS led a 120,000-comparison listener study) | not in this benchmark |

Per-language routing is the intended use: send each language to its row above. The first change that would move these results is a native-speaker listening pass (`listening_sheet.csv`) and a second ASR (IndicConformer) in place of Whisper-only scoring.

## Making Indic-Speak smaller: MLX quantization (tested 2026-10-19, Hindi + Marathi + Tamil, 28 sentences)

Indic-Speak is a Llama-style language model (3.2B parameters, bf16, 6.1 GB) plus a small SNAC/Vocos audio decoder. Only the language model is big, so `bench/indic_speak_quant.py` converts it with `mlx_lm convert -q` (4-bit and 8-bit, group size 64) and runs it on Apple's MLX; the original decoder is unchanged. Same prompts, sampling (temperature 0.6, top-p 0.9, top-k 50), token cap and runaway retry as the app.

| Variant | Weights | Peak memory | Mean CER (27 common sentences) | Speed (s of work per s of audio) | Seconds per sentence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original (torch, bf16, MPS) | 6.1 GB | about 7 GB (guard figure) | 0.115 | 6.6 | 35 |
| MLX 8-bit | 3.3 GB | 3.5 GB | 0.136 | 3.5 | 19 |
| MLX 4-bit | 1.8 GB | 2.1 GB | 0.126 | 1.8 | 10 |
| (DhVaani, same sentences) | n/a | about 1.5 GB | 0.090 | 1.5 | 8 |

Reading: 4-bit cut memory by about 70% and sped generation up 3.7x, at a CER cost of about 0.011 (not distinguishable from noise on 27 sentences; it was better than the original on 9 of 27). 8-bit was no better than 4-bit here, which is unexpected and also within noise. Even so, DhVaani remains more accurate (0.090) and as fast. The 4-bit variant makes Indic-Speak practical on a 16 GB Mac (about 2 GB instead of 7 GB, no reserve fight with other apps), which matters for its 22 languages and presets, not for beating DhVaani.

Pitfalls found: (1) `mlx_lm` ignores the `rope_parameters` key used by this model's transformers-5 config and silently falls back to rope_theta 10000, which makes the model run on without stopping; the converted `config.json` needs `rope_theta` 500000 and a `llama3` `rope_scaling` block added by hand. (2) The model needs a speaker tag; with none it never ends. (3) Runs are random: one sentence hit the token cap in one run and not another. Unverified: audio was not listened to; only 3 languages; one run each; the app does not use the quantized path (benchmark only).

## OmniVoice (Xiaomi / k2-fsa, arXiv 2604.00688): research-only test, 2026-10-20

Weights are CC-BY-NC (non-commercial); code is Apache-2.0. Run in its own environment (`backend/.venv-omni`, torch 2.8.0, `pip install omnivoice`) on the Apple GPU (MPS, float16) through `bench/omnivoice_worker.py`; all 82 sentences, Hindi synthetic reference, language code passed per sentence, no failures. The 0.6B model needed about 5 minutes to download.

| Language | OmniVoice | DhVaani | Indic-Speak | Pocket |
| --- | ---: | ---: | ---: | ---: |
| Hindi | 0.117 | **0.094** | 0.129 | 0.117 |
| Hinglish | 0.116 | **0.104** | 0.131 | 0.066 |
| Marathi | 0.149 | **0.121** | 0.147 | 0.182 |
| Tamil | **0.037** | 0.039 | 0.056 | 0.081 |
| Bengali | 0.151 | **0.127** | 0.187 | n/a |
| Telugu | 0.096 | **0.075** | 0.114 | 0.150 |
| Kannada | 0.134 | **0.033** | 0.074 | 0.110 |
| Gujarati | 0.084 | 0.084 | 0.106 | n/a |
| Malayalam | 0.583 | 0.676 | 0.562 | **0.555** |
| Punjabi | 0.435 | 0.343 | **0.270** | n/a |

Overall mean CER 0.190 (DhVaani 0.169); OmniVoice was better than DhVaani on 25 of 82 sentences. Speed 3.3 s of work per s of audio (16 s per sentence), slower than DhVaani on the same GPU (1.1 to 1.5). So its paper claims (4.0% average CER across 102 languages, measured with a different recogniser) were not reproduced as an advantage here; it is competitive only on Tamil and Gujarati. It does not beat DhVaani, is non-commercial, and is slower: not a candidate for the product. Limits: one run, 6 to 12 sentences per language, Whisper read-back, no listening; OmniVoice's own `normalize_text` option (numbers to words) was not tried.

## 4-bit Indic-Speak on all 10 languages, and what a "best model per language" router would do (2026-10-20)

Indic-Speak with the MLX 4-bit language model, now on all 82 sentences (81 succeeded; `mr-06` fails as in the original): mean CER 0.188, 1.9 s of work per s of audio, peak memory 2.05 GB. Paired with the bf16 original on the same 81 sentences the 4-bit cost is **+0.010 CER**, consistent across languages (Punjabi 0.283 against 0.270; Hinglish 0.051 against 0.131 is noise from 6 sentences). Malayalam 0.684 against 0.562 is within the usual 0.1 spread of that language.

Best measured engine per language (mean CER, 6 to 12 sentences): Hindi DhVaani 0.094; Hinglish Kokoro 0.035; Marathi DhVaani 0.121; Tamil OmniVoice 0.037 / DhVaani 0.039; Bengali DhVaani 0.127; Telugu DhVaani 0.075; Kannada DhVaani 0.033; Gujarati DhVaani/OmniVoice 0.084; Malayalam Pocket 0.555 / Indic-Speak 0.562 (all unreliable); Punjabi Indic-Speak 0.270 (4-bit 0.283).

Consequence for routing: DhVaani is best or tied in 8 of 10 languages. The app already sends cloning to DhVaani and preset voices to Indic-Speak (Kokoro as fallback), so re-ranking those two engines per language would change almost nothing (Hindi preset: Indic-Speak 0.129 / 4-bit 0.146 against Kokoro 0.150). The measured gains are elsewhere: (1) use DhVaani with a fixed synthetic house reference as the preset engine for the languages where it leads; (2) run Indic-Speak at 4-bit (2 GB instead of 7 GB, 3x faster, +0.01 CER) for the languages where it is still needed (Punjabi, Malayalam, the 12 languages DhVaani does not train on); (3) DhVaani at 6 steps; (4) number-to-words before DhVaani. Not done in the app yet.

## Using the 4-bit Indic-Speak in the app (built 2026-10-20, off unless configured)

```bash
bash backend/scripts/setup_indic_speak_q4.sh      # after the normal Indic-Speak setup; needs an Apple Silicon Mac
# add the printed line to backend/.env, then restart the API:
INDIC_SPEAK_MLX_MODEL=~/.cache/bonclay/indic-speak-q4
```

With the variable set (and the folder complete) the same worker runs the 4-bit model on the Apple GPU through `services/indic_speak_quant.py` in its own `backend/.venv-indic-q4`; unset it to return to the original. Tested through the app's own client: ready in 5 s, 1.8x real time, 3.1 GB for the whole worker process (the memory guard now asks for 3.5 GB instead of 7 GB when the variable is set; `MEMORY_NEED_INDIC_SPEAK_MB` still overrides), Marathi, Hindi and Punjabi sentences read back correctly. Two traps handled in code: `mlx-lm` ignores this model's `rope_parameters` config (fixed by `fix_rope_config`), and MLX's GPU stream is bound to one thread, so all MLX work runs on one dedicated thread. Not done: the original model is still the default; no listening test; the app's other engines were not changed.
