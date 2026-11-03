# Indian-language TTS and ASR: candidate survey (October 2026)

Purpose: which engines besides DhVaani and Indic-Speak could be added to the benchmark in `docs/BENCHMARK_INDIC.md`, and which checker could replace Whisper in the read-back. This is a shortlist, not a ranking.

## How to read the evidence labels

| Label | Meaning |
| --- | --- |
| **Measured here** | Our own run, `docs/BENCHMARK_INDIC.md` (read-back CER through Whisper, 6 to 12 sentences per language, one run). |
| **Full text** | A helper read the paper's HTML text and the passage was returned verbatim. |
| **Tool summary** | The fetch tool returned its own model-written summary of text that was not returned verbatim. Check against the paper before quoting. |
| **Search snippet** | Seen only in a web-search result (model card or blog page), not opened in full. Includes vendor blogs. |

## Papers

### arXiv 2604.21481: Preferences of a Voice-First Nation (Anand et al., IIT Madras, AI4Bharat, Josh Talks; v2, 23 Jun 2026)

Evidence: **Full text** (HTML v2 returned verbatim; the figures, which hold per-language scores, were not readable).

- Seven systems: Gemini 2.5 Pro TTS, GPT-4o-mini TTS, ElevenLabs v3, Sonic 3, Speech 2.8 HD, Bulbul v3 Beta, IndicF5. Only IndicF5 is called open-source in the paper. Indic Parler-TTS, DhVaani and Indic-Speak were not evaluated.
- Ten languages: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Tamil, Telugu, Urdu.
- Overall Bradley-Terry scores: Gemini 1128.53, ElevenLabs v3 1056.28, Sonic 3 1050.83, Bulbul v3 Beta 1021.91, Speech 2.8 HD 993.94, GPT-4o-mini 942.76, IndicF5 805.75. ElevenLabs and Sonic 3 are statistically indistinguishable. IndicF5 is last "despite extensive evaluation coverage", a large gap.
- Gemini is first in 9 of 10 languages, with near parity with ElevenLabs v3 in Marathi.
- Method: 5,357 sentences in 16 domains, 1,915 native raters, about 120,000 pairwise comparisons; each system used its default voice, so voice choice is part of the score. Data: `ai4bharat/SpeechArenaBench` on Hugging Face.
- What drives preference: expressiveness and intelligibility, then liveliness and voice quality (an XGBoost model on six axis ratings predicts the overall choice with 86.1% accuracy on held-out languages).
- Not stated: open-weights status of the commercial systems, any per-language numbers (figures only).

### arXiv 2604.25441: Praxy Voice (Menta, Praxel Ventures; 28 Apr 2026; CC BY 4.0)

Evidence is mixed. The HTML text was verbatim only up to the start of section IV-E.

- **Full text** (and confirmed by the abstract and introduction): the base is ResembleAI Chatterbox Multilingual, frozen, MIT-licensed, 810 M parameters in total; the adapter is a LoRA on the text-head attention projections (7.86 M trainable parameters, 0.97% of the base); the headline Telugu and licence claims.
- **Tool summary** (tables I to V, licence and release details, hardware, limitations): languages evaluated are Telugu, Tamil and Hindi only (Marathi is not covered); adapter weights Apache-2.0 and code MIT; about 1,220 hours of licensed Indic audio; training on one A100-80GB for about 11 hours; Telugu LLM-WER 0.033 for Praxy against 0.029 Sarvam Bulbul, 0.041 ElevenLabs v3, 0.029 Sonic-3, 0.144 Indic Parler-TTS; IndicF5 Telugu 0.079; Hindi LLM-WER 0.025 for plain Chatterbox while the adapter made Hindi worse (0.204 to 0.334); Tamil LLM-WER 0.041; each pilot set is n = 10 utterances per language and the authors state differences of about 5 points are not statistically separable; no MOS study; the metric is a Qwen-2.5-72B semantic judge on Whisper-based transcripts, not plain CER; code-mix uses IndicF5 plus an external LLM API (Claude Haiku 4.5) for transliteration; cloning uses an 8 to 20 second same-language clip and cross-language references hurt.
- Not stated: inference hardware, latency, real-time factor.

## Candidate TTS engines

Open weights, plausible on a 16 GB Mac:

| Model | Languages | Clones | Licence (as seen) | Evidence |
| --- | --- | --- | --- | --- |
| DhVaani 0.5 (IISc / ARTPARK) | 27 | yes | project-specific approval (see `docs/decisions/indian-language-cloning.md`) | **Measured here**; best of the tested engines in 8 of 10 languages |
| Indic-Speak (Bodhan AI / AI4Bharat) | 22 | no (presets) | Indic Open Model License, attribution string required | **Measured here**; better in Malayalam and Punjabi, about 4x slower |
| Kokoro (Hindi voice) | Hindi, Hinglish | no | Apache-2.0 | **Measured here**; best Hinglish CER (0.035) |
| Chatterbox Multilingual (Resemble AI) | 23+ incl. Hindi | yes | MIT | Hindi LLM-WER 0.025 (**Tool summary**, Praxy paper); Tamil and Telugu not covered |
| Praxy Voice (adapter on Chatterbox) | Telugu, Tamil, Hindi | yes | adapter Apache-2.0, code MIT | **Tool summary**; n = 10 per language |
| Indic Parler-TTS (AI4Bharat) | 20 + English | no | MIT or Apache-2.0 (sources disagree) | Telugu LLM-WER 0.144 (**Tool summary**); not in the listener study |
| IndicF5 (AI4Bharat) | 11 | yes | MIT (AIKosh listing) | Last of 7 in the listener study (**Full text**) |
| svara-TTS (kenpath) | 19 | yes | not checked | **Search snippet**; about 2,000 hours, about 50 speakers, 120 h Telugu, 55 h Tamil; no accuracy numbers found |
| Veena (Maya Research) | Hindi, English | no | Apache-2.0 | **Search snippet**; 3B parameters; no numbers found |
| Pocket TTS multilingual (community) | Hindi, Tamil, Telugu and more | yes | not checked | **Search snippet**; self-reported WER about 10.5% Hindi, 11.0% Tamil, 13.8% Telugu; about 418 MB, CPU |
| Voxtral TTS (Mistral) | Hindi among 9 | yes | CC BY-NC 4.0 (non-commercial) | **Search snippet**; no Tamil or Telugu |
| AI4Bharat Indic-TTS | 13 | no | not checked | **Search snippet**; 2022 VITS-style models |
| Qwen3-TTS | 10 | yes | Apache-2.0 | **Search snippet**; no Indian languages seen in its list |

Paid services, useful only as quality references:

| Service | Evidence |
| --- | --- |
| Gemini 2.5 Pro TTS | First in 9 of 10 languages (**Full text**) |
| ElevenLabs v3 | Second tier, tied with Sonic 3 (**Full text**) |
| Cartesia Sonic 3 | Second tier (**Full text**) |
| Sarvam Bulbul v3 | Fourth (**Full text**); about 250 ms first byte per vendor (**Search snippet**) |
| Gnani Vachana | Zero-shot cloning, 12 languages (**Search snippet**); no numbers |
| MiniMax Speech 2.8 HD | Two languages in the study (**Full text**) |
| GPT-4o-mini TTS | Sixth of seven (**Full text**) |
| Google Cloud TTS | Mentioned in a blog benchmark only (**Search snippet**) |

## Suggested engine per language (from the evidence above)

- Hindi: DhVaani; Chatterbox when cloning without DhVaani's approval scope.
- Hinglish: Kokoro (measured best), DhVaani second.
- Marathi, Tamil, Bengali, Telugu, Kannada, Gujarati: DhVaani. Try Praxy for Tamil and Telugu.
- Malayalam, Punjabi: Indic-Speak scored better, but both engines scored badly; check by ear before relying on either.

## ASR for the read-back (replacing or adding to Whisper)

All **Search snippet** unless noted.

- IndicConformer (AI4Bharat, MIT): described as the strongest open option on telephone audio in a 2026 evaluation; RNNT decoding far better than CTC.
- Voice of India benchmark (arXiv 2604.19151): Sarvam Audio lowest WER in 13 of 15 languages; open models in it were IndicConformer and Meta OmniASR (1B, 7B).
- SraVaani 1.0 (arXiv 2608.08235, 2026 preprint): 65 languages and dialects, authors report lowest WER on many pairs; not independently replicated.
- IndicWhisper / Vistaar (2023): historical baseline.

## What was not found

No paper showing a small model with cloning across all 22 languages; no results for Svara or Veena beyond model pages; no confirmation of whether Bulbul has open weights; per-language numbers from the listener study (figures only).

## Next steps if pursued

Add Indic Parler-TTS, Praxy, svara-TTS and Pocket TTS to `bench/indic_benchmark.py` one at a time; add IndicConformer as a second ASR so Marathi results no longer depend on Whisper alone; have native speakers fill the listening column. Check each model's licence on its own model card before publishing audio or a leaderboard.

## Sources

- https://arxiv.org/abs/2604.21481
- https://arxiv.org/abs/2604.25441
- https://arxiv.org/abs/2409.05356 (IndicVoices-R)
- https://arxiv.org/html/2604.19151v2 (Voice of India)
- https://arxiv.org/html/2608.08235v1 (SraVaani)
- https://huggingface.co/ai4bharat/IndicF5
- https://huggingface.co/kenpath/svara-tts-v1
- https://huggingface.co/maya-research/Veena
- https://huggingface.co/Thanushttk/pocket-tts-multilingual
- https://vaani.iisc.ac.in/models
- https://github.com/AI4Bharat/Indic-TTS
- https://github.com/AI4Bharat/IndicConformerASR
- https://www.kdnuggets.com/open-weight-text-to-speach-with-voxtral-tts
- https://www.sarvam.ai/text-to-speech
- https://analyticsindiamag.com/ai-news/gnaniai-unveils-vachana-tts-zero-shot-voice-cloning-model-for-12-indian-languages
- https://caller.digital/blog/indic-tts-benchmark-bulbul-elevenlabs-sarvam-google-ai4bharat-2026
