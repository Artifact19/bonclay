# Decision: which engine clones voices in Indian languages

**Date:** 15 October 2026, updated 17 October 2026 · **Status:** decided · **Owner:** BonClay team

## Decision

**DhVaani-0.5 (ARTPARK-IISc) is the voice-cloning engine for Indian languages.** It clones Hindi, Marathi, Tamil,
Bengali, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Odia, Urdu, Assamese and ten more, from a consented sample in
any language. **Chatterbox Multilingual (Resemble AI, MIT) stays as the Hindi fallback** when DhVaani is not installed.
Qwen3-TTS keeps its ten languages. Indic-Speak's preset voices stay the voice for Indian languages when there is no
cloned voice (and where DhVaani is missing). IndicF5 remains an optional engine nobody has to install.

None of the cloning engines is fast enough for a live phone call or agent turn on a laptop, so agents in a live
conversation speak with a preset voice (Kokoro). Cloned Indian-language speech is for Studio renders, Say it once and
pre-rendered clips.

Engine choice per request (`route_engine` in `backend/backend.py`):

| Voice | Language | Result |
|---|---|---|
| Cloned | Indian language DhVaani covers | DhVaani when set up |
| Cloned | Hindi, DhVaani missing | Chatterbox, else IndicF5 if it can run, else a clear `ENGINE_UNAVAILABLE` |
| Cloned | other Indian language, DhVaani missing | `ENGINE_UNAVAILABLE` with the setup command (preset voices still work) |
| Cloned | English, French, German, Spanish ... (Qwen3's ten) | Qwen3-TTS |
| Preset | Indian language | Indic-Speak when set up, else Kokoro's Hindi voice |
| Live turn / phone call | any | Kokoro preset voice, never DhVaani |

`INDIC_CLONE_ENGINE` overrides the preference: `auto` (default) is DhVaani first, then Chatterbox for Hindi;
`dhvaani` is DhVaani only, with an error rather than a silent substitute; `chatterbox` uses Chatterbox for Hindi (the
other Indian languages still go to DhVaani, because Chatterbox cannot clone them). `DHVAANI=off` turns DhVaani off.

## Licence

**Licence: the project owner obtained approval from ARTPARK-IISc for this use (15-17 Oct 2026); keep the written
approval on file.** The Hugging Face repository ARTPARK-IISc/DhVaani-0.5 is gated and access was granted for the
owner's token. This decision does not replace the written approval: anyone deploying BonClay elsewhere needs their own.

For transparency, the lineage that makes the approval necessary:

- The model card and repository metadata say Apache-2.0, "following the base model", and ship the Apache-2.0 text.
- The base model, [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice), releases its code under Apache-2.0 but says its
  pre-trained weights are **CC-BY-NC** because of its training data (Emilia). DhVaani is a fine-tune: only the text
  embedding was re-initialised and the flow-matching acoustic backbone was transferred, so its weights derive from the
  non-commercial checkpoint. That contradiction between the label and the lineage is why the written approval matters.
- DhVaani's own training data: the IISc SYSPIN and Rasa (CC-BY-4.0) corpora and IndicTTS, which the model card says is
  "subject to the original Indic TTS license terms" (IIT Madras). Those terms were not verified here.
- The 24 kHz vocoder (charactr/vocos-mel-24khz) is MIT.
- Credit: ARTPARK-IISc asks to be named. Every place DhVaani audio is offered shows "Voice cloning in Indian languages by
  DhVaani from ARTPARK-IISc." (Studio, Say it once, the landing credits, the README). Every clip is still watermarked
  and gets a signed receipt whose engine field is `dhvaani`.
- Voice-cloning consent is BonClay's own flow (spoken consent, speaker check, withdrawal deletes the voice and its
  cached DhVaani reference clip); the model card sets no usage restrictions of its own.

## Why

Qwen3-TTS, the engine BonClay already uses for cloning, supports ten languages and none of them is Indian, so "your voice,
in every language you speak" needed another engine. Chatterbox solved Hindi only, slowly. DhVaani covers 22 Indian
languages, is small (0.1B parameters, 491 MB), runs on the Apple GPU and is several times faster.

## How we tested

**1. Hindi bake-off (15 October).** On the team's Apple M4 (16 GB), five Hindi sentences, two reference conditions:
(A) a Hindi reference voice (a synthetic Kokoro voice) speaking Hindi; (B) the project owner's own consented English
voice sample speaking Hindi. Each output was transcribed by Whisper small and scored by character error rate (CER,
lower is better), and compared to the reference with a SpeechBrain ECAPA speaker embedding (cosine similarity). Whisper
small is itself a weak Hindi reader (CER 0.175 on a clean clip), so treat CER below about 0.15 as noise. The machine was
under memory pressure, so times are upper bounds.

| Engine (device) | Condition | Mean RTF | Median CER | Speaker similarity | Notes |
|---|---|---|---|---|---|
| Chatterbox Multilingual (CPU) | A: Hindi → Hindi | 3.9 | 0.21 | 0.77 | stable |
| Chatterbox Multilingual (CPU) | B: English sample → Hindi | 5.6 | 0.27 | 0.80 | intelligible, correct length |
| Chatterbox Multilingual (Apple GPU) | A / B | 13.9 / 22.8 | 0.24 / 0.33 | 0.78 / 0.79 | slower and flaky (NaN errors): avoid |
| IndicF5 (Apple GPU) | A: Hindi → Hindi | 6.9 | 0.13 | 0.76 | best Hindi accuracy |
| IndicF5 (Apple GPU) | B: English sample → Hindi | 9.6 | 0.62 | 0.83* | breaks: leaked English audio, loops |

\* inflated by leaked prompt audio in the output. RTF is generation seconds divided by audio seconds.

**2. DhVaani trial (15 October, `backend/bench/indic_clone_trial.py`, `docs/TRIAL_REPORT.md`).** Five clinic sentences
in each of Hindi, Marathi, Tamil, Bengali and Telugu, two synthetic Kokoro references (English `am_michael`, Hindi
`hf_alpha`), DhVaani at 16 steps on the Apple GPU, read back with mlx-whisper large-v3-turbo. **0 failures in 50.**

| Language | Read-back CER (range over the two references) | Speaker similarity |
|---|---|---|
| Hindi | 5-8 % | 0.69 from a Hindi reference, 0.41 from an English one |
| Tamil | 4-6 % | (same pooled figures) |
| Telugu | 7-11 % | |
| Marathi | 9-18 % | |
| Bengali | 12-18 % | |

BonClay's consent threshold for "same speaker" is 0.45; similarity from an English reference sits at or just under it, so
a Hindi (or other Indic) sample gives a much closer voice than an English one. Speed was 1.3-3 x real time with other jobs
running (the machine was at load average 5-20), and about 3-5 x faster than Chatterbox on equal footing. Chatterbox took
25-40 s per sentence on the CPU when idle.

**3. Live check of the integrated engine (17 October, `backend/bench/dhvaani_live_check.py`).** Only the DhVaani worker
was started (no app server), through the same client class the app uses. One sentence in each of Marathi, Tamil, Bengali,
Telugu, Kannada, Gujarati, Malayalam and Punjabi, from both synthetic references (16 renders per setting), a quiet M4.

| Steps | Reference | Mean seconds per sentence | Mean RTF | Mean CER | Mean similarity |
|---|---|---|---|---|---|
| 16 | Hindi | 5.9 | 1.25 | 0.12 | 0.68 |
| 16 | English | 4.5 | 1.42 | 0.17 | 0.39 |
| 8 | Hindi | 2.9 | 0.63 | 0.11 | 0.69 |
| 8 | English | 2.4 | 0.75 | 0.18 | 0.40 |

The worker is ready in about 4 s (the first request costs 3-4 s more for warm-up). Kannada, Gujarati and Tamil read back
like the earlier languages (CER 6-20 %); Malayalam and Punjabi were clearly worse (CER 18-39 %), which Whisper's own
weakness in those languages may exaggerate: listen before promising them. Eight sampling steps ran about twice as fast
with no visible loss on these 16 sentences (`DHVAANI_STEPS=8`); 16 stays the default because it is what the 50-render
trial measured.

## Consequences

- Any consented sample, including an English one, can speak the Indian languages above, under a licence the owner has
  approval for. A sample in an Indian language clones much closer.
- DhVaani needs the exact words of the reference clip as well as the audio. A voice sample of up to 10 s is used whole
  with the transcript stored on the voice; a longer one is cut at a pause into a clean 4-10 s clip, the cut is
  transcribed with the app's speech recognition, and the pair is cached next to the voice (`{id}.dhvaani.wav` and
  `.json` in `backend/data/audio`) and deleted with it. Without speech recognition a proportional slice of the stored
  transcript is used for that render only, and a clear error is returned when there are no words at all.
- The worker is a separate process (`backend/.venv-dhvaani`, about 0.8 GB, plus 0.5 GB of weights), started on first
  use, stopped by the model manager after 300 s idle and with the API server. It is not started when less than 1.5 GB
  of memory is available (`DHVAANI_MIN_FREE_MB`); the error says so.
- It is too slow for live turns, so those keep the preset voice.
- Numbers, ASCII digits and Latin words are in the model's vocabulary but were not trained as such: write numbers out
  as words where you can (the trial sentences did, and digits were never tested) and check a render before relying on it.
- Quality is uneven across languages: strongest in Hindi, Tamil, Telugu, Kannada; Marathi and Bengali fair; Malayalam
  and Punjabi weaker. Sanskrit, Dogri, Kashmiri and Rajasthani (about a tenth of the training data) are not routed to it.

## Not measured

Naturalness and accent (no listening test), real microphone recordings as references (all references so far were
synthetic Kokoro voices), longer paragraphs, and DhVaani on NVIDIA or CPU-only machines.
