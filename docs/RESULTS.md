# BonClay measured results

Measured on Apple M4 (10 logical cores), 16 GB RAM, macOS-15.8.1-arm64-arm-64bit; 15 and 16 October 2026 (updated 17 October 2026 with the 16 October run); newest speed run commit 68cb06a with uncommitted changes (the tables below name the commit of each run: 68cb06a, 7fde8dd, 9ff35a2). Every number here comes from the scripts in `backend/bench/`, which call the running API over HTTP, so they measure the real system. Rerun everything with:

```
./bonclay start
backend/.venv/bin/python backend/bench/latency.py
backend/.venv/bin/python backend/bench/latency.py --stream
backend/.venv/bin/python backend/bench/readback.py --engines kokoro-en,kokoro-hi,indic-hi
backend/.venv/bin/python backend/bench/clone_time.py
backend/.venv/bin/python backend/bench/asr_compare.py   # no API needed: starts its own MLX worker
backend/.venv/bin/python backend/bench/soak.py --minutes 45
backend/.venv/bin/python backend/bench/report.py
```

Raw per-item measurements are in `backend/bench/results/` (not committed). Anything not measured is marked "not measured".

## Current numbers (16 October 2026)

This section is the one to quote. The older tables further down are kept as history, with their dates, because they are what the October improvements were measured against. Where a line says "Source", the figure is in the named document or bench result. This section was written by hand on 17 October 2026; `backend/bench/report.py` regenerates the whole file from the newest bench results and would overwrite it, so merge rather than regenerate.

### Speed, 30 spoken questions, streaming, local models only

Method: the same 30 questions as the history tables (15 English, 15 Hindi, from the clinic receptionist template), posted to `/api/conversations/{agent}/turn/stream`, a new conversation each, after 2 warm-up turns, no cloud keys, LLM mlx-community/Llama-3.2-3B-Instruct-4bit. Failed turns: 0. Run 2026-10-16T05:35:45, commit 68cb06a (uncommitted changes), raw file `backend/bench/results/latency_stream-20261016-053741.json`. Same script as before: `backend/.venv/bin/python backend/bench/latency.py --stream`.

| Measure | Median (p50) | p95 | Morning 15 Oct (history) |
| --- | --- | --- | --- |
| First audio (the first sentence is ready to play; at the start of a call this is the spoken AI disclosure) | **0.84 s** | 1.34 s | 1.57 s median, 2.64 s p95 |
| First sentence of the answer itself | **2.23 s** | **3.58 s** | 4.20 s median, 6.73 s p95 |
| Full reply ready (stitched and watermarked) | **3.02 s** | 5.03 s | 4.68 s median, 8.44 s p95 |
| Speech recognition | 0.73 s | | 1.46 s |
| Knowledge search | 0.04 s | | 0.02 s |
| Voice synthesis per turn | 0.82 s | | 1.39 s |

By language (median): English first audio 0.48 s, first answer 1.89 s, full reply 2.37 s; Hindi first audio 1.28 s, first answer 3.07 s, full reply 3.47 s. Within 3 s: first audio 100% of turns, first answer 73% (English 100%, Hindi 47%), full reply 50% (English 80%, Hindi 20%). AI disclosure spoken: 30/30. Fallback replies: 0.

Reading it honestly: the caller hears the AI disclosure at 0.84 s, but the disclosure is pre-synthesised and plays while the model is still working. The first sentence of the answer is the number that matches "how long until it answers me": 2.23 s median, 3.58 s at p95. The quoted 1.6 s first-audio figure in earlier versions of the landing page was the disclosure sentence on the morning run; it is replaced by these numbers. These times start when the audio request is sent, not when the caller stops speaking.

Source for the third-wave changes behind the improvement (cache-friendly prompts, streamed tool agents, fp16 search): `docs/TRIAL_REPORT.md` section 9. The same table with the earlier steps: first audio 1.57 s (morning), 1.35 s (after the first wave), 1.02 s (second wave), 0.84 s (third wave); full reply median 4.68, 3.77, 3.10, 3.02 s (`docs/TRIAL_REPORT.md` sections 2, 8 and 9).

### Speed with Groq answering (15 October 2026)

With a `GROQ_API_KEY` set (the normal set-up), on the 10 turns Groq answered: first audio 0.50 s, first answer 2.52 s, **full reply 2.81 s** (under the 3-second target), speech recognition 0.25 s (Groq Whisper). Caveat: n = 10 only, because 20 of the 30 turns fell back to the local model after the Groq free tier's limits were used up that day (Qwen about 8,000 tokens per minute, gpt-oss-20b 200,000 tokens per day). Source: `docs/TRIAL_REPORT.md` section 2. A paid Groq tier has not been tried.

### Knowledge search

45 questions (13 English, 14 Hindi, 18 Hinglish) over three template documents, `backend/bench/retrieval_eval.py`: BGE-M3 plus keyword (hybrid) finds the correct passage first for 75.6% and in the top 3 for **97.8%** (MiniLM before: 64.4% and 88.9%). Reranker on: 84.4% and 97.8%, but costs 0.2 to 1.7 s, so it stays off. Source: `docs/TRIAL_REPORT.md` section 3, `STATUS.md`.

### Watermark robustness

20 clips, 21 attacks, AudioSeal at the app's own detection threshold, `backend/bench/watermark_attacks.py`. Detected in 20/20 clips after MP3 at 64 and 32 kbps, Opus at 16 kbps, a WhatsApp-style chain, an 8 kHz phone line and trimming; no false alarms on 420 clean checks (`docs/TRIAL_REPORT.md` section 4). After the second wave the common channels (MP3, Opus, WhatsApp-style, phone line, trimming) were still **70/70 detected in both English and Hindi** with no false positives (`docs/TRIAL_REPORT.md` section 8). Not survived: plus or minus 5% speed change, pitch shift, noise or music at 10 dB, AI re-encoding (EnCodec 6 kbps: 1/20). Hindi under heavy noise or music is weaker than English: 28/60 against 37/60 attack cases (`STATUS.md`, known issues). The watermark is evidence, not proof.

### DhVaani sampling steps (20 October 2026)

`backend/bench/dhvaani_live_check.py --steps N` on the same Apple M4: 8 sentences in 8 languages (Marathi, Tamil, Bengali,
Telugu, Kannada, Gujarati, Malayalam, Punjabi) from two synthetic Kokoro references, scored by Whisper read-back. One run each, so
differences of a few hundredths of CER are noise.

| Steps | Reference | Seconds per sentence | Speed (s of work per s of audio) | Mean read-back error | Speaker similarity |
|---|---|---|---|---|---|
| 16 | Hindi speech | 5.14 | 1.10 | 0.118 | 0.674 |
| 8 | Hindi speech | 2.68 | 0.57 | 0.146 | 0.684 |
| 6 | Hindi speech | 2.06 | 0.44 | 0.119 | 0.683 |
| 16 | English speech | 4.13 | 1.31 | 0.171 | 0.391 |
| 8 | English speech | 2.18 | 0.69 | 0.182 | 0.408 |
| 6 | English speech | 1.70 | 0.53 | 0.185 | 0.386 |

The default is now 8 steps (faster than real time: about 0.6 of the audio length). Accuracy and speaker similarity are not
distinguishable from 16 steps in this test, but the test cannot hear: if a cloned voice sounds rougher, set `DHVAANI_STEPS=16`. Not
solved by this: digits are not read correctly (write numbers as words), and Malayalam and Punjabi stay weaker than the other languages at every step count.

### DhVaani Marathi

Run 2026-10-19 on the same Apple M4 (16 GB), by `backend/bench/dhvaani_marathi_check.py`, which starts only the DhVaani worker (device `mps`, ready in 4.0 s) through the app's own client, never the API server. It builds on `video/dhvaani_test` (one clinic sentence, 16 steps: read-back CER 0.08 from the Hindi reference and 0.12 from the English one, speaker similarity 0.75 and 0.41). Here: 12 varied Marathi sentences (greetings, a date, a fee, a question, a 23-word sentence, English loanwords written in Devanagari, three sentences with ळ, one with ऱ्ह, a Hinglish-style sentence and one with Marathi-only forms) x 2 reference types x 2 step settings = 48 renders. References are synthetic Kokoro voices only (`hf_alpha` speaking Hindi, `am_michael` speaking English, 6.5 to 6.7 s each); no real person's recording was used. Audio is in `video/dhvaani_marathi/` (file list in its README.md), raw numbers in its `results.json`.

| Reference | Steps | Mean seconds per sentence | RTF | Read-back CER (mean / median) | Speaker similarity (ECAPA) |
| --- | --- | --- | --- | --- | --- |
| Hindi speech | 16 | 5.6 | 1.15 | 0.117 / 0.104 | 0.683 |
| Hindi speech | 32 | 17.9 (see timing note) | 3.68 | 0.114 / 0.099 | 0.642 |
| English speech | 16 | 5.1 | 1.64 | 0.196 / 0.200 | 0.409 |
| English speech | 32 | 22.8 (see timing note) | 7.38 | 0.205 / 0.200 | 0.406 |

Mean audio per sentence was 5.1 s from the Hindi reference and 3.4 s from the English one (the English-reference voice speaks faster, which is why its RTF is worse at the same seconds). The 23-word sentence took 11.3 s (Hindi ref) and 8.9 s (English ref) at 16 steps, RTF 0.83 and 1.00, so long sentences are the one case that runs faster than real time; short ones cost a roughly fixed 4.5 to 5 s.

Timing note for 32 steps: the first three 32-step renders took 8.9 to 9.9 s (about twice the 16-step cost, as expected), then every later render on the machine slowed to 20 to 25 s with nothing else of ours running (the machine's load average was 2 to 3; other agents were working in the same checkout). A separate 32-step rerun was slower still (mean 27.6 s Hindi ref, 30.5 s English ref). So the 32-step columns above are inflated by sustained slowdown (thermal or memory pressure on this Mac); the honest cost of 32 steps is about 2x the 16-step time, about 9 to 10 s per short sentence. The 32-step CER of the rerun (0.115 Hindi ref, 0.208 English ref) matches the first run, so quality figures are stable even though timing was not.

Per-sentence read-back CER (Hindi ref 16 / 32 steps, English ref 16 / 32 steps):

| # | Sentence type | Hi 16 | Hi 32 | En 16 | En 32 |
| --- | --- | --- | --- | --- | --- |
| 1 | greeting | 0.08 | 0.08 | 0.16 | 0.16 |
| 2 | greeting + question | 0.09 | 0.06 | 0.03 | 0.09 |
| 3 | date and number | 0.07 | 0.07 | 0.25 | 0.27 |
| 4 | fee (number) | 0.17 | 0.19 | 0.31 | 0.28 |
| 5 | question | 0.11 | 0.11 | 0.11 | 0.11 |
| 6 | long (23 words) | 0.12 | 0.10 | 0.25 | 0.26 |
| 7 | loanwords (अपॉइंटमेंट, क्लिनिक) | 0.09 | 0.09 | 0.16 | 0.16 |
| 8 | ळ (सकाळी, संध्याकाळी, गोळी) | 0.12 | 0.12 | 0.21 | 0.21 |
| 9 | ळ (वेळेवर, कळवू, काळजी) | 0.09 | 0.09 | 0.13 | 0.13 |
| 10 | ऱ्ह (वऱ्हाडी, कऱ्हाड, नऱ्हे) | 0.27 | 0.25 | 0.27 | 0.27 |
| 11 | Hinglish-style mix | 0.10 | 0.10 | 0.19 | 0.19 |
| 12 | Marathi-only forms | 0.11 | 0.11 | 0.29 | 0.33 |

Hindi-leakage scan (heard word compared with the reference word; 9 words with ळ and 3 with ऱ्ह per run):

| Pattern | Hi ref 16 | Hi ref 32 | En ref 16 | En ref 32 | Examples (what Whisper heard) |
| --- | --- | --- | --- | --- | --- |
| ळ heard as ल (or ड, र) | 8 of 9 | 7 of 9 | 6 of 9 | 5 of 9 | सकाळी as सकाली, संध्याकाळी as संध्याकाली, गोळी as गोली, वेळेवर as वेलेवर, कळवू as कलवू, काळजी as कालजी; English ref also गोडी, मंडडी, संध्याकारी |
| ऱ्ह reduced to रह / ह | 3 of 3 | 3 of 3 | 3 of 3 | 3 of 3 | वऱ्हाडी as वरहाडी / वहाडी, कऱ्हाडहून as करहाड, नऱ्हे as नरहे |
| ण heard as न | 1 | 1 | 4 | 3 | पण as पंड / पड़, जेवणानंतर as जेवनानंतर, आणि as यानी |
| Final ी written ि or Hindi-style endings | 4 | 3 | 2 | 1 | मंडळी as मंडली |
| Hindi grammar words (है, लिया) instead of Marathi | 1 sentence | 1 sentence | 5 sentences | 5 sentences | ठरली आहे as ठर लिया है, स्वागत आहे as स्वागता है, होत आहे as होता है |

Character-level substitutions across all 48 clips, most frequent: ळ to ल (16), त to द (12), short ि to long ी (10), ऱ् to र (9), झ to ज (8), े to ै (8), ळ to ड (6). The Hindi-reference voice leaks sounds (ळ to ल, ऱ्ह to रह) but keeps Marathi grammar; the English-reference voice additionally slips into Hindi grammar words in 5 of 12 sentences at either step count.

Caveats:
- Read-back CER is Whisper large-v3-turbo hearing the clip, not a Marathi listener. Whisper is weak on Marathi: it normalises toward Hindi spelling, so some of the ळ to ल and ऱ्ह to रह cases may be Whisper's own spelling bias and not what was spoken. The pattern is consistent and the ळ cases cluster in the same words in all four settings, so leakage is probably real, but the owner should listen to files 08, 09 and 10 (`s08_la1_*`, `s09_la2_*`, `s10_rha_*`) to confirm. Absolute CER (0.11 to 0.20) is therefore not comparable with the Hindi rows above, whose ASR is stronger.
- Numbers are scored after mapping Whisper's digits back to the sentence's number words, in order; the fee sentence (`4,85` for चारशे पन्नास) is a Whisper formatting problem as much as a speech one, and its CER is not trustworthy.
- Speaker similarity is the ECAPA cosine to the reference clip. The reference speaks a different language from the clone, so 0.68 for the Hindi reference and 0.41 for the English one are indicative only; it is not a listening test.
- One reference voice per type, one run per setting (plus a rerun of 32 steps), 12 sentences: the Hindi-versus-English gap is large (CER 0.12 vs 0.20, similarity 0.68 vs 0.41) and consistent across sentences, the 16-versus-32 difference is not.

Recommendation: the multilingual model is usable for Marathi as it is, and a Marathi-specific adapter or fine-tune is not worth doing yet. Default to a Hindi-speech reference at 16 steps: it is 2x cheaper than 32 steps, 32 steps changed neither CER (0.117 vs 0.114) nor leakage counts in a meaningful way and lowered speaker similarity slightly (0.683 vs 0.642), and the English reference is clearly worse (CER 0.20, similarity 0.41, Hindi grammar words appearing). When the user uploads their own voice in English, that is the weak case: say so in the UI or prefer asking for a short Marathi or Hindi sample. The one real defect is phonetic: ळ and ऱ्ह come out in the Hindi way in most words, which a native speaker will notice in common words (सकाळी, गोळी, वेळ). Before any fine-tune, a cheaper step is to confirm it by ear on the files above, and if confirmed, try a small text-side fix (for example spelling hints for ळ words) or a Marathi-native reference clip; a fine-tune needs a licensed Marathi speech set and GPU time that this 16 GB Mac cannot supply. Revisit the adapter only if the listening check shows ळ is wrong in the majority of words and the text-side fix does not help.

## Voice cloning time

Unchanged from the table further down (15 October, 10 runs): warm p50 2.4 s from consent clip to a usable voice, consent check included; first generation with the new voice 6.5 s warm p50. Indian-language cloning with DhVaani-0.5 (adopted 17 October, `docs/decisions/indian-language-cloning.md`): 0 failures in 50 trial runs; read-back error Hindi 5 to 8%, Tamil 4 to 6%, Telugu 7 to 11%, Marathi 9 to 18%, Bengali 12 to 18%; not fast enough for live calls (`docs/TRIAL_REPORT.md` section 5).

### Stability

45-minute soak on 15 October: 733 of 733 requests answered, 0 crashes (table further down).

### Test counts

Tests on 17 October 2026: 1,390 backend tests passed (3 skipped) with pytest, and 404 frontend tests passed with Jest.

### Hosted-platform reference points checked on 17 October 2026

These are other companies' published figures, retrieved on the date given, not measured by us. None of the pages shows a publication date, so the month is the month we read them.

| Source | Published figure |
| --- | --- |
| [Vapi pricing](https://vapi.ai/pricing), read 17 Oct 2026 | Hosting $0.05 per minute, plus provider pass-through costs listed per minute: speech recognition (Deepgram) $0.0095 to $0.0099, language model (OpenAI) $0.0077 to $0.0452, voice (ElevenLabs) $0.0146 to $0.0238, telephony free on Vapi's own transport or $0.0055 to $0.014 per minute through Twilio, Vonage or Telnyx. Currency shown as "$" only |
| [Retell AI pricing](https://www.retellai.com/pricing), read 17 Oct 2026 | Voice infrastructure $0.055 per minute; voice $0.015 per minute for most voices, $0.040 to $0.10 for ElevenLabs; telephony $0.015 per minute via Twilio, none on SIP trunking; language model varies by model. The estimator's default set-up totals $0.11 per minute (language model $0.04, voice infrastructure $0.055, voice $0.015, telephony $0) |
| [Twilio, "A Guide to Core Latency in AI Voice Agents (Cascaded Edition)"](https://www.twilio.com/en-us/blog/developers/best-practices/guide-core-latency-ai-voice-agents), 17 Nov 2025 | Mouth-to-ear turn gap: target 1,115 ms, upper limit 1,400 ms, given as "starting benchmarks" for the initial launch of a voice agent. It is a target, not a measurement of a product, and the earlier wording in this file ("median, as published") was corrected on 17 October 2026 |

## History: spoken-turn latency (15 October 2026, morning)

Method: 30 spoken questions (15 English, 15 Hindi, answerable from the clinic receptionist template) rendered with Kokoro beforehand, posted one by one to `/api/conversations/{agent}/turn`, a new conversation each, after 2 warm-up turn(s). Client time is request start until the JSON is received; server time is the API's own `total_ms`. This is speech in to the full reply audio ready: replies are not streamed, so it is not time to first audio. Failed turns: 0.

Run 2026-10-15T06:01:35, commit 9ff35a2 (uncommitted changes), LLM: mlx mlx-community/Llama-3.2-3B-Instruct-4bit.

| Questions | Turns | Client p50 (ms) | Client p95 (ms) | Server p50 (ms) | Server p95 (ms) | Within 3 s | ASR p50 | Retrieval p50 | LLM p50 | TTS p50 | Fallback replies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 30 | 4416 | 8108 | 4410 | 8101 | 23% | 1442 | 17 | 798 | 1392 | 0 |
| English | 15 | 3309 | 5779 | 3304 | 5772 | 47% | 794 | 13 | 704 | 1279 | 0 |
| Hindi | 15 | 4769 | 8650 | 4762 | 8643 | 0% | 2136 | 28 | 818 | 1451 | 0 |

## History: time to first audio (streaming, 15 October 2026, morning)

Method: the same 30 questions as above, posted to `/api/conversations/{agent}/turn/stream`, which answers with server-sent events, a new conversation each, after 2 warm-up turn(s). The client clock starts when the request is sent and stops when the first `sentence` event arrives (first audio: the caller can start hearing the reply, and every sentence clip is already watermarked), when the first sentence of the answer arrives, and when `done` arrives (the whole reply ready, stitched and watermarked, same as the table above). The agent says it is an AI at the start of each call (each question is a new conversation), so its first sentence is the spoken disclosure; it is synthesized once and replayed, and plays while the model is still writing the answer. "First answer" is the first sentence of the answer itself. Failed turns: 0.

Run 2026-10-15T07:25:22, commit 7fde8dd (uncommitted changes), LLM: mlx mlx-community/Llama-3.2-3B-Instruct-4bit.

| Questions | Turns | First audio p50 (ms) | First audio p95 (ms) | First audio within 3 s | First answer p50 (ms) | First answer p95 (ms) | First answer within 3 s | Full reply p50 (ms) | Full reply p95 (ms) | Server first audio p50 | ASR p50 | LLM first token p50 | AI disclosure spoken |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 30 | 1574 | 2642 | 100% | 4199 | 6728 | 30% | 4680 | 8436 | 1570 | 1456 | 162 | 30/30 |
| English | 15 | 891 | 1021 | 100% | 2704 | 5351 | 60% | 3386 | 6462 | 887 | 800 | 142 | 15/15 |
| Hindi | 15 | 2449 | 2651 | 100% | 4762 | 6940 | 0% | 5076 | 8163 | 2446 | 2131 | 174 | 15/15 |

## Voice intelligibility (read-back)

Method: 20 fixed sentences per engine (8 to 15 words, with numbers and names) are generated, then transcribed by the API's own `/api/transcribe`, and compared with the text. English is scored by word error rate (lowercase, no punctuation, numbers compared as written, so a transcript with digits counts as an error); Hindi by character error rate over Devanagari letters only. RTF is generation seconds divided by audio seconds (below 1.0 is faster than real time). Lower is better for both. Read-back measures intelligibility as the ASR hears it, not naturalness or likeness to a person.

Run 2026-10-15T06:04:46, commit 9ff35a2 (uncommitted changes), LLM: mlx mlx-community/Llama-3.2-3B-Instruct-4bit.

| Engine | Metric | Error rate | Mean RTF | Scored | Failures | Transcribed by |
| --- | --- | --- | --- | --- | --- | --- |
| Kokoro English (af_heart) | WER | 0.076 mean / 0.000 median | 0.18 | 20/20 | 0 | mlx-community/whisper-small-mlx |
| Kokoro Hindi (hf_alpha) | CER | 0.071 mean / 0.053 median | 0.18 | 20/20 | 0 | mlx-community/whisper-large-v3-turbo |
| Cloned voice, English | WER | 0.089 mean / 0.080 median | 1.69 | 20/20 | 0 | mlx-community/whisper-small-mlx |
| Indic-Speak Hindi (Kavya) | CER | 0.091 mean / 0.084 median | 6.96 | 20/20 | 0 | mlx-community/whisper-large-v3-turbo |

The Indic-Speak Hindi row is a separate run (2026-10-15T10:13:26, commit 92865b2 with uncommitted changes, same 20 sentences, voice Kavya, preset voice, transcribed by `/api/transcribe`). Its RTF means about 7 seconds of generation for each second of speech on this machine.

## Indic-Speak (Indian-language preset voices)

Measured by hand through `POST /api/indic/generate` plus the Indic-Speak row above, on the same Apple M4 (the model runs in its own virtualenv, on the Apple GPU).

| Measure | Result |
| --- | --- |
| Worker cold load (first request after the API started; weights already on disk) | 15.9 s until the worker reported ready; the first request, with two short sentences, took 62.9 s end to end |
| Generation per sentence, 20 Hindi read-back sentences | mean 35.0 s (range 22.8 to 54.1 s) for a mean of 5.0 s of speech |
| Real-time factor | 6.96 over the 20 sentences (6.2 to 7.2 for 19 of them; the first, 10.7, ran while the speech recogniser was also loading) |
| Speech rate | about 12 characters per second, so a clip is never longer than its text suggests (no runaway audio seen) |
| One sentence each in Marathi, Tamil, Bengali, Telugu (61 to 79 characters) | 5.5 to 6.1 s of speech in 39 to 42 s (about 7x) for Tamil, Bengali and Telugu; Marathi took 108.5 s for 5.7 s of speech (19x) while an unrelated test suite was running on the same machine |
| Long request, 421 characters of Hindi (8 sentences) | 34.5 s of speech in 215.7 s (RTF 6.3), split one sentence at a time, no timeout |
| Request over 600 characters | refused at once with `TEXT_TOO_LONG` (HTTP 422) and a message to split the text, before the model is loaded |

Indic-Speak is for Studio renders and pre-rendered clips, not live calls.

## Voice cloning time

Method: 10 runs of `/api/voices/clone` with a synthetic reference and consent clip (Kokoro am_michael reading a neutral passage and the consent sentence; no real person). Clone time includes the consent check, transcribing the reference and building the voice; then one short sentence is generated with the new voice. Run 1 may include model loading. Every created voice was deleted. Failed runs: 0.

Run 2026-10-15T06:12:15, commit 9ff35a2 (uncommitted changes), LLM: mlx mlx-community/Llama-3.2-3B-Instruct-4bit.

|  | Clone: consent clip to voice (s) | First generation with the new voice (s) |
| --- | --- | --- |
| First run (cold) | 1.8 | 8.0 |
| Warm p50 (n=9) | 2.4 | 6.5 |
| Warm p95 | 2.7 | 7.1 |

## Stability soak

Method: a weighted mix of Kokoro English and Hindi generation, spoken agent turns, transcription, one request at a time with short random pauses; process memory and `/api/health` sampled every 60 s. A crash is the API becoming unreachable and coming back, or its process id changing. Requests answered 429 by the API's rate limiter (0) are not counted as failures.

Run 2026-10-15T06:14:42, commit 9ff35a2 (uncommitted changes), LLM: mlx mlx-community/Llama-3.2-3B-Instruct-4bit.

| Measure | Result |
| --- | --- |
| Duration | 45.0 min (planned 45) |
| Requests | 733 (733 OK) |
| Failure rate | 0.00% (0 failures; by cause: none) |
| Crashes / restarts (target 0) | 0 |
| Health-check failures | 0 |

Latency by request type:

| Request | n | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- |
| kokoro_en | 260 | 682 | 774 |
| kokoro_hi | 178 | 935 | 1105 |
| turn | 176 | 3456 | 7202 |
| transcribe | 119 | 320 | 715 |

Memory (RSS, MB; trend is a linear fit):

|  | Start | End | Max | MB per minute |
| --- | --- | --- | --- | --- |
| API process | 2208 | 3160 | 3188 | +6.08 |
| Child processes (model workers) | 1782 | 2494 | 4065 | +9.88 |
| Total | 3990 | 5655 | 6138 | +15.96 |

## Model memory, measured 27 Oct 2026

Method: `backend/bench/model_memory.py` starts each engine the way the app does (the app's own worker classes and loader functions, one at a time, stopped and waited for between engines), runs one small real request, waits 4 s, then reads the process memory: **footprint** (macOS physical footprint, what Activity Monitor calls Memory: dirty + compressed + swapped pages, GPU buffers included) and **RSS** (pages resident right now). Footprint is the figure to size an engine by: RSS reads low for memory-mapped weights (Indic-Speak 66 MB, Whisper 162 MB) and for pages the system has compressed. Peak is the highest footprint sampled (every 0.25 s) from before the start to just after the first request. In-process models (the API's own Python process) are measured in a fresh process as a delta over the process after its imports. The memory guard was switched off and nothing was downloaded (`HF_HUB_OFFLINE=1`). Run on the Apple M4 MacBook, 16 GB, macOS 15.8.1, on commit 2148228 plus this branch's changes.

**The machine was not quiet.** Two other agents were running tests and builds on it (load average 2.6 at the start, 6.5 at the end; a 1.5 GB Python process and two browsers open), swap was already 3.2 GB at the start and 5.4 GB at the end, and "available" memory swung between 2.9 and 8.8 GB from one engine to the next. So the system columns below are context only (they move by 1 to 4 GB for reasons that are not the engine); the per-process footprint is not affected by that, except that a process under swap shows its swapped pages in the footprint but not in RSS. One run per engine; the raw file is `backend/bench/results/model_memory-20261027-run1.json` (git-ignored).

| Engine (app key) | Device | Footprint (MB) | RSS (MB) | Peak footprint (MB) | Start / first request (s) | System while running: available, swap, compressed (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| Kokoro 82M, English and Hindi (`kokoro`, MLX worker) | Apple GPU | 937 | 531 | 2,076 | 0.6 / 5.0 | 4,901 to 4,087; 3,191 to 3,191; 3,795 to 4,734 |
| Whisper large-v3-turbo only (the new default, strategy `single`) | Apple GPU | 1,712 | 162 | 2,284 | 4.6 / 8.0 | 4,531 to 4,604; 3,191 to 4,851; 4,581 to 3,771 |
| Whisper small + large-v3-turbo (before this change) | Apple GPU | 2,202 | 263 | 2,903 | 4.7 / 0.6 | 5,121 to 5,149; 4,851 to 4,835; 3,632 to 3,227 |
| Whisper small alone (information; the app never runs it alone) | Apple GPU | 612 | 885 | 1,153 | 2.1 / 0.7 | 6,234 to 5,260; 4,819 to 4,747; 2,540 to 2,471 |
| Qwen3-TTS 0.6B 8-bit voice clone (`qwen3_0_6b`, MLX worker) | Apple GPU | 2,393 | 1,298 | 2,838 | 5.2 / 2.4 | 8,197 to 3,746; 4,731 to 4,731; 2,085 to 3,249 |
| DhVaani 0.5 clone worker (`dhvaani`) | GPU / CPU, PyTorch | 874 | 745 | 1,968 | 6.2 / 10.8 | 4,238 to 5,636; 4,731 to 4,723; 2,682 to 2,764 |
| Chatterbox Multilingual clone worker (`chatterbox_ml`) | CPU, PyTorch | 4,074 | 3,438 | 4,495 | 19.2 / 41.9 | 4,748 to 4,536; 4,723 to 5,519; 2,570 to 2,659 |
| Indic-Speak 3.8B, full precision (`indic_speak`) | GPU / CPU, PyTorch | 7,658 | 66 | 7,985 | 28.8 / 36.7 | 7,464 to 3,857; 5,503 to 7,290; 1,449 to 4,022 |
| Local language model server, Llama 3.2 3B 4-bit (live replies) | Apple GPU | 2,151 | 2,338 | 2,255 | 3.7 / 1.5 | 7,806 to 4,980; 7,210 to 7,170; 1,826 to 1,609 |
| Background language model, Gemma 4 E4B 4-bit (`background_llm`) | Apple GPU | 4,565 | 4,731 | 4,791 | 6.4 / 0.3 | 6,792 to 2,889; 7,114 to 7,066; 1,511 to 2,023 |
| API process after `import backend`, nothing loaded | CPU | 232 | 297 | 232 | | |
| BGE-M3 embeddings, fp16 copy (in the API process; delta) | CPU | +592 | +1,303 | | 3.6 | |
| Reranker gte-multilingual (delta) | CPU/GPU per `RERANKER_DEVICE` | +1,525 | +706 | | 4.1 | |
| AudioSeal generator + detector (delta) | CPU | +1 | +81 | | 0.1 | |
| ECAPA speaker model (delta) | CPU | about 0 | +48 | | 0.0 | |
| MiniCheck-RoBERTa answer audit (delta) | CPU | +307 | +1,549 | | 5.8 | |

What follows from it:

- Every engine stopped cleanly (the worker processes were gone within seconds). The old guard estimates were low for the four biggest: Indic-Speak 7,000 (measured 7,658), Chatterbox 3,000 (4,074), Qwen3 clone 1,500 (2,393) and the background model 3,500 (4,565); DhVaani was high (1,500 against 874) and Kokoro low (600 against 937). The 27 Oct "2,540 MB free, 3,000 MB needed" refusal of the clone worker was therefore right for the wrong reason: it really needs about 2.4 GB plus the reserve, and nothing in the old numbers said so.
- With everything `./bonclay warm` loads and the local models running, the workers alone are Kokoro 0.9 + Whisper 2.2 + clone 2.4 + local model 2.2 GB = 7.7 GB, plus the API process (about 1.2 GB with BGE-M3, AudioSeal and the speaker model loaded): 9 GB on a 16 GB machine that macOS, the browser and the editor already take 5 to 6 GB of. That is the swap.
- The two Whisper sizes cost 490 MB more than the large one alone (2,202 against 1,712). The default strategy `single` never runs the small one.
- Peaks are well above steady state for Kokoro (2.1 GB), DhVaani (2.0 GB) and the Whisper worker (2.3 to 2.9 GB): the first request compiles MLX kernels. The guard's reserve (1,500 MB, x1.5 under warn pressure) is what covers that.
- Not measured: IndicF5 (not installed here; the estimate stays 3,000), the PyTorch Kokoro fallback, CPU Whisper (openai-whisper is not installed), CUDA and Windows.

### What the guard uses now

`core/memory_guard.py` defaults, each the steady footprint above plus 10 %, rounded to 100 MB: Indic-Speak 8,400 (was 7,000), Chatterbox 4,500 (3,000), Qwen3 clone 2,600 (1,500), background model 5,000 (3,500), DhVaani 1,000 (1,500), Kokoro 1,000 (600), reranker gte 1,700 (1,200). Every `MEMORY_NEED_<ENGINE>_MB` override still works.

On macOS the guard also reads the kernel's memory-pressure level (`kern.memorystatus_vm_pressure_level`: 1 normal, 2 warn, 4 critical): at warn the reserve is x1.5; at critical a heavy engine (estimate of 1,500 MB or more, `MEMORY_GUARD_HEAVY_MB`) is refused unless an idle engine was unloaded to make room. Idle engines are now unloaded largest (estimated size x idle seconds) first instead of least recently used first. Linux and Windows behave as before.

### What `./bonclay warm` loads now (with a Groq key set)

| | Before | Now | Saved (measured footprint) |
| --- | --- | --- | --- |
| Local Whisper | small + large-v3-turbo worker, 2,202 MB | not loaded; starts (4.6 s) on first use if Groq fails. `WARM_LOCAL_ASR=on` loads it | 2,202 MB |
| Local language model server | started, 2,151 MB | not started; starts (3.7 s) on first use if Groq fails. `WARM_LOCAL_LLM=on` starts it | 2,151 MB |
| Without a Groq key: Whisper | 2,202 MB | large-v3-turbo only, 1,712 MB (`MLX_ASR_RESIDENT=both` restores both) | 490 MB |

So with a Groq key `warm` holds 4.4 GB less (Whisper 2.2 + local model 2.2), without one 0.5 GB less. Kokoro, the embedder, AudioSeal, the speaker model and the clone worker are loaded as before.

### The 16 MB per minute of the 45-minute soak: not reproduced on this code

The soak above (15 Oct) measured RSS, which on a Mac under memory pressure moves with compression and swap, and it started with models not yet loaded (the API's RSS grew from 2.2 to 3.2 GB as BGE-M3, AudioSeal and the speaker model were first used). The same day's commit b9d5825 had already capped the `mlx_lm` prompt cache and the MLX buffer pools and kept Whisper models resident. Re-measured on the current code, 27 Oct:

| Test | Requests | Result |
| --- | --- | --- |
| API process, `bench/leak_hunt.py`: the real app, routes, retrieval with the real BGE-M3, AudioSeal, receipts and storage in this process; language model, speech recognition and Kokoro faked; 12 min, after 2 min of warm-up | 2,182 | footprint +2.1 MB/min (893 to 910 MB over 9 min) = 0.01 MB per request, then one step of +312 MB once, 9 min in (a lazy load; not a slope). RSS fell 30 MB/min (swap churn) |
| The same, 8 min, after this branch's bounds | 1,656 | footprint +1.5 MB/min (1,179 to 1,187 MB) |
| The same with `tracemalloc` (9 min) | 647 | Python heap +1.4 MB in total, all of it json for stored conversation rows and uuid strings; no structure grew |
| Model workers, `bench/worker_soak.py`: Kokoro, Whisper and the local model with real requests, 8 min, after 1.5 min of warm-up | 831 | footprint slope per minute: Kokoro -0.1, local model -2.0, Whisper -27 (it oscillates 1,720 to 2,020 MB with the MLX buffer pool); RSS also flat or falling |

The rate is two orders of magnitude lower per request than the soak (about 0.01 MB against 0.37 MB). What was found and bounded anyway: `_FILLER_CLIPS` (audio keyed by voice and text, no cap) now holds at most 64 clips, and the suggested-questions cache (`routers/templates._questions_cache`, one key per edit of every agent's documents) at most 128 entries. Everything else the grep of module-level dictionaries turned up already had a cap (`_known_calls`, caller memory, the widget and WhatsApp limiters, the consent challenges, the stream turn owners, the disclosure clips). `tests/test_memory_growth.py` runs 130 fake turns and fails if the Python heap grows more than 1.5 MB or threads pile up. A 45-minute soak with the real language model and speech recognition on the new defaults has not been run: that is the remaining check.

## Speech recognition strategy

This is the offline speech-recognition path (MLX Whisper on the Apple GPU). With a `GROQ_API_KEY` set, the live app uses Groq's Whisper instead and none of this applies. The worker keeps two models in memory (about 2 GB together): `whisper-small` (fast) and `whisper-large-v3-turbo` (accurate). Strategies, set with `MLX_ASR_STRATEGY=hybrid|accurate|auto|fast`:

- **hybrid** (the previous default): small decodes the whole clip first. If it heard an Indian language, large-v3-turbo transcribes the clip again.
- **accurate**: one large-v3-turbo pass, language auto-detected by that model.
- **auto** (the new default): small only detects the language (one encoder pass, no text decoded, the audio file is decoded once); an Indian language then gets one large-v3-turbo pass, anything else one small pass.
- **fast**: small only.
- A caller that passes an Indian-language hint (`language=hi`) always gets one large-v3-turbo pass; an English hint gets one small pass. That path is the same in every strategy and is column (c) below.

Method: `backend/bench/asr_compare.py` starts its own worker (no API), makes 40 synthetic clips with Kokoro (20 Hindi with `hf_alpha`; 10 English and 10 Hinglish, that is Roman-script Hindi, with `af_heart`) and adds a noisy copy (white noise, SNR 20 dB) of every second clip, 60 clips in all. Every clip goes through every strategy, interleaved clip by clip in a rotating order, 3 rounds, after one warm-up request per strategy; a clip's latency is the median of its 3 rounds (client wall time). Errors: CER over Devanagari letters for Hindi, WER (lowercase, no punctuation) for English and Hinglish, both with jiwer. Run 2026-10-15 on Apple M4, 16 GB, macOS 15.8.1, uncommitted changes on top of 88e5d4d. **The clips are synthetic text-to-speech, not human speech, and the machine was busy with other jobs during the run (load average 5.2 before, 3.6 after), so absolute seconds are higher than on an idle machine (an earlier single-round run on a quieter machine gave Hindi p50 1.9 s hybrid, 1.4 s hinted, 1.6 s auto). The differences between strategies are paired on the same clips and are the part to trust.**

| Clips | Strategy | n | Latency mean (s) | median (s) | p95 (s) | Error mean | Error median | Language id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hindi (CER) | (a) hybrid | 30 | 2.94 | 2.69 | 4.75 | 0.079 | 0.053 | 30/30 |
| Hindi (CER) | (b) accurate | 30 | 2.92 | 2.80 | 3.39 | 0.079 | 0.053 | 30/30 |
| Hindi (CER) | (c) hinted `hi` | 30 | 1.65 | 1.61 | 1.96 | 0.079 | 0.053 | given |
| Hindi (CER) | (d) auto | 30 | 2.09 | 1.91 | 3.45 | 0.079 | 0.053 | 30/30 |
| Hinglish (WER) | (a) hybrid | 15 | 0.83 | 0.74 | 1.16 | 0.799 | 0.778 | n/a |
| Hinglish (WER) | (b) accurate | 15 | 4.18 | 3.48 | 5.86 | 0.899 | 0.875 | n/a |
| Hinglish (WER) | (d) auto | 15 | 0.81 | 0.67 | 1.15 | 0.799 | 0.778 | n/a |
| English (WER) | (a) hybrid | 15 | 0.63 | 0.61 | 0.69 | 0.093 | 0.000 | 15/15 |
| English (WER) | (b) accurate | 15 | 2.58 | 2.44 | 2.91 | 0.061 | 0.000 | 15/15 |
| English (WER) | (c) hinted `en` | 15 | 0.56 | 0.46 | 0.81 | 0.093 | 0.000 | given |
| English (WER) | (d) auto | 15 | 0.65 | 0.62 | 0.81 | 0.093 | 0.000 | 15/15 |

Rows cover each language's 20 or 10 clips plus their noisy copies (Hindi 30, English and Hinglish 15 each); the 10 noisy Hindi clips alone scored CER 0.087 with every strategy, 0.083 to 0.087 across runs. Paired against hybrid on the same clips: Hindi accurate -0.02 s, auto -0.85 s, hinted -1.29 s, with CER difference 0.0000 for all three (the Hindi text comes from the same large-v3-turbo pass in all of them, so the transcripts are identical); English auto +0.02 s with identical WER; English accurate +1.95 s (large-v3-turbo on English is 4 times slower than small, with WER 0.061 against 0.093). Language id was right on every Hindi and English clip in every strategy; all 15 Hinglish clips were heard as English by every strategy and none was transcribed in Devanagari. Hinglish WER is poor for all strategies (0.80 to 0.90) because the English voice reads Roman Hindi with an English accent, so this row says little about real Hinglish speakers and nothing about Hindi-accented speech. 3 rounds, one machine, no human speech, English limited to 10 sentences: treat differences under about 0.1 s as noise.

Decision: the default is now **auto**. Single-pass large-v3-turbo without a hint (accurate) is not faster than hybrid (-0.02 s for Hindi) because (we did not time the parts separately) that model's own language detection appears to cost about as much as small's whole first pass, and it makes English 4 times slower, so it is not the default. Auto removes the wasted small decode: Hindi is about 0.85 s (about 30 percent) faster than hybrid at identical CER (difference 0.0000, inside the +0.01 limit), and English and Hinglish cost the same as hybrid. Callers that already know the language should pass it: Hindi with a hint is another 0.4 s faster. Set `MLX_ASR_STRATEGY=hybrid` for the old behaviour. Reproduce with `backend/.venv/bin/python backend/bench/asr_compare.py` (about 30 minutes on a busy machine; `--quick` for a smoke test, `--rounds 1` for a faster run).

## Speech recognition on a shorter encoder window (single pass)

Question (idea 3 in `docs/OPTIMIZATION_RESEARCH.md`): can one whisper-large-v3-turbo pass on an 8 to 10 s encoder window, with the language detected in the same pass, replace today's `auto` strategy (whisper-small language id, then one turbo pass for Hindi or one small pass for English) without losing accuracy?

Method: `backend/bench/asr_window_check.py`. 20 English clips (Kokoro af_heart) and 20 Hindi clips (Kokoro hf_alpha), fixed sentences from `bench/readback.py`, 4.0 to 6.9 s each, plus a 20 dB SNR noisy copy of every second clip (60 clips in all, 40 clean). No real person's voice. One process, models warm, audio already decoded, 3 interleaved rounds per clip (latency is the per-clip median). Error rate is corpus-level: WER for English (lowercase, punctuation removed), CER for Hindi (Devanagari letters and marks). The machine was in normal use by other work during the runs (load average about 2), so absolute latencies are higher than the research's idle-machine numbers; the comparison between variants is what counts. Run of 19 October 2026, Apple M4, 16 GB.

Variants: **baseline** = production `auto` today; **turbo30** = one turbo pass, 30 s window, language detected in the decoder; **trim8** = the research's proposal (windows 8/10/15/20/30 s, clip plus 1.5 s); **trim10** = windows 10/15/20/30 s. Both trim variants re-run on the 30 s window if the compression ratio is above 2.4, the average log-probability below -1.0 or the text empty.

| Clips | Variant | Error rate | Language id | Latency median (s) | p95 (s) |
|---|---|---|---|---|---|
| English clean, 20 (WER) | baseline | 7.39% | 20/20 | 0.94 | 1.00 |
| | turbo30 | 7.00% | 20/20 | 1.75 | 1.82 |
| | trim8 | 8.17% | 20/20 | 0.52 | 0.57 |
| | **trim10** | 7.39% | 20/20 | 0.59 | 0.63 |
| English noisy, 10 (WER) | baseline / trim10 | 5.38% / 6.15% | 10/10 each | 0.95 / 0.60 | 1.01 / 0.63 |
| Hindi clean, 20 (CER) | baseline | 7.29% | 20/20 | 2.03 | 2.82 |
| | turbo30 | 7.72% | 20/20 | 1.64 | 2.26 |
| | trim8 | 9.73% | 20/20 | 0.64 | 0.99 |
| | **trim10** | 7.93% | 20/20 | 0.70 | 1.01 |
| Hindi noisy, 10 (CER) | baseline / trim10 | 8.28% / 8.28% | 10/10 each | 2.04 / 0.71 | 2.53 / 0.92 |

Paired over all 30 clips per language (clean and noisy), trim10 minus baseline: English error +0.26 points, latency -0.35 s; Hindi error +0.42 points, latency -1.38 s. trim8 minus baseline: English +0.78 points, Hindi **+2.17 points** (latency -0.20 s and -1.44 s). Every clip in every variant was identified as the right language. trim10 never fell back to the 30 s window; trim8 fell back twice (one English clip and its noisy copy).

Readings:

- **The 8 s window is not safe.** With 4 to 6.9 s of speech and the 1.5 s margin, 36 of 40 clean clips used the 8 s window, and Hindi CER rose by 2.2 points (garbled words such as "पुने से मूं भी जाने वाली"). Moving the smallest window to 10 s brought it back to +0.4 points. A diagnostic pass (clean clips only, one run, no timing) with a 15 s smallest window scored Hindi 6.66% and English 7.39%, i.e. at or below baseline, so there is a trade between window size and accuracy that this set cannot resolve finely.
- **Most of the English difference is number formatting, not mishearing.** Turbo writes "10 minutes", "4 hours" where the small model writes "ten", "four" (the references are spelt out), so it is counted as an error. trim10 had 9 of 20 clean clips with at least one edit against 7 for baseline, and the extra ones are digits. The same happens in Hindi ("12 संतरे" for "बारह संतरे"), and one clip comes back with "order" in Latin script (turbo30 does the same). `language_policy` keys off script, so this matters if it is switched on.
- **Latency.** The single pass removes the small model's language-id pass and the 30 s encoder: Hindi about 2.0 s to 0.7 s, English about 0.94 s to 0.59 s. Turbo on the full 30 s window (turbo30) is faster than baseline for Hindi but slower for English (+0.8 s), because today's English path uses the small model; the short window is what makes the single pass pay off for English.
- Not covered: Hinglish, 8 kHz phone audio, real microphones, clips over 8.5 s (they take the 15 s window or more), more than one voice per language, and the memory saving (the worker still loads the small model for the other strategies).

Decision: trim10 is within the accuracy bar (+0.26 and +0.42 points, under the 1 point limit; trim8 is not) and faster, so the change is implemented as an **opt-in** strategy, `MLX_ASR_STRATEGY=single` (`services/asr_window.py`, `services/mlx_asr_worker.py`), **default off**: the default stays `auto`. The evidence is 40 synthetic clips from two voices, and the research itself asked for real, noisy, phone and Hinglish audio before this becomes the default. Reproduce with `backend/.venv/bin/python backend/bench/asr_window_check.py all --noisy`.

## Audio regression with synthetic callers

Method: the 60 cases of the Hindi/Hinglish adversarial suite are spoken by Kokoro and sent as audio through the real turn endpoint; CER compares what the API heard with the spoken text. Voice variants are different synthetic Kokoro voices and speeds, **not regional accents**. Reply verdicts: judged by groq/qwen/qwen3.8-27b, groq/openai/gpt-oss-20b, groq/openai/gpt-oss-120b with no retrieved context. Failed cases: 0. No previous full run to compare with: this is the baseline.

Run 2026-10-19T17:30:53, commit a9f150c (uncommitted changes), LLM: groq qwen/qwen3.8-27b.

| Script | Cases | CER mean | CER p95 | Audio pass rate | Text pass rate |
| --- | --- | --- | --- | --- | --- |
| all | 60 | 0.152 | 0.415 | 85.2% | 84.7% |
| devanagari | 20 | 0.037 | 0.126 | 72.2% | 70.0% |
| english | 10 | 0.003 | 0.015 | 90.0% | 90.0% |
| mixed | 10 | 0.665 | 2.175 | 100.0% | 90.0% |
| roman_hinglish | 20 | 0.085 | 0.192 | 88.9% | 94.7% |

| Voice variant | Cases | CER mean | Audio pass rate |
| --- | --- | --- | --- |
| American female | 5 | 0.005 | 80.0% |
| British female | 5 | 0.000 | 100.0% |
| Hindi female, fast | 20 | 0.357 | 88.2% |
| Hindi female, normal speed | 20 | 0.067 | 76.5% |
| Hindi female, slow | 10 | 0.060 | 90.0% |

Caveat: the `mixed` script rows are spoken as Devanagari with English words transliterated, while the recogniser writes those words in Latin, which the Devanagari-only CER drops; their high CER is a scoring artefact, not 66% mishearing. The fast Hindi variant's CER is also the worst (speed 1.15).

## Retrieval and answer-checker model choice, 19 Oct 2026

What was compared: BGE-M3's sparse and multi-vector (ColBERT) heads in the fusion, three multilingual rerankers, and three
answer checkers for the post-call audit. All runs were offline processes on the M4 (16 GB), one model at a time, while the
app's backend and frontend were running. Licences were checked on the Hub first: bge-m3 MIT; bge-reranker-v2-m3,
gte-multilingual-reranker-base and its code repository Alibaba-NLP/new-impl, Qwen3-Reranker-0.6B and HHEM-2.1-Open are
Apache-2.0; both MiniCheck models are MIT. Non-commercial models were not tried. Two question sets were used:
`retrieval_questions.json` has 45 questions on three templates, cut into 40-word passages (rq). `doc_qa_questions.json` has
113 answerable and 20 unanswerable questions on four documents, cut into structured passages (dq). Both are small, so
treat about ±3 questions as noise.

### 1. BGE-M3 sparse and multi-vector heads: on, multi-vector only

The head files are `sparse_linear.pt` and `colbert_linear.pt` (2.1 MB together), fetched with
`python -m services.m3_modes --download-heads`. The table gives answerable questions with the right passage first
(hit@1) and with it in the top 3 (hit@3), plus MRR. Ranking uses fused weights (dense : sparse : multi-vector : BM25),
renormalised. Script: `bench/retrieval_modes_bench.py` plus a weight grid in the same harness.

| Fusion | rq hit@1 / hit@3 / MRR (45) | dq hit@1 / hit@3 / MRR (113) |
|---|---|---|
| dense + BM25, 0.8 : 0.2 (previous behaviour; also what the new weights give without the heads) | 36 / 44 / 0.877 | 88 / 111 / 0.918 |
| + sparse (0.4 : 0.1 : 0 : 0.1) | 35 / 44 / 0.866 | 88 / 111 / 0.917 |
| + multi-vector (0.4 : 0 : 0.4 : 0.1) | 36 / 44 / 0.876 | 91 / 111 / 0.930 |
| all four (0.4 : 0.1 : 0.4 : 0.1, the old code default) | 36 / 43 / 0.874 | 91 / 111 / 0.933 |
| BGE-M3 paper (1 : 0.3 : 1, no BM25) | 34 / 43 / 0.856 | 90 / 112 / 0.928 |
| RRF over all four | 33 / 43 / 0.842 | 94 / 111 / 0.942 |
| **chosen: 0.4 : 0 : 0.3 : 0.1** | **36 / 44 / 0.880** | **91 / 111 / 0.932** |
| neighbours: 0.45 : 0 : 0.35 : 0.1, 0.4 : 0 : 0.25 : 0.1, 0.4 : 0 : 0.4 : 0.15 | 36 / 44 / 0.876-0.880 | 91 / 111-112 / 0.932 |
| 0.4 : 0.05 : 0.3 : 0.1 | 36 / 44 / 0.880 | 92 / 111 / 0.936 |

- **Decision.** The heads are on (`M3_MODES=on` was already the default and was inert without the files). The new default
  weights are `dense 0.4, sparse 0, colbert 0.3, bm25 0.1`. This is the only kind of setting that ties or beats dense +
  BM25 on every metric of both sets: dq gains 3 first-place hits and 0.014 MRR, and rq is level. Its neighbours are within
  one question, so it sits on a plateau rather than a lucky point.
- **Why sparse is off.** The sparse weights add nothing over BM25 on the Hinglish/Devanagari normaliser: at 0.1 they cost
  one rq question, and at 0.05 they gained one dq question, which is noise. Sparse is still computed and can be turned on
  with `RETRIEVAL_WEIGHTS`.
- **Fallback.** Without the heads, or before an agent's passage vectors are ready, dense 0.4 and BM25 0.1 renormalise to
  exactly the previous 0.8 : 0.2.
- **RRF over all four** gains 3 more on dq but loses 3 on rq. It does not win on both sets.
- **Cost per query.** Encoding a query in one pass with every output: 44 ms median, 82 ms p95, against 43 / 85 ms for the
  dense vector alone (the same forward pass). Fusion and scoring: 0.3-0.4 ms, against 0.05 ms before. On a synthetic
  50-page document (208 passages): 1.9 ms median, 2.7 ms p95, against 0.45 ms.
- **Passage-vector cache.** 108-204 KB per passage (4.3 MB for 40 passages, 8.0 MB for 39 structured passages). A
  50-page document is about 82 MB, which fits under the default `M3_CACHE_MB=256`. Vectors are computed once per agent in
  a background thread, using a temporary float32 copy (`M3_BULK_NEED_MB=2600` of free memory) and kept on disk.

### 2. Reranker: gte-multilingual-reranker-base, gated, top 3, on Apple's GPU

The table runs on the fused candidates of the chosen weights. "Gated" is the `RERANKER=auto` gate: confidence "medium",
or a fused top-1 / top-2 gap under 0.10. It fired on 129 of the 178 queries, unanswerable ones included. "All" reranks
every query. Latency is warm, per gated query, measured on the reranker call over the shown number of candidates cut to
180 words. The baseline, fusion only, is hit@1 127/158, hit@3 155/158, MRR 0.880 (rq) and 0.932 (dq).

| Reranker (size) | Candidates | Device | All: hit@1 / hit@3 | Gated: hit@1 / hit@3 | Gated MRR rq / dq | Gated p50 / p95 ms | RSS growth |
|---|---|---|---|---|---|---|---|
| bge-reranker-v2-m3 (568M) | 20 (RERANKER=on) | mps fp16 | 134 / 155 | - | - | - | - |
| | 8 | mps fp16 | 135 / 155 | 132 / 155 | 0.891 / 0.959 | 343 / 421 | 0.2-0.6 GB |
| | 8 | cpu fp32 | 135 / 155 | 132 / 155 | same | 1179 / 3751 | 1.6 GB |
| | 4 | mps fp16 | 135 / 156 | 132 / 156 | 0.891 / 0.960 | 159 / 184 | |
| | 3 | mps fp16 | 136 / 155 | **133** / 155 | 0.891 / 0.964 | 122 / 141 | 0.55 GB |
| | 3 | cpu fp32 | same | same | same | 382 / 438 | 2.1 GB |
| **gte-multilingual-reranker-base (306M)** | 20 (RERANKER=on) | cpu fp32 | 132 / 150 | - | - | - | - |
| | 8 | mps fp16 | 132 / 151 | 131 / 151 | 0.885 / 0.945 | 132 / 145 | 0.5 GB |
| | 8 | cpu fp32 | same | same | same | 399 / 525 | 0.85 GB |
| | 4 | mps fp16 | 133 / 153 | 132 / 153 | 0.893 / 0.951 | 67 / 77 | |
| | **3** | **mps fp16** | 133 / 155 | **132** / 155 | **0.895 / 0.951** | **50 / 59** | 0.5-0.9 GB |
| | 3 | cpu fp32 | same | same | same | 150 / 176 | 0.9-2.0 GB |
| | 3 | cpu fp16 | same | same | same | 782 / 910 | |
| Qwen3-Reranker-0.6B | 20 (RERANKER=on) | mps fp16 | 132 / 152 | - | - | - | - |
| | 8 | mps fp16 | 132 / 152 | 130 / 152 | 0.913 / 0.932 | 1190 / 1496 | |
| | 8 | cpu fp32 | same | same | same | 2607 / 5543 | 2.5 GB |
| | 4 | mps fp16 | 133 / 154 | 131 / 154 | 0.913 / 0.944 | 496 / 597 | |

Readings:

- **Cost scales with the number of passages.** Per passage it is about 17 ms for gte on the GPU, 40 ms for bge, and 125 ms
  for Qwen3. Cutting to 120 words saved nothing, because the passages are short.
- **Reranking 8 passages pushes right answers out of the top 3.** gte at 8 loses 4 top-3 hits and Qwen3 loses 3. With 3
  candidates the model receives the same three passages, only re-ordered, so top-3 recall cannot fall. With 4, bge gains
  one.
- **The budget was about 150 ms p95 on gated queries.** Inside it, only gte (3, 4 or 8 candidates on the GPU) and bge
  with 3 candidates on the GPU qualify. gte with 3 gives +5 first-place hits at 59 ms p95. bge with 3 gives +6 at 141 ms
  p95, so 2.4 times the time buys one question, which is within noise. Qwen3, the best on MMTEB-R in its own report, was
  the weakest here and about 10 times slower.
- **On the CPU nothing fits.** gte with 3 candidates runs 150 ms median and 176 ms p95. fp16 on the CPU is 5 times
  slower.
- **By language** (fusion, then gated with 3 candidates): gte goes en 43→44/53, hi 43→44/51, Hinglish 41→44/54. bge goes
  43→45, 43→45, 41→43.
- **Confidence calibration had to change.** bge's scores collapse on Hinglish: the median top score for answerable
  Hinglish questions is 0.001 (English 0.45, Hindi 0.36). With the old `CONF_RERANK_LOW=0.05`, 74 of 158 answerable
  questions became "low", and the grounding gate would refuse them. No threshold separates them: at 0.001, 35 answerable
  against 13 of 20 unanswerable. gte keeps every question between about 0.3 and 0.55, so it never trips 0.05. The
  reranker-based "low" is now off by default (`CONF_RERANK_LOW=0`, opt in with `RETRIEVAL_CONF_RERANK_LOW`). This was
  latent before: no reranker had been downloaded on this machine, so auto mode never ran.

Decision:

- **Model.** `RERANKER=auto` stays, with `RERANKER_MODEL` defaulting to **Alibaba-NLP/gte-multilingual-reranker-base**.
  It moves the right passage to first place on 5 more of the 158 questions, which meets the bar of 3 or more net, with
  hit@3 unchanged.
- **Settings.** `RERANK_AUTO_CANDIDATES` is 3 (was 8). `RERANKER_DEVICE` is auto: Apple's GPU when torch can use it, else
  `EMBEDDING_DEVICE`. `RERANKER_DTYPE` is auto: fp16 on a GPU, fp32 on a CPU.
- **Remote code.** gte needs `trust_remote_code`. Its model repository (8215cf04…) and its code repository
  Alibaba-NLP/new-impl (40ced75c…) are both pinned in `REMOTE_CODE_RERANKERS`. The code was read before it ran: plain
  torch, no network, no subprocess.
- **Loading.** Models load from the local cache folder. This avoids a Hub call that transformers 4.57 makes for every
  tokenizer loaded by name, which fails with `HF_HUB_OFFLINE=1`. The first call after loading is warmed up in the
  background, because a cold call takes about 0.43 s.
- **Thread safety.** Reranker calls are serialised with a lock.
- **Alternatives.** Both stay selectable: `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`, or `Qwen/Qwen3-Reranker-0.6B`, the
  model card's yes/no prompt and P(yes) scoring in `retrieval.Qwen3Reranker`.
- **Caveat.** The GPU is shared with the MLX speech models during a live call. These numbers were taken with the app idle.

### 3. Answer checker for the post-call audit: MiniCheck-RoBERTa-Large

The labelled set has 84 items. The 74 English items are 37 doc_qa questions, each with a faithful sentence (copied facts)
and an unfaithful one: 13 changed numbers, 5 changed days, 4 times, 3 dates, 1 name, 5 contradictions and 6 invented
details. The 10 Hindi or Hinglish items are 5 questions, each with a faithful and an unfaithful sentence. The premise is
the three passages the agent received for that question, joined and cut to 3000 characters, exactly as `evaluate()`
builds it. Every faithful sentence's evidence is in its premise. The checker runs on the CPU with two threads (as the
audit worker does), one sentence per call. It was loaded with `services/answer_audit.py`'s own code.

| Checker (size, licence) | AUC | Balanced acc. at 0.5 (TPR / TNR) | Best threshold, balanced acc. | ms per sentence p50 / p95 | 8 sentences | RSS |
|---|---|---|---|---|---|---|
| HHEM-2.1-Open (flan-t5-base, Apache-2.0, remote code) | 0.911 | 0.689 (0.97 / 0.41) | 0.90 → 0.851 | 175 / 249 | 1.6 s | 1.0 GB |
| **MiniCheck-RoBERTa-Large (355M, MIT)** | **0.965** | **0.932 (1.00 / 0.86)** | 0.5 → 0.932 | 573 / 670 | 4.5 s | 1.7 GB |
| MiniCheck-Flan-T5-Large (770M, MIT) | 0.947 | 0.905 (0.89 / 0.92) | 0.5 → 0.905 | 682 / 915 | 5.9 s | 4.2 GB |

- **Wrong sentences missed** at each model's threshold: HHEM missed 3 of 5 changed days, 4 of 13 numbers and the name.
  MiniCheck-RoBERTa missed 3 numbers, 1 contradiction and the name. MiniCheck-Flan-T5 missed 3 numbers.
- **HHEM's threshold.** At 0.5, HHEM passed 59% of the wrong sentences. Its 0.90 threshold was tuned on these same 74
  sentences, so 0.851 is optimistic. MiniCheck's 0.5 is the model's own decision point and was not tuned here.
- **Hindi and Hinglish (10 items)** at each model's threshold: HHEM had AUC 0.52 and balanced accuracy 0.50. MiniCheck-RoBERTa had AUC
  0.88 and balanced accuracy 0.70, but flagged 3 of 5 correct sentences. Flan-T5 scored 0.84 and 0.70. The audit stays
  English-only.
- **Chunk size.** A larger MiniCheck-RoBERTa chunk (480 tokens instead of 400) was faster at the median (382 ms) but
  slower at p95 (796 ms) and one item worse. 400 stays.

Decision:

- **Model.** `ANSWER_AUDIT_MODEL` defaults to **minicheck-roberta**, with `AUDIT_THRESHOLD` 0.5. It is the most accurate
  of the three, is neither the largest nor the slowest, and needs no remote code.
- **Speed.** About 0.6 s per sentence, roughly 2-5 s per reply. That is acceptable for a background job that pauses
  during live calls.
- **Pinning.** Every model is pinned to a commit in `answer_audit.MODELS` (`ANSWER_AUDIT_MODEL_REVISION` overrides). The
  pins are MiniCheck-RoBERTa 74c89196…, HHEM de9a5e6b…, and MiniCheck-Flan-T5 96eafd01….
- **Download and load.** The `.bin` weights are read with `torch.load(weights_only=True)`. The model is loaded from its
  cache folder. Loading it by name made transformers fetch a converted 1.4 GB safetensors copy in a background thread.
- **Alternatives.** `hhem` (threshold 0.9) and `minicheck-ft5` (threshold 0.5) stay selectable.

### Disk

- **Kept.** BGE-M3 heads (2.1 MB in the bge-m3 cache), gte-multilingual-reranker-base (600 MB, plus 80 KB of code in
  Alibaba-NLP/new-impl), and MiniCheck-RoBERTa-Large (1.3 GB).
- **Deleted from the Hugging Face cache.** bge-reranker-v2-m3 (2.1 GB), Qwen3-Reranker-0.6B (1.1 GB),
  MiniCheck-Flan-T5-Large (3.1 GB of weights, 5.9 GB on disk with a partial download), HHEM-2.1-Open (418 MB), the
  flan-t5-base tokenizer (3 MB), and the converted MiniCheck-RoBERTa safetensors copy (1.4 GB).
### Now the default (20 October 2026)

`MLX_ASR_STRATEGY=single` (the 10 s window, falling back to the full 30 s window when the result looks unsure) is now the
default; `MLX_ASR_STRATEGY=auto` restores the old behaviour. Two checks since the first run of this section:

* **Phone-line copies.** `asr_window_check.py all --phone` makes an 8 kHz-band copy with 25 dB line noise of every second
  clip (30 clips per language, the "noisy" rows). trim10 against the old strategy: English **-0.52 points** error (7.69% to
  6.15% on the 10 degraded clips) at the same speed; Hindi **+0.14 points** (8.28% to 7.04% degraded, 7.19% to 8.03% clean) and
  **-0.79 s** per clip (1.28 s to 0.48 s). trim8 stays rejected (+2.45 points in Hindi). Still synthetic voices; real
  callers, accents and Hinglish are not covered.
* **The live turn bench, local models only, 30 turns, same machine (Apple M4, 16 GB, no Groq key), before and after:**

| Median (p50) | `auto` (before) | `single` (after) | Change |
|---|---|---|---|
| First audio, all turns | 848 ms | 486 ms | -362 ms |
| First answer sentence, all turns | 2,358 ms | 2,136 ms | -222 ms |
| Full reply, all turns | 3,048 ms | 2,597 ms | -451 ms |
| Hindi: speech recognition | 1,162 ms | 432 ms | -730 ms |
| Hindi: first answer sentence | 3,138 ms | 2,478 ms | -660 ms |
| Hindi: full reply | 3,862 ms | 3,393 ms | -469 ms |
| English: speech recognition | 402 ms | 325 ms | -77 ms |
| English: first answer sentence | 1,828 ms | 1,777 ms | -51 ms |

Raw files: `backend/bench/results/latency_stream-20261020-211457.json` (before) and `-211802.json` (after). One run each,
so differences of about 100 ms are within the noise; the Hindi recognition change is far larger than that.

## Reference points

| Reference point | Target / published figure | BonClay (this report) |
| --- | --- | --- |
| Spoken reply time | under 3 s from the end of the question to the reply being ready (the plan's target) | full reply p50 3.02 s local, p95 5.03 s (16 Oct, client clock); 2.81 s with Groq answering (n = 10, 15 Oct). History: 4.4 s p50 on the morning of 15 Oct |
| Twilio, mouth-to-ear | 1,115 ms target (upper limit 1,400 ms) in Twilio's November 2025 guide to core latency in cascaded voice agents; a published benchmark target, not re-measured here. It is a different measurement from this report's times, so treat it as context, not a like-for-like comparison | not like-for-like: first audio p50 0.84 s, first sentence of the answer p50 2.23 s (client, streaming). Both start the clock when the audio request is sent, not when the caller stops speaking, and BonClay adds speech recognition time that a telephony figure does not include |
| Stability | 45 minutes of mixed load without a crash | 0 crash(es) in 45 min, 733/733 requests (15 Oct) |

### Marathi spelling hints for DhVaani (M1, 2026-10-19)

`backend/bench/pron_lexicon_check.py` renders the four Marathi sentences with ळ (two), ऱ्ह (one) and a plain control through the DhVaani worker (device `mps`, 16 steps, Hindi synthetic Kokoro reference `hf_alpha`, no real voice), once per candidate respelling, reads each back with mlx-whisper large-v3-turbo and scores against the ORIGINAL spelling (`leak_scan`, CER from `bench/scoring.py`). One run, 4 sentences per candidate: a difference of one flag is noise.

| candidate | ळ lost | ऱ्ह lost | flags | mean CER | plain-sentence CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (no hint) | 6 | 3 | 13 | 0.148 | 0.111 |
| ळ → ल़ | 5 | 3 | 12 | 0.129 | 0.111 |
| ळ → ड़ | 7 | 3 | 11 | 0.158 | 0.111 |
| ऱ्ह → र्ह | 6 | 3 | 13 | 0.143 | 0.111 |
| ऱ्ह → रह | 5 | 1 | 9 | 0.138 | 0.111 |
| ळ → ल़ and ऱ्ह → र्ह | 5 | 3 | 12 | 0.119 | 0.111 |
| ळ → ड़ and ऱ्ह → रह | 7 | 3 | 11 | 0.149 | 0.111 |

Reading: only ऱ्ह → रह moved a count beyond one (3 to 1); ळ → ल़ is within noise; ळ → ड़ is not better. The plain control is identical for every candidate, as it should be (no rule matches). **This does not show the hints sound better.** Whisper writes Marathi toward Hindi spelling, so "ळ lost" is high even when the audio is right, and a respelling can change what Whisper writes without changing what a Marathi speaker hears. The two rules in `services/pron_lexicon.py` are therefore hypotheses and the flag `PRON_LEXICON` stays **off**. Listening files from this run (baseline and every candidate, same sentences) are in the shared notes folder `m1_lexicon/` as `<sentence>__<candidate>.wav`; the earlier set `video/dhvaani_marathi/s08-s10` is the baseline for the same three sentences. Turn the flag on only after a Marathi speaker rates a hinted file as closer than its baseline.
