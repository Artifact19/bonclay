# BonClay measurement suite

Four scripts (latency has a streaming mode) that measure the running system over HTTP, and a report writer. They never import backend code, so
they measure what a user gets. Each writes raw per-item results to `backend/bench/results/<name>-<timestamp>.json`
(git-ignored) and prints a short summary. Each result file records the git commit, the machine (CPU, RAM, platform),
`/api/health` (platform, LLM provider and model, loaded models), the settings used and every raw measurement.

## Before you run

```
./bonclay start                      # from the repo root; the API must be on http://127.0.0.1:8000
export BONCLAY_API=http://127.0.0.1:8000   # optional; the default, or pass --api
```

Run everything with `backend/.venv/bin/python` (it needs `requests`, `numpy`, `soundfile`, `psutil`, `jiwer`;
`jiwer` is in `backend/requirements-dev.txt`). Every request sends `storage-mode: local`. Every script has `--quick`,
a tiny version for checking that it runs; quick numbers mean nothing and are flagged as such in the report.

| Script | What it measures | Full run |
| --- | --- | --- |
| `latency.py` | Spoken-turn latency: a spoken question in, the full reply audio ready | `backend/.venv/bin/python backend/bench/latency.py` |
| `readback.py` | Intelligibility: speak a sentence, transcribe it, score WER (English) or CER (Hindi); real-time factor | `backend/.venv/bin/python backend/bench/readback.py --engines kokoro-en,kokoro-hi` |
| `hindi_lexicon_check.py` | Spelling hints for Kokoro's Hindi voice (`services/pron_lexicon.py`): per mispronounced word, 3 carrier sentences spoken with the plain and each candidate spelling, read back through the API's Whisper; `--holdout` checks the shipped rules on other sentences. Start the API with `PRON_LEXICON` off | `backend/.venv/bin/python backend/bench/hindi_lexicon_check.py --api http://127.0.0.1:8003` |
| `kokoro_engine_compare.py` | Kokoro PyTorch vs MLX on the app's own code path, interleaved: per-sentence synthesis p50 for 10 English + 10 Hindi sentences, audio length, memory | `backend/.venv/bin/python backend/bench/kokoro_engine_compare.py` |
| `clone_time.py` | Seconds from consent clip to a usable cloned voice, and the first generation with it | `backend/.venv/bin/python backend/bench/clone_time.py` |
| `soak.py` | Stability: 45 minutes of mixed requests, memory trend, crashes | `backend/.venv/bin/python backend/bench/soak.py` |
| `model_memory.py` | Memory of each engine (footprint and RSS of the worker process, delta for in-process models, system memory, pressure) started the way the app starts it; offline, no API needed. `--list` shows what is installed | `HF_HUB_OFFLINE=1 backend/.venv/bin/python backend/bench/model_memory.py` |
| `leak_hunt.py` | The API process under a mixed load with the language model, speech recognition and Kokoro faked: RSS and footprint slope, `--tracemalloc` shows what grew | `backend/.venv/bin/python backend/bench/leak_hunt.py --minutes 12` |
| `worker_soak.py` | Footprint slope of the MLX workers (Kokoro, Whisper, local model) under real requests | `backend/.venv/bin/python backend/bench/worker_soak.py --minutes 8` |
| `retrieval_eval.py` | Knowledge retrieval hit@1 / hit@3 on `data/retrieval_questions.json` (English, Hindi, Hinglish): MiniLM vs BGE-M3 vs BGE-M3 + BM25 (+ reranker). Offline: imports `services/retrieval.py`, never calls the API | `backend/.venv/bin/python backend/bench/retrieval_eval.py [--reranker]` |
| `synthetic_callers.py` | Audio regression before a release: the 60 Hindi/Hinglish/English adversarial cases spoken by Kokoro (voice variants, not accents), sent as audio through the real turn endpoint; speech-recognition CER plus the Evaluation Lab judge verdict (`--no-judge` skips the judge, `--with-text` adds a typed-text control arm). Local only: needs the models and, for the judge, a Groq or OpenAI key. Compares with the previous full run | `./bonclay regression --with-text` |
| `report.py` | Writes `docs/RESULTS.md` from the newest result of each kind | `backend/.venv/bin/python backend/bench/report.py` |

## What each one does

**latency.py** creates an agent from the `clinic_receptionist` template (or uses `--agent-id`; an agent it creates is
deleted at the end unless `--keep-agent`). It renders 30 short questions (15 English, 15 Hindi, alternating, all
answerable from `backend/templates/agents/clinic_receptionist.md`) to WAV with Kokoro, which is not timed. After two
warm-up turns it posts each one to `/api/conversations/{agent}/turn`, a new conversation each time (or
`--same-conversation`). It records client wall time (request start to JSON received), the server's `total_ms` and
per-stage `timings`, engine, LLM provider, fallback flag, what the server heard and what it replied. The summary
gives p50 and p95 overall and per language, stage p50s and the share within 3 s. This is speech in to full reply
audio ready: nothing is streamed, so it is not time to first audio. Options: `-n`, `--warmup`.

With `--stream` the same questions go to `/api/conversations/{agent}/turn/stream` (server-sent events) and the result
is written to `latency_stream-*.json`. The client clock records when the first `sentence` event arrives (time to
first audio: the first sentence is synthesized, watermarked and ready to play), when the first sentence of the answer
itself arrives, and when `done` arrives (the whole reply ready, same meaning as the table above), next to the
server's `first_audio_ms`. Agents say they are an AI at the start of a call, and that disclosure is its own first
sentence, so first audio is usually the disclosure; "first answer" skips it. `--no-disclosure` turns the disclosure
off on the agent the script creates, so the first sentence is the answer. `report.py` writes this as the table
"Time to first audio (streaming)" and keeps the non-streaming table.

**readback.py** generates 20 fixed sentences per engine (8 to 15 words, with numbers and names), times the generation
and transcribes the audio with `/api/transcribe`. English is scored by WER (jiwer; lowercase, punctuation removed,
numbers compared as written: the sentences spell numbers out, so a transcript with digits counts as errors). Hindi is
scored by CER over Devanagari letters only. RTF is generation seconds divided by audio seconds, so lower is faster
and below 1.0 is faster than real time. Engines: `kokoro-en` (af_heart), `kokoro-hi` (hf_alpha), `clone-en` (needs
`--voice-id`), `indic-hi` (Indic-Speak; slow, about 8x real time on an Apple M4, and its model is several GB to
load on first use). The transcript also records which Whisper model answered.

**clone_time.py** makes a reference clip and a consent clip with the synthetic Kokoro voice am_michael (no real
person), then runs `/api/voices/clone` 10 times and, after each, generates one short sentence with the new voice.
Run 1 is cold, the rest warm. If the API has `GET /api/consent/challenge` it fetches a challenge for each run and
sends `challenge_id` with the challenge text; otherwise it reads the fixed consent sentence. Every voice it creates
is deleted at the end, also after a failure.

**soak.py** loops a weighted mix (Kokoro English 30, Kokoro Hindi 20, spoken turn 20, transcription 15, and with
`--voice-id` cloned-voice generation 15) with random 0.5 to 4 s pauses for `--minutes` (default 45). Every 60 s it
samples the API process RSS, its child processes (model workers) and `/api/health`. It counts requests, failures by
status code or exception, restarts (API unreachable and then back, or its process id changed) and latency p50 and
p95 per request type. The target is 0 crashes.

**report.py** takes the newest result file of each kind and writes `docs/RESULTS.md`: the machine, date and commit, one
table per measurement with its method, and reference points. Anything without a result file says "not measured".

## Caveats

- Numbers depend on the LLM in use: the local model (MLX) on a laptop is far slower than a hosted one (Groq). The
  health block in each result file says which was active; the report prints it with every table.
- First runs download and load models (Whisper, Qwen3-TTS, Indic-Speak). Run `./bonclay doctor` first, and expect
  the cold run of `clone_time.py` and the warm-up turns of `latency.py` to be slow. Warm-up turns are excluded.
- Close other apps. The machine's load average is recorded, but background work still changes the numbers. Run
  long measurements on an idle machine.
- The API rate-limits per route and per client address (transcribe 10 per minute, voices/generate 10, voices/clone 5,
  indic/generate 6). The scripts pace themselves under these limits and wait out a 429; the waiting is never
  counted in a latency number, but it makes `readback.py` and `clone_time.py` take longer than the generation alone.
- Memory (RSS) on macOS drops when the system compresses or swaps pages under pressure, so the soak's memory trend
  is only meaningful on a machine with free RAM. Each sample also records system memory use and swap.
- The soak talks to one API process on this machine (found by its listening port) to read memory; with a remote
  `--api` it skips the memory samples.
- Do not edit code or restart the API while a measurement runs: a restart is counted as a crash by the soak.
- The Hindi voice list of this server has `hf_alpha` only; `latency.py` uses `hm_omega` for half of the Hindi
  questions when the server offers it.
