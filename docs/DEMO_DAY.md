# BonClay — Demo Day Runbook

## 1. Tonight: setup (this Mac has no venv, no `node_modules`, no `.env.local`)

```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && cd ..
cd frontend && npm install && cd ..
```

The shorter route is `./bonclay setup` then `./bonclay start` (see the README). It runs in local mode: no keys, data stored on this computer, and on a Mac with Apple Silicon the agents run fully offline (Whisper and Llama 3.2 3B on the Apple GPU).

Hosted mode only: `frontend/.env.local` needs `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_API_URL`, and `backend/.env` needs `SUPABASE_SERVICE_ROLE_KEY` and `SUPABASE_JWT_SECRET`. `GROQ_API_KEY` is optional (faster speech recognition and a 70B model; without it the local model is used). `HF_TOKEN` is only used to download Indic-Speak. Without Indic-Speak, Hindi and Marathi speech falls back to Kokoro's Hindi voice.

**Indian-language speech:** run `bash backend/scripts/setup_indic_speak.sh` once (about 7 GB; request access on the model page first, see the README). Indic-Speak runs at about 8x real time on an M4, so a sentence takes about a minute, and the first request after idle also loads the 3.8B model (about 40 s). Pre-generate Hindi and Marathi clips in Studio before the demo, or run those parts on the Colab/CUDA machine.

**Hindi cloning:** run `./bonclay setup --chatterbox` once (about 3 GB, no token). Chatterbox runs at about 4-6x real time, so pre-render the Hindi clips in Studio before the demo.

Hosted mode only: run the migrations `backend/supabase/migrations/000` to `008` in order in the Supabase SQL editor (the `match_document_chunks` function is created by `000`; see `backend/supabase/README.md`). If the `consents` insert fails, cloning still works, but the voice card shows "No consent on file". Local mode needs none of this.

**Test Kokoro on this Mac before anything else.** The pip-installed espeak-ng loader looks for its data in a path that exists only on the machine that built it (`.../espeakng-loader/espeak-ng/_dynamic/share/espeak-ng-data/phontab: No such file or directory`), and loading Kokoro directly kills the process. The backend avoids this by pointing Kokoro at the Homebrew copy first (`services/espeak.py`), which needs `brew install espeak-ng`. Do not use the bare `KPipeline(...)` one-liner to check: it bypasses that fix and crashes even when the app is fine. Check it the way the app loads it, from `backend/`:

```bash
python -c "from services.espeak import configure_espeak; configure_espeak(); from kokoro import KPipeline; KPipeline(lang_code='a'); print('kokoro ok')"
```

If that fails, run `brew install espeak-ng` and check again. If Kokoro still fails, demo from the Colab/CUDA machine instead. Do not warm Kokoro live in front of the panel until this check passes. Japanese is an optional extra (`pip install -r requirements-ja.txt`); without it the Japanese voice is simply not offered.

Running on Colab or another GPU box? Set `NEXT_PUBLIC_API_URL` to the tunnel URL and add the frontend origin to `CORS_ORIGINS`. The frontend no longer hardcodes `localhost:8000`.

## 2. Tonight: fill the brackets on slide 17

- **Latency:** run 10 Playground turns and read p50 and p95 in the reply-time panel (Overview). Measured numbers go in `docs/RESULTS.md`; do not quote latency on a slide until they are there.
- **Stability:** with a valid `SUPABASE_TOKEN`, run `backend/.venv/bin/python backend/bench/soak.py --minutes 45` and record the crash count (see `docs/RESULTS.md`).
- **Resilience:** run the Evaluation Lab once on the demo agent and note the score.
- Record a backup screen capture of the full demo.

## 3. Pre-flight (30 min before)

1. Start the backend and frontend. Open **Overview** and click **Warm up** on every engine until all show *Warm*.
2. Log in. Confirm Chrome microphone permission (do one test recording).
3. Have these ready: the demo PDF (hotel FAQ), a clean 5–10 s reference clip, and the backup video in another tab.

## 4. Demo script (≈7 min)

| # | Page | Do | Point at |
|---|---|---|---|
| 1 | Overview | Show engines warm, GPU, storage (Local or Supabase) | Live model-manager state |
| 2 | Voices → Clone | Read the one-time consent sentence and code → upload/record sample → create | "Consent on file" shield |
| 2b | (bonus) | Clone again but read a wrong sentence, or have a different person read the consent | Refusal + "We heard: …", or the speaker-mismatch message |
| 3 | Agents | One-line role: "hotel receptionist for FAQs in English and Hindi" → assign voice → save | LLM-drafted persona |
| 4 | Knowledge | Upload PDF | chunks created |
| 5 | Playground | Hold mic: ask a question from the PDF | Engine badge, Grounded · N sources, stage timings |
| 6 | Playground | Ask something not in the PDF | Agent says it doesn't know (no hallucination) |
| 7 | Evaluations | Run evaluation | Resilience %, prompt-injection and PII rows |

## 4b. Demo: the phone call and the HOD office agent (≈3 min)

**Before (30 min ahead)**
1. `./bonclay start` (it prints the tunnel line) then `./bonclay warm`. In `./bonclay status` the tunnel must show.
2. If the call uses a cloned voice (`PHONE_CLONE_VOICE=1`), have one turn with that agent first so its voice engine is loaded;
   otherwise the first sentence waits.
3. Place **one test call** to the number you will use. A trial Exotel account only rings numbers whitelisted in its dashboard
   and refuses the rest. College Wi-Fi can block the tunnel: use a phone hotspot if the status line shows no tunnel.
4. Open **Approvals** (Agents mode) in a second tab so it is ready.

**Flow**

| # | Where | Do | Point at |
|---|---|---|---|
| 1 | Agents | **Call** on the HOD Office Assistant, pick the number, tick the consent box | The phone rings; the agent says it is an AI and greets in its own voice |
| 2 | The phone | "Is ma'am free on Monday at eleven?" | It checks the schedule, says that time is not free and offers the nearest other times |
| 3 | The phone | Pick one; give a name, that you are a student, the reason and a phone number | It reads back a request number and says it is **pending approval**, not confirmed |
| 4 | The phone | Talk over the agent mid-sentence | It stops at once (barge-in) |
| 5 | Approvals | Approve the request with a note | The request leaves the pending list; with Google Calendar connected it also lands on the calendar |
| 6 | The phone | "What happened to my request?" | The agent reads the head of department's decision and note |

**If the call fails:** open the agent's **Channels, Web call link** (`docs/WEB_CALL.md`) on a laptop and run the same conversation
in the browser: no phone network, no KYC. Keep a short screen recording of a good call as a last resort.

**Say honestly:** the schedule switch "Fill the week with a sample schedule" is on for demos, so the busy slots are made up; turn it
off before real callers. Google Calendar and Google Sheets logging have only been tested against fakes. Calls reach only whitelisted
numbers on a trial Exotel account, and the cloned phone voice answers a few seconds slower than the preset one. Details:
`docs/PHONE_AND_MEETINGS.md` and `docs/DEMO_CHECKLIST.md`.

## 5. Likely viva questions

- **Why not just use ElevenLabs?** It is closed and priced per minute, and its users' voices depend on the vendor (Play.ht users lost theirs in 2025). BonClay is open-weight, speaks 22 Indian languages through Indic-Speak, and costs about ₹0 to run.
- **How does the model manager avoid OOM?** Models load lazily. Under memory pressure the least-recently-used one is evicted. Idle models unload after 300 s. A 60 s minimum residence prevents thrashing. `expandable_segments` reduces CUDA fragmentation.
- **How do you stop someone cloning a voice without consent?** Several checks, in order. The speaker reads a fresh one-time sentence with a code, so an old recording cannot be replayed. Speech recognition checks that the sentence and code were said (the 60% threshold below). A speaker match checks that the consent recording and the voice sample are the same person. Each clone gets a signed receipt, and every generated clip carries an inaudible AudioSeal watermark, so a clip can be checked on the public `/verify` page. Deleting a voice deletes its sample and consent recording. Honest limit: a determined attacker with a live recording of the victim saying the new sentence could still pass, so this is a deterrent and an audit trail, not a guarantee.
- **Why is the threshold 60%?** It tolerates accents and ASR noise ("14th" vs "14"). In tests, unrelated speech scored 6% and the correct phrase scored 94%.
- **Is LLM-as-judge reliable?** It is a heuristic (Zheng et al., 2023) and the UI says so. Use it for regression testing, not certification.
- **Why Groq for ASR and the LLM?** It is optional. With a key, Groq keeps the GPU free for TTS and has low latency on a free tier. Without one, speech recognition runs on the Mac's GPU (Whisper through MLX) and the language model is Llama 3.2 3B on the Apple GPU, so the demo works with no keys and no internet.
- **What happens if Groq is down?** The language model falls through a chain: Groq (Llama 3.3 70B), any OpenAI-compatible server, then the local Llama 3.2 3B. If all of them fail, the agent speaks a short fallback line in the Kokoro voice. ASR failure returns a readable error and never crashes the session.
- **Why open models?** No vendor can take a voice away or change its price, everything can run offline and on your own data, and the licences are checked and allow research and commercial use with attribution (see Licences). It also lets us run the same agent on a laptop with no keys.
- **Data isolation?** A JWT is required on every route, there are ownership checks in the API, and Postgres RLS acts as a second wall.
- **Why Indic-Speak for Indian languages?** Qwen3-TTS supports no Indian languages and Kokoro has one Hindi voice. Indic-Speak (Bodhan AI with AI4Bharat) has a female and a male voice in each of 22 languages, recorded by consenting artists, and reads mixed Indian-language and English sentences naturally.
- **Can it clone a Hindi or Marathi voice?** Hindi yes, Marathi no. Hindi uses Chatterbox Multilingual (Resemble AI, MIT): a consented voice, even an English sample, can speak Hindi. It is slow (about 4-6x real time), so it is for Studio renders and pre-rendered clips, not live calls. Other Indian languages are not cloned yet; they use Indic-Speak preset voices. IndicF5 is an optional engine for a Hindi sample speaking Hindi and is not enabled by default (its dependencies clash with the rest of the backend). Cloning in English and nine other languages uses Qwen3-TTS. See `docs/decisions/indian-language-cloning.md`.
- **What does not work yet?** Cloning in Indian languages other than Hindi. Indic-Speak and Chatterbox are too slow for live calls on a laptop, so live agents speak with Kokoro and the slower engines are for Studio. Reply time is measured but, with the local model, above the 3 s target: see `docs/RESULTS.md`. The Evaluation Lab is a heuristic, not a certification.
- **How fast is Indic-Speak?** About 8x slower than real time on an M4 (a 3.8B-parameter model limited by memory bandwidth), so it suits Studio, not live conversation, on a laptop. On a CUDA GPU it is far faster.
- **Which licences apply?** Kokoro-82M and Qwen3-TTS are Apache-2.0. Chatterbox is MIT. AudioSeal (the watermark) is MIT, weights included. Indic-Speak uses the Indic Open Model License v1.0: free for research and commercial use, but the interface must show "Built with Indic-Speak from Bodhan AI / AI4Bharat." (the Studio and the site footer do), offering it as an API to other organisations needs Bodhan AI's written approval (academic and non-profit use is exempt), and the Llama 3.2 base-model terms also apply. The Llama models use the Llama community licence: show "Built with Llama" (the site footer does). IndicF5 is gated: check its licence on the model hub before any commercial use.
