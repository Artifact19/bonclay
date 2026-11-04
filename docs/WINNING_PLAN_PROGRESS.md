# Winning plan: progress

Tracks the work from "BonClay: The Winning Plan" (15 Oct 2026). Newest first inside each phase.
Status: ✅ done and verified · 🟡 in progress · ⏳ queued · ⛔ blocked (reason given)

## Assumptions made while executing

- **No Groq or Supabase keys are available yet**, so the four unrun modules (Agents, Playground, Knowledge,
  Evaluation) are made to run fully offline: a local language model on the Mac (MLX) and local storage.
  When keys are added to `backend/.env`, Groq and Supabase are used automatically.
- **Commits go straight to `main`** (the project's existing workflow), with private data (`backend/.env`,
  `backend/data/`) never committed because the repository is public.
- **Mac first.** Everything is tested on the team's Apple M4. Windows/Linux with NVIDIA keep working through
  PyTorch CUDA, but are not tested here.

## Phase 1: foundation (credibility)

| # | Item | Status | Notes |
|---|---|---|---|
| 0 | Commit and push the current work | ✅ | `9b5c609` on `main`; private voice data excluded |
| 1 | Run Agents, Playground, Knowledge, Evaluation end to end | ✅ | Run live with no keys: local Llama 3.2 3B (MLX) + local storage. Agent created in 11 s, grounded English and Hindi spoken answers, evaluation 83/100 after built-in guardrails and a rubric judge (was 44, then 28 with a misreading judge). Groq/Supabase used automatically when keys are added |
| 1a | Permanent environments and one-command setup/start | ✅ | `./bonclay setup/start/stop/status/doctor/logs/test` (macOS/Linux) and `bonclay.ps1` (Windows, untested); model workers exit with the API |
| 2 | Decide the Indian-language cloning story | ✅ | Bake-off done → Chatterbox Multilingual for Hindi cloning (works from an English sample, MIT); IndicF5 optional. See `docs/decisions/indian-language-cloning.md`. Integrated: your consented voice speaks Hindi (live check: Whisper read back the sentence with one spelling difference; ~28 s warm for 4.6 s of speech). Live calls keep preset voices |
| 3 | Core tables and vector search in migrations | ✅ | `000_core_schema.sql` + `match_document_chunks`; checked on stubbed Postgres (pgvector itself not exercised). Open: bucket is private but code returns public URLs → switch to signed URLs |
| 4 | Tag a demo-day release | ✅ | `v1.0-demo` (commit f506b5e, CI green): https://github.com/Artifact19/bonclay/releases/tag/v1.0-demo |
| 5 | Correct stale claims (deck, landing, README) | ✅ | Deck: measured numbers, corrected claims, current screenshots, redrawn architecture/turn/router/consent diagrams, Verify and Measured Numbers slides (now in `docs/college/`) |
| 6 | Measure reply time (p50/p95), WER, clone time, 45-minute soak | ✅ | `docs/RESULTS.md`: reply p50 4.4 s (English 3.3 s, Hindi 4.8 s) vs 18.7 s before the speed pass; read-back WER 7.6% (Kokoro en), CER 7.1% (Kokoro hi), WER 8.9% (cloned voice); clone 2.4 s warm; 45-min soak 733/733 OK, 0 crashes. Memory growth fixed (model workers bounded: 8.7 GB → 4.9 GB, flat over 15 min) |
| 7 | Re-check Indic-Speak on the real model | ✅ | No runaway (5.5-7 s of speech for 61-87 characters in hi/mr/ta/bn/te); 421-character Hindi split into 8 sentences and finished; 800 characters refused up front; cold load 15.9 s; ~35 s per sentence (RTF ~7); Hindi CER 0.091 over 20 sentences. Studio shows a loading note during cold start; attribution shown wherever Indic-Speak audio plays |
| 8 | Pre-render proof clips and a backup recording | ⏳ | |

## Phase 2: easy to operate

| # | Item | Status |
|---|---|---|
| 9 | First-run checklist on the Overview | ✅ - five steps tick themselves off (`GET /api/onboarding`) |
| 10 | Health strip that tells you how to fix each problem | ✅ - setup hints with copyable fix commands (`GET /api/setup-hints`); sidebar says "Local mode" |
| 11 | One-click demo mode (sample voice, PDF, agent, evaluation) | ✅ - two template agents with documents and a real attack test; reset button (no sample voice: a voice needs a real person's consent) |
| 12 | Live performance panel (stage timings, p50/p95) | ✅ - Overview and Playground; first honest numbers: p50 18.7 s, p95 22.2 s vs 3 s target (speed pass next) |
| 13 | Agent templates (clinic, EMI reminder, farmer helpline, admissions, tutor) | ✅ - instant, no language model needed; sample documents for fictional organisations |
| 14 | Hinglish mode | ✅ - Roman Hindi detected and converted (local LLM + 650-word dictionary fallback); Studio suggestion; agents answer Hinglish callers in Hinglish with a Hindi voice |

## Phase 3: trust

| # | Item | Status |
|---|---|---|
| 15 | Consent hardening: speaker match between consent and sample, random challenge word | ✅ - one-time challenge sentence with a code; ECAPA speaker match (live: same 0.88 accepted, different 0.14 refused); not yet tried on real microphone recordings |
| 16 | Plain-language notice, delete-my-voice, AI-generated label on every player | ✅ - delete removes sample, embedding and consent recording; agents say they are an AI on the first reply (toggle) |
| 17 | Watermark every output (AudioSeal) + consent receipt + Verify page | ✅ - AudioSeal (MIT incl. weights), HMAC-signed receipts, public `/verify`; survives MP3 down to 32 kbps and 8 kHz resampling in tests |
| 18 | Signature: say it once, hear it in four Indian languages | ✅ - `/dashboard/say-it-once`: one sentence, translated and spoken in each chosen language; your consented voice where an engine can clone it (Qwen3 for its 10 languages, Chatterbox for Hindi), credited preset voices otherwise, and the page says which. Every clip watermarked with a receipt. Live run with the owner's voice: Hindi read back essentially exact, French in the owner's voice, Marathi preset (local model's Marathi is weak; a Groq key improves it). Guards: refuses a translation whose weekday changed |

## Phase 4: front door and integrations

| # | Item | Status |
|---|---|---|
| 19 | Landing: playable languages, live consent demo, measured-numbers strip, honest comparison | ✅ - eight playable cards (preset voices only, Indic-Speak credited, 372 KB), numbers strip from RESULTS.md with its caveat, comparison with a "what BonClay does not do yet" block; Studio, Agents and consent screenshots retaken from the current app |
| 20a | Demo readiness: warm every model in one step | ✅ - `./bonclay warm` or the Overview card (`POST /api/warmup`): all ready in ~20 s; first streamed turn afterwards: first audio 0.8 s, full reply 3.7 s |
| 20 | Streaming replies / hands-free talk | ✅ - streaming: warm turn answers at 2.7 s, first audio 0.5 s; hands-free Playground mode: voice-activity detection sends each utterance, ignores the agent's own audio, speak to interrupt (echo and barge-in tested in unit tests only, not on real speakers) |
| 21 | Tools through MCP (local appointment server) | ✅ - clinic agent books, finds and cancels (cancel only after a spoken yes); live booking verified; tool calls stored with the turn |
| 22 | Phone channel (needs a telephony account; architecture slide until then) | 🟡 - architecture slide in the deck (Exotel/Twilio media stream → gateway → the streaming turn pipeline); not built |

## Research improvements and trials (15 October 2026)

Built from `docs/IMPROVEMENTS_RESEARCH.md`; measured before/after and trial verdicts in `docs/TRIAL_REPORT.md`
(local-only full reply p50 4.68 → 3.77 s, Hindi first audio 2.45 → 1.73 s, knowledge top-3 88.9 → 97.8%; Groq-answered
full reply 2.81 s).

## Robustness (15 October 2026)

- Groq works again (Cloudflare refused Python's default User-Agent; Llama 3.3 70B retired → qwen/qwen3.8-27b); the warm-up says when a provider fell back.
- Turns load only the current conversation's last 6 turns; a stuck local-model call can no longer block later calls; cloud audio links are signed.
- Repo clones on Windows (a folder name ended in a space); private voice files and old scripts removed; CI green (real requirements, Node 24).

## Landing and Indic-Speak follow-ups

Items 7 and 19 are done and committed (`5d572a5`). Follow-ups, resumed on 15 October 2026:

1. ✅ Marathi landing clip re-rendered with the same preset (70 s on a busy machine) and compared by read-back: no better than the original, so the original stays.
2. ✅ Indic-Speak speed comments in `backend/backend.py`, `backend/services/indic_speak.py` and the README now say ~7x (measured RTF 6.96).
3. ✅ `backend/bench/report.py` now uses the fullest real run of each kind (a later one-question or 15-minute run no longer replaces the full one) and builds read-back from each engine's newest run, so no row is lost. It does not write the hand-written Indic-Speak section of `docs/RESULTS.md`; keep that section if you regenerate.
4. ✅ `frontend/tsconfig.tsbuildinfo` is no longer tracked (`*.tsbuildinfo` in `.gitignore`).
