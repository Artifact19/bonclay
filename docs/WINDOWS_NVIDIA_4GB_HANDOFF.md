# BonClay: context transfer for the Windows + NVIDIA (4 GB VRAM) test machine

Written 1 November 2026 by the Claude Code session on the Mac that built most of BonClay. You (Claude Code on a Windows PC with an NVIDIA card, 4 GB VRAM) already have the repo. This file holds what the repo does **not**: the history, the measured numbers, the decisions, the gotchas, the owner's preferences, the honest status, and the exact job for you. Where the repo already explains something (setup, commands, API), this file points to it and does not repeat it.

Labels used throughout: **Measured** = a real run with numbers on record. **Tested with fakes** = unit tests against stand-ins. **Unverified** = never run for real. **Estimate** = my judgement.

---

## 1. Your job in one paragraph

Everything was built and tuned on an Apple-Silicon Mac (MLX). The only Windows port (23 Oct, branch `windows-nvidia-20261023`, see `docs/WINDOWS_NVIDIA_REPORT.md`) ran on a PC with **no NVIDIA GPU** (Intel Arc, CPU fallback). So the CUDA path has never executed once. Run the project on this machine, find out what works inside 4 GB of VRAM, fix Windows-only bugs with small cross-platform changes, and report numbers. Section 12 is the test plan, section 13 the list of what is still open, section 14 the report format.

## 2. Who the owner is and how they work

| Item | Detail |
| --- | --- |
| Owner | BonClay Core Team |
| Goal now | Showcase to the HOD of CSE and pitch for the college Best Project award; later pilots with clinics and colleges |
| Demo audience | Jayashri Prasad ma'am, HOD of the AI department; the HOD-office agent books her time |
| Preferences | Wants things done, not explained; runs agents in parallel; wants tables; wants honest scores; wants the cloned voice kept on phone calls (speed it up, do not switch to a preset) |
| Hard rules | Never print or commit `backend/.env`, keys, phone numbers, recordings. `backend/data/` is private and git-ignored. Commit or push only when asked. Do not break macOS when fixing Windows |
| Standing OK | The owner allowed calls to their own number only (the number in `PHONE_ALLOWED_NUMBERS`). You do not have that number or the Exotel credentials; do not ask for them unless the owner supplies them |

## 3. What BonClay is (the story)

A local-first voice-AI studio and agent platform for India. One laptop runs the whole chain: speech recognition, document retrieval, a language model, speech synthesis, a phone line, a calendar with human approval, and a public page that proves which audio it made. Target users: clinics, colleges and departments, coaching centres, creators. Languages: English, Hindi, Marathi, Hinglish first, up to 22 Indian languages for the Studio.

Two modes in one sidebar (a Studio | Agents toggle, terracotta vs green):

- **Studio:** preset voices, voice cloning with spoken consent, multi-speaker scripts, "Say it once" (translate and speak in up to 4 languages), Voices page, Models and memory panel, public Verify page.
- **Agents:** agent builder with 8 templates, knowledge base (hybrid retrieval, grounding, answer audit), Playground and Talk page (the agent speaks first), tools (meetings, approvals, Google Calendar), phone calls (Exotel), web call link, widget, WhatsApp voice notes, webhooks and Sheets, evaluation lab, analytics, API keys.

## 4. Architecture

```
Browser (Next.js 16 / React 19, :3000)
   | HTTP + WebSocket
FastAPI main app (backend/backend.py ~6.3k lines, :8000, meant for loopback only)
   |-- routers/      (phone_exotel, approvals, kokoro, system_resources, ...)
   |-- services/     (llm, retrieval, tools, model manager, provenance, gcal, ...)
   |-- core/         (shared helpers)
   |-- mcp_servers/  (meetings, others; JSON-protocol tool servers)
   |
   |-- model workers = separate processes, one model each, loaded on demand:
   |     Kokoro (MLX on Mac, PyTorch elsewhere) | Qwen3-TTS (MLX) | Chatterbox
   |     DhVaani (own venv) | Indic-Speak (own venv) | Whisper ASR (MLX on Mac)
   |
   |-- model manager: LRU + idle unload, memory guard
   |     (Mac: measured by physical footprint; elsewhere: see section 9)
   |
phone_public.py = phone-only public app (:8010) behind a tunnel (ngrok / cloudflared / own)
   -> Exotel Voicebot WebSocket (8 kHz PCM, `clear` message = barge-in)
```

Pipeline for one agent turn: caller audio -> ASR (Whisper MLX locally, or Groq) -> retrieval (BGE-M3 dense + keyword hybrid, reranker, instant answers, fact sheet, question index) -> grounding and an answer audit (MiniCheck style) -> LLM (Groq `qwen/qwen3.8-27b` by default; local Llama 3.2 3B / Gemma as offline fallback; tools via JSON-protocol MCP servers) -> language policy (reply in the caller's language) -> AI disclosure -> sentence-streamed TTS -> audio out.

Trust layer: AudioSeal watermark, signed receipt, public `/verify`, revocable provenance (withdrawing consent changes what an old clip verifies as), voice vault (encrypted samples, key destroyed on delete), optional C2PA manifest, licence named on every receipt.

Licences (`backend/services/licences.py`): `BONCLAY_COMMERCIAL=1` hides DhVaani (CC-BY-NC-4.0, approved for this project only), Indic-Speak (needs written approval from Bodhan AI, bans robocalls) and IndicF5 (unclear). Default is off.

## 5. Voice engines: measured comparison (Mac unless stated)

| Engine | Role | Language quality (Measured) | Speed (Measured) | Memory (Measured) | Licence | CUDA/4 GB status |
| --- | --- | --- | --- | --- | --- | --- |
| Kokoro 82M | Preset voices, live conversation | English word error 0.077, Hindi character error 0.083 | 0.5-0.7 s a sentence | small | permissive | **Unverified**; likely fits |
| Qwen3-TTS (8-bit MLX, streaming, 24 kHz reference, temp 0.5) | Cloning | similarity 0.64 to 0.75 after tuning; no Hindi speech | 4.6 to 2.4 s a sentence; warm clone ready in 2.4 s | peak 7.1 to 3.1 GB | permissive | **Unverified**; MLX is Mac-only, so a different backend is needed on CUDA. Check what the repo falls back to |
| Chatterbox Multilingual | Hindi cloning | character error 0.130 (last of five on accuracy) | about 33 s a sentence | large | permissive | **Unverified**; doubtful in 4 GB |
| DhVaani-0.5 (ARTPARK-IISc) | Cloning, 22 languages | error 0.169 over 82 sentences, best or tied in 8 of 10 languages; Malayalam and Punjabi poor | not benchmarked for speed | own venv | CC-BY-NC-4.0 | **Unverified**; the worker picks CUDA when present |
| Indic-Speak | Preset voices, 22 languages | error 0.188 | about 1.9x real time (4-bit) | own venv | approval-only | **Unverified** |

Speaker match for consent: same voice 0.88, different voice 0.14 (Measured, synthetic voices only).
Watermark: 20 of 20 clips detected over MP3, Opus, WhatsApp, 8 kHz line and trim; 0 false alarms in 420 clean checks. Fails on speed or pitch change and heavy noise.
Hinglish speech recognition on 15 synthetic clips: error 0.74. Hinglish speech output: 0.035.

## 6. Agent quality: measured

| Metric | Result |
| --- | --- |
| Answers graded correct | 89.8% of 108 |
| Invented content | 5.6% (Hindi: 83% correct, 11.4% invented) |
| Right passage in top 3 | 94.7% (multi-document agents: 62% at rank 1) |
| Reply in the caller's language | 100% |
| Forbidden bookings that got through | 0 of 38 |
| Phone: normal reply after caller stops | 1.4-1.5 s (bench, preset voice) |
| Phone: cloned voice | first audio median 3.0 s on a real call; TTS 7.3 s a reply |
| Tests | 3,287 backend collected, 745 frontend passed (77 suites) on the Mac, 1 Nov |

## 7. Status scorecard (Estimate, out of 10)

| Segment | Score | Why |
| --- | --- | --- |
| Demo readiness | 8 | calls, cloning, meeting booking and verify all ran live |
| Production readiness | 4 | several channels never ran for real; CI red; no data notice |
| Uniqueness | 7 | revocable provenance plus India-first offline stack are rare; consent plus speaker match is not new |
| Trust and provenance | 8 | |
| Meetings and tools | 8 | |
| Testing | 8 | |
| Agents and knowledge | 7.5 | |
| UI and UX | 7.5 | Agents and Talk pages overflow sideways on a phone; Call dialog shows developer text |
| Voice cloning | 7 | no anti-spoofing |
| Security | 7 | launcher `uvicorn.run(app, host="0.0.0.0", port=8000)` at the bottom of `backend.py` still binds all interfaces |
| Phone | 6.5 | real calls on a trial account that rings only whitelisted numbers |
| Voice quality, Indian languages | 6.5 | |
| Privacy and legal | 5 | no visible notice, no delete-everything route |
| Other channels (WhatsApp, other providers, cloud) | 4 | fakes only |
| Windows and NVIDIA | 4 | 2,573 backend tests passed on a CPU-only Windows PC |
| CI and release | 3 | GitHub refused to start jobs over a billing limit; not test failures |

## 8. Competitors (for context if you are asked to explain the pitch)

| Product | Strength | Where BonClay differs |
| --- | --- | --- |
| ElevenLabs | Best voice quality, large catalogue | Cloud and paid; no local run; BonClay adds consent, revocable provenance, Indian focus |
| Retell, Vapi | Polished phone agents | Cloud only, per-minute cost, not India-first, no on-device option |
| Plivo | Telephony in India | A carrier, not an agent studio |
| Dograh, Bolna | Open voice-agent platforms (Bolna India-focused) | BonClay adds cloning with consent, watermark and verify, HOD-style approvals |
| Sarvam | Indian models and agents | **Biggest commercial threat**; BonClay competes on local, private, consent-first |

Claims safe to make: runs on one laptop; measured answer accuracy on a stated sample; watermark survived the listed edits; commercial mode hides restricted-licence voices. Claims to avoid: "first" or "only" for consent or watermarking; "production-ready"; "open source" (repo is private); anything about WhatsApp, other phone providers or cloud mode.

## 9. Memory management, and why it matters on your machine

The model manager (LRU plus idle unload) and the memory guard were designed around macOS **physical footprint**. The sidebar indicator reads from `routers/system_resources.py` (it already has a CUDA branch using `torch.cuda.mem_get_info()`). On a 4 GB card the questions are:

1. Does the guard use VRAM or system RAM for its decisions on Windows? Check the code before trusting any "fits" message.
2. Do idle workers actually release VRAM (check `nvidia-smi` after idle)? Subprocess workers should free it on exit; in-process models need `torch.cuda.empty_cache()`.
3. `EMBEDDING_DEVICE` (default cpu) and `RERANKER_DEVICE` (default auto) decide whether BGE-M3 and the reranker take VRAM next to the TTS model. On 4 GB, keep both on CPU unless measured otherwise.
4. Mac numbers for reference: a 16 GB Mac swaps 3-5 GB with everything loaded; the demo keeps two models at a time.

Hardware detection lives in `backend.py` (`auto_detect_hardware`, `platform_info`): CUDA gets fp16, MPS fp32 (fp16 gave NaNs), else CPU fp32.

## 10. History: what was built, in order, and the lessons

| Phase | What happened | Lesson to carry |
| --- | --- | --- |
| Early | Studio, agents, retrieval, trust layer, 22 Oct critical review (65% complete, demo 80%, production 38%) | Reviews found ngrok exposure and licence blockers; both addressed |
| Exotel phone | Voicebot WebSocket, barge-in only when audio is playing (`IMPATIENT_MS=1500`), saved numbers file, clean Exotel errors | Account is on `api.exotel.com` (India endpoint gave 401); trial rings only whitelisted numbers (others return a KYC 403) |
| Public exposure | `phone_public.py` on :8010 is the only thing the tunnel reaches; main API loopback only; signed per-call tokens | Fixed the earlier "whole API on ngrok" risk |
| Clone tuning (23 Oct) | 8-bit, streaming, 24 kHz reference, temp 0.5 halved clone time | Tested on synthetic references only; nobody listened to the new settings |
| HOD meeting agent | `meetings` MCP server, approvals page, Google Calendar via service account | The model copied example values (a phone number) from the prompt, so prompts now use placeholders only; tool servers needed `BONCLAY_MEETINGS_FILE` and Google variables passed through `server_env` |
| Windows port (23 Oct) | Friend on a CPU-only PC fixed temp dirs, POSIX modes, timer granularity, encoding, ESM URLs, `torch._dynamo`, GPU detection in `bonclay.ps1` | `Test-NvidiaGpu` replaced a naive `nvidia-smi` check (System32 stub false positive) |
| Redesign (30-31 Oct) | Vaani-style site, Motion animations, 3-zone Studio, agents rail, Talk page | Pages fade in; screenshots taken early show blank pages |
| 1 Nov audits | Four audits of the repo plus UI; scorecard doc | See sections 7 and 13 |

Test-writing lessons: tests once leaked the real `.env` and wrote a real numbers file; the rig fixture now clears `PHONE_*`/`EXOTEL_*` and sets `BONCLAY_PHONE_NUMBERS_FILE`. A mutation test hung instead of failing; run with timeouts.

## 11. Known defects (from the audits, not Windows-specific)

| # | Defect | Severity |
| --- | --- | --- |
| 1 | `backend.py` launcher binds `0.0.0.0` | Medium |
| 2 | Agents page scrolls sideways on mobile (444 px); agent names clipped at ~800 px | Medium |
| 3 | Talk page overflows on mobile (469 px) | Medium |
| 4 | Call dialog shows developer text ("Pre-approved in backend/.env") | Low-Medium |
| 5 | Wrong tab titles (Campaigns says "Home", Analytics says "Results") | Low |
| 6 | Hero says 0.58 s first audio; text below says 0.84 s | Low |
| 7 | Approvals sample-schedule grid reads as broken when empty | Low |
| 8 | STATUS.md numbers stale; stray "Parrot AI" header | Low |
| 9 | No data-protection notice, no delete-all route | High for real customers |
| 10 | No anti-spoofing on the consent code | Medium |

You are not asked to fix these. Mention any you hit on Windows.

## 12. Test plan for this machine

Record for every step: PASS / FAIL / PARTIAL, exact error, time, **peak VRAM from `nvidia-smi`**, peak system RAM.

**A. Environment (setup steps are in the README and `bonclay.ps1`; do not redo them here)**
1. `doctor` must report the platform as NVIDIA with the card name and VRAM. If it says CPU, check `torch.cuda.is_available()`; a `+cpu` torch means the CUDA wheel did not install. Report the driver and torch versions.
2. Note which optional venvs exist (`.venv-dhvaani`, `.venv-chatterbox`, `.venv-indic`); test the base first, then add one at a time.

**B. Automated**
3. Backend `pytest`: Mac collected 3,287; the last Windows run (23 Oct, before 111 commits) was 2,573 passed. Report failures with full tracebacks.
4. Frontend `jest`: expect 77 suites, 745 tests.
5. Recheck the Windows guards from the 23 Oct report now that CUDA is present (`torch._dynamo` disabled for AudioSeal; permission asserts skipped on `nt`).

**C. Models on 4 GB (the core question)**

| Step | Action | Record |
| --- | --- | --- |
| 6 | Kokoro English then Hindi preset voice (needs espeak-ng) | device used, time to first audio, VRAM peak, mispronunciations |
| 7 | ASR path: which one Windows uses (local or Groq) and latency | path, latency |
| 8 | Embeddings and reranker on cpu vs cuda while Kokoro is loaded | VRAM, retrieval time |
| 9 | Cloning with the default engine (consent code, 10-20 s sample) | works or out-of-memory, time, VRAM; which engine actually ran |
| 10 | Try each optional cloning engine alone (DhVaani, Chatterbox) | fits or not, time, VRAM |
| 11 | Switch engines back to back | does VRAM return to baseline after idle unload |
| 12 | Say it once, Hindi plus Marathi | time, VRAM |

**D. Product flows**
13. Agents: create the Appointment template, upload the hospital sample PDFs from `docs/samples`, ask three English and one Hinglish question. Answers must come from the document.
14. Talk page: time from end of question to first audio (Mac: 1.4-1.5 s with Groq and a preset voice).
15. Verify: download a generated clip, upload to `/verify`, then withdraw consent and verify again.
16. Models and memory panel: shows GPU memory, not "CPU".

**E. Only if the owner supplies credentials:** Exotel call, web call link, Google Calendar booking.

## 13. Still open (whole project)

| Area | State |
| --- | --- |
| CUDA/NVIDIA run | Never executed. This is your job |
| 4 GB fit for cloning engines | Unknown |
| Non-Exotel phone providers (Twilio, Plivo, Vobiz, Telnyx), inbound, campaigns, voicemail, DTMF, transfer, takeover | Tested with fakes only |
| WhatsApp voice notes, Sheets, cloud mode with Supabase | Tested with fakes only |
| Real-caller measurements | None; all benchmarks use synthetic voices and small samples |
| Listening test of the faster clone settings | Not done |
| Anti-spoofing, data-protection notice, delete-all route | Missing |
| CI | Red because of a GitHub billing limit, not test failures |
| Release | Tags only v1.0-demo and v1.1 |
| Backend full test pass count on Mac | Run was about 87% done when this was written; not confirmed |

## 14. Report format

```
Machine: GPU name, VRAM, system RAM, driver, torch version, Python version
Automated: backend passed/failed/skipped, frontend suites/tests
Table: step | PASS/FAIL/PARTIAL | time | peak VRAM | peak RAM | notes
Failures: step, exact error, last 30 log lines, what you tried
Recommendation: which engines are usable on 4 GB, which to default to,
                what to change in the memory guard or engine routing
```

## 15. Working rules for you

- Branch `windows-nvidia-4gb-<date>`; do not commit or push unless asked.
- Small cross-platform fixes only (`os.name`, `pathlib`, `encoding="utf-8"`); do not weaken quality settings silently to fit memory. Record the number and propose options.
- Never print `.env`, keys, phone numbers or user recordings. Do not call real phone numbers.
- If something is unverified, say so. Do not report a pass you did not run.
