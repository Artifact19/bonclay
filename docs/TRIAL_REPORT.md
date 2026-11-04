# BonClay test report: what the October improvements changed, and which options to choose

Measured on the team's Apple M4 (16 GB), 15 October 2026. Every number comes from scripts in `backend/bench/` (raw
files in `backend/bench/results/` and `backend/bench/data/`, scratch notes from the trial runs). Voices in all trials are
synthetic presets - no real person's voice was used. Where the machine was busy, the comparison was run interleaved
(both options on the same items, alternating) so load affects both sides equally; absolute times are then noisier
than the differences. Nothing here is legal advice.

## 1. Bottom line

**The project got better on every measured axis except memory.** Replies are faster, Hindi most of all; knowledge
answers are found more often; agents can now take actions (booking); trust features are closer to what Indian rules
ask for; and the evaluation lab now tests Hindi/Hinglish attacks. The price is about **+2.2 GB of memory** for the new
knowledge-search model and **+0.2 s** per knowledge search.

**Recommended set-up (what to go with):**

| Decision | Choose | Why |
|---|---|---|
| Cloud language model | **Groq `qwen/qwen3.8-27b`**, then `gpt-oss-20b`, then `gpt-oss-120b`, then local | Fastest and most accurate in our checks (first token 0.12 s, only one to book "tomorrow" correctly); the fallbacks keep you on Groq when the free tier's per-minute or daily limits hit |
| Offline language model | **Keep Llama 3.2 3B on this 16 GB Mac for demos; switch to Gemma 4 E4B on a 24 GB+ machine or when offline quality matters most** | Gemma is far better at Indian languages (84% vs 49% judged) at the same speed, but adds ~2.5 GB and pushed swap from 2.8 to 4.7 GB in the full-app test |
| Live voice | **Keep Kokoro** | Nothing beat it on speed and intelligibility together; Kokoro-on-MLX is the same voice 10-47% faster - worth adding next |
| Knowledge search | **BGE-M3 + keyword (hybrid), reranker off** | top-3 accuracy 88.9% → 97.8%; reranker costs 0.2-1.7 s |
| Indian-language cloning beyond Hindi | **DhVaani-0.5: adopted (17 Oct), Chatterbox stays the Hindi fallback** | Clear Marathi/Tamil/Bengali/Telugu, 3-5× faster than Chatterbox; ARTPARK-IISc approved the use in writing (see docs/decisions/indian-language-cloning.md) |
| End-of-turn model (Smart Turn) | **Skip** | Ends turns 0.5 s sooner but cut callers off mid-sentence in 35-65% of pauses |
| C2PA content credentials | **Not now** | Works on WAV/MP3 without touching audio, but any re-encode strips it and a trusted certificate costs money |

## 2. Before vs after: speed (30 spoken questions, 15 English + 15 Hindi, streaming)

Like for like - **local models only, no cloud keys**, same 30 questions as the morning's `docs/RESULTS.md` run:

| | Before (morning) | After | Change |
|---|---|---|---|
| First audio, median | 1.57 s | **1.35 s** | −14% |
| First audio, Hindi | 2.45 s | **1.73 s** | −29% |
| First sentence of the answer, median | 4.20 s | **3.58 s** | −15% |
| Full reply, median | 4.68 s | **3.77 s** | −19% |
| Full reply, p95 | 8.44 s | 8.45 s | same |
| Speech recognition, median | 1.46 s | **0.92 s** | −37% (new Hindi "auto" strategy) |
| Knowledge search, median | 0.02 s | 0.24 s | +0.22 s (bigger, better model) |
| Failed turns | 0 / 30 | 0 / 30 | - |

**With Groq** (the normal set-up), on the turns Groq answered (n = 10): first audio **0.50 s**, first answer 2.52 s,
**full reply 2.81 s - under the 3-second target for the first time**; speech recognition 0.25 s (Groq Whisper).
Caveat: 20 of the 30 turns fell back to the local model because the free tier's limits were exhausted by the day's
testing (Qwen: ~8,000 tokens per minute; gpt-oss-20b: 200,000 tokens per day). `gpt-oss-120b` is now added as a
second Groq fallback, since each model has its own allowance. For a demo, avoid running evaluations on the same day,
or use a paid Groq tier.

Filler audio ("One moment" / "एक सेकंड") did not trigger in these runs because every test turn was a new call (the AI
disclosure plays first); it helps from the second turn of a call. Under heavy load it brought first sound to 1.4-1.7 s
when the answer itself took 8-17 s.

## 3. Before vs after: quality

**Knowledge search** (45 questions: 13 English, 14 Hindi, 18 Hinglish, over three template documents):

| | correct passage first | correct passage in top 3 |
|---|---|---|
| Before: MiniLM | 64.4% | 88.9% |
| **After: BGE-M3 + keyword hybrid** | **75.6%** | **97.8%** |
| BGE-M3 + hybrid + reranker (off) | 84.4% | 97.8% |

**Offline language model** (judged by Groq `gpt-oss-120b`, blind, both orders, 2 runs; 86% judge agreement):

| Task | Llama 3.2 3B | Gemma 4 E4B |
|---|---|---|
| Answers grounded in the document | 78% | **97%** |
| Safety (Hindi/Hinglish attacks) | 65% | **86%** |
| Translation to Hindi, Marathi, Tamil | 30% | **94%** |
| Hinglish → Devanagari | 38% | **100%** |
| Tool calls offline | 25% | 12% (both poor - tools need Groq) |
| **All** | **49%** | **84%** |
| Full-app reply time, local only (median) | 3.77 s | 4.03 s |
| Memory of the model server | 2.3 GB | 4.6-5.0 GB |

Gemma must run with "thinking" off (now automatic); with it on, replies took 15 s and many came back empty.
Licence: Google lists Gemma 4 as Apache-2.0, the MLX copy says "Gemma terms" - confirm before commercial use.

**Cloud models on Groq** (same prompts): Qwen 3.8 27B - first token 0.12 s, correct Hindi/Marathi/Hinglish, refused
the injection, booked "tomorrow" correctly. gpt-oss-20b/120b - 0.4 s first token, good answers, but booked *today*
instead of tomorrow. Qwen is a Preview model (no SLA, $0.80/$4.00 per M tokens vs $0.15/$0.60 for gpt-oss-20b).

**Speech recognition, offline Hindi** (60 synthetic clips incl. noisy copies): new `auto` strategy is **0.85 s faster**
than before for Hindi at identical character error rate (0.079); English unchanged.

**Evaluation lab, new 60-case Hindi/Hinglish attack suite** (clinic agent, 38 applicable cases): 61-76% across runs -
the spread comes from Groq limits pushing some replies to the 3B local model. Strong: system-prompt leaks, personal
data, impersonation, role-play, abuse (100%). Weak: switching language mid-call (25%) and tool misuse (33%, the suite
runs without tools). The original 6-persona quick suite: 83/100.

## 4. Trust and provenance

**Watermark robustness** (20 clips, 21 attacks, AudioSeal at the app's own threshold):

| Survives (20/20 detected, receipt found) | Partly | Removed |
|---|---|---|
| MP3 64 and 32 kbps, Opus 16 kbps, WhatsApp-style chain, 8 kHz phone line, trimming | Music at 20 dB below speech (20/20 detected, 18/20 receipt); broadband noise at 20 dB (40-45%); low-pass 3.4 kHz (detected, receipt lost) | ±5% speed, pitch shift, noise/music at 10 dB, neural codec (EnCodec 6 kbps: 1/20) |

No false alarms on 420 clean checks. **Hindi clips are much weaker against noise than English** (1/10 vs 7-8/10 at
20 dB) - the watermark sits lower under Hindi speech; worth fixing before claiming parity. Suggested public wording:
*"Detects BonClay audio after typical compression, phone lines and trimming. It is evidence, not proof: speed or pitch
changes, heavy noise or AI re-encoding can remove it."* Current landing claims (MP3 32 kbps, 8 kHz) are accurate.

**C2PA content credentials:** WAV and MP3 supported, audio bit-identical, ~3.6 KB overhead, licence MIT/Apache-2.0 -
but stripped by any re-encode (even an MP3 copy), Opus/WebM unsupported, and shows as untrusted without a paid
certificate. Not now.

**Compliance pack (built, functional):** itemised consent record per voice with download, plain notice before
recording, retention (365 days, report-only unless enforced), one-click withdrawal that /verify reports, and an
"AI-generated voice · BonClay · receipt …" label inside every WAV (audio unchanged, watermark intact).

## 5. Trials

| Trial | Result | Verdict |
|---|---|---|
| **Gemma 4 E4B** (offline model) | 84% vs 49% judged; same speed; +2.5 GB | Opt-in now (`LOCAL_LLM_MODEL=mlx-community/gemma-4-e4b-it-4bit`); default on bigger machines |
| **DhVaani-0.5** (cloning in Indian languages) | 0 failures in 50; read-back error Hindi 5-8%, Tamil 4-6%, Telugu 7-11%, Marathi 9-18%, Bengali 12-18%; voice similarity 0.69 from a Hindi reference, 0.41 from English; ~3-5× faster than Chatterbox | Adopted on 17 Oct after ARTPARK-IISc's written approval (its base model's weights are non-commercial, so keep the approval on file). Still to do: human listening, real-voice references |
| **Faster voices** | Kokoro on MLX: same voice, 10-47% faster. Pocket TTS: first audio 0.06 s but English only. Supertonic 3: Hindi OK at 4 steps, but licence bans medical-advice use (clinic template). OmniVoice: ~5 s per sentence | Keep Kokoro; add Kokoro-MLX next; Pocket TTS optional for English |
| **Smart Turn v3** | Complete turns end ~0.5 s sooner, but mid-thought pauses cut off 35-65% (Hindi worst) | Skip; retry only with real recorded calls |
| **Silero voice detection** (hands-free) | Built; loads only when hands-free is on; falls back to the old detector | Keep; tune on real speakers |

Listen for yourself: synthetic samples are in `trial-audio/` (not committed) - `indic-cloning/` (DhVaani in five
languages vs Chatterbox Hindi, plus the two reference voices) and `tts/` (Kokoro PyTorch vs MLX, Supertonic, OmniVoice,
Pocket TTS, Soprano).

## 6. What got worse, and what to watch

- **Memory:** BGE-M3 keeps ~2.2 GB resident; with Gemma on top, a 16 GB Mac swaps. Close other apps before demos.
- **Knowledge search is 0.2 s slower** per question (still small next to the 3-4 s reply).
- **Tool agents lose word-by-word streaming:** the booking decision is made first, then the answer is spoken sentence by
  sentence. Agents without tools are unaffected.
- **Groq free tier** runs out during heavy testing; replies then come from the slower, weaker local model.
- **First search after the upgrade re-embeds documents** (seconds per agent); `./bonclay warm` before a demo.
- **Cloud mode** needs Supabase migrations 009-011 applied before deploying.

## 7. Next steps, in order

1. Fix the Hindi watermark strength (raise it for Hindi or tune the loudness) and re-run the attack test.
2. Add Kokoro-on-MLX as the live voice on Apple Silicon (same voice, faster).
3. ~~Ask ARTPARK-IISc about DhVaani's licence; if clear, add it as a worker~~ Done on 17 Oct: approval received, DhVaani is integrated (`./bonclay setup --dhvaani`).
4. Store retrieval scores with each turn so the knowledge-gap report can also flag weak searches.
5. Teach agents to stay in the caller's language and add tool-misuse cases to a tools-on evaluation run.
6. Decide on a paid Groq tier (or Sarvam/other) before real customers; free-tier limits are reached within a day of testing.

## 8. Second wave (evening, 15 October 2026)

Built after the first report: website widget, WhatsApp voice notes, signed webhooks + Google Sheets, a language policy
(reply in the caller's language, remembered switches, polite decline), tools-on evaluation, Kokoro on MLX as the live
voice, a level-aware watermark, and retrieval scores stored per turn.

**Speed, same 30 questions, local models only, quiet machine:**

| | Morning | After first wave | **After second wave** |
|---|---|---|---|
| First audio (median) | 1.57 s | 1.35 s | **1.02 s** |
| Full reply (median) | 4.68 s | 3.77 s | **3.10 s** |
| Full reply, English | 3.39 s | 3.21 s | **2.14 s** |
| Full reply, Hindi | 5.08 s | 4.21 s | **4.06 s** |
| Voice synthesis per turn | 1.39 s | 1.17 s | **0.73 s** (Kokoro on MLX) |
| Knowledge search | 0.02 s | 0.24 s | **0.10 s** |

Kokoro on MLX also uses about 1.6 GB less memory than PyTorch Kokoro, which roughly offsets the knowledge-search model.

**Watermark, Hindi:** now embedded at the same strength relative to speech as English; detection under noise and music
went from 3/60 to 28/60 attack cases (English 37/60), with common channels (MP3, Opus, WhatsApp-style, phone, trim)
still 70/70 in both languages and no false positives. Cost: Hindi audio quality score PESQ 4.3 → 3.8 (still "good").

**Language policy:** the deterministic language check on language-switch attacks went from 6/12 to 12/12; with Groq
models the judged pass rate for that category went from 1/4 to 3/4. Tools-on evaluation: no misuse booking goes through
any more (3 did before). Judged scores from the local 3B judge are too noisy to compare.

**Channels:** website widget verified end to end in a browser (notice → spoken question → spoken reply; wrong site and
revoked key refused). WhatsApp verified end to end against a fake Graph API with a real OGG voice note (watermark still
detected after Opus at 24 kbps); a real Meta test number is needed to go live (`docs/WHATSAPP.md`). Webhooks verified
with a local receiver (signed events incl. a booking); Google Sheets tested against a fake only.

## 9. Third wave (16 October 2026): cache-friendly prompts, streamed tool agents, fp16 search

Same 30 questions, local models only, after a restart on the new code:

| | After second wave | **After third wave** |
|---|---|---|
| First audio (median) | 1.02 s | **0.84 s** |
| First sentence of the answer (median) | 2.89 s | **2.23 s** |
| First sentence of the answer (p95) | 6.46 s | **3.58 s** |
| First answer, Hindi (median) | 3.79 s | **3.07 s** |
| Full reply (median / p95) | 3.10 s / 6.87 s | **3.02 s / 5.03 s** |
| Knowledge search | 0.10 s | **0.04 s** |

Changes: a static system prompt with documents and the clock in the user turn (simulated uncached tokens per turn on
a three-document agent ~2,460 → ~500), ~120-word chunks and top-3 retrieval, streamed answers for tool agents with the
first phrase spoken first, BGE-M3 in fp16 on Apple Silicon, and no extra model call before the first Hinglish audio.
Swap fell from 5.3 GB to 2.4 GB during the run. "Say it once" was redesigned (one click, one card per language,
progress with ETA); a Fast run of Hindi, French and Spanish finished in 3 s.

