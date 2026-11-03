# Demo checklist (23 October 2026)

What the audience will feel, measured on this M4 on 22 October with the code on `main`, and the steps that keep it
that way on the day. Numbers come from `backend/bench/` runs against the merged code; the raw files are in
`backend/bench/results/`.

## What they will see and hear

| Moment | Measured | What makes it so |
| --- | --- | --- |
| Agent answers a typed or spoken question in the Playground | first audio 0.44 s median (p95 0.96 s); first sentence of the answer 1.56 s median (English 1.41 s, Hindi 1.84 s); 10 of 10 turns answered; one turn in ten hit Groq's per-minute limit and took 10.5 s via the local model | Groq Qwen3 with the gpt-oss fallbacks; the AI disclosure plays while the answer is written. Since 22 Oct a per-minute limit is waited out (up to 3 s) instead of dropping to the local 3B model |
| Phone call (bench, no real line) | first reply byte 1.91 s median after the caller stops; two callers at once 1.46 s median, p95 1.99 s | Pre-rendered greeting, speculative start at 250 ms, filler at 500 ms, engines pinned for the call |
| Clone a voice from a consent clip | 0.8 s warm; the first clone after a restart 1.8 s once `./bonclay warm` has run (12.8 s without it) | ECAPA speaker match plus the code read-back; the warm-up now starts the clone worker and renders one discarded sentence |
| A sentence in the cloned voice | about 0.6 s of generation per second of speech since 23 Oct: 2.4 s for a 4 s sentence when warm (was 4.6 s in the same paired run, 60 sentences each); speaker similarity to the reference 0.75 (was 0.64); the clone worker's peak memory 3.1 GB (was 7.1 GB) | Qwen3-TTS 0.6B on MLX with 8-bit weights, streaming decode inside the worker (the reference is no longer re-decoded with every sentence), temperature 0.5 and a 24 kHz reference (`MLX_CLONE_*` in `.env.example`). The reference clip's encoding (0.3 s) is done once per voice and cached. Measured on synthetic Kokoro references (`backend/bench/results/clone_trials_23oct/REPORT.md`); the similarity gain still has to be confirmed on a real voice |
| A Studio render of several sentences in a cloned voice | the first sentence plays while the rest renders (at about a fifth to a third of the old total wait); the download and receipt appear when the whole clip is done | `POST /api/voices/generate/stream`, the Speech panel's sentence queue; falls back to the one-shot render if the stream fails |
| English preset speech | read-back WER 0.077 (10 sentences) | Kokoro af_heart |
| Hindi preset speech | read-back CER 0.083 (10 sentences), fairly scored; 0.074 with `PRON_LEXICON=on`, which fixes मुंबई, छह and शांत (proven by read-back on held-out sentences; ट्रेन and झील still misread) | Kokoro hf_alpha, the only licence-clean fast Hindi voice; the three spelling hints are in services/pron_lexicon.py and never change the shown text or the receipt |
| Verify page | a watermarked clip verifies, a withdrawn voice's clip reads "consent withdrawn" on the next request | AudioSeal + Ed25519 receipts + the signed withdrawal log; C2PA manifest optional |

## Before the demo (the evening before)

1. **Run the merged code.** The laptop's main checkout was on `wip/local-uncommitted-changes`. From the repo root:
   `git checkout main && git pull`. Everything that was uncommitted there is already on `main`.
2. `./bonclay stop && ./bonclay start`, then `./bonclay warm` so the first turn is not a cold one. The warm-up now includes the
   clone worker (the "Voice cloning" line should say it rendered); without it the first clone takes 12.8 s instead of 1.8 s.
3. In `backend/.env`: `GROQ_API_KEY` set; `PHONE_PUBLIC_PORT` left at 8010; `TOOLS_READBACK` unset (on); `C2PA_MANIFEST`
   unset unless you want manifests in downloads; `PRON_LEXICON=on` if, after listening to the before/after pairs in
   `backend/bench/results/hindi_lexicon_listen/`, the hinted मुंबई, छह and शांत sound right to you (default off). Do not run any bench or the attack suite in the hour before: they use the
   same 8,000 tokens-a-minute Groq budget as the demo.
4. Open the Playground once and ask one question in English and one in Hindi; both should answer in under 2 s. Clone one
   voice and generate one sentence so the clone worker is up.
5. If a call is part of the demo: the trial Exotel account rings only whitelisted numbers; place one test call first.
   `./bonclay start` prints the tunnel line; the phone-only port is what the tunnel exposes.
6. Have `GET /api/health` open in a tab: `models_loaded`, `models_pinned` during a call, and `heartbeat` (off unless a ping
   URL is set) are the three things to glance at if something feels slow.

## During the demo, if something goes wrong

- A reply takes more than 4 s: Groq is rate-limited or down. The reply still comes (the local model answers); say so,
  ask the next question, it recovers on its own.
- A cloned-voice sentence takes long: the first sentence after cloning encodes the reference; the second is faster. In
  Studio the first sentence plays while the rest renders; if it does not, the stream fell back to the one-shot render.
- Hindi mispronounces a word: it is Kokoro's Hindi voice; the Devanagari text on screen is right (the receipt proves it).
- Nothing answers: `./bonclay logs api`, then `./bonclay stop && ./bonclay start`. Models reload in under a minute.

## What not to claim

- Real phone calls have not been measured; every call number above is the local bench playing Exotel.
- The Hinglish speech-recognition figures are from synthetic clips; the real-caller set is not recorded yet.
- The C2PA signer is self-signed: it proves integrity, not identity.
