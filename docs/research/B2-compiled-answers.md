# B2: Compiled knowledge base to instant spoken answers (phone-latency focus)

Date: 2026-10-17. Second pass: primary pages opened (arXiv abstract/HTML, GitHub READMEs). Markers:
**[V]** = number read on the cited page this pass; **[U]** = unverified (estimate, inference, or secondary source).

## 1. Findings with sources

### 1.1 Semantic caches and false hits
- **vCache** (Schroeder et al., ICLR 2026, arXiv:2502.03771, https://arxiv.org/html/2502.03771) learns a per-entry
  similarity threshold online and targets a user-set error bound delta. [V] numbers from the HTML paper:
  - Static-threshold GPTCache baseline on SemCacheLMArena (60k queries): threshold 0.99 -> 2.5% error at 37% hit;
    0.98 -> 4.1% error at 53% hit; 0.97 -> 5.2% error at 67% hit.
  - vCache: delta 2% -> 0.5% error at 41% hit; delta 3% -> 1.1% error at 46% hit (SemCacheLMArena).
    SemCacheClassification: delta 1% -> 0.4% error, 18.1% hit; delta 2% -> 1.2% error, 35.5% hit.
  - Overhead: embedding ~18 ms/request; threshold logistic regression ~1.7 ms; tau computation < 1.5 ms.
  - Headline: "up to 12.5x higher cache hit" and "26x lower error rates" vs baselines (abstract).
  - Lesson: even a 0.99 cosine threshold gives ~2.5% wrong answers on open chat traffic. Fixed thresholds are not safe.
- **vCache can run locally?** README (https://github.com/vcache-project/vCache): "uses OpenAI by default for both LLM
  inference and embedding generation", needs OPENAI_API_KEY in quick start, default vector DB HNSWLib, components are
  swappable, install `pip install -e .`. Licence badge reads CC NC-ND 3.0 [V for badge text; LICENSE file not opened].
  Verdict: technically swappable to BGE-M3 [U, no local example given], but **NC-ND is not commercial-friendly and
  forbids derivatives** -> do not vendor it. Reimplement the idea (per-entry threshold + abstention) from the paper.
- **GPTCache** (https://github.com/zilliztech/GPTCache): MIT [V]; local ONNX embedder (paraphrase-albert-onnx), FAISS,
  a cross-encoder-like `albert-duplicate-onnx` similarity evaluator [V]. README publishes **no measured hit or false-hit
  numbers** [V]; the "100x speed" claim is unsupported on the page.
- **Spoken / ASR-noisy queries:** no paper found that measures *cache* false hits on voice input. Closest:
  arXiv:2209.12944 (CIKM 2022) shows lexical and dense retrievers degrade under ASR noise, and "further degrade when
  dealing with natural ASR noise instead of synthetic" [V, abstract only; numeric MRR not read]. Practitioner blogs quote
  FPR 12% / 4% / 1% at thresholds 0.85 / 0.92 / 0.97 (https://tianpan.co/blog/2026-05-17-semantic-cache-confidently-wrong-answer)
  - explicitly illustrative, **[U]**. **Gap remains: BonClay must measure its own false-hit rate on 8 kHz phone ASR.**

### 1.2 Romanized Hinglish vs Devanagari
- No paper found that measures dense-retrieval accuracy Roman vs Devanagari Hindi directly (gap still open).
- RomanSetu (arXiv:2401.14280): romanization "reduces token fertility by 2x-4x" and romanized embeddings show
  "closer alignment with their English translations" [V, abstract; no Hindi-specific figures].
- arXiv:2305.03207 (MT for Indian languages): models "reasonably robust to script differences", Hindi most affected
  [V via search snippet only, paper not opened -> treat as U].
- **IndicXlit** (https://github.com/AI4Bharat/IndicXlit): MIT for code and models [V], 21 languages, ~11M params [V].
  Aksharantar top-1 for Hindi: En->Indic native words 55.59, named entities 58.87; Indic->En 52.30 / 59.22 [V].
  Marathi 63.72 / 53.57; Tamil 69.78; Telugu 84.69; Bengali 61.70 (Indic->En only 11.76) [V].
  Implication: top-1 is only ~55% for Hindi, so **do not transliterate then exact-match**; use it to generate both-script
  variants of each *stored question* offline (top-k candidates), and embed all variants.

### 1.3 IVR pre-computation and stitched audio
- No primary evaluation of carrier-phrase + slot quality found this pass (gap). Background only: concatenative-TTS patent
  US7308407 says prosody flaws are the main source of unnaturalness and read-aloud units differ from conversational
  prosody [V via search snippet, U for details]. Classic IVRs use pre-recorded prompts + slot concatenation (industry
  practice, [U]).
- Practical inference for BonClay: at 8 kHz mu-law/PCM telephone band, small joins are less audible than at 24 kHz [U].
  Safer pattern: avoid mid-sentence splicing; put slots in their **own short live-TTS sentence** ("Your slot is 4 pm.")
  after a cached full sentence, using the same voice, and add 80-150 ms silence between [U, test by listening].

## 2. Where the time goes on a phone turn (code read)
`backend/routers/phone_exotel.py::_turn` runs `b._turn_event_stream` (ASR -> retrieval -> LLM -> TTS per sentence),
then `_sentence_pcm` reads the WAV and resamples to the call rate in a thread, then `send_pcm`. Per-turn latency marks
already exist (`_mark("heard")`, `_mark("sentence")`, `_report_turn`). Shared context: first audio 0.84 s, first answer
2.23 s (p95 3.58 s), Hindi 3.07 s (browser, not phone).
A cache hit can skip: retrieval (~reranker), LLM first sentence, TTS of the whole answer, and WAV->8 kHz conversion.
It cannot skip VAD end-pointing or ASR.

## 3. Concrete design
1. **Compile (offline, per agent, on document save):** generate 30-150 likely questions per agent with the LLM
   (doc2query style) + owner-written FAQs; for each question create variants: English, Devanagari Hindi, romanized
   Hinglish (IndicXlit top-3 both directions), plus ASR-noised copies (run TTS -> 8 kHz -> ASR round trip on each
   question to get realistic phone transcripts). Generate one answer per language; **the owner approves each answer**.
2. **Pre-render audio:** synthesise approved answers in the agent's voice, store as **PCM16 at 8 kHz already**
   (and 16 kHz for browser), watermark at render time, store hash for the receipt. Saves conversion per call.
3. **Match (online, after final ASR):** BGE-M3 dense (already loaded) + BM25 over all variants -> top-5 -> local
   cross-encoder (the existing reranker) -> accept only if (a) reranker score >= per-entry threshold learnt vCache-style
   from logged confirmations, (b) margin to 2nd best intent >= m, (c) no time/personal slot needed, (d) answer still
   valid (doc hash unchanged). Otherwise fall through to the live pipeline with zero penalty except ~20-60 ms [U].
4. **Freshness:** each answer stores source chunk hashes; any doc edit marks it `stale` -> not served until re-approved.
   Time-dependent answers (today's slots, meetings) are never cached; they go to tools/live LLM.
5. **Approvals interaction:** reuse the existing approvals pattern (routers/approvals.py statuses
   pending/approved/declined) with a new kind `compiled_answer`. Rules: only `approved` + not stale entries are
   servable; a cache hit must **never** perform an action (book/confirm a meeting) - anything that would create a
   meeting request goes live, so the existing "agent never confirms, owner approves" invariant is untouched. Log each
   hit (question, matched id, score) so owners can flag wrong hits; a flag auto-demotes the entry to pending.
6. **Receipts:** served clip gets the same HMAC receipt, with `source: "compiled", answer_id, version`.
7. **Speculative bonus (optional):** start matching on partial transcripts; begin playback only on final ASR.

## 4. Expected numbers (all estimates unless marked)
- **Hit rate:** narrow SMB / HOD-office agents: 25-45% of turns [U]; vCache shows 35-46% hits at ~0.5-1.2% error on
  general traffic [V], narrow domains should do better [U].
- **False-hit rate:** target <= 1% of served hits with reranker + margin + abstention [U]; naive cosine 0.97 could be
  ~5% [V for text, worse on noisy ASR per 2209.12944 qualitatively].
- **Latency saved per hit:** ~1.0-1.6 s off time-to-first-audio vs live (removes LLM first token + first TTS sentence +
  resample; ASR and end-pointing stay) [U, derived from 0.84 s / 2.23 s figures]. Cached path ~ASR + 30-80 ms [U].
  Averaged over all turns at 35% hit: ~0.4-0.55 s mean saving [U]. Measure with the existing `_report_turn` line.

## 5. Effort
- Compiler + approval kind + UI list: 4-6 days. Matcher with reranker/margin/abstention: 2-3 days. 8 kHz pre-render +
  receipts: 1-2 days. Eval harness (TTS->8 kHz->ASR noised Hinglish set, 300 questions incl. 100 near-miss negatives):
  2-3 days. Total ~2-2.5 weeks for one student [U].

## 6. Risks
- Confident wrong answers (the main risk); mitigated by abstention, owner approval, near-miss negatives in eval.
- Stale answers after doc edits; mitigated by hash invalidation.
- Code-mixed/8 kHz ASR noise breaks matching more than text benchmarks suggest (no direct measurement exists).
- Voice change or clone revocation invalidates rendered audio (consent: delete clips when consent is revoked).
- Licence trap: vCache code is NC-ND; do not ship it.
- Answers sounding canned / prosody mismatch with live sentences that follow.

## 7. Evaluation protocol
300+ realistic questions (EN/HI-Devanagari/Hinglish-Roman, Marathi subset), each spoken via TTS and a few real voices,
passed through 8 kHz mu-law and phone ASR. Include 1/3 near-miss negatives (same words, different intent, e.g. "cancel
appointment" vs "book appointment"). Report hit rate, false-hit rate with 95% CI, abstention rate, p50/p95
time-to-first-audio from `_report_turn`, listening MOS for cached vs live sentences. Ship gate: false-hit <= 1% with CI
upper bound <= 2%.

## 8. Verdict
**Build it, narrowly.** It is the single largest phone-latency win available for common questions (~1-1.6 s per hit,
[U]) and fits the consent/approval story (owner-approved, watermarked answers). Implement our own matcher (MIT/Apache
pieces: BGE-M3, existing reranker, IndicXlit MIT) using vCache's per-entry threshold idea, not its code. Ship behind a
per-agent flag, start in "shadow mode" (match and log, but play live answer) for a week of calls to measure the
real false-hit rate before serving.

## Sources opened this pass
- https://arxiv.org/abs/2502.03771 and https://arxiv.org/html/2502.03771 (vCache)
- https://github.com/vcache-project/vCache
- https://github.com/zilliztech/GPTCache
- https://github.com/AI4Bharat/IndicXlit
- https://arxiv.org/abs/2209.12944 (abstract only)
- https://arxiv.org/abs/2401.14280 (abstract only)
Secondary (search snippets, not opened): tianpan.co semantic-cache blog; arXiv:2305.03207; US patent 7308407.
