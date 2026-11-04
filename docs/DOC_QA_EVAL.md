# Document question answering: accuracy evaluation

Measured on 19 October 2026 against the original code (commit `a9f150c`), on a 16 GB M4 Mac, without touching the
running app or its data. Script: `backend/bench/doc_qa_eval.py`; questions: `backend/bench/data/doc_qa_questions.json`;
raw results: `backend/bench/results/doc_qa_*` (git-ignored).

## Summary in plain language

- **Answers are right about 9 times in 10.** Of 108 caller questions answered by the production model (Qwen 3.8 27B on
  Groq), 97 were fully correct (89.8%), 3 partly correct and 8 wrong. English 93%, Hinglish 94%, Hindi 83%.
- **About 1 reply in 18 states something the documents do not say** (6 of 108, 5.6%). These are the replies that
  matter most on a phone line: "yes, the bone doctor sits on Tuesday" (he does not), "Ayushman Bharat is not
  accepted" (the document never says so), an MBA fee of 1,06,000 spoken instead of 1,60,000.
- **Most of those come from one habit.** When the right passage is not among the three the model is given, it fills
  the gap instead of saying "I don't know". The other cause is a caller's leading question in Hindi ("Thursday ko
  milengi na?"), which the model agreed with although the passage in front of it said otherwise.
- **Finding the right passage is good but not perfect.** The right passage is among the three handed to the model
  for 94.7% of questions, and among the top four for 98.2%. Every miss sat at rank 4, so **passing four passages
  instead of three would have fixed all four misses**, at about +10% prompt tokens and no added delay.
- **Language handling is solid.** Every reply was in the caller's language and script (English, Hindi in Devanagari,
  Hinglish in Roman letters). The app's own language check wrongly flagged one correct Hindi reply and regenerated it.
- **Unanswerable questions are mostly handled well.** 13 of 15 were declined with an offer to connect to staff; 2 got
  an invented answer.
- **It is fast.** Retrieval plus the language model took 0.43 s at the median and 0.68 s at the 90th percentile
  (speech synthesis is not included).
- **The free Groq tier is the real limit.** Qwen allows 200,000 tokens a day. A turn of an agent with booking tools
  sends about 2,200 tokens, so the quota covers about 90 calls a day. This evaluation used it up, after which the
  production fallback chain failed in two ways (details under Operational findings). The HOD-office document could
  only be tested on 6 of its 31 questions because of this.

## What was measured, and how

### Test set

133 questions written by hand from the four documents (no model wrote them). 36 are about the Sahyadri Care Hospital
PDF (`docs/demo`), 33 about the clinic receptionist template, 33 about college admissions and 31 about the HOD office.

| | English | Hindi (Devanagari) | Hinglish (Roman) | total |
|---|---:|---:|---:|---:|
| questions | 47 | 44 | 42 | 133 |

The question types are direct (15), paraphrase with no keyword overlap (25), numbers (27), yes/no (17), two facts (13),
near-miss distractors that state a wrong day, value or department (16) and unanswerable (20, 15%). They are phrased the
way callers speak, for example "bachche ke doctor ki pehli baar ki fees kitni hai?", "हड्डी वाले डॉक्टर मंगलवार को बैठते हैं
ना?" or "how much to see a heart specialist there". Every answerable question has gold evidence strings, which are exact
substrings of the text the app stores (the PDF extracted with the app's pypdf code, a template with its sample
banner). The script checks that every string is present before it runs.

### Retrieval (offline, no Groq)

The production pipeline ran in a separate `python -I` process: `services/chunking.py` (120 words, 20 overlap),
BGE-M3 (float16 for questions, float32 for passages, as the app does), the passages stored in a temporary
`LocalStore`, and ranking by `LocalStore.match_chunks` (dense cosine and BM25 fused with RRF, which is what
`retrieve_context` calls). Each agent holds its own document, as in the app, so a question is searched among 7 to 11
passages. A question is a hit at k when ALL its evidence strings are in the top k passages. MRR uses the first passage
with any evidence string.

### End to end (real answer path, isolated data)

`backend.py` was imported in a separate process with every path it writes to moved to a temp directory. Instant answers
were off, Google Calendar was off, the tool servers wrote to temp files and only the Groq provider was allowed. The agents
were made by the real template code (`routers.templates._create_from_template`). The hospital agent is the clinic
template personalised for "Sahyadri Care Hospital, Pune", with the PDF uploaded through `backend._extract_knowledge_text`
and `backend.ingest_document`. Each question was one typed turn through **`backend._turn_blocking`** (the path of
`POST /api/agents/{id}/turn` with `text`). That covers the language policy (`_turn_language`), retrieval
(`retrieve_context`, top 3), the agent's MCP tools (`_turn_tools`, `_tool_turn`), the system prompt (`_turn_messages`),
`services/llm.py` with its Groq fallbacks, and `_enforce_language`. Only speech synthesis, the watermark and the audio
file were stubbed (`_turn_spoken`, `_turn_voice`, `_turn_synth`, `provenance.apply_provenance`, `_turn_store`). Every
question was a new conversation, so every reply starts with the AI disclosure, which is removed before grading.

### Grading

- **By hand (primary).** I graded all 108 replies against the gold answer and the passages the model was given, for
  correct (yes, partial or no) and grounded (no fact the passages or the agent's instructions do not support).
- **LLM judge.** Groq `openai/gpt-oss-120b`, a different model from the answering one, with a strict rubric and two
  answers of the same document per call. It graded 87 of the 108 answers before its daily quota ran out; it agreed
  with the hand grades on 94% (see [Judge validation](#judge-validation)).
- **Deterministic checks.** Reply language via `language_policy.reply_matches`, the app's own check, against the
  caller's real language. Key numbers of the gold answer present in the reply. Numbers in the reply that appear in no
  document, prompt or question.

### Which code ran

Other work changed the backend while this evaluation was running. The baseline is therefore pinned to the original
code. The retrieval numbers and 10 control answers came from a `git archive` export of HEAD `a9f150c`; the working
tree matched HEAD when this work began. The 108 graded answers come from the first full run, which started at 16:16,
before the edits reached the answer path. Their system prompts, passages and user messages were checked to be
identical to what the HEAD code builds. Every college prompt was rebuilt with HEAD code (66 of 66 identical), the
hospital prompts were compared with the HEAD-export run, and all 318 passages are output of the HEAD chunker.

The script pins the newer switches to the original behaviour by default (`QUESTION_INDEX=off`,
`STRUCTURED_CHUNKING=off`, `GROUNDING_PROMPT=v1`, `CONFIDENCE_GATE=off`, `ANSWER_GUARD=off`, `FACT_SHEET=off`,
`QUERY_CONDENSE=off`, `ASR_VOCAB_PROMPT=off`, `RERANKER=off`, `RETRIEVAL_FUSION=rrf`, `LLM_ANSWER_TEMPERATURE=0.3`).
Every result file records the git commit, `git status`, the pins and the retrieval and answer settings. Use
`--no-baseline-pins --set KEY=VALUE` to measure the new code.

## Retrieval results

113 answerable questions (unanswerable ones have nothing to find).

| variant | hit@1 | hit@3 | hit@4 | hit@5 | MRR | notes |
|---|---:|---:|---:|---:|---:|---|
| **production**: 120-word passages, hybrid (the model gets the top 3) | 76.1% | **94.7%** | 98.2% | 98.2% | 0.903 | 7-11 passages per agent |
| dense only (`HYBRID_SEARCH=off`) | 72.6% | 92.9% | 97.3% | 97.3% | 0.877 | |
| BM25 only | 60.2% | 76.1% | - | 77.9% | 0.715 | finds little for Hindi |
| 80-word passages, hybrid | 77.0% | 96.5% | 99.1% | 99.1% | 0.909 | smaller prompt than production |
| 200-word passages, hybrid | 78.8% | 97.3% | 99.1% | 99.1% | 0.906 | about 60% more prompt tokens per turn |
| all four documents in ONE agent, hybrid | 61.9% | 91.2% | 91.2% | 92.0% | 0.806 | 35 passages: a multi-document agent |
| production + reranker (bge-reranker-v2-m3) | not run | | | | | model not downloaded; see recommendations |

Hit@1 counts a question only when ALL its evidence is in the first passage. A two-fact question whose facts sit in two
passages can never score there, which is why hit@1 for "two facts" is 31% while its MRR is 1.00.

Production retrieval by slice (hit@1 / hit@3 / hit@5, MRR):

| slice | n | production | dense only | BM25 only |
|---|---:|---|---|---|
| English | 40 | 78 / 95 / 100, 0.91 | 72 / 98 / 100, 0.89 | 75 / 92 / 95, 0.86 |
| Hindi | 37 | 70 / 92 / 95, 0.86 | 70 / 92 / 95, 0.86 | **24 / 43 / 43, 0.36** |
| Hinglish | 36 | 81 / 97 / 100, 0.94 | 75 / 89 / 97, 0.89 | 81 / 92 / 94, 0.91 |
| direct | 15 | 93 / 100 / 100, 0.97 | | |
| paraphrase | 25 | 76 / **84** / 96, 0.83 | | |
| number | 27 | 81 / 96 / 96, 0.89 | | |
| yes/no | 17 | 82 / 100 / 100, 0.91 | | |
| two facts | 13 | 31 / 100 / 100, 1.00 | | |
| distractor | 16 | 81 / 94 / 100, 0.89 | | |
| hospital PDF | 31 | 68 / **87** / 97, 0.83 | | |
| clinic | 28 | 79 / 96 / 96, 0.93 | | |
| college | 28 | 75 / 96 / 100, 0.90 | | |
| HOD office | 26 | 85 / 100 / 100, 0.96 | | |

What this shows:

- **Hybrid search pays for itself.** It matches dense-only on Hindi, where BM25 finds almost nothing because the
  documents are in English, and adds 8 points at hit@3 on Hinglish, where BM25 matches the English words callers mix
  in.
- **The weak spot is the PDF.** pypdf gives no headings, so a 120-word window mixes contacts, parking and the whole OPD
  table. "Bone doctor" and "aankhon ke doctor" land at rank 4. The six production misses at top 3 are five
  paraphrases or leading questions about the PDF's OPD table or contacts and one college paraphrase ("lock my seat").
- **Multi-document agents lose about 4 points at hit@3 and 14 at hit@1**, Hindi most of all (84% at hit@3).
- Query encoding takes 46 ms at the median and 85 ms at p95 (BGE-M3 fp16 on the CPU); ranking 7-35 passages is under
  10 ms.

## End-to-end results (108 answered questions)

| slice | n | correct | correct or partial | hallucination | right language | unanswerable handled |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 108 | **89.8%** | 92.6% | **5.6%** | **100%** (app check: 99.1%) | **13 / 15** |
| English | 41 | 92.7% | 95.1% | 2.4% | 100% | 5 / 5 |
| Hindi | 35 | 82.9% | 88.6% | 11.4% | 100% (app check: 97.1%) | 5 / 6 |
| Hinglish | 32 | 93.8% | 93.8% | 3.1% | 100% | 3 / 4 |
| direct | 11 | 100% | 100% | 0% | | |
| paraphrase | 20 | 80.0% | 85.0% | 5.0% | | |
| number | 23 | 95.7% | 95.7% | 4.3% | | |
| yes/no | 15 | 100% | 100% | 0% | | |
| two facts | 11 | 90.9% | 100% | 0% | | |
| distractor | 13 | **76.9%** | 84.6% | **15.4%** | | |
| unanswerable | 15 | 86.7% | 86.7% | 13.3% | | |
| hospital PDF | 36 | 88.9% | 88.9% | 8.3% | | 4 / 5 |
| clinic | 33 | 90.9% | 93.9% | 6.1% | | 4 / 5 |
| college | 33 | 87.9% | 93.9% | 3.0% | | 5 / 5 |
| HOD office | 6 | 100% | 100% | 0% | | - |

Correct and hallucination are the hand grades. Right language is my reading of all 108 replies; the app's own check
agreed on 107 (see failure analysis). All 108 replies came from `qwen/qwen3.8-27b`; the two answers that a fallback
model had touched were asked again and the qwen reply kept. The missing questions are HOD 7-31 (25), which were not
answered by qwen because the daily quota ran out.

**Latency per answer** (retrieval plus the language-model call or calls, measured inside the turn): median 426 ms,
p90 683 ms, max 1.9 s. Retrieval alone takes 81 ms at the median and 204 ms at p90; the model 328 ms and 533 ms.
Speech synthesis and network transfer to the caller are not included. Prompt size per call (median): college 774
tokens, clinic 1,858, hospital 2,209, HOD 2,610. The difference is the tools section of the system prompt, which is 60%
of the system prompt of an agent with booking tools.

## Failure analysis

Every reply not graded fully correct (11 of 108), grouped by cause.

| cause | count | questions |
|---|---:|---|
| Retrieval miss, then the model **invented** an answer | 2 | hosp-02, hosp-17 |
| Retrieval miss, then the model **declined** safely | 2 | hosp-27, college-03 |
| Right passage in context, wrong or incomplete answer | 5 | clinic-15 (wrong), college-13 (wrong), clinic-21, college-12, college-20 (partial) |
| Unanswerable question, model invented an answer | 2 | hosp-18, clinic-30 |
| Language slip | 0 | (one false alarm by the app's checker: college-18) |
| Judge error | see [Judge validation](#judge-validation) | |

**1. Retrieval miss followed by an invented answer (2).** The model does not say "I don't know" when the passage it
needs is missing; it reaches for something plausible.
- hosp-02, "When does the bone doctor sit?": the orthopaedics row was in the 4th passage. Reply: *"Our Orthopaedics
  clinic runs Monday to Friday, 9 am to 1 pm."* The document says Monday, Wednesday and Friday, 11 am to 3 pm.
- hosp-17, "हड्डी वाले डॉक्टर मंगलवार को बैठते हैं ना?": same missing passage. Reply: *"जी हाँ ... मंगलवार को भी बैठते हैं"*,
  agreeing with the caller's wrong day.

**2. Retrieval miss followed by a safe decline (2).** hosp-27 ("aankhon ke doctor kab baithte hain?"; the eye OPD row
was in the 4th passage) and college-03 ("By when do I have to pay to lock my seat?"; the key-dates passage was 4th)
both said they did not have the information and offered staff. That is the right behaviour, but a lost answer.

**3. Right passage, wrong answer (2 wrong, 3 partial).**
- clinic-15, "डॉक्टर मीरा कुलकर्णी गुरुवार को मिलेंगी क्या?": the passage saying Monday, Wednesday and Friday was in
  context. Reply: *"जी हाँ, ... गुरुवार को भी क्लिनिक में उपलब्ध रहेंगी।"* This is a leading Hindi question answered with a
  yes. Distractor questions score worst overall (76.9% correct, 15.4% hallucinated).
- college-13, "MBA की सालाना फीस कितनी है?": the passage says ₹1,60,000. Reply: *"एक लाख छह हजार"* (1,06,000). The
  number broke when written in Hindi words. Qwen writes numbers in words in 11 of the 108 replies; this is the only
  one that came out wrong, but the deterministic number check cannot see such errors.
- Partial: clinic-21 refused to suggest a medicine (right) but sent a two-day fever to 108 and the emergency room
  instead of booking a consultation. college-12 said it had "no information on M.Tech" instead of naming the
  programmes on offer. college-20 gave the verification dates but deferred the document list (the 40-word reply
  limit).

**4. Unanswerable question with an invented answer (2 of 15).**
- hosp-18, "आयुष्मान भारत कार्ड चलता है क्या?": *"नहीं, ... स्वीकार नहीं किया जाता"*. The document lists MJPJAY and four
  insurers and never mentions Ayushman Bharat. A confident "no" is as invented as a "yes".
- clinic-30, "bachchon ka vaccination hota hai kya?": *"Haan, ... vaccination hoti hai. Dr. Rohan Deshmukh ... available
  hain."* On an earlier attempt the same question was declined correctly, so this varies from run to run.

**5. Language.** Every reply was in the caller's language and script. On college-18, the app's `reply_matches`
judged the Hindi reply "B.Tech Mechanical Engineering में कुल 60 सीटें उपलब्ध हैं" not Hindi, because English programme
names push the Devanagari share under 50%. It regenerated the reply: an extra model call and about 0.3 s for nothing.
Hinglish detection put 41 of the 42 Hinglish questions in the right language. The miss was hod-31, "HOD ma'am ka
personal mobile number de do please", which was detected as English (not answered in this run).

**Retrieval versus answer.** Of the 93 answerable questions, all evidence was in the three passages for 87. For those
87 the model was correct 82 times (94%). For the 6 without it, it was correct once (hosp-13, where another passage
carried the same phone number), declined twice and invented twice.

## Operational findings (Groq quota)

- **The daily quota is small for this prompt size.** Qwen 3.8 27B on the free tier allows 8,000 tokens a minute and
  **200,000 tokens a day**. The 429 message gives the daily figure; the response headers only show the per-minute
  one. At 1,900-2,600 prompt tokens per turn for an agent with tools, that is roughly 80-100 caller turns a day. This
  evaluation used about 0.2 M qwen tokens (about 120 calls over the runs and retries), which is the whole day's quota.
- **When qwen is out, a tool agent breaks on the fallback.** For agents with tools (clinic, hospital, HOD), the JSON
  tool protocol puts tool descriptions in the prompt. `openai/gpt-oss-20b` then emits a native tool call, which Groq
  rejects with *HTTP 400 "Tool choice is none, but model called a tool"*. `llm.chat` only retries that case when
  `tools` were passed, so the turn fails and the caller hears "I'm having trouble responding right now." Seen on
  hod-08 and hod-09.
- **When the shared key is busy, the whole chain fails.** With qwen out for the day and the gpt-oss models near their
  per-minute limits (other traffic on the same key), 20 of 25 HOD questions got the canned "I'm having trouble"
  reply (`doc_qa_answers-…-baseline-hod-quota-out.json`).
- **Impact on the live app.** The running app uses the same key, so for the rest of the day it answers with the
  fallback models, with the tool-agent failure above.

## Recommendations

Measurement only; nothing in the app was changed. Each item says how to test it with this script.

1. **Pass 4 passages instead of 3 (`RETRIEVAL_TOP_K = 4`).** All four top-3 misses sat at rank 4, so hit@3 goes from
   94.7% to 98.2% on this set. On today's answers that converts two hallucinations (hosp-02, hosp-17) and two lost
   answers into answerable turns: about +4 points of accuracy and −2 points of hallucination. Cost: one more passage,
   about 170 tokens or +8-20% of the prompt, and no added latency. Test: `--answer-top-k 4` against this baseline.
2. **Make "not in the passages" an explicit instruction for leading questions.** Add to the rules: *"If the caller
   assumes a day, time, price or service, check it against the context; if the context does not confirm it, say so.
   Never answer yes or no about something the context does not mention."* This targets clinic-15, hosp-17, hosp-18
   and clinic-30 (4 of the 6 hallucinations). Test: the 16 distractor and 20 unanswerable questions
   (`--ids` or `--langs hi`) with the prompt variant; expect distractor accuracy above 90% and unanswerables handled
   20 of 20.
3. **Ask for numbers in digits.** For example: *"Write prices, phone numbers and times in digits (₹1,60,000), never in
   words."* That removes the college-13 failure mode and makes the number check meaningful (11 replies used words).
   Check first that the Hindi voice reads Indian digit grouping well.
4. **Add heading detection for PDF text.** A short line on its own followed by table rows ("Outpatient (OPD) timings",
   "Consultation fees") could be treated as a heading by `chunking.chunk_text`. The PDF scores 87% at hit@3 against
   96-100% for the markdown documents. This is what `STRUCTURED_CHUNKING` appears to target, and this set measures it
   directly (`--retrieval-only --no-baseline-pins --set STRUCTURED_CHUNKING=on`).
5. **Skip the reranker for single-document agents.** It was not measured here, because bge-reranker-v2-m3 (about
   2.3 GB) is not downloaded and this run does not download models. The earlier measurement was about 945 ms per
   question for 20 candidates on this Mac. With 7-11 passages per agent and every miss at rank 4, top-4 gets the same
   gain for no latency, while a reranker would roughly double the 0.43 s median turn time. Re-measure only for
   multi-document agents (the pooled variant, 91% at hit@3), after downloading the model with your OK
   (`--retrieval-only --variants prod,pooled,rerank`).
6. **Leave chunk size at 120 words.** 80 and 200 words differ by 2-3 questions at hit@3 (within noise for 113
   questions), and 200 words costs about 60% more passage tokens. 80 words is worth a re-test together with top-4.
7. **No Hinglish query normalisation is needed for retrieval.** Hinglish is already the best-retrieved language (97%
   at hit@3), helped by BM25 on its English words. The open item is hod-31-style mixed sentences that the language
   detector reads as English.
8. **Fix the app's language check for Hindi with English names.** Latin words that also appear in the passages or the
   question (programme names, "B.Tech") should not count against a Hindi reply. That saves a regeneration (a model
   call and about 0.3 s) on such replies.
9. **Fix the Groq fallback robustness (most urgent for real calls):**
   (a) on the HTTP 400 "Tool choice is none" error, retry with the tool section removed, or prefer a fallback model
   that follows the JSON protocol;
   (b) read "tokens per day" from the 429 and mark the model unavailable until the reset (a `GROQ_DEAD_MODEL_S`
   switch is being added);
   (c) shrink the tools section, which is 60% of the system prompt, or send it only when the turn looks like a booking.
   That would roughly double the number of turns the daily quota covers.

## Fixes after the doc QA evaluation

Made after this evaluation; the numbers above are the baseline and were not re-measured (Groq's daily quota was used
up). Re-run `--no-baseline-pins` (see How to re-run) to measure them. Every fix has offline tests with mocked HTTP.

| # | fix | where | tests |
|---|---|---|---|
| 9a | **gpt-oss tool calls no longer fail the turn.** Groq answers the JSON tool prompt's native call with HTTP 400 `tool_use_failed` (blocking) or HTTP 200 then `event: error` (streamed); both carry the attempted call in `failed_generation` (checked against Groq on 2026-10-19). That call is now used as the model's reply in the JSON protocol when it names one of the agent's tools, so the tool turn carries on. Otherwise the model is asked once more with no tools and an instruction to answer in words; a second failure moves on to the next Groq model. A malformed native call (tool_choice "auto") becomes `tool_calls`. | `services/llm.py` (`failed_tool_call`, `_recover_tool_use`, `_openai_stream`), `services/tools.py` (passes `json_tools`) | `tests/test_llm_tool_use_recovery.py` |
| 9b | **Daily quota.** A 429 for tokens or requests per DAY pauses that model until the reset time. The time comes from "try again in 12m35s", else retry-after, else x-ratelimit-reset-requests for RPD, else the next midnight UTC. It is logged once and listed as `llm.groq_paused` in `/api/health`. When every Groq model is paused, Groq is skipped without a request. A per-minute 429 behaves as before. | `services/llm.py` (`daily_quota_pause`, `_mark_gone`, `_groq_models`) | `tests/test_llm_daily_quota.py` |
| 1 | **Four passages.** `RETRIEVAL_TOP_K` defaults to 4 and can be set in the environment; the v2 prompt's passage cap (`GROUNDING_TOP_K`) follows it. Order unchanged: best passage first, question last. | `backend.py`, `services/grounding.py` | `tests/test_prompt_layout.py`, `tests/test_grounding.py` |
| 2, 3 | **Grounding prompt v2.** It now never confirms a day, time, price, person or service the caller assumes unless the context states it, and says what the context says instead. A new Hindi example corrects a wrong day. It never says yes or no about something the context does not mention. It writes numbers, amounts, times and phone numbers in digits in every language. The prompt is 417 words (was 370), 671 Qwen3 tokens (was 562). `GROUNDING_PROMPT=v1` still switches back. | `services/grounding.py` | `tests/test_grounding.py` |
| 3 | **Hindi speech of digits.** In espeak-ng's Hindi (Kokoro's Hindi G2P), "1,60,000" was read as "ek, saath, shunya shunya shunya", "₹" was silent and a phone number came out as one large amount. Before speech only, the digits of a Hindi or Hinglish reply are now rewritten: grouped amounts lose their commas (read "ek lakh saath hazaar"), ₹ and Rs become "रुपये", phone numbers are read digit by digit, clock times become "सुबह साढ़े 9 बजे" and "1.5 लाख" becomes "डेढ़ लाख". | `services/speech_numbers.py`, `backend._turn_spoken` | `tests/test_speech_numbers.py` (includes the real G2P when installed) |
| 8 | **Language check.** In a Hindi reply, Latin names and terms no longer count against it: capitalised words, acronyms, words with a dot or a digit, and words the caller used. At least a fifth of the letters must still be Devanagari. On the 108 graded replies, only college-18 changes (now accepted). | `services/language_policy.py` (`reply_matches`), `backend._language_known_text` | `tests/test_language_policy.py` |
| 9c | **Shorter tools section.** The rules were rewritten more tightly. The clinic-only lines now appear only for agents with `book_appointment`, and the meeting lines only for agents with `request_meeting`. Argument notes that only restate the date and time format were dropped, string types are no longer printed, and docstrings are squashed onto one line. The built-in MCP servers' descriptions were shortened, which also shrinks the native schemas. | `services/tools.py`, `mcp_servers/appointments.py`, `mcp_servers/meetings.py` | `tests/test_tools_prompt.py`, the tool suites |

Tools section of the system prompt (JSON protocol). Counts use the Qwen3 tokenizer (`tokenizer.json` of the cached
Qwen3-Reranker-0.6B). On the 106 graded prompts it came within 4% of the `prompt_tokens` Groq reported (median ratio
0.979).

| agent | before | after | change |
|---|---:|---:|---:|
| clinic / hospital (appointments, 4 tools) | 849 tokens (606 words) | 614 tokens (429 words) | −235 (−28%) |
| HOD office (meetings, 4 tools) | 1,106 tokens (801 words) | 776 tokens (535 words) | −330 (−30%) |
| native schemas (`tools` JSON), clinic / HOD | 688 / 873 | 595 / 772 | −93 / −101 |

Net prompt per turn, against the code before these fixes:
- An agent with tools: −235 or −330 (tools section), +109 (grounding rules) and about +170 (the fourth passage). That
  is about +45 tokens for the clinic and −50 for the HOD office.
- An agent without tools: about +280.

## Caveats

- **Sample size.** 108 graded answers (113 for retrieval). A 95% interval on 90% correct is about ±6 points, and the
  slices (11-41 questions) are wider still. The per-type and per-language differences above are indications, not
  proofs.
- **Single run, temperature 0.3.** Answers vary between runs: clinic-30 was declined correctly on one attempt and
  answered with an invented "yes" on the next.
- **HOD office under-sampled.** Only 6 of its 31 questions got a production-model answer, so the document-level
  numbers cover hospital, clinic and college. HOD retrieval, though, was measured on all 26 answerable questions (100%
  at hit@3).
- **Gold evidence is strict.** hosp-13 counts as a retrieval miss because the contact table was not retrieved, but
  the booking passage gave the same number and the answer was correct.
- **Fallback-model share.** All 108 graded replies are from qwen. Two first attempts went through gpt-oss-20b
  (clinic-12 and clinic-30, asked again). Of the 25 HOD questions run after the quota ran out, 4 were answered by
  gpt-oss-20b, 1 by gpt-oss-120b and 20 failed. Those 25 are reported only under Operational findings.
- **Judge limits.** The LLM judge reads passages and replies but cannot hear the voice. It shares a quota with
  everything else on the key; see [Judge validation](#judge-validation).
- **Not measured.** Speech recognition errors (questions were typed), speech synthesis, multi-turn conversations
  (every question was a first turn) and the streaming phone path (`_turn_event_stream`), which uses the same prompt
  and retrieval but streams the model reply.

## Judge validation

`openai/gpt-oss-120b` (low reasoning effort, two answers per call) graded 87 of the 108 answers: all hospital and
clinic answers and college-01 to college-18. Then its own 200,000-token daily limit ran out, because the key is shared
with the live app and other work. I graded all 108 by hand before reading its verdicts. On the 87 graded by both:

| | agreement |
|---|---:|
| correct (yes / partial / no) | 82 / 87 (94%) |
| correct, yes vs not yes | 83 / 87 (95%) |
| grounded | 83 / 87 (95%) |
| right language | 87 / 87 (100%) |

Both found 77 of the 87 fully correct. The disagreements:

- **hosp-18 (a real judge error).** The judge accepted *"Ayushman Bharat card is not accepted"* as a correct,
  grounded answer to an unanswerable question. The document never says that, so this is the most important kind of
  miss for a judge. **An LLM judge alone undercounts invented negative answers.**
- **hosp-27, college-03 (judge too harsh on grounding).** Both replies declined when the passage was missing. The judge
  counted "I don't have that information" as an unsupported claim, which the rubric says it is not. The correct grade
  ("no") was the same.
- **clinic-21 (judgement call).** The judge said "no, ungrounded" where I said "partial" for the 108 advice on a fever.
  Either is defensible.
- **clinic-28, college-06 (judge stricter).** It marked an answer partial for leaving out a detail of the reference
  answer that the caller did not ask for (carrying old reports; the separate exam fee).
- **college-12 (judge more lenient).** It accepted "no information on M.Tech" as a full correction of the premise.

In the smoke test (4 answers) the judge also once called a Roman-script Hinglish reply "Devanagari". The deterministic
language check is the reliable one for script. On the 87 answers, judge-only scoring would give the same accuracy
(88.5%) but a hallucination rate of 9.2% against 6.9% by hand, because of the grounding over-calls and one missed
invention. The headline numbers in this report use the hand grades. The judge verdicts are stored next to them in the
results file (`judge`, `hand_check`), so `--summary` prints both.

## How to re-run

```
# retrieval only (no Groq; about 1 minute, about 2.5 GB of RAM)
backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --retrieval-only
# end to end + judge (about 133 x 2,000 Groq tokens: more than half of qwen's free daily quota)
backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --answers-only --label=-myrun
# "after" numbers for new code: drop the pins, set what is under test
backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --no-baseline-pins --set STRUCTURED_CHUNKING=on --label=-after
# a slice, a smaller smoke test, or re-judging saved answers
backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --answers-only --docs hod --langs hi --limit 10
backend/.venv/bin/python -I backend/bench/doc_qa_eval.py --judge-only backend/bench/results/doc_qa_answers-<stamp>.json
```

Result files: `doc_qa_retrieval-20261019-171211-baseline.json` (retrieval, HEAD export),
`doc_qa_answers-20261019-161639-k3-baseline-main.json` (the 108 graded answers, with judge verdicts and the hand
grades under `hand_check`), `doc_qa_answers-20261019-171218-k3-baseline.json` (10 control answers from the HEAD export)
and `doc_qa_answers-20261019-172557-k3-baseline-hod-quota-out.json` (HOD questions after the quota ran out). Files
marked `-superseded` hold the earlier raw runs they were assembled from, including a run on mixed code that is not
used.
