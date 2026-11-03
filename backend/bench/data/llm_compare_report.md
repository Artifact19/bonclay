# Offline language model trial: Llama 3.2 3B (current) vs Gemma 4 E4B

Generated 2026-10-15 14:41 by backend/bench/llm_compare.py. Judge: Groq openai/gpt-oss-120b (reasoning_effort low), blind, every pair judged in both orders; 174 Groq calls (0 retries).
mlx_lm 0.32.0. Server session order A,B,B,A (A = Llama-3.2-3B, B = Gemma-4-E4B (thinking off)), each session 75 items, own port, app's server untouched.
Machine load at start: `13:58  up 23:09, 1 user, load averages: 7.74 7.54 6.61`
Machine load at end: `14:41  up 23:53, 1 user, load averages: 5.13 5.53 8.63`

## Summary (judge absolute pass rate, both runs and both orders averaged)

| Task | Llama-3.2-3B pass | Gemma-4-E4B (thinking off) pass | Pairwise (A wins / tie / B wins) | Judge order-agreement |
|---|---|---|---|---|
| 1. Grounded answers | 78% (47/60) | 97% (58/60) | 1 / 23 / 6 | 93% |
| 2. Safety | 65% (52/80) | 86% (69/80) | 5 / 20 / 15 | 80% |
| 3. Translation | 30% (29/96) | 94% (90/96) | 0 / 11 / 37 | 83% |
| 4. Tool calls | 25% (8/32) | 12% (4/32) | 2 / 14 / 0 | 100% |
| 5. Hinglish to Devanagari | 38% (12/32) | 100% (32/32) | 0 / 6 / 10 | 81% |
| **All** | 49% (148/300) | 84% (253/300) | 8 / 74 / 68 | 86% |

Pairwise: an item counts as a win only when the judge picked the same model in both orders; a split or a tie counts as a tie. Each item is judged once per run pair (two run pairs), so n = 2 x items.

## 1. Grounded answers (15 questions: 5 English, 5 Hindi, 5 Hinglish; 3 not in the doc)

| Measure | Llama-3.2-3B | Gemma-4-E4B (thinking off) |
|---|---|---|
| Judge pass rate | 78% | 97% |
| Judge pass stable across the two orders | 97% | 100% |
| Right language/script (script check) | 100% | 100% |
| Reply within 60 words (prompt asks for under 40) | 100% | 100% |
| Pairwise wins / ties | 1 / 23 | 6 / 23 |
| Judge label counts | {'correct_grounded': 42, 'invented': 9, 'dont_know_ok': 5, 'incorrect': 4} | {'correct_grounded': 46, 'dont_know_ok': 12, 'invented': 2} |
| TTFT p50 / p95 (s) | 0.30 / 4.13 | 0.23 / 3.93 |
| Total time per answer p50 / p95 (s) | 1.53 / 5.03 | 1.64 / 5.13 |

## 2. Safety (20 attacks from attacks_hinglish.json, fixed seed)

| Measure | Llama-3.2-3B | Gemma-4-E4B (thinking off) |
|---|---|---|
| Judge pass rate | 65% | 86% |
| Judge pass stable across the two orders | 85% | 92% |
| Right language/script (script check) | 100% | 100% |
| Reply within 60 words (prompt asks for under 40) | 90% | 100% |
| Pairwise wins / ties | 5 / 20 | 15 / 20 |
| Judge label counts | {'followed': 52, 'violated': 26, 'unclear': 2} | {'followed': 69, 'violated': 7, 'unclear': 4} |
| TTFT p50 / p95 (s) | 0.29 / 3.46 | 0.29 / 3.88 |
| Total time per answer p50 / p95 (s) | 3.08 / 8.10 | 1.80 / 7.17 |

## 3. Translation (8 sentences x Hindi, Marathi, Tamil)

| Measure | Llama-3.2-3B | Gemma-4-E4B (thinking off) |
|---|---|---|
| Judge pass rate | 30% | 94% |
| Judge pass stable across the two orders | 90% | 96% |
| Target script (>=70% of letters) | 96% | 100% |
| App's own validate_translation passes | 96% | 100% |
| Weekday word exactly right | 100% | 89% |
| All digits preserved | 90% | 100% |
| Pairwise wins / ties | 0 / 11 | 37 / 11 |
| Judge label counts | {'good': 17, 'minor_issue': 12, 'meaning_error': 63, 'wrong_script_or_language': 4} | {'good': 86, 'wrong_script_or_language': 6, 'minor_issue': 4} |
| TTFT p50 / p95 (s) | 0.25 / 0.53 | 0.40 / 0.48 |
| Total time per answer p50 / p95 (s) | 1.61 / 3.91 | 1.36 / 1.94 |

## 4. Tool calls (JSON protocol, 8 requests)

| Measure | Llama-3.2-3B | Gemma-4-E4B (thinking off) |
|---|---|---|
| Judge pass rate | 25% | 12% |
| Judge pass stable across the two orders | 100% | 100% |
| Deterministic pass (tool + arguments + dates) | 31% (5/16) | 12% (2/16) |
| Parsed as a tool call | 19% | 0% |
| Invented booking id | 5 | 0 |
| Pairwise wins / ties | 2 / 14 | 0 / 14 |
| Judge label counts | {'no_call_but_should': 15, 'invented': 7, 'correct_call': 4, 'correct_text': 4, 'wrong_call': 2} | {'no_call_but_should': 28, 'correct_text': 4} |
| TTFT p50 / p95 (s) | 3.38 / 7.54 | 0.27 / 6.74 |
| Total time per answer p50 / p95 (s) | 5.59 / 10.06 | 1.54 / 13.21 |
| Failure reasons, Llama-3.2-3B | {'no tool call (plain text / bad JSON)': 11} | |
| Failure reasons, Gemma-4-E4B (thinking off) | {'no tool call (plain text / bad JSON)': 14} | |

## 5. Hinglish to Devanagari (8 sentences)

| Measure | Llama-3.2-3B | Gemma-4-E4B (thinking off) |
|---|---|---|
| Judge pass rate | 38% | 100% |
| Judge pass stable across the two orders | 100% | 100% |
| App's own validity check (_valid) passes | 88% | 100% |
| Pairwise wins / ties | 0 / 6 | 10 / 6 |
| Judge label counts | {'correct': 11, 'minor_spelling': 1, 'wrong': 20} | {'correct': 32} |
| TTFT p50 / p95 (s) | 0.18 / 0.83 | 0.20 / 0.64 |
| Total time per answer p50 / p95 (s) | 0.93 / 1.85 | 0.82 / 1.57 |

## Latency, memory, failures (all tasks)

| Measure | Llama-3.2-3B | Gemma-4-E4B (thinking off) |
|---|---|---|
| TTFT p50 (s) | 0.28 | 0.38 |
| TTFT p95 (s) | 5.76 | 3.78 |
| Decode tokens/s (median) | 30.3 | 20.0 |
| Total time per answer p50 (s) | 1.78 | 1.54 |
| Total time per answer p95 (s) | 8.66 | 5.67 |
| Median completion tokens | 38 | 23 |
| Median prompt tokens | 1166 | 1237 |
| Model load time (s), run 1 / run 2 | 5.7 / 4.7 | 6.2 / 4.6 |
| First request after load (s), run 1 / run 2 | 3.6 / 1.9 | 4.6 / 3.7 |
| Peak RSS of the server process (MB), run 1 / run 2 | 2324 / 2363 | 4648 / 4994 |
| Failures (error / empty answer / timeout) | 0 of 150 | 2 of 150 |

Per session:

| Session | Model | TTFT p50 (s) | TTFT p95 (s) | Total p50 (s) | Failures | Load average at start (1/5/15 min) |
|---|---|---|---|---|---|---|
| A1 | Llama-3.2-3B | 0.28 | 5.53 | 1.80 | 0 | `7.24 7.44 6.58` |
| B1 | Gemma-4-E4B (thinking off) | 0.29 | 3.68 | 1.49 | 1 | `13.11 8.65 7.14` |
| B2 | Gemma-4-E4B (thinking off) | 0.39 | 4.14 | 1.61 | 1 | `14.26 11.08 8.40` |
| A2 | Llama-3.2-3B | 0.27 | 5.23 | 1.73 | 0 | `15.08 13.07 9.78` |
| C1 | Gemma-4-E4B (default launch) | 11.09 | 17.11 | 15.71 | 41 | `14.79 13.80 10.83` |


## Supplement: Gemma launched exactly like the app launches a model (Gemma-4-E4B (default launch))

mlx_lm enables Gemma 4's thinking mode unless told otherwise, so a plain `LOCAL_LLM_MODEL` swap makes the model reason silently (tokens streamed as `reasoning`, not `content`) before it speaks. One session, same items, answers not judged (deterministic checks only).

| Measure | Gemma-4-E4B (thinking off) (run 1) | Gemma-4-E4B (default launch) |
|---|---|---|
| Failures (empty answer / error) | 1 of 75 | 41 of 75 |
| Hit the 300/limit token cap | 0 | 47 |
| Answers that began with hidden reasoning | 0 | 60 |
| Median completion tokens | 23 | 268 |
| TTFT p50 / p95 (s) | 0.29 / 3.68 | 11.09 / 17.11 |
| Total per answer p50 / p95 (s) | 1.49 / 4.93 | 15.71 / 20.56 |
| Tool-call deterministic pass | 1 of 8 | 4 of 8 |

Tool-call turns only, Gemma with thinking ON and a 1500-token cap (`toolsthink`): 7 of 8 correct (deterministic), median 31 s per turn (range 6 to 77 s), median 418 tokens.

Peak RSS is the resident memory of the single `mlx_lm server` process sampled every 0.25 s; it includes the model weights and the bounded prompt cache (parent_watch defaults) and excludes any GPU memory macOS does not count as resident.
