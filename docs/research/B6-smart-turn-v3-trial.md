# B6. Smart Turn v3 against the silence rule: trial of 22 October 2026

**Question.** Would Pipecat's Smart Turn v3 (BSD-2-Clause, 8 MB int8 ONNX, `pipecat-ai/smart-turn-v3`) end a caller's
turn sooner than the fixed silence rule without cutting people off, in Hindi in particular?

**How.** `backend/bench/turn_end_trial.py` (gen, bench, eval, report) on 140 synthetic Kokoro clips: 20 complete
sentences (7 English, 6 Hindi, 7 Hinglish, that is Devanagari with English loanwords through the Hindi voice) and 20
incomplete sentences cut in two with a 300, 600 or 900 ms pause, either rendered as a pause by the voice ("tts") or
spliced at a quiet point ("cut"). Every clip is replayed through the browser's Silero state machine (positive 0.5,
negative 0.35, redemption 700 ms); Smart Turn is asked at the 224, 320, 416, 512 and 608 ms silence checkpoints.
Policies: the current 700 ms rule, shorter silence-only rules (512, 416 ms), Smart Turn once at 224 ms (x1) or at
every checkpoint (x5), thresholds 0.5 / 0.8 / 0.9, each falling back to the 700 ms rule.

**Verdict: not adopted.** On this set Smart Turn ends complete Hindi turns about 320 ms sooner (mean delay after the
true end of speech 472 ms at threshold 0.8 against 792 ms today) but cuts Hindi callers off in the middle of a thought
far more often than the current rule: at a 300 ms mid-sentence pause 3 of 6 Hindi clips are ended (current rule 0 of
6), at 600 ms 3 of 6 (current 0 of 6). Hindi is the language the model separates worst: P(complete) at the true end is
0.84 on average (English 0.96) and 0.51-0.84 in the middle of a pause, so no threshold divides the two. The plan's kill
criterion for this item ("cut-ins rise") is met. Inference itself is cheap: 13 ms median on the M4 with 4 threads.

**Caveats.** Synthetic speech: Kokoro's Hindi voices are graded C by their author, and a pause rendered by a TTS voice
is not a human hesitation. Smart Turn reports 93% accuracy on real Hindi. A real-caller set (plan 3.5) could overturn
this; until then the fixed rule stays.

**Side finding worth acting on.** A 512 ms silence rule, close to the phone path's `PHONE_END_SILENCE_MS` (500 ms),
cuts Hindi 600 ms mid-thought pauses 3 of 6 times on this set, where 700 ms cuts none. On a real call the speculative
start (250 ms) hides most of a longer wait, so `PHONE_END_SILENCE_MS=700` for Hindi-first agents is the next thing to
measure, on real calls.

Raw report (the three tables the bench prints; "end of B" is the second half of a split sentence):


## model smart-turn-v3.0  (140 files)

### [tts] false cut-offs (turn ended during the mid-thought pause) / n
| policy | pause | en | hi | hinglish | all |
|---|---|---|---|---|---|
| silence 700 ms (current) | 300 ms | 0/7 | 0/6 | 0/7 | 0/20 |
| silence 700 ms (current) | 600 ms | 0/7 | 0/6 | 0/7 | 0/20 |
| silence 700 ms (current) | 900 ms | 6/7 | 6/6 | 7/7 | 19/20 |
| silence 512 ms | 300 ms | 0/7 | 0/6 | 0/7 | 0/20 |
| silence 512 ms | 600 ms | 1/7 | 3/6 | 6/7 | 10/20 |
| silence 512 ms | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| silence 416 ms | 300 ms | 0/7 | 0/6 | 0/7 | 0/20 |
| silence 416 ms | 600 ms | 6/7 | 6/6 | 7/7 | 19/20 |
| silence 416 ms | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| SmartTurn x1 th=0.5 | 300 ms | 1/7 | 3/6 | 5/7 | 9/20 |
| SmartTurn x1 th=0.5 | 600 ms | 6/7 | 5/6 | 6/7 | 17/20 |
| SmartTurn x1 th=0.5 | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| SmartTurn x1 th=0.8 | 300 ms | 1/7 | 3/6 | 3/7 | 7/20 |
| SmartTurn x1 th=0.8 | 600 ms | 6/7 | 3/6 | 3/7 | 12/20 |
| SmartTurn x1 th=0.8 | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| SmartTurn x1 th=0.9 | 300 ms | 1/7 | 3/6 | 2/7 | 6/20 |
| SmartTurn x1 th=0.9 | 600 ms | 6/7 | 3/6 | 3/7 | 12/20 |
| SmartTurn x1 th=0.9 | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| SmartTurn x5 th=0.5 | 300 ms | 1/7 | 3/6 | 5/7 | 9/20 |
| SmartTurn x5 th=0.5 | 600 ms | 6/7 | 5/6 | 6/7 | 17/20 |
| SmartTurn x5 th=0.5 | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| SmartTurn x5 th=0.8 | 300 ms | 1/7 | 3/6 | 3/7 | 7/20 |
| SmartTurn x5 th=0.8 | 600 ms | 6/7 | 3/6 | 3/7 | 12/20 |
| SmartTurn x5 th=0.8 | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |

### [cut] false cut-offs (turn ended during the mid-thought pause) / n
| policy | pause | en | hi | hinglish | all |
|---|---|---|---|---|---|
| silence 700 ms (current) | 300 ms | 0/7 | 0/6 | 0/7 | 0/20 |
| silence 700 ms (current) | 600 ms | 4/7 | 2/6 | 3/7 | 9/20 |
| silence 700 ms (current) | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| silence 512 ms | 300 ms | 1/7 | 0/6 | 0/7 | 1/20 |
| silence 512 ms | 600 ms | 5/7 | 3/6 | 7/7 | 15/20 |
| silence 512 ms | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| silence 416 ms | 300 ms | 4/7 | 2/6 | 3/7 | 9/20 |
| silence 416 ms | 600 ms | 6/7 | 6/6 | 7/7 | 19/20 |
| silence 416 ms | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| SmartTurn x1 th=0.5 | 300 ms | 4/7 | 4/6 | 4/7 | 12/20 |
| SmartTurn x1 th=0.5 | 600 ms | 4/7 | 4/6 | 5/7 | 13/20 |
| SmartTurn x1 th=0.5 | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| SmartTurn x1 th=0.8 | 300 ms | 2/7 | 3/6 | 3/7 | 8/20 |
| SmartTurn x1 th=0.8 | 600 ms | 4/7 | 2/6 | 4/7 | 10/20 |
| SmartTurn x1 th=0.8 | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| SmartTurn x1 th=0.9 | 300 ms | 2/7 | 3/6 | 2/7 | 7/20 |
| SmartTurn x1 th=0.9 | 600 ms | 4/7 | 2/6 | 3/7 | 9/20 |
| SmartTurn x1 th=0.9 | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| SmartTurn x5 th=0.5 | 300 ms | 4/7 | 4/6 | 4/7 | 12/20 |
| SmartTurn x5 th=0.5 | 600 ms | 4/7 | 4/6 | 5/7 | 13/20 |
| SmartTurn x5 th=0.5 | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |
| SmartTurn x5 th=0.8 | 300 ms | 4/7 | 3/6 | 3/7 | 10/20 |
| SmartTurn x5 th=0.8 | 600 ms | 4/7 | 3/6 | 4/7 | 11/20 |
| SmartTurn x5 th=0.8 | 900 ms | 7/7 | 6/6 | 7/7 | 20/20 |

### end-of-turn delay after true end of speech (ms): mean [p50 / max]
| policy | set | en | hi | hinglish | all |
|---|---|---|---|---|---|
| silence 700 ms (current) | complete | 900 [922/970] | 792 [796/808] | 792 [788/826] | 830 [800/970] |
| silence 700 ms (current) | end of B, tts | 887 [892/944] | 798 [797/846] | 785 [785/824] | 826 [812/944] |
| silence 700 ms (current) | end of B, cut | 868 [857/944] | 804 [805/848] | 793 [796/812] | 821 [806/944] |
| silence 512 ms | complete | 708 [730/778] | 600 [604/616] | 600 [596/634] | 638 [608/778] |
| silence 512 ms | end of B, tts | 689 [690/744] | 603 [602/634] | 591 [587/632] | 637 [626/744] |
| silence 512 ms | end of B, cut | 679 [665/752] | 614 [614/656] | 596 [600/612] | 631 [615/752] |
| silence 416 ms | complete | 612 [634/682] | 504 [508/520] | 504 [500/538] | 542 [512/682] |
| silence 416 ms | end of B, tts | 605 [615/648] | 506 [505/538] | 493 [484/536] | 540 [518/648] |
| silence 416 ms | end of B, cut | 601 [601/656] | 516 [514/560] | 502 [501/516] | 539 [524/656] |
| SmartTurn x1 th=0.5 | complete | 409 [437/482] | 312 [316/328] | 312 [308/346] | 342 [318/482] |
| SmartTurn x1 th=0.5 | end of B, tts | 540 [446/846] | 308 [307/332] | 319 [326/326] | 426 [351/846] |
| SmartTurn x1 th=0.5 | end of B, cut | 390 [368/464] | 313 [315/340] | 304 [306/326] | 341 [328/464] |
| SmartTurn x1 th=0.8 | complete | 409 [437/482] | 472 [322/802] | 380 [318/788] | 418 [334/802] |
| SmartTurn x1 th=0.8 | end of B, tts | 540 [446/846] | 305 [307/332] | 306 [309/326] | 384 [322/846] |
| SmartTurn x1 th=0.8 | end of B, cut | 383 [368/464] | 388 [324/836] | 309 [312/332] | 361 [330/836] |
| SmartTurn x1 th=0.9 | complete | 489 [444/922] | 552 [545/802] | 380 [318/788] | 470 [346/922] |
| SmartTurn x1 th=0.9 | end of B, tts | 540 [446/846] | 305 [307/332] | 364 [312/824] | 404 [324/846] |
| SmartTurn x1 th=0.9 | end of B, cut | 443 [368/944] | 388 [324/836] | 364 [312/812] | 397 [327/944] |
| SmartTurn x5 th=0.5 | complete | 409 [437/482] | 312 [316/328] | 312 [308/346] | 342 [318/482] |
| SmartTurn x5 th=0.5 | end of B, tts | 540 [446/846] | 308 [307/332] | 319 [326/326] | 426 [351/846] |
| SmartTurn x5 th=0.5 | end of B, cut | 390 [368/464] | 313 [315/340] | 304 [306/326] | 341 [328/464] |
| SmartTurn x5 th=0.8 | complete | 409 [437/482] | 408 [322/802] | 380 [318/788] | 398 [334/802] |
| SmartTurn x5 th=0.8 | end of B, tts | 540 [446/846] | 305 [307/332] | 306 [309/326] | 384 [322/846] |
| SmartTurn x5 th=0.8 | end of B, cut | 390 [368/464] | 313 [314/340] | 309 [312/332] | 336 [326/464] |

### Smart Turn P(complete) at the 224 ms checkpoint
| point | lang | n | mean P | min | share >= 0.5 | share >= 0.8 |
|---|---|---|---|---|---|---|
| true end (complete) | en | 7 | 0.96 | 0.90 | 100% | 100% |
| true end (complete) | hi | 6 | 0.84 | 0.59 | 100% | 67% |
| true end (complete) | hinglish | 7 | 0.95 | 0.78 | 100% | 86% |
| true end (complete) | all | 20 | 0.92 | 0.59 | 100% | 85% |
| true end after pause [tts] | en | 21 | 0.84 | 0.01 | 86% | 86% |
| true end after pause [tts] | hi | 18 | 0.76 | 0.02 | 78% | 72% |
| true end after pause [tts] | hinglish | 21 | 0.95 | 0.69 | 100% | 95% |
| true end after pause [tts] | all | 60 | 0.85 | 0.01 | 88% | 85% |
| true end after pause [cut] | en | 21 | 0.96 | 0.85 | 100% | 100% |
| true end after pause [cut] | hi | 18 | 0.80 | 0.02 | 83% | 78% |
| true end after pause [cut] | hinglish | 21 | 0.95 | 0.81 | 100% | 100% |
| true end after pause [cut] | all | 60 | 0.91 | 0.02 | 95% | 93% |
| mid pause 300 [tts] | en | 1 | 0.93 | 0.93 | 100% | 100% |
| mid pause 300 [tts] | hi | 4 | 0.84 | 0.47 | 75% | 75% |
| mid pause 300 [tts] | hinglish | 6 | 0.69 | 0.14 | 83% | 50% |
| mid pause 300 [tts] | all | 11 | 0.76 | 0.14 | 82% | 64% |
| mid pause 600 [tts] | en | 7 | 0.83 | 0.01 | 86% | 86% |
| mid pause 600 [tts] | hi | 6 | 0.75 | 0.43 | 83% | 50% |
| mid pause 600 [tts] | hinglish | 7 | 0.74 | 0.36 | 86% | 43% |
| mid pause 600 [tts] | all | 20 | 0.77 | 0.01 | 85% | 60% |
| mid pause 900 [tts] | en | 7 | 0.83 | 0.01 | 86% | 86% |
| mid pause 900 [tts] | hi | 6 | 0.71 | 0.15 | 83% | 50% |
| mid pause 900 [tts] | hinglish | 7 | 0.74 | 0.32 | 86% | 43% |
| mid pause 900 [tts] | all | 20 | 0.76 | 0.01 | 85% | 60% |
| mid pause 300 [cut] | en | 4 | 0.82 | 0.63 | 100% | 50% |
| mid pause 300 [cut] | hi | 3 | 0.65 | 0.04 | 67% | 67% |
| mid pause 300 [cut] | hinglish | 6 | 0.51 | 0.02 | 50% | 50% |
| mid pause 300 [cut] | all | 13 | 0.64 | 0.02 | 69% | 54% |
| mid pause 600 [cut] | en | 7 | 0.55 | 0.06 | 57% | 43% |
| mid pause 600 [cut] | hi | 6 | 0.51 | 0.05 | 50% | 33% |
| mid pause 600 [cut] | hinglish | 7 | 0.46 | 0.02 | 43% | 43% |
| mid pause 600 [cut] | all | 20 | 0.51 | 0.02 | 50% | 40% |
| mid pause 900 [cut] | en | 7 | 0.61 | 0.13 | 57% | 57% |
| mid pause 900 [cut] | hi | 6 | 0.57 | 0.05 | 50% | 33% |
| mid pause 900 [cut] | hinglish | 7 | 0.45 | 0.02 | 43% | 43% |
| mid pause 900 [cut] | all | 20 | 0.54 | 0.02 | 50% | 45% |

Smart Turn inference inside eval (features+ORT, concurrent load): n=1112 p50=13 ms p95=16 ms
incidental silent stretches inside speech that reached a checkpoint: 9
