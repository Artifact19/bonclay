# Indic voice experiments (2026-10-20, Apple M4 16 GB, DhVaani on the Apple GPU)

Method as in `docs/BENCHMARK_INDIC.md`: synthetic Kokoro references, Whisper large-v3-turbo read-back, CER from `backend/bench/scoring.py`. Raw files are outside the repo (shared notes folder). Scripts: `backend/bench/exp_digits_vs_words.py`, `exp_reference_language.py`, `exp_steps.py`. None of the audio has been listened to.

## A. Digits versus number words (25 sentences with numbers)

| Input | Mean CER | Number spoken correctly |
| --- | ---: | ---: |
| Number words | 0.109 | 19 of 25 (misses are Whisper spelling variants) |
| ASCII digits (450) | 0.213 | 0 of 25 |
| Native digits (४५०), hi and mr | 0.381 | 0 of 4 |

DhVaani cannot read digits: "फीस 450 रुपये" was heard as "फीस जह रुपे है". The app has no number-to-words step. A deterministic normalizer (digits, amounts, dates, times, phone numbers) must run before DhVaani. Limits: 25 sentences, 5 distinct numbers, one run each. Sentence CER understates the problem because the number is a few characters.

## B. Reference language for Malayalam and Punjabi (7 shared sentences per language)

| Language | Hindi reference | English reference | Same-language bootstrapped reference |
| --- | ---: | ---: | ---: |
| Malayalam | 0.866 | 0.878 | 0.927 |
| Punjabi | 0.301 | 0.753 | 0.382 |
| Tamil (control) | 0.039 | 0.074 | 0.042 |
| Kannada (control) | 0.036 | 0.273 | 0.037 |

Hypothesis not supported: a different reference does not fix Malayalam or Punjabi, and an English reference is clearly worse (and makes DhVaani speak about 30% faster). Malayalam failures follow the sentence (sentences 5 to 8 fail under every reference) and the failed transcripts are in the wrong script and loop, which is typical of Whisper failing; but even good Malayalam sentences lose conjuncts, which could also be the engine. Cannot be separated without listening or a second recognizer. Run-to-run spread on Malayalam is about 0.1; the bootstrapped reference copies the engine's own flaws.

## C. Sampling steps (Hindi, Marathi, Tamil; 28 sentences; quiet machine)

| Steps | Mean CER | Seconds per sentence | Speed (s of work per s of audio) |
| ---: | ---: | ---: | ---: |
| 6 | 0.086 | 2.2 | 0.44 |
| 8 | 0.086 | 2.9 | 0.57 |
| 12 | 0.085 | 4.2 | 0.83 |
| 16 | 0.087 | 5.6 | 1.10 |

Read-back accuracy is flat from 6 to 16 steps while speed improves 2.5x: 6 steps runs faster than real time. Per language the CER is also flat (hi 0.093 to 0.094, mr 0.112 to 0.116, ta 0.035 to 0.037). Limits: CER measures intelligibility only, not naturalness or speaker similarity, which fewer steps may hurt; one run per cell; speed here (1.1 at 16 steps) is faster than the 1.5 in the main benchmark because the machine was otherwise idle. Next: listen to 6-step against 16-step files, measure speaker similarity (ECAPA), try 4 steps, and only then change `DHVAANI_STEPS`.
