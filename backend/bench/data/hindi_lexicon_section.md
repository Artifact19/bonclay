## Hindi Kokoro lexicon

Kokoro's Hindi voice `hf_alpha` reads from a phoneme front end (espeak), so a word it mispronounces can only be fixed by writing it differently. `backend/services/pron_lexicon.py` has a rule set for `("hi", "kokoro")` with three rules. It is **off** (`PRON_LEXICON=on` enables it); the numbers below are a read-back through Whisper, so they show what an ASR hears, not what a listener prefers. Listen to the before/after pairs first (paths at the end).

**What was tried.** The read-back bench (`readback.py`, engine `kokoro-hi`, 10 sentences, ASR `mlx-community/whisper-large-v3-turbo`) scored CER 0.083. Its misread words were मुंबई (heard मूबी), छह (च्छ), ट्रेन (त्रेन), झील (जील), शांत (शाद), दोपहर (दोभर), पेड़ (पेर), पुणे (पुने) and सात (साथ). `bench/hindi_lexicon_check.py` takes each word, writes 3 short carrier sentences, speaks the plain spelling and 3 to 5 candidate spellings (chosen from what espeak turns each spelling into), transcribes each clip with the API's own Whisper and scores it against the sentence.

**Scoring and the keep rule.** Whisper spells what it hears, sometimes differently from the reference for the same sound (शान्त for शांत, मुम्बई for मुंबई), so the check unifies ँ and nasal + virama into ं on both sides first, and counts a few extra spellings of a correctly read word (ACCEPT in the script: मुमबई, मुमबाई, मुम्बाई, मुमबी for मुंबई; छः, छे for छह). The whole-sentence CER on 3 sentences is dominated by the other words, which the rule does not touch and which move with prosody and ASR noise (the same carrier came back "वेह" or "वेहे" depending only on how मुंबई was spelled), so the keep rule is applied to the edits that fall on the target word. A candidate is kept when, against the plain spelling, (1) the target word has no more errors on any carrier sentence and fewer on at least one, (2) the word is transcribed right in at least 2 of 3 sentences and no fewer than the plain spelling got, and (3) mean sentence CER does not rise by more than 0.02. Among kept spellings of a word, the one with the lowest mean CER on the transcript as Whisper wrote it ships. The literal rule "whole-sentence CER lower on every carrier sentence" is printed too (`strict`); only मुंबई passes it, because for the other words some sentence already had CER 0 or moved on unrelated words.

**Results (3 carrier sentences per word, the API's own ASR, rendered by the server's Kokoro).** CER s1..s3 is the whole sentence; "edits" are the edits on the target word per sentence; "right" is how many of 3 transcripts hold the word.

| word | spelling | CER s1 | CER s2 | CER s3 | mean CER | edits on the word | word right | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| मुंबई | baseline | 0.200 | 0.136 | 0.227 | 0.188 | 3/3/3 | 0/3 | baseline |
| मुंबई | मुम्बई | 0.050 | 0.000 | 0.136 | 0.062 | 0/0/1 | 2/3 | **ship** |
| मुंबई | मुमबई | 0.200 | 0.045 | 0.136 | 0.127 | 3/1/1 | 0/3 | dropped (right 0/3) |
| मुंबई | मुम्बइ | 0.250 | 0.000 | 0.136 | 0.129 | 1/0/1 | 1/3 | dropped (right 1/3) |
| मुंबई | मुम्बाई | 0.050 | 0.000 | 0.091 | 0.047 | 0/0/0 | 3/3 | kept, not shipped (only right through ACCEPT: Whisper writes मुंबाई) |
| मुंबई | मुमबाई | 0.100 | 0.000 | 0.091 | 0.064 | 0/0/0 | 3/3 | kept, not shipped (same) |
| छह | baseline | 0.056 | 0.118 | 0.357 | 0.177 | 0/1/1 | 1/3 | baseline |
| छह | छे | 0.056 | 0.000 | 0.000 | 0.019 | 0/0/0 | 3/3 | **ship** (छह is spoken "chhe") |
| छह | छै | 0.056 | 0.000 | 0.286 | 0.114 | 0/0/1 | 2/3 | kept, not shipped |
| छह | छेह | 0.056 | 0.118 | 1.286 | 0.486 | 0/1/1 | 1/3 | dropped (no fewer errors anywhere) |
| ट्रेन | baseline | 0.087 | 0.000 | 0.105 | 0.064 | 1/0/0 | 2/3 | baseline |
| ट्रेन | टरेन | 0.130 | 0.050 | 0.053 | 0.078 | 1/1/1 | 0/3 | dropped (more errors, sentence 2) |
| ट्रेन | टेरेन | 0.130 | 0.100 | 0.053 | 0.094 | 2/2/1 | 0/3 | dropped (more errors, sentence 1) |
| ट्रेन | टिरेन | 0.130 | 0.050 | 0.053 | 0.078 | 2/1/1 | 0/3 | dropped (more errors, sentence 1) |
| झील | baseline | 0.136 | 0.111 | 0.045 | 0.098 | 1/1/1 | 0/3 | baseline |
| झील | झ्हील | 0.136 | 0.111 | 0.045 | 0.098 | 1/1/1 | 0/3 | dropped (no change) |
| झील | झहील | 0.182 | 0.167 | 0.091 | 0.146 | 1/1/1 | 0/3 | dropped (no change on the word, CER up) |
| झील | झीईल | 0.136 | 0.111 | 0.045 | 0.098 | 1/1/1 | 0/3 | dropped (no change) |
| शांत | baseline | 0.000 | 0.077 | 0.125 | 0.067 | 0/0/2 | 2/3 | baseline |
| शांत | शान्त | 0.000 | 0.077 | 0.042 | 0.040 | 0/0/0 | 3/3 | **ship** |
| शांत | शाँत | 0.000 | 0.077 | 0.083 | 0.053 | 0/0/1 | 2/3 | kept, not shipped |
| शांत | शानत | 0.143 | 0.154 | 0.083 | 0.127 | 2/1/1 | 0/3 | dropped (more errors, sentence 1) |
| दोपहर | baseline | 0.000 | 0.056 | 0.000 | 0.019 | 0/0/0 | 3/3 | baseline (already right) |
| दोपहर | दो पहर | 0.000 | 0.056 | 0.000 | 0.019 | 0/0/0 | 3/3 | dropped (no gain) |
| दोपहर | दो-पहर | 0.000 | 0.056 | 0.000 | 0.019 | 0/0/0 | 3/3 | dropped (no gain) |
| दोपहर | दुपहर | 0.053 | 0.111 | 0.050 | 0.071 | 1/1/1 | 0/3 | dropped (more errors, sentence 1) |
| दोपहर | दोपेहर | 0.053 | 0.111 | 0.050 | 0.071 | 1/1/1 | 0/3 | dropped (more errors, sentence 1) |
| दोपहर | दोपहार | 0.105 | 0.167 | 0.100 | 0.124 | 2/2/2 | 0/3 | dropped (more errors, sentence 1) |
| पेड़ | baseline | 0.053 | 0.100 | 0.042 | 0.065 | 1/1/1 | 0/3 | baseline |
| पेड़ | पेड | 0.158 | 0.100 | 0.042 | 0.100 | 3/1/1 | 0/3 | dropped (more errors, sentence 1) |
| पेड़ | पेढ | 0.158 | 0.100 | 0.042 | 0.100 | 3/1/1 | 0/3 | dropped (more errors, sentence 1) |
| पेड़ | पेड् | 0.158 | 0.100 | 0.042 | 0.100 | 3/1/1 | 0/3 | dropped (more errors, sentence 1) |
| पुणे | baseline | 0.100 | 0.143 | 0.118 | 0.120 | 2/1/2 | 0/3 | baseline |
| पुणे | पूणे | 0.050 | 0.191 | 0.118 | 0.119 | 1/2/2 | 0/3 | dropped (more errors, sentence 2) |
| पुणे | पुंणे | 0.050 | 0.191 | 0.118 | 0.119 | 1/2/2 | 0/3 | dropped (more errors, sentence 2) |
| पुणे | पुणें | 0.100 | 0.143 | 0.118 | 0.120 | 2/1/2 | 0/3 | dropped (no change) |
| सात | baseline | 0.235 | 0.071 | 0.071 | 0.126 | 1/1/1 | 0/3 | baseline |
| सात | साअत | 0.235 | 0.071 | 0.071 | 0.126 | 1/1/1 | 0/3 | dropped (no change) |
| सात | सात् | 0.235 | 0.071 | 0.071 | 0.126 | 1/1/1 | 0/3 | dropped (no change) |
| सात | सातः | 0.176 | 0.071 | 0.071 | 0.106 | 0/0/0 | 0/3 | dropped (right 0/3: Whisper writes सातह) |

Reading: three words have a spelling the read-back proves. For ट्रेन, झील, पेड़, पुणे and सात no spelling helped. The likely reason for ट्रेन, झील, पेड़ and पुणे is a sound the voice does not render (retroflex ट, ड़ and ण, aspirated झ), and a respelling does not add it; that is a guess from the transcripts, not something measured. Whisper wrote पुने for पुणे and साथ for सात in every transcript, whichever spelling was spoken, which suggests it prefers those spellings, so these two may not be provable with this recogniser; none of this was checked by ear. दोपहर, which the first bench scored badly, was read right in all 3 plain carrier sentences in this run, so it gets no rule.

**Held-out check.** The spellings were chosen on the 3 sentences above, so they were checked on 3 other sentences per word (`hindi_lexicon_check.py --holdout`, shipped rules against the plain spelling):

| word | CER s1 | CER s2 | CER s3 | mean CER | edits on the word | word right |
| --- | --- | --- | --- | --- | --- | --- |
| मुंबई plain | 0.080 | 0.120 | 0.240 | 0.147 | 2/3/3 | 0/3 |
| मुंबई -> मुम्बई | 0.000 | 0.000 | 0.120 | 0.040 | 0/0/0 | 3/3 |
| छह plain | 0.095 | 0.158 | 0.278 | 0.177 | 1/1/1 | 0/3 |
| छह -> छे | 0.048 | 0.105 | 0.000 | 0.051 | 0/0/0 | 3/3 |
| शांत plain | 0.182 | 0.091 | 0.091 | 0.121 | 2/0/1 | 1/3 |
| शांत -> शान्त | 0.045 | 0.091 | 0.000 | 0.045 | 0/0/0 | 3/3 |

**Depends on the ASR.** A first pass was run while the server was using the CPU `openai-whisper small` fallback instead of the Whisper large-v3-turbo the bench normally uses (the worktree had no MLX environment). It kept different spellings (मुमबई, शान्त, शाँत, दो पहर) and dropped मुम्बई and छे, and its plain-spelling CERs were mostly 2 to 3 times higher. The table above is the second pass, with `MLX_PYTHON` pointing at the MLX environment, which reproduces the 0.083 baseline exactly. A rule proven with one recogniser can fail with another.

**Full read-back, 10 sentences, `PRON_LEXICON` off vs on** (`readback.py --engines kokoro-hi -n 10`, server restarted between the settings; three runs each). Only sentence 2 (पुणे से मुंबई जाने वाली ट्रेन सुबह छह बजे निकलती है) contains a rule word with a changed result.

| sentence | off run 1 | off run 2 | off run 3 | on run 1 | on run 2 | on run 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.020 | 0.020 | 0.020 | 0.020 | 0.020 | 0.020 |
| 2 (मुंबई, छह) | 0.195 | 0.195 | 0.195 | 0.098 | 0.098 | 0.098 |
| 3 | 0.133 | 0.133 | 0.133 | 0.133 | 0.133 | 0.133 |
| 4 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 |
| 5 | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 |
| 6 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 |
| 7 (शांत) | 0.071 | 0.071 | 0.071 | 0.071 | 0.071 | 0.071 |
| 8 | 0.146 | 0.042 | 0.146 | 0.146 | 0.146 | 0.146 |
| 9 | 0.060 | 0.060 | 0.060 | 0.060 | 0.060 | 0.060 |
| 10 | 0.065 | 0.065 | 0.065 | 0.065 | 0.065 | 0.065 |
| **mean CER** | **0.083** | **0.073** | **0.083** | **0.074** | **0.074** | **0.074** |

Reading: the rules fix sentence 2 (मुंबई now comes back as मुंबई, छह as छे) and move no sentence the wrong way, so the mean falls from 0.083 to 0.074. Two cautions. Sentence 8 has no rule word and its audio has the same length in every run (5.375 s); its CER swings between 0.146 and 0.042 because Whisper in some runs writes the loan word "ऑर्डर" in Latin letters ("order", scored as missing) and the number as a digit, and in one run writes both in Devanagari. That swing alone moves the mean by 0.010, so the second off run (0.073) is as low as the on runs; the reliable evidence is sentence 2, not the mean. Sentence 7 (शांत): the plain audio came back शाद and the hinted audio शान, with the same CER, so the rule did not help that sentence; it did on the held-out sentences above.

**Not proven.** Nobody has listened. The read-back shows the transcript is closer, not that a Hindi speaker prefers the audio. Selecting the best of 3 to 5 candidates on 3 sentences is optimistic; the held-out table is the check. The rules replace a word only as a whole word (मुंबईकर, छहों and शांति are untouched), and only for `("hi", "kokoro")`: Marathi or other text spoken by Kokoro's Hindi voice, DhVaani and Indic-Speak are not covered. Recommendation: leave `PRON_LEXICON` off until a Hindi speaker has rated the three before/after pairs; with it on, the gain on the 10-sentence bench is one sentence.

**Reproduce.**

```bash
# server on 8003 from backend/, PRON_LEXICON off for the candidate search
# (MLX_PYTHON=<repo>/backend/.venv-mlx/bin/python if the checkout has no .venv-mlx)
PHONE_PUBLIC_PORT=0 WARM_ON_START=off backend/.venv/bin/python -m uvicorn backend:app --host 127.0.0.1 --port 8003
backend/.venv/bin/python backend/bench/hindi_lexicon_check.py --api http://127.0.0.1:8003             # candidates; writes wav/, table.md, best/ (before/after pairs)
backend/.venv/bin/python backend/bench/hindi_lexicon_check.py --api http://127.0.0.1:8003 --holdout   # the shipped rules on other sentences
# restart the server with PRON_LEXICON=on, then with it off, and each time:
backend/.venv/bin/python backend/bench/readback.py --api http://127.0.0.1:8003 --engines kokoro-hi -n 10
```

The server must see the MLX environment (`backend/.venv-mlx`, or `MLX_PYTHON`) for the bench's Whisper to be the large-v3-turbo model. Listening files (before and after, same sentence) are written to `bench/results/hindi_lexicon-<stamp>/best/`.
