# Indian-language benchmark sentences

Original sentences for the BonClay voice benchmark (`backend/bench/indic_benchmark.py`, results in
`docs/BENCHMARK_INDIC.md`). One JSON file per language: `hi` Hindi, `hinglish`, `mr` Marathi, `ta` Tamil, `bn` Bengali,
`te` Telugu, `kn` Kannada, `gu` Gujarati, `ml` Malayalam, `pa` Punjabi.

| field | meaning |
| --- | --- |
| `language` | file's language code |
| `id` | `<language>-NN`, unique |
| `text` | the sentence (for `hinglish`: Roman script as a caller would type it) |
| `number_words` | the number words in the sentence, in order. ASR writes numbers as digits; the scorer replaces digit runs with these before computing CER |
| `kind` | greeting, date, fee, question, instruction, long, loanwords, mixed, plus targeted sets: `nukta` (Hindi), `la` and `rha` (Marathi ळ, ऱ्ह), `zha` (Tamil ழ) |
| `spoken` | optional; for `hinglish` the Devanagari reading an engine speaks and the scorer compares against (written by hand, not by a model) |

## Licence

The sentences are original to this repository and released under **CC-BY-4.0**: use, copy and adapt them with credit
to "BonClay benchmark sentences". The repository's code licence does not change. No sentence is copied from a corpus,
a model's output, or a real person's speech.

## Quality note

Drafted by an AI assistant for this repo (about half the Hindi, Marathi, Tamil, Bengali and Telugu lines come from
earlier project trials) and **not reviewed by a native speaker in any language yet**. Expect the odd unnatural phrase.
Corrections by pull request are welcome and will be credited. Dates and amounts are written as words on purpose.
