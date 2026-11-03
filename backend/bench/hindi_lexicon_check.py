#!/usr/bin/env python3
"""Find spellings that Kokoro's Hindi voice (hf_alpha) reads better, by read-back (services/pron_lexicon.py).

For each target word (one Kokoro mispronounces, see docs/BENCHMARK_INDIC.md, "Hindi Kokoro lexicon") there are 3 short
carrier sentences. Each sentence is spoken with the baseline spelling and with every candidate spelling through the
running API (POST /api/kokoro/generate), transcribed by the API's own Whisper (POST /api/transcribe) and scored against
the sentence as written. The script reports, per candidate, the CER on each carrier sentence, the edits that fall on the
target word, and how many of the 3 transcripts hold the target word right.

Scoring. Whisper spells what it hears, sometimes differently from the reference for the same sound (शान्त for शांत,
मुम्बई for मुंबई), so both sides are canonicalised first: ँ -> ं, and a nasal consonant + virama before a consonant ->
ं. A few more spellings Whisper uses for a word that was read right are listed in ACCEPT. Two numbers per sentence:
`cer` over the whole sentence (readback.score, canonicalised), and `word_errors`, the edits of one minimum-cost
alignment that land on the target word's characters. The rule changes only that word; the other words move with prosody
and ASR noise (in the first run, the same carrier came back "वेह" or "वेहे" depending only on how मुंबई was spelled).

A candidate is KEPT only when, against the baseline spelling:
  * word_errors is not higher on any carrier sentence and is lower on at least one, and
  * the target word is transcribed correctly in at least 2 of 3 sentences, and in no fewer than the baseline got right, and
  * the mean whole-sentence CER does not rise by more than 0.02.
`strict` also reports whether the literal rule "whole-sentence CER lower on every sentence that had errors" holds.

Start the API with PRON_LEXICON off (the script sends the already-rewritten text; a lexicon in the server would rewrite
it again). Whisper output is cached per spoken text in <out>/cache.json, so a rerun with the same --out costs nothing.

    backend/.venv/bin/python backend/bench/hindi_lexicon_check.py --api http://127.0.0.1:8003 [--only ट्रेन,छह]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
sys.path.insert(0, HERE)

import jiwer  # noqa: E402

from common import LIMITS, RESULTS_DIR, Api, ApiError, RateGate, err_text, now_stamp, require_api  # noqa: E402
from readback import norm_hi  # noqa: E402
from services.pron_lexicon import RULES, apply_rules, rule, whole_word  # noqa: E402

VOICE = "hf_alpha"

# word -> (carrier sentences, [candidate spellings]). Candidates avoid the nukta (espeak reads a bare U+093C aloud as its
# Unicode name) and were picked from what misaki's espeak front end turns each spelling into.
TARGETS = {
    "मुंबई": (["वह कल सुबह मुंबई जा रहा है।", "मुंबई में आज बारिश हो रही है।", "हमारा दफ्तर मुंबई के पास है।"],
              ["मुम्बई", "मुमबई", "मुम्बइ", "मुम्बाई", "मुमबाई"]),
    "छह": (["मुझे छह किताबें चाहिए।", "बैठक छह बजे शुरू होगी।", "उनके छह बच्चे हैं।"], ["छे", "छै", "छेह"]),
    "ट्रेन": (["ट्रेन समय पर स्टेशन पहुंच गई।", "हम रात की ट्रेन से जाएंगे।", "यह ट्रेन दिल्ली जाती है।"],
              ["टरेन", "टेरेन", "टिरेन"]),
    "झील": (["हम शाम को झील के किनारे घूमे।", "झील का पानी बहुत साफ है।", "बच्चे झील में नाव चला रहे थे।"],
            ["झ्हील", "झहील", "झीईल"]),
    "शांत": (["कमरा बहुत शांत है।", "कृपया शांत रहिए।", "गांव की सुबह बहुत शांत होती है।"], ["शान्त", "शाँत", "शानत"]),
    "दोपहर": (["दोपहर को हम खाना खाएंगे।", "आज दोपहर बहुत गर्मी थी।", "वे दोपहर तक यहां आ जाएंगे।"],
              ["दो पहर", "दो-पहर", "दुपहर", "दोपेहर", "दोपहार"]),
    "पेड़": (["आंगन में एक बड़ा पेड़ है।", "पेड़ की छाया बहुत ठंडी है।", "बच्चे पेड़ के नीचे खेल रहे हैं।"],
            ["पेड", "पेढ", "पेड्"]),
    "पुणे": (["मेरा भाई पुणे में रहता है।", "हम अगले हफ्ते पुणे जाएंगे।", "पुणे का मौसम अच्छा है।"],
             ["पूणे", "पुंणे", "पुणें"]),
    "सात": (["मेरे सात दोस्त आए हैं।", "बैठक सात बजे होगी।", "उसने सात केले खाए।"], ["साअत", "सात्", "सातः"]),
}

# Sentences not used to choose the spellings: --holdout speaks them with the shipped rules (pron_lexicon.RULES) against the
# plain spelling, the same scoring and the same keep rule.
HOLDOUT = {
    "मुंबई": ["मुंबई की बारिश बहुत तेज होती है।", "उसकी बहन अगले महीने मुंबई आएगी।", "मुंबई से गोवा की उड़ान शाम को है।"],
    "छह": ["छह दिन बाद मेरी परीक्षा है।", "मैंने छह रोटियां बनाईं।", "टोकरी में छह आम रखे हैं।"],
    "शांत": ["यह जगह बहुत शांत और सुंदर है।", "बच्चा सोते समय शांत रहता है।", "सभी लोग शांत होकर सुन रहे थे।"],
}

# Other spellings Whisper writes for a word that was read right (same sound, different orthography).
ACCEPT = {"मुंबई": ["मुमबई", "मुमबाई", "मुम्बाई", "मुमबी"], "छह": ["छः", "छे"]}

_NASAL_CONJ = "[ङञणनम]्(?=[क-ह])"


def canon(s: str) -> str:
    """Devanagari letters only (readback.norm_hi), anusvara spellings unified (see the module doc)."""
    s = unicodedata.normalize("NFC", norm_hi(s)).replace("ँ", "ं")
    return re.sub(_NASAL_CONJ, "ं", s)


def sentence_cer(ref: str, hyp: str, target: str) -> float:
    """Whole-sentence CER (jiwer, as readback.py) on canonicalised letters; ACCEPT spellings of the target count as it."""
    alts = {canon(a) for a in ACCEPT.get(target, [])}
    toks = [(target if canon(t) in alts else t) for t in (hyp or "").split()]
    r, h = canon(ref), canon(" ".join(toks))
    return 1.0 if not h else float(jiwer.cer(r, h))


def plain_cer(ref: str, hyp: str) -> float:
    """Whole-sentence CER on canonicalised letters without the ACCEPT leniency (the transcript spelled as it came)."""
    r, h = canon(ref), canon(hyp)
    return 1.0 if not h else float(jiwer.cer(r, h))


def word_errors(ref: str, hyp: str, target: str) -> int:
    """Edits (substitution, deletion, insertion) of one minimum-cost alignment that land on the target word's characters
    in the reference. An extra transcript character counts when it sits strictly inside the word."""
    alts = {canon(a) for a in ACCEPT.get(target, [])}
    toks = [(target if canon(t) in alts else t) for t in (hyp or "").split()]
    r, h, w = canon(ref), canon(" ".join(toks)), canon(target)
    start = r.find(w)
    if start < 0:
        raise ValueError(f"{target!r} is not in {ref!r}")
    end = start + len(w)
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    i, j, errs = n, m, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            errs += int(r[i - 1] != h[j - 1] and start <= i - 1 < end)
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:  # a reference character is missing from the transcript
            errs += int(start <= i - 1 < end)
            i -= 1
        else:  # an extra transcript character
            errs += int(start < i < end)
            j -= 1
    return errs


def tokens_hold(hyp: str, target: str) -> bool:
    """True when the transcript has the target word: one token equal to it (or to an ACCEPT spelling), or up to three
    adjacent tokens that join into it (Whisper may split दोपहर into दो पहर). Compared canonicalised."""
    want = canon(target)
    alts = {canon(a) for a in ACCEPT.get(target, [])}
    toks = [t for t in (canon(w) for w in (hyp or "").split()) if t]
    for i in range(len(toks)):
        joined = ""
        for j in range(i, min(i + 3, len(toks))):
            joined += toks[j]
            if joined == want or joined in alts:
                return True
    return False


def decide(base_rows, cand_rows):
    """(kept, strict, reason) for one candidate against the baseline; rows are the 3 carrier results in order."""
    b_ok = sum(r["target_ok"] for r in base_rows)
    c_ok = sum(r["target_ok"] for r in cand_rows)
    pairs = list(zip(base_rows, cand_rows))
    strict = (any(b["cer"] > 0 for b, _ in pairs)
              and all(c["cer"] < b["cer"] - 1e-9 for b, c in pairs if b["cer"] > 0))
    worse = [i for i, (b, c) in enumerate(pairs) if c["word_errors"] > b["word_errors"]]
    better = [i for i, (b, c) in enumerate(pairs) if c["word_errors"] < b["word_errors"]]
    mean_b = sum(b["cer"] for b in base_rows) / len(base_rows)
    mean_c = sum(c["cer"] for c in cand_rows) / len(cand_rows)
    if worse:
        return False, strict, f"word errors higher on sentence {worse[0] + 1}"
    if not better:
        return False, strict, "word errors lower on no sentence"
    if c_ok < 2:
        return False, strict, f"word right in {c_ok}/3 (needs 2)"
    if c_ok < b_ok:
        return False, strict, f"word right in {c_ok}/3, baseline {b_ok}/3"
    if mean_c > mean_b + 0.02:
        return False, strict, f"mean sentence CER rises {mean_b:.3f} -> {mean_c:.3f}"
    return True, strict, f"word right {b_ok}/3 -> {c_ok}/3, word errors lower on sentence(s) {[i + 1 for i in better]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=os.getenv("BONCLAY_API", "http://127.0.0.1:8003"))
    ap.add_argument("--out", default=None, help="folder for audio, cache and results (default: bench/results/hindi_lexicon-<stamp>)")
    ap.add_argument("--holdout", action="store_true", help="speak the HOLDOUT sentences with the shipped rules instead of trying candidates")
    ap.add_argument("--only", default=None, help="comma list of target words")
    ap.add_argument("--best", type=int, default=3, help="how many best cases to copy to <out>/best (default 3)")
    a = ap.parse_args()
    out = a.out or os.path.join(str(RESULTS_DIR), f"hindi_lexicon-{now_stamp()}")
    os.makedirs(os.path.join(out, "wav"), exist_ok=True)
    only = {unicodedata.normalize("NFC", w.strip()) for w in a.only.split(",")} if a.only else None
    api = Api(a.api)
    health = require_api(api)
    gate = RateGate(LIMITS["transcribe"])
    cache_path = os.path.join(out, "cache.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}

    def read_back(spoken: str):
        key = hashlib.sha1(spoken.encode()).hexdigest()[:16]
        wav_path = os.path.join(out, "wav", key + ".wav")
        if key not in cache or not os.path.exists(wav_path):
            r, _, _ = api.post("/api/kokoro/generate", json={"voice": VOICE, "text": spoken}, timeout=180)
            if r.status_code != 200:
                raise ApiError(r.status_code, err_text(r))
            open(wav_path, "wb").write(r.content)
            gate.wait()
            r, _, _ = api.post("/api/transcribe", files={"audio_file": ("hi.wav", r.content, "audio/wav")},
                               data={"language": "hi"}, timeout=300)
            if r.status_code != 200:
                raise ApiError(r.status_code, "transcribe: " + err_text(r))
            cache[key] = r.json().get("text", "")
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
        return cache[key], wav_path

    results = {}
    targets = {w: (c, ["shipped"]) for w, c in HOLDOUT.items()} if a.holdout else TARGETS
    for word, (carriers, cands) in targets.items():
        if only and unicodedata.normalize("NFC", word) not in only:
            continue
        variants = {"baseline": []}
        for repl in cands:
            variants[repl] = (list(RULES[("hi", "kokoro")]) if a.holdout
                              else [rule(f"{word}->{repl}", whole_word(word), repl)])
        per = {}
        for vname, rules in variants.items():
            rows = []
            for sent in carriers:
                spoken = apply_rules(sent, rules)
                if vname != "baseline" and spoken == unicodedata.normalize("NFC", sent):
                    raise SystemExit(f"rule for {word} did not change {sent!r}")
                hyp, wav = read_back(spoken)
                rows.append({"sentence": sent, "spoken": spoken, "heard": hyp, "cer": round(sentence_cer(sent, hyp, word), 4),
                             "cer_plain": round(plain_cer(sent, hyp), 4), "word_errors": word_errors(sent, hyp, word), "target_ok": tokens_hold(hyp, word), "wav": wav})
            per[vname] = rows
            print(f"  {word} {vname:<9} CER {[r['cer'] for r in rows]} word edits {[r['word_errors'] for r in rows]} "
                  f"right {sum(r['target_ok'] for r in rows)}/3  heard: {' | '.join(r['heard'] for r in rows)}", flush=True)
        verdicts = {v: decide(per["baseline"], per[v]) for v in cands}
        results[word] = {"variants": per, "verdicts": {k: {"kept": v[0], "strict": v[1], "reason": v[2]}
                                                       for k, v in verdicts.items()}}

    ship = {}
    for word, res in results.items():
        kept = [v for v, x in res["verdicts"].items() if x["kept"]]
        if kept:  # one spelling per word: the lowest mean CER on the transcript as Whisper spelled it, then most right
            ship[word] = min(kept, key=lambda v: (sum(r["cer_plain"] for r in res["variants"][v]),
                                                  -sum(r["target_ok"] for r in res["variants"][v])))
    print(f"\n{'word':<7}{'spelling':<10}{'CER s1':>7}{'s2':>7}{'s3':>7}{'mean':>7}  {'word edits':<11}{'right':>6}  verdict")
    best = []
    for word, res in results.items():
        base = res["variants"]["baseline"]
        for vname, rows in res["variants"].items():
            cers = [r["cer"] for r in rows]
            v = res["verdicts"].get(vname)
            tag = ("baseline" if v is None else f"{'KEEP' if v['kept'] else 'drop'} (strict {'yes' if v['strict'] else 'no'}): {v['reason']}"
                   + ("  <- SHIP" if ship.get(word) == vname else ""))
            print(f"{word:<7}{vname:<10}{cers[0]:>7.3f}{cers[1]:>7.3f}{cers[2]:>7.3f}{sum(cers) / 3:>7.3f}  "
                  f"{str([r['word_errors'] for r in rows]):<11}{sum(r['target_ok'] for r in rows):>4}/3  {tag}")
            if v and v["kept"] and ship.get(word) == vname:
                for b, c in zip(base, rows):
                    best.append((b["word_errors"] - c["word_errors"], word, vname, b, c))
    best.sort(key=lambda x: -x[0])
    bdir = os.path.join(out, "best")
    shutil.rmtree(bdir, ignore_errors=True)
    os.makedirs(bdir, exist_ok=True)
    seen = set()
    for gain, word, vname, b, c in best:
        if (word, vname) in seen:
            continue
        if len(seen) >= a.best:
            break
        seen.add((word, vname))
        stem = f"{len(seen)}_{word}_{vname}"
        shutil.copy(b["wav"], os.path.join(bdir, stem + "__before.wav"))
        shutil.copy(c["wav"], os.path.join(bdir, stem + "__after.wav"))
        print(f"best {len(seen)}: {word} -> {vname} (word edits {b['word_errors']} -> {c['word_errors']}): {c['sentence']}")
    md = ["| word | spelling | CER s1 | CER s2 | CER s3 | mean CER | edits on the word | word right | verdict |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for word, res in results.items():
        for vname, rows in res["variants"].items():
            v = res["verdicts"].get(vname)
            cers = [r["cer"] for r in rows]
            verdict = ("baseline" if v is None else ("**ship**" if ship.get(word) == vname else "kept, not shipped" if v["kept"]
                                                    else "dropped") + f" ({v['reason']})")
            md.append(f"| {word} | {vname} | {cers[0]:.3f} | {cers[1]:.3f} | {cers[2]:.3f} | {sum(cers) / 3:.3f} | "
                      f"{'/'.join(str(r['word_errors']) for r in rows)} | {sum(r['target_ok'] for r in rows)}/3 | {verdict} |")
    open(os.path.join(out, "table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump({"voice": VOICE, "api": a.api, "health": health.get("status"), "ship": ship, "results": results},
              open(os.path.join(out, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nlistening files (before/after pairs): {bdir}\nall audio and results: {out}")


if __name__ == "__main__":
    main()
