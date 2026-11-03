"""Split streamed text into speakable sentences as the text arrives.

A sentence ends at . ? ! … or the Devanagari danda ।, plus any closing quote or bracket, when whitespace follows (so
decimals such as 4.5 and times such as 10.30 stay whole), or at a newline. A full stop after an abbreviation (Dr.,
Mr., a.m., p.m., e.g., i.e., etc.), an initial (A. Kulkarni) or a list number (1.) does not end one. A sentence that
runs past MAX_CHARS without any of these is cut at its last comma, semicolon or colon so speech can still start.

`turn_splitter()` is the splitter of a spoken turn. Kokoro's time is linear in the audio length (about 20 ms plus 70 ms
per second of audio), so shorter pieces reach the caller sooner: pieces are capped at TURN_MAX_CHARS (110) at phrase
boundaries, and the first piece is a short clause (see `first_phrase`) instead of a whole sentence: it is cut at the
first comma, semicolon, colon or dash once 4 words have arrived, at a conjunction or the 8th word when none came
(TURN_FIRST_CLAUSE=off restores the older rule, a cut only after 8 words have arrived).

    sp = SentenceSplitter()
    for delta in stream:
        for sentence in sp.feed(delta):
            speak(sentence)
    for sentence in sp.flush():
        speak(sentence)
"""
import os
import re
from typing import List, Optional

ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "mt", "vs", "etc", "no", "nos", "approx", "dept", "est", "inc",
    "ltd", "co", "corp", "fig", "vol", "ph", "rs", "ave", "rd", "capt", "col", "gen", "lt", "sgt", "gov", "rev",
    "hon", "mx", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mon", "tue",
    "tues", "wed", "thu", "thur", "thurs", "fri", "sat",
    "श्री", "श्रीमती", "डॉ", "कु",
}
MAX_CHARS = 220
SOFT_MIN = 60  # a run-on is only cut at a comma at least this far in
TURN_MAX_CHARS = 110   # spoken turns: the longest piece sent to the voice
TURN_SOFT_MIN = 40
FIRST_PHRASE_WORDS = 8      # the first piece is cut at a clause boundary once this many words have arrived without a full stop
FIRST_PHRASE_MIN_WORDS = 4  # ... and never before this many words (a verb must not be parted from its object)

FIRST_CLAUSE_MIN_WORDS = 4   # TURN_FIRST_CLAUSE=on: a clause mark ends the first piece once this many words are in
FIRST_CLAUSE_WORDS = 8       # ... and with no clause mark the first piece is cut at the 8th word (or the conjunction before it)
FIRST_CLAUSE_MAX_WORDS = 11  # ... moved on past a word that cannot end a phrase ("the", "of", "का"), but never beyond this

_CLAUSE_MARKS = ",;:،—–"
# Words a clause can start with; the first phrase may end just before one of them (English, Hindi, Hinglish).
_CONJUNCTIONS = re.compile(r"\s(?=(?:and|but|or|so|because|while|which|then|और|लेकिन|या|क्योंकि|जबकि|तो|aur|lekin|ya|kyunki|toh)\s)",
                           re.IGNORECASE)

# Words that cannot end a phrase: a first piece that would stop on one is extended by a word.
_DANGLING = frozenset(
    "a an the of to in on at for with and or but so if as by from into than that this these those my your our their "
    "its his her i we you "
    "का की के को में से पर तक लिए ने और या कि जो एक यह वह मेरा आपका हमारा "
    "ka ki ke ko mein se par tak liye ne aur ya jo ek yeh woh mera aapka hamara".split())
# Hindi and Hinglish postpositions and particles belong to the word before them: a piece is never cut just ahead of one.
_ATTACHING = frozenset("से को का की के में पर तक ने भी ही se ko ka ki ke mein me par pe tak ne bhi hi "
                         "है हैं था थी थे हो होगा होगी रहा रही रहे hai hain tha thi the ho hoga hogi raha rahi rahe".split())
# Words a clause can start with: the first piece may end just before one of them (English, Hindi, Hinglish).
_CONJUNCTION_WORDS = frozenset("and but or so because while which then और लेकिन या क्योंकि जबकि तो aur lekin ya kyunki toh".split())
# "e.g.," and "i.e.," introduce what follows: the comma after them is not a place to stop.
_INTRODUCERS = frozenset({"e.g.", "i.e.", "viz.", "eg.", "ie."})
_TOKEN = re.compile(r"\S+")

_END = ".?!…।"
_CLOSERS = "\"'”’)]}»"
_WORD_CHARS = re.compile(r"[^\s]+$")
_DOTTED = re.compile(r"^(?:[A-Za-z]\.){1,}[A-Za-z]?$")  # a.m., p.m., e.g., i.e., U.S.A


def _bare(token: str) -> str:
    """A word without the punctuation around it, lower case."""
    return token.strip(_CLOSERS + "\"'“‘([{«" + _CLAUSE_MARKS + ".!?…।").lower()


def _speakable(text: str) -> bool:
    """Has at least one letter or digit (a stray "..." or "-" is not worth a synthesis call)."""
    return any(ch.isalnum() for ch in text)


def _ends_sentence(buf: str, i: int) -> bool:
    """Is the terminator at buf[i] the end of a sentence? Only called when whitespace follows its closers."""
    ch = buf[i]
    if ch != ".":
        return True
    word = _WORD_CHARS.search(buf[:i + 1])
    token = word.group(0) if word else "."
    token = token.lstrip("\"'“‘([{«")
    core = token.rstrip(".")
    if not core:
        return True  # a lone "." after a space
    if core.isdigit():
        return False  # "1." of a numbered list
    if core.lower() in ABBREVIATIONS:
        return False
    if _DOTTED.match(token):
        return False  # a.m. p.m. e.g.
    if len(core) == 1 and core.isalpha() and core != "I":
        return False  # an initial: "A. Kulkarni"
    return True


def first_clause_enabled() -> bool:
    """TURN_FIRST_CLAUSE=on|off (default on), read on every call."""
    return (os.getenv("TURN_FIRST_CLAUSE", "on") or "on").strip().lower() not in ("off", "0", "false", "no")


def turn_splitter(first_phrase: bool = True, first_clause: Optional[bool] = None) -> "SentenceSplitter":
    """The splitter of a spoken turn: pieces of at most TURN_MAX_CHARS and (`first_phrase`) a short first clause. A text
    that is already complete (a regenerated reply) has no use for a short first clause: pass False.
    `first_clause` None follows TURN_FIRST_CLAUSE (default on)."""
    return SentenceSplitter(max_chars=TURN_MAX_CHARS, soft_min=TURN_SOFT_MIN, first_phrase=first_phrase,
                            split_long=True,
                            first_clause=first_clause_enabled() if first_clause is None else first_clause)


class SentenceSplitter:
    def __init__(self, max_chars: int = MAX_CHARS, soft_min: int = SOFT_MIN, first_phrase: bool = False,
                 split_long: bool = False, first_clause: bool = False):
        """`first_phrase`: the first piece of the text is cut short at a natural boundary so speech can start on a
        phrase instead of a whole sentence. With `first_clause` that is the first comma, semicolon, colon or dash after
        FIRST_CLAUSE_MIN_WORDS words (a sentence end earlier than that wins, as always), else the conjunction or the
        FIRST_CLAUSE_WORDS-th word, whatever the order the text arrives in. Without it (the older rule) it is a comma, or
        just before a conjunction, once FIRST_PHRASE_WORDS words have arrived without a sentence end. `split_long`: a finished sentence longer than `max_chars` is cut at clause boundaries (`max_chars`
        is then a real cap on every piece, not only on a sentence that is still arriving)."""
        self.buf = ""
        self.max_chars = max_chars
        self.soft_min = soft_min
        self.first_phrase = first_phrase
        self.first_clause = first_clause
        self.split_long = split_long
        self._first_done = False

    def feed(self, delta: str) -> List[str]:
        """Add streamed text; returns the sentences that are now complete (in order)."""
        self.buf += delta
        return self._drain(final=False)

    def flush(self) -> List[str]:
        """End of the stream: whatever is left is the last sentence."""
        out = self._drain(final=True)
        rest = self.buf.strip()
        self.buf = ""
        if rest and _speakable(rest):
            out.append(rest)
        return out

    def _drain(self, final: bool) -> List[str]:
        out: List[str] = []
        while True:
            cut = self._find_cut(final)
            if cut is None:
                return out
            piece, self.buf = self.buf[:cut], self.buf[cut:]
            piece = piece.strip()
            if piece and _speakable(piece):
                parts = self._cap(piece) if self.split_long else [piece]
                out.extend(parts)
                self._first_done = True

    def _find_cut(self, final: bool):
        """Index just past the first complete piece in the buffer, or None."""
        buf, n = self.buf, len(self.buf)
        s = self._sentence_cut()
        if self.first_phrase and not self._first_done:
            if self.first_clause:
                c = self._first_clause_cut()
                if c is not None and (s is None or c < s):  # a sentence that ends first is still a whole piece
                    return c
            elif s is None:
                c = self._first_phrase_cut()
                if c is not None:
                    return c
        if s is not None:
            return s
        if n > self.max_chars:
            cut = max(buf.rfind(c, 0, self.max_chars) for c in (",", ";", ":", "،", "、"))
            if cut >= self.soft_min:
                return cut + 1
            space = buf.rfind(" ", 0, self.max_chars)
            if space >= self.soft_min:
                return space + 1
        return None

    def _sentence_cut(self):
        """Index just past the first complete sentence in the buffer (a sentence end, or a newline), or None."""
        buf, n = self.buf, len(self.buf)
        i = 0
        while i < n:
            ch = buf[i]
            if ch == "\n":
                return i + 1
            if ch in _END:
                j = i
                while j + 1 < n and buf[j + 1] in _END:  # "?!" "..." stay together
                    j += 1
                k = j
                while k + 1 < n and buf[k + 1] in _CLOSERS:
                    k += 1
                if k + 1 >= n:
                    break  # the next character decides (a decimal, or more text): wait for it
                if buf[k + 1].isspace() and _ends_sentence(buf, j):
                    return k + 1
                i = k
            i += 1
        return None

    def _first_clause_cut(self):
        """Where the first clause of the reply may end, or None (the text so far is too short, or has no boundary yet).

        In order of the words as they arrive: (1) a clause mark - comma, semicolon, colon, dash, Arabic comma - at the
        end of a word once FIRST_CLAUSE_MIN_WORDS words are in; the mark must be followed by whitespace (so 1,500 and
        10:30 and http://x are never cut), or be the last character of the text so far after a non-digit; the comma
        after "e.g." or "i.e." does not count. (2) FIRST_CLAUSE_WORDS complete words with no mark: the piece ends before
        the first conjunction among words 5 to 8, else after the 8th word, carried on over words that cannot end a
        phrase (the, of, का) up to FIRST_CLAUSE_MAX_WORDS. The result depends only on the text, not on how it was
        chopped into deltas. A sentence end is the caller's business (`_sentence_cut`)."""
        buf, n = self.buf, len(self.buf)
        toks = [(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(buf)]
        words = 0
        for idx, (start, end, tok) in enumerate(toks):
            complete = end < n  # whitespace follows: the word is finished
            if _speakable(tok):
                words += 1
            stem = tok.rstrip(_CLOSERS)
            if stem and stem[-1] in _CLAUSE_MARKS and words >= FIRST_CLAUSE_MIN_WORDS:
                digit_before = len(stem) > 1 and stem[-2].isdigit()
                if (complete or not digit_before) and stem.lower() not in _INTRODUCERS \
                        and stem[:-1].lower() not in _INTRODUCERS:
                    return end
            if not complete:
                return None  # the last word is still arriving
            if words >= FIRST_CLAUSE_WORDS and _speakable(tok):
                if words == FIRST_CLAUSE_WORDS:
                    seen = 0
                    for s2, _e2, t2 in toks[:idx + 1]:
                        if seen >= FIRST_CLAUSE_MIN_WORDS and t2.lower() in _CONJUNCTION_WORDS:
                            return s2
                        if _speakable(t2):
                            seen += 1
                if words >= FIRST_CLAUSE_MAX_WORDS:
                    return end
                if _bare(tok) in _DANGLING or any(ch.isdigit() for ch in tok):
                    continue  # "... open from nine to" cannot stop here, nor can "... costs 4.5" (the unit follows)
                if idx + 1 >= len(toks) or toks[idx + 1][1] >= n:
                    return None  # the next word decides (a postposition belongs to this one): wait for it
                if _bare(toks[idx + 1][2]) in _ATTACHING:
                    continue
                return end
        return None

    def _first_phrase_cut(self):
        """Where the first piece may end, or None. Called only while the first sentence is still arriving."""
        buf = self.buf
        if len(buf.split()) < FIRST_PHRASE_WORDS:
            return None
        for i, ch in enumerate(buf):
            if ch in _CLAUSE_MARKS and i + 1 < len(buf) and buf[i + 1].isspace() \
                    and len(buf[:i].split()) >= FIRST_PHRASE_MIN_WORDS:
                return i + 1
        for m in _CONJUNCTIONS.finditer(buf):
            if len(buf[:m.start()].split()) >= FIRST_PHRASE_MIN_WORDS:
                return m.start() + 1
        return None

    def _cap(self, piece: str) -> List[str]:
        """A finished sentence longer than max_chars, cut at clause boundaries into pieces of at most max_chars (a clause
        with no boundary in it is cut at a space)."""
        if len(piece) <= self.max_chars:
            return [piece]
        out: List[str] = []
        cur = ""
        for clause in re.split(r"(?<=[,;:،])\s+", piece):
            if cur and len(cur) + 1 + len(clause) > self.max_chars:
                out.append(cur)
                cur = clause
            else:
                cur = f"{cur} {clause}".strip()
        if cur:
            out.append(cur)
        final: List[str] = []
        for part in out:
            while len(part) > self.max_chars:
                space = part.rfind(" ", self.soft_min, self.max_chars)
                if space < 0:
                    break
                final.append(part[:space].strip())
                part = part[space:].strip()
            final.append(part)
        if len(final) > 1 and len(final[-1]) < self.soft_min and len(final[-2]) + 1 + len(final[-1]) <= self.max_chars * 1.3:
            final[-2:] = [f"{final[-2]} {final[-1]}"]    # no stub at the end
        return final
