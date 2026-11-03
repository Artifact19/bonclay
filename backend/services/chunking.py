"""Split a document into passages for retrieval.

Small passages (about 120 words) keep the prompt short, so the language model's prompt cache stays useful and a turn
costs fewer tokens; a heading-aware split keeps a passage inside one section, and every passage of a section starts
with its heading so the section title reaches the embedding (the same packer backend/bench/retrieval_eval.py uses).

Lines are the unit: a passage is made of whole lines (a table row or a list item is never cut), packed up to
`chunk_size` words. A single line longer than that is cut into word windows. The next passage of a section starts with
the last `overlap` words of the previous one, so a sentence that straddles a boundary is still found whole.

Passages already stored with the old 500-word size keep working: nothing here changes them. `rechunk_needed` tells the
document code that a stored document has such big passages, and `backend.rechunk_agent_documents` re-splits them.
"""
import re
from typing import List, Optional

DEFAULT_CHUNK_WORDS = 120
DEFAULT_OVERLAP_WORDS = 20
LEGACY_CHUNK_WORDS = 500          # the size passages were stored with before this module (see backend.chunk_text)

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
MAX_HEADING_WORDS = 30            # a "# ..." line longer than this is body text (e.g. an old document flattened to one line)


def _words(line: str) -> int:
    return len(line.split())


def _windows(line: str, size: int, overlap: int) -> List[str]:
    """A line longer than `size` words, cut into overlapping word windows."""
    words = line.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_WORDS, overlap: int = DEFAULT_OVERLAP_WORDS) -> List[str]:
    """Passages of at most about `chunk_size` words (plus the heading and the overlap)."""
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    chunks: List[str] = []
    heading = ""
    buf: List[str] = []
    count = 0
    carry = ""  # the tail of the previous passage of this section

    def emit(body: str, with_carry: bool = True) -> None:
        nonlocal carry
        head = heading + "\n" if heading else ""
        chunks.append(head + (carry + " " if carry and with_carry else "") + body)
        tail = body.split()
        carry = " ".join(tail[-overlap:]) if overlap else ""

    def flush() -> None:
        nonlocal buf, count
        if buf:
            emit("\n".join(buf))
        buf, count = [], 0

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if _HEADING.match(raw) and _words(line) <= MAX_HEADING_WORDS:
            flush()
            carry = ""      # never across a heading
            heading = line.lstrip("# ").strip()
            continue
        if _words(line) > chunk_size:   # one huge line (a PDF paragraph): its own overlapping windows
            flush()
            for window in _windows(line, chunk_size, overlap):
                emit(window, with_carry=False)
            continue
        n = _words(line)
        if buf and count + n > chunk_size:
            flush()
        buf.append(line)
        count += n
    flush()
    if not chunks and (text or "").strip():
        chunks = [" ".join(text.split())]
    return chunks


def rechunk_needed(chunk_texts: List[str], max_words: int = DEFAULT_CHUNK_WORDS * 2) -> bool:
    """True when any stored passage is much bigger than the current chunk size (a document ingested before the
    120-word split), so it would still put thousands of tokens in front of the model."""
    return any(_words(c) > max_words for c in chunk_texts)


LEGACY_OVERLAP_WORDS = 50         # the overlap of the old 500-word passages (step 450)


def text_from_legacy_chunks(chunk_texts: List[str]) -> str:
    """The document text rebuilt from its stored passages, in order (the old passages overlapped by 50 words; the
    overlap is dropped). Line breaks were already lost when those passages were made, so the result is one block of
    words that `chunk_text` cuts into word windows."""
    words: List[str] = []
    for i, c in enumerate(chunk_texts):
        w = c.split()
        words += w[LEGACY_OVERLAP_WORDS:] if i else w
    return " ".join(words)


# =====================================================================================================================
# Structure-aware chunking: `chunk_document(text, title)` -> [{"content", "header", "kind": "passage"}]
#
# `chunk_text` above cuts by lines and prefixes the heading to every passage. `chunk_document` reads the document's own
# structure instead: headings (markdown, numbered, ALL-CAPS, short title-like lines), FAQ pairs (kept whole), tables (one
# row per passage with the header row repeated) and prose (whole sentences, ~120 words, 20 words of overlap). The
# passage text is the content only; the title and section path travel in `header` ("Title > Section > FAQ question") and
# the text that is embedded is `embedding_text(header, content)` (header + newline + content).
# =====================================================================================================================
HEADER_SEP = " › "           # " > " as the single guillemet used in headers: "Hospital > OPD timings"
MAX_HEADER_PART_CHARS = 160
_OVERSIZE_FACTOR = 1.6           # a sentence up to 1.6x the target is kept whole; a longer one is cut at clauses, then words
_TERMINAL_END = ".?!।॥;:"
_DANDA = "।॥"
_CLOSERS = "\"')”’]"
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "st", "no", "vs", "etc", "e.g", "i.e", "rs", "sr", "jr", "mt", "approx",
                  "dept", "fig", "inc", "ltd", "co", "viz", "cf", "ph", "smt", "shri", "sri", "ave", "rd", "min", "max"}
_WS_FIX = str.maketrans({" ": " ", " ": " ", " ": " ", " ": " ", "​": None, "﻿": None})

_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_RULE = re.compile(r"^\s*[-=_*—]{3,}\s*$")
_SETEXT = re.compile(r"^\s*(=+|-+)\s*$")
_PAGE_NO = re.compile(r"^\s*(?:page\s+\d+(?:\s*(?:of|/)\s*\d+)?|-\s*\d+\s*-)\s*$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•●▪–—]|\d{1,3}[.)])\s+\S")
_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+(\S.*)$")
_Q_PREFIX = re.compile(r"^(?:\*\*|__)?\s*(?:Q(?:uestion)?\s*\.?\s*\d*\s*[:.)\-–—]|प्रश्न\s*\d*\s*[:.)\-–—])"
                       r"\s*(?:\*\*|__)?\s*(.*)$", re.I)
_A_PREFIX = re.compile(r"^(?:\*\*|__)?\s*(?:A(?:nswer|ns)?\s*\.?\s*\d*\s*[:.)\-–—]|उत्तर\s*[:.)\-–—])"
                       r"\s*(?:\*\*|__)?\s*(.*)$", re.I)
_INLINE_A = re.compile(r"\s+(?:\*\*)?(?:A(?:nswer)?\s*[:.)]|उत्तर\s*[:.)])(?:\*\*)?\s+", re.I)
_LIST_PREFIX = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*•])\s+")
_SEP_CELL = re.compile(r"^:?-{2,}:?$")
_CELL_RUN = re.compile(r"\S+(?: \S+)*")
MAX_HEADING_WORDS_PLAIN = 8
MAX_QUESTION_WORDS = 35


def embedding_text(header: Optional[str], content: str) -> str:
    """The text that is embedded and BM25-indexed for a passage: its header, a newline, then the content."""
    header = (header or "").strip()
    return f"{header}\n{content}" if header else content


def pretty_title(filename: str) -> str:
    """A readable title from a file name: 'Sahyadri_Care_Hospital_Pune_FAQ.pdf' -> 'Sahyadri Care Hospital Pune FAQ'."""
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", (filename or "").strip())
    stem = re.sub(r"[_\-]+", " ", stem)
    return " ".join(stem.split())


def _clean_line(raw: str) -> str:
    return raw.translate(_WS_FIX)


def _strip_md(s: str) -> str:
    s = s.strip()
    for mark in ("**", "__"):
        if s.startswith(mark) and s.endswith(mark) and len(s) > 2 * len(mark):
            s = s[len(mark):-len(mark)].strip()
    return s.strip("`").strip()


def _trim_part(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= MAX_HEADER_PART_CHARS else s[:MAX_HEADER_PART_CHARS].rsplit(" ", 1)[0] + "..."


def _norm_key(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.casefold()).strip()


def guess_title(text: str, filename: str = "") -> str:
    """The document's title: its first line when that is short and title-like (no sentence punctuation at the end), else
    the file name made readable."""
    for raw in (text or "").splitlines():
        line = _strip_md(_clean_line(raw).strip().lstrip("#").strip())
        if not line:
            continue
        if (1 <= _words(line) <= 10 and line[-1] not in _TERMINAL_END and not _table_like(raw)
                and any(c.isalpha() for c in line) and not _Q_PREFIX.match(line)):
            return line
        break
    return pretty_title(filename)


def _table_like(raw: str) -> bool:
    s = raw.strip()
    return "|" in s or "\t" in s or bool(re.search(r"\S {2,}\S", s))


# ----------------------------------------------------------------------------------------------- sentences

def split_sentences(text: str) -> List[str]:
    """Whole sentences. A sentence ends at . ? ! (followed by a space and a new sentence, not after an abbreviation such as
    Dr. or Rs., not before a lowercase word as in '9 a.m. to 5 p.m.') and at the Hindi danda and double danda (always)."""
    text = " ".join(_clean_line(text).split())
    out: List[str] = []
    n, start, i = len(text), 0, 0
    while i < n:
        ch = text[i]
        if ch in _DANDA:
            j = i + 1
            while j < n and text[j] in _DANDA:
                j += 1
            piece = text[start:j].strip()
            if piece:
                out.append(piece)
            start = i = j
            continue
        if ch in ".?!":
            j = i + 1
            while j < n and text[j] in ".?!":
                j += 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            if j < n and text[j] == " ":
                k = j
                while k < n and text[k] == " ":
                    k += 1
                nxt = text[k] if k < n else ""
                boundary = bool(nxt) and not ("a" <= nxt <= "z")
                if boundary and ch == "." and text[i:j].rstrip(_CLOSERS) == ".":
                    before = text[start:i]
                    token = before.split()[-1].strip("([\"'").lower() if before.split() else ""
                    if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                        boundary = False
                    elif token.isdigit() and before.strip() == token:   # "1. Bring your card": a list number
                        boundary = False
                if boundary:
                    piece = text[start:j].strip()
                    if piece:
                        out.append(piece)
                    start = k
                    i = k
                    continue
            i = j
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _split_long(sentence: str, size: int, overlap: int) -> List[str]:
    """A sentence longer than the oversize limit: cut at clause marks (, ; : dashes), greedily up to `size` words; a clause
    that is still too long is cut into word windows."""
    clauses = [c for c in re.split(r"(?<=[,;:–—])\s+", sentence) if c.strip()]
    pieces: List[str] = []
    cur: List[str] = []
    count = 0
    for c in clauses:
        n = _words(c)
        if n > size:
            if cur:
                pieces.append(" ".join(cur))
                cur, count = [], 0
            pieces.extend(_windows(c, size, overlap))
            continue
        if cur and count + n > size:
            pieces.append(" ".join(cur))
            cur, count = [], 0
        cur.append(c)
        count += n
    if cur:
        pieces.append(" ".join(cur))
    return pieces or [sentence]


# ------------------------------------------------------------------------------------------------ prose units

def _prose_units(lines: List[str]) -> List[tuple]:
    """Units for packing: (text, starts_new_line, paragraph_words). A unit is a whole sentence, or a line without sentence
    punctuation (a list item, a "Label: value" line). Wrapped PDF lines are joined; a line much shorter than the longest
    line of its paragraph, or a bullet, ends a logical line."""
    paragraphs: List[List[str]] = []
    cur: List[str] = []
    for raw in lines:
        s = raw.strip()
        if s:
            if (not cur and paragraphs and s[0].islower() and paragraphs[-1][-1][-1:] not in _TERMINAL_END
                    and not _BULLET.match(s)):
                cur = paragraphs.pop()      # a sentence that continues after a blank line (a page break)
            cur.append(s)
        elif cur:
            paragraphs.append(cur)
            cur = []
    if cur:
        paragraphs.append(cur)
    units: List[tuple] = []
    for para in paragraphs:
        maxlen = max(len(l) for l in para)
        logical: List[str] = []
        buf = ""
        prev_short = False
        for l in para:
            if buf and (_BULLET.match(l) or prev_short):
                logical.append(buf)
                buf = l
            else:
                buf = f"{buf} {l}" if buf else l
            prev_short = len(l) < 0.55 * maxlen and len(para) > 1
        if buf:
            logical.append(buf)
        para_words = sum(_words(l) for l in logical)
        first_in_para = True
        for line in logical:
            sentences = split_sentences(line)
            for j, sentence in enumerate(sentences):
                units.append((sentence, j == 0, para_words if first_in_para else 0))
                first_in_para = False
    return units


def _join_units(units: List[tuple]) -> str:
    out = ""
    for text, nl, _ in units:
        out = text if not out else out + ("\n" if nl else " ") + text
    return out


def _pack(units: List[tuple], size: int, overlap: int) -> List[str]:
    """Pack units into passages of about `size` words. Whole units only; the next passage starts with the trailing whole
    sentences of the previous one that fit in `overlap` words; a paragraph that would overflow starts a new passage once
    the current one is at least 40% full."""
    limit = int(size * _OVERSIZE_FACTOR)
    expanded: List[tuple] = []
    for text, nl, pw in units:
        if _words(text) > limit:
            for k, piece in enumerate(_split_long(text, size, overlap)):
                expanded.append((piece, nl if k == 0 else True, pw if k == 0 else 0))
        else:
            expanded.append((text, nl, pw))
    passages: List[str] = []
    buf: List[tuple] = []
    count = 0
    fresh = 0      # units in `buf` that are new (not carried over from the previous passage)

    def flush() -> None:
        nonlocal buf, count, fresh
        if fresh:
            passages.append(_join_units(buf))
        carry: List[tuple] = []
        total = 0
        for u in reversed(buf if fresh else []):
            w = _words(u[0])
            if total + w > overlap:
                break
            carry.insert(0, u)
            total += w
        buf, count, fresh = carry, total, 0

    for unit in expanded:
        n = _words(unit[0])
        pw = unit[2]
        if fresh and pw and count + pw > size and count >= 0.4 * size:
            flush()
        if fresh and count + n > size:
            flush()
        if count + n > size:       # the carried overlap does not leave room for this unit
            buf, count = [], 0
        buf.append(unit)
        count += n
        fresh += 1
    if fresh:
        passages.append(_join_units(buf))
    return passages


# ---------------------------------------------------------------------------------------------------- tables

def _line_cells(raw: str) -> tuple:
    """(kind, [(key, text)]) with kind 'pipe', 'tab', 'space' or None. `key` is the column index for pipe/tab tables and
    the start offset for space-aligned ones (cells separated by two or more spaces, as PDF layout extraction gives)."""
    s = raw.rstrip()
    st = s.strip()
    if not st:
        return None, []
    if "|" in st and st.count("|") >= 1 and not st.startswith("#"):
        parts = st.split("|")
        if parts and parts[0].strip() == "":
            parts = parts[1:]
        if parts and parts[-1].strip() == "":
            parts = parts[:-1]
        if len(parts) >= 2:
            return "pipe", [(i, p.strip()) for i, p in enumerate(parts)]
    if "\t" in st:
        parts = [p.strip() for p in st.split("\t")]
        if sum(1 for p in parts if p) >= 2:
            return "tab", [(i, p) for i, p in enumerate(parts)]
    cells = [(m.start(), m.group()) for m in _CELL_RUN.finditer(s)]
    if len(cells) >= 2:
        return "space", cells
    return None, cells


def _col_of(starts: List[int], pos: int, tol: int = 3) -> Optional[int]:
    """The column a cell that starts at `pos` belongs to: the right-most column that starts at or before it (a few
    characters of slack), or None when it starts left of the first column."""
    col = None
    for idx, st in enumerate(starts):
        if pos + tol >= st:
            col = idx
    return col


def _scan_table(lines: List[str], i: int):
    """Try to read a table starting at line i. Returns (header_cells, rows, next_index) or None."""
    kind, first = _line_cells(lines[i])
    if kind is None:
        return None
    header = [t for _, t in first]
    if kind in ("pipe", "tab"):
        rows: List[List[str]] = []
        j = i + 1
        while j < len(lines):
            k2, cells = _line_cells(lines[j])
            if k2 != kind:
                break
            texts = [t for _, t in cells]
            if all(_SEP_CELL.match(t) for t in texts if t):
                j += 1
                continue
            if len(texts) < len(header):
                texts = texts + [""] * (len(header) - len(texts))
            rows.append(texts[:max(len(header), len(texts))])
            j += 1
        return (header, rows, j) if rows else None
    # space-aligned
    if len(header) > 8 or any(_words(h) > 6 or h[-1] in ".?!।" for h in header):
        return None
    starts = [k for k, _ in first]
    rows = []
    j = i + 1
    last_end = j
    prev_blank = False
    while j < len(lines):
        raw = lines[j]
        if not raw.strip():
            prev_blank = True
            j += 1
            continue
        k2, cells = _line_cells(raw)
        cols = [(_col_of(starts, key) if k2 == "space" or k2 is None else None) for key, _ in cells]
        if k2 == "space" and all(c is not None for c in cols) and len(set(cols)) == len(cols):
            cell_map = {c: t for c, (_, t) in zip(cols, cells)}
            if 0 in cell_map:
                rows.append([cell_map.get(c, "") for c in range(len(header))])
            elif rows and not prev_blank:
                for c, t in cell_map.items():
                    rows[-1][c] = f"{rows[-1][c]} {t}".strip()
            else:
                break
        elif k2 is None and len(cells) == 1 and rows and not prev_blank:
            c = _col_of(starts, cells[0][0])
            if c is None:
                break
            rows[-1][c] = f"{rows[-1][c]} {cells[0][1]}".strip()
        else:
            break
        prev_blank = False
        j += 1
        last_end = j
    return (header, rows, last_end) if len(rows) >= 2 else None


# ---------------------------------------------------------------------------------------------------- scanning

def _heading_at(raw: str, prev_blank: bool, prev_terminal: bool, has_blank_structure: bool,
                next_line: Optional[str], next_is_list: bool):
    """(level, text) when this line is a heading, else None. Levels: markdown # count, numbered 1.2.3 depth, ALL-CAPS 1,
    short plain / bold line 2-3."""
    s = raw.strip()
    m = _MD_HEADING.match(raw)
    if m and _words(m.group(2)) <= MAX_HEADING_WORDS:
        return len(m.group(1)), _strip_md(m.group(2))
    if m:
        return None
    if _words(s) > 12 or not any(ch.isalpha() for ch in s):
        return None
    bold = _strip_md(s)
    is_bold = bold != s
    core = bold.rstrip(":").strip()
    if not core or core[-1] in _TERMINAL_END or core.endswith(","):
        return None
    if not (prev_blank or prev_terminal) or next_line is None:
        return None
    nm = _NUMBERED.match(core)
    if nm:
        number, title = nm.group(1), nm.group(2).strip()
        if _words(title) > 10 or title[-1:] in _TERMINAL_END:
            return None
        multi = "." in number
        if next_is_list and not multi:
            return None
        if title[0].islower():
            return None
        return number.count(".") + 1, title
    letters = [c for c in core if c.isalpha()]
    if len(letters) >= 3 and all(not c.islower() for c in letters) and any(c.isupper() for c in letters):
        return 1, core
    if is_bold and _words(core) <= 12:
        return 3, core
    if _words(core) > MAX_HEADING_WORDS_PLAIN or (s.endswith(":") and _words(core) > 6):
        return None
    if core[0].islower():
        return None
    if has_blank_structure:
        if not prev_blank:
            return None
    else:
        if _words(next_line) < 5:
            return None
    if not (core[0].isalpha() or core[0].isdigit()) or ":" in core:
        return None
    return 2, core


def _faq_question(raw: str, prev_ok: bool, next_line: Optional[str]):
    """The question text when the line starts a FAQ entry (and, for 'Q: q? A: a' on one line, the inline answer)."""
    s = raw.strip()
    m = _Q_PREFIX.match(s)
    if m and m.group(1).strip():
        rest = m.group(1).strip()
        parts = _INLINE_A.split(rest, maxsplit=1)
        return _strip_md(parts[0]).strip(), (parts[1].strip() if len(parts) > 1 else "")
    body = _strip_md(_LIST_PREFIX.sub("", s, count=1))
    if body.endswith(("?", "？")) and 2 <= _words(body) <= MAX_QUESTION_WORDS and prev_ok and next_line is not None:
        if not ("a" <= body[0] <= "z"):
            return body, ""
    return None


class _Section:
    """The heading path of the current position in the document."""

    def __init__(self, title: str):
        self.title = _trim_part(title) if title else ""
        self.stack: List[tuple] = []

    def push(self, level: int, text: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, _trim_part(text)))

    def header(self, extra: str = "") -> str:
        parts = ([self.title] if self.title else []) + [t for _, t in self.stack] + ([_trim_part(extra)] if extra else [])
        out: List[str] = []
        for p in parts:
            if p and (not out or _norm_key(p) != _norm_key(out[-1])):
                out.append(p)
        return HEADER_SEP.join(out)


def chunk_document(text: str, title: str = "", chunk_size: int = DEFAULT_CHUNK_WORDS,
                   overlap: int = DEFAULT_OVERLAP_WORDS) -> List[dict]:
    """Passages of a document, cut along its structure. Returns [{"content", "header", "kind": "passage"}].

    `header` is "title > section > sub-section" (and "> the FAQ question" inside a FAQ entry); `content` is the passage text
    without it. FAQ entries stay together as "Q: ...\\nA: ..."; each table row is its own passage with the header row of
    the table repeated ("Department | Days\\nCardiology | Mon-Sat"); prose is packed from whole sentences (. ? ! and the
    Hindi danda) up to `chunk_size` words with `overlap` words carried into the next passage, never across a heading, FAQ
    entry or table."""
    chunk_size = max(8, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 3))
    lines = [_clean_line(l).replace("\r", "") for l in (text or "").splitlines()]
    nonblank = sum(1 for l in lines if l.strip())
    if not nonblank:
        return []
    has_blank_structure = (len(lines) - nonblank) >= 0.1 * nonblank
    section = _Section(title)
    title_key = _norm_key(title) if title else ""
    out: List[dict] = []
    prose: List[str] = []              # raw lines of the current prose block (blank lines kept)
    faq: Optional[dict] = None         # {"q": str, "lines": [...]}
    seen_content = False

    def emit(header: str, content: str) -> None:
        if content.strip():
            out.append({"content": content.strip(), "header": header, "kind": "passage"})

    def flush_prose() -> None:
        nonlocal prose
        if any(l.strip() for l in prose):
            for body in _pack(_prose_units(prose), chunk_size, overlap):
                emit(section.header(), body)
        prose = []

    def flush_faq() -> None:
        nonlocal faq
        if faq is None:
            return
        q, ans = faq["q"], faq["lines"]
        faq = None
        units = _prose_units(ans)
        if not units:                           # a question with no answer text: keep it as prose
            prose.append(q)
            flush_prose()
            return
        head = section.header(q)
        budget = max(20, chunk_size - _words(q) - 2)
        total = sum(_words(u[0]) for u in units)
        bodies = [_join_units(units)] if total <= budget * _OVERSIZE_FACTOR else _pack(units, budget, min(overlap, budget // 3))
        for body in bodies:
            emit(head, f"Q: {q}\nA: {body}")

    i = 0
    prev_blank, prev_terminal = True, False
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if not s:
            if faq is not None:
                faq["lines"].append("")
            else:
                prose.append("")
            prev_blank = True
            i += 1
            continue
        if _RULE.match(s) or _PAGE_NO.match(s):
            i += 1
            continue
        # a table (before anything else: its cells would look like headings)
        tbl = _scan_table(lines, i)
        if tbl is not None:
            header_cells, rows, nxt = tbl
            flush_faq()
            flush_prose()
            head_row = " | ".join(header_cells)
            hdr = section.header()
            for row in rows:
                emit(hdr, f"{head_row}\n" + " | ".join(row))
            seen_content = True
            prev_blank, prev_terminal = True, False
            i = nxt
            continue
        # look ahead
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        next_line = lines[j] if j < len(lines) else None
        next_is_list = bool(next_line and _NUMBERED.match(next_line))
        # setext heading: "Title" then "=====" / "-----"
        if next_line is not None and j == i + 1 and _SETEXT.match(next_line) and len(next_line.strip()) >= 3 \
                and not _BULLET.match(raw) and _words(s) <= 12:
            flush_faq()
            flush_prose()
            level = 1 if next_line.strip()[0] == "=" else 2
            text_h = _strip_md(s)
            if not seen_content and not out and title_key and _norm_key(text_h) == title_key:
                pass
            else:
                section.push(level, text_h)
            seen_content = True
            prev_blank, prev_terminal = True, False
            i = j + 1
            continue
        # FAQ question (explicit "Q:" or a line that ends with "?")
        mh = _MD_HEADING.match(raw)
        if mh and _strip_md(mh.group(2)).endswith(("?", "？")):
            q = _strip_md(mh.group(2))
            flush_faq()
            flush_prose()
            faq = {"q": q, "lines": []}
            seen_content, prev_blank, prev_terminal = True, True, False
            i += 1
            continue
        if not mh:
            fq = _faq_question(raw, prev_blank or prev_terminal or faq is not None, next_line)
            if fq is not None:
                flush_faq()
                flush_prose()
                faq = {"q": fq[0], "lines": [fq[1]] if fq[1] else []}
                seen_content, prev_blank, prev_terminal = True, False, False
                i += 1
                continue
        # heading
        h = _heading_at(raw, prev_blank, prev_terminal, has_blank_structure, next_line, next_is_list)
        if h is not None:
            level, text_h = h
            flush_faq()
            flush_prose()
            if not seen_content and not out and title_key and _norm_key(text_h) == title_key:
                pass                               # the title line itself
            else:
                section.push(level, text_h)
            seen_content = True
            prev_blank, prev_terminal = True, False
            i += 1
            continue
        # plain text line (an FAQ answer or prose)
        if faq is not None:
            am = _A_PREFIX.match(s)
            faq["lines"].append(am.group(1) if am and not any(l.strip() for l in faq["lines"]) else s)
        else:
            prose.append(s)
        seen_content = True
        prev_blank = False
        prev_terminal = s[-1] in ".?!।॥"
        i += 1
    flush_faq()
    flush_prose()
    if not out and (text or "").strip():
        out = [{"content": " ".join(text.split()), "header": section.header(), "kind": "passage"}]
    return out


def pdf_text_from_reader(reader) -> str:
    """Text of a pypdf reader's pages. Uses pypdf's layout mode, which keeps table columns apart (two or more spaces between
    cells, rows one under the other) so tables can be told from prose; a page whose layout text lost words falls back to the
    plain text. Pages are joined with a single newline (a sentence may continue on the next page)."""
    pages: List[str] = []
    for page in reader.pages:
        plain = ""
        try:
            plain = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            pass
        laid = ""
        try:
            laid = page.extract_text(extraction_mode="layout") or ""
        except Exception:  # noqa: BLE001 - old pypdf or an odd page: the plain text will do
            laid = ""
        use = laid if laid.strip() and len(laid.split()) >= 0.9 * len(plain.split()) else plain
        if use.strip():
            pages.append(re.sub(r"\A(?:[ \t]*\n)+", "", use).rstrip())
    return "\n".join(pages).strip()
