"""Knowledge retrieval building blocks: embedding model config, a Hindi/English/Hinglish normaliser and tokenizer,
BM25, score fusion, question-row -> passage mapping, a retrieval confidence rating and a gated cross-encoder reranker.

Nothing heavy is imported at module level: the embedding model and the reranker are loaded on first use, so the
tokenizer, BM25 and fusion code run (and are unit-tested) without torch.

Pipeline (see rank_rows, LocalStore.match_chunks and backend.retrieve_context):
  dense top-20 ∪ BM25 top-20 (passages and question rows) = candidates
     -> scores: dense cosine, BM25, [BGE-M3 sparse + multi-vector, services/m3_modes.py, when its heads are present]
     -> weighted fusion of min-max normalised scores (RETRIEVAL_FUSION=rrf: the previous Reciprocal Rank Fusion)
     -> question rows stand for their parent passage (best score wins) -> assess_confidence
     -> [reranker: RERANKER=on always, auto only when uncertain] -> top k
"""
import logging
import math
import os
import platform
import re
import threading
import time
import unicodedata
from collections import Counter, OrderedDict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------ configuration

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"          # 1024-d, multilingual (100+ languages incl. Hindi), MIT licence
LEGACY_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384-d, the previous default

# Apache-2.0, 306M, multilingual; runs code from its repositories (pinned, see REMOTE_CODE_RERANKERS). Chosen over
# BAAI/bge-reranker-v2-m3 and Qwen/Qwen3-Reranker-0.6B in docs/RESULTS.md (19 Oct 2026): gated, on the top 3, it moves the
# right passage to first place on 5 more of 158 bench questions at 50 ms median / 59 ms p95 on an M4 GPU (bge: 6 more at
# 141 ms p95; Qwen3: 4 more at about 600 ms p95 for 4 passages).
DEFAULT_RERANKER_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"
RERANK_CANDIDATES = 20                           # candidates the cross-encoder scores before cutting to top k

RRF_K = 60


# Settings are read from the environment when used, not at import: backend.py loads .env after it imports this module.

def embedding_model_name() -> str:
    return (os.getenv("EMBEDDING_MODEL") or "").strip() or DEFAULT_EMBEDDING_MODEL


def embedding_device() -> str:
    return (os.getenv("EMBEDDING_DEVICE") or "cpu").strip()      # cpu | mps | cuda


def embedding_dtype() -> str:
    """"fp16" or "fp32": the precision the embedding model runs in (EMBEDDING_DTYPE=auto|fp16|fp32, default auto).

    fp16 halves BGE-M3's resident memory (RSS about 1 GB less) and runs a query about 2x faster on Apple Silicon, with
    identical retrieval hits (backend/bench/retrieval_eval.py); see fp16_cache_dir for how it is loaded. Other CPUs (x86 laptops) mostly lack fast fp16 matrix
    multiplication, so there fp16 would be slower than fp32: `auto` picks fp16 only on arm64 macOS, or on a GPU
    device (EMBEDDING_DEVICE=mps|cuda)."""
    choice = (os.getenv("EMBEDDING_DTYPE") or "auto").strip().lower()
    if choice in ("fp16", "float16", "half"):
        return "fp16"
    if choice in ("fp32", "float32", "full"):
        return "fp32"
    if embedding_device().split(":")[0] in ("mps", "cuda"):
        return "fp16"
    return "fp16" if platform.system() == "Darwin" and platform.machine() == "arm64" else "fp32"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def reranker_model_name() -> str:
    return (os.getenv("RERANKER_MODEL") or "").strip() or DEFAULT_RERANKER_MODEL


def hybrid_enabled() -> bool:
    return (os.getenv("HYBRID_SEARCH", "on") or "on").strip().lower() not in ("off", "0", "false", "no")


# ------------------------------------------------------------------------------------------------ shared embedder

_embedder_lock = threading.Lock()
_embedders: Dict[str, object] = {}
_half_models: set = set()          # model ids whose resident copy runs in float16
_bulk_lock = threading.Lock()      # one float32 bulk encode at a time (it holds a second copy of the weights)

BULK_WORDS = 200                   # texts this long (in total) are encoded in float32, see encode_texts
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def fp16_cache_dir(model_name: str) -> str:
    """Where the float16 copy of an embedding model is kept (backend/data/models, git-ignored): about 1.1 GB for BGE-M3."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip("/")) or "model"
    return os.path.join(os.getenv("EMBEDDING_FP16_DIR") or os.path.join(_DATA_DIR, "models"), safe + "-fp16")


_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def fp16_copy_ready(model_name: str) -> bool:
    """The float16 copy is usable only when it was finished AND its weights are still on disk (a cleanup that removes
    the large file but leaves the folder must not make every later load fail)."""
    path = fp16_cache_dir(model_name)
    return os.path.exists(os.path.join(path, ".complete")) and any(
        os.path.getsize(os.path.join(path, f)) > 0 for f in _WEIGHT_FILES if os.path.exists(os.path.join(path, f)))


def _st_kwargs(dtype) -> dict:
    return {"model_kwargs": {"dtype": dtype}}


class _DropRegexWarning(logging.Filter):
    """transformers warns about a "Mistral regex" for every tokenizer saved by the library itself, BGE-M3's float16 copy
    included. The warning is wrong here (the token ids are identical, checked on 46 texts) and noisy."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "incorrect regex pattern" not in record.getMessage()


def _new_sentence_transformer(source: str, dtype=None):
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - heavy, lazy
    logging.getLogger("transformers.tokenization_utils_base").addFilter(_DropRegexWarning())
    if dtype is None:
        return SentenceTransformer(source, device=embedding_device())
    try:
        return SentenceTransformer(source, device=embedding_device(), **_st_kwargs(dtype))
    except TypeError:       # older transformers: the argument is still called torch_dtype
        return SentenceTransformer(source, device=embedding_device(), model_kwargs={"torch_dtype": dtype})


def _fp16_copy(model_name: str):
    """The model as float16, loaded from a converted copy on disk when there is one (made on first use).

    Converting in memory (`.half()`, or loading with dtype=float16) leaves the weights in ordinary process memory:
    RSS drops by about 1 GB, but the process footprint macOS counts for memory pressure rises by 1.1 GB, because
    the float32 weights were memory-mapped from the checkpoint file (free to drop) and the float16 ones are not. A
    float16 checkpoint on disk is memory-mapped the same way: footprint unchanged, RSS about 1 GB lower, loads in 0.3 s
    instead of 8. Returns None when the copy cannot be made (read-only disk): the caller then keeps float32."""
    import torch  # noqa: PLC0415 - heavy, lazy
    path = fp16_cache_dir(model_name)
    if not fp16_copy_ready(model_name):
        try:
            import shutil  # noqa: PLC0415
            tmp = path + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            _new_sentence_transformer(model_name, torch.float16).save(tmp, safe_serialization=True)
            open(os.path.join(tmp, ".complete"), "w").close()
            shutil.rmtree(path, ignore_errors=True)
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001 - a full or read-only disk: float32 still works
            log.warning("could not write the float16 embedding model to %s (%s); using float32", path, e)
            return None
    try:
        return _new_sentence_transformer(path, torch.float16)
    except Exception as e:  # noqa: BLE001 - a damaged copy: drop it, float32 still works
        log.warning("the float16 embedding model at %s could not be loaded (%s); using float32", path, e)
        return None


def load_sentence_transformer(model_name: str):
    """One SentenceTransformer per model id for the whole process (BGE-M3 is ~2.2 GB in fp32: never load it twice).
    With EMBEDDING_DTYPE resolving to fp16 (see embedding_dtype) the resident copy is the float16 one."""
    with _embedder_lock:
        model = _embedders.get(model_name)
        if model is None:
            if embedding_dtype() == "fp16":
                model = _fp16_copy(model_name)
                if model is not None:
                    _half_models.add(model_name)
            if model is None:
                model = _new_sentence_transformer(model_name)
            try:
                cap = _int_env("EMBEDDING_MAX_SEQ_LENGTH", 512)   # BGE-M3 accepts 8192 tokens: cap it, cost grows with length
                if cap > 0:
                    model.max_seq_length = min(int(model.max_seq_length or cap), cap)
            except Exception:  # noqa: BLE001
                pass
            _embedders[model_name] = model
        return model


def _is_bulk(texts: Sequence[str]) -> bool:
    return sum(len(t.split()) for t in texts) >= BULK_WORDS


def _encode_bulk_fp32(model_name: str, texts: List[str], kwargs: dict):
    """Encode passage-sized text with a temporary float32 copy of the float16 weights. On a CPU, float16 is 2.5x faster
    than float32 for a question (a few tokens) but 4-7x SLOWER for a 120-word passage (measured on an M4: 770 ms against
    170 ms), so a document being ingested must not use the resident float16 model. The copy lives for the call."""
    import gc  # noqa: PLC0415
    import torch  # noqa: PLC0415 - heavy, lazy
    with _bulk_lock:
        # The float16 copy on disk is the fast source; if it has gone missing, read the original checkpoint instead.
        source = fp16_cache_dir(model_name) if fp16_copy_ready(model_name) else model_name
        model = _new_sentence_transformer(source, torch.float32)
        try:
            cap = _int_env("EMBEDDING_MAX_SEQ_LENGTH", 512)
            if cap > 0:
                model.max_seq_length = min(int(model.max_seq_length or cap), cap)
            return model.encode(texts, **kwargs)
        finally:
            del model
            gc.collect()


def encode_texts(model_name: str, texts: Sequence[str], **kwargs):
    """Unit-normalised float32 embeddings (cosine == dot product) from the shared model. A half-precision model is
    normalised again in float32, so the vectors stored and compared never depend on the model's precision."""
    normalise = bool(kwargs.pop("normalize_embeddings", True))
    kwargs.setdefault("batch_size", max(1, _int_env("EMBEDDING_BATCH_SIZE", 8)))
    kwargs.setdefault("show_progress_bar", False)
    texts = list(texts)
    model = load_sentence_transformer(model_name)
    half = model_name in _half_models
    kwargs["normalize_embeddings"] = False if half else normalise
    if half and embedding_device().split(":")[0] == "cpu" and _is_bulk(texts):
        out = _encode_bulk_fp32(model_name, texts, kwargs)
    else:
        out = model.encode(texts, **kwargs)
    if half and isinstance(out, np.ndarray):
        out = out.astype(np.float32, copy=False)
        if normalise:
            out = out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)
    return out


def embedder_loaded(model_name: str) -> bool:
    return model_name in _embedders


# ------------------------------------------------------------------------------------------------ tokenizer

# The ONE place where Hindi (Devanagari), Hinglish (Roman Hindi) and English text is normalised for matching (BM25, the
# BGE-M3 sparse weights, the confidence keyword overlap; other modules import `tokenize` / `normalise_text`).
#
# Every word ends up as a Roman "key": a Devanagari word is transliterated (कितना -> kitna), every Roman word is folded
# the same way (ee -> i, oo -> u, ph -> f, w -> v, z -> j, doubled letters collapsed, English plural -s dropped), and
# a curated table maps the frequent spelling variants of one word, in both scripts, to one key (kb/kab/कब, doctor/dr/
# dactor/डॉक्टर, fees/phees/फीस/शुल्क). For a handful of words a caller says in Hindi about an English document (days,
# fee, doctor, timings, medicine...) the key is the English word, so "ravivar ko" finds "Sunday". Numbers: any Indian
# script digits become ASCII, 45,000 == 45000, a phone number written in groups also gives its digits joined, and
# ₹ / Rs / rupees / रुपये are one word. Both sides (questions and passages) go through exactly this code.

# Devanagari: letters, vowel signs, virama (U+0900-U+097F) except the danda marks U+0964/U+0965 and the digits
# U+0966-U+096F (digits are mapped to ASCII below so "४००" and "400" match).
_DEVANAGARI = r"ऀ-ॣ॰-ॿ"
_TOKEN_RE = re.compile(rf"[{_DEVANAGARI}]+|[a-z]+|[0-9]+")
_DEVA_RE = re.compile(rf"[{_DEVANAGARI}]")
# Decimal digits of the Indian scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada,
# Malayalam) -> ASCII. Kept as _DEVA_DIGITS for older callers.
_INDIC_DIGIT_BASES = (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66)
_DEVA_DIGITS = {base + i: ord("0") + i for base in _INDIC_DIGIT_BASES for i in range(10)}
_ZERO_WIDTH_RE = re.compile("[​-‍⁠﻿­]")      # ZWJ/ZWNJ inside Devanagari words, soft hyphen
_LATIN_ACCENT_RE = re.compile("[À-ɏḀ-ỿ̀-ͯ]")
_COMBINING_RE = re.compile("[̀-ͯ]")                       # Latin combining accents only (not Devanagari signs)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d\d)")                       # 45,000 / 1,20,000 -> 45000 / 120000
_RUPEE_RE = re.compile("[₹₨]")                                   # ₹ and the old "Rs" sign
_PHONE_RE = re.compile(r"\d+(?:[ \-]\d+)+")                          # 020-5550-1299, 98200 12345

_NUKTA, _CHANDRABINDU, _ANUSVARA = "़", "ँ", "ं"
_DEVA_PRE = str.maketrans({"ॉ": "ो", "ऑ": "ओ",   # candra O (डॉक्टर, ऑफिस) -> plain o
                           "ॅ": "े", "ऍ": "ए"})   # candra E -> e


def _collapse_repeats(word: str) -> str:
    """'kitnaa' -> 'kitna', 'accha' -> 'acha', 'feee' -> 'fe': a run of the same letter becomes one letter."""
    out: List[str] = []
    for ch in word:
        if not out or out[-1] != ch or ch.isdigit():
            out.append(ch)
    return "".join(out)


_VOWELS = set("aeiou")
_DIGRAPHS = (("kh", "K"), ("gh", "G"), ("chh", "Q"), ("ch", "C"), ("jh", "J"), ("th", "T"), ("dh", "D"), ("ph", "P"),
             ("bh", "B"), ("sh", "S"))


@lru_cache(maxsize=50000)
def _transliterate_devanagari(word: str) -> str:
    """A Devanagari word as a Roman 'Hinglish-like' spelling (before folding), so Hinglish questions can match Hindi
    text and vice versa: long vowels are written like short ones (ई -> i, ऊ -> u, आ -> a), the silent word-final and
    interior schwa is dropped (समय -> samay, कितना -> kitna), a word-final nasal is dropped (नहीं -> nahi)."""
    deva = word.replace(_NUKTA, "").replace(_CHANDRABINDU, _ANUSVARA).translate(_DEVA_PRE)
    if len(deva) > 1 and deva.endswith(_ANUSVARA):
        deva = deva[:-1]                                  # nasalised final vowel: नहीं -> नही, हैं -> है, बच्चों -> बच्चो
    try:
        from indic_transliteration import sanscript  # noqa: PLC0415 - optional dependency
        roman = sanscript.transliterate(deva, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception:  # noqa: BLE001 - without the package the Devanagari token is used as it is
        return deva
    roman = roman.replace("j~n", "gy").replace("GY", "gy")            # ज्ञ: gyan
    roman = re.sub(r"[^A-Za-z]", "", roman)              # nasal marks (.N), stray signs, unconverted letters
    if roman.endswith("a") and len(roman) > 2:           # word-final inherent 'a' is silent in Hindi: bhojana -> bhojan
        roman = roman[:-1]
    roman = (roman.replace("RRi", "ri").replace("A", "a").replace("I", "i").replace("U", "u").replace("M", "n")
             .replace("x", "ksh").lower())
    # Interior schwa deletion: V C a C V -> V C C V (kitana -> kitna, baiThate -> baithte). Aspirated consonants and
    # sh are one sound written with two letters: they are folded to one placeholder letter while the rule runs.
    for dg, ph in _DIGRAPHS:
        roman = roman.replace(dg, ph)
    chars = list(roman)
    keep: List[str] = []
    for i, ch in enumerate(chars):
        if (ch == "a" and 2 <= i < len(chars) - 2 and chars[i - 1] not in _VOWELS and chars[i - 2] in _VOWELS
                and chars[i + 1] not in _VOWELS and chars[i + 2] in _VOWELS):
            continue
        keep.append(ch)
    roman = "".join(keep)
    for dg, ph in _DIGRAPHS:
        roman = roman.replace(ph, dg)
    return roman


def _fold_roman(word: str) -> str:
    """The spelling-insensitive key of one Roman word (English, Hinglish, or a transliterated Hindi word)."""
    if word.isdigit():
        return word
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]                                  # fees -> fee, doctors -> doctor
    word = (word.replace("ph", "f").replace("w", "v").replace("z", "j").replace("q", "k")
            .replace("ee", "i").replace("oo", "u"))
    word = re.sub(r"n(?=b|p)", "m", word)                # sanbandh / sambandh, chanpa / champa
    return _collapse_repeats(word)


# Spelling variants -> one key. Canonical word first (written naturally; folded when the table is built), then its
# variants in Roman and Devanagari. Where the canonical is English, a Hindi or Hinglish word finds the English
# document word. Keep entries unambiguous: a variant must not also be a common, different English or Hindi word.
_VARIANTS: Dict[str, Sequence[str]] = {
    # question and function words (mostly stopwords, unified so the stopword list catches every spelling)
    "hai": ("he", "h", "hain", "hein", "hay", "hy", "hei", "hae", "है", "हैं"),
    "nahi": ("nahin", "nai", "nhi", "nahe", "nahee", "nehi", "nahi", "नहीं", "नही", "ना"),
    "kya": ("kia", "kyaa", "kiya", "क्या", "किया"),
    "kyon": ("kyo", "kyu", "kyun", "kyoon", "क्यों", "क्यूं", "क्यूँ"),
    "kab": ("kb", "kabb", "कब"),
    "kitna": ("kitni", "kitne", "kitnaa", "kitana", "ktna", "kitney", "kitnee", "कितना", "कितनी", "कितने"),
    "kaise": ("kese", "kaisa", "kaisi", "kaisay", "kayse", "kaise", "कैसे", "कैसा", "कैसी"),
    "kahan": ("kaha", "kahaan", "kahaa", "kidhar", "किधर", "कहाँ", "कहां", "कहा"),
    "kaun": ("kon", "koun", "kaunsa", "konsa", "kaunsi", "konsi", "kaunse", "konse", "कौन", "कौनसा", "कौनसी", "कौनसे"),
    "ka": ("ke", "ki", "ko", "का", "की", "के", "को"),
    "mein": ("men", "me", "mai", "mei", "में", "मैं"),
    "aap": ("ap", "aapka", "aapki", "aapke", "apka", "apki", "apke", "आप", "आपका", "आपकी", "आपके"),
    "mujhe": ("muje", "mujhey", "mujhko", "mereko", "mujhe", "मुझे", "मुझको"),
    "chahiye": ("chaiye", "chahie", "chahiyee", "chahye", "chahiya", "चाहिए", "चाहिये"),
    "bataiye": ("batao", "bataye", "batayein", "batayen", "bataiyee", "batana", "बताइए", "बताइये", "बताओ", "बताएं",
                "बताये", "बताना"),
    "hota": ("hoti", "hote", "hoga", "hogi", "honge", "hona", "होता", "होती", "होते", "होगा", "होगी", "होंगे", "होना"),
    "hun": ("hoon", "hu", "hoo", "हूँ", "हूं", "हू"),
    "baje": ("bje", "बजे"),
    # money
    "rupee": ("rupees", "rs", "inr", "rupaye", "rupye", "rupay", "rupaya", "rupiya", "rupiye", "rupe", "रुपये", "रुपए",
              "रुपया", "रुपयों", "रु"),
    "money": ("paisa", "paise", "paisha", "pese", "पैसा", "पैसे"),
    "fee": ("fees", "phees", "phee", "fis", "shulk", "फीस", "फ़ीस", "शुल्क"),
    "price": ("keemat", "kimat", "daam", "mulya", "कीमत", "क़ीमत", "दाम", "मूल्य"),
    "free": ("muft", "nishulk", "मुफ्त", "मुफ़्त", "निःशुल्क", "निशुल्क"),
    "discount": ("chhoot", "chhut", "छूट"),
    "payment": ("bhugtan", "भुगतान"),
    # people, places, services (English loanwords first: their Devanagari spelling transliterates differently)
    "doctor": ("dr", "doktar", "daktar", "dactor", "dactar", "docter", "doktor", "daktor", "doctr", "doctoron",
               "डॉक्टर", "डाक्टर", "डॉक्टरों", "डाक्टरों"),
    "clinic": ("klinik", "clinik", "kilinik", "क्लिनिक", "क्लीनिक", "davakhana", "दवाखाना"),
    "hospital": ("aspatal", "haspatal", "hospitl", "aspataal", "अस्पताल", "हॉस्पिटल", "हास्पिटल"),
    "medicine": ("dawai", "davai", "dava", "dawa", "davaiyan", "dawaiyan", "aushadhi", "दवा", "दवाई", "दवाइयां",
                 "दवाइयाँ", "औषधि"),
    "pharmacy": ("chemist", "farmesi", "pharmesi", "फार्मेसी", "फ़ार्मेसी", "केमिस्ट"),
    "appointment": ("apointment", "appoinment", "apoinment", "appointmant", "अपॉइंटमेंट", "अपोइंटमेंट", "अपाइंटमेंट"),
    "emergency": ("imarjensi", "imrjensi", "emarjensi", "emergancy", "emarjency", "इमरजेंसी", "इमर्जेंसी",
                  "aapatkal", "आपातकाल"),
    "ambulance": ("embulens", "ambulans", "embulance", "एम्बुलेंस", "एंबुलेंस"),
    "dentist": ("dental", "teeth", "tooth", "dant", "daant", "दांत", "दाँत", "डेंटिस्ट", "डेंटल"),
    "test": ("janch", "jaanch", "जांच", "जाँच", "टेस्ट"),
    "report": ("riport", "रिपोर्ट"),
    "blood": ("khoon", "khun", "rakt", "खून", "ख़ून", "रक्त"),
    "eye": ("aankh", "ankh", "aankhon", "आंख", "आँख", "आंखों", "आँखों"),
    "child": ("children", "kid", "kids", "bachcha", "bachche", "bachchon", "bacche", "bachhe", "baccha", "bachon",
              "बच्चा", "बच्चे", "बच्चों", "बच्ची"),
    "lab": ("laib", "लैब", "prayogshala", "प्रयोगशाला"),
    "college": ("kolej", "kalej", "कॉलेज", "कालेज", "mahavidyalay", "महाविद्यालय"),
    "course": ("kors", "कोर्स", "pathyakram", "पाठ्यक्रम"),
    "hostel": ("hostal", "हॉस्टल", "होस्टल", "chhatravas", "छात्रावास"),
    "admission": ("admisan", "dakhila", "daakhila", "pravesh", "प्रवेश", "दाखिला", "एडमिशन"),
    "exam": ("exams", "examination", "pariksha", "imtihan", "imtehaan", "परीक्षा", "इम्तिहान"),
    "result": ("parinam", "natija", "nateeja", "परिणाम", "नतीजा", "रिजल्ट"),
    "scholarship": ("scolarship", "skolarship", "chhatravritti", "छात्रवृत्ति", "स्कॉलरशिप"),
    "office": ("ofis", "daftar", "karyalay", "दफ्तर", "दफ़्तर", "कार्यालय", "ऑफिस", "ऑफ़िस"),
    "meeting": ("baithak", "mulakat", "mulaqat", "बैठक", "मुलाकात", "मुलाक़ात", "मीटिंग"),
    "holiday": ("chhutti", "chutti", "छुट्टी"),
    "document": ("dastavez", "dastavej", "दस्तावेज़", "दस्तावेज"),
    "phone": ("fon", "फोन", "फ़ोन"),
    "number": ("nambar", "numbar", "nanbar", "नंबर", "नम्बर"),
    "name": ("naam", "नाम"),
    "available": ("milega", "milegi", "milenge", "milta", "milti", "milte", "uplabdh", "उपलब्ध", "मिलेगा", "मिलेगी",
                  "मिलेंगे", "मिलता", "मिलती", "मिलते"),
    "open": ("khula", "khuli", "khule", "khulta", "khulti", "khulte", "khulna", "khulega", "khulegi", "khulenge",
             "opan", "खुला", "खुली", "खुले", "खुलता", "खुलती", "खुलते", "खुलना", "खुलेगा", "खुलेगी", "ओपन"),
    "closed": ("band", "बंद"),
    "cancel": ("cancellation", "kainsil", "radd", "रद्द", "कैंसिल"),
    # time
    "time": ("timing", "timings", "samay", "samai", "taim", "tym", "टाइम", "समय", "टाइमिंग"),
    "hour": ("ghanta", "ghante", "ghanton", "घंटा", "घंटे", "घंटों"),
    "today": ("aaj", "आज"),
    "now": ("abhi", "अभी"),
    "day": ("din", "dino", "dinon", "दिन", "दिनों"),
    "week": ("hafta", "hafte", "saptah", "हफ्ता", "हफ़्ता", "हफ्ते", "हफ़्ते", "सप्ताह"),
    "month": ("mahina", "mahine", "mahino", "महीना", "महीने", "महीनों"),
    "year": ("saal", "varsh", "साल", "वर्ष"),
    "morning": ("subah", "subha", "सुबह"),
    "evening": ("shaam", "sham", "शाम"),
    "night": ("raat", "रात"),
    "afternoon": ("dopahar", "dopehar", "दोपहर"),
    "sunday": ("ravivar", "raviwar", "itvar", "itwar", "इतवार", "रविवार"),
    "monday": ("somvar", "somwar", "सोमवार"),
    "tuesday": ("mangalvar", "mangalwar", "मंगलवार"),
    "wednesday": ("budhvar", "budhwar", "बुधवार"),
    "thursday": ("guruvar", "guruwar", "brihaspativar", "गुरुवार", "बृहस्पतिवार"),
    "friday": ("shukravar", "shukrawar", "शुक्रवार"),
    "saturday": ("shanivar", "shaniwar", "शनिवार"),
}


def _raw_key(word: str) -> str:
    """Script conversion + folding, without the variant table."""
    if _DEVA_RE.search(word):
        word = _transliterate_devanagari(word)
        if _DEVA_RE.search(word):                         # indic_transliteration missing: keep the Devanagari word
            return word.replace(_NUKTA, "").replace(_CHANDRABINDU, _ANUSVARA)
    return _fold_roman(word)


@lru_cache(maxsize=1)
def _variant_table() -> Dict[str, str]:
    """folded variant -> folded canonical. Built once (lazily: it transliterates the Devanagari entries)."""
    table: Dict[str, str] = {}
    for canonical, variants in _VARIANTS.items():
        canon = _raw_key(canonical.lower())
        for v in (canonical, *variants):
            key = _raw_key(v.lower())
            if key and key != canon:
                table.setdefault(key, canon)              # the first entry wins a clash (tests check there is none)
    return table


@lru_cache(maxsize=100000)
def normalise_token(token: str) -> str:
    """Canonical key of one lowercase token (Latin, digits or Devanagari): the same key for the spelling variants of a
    word in both scripts (kitni / kitne / कितनी -> kitna)."""
    if not token:
        return ""
    if token.isdigit():
        return token
    key = _raw_key(token)
    return _variant_table().get(key, key)


def _prepare(text: str) -> str:
    """Unicode clean-up shared by every tokenizer call: NFKC, zero-width characters out, Latin accents stripped
    (Pāṇḍey -> Pandey), Indian-script digits to ASCII, thousands separators out, ₹ -> 'rupee', lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    if _LATIN_ACCENT_RE.search(text):
        text = unicodedata.normalize("NFC", _COMBINING_RE.sub("", unicodedata.normalize("NFD", text)))
    text = text.translate(_DEVA_DIGITS)
    text = _THOUSANDS_RE.sub("", text)
    text = _RUPEE_RE.sub(" rupee ", text)
    return text.lower()


# Function words dropped before BM25 (both scripts). They carry no topic and, in short chunks, only add noise.
_STOPWORDS = frozenset("""
a an the of in on at to for and or is are was were be been it this that these those with by from as i me my we you your
do does did can could will would should what when where which who how why not no yes please tell about there any
hai ka ka mein se par ko ne bhi aur ya to toh ye yeh wo vo kya kab kaise kitna kyo koi kaun kahan aap mujhe chahiye
bataiye hota hun hum ham tha thi the raha rahi rahe sakta sakti sakte
है हैं था थे थी का की के को में से पर ने भी और या तो यह ये वह वो क्या कब कैसे कितना कितने कितनी क्यों कोई एक
कौन कहाँ कहां आप मुझे चाहिए बताइए होता हूँ हम मैं रहा रही रहे सकता सकती सकते
""".split())


@lru_cache(maxsize=1)
def _stopwords() -> frozenset:
    """The stopword list in normalised form (so 'है' and 'hai', 'का' and 'ka' are the same stopword)."""
    return frozenset(normalise_token(w) for w in _STOPWORDS)


def _phone_joins(text: str) -> List[str]:
    """A number written in groups ('020-5550-1299', '98200 12345') also as one token, as a caller may say or type it
    without the separators."""
    out = []
    for m in _PHONE_RE.finditer(text):
        joined = re.sub(r"[ \-]", "", m.group(0))
        if len(joined) >= 7:
            out.append(joined)
    return out


@lru_cache(maxsize=20000)
def _tokenize_cached(text: str, drop_stopwords: bool) -> tuple:
    text = _prepare(text)
    tokens = [normalise_token(t) for t in _TOKEN_RE.findall(text)]
    tokens = [t for t in tokens if t] + _phone_joins(text)
    if drop_stopwords:
        stop = _stopwords()
        kept = [t for t in tokens if t not in stop]
        return tuple(kept or tokens)    # a question made only of function words keeps them
    return tuple(tokens)


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    """Normalised tokens of a text (see the notes above this section): punctuation removed, digits of Indian scripts
    mapped to ASCII, Devanagari words transliterated to Roman, Roman words folded, spelling variants in both scripts
    unified. Stopwords (both scripts) are dropped unless that would leave nothing."""
    return list(_tokenize_cached(text or "", drop_stopwords))


def content_tokens(text: str) -> List[str]:
    """Normalised tokens without stopwords, even when that leaves nothing (for keyword-overlap checks)."""
    stop = _stopwords()
    return [t for t in _tokenize_cached(text or "", False) if t not in stop]


def normalise_text(text: str) -> str:
    """The whole text as space-separated normalised keys (stopwords kept): two texts that differ only in script,
    Hinglish spelling, digits or ₹/Rs give the same string. For other modules that compare spoken text (contract 5)."""
    return " ".join(_tokenize_cached(text or "", False))


# ------------------------------------------------------------------------------------------------ BM25

class BM25:
    """Okapi BM25 (k1=1.5, b=0.75) over pre-tokenised documents. Pure Python, no dependencies."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.n = len(docs)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        self.idf = {t: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: Iterable[str]) -> List[float]:
        q = [t for t in dict.fromkeys(query_tokens) if t in self.idf]
        out = [0.0] * self.n
        if not q or not self.avgdl:
            return out
        for i, tf in enumerate(self.docs):
            norm = self.k1 * (1.0 - self.b + self.b * self.lengths[i] / self.avgdl)
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1.0) / (f + norm)
            out[i] = s
        return out


_bm25_cache: "OrderedDict[tuple, BM25]" = OrderedDict()
_bm25_lock = threading.Lock()
_BM25_CACHE_SIZE = 16


def bm25_index(texts: Sequence[str]) -> BM25:
    """BM25 over `texts`, built once per distinct list of texts (an agent's passages are the same on every turn, so the
    index is not rebuilt per question; a changed document gives a new list and a new index)."""
    key = tuple(texts)
    with _bm25_lock:
        bm = _bm25_cache.get(key)
        if bm is not None:
            _bm25_cache.move_to_end(key)
            return bm
    bm = BM25([tokenize(t) for t in key])
    with _bm25_lock:
        _bm25_cache[key] = bm
        while len(_bm25_cache) > _BM25_CACHE_SIZE:
            _bm25_cache.popitem(last=False)
    return bm


def bm25_ranking(query: str, texts: Sequence[str]) -> List[int]:
    """Indices of `texts` that share at least one token with the query, best first (empty when nothing matches)."""
    scores = bm25_index(texts).scores(tokenize(query))
    order = [i for i in sorted(range(len(texts)), key=lambda i: -scores[i]) if scores[i] > 0]
    return order


# ------------------------------------------------------------------------------------------------ fusion

def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> List[int]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)), ranks starting at 1. Returns the ids
    best first; ties keep the order of first appearance (so the first ranking, the dense one, wins a tie)."""
    scores = rrf_scores(rankings, k)
    first_seen: Dict[int, int] = {}
    for ranking in rankings:
        for doc in ranking:
            first_seen.setdefault(doc, len(first_seen))
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))


def rrf_scores(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return scores


def dense_ranking(similarities: Sequence[float]) -> List[int]:
    return sorted(range(len(similarities)), key=lambda i: -similarities[i])


# Weighted score fusion (Bruch, Gai & Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval", TOIS 2023: a
# convex combination of normalised scores beats RRF in and out of domain, and needs only a small labelled set to
# tune). Each signal is scaled to 0..1 over the candidates (theoretical min-max: score / best score, the paper's
# TM2C2; RETRIEVAL_NORM=minmax for observed min-max), then fused = sum(w_s * norm_s) / sum(active w_s).
# Defaults (bench/retrieval_modes_bench.py, docs/RESULTS.md "Retrieval and answer-checker model choice, 19 Oct 2026"):
# dense 0.4, BGE-M3 multi-vector (ColBERT) 0.3, BM25 0.1, BGE-M3 sparse 0. On the two bench sets (45 + 113 answerable
# questions) this beats dense + BM25 on both: hit@1 36 -> 36 and 88 -> 91, MRR 0.877 -> 0.880 and 0.918 -> 0.932;
# the grid around it (dense 0.35-0.45, ColBERT 0.25-0.4, BM25 0.1-0.15) is within one question everywhere, so it is a
# plateau, not a lucky point. The sparse weights add nothing over BM25 on the Hinglish/Devanagari normaliser (0.1 cost
# one question on the 45-question set, 0.05 gained one on the 113-question set): 0, kept selectable. BM25 stays for exact
# numbers, names and Hinglish spellings. Without the M3 heads (or before an agent's passage vectors exist) the weights
# renormalise to dense 0.8 : BM25 0.2, the best point of the BM25-weight grid on both sets (MRR 0.877 / 0.918, against
# 0.862 / 0.908 for the previous RRF). Small sets: about +-3 questions of noise.
DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {"dense": 0.4, "sparse": 0.0, "colbert": 0.3, "bm25": 0.1}
SIGNALS = ("dense", "sparse", "colbert", "bm25")
CANDIDATES_PER_SIGNAL = 20        # candidates = union of the dense top-N and the BM25 top-N


def fusion_mode() -> str:
    """RETRIEVAL_FUSION=weighted (default) | rrf (the previous Reciprocal Rank Fusion, k=60)."""
    mode = (os.getenv("RETRIEVAL_FUSION") or "weighted").strip().lower()
    return mode if mode in ("weighted", "rrf") else "weighted"


def fusion_weights() -> Dict[str, float]:
    """RETRIEVAL_WEIGHTS="dense:0.4,sparse:0,colbert:0.3,bm25:0.1" (any subset; missing ones keep the default,
    0 switches a signal off). Malformed entries are ignored."""
    weights = dict(DEFAULT_FUSION_WEIGHTS)
    for part in (os.getenv("RETRIEVAL_WEIGHTS") or "").split(","):
        name, _, value = part.partition(":")
        name = name.strip().lower()
        try:
            if name in weights:
                weights[name] = max(0.0, float(value))
        except ValueError:
            continue
    return weights


def candidates_per_signal() -> int:
    return max(1, _int_env("RETRIEVAL_CANDIDATES", CANDIDATES_PER_SIGNAL))


def minmax(values: Sequence[Optional[float]]) -> List[float]:
    """Min-max normalisation to 0..1; a missing value counts as 0; all-equal present values become 1."""
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values)
    lo, hi = min(present), max(present)
    if hi - lo < 1e-12:
        return [1.0 if v is not None else 0.0 for v in values]
    return [((v - lo) / (hi - lo)) if v is not None else 0.0 for v in values]


def tmm(values: Sequence[Optional[float]], floor: float = 0.0) -> List[float]:
    """Theoretical-min-max normalisation (Bruch et al.'s TM2C2): (v - floor) / (max - floor), with a fixed lower bound
    instead of the observed minimum, so a small difference between two close candidates is not stretched to 0 vs 1.
    Every score fused here is >= 0 for a relevant text (cosine, BM25, M3 sparse, MaxSim), so the floor is 0."""
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values)
    hi = max(present)
    if hi - floor < 1e-12:
        return [0.0] * len(values)
    return [max(0.0, (v - floor) / (hi - floor)) if v is not None else 0.0 for v in values]


def score_norm() -> str:
    """RETRIEVAL_NORM=tmm (default) | minmax: how each signal is scaled to 0..1 before the weighted sum."""
    mode = (os.getenv("RETRIEVAL_NORM") or "tmm").strip().lower()
    return mode if mode in ("tmm", "minmax") else "tmm"


def weighted_fuse(signals: Dict[str, Sequence[Optional[float]]], weights: Dict[str, float]) -> List[float]:
    """Convex combination of normalised signals (RETRIEVAL_NORM: tmm by default, or minmax). A signal with no value at all, or weight 0, is left out and
    the remaining weights are renormalised, so the result is always in 0..1."""
    n = max((len(v) for v in signals.values()), default=0)
    active = [s for s, vals in signals.items() if weights.get(s, 0.0) > 0 and any(v is not None for v in vals)]
    total = sum(weights[s] for s in active)
    out = [0.0] * n
    if not active or total <= 0:
        return out
    norm = minmax if score_norm() == "minmax" else tmm
    for s in active:
        w = weights[s] / total
        for i, v in enumerate(norm(signals[s])):
            out[i] += w * v
    return out


# -- passage rows: question-index rows (contract 3) and headers (contract 4)

def row_text(row: dict) -> str:
    return str(row.get("chunk_text") or row.get("content") or "")


def passage_text(row: dict) -> str:
    """The text of a passage as it is matched and shown: `header + "\\n" + text` when the row has a header."""
    header = str(row.get("header") or "").strip()
    text = row_text(row)
    return f"{header}\n{text}" if header else text


def is_question_row(row: dict) -> bool:
    return str(row.get("kind") or "passage").lower() == "question"


def _parents(rows: Sequence[dict]) -> List[Optional[int]]:
    """For each row, the index of the passage it stands for: itself for a passage, its parent passage for a question
    row (None when the parent is not among the rows: such a question can never be returned)."""
    by_id = {r.get("id"): i for i, r in enumerate(rows) if not is_question_row(r) and r.get("id") is not None}
    return [by_id.get(r.get("parent_chunk_id")) if is_question_row(r) else i for i, r in enumerate(rows)]


def rank_rows(query: str, rows: Sequence[dict], similarities: Sequence[Optional[float]], *, query_m3=None,
              info: Optional[dict] = None) -> List[dict]:
    """Rank knowledge rows for a question. Returns one entry per PASSAGE, best first:
    {"index": i (into rows), "score": fused 0..1, "similarity": best dense cosine of the passage or its questions,
     "signals": {"dense", "bm25", "sparse", "colbert"} of its best-scoring row, "via": "passage" | "question"}.

    `similarities[i]` is the dense cosine of row i, or None when it has no comparable embedding (then it takes part
    through BM25 only, so vectors of different dimensions are never compared). Question rows (kind="question") are
    scored like passages but stand for their parent passage (best score wins); they are never returned themselves.
    `query_m3` (services.m3_modes.QueryM3) adds BGE-M3's sparse and multi-vector scores when every candidate has them.
    HYBRID_SEARCH=off: dense order only. RETRIEVAL_FUSION=rrf: the previous Reciprocal Rank Fusion (k=60) of the full
    dense and BM25 rankings (plus the M3 rankings of the candidates when present). `info`, when given, is filled with
    what the fusion did (signals used, agreement of the dense and BM25 tops) for the confidence assessment."""
    n = len(rows)
    info = info if info is not None else {}
    if not n:
        info.update({"fusion": None, "signals": [], "n_candidates": 0})
        return []
    parent = _parents(rows)
    dense_order = sorted((i for i in range(n) if similarities[i] is not None and parent[i] is not None),
                         key=lambda i: -similarities[i])  # type: ignore[operator]

    def best_dense(p_rows: List[int]) -> Optional[float]:
        vals = [similarities[i] for i in p_rows if similarities[i] is not None]
        return max(vals) if vals else None  # type: ignore[type-var]

    members: Dict[int, List[int]] = {}
    for i, p in enumerate(parent):
        if p is not None:
            members.setdefault(p, []).append(i)

    if not hybrid_enabled() or not (query or "").strip():
        out, seen = [], set()
        for i in dense_order:
            p = parent[i]
            if p in seen:
                continue
            seen.add(p)
            out.append({"index": p, "score": max(0.0, float(similarities[i])), "similarity": best_dense(members[p]),  # type: ignore[arg-type]
                        "signals": {"dense": similarities[i]}, "via": "question" if p != i else "passage"})
        info.update({"fusion": "dense", "signals": ["dense"], "n_candidates": len(out), "agree": None})
        return out

    texts = [passage_text(r) if not is_question_row(r) else row_text(r) for r in rows]
    bm = bm25_index(texts).scores(tokenize(query))
    bm_order = sorted((i for i in range(n) if bm[i] > 0 and parent[i] is not None), key=lambda i: -bm[i])
    dense_top = parent[dense_order[0]] if dense_order else None
    bm_top = parent[bm_order[0]] if bm_order else None
    info.update({"dense_top": dense_top, "bm25_top": bm_top,
                 "agree": (dense_top == bm_top) if (dense_top is not None and bm_top is not None) else None})

    cap = candidates_per_signal()
    cand_parents: List[int] = []
    for i in dense_order[:cap] + bm_order[:cap]:
        if parent[i] not in cand_parents:
            cand_parents.append(parent[i])  # type: ignore[arg-type]
    cand_rows = [i for p in cand_parents for i in members[p]]

    signals: Dict[str, List[Optional[float]]] = {"dense": [similarities[i] for i in cand_rows],
                                                 "bm25": [bm[i] for i in cand_rows]}
    m3_used = False
    if query_m3 is not None and cand_rows:
        try:
            m3 = query_m3.score_rows([rows[i] for i in cand_rows], [texts[i] for i in cand_rows], all_rows=rows,
                                     all_texts=texts)
        except Exception as e:  # noqa: BLE001 - never lose retrieval because the optional M3 scores failed
            log.warning("BGE-M3 sparse/multi-vector scoring failed: %s", e)
            m3 = None
        if m3:
            signals.update(m3)
            m3_used = True

    weights = fusion_weights()
    mode = fusion_mode()
    if mode == "rrf":
        # the previous behaviour, exactly, when only dense and BM25 are present: RRF of the full rankings
        rankings = [dense_order, bm_order]
        for s in ("sparse", "colbert"):
            if s in signals:
                vals = signals[s]
                rankings.append([cand_rows[j] for j in sorted(range(len(cand_rows)), key=lambda j: -(vals[j] or 0.0))])
        rr = rrf_scores(rankings)
        max_possible = len(rankings) / (RRF_K + 1)
        ordered = rrf_fuse(rankings)
        row_score = {i: rr[i] / max_possible for i in ordered}
        scored_rows = ordered
    else:
        fused = weighted_fuse(signals, weights)
        row_score = {i: fused[j] for j, i in enumerate(cand_rows)}
        scored_rows = cand_rows

    best: Dict[int, int] = {}            # parent -> its best-scoring row
    for i in scored_rows:
        p = parent[i]
        if p is None:
            continue
        if p not in best or row_score[i] > row_score[best[p]]:
            best[p] = i
    first_seen = {p: k for k, p in enumerate(dict.fromkeys(parent[i] for i in scored_rows if parent[i] is not None))}
    ranked = sorted(best, key=lambda p: (-row_score[best[p]], -(best_dense(members[p]) or -2.0), first_seen[p]))

    def entry(p: int, score: float) -> dict:
        i = best.get(p, p)
        sig = {"dense": similarities[i], "bm25": round(bm[i], 4)}
        if i in cand_rows:
            j = cand_rows.index(i)
            for s in ("sparse", "colbert"):
                if s in signals and signals[s][j] is not None:
                    sig[s] = round(float(signals[s][j]), 4)  # type: ignore[arg-type]
        return {"index": p, "score": round(float(score), 4), "similarity": best_dense(members[p]), "signals": sig,
                "via": "question" if i != p else "passage"}

    out = [entry(p, row_score[best[p]]) for p in ranked]
    # Passages outside the candidates follow (a caller may ask for the full ranking): dense order, then the rest.
    seen = set(ranked)
    for i in dense_order:
        p = parent[i]
        if p not in seen:
            seen.add(p)
            out.append(entry(p, 0.0))
    for p in sorted(members):
        if p not in seen and mode == "weighted" and any(bm[i] > 0 for i in members[p]):
            seen.add(p)
            out.append(entry(p, 0.0))
    used = [s for s in SIGNALS if s in signals and (mode == "rrf" or weights.get(s, 0) > 0)
            and any(v is not None for v in signals[s])]
    info.update({"fusion": mode, "signals": used, "m3": m3_used, "n_candidates": len(cand_parents)})
    return out


def hybrid_order(query: str, texts: Sequence[str], similarities: Sequence[Optional[float]]) -> List[int]:
    """Fused order of `texts`. `similarities[i]` is the dense cosine of chunk i, or None when that chunk has no usable
    embedding (different dimension): such a chunk takes part through BM25 only, so mixed dimensions are never
    compared. Same fusion as rank_rows (RETRIEVAL_FUSION, RETRIEVAL_WEIGHTS)."""
    return [e["index"] for e in rank_rows(query, [{"chunk_text": t} for t in texts], similarities)]


# ------------------------------------------------------------------------------------------------ confidence

# How sure retrieval is that the passages answer the question (contract 2, read by services/grounding.py). Inputs:
# the dense cosine of the best passage (BGE-M3: unrelated text ~0.3-0.45, a passage that answers ~0.55-0.8), the
# fused top score and the gap to the second passage, the share of the question's content words found in the best
# passage (after the Hinglish/Devanagari normaliser), and whether the dense and BM25 rankings agree on the best
# passage. Calibrated on bench/retrieval_modes_bench.py (158 answerable + 20 unanswerable questions, 2026-10-19): the
# unanswerable ones there are topical near-misses whose features overlap the answerable ones (top dense 0.37-0.56
# against 0.27-0.77), so "low" is kept for clear cases only (it fired on no question there) and "high" (dense >= 0.58
# plus agreement, a clear gap or keyword overlap) on 46 of 158 answerable and none of the unanswerable. Every
# threshold has an env override (RETRIEVAL_CONF_DENSE_HIGH=...).
CONF_DENSE_HIGH = 0.58      # best passage this close in meaning: high, when another signal agrees
CONF_DENSE_LOW = 0.40       # below this the passage is probably about something else
CONF_DENSE_NONE = 0.30      # below this, with no shared keyword and no BM25 match at all: nothing relevant
CONF_GAP_CLEAR = 0.15       # fused top1 - top2: a clear winner
CONF_OVERLAP_HIGH = 0.5     # half the question's content words are in the best passage
CONF_OVERLAP_LOW = 0.15
CONF_RERANK_HIGH = 0.7      # cross-encoder probability (sigmoid) of the reranked top passage
# 0 = off. A reranker score must not make a search "low" (the grounding gate then refuses without asking the model):
# bge-reranker-v2-m3 gives answerable Hinglish questions a median of 0.001 (RESULTS.md, 19 Oct 2026), so 0.05 rated 74
# of 158 answerable bench questions "low"; gte's scores sit between 0.3 and 0.55 for every question. Opt in per model.
CONF_RERANK_LOW = 0.0


def _conf_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(f"RETRIEVAL_{name}") or default)
    except ValueError:
        return default


def confidence_thresholds() -> Dict[str, float]:
    """The thresholds in force (RETRIEVAL_CONF_DENSE_HIGH=0.62 etc. override the constants above)."""
    names = ("CONF_DENSE_HIGH", "CONF_DENSE_LOW", "CONF_DENSE_NONE", "CONF_GAP_CLEAR", "CONF_OVERLAP_HIGH",
             "CONF_OVERLAP_LOW", "CONF_RERANK_HIGH", "CONF_RERANK_LOW")
    return {n.lower(): _conf_env(n, globals()[n]) for n in names}


def keyword_overlap(query: str, passage: str) -> Optional[float]:
    """Share of the question's content words (normalised, stopwords out) that occur in the passage; None when the
    question has no content word."""
    q = set(content_tokens(query))
    if not q:
        return None
    return len(q & set(content_tokens(passage))) / len(q)


def assess_confidence(chunks: Sequence[Any], query: str, info: Optional[dict] = None) -> dict:
    """{"level": "high"|"medium"|"low"|"none", "top": fused score of the best passage, "gap": top1 - top2,
    "reason": short text, plus "dense", "overlap", "agree" (and "rerank" when the reranker ran)}.

    `chunks` are the ranked passage rows (dicts with `score`, `similarity`, `chunk_text`/`header`, optional
    `rerank_score`); plain strings are accepted (only the keyword overlap is then known). `info` is what rank_rows
    reported (dense/BM25 agreement). Never raises."""
    t = confidence_thresholds()
    rows = [c if isinstance(c, dict) else {"chunk_text": str(c)} for c in (chunks or [])]
    if not rows:
        return {"level": "none", "top": 0.0, "gap": 0.0, "reason": "no passages", "dense": None, "overlap": None,
                "agree": None}
    try:
        scores = [r.get("score") for r in rows]
        top = float(scores[0]) if isinstance(scores[0], (int, float)) else None
        second = float(scores[1]) if len(scores) > 1 and isinstance(scores[1], (int, float)) else 0.0
        gap = (top - second) if top is not None else 0.0
        dense = rows[0].get("similarity")
        dense = float(dense) if isinstance(dense, (int, float)) else None
        overlap = keyword_overlap(query, passage_text(rows[0]))
        agree = (info or {}).get("agree")
        rr = rows[0].get("rerank_score")
        rr = float(rr) if isinstance(rr, (int, float)) else None
        ov = overlap if overlap is not None else 0.0
        reasons = []
        if dense is None and top is None:
            level = "high" if ov >= t["conf_overlap_high"] else "medium" if ov > 0 else "low"
            reasons.append(f"keyword overlap {ov:.2f} only")
        elif dense is not None and dense < t["conf_dense_none"] and ov == 0 and agree is None:
            level = "none"
            reasons.append(f"dense {dense:.2f} < {t['conf_dense_none']}, no shared keyword, no BM25 match")
        elif rr is not None and rr < t["conf_rerank_low"]:
            level = "low"
            reasons.append(f"reranker {rr:.2f} < {t['conf_rerank_low']}")
        elif dense is not None and dense < t["conf_dense_low"] and ov < t["conf_overlap_low"] and not agree:
            level = "low"
            reasons.append(f"dense {dense:.2f} < {t['conf_dense_low']}, overlap {ov:.2f}")
        elif dense is not None and dense >= t["conf_dense_high"] and (
                agree or gap >= t["conf_gap_clear"] or ov >= t["conf_overlap_high"]
                or (rr is not None and rr >= t["conf_rerank_high"])):
            level = "high"
            reasons.append(f"dense {dense:.2f}" + (", dense and BM25 agree" if agree else "")
                           + (f", gap {gap:.2f}" if gap >= t["conf_gap_clear"] else "")
                           + (f", overlap {ov:.2f}" if ov >= t["conf_overlap_high"] else ""))
        else:
            level = "medium"
            reasons.append(f"dense {dense:.2f}" if dense is not None else "no dense score")
            reasons.append(f"gap {gap:.2f}, overlap {ov:.2f}")
        out = {"level": level, "top": round(top, 4) if top is not None else (round(dense, 4) if dense else 0.0),
               "gap": round(gap, 4), "reason": "; ".join(reasons), "dense": round(dense, 4) if dense is not None else None,
               "overlap": round(overlap, 3) if overlap is not None else None, "agree": agree}
        if rr is not None:
            out["rerank"] = round(rr, 4)
        return out
    except Exception as e:  # noqa: BLE001 - a confidence bug must never break a turn
        log.warning("assess_confidence failed: %s", e)
        return {"level": "medium", "top": 0.0, "gap": 0.0, "reason": "assessment failed"}


# ------------------------------------------------------------------------------------------------ reranker

# RERANKER=off | on | auto (default auto).
#   on:   every search: the fused top RERANK_CANDIDATES (20) are re-scored by the cross-encoder (the previous RERANKER=on).
#   auto: only an uncertain search (fused top1-top2 gap < RERANK_AUTO_GAP, or confidence "medium"), at most
#         RERANK_AUTO_CANDIDATES (8) passages cut to RERANK_AUTO_WORDS words, and only when the model is already on
#         disk and in memory: the first uncertain search loads it in the background (if memory_guard allows) and goes
#         without; it is never downloaded by the app.
# RERANKER_MODEL (see DEFAULT_RERANKER_MODEL and docs/RESULTS.md, 19 Oct 2026, for the comparison):
#   BAAI/bge-reranker-v2-m3                      Apache-2.0, 568M, a cross-encoder (sentence-transformers CrossEncoder)
#   Alibaba-NLP/gte-multilingual-reranker-base   Apache-2.0, 306M, a cross-encoder that runs code from the model
#                                                repositories (trust_remote_code; the model and its code repository
#                                                Alibaba-NLP/new-impl are pinned in REMOTE_CODE_RERANKERS,
#                                                RERANKER_REVISION overrides the model's)
#   Qwen/Qwen3-Reranker-0.6B                     Apache-2.0, 0.6B, a causal LM asked "yes"/"no": the score is
#                                                P(yes) from the last-position logits (Qwen3Reranker below, the
#                                                model card's prompt and scoring)
# RERANKER_DEVICE (auto: Apple's GPU, "mps", on Apple Silicon when torch can use it, else EMBEDDING_DEVICE) and
# RERANKER_DTYPE (auto: fp16 on mps/cuda, fp32 on a CPU, where fp16 was 5x slower) set where and in what precision it
# runs. On the M4's CPU every model was over the 150 ms budget (gte 150 ms median / 176 ms p95 for 3 passages).
# 3 (was 8): the cost is per passage (about 17 ms each for gte on an M4 GPU, 40 ms for bge, 130 ms for bge on the CPU),
# and on the bench sets 3 gave the same first-place gains as 8 without pushing a right passage out of the top 3 (with 8,
# gte lost 4 top-3 hits; with 3 the passages handed to the model are the same three, re-ordered).
RERANK_AUTO_CANDIDATES = 3
RERANK_AUTO_WORDS = 180            # about 256 XLM-R tokens of English; Hindi is cut by the model's own max_length
RERANK_AUTO_GAP = 0.10
RERANK_MAX_LENGTH = 320            # query + passage tokens in auto mode
RERANKER_NEED_MB = 1300            # memory to have free before loading a reranker not listed below
# Resident memory per model once loaded and warm (MB; process RSS growth measured on an M4, RESULTS.md, 19 Oct 2026:
# gte 0.5-0.9 GB on the GPU in fp16, up to 2 GB on the CPU in fp32; bge 1.6-2.1 GB on the CPU; Qwen3 2.5 GB on the CPU).
RERANKER_NEED_MB_BY_MODEL: Dict[str, int] = {      # keys as on the Hub; looked up case-insensitively
    "BAAI/bge-reranker-v2-m3": 2000,
    "Alibaba-NLP/gte-multilingual-reranker-base": 1700,   # measured 27 Oct: footprint +1525 MB (x1.1)
    "Qwen/Qwen3-Reranker-0.6B": 2500,
}
# Rerankers whose repository code must run (trust_remote_code): the model-repository commit and the commit of the
# repository the code comes from (gte keeps its modelling code in Alibaba-NLP/new-impl), both pinned.
REMOTE_CODE_RERANKERS: Dict[str, Dict[str, str]] = {
    "alibaba-nlp/gte-multilingual-reranker-base": {"revision": "8215cf04918ba6f7b6a62bb44238ce2953d8831c",
                                                   "code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4"},
}

# Qwen3-Reranker prompt (model card, Qwen/Qwen3-Reranker-0.6B): a system turn asking for "yes"/"no", the
# instruction / query / document as the user turn, and an empty thinking block in the assistant turn.
QWEN3_RERANKER_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
                         "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                         "<|im_start|>user\n")
QWEN3_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN3_RERANKER_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def reranker_kind(model_name: Optional[str] = None) -> str:
    """"qwen3" for a Qwen3-Reranker (yes/no causal LM), "cross-encoder" for everything else."""
    return "qwen3" if "qwen3-reranker" in (model_name or reranker_model_name()).lower() else "cross-encoder"


def _mps_usable() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import torch  # noqa: PLC0415 - heavy, lazy (only when the reranker is loaded)
        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def reranker_device() -> str:
    """RERANKER_DEVICE=auto (default) | cpu | mps | cuda. auto: "mps" on Apple Silicon (the gated reranker is about 2.5x
    faster there than on the CPU), else EMBEDDING_DEVICE."""
    choice = (os.getenv("RERANKER_DEVICE") or "auto").strip()
    if choice.lower() != "auto":
        return choice
    return "mps" if _mps_usable() else embedding_device()


def reranker_dtype() -> str:
    """"fp16" | "bf16" | "fp32" (RERANKER_DTYPE=auto|fp16|bf16|fp32; auto: fp16 on a GPU device, fp32 on a CPU, where
    half precision is slower for passage-length inputs, see _encode_bulk_fp32)."""
    choice = (os.getenv("RERANKER_DTYPE") or "auto").strip().lower()
    if choice in ("fp16", "float16", "half"):
        return "fp16"
    if choice in ("bf16", "bfloat16"):
        return "bf16"
    if choice in ("fp32", "float32", "full"):
        return "fp32"
    return "fp16" if reranker_device().split(":")[0] in ("mps", "cuda") else "fp32"


def reranker_need_mb(model_name: Optional[str] = None) -> int:
    name = (model_name or reranker_model_name()).lower()
    by_lower = {k.lower(): v for k, v in RERANKER_NEED_MB_BY_MODEL.items()}
    return _int_env("MEMORY_NEED_RERANKER_MB", by_lower.get(name, RERANKER_NEED_MB))


def qwen3_yes_probability(last_logits, yes_id: int, no_id: int) -> List[float]:
    """P("yes") per row from the last-position logits (batch, vocab): softmax over the "no" and "yes" logits only, as
    in the model card's compute_logits."""
    import torch  # noqa: PLC0415 - heavy, lazy
    two = torch.stack([last_logits[:, no_id], last_logits[:, yes_id]], dim=1).float()
    return [float(x) for x in torch.log_softmax(two, dim=1)[:, 1].exp().tolist()]


class Qwen3Reranker:
    """Qwen3-Reranker as a pointwise reranker with the CrossEncoder interface (`predict(pairs) -> scores 0..1`).

    Each (query, passage) becomes prefix + "<Instruct>: ...\n<Query>: ...\n<Document>: ..." + suffix (the passage is
    cut so the whole input fits `max_length` tokens), the batch is padded on the LEFT so the last position is every
    row's last token, and the score is P(yes) against P(no) at that position. The tokenizer and the model are passed in
    (load() builds the real ones), so the scoring is testable with fakes."""

    def __init__(self, tokenizer, model, *, max_length: int = 512, instruction: Optional[str] = None,
                 device: Optional[str] = None, batch_size: int = 8):
        self.tokenizer, self.model = tokenizer, model
        self.max_length, self.device, self.batch_size = int(max_length), device, max(1, int(batch_size))
        self.instruction = instruction or QWEN3_RERANKER_INSTRUCTION
        self.yes_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_id = tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = list(tokenizer.encode(QWEN3_RERANKER_PREFIX, add_special_tokens=False))
        self.suffix_ids = list(tokenizer.encode(QWEN3_RERANKER_SUFFIX, add_special_tokens=False))

    @classmethod
    def load(cls, model_name: str, device: str, dtype=None, max_length: int = 512, revision: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415 - heavy, lazy
        extra = {"revision": revision} if revision and not os.path.isdir(model_name) else {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **extra)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **extra)
        except TypeError:       # older transformers: the argument is still called torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **extra)
        model = model.to(device).eval()
        return cls(tokenizer, model, max_length=max_length, device=device,
                   instruction=(os.getenv("RERANKER_INSTRUCTION") or "").strip() or None)

    def format(self, query: str, passage: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {passage}"

    def build_inputs(self, pairs: Sequence[Tuple[str, str]]):
        budget = max(8, self.max_length - len(self.prefix_ids) - len(self.suffix_ids))
        enc = self.tokenizer([self.format(q, p) for q, p in pairs], padding=False, truncation="longest_first",
                             return_attention_mask=False, max_length=budget)
        ids = [self.prefix_ids + list(x) + self.suffix_ids for x in enc["input_ids"]]
        return self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")

    def predict(self, pairs: Sequence[Tuple[str, str]], batch_size: Optional[int] = None, **_ignored) -> List[float]:
        import torch  # noqa: PLC0415 - heavy, lazy
        pairs = list(pairs)
        size = max(1, int(batch_size or self.batch_size))
        out: List[float] = []
        for i in range(0, len(pairs), size):
            batch = self.build_inputs(pairs[i:i + size])
            if self.device:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                try:
                    logits = self.model(**batch, logits_to_keep=1).logits    # only the last position's logits
                except TypeError:
                    logits = self.model(**batch).logits
            out.extend(qwen3_yes_probability(logits[:, -1, :], self.yes_id, self.no_id))
        return out


def _torch_dtype(name: str):
    import torch  # noqa: PLC0415 - heavy, lazy
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _reranker_revision(model_name: str) -> Optional[str]:
    """RERANKER_REVISION, else the pinned commit of a remote-code reranker, else None (the cached main)."""
    pinned = REMOTE_CODE_RERANKERS.get(model_name.lower())
    return (os.getenv("RERANKER_REVISION") or "").strip() or (pinned["revision"] if pinned else None)


def _local_snapshot(model_name: str, revision: Optional[str]) -> str:
    """The model's folder in the Hugging Face cache (never downloads), or the name itself when it is not cached as a
    whole. Loading from the folder also avoids a Hub call transformers 4.57 makes for every tokenizer loaded by repo
    name (it fails with HF_HUB_OFFLINE=1)."""
    if os.path.isdir(model_name):
        return model_name
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        return snapshot_download(model_name, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached under that revision: let the loader resolve the name
        return model_name


def _new_reranker(model_name: str):
    """The reranker model for RERANKER_MODEL (see the notes above RERANK_AUTO_CANDIDATES)."""
    device, dtype = reranker_device(), _torch_dtype(reranker_dtype())
    max_length = _int_env("EMBEDDING_MAX_SEQ_LENGTH", 512)
    revision = _reranker_revision(model_name)
    source = _local_snapshot(model_name, revision)
    if reranker_kind(model_name) == "qwen3":
        return Qwen3Reranker.load(source, device, dtype, max_length=max_length, revision=revision)
    from sentence_transformers import CrossEncoder  # noqa: PLC0415 - heavy, lazy
    kwargs: Dict[str, Any] = {}
    code: Dict[str, str] = {}
    pinned = REMOTE_CODE_RERANKERS.get(model_name.lower())
    if pinned is not None:
        kwargs["trust_remote_code"] = True
        code = {"code_revision": pinned["code_revision"]} if pinned.get("code_revision") else {}
    if revision and source == model_name:
        kwargs["revision"] = revision
    if code:    # the code repository's commit, for the config, the tokenizer/processor and the model
        kwargs.update(config_kwargs=dict(code), processor_kwargs=dict(code))
    try:
        return CrossEncoder(source, device=device, max_length=max_length, model_kwargs={"dtype": dtype, **code},
                            **kwargs)
    except TypeError:       # older transformers: the argument is still called torch_dtype
        return CrossEncoder(source, device=device, max_length=max_length,
                            model_kwargs={"torch_dtype": dtype, **code}, **kwargs)


_reranker = None
_reranker_failed: Optional[str] = None
_reranker_loading = False


def reranker_mode() -> str:
    raw = (os.getenv("RERANKER", "auto") or "auto").strip().lower()
    if raw in ("on", "1", "true", "yes"):
        return "on"
    if raw in ("off", "0", "false", "no"):
        return "off"
    return "auto"


def reranker_enabled() -> bool:
    """RERANKER=on: the reranker runs on every search (read on every call, so it can be flipped in tests)."""
    return reranker_mode() == "on"


@lru_cache(maxsize=8)
def _reranker_on_disk(model_name: str) -> bool:
    if os.path.isdir(model_name):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
        revision = _reranker_revision(model_name)
        for f in ("model.safetensors", "pytorch_model.bin"):
            hit = try_to_load_from_cache(model_name, f, revision=revision) if revision else \
                try_to_load_from_cache(model_name, f)
            if isinstance(hit, str) and os.path.exists(hit):
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def reranker_available() -> bool:
    """The reranker model is on disk (never downloaded by auto mode) and has not failed to load."""
    return _reranker_failed is None and _reranker_on_disk(reranker_model_name())


def reranker_loaded() -> bool:
    return _reranker is not None


def _memory_allows_reranker() -> bool:
    try:
        from core import memory_guard  # noqa: PLC0415
    except ImportError:  # pragma: no cover - imported as backend.services.retrieval
        try:
            from backend.core import memory_guard  # type: ignore  # noqa: PLC0415
        except ImportError:
            return True
    if not memory_guard.enabled():
        return True
    avail = memory_guard.available_mb()
    need = reranker_need_mb() + memory_guard.reserve_mb()
    return avail is None or avail >= need


def _load_reranker():
    global _reranker, _reranker_failed
    with _embedder_lock:
        if _reranker is None:
            _reranker = _new_reranker(reranker_model_name())
    return _reranker


def _load_reranker_background() -> bool:
    """Start loading the reranker in a thread (auto mode); True when a load was started."""
    global _reranker_loading
    if _reranker is not None or _reranker_loading or not reranker_available() or not _memory_allows_reranker():
        return False
    _reranker_loading = True

    def work():
        global _reranker_loading, _reranker_failed
        t = time.perf_counter()
        try:
            _load_reranker()
            # one throw-away call: the first one pays for kernel set-up (about 0.3 s more on Apple's GPU)
            rerank_scores("warm-up", ["warm-up passage"] * max(1, _int_env("RERANK_AUTO_CANDIDATES", RERANK_AUTO_CANDIDATES)))
            log.info("reranker %s loaded in the background in %.1f s (%s, %s)", reranker_model_name(),
                     time.perf_counter() - t, reranker_device(), reranker_dtype())
        except Exception as e:  # noqa: BLE001
            _reranker_failed = f"{type(e).__name__}: {e}"[:200]
            log.warning("reranker could not load (%s); searches go without it", _reranker_failed)
        finally:
            _reranker_loading = False

    threading.Thread(target=work, name="reranker-load", daemon=True).start()
    return True


_rerank_lock = threading.Lock()    # one reranker call at a time: a model (on the GPU above all) is not thread-safe


def rerank_scores(query: str, texts: Sequence[str]) -> List[float]:
    model = _load_reranker()
    with _rerank_lock:
        return [float(s) for s in model.predict([(query, t) for t in texts], show_progress_bar=False)]


_reranker_warm_lock = threading.Lock()   # call starts that come together warm the reranker one after the other


def warm_reranker() -> Optional[str]:
    """RERANKER=on: load the reranker (once) and score one tiny pair, so the first real rerank of a call is not a cold
    start (0.1-0.9 s on MPS, measured 21 Oct). Returns the model name, or None when the reranker is off or failed to
    load earlier. Serialised by _reranker_warm_lock and then by rerank_scores' own lock, so it never runs twice at once.
    Raises what the load raises; the caller reports it."""
    if not reranker_enabled() or _reranker_failed:
        return None
    with _reranker_warm_lock:
        rerank_scores("warm-up", ["warm-up passage"])
    return reranker_model_name()


def rerank(query: str, rows: List[dict], top_k: int) -> List[dict]:
    """Cross-encoder rerank of candidate rows (dicts with `chunk_text`) down to top_k. With the reranker off, or when
    it cannot load, the rows come back unchanged (cut to top_k)."""
    global _reranker_failed
    if not reranker_enabled() or len(rows) <= 1 or _reranker_failed:
        return rows[:top_k]
    try:
        scores = rerank_scores(query, [passage_text(r) for r in rows])
    except Exception as e:  # noqa: BLE001 - never lose retrieval because the optional reranker is broken
        _reranker_failed = f"{type(e).__name__}: {e}"[:200]
        return rows[:top_k]
    order = sorted(range(len(rows)), key=lambda i: -scores[i])[:top_k]
    return [{**rows[i], "rerank_score": scores[i]} for i in order]


def _truncate_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n])


def rerank_gate(rows: Sequence[dict], confidence: Optional[dict]) -> Tuple[bool, str]:
    """Auto mode: should this search be reranked? Only when it is uncertain: a small fused top1-top2 gap or a
    "medium" confidence. Not for one passage, and not for "none"/"high" (nothing to fix, or nothing to gain)."""
    if len(rows) < 2:
        return False, "one passage"
    level = (confidence or {}).get("level")
    if level in ("none", "high"):
        return False, f"confidence {level}"
    try:
        gap_limit = float(os.getenv("RERANK_AUTO_GAP") or RERANK_AUTO_GAP)
    except ValueError:
        gap_limit = RERANK_AUTO_GAP
    s0, s1 = rows[0].get("score"), rows[1].get("score")
    gap = (float(s0) - float(s1)) if isinstance(s0, (int, float)) and isinstance(s1, (int, float)) else None
    if level == "medium" or (gap is not None and gap < gap_limit):
        return True, f"confidence {level}, gap {gap if gap is None else round(gap, 3)}"
    return False, f"gap {gap if gap is None else round(gap, 3)}"


def maybe_rerank(query: str, rows: List[dict], top_k: int, confidence: Optional[dict] = None) -> Tuple[List[dict], dict]:
    """Apply the reranker per RERANKER (see above). Returns (rows cut to top_k, info) where info says whether it ran,
    why not, and how long it took. Never raises."""
    global _reranker_failed
    mode = reranker_mode()
    info: dict = {"mode": mode, "ran": False}
    if mode == "off":
        return rows[:top_k], info
    if mode == "on":
        t = time.perf_counter()
        out = rerank(query, rows, top_k)
        info.update({"ran": any("rerank_score" in r for r in out), "ms": round((time.perf_counter() - t) * 1000, 1)})
        return out, info
    run, why = rerank_gate(rows, confidence)
    info["why"] = why
    if not run:
        return rows[:top_k], info
    if not reranker_available():
        info["why"] = "model not downloaded" if _reranker_failed is None else "model failed to load"
        return rows[:top_k], info
    if _reranker is None:
        info["why"] = "loading in the background" if _load_reranker_background() else "not loaded (memory)"
        return rows[:top_k], info
    m = max(2, min(len(rows), _int_env("RERANK_AUTO_CANDIDATES", RERANK_AUTO_CANDIDATES)))
    head, tail = rows[:m], rows[m:]
    words = _int_env("RERANK_AUTO_WORDS", RERANK_AUTO_WORDS)
    t = time.perf_counter()
    try:
        scores = rerank_scores(query, [_truncate_words(passage_text(r), words) for r in head])
    except Exception as e:  # noqa: BLE001
        _reranker_failed = f"{type(e).__name__}: {e}"[:200]
        info["why"] = "failed"
        return rows[:top_k], info
    ms = (time.perf_counter() - t) * 1000
    order = sorted(range(len(head)), key=lambda i: -scores[i])
    out = [{**head[i], "rerank_score": scores[i]} for i in order] + list(tail)
    info.update({"ran": True, "ms": round(ms, 1), "n": len(head)})
    log.info("reranked %d passages in %.0f ms (%s)", len(head), ms, why)
    return out[:top_k], info


def candidate_count(top_k: int) -> int:
    """How many fused candidates to fetch: RERANK_CANDIDATES with RERANKER=on; in auto mode enough for the gated
    reranker when its model is on disk; top_k otherwise."""
    mode = reranker_mode()
    if mode == "on":
        return max(top_k, RERANK_CANDIDATES)
    if mode == "auto" and reranker_available():
        return max(top_k, _int_env("RERANK_AUTO_CANDIDATES", RERANK_AUTO_CANDIDATES))
    return top_k
