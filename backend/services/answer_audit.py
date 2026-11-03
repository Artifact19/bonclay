"""Answer audit: after a conversation is over, check each reply the agent gave from its documents against the passages it
was given, and flag the ones the documents do not support. It never runs on a live turn.

What is checked. A stored agent turn is checked when it was grounded on passages (the turn's `retrieval` has
`n_chunks` > 0), is in English, and is not one of the owner-approved or fixed kinds: an instant answer (the owner wrote
and approved it), a refusal ("I don't have that information"), the first greeting, a model failure. Inside a reply, the
AI-disclosure line, pleasantries, questions the agent asks and tool confirmations ("Your booking is confirmed") are left
out, and the remaining sentences are scored one by one. The reply's score is its weakest sentence.

The verifier (ANSWER_AUDIT_MODEL) is MiniCheck-RoBERTa-Large by default (`lytang/MiniCheck-RoBERTa-Large`, MIT, 355 M
parameters, Tang et al. 2024): it returns P(supported) 0..1 for (passages, sentence); a sentence below `AUDIT_THRESHOLD`
(default 0.5, MiniCheck's own decision point) is unsupported. Chosen on 19 Oct 2026 (docs/RESULTS.md) on 74 hand-labelled
English sentences about the doc_qa documents (half copied facts, half with a changed number, time, day, date or name, or
an invented detail), premise = the three passages the agent was given: balanced accuracy 0.93 at 0.5 (AUC 0.97), about
0.57 s per sentence on two CPU threads, 1.7 GB resident. The alternatives stay selectable:
  hhem                vectara/hallucination_evaluation_model (HHEM-2.1-Open; Apache-2.0, flan-t5-base, runs code from its
                      repository: trust_remote_code). AUC 0.91, but at 0.5 it passed 60% of the wrong sentences; its
                      calibrated threshold is 0.9 (balanced accuracy 0.85, tuned on the same 74 sentences). 0.17 s/sentence.
  minicheck-ft5       lytang/MiniCheck-Flan-T5-Large (MIT, 770 M): balanced accuracy 0.91 at 0.5, 0.68 s/sentence, 4.2 GB.
Every model is pinned to a commit (MODELS[...]["revision"]; ANSWER_AUDIT_MODEL_REVISION overrides it). It is English-only:
on 10 Hindi / Hinglish sentences every model was near chance or flagged most correct ones, so a reply in Hindi, Devanagari
or Hinglish, or a document that is not in English, is stored as "not checked (language)" and the page says so. The
owner's "It was fine" / "It was wrong" answers are kept so the threshold can be tuned from real data.

Where the passages come from. The turn stores `retrieval.chunk_ids` (added to backend._retrieval_signals); the passage
text is re-read from the document store by id. Turns stored before that have only `doc_ids`: for those the best BM25
passages of those documents (for question + reply) stand in, and the result is marked `passage_source: "approx"`.

Model handling: loaded lazily inside the audit worker only (never at import, never in tests), guarded by
core/memory_guard.py, run on the CPU (ANSWER_AUDIT_DEVICE=mps tries Apple's GPU and falls back), one check at a time, and
unloaded after ANSWER_AUDIT_IDLE_S seconds (default 300) without use. The first load downloads the model (MiniCheck-RoBERTa
about 1.4 GB; HHEM about 0.45 GB and runs the code in its repository) at the pinned commit. The background job therefore
waits until the owner has run "Check recent answers" once (the model is then in the local cache), unless
ANSWER_AUDIT_AUTO_DOWNLOAD=on.

Results are stored per turn in backend/data/answer_audit.json (BONCLAY_ANSWER_AUDIT_FILE overrides) in local mode, and in
the `answer_audit` table (migration 019, owner-only RLS) in hosted mode: new storage only, no existing table is touched.

Switches (backend/.env.example): ANSWER_AUDIT=off, ANSWER_AUDIT_AUTO=off, ANSWER_AUDIT_MODEL, AUDIT_THRESHOLD, ANSWER_AUDIT_IDLE_S,
ANSWER_AUDIT_DEVICE, ANSWER_AUDIT_MODEL_REVISION, ANSWER_AUDIT_AUTO_DOWNLOAD, ANSWER_AUDIT_RUN_MAX,
ANSWER_AUDIT_POLL_S, ANSWER_AUDIT_PHONE_IDLE_S, ANSWER_AUDIT_IDLE_AFTER_S, ANSWER_AUDIT_SWEEP_DAYS."""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from core.errors import AppError
    from services import knowledge_gaps as kg
    from services import sentences as sentence_split
except ImportError:  # pragma: no cover - imported as backend.services.answer_audit
    from backend.core.errors import AppError
    from backend.services import knowledge_gaps as kg
    from backend.services import sentences as sentence_split

log = logging.getLogger("bonclay.answer_audit")

# The checking models (ANSWER_AUDIT_MODEL picks one; see the module docstring and docs/RESULTS.md, 19 Oct 2026, for the
# comparison). Each is pinned to a commit of its repository (ANSWER_AUDIT_MODEL_REVISION overrides the pin), has its own
# calibrated threshold (AUDIT_THRESHOLD overrides it) and its own memory estimate for core/memory_guard.py.
MODELS: Dict[str, Dict[str, Any]] = {
    "hhem": {
        "repo": "vectara/hallucination_evaluation_model", "label": "HHEM-2.1-Open", "kind": "hhem",
        "revision": "de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8", "remote_code": True,
        "needs": (("google/flan-t5-base", "tokenizer_config.json"),),   # HHEM builds its tokenizer from this repository
        "threshold": 0.9, "need_mb": 1200,
    },
    "minicheck-ft5": {
        "repo": "lytang/MiniCheck-Flan-T5-Large", "label": "MiniCheck-Flan-T5-Large", "kind": "minicheck-ft5",
        "revision": "96eafd01cee2d16cf81aaa2fb226b14f422a37b3", "remote_code": False, "needs": (),
        "threshold": 0.5, "need_mb": 4200,
    },
    "minicheck-roberta": {
        "repo": "lytang/MiniCheck-RoBERTa-Large", "label": "MiniCheck-RoBERTa-Large", "kind": "minicheck-roberta",
        "revision": "74c8919647e61ed0f71bc177d94f10930f090068", "remote_code": False, "needs": (),
        "threshold": 0.5, "need_mb": 2200,
    },
}
DEFAULT_MODEL = "minicheck-roberta"

MODEL_NAME = MODELS[DEFAULT_MODEL]["repo"]
TOKENIZER_NAME = "google/flan-t5-base"   # HHEM-2.1-Open (ANSWER_AUDIT_MODEL=hhem) builds its tokenizer from this repository
MODEL_LABEL = MODELS[DEFAULT_MODEL]["label"]
ENGINE = "answer_audit"                  # memory_guard engine key
DEFAULT_THRESHOLD = MODELS[DEFAULT_MODEL]["threshold"]
DEFAULT_IDLE_UNLOAD_S = 300
MODEL_NEED_MB = MODELS[DEFAULT_MODEL]["need_mb"]   # weights and activations plus headroom
MAX_PREMISE_CHARS = 3000
MAX_PASSAGES = 3
MAX_SENTENCES = 8                        # a reply is at most about 40 words; this bounds a runaway one
SNIPPET_CHARS = 320
MAX_ROWS = 20000

STATUS_CHECKED = "checked"
STATUS_SKIPPED = "skipped"
STATUS_LANGUAGE = "not_checked_language"
FEEDBACK = ("fine", "wrong")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# -- settings (read on every call so they can change without a restart) --------------------------------------------

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name) or default))
    except ValueError:
        return default


def _off(name: str) -> bool:
    return (os.getenv(name, "on") or "on").strip().lower() in ("off", "0", "false", "no")


def enabled() -> bool:
    """ANSWER_AUDIT=off turns the whole feature off (no background job, runs are refused)."""
    return not _off("ANSWER_AUDIT")


def auto_enabled() -> bool:
    """ANSWER_AUDIT_AUTO=off keeps only the owner-triggered check."""
    return enabled() and not _off("ANSWER_AUDIT_AUTO")


def model_key() -> str:
    """ANSWER_AUDIT_MODEL: hhem | minicheck-ft5 | minicheck-roberta (default DEFAULT_MODEL); unknown values fall back."""
    key = (os.getenv("ANSWER_AUDIT_MODEL") or "").strip().lower()
    return key if key in MODELS else DEFAULT_MODEL


def model_spec() -> Dict[str, Any]:
    return MODELS[model_key()]


def model_label() -> str:
    return str(model_spec()["label"])


def model_revision() -> str:
    """ANSWER_AUDIT_MODEL_REVISION, else the commit the selected model is pinned to."""
    return (os.getenv("ANSWER_AUDIT_MODEL_REVISION") or "").strip() or str(model_spec()["revision"])


def threshold() -> float:
    """AUDIT_THRESHOLD (default: the selected model's calibrated threshold, see MODELS): a reply whose weakest sentence
    scores below this is flagged."""
    default = float(model_spec()["threshold"])
    v = _float_env("AUDIT_THRESHOLD", default)
    return v if 0.0 < v < 1.0 else default


def idle_unload_s() -> float:
    return max(5.0, _float_env("ANSWER_AUDIT_IDLE_S", DEFAULT_IDLE_UNLOAD_S))


def run_max() -> int:
    return max(1, min(500, _int_env("ANSWER_AUDIT_RUN_MAX", 40)))


def poll_s() -> float:
    return max(10.0, _float_env("ANSWER_AUDIT_POLL_S", 60.0))


def phone_idle_s() -> float:
    """A phone call is over this many seconds after its last turn (the call stream closes, so this is short)."""
    return max(0.0, _float_env("ANSWER_AUDIT_PHONE_IDLE_S", 45.0))


def other_idle_s() -> float:
    """Widget, WhatsApp and app conversations are over this many seconds after their last turn (2 minutes)."""
    return max(0.0, _float_env("ANSWER_AUDIT_IDLE_AFTER_S", 120.0))


def sweep_days() -> float:
    return max(0.1, _float_env("ANSWER_AUDIT_SWEEP_DAYS", 2.0))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


now_iso = _now


def parse_ts(value: Any) -> Optional[datetime]:
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# -- the verifier -------------------------------------------------------------------------------------------------------

def model_cached() -> bool:
    """True when the selected checking model (and any repository it needs, HHEM's tokenizer) is already in the local
    Hugging Face cache (so loading downloads nothing). Never touches the network."""
    spec = model_spec()
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
        return (isinstance(try_to_load_from_cache(spec["repo"], "config.json", revision=model_revision()), str)
                and all(isinstance(try_to_load_from_cache(repo, name), str) for repo, name in spec["needs"]))
    except Exception:  # noqa: BLE001
        return False


# MiniCheck (Tang, Laban & Durrett, EMNLP 2024; github.com/Liyan06/MiniCheck, minicheck/inference.py): the document and
# the claim are joined with the tokenizer's end-of-sequence token; the Flan-T5 model is prompted with "predict: " and
# reads one decoder step whose logits for token 3 ("no support") and 209 ("support") give the probability; the RoBERTa
# model is a 2-class classifier (class 1 = supported). A long document is cut into chunks and the best chunk counts.
MINICHECK_FT5_IDS = (3, 209)
MINICHECK_CHUNK_TOKENS = {"minicheck-ft5": 1500, "minicheck-roberta": 400}
MINICHECK_MAX_LENGTH = {"minicheck-ft5": 2048, "minicheck-roberta": 512}
_SENT_END = re.compile(r"(?<=[.!?;:\n])\s+")


class MiniCheckModel:
    """A MiniCheck checkpoint with HHEM's interface: `predict([(premise, claim), ...]) -> [P(supported)]`. The tokenizer
    and model are passed in, so the input building and the scoring are testable with fakes."""

    def __init__(self, tokenizer: Any, model: Any, kind: str, batch_size: int = 4):
        if kind not in MINICHECK_CHUNK_TOKENS:
            raise ValueError(f"unknown MiniCheck kind {kind!r}")
        self.tokenizer, self.model, self.kind, self.batch_size = tokenizer, model, kind, max(1, batch_size)

    def _n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def chunks(self, doc: str, claim: str) -> List[str]:
        """The document cut at sentence ends into pieces that fit the model next to the claim (whole when it fits)."""
        budget = max(32, min(MINICHECK_CHUNK_TOKENS[self.kind],
                             MINICHECK_MAX_LENGTH[self.kind] - self._n_tokens(claim) - 16))
        if self._n_tokens(doc) <= budget:
            return [doc]
        out: List[str] = []
        cur: List[str] = []
        used = 0
        for sent in (p for p in _SENT_END.split(doc) if p.strip()):
            n = self._n_tokens(sent)
            if cur and used + n > budget:
                out.append(" ".join(cur))
                cur, used = [], 0
            cur.append(sent)
            used += n
        if cur:
            out.append(" ".join(cur))
        return out or [doc]

    def texts(self, doc: str, claim: str) -> List[str]:
        eos = self.tokenizer.eos_token or "</s>"
        joined = [eos.join([c, claim]) for c in self.chunks(doc, claim)]
        return ["predict: " + t for t in joined] if self.kind == "minicheck-ft5" else joined

    def _probabilities(self, texts: List[str]) -> List[float]:
        import torch  # noqa: PLC0415 - heavy, lazy
        enc = self.tokenizer(texts, max_length=MINICHECK_MAX_LENGTH[self.kind], truncation=True, padding=True,
                             return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            if self.kind == "minicheck-ft5":
                start = torch.zeros((enc["input_ids"].shape[0], 1), dtype=torch.long, device=enc["input_ids"].device)
                logits = self.model(**enc, decoder_input_ids=start).logits[:, 0, :]
                logits = logits[:, list(MINICHECK_FT5_IDS)]
            else:
                logits = self.model(**enc).logits
            return [float(x) for x in torch.softmax(logits.float(), dim=-1)[:, 1].tolist()]

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        flat: List[Tuple[int, str]] = []
        for i, (doc, claim) in enumerate(pairs):
            flat.extend((i, t) for t in self.texts(doc, claim))
        best = [0.0] * len(pairs)
        for k in range(0, len(flat), self.batch_size):
            part = flat[k:k + self.batch_size]
            for (i, _), p in zip(part, self._probabilities([t for _, t in part])):
                best[i] = max(best[i], p)
        return best


MINICHECK_FILES = ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "spiece.model")


def _snapshot(repo: str, revision: str) -> str:
    """The model's folder at `revision`: from the local cache, else downloaded once (only the files the model needs).
    Loading from the folder keeps transformers from calling the Hub on every load and from fetching a converted
    safetensors copy of the weights in the background (about 1.4 GB more for MiniCheck-RoBERTa). The .bin weights are
    read with torch.load(weights_only=True) by transformers: no pickled code runs."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    try:
        return snapshot_download(repo, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet: the first check downloads it
        return snapshot_download(repo, revision=revision, allow_patterns=list(MINICHECK_FILES))


def _load_model(spec: Dict[str, Any], revision: str) -> Any:
    """The selected checking model with a `predict(pairs)` method (downloads it the first time)."""
    if spec["kind"] == "hhem":
        from transformers import AutoModelForSequenceClassification  # noqa: PLC0415
        return AutoModelForSequenceClassification.from_pretrained(spec["repo"], trust_remote_code=True,
                                                                  revision=revision)
    from transformers import AutoTokenizer  # noqa: PLC0415
    source = _snapshot(spec["repo"], revision)
    tokenizer = AutoTokenizer.from_pretrained(source)
    if spec["kind"] == "minicheck-ft5":
        from transformers import AutoModelForSeq2SeqLM  # noqa: PLC0415
        model = AutoModelForSeq2SeqLM.from_pretrained(source)
    else:
        from transformers import AutoModelForSequenceClassification  # noqa: PLC0415
        model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2)
    model.eval()
    return MiniCheckModel(tokenizer, model, spec["kind"])


def _default_loader() -> Any:
    """Load the selected checking model (downloads it the first time). Called only from the audit worker."""
    try:
        from core import memory_guard  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from backend.core import memory_guard  # noqa: PLC0415
    spec = model_spec()
    memory_guard.DEFAULT_NEED_MB[ENGINE] = int(spec["need_mb"])
    memory_guard.ensure_headroom(ENGINE, [], lambda _key: None)   # nothing to unload: refuse rather than swap
    model = _load_model(spec, model_revision())
    inner = getattr(model, "model", model)
    if hasattr(inner, "eval"):
        inner.eval()
    if (os.getenv("ANSWER_AUDIT_DEVICE", "cpu") or "cpu").strip().lower() == "mps":
        try:
            import torch  # noqa: PLC0415
            if torch.backends.mps.is_available():
                inner.to("mps")
        except Exception as e:  # noqa: BLE001 - the model's own predict() may not move inputs: stay on the CPU
            log.warning("answer audit: could not use the GPU (%s); using the CPU", e)
            inner.to("cpu")
    return model


class HHEMVerifier:
    """The checking model (ANSWER_AUDIT_MODEL, MiniCheck-RoBERTa-Large by default; the name is historical), loaded on first use and unloaded after `idle_s` seconds without use. `score(pairs)` takes a list of
    (premise, hypothesis) and returns one 0..1 consistency score per pair. One call at a time (a lock); the loader is
    injectable so the lifecycle is testable without the model."""

    @property
    def name(self) -> str:
        spec = model_spec()
        return f"{spec['repo']} ({spec['label']})"

    def __init__(self, loader: Optional[Callable[[], Any]] = None, idle_s: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._loader = loader or _default_loader
        self._idle_s = idle_s
        self._clock = clock
        self._lock = threading.Lock()
        self._model: Any = None
        self.last_used = 0.0
        self._timer: Optional[threading.Timer] = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        out: List[float] = []
        try:
            import torch  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            torch = None  # type: ignore[assignment]
        previous = None
        if torch is not None:
            previous = torch.get_num_threads()
            torch.set_num_threads(max(1, min(2, previous)))   # leave the cores to live calls
        try:
            for i in range(0, len(pairs), 8):
                batch = list(pairs[i:i + 8])
                if torch is not None:
                    with torch.inference_mode():
                        scores = self._model.predict(batch)
                else:
                    scores = self._model.predict(batch)
                out.extend(float(s) for s in (scores.tolist() if hasattr(scores, "tolist") else scores))
                time.sleep(float(os.getenv("ANSWER_AUDIT_PAUSE_S", "0.05") or 0))   # low priority: let other work in
        finally:
            if torch is not None and previous is not None:
                torch.set_num_threads(previous)
        return out

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        with self._lock:
            if self._model is None:
                log.info("answer audit: loading %s", model_spec()["repo"])
                self._model = self._loader()
            self.last_used = self._clock()
            try:
                return self._predict(pairs)
            finally:
                self.last_used = self._clock()
                self._arm_timer()

    def _arm_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        delay = (self._idle_s if self._idle_s is not None else idle_unload_s()) + 1.0
        t = threading.Timer(delay, self.unload_if_idle)
        t.daemon = True
        self._timer = t
        t.start()

    def unload_if_idle(self, now: Optional[float] = None) -> bool:
        """Drop the model when it has been idle long enough. True when it was unloaded."""
        idle = self._idle_s if self._idle_s is not None else idle_unload_s()
        with self._lock:
            if self._model is None:
                return False
            if (now if now is not None else self._clock()) - self.last_used < idle:
                return False
            self._release()
            return True

    def unload(self) -> None:
        with self._lock:
            self._release()

    def _release(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._model is not None:
            self._model = None
            try:
                import gc  # noqa: PLC0415
                gc.collect()
                import torch  # noqa: PLC0415
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            log.info("answer audit: model unloaded")


_verifier: Any = None
_verifier_forced = False
_verifier_lock = threading.Lock()
_job_lock = threading.Lock()   # one audit at a time, whoever started it


def get_verifier() -> Any:
    global _verifier
    with _verifier_lock:
        if _verifier is None:
            _verifier = HHEMVerifier()
        return _verifier


def set_verifier(verifier: Any) -> None:
    """Replace the verifier (tests pass a fake with `.score(pairs)` and `.name`); None restores the default checker."""
    global _verifier, _verifier_forced
    with _verifier_lock:
        _verifier, _verifier_forced = verifier, verifier is not None


def auto_ready() -> bool:
    """The background job may run: it is on, and the model needs no first download (or that was allowed)."""
    if not auto_enabled():
        return False
    return _verifier_forced or model_cached() or auto_download()


def auto_download() -> bool:
    """ANSWER_AUDIT_AUTO_DOWNLOAD=on lets the background job download the model by itself (default off)."""
    return (os.getenv("ANSWER_AUDIT_AUTO_DOWNLOAD", "off") or "off").strip().lower() in ("on", "1", "true", "yes")


def model_ready() -> bool:
    return _verifier_forced or model_cached() or (_verifier is not None and getattr(_verifier, "loaded", False))


# -- text rules ---------------------------------------------------------------------------------------------------------

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_WS = re.compile(r"\s+")
# The disclosure lines of backend.AI_DISCLOSURES and the ways an agent greets or closes.
_DISCLOSURE = re.compile(r"\b(?:this is an? (?:ai|artificial intelligence)|i(?:'m| am) an? (?:ai|virtual|automated)|"
                         r"you(?:'re| are) (?:talking|speaking|chatting) (?:to|with) an? (?:ai|virtual|automated))", re.I)
_PLEASANTRY = re.compile(
    r"^(?:hello|hi|hey|namaste|good (?:morning|afternoon|evening)|thanks|thank you|you(?:'re| are) welcome|sure|of course|"
    r"certainly|okay|ok|alright|great|sorry|sure thing|no problem|goodbye|bye|please hold|one moment|just a moment|"
    r"let me check|how can i (?:help|assist)|is there anything else|anything else|have a (?:nice|great|good) day)\b", re.I)
_CLOSER = re.compile(r"\banything else (?:i|we) can (?:help|assist)|how (?:may|can) i (?:help|assist)", re.I)
_CONFIRM = re.compile(r"\b(?:booked|booking|confirmed|scheduled|cancel(?:l)?ed|submitted|reserved|noted|request(?:ed)?|"
                      r"reference|appointment is)\b", re.I)


def _words(text: str) -> List[str]:
    return _WORD.findall(text or "")


def split_sentences(text: str) -> List[str]:
    sp = sentence_split.SentenceSplitter()
    out = sp.feed(text or "") + sp.flush()
    return [s.strip() for s in out if s and s.strip()]


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("’", "'").replace("‘", "'")).strip()


def _tool_results(turn: dict) -> str:
    calls = turn.get("tool_calls")
    if not isinstance(calls, list):
        return ""
    return " ".join(str(c.get("result") or "") for c in calls if isinstance(c, dict) and c.get("ok"))


def has_tool_calls(turn: dict) -> bool:
    return bool(_tool_results(turn))


def checkable_sentences(reply: str, turn: Optional[dict] = None) -> List[str]:
    """The sentences of a reply that make a claim the documents should support: not the AI-disclosure line, a
    pleasantry, a question the agent asks, a refusal, a very short fragment or (when the turn used a tool) a confirmation
    of what the tool did."""
    turn = turn or {}
    tool_text = _tool_results(turn).lower()
    tool_tokens = set(_words(tool_text)) if tool_text else set()
    out: List[str] = []
    for raw in split_sentences(reply):
        s = _norm(raw)
        words = _words(s)
        if len(words) < 3 or s.endswith("?"):
            continue
        if (_DISCLOSURE.search(s) and len(words) <= 12) or _CLOSER.search(s):
            continue
        if _PLEASANTRY.match(s) and len(words) <= 9:
            continue
        if kg.is_gap_reply(s):
            continue
        if tool_text:
            low = [w.lower() for w in words]
            overlap = sum(1 for w in low if w in tool_tokens) / max(1, len(low))
            if _CONFIRM.search(s) or overlap >= 0.5:
                continue
        out.append(s)
    return out[:MAX_SENTENCES]


def _latin_ratio(text: str) -> float:
    letters = [c for c in (text or "") if c.isalpha()]
    return 1.0 if not letters else sum(1 for c in letters if ord(c) < 0x250) / len(letters)


def reply_is_english(reply: str, language: Optional[str]) -> bool:
    """False for a reply in another language, any non-Latin script, or Hinglish (Hindi in Roman letters)."""
    lang = (language or "").strip().lower().replace("_", "-")
    if lang and not (lang == "en" or lang.startswith("en-")):
        return False
    if _latin_ratio(reply) < 0.9:
        return False
    try:
        try:
            from services import language_policy  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            from backend.services import language_policy  # noqa: PLC0415
        if language_policy._hinglish_reply(reply):  # noqa: SLF001
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


# -- candidates ---------------------------------------------------------------------------------------------------------

@dataclass
class Candidate:
    user_id: str
    agent_id: Optional[str]
    conversation_id: str
    channel: Optional[str]
    turn: dict
    question: str
    index: int = 0   # position of the turn in its conversation


def _retrieval(turn: dict) -> dict:
    r = turn.get("retrieval")
    if isinstance(r, str):
        try:
            r = json.loads(r)
        except ValueError:
            r = None
    return r if isinstance(r, dict) else {}


def _timings(turn: dict) -> dict:
    t = turn.get("timings")
    if isinstance(t, str):
        try:
            t = json.loads(t)
        except ValueError:
            t = None
    return t if isinstance(t, dict) else {}


def skip_reason(turn: dict, question: str) -> Optional[str]:
    """Why a turn is not checked at all (None: check it). These need no model and no passages."""
    reply = str(turn.get("text_content") or "")
    if not reply.strip():
        return "empty reply"
    if not question.strip():
        return "greeting"
    if _timings(turn).get("instant_answer_id"):
        return "instant answer (approved by the owner)"
    if turn.get("fallback"):
        return "model failure"
    r = _retrieval(turn)
    if not r or r.get("error") or not (r.get("n_chunks") or 0):
        return "not answered from documents"
    if kg.is_gap_reply(reply):
        return "refusal"
    return None


def collect(convs: Iterable[Tuple[dict, List[dict]]], known: Set[Tuple[str, str]], *,
            limit: Optional[int] = None, agent_id: Optional[str] = None) -> List[Candidate]:
    """Agent turns that have not been audited yet, newest conversations first (at most `limit`), from
    (conversation, turns oldest first) pairs. Practice (Test) calls are left out, like on the Results page."""
    out: List[Candidate] = []
    for conv, turns in convs:
        if conv.get("is_test") or not conv.get("user_id"):
            continue
        if agent_id and conv.get("agent_id") != agent_id:
            continue
        question = ""
        for i, t in enumerate(turns):
            role = t.get("role")
            if role == "user":
                question = str(t.get("text_content") or "")
            elif role == "agent":
                tid = t.get("id")
                if tid and (str(conv["user_id"]), str(tid)) not in known:
                    out.append(Candidate(str(conv["user_id"]), conv.get("agent_id"), str(conv.get("id")),
                                         conv.get("source"), t, question, i))
                question = ""   # a second reply with no new question is not a reply to this one
    out.sort(key=lambda c: str(c.turn.get("created_at") or ""), reverse=True)
    return out[:limit] if limit else out


def conversation_over(conv: dict, turns: List[dict], now: Optional[datetime] = None,
                      due_now: Optional[Set[str]] = None) -> bool:
    """A phone call is over ANSWER_AUDIT_PHONE_IDLE_S (45) seconds after its last turn, any other conversation after
    ANSWER_AUDIT_IDLE_AFTER_S (120): the owner may still be talking to it before that. `due_now` holds conversation ids a
    channel reported as ended (notify_conversation_end)."""
    if due_now and str(conv.get("id")) in due_now:
        return True
    stamps = [d for d in (parse_ts(t.get("created_at")) for t in turns) if d is not None]
    if not stamps:
        return False
    idle = phone_idle_s() if conv.get("source") == "phone" else other_idle_s()
    return ((now or datetime.now(timezone.utc)) - max(stamps)).total_seconds() >= idle


_due_now: Set[str] = set()


def notify_conversation_end(conversation_id: str) -> None:
    """A channel (a phone call hanging up) tells the audit its conversation is over: the next sweep checks it without
    waiting for the idle period."""
    if conversation_id:
        _due_now.add(str(conversation_id))
        if len(_due_now) > 1000:
            _due_now.clear()


def take_due_now() -> Set[str]:
    return set(_due_now)


def live_traffic(convs: Iterable[Tuple[dict, List[dict]]], now: Optional[datetime] = None, quiet_s: float = 10.0) -> bool:
    """True when some conversation had a turn in the last `quiet_s` seconds: a live call is using the machine."""
    now = now or datetime.now(timezone.utc)
    for _conv, turns in convs:
        for t in turns[-2:]:
            d = parse_ts(t.get("created_at"))
            if d is not None and 0 <= (now - d).total_seconds() < quiet_s:
                return True
    return False


# -- passages -----------------------------------------------------------------------------------------------------------

def _chunk_text(chunk: dict) -> str:
    text = str(chunk.get("chunk_text") or chunk.get("content") or "").strip()
    header = str(chunk.get("header") or "").strip()
    return f"{header}\n{text}" if header and text else text


def passages_for(turn: dict, question: str, reply: str, chunks: Any) -> Tuple[List[str], str]:
    """(passages, "stored" | "approx" | "none"). `chunks` has by_ids(ids) and by_docs(doc_ids) returning chunk rows."""
    r = _retrieval(turn)
    ids = [str(i) for i in (r.get("chunk_ids") or []) if i]
    if ids:
        try:
            rows = chunks.by_ids(ids)
        except Exception as e:  # noqa: BLE001
            log.warning("answer audit: could not read passages: %s", e)
            rows = []
        by_id = {str(c.get("id")): c for c in rows}
        texts = [t for t in (_chunk_text(by_id[i]) for i in ids if i in by_id) if t]
        if texts:
            return texts[:MAX_PASSAGES], "stored"
    docs = [str(d) for d in (r.get("doc_ids") or []) if d]
    if docs:
        try:
            rows = chunks.by_docs(docs)
        except Exception as e:  # noqa: BLE001
            log.warning("answer audit: could not read documents: %s", e)
            rows = []
        texts = [t for t in (_chunk_text(c) for c in rows) if t]
        if texts:
            try:
                try:
                    from services import retrieval  # noqa: PLC0415
                except ImportError:  # pragma: no cover
                    from backend.services import retrieval  # noqa: PLC0415
                order = retrieval.bm25_ranking(f"{question} {reply}", texts)
            except Exception:  # noqa: BLE001
                order = []
            if order:
                return [texts[i] for i in order[:MAX_PASSAGES]], "approx"
    return [], "none"


def _overlap(a: Sequence[str], b: Sequence[str]) -> int:
    return len(set(a) & set(b))


def best_snippet(passages: Sequence[str], sentence: str, width: int = SNIPPET_CHARS) -> str:
    """The part of the passages closest to a sentence (what "the document says" should show)."""
    if not passages:
        return ""
    try:
        try:
            from services import retrieval  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            from backend.services import retrieval  # noqa: PLC0415
        tok = retrieval.tokenize
    except Exception:  # noqa: BLE001
        tok = lambda s: [w.lower() for w in _words(s)]  # noqa: E731
    want = tok(sentence)
    best: Tuple[int, str] = (-1, "")
    for p in passages:
        sents = split_sentences(p) or [p]
        for i, s in enumerate(sents):
            score = _overlap(want, tok(s))
            if score > best[0]:
                text = s
                if len(text) < width // 2 and i + 1 < len(sents):
                    text = f"{s} {sents[i + 1]}"
                best = (score, text)
    text = _norm(best[1] or passages[0])
    return text if len(text) <= width else text[:width - 1].rstrip() + "…"


# -- evaluation ---------------------------------------------------------------------------------------------------------

def _base_row(c: Candidate) -> Dict[str, Any]:
    t = c.turn
    return {"id": str(t["id"]), "turn_id": str(t["id"]), "user_id": c.user_id, "agent_id": c.agent_id,
            "conversation_id": c.conversation_id, "channel": c.channel or "app", "status": STATUS_SKIPPED,
            "skip_reason": None, "score": None, "flagged": False, "question": None, "reply": None,
            "weakest_sentence": None, "passage": None, "passage_source": None, "sentences": [],
            "threshold": None, "model": None, "feedback": None, "feedback_at": None,
            "turn_at": t.get("created_at"), "checked_at": _now()}


def evaluate(c: Candidate, chunks: Any, verifier: Any, thr: Optional[float] = None) -> Dict[str, Any]:
    """The audit row of one candidate turn. Skips need no model; the verifier is called once, for all sentences."""
    thr = threshold() if thr is None else thr
    row = _base_row(c)
    reply = str(c.turn.get("text_content") or "")
    why = skip_reason(c.turn, c.question)
    if why:
        row["skip_reason"] = why
        return row
    if not reply_is_english(reply, c.turn.get("language")):
        row.update(status=STATUS_LANGUAGE, skip_reason="reply is not in English",
                   question=c.question, reply=reply)
        return row
    sentences = checkable_sentences(reply, c.turn)
    if not sentences:
        row["skip_reason"] = "nothing to check"
        return row
    passages, source = passages_for(c.turn, c.question, reply, chunks)
    if not passages:
        row["skip_reason"] = "passages not available"
        return row
    premise = "\n\n".join(passages)[:MAX_PREMISE_CHARS]
    if _latin_ratio(premise) < 0.8:
        row.update(status=STATUS_LANGUAGE, skip_reason="documents are not in English", question=c.question, reply=reply)
        return row
    scores = verifier.score([(premise, s) for s in sentences])
    if len(scores) != len(sentences):
        raise RuntimeError("the verifier returned the wrong number of scores")
    per = [{"text": s, "score": round(float(x), 4)} for s, x in zip(sentences, scores)]
    weakest = min(per, key=lambda p: p["score"])
    row.update(status=STATUS_CHECKED, score=weakest["score"], flagged=weakest["score"] < thr,
               question=c.question, reply=reply, weakest_sentence=weakest["text"],
               passage=best_snippet(passages, weakest["text"]), passage_source=source, sentences=per,
               threshold=thr, model=getattr(verifier, "name", MODEL_LABEL))
    return row


def run_candidates(cands: Sequence[Candidate], chunks_for: Callable[[Candidate], Any], verifier: Any, store_for: Callable[[Candidate], Any],
                   *, progress: Optional[Callable[[Dict[str, int]], None]] = None,
                   should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, int]:
    """Audit candidates one at a time (a process-wide lock: never two audits at once), saving each row as it is made.
    Returns counts {total, done, checked, flagged, skipped, language}. A verifier failure stops the run and raises: nothing
    is stored for the turn that failed, so a later run tries it again."""
    stats = {"total": len(cands), "done": 0, "checked": 0, "flagged": 0, "skipped": 0, "language": 0}
    if progress:
        progress(dict(stats))
    with _job_lock:
        for c in cands:
            if should_stop and should_stop():
                break
            row = evaluate(c, chunks_for(c), verifier)
            store_for(c).upsert(row)
            stats["done"] += 1
            if row["status"] == STATUS_CHECKED:
                stats["checked"] += 1
                stats["flagged"] += 1 if row["flagged"] else 0
            elif row["status"] == STATUS_LANGUAGE:
                stats["language"] += 1
            else:
                stats["skipped"] += 1
            if progress:
                progress(dict(stats))
    return stats


# -- views --------------------------------------------------------------------------------------------------------------

def _mask(text: Any) -> str:
    try:
        try:
            from routers.conversations import mask_text  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            from backend.routers.conversations import mask_text  # noqa: PLC0415
        return mask_text(text)
    except Exception:  # noqa: BLE001
        return kg.redact(str(text or ""))


def public_view(row: dict) -> dict:
    """One flagged answer as the owner sees it. Phone numbers are masked, like on the Calls page."""
    return {
        "turn_id": row.get("turn_id") or row.get("id"), "agent_id": row.get("agent_id"),
        "conversation_id": row.get("conversation_id"), "channel": row.get("channel"),
        "question": _mask(row.get("question")), "reply": _mask(row.get("reply")),
        "weakest_sentence": _mask(row.get("weakest_sentence")), "passage": _mask(row.get("passage")),
        "passage_source": row.get("passage_source"), "score": row.get("score"), "threshold": row.get("threshold"),
        "feedback": row.get("feedback"), "at": row.get("turn_at"), "checked_at": row.get("checked_at"),
    }


def summarize(rows: Sequence[dict]) -> Dict[str, int]:
    checked = [r for r in rows if r.get("status") == STATUS_CHECKED]
    return {
        "checked": len(checked),
        "flagged": sum(1 for r in checked if r.get("flagged") and r.get("feedback") != "fine"),
        "not_checked_language": sum(1 for r in rows if r.get("status") == STATUS_LANGUAGE),
        "marked_fine": sum(1 for r in rows if r.get("feedback") == "fine"),
        "marked_wrong": sum(1 for r in rows if r.get("feedback") == "wrong"),
    }


def flagged_rows(rows: Sequence[dict]) -> List[dict]:
    """Flagged and not marked fine, lowest score first."""
    out = [r for r in rows if r.get("status") == STATUS_CHECKED and r.get("flagged") and r.get("feedback") != "fine"]
    out.sort(key=lambda r: (r.get("score") if r.get("score") is not None else 1.0, str(r.get("turn_at") or "")))
    return out


# -- storage ------------------------------------------------------------------------------------------------------------

def default_path() -> str:
    return os.getenv("BONCLAY_ANSWER_AUDIT_FILE") or os.path.join(_DATA_DIR, "answer_audit.json")


class LocalFileStore:
    """All audit rows in one JSON file (list of rows, keyed by user_id + turn id). Atomic writes, one lock. Every method
    takes the owner's user id: other users' rows are invisible."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_path()
        self._lock = threading.RLock()

    def _read(self) -> List[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            return rows if isinstance(rows, list) else []
        except (OSError, ValueError):
            return []

    def _write(self, rows: List[dict]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rows[-MAX_ROWS:], f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def list(self, user_id: str, agent_id: Optional[str] = None, since_iso: Optional[str] = None) -> List[dict]:
        with self._lock:
            rows = self._read()
        return [r for r in rows if r.get("user_id") == user_id and (agent_id is None or r.get("agent_id") == agent_id)
                and (not since_iso or str(r.get("turn_at") or r.get("checked_at") or "") >= since_iso)]

    def known(self, user_id: Optional[str] = None) -> Set[Tuple[str, str]]:
        with self._lock:
            return {(str(r.get("user_id")), str(r.get("turn_id"))) for r in self._read()
                    if user_id is None or r.get("user_id") == user_id}

    def get(self, user_id: str, turn_id: str) -> Optional[dict]:
        with self._lock:
            return next((r for r in self._read() if r.get("user_id") == user_id and r.get("turn_id") == turn_id), None)

    def upsert(self, row: dict) -> dict:
        with self._lock:
            rows = self._read()
            for i, r in enumerate(rows):
                if r.get("user_id") == row["user_id"] and r.get("turn_id") == row["turn_id"]:
                    rows[i] = {**row, "feedback": r.get("feedback"), "feedback_at": r.get("feedback_at")}  # keep the owner's mark
                    break
            else:
                rows.append(row)
            self._write(rows)
        return row

    def set_feedback(self, user_id: str, turn_id: str, feedback: Optional[str]) -> Optional[dict]:
        with self._lock:
            rows = self._read()
            for r in rows:
                if r.get("user_id") == user_id and r.get("turn_id") == turn_id:
                    r["feedback"] = feedback
                    r["feedback_at"] = _now() if feedback else None
                    self._write(rows)
                    return r
        return None

    def purge_agent(self, user_id: str, agent_id: str) -> int:
        with self._lock:
            rows = self._read()
            keep = [r for r in rows if not (r.get("user_id") == user_id and r.get("agent_id") == agent_id)]
            if len(keep) != len(rows):
                self._write(keep)
            return len(rows) - len(keep)


class SupabaseStore:
    """The hosted twin: table `answer_audit` (migration 019). Same methods; every query filters by user_id. Not exercised
    by the test suite."""

    TABLE = "answer_audit"
    COLUMNS = ("id", "turn_id", "user_id", "agent_id", "conversation_id", "channel", "status", "skip_reason", "score",
               "flagged", "question", "reply", "weakest_sentence", "passage", "passage_source", "sentences", "threshold",
               "model", "feedback", "feedback_at", "turn_at", "checked_at")

    def __init__(self, client: Any):
        self.client = client

    def _t(self) -> Any:
        return self.client.table(self.TABLE)

    def list(self, user_id: str, agent_id: Optional[str] = None, since_iso: Optional[str] = None) -> List[dict]:
        q = self._t().select("*").eq("user_id", user_id)
        if agent_id:
            q = q.eq("agent_id", agent_id)
        if since_iso:
            q = q.gte("turn_at", since_iso)
        return list(q.order("turn_at", desc=True).limit(2000).execute().data or [])

    def known(self, user_id: Optional[str] = None) -> Set[Tuple[str, str]]:
        q = self._t().select("user_id, turn_id")
        if user_id:
            q = q.eq("user_id", user_id)
        since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        rows = q.gte("checked_at", since).limit(10000).execute().data or []
        return {(str(r["user_id"]), str(r["turn_id"])) for r in rows}

    def get(self, user_id: str, turn_id: str) -> Optional[dict]:
        rows = self._t().select("*").eq("user_id", user_id).eq("turn_id", turn_id).limit(1).execute().data or []
        return rows[0] if rows else None

    def upsert(self, row: dict) -> dict:
        old = self.get(row["user_id"], row["turn_id"])
        data = {k: row.get(k) for k in self.COLUMNS if k in row}
        if old:
            data["feedback"], data["feedback_at"] = old.get("feedback"), old.get("feedback_at")
        self._t().upsert(data, on_conflict="id").execute()
        return row

    def set_feedback(self, user_id: str, turn_id: str, feedback: Optional[str]) -> Optional[dict]:
        if self.get(user_id, turn_id) is None:
            return None
        rows = (self._t().update({"feedback": feedback, "feedback_at": _now() if feedback else None})
                .eq("user_id", user_id).eq("turn_id", turn_id).execute().data or [])
        return rows[0] if rows else self.get(user_id, turn_id)

    def purge_agent(self, user_id: str, agent_id: str) -> int:
        return 0   # rows go with the agent (on delete cascade)


_local_store: Optional[LocalFileStore] = None
_store_lock = threading.Lock()
_supabase_getter: Optional[Callable[[], Any]] = None


def configure(*, supabase: Optional[Callable[[], Any]] = None) -> None:
    global _supabase_getter
    _supabase_getter = supabase


def get_store(storage_mode: str = "local") -> Any:
    """The store for a storage mode. Hosted mode without a Supabase client falls back to the local file."""
    global _local_store
    if storage_mode != "local" and _supabase_getter is not None:
        try:
            client = _supabase_getter()
        except Exception:  # noqa: BLE001
            client = None
        if client is not None:
            return SupabaseStore(client)
    with _store_lock:
        path = default_path()
        if _local_store is None or _local_store.path != path:
            _local_store = LocalFileStore(path)
        return _local_store


# -- reading conversations and passages ---------------------------------------------------------------------------------

TURN_COLUMNS = ("id, conversation_id, role, text_content, created_at, fallback, language, timings, retrieval, tool_calls",
                "id, conversation_id, role, text_content, created_at, fallback, language, timings, tool_calls",
                "id, conversation_id, role, text_content, created_at, fallback")


def load_local(db: Any, user_id: Optional[str], agent_id: Optional[str], since_iso: str,
               limit: int = 500) -> List[Tuple[dict, List[dict]]]:
    """(conversation, turns oldest first) of the local store: one user's (or everyone's when user_id is None)."""
    def keep(r: dict) -> bool:
        return ((user_id is None or r.get("user_id") == user_id) and (agent_id is None or r.get("agent_id") == agent_id)
                and str(r.get("created_at") or "") >= since_iso)

    convs = db.select("conversations", keep, order="created_at", desc=True, limit=limit)
    ids = {c["id"] for c in convs}
    turns: Dict[str, List[dict]] = {}
    for t in db.select("conversation_turns", lambda r: r.get("conversation_id") in ids, order="created_at"):
        turns.setdefault(t["conversation_id"], []).append(t)
    return [(c, turns.get(c["id"], [])) for c in convs]


def load_cloud(sb: Any, user_id: Optional[str], agent_id: Optional[str], since_iso: str,
               limit: int = 500) -> List[Tuple[dict, List[dict]]]:
    tiers = ("id, agent_id, user_id, started_at, source, is_test", "id, agent_id, user_id, started_at")
    rows: List[dict] = []
    for n, cols in enumerate(tiers):
        try:
            q = sb.table("conversations").select(cols).gte("started_at", since_iso)
            if user_id:
                q = q.eq("user_id", user_id)
            if agent_id:
                q = q.eq("agent_id", agent_id)
            rows = q.order("started_at", desc=True).limit(limit).execute().data or []
            break
        except Exception:  # noqa: BLE001
            if n == len(tiers) - 1:
                raise
    convs = [{**r, "created_at": r.get("started_at")} for r in rows]
    by_conv: Dict[str, List[dict]] = {}
    ids = [c["id"] for c in convs]
    for i in range(0, len(ids), 100):
        got: List[dict] = []
        for n, cols in enumerate(TURN_COLUMNS):
            try:
                got = (sb.table("conversation_turns").select(cols).in_("conversation_id", ids[i:i + 100])
                       .order("created_at").execute().data or [])
                break
            except Exception:  # noqa: BLE001
                if n == len(TURN_COLUMNS) - 1:
                    raise
        for t in got:
            by_conv.setdefault(t["conversation_id"], []).append(t)
    return [(c, by_conv.get(c["id"], [])) for c in convs]


class LocalChunks:
    def __init__(self, db: Any):
        self.db = db

    def by_ids(self, ids: Sequence[str]) -> List[dict]:
        want = set(ids)
        return self.db.select("document_chunks", lambda r: r.get("id") in want)

    def by_docs(self, doc_ids: Sequence[str]) -> List[dict]:
        want = set(doc_ids)
        rows = self.db.select("document_chunks", lambda r: r.get("document_id") in want, order="chunk_index")
        return [r for r in rows if (r.get("kind") or "passage") == "passage"][:300]


class CloudChunks:
    def __init__(self, sb: Any):
        self.sb = sb

    def by_ids(self, ids: Sequence[str]) -> List[dict]:
        return self.sb.table("document_chunks").select("id, document_id, chunk_text").in_("id", list(ids)).execute().data or []

    def by_docs(self, doc_ids: Sequence[str]) -> List[dict]:
        return (self.sb.table("document_chunks").select("id, document_id, chunk_text").in_("document_id", list(doc_ids))
                .limit(300).execute().data or [])
