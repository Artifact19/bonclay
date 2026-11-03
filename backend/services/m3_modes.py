"""BGE-M3's lexical (sparse) weights and multi-vector (ColBERT) vectors, next to its dense vector.

BGE-M3 (Chen et al. 2024, arXiv 2402.03216) produces three outputs from ONE forward pass: the dense vector (the [CLS]
hidden state, what services/retrieval.py already uses), a weight per token (`sparse_linear`, a 1024 -> 1 layer, ReLU)
and a vector per token (`colbert_linear`, 1024 -> 1024, L2-normalised; relevance = MaxSim). The two heads ship as
`sparse_linear.pt` and `colbert_linear.pt` in the BAAI/bge-m3 repository; sentence-transformers does not load them,
so this module applies them to the token embeddings of the same forward pass. On MIRACL (incl. its Hindi split) the
paper's dense + sparse + multi-vector combination (1 : 0.3 : 1) beats dense alone by 2.3 nDCG@10 on average and 3.8
on Hindi.

Query side (live turn): encode_query runs the shared embedding model once with every output (no second model, no
second pass) and returns the dense vector (identical to retrieval.encode_texts) and a QueryM3 (word weights + token
vectors). Passage side: the weights and vectors of an agent's passages are computed lazily in a background thread the
first time the agent is searched, cached in memory (LRU, M3_CACHE_MB) and on disk as one small .npz per passage
(M3_CACHE_DIR, keyed by chunk id + a hash of the text, so an edited passage is recomputed; never inside the user's JSON
tables). Until every candidate of a search has them, that search fuses dense + BM25 only.

Sparse weights are kept per WORD, keyed by the Hinglish/Devanagari normaliser of services/retrieval.py (the word's
weight is the largest of its sub-word pieces), so kitni / kitne / कितनी and fees / फीस share a key, like in BM25.

Switches: M3_MODES=on|off (default on; inert while the head files are missing). The head files are never downloaded
by the app; fetch them once (about 2 MB, from the same repository as the model) with

    backend/.venv/bin/python -m services.m3_modes --download-heads

Nothing heavy is imported at module level; torch is imported where the heads are used.
"""
from __future__ import annotations

import bisect
import gc
import hashlib
import logging
import os
import queue
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    from services import retrieval
except ImportError:  # imported as backend.services.m3_modes
    from backend.services import retrieval  # type: ignore

log = logging.getLogger(__name__)

HEAD_FILES = ("sparse_linear.pt", "colbert_linear.pt")
HF_REPO = "BAAI/bge-m3"
DEFAULT_CACHE_MB = 256          # passage vectors kept in memory (a 50-page document is about 80 MB)
DEFAULT_DISK_MB = 2048          # sidecar files kept on disk before the oldest are removed
BULK_NEED_MB = 2600             # a temporary float32 copy of BGE-M3 for passages (see retrieval._encode_bulk_fp32)
PASSAGE_BATCH = 8
PAUSE_MS = 30                   # pause between background batches, so a live turn gets the CPU


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def enabled() -> bool:
    return (os.getenv("M3_MODES", "on") or "on").strip().lower() not in ("off", "0", "false", "no")


def is_m3_model(model_name: str) -> bool:
    return "bge-m3" in (model_name or "").lower()


# ------------------------------------------------------------------------------------------------ heads

def _head_candidates(model_name: str) -> Iterator[str]:
    custom = (os.getenv("M3_HEADS_DIR") or "").strip()
    if custom:
        yield custom
    yield retrieval.fp16_cache_dir(model_name)
    if os.path.isdir(model_name):
        yield model_name


_heads_paths_cache: Dict[str, Optional[Tuple[str, str]]] = {}


def heads_paths(model_name: str) -> Optional[Tuple[str, str]]:
    """Paths of (sparse_linear.pt, colbert_linear.pt) on disk, or None. Looks in M3_HEADS_DIR, next to the float16 copy
    of the model, and in the Hugging Face cache (local files only: never downloads)."""
    if model_name in _heads_paths_cache:
        return _heads_paths_cache[model_name]
    found: Optional[Tuple[str, str]] = None
    for d in _head_candidates(model_name):
        paths = tuple(os.path.join(d, f) for f in HEAD_FILES)
        if all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in paths):
            found = paths  # type: ignore[assignment]
            break
    if found is None:
        try:
            from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
            hits = [try_to_load_from_cache(model_name, f) for f in HEAD_FILES]
            if all(isinstance(h, str) and os.path.isfile(h) for h in hits):
                found = (hits[0], hits[1])  # type: ignore[assignment]
        except Exception:  # noqa: BLE001
            pass
    _heads_paths_cache[model_name] = found
    return found


def heads_available(model_name: str) -> bool:
    return heads_paths(model_name) is not None


_failed: Dict[str, str] = {}


def active(model_name: str) -> bool:
    """M3 scoring is used for this embedding model: switched on, a BGE-M3 model, head files present, not failed."""
    return enabled() and is_m3_model(model_name) and model_name not in _failed and heads_available(model_name)


@dataclass
class Heads:
    sparse: Callable          # (n, d) float32 tensor -> (n,) token weights (ReLU applied)
    colbert: Callable         # (n, d) float32 tensor -> (n, d') unit vectors
    dim: int


_heads: Dict[str, Heads] = {}
_heads_lock = threading.Lock()


def _linear_from_state(path: str):
    import torch  # noqa: PLC0415
    state = torch.load(path, map_location="cpu", weights_only=True)   # weights only: no pickled code is run
    weight, bias = state["weight"].float(), state.get("bias")
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            layer.bias.copy_(bias.float())
    layer.eval()
    return layer


def load_heads(model_name: str) -> Heads:
    with _heads_lock:
        heads = _heads.get(model_name)
        if heads is None:
            paths = heads_paths(model_name)
            if paths is None:
                raise FileNotFoundError(f"BGE-M3 head files {HEAD_FILES} not found for {model_name}")
            import torch  # noqa: PLC0415
            sparse_layer, colbert_layer = _linear_from_state(paths[0]), _linear_from_state(paths[1])

            def sparse(h):
                with torch.no_grad():
                    return torch.relu(sparse_layer(h)).squeeze(-1)

            def colbert(h):
                with torch.no_grad():
                    return torch.nn.functional.normalize(colbert_layer(h), dim=-1)

            heads = Heads(sparse=sparse, colbert=colbert, dim=int(colbert_layer.weight.shape[0]))
            _heads[model_name] = heads
        return heads


def download_heads(model_name: str = HF_REPO) -> Tuple[str, str]:
    """Fetch the two head files into the Hugging Face cache (explicit, from the command line only)."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    paths = tuple(hf_hub_download(model_name, f) for f in HEAD_FILES)
    _heads_paths_cache.pop(model_name, None)
    return paths  # type: ignore[return-value]


# ------------------------------------------------------------------------------------------------ representations

@dataclass
class PassageM3:
    sparse: Dict[str, float]          # normalised word key -> weight
    colbert: np.ndarray               # (tokens, dim) float16, unit rows

    @property
    def nbytes(self) -> int:
        return int(self.colbert.nbytes) + 64 * len(self.sparse) + 200


_SPAN_RE = re.compile(r"[ऀ-ॣ॰-ॿ‌‍]+|[A-Za-zÀ-ɏ]+|\d+(?:,\d{2,})*")


def word_weights(text: str, offsets: Optional[Sequence[Tuple[int, int]]], ids: Sequence[int],
                 weights: Sequence[float], special_ids: Iterable[int]) -> Dict[str, float]:
    """Token weights -> {normalised word key: weight}: a word's weight is the largest weight of the sub-word pieces
    that overlap it; the key is retrieval's normaliser of the word (stopwords dropped). Without offsets (a tokenizer
    that cannot give them) the pieces are keyed by token id, like the paper."""
    special = set(special_ids)
    if offsets is None:
        out: Dict[str, float] = {}
        for tid, w in zip(ids, weights):
            if tid not in special and w > 0:
                key = f"#{tid}"
                out[key] = max(out.get(key, 0.0), float(w))
        return out
    spans = [(m.start(), m.end(), m.group(0)) for m in _SPAN_RE.finditer(text)]
    if not spans:
        return {}
    starts = [s for s, _, _ in spans]
    span_w = [0.0] * len(spans)
    for (a, b), tid, w in zip(offsets, ids, weights):
        if tid in special or b <= a or w <= 0:
            continue
        k = bisect.bisect_right(starts, a) - 1
        # every word the piece overlaps (a piece may start on the space before a word)
        k = max(k, 0)
        while k < len(spans) and spans[k][0] < b:
            if spans[k][1] > a:
                span_w[k] = max(span_w[k], float(w))
            k += 1
    out = {}
    for (_, _, word), w in zip(spans, span_w):
        if w <= 0:
            continue
        for key in retrieval.content_tokens(word):
            out[key] = max(out.get(key, 0.0), w)
    return out


def _special_ids(tokenizer) -> List[int]:
    ids = [getattr(tokenizer, n, None) for n in ("cls_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
                                                  "bos_token_id", "sep_token_id")]
    return [i for i in ids if isinstance(i, int)]


def _offsets(tokenizer, text: str, ids: Sequence[int], max_length: int) -> Optional[List[Tuple[int, int]]]:
    try:
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length,
                        add_special_tokens=True)
    except Exception:  # noqa: BLE001 - a slow tokenizer has no offsets
        return None
    if list(enc["input_ids"]) != list(ids):
        return None
    return [tuple(o) for o in enc["offset_mapping"]]  # type: ignore[misc]


def reps_from_output(out: dict, text: str, heads: Heads, tokenizer, max_length: int) -> Tuple[np.ndarray, PassageM3]:
    """(unit dense vector float32, PassageM3) from one sentence-transformers output (encode(output_value=None))."""
    h = out["token_embeddings"].float().cpu()
    ids = [int(i) for i in out["input_ids"].tolist()]
    w = heads.sparse(h).numpy()
    vecs = heads.colbert(h[1:]).numpy().astype(np.float16)            # [CLS] excluded, like FlagEmbedding
    dense = out["sentence_embedding"].float().cpu().numpy().astype(np.float32)
    dense = dense / max(float(np.linalg.norm(dense)), 1e-12)
    sparse = word_weights(text, _offsets(tokenizer, text, ids, max_length), ids, w.tolist(), _special_ids(tokenizer))
    return dense, PassageM3(sparse=sparse, colbert=vecs)


def _st_max_length(model) -> int:
    try:
        return int(model.max_seq_length or 512)
    except Exception:  # noqa: BLE001
        return 512


def forward_outputs(model, texts: Sequence[str]) -> List[dict]:
    """Per-text {"token_embeddings", "input_ids", "sentence_embedding"} for a batch, from one forward pass of a
    sentence-transformers model (encode(output_value=None) cannot batch in sentence-transformers 5.x: it indexes the
    string `modality` field). Padding is on the right, so each text's tokens are its first attention_mask.sum()."""
    import torch  # noqa: PLC0415
    texts = list(texts)
    try:
        prep = getattr(model, "preprocess", None)
        features = prep(texts) if prep is not None else model.tokenize(texts)
    except Exception:  # noqa: BLE001 - older/newer API
        features = model.tokenize(texts)
    device = getattr(model, "device", None)
    features = {k: (v.to(device) if device is not None and hasattr(v, "to") else v) for k, v in features.items()}
    with torch.no_grad():
        out = model.forward(features)
    ids = out.get("input_ids", features["input_ids"])
    items = []
    for i in range(len(texts)):
        n = int(out["attention_mask"][i].sum())
        items.append({"token_embeddings": out["token_embeddings"][i, :n], "input_ids": ids[i, :n],
                      "sentence_embedding": out["sentence_embedding"][i]})
    return items


def encode_query(model_name: str, text: str) -> Tuple[np.ndarray, Optional["QueryM3"]]:
    """The question's dense vector (identical to retrieval.encode_texts(model_name, [text])[0]) and, when M3 is
    active, its QueryM3, from one forward pass of the shared model. On any M3 problem: the dense vector and None."""
    if not active(model_name):
        return np.asarray(retrieval.encode_texts(model_name, [text])[0], dtype=np.float32), None
    try:
        heads = load_heads(model_name)
        model = retrieval.load_sentence_transformer(model_name)
        outs = model.encode([text], output_value=None, normalize_embeddings=False, show_progress_bar=False,
                            batch_size=1)
        dense, rep = reps_from_output(outs[0], text, heads, model.tokenizer, _st_max_length(model))
        return dense, QueryM3(sparse=rep.sparse, colbert=rep.colbert.astype(np.float32), store=get_store(model_name))
    except Exception as e:  # noqa: BLE001 - the dense search must work without M3
        _failed[model_name] = f"{type(e).__name__}: {e}"[:200]
        log.warning("BGE-M3 sparse/multi-vector query encoding failed, using dense + BM25 only: %s", _failed[model_name])
        return np.asarray(retrieval.encode_texts(model_name, [text])[0], dtype=np.float32), None


def _memory_allows(need_mb: int) -> bool:
    try:
        from core import memory_guard  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        try:
            from backend.core import memory_guard  # type: ignore  # noqa: PLC0415
        except ImportError:
            return True
    if not memory_guard.enabled():
        return True
    avail = memory_guard.available_mb()
    return avail is None or avail >= need_mb + memory_guard.reserve_mb()


def encode_passages(model_name: str, texts: Sequence[str], batch: int = PASSAGE_BATCH) -> Iterator[List[PassageM3]]:
    """PassageM3 of `texts`, yielded batch by batch. On a CPU with the float16 model resident, passages go through a
    temporary float32 copy (fp16 is several times slower than fp32 for passage-length text on CPU, see
    retrieval._encode_bulk_fp32) when memory_guard says there is room; otherwise through the resident model."""
    heads = load_heads(model_name)
    resident = retrieval.load_sentence_transformer(model_name)
    model, temp = resident, None
    half = model_name in retrieval._half_models
    if half and retrieval.embedding_device().split(":")[0] == "cpu" and len(texts) > 2 \
            and _memory_allows(_env_int("M3_BULK_NEED_MB", BULK_NEED_MB)):
        import torch  # noqa: PLC0415
        retrieval._bulk_lock.acquire()
        try:
            source = retrieval.fp16_cache_dir(model_name) if retrieval.fp16_copy_ready(model_name) else model_name
            temp = retrieval._new_sentence_transformer(source, torch.float32)
            temp.max_seq_length = _st_max_length(resident)
            model = temp
        except Exception as e:  # noqa: BLE001
            log.warning("float32 copy for M3 passages unavailable (%s); using the resident model", e)
            retrieval._bulk_lock.release()
            temp, model = None, resident
    try:
        max_len = _st_max_length(model)
        for i in range(0, len(texts), batch):
            part = list(texts[i:i + batch])
            outs = forward_outputs(model, part)
            yield [reps_from_output(o, t, heads, model.tokenizer, max_len)[1] for o, t in zip(outs, part)]
    finally:
        if temp is not None:
            del temp, model
            gc.collect()
            retrieval._bulk_lock.release()


# ------------------------------------------------------------------------------------------------ scoring

def sparse_score(query: Dict[str, float], passage: Dict[str, float]) -> float:
    """BGE-M3 lexical matching score: sum over shared keys of query weight x passage weight."""
    if len(passage) < len(query):
        return float(sum(w * query[k] for k, w in passage.items() if k in query))
    return float(sum(w * passage[k] for k, w in query.items() if k in passage))


def maxsim_scores(query: np.ndarray, passages: Sequence[np.ndarray]) -> List[float]:
    """ColBERT late interaction: for each passage, the mean over query tokens of the best dot product with any passage
    token. One matrix product over the concatenated candidates (float16 in torch when available: ~2 ms for 40
    passages of 200 tokens on an M4)."""
    if not passages:
        return []
    lengths = [int(p.shape[0]) for p in passages]
    nq = max(int(query.shape[0]), 1)
    try:
        import torch  # noqa: PLC0415
        P = torch.from_numpy(np.ascontiguousarray(np.concatenate(passages).astype(np.float16, copy=False)))
        Q = torch.from_numpy(np.ascontiguousarray(query)).to(torch.float16)
        S = (Q @ P.T).float().numpy()
    except Exception:  # noqa: BLE001 - numpy fallback
        S = query.astype(np.float32) @ np.concatenate(passages).astype(np.float32).T
    out, start = [], 0
    for n in lengths:
        out.append(float(S[:, start:start + n].max(axis=1).sum() / nq) if n else 0.0)
        start += n
    return out


@dataclass
class QueryM3:
    sparse: Dict[str, float]
    colbert: np.ndarray               # (tokens, dim) float32, unit rows
    store: "M3Store" = field(repr=False, default=None)  # type: ignore[assignment]

    def score_rows(self, rows: Sequence[dict], texts: Sequence[str], all_rows: Optional[Sequence[dict]] = None,
                   all_texts: Optional[Sequence[str]] = None) -> Optional[Dict[str, List[Optional[float]]]]:
        """{"sparse": [...], "colbert": [...]} for the candidate rows, or None when any of them has no passage vectors
        yet (then the passages of the whole search, `all_rows`, are queued for the background worker)."""
        keys = [self.store.key(r.get("id"), t) for r, t in zip(rows, texts)]
        reps = self.store.get_many(keys)
        if any(r is None for r in reps):
            pool_rows = all_rows if all_rows is not None else rows
            pool_texts = all_texts if all_texts is not None else texts
            self.store.schedule((self.store.key(r.get("id"), t), t) for r, t in zip(pool_rows, pool_texts))
            return None
        return {"sparse": [sparse_score(self.sparse, r.sparse) for r in reps],  # type: ignore[union-attr]
                "colbert": maxsim_scores(self.colbert, [r.colbert for r in reps])}  # type: ignore[union-attr]


# ------------------------------------------------------------------------------------------------ passage cache

def default_cache_dir(model_name: str) -> str:
    base = (os.getenv("M3_CACHE_DIR") or "").strip() or os.path.join(retrieval._DATA_DIR, "m3_cache")
    return os.path.join(base, re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip("/")) or "model")


class M3Store:
    """Passage weights and vectors: memory LRU (cap in MB) over one .npz sidecar per passage on disk, plus a
    background worker that computes missing ones. `encoder(texts)` yields lists of PassageM3 (encode_passages)."""

    def __init__(self, root: str, model_name: str, encoder: Optional[Callable[[Sequence[str]], Iterator[List[PassageM3]]]] = None,
                 mem_mb: Optional[int] = None, disk_mb: Optional[int] = None, pause_ms: Optional[int] = None):
        self.root, self.model_name = root, model_name
        self.encoder = encoder or (lambda texts: encode_passages(model_name, texts))
        self.mem_cap = (mem_mb if mem_mb is not None else _env_int("M3_CACHE_MB", DEFAULT_CACHE_MB)) * 2 ** 20
        self.disk_cap = (disk_mb if disk_mb is not None else _env_int("M3_DISK_MB", DEFAULT_DISK_MB)) * 2 ** 20
        self.pause = (pause_ms if pause_ms is not None else _env_int("M3_PAUSE_MS", PAUSE_MS)) / 1000.0
        self._mem: "OrderedDict[str, PassageM3]" = OrderedDict()
        self._mem_bytes = 0
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._pending: set = set()
        self._worker: Optional[threading.Thread] = None
        self._missing_on_disk: set = set()
        self.computed = 0
        self.last_error: Optional[str] = None

    # -- keys and files
    def key(self, chunk_id, text: str) -> str:
        h = hashlib.sha1(f"{self.model_name}\x00{chunk_id or ''}\x00{text}".encode("utf-8")).hexdigest()
        return h[:32]

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key[:2], key + ".npz")

    # -- memory LRU
    def _remember(self, key: str, rep: PassageM3) -> None:
        with self._lock:
            old = self._mem.pop(key, None)
            if old is not None:
                self._mem_bytes -= old.nbytes
            self._mem[key] = rep
            self._mem_bytes += rep.nbytes
            while self._mem_bytes > self.mem_cap and len(self._mem) > 1:
                _, dropped = self._mem.popitem(last=False)
                self._mem_bytes -= dropped.nbytes

    @property
    def memory_bytes(self) -> int:
        return self._mem_bytes

    def get(self, key: str) -> Optional[PassageM3]:
        with self._lock:
            rep = self._mem.get(key)
            if rep is not None:
                self._mem.move_to_end(key)
                return rep
        path = self._path(key)
        if key in self._missing_on_disk and key not in self._pending:
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                rep = PassageM3(sparse=dict(zip(z["sk"].tolist(), z["sv"].astype(float).tolist())),
                                colbert=np.array(z["colbert"], dtype=np.float16))
        except FileNotFoundError:
            self._missing_on_disk.add(key)
            return None
        except Exception as e:  # noqa: BLE001 - a damaged sidecar is recomputed
            log.warning("unreadable M3 sidecar %s (%s); recomputing", path, e)
            try:
                os.remove(path)
            except OSError:
                pass
            return None
        self._remember(key, rep)
        return rep

    def get_many(self, keys: Sequence[str]) -> List[Optional[PassageM3]]:
        return [self.get(k) for k in keys]

    def put(self, key: str, rep: PassageM3) -> None:
        path = self._path(key)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            keys = list(rep.sparse)
            with open(tmp, "wb") as f:
                np.savez(f, colbert=rep.colbert.astype(np.float16),
                         sk=np.asarray(keys, dtype=np.str_) if keys else np.zeros(0, dtype="<U1"),
                         sv=np.asarray([rep.sparse[k] for k in keys], dtype=np.float32))
            os.replace(tmp, path)
        except OSError as e:   # a read-only or full disk: keep it in memory only
            log.warning("could not write M3 sidecar %s: %s", path, e)
        self._missing_on_disk.discard(key)
        self._remember(key, rep)

    # -- background computation
    def schedule(self, items: Iterable[Tuple[str, str]]) -> int:
        """Queue (key, text) pairs that have no vectors yet; returns how many were queued."""
        n = 0
        with self._lock:
            for key, text in items:
                if key in self._pending or key in self._mem:
                    continue
                if os.path.exists(self._path(key)):
                    continue
                self._pending.add(key)
                self._queue.put((key, text))
                n += 1
            if n and (self._worker is None or not self._worker.is_alive()):
                self._worker = threading.Thread(target=self._drain, name="m3-passages", daemon=True)
                self._worker.start()
        return n

    def _drain(self) -> None:
        while True:
            items: List[Tuple[str, str]] = []
            try:
                while len(items) < 512:
                    items.append(self._queue.get_nowait())
            except queue.Empty:
                pass
            if not items:
                return
            t = time.perf_counter()
            done = 0
            try:
                it = iter(items)
                gen = self.encoder([text for _, text in items])
                for reps in gen:
                    for rep in reps:
                        key, _ = next(it)
                        self.put(key, rep)
                        with self._lock:
                            self._pending.discard(key)
                        done += 1
                    if self.pause:
                        time.sleep(self.pause)
            except Exception as e:  # noqa: BLE001 - leave the rest for the next search to queue again
                self.last_error = f"{type(e).__name__}: {e}"[:200]
                log.warning("M3 passage vectors failed after %d of %d: %s", done, len(items), self.last_error)
            finally:
                with self._lock:
                    for key, _ in items:
                        self._pending.discard(key)
            self.computed += done
            if done:
                log.info("M3 passage vectors: %d computed in %.1f s", done, time.perf_counter() - t)
                self.prune()

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until the background worker has nothing left (tests and the bench)."""
        end = time.time() + timeout
        while time.time() < end:
            w = self._worker
            if not self._pending and (w is None or not w.is_alive()):
                return True
            time.sleep(0.01)
        return False

    def compute_now(self, items: Sequence[Tuple[str, str]]) -> int:
        """Compute missing vectors synchronously (bench / warm-up). Returns how many were computed."""
        todo = [(k, t) for k, t in items if self.get(k) is None]
        if not todo:
            return 0
        it = iter(todo)
        n = 0
        for reps in self.encoder([t for _, t in todo]):
            for rep in reps:
                key, _ = next(it)
                self.put(key, rep)
                n += 1
        return n

    def prune(self) -> int:
        """Remove the oldest sidecar files while the folder is over M3_DISK_MB. Returns how many were removed."""
        files = []
        total = 0
        for dirpath, _, names in os.walk(self.root):
            for name in names:
                if name.endswith(".npz"):
                    p = os.path.join(dirpath, name)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    files.append((st.st_mtime, st.st_size, p))
                    total += st.st_size
        removed = 0
        if total <= self.disk_cap:
            return 0
        for _, size, p in sorted(files):
            if total <= self.disk_cap * 0.9:
                break
            try:
                os.remove(p)
                total -= size
                removed += 1
            except OSError:
                pass
        return removed


_stores: Dict[str, M3Store] = {}
_stores_lock = threading.Lock()


def get_store(model_name: str) -> M3Store:
    with _stores_lock:
        store = _stores.get(model_name)
        if store is None:
            store = M3Store(default_cache_dir(model_name), model_name)
            _stores[model_name] = store
        return store


def status(model_name: str) -> dict:
    """For logs and the warm-up report."""
    store = _stores.get(model_name)
    return {"enabled": enabled(), "model": model_name, "heads": heads_available(model_name),
            "failed": _failed.get(model_name), "cached_mb": round(store.memory_bytes / 2 ** 20, 1) if store else 0.0,
            "computed": store.computed if store else 0}


if __name__ == "__main__":   # backend/.venv/bin/python -m services.m3_modes --download-heads
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="BGE-M3 sparse / multi-vector heads")
    ap.add_argument("--download-heads", action="store_true", help=f"fetch {HEAD_FILES} (~2 MB) from {HF_REPO}")
    ap.add_argument("--model", default=retrieval.embedding_model_name())
    args = ap.parse_args()
    if args.download_heads:
        print("downloaded:", download_heads(args.model))
    print(status(args.model))
    sys.exit(0)
