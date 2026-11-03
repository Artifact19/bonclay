"""BGE-M3 sparse + multi-vector modes (services/m3_modes.py) with a tiny fake sentence-transformers model and fake head
files (no real model): one forward pass gives the dense vector, word weights and token vectors; word weights use the
Hinglish/Devanagari normaliser; passage vectors are computed in the background, cached in memory under a cap and on
disk as sidecars keyed by chunk id + text; rank_rows fuses them in only when every candidate has them.

The one slow test with the real BGE-M3 runs only with BONCLAY_SLOW_TESTS=1 and when the head files are on disk."""
import os
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from services import m3_modes, retrieval  # noqa: E402

DIM = 8
FAKE_NAME = "test/bge-m3-fake"


@pytest.fixture(autouse=True)
def _m3_on(monkeypatch):
    """conftest switches M3_MODES off for the whole suite; this file tests it (with fake heads and a fake model)."""
    monkeypatch.delenv("M3_MODES", raising=False)


class FakeTokenizer:
    """Whitespace/punctuation tokenizer with offsets: [CLS]=0, words, [EOS]=2."""
    cls_token_id, eos_token_id, pad_token_id, unk_token_id = 0, 2, 1, 3

    def _pieces(self, text):
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    @staticmethod
    def _id(word):
        return 10 + (sum(ord(c) for c in word) % 500)

    def __call__(self, text, return_offsets_mapping=False, **kw):
        spans = self._pieces(text)
        ids = [0] + [self._id(text[a:b]) for a, b in spans] + [2]
        enc = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            enc["offset_mapping"] = [(0, 0)] + spans + [(0, 0)]
        return enc


def _token_vec(tid):
    rng = np.random.default_rng(tid)
    v = rng.normal(size=DIM).astype(np.float32)
    v[0] = (tid % 7) / 7.0 + 0.05            # feature 0 drives the fake sparse weight
    return v


class FakeST:
    """encode(output_value=None) for one text, and preprocess + forward for batches, like sentence-transformers 5."""
    max_seq_length = 64
    device = "cpu"

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.forward_calls = 0
        self.encode_calls = 0

    def _out(self, text):
        ids = self.tokenizer(text)["input_ids"]
        h = torch.tensor(np.stack([_token_vec(i) for i in ids]))
        return {"input_ids": torch.tensor(ids), "token_embeddings": h.half(), "sentence_embedding": h[0].half()}

    def encode(self, texts, output_value="sentence_embedding", **kw):
        self.encode_calls += 1
        outs = [self._out(t) for t in texts]
        if output_value is None:
            return outs
        return np.stack([o["sentence_embedding"].float().numpy() for o in outs])

    def preprocess(self, texts):
        outs = [self._out(t) for t in texts]
        n = max(len(o["input_ids"]) for o in outs)
        ids = torch.ones((len(texts), n), dtype=torch.long)
        mask = torch.zeros((len(texts), n), dtype=torch.long)
        for i, o in enumerate(outs):
            ids[i, :len(o["input_ids"])] = o["input_ids"]
            mask[i, :len(o["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask, "modality": "text"}

    def forward(self, features):
        self.forward_calls += 1
        ids = features["input_ids"]
        h = torch.tensor(np.stack([[_token_vec(int(t)) for t in row] for row in ids])).half()
        return {**features, "token_embeddings": h, "sentence_embedding": h[:, 0]}


@pytest.fixture()
def fake_m3(monkeypatch, tmp_path):
    """Fake heads on disk (sparse = ReLU(feature 0), colbert = identity) and a fake model under a BGE-M3-like name."""
    heads = tmp_path / "heads"
    heads.mkdir()
    sw = torch.zeros(1, DIM)
    sw[0, 0] = 1.0
    torch.save({"weight": sw, "bias": torch.zeros(1)}, heads / "sparse_linear.pt")
    torch.save({"weight": torch.eye(DIM), "bias": torch.zeros(DIM)}, heads / "colbert_linear.pt")
    monkeypatch.setenv("M3_HEADS_DIR", str(heads))
    monkeypatch.setenv("M3_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("M3_PAUSE_MS", "0")
    model = FakeST()
    monkeypatch.setattr(retrieval, "load_sentence_transformer", lambda name: model)
    monkeypatch.setattr(retrieval, "encode_texts", lambda name, texts, **k: np.stack([
        (lambda v: v / np.linalg.norm(v))(model._out(t)["sentence_embedding"].float().numpy()) for t in texts]))
    for cache in (m3_modes._heads_paths_cache, m3_modes._heads, m3_modes._failed, m3_modes._stores):
        cache.clear()
    yield model
    for store in list(m3_modes._stores.values()):
        store.wait_idle(5)
    for cache in (m3_modes._heads_paths_cache, m3_modes._heads, m3_modes._failed, m3_modes._stores):
        cache.clear()


# ------------------------------------------------------------------------------------------------ switches and heads

def test_inactive_without_heads_or_when_switched_off(monkeypatch, tmp_path):
    m3_modes._heads_paths_cache.clear()
    monkeypatch.setenv("M3_HEADS_DIR", str(tmp_path))                 # empty folder
    monkeypatch.setattr(retrieval, "fp16_cache_dir", lambda name: str(tmp_path / "none"))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, f: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("must never download"))
    assert m3_modes.heads_paths("BAAI/bge-m3") is None and not m3_modes.active("BAAI/bge-m3")
    m3_modes._heads_paths_cache.clear()
    assert not m3_modes.active(retrieval.LEGACY_EMBEDDING_MODEL)       # not a BGE-M3 model


def test_switch_and_model_name(fake_m3, monkeypatch):
    assert m3_modes.active(FAKE_NAME)
    monkeypatch.setenv("M3_MODES", "off")
    assert not m3_modes.active(FAKE_NAME)
    assert not m3_modes.is_m3_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def test_heads_load_with_weights_only(fake_m3):
    h = m3_modes.load_heads(FAKE_NAME)
    x = torch.tensor([[0.5] + [0.0] * (DIM - 1), [-0.3] + [1.0] * (DIM - 1)])
    assert h.sparse(x).tolist() == pytest.approx([0.5, 0.0])          # ReLU
    assert torch.allclose(h.colbert(x).norm(dim=-1), torch.ones(2))   # unit vectors
    assert h.dim == DIM and m3_modes.load_heads(FAKE_NAME) is h


# ------------------------------------------------------------------------------------------------ query side

def test_query_is_one_forward_pass_and_dense_matches_encode_texts(fake_m3):
    dense, qm3 = m3_modes.encode_query(FAKE_NAME, "Doctor ki fees kitni hai?")
    assert fake_m3.encode_calls == 1 and fake_m3.forward_calls == 0
    ref = retrieval.encode_texts(FAKE_NAME, ["Doctor ki fees kitni hai?"])[0]
    assert np.allclose(dense, ref, atol=1e-3) and abs(np.linalg.norm(dense) - 1) < 1e-5
    assert qm3 is not None and qm3.colbert.shape == (len("Doctor ki fees kitni hai?".split()) + 1, DIM)
    assert set(qm3.sparse) == {"doctor", retrieval.normalise_token("fees")}   # stopwords (ki, kitni, hai) dropped
    assert np.allclose(np.linalg.norm(qm3.colbert, axis=1), 1.0, atol=1e-2)


def test_query_falls_back_to_dense_when_m3_fails(fake_m3, monkeypatch):
    monkeypatch.setattr(m3_modes, "load_heads", lambda name: (_ for _ in ()).throw(RuntimeError("bad heads")))
    dense, qm3 = m3_modes.encode_query(FAKE_NAME, "hello")
    assert qm3 is None and dense.shape == (DIM,)
    assert not m3_modes.active(FAKE_NAME) and "bad heads" in m3_modes.status(FAKE_NAME)["failed"]


def test_query_dense_only_when_inactive(fake_m3, monkeypatch):
    monkeypatch.setenv("M3_MODES", "off")
    dense, qm3 = m3_modes.encode_query(FAKE_NAME, "hello")
    assert qm3 is None and fake_m3.encode_calls == 0


# ------------------------------------------------------------------------------------------------ word weights

def test_word_weights_take_the_max_piece_and_normalise_keys():
    text = "कितनी fees? Dr. Sharma"
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    # two pieces inside "fees?" (0.2 and 0.6), one for each other word
    offsets = [(0, 0), spans[0], (spans[1][0], spans[1][0] + 2), (spans[1][0] + 2, spans[1][1]), spans[2], spans[3], (0, 0)]
    ids = [0, 11, 12, 13, 14, 15, 2]
    w = m3_modes.word_weights(text, offsets, ids, [0.9, 0.3, 0.2, 0.6, 0.5, 0.4, 0.9], [0, 2])
    assert w == {retrieval.normalise_token("fees"): 0.6, "doctor": 0.5, "sharma": 0.4}   # कितनी is a stopword
    assert m3_modes.word_weights(text, None, ids, [0.9, 0.3, 0, 0, 0, 0, 0.9], [0, 2]) == {"#11": 0.3}
    assert m3_modes.word_weights("", [], [], [], []) == {}


def test_word_weights_numbers_keep_thousands_together():
    text = "Fee ₹45,000"
    offsets = [(0, 0), (0, 3), (4, 5), (5, 11), (0, 0)]
    w = m3_modes.word_weights(text, offsets, [0, 5, 6, 7, 2], [0, 0.4, 0.1, 0.7, 0], [0, 2])
    assert w["45000"] == 0.7 and "rupi" not in w


def test_sparse_and_maxsim_scores():
    assert m3_modes.sparse_score({"a": 0.5, "b": 0.2}, {"a": 0.4, "c": 1.0}) == pytest.approx(0.2)
    assert m3_modes.sparse_score({}, {"a": 1}) == 0.0
    q = np.eye(3, dtype=np.float32)[:2]                                  # two query tokens
    p1 = np.eye(3, dtype=np.float16)                                     # matches both perfectly
    p2 = np.array([[1, 0, 0]], dtype=np.float16)                         # matches the first only
    assert m3_modes.maxsim_scores(q, [p1, p2]) == pytest.approx([1.0, 0.5], abs=1e-3)
    assert m3_modes.maxsim_scores(q, []) == []


# ------------------------------------------------------------------------------------------------ passage store

def _rep(n=3, seed=0, sparse=None):
    v = np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return m3_modes.PassageM3(sparse=sparse or {"fe": 0.3}, colbert=v.astype(np.float16))


def test_sidecar_roundtrip_key_and_memory_cap(tmp_path):
    store = m3_modes.M3Store(str(tmp_path), "m", encoder=lambda t: iter(()), mem_mb=0)
    k = store.key("chunk-1", "text")
    assert k != store.key("chunk-1", "text edited") and k != store.key("chunk-2", "text")
    rep = _rep(sparse={"fe": 0.25, "डॉक": 0.5})
    store.put(k, rep)
    assert os.path.exists(store._path(k)) and not str(store._path(k)).startswith(str(tmp_path / "local_db"))
    fresh = m3_modes.M3Store(str(tmp_path), "m", encoder=lambda t: iter(()))
    got = fresh.get(k)
    assert got.sparse == pytest.approx({"fe": 0.25, "डॉक": 0.5}) and got.colbert.dtype == np.float16
    assert np.array_equal(got.colbert, rep.colbert)
    assert fresh.get(store.key("nope", "x")) is None
    capped = m3_modes.M3Store(str(tmp_path / "c"), "m", encoder=lambda t: iter(()), mem_mb=1)
    big = [_rep(n=200, seed=i) for i in range(5)]                      # 200 x 8 x 2 bytes each: all fit
    for i, r in enumerate(big):
        capped._remember(f"k{i}", r)
    assert len(capped._mem) == 5
    capped.mem_cap = big[0].nbytes * 2
    capped._remember("k5", _rep(n=200, seed=9))
    assert len(capped._mem) == 2 and "k5" in capped._mem and capped.memory_bytes <= capped.mem_cap


def test_damaged_sidecar_is_dropped(tmp_path):
    store = m3_modes.M3Store(str(tmp_path), "m", encoder=lambda t: iter(()))
    k = store.key("c", "t")
    os.makedirs(os.path.dirname(store._path(k)), exist_ok=True)
    open(store._path(k), "wb").write(b"not a zip")
    assert store.get(k) is None and not os.path.exists(store._path(k))


def test_prune_keeps_the_disk_under_its_cap(tmp_path):
    store = m3_modes.M3Store(str(tmp_path), "m", encoder=lambda t: iter(()), disk_mb=0)
    for i in range(4):
        store.put(store.key(i, "t"), _rep(n=50, seed=i))
    assert store.prune() == 4


def test_background_worker_computes_and_rank_rows_uses_m3_only_when_complete(tmp_path, monkeypatch):
    monkeypatch.delenv("RETRIEVAL_WEIGHTS", raising=False)
    calls = []

    def encoder(texts):
        calls.append(list(texts))
        for i in range(0, len(texts), 2):
            yield [_rep(seed=0, sparse={retrieval.normalise_token(w): 0.3 for w in t.lower().split()})
                   for t in texts[i:i + 2]]

    store = m3_modes.M3Store(str(tmp_path), "m", encoder=encoder, pause_ms=0)
    rows = [{"id": f"p{i}", "chunk_text": t} for i, t in enumerate(["cardiology opd tuesday", "dentist fee 300",
                                                                    "pharmacy open late"])]
    qv = np.eye(DIM, dtype=np.float32)[:2]
    qm3 = m3_modes.QueryM3(sparse={"dentist": 0.5, retrieval.normalise_token("fee"): 0.4}, colbert=qv, store=store)
    info: dict = {}
    first = retrieval.rank_rows("dentist fee", rows, [0.5, 0.5, 0.5], query_m3=qm3, info=info)
    assert info["m3"] is False and "sparse" not in info["signals"]          # nothing cached yet: dense + BM25
    assert store.wait_idle(10) and store.computed == 3 and len(calls) == 1  # every passage of the search, once
    info = {}
    second = retrieval.rank_rows("dentist fee", rows, [0.5, 0.5, 0.5], query_m3=qm3, info=info)
    # default weights: the multi-vector score is fused, the sparse one is computed but weighted 0 (RESULTS.md, 19 Oct 2026)
    assert info["m3"] is True and info["signals"] == ["dense", "colbert", "bm25"]
    assert second[0]["index"] == 1 and "sparse" in second[0]["signals"] and first[0]["index"] == 1
    monkeypatch.setenv("RETRIEVAL_WEIGHTS", "sparse:0.1")                    # still selectable
    info = {}
    retrieval.rank_rows("dentist fee", rows, [0.5, 0.5, 0.5], query_m3=qm3, info=info)
    assert info["signals"] == ["dense", "sparse", "colbert", "bm25"]
    assert store.schedule([(store.key("p0", rows[0]["chunk_text"]), rows[0]["chunk_text"])]) == 0  # already there


def test_worker_failure_is_recorded_not_raised(tmp_path):
    def encoder(texts):
        raise RuntimeError("model gone")
        yield  # pragma: no cover

    store = m3_modes.M3Store(str(tmp_path), "m", encoder=encoder, pause_ms=0)
    assert store.schedule([("k1", "a"), ("k2", "b")]) == 2
    assert store.wait_idle(5) and "model gone" in store.last_error and store.computed == 0
    assert store.schedule([("k1", "a")]) == 1                              # queued again by a later search
    store.wait_idle(5)


def test_encode_passages_batches_through_forward(fake_m3, monkeypatch):
    monkeypatch.setattr(m3_modes, "_memory_allows", lambda need: False)   # resident model path
    texts = ["Dentist fee 300", "Cardiology OPD Tuesday and Thursday", "Pharmacy"]
    reps = [r for batch in m3_modes.encode_passages(FAKE_NAME, texts, batch=2) for r in batch]
    assert len(reps) == 3 and fake_m3.forward_calls == 2 and fake_m3.encode_calls == 0
    assert reps[1].colbert.shape == (len(texts[1].split()) + 1, DIM)        # padding cut off, [CLS] dropped
    assert "dentist" in reps[0].sparse and reps[2].colbert.shape == (2, DIM)
    # the same text alone gives the same vectors (padding does not leak in)
    alone = next(m3_modes.encode_passages(FAKE_NAME, [texts[0]], batch=1))[0]
    assert np.allclose(alone.colbert.astype(np.float32), reps[0].colbert.astype(np.float32), atol=1e-2)


def test_store_wiring_through_encode_query_and_local_store(fake_m3, tmp_path):
    from services.local_store import LocalStore
    db = LocalStore(str(tmp_path / "db"))
    doc = db.insert("documents", {"agent_id": "a1", "user_id": "u", "filename": "f"})
    texts = ["Dentist fee is 300 rupees", "Cardiology OPD on Tuesday", "Pharmacy is open all night"]
    for i, t in enumerate(texts):
        v = retrieval.encode_texts(FAKE_NAME, [t])[0]
        db.insert("document_chunks", {"document_id": doc["id"], "chunk_text": t, "chunk_index": i,
                                      "kind": "passage", "embedding": v.tolist()})
    dense, qm3 = m3_modes.encode_query(FAKE_NAME, "dentist fees")
    info: dict = {}
    db.match_chunks("a1", dense, 3, "dentist fees", query_m3=qm3, info=info)
    assert info["m3"] is False
    store = m3_modes.get_store(FAKE_NAME)
    assert store.wait_idle(10) and store.computed == 3 and str(tmp_path / "cache") in store.root
    info = {}
    out = db.match_chunks("a1", dense, 3, "dentist fees", query_m3=qm3, info=info)
    assert info["m3"] is True and out[0]["chunk_text"] == "Dentist fee is 300 rupees"
    assert db.select("document_chunks")[0].keys() == {"id", "created_at", "document_id", "chunk_text", "chunk_index",
                                                      "kind", "embedding"}                  # user tables untouched


# ------------------------------------------------------------------------------------------------ real model (opt-in)

@pytest.mark.skipif(os.getenv("BONCLAY_SLOW_TESTS") != "1", reason="set BONCLAY_SLOW_TESTS=1 to load the real BGE-M3")
def test_real_bge_m3_one_pass_matches_dense_and_ranks_the_answer_first():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    name = retrieval.embedding_model_name()
    m3_modes._heads_paths_cache.clear()
    if not m3_modes.active(name):
        pytest.skip("BGE-M3 head files not on disk (python -m services.m3_modes --download-heads)")
    import psutil
    if psutil.virtual_memory().available < 2.5 * 2 ** 30:
        pytest.skip("not enough free memory to load BGE-M3")
    q = "dentist ki fees kitni hai?"
    dense, qm3 = m3_modes.encode_query(name, q)
    assert float(np.dot(dense, retrieval.encode_texts(name, [q])[0])) > 0.999 and qm3 is not None
    texts = ["Consultation fees: Dentist ₹300, General physician ₹400.", "Sunday: 10 AM to 1 PM.",
             "Ambulance: call 108."]
    reps = [r for b in m3_modes.encode_passages(name, texts) for r in b]
    sp = [m3_modes.sparse_score(qm3.sparse, r.sparse) for r in reps]
    cb = m3_modes.maxsim_scores(qm3.colbert, [r.colbert for r in reps])
    assert int(np.argmax(sp)) == 0 and int(np.argmax(cb)) == 0
