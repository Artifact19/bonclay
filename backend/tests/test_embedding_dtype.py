"""Embedding model precision (services/retrieval.py): EMBEDDING_DTYPE=auto|fp16|fp32, the float16 copy on disk, the
float32 bulk path and the float32 re-normalisation. No model is loaded: SentenceTransformer is faked."""
import os
import sys
import types

import numpy as np
import pytest

from services import retrieval


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DTYPE", raising=False)
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.setattr(retrieval, "_embedders", {})
    monkeypatch.setattr(retrieval, "_half_models", set())


def _platform(monkeypatch, system, machine):
    monkeypatch.setattr(retrieval.platform, "system", lambda: system)
    monkeypatch.setattr(retrieval.platform, "machine", lambda: machine)


def test_auto_is_fp16_on_apple_silicon_only(monkeypatch):
    _platform(monkeypatch, "Darwin", "arm64")
    assert retrieval.embedding_dtype() == "fp16"
    _platform(monkeypatch, "Windows", "AMD64")
    assert retrieval.embedding_dtype() == "fp32"       # x86 CPUs mostly lack fast fp16 matmul
    _platform(monkeypatch, "Linux", "x86_64")
    assert retrieval.embedding_dtype() == "fp32"
    _platform(monkeypatch, "Darwin", "x86_64")
    assert retrieval.embedding_dtype() == "fp32"


def test_auto_is_fp16_on_a_gpu_device(monkeypatch):
    _platform(monkeypatch, "Windows", "AMD64")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
    assert retrieval.embedding_dtype() == "fp16"
    monkeypatch.setenv("EMBEDDING_DEVICE", "mps")
    assert retrieval.embedding_dtype() == "fp16"


@pytest.mark.parametrize("value,expected", [("fp16", "fp16"), ("FP16", "fp16"), ("float16", "fp16"), ("fp32", "fp32"),
                                            ("float32", "fp32"), (" fp32 ", "fp32")])
def test_explicit_choice_wins(monkeypatch, value, expected):
    _platform(monkeypatch, "Darwin" if expected == "fp32" else "Windows", "arm64" if expected == "fp32" else "AMD64")
    monkeypatch.setenv("EMBEDDING_DTYPE", value)
    assert retrieval.embedding_dtype() == expected


def test_unknown_value_behaves_like_auto(monkeypatch):
    _platform(monkeypatch, "Windows", "AMD64")
    monkeypatch.setenv("EMBEDDING_DTYPE", "bf16-please")
    assert retrieval.embedding_dtype() == "fp32"


class HalfModel:
    """Returns float16, un-normalised vectors, like a half-precision SentenceTransformer."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kw):
        self.calls.append((list(texts), kw))
        return (np.arange(1, len(texts) * 4 + 1, dtype=np.float16).reshape(len(texts), 4)) * np.float16(3)


def _half(monkeypatch, model):
    monkeypatch.setattr(retrieval, "load_sentence_transformer", lambda name: model)
    retrieval._half_models.add("m")


def test_half_model_vectors_are_float32_and_unit_length(monkeypatch):
    model = HalfModel()
    _half(monkeypatch, model)
    out = retrieval.encode_texts("m", ["what time", "fees"])
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    assert model.calls[0][1]["normalize_embeddings"] is False       # normalised afterwards, in float32


def test_half_model_respects_normalize_false(monkeypatch):
    _half(monkeypatch, HalfModel())
    out = retrieval.encode_texts("m", ["a"], normalize_embeddings=False)
    assert out.dtype == np.float32 and not np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_full_precision_model_is_left_alone(monkeypatch):
    seen = {}

    class M:
        def encode(self, texts, **kw):
            seen.update(kw)
            return np.zeros((len(texts), 4))

    monkeypatch.setattr(retrieval, "load_sentence_transformer", lambda name: M())
    retrieval.encode_texts("m", ["a"])
    assert seen["normalize_embeddings"] is True


def test_questions_use_the_half_model_and_passages_the_float32_copy(monkeypatch):
    model = HalfModel()
    _half(monkeypatch, model)
    bulk = []

    def fake_bulk(name, texts, kwargs):
        bulk.append(texts)
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(retrieval, "_encode_bulk_fp32", fake_bulk)
    retrieval.encode_texts("m", ["क्लिनिक सोमवार को कितने बजे खुलता है?"])          # a question
    assert len(model.calls) == 1 and not bulk
    passage = " ".join(["word"] * 120)
    out = retrieval.encode_texts("m", [passage, passage])                            # a document being ingested
    assert len(bulk) == 1 and len(model.calls) == 1
    assert out.dtype == np.float32 and np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_bulk_threshold_counts_words_across_the_batch():
    assert not retrieval._is_bulk(["a few words"])
    assert not retrieval._is_bulk([" ".join(["w"] * 40)] * 4)
    assert retrieval._is_bulk([" ".join(["w"] * 120)] * 2)


def test_no_bulk_float32_copy_off_the_cpu(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", "mps")
    model = HalfModel()
    _half(monkeypatch, model)
    monkeypatch.setattr(retrieval, "_encode_bulk_fp32", lambda *a: pytest.fail("a GPU keeps fp16 for everything"))
    retrieval.encode_texts("m", [" ".join(["w"] * 300)])
    assert len(model.calls) == 1


# ------------------------------------------------------------------------------------------------ loading

class FakeST:
    instances = []

    def __init__(self, source, device=None, model_kwargs=None):
        self.source, self.device, self.model_kwargs = source, device, model_kwargs
        self.max_seq_length = 8192
        self.saved = None
        FakeST.instances.append(self)

    def save(self, path, safe_serialization=True):
        import os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.safetensors"), "w") as f:
            f.write("weights")
        self.saved = path


@pytest.fixture()
def fake_st(monkeypatch, tmp_path):
    FakeST.instances = []
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    monkeypatch.setenv("EMBEDDING_FP16_DIR", str(tmp_path))
    torch = pytest.importorskip("torch")
    return torch


def test_first_fp16_load_converts_saves_and_reloads_from_the_copy(monkeypatch, fake_st, tmp_path):
    monkeypatch.setenv("EMBEDDING_DTYPE", "fp16")
    model = retrieval.load_sentence_transformer("BAAI/bge-m3")
    path = retrieval.fp16_cache_dir("BAAI/bge-m3")
    assert path.startswith(str(tmp_path)) and path.endswith("BAAI_bge-m3-fp16")
    converted, resident = FakeST.instances
    assert converted.source == "BAAI/bge-m3" and converted.model_kwargs == {"dtype": fake_st.float16}
    assert converted.saved.endswith(".tmp")
    assert resident is model and resident.source == path and resident.model_kwargs == {"dtype": fake_st.float16}
    assert "BAAI/bge-m3" in retrieval._half_models and model.max_seq_length == 512
    import os
    assert os.path.exists(os.path.join(path, ".complete")) and not os.path.exists(path + ".tmp")


def test_second_load_uses_the_copy_without_converting(monkeypatch, fake_st):
    monkeypatch.setenv("EMBEDDING_DTYPE", "fp16")
    retrieval.load_sentence_transformer("BAAI/bge-m3")
    retrieval._embedders.clear()
    retrieval._half_models.clear()
    FakeST.instances.clear()
    model = retrieval.load_sentence_transformer("BAAI/bge-m3")
    assert len(FakeST.instances) == 1 and model.source == retrieval.fp16_cache_dir("BAAI/bge-m3")


def test_fp32_choice_loads_the_original_only(monkeypatch, fake_st):
    monkeypatch.setenv("EMBEDDING_DTYPE", "fp32")
    model = retrieval.load_sentence_transformer("BAAI/bge-m3")
    assert model.source == "BAAI/bge-m3" and model.model_kwargs is None
    assert "BAAI/bge-m3" not in retrieval._half_models


def test_unwritable_copy_falls_back_to_float32(monkeypatch, fake_st):
    monkeypatch.setenv("EMBEDDING_DTYPE", "fp16")

    def broken_save(self, path, safe_serialization=True):
        raise OSError("read-only file system")

    monkeypatch.setattr(FakeST, "save", broken_save)
    model = retrieval.load_sentence_transformer("BAAI/bge-m3")
    assert model.source == "BAAI/bge-m3" and model.model_kwargs is None
    assert "BAAI/bge-m3" not in retrieval._half_models


def test_a_copy_without_its_weights_is_not_trusted(tmp_path, monkeypatch):
    from services import retrieval
    monkeypatch.setenv("EMBEDDING_FP16_DIR", str(tmp_path))
    path = retrieval.fp16_cache_dir("BAAI/bge-m3")
    os.makedirs(path)
    open(os.path.join(path, ".complete"), "w").close()
    assert retrieval.fp16_copy_ready("BAAI/bge-m3") is False          # marker alone: weights were removed
    with open(os.path.join(path, "model.safetensors"), "wb") as f:
        f.write(b"x")
    assert retrieval.fp16_copy_ready("BAAI/bge-m3") is True


def test_bulk_encode_falls_back_to_the_original_model_when_the_copy_is_gone(tmp_path, monkeypatch):
    from services import retrieval
    monkeypatch.setenv("EMBEDDING_FP16_DIR", str(tmp_path))
    seen = []

    class Fake:
        max_seq_length = 512

        def encode(self, texts, **kw):
            return [[0.0]] * len(texts)

    monkeypatch.setattr(retrieval, "_new_sentence_transformer", lambda source, dtype=None: seen.append(source) or Fake())
    retrieval._encode_bulk_fp32("BAAI/bge-m3", ["word " * 250], {})
    assert seen == ["BAAI/bge-m3"]
