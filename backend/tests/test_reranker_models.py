"""Reranker model options in services/retrieval.py: the Qwen3-Reranker yes/no scoring (with a fake tokenizer and model),
the remote-code pins of gte-multilingual-reranker-base, device / precision / memory settings. Nothing is downloaded."""
import sys
import types

import pytest

from services import retrieval

torch = pytest.importorskip("torch")

YES, NO, PAD, MARK = 1, 2, 0, ord("Z")


class FakeTokenizer:
    """Character-level: every character is one token (its code point)."""

    def convert_tokens_to_ids(self, token):
        return {"yes": YES, "no": NO}[token]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self, texts, padding=False, truncation=None, return_attention_mask=False, max_length=None):
        ids = [[ord(c) for c in t] for t in texts]
        if truncation and max_length:
            ids = [x[:max_length] for x in ids]
        return {"input_ids": ids}

    def pad(self, enc, padding=True, return_tensors="pt"):
        rows = enc["input_ids"]
        n = max(len(r) for r in rows)
        ids = [[PAD] * (n - len(r)) + list(r) for r in rows]                 # left padding, like padding_side="left"
        mask = [[0] * (n - len(r)) + [1] * len(r) for r in rows]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class FakeCausalLM:
    """'yes' logit = how often the letter Z occurs in the input, 'no' logit = 0, at the LAST position only."""

    def __init__(self, accept_logits_to_keep=True, vocab=128):
        self.accept, self.vocab, self.calls = accept_logits_to_keep, vocab, []

    def __call__(self, input_ids, attention_mask, **kw):
        if "logits_to_keep" in kw and not self.accept:
            raise TypeError("unexpected keyword argument 'logits_to_keep'")
        self.calls.append({"input_ids": input_ids, **kw})
        b, n = input_ids.shape
        positions = 1 if kw.get("logits_to_keep") == 1 else n
        logits = torch.full((b, positions, self.vocab), -5.0)
        logits[:, -1, YES] = (input_ids == MARK).sum(dim=1).float()
        logits[:, -1, NO] = 0.0
        return types.SimpleNamespace(logits=logits)


def make(max_length=400, **kw):
    return retrieval.Qwen3Reranker(FakeTokenizer(), FakeCausalLM(**kw), max_length=max_length)


def test_qwen3_input_is_prefix_instruction_query_document_suffix_left_padded():
    r = make()
    batch = r.build_inputs([("fee?", "short"), ("fee?", "a much longer document text")])
    ids = batch["input_ids"].tolist()
    text = "".join(chr(i) for i in ids[1])
    assert text == (retrieval.QWEN3_RERANKER_PREFIX + "<Instruct>: " + retrieval.QWEN3_RERANKER_INSTRUCTION
                    + "\n<Query>: fee?\n<Document>: a much longer document text" + retrieval.QWEN3_RERANKER_SUFFIX)
    assert ids[0][0] == PAD and ids[0][-len(r.suffix_ids):] == r.suffix_ids     # short row padded on the left
    assert batch["attention_mask"][0, 0].item() == 0 and batch["attention_mask"][1, 0].item() == 1


def test_qwen3_truncates_the_document_to_fit_max_length():
    r = make(max_length=len(retrieval.QWEN3_RERANKER_PREFIX) + len(retrieval.QWEN3_RERANKER_SUFFIX) + 120)
    batch = r.build_inputs([("q", "x" * 5000)])
    ids = batch["input_ids"][0].tolist()
    assert len(ids) == r.max_length and ids[-len(r.suffix_ids):] == r.suffix_ids  # the suffix is never cut


def test_qwen3_score_is_p_yes_against_no_and_orders_passages():
    r = make()
    scores = r.predict([("q", "nothing"), ("q", "one Z"), ("q", "two Z Z")])
    assert scores[0] == pytest.approx(0.5) and scores[1] > 0.5 and scores[2] > scores[1]
    assert scores[1] == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item(), abs=1e-6)
    assert r.model.calls[0]["logits_to_keep"] == 1                              # only the last position is computed


def test_qwen3_batches_and_falls_back_without_logits_to_keep():
    r = make(accept_logits_to_keep=False)
    r.batch_size = 2
    assert len(r.predict([("q", "Z")] * 5)) == 5 and len(r.model.calls) == 3


def test_qwen3_yes_probability():
    logits = torch.zeros((2, 4))
    logits[0, YES], logits[1, NO] = 2.0, 2.0
    p = retrieval.qwen3_yes_probability(logits, YES, NO)
    assert p[0] == pytest.approx(1 / (1 + 2.718281828 ** -2), rel=1e-5) and p[1] == pytest.approx(1 - p[0], rel=1e-5)


def test_reranker_kind_device_dtype_and_memory(monkeypatch):
    assert retrieval.reranker_kind("Qwen/Qwen3-Reranker-0.6B") == "qwen3"
    assert retrieval.reranker_kind("BAAI/bge-reranker-v2-m3") == "cross-encoder"
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    assert retrieval.reranker_model_name() == "Alibaba-NLP/gte-multilingual-reranker-base"
    monkeypatch.delenv("RERANKER_DEVICE", raising=False)
    monkeypatch.delenv("RERANKER_DTYPE", raising=False)
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr(retrieval, "_mps_usable", lambda: True)
    assert retrieval.reranker_device() == "mps" and retrieval.reranker_dtype() == "fp16"
    monkeypatch.setattr(retrieval, "_mps_usable", lambda: False)
    assert retrieval.reranker_device() == "cpu" and retrieval.reranker_dtype() == "fp32"
    monkeypatch.setenv("RERANKER_DEVICE", "cuda")
    monkeypatch.setenv("RERANKER_DTYPE", "bf16")
    assert retrieval.reranker_device() == "cuda" and retrieval.reranker_dtype() == "bf16"
    monkeypatch.delenv("MEMORY_NEED_RERANKER_MB", raising=False)
    assert retrieval.reranker_need_mb("BAAI/bge-reranker-v2-m3") == 2000
    assert retrieval.reranker_need_mb("someone/else") == retrieval.RERANKER_NEED_MB
    monkeypatch.setenv("MEMORY_NEED_RERANKER_MB", "777")
    assert retrieval.reranker_need_mb() == 777


class FakeCrossEncoder:
    made = []

    def __init__(self, name, **kw):
        FakeCrossEncoder.made.append((name, kw))


@pytest.fixture()
def fake_st(monkeypatch):
    FakeCrossEncoder.made = []
    mod = types.ModuleType("sentence_transformers")
    mod.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    monkeypatch.setattr(retrieval, "_local_snapshot", lambda name, revision: f"/cache/{name}@{revision}")
    monkeypatch.setenv("RERANKER_DEVICE", "cpu")
    monkeypatch.delenv("RERANKER_DTYPE", raising=False)
    monkeypatch.delenv("RERANKER_REVISION", raising=False)
    return FakeCrossEncoder


def test_gte_runs_remote_code_only_at_the_pinned_commits(fake_st):
    retrieval._new_reranker("Alibaba-NLP/gte-multilingual-reranker-base")
    name, kw = fake_st.made[-1]
    pins = retrieval.REMOTE_CODE_RERANKERS["alibaba-nlp/gte-multilingual-reranker-base"]
    assert name == f"/cache/Alibaba-NLP/gte-multilingual-reranker-base@{pins['revision']}"
    assert kw["trust_remote_code"] is True and kw["device"] == "cpu"
    for k in ("config_kwargs", "processor_kwargs"):
        assert kw[k] == {"code_revision": pins["code_revision"]}
    assert kw["model_kwargs"]["code_revision"] == pins["code_revision"]
    assert kw["model_kwargs"]["dtype"] == torch.float32


def test_other_cross_encoders_never_run_remote_code(fake_st):
    retrieval._new_reranker("BAAI/bge-reranker-v2-m3")
    name, kw = fake_st.made[-1]
    assert name == "/cache/BAAI/bge-reranker-v2-m3@None" and "trust_remote_code" not in kw
    assert "config_kwargs" not in kw


def test_qwen3_model_name_builds_the_yes_no_reranker(fake_st, monkeypatch):
    seen = {}
    monkeypatch.setattr(retrieval.Qwen3Reranker, "load",
                        classmethod(lambda cls, source, device, dtype, max_length, revision: seen.update(
                            source=source, device=device, max_length=max_length) or "qwen"))
    assert retrieval._new_reranker("Qwen/Qwen3-Reranker-0.6B") == "qwen"
    assert seen["source"].startswith("/cache/Qwen/Qwen3-Reranker-0.6B") and not fake_st.made


def test_on_disk_check_uses_the_pinned_revision(monkeypatch):
    import huggingface_hub
    retrieval._reranker_on_disk.cache_clear()
    calls = []
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, f, revision=None: calls.append((repo, f, revision)) or None)
    monkeypatch.delenv("RERANKER_REVISION", raising=False)
    assert retrieval._reranker_on_disk("Alibaba-NLP/gte-multilingual-reranker-base") is False
    pin = retrieval.REMOTE_CODE_RERANKERS["alibaba-nlp/gte-multilingual-reranker-base"]["revision"]
    assert calls and all(c[2] == pin for c in calls)
    retrieval._reranker_on_disk.cache_clear()


def test_reranker_calls_are_serialised_across_threads(monkeypatch):
    import threading
    import time as _time
    state = {"inside": 0, "max": 0}

    class Slow:
        def predict(self, pairs, **kw):
            state["inside"] += 1
            state["max"] = max(state["max"], state["inside"])
            _time.sleep(0.02)
            state["inside"] -= 1
            return [0.5] * len(pairs)

    monkeypatch.setattr(retrieval, "_reranker", Slow())
    threads = [threading.Thread(target=retrieval.rerank_scores, args=("q", ["a", "b"])) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] == 1
