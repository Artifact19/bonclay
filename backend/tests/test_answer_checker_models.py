"""Answer-audit checking models (services/answer_audit.py): model selection, pinned revisions, calibrated thresholds and
MiniCheck's input format and scoring, with a fake tokenizer and fake models. Nothing is downloaded."""
import types

import pytest

from services import answer_audit as aa

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("ANSWER_AUDIT_MODEL", "ANSWER_AUDIT_MODEL_REVISION", "AUDIT_THRESHOLD"):
        monkeypatch.delenv(k, raising=False)


def test_default_is_minicheck_roberta_pinned_with_its_calibrated_threshold(monkeypatch):
    assert aa.model_key() == aa.DEFAULT_MODEL == "minicheck-roberta"
    assert aa.model_label() == "MiniCheck-RoBERTa-Large" and aa.threshold() == 0.5
    assert aa.model_revision() == aa.MODELS["minicheck-roberta"]["revision"] and len(aa.model_revision()) == 40
    assert aa.MODEL_NEED_MB == aa.MODELS["minicheck-roberta"]["need_mb"]
    monkeypatch.setenv("ANSWER_AUDIT_MODEL", "HHEM")
    assert aa.model_label() == "HHEM-2.1-Open" and aa.threshold() == 0.9     # HHEM's scores sit high: 0.5 misses most
    assert aa.MODELS["hhem"]["remote_code"] is True and aa.model_revision() == aa.MODELS["hhem"]["revision"]
    monkeypatch.setenv("AUDIT_THRESHOLD", "0.7")
    assert aa.threshold() == 0.7
    monkeypatch.setenv("ANSWER_AUDIT_MODEL_REVISION", "abc123")
    assert aa.model_revision() == "abc123"
    monkeypatch.setenv("ANSWER_AUDIT_MODEL", "nonsense")
    assert aa.model_key() == "minicheck-roberta"


def test_verifier_name_follows_the_selected_model(monkeypatch):
    v = aa.HHEMVerifier(loader=lambda: None)
    assert v.name == "lytang/MiniCheck-RoBERTa-Large (MiniCheck-RoBERTa-Large)"
    monkeypatch.setenv("ANSWER_AUDIT_MODEL", "minicheck-ft5")
    assert "MiniCheck-Flan-T5-Large" in v.name


def test_model_cached_looks_for_the_pinned_commit_and_hhems_tokenizer(monkeypatch):
    import huggingface_hub
    calls = []

    def fake(repo, name, revision=None):
        calls.append((repo, name, revision))
        return "/x"

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake)
    assert aa.model_cached() is True
    assert calls == [("lytang/MiniCheck-RoBERTa-Large", "config.json", aa.MODELS["minicheck-roberta"]["revision"])]
    calls.clear()
    monkeypatch.setenv("ANSWER_AUDIT_MODEL", "hhem")
    assert aa.model_cached() is True and ("google/flan-t5-base", "tokenizer_config.json", None) in calls


def test_default_loader_reserves_the_selected_models_memory(monkeypatch):
    from core import memory_guard
    from core.errors import AppError

    def refuse(engine, candidates, unload, **kw):
        raise AppError("INSUFFICIENT_MEMORY", "no room", 503)

    monkeypatch.setattr(memory_guard, "ensure_headroom", refuse)
    monkeypatch.setattr(aa, "_load_model", lambda *a: pytest.fail("must not load"))
    monkeypatch.setenv("ANSWER_AUDIT_MODEL", "minicheck-ft5")
    with pytest.raises(AppError):
        aa._default_loader()
    assert memory_guard.need_mb(aa.ENGINE) == aa.MODELS["minicheck-ft5"]["need_mb"]
    monkeypatch.delenv("ANSWER_AUDIT_MODEL")
    with pytest.raises(AppError):
        aa._default_loader()
    assert memory_guard.need_mb(aa.ENGINE) == aa.MODEL_NEED_MB


# ---------------------------------------------------------------------------------------------------- MiniCheck scoring

class WordTokenizer:
    """One token per whitespace word; id = 10 + position in a growing vocabulary."""
    eos_token = "</s>"

    def __init__(self):
        self.vocab = {}
        self.calls = []

    def _ids(self, text):
        return [self.vocab.setdefault(w, 10 + len(self.vocab)) for w in text.split()]

    def __call__(self, text, add_special_tokens=True, max_length=None, truncation=False, padding=False,
                 return_tensors=None):
        if isinstance(text, str):
            return {"input_ids": self._ids(text)}
        self.calls.append(list(text))
        rows = [self._ids(t)[:max_length] if truncation and max_length else self._ids(t) for t in text]
        n = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (n - len(r)) for r in rows])
        mask = torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in rows])
        return {"input_ids": ids, "attention_mask": mask}


class FakeT5:
    """Seq2seq: one decoder step; logit of token 209 ('support') = 3 when the text contains 'GOOD', token 3 = 0."""

    def __init__(self, tok):
        self.tok, self.seen = tok, []

    def __call__(self, input_ids, attention_mask, decoder_input_ids):
        self.seen.append(decoder_input_ids)
        good = self.tok.vocab.get("GOOD", -1)
        logits = torch.zeros((input_ids.shape[0], 1, 300))
        logits[:, 0, 209] = 3.0 * (input_ids == good).any(dim=1).float()
        return types.SimpleNamespace(logits=logits)


class FakeRoberta:
    """2-class: class 1 logit = 2 when the text contains 'GOOD'."""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, input_ids, attention_mask):
        good = self.tok.vocab.get("GOOD", -1)
        logits = torch.zeros((input_ids.shape[0], 2))
        logits[:, 1] = 2.0 * (input_ids == good).any(dim=1).float()
        return types.SimpleNamespace(logits=logits)


def test_minicheck_ft5_input_format_and_support_probability():
    tok = WordTokenizer()
    m = aa.MiniCheckModel(tok, FakeT5(tok), "minicheck-ft5")
    assert m.texts("The fee is 300.", "Fee is 300.") == ["predict: The fee is 300.</s>Fee is 300."]
    scores = m.predict([("doc GOOD here.", "claim"), ("doc here.", "claim")])
    assert scores[0] == pytest.approx(torch.softmax(torch.tensor([0.0, 3.0]), 0)[1].item())
    assert scores[1] == pytest.approx(0.5)
    assert all(int(d.abs().sum()) == 0 and d.shape[1] == 1 for d in m.model.seen)   # one start token (id 0)


def test_minicheck_roberta_chunks_a_long_document_and_keeps_the_best_chunk(monkeypatch):
    tok = WordTokenizer()
    m = aa.MiniCheckModel(tok, FakeRoberta(tok), "minicheck-roberta", batch_size=2)
    monkeypatch.setitem(aa.MINICHECK_CHUNK_TOKENS, "minicheck-roberta", 40)
    doc = " ".join(f"Sentence {i} has filler words to pad it out." for i in range(20)) + " The GOOD answer is here."
    chunks = m.chunks(doc, "the claim")
    assert len(chunks) > 3 and all(len(c.split()) <= 40 for c in chunks)
    assert " ".join(chunks) == " ".join(doc.split())                                 # nothing lost, nothing repeated
    texts = m.texts(doc, "the claim")
    assert all(t.endswith("</s>the claim") and not t.startswith("predict:") for t in texts)
    score = m.predict([(doc, "the claim")])[0]
    assert score == pytest.approx(torch.softmax(torch.tensor([0.0, 2.0]), 0)[1].item())   # the GOOD chunk wins
    assert m.predict([("short doc.", "the claim")])[0] == pytest.approx(0.5)


def test_minicheck_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        aa.MiniCheckModel(WordTokenizer(), None, "bert")
