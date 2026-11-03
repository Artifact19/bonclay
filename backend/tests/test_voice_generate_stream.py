"""POST /api/voices/generate/stream: a Studio render as server-sent events, one `sentence` per rendered piece and a
`done` with the generation of the stitched clip (one receipt, one watermark). Kokoro and the clone engine are faked."""
import io
import json

import numpy as np
import pytest
import soundfile as sf

from services import provenance
from tests.test_offline_agents import LOCAL
from tests.test_provenance import backend_mod, fake_seal  # noqa: F401  (fixtures)

TEXT = "The clinic opens at nine. Dr. Kulkarni sees patients on Tuesdays. Please bring your insurance card."
CLIP = 2400          # samples per faked sentence (0.1 s at 24 kHz)


def parse(r):
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    events = []
    for block in r.text.strip().split("\n\n"):
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ") and len(lines) == 2
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


@pytest.fixture()
def rig(app_client, backend_mod, monkeypatch):
    spoken = []

    async def fake_load(key):
        return object()

    def fake_kokoro(model, text, lang="en"):
        spoken.append((text, lang))
        return np.full(CLIP, 0.05, dtype=np.float32)

    monkeypatch.setattr(backend_mod, "_kokoro_generate", fake_kokoro)
    monkeypatch.setattr(backend_mod.advanced_manager, "load_model", fake_load)
    app_client.spoken = spoken
    return app_client


def post(client, **data):
    return client.post("/api/voices/generate/stream", headers=LOCAL, data={"text": TEXT, **data})


def test_three_sentences_stream_then_one_receipted_clip(rig):
    events = parse(post(rig))
    assert [e for e, _ in events] == ["sentence", "sentence", "sentence", "done"]
    sentences = [d for e, d in events if e == "sentence"]
    assert [s["index"] for s in sentences] == [0, 1, 2]
    assert [s["text"] for s in sentences] == ["The clinic opens at nine.",
                                              "Dr. Kulkarni sees patients on Tuesdays.",
                                              "Please bring your insurance card."]
    assert all(s["total"] == 3 for s in sentences)
    urls = [s["audio_url"] for s in sentences]
    assert len(set(urls)) == 3
    assert [t for t, _ in rig.spoken] == [s["text"] for s in sentences]   # each sentence spoken once, in order

    for url in urls:  # each sentence clip is served (owner-checked) and is a watermarked WAV
        clip = rig.get(url, headers=LOCAL)
        assert clip.status_code == 200 and clip.headers["content-type"] == "audio/wav"
        assert len(sf.read(io.BytesIO(clip.content))[0]) == CLIP

    done = events[-1][1]
    assert done["engine"] == "kokoro" and done["sentences"] == 3
    assert done["receipt_id"] and done["audio_url"] and done["generation_id"] == done["id"]
    full = rig.get(done["audio_url"], headers=LOCAL)
    assert full.status_code == 200
    data, sr = sf.read(io.BytesIO(full.content))
    gap = int(sr * 0.08)
    assert len(data) == pytest.approx(3 * CLIP + 2 * gap, abs=2)
    assert done["duration_s"] == pytest.approx(len(data) / sr, abs=0.001)

    # one receipt, for exactly these bytes and this text, and it resolves through the public route
    r = rig.get(f"/api/receipts/{done['receipt_id']}")
    assert r.status_code == 200, r.text
    receipt = provenance.get_receipt(done["receipt_id"])
    assert receipt["kind"] == "generation" and receipt["audio_sha256"] == provenance.sha256_hex(full.content)
    assert receipt["text_sha256"] == provenance.sha256_hex(TEXT.encode("utf-8"))


def test_the_stitched_clip_is_receipted_once_through_persist_generation(rig, backend_mod, monkeypatch):
    calls = []
    real = backend_mod.persist_generation

    async def spy(*args, **kw):
        calls.append((kw.get("text"), kw.get("engine")))
        return await real(*args, **kw)

    monkeypatch.setattr(backend_mod, "persist_generation", spy)
    parse(post(rig))
    assert calls == [(TEXT, "kokoro")]


def test_the_non_streaming_endpoint_is_unchanged(rig):
    r = rig.post("/api/voices/generate", headers=LOCAL, data={"text": TEXT})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.headers.get("x-receipt-id")


def test_a_single_sentence_is_one_sentence_event(rig):
    events = parse(post(rig, text="Just one short sentence."))
    assert [e for e, _ in events] == ["sentence", "done"]


def test_an_unknown_voice_is_an_error_event(rig):
    events = parse(post(rig, voice_id="no-such-voice", mode="premium"))
    assert [e for e, _ in events] == ["error"]
    assert events[0][1]["code"] == "VOICE_NOT_FOUND"


def test_a_failing_engine_is_an_error_event_after_the_sentences_that_were_made(rig, backend_mod, monkeypatch):
    n = []

    def flaky(model, text, lang="en"):
        n.append(text)
        if len(n) == 2:
            raise RuntimeError("engine died")
        return np.full(CLIP, 0.05, dtype=np.float32)

    monkeypatch.setattr(backend_mod, "_kokoro_generate", flaky)
    events = parse(post(rig))
    assert [e for e, _ in events] == ["sentence", "error"]
    assert events[1][1]["code"] == "GENERATION_FAILED"


def test_request_problems_are_http_errors(rig):
    assert rig.post("/api/voices/generate/stream", headers=LOCAL, data={"text": "   "}).status_code in (400, 422)
    assert rig.post("/api/voices/generate/stream", headers=LOCAL,
                    data={"text": "hi", "voice_id": "bad id!"}).status_code in (400, 422)


def test_a_cloned_voice_speaks_each_sentence_through_the_clone_engine(rig, backend_mod, monkeypatch):
    calls = []

    class FakeQwen:
        def generate_voice_clone(self, text, language, voice_clone_prompt, **kw):
            calls.append((text, language, voice_clone_prompt))
            return [np.full(CLIP, 0.05, dtype=np.float32)], 24000

    async def lookup(voice_id, storage_mode, user_id=None):
        return {"id": voice_id, "display_name": "Ayush", "transcript": "good morning", "engine": "qwen3_0_6b"}

    async def prompt(voice_id, user_id, storage_mode, row, model):
        return {"prompt": voice_id}

    async def load(key):
        return FakeQwen()

    monkeypatch.setattr(backend_mod, "_lookup_voice", lookup)
    monkeypatch.setattr(backend_mod, "load_clone_prompt", prompt)
    monkeypatch.setattr(backend_mod.advanced_manager, "load_model", load)
    events = parse(post(rig, voice_id="v-1", mode="premium"))
    assert [e for e, _ in events] == ["sentence", "sentence", "sentence", "done"]
    assert [c[0] for c in calls] == [d["text"] for e, d in events if e == "sentence"]
    assert all(c[2] == {"prompt": "v-1"} for c in calls)
    done = events[-1][1]
    assert done["engine"] == backend_mod.CLONE_ENGINE and done["voice_name"] == "Ayush"
