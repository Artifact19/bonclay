"""POST /api/conversations/{agent}/turn/stream: the spoken turn as server-sent events, with speech recognition, the
language model stream and TTS faked. Covers event order, sentence splitting, persistence and perf fields, the
fallback line, a stream that dies mid-reply, and who may fetch the sentence audio."""
import io
import json
import time

import numpy as np
import pytest
import soundfile as sf

from services import llm
from services.sentences import SentenceSplitter
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)

REPLY = ["The clinic opens ", "at 9 a.m. on Monday. ", "Dr. Kulkarni sees ", "patients on Tuesdays, ", "for 4.5 hours?"]


class FakeStream:
    """Stands in for llm.ChatStream."""

    def __init__(self, deltas, *, fail_after=None, delay=0.0, first_delay=0.0):
        self.deltas, self.fail_after, self.delay, self.first_delay = deltas, fail_after, delay, first_delay
        self.info = {}
        self.result = None
        self.closed = False

    def __iter__(self):
        text = ""
        time.sleep(self.first_delay)
        for i, d in enumerate(self.deltas):
            if self.fail_after is not None and i >= self.fail_after:
                raise llm.LLMStreamInterrupted("fake", "fake-model", text, ConnectionResetError("reset"))
            time.sleep(self.delay)
            text += d
            yield d
        self.result = llm.StreamResult(text, "fake", "fake-model", 7, {}, 3)

    def close(self):
        self.closed = True


class Rig:
    def __init__(self, env, monkeypatch, app_client, ref_wav):
        self.env, self.client, self.wav = env, app_client, ref_wav.read_bytes()
        self.spoken = []        # (text, language) handed to Kokoro
        self.asked = []         # messages sent to the language model
        self.asr_text = "When does the clinic open?"
        self.stream = lambda: FakeStream(REPLY)

        async def fake_asr(audio_bytes, name, language=None):
            return self.asr_text

        async def fake_load(key):
            return object()

        def fake_kokoro(model, text, lang="en"):
            self.spoken.append((text, lang))
            return np.full(2400, 0.05, dtype=np.float32)

        def chat_stream(messages, **kw):
            self.asked.append({"messages": messages, **kw})
            s = self.stream()
            if isinstance(s, Exception):
                raise s
            return s

        env._DISCLOSURE_CLIPS.clear()
        monkeypatch.setattr(env, "_speech_to_text", fake_asr)
        monkeypatch.setattr(env, "_kokoro_generate", fake_kokoro)
        monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
        monkeypatch.setattr(llm, "chat_stream", chat_stream)

    def post(self, agent_id, conversation_id=None, wav=None):
        data = {"conversation_id": conversation_id} if conversation_id else None
        return self.client.post(f"/api/conversations/{agent_id}/turn/stream", headers=LOCAL, data=data,
                                files={"audio": ("rec.wav", wav or self.wav, "audio/wav")})


@pytest.fixture()
def rig(app_client, env, monkeypatch, ref_wav):
    return Rig(env, monkeypatch, app_client, ref_wav)


def parse(r):
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    events = []
    for block in r.text.strip().split("\n\n"):
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ") and len(lines) == 2
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


def names(events):
    return [e for e, _ in events]


# ------------------------------------------------------------------------------------ the happy path

def test_events_arrive_in_order_with_the_disclosure_first(rig):
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"]))
    assert names(events) == ["transcript", "sentence", "sentence", "sentence", "done"]
    transcript = events[0][1]
    assert transcript == {"user_text": "When does the clinic open?", "language": "en"}

    sentences = [d for e, d in events if e == "sentence"]
    assert [s["index"] for s in sentences] == [0, 1, 2]
    assert sentences[0]["text"] == "This is an AI assistant."
    assert sentences[1]["text"] == "The clinic opens at 9 a.m. on Monday."
    assert sentences[2]["text"] == "Dr. Kulkarni sees patients on Tuesdays, for 4.5 hours?"
    assert all(s["audio_url"] == f"/api/conversations/turns/{events[-1][1]['turn_id']}/sentences/{s['index']}/audio"
               for s in sentences)


def test_done_summary_and_stitched_audio(rig):
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"]))
    done = events[-1][1]
    sentences = [d for e, d in events if e == "sentence"]
    assert done["agent_text"] == " ".join(s["text"] for s in sentences)
    assert done["fallback"] is False and done["interrupted"] is False
    assert done["engine"] == "kokoro" and done["llm_provider"] == "fake" and done["llm_model"] == "fake-model"
    assert done["language"] == "en" and done["receipt_id"] and done["conversation_id"] and done["turn_id"]
    t = done["timings"]
    assert {"asr_ms", "retrieval_ms", "llm_first_token_ms", "llm_ms", "first_audio_ms", "tts_ms",
            "total_ms"} <= set(t)
    assert all(isinstance(v, int) and v >= 0 for v in t.values())
    assert t["first_audio_ms"] <= t["total_ms"] and t["llm_first_token_ms"] <= t["llm_ms"]

    # The stitched reply is served by the existing turn-audio route and is as long as its sentences plus the gaps.
    full = rig.client.get(done["agent_audio_url"], headers=LOCAL)
    assert full.status_code == 200 and full.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(full.content))
    clip_len = 2400 * len(sentences)
    gaps = int(sr * rig.env.SENTENCE_GAP_S) * (len(sentences) - 1)
    assert len(data) == clip_len + gaps
    # ...and its receipt is for exactly these bytes
    from services import provenance
    receipt = provenance.get_receipt(done["receipt_id"])
    assert receipt["turn_id"] == done["turn_id"] and receipt["kind"] == "agent_reply"
    assert receipt["audio_sha256"] == provenance.sha256_hex(full.content)
    assert receipt["text_sha256"] == provenance.sha256_hex(done["agent_text"].encode("utf-8"))


def test_the_stitched_reply_goes_through_the_existing_provenance_path_once(rig, monkeypatch):
    from services import provenance
    calls = []
    real = provenance.apply_provenance

    def spy(**kw):
        calls.append(kw)
        return real(**kw)
    monkeypatch.setattr(provenance, "apply_provenance", spy)
    a = make_agent(rig.client)
    done = parse(rig.post(a["id"]))[-1][1]
    assert len(calls) == 1
    c = calls[0]
    assert c["kind"] == "agent_reply" and c["subject_id"] == done["turn_id"] and c["text"] == done["agent_text"]
    assert c["engine"] == "kokoro" and c["voice_id"] is None and c["user_id"] == "local-user" and c["cloud"] is False


def test_each_sentence_is_synthesized_on_its_own_and_served(rig):
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"]))
    assert [t for t, _ in rig.spoken] == [
        "This is an AI assistant.", "The clinic opens at 9 a.m. on Monday.",
        "Dr. Kulkarni sees patients on Tuesdays, for 4.5 hours?"]
    for e, d in events:
        if e == "sentence":
            r = rig.client.get(d["audio_url"], headers=LOCAL)
            assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
            assert len(sf.read(io.BytesIO(r.content))[0]) == 2400


def test_sentences_are_emitted_before_the_model_finishes(rig):
    """The first sentence's event is produced while the model is still writing the rest."""
    rig.stream = lambda: FakeStream(REPLY, delay=0.15)
    a = make_agent(rig.client, disclose_ai=False)
    seen = []
    orig = rig.env._kokoro_generate

    def spy(model, text, lang="en"):
        seen.append((text, time.time()))
        return orig(model, text, lang)
    rig.env._kokoro_generate = spy
    events = parse(rig.post(a["id"]))
    done = events[-1][1]
    assert [d["index"] for e, d in events if e == "sentence"] == [0, 1]
    # The first sentence was spoken well before the model's last delta (5 deltas x 0.15 s = 0.75 s).
    assert done["timings"]["first_audio_ms"] < done["timings"]["llm_ms"] - 200
    assert done["timings"]["llm_ms"] >= 700 and done["timings"]["llm_first_token_ms"] >= 140


def test_the_disclosure_is_synthesized_once_and_still_watermarked_per_turn(rig):
    a = make_agent(rig.client)
    parse(rig.post(a["id"]))
    parse(rig.post(a["id"]))  # a new conversation: the disclosure is spoken again
    assert [t for t, _ in rig.spoken].count("This is an AI assistant.") == 1
    marks = []
    orig = rig.env._watermark_clip

    def spy(wav, sr, turn_id):
        out = orig(wav, sr, turn_id)
        marks.append((turn_id, out[1]))
        return out
    rig.env._watermark_clip = spy
    try:
        parse(rig.post(a["id"]))
    finally:
        rig.env._watermark_clip = orig
    assert len(marks) == 3 and len({m[0] for m in marks}) == 1  # every clip carries this turn's id


def test_no_disclosure_after_the_first_reply_and_history_is_sent(rig):
    a = make_agent(rig.client)
    first = parse(rig.post(a["id"]))
    conv = first[-1][1]["conversation_id"]
    second = parse(rig.post(a["id"], conv))
    sentences = [d["text"] for e, d in second if e == "sentence"]
    assert sentences[0] == "The clinic opens at 9 a.m. on Monday."
    assert second[-1][1]["conversation_id"] == conv
    roles = [m["role"] for m in rig.asked[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]  # the first exchange is in the history
    assert rig.asked[-1]["max_tokens"] == 160 and rig.asked[-1]["temperature"] == llm.answer_temperature() == 0.2


def test_disclosure_can_be_switched_off(rig):
    a = make_agent(rig.client, disclose_ai=False)
    events = parse(rig.post(a["id"]))
    texts = [d["text"] for e, d in events if e == "sentence"]
    assert "AI assistant" not in " ".join(texts) and texts[0].startswith("The clinic opens")


# ------------------------------------------------------------------------------------ persistence and perf

def test_turn_is_persisted_like_the_blocking_endpoint_and_feeds_perf(rig):
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"]))
    done = events[-1][1]
    rows = rig.env.local_db.select("conversation_turns", lambda r: r["conversation_id"] == done["conversation_id"])
    user, agent = sorted(rows, key=lambda r: r["role"] != "user")
    assert user["role"] == "user" and user["text_content"] == "When does the clinic open?"
    assert agent["id"] == done["turn_id"] and agent["text_content"] == done["agent_text"]
    assert agent["audio_storage_path"] == f"local:turn_{done['turn_id']}.wav"
    assert agent["engine"] == "kokoro" and agent["llm_provider"] == "fake" and agent["llm_model"] == "fake-model"
    assert agent["fallback"] is False and agent["language"] == "en"
    assert agent["total_ms"] == agent["latency_ms"] == done["total_ms"]
    stored = agent["timings"]
    assert stored["first_audio_ms"] == done["timings"]["first_audio_ms"]
    assert stored["llm_first_token_ms"] == done["timings"]["llm_first_token_ms"]
    assert "total_ms" not in stored  # that lives in its own column

    perf = rig.client.get("/api/perf/turns", headers=LOCAL).json()
    t = perf["turns"][0]
    assert t["first_audio_ms"] == stored["first_audio_ms"] and t["llm_first_token_ms"] == stored["llm_first_token_ms"]
    assert set(t["timings"]) == {"asr_ms", "retrieval_ms", "llm_ms", "tts_ms"}
    assert perf["first_audio_count"] == 1 and perf["first_audio_p50_ms"] == t["first_audio_ms"]
    assert perf["first_audio_p95_ms"] == t["first_audio_ms"] and perf["first_audio_within_target_pct"] == 100.0


# ------------------------------------------------------------------------------------ failures

def test_llm_unavailable_gives_one_fallback_sentence(rig):
    rig.stream = lambda: FakeStream([], fail_after=None)  # no text at all: the empty-reply failure
    rig.stream = lambda: llm.LLMUnavailable({"mlx": "not downloaded"})
    a = make_agent(rig.client, disclose_ai=False)
    events = parse(rig.post(a["id"]))
    assert names(events) == ["transcript", "sentence", "done"]
    assert events[1][1]["text"] == "I'm having trouble responding right now."
    done = events[-1][1]
    assert done["fallback"] is True and done["llm_provider"] is None and done["engine"] == "kokoro"
    assert done["agent_text"] == "I'm having trouble responding right now."
    assert done["route_reason"] == "LLM fallback reply"
    stored = rig.env.local_db.select("conversation_turns", lambda r: r["role"] == "agent")[-1]
    assert stored["fallback"] is True and stored["llm_provider"] is None


def test_fallback_after_the_disclosure(rig):
    rig.stream = lambda: llm.LLMUnavailable({"mlx": "off"})
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"]))
    texts = [d["text"] for e, d in events if e == "sentence"]
    assert texts == ["This is an AI assistant.", "I'm having trouble responding right now."]
    assert events[-1][1]["fallback"] is True


def test_stream_that_dies_before_a_complete_sentence_falls_back(rig):
    rig.stream = lambda: FakeStream(["The clinic op", "ens at"], fail_after=1)
    a = make_agent(rig.client, disclose_ai=False)
    events = parse(rig.post(a["id"]))
    assert [d["text"] for e, d in events if e == "sentence"] == ["I'm having trouble responding right now."]
    assert events[-1][1]["fallback"] is True


def test_stream_that_dies_mid_reply_stops_and_reports(rig):
    rig.stream = lambda: FakeStream(["The clinic opens at nine. ", "Dr. Kulkarni sees pat", "ients"], fail_after=2)
    a = make_agent(rig.client, disclose_ai=False)
    events = parse(rig.post(a["id"]))
    assert [d["text"] for e, d in events if e == "sentence"] == ["The clinic opens at nine."]  # no half sentence
    done = events[-1][1]
    assert done["interrupted"] is True and done["fallback"] is False
    assert done["agent_text"] == "The clinic opens at nine." and done["llm_provider"] == "fake"


def test_empty_model_output_falls_back(rig):
    rig.stream = lambda: FakeStream(["  ", "\n"])
    a = make_agent(rig.client, disclose_ai=False)
    events = parse(rig.post(a["id"]))
    assert events[-1][1]["fallback"] is True


def test_no_speech_and_asr_unavailable_are_error_events(rig):
    a = make_agent(rig.client)

    async def silent(audio_bytes, name, language=None):
        return ""

    async def broken(audio_bytes, name, language=None):
        return None
    rig.env._speech_to_text = silent
    events = parse(rig.post(a["id"]))
    assert names(events) == ["error"] and events[0][1]["code"] == "NO_SPEECH"
    rig.env._speech_to_text = broken
    events = parse(rig.post(a["id"]))
    assert names(events) == ["error"] and events[0][1]["code"] == "ASR_UNAVAILABLE"
    assert rig.env.local_db.select("conversation_turns") == []  # nothing is stored for a failed turn


def test_tts_failure_is_an_error_event(rig):
    def boom(model, text, lang="en"):
        raise RuntimeError("synthesis exploded")
    rig.env._kokoro_generate = boom
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"]))
    assert names(events)[-1] == "error" and events[-1][1]["code"] == "TURN_FAILED"
    assert "exploded" not in json.dumps(events[-1][1])  # no internals in the message


def test_problems_before_the_stream_are_http_errors(rig):
    assert rig.post("not-an-agent").status_code == 404
    a = make_agent(rig.client)
    events = parse(rig.post(a["id"], "no-such-conversation"))
    assert events[-1][0] == "error" and events[-1][1]["code"] == "NOT_FOUND"
    bad = rig.client.post(f"/api/conversations/{a['id']}/turn/stream", headers=LOCAL,
                          files={"audio": ("rec.wav", b"this is not audio", "audio/wav")})
    assert bad.status_code in (400, 415, 422)


# ------------------------------------------------------------------------------------ languages

def test_hinglish_sentences_are_spoken_in_devanagari_with_the_hindi_voice(rig):
    rig.asr_text = "mera refund kab milega"
    rig.stream = lambda: FakeStream(["Aapka refund ", "tees din mein aayega. ", "Aur kuch?"])
    a = make_agent(rig.client, disclose_ai=False)
    events = parse(rig.post(a["id"]))
    assert events[0][1]["language"] == "hinglish"
    shown = [d["text"] for e, d in events if e == "sentence"]
    assert shown == ["Aapka refund tees din mein aayega.", "Aur kuch?"]  # shown as the agent wrote it
    assert [lang for _, lang in rig.spoken] == ["hi", "hi"]
    assert all(any("ऀ" <= ch <= "ॿ" for ch in text) for text, _ in rig.spoken)  # Devanagari for the voice
    assert events[-1][1]["language"] == "hinglish" and events[-1][1]["engine"] == "kokoro"


def test_hindi_reply_splits_at_the_danda(rig):
    rig.asr_text = "क्लिनिक कब खुलता है?"
    rig.stream = lambda: FakeStream(["क्लिनिक सुबह नौ बजे ", "खुलता है। ", "आप आ सकते हैं।"])
    a = make_agent(rig.client, disclose_ai=False, suggested_languages=["en", "hi"])
    events = parse(rig.post(a["id"]))
    assert events[0][1]["language"] == "hi"
    assert [d["text"] for e, d in events if e == "sentence"] == ["क्लिनिक सुबह नौ बजे खुलता है।", "आप आ सकते हैं।"]
    assert [lang for _, lang in rig.spoken] == ["hi", "hi"]


# ------------------------------------------------------------------------------------ sentence audio ownership

def test_sentence_audio_is_owner_checked(rig, app_client):
    a = make_agent(rig.client)
    done = parse(rig.post(a["id"]))[-1][1]
    url = f"/api/conversations/turns/{done['turn_id']}/sentences/0/audio"
    assert app_client.get(url, headers=LOCAL).status_code == 200

    # another user: the stream registry and the stored turn both say it is not theirs
    for c in rig.env.local_db.select("conversations"):
        rig.env.local_db.update("conversations", c["id"], {"user_id": "someone-else"})
    rig.env._STREAM_TURN_OWNERS[done["turn_id"]] = "someone-else"
    assert app_client.get(url, headers=LOCAL).status_code == 404
    assert app_client.get(f"/api/conversations/turns/{done['turn_id']}/audio", headers=LOCAL).status_code == 404


def test_sentence_audio_rejects_unknown_turns_and_indexes(rig, app_client):
    a = make_agent(rig.client)
    done = parse(rig.post(a["id"]))[-1][1]
    tid = done["turn_id"]
    assert app_client.get(f"/api/conversations/turns/{tid}/sentences/99/audio", headers=LOCAL).status_code == 404
    assert app_client.get(f"/api/conversations/turns/{tid}/sentences/-1/audio", headers=LOCAL).status_code == 404
    other = "00000000-0000-0000-0000-000000000000"
    assert app_client.get(f"/api/conversations/turns/{other}/sentences/0/audio", headers=LOCAL).status_code == 404
    assert app_client.get(f"/api/conversations/turns/..%2F..%2Fetc/sentences/0/audio", headers=LOCAL).status_code in (404, 422)


def test_old_sentence_clips_are_pruned(rig, tmp_path):
    import os
    a = make_agent(rig.client)
    done = parse(rig.post(a["id"]))[-1][1]
    clip = tmp_path / f"turn_{done['turn_id']}_0.wav"
    assert clip.exists()
    os.utime(clip, (time.time() - 7200, time.time() - 7200))
    keep = tmp_path / f"turn_{done['turn_id']}.wav"  # the stitched reply is never pruned
    parse(rig.post(a["id"]))
    assert not clip.exists() and keep.exists()


# ------------------------------------------------------------------------------------ the sentence splitter

def split(text, step=None):
    sp = SentenceSplitter()
    out = []
    for i in range(0, len(text), step or len(text) or 1):
        out += sp.feed(text[i:i + (step or len(text) or 1)])
    return out + sp.flush()


@pytest.mark.parametrize("text,expected", [
    ("Hello there. How are you? Fine!", ["Hello there.", "How are you?", "Fine!"]),
    ("Dr. Meera sees patients. Mr. Rao does not.", ["Dr. Meera sees patients.", "Mr. Rao does not."]),
    ("We open at 9 a.m. and close at 5 p.m. daily.", ["We open at 9 a.m. and close at 5 p.m. daily."]),
    ("The fee is 4.5 dollars. Pay at 10.30 sharp.", ["The fee is 4.5 dollars.", "Pay at 10.30 sharp."]),
    ("क्लिनिक खुलता है। फीस 500 रुपये है। धन्यवाद", ["क्लिनिक खुलता है।", "फीस 500 रुपये है।", "धन्यवाद"]),
    ("First line\nSecond line\n\nThird", ["First line", "Second line", "Third"]),
    ("Call A. Kulkarni today. Thanks.", ["Call A. Kulkarni today.", "Thanks."]),
    ('She said "Stop." Then left.', ['She said "Stop."', "Then left."]),
    ("Wait... what? Really?!", ["Wait...", "what?", "Really?!"]),
    ("Steps: 1. Sign in 2. Pay", ["Steps: 1. Sign in 2. Pay"]),
    ("No trailing punctuation", ["No trailing punctuation"]),
    ("...", []),
    ("", []),
])
def test_splitter_edge_cases(text, expected):
    assert split(text) == expected
    assert split(text, 1) == expected    # one character at a time
    assert split(text, 3) == expected    # arbitrary token boundaries


def test_splitter_emits_a_sentence_as_soon_as_the_next_word_starts():
    sp = SentenceSplitter()
    assert sp.feed("Hello there.") == []            # the full stop may be a decimal point: wait for the next char
    assert sp.feed(" How") == ["Hello there."]
    assert sp.flush() == ["How"]


def test_splitter_cuts_a_run_on_at_a_comma():
    text = "word " * 20 + "and, " + "more " * 60
    sp = SentenceSplitter(max_chars=120)
    out = sp.feed(text) + sp.flush()
    assert len(out) > 1 and all(len(s) <= 125 for s in out[:-1])
    assert " ".join(out).split() == text.split()   # nothing lost
