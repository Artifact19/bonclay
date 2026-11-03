"""The streaming turn's filler clip ("One moment."): sent as a `sentence` event with `filler: true` only when the
answer is slow, never while the AI disclosure plays, never twice, and never part of the reply text or its audio."""
import io

import pytest
import soundfile as sf

from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import REPLY, FakeStream, Rig, parse  # noqa: F401  (rig fixture is rebuilt below)


@pytest.fixture()
def rig(app_client, env, monkeypatch, ref_wav):
    env._FILLER_CLIPS.clear()
    env._FILLER_LAST.clear()
    monkeypatch.setenv("FILLER_AFTER_MS", "150")
    monkeypatch.delenv("FILLER_AUDIO", raising=False)
    return Rig(env, monkeypatch, app_client, ref_wav)


def sentences(events):
    return [d for e, d in events if e == "sentence"]


def slow(rig, delay=0.6):
    rig.stream = lambda: FakeStream(REPLY, first_delay=delay)


def test_a_slow_answer_is_preceded_by_a_filler(rig):
    slow(rig)
    a = make_agent(rig.client, disclose_ai=False)
    events = parse(rig.post(a["id"]))
    s = sentences(events)
    assert [x["index"] for x in s] == [0, 1, 2]
    assert s[0]["filler"] is True and s[0]["text"] in ("One moment.", "Let me check.")
    assert all("filler" not in x for x in s[1:])
    assert s[1]["text"] == "The clinic opens at 9 a.m. on Monday."
    assert rig.client.get(s[0]["audio_url"], headers=LOCAL).status_code == 200  # served like any sentence clip


def test_a_fast_answer_gets_no_filler(rig):
    a = make_agent(rig.client, disclose_ai=False)
    s = sentences(parse(rig.post(a["id"])))
    assert [x["index"] for x in s] == [0, 1] and not any(x.get("filler") for x in s)
    assert all(t not in ("One moment.", "Let me check.") for t, _ in rig.spoken)


def test_the_filler_is_not_part_of_the_reply_text_the_transcript_or_the_stitched_audio(rig):
    slow(rig)
    a = make_agent(rig.client, disclose_ai=False)
    events = parse(rig.post(a["id"]))
    done = events[-1][1]
    s = sentences(events)
    answer = [x for x in s if not x.get("filler")]
    assert len(answer) == 2 and len(s) == 3
    assert done["agent_text"] == " ".join(x["text"] for x in answer)
    assert s[0]["text"] not in done["agent_text"]
    rows = rig.env.local_db.select("conversation_turns", lambda r: r["conversation_id"] == done["conversation_id"])
    assert s[0]["text"] not in " ".join(r["text_content"] for r in rows)
    full = rig.client.get(done["agent_audio_url"], headers=LOCAL)
    data, sr = sf.read(io.BytesIO(full.content))
    assert len(data) == 2400 * len(answer) + int(sr * rig.env.SENTENCE_GAP_S) * (len(answer) - 1)
    # the first answer audio is reported apart from the first audio (the filler)
    t = done["timings"]
    assert t["first_answer_audio_ms"] > t["first_audio_ms"]


def test_the_filler_is_watermarked_with_the_turns_message(rig):
    slow(rig)
    a = make_agent(rig.client, disclose_ai=False)
    marks = []
    orig = rig.env._watermark_clip

    def spy(wav, sr, turn_id):
        out = orig(wav, sr, turn_id)
        marks.append(turn_id)
        return out
    rig.env._watermark_clip = spy
    try:
        events = parse(rig.post(a["id"]))
    finally:
        rig.env._watermark_clip = orig
    assert len(marks) == 3 and marks == [events[-1][1]["turn_id"]] * 3


def test_never_during_the_disclosure(rig):
    slow(rig, 1.0)
    a = make_agent(rig.client)  # discloses: the first sentence is the AI disclosure
    s = sentences(parse(rig.post(a["id"])))
    assert s[0]["text"] == "This is an AI assistant." and not any(x.get("filler") for x in s)


def test_at_most_one_filler_per_turn(rig):
    rig.stream = lambda: FakeStream(REPLY, first_delay=0.5, delay=0.4)
    a = make_agent(rig.client, disclose_ai=False)
    s = sentences(parse(rig.post(a["id"])))
    assert sum(1 for x in s if x.get("filler")) == 1


def test_it_can_be_switched_off(rig, monkeypatch):
    monkeypatch.setenv("FILLER_AUDIO", "off")
    slow(rig)
    a = make_agent(rig.client, disclose_ai=False)
    s = sentences(parse(rig.post(a["id"])))
    assert not any(x.get("filler") for x in s) and len(s) == 2


def test_the_filler_lines_alternate_within_a_conversation(rig):
    slow(rig, 0.4)
    a = make_agent(rig.client, disclose_ai=False)
    first = parse(rig.post(a["id"]))
    conv = first[-1][1]["conversation_id"]
    second = parse(rig.post(a["id"], conv))
    third = parse(rig.post(a["id"], conv))
    texts = [sentences(e)[0]["text"] for e in (first, second, third)]
    assert texts[0] != texts[1] and texts[1] != texts[2] and texts[0] == texts[2]


def test_the_filler_is_rendered_once_and_reused(rig):
    slow(rig, 0.4)
    a = make_agent(rig.client, disclose_ai=False)
    texts = []
    for _ in range(4):  # new conversations each time: the first line every time
        texts.append(sentences(parse(rig.post(a["id"])))[0])
    assert all(t.get("filler") for t in texts)
    assert [t for t, _ in rig.spoken].count(texts[0]["text"]) == 1


def test_hindi_and_hinglish_use_the_hindi_fillers(rig):
    slow(rig, 0.4)
    rig.asr_text = "क्लिनिक कब खुलता है?"
    rig.stream = lambda: FakeStream(["क्लिनिक सुबह नौ बजे ", "खुलता है। "], first_delay=0.4)
    a = make_agent(rig.client, disclose_ai=False, suggested_languages=["en", "hi"])
    s = sentences(parse(rig.post(a["id"])))
    assert s[0]["filler"] is True and s[0]["text"] in ("एक सेकंड।", "देखती हूँ।")
    assert rig.spoken[0] == (s[0]["text"], "hi")

    rig.spoken.clear()
    rig.asr_text = "mera refund kab milega"
    rig.stream = lambda: FakeStream(["Aapka refund ", "tees din mein aayega. "], first_delay=0.4)
    b = make_agent(rig.client, disclose_ai=False)
    s = sentences(parse(rig.post(b["id"])))
    assert s[0]["filler"] is True and s[0]["text"] in ("एक सेकंड।", "देखती हूँ।")
    assert rig.spoken[0][1] == "hi"
    assert s[1]["text"] == "Aapka refund tees din mein aayega."  # the answer is still shown as the agent wrote it


def test_english_fillers_are_english_and_the_language_helpers(env):
    assert env._pick_filler("en", "c1") == "One moment."
    assert env._pick_filler("en", "c1") == "Let me check."
    assert env._pick_filler("hinglish", None) == "एक सेकंड।"
    assert env._pick_filler("mr", "c2") is None


def test_a_slow_retrieval_also_gets_a_filler_before_the_model_starts(rig, monkeypatch):
    real = rig.env._turn_context

    async def slow_context(run, conversation_id, t, **kw):
        import asyncio
        await asyncio.sleep(0.6)
        return await real(run, conversation_id, t, **kw)
    monkeypatch.setattr(rig.env, "_turn_context", slow_context)
    a = make_agent(rig.client, disclose_ai=False)
    s = sentences(parse(rig.post(a["id"])))
    assert [x["index"] for x in s] == [0, 1, 2] and s[0]["filler"] is True
    assert "filler" not in s[1] and s[1]["text"] == "The clinic opens at 9 a.m. on Monday."


def test_a_slow_retrieval_on_a_disclosing_turn_gets_no_filler(rig, monkeypatch):
    real = rig.env._turn_context

    async def slow_context(run, conversation_id, t, **kw):
        import asyncio
        await asyncio.sleep(0.6)
        return await real(run, conversation_id, t, **kw)
    monkeypatch.setattr(rig.env, "_turn_context", slow_context)
    a = make_agent(rig.client)
    s = sentences(parse(rig.post(a["id"])))
    assert s[0]["text"] == "This is an AI assistant." and not any(x.get("filler") for x in s)

