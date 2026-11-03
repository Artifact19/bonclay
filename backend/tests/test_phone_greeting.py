"""The instant phone greeting (services/phone_greeting.py), the warm-up while the phone rings, and the speculative turn
start at the caller's early pause (routers/phone_exotel.py, services/phone_audio.py). Models are faked like in
test_phone_exotel.py: no speech recognition, language model, voice or network."""
import asyncio
import json
import os
import time

import numpy as np
import pytest

from routers import phone_exotel as px
from services import llm
from services import phone_audio as pa
from services import phone_greeting as pg
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_phone_exotel import (STREAM_SID, configure, exotel, feed, is_mark, pcm_silence, pcm_tone,  # noqa: F401
                                     read_until, rig, start_msg, token_for)
from tests.test_turn_stream import REPLY, FakeStream


def stop(ws):
    ws.send_text(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))


def wait_until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def conversation_turns(env, conversation_id):
    return env.local_db.select("conversation_turns", lambda r: r.get("conversation_id") == conversation_id)


def receipts(tmp_path):
    p = tmp_path / "receipts.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def agent_row(env, a):
    return env.local_db.get("agents", a["id"])


@pytest.fixture()
def sessions(monkeypatch):
    """Every CallSession of the test, with the time of each message it sent."""
    made = []

    class Recording(px.CallSession):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.sent_at = []
            self.confirmed_at = None
            made.append(self)

        async def _send(self, obj):
            self.sent_at.append((time.monotonic(), obj.get("event"), (obj.get("mark") or {}).get("name")))
            await super()._send(obj)

        def _confirm(self, spec, end):
            self.confirmed_at = time.monotonic()
            super()._confirm(spec, end)

    monkeypatch.setattr(px, "CallSession", Recording)
    return made


@pytest.fixture()
def watch(rig, env, monkeypatch):
    """Counts the knowledge searches and speech recognitions (the language model is counted by the rig)."""
    seen = {"retrieval": [], "asr": 0}
    real_retrieve = env.retrieve_context

    async def retrieve(agent_id, query, *a, **k):
        seen["retrieval"].append(query)
        return await real_retrieve(agent_id, query, *a, **k)

    async def asr(audio_bytes, name, language=None):
        seen["asr"] += 1
        return rig.asr_text

    monkeypatch.setattr(env, "retrieve_context", retrieve)
    monkeypatch.setattr(env, "_speech_to_text", asr)
    return seen


# --------------------------------------------------------------------------------------------- the detector

def detector(early=250, end=500):
    return pa.UtteranceDetector(8000, vad=pa.EnergyVad(8000),
                                config=pa.DetectorConfig(end_silence_ms=end, early_silence_ms=early))


def kinds(events):
    return [e.kind for e in events]


def test_a_pause_then_more_speech_gives_pause_resume_and_one_utterance_at_the_end():
    d = detector()
    first = d.feed(pcm_silence(200) + pcm_tone(800) + pcm_silence(300))
    assert kinds(first) == ["barge", "pause"]
    assert first[1].audio and first[1].speech_ms >= 400
    assert kinds(d.feed(pcm_tone(400))) == ["resume"]
    last = d.feed(pcm_silence(600))
    assert kinds(last) == ["pause", "utterance"]
    assert len(last[1].audio) > len(first[1].audio)     # the utterance carries all of it


def test_a_confirmed_pause_carries_the_same_audio_as_the_utterance():
    d = detector()
    events = d.feed(pcm_silence(200) + pcm_tone(900) + pcm_silence(700))
    assert kinds(events) == ["barge", "pause", "utterance"]
    assert events[1].audio == events[2].audio            # what the speculative turn heard is what the caller said


def test_no_pause_without_early_silence_or_for_a_short_noise():
    assert kinds(detector(early=0).feed(pcm_tone(900) + pcm_silence(700))) == ["barge", "utterance"]
    assert "pause" not in kinds(detector().feed(pcm_tone(200) + pcm_silence(700)))   # too short to be speech


# ------------------------------------------------------------------------------------------- the greeting

def test_the_greeting_goes_out_on_start_without_the_model_search_or_recognition(rig, env, watch, caplog):
    caplog.set_level("INFO", logger="bonclay.phone.exotel")
    a = make_agent(rig.client)
    # first call: the cache is cold, so the greeting is rendered now (still no model, no search, no recognition)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(start_msg())
        msgs = read_until(ws, is_mark)
        stop(ws)
    assert any(m["event"] == "media" for m in msgs) and msgs[-1]["mark"]["name"] == "turn-1-end"
    assert rig.asked == [] and watch["retrieval"] == [] and watch["asr"] == 0
    spoken = [t for t, _ in rig.spoken]
    assert spoken == ["This is an AI assistant.", "Hi, I'm Ria."]         # the disclosure, then the agent's greeting
    assert "(rendered;" in caplog.text

    # second call: straight from memory, nothing is synthesized again
    caplog.clear()
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        msgs = read_until(ws, is_mark)
        stop(ws)
    assert any(m["event"] == "media" for m in msgs)
    assert rig.asked == [] and watch["retrieval"] == [] and watch["asr"] == 0 and len(rig.spoken) == 2
    assert "(memory;" in caplog.text and "start->first byte" in caplog.text


def test_the_greeting_is_stored_as_the_first_turn_and_the_reply_does_not_disclose_again(rig, env, sessions):
    rig.asr_text = "When does the clinic open?"
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in sessions[0].log))
        feed(ws, pcm_silence(300) + pcm_tone(1200) + pcm_silence(900))
        msgs = read_until(ws, is_mark)
        assert wait_until(lambda: any(e["event"] == "said" for e in sessions[0].log))
        stop(ws)
    assert msgs[-1]["mark"]["name"] == "turn-2-end"
    conv = sessions[0].conversation_id
    rows = conversation_turns(env, conv)
    assert [r["role"] for r in rows] == ["user", "agent", "user", "agent"]
    assert rows[0]["text_content"] == pg.CALL_MARKER
    assert rows[1]["text_content"] == "This is an AI assistant. Hi, I'm Ria."
    assert not rows[3]["text_content"].startswith("This is an AI assistant.")   # disclosed once, in the greeting
    history = rig.asked[-1]["messages"]
    assert {"role": "assistant", "content": "This is an AI assistant. Hi, I'm Ria."} in history
    conv_row = env.local_db.get("conversations", conv)
    assert conv_row["source"] == "phone"


def test_the_greeting_clip_is_watermarked_once_with_a_receipt(rig, env, tmp_path):
    a = agent_row(env, make_agent(rig.client))
    g = asyncio.run(pg.ensure(env, a, "local-user", "local"))
    rs = [r for r in receipts(tmp_path) if r.get("turn_id") == g.clip_id]
    assert len(rs) == 1 and rs[0]["receipt_id"] == g.receipt_id and rs[0]["kind"] == "agent_reply"
    from services import provenance
    assert provenance.read_wav_label(g.wav)["ICMT"].endswith(g.receipt_id)   # the file says it is AI and names it
    assert g.pcm[8000] and len(g.pcm[8000]) % 2 == 0
    asyncio.run(pg.ensure(env, a, "local-user", "local"))                      # cached: no second receipt
    assert len([r for r in receipts(tmp_path) if r.get("turn_id")]) == 1


def test_the_disk_cache_is_used_after_a_restart_and_for_another_rate(rig, env, tmp_path):
    a = agent_row(env, make_agent(rig.client))
    g = asyncio.run(pg.ensure(env, a, "local-user", "local"))
    folder = tmp_path / "phone_greetings"
    assert {f"{g.key}.wav", f"{g.key}.json", f"{g.key}.8000.pcm"} <= set(os.listdir(folder))
    pg._mem.clear()                                                            # a restart forgets the memory
    spoken = len(rig.spoken)
    g2, pcm, source = asyncio.run(pg.for_call(env, a, "local-user", "local", 8000))
    assert source == "disk" and pcm == g.pcm[8000] and g2.receipt_id == g.receipt_id and len(rig.spoken) == spoken
    pg._mem.clear()
    _g3, pcm16, source = asyncio.run(pg.for_call(env, a, "local-user", "local", 16000))   # Exotel announced 16 kHz
    assert source == "disk" and len(rig.spoken) == spoken
    assert abs(len(pcm16) - 2 * len(pcm)) <= 8 and (folder / f"{g.key}.16000.pcm").exists()


def test_the_cache_key_follows_the_words_the_voice_and_the_settings(rig, env, monkeypatch, tmp_path):
    a = agent_row(env, make_agent(rig.client))
    key = pg.greeting_key(env, a)
    assert pg.greeting_key(env, dict(a)) == key
    assert pg.greeting_key(env, {**a, "greeting": "Welcome to the clinic."}) != key
    assert pg.greeting_key(env, {**a, "voice_id": "v-1"}) != key
    assert pg.greeting_key(env, {**a, "disclose_ai": False}) != key
    assert pg.greeting_key(env, {**a, "id": "another"}) != key
    monkeypatch.setenv("PHONE_CLONE_VOICE", "1")
    assert pg.greeting_key(env, a) != key
    monkeypatch.delenv("PHONE_CLONE_VOICE")
    monkeypatch.setenv("WATERMARK", "off")
    assert pg.greeting_key(env, a) != key
    monkeypatch.delenv("WATERMARK")
    # a changed greeting renders a new clip and the old one of that agent is removed from disk
    g1 = asyncio.run(pg.ensure(env, a, "local-user", "local"))
    env.local_db.update("agents", a["id"], {"greeting": "Welcome to the clinic."})
    a2 = agent_row(env, a)
    g2 = asyncio.run(pg.ensure(env, a2, "local-user", "local"))
    assert g2.key != g1.key and g2.text == "This is an AI assistant. Welcome to the clinic."
    names = os.listdir(tmp_path / "phone_greetings")
    assert not any(n.startswith(g1.key) for n in names) and f"{g2.key}.wav" in names


def test_an_empty_greeting_and_no_disclosure(rig, env):
    a = agent_row(env, make_agent(rig.client))
    lang, disclosure, greeting = pg.opening(env, {**a, "greeting": "", "supported_languages": ["hi", "en"]})
    assert (lang, disclosure, greeting) == ("hi", "यह एक AI सहायक है।", pg.GENERIC_GREETINGS["hi"])
    assert pg.opening(env, {**a, "disclose_ai": False})[1] is None
    assert pg.opening(env, {**a, "greeting": "", "supported_languages": ["ta"]})[2] == pg.GENERIC_GREETINGS["en"]


def test_talking_over_the_greeting_cuts_it_and_the_first_reply_discloses(rig, env, monkeypatch, sessions):
    long_sentence = np.full(24000 * 6, 0.05, dtype=np.float32)    # 6 s per sentence: the disclosure is still playing
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": long_sentence)
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, lambda m: m["event"] == "media")             # the greeting is playing
        feed(ws, pcm_tone(800))                                     # the caller talks over it
        msgs = read_until(ws, lambda m: m["event"] == "clear")
        assert msgs[-1] == {"event": "clear", "stream_sid": STREAM_SID}
        feed(ws, pcm_silence(900))                                  # ...and finishes their sentence
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-2-end")
        assert wait_until(lambda: any(e["event"] == "said" for e in sessions[0].log))
        stop(ws)
    s = sessions[0]
    assert s.barges == 1 and any(e["event"] == "greeting_not_recorded" for e in s.log)
    rows = conversation_turns(env, s.conversation_id)
    assert [r["role"] for r in rows] == ["user", "agent"]
    assert rows[1]["text_content"].startswith("This is an AI assistant.")   # the reply said the disclosure instead


# ------------------------------------------------------------------------------------ speculative turn start

def slow_first_stream(rig, first_delay):
    calls = {"n": 0}

    def stream():
        calls["n"] += 1
        return FakeStream(REPLY, first_delay=first_delay if calls["n"] == 1 else 0.0)
    rig.stream = stream


def test_a_speculative_turn_is_cancelled_when_the_caller_goes_on_and_stores_nothing(rig, env, sessions, tmp_path):
    rig.asr_text = "When does the clinic open?"
    slow_first_stream(rig, 1.0)        # the speculative turn is inside the language model when the caller goes on
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        receipts_before = len(receipts(tmp_path))
        feed(ws, pcm_silence(300) + pcm_tone(1000) + pcm_silence(320))     # a pause: the turn starts speculatively
        assert wait_until(lambda: len(rig.asked) == 1)                     # ...and has reached the language model
        feed(ws, pcm_tone(600) + pcm_silence(900))                         # the caller goes on, then really finishes
        msgs = read_until(ws, is_mark)
        assert wait_until(lambda: any(e["event"] == "said" for e in s.log))
        stop(ws)
    events = [e["event"] for e in s.log]
    assert "speculative_start" in events and "speculative_cancelled" in events
    assert msgs[-1]["mark"]["name"] == "turn-2-end"                    # the cancelled turn's number is reused
    rows = conversation_turns(env, s.conversation_id)
    assert [r["role"] for r in rows] == ["user", "agent", "user", "agent"]   # the greeting and ONE reply
    assert len(receipts(tmp_path)) == receipts_before + 1                    # one receipt: the reply that was spoken
    stored = {r["id"] for r in rows}
    clips = [n for n in os.listdir(tmp_path) if n.startswith("turn_") and n.count("_") == 2]
    assert all(n.split("_")[1] in stored for n in clips)                     # no sentence clip of the cancelled turn
    assert s.spec_stats["cancelled"] >= 1 and s.spec_stats["confirmed"] >= 1


def test_speculative_audio_is_held_until_the_end_of_speech_is_confirmed(rig, env, sessions):
    rig.asr_text = "When does the clinic open?"
    rig.stream = lambda: FakeStream(REPLY, delay=0.15)   # the first sentence is ready quickly, the reply ends later
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        feed(ws, pcm_silence(300) + pcm_tone(1000) + pcm_silence(320))     # pause: the speculative turn starts
        assert wait_until(lambda: "sentence" in s.tm)                       # its first sentence is ready...
        time.sleep(0.2)
        sent_before = [x for x in s.sent_at if x[1] == "media"]
        feed(ws, pcm_silence(400))                                          # the end of speech is confirmed
        msgs = read_until(ws, is_mark)
        stop(ws)
    assert s.confirmed_at is not None
    greeting_media = len(sent_before)
    reply_media = [t for t, kind, _ in s.sent_at[greeting_media:] if kind == "media"]
    assert reply_media and all(t >= s.confirmed_at for t in reply_media)      # ...but nothing went out before
    assert msgs[-1]["mark"]["name"] == "turn-2-end"
    [row] = [e for e in s.log if e["event"] == "latency"]
    assert row["speculative"] is True and row["start_ms"] < 0 and row["sentence_ms"] < 0 <= row["audio_ms"]


@pytest.fixture()
def dispatched(env, monkeypatch):
    """The turns handed to the webhook / Sheets dispatcher (integrations.dispatch_turn), by turn id."""
    seen = []
    monkeypatch.setattr(env.integrations_mod, "dispatch_turn", lambda run, turn_id, text, latency: seen.append(turn_id))
    return seen


def finished_speculation(rig, ws, s):
    """Start a speculative turn and let it finish the model and every sentence while the caller is still pausing. Returns
    its turn id. The fake model answers at once, so all that is left when this returns is the commit (store, receipt)."""
    spoken_before = len(rig.spoken)
    feed(ws, pcm_silence(300) + pcm_tone(1000) + pcm_silence(320))     # pause: the speculative turn starts
    assert wait_until(lambda: s.spec is not None)
    turn_id = s.spec.turn_id
    assert wait_until(lambda: len(rig.spoken) >= spoken_before + 2)   # REPLY is two sentences (the greeting disclosed)
    time.sleep(0.4)                                                    # well past the end of the reply
    assert not s.spec.task.done()                                      # ...and it is waiting at the commit gate
    return turn_id


def test_a_speculative_turn_that_finished_early_stores_nothing_until_confirmed(rig, env, sessions, tmp_path, dispatched):
    rig.asr_text = "When does the clinic open?"
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        receipts_before = len(receipts(tmp_path))
        dispatched.clear()                                                 # the greeting turn was dispatched
        turn_id = finished_speculation(rig, ws, s)
        assert len(rig.asked) == 1                                         # the model has answered...
        assert [r["role"] for r in conversation_turns(env, s.conversation_id)] == ["user", "agent"]   # ...not stored
        assert len(receipts(tmp_path)) == receipts_before and dispatched == []
        feed(ws, pcm_silence(400))                                         # the end of speech is confirmed
        read_until(ws, is_mark)
        assert wait_until(lambda: any(e["event"] == "said" for e in s.log))
        stop(ws)
    rows = conversation_turns(env, s.conversation_id)
    assert [r["role"] for r in rows] == ["user", "agent", "user", "agent"]
    assert rows[-1]["id"] == turn_id and rows[-1]["timings"]["commit_wait_ms"] >= 300
    assert len(receipts(tmp_path)) == receipts_before + 1 and dispatched == [turn_id]
    assert not any(e["event"] == "speculative_stored_early" for e in s.log)


def test_a_cancelled_speculative_turn_that_had_finished_leaves_nothing_behind(rig, env, sessions, tmp_path, dispatched):
    rig.asr_text = "When does the clinic open?"
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        receipts_before = len(receipts(tmp_path))
        dispatched.clear()                                                 # the greeting turn was dispatched
        spec_id = finished_speculation(rig, ws, s)
        feed(ws, pcm_tone(600) + pcm_silence(900))                         # the caller goes on, then really finishes
        msgs = read_until(ws, is_mark)
        assert wait_until(lambda: any(e["event"] == "said" for e in s.log))
        stop(ws)
    assert "speculative_cancelled" in [e["event"] for e in s.log]
    assert msgs[-1]["mark"]["name"] == "turn-2-end"
    rows = conversation_turns(env, s.conversation_id)
    assert [r["role"] for r in rows] == ["user", "agent", "user", "agent"]   # the greeting and the ONE confirmed reply
    assert spec_id not in {r["id"] for r in rows}
    assert len(receipts(tmp_path)) == receipts_before + 1                    # only the reply that was spoken
    assert spec_id not in {r.get("turn_id") for r in receipts(tmp_path)}
    assert len(dispatched) == 1 and dispatched[0] != spec_id                 # no webhook for the cancelled turn
    assert not [n for n in os.listdir(tmp_path) if n.startswith(f"turn_{spec_id}")]   # no clip, no stitched reply


def test_a_speculative_instant_answer_also_waits_for_the_commit_gate(rig, env, sessions, tmp_path, dispatched):
    from services import instant_answers as ia
    from tests.test_offline_agents import fake_embed
    ia.set_embedder(lambda texts: fake_embed(list(texts)))
    ia.set_fingerprinter(lambda *a: "fp")
    try:
        rig.asr_text = "When does the clinic open?"
        a = make_agent(rig.client)
        ia.get_store("local").insert(ia.new_row(
            agent_id=a["id"], user_id="local-user", question="When does the clinic open?",
            answer="We open at nine in the morning.", language="en", source="owner", status=ia.STATUS_APPROVED,
            fingerprint="fp", embed=lambda t: fake_embed(list(t))))
        with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
            ws.send_text(start_msg())
            read_until(ws, is_mark)
            s = sessions[0]
            assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
            dispatched.clear()
            spoken_before = len(rig.spoken)
            feed(ws, pcm_silence(300) + pcm_tone(1000) + pcm_silence(320))     # pause: the speculative turn starts
            assert wait_until(lambda: s.spec is not None)
            turn_id = s.spec.turn_id
            assert wait_until(lambda: len(rig.spoken) > spoken_before)        # the instant answer is rendered...
            time.sleep(0.4)
            assert len(conversation_turns(env, s.conversation_id)) == 2 and dispatched == []   # ...but not stored
            feed(ws, pcm_silence(400))                                          # the end of speech is confirmed
            read_until(ws, is_mark)
            assert wait_until(lambda: any(e["event"] == "said" for e in s.log))
            stop(ws)
        assert rig.asked == []                                                  # no language model at all
        rows = conversation_turns(env, s.conversation_id)
        assert rows[-1]["id"] == turn_id and rows[-1]["text_content"] == "We open at nine in the morning."
        assert rows[-1]["timings"]["instant_answer_id"] and dispatched == [turn_id]
    finally:
        ia.set_embedder(None)
        ia.set_fingerprinter(None)


def test_a_turn_without_a_commit_gate_does_not_wait(rig, env):
    """Every turn that is not a speculative phone turn (playground, widget) has no gate and is stored as before."""
    from tests.test_turn_stream import parse
    a = make_agent(rig.client)
    r = rig.client.post(f"/api/conversations/{a['id']}/turn/stream", headers=LOCAL,
                        files={"audio": ("rec.wav", rig.wav, "audio/wav")})
    done = parse(r)[-1][1]
    assert "commit_wait_ms" not in done["timings"]
    assert len(conversation_turns(env, done["conversation_id"])) == 2


def test_in_asr_mode_a_speculative_start_stops_after_speech_recognition(rig, env, sessions, watch, monkeypatch):
    monkeypatch.setenv("PHONE_SPECULATIVE", "asr")
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        feed(ws, pcm_silence(300) + pcm_tone(1000) + pcm_silence(320))     # pause: recognition runs...
        assert wait_until(lambda: "heard" in s.tm)
        time.sleep(0.3)
        assert rig.asked == []                                              # ...but nothing after it
        feed(ws, pcm_tone(600) + pcm_silence(900))                          # the caller goes on, then finishes
        read_until(ws, is_mark)
        stop(ws)
    assert s.spec_stats["cancelled"] >= 1 and len(rig.asked) == 1 and watch["asr"] >= 2


def test_speculation_can_be_switched_off(rig, env, sessions, monkeypatch):
    monkeypatch.setenv("PHONE_SPECULATIVE", "off")
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        feed(ws, pcm_silence(300) + pcm_tone(1200) + pcm_silence(900))
        read_until(ws, is_mark)
        stop(ws)
    s = sessions[0]
    assert s.spec_stats["started"] == 0
    [row] = [e for e in s.log if e["event"] == "latency"]
    assert row["speculative"] is False and row["start_ms"] >= 0


def test_no_speculation_for_an_agent_with_tools_or_without_a_conversation(rig, env):
    s = px.CallSession(ws=None, claims={"u": "local-user", "m": "local", "a": "x"}, agent={"id": "x", "tools": []})
    ev = pa.VadEvent("pause", audio=b"\x00" * 3200, speech_ms=800)
    assert s._may_speculate(ev) is False             # no conversation yet
    s.conversation_id = "c1"
    assert s._may_speculate(ev) is True
    s.agent = {"id": "x", "tools": ["appointments"]}
    assert s._may_speculate(ev) is False             # a tool may act (book): never on an unfinished sentence


# --------------------------------------------------------------------------------- triggers and warm-up

def test_placing_a_call_starts_the_greeting_and_warm_up_without_waiting(rig, exotel, monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv("PHONE_PREWARM", "on")
    started = []

    async def slow_prepare(b, agent, user_id, mode, *, warm=True, rates=(8000,)):
        started.append({"agent": agent["id"], "user": user_id, "mode": mode, "warm": warm})
        await asyncio.sleep(1.5)

    monkeypatch.setattr(pg, "prepare", slow_prepare)
    a = make_agent(rig.client)
    t = time.monotonic()
    r = rig.client.post("/api/phone/exotel/call", headers=LOCAL,
                        json={"agent_id": a["id"], "to": "+919876543210", "caller_agreed": True})
    took = time.monotonic() - t
    assert r.status_code == 200, r.text
    assert took < 1.0                                                  # the response did not wait for the warm-up
    assert started == [{"agent": a["id"], "user": "local-user", "mode": "local", "warm": True}]
    r = rig.client.post("/api/phone/exotel/inbound-link", headers=LOCAL, json={"agent_id": a["id"]})
    assert r.status_code == 200 and started[-1]["warm"] is False       # an inbound link renders the greeting only


def test_prewarm_is_off_when_switched_off(rig, exotel, monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv("PHONE_PREWARM", "off")
    called = []

    async def prepare(*a, **k):
        called.append(1)
    monkeypatch.setattr(pg, "prepare", prepare)
    a = make_agent(rig.client)
    assert rig.client.post("/api/phone/exotel/call", headers=LOCAL,
                           json={"agent_id": a["id"], "to": "+919876543210", "caller_agreed": True}).status_code == 200
    assert called == []


def test_warm_up_loads_what_the_first_reply_needs_once(rig, env, monkeypatch):
    monkeypatch.setenv("PHONE_VAD", "energy")
    monkeypatch.delattr(env, "warm_for_live_turns", raising=False)        # the fallback: this module's own steps
    loaded, prewarmed, searched = [], [], []

    async def load(key):
        loaded.append(key)
        return object()

    async def retrieve(agent_id, query, *a, **k):
        searched.append(query)
        return []

    monkeypatch.setattr(env.advanced_manager, "load_model", load)
    monkeypatch.setattr(env, "retrieve_context", retrieve)
    monkeypatch.setattr(env, "groq_client", object(), raising=False)   # speech recognition is Groq: nothing to load
    monkeypatch.setattr(llm, "prewarm", lambda: prewarmed.append(1), raising=False)
    a = agent_row(env, make_agent(rig.client))
    res = asyncio.run(pg.warm_up(env, a, "local-user", "local"))
    assert prewarmed == [1] and searched == ["Hello"] and res["asr"].startswith("skipped")
    assert set(loaded) == {"kokoro"}                                    # no clone, Indic or other heavy engine
    again = asyncio.run(pg.warm_up(env, a, "local-user", "local"))
    assert again["skipped"] == "warm" and set(again) == {"llm", "skipped"}   # only the connection is refreshed
    assert prewarmed == [1, 1] and searched == ["Hello"]


def test_warm_up_uses_the_backends_shared_live_warm_up_when_there_is_one(rig, env, monkeypatch):
    monkeypatch.setenv("PHONE_VAD", "energy")
    shared, loaded = [], []

    async def warm_for_live_turns(reason="manual", agent_id=None):
        shared.append((reason, agent_id))
        return {"llm": {"state": "ready"}, "knowledge": {"state": "ready"}}

    async def load(key):
        loaded.append(key)
        return object()

    monkeypatch.setattr(env, "warm_for_live_turns", warm_for_live_turns, raising=False)
    monkeypatch.setattr(env.advanced_manager, "load_model", load)
    a = agent_row(env, make_agent(rig.client))
    res = asyncio.run(pg.warm_up(env, a, "local-user", "local"))
    assert shared == [("phone", a["id"])] and "knowledge:ready" in res["live"]
    assert set(res) == {"live", "vad", "tts"} and set(loaded) == {"kokoro"}


def test_warm_up_works_without_llm_prewarm(rig, env, monkeypatch):
    monkeypatch.setenv("PHONE_VAD", "energy")
    monkeypatch.delattr(env, "warm_for_live_turns", raising=False)
    monkeypatch.delattr(llm, "prewarm", raising=False)
    monkeypatch.setattr(env, "groq_client", object(), raising=False)
    a = agent_row(env, make_agent(rig.client))
    res = asyncio.run(pg.warm_up(env, a, "local-user", "local"))
    assert res["llm"].startswith("skipped") and not any(str(v).startswith("failed") for v in res.values())


def test_prewarm_all_renders_the_greetings_of_agents_that_take_calls(rig, env, monkeypatch):
    monkeypatch.setenv("PHONE_PREWARM", "on")
    a = agent_row(env, make_agent(rig.client))
    b = agent_row(env, make_agent(rig.client))
    pg.remember_agent(env, a["id"], "local-user", "local")
    env.local_db.insert("conversations", {"agent_id": b["id"], "user_id": "local-user", "source": "phone"})
    env.local_db.insert("agents", {"user_id": "someone-else", "name": "x"})   # not on any call: left alone
    ready = asyncio.run(pg.prewarm_all(env))
    assert set(ready) == {a["id"], b["id"]}
    assert pg.cached(env, a) is not None and pg.cached(env, b) is not None
    pg._mem.clear()
    spoken = len(rig.spoken)
    assert set(asyncio.run(pg.prewarm_all(env))) == {a["id"], b["id"]}    # after a restart: from disk, no new render
    assert len(rig.spoken) == spoken


def test_the_startup_hook_is_installed_and_schedules_the_prewarm(monkeypatch):
    """On a real FastAPI app (Starlette 1.x has no `app.add_event_handler`): install() registers a startup handler, and
    starting the app schedules prewarm_all_later in the background without awaiting it. No model is touched."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    started = []

    async def fake_later(b):
        started.append(b)

    spawned = []
    real_spawn = pg._spawn

    def spawn(coro):
        spawned.append(coro)
        return real_spawn(coro)
    monkeypatch.setattr(pg, "prewarm_all_later", fake_later)
    monkeypatch.setattr(pg, "_spawn", spawn)
    app = FastAPI()
    before = len(app.router.on_startup)
    px.install(app)
    assert len(app.router.on_startup) == before + 1
    assert app.router.on_startup[-1].__name__ == "_phone_greetings_at_startup"
    with TestClient(app):        # runs the startup handlers
        assert wait_until(lambda: len(started) == 1)
    assert len(spawned) == 1


def test_the_real_app_registers_the_phone_startup_hook(app_client):
    import backend
    names = [getattr(fn, "__name__", "") for fn in backend.app.router.on_startup]
    assert "_phone_greetings_at_startup" in names
