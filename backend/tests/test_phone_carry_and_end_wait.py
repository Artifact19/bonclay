"""A caller who gives details in pieces on a phone call (routers/phone_exotel.py): the words of a turn that was cancelled
before it was stored are carried into the next turn (PHONE_CARRY_UNANSWERED), the wait for the end of a turn follows what
the agent just asked (PHONE_ADAPTIVE_END), and an answer that looks unfinished is waited for once more. Models are faked
like in test_phone_exotel.py and test_phone_greeting.py: no speech recognition, language model, voice or network."""
import asyncio
import io
import json
import time
import wave

import numpy as np
import pytest

from routers import phone_exotel as px
from services import phone_audio as pa
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_phone_exotel import (STREAM_SID, feed, is_mark, pcm_silence, pcm_tone, read_until,  # noqa: F401
                                     rig, start_msg, token_for)
from tests.test_phone_greeting import sessions, slow_first_stream, stop, wait_until  # noqa: F401
from tests.test_turn_stream import REPLY, FakeStream


def wav_samples(data: bytes):
    with wave.open(io.BytesIO(data)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0, w.getframerate()


def seconds(data: bytes) -> float:
    x, sr = wav_samples(data)
    return len(x) / sr


def regions(data: bytes, floor=0.05, min_gap_s=0.2):
    """[(start_s, end_s, dominant_hz)] of the stretches of sound in a wav, separated by at least min_gap_s of quiet."""
    x, sr = wav_samples(data)
    loud = np.abs(x) > floor
    out, start, last = [], None, None
    for i in np.flatnonzero(loud):
        if start is None:
            start = last = i
        elif i - last > min_gap_s * sr:
            out.append((start, last))
            start = i
        last = i
    if start is not None:
        out.append((start, last))
    res = []
    for a, b in out:
        seg = x[a:b + 1]
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        res.append((a / sr, b / sr, float(freqs[np.argmax(np.abs(np.fft.rfft(seg)))])))
    return res


@pytest.fixture()
def asr(rig, env, monkeypatch):
    """Every recording that reaches speech recognition, in order; the text it returns is rig.asr_text, or the next of
    rig.asr_texts when that list is not empty."""
    seen = []
    rig.asr_texts = []

    async def fake(audio_bytes, name, language=None):
        seen.append(audio_bytes)
        return rig.asr_texts.pop(0) if rig.asr_texts else rig.asr_text

    monkeypatch.setattr(env, "_speech_to_text", fake)
    return seen


def user_rows(env, conversation_id):
    rows = env.local_db.select("conversation_turns", lambda r: r.get("conversation_id") == conversation_id)
    return [r for r in rows if r["role"] == "user"]


def tone_piece(ms, freq):
    return pcm_tone(ms, freq=freq)


# =========================================================================================== A: carry the words

def test_a_turn_cancelled_by_new_speech_before_it_is_stored_hands_its_words_to_the_next_turn(rig, env, sessions, asr, monkeypatch):
    monkeypatch.setenv("PHONE_SPECULATIVE", "off")
    slow_first_stream(rig, 1.2)                 # turn 1 is inside the language model when the caller goes on
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        feed(ws, pcm_silence(300) + tone_piece(1800, 220) + pcm_silence(600))          # "the name and the number"
        assert wait_until(lambda: len(rig.asked) == 1)                                   # turn 2 is thinking
        feed(ws, tone_piece(1800, 660) + pcm_silence(700))                               # "...tomorrow at five"
        msgs = read_until(ws, is_mark)
        assert msgs[-1]["mark"]["name"] == "turn-3-end"
        assert wait_until(lambda: any(e["event"] == "said" for e in s.log))
        stop(ws)
    assert len(asr) == 2
    first, second = regions(asr[0]), regions(asr[1])
    assert len(first) == 1 and abs(first[0][2] - 220) < 15
    assert [round(r[2] / 10) * 10 for r in second] == [220, 660]                          # both pieces, in order
    gap = second[1][0] - second[0][1]
    assert 0.3 < gap < 0.9           # the 300 ms of silence between, plus piece 1's tail and piece 2's lead-in
    assert seconds(asr[1]) > seconds(asr[0]) + 1.5
    carried = [e for e in s.log if e["event"] == "carried"]
    assert len(carried) == 1 and carried[0]["pieces"] == 1 and carried[0]["ms"] > 1500
    rows = user_rows(env, s.conversation_id)
    assert len(rows) == 2 and "clinic" in rows[-1]["text_content"]                      # greeting + ONE answer for both


def test_a_turn_that_was_stored_is_not_carried(rig, env, sessions, asr, monkeypatch):
    monkeypatch.setenv("PHONE_SPECULATIVE", "off")
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        feed(ws, pcm_silence(300) + tone_piece(1800, 220) + pcm_silence(600))
        read_until(ws, is_mark)                                                          # the whole reply, stored
        assert wait_until(lambda: any(e["event"] == "said" for e in s.log))
        feed(ws, pcm_silence(300) + tone_piece(1800, 660) + pcm_silence(700))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        stop(ws)
    assert len(asr) == 2
    assert [round(r[2] / 10) * 10 for r in regions(asr[1])] == [660]                      # the second piece alone
    assert not any(e["event"] == "carried" for e in s.log)
    assert len(user_rows(env, s.conversation_id)) == 3


def test_carrying_can_be_switched_off(rig, env, sessions, asr, monkeypatch):
    monkeypatch.setenv("PHONE_SPECULATIVE", "off")
    monkeypatch.setenv("PHONE_CARRY_UNANSWERED", "off")
    slow_first_stream(rig, 1.2)
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        feed(ws, pcm_silence(300) + tone_piece(1800, 220) + pcm_silence(600))
        assert wait_until(lambda: len(rig.asked) == 1)
        feed(ws, tone_piece(1800, 660) + pcm_silence(700))
        read_until(ws, is_mark)
        stop(ws)
    assert [round(r[2] / 10) * 10 for r in regions(asr[-1])] == [660]                     # as before: the new piece only
    assert not any(e["event"] == "carried" for e in sessions[0].log)


def test_talking_over_a_reply_that_was_not_stored_yet_carries_the_words(rig, env, sessions, asr, monkeypatch):
    """Barge-in: the agent is already speaking, but its turn is not in the history yet."""
    monkeypatch.setenv("PHONE_SPECULATIVE", "off")
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": np.full(24000 * 6, 0.05, dtype=np.float32))
    real_store = env._turn_store

    async def slow_store(*a, **k):
        await asyncio.sleep(2.0)                      # the turn is stored late
        return await real_store(*a, **k)

    monkeypatch.setattr(env, "_turn_store", slow_store)
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, lambda m: m["event"] == "media")                                  # the greeting
        s = sessions[0]
        feed(ws, pcm_silence(300) + tone_piece(1800, 220) + pcm_silence(600))
        read_until(ws, lambda m: m["event"] == "media" and s.turns == 2 and s.turn_audio)   # the reply is being spoken
        feed(ws, tone_piece(1800, 660) + pcm_silence(700))                               # the caller talks over it
        read_until(ws, lambda m: m["event"] == "clear")
        assert wait_until(lambda: len(asr) == 2, timeout=8)
        stop(ws)
    assert s.barges >= 1
    assert [round(r[2] / 10) * 10 for r in regions(asr[1])] == [220, 660]
    assert any(e["event"] == "carried" for e in s.log)


def test_a_barge_in_on_a_reply_that_is_stored_carries_nothing(rig, env, sessions, asr, monkeypatch):
    monkeypatch.setenv("PHONE_SPECULATIVE", "off")
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": np.full(24000 * 6, 0.05, dtype=np.float32))
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, lambda m: m["event"] == "media")
        s = sessions[0]
        feed(ws, pcm_silence(300) + tone_piece(1800, 220) + pcm_silence(600))
        read_until(ws, lambda m: m["event"] == "media" and s.turns == 2 and s.turn_audio)
        assert wait_until(lambda: len(user_rows(env, s.conversation_id)) == 1)           # stored while it still speaks
        feed(ws, tone_piece(1800, 660) + pcm_silence(700))
        read_until(ws, lambda m: m["event"] == "clear")
        assert wait_until(lambda: len(asr) == 2, timeout=8)
        stop(ws)
    assert [round(r[2] / 10) * 10 for r in regions(asr[1])] == [660]
    assert not any(e["event"] == "carried" for e in s.log)


def bare_session():
    return px.CallSession(ws=None, claims={"u": "local-user", "m": "local", "a": "x"}, agent={"id": "x", "tools": []})


def piece(seconds_, value=1000, rate=8000):
    return (np.full(int(seconds_ * rate), value, dtype="<i2")).tobytes()


def test_at_most_three_cancelled_pieces_are_carried_and_the_oldest_goes_first():
    s = bare_session()
    infl = px._Inflight(task=None, n=4, pieces=[piece(1, 1), piece(1, 2), piece(1, 3), piece(1, 4)])
    assert s._keep_words(infl, None) is True
    assert [p[:2] for p in s._carry] == [piece(0.001, v)[:2] for v in (2, 3, 4)]


def test_the_clip_for_speech_recognition_stays_under_25_seconds():
    s = bare_session()
    pieces = [piece(10, 1), piece(10, 2), piece(10, 3), piece(8, 4)]          # 38 s and three gaps
    fitted = s._fit_pieces(pieces)
    total = sum(len(p) for p in fitted) / 2 / s.rate + 0.3 * (len(fitted) - 1)
    assert total <= px.CARRY_MAX_S and fitted[-1] == pieces[-1]               # the newest is kept whole
    assert [p[:2] for p in fitted] == [piece(0.001, 3)[:2], piece(0.001, 4)[:2]]   # the oldest two went first
    one_long = [piece(40, 7)]
    assert len(s._joined(s._fit_pieces(one_long))) / 2 / s.rate <= px.CARRY_MAX_S   # one piece: its start is dropped
    assert s._fit_pieces([piece(1, 1)]) == [piece(1, 1)]


def test_a_piece_of_noise_or_a_stored_turn_is_not_carried():
    s = bare_session()
    assert s._keep_words(px._Inflight(None, 2, [piece(1)], noise=True), None) is False
    assert s._keep_words(px._Inflight(None, 2, [piece(1)], done=True), None) is False
    stored_run = type("R", (), {"stored": True})()
    assert s._keep_words(px._Inflight(None, 2, [piece(1)], run=stored_run), None) is False
    assert s._keep_words(None, None) is False
    assert s._carry == []


def test_a_speculative_turn_cancelled_by_resume_does_not_double_the_audio(rig, env, sessions, asr):
    """The detector already holds the whole utterance: carrying the cancelled speculative turn too would say it twice."""
    rig.asr_text = "When does the clinic open?"
    slow_first_stream(rig, 1.0)
    a = make_agent(rig.client)
    audio = tone_piece(1000, 220) + pcm_silence(320) + tone_piece(900, 220) + pcm_silence(900)
    det = pa.UtteranceDetector(8000, vad=pa.EnergyVad(8000),
                               config=pa.DetectorConfig(end_silence_ms=400, early_silence_ms=250))
    events = det.feed(pcm_silence(300) + audio)
    whole = [e for e in events if e.kind == "utterance"][-1].audio
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        feed(ws, pcm_silence(300) + tone_piece(1000, 220) + pcm_silence(320))       # a pause: speculation starts
        assert wait_until(lambda: len(rig.asked) == 1)
        feed(ws, tone_piece(900, 220) + pcm_silence(900))                           # the caller goes on
        read_until(ws, is_mark)
        stop(ws)
    assert s.spec_stats["cancelled"] >= 1 and not any(e["event"] == "carried" for e in s.log)
    expect = len(whole) / 2 / 8000
    assert abs(seconds(asr[-1]) - expect) < 0.1                                      # once, not twice


# ===================================================================================== B: how long to wait

def ask(rig, text):
    rig.stream = lambda: FakeStream([text])


def test_the_wait_after_each_reply_follows_what_the_agent_asked(rig, env, sessions, asr, monkeypatch):
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert s.detector.end_silence_ms == 400 and s.end_wait.kind == "normal"           # the base, as before
        ask(rig, "What are the patient details?")
        feed(ws, pcm_silence(300) + tone_piece(1500, 220) + pcm_silence(900))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-2-end")
        assert s.detector.end_silence_ms == 1500 and s.end_wait.kind == "details"
        ask(rig, "Please tell me your phone number.")
        feed(ws, tone_piece(1500, 220) + pcm_silence(1700))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        assert s.detector.end_silence_ms == 1100 and s.end_wait.kind == "detail"
        ask(rig, "Thank you. Your appointment is booked.")
        feed(ws, tone_piece(1500, 220) + pcm_silence(1300))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-4-end")
        assert s.detector.end_silence_ms == 400 and s.end_wait.kind == "normal"           # back to the normal wait
        stop(ws)
    rows = [e for e in s.log if e["event"] == "latency"]
    assert [r["end_wait"] for r in rows] == ["400(normal)", "1500(details)", "1100(detail)"]


def test_the_wait_is_in_the_latency_log_line(rig, env, sessions, asr, caplog):
    caplog.set_level("INFO", logger="bonclay.phone.exotel")
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        feed(ws, pcm_silence(300) + tone_piece(1200, 220) + pcm_silence(900))
        read_until(ws, is_mark)
        stop(ws)
    assert any("end_wait=400(normal)" in r.getMessage() for r in caplog.records)


def test_a_greeting_that_asks_for_something_is_waited_on_longer(rig, env, sessions, asr):
    a = make_agent(rig.client)
    env.local_db.update("agents", a["id"], {"greeting": "Hello! Please tell me the patient's name."})
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: s.detector is not None and s.detector.end_silence_ms == 1100)
        stop(ws)


def test_a_short_piece_while_it_thinks_is_not_ignored_after_a_request_for_details(rig, env, sessions, asr, monkeypatch):
    """"5 baje" is the next item of an answer, not "hello?": it must not be thrown away while the agent thinks."""
    monkeypatch.setenv("PHONE_SPECULATIVE", "off")
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        ask(rig, "What are the patient details?")
        feed(ws, pcm_silence(300) + tone_piece(1500, 220) + pcm_silence(900))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-2-end")
        slow_first_stream(rig, 1.2)
        rig.stream, first = (lambda: FakeStream(REPLY, first_delay=1.2)), rig.stream
        feed(ws, tone_piece(1500, 220) + pcm_silence(1700))                              # the details: turn 3 thinks
        assert wait_until(lambda: len(rig.asked) == 2)
        feed(ws, tone_piece(700, 660) + pcm_silence(1700))                               # "5 baje": shorter than 1.5 s
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-4-end")
        stop(ws)
    assert not any(e["event"] == "ignored" for e in s.log)
    assert [round(r[2] / 10) * 10 for r in regions(asr[-1])] == [220, 660]


def test_a_short_hello_while_it_thinks_is_still_ignored_when_nothing_was_asked(rig, env, sessions, asr, monkeypatch):
    monkeypatch.setenv("PHONE_SPECULATIVE", "off")
    monkeypatch.setenv("FILLER_AUDIO", "off")          # a filler ("one moment") would make the agent "speaking"
    slow_first_stream(rig, 1.0)
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))   # no disclosure to say again
        feed(ws, pcm_silence(300) + tone_piece(1500, 220) + pcm_silence(600))
        assert wait_until(lambda: len(rig.asked) == 1)
        feed(ws, tone_piece(600, 660) + pcm_silence(700))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-2-end")
        stop(ws)
    assert any(e["event"] == "ignored" for e in s.log) and len(asr) == 1


# ------------------------------------------------------------------------ an answer that looks unfinished

def speculate_and_hear(ws, s, rig, text, *, tone_ms=1200):
    """Start a speculative turn at the caller's pause and wait until it has been through speech recognition, the way a
    real call does (the audio streams in real time; here it is fed faster, so the pause is waited out)."""
    rig.asr_text = text
    feed(ws, pcm_silence(300) + tone_piece(tone_ms, 220) + pcm_silence(300))          # pause: the turn starts
    assert wait_until(lambda: s.spec is not None and s.spec.heard is not None)


def test_a_transcript_ending_in_and_is_waited_for_once_more_then_answered(rig, env, sessions, asr, monkeypatch):
    monkeypatch.setenv("PHONE_INCOMPLETE_EXTRA_MS", "700")
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        speculate_and_hear(ws, s, rig, "My name is Priya Sharma and")
        t0 = time.monotonic()
        feed(ws, pcm_silence(300))                                                      # the end silence is reached
        assert wait_until(lambda: any(e["event"] == "end_extended" for e in s.log))
        assert s.confirmed_at is None and s.spec is not None and s.spec.deferred          # not confirmed
        msgs = read_until(ws, is_mark)
        assert s.confirmed_at is not None and s.confirmed_at - t0 >= 0.6                  # after the extra wait
        stop(ws)
    extended = [e for e in s.log if e["event"] == "end_extended"]
    assert len(extended) == 1 and extended[0]["reason"] == "trailing" and extended[0]["ms"] == 700   # once
    assert any(m["event"] == "media" for m in msgs) and msgs[-1]["mark"]["name"] == "turn-2-end"
    [row] = [e for e in s.log if e["event"] == "latency"]
    assert row["end_wait"] == "400(normal)+700(incomplete:trailing)"


def test_seven_digits_of_a_phone_number_are_waited_for_when_a_number_was_asked_for(rig, env, sessions, asr, monkeypatch):
    monkeypatch.setenv("PHONE_INCOMPLETE_EXTRA_MS", "600")
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        ask(rig, "Please tell me your phone number.")
        feed(ws, pcm_silence(300) + tone_piece(1200, 220) + pcm_silence(900))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-2-end")
        assert s.detector.end_silence_ms == 1100
        speculate_and_hear(ws, s, rig, "nine eight seven six five four three", tone_ms=1500)
        feed(ws, pcm_silence(900))                                                      # 1200 ms of silence in all
        assert wait_until(lambda: any(e["event"] == "end_extended" for e in s.log))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        stop(ws)
    [ext] = [e for e in s.log if e["event"] == "end_extended"]
    assert ext["reason"] == "phone"


def test_a_complete_answer_is_confirmed_at_the_end_silence_with_no_extra_wait(rig, env, sessions, asr):
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        speculate_and_hear(ws, s, rig, "When does the clinic open?")
        feed(ws, pcm_silence(300))
        read_until(ws, is_mark)
        stop(ws)
    assert not any(e["event"] == "end_extended" for e in s.log)
    [row] = [e for e in s.log if e["event"] == "latency"]
    assert row["end_wait"] == "400(normal)"


def test_the_caller_going_on_during_the_extra_wait_gets_one_answer_to_everything(rig, env, sessions, asr, monkeypatch):
    monkeypatch.setenv("PHONE_INCOMPLETE_EXTRA_MS", "2500")
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        speculate_and_hear(ws, s, rig, "My name is Priya Sharma and", tone_ms=1500)
        feed(ws, pcm_silence(300))
        assert wait_until(lambda: any(e["event"] == "end_extended" for e in s.log))
        rig.asr_text = "My name is Priya Sharma and I want to come tomorrow at five"
        feed(ws, tone_piece(1800, 660) + pcm_silence(700))                              # ...and the rest, in time
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-2-end")
        assert wait_until(lambda: any(e["event"] == "said" for e in s.log))
        stop(ws)
    assert len([e for e in s.log if e["event"] == "end_extended"]) == 1                 # never twice
    carried = [e for e in s.log if e["event"] == "carried"]
    assert len(carried) == 1
    assert [round(r[2] / 10) * 10 for r in regions(asr[-1])] == [220, 660]               # the whole of it, once
    rows = user_rows(env, s.conversation_id)
    assert len(rows) == 2 and "tomorrow" in rows[-1]["text_content"]                     # greeting + one answer


def test_the_extra_wait_happens_once_per_turn_and_can_be_switched_off(rig, env, monkeypatch):
    s = bare_session()
    spec = px._Speculation(task=None, gate=None, turn_id="t", n=2, started=0.0, heard="Priya, my number is")
    assert s._unfinished(spec) is None or s._unfinished(spec) == "trailing"
    spec.heard = "Priya Sharma and"
    assert s._unfinished(spec) == "trailing"
    spec.extended = True
    assert s._unfinished(spec) is None                                                   # once
    spec.extended = False
    monkeypatch.setenv("PHONE_INCOMPLETE_EXTRA_MS", "0")
    assert s._unfinished(spec) is None
    monkeypatch.setenv("PHONE_INCOMPLETE_EXTRA_MS", "1200")
    monkeypatch.setenv("PHONE_ADAPTIVE_END", "off")
    assert s._unfinished(spec) is None
    monkeypatch.delenv("PHONE_ADAPTIVE_END")
    spec.heard = None                                                                   # no transcript yet: as before
    assert s._unfinished(spec) is None


# ============================================================== C and D on a call: the details and the prompt

def test_the_small_model_is_asked_after_the_reply_and_never_holds_it_up(rig, env, sessions, asr, monkeypatch):
    import threading
    from services import caller_memory as cm
    from tests.test_caller_memory import FakeModel
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("CALLER_MEMORY_LLM", "on")
    gate = threading.Event()
    fake = FakeModel({"name": "Priya Sharma", "doctor": "Dr Rao"}, gate=gate)
    monkeypatch.setattr(cm, "_llm_call", fake)
    rig.asr_text = "Priya Sharma, I want to see Dr Rao"                      # no cue: only the small model reads the name
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        feed(ws, pcm_silence(300) + tone_piece(1200, 220) + pcm_silence(900))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-2-end")   # the whole reply, model still blocked
        assert wait_until(lambda: len(fake.calls) == 1)                              # asked once the turn was stored
        assert "name" not in cm.load(s.conversation_id)
        gate.set()
        assert wait_until(lambda: "name" in cm.load(s.conversation_id))
        rig.asr_text = "tomorrow at 5 baje"
        feed(ws, pcm_silence(300) + tone_piece(1200, 220) + pcm_silence(900))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        stop(ws)
    last = rig.asked[-1]["messages"][-1]["content"]
    assert "name Priya Sharma" in last and "doctor Dr Rao" in last and "time 17:00" in last


def test_a_call_gives_the_speech_recognition_prompt_the_agents_last_words(rig, env, sessions, monkeypatch):
    from services import query_understanding as qu
    seen = []

    async def fake_asr(audio_bytes, name, language=None):
        seen.append((qu.asr_hints() or {}).get("prompt"))
        return "hello"

    monkeypatch.setattr(env, "_speech_to_text", fake_asr)
    rig.stream = lambda: FakeStream(["Please tell me your phone number."])
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        feed(ws, pcm_silence(300) + tone_piece(1200, 220) + pcm_silence(900))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-2-end")
        rig.stream = lambda: FakeStream(["Thank you."])
        feed(ws, tone_piece(1200, 220) + pcm_silence(1700))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        rig.stream = lambda: FakeStream(["Anything else?"])
        feed(ws, tone_piece(1200, 220) + pcm_silence(900))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-4-end")
        stop(ws)
    assert len(seen) == 3
    assert not (seen[0] and "Phone:" in seen[0])                      # the greeting asked for nothing
    assert seen[1] and seen[1].endswith("Phone: 98765 43210.")        # "please tell me your phone number" came before
    assert not (seen[2] and "Phone:" in seen[2])                      # "Thank you." did not


def test_a_turn_that_failed_before_it_answered_hands_its_words_on_too(rig, env, sessions, monkeypatch):
    monkeypatch.setenv("PHONE_SPECULATIVE", "off")
    heard = []
    answers = iter([None, "My name is Priya and tomorrow at five"])        # speech recognition fails once

    async def flaky_asr(audio_bytes, name, language=None):
        heard.append(audio_bytes)
        return next(answers)

    monkeypatch.setattr(env, "_speech_to_text", flaky_asr)
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        feed(ws, pcm_silence(300) + tone_piece(1500, 220) + pcm_silence(600))
        assert wait_until(lambda: any(e["event"] == "error" for e in s.log))
        feed(ws, tone_piece(1500, 660) + pcm_silence(700))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        stop(ws)
    assert [round(r[2] / 10) * 10 for r in regions(heard[1])] == [220, 660]
    assert any(e.get("after") == "failure" for e in s.log if e["event"] == "carried")


def test_noise_that_held_no_speech_is_not_carried(rig, env, sessions, monkeypatch):
    monkeypatch.setenv("PHONE_SPECULATIVE", "off")
    heard = []
    answers = iter(["", "When does the clinic open?"])                      # the first recording had no speech in it

    async def asr(audio_bytes, name, language=None):
        heard.append(audio_bytes)
        return next(answers)

    monkeypatch.setattr(env, "_speech_to_text", asr)
    a = make_agent(rig.client)
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
        feed(ws, pcm_silence(300) + tone_piece(1500, 220) + pcm_silence(600))
        assert wait_until(lambda: any(e.get("code") == "NO_SPEECH" for e in s.log))
        feed(ws, tone_piece(1500, 660) + pcm_silence(700))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        stop(ws)
    assert [round(r[2] / 10) * 10 for r in regions(heard[1])] == [660]
    assert not any(e["event"] == "carried" for e in s.log)


# ================================================== an unfinished answer, for an agent that does not reply early

DETAILS_ASK = "Please tell me the patient details."


def tool_agent_call(rig, env, monkeypatch, sessions, asr, texts, *, tools=True, speculative=None, extra_ms="2500"):
    """A call to an agent with tools (so no speculative replies) whose first reply asks for the patient details. Returns
    (agent, ws context manager args)."""
    async def no_tools(run):
        return 0.0                                                # no tool server is started: the plain model answers

    monkeypatch.setattr(env, "_turn_tools", no_tools)
    monkeypatch.setenv("PHONE_INCOMPLETE_EXTRA_MS", extra_ms)
    if speculative:
        monkeypatch.setenv("PHONE_SPECULATIVE", speculative)
    ask(rig, DETAILS_ASK)
    rig.asr_texts[:] = ["Hello"] + list(texts)
    return make_agent(rig.client, **({"tools": ["appointments"]} if tools else {}))


def open_and_ask_details(ws, rig, sessions):
    ws.send_text(start_msg())
    read_until(ws, is_mark)
    s = sessions[0]
    assert wait_until(lambda: any(e["event"] == "greeting_recorded" for e in s.log))
    feed(ws, pcm_silence(300) + tone_piece(1200, 330) + pcm_silence(700))
    read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-2-end")
    assert s.end_wait.kind == "details" and s.detector.end_silence_ms == 1500
    rig.stream = lambda: FakeStream(["Thank you, I have noted that."])
    return s


def test_a_booking_agent_waits_for_the_rest_of_the_details_and_answers_once(rig, env, sessions, asr, monkeypatch):
    full = "Priya Sharma 9876543210 tomorrow at five"
    a = tool_agent_call(rig, env, monkeypatch, sessions, asr, ["Priya Sharma 9876543", full])
    assert a["tools"] == ["appointments"]
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        s = open_and_ask_details(ws, rig, sessions)
        asked_before = len(rig.asked)
        feed(ws, tone_piece(1500, 220) + pcm_silence(1600))                       # a name and 7 digits, then a 1.6 s pause
        assert wait_until(lambda: any(e["event"] == "end_extended" for e in s.log))
        time.sleep(0.3)
        assert len(rig.asked) == asked_before and s.turns == 2                  # no reply, no turn started on half of it
        feed(ws, tone_piece(1800, 660) + pcm_silence(1700))                       # the rest
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        assert wait_until(lambda: any(e["event"] == "said" for e in s.log[-3:]))
        stop(ws)
    assert [e["reason"] for e in s.log if e["event"] == "end_extended"] == ["phone"]       # once
    assert len(asr) == 3                                                       # the details question, part 1, then all of it
    assert [round(r[2] / 10) * 10 for r in regions(asr[-1])] == [220, 660]
    assert len(rig.asked) == asked_before + 1                                  # ONE reply, to the whole answer
    assert full in rig.asked[-1]["messages"][-1]["content"]
    rows = user_rows(env, s.conversation_id)
    assert [r["text_content"] for r in rows][-1] == full and len(rows) == 3
    assert any(e["event"] == "carried" for e in s.log)


def test_a_complete_answer_to_a_request_for_details_gets_no_extra_wait_and_one_recognition(rig, env, sessions, asr, monkeypatch):
    full = "Priya Sharma 9876543210 tomorrow at five"
    a = tool_agent_call(rig, env, monkeypatch, sessions, asr, [full])
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        s = open_and_ask_details(ws, rig, sessions)
        feed(ws, tone_piece(1800, 220) + pcm_silence(1700))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        stop(ws)
    assert not any(e["event"] == "end_extended" for e in s.log)
    assert len(asr) == 2                                                       # the question, and this answer ONCE
    assert full in rig.asked[-1]["messages"][-1]["content"]
    [_, row] = [e for e in s.log if e["event"] == "latency"]
    assert row["end_wait"] == "1500(details)"


def test_without_speculation_at_all_the_answer_is_looked_at_at_the_end_silence(rig, env, sessions, asr, monkeypatch):
    full = "My name is Priya Sharma and my number is 9876543210"
    a = tool_agent_call(rig, env, monkeypatch, sessions, asr, ["My name is Priya Sharma and", full],
                        tools=False, speculative="off")
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        s = open_and_ask_details(ws, rig, sessions)                             # (no pause event: the detector has none)
        feed(ws, tone_piece(1500, 220) + pcm_silence(1600))
        assert wait_until(lambda: any(e["event"] == "end_extended" for e in s.log))
        feed(ws, tone_piece(1800, 660) + pcm_silence(1700))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        stop(ws)
    assert [e["reason"] for e in s.log if e["event"] == "end_extended"] == ["trailing"]
    assert [round(r[2] / 10) * 10 for r in regions(asr[-1])] == [220, 660]
    assert full in rig.asked[-1]["messages"][-1]["content"]


def test_the_extra_wait_ends_and_the_turn_starts_when_the_caller_stays_quiet(rig, env, sessions, asr, monkeypatch):
    a = tool_agent_call(rig, env, monkeypatch, sessions, asr, ["Priya Sharma 9876543"], extra_ms="500")
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        s = open_and_ask_details(ws, rig, sessions)
        feed(ws, tone_piece(1500, 220) + pcm_silence(1600))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")   # after the 500 ms, it answers
        stop(ws)
    assert len(asr) == 2                                                       # recognised once, the result is the turn's
    [_, row] = [e for e in s.log if e["event"] == "latency"]
    assert row["end_wait"] == "1500(details)+500(incomplete:phone)"
    assert "Priya Sharma 9876543" in rig.asked[-1]["messages"][-1]["content"]


def test_a_turn_that_asked_for_nothing_is_not_looked_at_and_keeps_its_latency_line(rig, env, sessions, asr, monkeypatch):
    async def no_tools(run):
        return 0.0

    monkeypatch.setattr(env, "_turn_tools", no_tools)
    a = make_agent(rig.client, tools=["appointments"])
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        ws.send_text(start_msg())
        read_until(ws, is_mark)
        s = sessions[0]
        feed(ws, pcm_silence(300) + tone_piece(1200, 220) + pcm_silence(900))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-2-end")
        stop(ws)
    assert not any(e["event"] in ("precheck_start", "end_extended") for e in s.log)
    assert len(asr) == 1
    [row] = [e for e in s.log if e["event"] == "latency"]
    assert row["end_wait"] == "400(normal)" and row["speculative"] is False


def test_the_look_at_the_answer_can_be_switched_off(rig, env, sessions, asr, monkeypatch):
    a = tool_agent_call(rig, env, monkeypatch, sessions, asr, ["Priya Sharma 9876543", "all of it"], extra_ms="0")
    with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}") as ws:
        s = open_and_ask_details(ws, rig, sessions)
        feed(ws, tone_piece(1500, 220) + pcm_silence(1600))
        read_until(ws, lambda m: is_mark(m) and m["mark"]["name"] == "turn-3-end")
        stop(ws)
    assert not any(e["event"] in ("precheck_start", "end_extended") for e in s.log)
