"""The WebSocket dialect of each phone provider (services/telephony): the messages the carrier sends are read into events,
and what we send back is built in the carrier's shape. The incoming messages follow the shapes in each provider's docs
(see the module docstrings for the URLs); audio is mu-law on the wire and PCM16 inside."""
import base64
import json

import numpy as np
import pytest

from services import phone_audio as pa
from services.telephony.exotel import ExotelCodec
from services.telephony.plivo import PlivoCodec, VobizCodec
from services.telephony.telnyx import TelnyxCodec
from services.telephony.twilio import TwilioCodec


def tone(ms=100, amp=0.3, freq=440.0, rate=8000):
    t = np.arange(int(rate * ms / 1000)) / rate
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype("<i2").tobytes()


def b64(raw):
    return base64.b64encode(raw).decode()


def close_to(a: bytes, b: bytes, tol=0.08):
    """mu-law is lossy: the signals agree to within a few percent of full scale."""
    x = np.frombuffer(a, "<i2").astype(float) / 32768
    y = np.frombuffer(b, "<i2").astype(float) / 32768
    return len(x) == len(y) and float(np.max(np.abs(x - y))) < tol


# ------------------------------------------------------------------------------------------------------- Exotel

def test_exotel_reads_start_media_dtmf_mark_and_stop_and_builds_raw_pcm_frames():
    c = ExotelCodec()
    ev = c.parse(json.dumps({"event": "start", "stream_sid": "S1", "start": {"stream_sid": "S1", "call_sid": "CA1",
                                                                              "from": "+919876543210", "media_format": {"sample_rate": "16000"}}}))
    assert (ev.kind, ev.stream_id, ev.call_id, ev.caller, ev.rate) == ("start", "S1", "CA1", "+919876543210", 16000)
    pcm = tone()
    assert c.parse(json.dumps({"event": "media", "media": {"payload": b64(pcm)}})).pcm == pcm
    assert c.parse(json.dumps({"event": "dtmf", "dtmf": {"digit": "5"}})).digit == "5"
    assert c.parse(json.dumps({"event": "dtmf", "digit": "0"})).digit == "0"
    assert c.parse(json.dumps({"event": "mark", "mark": {"name": "m"}})).name == "m"
    assert c.parse(json.dumps({"event": "stop"})).kind == "stop"
    assert c.parse("not json") is None and c.parse(json.dumps({"event": "connected"})) is None
    assert c.media(pcm, "S1") == {"event": "media", "stream_sid": "S1", "media": {"payload": b64(pcm)}}
    assert c.clear("S1") == {"event": "clear", "stream_sid": "S1"}
    assert c.mark("turn-1-end", "S1") == {"event": "mark", "stream_sid": "S1", "mark": {"name": "turn-1-end"}}


# ------------------------------------------------------------------------------------------------------- Twilio
# https://www.twilio.com/docs/voice/media-streams/websocket-messages

TWILIO_START = {"event": "start", "sequenceNumber": "1", "streamSid": "MZ1",
                "start": {"accountSid": "AC1", "streamSid": "MZ1", "callSid": "CA1", "tracks": ["inbound"],
                          "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                          "customParameters": {"agent": "a1"}}}


def test_twilio_start_carries_the_stream_and_call_ids_and_custom_parameters():
    ev = TwilioCodec().parse(json.dumps(TWILIO_START))
    assert (ev.kind, ev.stream_id, ev.call_id, ev.rate) == ("start", "MZ1", "CA1", 8000)
    assert ev.params == {"agent": "a1"}


def test_twilio_media_is_mulaw_in_and_out_and_round_trips():
    c = TwilioCodec()
    pcm = tone(100)
    wire = pa.ulaw_encode(pcm)
    assert len(wire) == len(pcm) // 2                                   # one byte per sample
    ev = c.parse(json.dumps({"event": "media", "sequenceNumber": "2", "streamSid": "MZ1",
                             "media": {"track": "inbound", "chunk": "1", "timestamp": "5", "payload": b64(wire)}}))
    assert ev.kind == "media" and close_to(ev.pcm, pcm)
    out = c.media(pcm, "MZ1")
    assert out["event"] == "media" and out["streamSid"] == "MZ1" and set(out["media"]) == {"payload"}
    assert close_to(pa.ulaw_decode(base64.b64decode(out["media"]["payload"])), pcm)


def test_twilio_ignores_connected_and_the_outbound_track_and_reads_dtmf_mark_stop():
    c = TwilioCodec()
    assert c.parse(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"})) is None
    assert c.parse(json.dumps({"event": "media", "media": {"track": "outbound", "payload": b64(b"\xff" * 80)}})) is None
    assert c.parse(json.dumps({"event": "dtmf", "streamSid": "MZ1", "dtmf": {"track": "inbound", "digit": "1"}})).digit == "1"
    assert c.parse(json.dumps({"event": "mark", "streamSid": "MZ1", "mark": {"name": "turn-1-end"}})).name == "turn-1-end"
    assert c.parse(json.dumps({"event": "stop", "streamSid": "MZ1", "stop": {"callSid": "CA1"}})).kind == "stop"


def test_twilio_clear_and_mark_builders():
    c = TwilioCodec()
    assert c.clear("MZ1") == {"event": "clear", "streamSid": "MZ1"}
    assert c.mark("m1", "MZ1") == {"event": "mark", "streamSid": "MZ1", "mark": {"name": "m1"}}


# -------------------------------------------------------------------------------------------------- Plivo / Vobiz
# https://docs.plivo.com/docs/voice-agents/audio-streaming/concepts/audio-streaming-reference.md

PLIVO_START = {"event": "start", "sequenceNumber": 1,
               "start": {"callId": "call-uuid-1", "streamId": "stream-1", "accountId": "MA1", "tracks": ["inbound"],
                         "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000}}}


def test_plivo_start_media_dtmf_checkpoint_and_the_events_it_ignores():
    c = PlivoCodec()
    ev = c.parse(json.dumps(PLIVO_START))
    assert (ev.kind, ev.stream_id, ev.call_id, ev.rate) == ("start", "stream-1", "call-uuid-1", 8000)
    pcm = tone()
    m = c.parse(json.dumps({"event": "media", "sequenceNumber": 42, "streamId": "stream-1",
                            "media": {"track": "inbound", "timestamp": "1", "chunk": 41, "payload": b64(pa.ulaw_encode(pcm))}}))
    assert m.kind == "media" and close_to(m.pcm, pcm)
    assert c.parse(json.dumps({"event": "dtmf", "streamId": "stream-1", "dtmf": {"track": "inbound", "digit": "5"}})).digit == "5"
    played = c.parse(json.dumps({"event": "playedStream", "streamId": "stream-1", "name": "turn-1-end"}))
    assert (played.kind, played.name) == ("mark", "turn-1-end")
    assert c.parse(json.dumps({"event": "clearedAudio", "streamId": "stream-1"})) is None


def test_plivo_sends_play_audio_clear_audio_and_checkpoint():
    c = PlivoCodec()
    c.parse(json.dumps(PLIVO_START))
    pcm = tone()
    out = c.media(pcm, "stream-1")
    assert out["event"] == "playAudio" and "streamId" not in out
    assert out["media"]["contentType"] == "audio/x-mulaw" and out["media"]["sampleRate"] == 8000
    assert close_to(pa.ulaw_decode(base64.b64decode(out["media"]["payload"])), pcm)
    assert c.clear("stream-1") == {"event": "clearAudio", "streamId": "stream-1"}
    assert c.mark("turn-1-end", "stream-1") == {"event": "checkpoint", "streamId": "stream-1", "name": "turn-1-end"}


def test_plivo_follows_the_format_named_in_start_linear_16_bit():
    c = PlivoCodec()
    c.parse(json.dumps({"event": "start", "start": {"callId": "c", "streamId": "s",
                                                    "mediaFormat": {"encoding": "audio/x-l16", "sampleRate": 16000}}}))
    pcm = tone(rate=16000)
    assert c.parse(json.dumps({"event": "media", "media": {"payload": b64(pcm)}})).pcm == pcm
    out = c.media(pcm, "s")
    assert out["media"]["contentType"] == "audio/x-l16" and out["media"]["sampleRate"] == 16000
    assert base64.b64decode(out["media"]["payload"]) == pcm


def test_vobiz_play_audio_names_the_stream_as_its_docs_show():
    # https://www.vobiz.ai/docs/integrations/websockets.md: playAudio carries streamId beside media
    c = VobizCodec()
    ev = c.parse(json.dumps({"event": "start", "start": {"callId": "CALL", "streamId": "STREAM", "tracks": ["inbound"],
                                                          "mediaFormat": {"encoding": "audio/x-l16", "sampleRate": 8000}}}))
    assert (ev.stream_id, ev.call_id, ev.rate) == ("STREAM", "CALL", 8000)
    out = c.media(tone(), "STREAM")
    assert out["event"] == "playAudio" and out["streamId"] == "STREAM"
    assert out["media"]["contentType"] == "audio/x-l16" and out["media"]["sampleRate"] == 8000
    assert c.clear("STREAM") == {"event": "clearAudio", "streamId": "STREAM"}
    assert c.mark("response-1", "STREAM") == {"event": "checkpoint", "streamId": "STREAM", "name": "response-1"}


# ------------------------------------------------------------------------------------------------------- Telnyx
# https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming

TELNYX_START = {"event": "start", "sequence_number": "1", "stream_id": "st-1",
                "start": {"user_id": "u", "call_control_id": "cc-1", "call_session_id": "cs-1", "from": "+15551234567",
                          "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1}}}


def test_telnyx_start_media_dtmf_mark_stop_and_the_frames_it_ignores():
    c = TelnyxCodec()
    ev = c.parse(json.dumps(TELNYX_START))
    assert (ev.kind, ev.stream_id, ev.call_id, ev.rate, ev.caller) == ("start", "st-1", "cc-1", 8000, "+15551234567")
    pcm = tone()
    m = c.parse(json.dumps({"event": "media", "sequence_number": "2", "stream_id": "st-1",
                            "media": {"track": "inbound", "chunk": "1", "timestamp": "5", "payload": b64(pa.ulaw_encode(pcm))}}))
    assert m.kind == "media" and close_to(m.pcm, pcm)
    assert c.parse(json.dumps({"event": "dtmf", "stream_id": "st-1", "dtmf": {"digit": "1"}, "occurred_at": "x"})).digit == "1"
    assert c.parse(json.dumps({"event": "mark", "mark": {"name": "m1"}})).name == "m1"
    assert c.parse(json.dumps({"event": "stop", "stop": {"call_control_id": "cc-1", "user_id": "u"}})).kind == "stop"
    assert c.parse(json.dumps({"event": "connected", "version": "1.0.0"})) is None
    assert c.parse(json.dumps({"event": "error", "payload": {"code": 1}})) is None


def test_telnyx_builders_carry_no_stream_id_and_send_pcmu():
    c = TelnyxCodec()
    pcm = tone()
    out = c.media(pcm, "st-1")
    assert set(out) == {"event", "media"} and out["event"] == "media"
    assert close_to(pa.ulaw_decode(base64.b64decode(out["media"]["payload"])), pcm)
    assert c.clear("st-1") == {"event": "clear"}
    assert c.mark("m1", "st-1") == {"event": "mark", "mark": {"name": "m1"}}


@pytest.mark.parametrize("codec", [TwilioCodec, PlivoCodec, VobizCodec, TelnyxCodec, ExotelCodec])
def test_every_codec_shrugs_off_garbage(codec):
    c = codec()
    for junk in ("", "nope", "[]", "42", json.dumps({"no": "event"}), json.dumps({"event": "media", "media": {"payload": "!!"}})):
        c.parse(junk)                      # never raises
    assert c.parse("nope") is None
