"""WhatsApp voice-note channel (routers/whatsapp.py, services/whatsapp.py), tested against a fake Graph API: a real local
HTTP server that speaks the parts of Meta's WhatsApp Cloud API the channel uses (media lookup and download, media upload,
send message, phone number lookup) and records every request. Speech recognition, the language model and TTS are faked
like in test_offline_agents.py; the audio conversion is real (ffmpeg)."""
import email.parser
import email.policy
import json
import logging
import os
import shutil
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from routers import whatsapp as wa_router
from services import whatsapp as wa
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed for WhatsApp voice notes")

PNID = "109876543210987"
TOKEN = "EAAGtestaccesstoken1234567890XYZ"
APP_SECRET = "app-secret-0123456789abcdef"
CALLER = "919876543210"
REPLY = "Refunds are possible within thirty days."
DISCLOSED = "This is an AI assistant. " + REPLY


def make_ogg(seconds=1.5, freq=440) -> bytes:
    """A real OGG/Opus voice note, like WhatsApp sends."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}", "-c:a", "libopus",
         "-ac", "1", "-ar", "48000", "-b:a", "24k", "-f", "ogg", "pipe:1"], capture_output=True, check=True).stdout


def probe(data: bytes) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,channels,sample_rate,bit_rate",
                        "-show_entries", "format=format_name,duration", "-of", "json", "pipe:0"],
                       input=data, capture_output=True, check=True)
    j = json.loads(p.stdout)
    return {**j["streams"][0], **j["format"]}


# ------------------------------------------------------------------------------------------------ fake Graph API

class FakeGraph:
    """Records every request; serves media by id; hands out ids for uploads."""

    def __init__(self):
        self.requests = []          # {"method", "path", "headers", "body", "json"?, "parts"?}
        self.media = {}             # media id -> (bytes, mime)
        self.media_info = {}        # media id -> overrides for the lookup answer (url, file_size, ...)
        self.uploads = 0
        self.fail = {}              # path suffix -> HTTP status to answer with
        graph = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _read(self):
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            def _send(self, status, payload, ctype="application/json"):
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method):
                body = self._read()
                rec = {"method": method, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                       "body": body}
                ctype = rec["headers"].get("content-type", "")
                if ctype.startswith("application/json") and body:
                    rec["json"] = json.loads(body)
                if ctype.startswith("multipart/form-data"):
                    msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(
                        b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
                    rec["parts"] = {p.get_param("name", header="content-disposition"): {
                        "filename": p.get_filename(), "type": p.get_content_type(),
                        "ctype_header": p["content-type"], "data": p.get_payload(decode=True)} for p in msg.iter_parts()}
                graph.requests.append(rec)
                path = self.path.split("?")[0]
                if path.startswith("/v99.0"):
                    path = path[len("/v99.0"):]
                for suffix, status in graph.fail.items():
                    if path.endswith(suffix):
                        return self._send(status, {"error": {"message": "Simulated failure", "code": 131000}})
                if rec["headers"].get("authorization") != f"Bearer {TOKEN}":
                    return self._send(401, {"error": {"message": "Invalid OAuth access token.", "code": 190}})
                if method == "GET" and path.startswith("/media/"):
                    mid = path[len("/media/"):]
                    data, mime = graph.media[mid]
                    return self._send(200, data, mime)
                if method == "GET" and path.strip("/") in graph.media:
                    mid = path.strip("/")
                    data, mime = graph.media[mid]
                    info = {"url": f"http://127.0.0.1:{graph.port}/media/{mid}", "mime_type": mime,
                            "file_size": len(data), "id": mid, "messaging_product": "whatsapp"}
                    info.update(graph.media_info.get(mid, {}))
                    return self._send(200, info)
                if method == "GET" and path.strip("/") == PNID:
                    return self._send(200, {"display_phone_number": "+1 555 010 9999", "verified_name": "Test Shop",
                                            "id": PNID})
                if method == "POST" and path == f"/{PNID}/media":
                    graph.uploads += 1
                    return self._send(200, {"id": f"OUT{graph.uploads}"})
                if method == "POST" and path == f"/{PNID}/messages":
                    return self._send(200, {"messaging_product": "whatsapp", "messages": [{"id": "wamid.out"}]})
                return self._send(404, {"error": {"message": "Unknown path", "code": 100}})

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/v99.0"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def sends(self, kind=None):
        out = [r for r in self.requests if r["method"] == "POST" and r["path"].endswith("/messages")
               and r.get("json", {}).get("status") != "read"]
        return [r["json"] for r in out if kind is None or r["json"]["type"] == kind]

    def uploads_seen(self):
        return [r for r in self.requests if r["method"] == "POST" and r["path"].endswith("/media")]

    def reads(self):
        return [r["json"] for r in self.requests if r.get("json", {}).get("status") == "read"]


@pytest.fixture()
def graph(monkeypatch):
    g = FakeGraph()  # like the real API, its paths start with a version segment (/v99.0/{id})
    monkeypatch.setenv("WHATSAPP_GRAPH_URL", g.url)
    yield g
    g.close()


class Rig:
    def __init__(self, client, env, monkeypatch, tmp_path, graph):
        self.client, self.env, self.graph = client, env, graph
        self.asr_text = "What is the refund window?"
        self.asr_inputs = []
        self.chat_calls = env.chat_calls

        async def fake_asr(audio_bytes, name, language=None):
            self.asr_inputs.append(audio_bytes)
            return self.asr_text

        async def fake_load(key):
            return object()

        monkeypatch.setattr(env, "_speech_to_text", fake_asr)
        monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": np.full(12000, 0.05, np.float32))
        monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
        env._DISCLOSURE_CLIPS.clear()
        monkeypatch.setenv("BONCLAY_WHATSAPP_DIR", str(tmp_path / "wa"))
        monkeypatch.delenv("WHATSAPP_APP_SECRET", raising=False)
        monkeypatch.delenv("WHATSAPP_VERIFY_TOKEN", raising=False)
        monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN", raising=False)
        wa_router.seen.clear()
        wa_router.sender_limiter.clear()
        wa_router._CHATS.clear()

    def agent(self, **settings):
        a = make_agent(self.client)
        self.configure(a["id"], **settings)
        return a

    def configure(self, agent_id, enable=True, **extra):
        body = {"phone_number_id": PNID, "access_token": TOKEN, "app_secret": APP_SECRET, **extra}
        r = self.client.put(f"/api/agents/{agent_id}/whatsapp", headers=LOCAL, json=body)
        assert r.status_code == 200, r.text
        if enable:
            r = self.client.put(f"/api/agents/{agent_id}/whatsapp", headers=LOCAL, json={"enabled": True})
            assert r.status_code == 200, r.text
        return r.json()

    @staticmethod
    def payload(messages, pnid=PNID, statuses=None):
        value = {"messaging_product": "whatsapp", "metadata": {"display_phone_number": "15550109999",
                                                                "phone_number_id": pnid},
                 "contacts": [{"profile": {"name": "Priya"}, "wa_id": CALLER}]}
        if messages:
            value["messages"] = messages
        if statuses:
            value["statuses"] = statuses
        return {"object": "whatsapp_business_account",
                "entry": [{"id": "WABA1", "changes": [{"field": "messages", "value": value}]}]}

    @staticmethod
    def text_msg(text, mid="wamid.T1", ts=None, sender=CALLER):
        return {"from": sender, "id": mid, "timestamp": str(int(ts or time.time())), "type": "text",
                "text": {"body": text}}

    @staticmethod
    def audio_msg(media_id="MEDIA1", mid="wamid.A1", ts=None):
        return {"from": CALLER, "id": mid, "timestamp": str(int(ts or time.time())), "type": "audio",
                "audio": {"mime_type": "audio/ogg; codecs=opus", "sha256": "x", "id": media_id, "voice": True}}

    def post(self, payload, secret=APP_SECRET, signature=None, raw=None):
        body = raw if raw is not None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if signature is None and secret:
            signature = wa.sign(secret, body)
        if signature:
            headers["X-Hub-Signature-256"] = signature
        return self.client.post("/api/whatsapp/webhook", content=body, headers=headers)

    def send(self, *messages, **kw):
        r = self.post(self.payload(list(messages)), **kw)
        assert r.status_code == 200, r.text
        return r


@pytest.fixture()
def rig(app_client, env, monkeypatch, tmp_path, graph):
    return Rig(app_client, env, monkeypatch, tmp_path, graph)


# ------------------------------------------------------------------------------------------------ verification

def test_verification_handshake(rig):
    a = rig.agent()
    token = rig.client.get(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL).json()["verify_token"]
    assert token.startswith("avt_") and len(token) > 20

    ok = rig.client.get("/api/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": token,
                                                         "hub.challenge": "1158201444"})
    assert ok.status_code == 200 and ok.text == "1158201444" and ok.headers["content-type"].startswith("text/plain")

    bad = rig.client.get("/api/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "nope",
                                                          "hub.challenge": "1"})
    assert bad.status_code == 403
    wrong_mode = rig.client.get("/api/whatsapp/webhook", params={"hub.mode": "unsubscribe", "hub.verify_token": token,
                                                                 "hub.challenge": "1"})
    assert wrong_mode.status_code == 403
    assert rig.client.get("/api/whatsapp/webhook").status_code == 403
    assert rig.client.get("/api/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "",
                                                           "hub.challenge": "1"}).status_code == 403
    # a challenge that is not a plain token is not echoed back
    weird = rig.client.get("/api/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": token,
                                                            "hub.challenge": "<script>alert(1)</script>"})
    assert weird.status_code == 400


def test_server_wide_verify_token(rig, monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "global-verify-token")
    r = rig.client.get("/api/whatsapp/webhook", params={"hub.mode": "subscribe",
                                                        "hub.verify_token": "global-verify-token", "hub.challenge": "42"})
    assert r.status_code == 200 and r.text == "42"


# ------------------------------------------------------------------------------------------------ signature

def test_missing_or_bad_signature_is_rejected_and_nothing_runs(rig):
    rig.agent()
    p = rig.payload([rig.text_msg("hello")])
    body = json.dumps(p).encode()
    missing = rig.client.post("/api/whatsapp/webhook", content=body, headers={"Content-Type": "application/json"})
    assert missing.status_code == 403 and missing.json()["code"] == "WHATSAPP_SIGNATURE_INVALID"
    bad = rig.post(p, signature="sha256=" + "0" * 64)
    wrong_secret = rig.post(p, secret="some-other-secret")
    junk = rig.post(p, signature="not-a-signature")
    wrong_body = rig.post(p, signature=wa.sign(APP_SECRET, body + b" "))
    assert {bad.status_code, wrong_secret.status_code, junk.status_code, wrong_body.status_code} == {403}
    assert rig.graph.requests == []  # nothing was answered
    assert all(c.get("json_mode") for c in rig.env.chat_calls)  # (only the agent's creation asked the model)
    assert rig.env.local_db.select("conversations") == []


def test_unsigned_call_is_rejected_even_without_any_secret_configured(rig):
    """No app secret anywhere: the check fails closed, it is never skipped."""
    a = make_agent(rig.client)
    rig.env.local_db.update("agents", a["id"], {"whatsapp": {"enabled": True, "phone_number_id": PNID}})
    r = rig.client.post("/api/whatsapp/webhook", json=rig.payload([rig.text_msg("hi")]))
    assert r.status_code == 403
    r = rig.post(rig.payload([rig.text_msg("hi")]), secret="", signature="sha256=" + "0" * 64)
    assert r.status_code == 403 and rig.graph.requests == []


def test_unknown_phone_number_and_invalid_json_and_huge_body(rig):
    rig.agent()
    other = rig.payload([rig.text_msg("x")], pnid="555000111222")
    assert rig.post(other).status_code == 403  # nobody owns this number: nothing proves who signed it
    assert rig.client.post("/api/whatsapp/webhook", content=b"not json",
                           headers={"X-Hub-Signature-256": wa.sign(APP_SECRET, b"not json")}).status_code == 403
    big = b"x" * (wa.MAX_BODY_BYTES + 1)
    assert rig.client.post("/api/whatsapp/webhook", content=big).status_code == 413


def test_server_wide_app_secret(rig, monkeypatch):
    a = make_agent(rig.client)
    rig.env.local_db.update("agents", a["id"], {"whatsapp": {"enabled": True, "phone_number_id": PNID,
                                                              "access_token_set": True}})
    wa.save_secrets(a["id"], {"access_token": TOKEN}, local=True)
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "global-app-secret")
    p = rig.payload([rig.text_msg("hello")])
    assert rig.post(p, secret=APP_SECRET).status_code == 403
    assert rig.post(p, secret="global-app-secret").status_code == 200
    assert len(rig.graph.sends("text")) == 1


def test_statuses_are_ignored(rig):
    rig.agent()
    p = rig.payload([], statuses=[{"id": "wamid.out", "status": "delivered", "timestamp": "1", "recipient_id": CALLER}])
    assert rig.post(p).status_code == 200
    assert rig.graph.requests == []
    assert rig.env.local_db.select("conversations") == []


def test_disabled_agent_does_not_answer(rig):
    a = rig.agent(enable=False)
    assert rig.post(rig.payload([rig.text_msg("hi")])).status_code == 200
    assert rig.graph.requests == []
    rig.client.put(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL, json={"enabled": True})
    rig.send(rig.text_msg("hi", mid="wamid.later"))
    assert rig.graph.sends("text")


# ------------------------------------------------------------------------------------------------ dedupe, window, limits

def test_a_repeated_message_id_is_answered_once(rig):
    rig.agent()
    msg = rig.text_msg("What is the refund window?", mid="wamid.DUP")
    rig.send(msg)
    rig.send(msg)
    rig.send(msg)
    assert len(rig.graph.sends("text")) == 1
    assert len(rig.env.local_db.select("conversation_turns", lambda r: r["role"] == "agent")) == 1


def test_messages_outside_the_24_hour_window_are_not_answered(rig):
    rig.agent()
    rig.send(rig.text_msg("old one", mid="wamid.OLD", ts=time.time() - 30 * 3600))
    assert rig.graph.sends() == []
    rig.send(rig.text_msg("new one", mid="wamid.NEW"))
    assert len(rig.graph.sends("text")) == 1


def test_a_sender_over_the_hourly_limit_is_ignored(rig, monkeypatch):
    rig.agent()
    monkeypatch.setattr(wa_router, "sender_limiter", wa.SenderLimiter(2))
    for i in range(4):
        rig.send(rig.text_msg("hello", mid=f"wamid.L{i}"))
    assert len(rig.graph.sends("text")) == 2


# ------------------------------------------------------------------------------------------------ text turn

def test_text_message_gets_a_voice_note_and_the_text(rig):
    a = rig.agent()
    rig.send(rig.text_msg("What is the refund window?"))
    g = rig.graph

    # voice note first: upload, then the audio message that points at the uploaded media; then the text
    assert rig.env.chat_calls and g.uploads == 1
    audio, text = g.sends("audio"), g.sends("text")
    assert len(audio) == 1 and len(text) == 1
    assert audio[0] == {"messaging_product": "whatsapp", "recipient_type": "individual", "to": CALLER, "type": "audio",
                        "audio": {"id": "OUT1"}}
    assert text[0] == {"messaging_product": "whatsapp", "recipient_type": "individual", "to": CALLER, "type": "text",
                       "text": {"preview_url": False, "body": DISCLOSED}}
    order = [("upload" if r["path"].endswith("/media") else r["json"]["type"]) for r in g.requests
             if r["method"] == "POST" and r.get("json", {}).get("status") != "read"]
    assert order == ["upload", "audio", "text"]
    assert all(r["headers"]["authorization"] == f"Bearer {TOKEN}" for r in g.requests)
    assert g.reads() == [{"messaging_product": "whatsapp", "status": "read", "message_id": "wamid.T1"}]

    # the turn is a normal BonClay turn: stored with the whatsapp source and the caller, AI disclosure on the first reply
    conv = rig.env.local_db.select("conversations")[0]
    assert conv["source"] == "whatsapp" and conv["external_id"] == CALLER and conv["agent_id"] == a["id"]
    assert conv["needs_human"] is False
    turns = rig.env.local_db.select("conversation_turns")
    assert [t["role"] for t in turns] == ["user", "agent"]
    assert turns[0]["text_content"] == "What is the refund window?" and turns[1]["text_content"] == DISCLOSED
    assert turns[1]["audio_storage_path"].startswith("local:turn_") and turns[1]["language"] == "en"

    # same caller again: same conversation, history is used, the disclosure is not repeated
    rig.send(rig.text_msg("And for sale items?", mid="wamid.T2"))
    assert len(rig.env.local_db.select("conversations")) == 1
    assert len(rig.env.chat_calls[-1]["messages"]) == 4
    assert g.sends("text")[-1]["text"]["body"] == REPLY


def test_the_reply_is_the_stored_signed_turn_audio(rig):
    """The voice note is made from the same watermarked audio that is stored and that the receipt is signed for."""
    from services import provenance
    a = make_agent(rig.client)
    out = rig.client.post(f"/api/agents/{a['id']}/whatsapp/simulate", headers=LOCAL,
                          json={"text": "What is the refund window?"}).json()
    receipt = provenance.get_receipt(out["receipt_id"])
    turn = rig.env.local_db.select("conversation_turns", lambda r: r["role"] == "agent")[0]
    assert receipt["kind"] == "agent_reply" and receipt["turn_id"] == turn["id"]
    stored = open(os.path.join(rig.env.LOCAL_AUDIO_DIR, f"turn_{turn['id']}.wav"), "rb").read()
    assert receipt["audio_sha256"] == provenance.sha256_hex(stored)
    import base64
    wav = base64.b64decode(out["messages"][0]["audio_url"].split(",", 1)[1])
    assert wav == stored


def test_text_only_when_voice_for_text_is_off(rig):
    rig.agent(voice_for_text=False)
    rig.send(rig.text_msg("What is the refund window?"))
    assert rig.graph.uploads == 0 and rig.graph.sends("audio") == []
    assert [m["text"]["body"] for m in rig.graph.sends("text")] == [DISCLOSED]


def test_voice_failure_still_sends_the_text(rig):
    rig.agent()
    rig.graph.fail["/media"] = 500
    rig.send(rig.text_msg("What is the refund window?"))
    assert rig.graph.sends("audio") == []
    assert [m["text"]["body"] for m in rig.graph.sends("text")] == [DISCLOSED]


def test_interactive_and_unsupported_messages(rig):
    rig.agent()
    button = {"from": CALLER, "id": "wamid.B1", "timestamp": str(int(time.time())), "type": "interactive",
              "interactive": {"type": "button_reply", "button_reply": {"id": "1", "title": "Refund window"}}}
    rig.send(button)
    assert rig.env.chat_calls[-1]["messages"][-1]["content"].endswith("Caller says: Refund window")
    image = {"from": CALLER, "id": "wamid.I1", "timestamp": str(int(time.time())), "type": "image",
             "image": {"id": "IMG", "mime_type": "image/jpeg"}}
    before = len(rig.env.chat_calls)
    rig.send(image)
    assert len(rig.env.chat_calls) == before
    assert rig.graph.sends("text")[-1]["text"]["body"] == wa.UNSUPPORTED_REPLY["en"]


# ------------------------------------------------------------------------------------------------ voice note in

def test_voice_note_is_downloaded_decoded_and_answered_with_a_voice_note(rig):
    a = rig.agent()
    rig.graph.media["MEDIA1"] = (make_ogg(1.5), "audio/ogg; codecs=opus")
    rig.send(rig.audio_msg("MEDIA1"))
    g = rig.graph

    # media lookup, then the download with the bearer token
    gets = [r for r in g.requests if r["method"] == "GET"]
    assert [r["path"] for r in gets] == ["/v99.0/MEDIA1", "/media/MEDIA1"] or [r["path"] for r in gets][-1] == "/media/MEDIA1"
    assert all(r["headers"]["authorization"] == f"Bearer {TOKEN}" for r in gets)

    # the recording reached speech recognition as 16 kHz mono WAV of the right length
    assert len(rig.asr_inputs) == 1
    import io
    import wave
    with wave.open(io.BytesIO(rig.asr_inputs[0])) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert 1.3 < w.getnframes() / 16000 < 1.8

    turns = rig.env.local_db.select("conversation_turns")
    assert turns[0]["text_content"] == "What is the refund window?"
    assert g.uploads == 1 and len(g.sends("audio")) == 1 and g.sends("text")[0]["text"]["body"] == DISCLOSED


def test_voice_note_replies_even_when_voice_for_text_is_off(rig):
    rig.agent(voice_for_text=False)
    rig.graph.media["MEDIA1"] = (make_ogg(1.0), "audio/ogg")
    rig.send(rig.audio_msg("MEDIA1"))
    assert len(rig.graph.sends("audio")) == 1


def test_the_reply_voice_note_is_ogg_opus_mono_48k_about_24kbps(rig):
    rig.agent()
    rig.send(rig.text_msg("What is the refund window?"))
    (up,) = rig.graph.uploads_seen()
    parts = up["parts"]
    assert parts["messaging_product"]["data"] == b"whatsapp"
    assert parts["type"]["data"] == b"audio/ogg"
    f = parts["file"]
    assert b"Content-Type: audio/ogg; codecs=opus\r\n" in up["body"]  # the mime Meta wants for a voice note
    assert f["data"][:4] == b"OggS" and f["filename"].endswith(".ogg")
    info = probe(f["data"])
    assert info["codec_name"] == "opus" and info["channels"] == 1 and info["sample_rate"] == "48000"
    assert "ogg" in info["format_name"]
    assert up["headers"]["authorization"] == f"Bearer {TOKEN}"


def test_wav_to_voice_note_bitrate():
    import io
    import wave
    n = 48000 * 4
    t = np.arange(n) / 48000
    pcm = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm)
    ogg = wa.wav_to_voice_note(buf.getvalue())
    kbps = len(ogg) * 8 / 4 / 1000
    assert 6 < kbps < 40  # about 24 kbps of voice, with the container overhead
    assert probe(ogg)["codec_name"] == "opus"


def test_voice_note_decoding_edge_cases():
    wav, seconds = wa.voice_note_to_wav(make_ogg(1.0))
    assert wav[:4] == b"RIFF" and 0.9 < seconds < 1.2
    with pytest.raises(wa.WhatsAppError) as e:
        wa.voice_note_to_wav(make_ogg(3.0), max_seconds=2.0)
    assert e.value.code == "TOO_LONG"
    with pytest.raises(wa.WhatsAppError) as e:
        wa.voice_note_to_wav(b"OggS" + b"\x00" * 200)
    assert e.value.code == "UNDECODABLE"


def test_an_oversized_voice_note_is_refused_without_downloading_it(rig):
    rig.agent()
    rig.graph.media["BIG"] = (make_ogg(1.0), "audio/ogg")
    rig.graph.media_info["BIG"] = {"file_size": wa.MAX_MEDIA_BYTES + 1}
    rig.send(rig.audio_msg("BIG"))
    assert not [r for r in rig.graph.requests if r["path"].startswith("/media/")]
    assert rig.graph.sends("text")[-1]["text"]["body"] == wa.TOO_LONG_REPLY["en"]
    assert rig.asr_inputs == []


def test_a_media_url_on_a_foreign_host_is_never_fetched_with_the_token(rig):
    rig.agent()
    rig.graph.media["EVIL"] = (make_ogg(1.0), "audio/ogg")
    rig.graph.media_info["EVIL"] = {"url": "http://169.254.169.254/latest/meta-data/"}
    rig.send(rig.audio_msg("EVIL"))
    assert [r["path"] for r in rig.graph.requests if r["method"] == "GET"] == ["/v99.0/EVIL"]
    assert rig.graph.sends("text")[-1]["text"]["body"] == wa.FAILED_REPLY["en"]
    assert not wa.media_url_ok("https://evil.example.com/x") and wa.media_url_ok("https://lookaside.fbsbx.com/x")
    assert not wa.media_url_ok("http://lookaside.fbsbx.com/x") and not wa.media_url_ok("https://fbsbx.com.evil.io/x")


def test_silence_or_no_speech_gets_a_polite_text(rig):
    rig.agent()
    rig.graph.media["MEDIA1"] = (make_ogg(1.0), "audio/ogg")
    rig.asr_text = ""
    rig.send(rig.audio_msg("MEDIA1"))
    assert rig.graph.sends("text")[-1]["text"]["body"] == wa.UNHEARD_REPLY["en"]
    assert rig.graph.uploads == 0


# ------------------------------------------------------------------------------------------------ handoff

@pytest.mark.parametrize("text", [
    "I want to talk to a human", "can I speak to the manager?", "agent please", "Human", "talk to someone",
    "please connect me to a real person", "I need a representative", "get me a supervisor",
    "Are you a bot? I want a human", "transfer me to a human agent",
    "मुझे मैनेजर से बात करनी है", "किसी इंसान से बात कराइए", "एजेंट से बात करवाओ", "इंसान चाहिए", "मैनेजर",
    "kisi insaan se baat karao", "manager chahiye", "agent se baat karao", "mujhe human chahiye",
    "mujhe kisi se baat karni hai", "human please",
])
def test_handoff_requests_are_recognised(text):
    assert wa.wants_human(text) is True


@pytest.mark.parametrize("text", [
    "What do your agents charge?", "I am a travel agent", "What are your opening hours?", "koi baat nahi",
    "The manager's name is Raj", "I booked via an agent last week", "मेरा ऑर्डर कहाँ है?", "aapka manager kaun hai",
    "how do I get a refund", "kya aap human ho?", "tell me about the human resources policy", "",
])
def test_ordinary_questions_are_not_handoffs(text):
    assert wa.wants_human(text) is False


def test_handoff_sends_the_configured_message_and_flags_the_conversation(rig):
    a = rig.agent(handoff_text="Our team will call you back within the hour.")
    before = len(rig.env.chat_calls)
    rig.send(rig.text_msg("I want to talk to a human"))
    assert len(rig.env.chat_calls) == before  # the model was not asked
    assert [m["text"]["body"] for m in rig.graph.sends("text")] == ["Our team will call you back within the hour."]
    assert rig.graph.uploads == 0
    conv = rig.env.local_db.select("conversations")[0]
    assert conv["needs_human"] is True and conv["handoff_at"] and conv["external_id"] == CALLER
    assert [t["role"] for t in rig.env.local_db.select("conversation_turns")] == ["user", "agent"]

    listed = rig.client.get(f"/api/agents/{a['id']}/whatsapp/handoffs", headers=LOCAL).json()["handoffs"]
    assert [(h["conversation_id"], h["contact"]) for h in listed] == [(conv["id"], CALLER)]
    r = rig.client.post(f"/api/agents/{a['id']}/whatsapp/handoffs/{conv['id']}/resolve", headers=LOCAL)
    assert r.status_code == 200 and r.json()["needs_human"] is False
    assert rig.client.get(f"/api/agents/{a['id']}/whatsapp/handoffs", headers=LOCAL).json() == {"handoffs": []}
    assert rig.client.post(f"/api/agents/{a['id']}/whatsapp/handoffs/does-not-exist-1/resolve",
                           headers=LOCAL).status_code == 404


def test_handoff_default_message_follows_the_callers_language(rig):
    rig.agent()
    rig.send(rig.text_msg("मुझे मैनेजर से बात करनी है", mid="wamid.H1"))
    rig.send(rig.text_msg("kisi insaan se baat karao", mid="wamid.H2", sender="919000000001"))
    rig.send(rig.text_msg("I want a human", mid="wamid.H3", sender="919000000002"))
    bodies = [m["text"]["body"] for m in rig.graph.sends("text")]
    assert bodies == [wa.DEFAULT_HANDOFF["hi"], wa.DEFAULT_HANDOFF["hinglish"], wa.DEFAULT_HANDOFF["en"]]


def test_a_voice_note_asking_for_a_person_is_a_handoff(rig):
    a = rig.agent()
    rig.graph.media["MEDIA1"] = (make_ogg(1.0), "audio/ogg")
    rig.asr_text = "Mujhe kisi insaan se baat karni hai"
    rig.send(rig.audio_msg("MEDIA1"))
    assert rig.env.local_db.select("conversations")[0]["needs_human"] is True
    assert rig.graph.sends("text")[-1]["text"]["body"] == wa.DEFAULT_HANDOFF["hinglish"]
    assert len(rig.client.get(f"/api/agents/{a['id']}/whatsapp/handoffs", headers=LOCAL).json()["handoffs"]) == 1


def test_chats_of_two_callers_stay_separate(rig):
    rig.agent()
    rig.send(rig.text_msg("hello", mid="wamid.X1", sender="919000000001"))
    rig.send(rig.text_msg("hello", mid="wamid.X2", sender="919000000002"))
    convs = rig.env.local_db.select("conversations")
    assert {c["external_id"] for c in convs} == {"919000000001", "919000000002"}
    assert [m["to"] for m in rig.graph.sends("text")] == ["919000000001", "919000000002"]


# ------------------------------------------------------------------------------------------------ owner API and secrets

def test_secrets_are_write_only(rig, caplog):
    caplog.set_level(logging.DEBUG)
    a = rig.agent()
    cfg = rig.client.get(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL)
    assert cfg.status_code == 200
    body = cfg.json()
    assert body["access_token_set"] is True and body["app_secret_set"] is True and body["ready"] is True
    assert body["access_token_hint"] == "••••" + TOKEN[-4:] and body["enabled"] is True
    assert body["phone_number_id"] == PNID and body["webhook_url"].endswith("/api/whatsapp/webhook")

    # not in any response the API gives: settings, the agent, the agent list, a repeated save, the check
    seen_texts = [cfg.text, rig.client.get(f"/api/agents/{a['id']}", headers=LOCAL).text,
                  rig.client.get("/api/agents", headers=LOCAL).text,
                  rig.client.put(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL, json={"handoff_text": "x"}).text,
                  rig.client.post(f"/api/agents/{a['id']}/whatsapp/verify-token", headers=LOCAL).text,
                  rig.client.post(f"/api/agents/{a['id']}/whatsapp/check", headers=LOCAL).text]
    for text in seen_texts:
        assert TOKEN not in text and APP_SECRET not in text and "secrets_enc" not in text
    assert TOKEN[:-4] not in cfg.text

    # not in the agent row, only in the private file
    row = rig.env.local_db.get("agents", a["id"])
    assert TOKEN not in json.dumps(row) and APP_SECRET not in json.dumps(row)
    path = wa.secrets_dir() / f"{a['id']}.secrets.json"
    assert json.loads(path.read_text()) == {"access_token": TOKEN, "app_secret": APP_SECRET}
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    # not in the logs of a full conversation, nor in an error from Meta
    rig.send(rig.text_msg("What is the refund window?"))
    rig.graph.fail["/messages"] = 400
    rig.send(rig.text_msg("again", mid="wamid.T9"))
    assert TOKEN not in caplog.text and APP_SECRET not in caplog.text

    # deleting the settings deletes the secrets
    assert rig.client.delete(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL).status_code == 200
    assert not path.exists()
    assert rig.client.get(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL).json()["access_token_set"] is False


def test_saving_other_fields_keeps_the_stored_secrets(rig):
    a = rig.agent()
    rig.client.put(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL, json={"handoff_text": "Hello", "access_token": ""})
    assert wa.load_secrets(a["id"], local=True) == {"access_token": TOKEN, "app_secret": APP_SECRET}
    rig.client.put(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL, json={"access_token": "EAAGnewtoken987654321"})
    assert wa.load_secrets(a["id"], local=True)["access_token"] == "EAAGnewtoken987654321"
    assert wa.load_secrets(a["id"], local=True)["app_secret"] == APP_SECRET


def test_enabling_needs_the_settings_and_validation(rig):
    a = make_agent(rig.client)
    url = f"/api/agents/{a['id']}/whatsapp"
    first = rig.client.get(url, headers=LOCAL).json()
    assert first["enabled"] is False and first["ready"] is False and first["access_token_set"] is False
    assert rig.client.put(url, headers=LOCAL, json={"enabled": True}).json()["code"] == "PHONE_NUMBER_ID_REQUIRED"
    rig.client.put(url, headers=LOCAL, json={"phone_number_id": PNID})
    assert rig.client.put(url, headers=LOCAL, json={"enabled": True}).json()["code"] == "ACCESS_TOKEN_REQUIRED"
    rig.client.put(url, headers=LOCAL, json={"access_token": TOKEN})
    assert rig.client.put(url, headers=LOCAL, json={"enabled": True}).json()["code"] == "APP_SECRET_REQUIRED"
    rig.client.put(url, headers=LOCAL, json={"app_secret": APP_SECRET})
    ok = rig.client.put(url, headers=LOCAL, json={"enabled": True})
    assert ok.status_code == 200 and ok.json()["enabled"] is True and ok.json()["ready"] is True

    assert rig.client.put(url, headers=LOCAL, json={"phone_number_id": "+91 98765 43210"}).status_code == 422
    assert rig.client.put(url, headers=LOCAL, json={"access_token": "short"}).status_code == 422
    assert rig.client.put(url, headers=LOCAL, json={"app_secret": "has space inside it"}).status_code == 422
    assert rig.client.get("/api/agents/nope/whatsapp", headers=LOCAL).status_code == 404


def test_a_phone_number_belongs_to_one_agent(rig):
    a = rig.agent()
    b = make_agent(rig.client)
    r = rig.client.put(f"/api/agents/{b['id']}/whatsapp", headers=LOCAL, json={"phone_number_id": PNID})
    assert r.status_code == 409 and r.json()["code"] == "PHONE_NUMBER_IN_USE"
    assert rig.client.put(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL,
                          json={"phone_number_id": PNID}).status_code == 200  # the same agent may re-save it


def test_verify_token_rotation_invalidates_the_old_one(rig):
    a = rig.agent()
    old = rig.client.get(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL).json()["verify_token"]
    new = rig.client.post(f"/api/agents/{a['id']}/whatsapp/verify-token", headers=LOCAL).json()["verify_token"]
    assert new != old
    ask = lambda t: rig.client.get("/api/whatsapp/webhook", params={  # noqa: E731
        "hub.mode": "subscribe", "hub.verify_token": t, "hub.challenge": "7"}).status_code
    assert ask(old) == 403 and ask(new) == 200


def test_check_connection_asks_meta(rig):
    a = rig.agent()
    r = rig.client.post(f"/api/agents/{a['id']}/whatsapp/check", headers=LOCAL).json()
    assert r == {"ok": True, "display_phone_number": "+1 555 010 9999", "verified_name": "Test Shop"}
    rig.client.put(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL, json={"access_token": "EAAGwrongtoken000000"})
    bad = rig.client.post(f"/api/agents/{a['id']}/whatsapp/check", headers=LOCAL).json()
    assert bad["ok"] is False and "Invalid OAuth access token" in bad["error"]
    assert "EAAGwrongtoken000000" not in json.dumps(bad)


def test_webhook_url_uses_the_public_base_url(rig, monkeypatch):
    a = rig.agent()
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://abc123.trycloudflare.com/")
    assert rig.client.get(f"/api/agents/{a['id']}/whatsapp", headers=LOCAL).json()["webhook_url"] == \
        "https://abc123.trycloudflare.com/api/whatsapp/webhook"


def test_owner_endpoints_need_the_owner(rig):
    a = rig.agent()
    other = rig.env.local_db.insert("agents", {"user_id": "someone-else", "name": "X", "system_instructions": "x",
                                               "greeting": "", "supported_languages": ["en"]})
    assert rig.client.get(f"/api/agents/{other['id']}/whatsapp", headers=LOCAL).status_code == 404
    assert rig.client.put(f"/api/agents/{other['id']}/whatsapp", headers=LOCAL, json={"handoff_text": "x"}).status_code == 404
    assert rig.client.post(f"/api/agents/{other['id']}/whatsapp/simulate", headers=LOCAL,
                           json={"text": "hi"}).status_code == 404


# ------------------------------------------------------------------------------------------------ dry run

def test_simulate_runs_the_flow_without_calling_meta(rig):
    a = make_agent(rig.client)  # no Meta settings at all
    r = rig.client.post(f"/api/agents/{a['id']}/whatsapp/simulate", headers=LOCAL,
                        json={"text": "What is the refund window?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dry_run"] is True and body["handoff"] is False and body["voice_sent"] is True
    assert body["agent_text"] == DISCLOSED and body["language"] == "en"
    kinds = [m["type"] for m in body["messages"]]
    assert kinds == ["audio", "text"]
    audio, text = body["messages"]
    assert audio["mime"] == "audio/ogg; codecs=opus" and audio["bytes"] > 500 and audio["audio_url"].startswith("data:audio/wav;base64,")
    assert text["body"] == DISCLOSED
    assert rig.graph.requests == []

    # an owner test: kept out of Analytics and out of the handoff list, and it keeps its own conversation
    conv = rig.env.local_db.select("conversations")[0]
    assert conv["is_test"] is True and conv["source"] == "whatsapp"
    r2 = rig.client.post(f"/api/agents/{a['id']}/whatsapp/simulate", headers=LOCAL, json={"text": "And sale items?"})
    assert r2.json()["agent_text"] == REPLY and len(rig.env.local_db.select("conversations")) == 1


def test_simulate_handoff_and_validation(rig):
    a = make_agent(rig.client)
    r = rig.client.post(f"/api/agents/{a['id']}/whatsapp/simulate", headers=LOCAL, json={"text": "human please"}).json()
    assert r["handoff"] is True and [m["type"] for m in r["messages"]] == ["text"]
    assert r["messages"][0]["body"] == wa.DEFAULT_HANDOFF["en"]
    assert rig.client.get(f"/api/agents/{a['id']}/whatsapp/handoffs", headers=LOCAL).json() == {"handoffs": []}
    assert rig.client.post(f"/api/agents/{a['id']}/whatsapp/simulate", headers=LOCAL, json={"text": ""}).status_code == 422


def test_pipeline_failure_gets_an_apology_not_silence(rig, monkeypatch):
    rig.agent()

    async def boom(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(rig.env, "_turn_blocking", boom)
    rig.send(rig.text_msg("hello"))
    assert rig.graph.sends("text")[-1]["text"]["body"] == wa.FAILED_REPLY["en"]
