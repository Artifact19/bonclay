"""Phone calls through several providers (routers/phone_exotel.py + services/telephony): which provider and account carry a
call (PHONE_PROVIDERS, PHONE_ROUTES), failover across providers and accounts, the routes each provider uses (stream, status
callback, answer document, inbound link), and whole calls over a Twilio-, Plivo- and Telnyx-shaped stream through the same call
runner as Exotel, with the same fakes as test_phone_exotel.py (no model, no network)."""
import base64
import json
import logging
import os
import stat
import time
import xml.etree.ElementTree as ET

import httpx
import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

import phone_public
from routers import phone_exotel as px
from services import phone_audio as pa
from services.telephony import registry
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_phone_exotel import call, pcm_silence, pcm_tone, read_until, rig  # noqa: F401  (rig is a fixture)

ALL_VARS = {
    "exotel": ("EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_ACCOUNT_SID", "EXOTEL_CALLER_ID", "EXOTEL_SUBDOMAIN"),
    "twilio": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_CALLER_ID"),
    "plivo": ("PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN", "PLIVO_CALLER_ID"),
    "vobiz": ("VOBIZ_AUTH_ID", "VOBIZ_AUTH_TOKEN", "VOBIZ_CALLER_ID", "VOBIZ_BASE_URL"),
    "telnyx": ("TELNYX_API_KEY", "TELNYX_CONNECTION_ID", "TELNYX_CALLER_ID"),
}
KYC = "Your account is not yet KYC compliant. This is mandatory before making outbound calls."


@pytest.fixture()
def clean(monkeypatch):
    """No provider setting from a real backend/.env; a public address; every provider's rotation and routes at the default."""
    for names in ALL_VARS.values():
        for n in range(1, 10):
            for name in names:
                monkeypatch.delenv(name + ("" if n == 1 else f"_{n}"), raising=False)
    for name in ("PHONE_PROVIDERS", "PHONE_ROUTES", "PHONE_ALLOWED_NUMBERS", "EXOTEL_ROTATION", "TWILIO_ROTATION",
                 "EXOTEL_STATUS_CALLBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://abc.ngrok.app")
    monkeypatch.setattr(px, "_rr_counter", 0)
    registry._warned.clear()
    registry._books.clear()
    px._warned_partial.clear()


def account(monkeypatch, provider, n=1, **over):
    suffix = "" if n == 1 else f"_{n}"
    vals = {
        "exotel": {"EXOTEL_API_KEY": f"ekey{n}", "EXOTEL_API_TOKEN": f"etok{n}", "EXOTEL_ACCOUNT_SID": f"esid{n}",
                   "EXOTEL_CALLER_ID": f"0804749189{n}"},
        "twilio": {"TWILIO_ACCOUNT_SID": f"ACtw{n}", "TWILIO_AUTH_TOKEN": f"twtok{n}", "TWILIO_CALLER_ID": f"+1500555000{n}"},
        "plivo": {"PLIVO_AUTH_ID": f"MAplivo{n}", "PLIVO_AUTH_TOKEN": f"pltok{n}", "PLIVO_CALLER_ID": f"+1415555000{n}"},
        "vobiz": {"VOBIZ_AUTH_ID": f"VBid{n}", "VOBIZ_AUTH_TOKEN": f"vbtok{n}", "VOBIZ_CALLER_ID": f"+91224000000{n}"},
        "telnyx": {"TELNYX_API_KEY": f"KEYtx{n}", "TELNYX_CONNECTION_ID": f"conn{n}", "TELNYX_CALLER_ID": f"+1312555000{n}"},
    }[provider]
    for name, v in {**vals, **over}.items():
        if v is None:
            monkeypatch.delenv(name + suffix, raising=False)
        else:
            monkeypatch.setenv(name + suffix, v)


class Routed:
    """A fake carrier per host: `replies` maps a part of the URL to the answers its next requests get (the last repeats)."""
    replies: dict = {}
    sent: list = []
    error_for: set = set()

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def _do(self, method, url, **kw):
        Routed.sent.append({"method": method, "url": url, **kw})
        for key in Routed.error_for:
            if key in url:
                raise httpx.ConnectError("down")
        for key, queue in Routed.replies.items():
            if key in url:
                status, body = queue.pop(0) if len(queue) > 1 else queue[0]
                return httpx.Response(status, text=body) if isinstance(body, str) else httpx.Response(status, json=body)
        defaults = {"exotel": {"Call": {"Sid": "CA-exotel"}}, "twilio.com": {"sid": "CA-twilio", "status": "queued"},
                    "plivo.com": {"request_uuid": "uuid-plivo"}, "vobiz.ai": {"request_uuid": "uuid-vobiz"},
                    "telnyx.com": {"data": {"call_control_id": "cc-telnyx"}}}
        for key, body in defaults.items():
            if key in url:
                return httpx.Response(200, json=body)
        return httpx.Response(200, json={})

    async def post(self, url, **kw):
        return await self._do("post", url, **kw)

    async def get(self, url, **kw):
        return await self._do("get", url, **kw)

    async def delete(self, url, **kw):
        return await self._do("delete", url, **kw)


@pytest.fixture()
def fake(monkeypatch):
    Routed.replies, Routed.sent, Routed.error_for = {}, [], set()
    monkeypatch.setattr(httpx, "AsyncClient", Routed)
    return Routed


def hosts():
    return [next(h for h in ("exotel", "twilio", "plivo", "vobiz", "telnyx") if h in s["url"]) for s in Routed.sent]


@pytest.fixture()
def ready(rig, clean, monkeypatch):
    monkeypatch.setenv("PHONE_ALLOWED_NUMBERS", "+919876543210")
    monkeypatch.setattr(px, "CALLS_PER_HOUR", 100)
    return rig, make_agent(rig.client)


def state_of(provider, label="1"):
    a = next(a for a in registry.provider(provider).accounts() if a["label"] == label)
    return registry.book(provider).state(a)["state"]


# ------------------------------------------------------------------------------------------------ settings and order

def test_the_default_is_exotel_only_and_the_old_settings_are_unchanged(clean, monkeypatch):
    assert registry.enabled() == ["exotel"]
    assert px.missing_settings() == ["EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_ACCOUNT_SID", "EXOTEL_CALLER_ID"]
    account(monkeypatch, "exotel")
    assert px.missing_settings() == [] and [a.label for a in registry.attempts(None)] == ["exotel 1"]


def test_providers_are_listed_in_the_order_of_the_setting_unknown_names_are_ignored_with_one_warning(clean, monkeypatch, caplog):
    monkeypatch.setenv("PHONE_PROVIDERS", " Twilio, exotel ,plivo,twilio,nonsense")
    with caplog.at_level(logging.WARNING, logger="bonclay.phone"):
        assert registry.enabled() == ["twilio", "exotel", "plivo"]
        registry.enabled()
    assert len([r for r in caplog.records if "nonsense" in r.getMessage()]) == 1
    monkeypatch.setenv("PHONE_PROVIDERS", " , ")
    assert registry.enabled() == ["exotel"]


@pytest.mark.parametrize("provider,missing", [
    ("twilio", ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_CALLER_ID"]),
    ("plivo", ["PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN", "PLIVO_CALLER_ID"]),
    ("vobiz", ["VOBIZ_AUTH_ID", "VOBIZ_AUTH_TOKEN", "VOBIZ_CALLER_ID"]),
    ("telnyx", ["TELNYX_API_KEY", "TELNYX_CONNECTION_ID", "TELNYX_CALLER_ID"]),
])
def test_missing_settings_names_the_first_providers_variables(clean, monkeypatch, provider, missing):
    monkeypatch.setenv("PHONE_PROVIDERS", f"{provider},exotel")
    assert px.missing_settings() == missing + []
    account(monkeypatch, provider)
    assert px.missing_settings() == []
    monkeypatch.delenv("PUBLIC_BASE_URL")
    assert px.missing_settings() == ["PUBLIC_BASE_URL"]


def test_accounts_use_the_same_suffix_scheme_for_every_provider(clean, monkeypatch):
    for n in (1, 3):
        account(monkeypatch, "twilio", n)
    account(monkeypatch, "vobiz", 2, VOBIZ_BASE_URL="https://vb.example/api/v1")
    account(monkeypatch, "telnyx", 1)
    account(monkeypatch, "plivo", 2, PLIVO_AUTH_TOKEN=None)               # half filled: ignored
    assert [a["label"] for a in registry.provider("twilio").accounts()] == ["1", "3"]
    assert registry.provider("vobiz").accounts()[0]["base_url"] == "https://vb.example/api/v1"
    assert registry.provider("plivo").accounts() == []
    assert registry.provider("telnyx").accounts()[0]["connection_id"] == "conn1"
    account(monkeypatch, "vobiz", 2, VOBIZ_BASE_URL=None)
    assert registry.provider("vobiz").accounts()[0]["base_url"] == "https://api.vobiz.ai/api/v1"


def test_attempts_go_providers_in_order_each_with_its_accounts(clean, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio,exotel")
    for p, n in (("exotel", 1), ("exotel", 2), ("twilio", 1), ("twilio", 2)):
        account(monkeypatch, p, n)
    assert [a.label for a in registry.attempts(None)] == ["twilio 1", "twilio 2", "exotel 1", "exotel 2"]


def test_marked_accounts_go_after_every_active_one_even_of_the_next_provider(clean, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio,exotel")
    for p, n in (("exotel", 1), ("twilio", 1), ("twilio", 2)):
        account(monkeypatch, p, n)
    tw = registry.provider("twilio").accounts()
    registry.book("twilio").set(tw[0], "exhausted")
    assert [a.label for a in registry.attempts(None)] == ["twilio 2", "exotel 1", "twilio 1"]


def test_routes_pin_a_number_to_a_provider_or_account_first(clean, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "exotel,twilio")
    for p, n in (("exotel", 1), ("exotel", 2), ("twilio", 1)):
        account(monkeypatch, p, n)
    monkeypatch.setenv("PHONE_ROUTES", "+919876543210:exotel#2, 09812345678:twilio ,garbage,+919800000000:nope#x")
    assert registry.routes() == {"+919876543210": [("exotel", "2")], "+919812345678": [("twilio", None)]}
    assert [a.label for a in registry.attempts("+919876543210")] == ["exotel 2", "exotel 1", "twilio 1"]
    assert [a.label for a in registry.attempts("+919812345678")] == ["twilio 1", "exotel 1", "exotel 2"]
    assert [a.label for a in registry.attempts("+919811111111")] == ["exotel 1", "exotel 2", "twilio 1"]


def test_a_route_to_a_provider_that_is_not_configured_is_skipped(clean, monkeypatch):
    account(monkeypatch, "exotel")
    monkeypatch.setenv("PHONE_ROUTES", "+919876543210:telnyx")
    assert [a.label for a in registry.attempts("+919876543210")] == ["exotel 1"]


def test_round_robin_is_per_provider(clean, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio")
    monkeypatch.setenv("TWILIO_ROTATION", "round_robin")
    for n in (1, 2, 3):
        account(monkeypatch, "twilio", n)
    firsts = [registry.attempts(None)[0].label for _ in range(4)]
    assert firsts == ["twilio 1", "twilio 2", "twilio 3", "twilio 1"]


# ------------------------------------------------------------------------------------------------------- dialling

def test_one_exotel_account_behaves_as_before_and_the_reply_names_provider_and_account(ready, fake, monkeypatch):
    rig, a = ready
    account(monkeypatch, "exotel")
    r = call(rig, a)
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "calling", "to": "91******3210", "call_sid": "CA-exotel", "agent_id": a["id"],
                        "provider": "exotel", "account": "1"}
    assert hosts() == ["exotel"]
    assert px._known_calls["CA-exotel"]["provider"] == "exotel"
    rec = px.call_records.get("CA-exotel")
    assert rec["provider"] == "exotel" and rec["account"] == "1" and rec["exotel_account"] == "1"


def test_a_call_goes_through_the_first_provider_in_the_setting(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio,exotel")
    account(monkeypatch, "exotel")
    account(monkeypatch, "twilio")
    r = call(rig, a)
    assert r.status_code == 200, r.text
    assert r.json()["provider"] == "twilio" and r.json()["account"] == "1" and r.json()["call_sid"] == "CA-twilio"
    assert hosts() == ["twilio"]
    sent = Routed.sent[0]
    claims = px.read_token(ET.fromstring(sent["data"]["Twiml"]).find("Connect/Stream").get("url").rsplit("/", 1)[1])
    assert claims["p"] == "twilio" and claims["a"] == a["id"] and "k" not in claims
    assert sent["data"]["StatusCallback"].startswith("https://abc.ngrok.app/api/phone/twilio/status-callback/")
    assert px.call_records.get("CA-twilio")["provider"] == "twilio"
    assert "exotel_account" not in px.call_records.get("CA-twilio")


def test_plivo_is_given_an_answer_url_that_returns_the_stream_xml(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "plivo")
    account(monkeypatch, "plivo")
    assert call(rig, a).status_code == 200
    body = Routed.sent[0]["json"]
    assert body["answer_url"].startswith("https://abc.ngrok.app/api/phone/plivo/answer/")
    r = rig.client.get(body["answer_url"].replace("https://abc.ngrok.app", ""))
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/xml")
    stream = ET.fromstring(r.text).find("Stream")
    assert stream.get("bidirectional") == "true"
    wss = stream.text.strip()
    assert wss.startswith("wss://abc.ngrok.app/api/phone/plivo/stream/")
    claims = px.read_token(wss.rsplit("/", 1)[1])
    assert claims["p"] == "plivo" and claims["a"] == a["id"] and "k" not in claims


def test_telnyx_and_vobiz_calls_name_themselves_too(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "telnyx,vobiz")
    account(monkeypatch, "telnyx")
    account(monkeypatch, "vobiz")
    r = call(rig, a)
    assert r.json()["provider"] == "telnyx" and r.json()["call_sid"] == "cc-telnyx"
    assert Routed.sent[0]["json"]["stream_url"].startswith("wss://abc.ngrok.app/api/phone/telnyx/stream/")
    registry.book("telnyx").set(registry.provider("telnyx").accounts()[0], "exhausted")
    Routed.replies = {"telnyx.com": [(402, {"errors": [{"code": "20100", "detail": "Insufficient Funds"}]})]}
    r = call(rig, a)
    assert r.status_code == 200 and r.json()["provider"] == "vobiz" and r.json()["call_sid"] == "uuid-vobiz"


def test_failover_moves_from_a_provider_that_refuses_the_account_to_the_next(ready, fake, monkeypatch, caplog):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "exotel,twilio")
    account(monkeypatch, "exotel")
    account(monkeypatch, "twilio")
    fake.replies = {"exotel": [(403, {"Message": KYC})]}
    with caplog.at_level(logging.WARNING, logger="bonclay.phone.exotel"):
        r = call(rig, a)
    assert r.status_code == 200 and r.json()["provider"] == "twilio"
    assert hosts() == ["exotel", "twilio"]
    assert "Exotel account 1 not KYC compliant; trying Twilio account 1" in "\n".join(x.getMessage() for x in caplog.records)
    assert state_of("exotel") == "active"                                # KYC marks nothing


def test_failover_walks_accounts_and_providers_and_marks_the_ones_out_of_credit(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio,plivo")
    account(monkeypatch, "twilio", 1)
    account(monkeypatch, "twilio", 2)
    account(monkeypatch, "plivo", 1)
    fake.replies = {"AC" + "tw1": [(402, {"message": "Payment required"})],
                    "ACtw2": [(401, {"code": 20003, "message": "Authenticate"})]}
    r = call(rig, a)
    assert r.status_code == 200 and r.json()["provider"] == "plivo"
    assert hosts() == ["twilio", "twilio", "plivo"]
    assert state_of("twilio", "1") == "exhausted" and state_of("twilio", "2") == "auth_failed"
    fake.sent.clear()
    assert call(rig, a).json()["provider"] == "plivo"                      # the marked accounts are tried last
    assert hosts() == ["plivo"]                                             # ...so plivo took it at the first try


def test_when_every_provider_refuses_the_error_names_provider_and_account(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "exotel,twilio")
    account(monkeypatch, "exotel")
    account(monkeypatch, "twilio")
    fake.replies = {"exotel": [(403, {"Message": KYC})], "twilio": [(401, {"code": 20003, "message": "Authenticate"})]}
    r = call(rig, a)
    assert r.status_code == 503 and r.json()["code"] == "PHONE_ALL_ACCOUNTS_FAILED"
    msg = r.json()["error"]
    assert "exotel 1 not KYC compliant, twilio 1 credentials refused" in msg
    for secret in ("ekey1", "etok1", "esid1", "ACtw1", "twtok1"):
        assert secret not in msg
    assert hosts() == ["exotel", "twilio"]                                  # one try per account


def test_one_provider_with_two_accounts_keeps_the_exotel_style_message(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio")
    account(monkeypatch, "twilio", 1)
    account(monkeypatch, "twilio", 2)
    fake.replies = {"ACtw1": [(402, {"message": "Payment required"})], "ACtw2": [(401, {"message": "Authenticate"})]}
    r = call(rig, a)
    assert r.status_code == 503 and r.json()["code"] == "TWILIO_ALL_ACCOUNTS_EXHAUSTED"
    assert r.json()["error"].startswith("All 2 Twilio accounts are out of credit or refused (1 out of credit, 2 credentials refused)")


def test_any_other_refusal_or_a_network_error_stops_without_trying_the_next_provider(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio,exotel")
    account(monkeypatch, "exotel")
    account(monkeypatch, "twilio")
    fake.replies = {"twilio": [(400, {"code": 21211, "message": "Invalid 'To' number"})]}
    r = call(rig, a)
    assert r.status_code == 502 and r.json()["code"] == "TWILIO_REFUSED" and "Invalid 'To' number" in r.json()["error"]
    assert hosts() == ["twilio"] and state_of("twilio") == "active"
    fake.sent.clear()
    fake.replies = {}
    fake.error_for = {"twilio.com"}
    r = call(rig, a)
    assert r.status_code == 502 and r.json()["code"] == "TWILIO_UNREACHABLE"
    assert hosts() == ["twilio"]


def test_a_single_account_gives_the_plain_refusal_and_marks_it(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "telnyx")
    account(monkeypatch, "telnyx")
    fake.replies = {"telnyx.com": [(402, {"errors": [{"code": "20100", "title": "Insufficient Funds", "detail": "Top up"}]})]}
    r = call(rig, a)
    assert r.status_code == 502 and r.json()["code"] == "TELNYX_REFUSED" and "Top up" in r.json()["error"]
    assert state_of("telnyx") == "exhausted"


def test_routed_numbers_use_their_provider_first_and_fall_through_when_it_fails(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "exotel,twilio")
    monkeypatch.setenv("PHONE_ROUTES", "+919876543210:twilio")
    account(monkeypatch, "exotel")
    account(monkeypatch, "twilio")
    assert call(rig, a).json()["provider"] == "twilio" and hosts() == ["twilio"]
    fake.sent.clear()
    fake.replies = {"twilio": [(401, {"code": 20003, "message": "Authenticate"})]}
    r = call(rig, a)
    assert r.json()["provider"] == "exotel" and hosts() == ["twilio", "exotel"]


def test_the_unconfigured_message_names_phone_calling_when_several_providers_are_listed(rig, clean, monkeypatch):
    monkeypatch.setenv("PHONE_ALLOWED_NUMBERS", "+919876543210")
    a = make_agent(rig.client)
    r = call(rig, a)
    assert r.status_code == 503 and r.json()["error"].startswith("Exotel is not set up: add EXOTEL_API_KEY")
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio,exotel")
    r = call(rig, a)
    assert r.status_code == 503 and r.json()["error"].startswith("Phone calling is not set up: add TWILIO_ACCOUNT_SID")


def test_state_of_other_providers_is_kept_privately_apart_from_exotels(ready, fake, monkeypatch, tmp_path):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio,exotel")
    account(monkeypatch, "twilio")
    account(monkeypatch, "exotel")
    fake.replies = {"twilio": [(402, {"message": "Payment required"})]}
    assert call(rig, a).json()["provider"] == "exotel"
    path = tmp_path / "phone_accounts_twilio.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    raw = path.read_text()
    assert "exhausted" in raw and "ACtw1" not in raw and "twtok1" not in raw
    assert not (tmp_path / "exotel_accounts.json").exists()


# ---------------------------------------------------------------------------------------- status, polling, callbacks

def test_the_status_route_lists_every_provider_without_a_secret(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio,exotel")
    account(monkeypatch, "exotel")
    account(monkeypatch, "twilio", 1)
    account(monkeypatch, "twilio", 2)
    fake.replies = {"ACtw1": [(402, {"message": "Payment required"})]}
    call(rig, a)
    body = rig.client.get("/api/phone/exotel/status", headers=LOCAL).json()
    assert body["configured"] is True and body["provider_order"] == ["twilio", "exotel"]
    assert [x["label"] for x in body["accounts"]] == ["1"]                  # the old Exotel-only field, unchanged
    by = {p["name"]: p for p in body["providers"]}
    assert list(by) == ["exotel", "twilio", "plivo", "vobiz", "telnyx"]
    assert by["twilio"]["configured"] is True and by["plivo"]["configured"] is False
    assert [(x["label"], x["state"]) for x in by["twilio"]["accounts"]] == [("1", "exhausted"), ("2", "active")]
    assert by["twilio"]["accounts"][0]["caller_id_masked"].endswith("0001")
    assert by["twilio"]["enabled"] is True and by["telnyx"]["enabled"] is False and by["twilio"]["verified"] is False
    raw = json.dumps(body)
    for secret in ("ACtw1", "ACtw2", "twtok1", "ekey1", "etok1", "esid1", "+15005550001"):
        assert secret not in raw


def test_call_status_asks_the_provider_and_account_that_placed_the_call(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio")
    account(monkeypatch, "twilio", 1)
    account(monkeypatch, "twilio", 2)
    fake.replies = {"ACtw1": [(402, {"message": "Payment required"})],
                    "ACtw2": [(201, {"sid": "CAsecond"})]}
    sid = call(rig, a).json()["call_sid"]
    assert sid == "CAsecond" and px._known_calls[sid]["account"] == "2"
    fake.replies = {"Calls/CAsecond": [(200, {"status": "in-progress", "duration": None})]}
    r = rig.client.get(f"/api/phone/exotel/call/{sid}", headers=LOCAL)
    assert r.status_code == 200 and r.json()["status"] == "in-progress"
    got = fake.sent[-1]
    assert got["method"] == "get" and got["url"] == "https://api.twilio.com/2010-04-01/Accounts/ACtw2/Calls/CAsecond.json"
    assert got["auth"] == ("ACtw2", "twtok2")
    fake.replies = {"Calls/CAsecond": [(500, "boom")]}
    r = rig.client.get(f"/api/phone/exotel/call/{sid}", headers=LOCAL)
    assert r.status_code == 502 and r.json()["code"] == "TWILIO_REFUSED" and "Twilio could not give the call status" in r.json()["error"]


@pytest.mark.parametrize("provider,form,sid,status", [
    ("twilio", {"CallSid": "CAcb", "CallStatus": "completed", "CallDuration": "9"}, "CAcb", "completed"),
    ("plivo", {"CallUUID": "uuid-cb", "CallStatus": "busy", "Duration": "0"}, "uuid-cb", "busy"),
    ("vobiz", {"CallUUID": "uuid-vb", "Event": "Hangup", "Status": "completed", "Duration": "3"}, "uuid-vb", "completed"),
])
def test_status_callbacks_update_the_record_through_the_providers_own_fields(rig, clean, monkeypatch, provider, form, sid, status):
    monkeypatch.setenv("PHONE_PROVIDERS", provider)
    a = make_agent(rig.client)
    token = px.make_token(a["id"], "local-user", "local", 300, provider=provider)
    r = rig.client.post(f"/api/phone/{provider}/status-callback/{token}", data=form)
    assert r.status_code == 200 and r.json() == {"ok": True}
    rec = px.call_records.get(sid)
    assert rec["status"] == status and rec["owner"] == "local-user" and rec["agent_id"] == a["id"]
    # a token for another provider, or a forged one, files nothing
    other = px.make_token(a["id"], "local-user", "local", 300)
    assert rig.client.post(f"/api/phone/{provider}/status-callback/{other}", data=form).json() == {"ok": False}
    assert rig.client.post(f"/api/phone/{provider}/status-callback/forged", data=form).json() == {"ok": False}


def test_a_telnyx_webhook_is_json(rig, clean, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "telnyx")
    a = make_agent(rig.client)
    token = px.make_token(a["id"], "local-user", "local", 300, provider="telnyx")
    hook = {"data": {"event_type": "call.hangup", "payload": {"call_control_id": "cc-w", "hangup_cause": "user_busy"}}}
    r = rig.client.post(f"/api/phone/telnyx/status-callback/{token}", json=hook)
    assert r.json() == {"ok": True} and px.call_records.get("cc-w")["status"] == "busy"


# ----------------------------------------------------------------------------------- inbound: answer documents, links

def test_the_inbound_link_for_twilio_returns_twiml_pointing_at_a_fresh_stream(rig, clean, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio")
    a = make_agent(rig.client)
    r = rig.client.post("/api/phone/twilio/inbound-link", headers=LOCAL, json={"agent_id": a["id"]})
    assert r.status_code == 200, r.text
    link = r.json()
    assert link["provider"] == "twilio" and link["answer_url"].startswith("https://abc.ngrok.app/api/phone/twilio/answer/")
    assert link["valid_days"] >= 20 and "Voice Configuration" in link["how"]
    path = link["answer_url"].replace("https://abc.ngrok.app", "")
    for method in (rig.client.post, rig.client.get):
        doc = method(path)
        assert doc.status_code == 200 and doc.headers["content-type"].startswith("application/xml")
        url = ET.fromstring(doc.text).find("Connect/Stream").get("url")
        claims = px.read_token(url.rsplit("/", 1)[1])
        assert url.startswith("wss://abc.ngrok.app/api/phone/twilio/stream/")
        assert claims["p"] == "twilio" and claims["a"] == a["id"] and "k" not in claims
        assert claims["x"] < time.time() + 3600           # fresh and short lived, not the 30-day link


def test_an_answer_token_opens_no_stream_and_a_stream_token_fetches_no_answer(rig, clean, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio,plivo")
    a = make_agent(rig.client)
    answer = px.make_token(a["id"], "local-user", "local", 300, provider="twilio", kind="ans")
    stream = px.make_token(a["id"], "local-user", "local", 300, provider="twilio")
    other = px.make_token(a["id"], "local-user", "local", 300, provider="plivo", kind="ans")
    with pytest.raises(WebSocketDisconnect) as e:
        with rig.client.websocket_connect(f"/api/phone/twilio/stream/{answer}"):
            pass
    assert e.value.code == 4401
    assert rig.client.get(f"/api/phone/twilio/answer/{stream}").status_code == 404
    assert rig.client.get(f"/api/phone/twilio/answer/{other}").status_code == 404      # another provider's token
    assert rig.client.get("/api/phone/twilio/answer/garbage").status_code == 404


def test_plivo_vobiz_links_return_plivo_xml_and_name_the_console_step(rig, clean, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "plivo,vobiz")
    a = make_agent(rig.client)
    for name in ("plivo", "vobiz"):
        link = rig.client.post(f"/api/phone/{name}/inbound-link", headers=LOCAL, json={"agent_id": a["id"]}).json()
        assert "Answer URL" in link["how"]
        doc = rig.client.get(link["answer_url"].replace("https://abc.ngrok.app", ""))
        assert ET.fromstring(doc.text).find("Stream").text.strip().startswith(f"wss://abc.ngrok.app/api/phone/{name}/stream/")


def test_a_telnyx_inbound_call_is_answered_through_the_api_with_the_stream_attached(rig, clean, fake, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "telnyx")
    account(monkeypatch, "telnyx", 1)
    account(monkeypatch, "telnyx", 2)
    a = make_agent(rig.client)
    link = rig.client.post("/api/phone/telnyx/inbound-link", headers=LOCAL, json={"agent_id": a["id"]}).json()
    path = link["answer_url"].replace("https://abc.ngrok.app", "")
    hook = {"data": {"event_type": "call.initiated", "payload": {"call_control_id": "cc-in", "connection_id": "conn2",
                                                               "direction": "incoming", "from": "+919800000000"}}}
    assert rig.client.post(path, json=hook).json() == {"ok": True}
    sent = fake.sent[0]
    assert sent["url"] == "https://api.telnyx.com/v2/calls/cc-in/actions/answer"
    assert sent["headers"]["Authorization"] == "Bearer KEYtx2"             # the account whose connection took the call
    assert sent["json"]["stream_url"].startswith("wss://abc.ngrok.app/api/phone/telnyx/stream/")
    rec = px.call_records.get("cc-in")
    assert rec["direction"] == "inbound" and rec["provider"] == "telnyx" and rec["account"] == "2"


@pytest.mark.parametrize("name", ["twilio", "plivo", "vobiz", "telnyx"])
def test_the_inbound_link_of_an_unlisted_provider_still_works_but_its_routes_stay_inert(rig, clean, monkeypatch, name):
    a = make_agent(rig.client)                                              # PHONE_PROVIDERS is the default: exotel
    link = rig.client.post(f"/api/phone/{name}/inbound-link", headers=LOCAL, json={"agent_id": a["id"]}).json()
    assert rig.client.get(link["answer_url"].replace("https://abc.ngrok.app", "")).status_code == 404
    token = px.make_token(a["id"], "local-user", "local", 300, provider=name)
    with pytest.raises(WebSocketDisconnect) as e:
        with rig.client.websocket_connect(f"/api/phone/{name}/stream/{token}"):
            pass
    assert e.value.code == 4404


# ------------------------------------------------------------------------------------------- the public (tunnel) app

@pytest.fixture()
def public(monkeypatch):
    monkeypatch.setenv("WIDGET_TOKEN_SECRET", "test-secret")
    from fastapi.testclient import TestClient
    return TestClient(phone_public.public_app, raise_server_exceptions=False)


def test_the_public_app_exposes_only_stream_callback_and_answer_routes_for_every_provider(public, clean, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "exotel,twilio,plivo,vobiz,telnyx")
    assert len(phone_public.public_app.routes) == 3                          # the health probe, the phone router, the /call/ router (tests/test_web_call.py)
    routes = {getattr(r, "path", "") for r in px.public_router.routes} | {"/api/phone/public/health"}
    expected = {"/api/phone/public/health", px.STREAM_PATH + "/{token}", px.STATUS_CALLBACK_PATH + "/{token}"}
    for n in px.PROVIDER_NAMES:
        expected |= {f"/api/phone/{n}/stream/{{token}}", f"/api/phone/{n}/status-callback/{{token}}",
                     f"/api/phone/{n}/answer/{{token}}"}
    assert routes == expected
    assert public.get("/api/phone/public/health").json()["routes"] == sorted(expected - {"/api/phone/public/health"})
    for n in px.PROVIDER_NAMES:
        for path in (f"/api/phone/{n}/inbound-link", f"/api/phone/{n}/call/x", f"/api/phone/{n}/status", f"/api/phone/{n}/numbers",
                     f"/api/phone/{n}/live", f"/api/phone/{n}/calls", f"/api/phone/{n}/monitor/x"):
            assert public.get(path).status_code == 404, path
            assert public.post(path, json={}).status_code == 404, path
        assert public.post(f"/api/phone/{n}/status-callback/bad", data={}).json() == {"ok": False}
        assert public.get(f"/api/phone/{n}/answer/bad").status_code == 404
        with pytest.raises(WebSocketDisconnect) as e:
            with public.websocket_connect(f"/api/phone/{n}/stream/bad"):
                pass
        assert e.value.code == 4401


def test_the_health_probe_lists_only_the_routes_of_providers_in_use(public, clean, monkeypatch):
    base = sorted([px.STREAM_PATH + "/{token}", px.STATUS_CALLBACK_PATH + "/{token}"])
    assert public.get("/api/phone/public/health").json()["routes"] == base
    monkeypatch.setenv("PHONE_ROUTES", "+919876543210:telnyx")
    routes = public.get("/api/phone/public/health").json()["routes"]
    assert routes == sorted(base + [f"/api/phone/telnyx/{w}/{{token}}" for w in ("stream", "status-callback", "answer")])


# --------------------------------------------------------------------------------------- a whole call, per provider

STREAM_ID = "STREAM1"


def ulaw_frames(pcm, frame_ms=100):
    step = int(8000 * 2 * frame_ms / 1000)
    return [pa.ulaw_encode(pcm[i:i + step]) for i in range(0, len(pcm), step)]


class Dialect:
    """One provider's wire format, as a fake carrier would speak it."""

    def __init__(self, name):
        self.name = name

    def start(self):
        if self.name == "twilio":
            return {"event": "start", "sequenceNumber": "1", "streamSid": STREAM_ID,
                    "start": {"accountSid": "AC1", "streamSid": STREAM_ID, "callSid": "CAtw", "tracks": ["inbound"],
                              "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                              "customParameters": {}}}
        if self.name == "plivo":
            return {"event": "start", "sequenceNumber": 1,
                    "start": {"callId": "uuid-call", "streamId": STREAM_ID, "accountId": "MA1", "tracks": ["inbound"],
                              "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000}}}
        return {"event": "start", "sequence_number": "1", "stream_id": STREAM_ID,
                "start": {"call_control_id": "cc-call", "call_session_id": "s", "from": "+919876543210",
                          "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1}}}

    def media(self, wire):
        payload = base64.b64encode(wire).decode()
        if self.name == "twilio":
            return {"event": "media", "streamSid": STREAM_ID, "media": {"track": "inbound", "payload": payload}}
        if self.name == "plivo":
            return {"event": "media", "streamId": STREAM_ID, "media": {"track": "inbound", "payload": payload}}
        return {"event": "media", "stream_id": STREAM_ID, "media": {"track": "inbound", "payload": payload}}

    def stop(self):
        return {"event": "stop"} if self.name != "plivo" else {"event": "stop", "streamId": STREAM_ID}

    # what the server sends
    @property
    def media_event(self):
        return "playAudio" if self.name == "plivo" else "media"

    @property
    def mark_event(self):
        return "checkpoint" if self.name == "plivo" else "mark"

    @property
    def clear_event(self):
        return "clearAudio" if self.name == "plivo" else "clear"

    def mark_name(self, m):
        return m["name"] if self.name == "plivo" else m["mark"]["name"]

    def pcm_of(self, m):
        return pa.ulaw_decode(base64.b64decode(m["media"]["payload"]))


@pytest.fixture(params=["twilio", "plivo", "telnyx"])
def dialect(request, clean, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", request.param)
    return Dialect(request.param)


def connect(rig, dialect, agent):
    token = px.make_token(agent["id"], "local-user", "local", 300, provider=dialect.name)
    return rig.client.websocket_connect(f"/api/phone/{dialect.name}/stream/{token}")


def send(ws, obj):
    ws.send_text(json.dumps(obj))


def feed(ws, dialect, pcm):
    for wire in ulaw_frames(pcm):
        send(ws, dialect.media(wire))


def test_the_agent_greets_in_the_providers_audio_format(rig, dialect):
    a = make_agent(rig.client)
    with connect(rig, dialect, a) as ws:
        send(ws, {"event": "connected", "protocol": "Call", "version": "1.0.0"})
        send(ws, dialect.start())
        msgs = read_until(ws, lambda m: m["event"] == dialect.mark_event)
        send(ws, dialect.stop())
    audio = [m for m in msgs if m["event"] == dialect.media_event]
    assert audio, "the agent said nothing"
    total = 0
    for m in audio:
        pcm = dialect.pcm_of(m)
        assert len(pcm) % 2 == 0 and len(pcm) > 0
        total += len(pcm)
        if dialect.name == "twilio":
            assert m["streamSid"] == STREAM_ID and set(m["media"]) == {"payload"}
        if dialect.name == "plivo":
            assert m["media"]["contentType"] == "audio/x-mulaw" and m["media"]["sampleRate"] == 8000
        if dialect.name == "telnyx":
            assert set(m) == {"event", "media"}
    assert total >= 3200                                                    # real audio, not a click
    last = msgs[-1]
    assert dialect.mark_name(last) == "turn-1-end"
    if dialect.name == "plivo":
        assert last["streamId"] == STREAM_ID
    assert any("AI" in t[0] or "assistant" in t[0].lower() for t in rig.spoken)       # the disclosure was spoken


def test_one_full_turn_over_the_providers_stream_reaches_the_model_and_comes_back_as_audio(rig, dialect):
    rig.asr_text = "When does the clinic open?"
    a = make_agent(rig.client)
    with connect(rig, dialect, a) as ws:
        send(ws, dialect.start())
        read_until(ws, lambda m: m["event"] == dialect.mark_event)                    # the greeting is done
        feed(ws, dialect, pcm_silence(300) + pcm_tone(1200) + pcm_silence(900))
        msgs = read_until(ws, lambda m: m["event"] == dialect.mark_event)
        send(ws, dialect.stop())
    assert any(m["event"] == dialect.media_event for m in msgs)
    assert dialect.mark_name(msgs[-1]) == "turn-2-end"
    asked = [q["messages"][-1]["content"] for q in rig.asked]
    assert any("When does the clinic open?" in c for c in asked)                   # the caller's mu-law words were understood
    rec = px.call_records.get({"twilio": "CAtw", "plivo": "uuid-call", "telnyx": "cc-call"}[dialect.name])
    assert rec["provider"] == dialect.name and rec["hangup_reason"] == "caller_hung_up"


@pytest.mark.parametrize("name", ["twilio", "plivo"])
def test_talking_over_the_agent_clears_the_providers_queue(rig, clean, monkeypatch, env, name):
    monkeypatch.setenv("PHONE_PROVIDERS", name)
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": np.full(24000 * 6, 0.05, dtype=np.float32))
    d = Dialect(name)
    a = make_agent(rig.client)
    with connect(rig, d, a) as ws:
        send(ws, d.start())
        read_until(ws, lambda m: m["event"] == d.media_event)                         # the agent is talking
        feed(ws, d, pcm_tone(800))                                                  # the caller talks over it
        msgs = read_until(ws, lambda m: m["event"] == d.clear_event)
        want = {"event": "clear", "streamSid": STREAM_ID} if name == "twilio" else {"event": "clearAudio", "streamId": STREAM_ID}
        assert msgs[-1] == want
        send(ws, d.stop())


def test_a_provider_dtmf_press_is_counted_but_never_stored(rig, dialect):
    a = make_agent(rig.client)
    with connect(rig, dialect, a) as ws:
        send(ws, dialect.start())
        read_until(ws, lambda m: m["event"] == dialect.mark_event)
        send(ws, {"event": "dtmf", "dtmf": {"digit": "0"}, "streamSid": STREAM_ID, "streamId": STREAM_ID})
        send(ws, {"event": "dtmf", "dtmf": {"digit": "7"}, "streamSid": STREAM_ID, "streamId": STREAM_ID})
        time.sleep(0.5)
        send(ws, dialect.stop())
    time.sleep(0.3)
    rec = px.call_records.get({"twilio": "CAtw", "plivo": "uuid-call", "telnyx": "cc-call"}[dialect.name])
    assert rec["dtmf_presses"] == 2 and rec["outcome"] == "human_requested"
    assert not any("digit" in k for k in rec) and "7" not in rec.values()   # only a count is kept


def test_an_outbound_call_is_recognised_by_its_provider_call_id(rig, clean, fake, monkeypatch):
    monkeypatch.setenv("PHONE_PROVIDERS", "twilio")
    monkeypatch.setenv("PHONE_ALLOWED_NUMBERS", "+919876543210")
    account(monkeypatch, "twilio")
    a = make_agent(rig.client)
    fake.replies = {"twilio.com": [(201, {"sid": "CAtw"})]}
    assert call(rig, a).json()["call_sid"] == "CAtw"
    d = Dialect("twilio")
    with connect(rig, d, a) as ws:
        send(ws, d.start())
        read_until(ws, lambda m: m["event"] == "mark")
        send(ws, d.stop())
    assert px.call_records.get("CAtw")["direction"] == "outbound"


# -------------------------------------------------------------------------------------------- hanging up through the API

@pytest.mark.asyncio
@pytest.mark.parametrize("reason,expected", [("ended_by_agent", True), ("voicemail", True), ("time_limit", True),
                                              ("caller_hung_up", False), (None, False)])
async def test_a_telnyx_call_is_hung_up_through_the_api_when_the_agent_ends_it(clean, fake, monkeypatch, reason, expected):
    monkeypatch.setenv("PHONE_PROVIDERS", "telnyx")
    account(monkeypatch, "telnyx", 1)
    account(monkeypatch, "telnyx", 2)
    s = px.CallSession.__new__(px.CallSession)
    s.provider_name, s.call_sid, s.end_reason, s.log = "telnyx", "cc-1", reason, []
    px._known_calls["cc-1"] = {"owner": "u", "agent_id": "a", "account": "2", "provider": "telnyx"}
    await s._api_hangup()
    if expected:
        assert fake.sent[0]["url"] == "https://api.telnyx.com/v2/calls/cc-1/actions/hangup"
        assert fake.sent[0]["headers"]["Authorization"] == "Bearer KEYtx2"
        assert s.log == [{"event": "api_hangup"}]
    else:
        assert fake.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["exotel", "twilio", "plivo"])
async def test_exotel_twilio_and_plivo_calls_end_when_the_stream_closes_so_no_api_call_is_made(clean, fake, monkeypatch, name):
    s = px.CallSession.__new__(px.CallSession)
    s.provider_name, s.call_sid, s.end_reason, s.log = name, "c-1", "ended_by_agent", []
    await s._api_hangup()
    assert fake.sent == []
