"""The busy branch: beyond PHONE_MAX_CALLS live calls, a new stream is refused before it is accepted (close code 4429),
so the Exotel flow can continue to a Connect applet that dials the owner's mobile. The status and inbound-link routes
tell the owner about the limit and the fallback."""
import pytest
from starlette.websockets import WebSocketDisconnect

from routers import phone_exotel as px
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_phone_exotel import rig, token_for  # noqa: F401  (rig is a fixture)


def test_a_call_beyond_the_limit_is_refused_before_accept(rig, monkeypatch):
    a = make_agent(rig.client)
    monkeypatch.setenv("PHONE_MAX_CALLS", "1")
    monkeypatch.setattr(px, "_active_calls", 1)
    with pytest.raises(WebSocketDisconnect) as e:
        with rig.client.websocket_connect(f"{px.STREAM_PATH}/{token_for(a)}"):
            pass
    assert e.value.code == px.BUSY_CLOSE_CODE == 4429
    assert px._active_calls == 1                       # a refused call never touches the counter


def test_the_default_limit_is_two_and_nonsense_falls_back_to_it(monkeypatch):
    monkeypatch.delenv("PHONE_MAX_CALLS", raising=False)
    assert px.max_calls() == 2
    monkeypatch.setenv("PHONE_MAX_CALLS", "0")
    assert px.max_calls() == 1
    monkeypatch.setenv("PHONE_MAX_CALLS", "many")
    assert px.max_calls() == 2


def test_status_reports_the_limit_the_live_count_and_the_fallback(rig):
    r = rig.client.get("/api/phone/exotel/status", headers=LOCAL).json()
    assert r["max_calls"] == 2 and r["active_calls"] == 0
    assert "Connect" in r["busy_fallback"] and "Voicebot" in r["busy_fallback"]


def test_the_inbound_link_explains_the_flow(rig, monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://phone.example.test")
    a = make_agent(rig.client)
    r = rig.client.post("/api/phone/exotel/inbound-link", headers=LOCAL, json={"agent_id": a["id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stream_url"].startswith("wss://phone.example.test" + px.STREAM_PATH)
    assert body["flow"]["applet"] == "Voicebot" and "Connect" in body["flow"]["then"] and body["flow"]["max_calls"] == 2
