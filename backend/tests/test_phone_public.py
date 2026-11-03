"""The phone-only public app (phone_public.py): only the stream and the status callback exist on it; everything the
tunnel used to expose answers 404. The main app keeps serving the same two routes on the loopback port."""
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import backend
import phone_public
from routers import phone_exotel as px


@pytest.fixture()
def public(monkeypatch):
    monkeypatch.setenv("WIDGET_TOKEN_SECRET", "test-secret")
    return TestClient(phone_public.public_app, raise_server_exceptions=False)


@pytest.mark.parametrize("path", [
    "/api/voices", "/api/agents", "/api/generate", "/api/keys", "/api/health", "/docs", "/openapi.json",
    "/api/phone/exotel/numbers", "/api/phone/exotel/calls", "/api/phone/exotel/status", "/api/phone/exotel/live",
])
def test_everything_but_the_phone_routes_is_absent(public, path):
    assert public.get(path).status_code == 404, path


def test_the_call_endpoint_is_absent_too(public):
    r = public.post("/api/phone/exotel/call", json={"agent_id": "a", "to": "+919876543210", "caller_agreed": True})
    assert r.status_code == 404


def test_health_probe_names_only_the_two_public_routes(public):
    r = public.get("/api/phone/public/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["routes"] == sorted([px.STREAM_PATH + "/{token}", px.STATUS_CALLBACK_PATH + "/{token}"])


def test_status_callback_exists_and_ignores_a_bad_token(public):
    r = public.post(px.STATUS_CALLBACK_PATH + "/not-a-token", data={"CallSid": "x", "Status": "completed"})
    assert r.status_code == 200 and r.json() == {"ok": False}


def test_stream_exists_and_refuses_a_bad_token(public):
    with pytest.raises(WebSocketDisconnect) as info:
        with public.websocket_connect(px.STREAM_PATH + "/not-a-token"):
            pass
    assert info.value.code == 4401


def test_main_app_still_serves_the_public_routes(app_client, monkeypatch):
    monkeypatch.setenv("WIDGET_TOKEN_SECRET", "test-secret")
    with pytest.raises(WebSocketDisconnect) as info:   # the route exists: it is the token check that refuses
        with app_client.websocket_connect(px.STREAM_PATH + "/not-a-token"):
            pass
    assert info.value.code == 4401
    r = app_client.post(px.STATUS_CALLBACK_PATH + "/not-a-token", data={"CallSid": "x"})
    assert r.status_code == 200 and r.json() == {"ok": False}
    # and the owner routes stay on the main app only (a request without a body is a 422, not a 404)
    assert app_client.post("/api/phone/exotel/call", headers={"storage-mode": "local"}).status_code == 422


def test_port_setting(monkeypatch):
    monkeypatch.delenv("PHONE_PUBLIC_PORT", raising=False)
    assert phone_public.port() == phone_public.DEFAULT_PORT
    monkeypatch.setenv("PHONE_PUBLIC_PORT", "0")
    assert phone_public.port() is None
    monkeypatch.setenv("PHONE_PUBLIC_PORT", "nope")
    assert phone_public.port() is None
    monkeypatch.setenv("PHONE_PUBLIC_PORT", "8011")
    assert phone_public.port() == 8011


@pytest.mark.asyncio
async def test_public_server_starts_on_a_free_port_and_stops(monkeypatch):
    import socket
    import httpx

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    monkeypatch.setenv("PHONE_PUBLIC_PORT", str(free))
    srv = phone_public.PublicServer()
    assert await srv.start() == free
    try:
        async with httpx.AsyncClient() as client:
            for _ in range(50):  # the server binds in the background; give it a moment
                try:
                    r = await client.get(f"http://127.0.0.1:{free}/api/phone/public/health")
                    break
                except httpx.ConnectError:
                    import asyncio
                    await asyncio.sleep(0.05)
            assert r.status_code == 200 and r.json()["ok"] is True
            assert (await client.get(f"http://127.0.0.1:{free}/api/voices")).status_code == 404
    finally:
        await srv.stop()
    assert not srv.running
