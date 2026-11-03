"""Local mode answers only on this computer. A request from another address, or one that came through a tunnel or
proxy, needs BONCLAY_LOCAL_TOKEN (services/auth.py)."""
import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from services import auth


def _code(exc: HTTPException) -> str:
    return exc.detail["code"]


def _request(host, headers=None):
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {"type": "http", "method": "GET", "path": "/api/voices", "headers": raw, "query_string": b"",
             "scheme": "http", "server": ("127.0.0.1", 8000), "client": (host, 50000) if host else None}
    return Request(scope)


@pytest.fixture(autouse=True)
def no_supabase(monkeypatch):
    for name in ("NEXT_PUBLIC_SUPABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_JWT_SECRET",
                 "BONCLAY_LOCAL_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def resolve(request):
    return asyncio.run(auth._resolve_user_id(request))


def test_loopback_is_the_local_user():
    assert resolve(_request("127.0.0.1", {"storage-mode": "local"})) == auth.LOCAL_USER
    assert resolve(_request("::1")) == auth.LOCAL_USER
    assert resolve(_request("testclient")) == auth.LOCAL_USER


def test_another_address_is_refused_without_a_token():
    with pytest.raises(HTTPException) as info:
        resolve(_request("10.0.0.5", {"storage-mode": "local"}))
    assert info.value.status_code == 401
    assert _code(info.value) == "LOCAL_TOKEN_REQUIRED"


def test_a_tunnelled_request_counts_as_outside_even_from_loopback():
    for header in ("X-Forwarded-For", "Cf-Connecting-Ip", "X-Real-Ip"):
        with pytest.raises(HTTPException) as info:
            resolve(_request("127.0.0.1", {header: "203.0.113.9"}))
        assert _code(info.value) == "LOCAL_TOKEN_REQUIRED", header


def test_the_token_lets_another_device_in(monkeypatch):
    monkeypatch.setenv("BONCLAY_LOCAL_TOKEN", "s3cret-token")
    assert resolve(_request("10.0.0.5", {"Authorization": "Bearer s3cret-token"})) == auth.LOCAL_USER
    assert resolve(_request("10.0.0.5", {"X-BonClay-Token": "s3cret-token"})) == auth.LOCAL_USER
    with pytest.raises(HTTPException):
        resolve(_request("10.0.0.5", {"Authorization": "Bearer wrong"}))
    with pytest.raises(HTTPException):
        resolve(_request("10.0.0.5"))


def test_a_request_without_transport_is_trusted():
    assert auth.is_loopback(_request(None)) is True
