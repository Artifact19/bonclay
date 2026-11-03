"""Personal API keys (`aur_...`) are accepted on EVERY route: the ones in backend.py and the ones in the routers.

Regression: only backend.get_current_user knew about API keys, so a key that worked on /api/voices got a 401 on
/api/kokoro/generate, /api/rag/* and /api/analytics/summary, which authenticate through the shared auth service."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import backend  # noqa: E402
from core import apikeys  # noqa: E402

KOKORO = ("post", "/api/kokoro/generate", {"json": {"voice": "no-such-voice", "text": "hi"}})  # 400 once authenticated
ROUTER_ROUTES = [
    ("get", "/api/rag/documents", {}),
    ("get", "/api/analytics/summary", {}),
    KOKORO,
]


@pytest.fixture
def auth_env(monkeypatch):
    """Supabase auth configured, so there is no local-user shortcut to hide a missing credential behind."""
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "x" * 40)
    monkeypatch.delenv("BONCLAY_ALLOW_LOCAL_MODE", raising=False)
    monkeypatch.delenv("BONCLAY_ALLOW_DEMO_TOKENS", raising=False)


@pytest.fixture
def keys(tmp_path, monkeypatch):
    """Two users' keys in a throwaway store (never the real backend/data/api_keys.json)."""
    monkeypatch.setattr(apikeys, "_data_dir", str(tmp_path))
    monkeypatch.setattr(apikeys, "_get_supabase", lambda: None)
    monkeypatch.setattr(apikeys, "_verify_cache", {})
    a1, a2 = apikeys.create_key("user-a", "first"), apikeys.create_key("user-a", "second")
    b1 = apikeys.create_key("user-b", "only")
    return {"a": a1["key"], "a_id": a1["id"], "a2": a2["key"], "b": b1["key"]}


@pytest.fixture
def client():
    return TestClient(backend.app, raise_server_exceptions=False)


def call(client, route, key=None):
    method, path, kw = route
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return getattr(client, method)(path, headers=headers, **kw)


@pytest.mark.parametrize("route", ROUTER_ROUTES, ids=lambda r: r[1])
def test_a_valid_key_is_accepted_on_the_router_routes(auth_env, keys, client, route):
    r = call(client, route, keys["a"])
    assert r.status_code != 401, r.text  # authenticated: whatever the route then says about the request is its own
    if route is KOKORO:
        assert (r.status_code, r.json()["code"]) == (400, "UNKNOWN_VOICE")  # got past auth and into the handler


@pytest.mark.parametrize("route", ROUTER_ROUTES, ids=lambda r: r[1])
def test_the_router_routes_still_refuse_everyone_else(auth_env, keys, client, route):
    assert call(client, route).status_code == 401                                   # no credential
    r = call(client, route, "aur_deadbeef_notarealkey")                              # right shape, wrong secret
    assert (r.status_code, r.json()["code"]) == (401, "INVALID_API_KEY")


def test_a_revoked_key_stops_working_on_router_routes(auth_env, keys, client):
    assert call(client, ROUTER_ROUTES[0], keys["a"]).status_code == 200
    assert apikeys.delete_key("user-a", keys["a_id"]) is True
    assert call(client, ROUTER_ROUTES[0], keys["a"]).status_code == 401


def test_a_key_still_works_on_the_core_routes_and_each_key_acts_as_its_owner(auth_env, keys, client):
    def listed(key):
        r = client.get("/api/keys", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200, r.text
        return sorted(k["name"] for k in r.json()["keys"])

    assert listed(keys["a"]) == ["first", "second"]  # user A sees A's keys,
    assert listed(keys["a2"]) == ["first", "second"]  # whichever of A's keys is used,
    assert listed(keys["b"]) == ["only"]              # and user B sees only B's


def test_an_invalid_key_is_refused_even_when_auth_is_not_configured(keys, monkeypatch, client):
    """No Supabase configured = the local-user shortcut applies to anonymous calls, but a key that is presented
    must still be valid (this is what the core routes always did)."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("NEXT_PUBLIC_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert call(client, ROUTER_ROUTES[0]).status_code == 200                                  # anonymous local user
    assert call(client, ROUTER_ROUTES[0], keys["a"]).status_code == 200                       # a valid key
    assert call(client, ROUTER_ROUTES[0], "aur_deadbeef_notarealkey").status_code == 401      # a presented bad key
