"""Website widget (routers/widget.py): owner settings, the origin allowlist, signed session tokens, rate limits, key
revocation, CORS, and a turn that goes through the same internal streaming pipeline with source="widget". The language
model, speech recognition and TTS are faked exactly like in test_turn_stream.py."""
import json
import re

import pytest

from routers import widget
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import REPLY, FakeStream, Rig

ORIGIN = "https://shop.example.com"
OTHER = "https://other.example.org"


@pytest.fixture()
def rig(app_client, env, monkeypatch, ref_wav):
    monkeypatch.setenv("WIDGET_TOKEN_SECRET", "test-secret")
    widget.limiter.reset()
    widget._TURN_OWNERS.clear()
    widget._VISITOR_CONV.clear()
    return Rig(env, monkeypatch, app_client, ref_wav)


def publish(client, agent_id, origins=(ORIGIN,), **extra):
    r = client.put(f"/api/agents/{agent_id}/widget", headers=LOCAL,
                   json={"enabled": True, "origins": list(origins), **extra})
    assert r.status_code == 200, r.text
    return r.json()


def session(client, key, origin=ORIGIN, **body):
    headers = {"Origin": origin} if origin else {}
    return client.post("/api/widget/session", headers=headers, json={"key": key, **body})


def turn(client, token, wav, origin=ORIGIN, bearer=False):
    headers = {"Origin": origin} if origin else {}
    data = {}
    if bearer:
        headers["Authorization"] = f"Bearer {token}"
    else:
        data["session"] = token
    return client.post("/api/widget/turn/stream", headers=headers, data=data,
                       files={"audio": ("recording.wav", wav, "audio/wav")})


def events(r):
    assert r.status_code == 200, r.text
    out = []
    for block in r.text.strip().split("\n\n"):
        head, data = block.split("\n")
        out.append((head[7:], json.loads(data[6:])))
    return out


# ------------------------------------------------------------------------------------------------ origins

@pytest.mark.parametrize("raw,expected", [
    ("https://Shop.Example.com", "https://shop.example.com"),
    ("https://shop.example.com/", "https://shop.example.com"),
    ("https://shop.example.com:443", "https://shop.example.com"),
    ("http://shop.example.com:8080", "http://shop.example.com:8080"),
    ("shop.example.com", "https://shop.example.com"),
    ("localhost:3000", "http://localhost:3000"),
    ("http://[::1]:3000", "http://[::1]:3000"),
])
def test_owner_origins_are_normalised(raw, expected):
    assert widget.normalize_origin(raw, lenient=True) == expected


@pytest.mark.parametrize("raw", ["", "null", "*", "https://*.example.com", "https://example.com/path",
                                 "https://example.com?x=1", "ftp://example.com", "https://user:pw@example.com",
                                 "javascript:alert(1)", "https://exa mple.com", "https://example.com:99999"])
def test_bad_origins_are_rejected(raw):
    assert widget.normalize_origin(raw, lenient=True) is None


def test_the_browsers_origin_header_is_parsed_strictly():
    assert widget.normalize_origin("example.com") is None  # a real Origin always has a scheme
    assert widget.normalize_origin("https://example.com") == "https://example.com"


# ------------------------------------------------------------------------------------------ owner settings

def test_publish_creates_a_key_and_saves_with_the_agent(rig):
    a = make_agent(rig.client)
    before = rig.client.get(f"/api/agents/{a['id']}/widget", headers=LOCAL).json()
    assert before["enabled"] is False and before["key"] is None
    assert before["per_visitor_hour"] == 30 and before["per_key_day"] == 300
    assert before["script_path"] == "/widget/v1/bonclay-agent.js"

    cfg = publish(rig.client, a["id"], origins=["Shop.Example.com", "https://shop.example.com/"],
                  business_name="Shop Ltd")
    assert cfg["enabled"] and re.fullmatch(r"awk_[A-Za-z0-9_-]{24}", cfg["key"])
    assert cfg["origins"] == [ORIGIN]  # normalised and de-duplicated
    assert cfg["business_name"] == "Shop Ltd"
    row = rig.env.local_db.get("agents", a["id"])
    assert row["widget_key"] == cfg["key"] and row["widget"]["origins"] == [ORIGIN]
    # The key never leaks through the ordinary agent API.
    assert cfg["key"] not in json.dumps(rig.client.get(f"/api/agents/{a['id']}", headers=LOCAL).json())


def test_publishing_needs_an_allowed_site_and_valid_values(rig):
    a = make_agent(rig.client)
    url = f"/api/agents/{a['id']}/widget"
    r = rig.client.put(url, headers=LOCAL, json={"enabled": True})
    assert r.status_code == 422 and r.json()["code"] == "ORIGINS_REQUIRED"
    r = rig.client.put(url, headers=LOCAL, json={"enabled": True, "origins": ["https://*.example.com"]})
    assert r.status_code == 422 and r.json()["code"] == "INVALID_ORIGIN"
    r = rig.client.put(url, headers=LOCAL, json={"per_visitor_hour": 0})
    assert r.status_code == 422
    r = rig.client.put(url, headers=LOCAL, json={"origins": [f"https://s{i}.example.com" for i in range(21)]})
    assert r.status_code == 422 and r.json()["code"] == "TOO_MANY_ORIGINS"
    assert rig.client.get(url, headers=LOCAL).json()["key"] is None  # nothing was half-saved


def test_unknown_agent_is_a_404(rig):
    assert rig.client.get("/api/agents/does-not-exist/widget", headers=LOCAL).status_code == 404
    assert rig.client.put("/api/agents/does-not-exist/widget", headers=LOCAL, json={"enabled": False}).status_code == 404


def test_owner_routes_need_the_owner_and_public_routes_never_take_an_agent_id():
    """Settings routes depend on the shared auth helper; the public routes have no agent or user parameter at all."""
    from services.auth import get_current_user_id
    for route in widget.router.routes:
        deps = {d.call for d in route.dependant.dependencies}
        if route.path.startswith("/api/agents/"):
            assert get_current_user_id in deps, route.path
        else:
            assert get_current_user_id not in deps, route.path
            names = {p.name for p in route.dependant.path_params + route.dependant.query_params}
            assert not names & {"agent_id", "user_id", "conversation_id"}, route.path


def test_rotate_and_delete_revoke_the_key(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    assert session(rig.client, cfg["key"]).status_code == 200
    r = rig.client.post(f"/api/agents/{a['id']}/widget/rotate", headers=LOCAL)
    new = r.json()["key"]
    assert new != cfg["key"]
    assert session(rig.client, cfg["key"]).status_code == 403
    assert session(rig.client, new).status_code == 200
    r = rig.client.delete(f"/api/agents/{a['id']}/widget", headers=LOCAL)
    assert r.json()["enabled"] is False and r.json()["key"] is None
    assert session(rig.client, new).status_code == 403
    assert rig.client.post(f"/api/agents/{a['id']}/widget/rotate", headers=LOCAL).status_code == 409


# -------------------------------------------------------------------------------------------- the session

def test_session_for_an_allowed_origin_returns_a_signed_bound_token(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"], business_name="Shop Ltd")
    r = session(rig.client, cfg["key"])
    assert r.status_code == 200
    body = r.json()
    assert body["expires_in"] == 600 and body["agent"]["business_name"] == "Shop Ltd"
    claims = widget.read_session(body["token"])
    assert claims["k"] == cfg["key"] and claims["a"] == a["id"] and claims["o"] == ORIGIN
    assert re.fullmatch(r"[0-9a-f]{24}", claims["v"]) and claims["exp"] - claims["iat"] == 600
    other = widget.read_session(session(rig.client, cfg["key"]).json()["token"])
    assert other["v"] != claims["v"]  # a random visitor id per visitor
    # An old token keeps the visitor (and so the conversation) when the widget renews its session.
    renewed = widget.read_session(session(rig.client, cfg["key"], previous=body["token"]).json()["token"])
    assert renewed["v"] == claims["v"]
    stolen = widget.read_session(session(rig.client, cfg["key"], previous=body["token"] + "x").json()["token"])
    assert stolen["v"] != claims["v"]


@pytest.mark.parametrize("origin", [OTHER, "http://shop.example.com", "https://evil.shop.example.com",
                                    "https://shop.example.com:8443", "http://localhost:3000", "null", None])
def test_every_other_origin_is_refused(rig, origin):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    r = session(rig.client, cfg["key"], origin=origin)
    assert r.status_code == 403 and r.json()["code"] == "WIDGET_FORBIDDEN"
    assert "access-control-allow-origin" not in r.headers
    assert "token" not in r.text


def test_localhost_works_only_when_the_owner_lists_it(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"], origins=["http://localhost:8080"])
    assert session(rig.client, cfg["key"], origin="http://localhost:8080").status_code == 200
    assert session(rig.client, cfg["key"], origin="http://localhost:3000").status_code == 403
    assert session(rig.client, cfg["key"], origin="http://127.0.0.1:8080").status_code == 403


def test_unknown_malformed_and_disabled_keys_look_the_same(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    unknown = session(rig.client, "awk_" + "x" * 24)
    malformed = session(rig.client, "not-a-key")
    rig.client.put(f"/api/agents/{a['id']}/widget", headers=LOCAL, json={"enabled": False})
    disabled = session(rig.client, cfg["key"])
    assert [r.status_code for r in (unknown, malformed, disabled)] == [403, 403, 403]
    assert unknown.json() == malformed.json() == disabled.json()
    assert unknown.json()["code"] == "WIDGET_FORBIDDEN"
    # Switching it back on restores the same key.
    again = rig.client.put(f"/api/agents/{a['id']}/widget", headers=LOCAL, json={"enabled": True}).json()
    assert again["key"] == cfg["key"] and session(rig.client, cfg["key"]).status_code == 200


# ------------------------------------------------------------------------------------------------- CORS

def test_cors_is_for_the_validated_origin_only(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"], origins=[ORIGIN, "http://localhost:3000"])
    ok = session(rig.client, cfg["key"])
    assert ok.headers["access-control-allow-origin"] == ORIGIN
    assert "Origin" in ok.headers["vary"]
    # The dashboard's own origin gets no special treatment on widget routes: only what the key lists.
    dash = session(rig.client, cfg["key"], origin="http://localhost:3000")
    assert dash.headers["access-control-allow-origin"] == "http://localhost:3000"
    denied = session(rig.client, cfg["key"], origin=OTHER)
    assert "access-control-allow-origin" not in denied.headers
    # ...and for a key that does not list the dashboard origin, the app-wide CORS list does not leak in.
    b = make_agent(rig.client)
    cfg2 = publish(rig.client, b["id"], origins=[ORIGIN])
    leaked = session(rig.client, cfg2["key"], origin="http://localhost:3000")
    assert leaked.status_code == 403 and "access-control-allow-origin" not in leaked.headers


def test_preflight_names_methods_and_headers_but_grants_nothing_by_itself(rig):
    r = rig.client.options("/api/widget/session", headers={
        "Origin": OTHER, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == OTHER
    assert "POST" in r.headers["access-control-allow-methods"]
    assert "authorization" in r.headers["access-control-allow-headers"].lower()
    # The real request from that origin is still refused and carries no allow header.
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    assert "access-control-allow-origin" not in session(rig.client, cfg["key"], origin=OTHER).headers


def test_other_routes_keep_the_app_wide_cors_rules(rig):
    r = rig.client.get("/api/agents", headers={**LOCAL, "Origin": OTHER})
    assert "access-control-allow-origin" not in r.headers


# -------------------------------------------------------------------------------------- token properties

def test_tampered_foreign_and_garbage_tokens_are_invalid(rig, monkeypatch):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    token = session(rig.client, cfg["key"]).json()["token"]
    payload, mac = token.split(".")
    forged_payload = widget._b64(json.dumps({**json.loads(widget._unb64(payload)), "a": "other-agent"}).encode())
    for bad in (f"{payload}.{'A' * len(mac)}", f"{forged_payload}.{mac}", token[:-2], "", "a.b.c", "xyz"):
        r = turn(rig.client, bad, rig.wav)
        assert r.status_code == 401 and r.json()["code"] == "WIDGET_SESSION_INVALID", bad
    # Signed with another secret = not ours.
    other = widget.sign_session({"k": cfg["key"], "a": a["id"], "m": "local", "o": ORIGIN, "v": "x", "exp": 9e12})
    assert turn(rig.client, other, rig.wav).status_code == 200
    monkeypatch.setenv("WIDGET_TOKEN_SECRET", "another-secret")
    assert turn(rig.client, other, rig.wav).status_code == 401


def test_token_expires_after_ten_minutes_and_the_widget_can_still_read_why(rig, monkeypatch):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    token = session(rig.client, cfg["key"]).json()["token"]
    real = widget._now()
    monkeypatch.setattr(widget, "_now", lambda: real + 599)
    assert turn(rig.client, token, rig.wav).status_code == 200
    monkeypatch.setattr(widget, "_now", lambda: real + 601)
    r = turn(rig.client, token, rig.wav)
    assert r.status_code == 401 and r.json()["code"] == "WIDGET_SESSION_EXPIRED"
    assert r.headers["access-control-allow-origin"] == ORIGIN
    assert "claims" not in r.text
    # A renewal still works within a day (it keeps the visitor) but not later.
    renewed = session(rig.client, cfg["key"], previous=token)
    assert widget.read_session(renewed.json()["token"])["v"] == widget.read_session(token, allow_expired_for=60)["v"]
    monkeypatch.setattr(widget, "_now", lambda: real + 601 + 25 * 3600)
    fresh = widget.read_session(session(rig.client, cfg["key"], previous=token).json()["token"])
    assert fresh["v"] != widget.read_session(token, allow_expired_for=9e9)["v"]


def test_token_is_bound_to_its_origin_and_its_key(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"], origins=[ORIGIN, OTHER])
    token = session(rig.client, cfg["key"]).json()["token"]
    # Same token presented from another (also allowed) site, or with no Origin at all: refused.
    r = turn(rig.client, token, rig.wav, origin=OTHER)
    assert r.status_code == 403 and "access-control-allow-origin" not in r.headers
    assert turn(rig.client, token, rig.wav, origin=None).status_code == 403
    # Removing the origin from the allowlist (or rotating the key) ends existing sessions at once.
    rig.client.put(f"/api/agents/{a['id']}/widget", headers=LOCAL, json={"origins": [OTHER]})
    assert turn(rig.client, token, rig.wav).status_code == 401
    rig.client.put(f"/api/agents/{a['id']}/widget", headers=LOCAL, json={"origins": [ORIGIN]})
    assert turn(rig.client, token, rig.wav).status_code == 200
    rig.client.post(f"/api/agents/{a['id']}/widget/rotate", headers=LOCAL)
    r = turn(rig.client, token, rig.wav)
    assert r.status_code == 401 and r.json()["code"] == "WIDGET_SESSION_INVALID"


def test_revoked_or_disabled_key_stops_an_open_session(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    token = session(rig.client, cfg["key"]).json()["token"]
    rig.client.put(f"/api/agents/{a['id']}/widget", headers=LOCAL, json={"enabled": False})
    assert turn(rig.client, token, rig.wav).status_code == 401
    rig.client.delete(f"/api/agents/{a['id']}/widget", headers=LOCAL)
    assert turn(rig.client, token, rig.wav).status_code == 401


def test_deleting_the_agent_ends_its_sessions(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    token = session(rig.client, cfg["key"]).json()["token"]
    assert rig.client.delete(f"/api/agents/{a['id']}", headers=LOCAL).status_code == 200
    assert turn(rig.client, token, rig.wav).status_code == 401
    assert session(rig.client, cfg["key"]).status_code == 403


# ---------------------------------------------------------------------------------------------- limits

def test_visitor_hourly_limit_returns_429_with_retry_after(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"], per_visitor_hour=2)
    token = session(rig.client, cfg["key"]).json()["token"]
    assert turn(rig.client, token, rig.wav).status_code == 200
    assert turn(rig.client, token, rig.wav).status_code == 200
    r = turn(rig.client, token, rig.wav)
    assert r.status_code == 429 and r.json()["code"] == "WIDGET_RATE_LIMITED"
    assert 0 < int(r.headers["retry-after"]) <= 3600
    assert r.headers["access-control-allow-origin"] == ORIGIN  # the widget can read the refusal
    assert len(rig.env.local_db.select("conversations")) == 1  # the refused turn did no work


def test_a_visitor_cannot_dodge_the_limit_with_fresh_sessions(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"], per_visitor_hour=2)
    for _ in range(2):
        assert turn(rig.client, session(rig.client, cfg["key"]).json()["token"], rig.wav).status_code == 200
    # a new session = a new visitor id, but the same IP is still counted
    assert turn(rig.client, session(rig.client, cfg["key"]).json()["token"], rig.wav).status_code == 429


def test_daily_limit_is_per_key(rig, monkeypatch):
    a, b = make_agent(rig.client), make_agent(rig.client)
    ka = publish(rig.client, a["id"], per_visitor_hour=100, per_key_day=2)["key"]
    kb = publish(rig.client, b["id"], per_visitor_hour=100, per_key_day=2)["key"]
    ta, tb = session(rig.client, ka).json()["token"], session(rig.client, kb).json()["token"]
    assert turn(rig.client, ta, rig.wav).status_code == 200
    assert turn(rig.client, ta, rig.wav).status_code == 200
    assert turn(rig.client, ta, rig.wav).status_code == 429
    assert turn(rig.client, tb, rig.wav).status_code == 200  # another business is unaffected
    # The window slides: a day later the key works again.
    real = widget._now()
    monkeypatch.setattr(widget, "_now", lambda: real + 86400 + 5)
    assert turn(rig.client, session(rig.client, ka).json()["token"], rig.wav).status_code == 200


def test_session_creation_is_limited_per_ip(rig, monkeypatch):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    monkeypatch.setattr(widget, "SESSIONS_PER_IP_HOUR", 3)
    monkeypatch.setattr(widget, "SESSIONS_PER_IP_ALL_KEYS_HOUR", 6)
    assert [session(rig.client, cfg["key"]).status_code for _ in range(4)] == [200, 200, 200, 429]
    # Guessing other keys from the same IP is throttled as a whole (the refused attempt above counted too).
    guesses = [session(rig.client, f"awk_{i:02d}" + "y" * 20).status_code for i in range(4)]
    assert guesses == [403, 403, 403, 429]


def test_oversized_uploads_are_refused_before_they_are_read(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    token = session(rig.client, cfg["key"]).json()["token"]
    r = turn(rig.client, token, b"RIFF" + b"\0" * (widget.MAX_UPLOAD_BYTES + 100_000))
    assert r.status_code == 413


# ----------------------------------------------------------------------------------------------- the turn

def test_turn_runs_the_shared_pipeline_as_the_owner_with_source_widget(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    token = session(rig.client, cfg["key"]).json()["token"]
    r = turn(rig.client, token, rig.wav)
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["access-control-allow-origin"] == ORIGIN
    ev = events(r)
    assert [n for n, _ in ev] == ["transcript", "sentence", "sentence", "sentence", "done"]  # same as the owner's route
    assert ev[0][1] == {"user_text": "When does the clinic open?", "language": "en"}
    assert [d["text"] for n, d in ev if n == "sentence"][0] == "This is an AI assistant."

    convs = rig.env.local_db.select("conversations")
    assert len(convs) == 1
    assert convs[0]["agent_id"] == a["id"] and convs[0]["source"] == "widget" and not convs[0].get("is_test")
    assert convs[0]["user_id"] == rig.env.local_db.get("agents", a["id"])["user_id"]  # owned by the agent's owner
    turns = rig.env.local_db.select("conversation_turns")
    assert [t["role"] for t in turns] == ["user", "agent"]


def test_owner_turns_are_not_marked_as_widget(rig):
    a = make_agent(rig.client)
    assert rig.post(a["id"]).status_code == 200
    assert "source" not in rig.env.local_db.select("conversations")[0]


def test_visitor_gets_only_what_the_page_needs(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    token = session(rig.client, cfg["key"]).json()["token"]
    ev = events(turn(rig.client, token, rig.wav))
    done = ev[-1][1]
    assert set(done) == {"turn_id", "user_text", "agent_text", "language", "fallback", "interrupted"}
    assert done["agent_text"].startswith("This is an AI assistant.")
    text = json.dumps(ev)
    for secret in ("receipt", "timings", "llm_provider", "fake-model", "conversation_id", "retrieval", "tool_calls",
                   "agent_audio_url", cfg["key"], token, a["id"]):
        assert secret not in text, secret


def test_clips_are_served_only_to_the_session_that_started_the_turn(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"], origins=[ORIGIN, OTHER])
    mine = session(rig.client, cfg["key"]).json()["token"]
    theirs = session(rig.client, cfg["key"], origin=OTHER).json()["token"]
    ev = events(turn(rig.client, mine, rig.wav))
    urls = [d["audio_url"] for n, d in ev if n == "sentence"]
    assert all(re.fullmatch(r"/api/widget/clip/[0-9a-f-]{36}/\d+", u) for u in urls)

    def get(url, token, origin=ORIGIN):
        return rig.client.get(url, headers={"Origin": origin, "Authorization": f"Bearer {token}"})

    ok = get(urls[0], mine)
    assert ok.status_code == 200 and ok.headers["content-type"] == "audio/wav" and ok.content[:4] == b"RIFF"
    assert ok.headers["access-control-allow-origin"] == ORIGIN
    assert get(urls[0], theirs, OTHER).status_code == 404       # another visitor's session
    assert rig.client.get(urls[0], headers={"Origin": ORIGIN}).status_code == 401   # no token
    assert get(urls[0], mine, OTHER).status_code == 403          # token used from the wrong site
    assert get(urls[0].rsplit("/", 1)[0] + "/99", mine).status_code == 404


def test_a_visitors_follow_up_continues_their_conversation_and_cannot_pick_one(rig):
    a = make_agent(rig.client)
    cfg = publish(rig.client, a["id"])
    mine = session(rig.client, cfg["key"]).json()["token"]
    other = session(rig.client, cfg["key"]).json()["token"]
    events(turn(rig.client, mine, rig.wav))
    events(turn(rig.client, mine, rig.wav))
    convs = rig.env.local_db.select("conversations")
    assert len(convs) == 1 and len(rig.env.local_db.select("conversation_turns")) == 4
    # Another visitor, even when naming the first visitor's conversation, gets a conversation of their own.
    r = rig.client.post("/api/widget/turn/stream", headers={"Origin": ORIGIN},
                        data={"session": other, "conversation_id": convs[0]["id"]},
                        files={"audio": ("r.wav", rig.wav, "audio/wav")})
    events(r)
    assert len(rig.env.local_db.select("conversations")) == 2
    assert len(rig.env.local_db.select("conversation_turns", lambda t: t["conversation_id"] == convs[0]["id"])) == 4


def test_a_session_only_reaches_its_own_agent(rig):
    a, b = make_agent(rig.client), make_agent(rig.client)
    ka = publish(rig.client, a["id"])["key"]
    publish(rig.client, b["id"])
    token = session(rig.client, ka).json()["token"]
    events(turn(rig.client, token, rig.wav))
    assert {c["agent_id"] for c in rig.env.local_db.select("conversations")} == {a["id"]}


def test_bearer_header_works_like_the_form_field(rig):
    a = make_agent(rig.client)
    token = session(rig.client, publish(rig.client, a["id"])["key"]).json()["token"]
    assert turn(rig.client, token, rig.wav, bearer=True).status_code == 200


def test_a_failing_turn_hides_internal_details(rig):
    async def broken(audio_bytes, name, language=None):
        return None  # speech recognition unavailable: the owner's message would tell how to fix the server

    a = make_agent(rig.client)
    token = session(rig.client, publish(rig.client, a["id"])["key"]).json()["token"]
    rig.env._speech_to_text = broken
    ev = events(turn(rig.client, token, rig.wav))
    assert ev[-1][0] == "error" and ev[-1][1]["code"] == "ASR_UNAVAILABLE"
    assert "GROQ" not in ev[-1][1]["message"] and "whisper" not in ev[-1][1]["message"].lower()


# ------------------------------------------------------------------------------------------------ script

def test_the_bundle_is_served_from_the_api_host(rig):
    r = rig.client.get("/widget/v1/bonclay-agent.js")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/javascript")
    assert "max-age" in r.headers["cache-control"] and r.headers["x-content-type-options"] == "nosniff"
    assert "customElements.define('bonclay-agent'" in r.text
    assert len(r.content) < 30 * 1024
    assert "https://" not in re.sub(r"https?://www\.w3\.org/\S+", "", r.text.replace("https://<", ""))  # no external hosts
