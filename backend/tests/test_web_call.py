"""Web call link (routers/web_call.py): owner routes (create / list / revoke / delete), the visitor routes on the
phone-only public app (page, session, turn, clip, end), the limits, and what a link can and cannot reach. The language
model, speech recognition and TTS are faked exactly like in test_widget.py / test_turn_stream.py."""
import json
import os
import re
import stat
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import phone_public
from routers import web_call, widget
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import REPLY, Rig

ORIGIN = "http://testserver"           # the TestClient's own host: the page's own origin
BASE = "https://phone.example.test"    # PUBLIC_BASE_URL (the tunnel)


@pytest.fixture()
def rig(app_client, env, monkeypatch, ref_wav):
    monkeypatch.setenv("WIDGET_TOKEN_SECRET", "test-secret")
    monkeypatch.setenv("PUBLIC_BASE_URL", BASE)
    for obj in (widget.limiter, web_call.limiter):
        obj.reset()
    widget._TURN_OWNERS.clear()
    widget._VISITOR_CONV.clear()
    web_call._LAST_SEEN.clear()
    web_call._VISITOR_USED.clear()
    r = Rig(env, monkeypatch, app_client, ref_wav)
    r.public = TestClient(phone_public.public_app, raise_server_exceptions=False)
    return r


def make_link(rig, agent_id, **body):
    r = rig.client.post(f"/api/agents/{agent_id}/web-call-links", headers=LOCAL, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def start(rig, link, origin=ORIGIN, **headers):
    h = {"Origin": origin, **headers} if origin else dict(headers)
    return rig.public.post(f"/call/api/{link['id']}/session", headers=h, json={})


def token_for(rig, link, **headers):
    r = start(rig, link, **headers)
    assert r.status_code == 200, r.text
    return r.json()["token"]


def say(rig, link_id, token, origin=ORIGIN, wav=None, **headers):
    h = {"Authorization": f"Bearer {token}", **headers}
    if origin:
        h["Origin"] = origin
    return rig.public.post(f"/call/api/{link_id}/turn", headers=h,
                           files={"audio": ("recording.wav", wav or rig.wav, "audio/wav")})


def events(r):
    assert r.status_code == 200, r.text
    out = []
    for block in r.text.strip().split("\n\n"):
        head, data = block.split("\n")
        out.append((head[7:], json.loads(data[6:])))
    return out


# --------------------------------------------------------------------------------------------- owner routes

def test_create_list_with_defaults_and_a_private_store(rig, env):
    a = make_agent(rig.client)
    url = f"/api/agents/{a['id']}/web-call-links"
    empty = rig.client.get(url, headers=LOCAL).json()
    assert empty["links"] == [] and empty["base_url"] == BASE and empty["public"] is True
    assert empty["defaults"] == {"expires_in": "7d", "max_session_minutes": 10, "max_minutes_per_day": 60}

    link = make_link(rig, a["id"], label="  Dr. Mehta demo ")
    assert re.fullmatch(r"awc_[A-Za-z0-9_-]{32}", link["id"])            # 192 random bits
    assert link["url"] == f"{BASE}/call/{link['id']}" and link["label"] == "Dr. Mehta demo"
    assert link["status"] == "active" and link["sessions"] == 0 and link["minutes_today"] == 0
    assert link["max_session_minutes"] == 10 and link["max_minutes_per_day"] == 60
    assert link["max_minutes_per_visitor"] is None
    assert link["expires_at"] is not None and link["revoked_at"] is None   # 7 days by default

    listed = rig.client.get(url, headers=LOCAL).json()["links"]
    assert [x["id"] for x in listed] == [link["id"]]
    other = make_link(rig, a["id"], expires_in="never", max_session_minutes=3, max_minutes_per_day=5,
                      max_minutes_per_visitor=2)
    assert other["expires_at"] is None and other["max_minutes_per_visitor"] == 2
    assert [x["id"] for x in rig.client.get(url, headers=LOCAL).json()["links"]] == [other["id"], link["id"]]

    path = web_call.store.path()                                   # next to the local database, owner-only
    assert os.path.dirname(path) == os.path.dirname(env.local_db.root)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    stored = json.load(open(path))["links"][link["id"]]
    assert set(stored) == {"agent_id", "user_id", "mode", "label", "created", "expires", "revoked",
                           "max_session_minutes", "max_minutes_per_day", "max_minutes_per_visitor", "sessions",
                           "turns", "minutes_total", "day", "minutes_today", "last_used"}


def test_expiry_choices_and_validation(rig, monkeypatch):
    a = make_agent(rig.client)
    now = 1_800_000_000.0
    monkeypatch.setattr(widget, "_now", lambda: now)
    day = make_link(rig, a["id"], expires_in="1d")
    week = make_link(rig, a["id"], expires_in="7d")
    assert day["expires_at"].startswith("2027-01-16") and week["expires_at"].startswith("2027-01-22")
    url = f"/api/agents/{a['id']}/web-call-links"
    for bad in ({"expires_in": "forever"}, {"max_session_minutes": 0}, {"max_session_minutes": 31},
                {"max_minutes_per_day": 0}, {"max_minutes_per_day": 5000}, {"max_minutes_per_visitor": 0},
                {"label": "x" * 81}):
        assert rig.client.post(url, headers=LOCAL, json=bad).status_code == 422, bad


def test_at_most_twenty_active_links_per_agent(rig):
    a = make_agent(rig.client)
    url = f"/api/agents/{a['id']}/web-call-links"
    first = [make_link(rig, a["id"]) for _ in range(20)]
    r = rig.client.post(url, headers=LOCAL, json={})
    assert r.status_code == 409 and r.json()["code"] == "TOO_MANY_LINKS"
    rig.client.post(f"{url}/{first[0]['id']}/revoke", headers=LOCAL)
    assert rig.client.post(url, headers=LOCAL, json={}).status_code == 200


def test_unknown_agent_and_other_agents_links_are_404(rig):
    a, b = make_agent(rig.client), make_agent(rig.client)
    assert rig.client.get("/api/agents/nope/web-call-links", headers=LOCAL).status_code == 404
    assert rig.client.post("/api/agents/nope/web-call-links", headers=LOCAL, json={}).status_code == 404
    link = make_link(rig, a["id"])
    wrong = f"/api/agents/{b['id']}/web-call-links/{link['id']}"
    assert rig.client.post(wrong + "/revoke", headers=LOCAL).status_code == 404
    assert rig.client.delete(wrong, headers=LOCAL).status_code == 404
    assert rig.client.get(f"/api/agents/{b['id']}/web-call-links", headers=LOCAL).json()["links"] == []
    assert rig.client.post(f"/api/agents/{a['id']}/web-call-links/awc_nope/revoke", headers=LOCAL).status_code == 404
    assert rig.client.get(link["url"].replace(BASE, ""), headers=LOCAL).status_code == 404   # not on the main app


def test_revoke_keeps_the_record_and_delete_removes_it(rig):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    url = f"/api/agents/{a['id']}/web-call-links"
    r = rig.client.post(f"{url}/{link['id']}/revoke", headers=LOCAL)
    assert r.status_code == 200 and r.json()["status"] == "revoked" and r.json()["revoked_at"]
    assert rig.client.get(url, headers=LOCAL).json()["links"][0]["status"] == "revoked"
    assert rig.client.delete(f"{url}/{link['id']}", headers=LOCAL).json() == {"ok": True}
    assert rig.client.get(url, headers=LOCAL).json()["links"] == []
    assert rig.client.delete(f"{url}/{link['id']}", headers=LOCAL).status_code == 404


def test_base_url_falls_back_to_the_local_public_port(rig, monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL")
    monkeypatch.setenv("PHONE_PUBLIC_PORT", "8010")
    a = make_agent(rig.client)
    body = rig.client.get(f"/api/agents/{a['id']}/web-call-links", headers=LOCAL).json()
    assert body["base_url"] == "http://localhost:8010" and body["public"] is False


def test_owner_routes_are_authenticated_and_public_routes_take_no_agent_or_user():
    from services.auth import get_current_user_id
    for route in web_call.router.routes:
        assert get_current_user_id in {d.call for d in route.dependant.dependencies}, route.path
        assert route.path.startswith("/api/agents/{agent_id}/web-call-links")
    for route in web_call.public_router.routes:
        assert route.path.startswith("/call/"), route.path
        assert get_current_user_id not in {d.call for d in route.dependant.dependencies}, route.path
        names = {p.name for p in route.dependant.path_params + route.dependant.query_params}
        assert not names & {"agent_id", "user_id", "conversation_id"}, route.path


# ------------------------------------------------------------------------------------------------ the page

def test_the_page_is_self_contained_branded_and_safe(rig, env):
    a = make_agent(rig.client)
    env.local_db.update("agents", a["id"], {"name": '<img src=x onerror=alert(1)> & "Co"'}, "local-user")
    link = make_link(rig, a["id"], max_session_minutes=7)
    r = rig.public.get(f"/call/{link['id']}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    page = r.text
    assert "&lt;img src=x onerror=alert(1)&gt; &amp; &quot;Co&quot;" in page and "<img src=x" not in page
    for needle in ("BonClay", "AI voice agent", "You are talking to an AI, not a person", "Start talking",
                   "End call", "Mute microphone", 'role="log"', 'aria-live="polite"', "up to 7 minutes",
                   "Voice is AI-generated, watermarked and signed; check a clip at",
                   "bonclay.ayushchougula.in/verify"):
        assert needle in page, needle
    assert "{{" not in page
    # no inline script or style: everything is a same-origin file
    assert not re.search(r"<script(?![^>]*\bsrc=)", page) and " style=" not in page and "onclick" not in page
    assert re.findall(r'(?:src|href)="([^"]+)"', page) == [
        "data:,", "/call/assets/call.css", "https://bonclay.ayushchougula.in/verify", "/call/assets/call.js"]
    h = r.headers
    assert h["cache-control"] == "no-store" and h["x-content-type-options"] == "nosniff"
    assert h["referrer-policy"] == "no-referrer" and h["x-frame-options"] == "DENY"
    assert "microphone=(self)" in h["permissions-policy"]
    csp = h["content-security-policy"]
    for part in ("default-src 'none'", "script-src 'self'", "style-src 'self'", "connect-src 'self'",
                 "frame-ancestors 'none'", "base-uri 'none'"):
        assert part in csp
    assert "unsafe-inline" not in csp and "http:" not in csp and "*" not in csp
    assert not any(k.lower().startswith("access-control-") for k in h)


def test_the_page_leaks_no_secret_id_or_number(rig, env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_SECRETSECRETSECRET")
    a = make_agent(rig.client, system_instructions="Call the owner on +919876543210 with key sk-abcdef")
    link = make_link(rig, a["id"], label="private label")
    page = rig.public.get(f"/call/{link['id']}").text
    css = rig.public.get("/call/assets/call.css").text
    js = rig.public.get("/call/assets/call.js").text
    for text in (page, css, js):
        for needle in (a["id"], "local-user", "local_project", BASE, "private label", "9876543210", "gsk_", "sk-",
                       "hf_", "BEGIN PRIVATE KEY", "SECRETSECRET", "test-secret"):
            assert needle not in text, needle
        assert not re.search(r"\+?\d{10,}", text)
    assert link["id"] in page   # the visitor holds the link already; it is the only identifier on the page
    assert "/api/" not in js.replace("/call/api/", "")   # the script talks to /call/api/... and nothing else


def test_assets_are_served_with_the_right_types(rig):
    js = rig.public.get("/call/assets/call.js")
    css = rig.public.get("/call/assets/call.css")
    assert js.status_code == css.status_code == 200
    assert js.headers["content-type"].startswith("application/javascript")
    assert css.headers["content-type"].startswith("text/css")
    assert js.headers["cache-control"] == "public, max-age=300" and "default-src 'none'" in js.headers["content-security-policy"]
    assert rig.public.get("/call/assets/other.js").status_code == 404
    assert rig.public.get("/call/assets/../backend.py").status_code == 404


def test_unknown_malformed_revoked_expired_and_orphaned_links_refuse(rig, env, monkeypatch):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"], expires_in="1d")
    for bad in ("awc_" + "x" * 32, "nope", "awk_" + "x" * 32, "awc_short"):
        r = rig.public.get(f"/call/{bad}")
        assert r.status_code == 404 and "Check the link" in r.text, bad
        assert start(rig, {"id": bad}).status_code == 404
    assert rig.public.get(f"/call/{link['id']}").status_code == 200

    real = widget._now()
    monkeypatch.setattr(widget, "_now", lambda: real + 86400 + 5)
    r = rig.public.get(f"/call/{link['id']}")
    assert r.status_code == 410 and "no longer active" in r.text and "Start talking" not in r.text
    assert start(rig, link).status_code == 410
    monkeypatch.setattr(widget, "_now", lambda: real)

    rig.client.post(f"/api/agents/{a['id']}/web-call-links/{link['id']}/revoke", headers=LOCAL)
    r = rig.public.get(f"/call/{link['id']}")
    assert r.status_code == 410 and "Start talking" not in r.text
    r = start(rig, link)
    assert r.status_code == 410 and r.json()["code"] == "CALL_LINK_INACTIVE" and "token" not in r.text

    b = make_agent(rig.client)
    other = make_link(rig, b["id"])
    assert rig.public.get(f"/call/{other['id']}").status_code == 200
    rig.client.delete(f"/api/agents/{b['id']}", headers=LOCAL)         # the agent is gone: the link is dead
    assert rig.public.get(f"/call/{other['id']}").status_code == 410
    assert start(rig, other).status_code == 410


# ---------------------------------------------------------------------------------------------- the session

def test_session_token_is_signed_bound_and_cut_down(rig):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"], max_session_minutes=3)
    r = start(rig, link)
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()
    assert set(body) == {"token", "expires_in", "max_session_s", "agent"} and set(body["agent"]) == {"name"}
    assert body["expires_in"] == body["max_session_s"] == 180
    claims = widget.read_session(body["token"])
    assert claims["k"] == link["id"] and claims["a"] == a["id"] and claims["o"] == ORIGIN and claims["kind"] == "call"
    assert claims["exp"] - claims["iat"] == 180 and re.fullmatch(r"[0-9a-f]{24}", claims["v"])
    assert widget.read_session(token_for(rig, link))["v"] != claims["v"]
    for secret in (a["id"], "local-user"):
        assert secret not in r.text
    assert not any(k.lower().startswith("access-control-") for k in r.headers)
    assert rig.client.get("/api/agents/" + a["id"] + "/web-call-links", headers=LOCAL).json()["links"][0]["sessions"] == 2


@pytest.mark.parametrize("origin", ["https://evil.example.com", "http://testserver:8000", "http://testserver.evil.test",
                                    "null", None])
def test_only_the_pages_own_origin_may_start_a_call(rig, origin):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    r = start(rig, link, origin=origin)
    assert r.status_code == 403 and r.json()["code"] == "CALL_FORBIDDEN" and "token" not in r.text
    assert not any(k.lower().startswith("access-control-") for k in r.headers)
    # and a preflight from another site grants nothing
    pre = rig.public.options(f"/call/api/{link['id']}/session", headers={
        "Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in pre.headers


def test_the_tunnel_host_counts_as_the_pages_own_origin(rig, monkeypatch):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    # the tunnel passes a different Host (or keeps it): PUBLIC_BASE_URL and X-Forwarded-Host both name the page's host
    assert start(rig, link, origin=BASE).status_code == 200
    assert start(rig, link, origin="https://phone.other.test", **{"X-Forwarded-Host": "phone.other.test"}).status_code == 200
    monkeypatch.delenv("PUBLIC_BASE_URL")
    assert start(rig, link, origin=BASE).status_code == 403
    assert start(rig, link, origin="https://evil.example.com", **{"X-Forwarded-Host": "phone.other.test"}).status_code == 403


def test_client_ip_trusts_the_tunnel_headers_only_from_this_computer():
    def req(peer, **headers):
        return SimpleNamespace(client=SimpleNamespace(host=peer), headers={k.lower(): v for k, v in headers.items()})

    assert web_call.client_ip(req("127.0.0.1", **{"CF-Connecting-IP": "203.0.113.9"})) == "203.0.113.9"
    assert web_call.client_ip(req("127.0.0.1", **{"X-Forwarded-For": "1.2.3.4, 198.51.100.7"})) == "198.51.100.7"
    assert web_call.client_ip(req("127.0.0.1", **{"CF-Connecting-IP": "not an ip"})) == "127.0.0.1"
    assert web_call.client_ip(req("::1")) == "::1"
    assert web_call.client_ip(req("192.0.2.1", **{"CF-Connecting-IP": "203.0.113.9"})) == "192.0.2.1"   # LAN: no trust


# ------------------------------------------------------------------------------------------------ the turn

def test_a_turn_streams_trimmed_events_clips_and_counts(rig):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    token = token_for(rig, link)
    r = say(rig, link["id"], token)
    ev = events(r)
    assert [e for e, _ in ev] == ["transcript", "sentence", "sentence", "sentence", "done"]
    assert ev[0][1] == {"user_text": "When does the clinic open?", "language": "en"}
    done = ev[-1][1]
    assert set(done) == {"turn_id", "user_text", "agent_text", "language", "fallback", "interrupted"}
    sentences = [d for e, d in ev if e == "sentence"]
    assert sentences[1]["text"] == "The clinic opens at 9 a.m. on Monday."
    for s in sentences:
        assert s["audio_url"] == f"/call/api/{link['id']}/clip/{done['turn_id']}/{s['index']}"
    assert r.headers["cache-control"] == "no-store" and "content-security-policy" in r.headers
    for secret in (a["id"], "local-user", "receipt", "model", "timings"):
        assert secret not in r.text
    # the conversation is stored for the owner, marked as a website visit
    conv = rig.env.local_db.select("conversations")[0]
    assert conv["agent_id"] == a["id"] and conv.get("source") == "widget"
    # a clip: only with the session's own token, only for this link
    url = sentences[1]["audio_url"]
    ok = rig.public.get(url, headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200 and ok.headers["content-type"] == "audio/wav" and ok.content[:4] == b"RIFF"
    assert rig.public.get(url).status_code == 401
    assert rig.public.get(url, headers={"Authorization": f"Bearer {token_for(rig, link)}"}).status_code == 404
    assert rig.public.get(f"/call/api/{link['id']}/clip/{done['turn_id']}/9999",
                          headers={"Authorization": f"Bearer {token}"}).status_code == 404
    shown = rig.client.get(f"/api/agents/{a['id']}/web-call-links", headers=LOCAL).json()["links"][0]
    assert shown["sessions"] == 2 and shown["turns"] == 1 and shown["minutes_today"] > 0
    assert shown["last_used_at"]
    # the second turn carries on the visitor's conversation
    events(say(rig, link["id"], token))
    assert len(rig.env.local_db.select("conversations")) == 1


def test_a_turn_needs_the_pages_origin_and_a_valid_token(rig, monkeypatch):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    token = token_for(rig, link)
    r = say(rig, link["id"], token, origin="https://evil.example.com")
    assert r.status_code == 403 and r.json()["code"] == "CALL_FORBIDDEN"
    assert say(rig, link["id"], token, origin=None).status_code == 403
    payload, mac = token.split(".")
    forged = widget._b64(json.dumps({**json.loads(widget._unb64(payload)), "a": "other-agent"}).encode())
    for bad in (f"{payload}.{'A' * len(mac)}", f"{forged}.{mac}", token[:-2], "", "a.b.c"):
        r = say(rig, link["id"], bad)
        assert r.status_code == 401 and r.json()["code"] == "CALL_SESSION_INVALID", bad
    assert rig.public.post(f"/call/api/{link['id']}/turn", headers={"Origin": ORIGIN},
                           files={"audio": ("r.wav", rig.wav, "audio/wav")}).status_code == 401
    monkeypatch.setenv("WIDGET_TOKEN_SECRET", "another-secret")
    assert say(rig, link["id"], token).status_code == 401


def test_a_link_token_works_for_that_link_only(rig):
    a, b = make_agent(rig.client), make_agent(rig.client)
    la, lb = make_link(rig, a["id"]), make_link(rig, b["id"])
    ta, tb = token_for(rig, la), token_for(rig, lb)
    r = say(rig, lb["id"], ta)                               # A's token on B's link
    assert r.status_code == 401 and r.json()["code"] == "CALL_SESSION_INVALID"
    assert say(rig, la["id"], tb).status_code == 401
    assert say(rig, la["id"], ta).status_code == 200 and say(rig, lb["id"], tb).status_code == 200
    spoken = [c["agent_id"] for c in rig.env.local_db.select("conversations")]
    assert sorted(spoken) == sorted([a["id"], b["id"]])
    # a token cannot be re-pointed at another agent either: the agent is part of the signed claims and of the link
    payload, mac = ta.split(".")
    swapped = widget._b64(json.dumps({**json.loads(widget._unb64(payload)), "k": lb["id"]}).encode()) + "." + mac
    assert say(rig, lb["id"], swapped).status_code == 401


def test_widget_tokens_and_call_tokens_are_not_interchangeable(rig):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    call_token = token_for(rig, link)
    key = rig.client.put(f"/api/agents/{a['id']}/widget", headers=LOCAL,
                         json={"enabled": True, "origins": [ORIGIN]}).json()["key"]
    wtoken = widget.sign_session({"k": key, "a": a["id"], "m": "local", "o": ORIGIN, "v": "v1", "exp": 9e12})
    assert say(rig, link["id"], wtoken).status_code == 401          # a widget session on a call link
    # a call session on the widget's turn route (main app): the key claim is the link id, never a widget key
    r = rig.client.post("/api/widget/turn/stream", headers={"Origin": ORIGIN, "Authorization": f"Bearer {call_token}"},
                        files={"audio": ("r.wav", rig.wav, "audio/wav")})
    assert r.status_code == 401
    # a widget token forged with a link id as its key still has no "call" kind
    forged = widget.sign_session({"k": link["id"], "a": a["id"], "m": "local", "o": ORIGIN, "v": "v1", "exp": 9e12})
    assert say(rig, link["id"], forged).status_code == 401


def test_revoking_or_expiring_stops_a_call_in_progress(rig, monkeypatch):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"], expires_in="1d")
    token = token_for(rig, link)
    assert say(rig, link["id"], token).status_code == 200
    real = widget._now()
    monkeypatch.setattr(widget, "_now", lambda: real + 86400 + 1)
    r = say(rig, link["id"], token)
    assert r.status_code == 401                                                  # the token itself ran out first
    monkeypatch.setattr(widget, "_now", lambda: real)
    rig.client.post(f"/api/agents/{a['id']}/web-call-links/{link['id']}/revoke", headers=LOCAL)
    r = say(rig, link["id"], token)
    assert r.status_code == 410 and r.json()["code"] == "CALL_LINK_INACTIVE"
    assert rig.public.get(f"/call/api/{link['id']}/clip/{'0' * 8}-0000-0000-0000-{'0' * 12}/0",
                          headers={"Authorization": f"Bearer {token}"}).status_code == 410
    assert len(rig.env.local_db.select("conversations")) == 1                     # nothing ran after the revoke


def test_session_length_is_capped_in_the_token(rig, monkeypatch):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"], max_session_minutes=2)
    token = token_for(rig, link)
    real = widget._now()
    monkeypatch.setattr(widget, "_now", lambda: real + 119)
    assert say(rig, link["id"], token).status_code == 200
    monkeypatch.setattr(widget, "_now", lambda: real + 121)
    r = say(rig, link["id"], token)
    assert r.status_code == 401 and r.json()["code"] == "CALL_SESSION_EXPIRED"
    assert "time limit" in r.json()["error"] and "claims" not in r.text


# ---------------------------------------------------------------------------------------------- the limits

def test_daily_minutes_for_the_link_run_out(rig, monkeypatch):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"], max_minutes_per_day=1)
    token = token_for(rig, link)
    real = widget._now()
    monkeypatch.setattr(widget, "_now", lambda: real + 60)       # 60 s of talking since the session began
    assert say(rig, link["id"], token).status_code == 200
    r = say(rig, link["id"], token)
    assert r.status_code == 429 and r.json()["code"] == "CALL_DAY_LIMIT" and "tomorrow" in r.json()["error"]
    r = start(rig, link)                                          # nobody new can start either
    assert r.status_code == 429 and r.json()["code"] == "CALL_DAY_LIMIT"
    shown = rig.client.get(f"/api/agents/{a['id']}/web-call-links", headers=LOCAL).json()["links"][0]
    assert shown["minutes_today"] == 1.0 and shown["turns"] == 1
    monkeypatch.setattr(widget, "_now", lambda: real + 86400 + 60)   # the next day starts afresh
    assert start(rig, link).status_code == 200


def test_every_turn_costs_at_least_a_few_seconds(rig):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"], max_minutes_per_day=1)         # 60 s = twelve 5 s turns
    token = token_for(rig, link)
    codes = [say(rig, link["id"], token).status_code for _ in range(14)]
    assert codes[:12] == [200] * 12 and codes[12:] == [429, 429]


def test_minutes_per_visitor_are_counted_per_ip(rig, monkeypatch):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"], max_minutes_per_visitor=1)
    mine = {"CF-Connecting-IP": "203.0.113.5"}
    token = token_for(rig, link, **mine)
    real = widget._now()
    monkeypatch.setattr(widget, "_now", lambda: real + 60)
    assert say(rig, link["id"], token, **mine).status_code == 200
    r = say(rig, link["id"], token, **mine)
    assert r.status_code == 429 and r.json()["code"] == "CALL_VISITOR_LIMIT"
    assert start(rig, link, **mine).status_code == 429
    theirs = {"CF-Connecting-IP": "203.0.113.6"}                  # somebody else is not affected
    assert say(rig, link["id"], token_for(rig, link, **theirs), **theirs).status_code == 200


def test_end_settles_the_minutes_once(rig, monkeypatch):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    token = token_for(rig, link)
    real = widget._now()
    monkeypatch.setattr(widget, "_now", lambda: real + 30)
    h = {"Authorization": f"Bearer {token}", "Origin": ORIGIN}
    assert rig.public.post(f"/call/api/{link['id']}/end", headers=h).json() == {"ok": True}
    assert rig.public.post(f"/call/api/{link['id']}/end", headers=h).json() == {"ok": True}
    shown = rig.client.get(f"/api/agents/{a['id']}/web-call-links", headers=LOCAL).json()["links"][0]
    assert shown["minutes_today"] == 0.5 and shown["turns"] == 0
    assert rig.public.post(f"/call/api/{link['id']}/end", headers={"Origin": ORIGIN}).status_code == 401


def test_session_starts_are_limited_per_ip_and_per_link(rig):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    codes = [start(rig, link).status_code for _ in range(12)]
    assert codes[:10] == [200] * 10 and codes[10:] == [429, 429]
    r = start(rig, link)
    assert r.json()["code"] == "CALL_RATE_LIMITED" and 0 < int(r.headers["retry-after"]) <= 3600
    assert start(rig, link, **{"CF-Connecting-IP": "203.0.113.77"}).status_code == 200    # another visitor is fine


def test_turns_are_limited_per_visitor_and_a_refused_turn_is_not_charged(rig):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"], max_minutes_per_day=1000)
    token = token_for(rig, link)
    codes = [say(rig, link["id"], token).status_code for _ in range(22)]
    assert codes[:20] == [200] * 20 and codes[20:] == [429, 429]
    r = say(rig, link["id"], token)
    assert r.json()["code"] == "CALL_RATE_LIMITED" and int(r.headers["retry-after"]) >= 1
    shown = rig.client.get(f"/api/agents/{a['id']}/web-call-links", headers=LOCAL).json()["links"][0]
    assert shown["turns"] == 20


def test_page_loads_are_rate_limited_per_ip(rig):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    codes = [rig.public.get(f"/call/{link['id']}").status_code for _ in range(92)]
    assert codes[:90] == [200] * 90 and codes[90:] == [429, 429]
    assert rig.public.get(f"/call/{link['id']}", headers={"CF-Connecting-IP": "203.0.113.50"}).status_code == 200


def test_oversized_uploads_are_refused_before_they_are_read(rig):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    token = token_for(rig, link)
    r = rig.public.post(f"/call/api/{link['id']}/turn", headers={
        "Authorization": f"Bearer {token}", "Origin": ORIGIN, "Content-Length": str(20 * 1024 * 1024)}, content=b"x")
    assert r.status_code == 413 and r.json()["code"] == "FILE_TOO_LARGE"
    assert "content-security-policy" in r.headers


def test_a_failed_asr_turn_gives_a_friendly_error_event(rig, monkeypatch):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    token = token_for(rig, link)

    async def no_asr(audio_bytes, name, language=None):
        return None

    monkeypatch.setattr(rig.env, "_speech_to_text", no_asr)
    r = say(rig, link["id"], token)
    assert r.status_code in (200, 503)
    assert "Traceback" not in r.text and "gsk_" not in r.text


# ------------------------------------------------------------ the public app exposes nothing else (security)

EXPECTED_CALL_ROUTES = {
    "/call/assets/call.js", "/call/assets/call.css", "/call/{link_id}", "/call/api/{link_id}/session",
    "/call/api/{link_id}/turn", "/call/api/{link_id}/clip/{turn_id}/{index}", "/call/api/{link_id}/end"}


def flatten(routes, prefix=""):
    """(path, methods) of every route of an app, through nested (included) routers."""
    for r in routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            yield from flatten(inner.routes, prefix + (r.include_context.prefix or ""))
        elif hasattr(r, "path"):
            yield prefix + r.path, set(getattr(r, "methods", None) or ["GET"])


def public_route_paths():
    return {p for p, _ in flatten(phone_public.public_app.routes)}


def test_public_app_has_exactly_the_phone_and_call_routes():
    paths = public_route_paths()
    call = {p for p in paths if p.startswith("/call")}
    assert call == EXPECTED_CALL_ROUTES
    assert all(p.startswith(("/call/", "/api/phone/")) for p in paths), sorted(paths)
    assert web_call.public_paths() == sorted(EXPECTED_CALL_ROUTES)


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/voices"), ("POST", "/api/voices"), ("GET", "/api/keys"), ("POST", "/api/keys"),
    ("POST", "/api/generate"), ("POST", "/api/generate-stream"), ("GET", "/api/agents"), ("POST", "/api/agents"),
    ("GET", "/api/agents/x"), ("POST", "/api/agents/x/clone"), ("GET", "/api/system/resources"),
    ("POST", "/api/system/models/kokoro/unload"), ("POST", "/api/phone/exotel/call"),
    ("GET", "/api/phone/exotel/calls"), ("GET", "/api/health"), ("GET", "/api/calls"),
    ("GET", "/api/agents/x/web-call-links"), ("POST", "/api/agents/x/web-call-links"),
    ("DELETE", "/api/agents/x/web-call-links/awc_" + "a" * 32), ("POST", "/api/agents/x/web-call-links/y/revoke"),
    ("GET", "/api/agents/x/widget"), ("POST", "/api/widget/session"), ("POST", "/api/widget/turn/stream"),
    ("POST", "/api/conversations/x/turn/stream"), ("GET", "/api/conversations/turns/x/sentences/0/audio"),
    ("GET", "/widget/v1/bonclay-agent.js"), ("GET", "/docs"), ("GET", "/openapi.json"), ("GET", "/"),
    ("GET", "/call"), ("GET", "/call/"), ("GET", "/call/api"), ("POST", "/call/api/x/other"),
    ("GET", "/call/api/awc_" + "a" * 32 + "/turn/extra/more"), ("GET", "/call/api/awc_" + "a" * 32),
])
def test_everything_but_the_call_and_phone_routes_is_absent(rig, method, path):
    r = rig.public.request(method, path)
    assert r.status_code in (404, 405), (method, path, r.status_code)
    assert "Start talking" not in r.text


def test_every_main_app_route_is_unreachable_on_the_public_app(app_client, rig):
    """Walk the real app's route table: nothing on it (except the shared phone paths) answers on the public app."""
    import backend
    public_paths = public_route_paths()
    checked = 0
    for path, methods in flatten(backend.app.routes):
        if path in public_paths:
            continue
        for method in sorted(methods):
            if method in ("HEAD", "OPTIONS"):
                continue
            url = re.sub(r"\{[^}]+\}", "x", path)
            r = rig.public.request(method, url)
            assert r.status_code in (404, 405), (method, path, r.status_code)
            checked += 1
    assert checked > 150


def test_a_link_does_not_unlock_the_rest_of_the_api_even_with_its_token(rig):
    a = make_agent(rig.client)
    link = make_link(rig, a["id"])
    token = token_for(rig, link)
    h = {"Authorization": f"Bearer {token}", "Origin": ORIGIN}
    for path in ("/api/voices", "/api/agents", f"/api/agents/{a['id']}", "/api/keys", "/api/system/resources",
                 f"/api/agents/{a['id']}/web-call-links", "/api/calls"):
        assert rig.public.get(path, headers=h).status_code == 404, path
