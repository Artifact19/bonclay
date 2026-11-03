"""Several Exotel accounts (routers/phone_exotel.py): the env format, failover when an account is out of credit or its
credentials are refused, round robin, the saved account state, and which account a call's status is read from. Exotel is
faked per account (by the SID in the URL); no network."""
import json
import logging
import os
import stat

import httpx
import pytest

from routers import phone_exotel as px
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_phone_exotel import call, rig  # noqa: F401  (rig is a fixture)

NAMES = ("EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_ACCOUNT_SID", "EXOTEL_CALLER_ID", "EXOTEL_SUBDOMAIN")


def set_account(monkeypatch, n, **over):
    suffix = "" if n == 1 else f"_{n}"
    vals = {"EXOTEL_API_KEY": f"key{n}", "EXOTEL_API_TOKEN": f"token{n}", "EXOTEL_ACCOUNT_SID": f"sid{n}",
            "EXOTEL_CALLER_ID": f"0804749189{n}", **over}
    for name, v in vals.items():
        if v is None:
            monkeypatch.delenv(name + suffix, raising=False)
        else:
            monkeypatch.setenv(name + suffix, v)


@pytest.fixture()
def clean(monkeypatch):
    """No Exotel setting from a real backend/.env; a public address; the rotation back at its default."""
    for n in range(1, 10):
        for name in NAMES:
            monkeypatch.delenv(name + ("" if n == 1 else f"_{n}"), raising=False)
    for name in ("EXOTEL_ROTATION", "EXOTEL_EXHAUSTED_RESET_HOURS", "PHONE_ALLOWED_NUMBERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://abc.ngrok.app")
    monkeypatch.setattr(px, "_rr_counter", 0)
    px._warned_partial.clear()


class MultiHttp:
    """Exotel by account: `replies` maps an account SID to the answers its next calls get (the last one repeats)."""
    replies: dict = {}
    sent: list = []
    gets: list = []
    error_for: set = set()

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    @staticmethod
    def _sid(url):
        return url.split("/Accounts/")[1].split("/")[0]

    async def post(self, url, auth=None, data=None):
        sid = self._sid(url)
        MultiHttp.sent.append({"sid": sid, "auth": auth, "data": data})
        if sid in MultiHttp.error_for:
            raise httpx.ConnectError("down")
        queue = MultiHttp.replies.get(sid) or [(200, {"Call": {"Sid": f"CA-{sid}"}})]
        status, body = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    async def get(self, url, auth=None):
        MultiHttp.gets.append({"sid": self._sid(url), "auth": auth, "url": url})
        return httpx.Response(200, json={"Call": {"Status": "ringing"}})


@pytest.fixture()
def fake(monkeypatch):
    MultiHttp.replies, MultiHttp.sent, MultiHttp.gets, MultiHttp.error_for = {}, [], [], set()
    monkeypatch.setattr(httpx, "AsyncClient", MultiHttp)
    return MultiHttp


def sent_sids():
    return [s["sid"] for s in MultiHttp.sent]


def balance(sid):
    return {sid: [(400, {"RestException": {"Status": 400, "Message": "Insufficient balance in your account"}})]}


def state_of(label):
    acct = next(a for a in px.exotel_accounts() if a["label"] == label)
    return px.account_state(acct)["state"]


@pytest.fixture()
def ready(rig, clean, monkeypatch):
    """The test client and an agent, accounts 1 and 2 configured (account 1 first)."""
    monkeypatch.setenv("PHONE_ALLOWED_NUMBERS", "+919876543210")
    set_account(monkeypatch, 1)
    set_account(monkeypatch, 2)
    return rig, make_agent(rig.client)


# ---------------------------------------------------------------------------------------------- parsing

def test_one_account_with_the_old_names_still_works(clean, monkeypatch):
    set_account(monkeypatch, 1, EXOTEL_SUBDOMAIN="api.exotel.com")
    accounts = px.exotel_accounts()
    assert [a["label"] for a in accounts] == ["1"]
    assert accounts[0]["sid"] == "sid1" and accounts[0]["subdomain"] == "api.exotel.com"
    assert px.exotel_settings()["key"] == "key1" and px.exotel_settings()["public"] == "https://abc.ngrok.app"
    assert px.missing_settings() == []


def test_three_accounts_are_read_in_order_with_their_own_subdomain(clean, monkeypatch):
    for n in (1, 2, 3):
        set_account(monkeypatch, n)
    set_account(monkeypatch, 3, EXOTEL_SUBDOMAIN="api.exotel.com")
    accounts = px.exotel_accounts()
    assert [a["label"] for a in accounts] == ["1", "2", "3"]
    assert accounts[1]["subdomain"] == "api.in.exotel.com" and accounts[2]["subdomain"] == "api.exotel.com"


def test_a_gap_in_the_numbers_is_allowed(clean, monkeypatch):
    for n in (1, 3, 9):
        set_account(monkeypatch, n)
    assert [a["label"] for a in px.exotel_accounts()] == ["1", "3", "9"]


def test_a_partial_account_is_ignored_with_one_warning_that_names_only_variables(clean, monkeypatch, caplog):
    set_account(monkeypatch, 1)
    set_account(monkeypatch, 2, EXOTEL_API_TOKEN=None, EXOTEL_CALLER_ID=None)
    with caplog.at_level(logging.WARNING, logger="bonclay.phone.exotel"):
        assert [a["label"] for a in px.exotel_accounts()] == ["1"]
        px.exotel_accounts()                                   # asked again: not logged again
    warnings = [r.getMessage() for r in caplog.records if "account 2" in r.getMessage()]
    assert len(warnings) == 1
    assert "EXOTEL_API_TOKEN_2" in warnings[0] and "EXOTEL_CALLER_ID_2" in warnings[0]
    assert "key2" not in warnings[0] and "sid2" not in warnings[0]


def test_missing_settings_needs_one_complete_account_and_the_public_address(clean, monkeypatch):
    assert px.missing_settings() == ["EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_ACCOUNT_SID", "EXOTEL_CALLER_ID"]
    set_account(monkeypatch, 2)                                   # only account 2 is complete: that is enough
    assert px.missing_settings() == []
    monkeypatch.delenv("PUBLIC_BASE_URL")
    assert px.missing_settings() == ["PUBLIC_BASE_URL"]
    set_account(monkeypatch, 2, EXOTEL_API_KEY=None)
    assert "EXOTEL_API_KEY" in px.missing_settings() and "PUBLIC_BASE_URL" in px.missing_settings()


def test_the_startup_check_accepts_any_complete_account(clean, monkeypatch):
    import backend
    assert backend._exotel_configured() is False
    set_account(monkeypatch, 3)
    assert backend._exotel_configured() is True
    set_account(monkeypatch, 3, EXOTEL_ACCOUNT_SID=None)
    assert backend._exotel_configured() is False


# ---------------------------------------------------------------------------------------------- failover

def test_a_balance_message_moves_to_the_next_account_and_marks_the_first_exhausted(ready, fake, caplog):
    rig, a = ready
    fake.replies = balance("sid1")
    with caplog.at_level(logging.INFO, logger="bonclay.phone.exotel"):
        r = call(rig, a)
    assert r.status_code == 200, r.text
    assert sent_sids() == ["sid1", "sid2"]
    assert fake.sent[1]["auth"] == ("key2", "token2") and fake.sent[1]["data"]["CallerId"] == "08047491892"
    assert state_of("1") == "exhausted" and state_of("2") == "active"
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Exotel account 1 out of credit; trying account 2" in text and "via Exotel account 2" in text
    assert "key1" not in text and "token1" not in text and "key2" not in text and "token2" not in text


def test_the_next_call_skips_the_exhausted_account(ready, fake):
    rig, a = ready
    fake.replies = balance("sid1")
    assert call(rig, a).status_code == 200
    fake.sent.clear()
    assert call(rig, a).status_code == 200
    assert sent_sids() == ["sid2"]                                # account 1 is not asked again


@pytest.mark.parametrize("word", ["credit", "recharge", "quota", "limit exceeded", "BALANCE"])
def test_each_credit_word_counts(ready, fake, word):
    rig, a = ready
    fake.replies = {"sid1": [(400, {"Message": f"Your {word} is over"})]}
    assert call(rig, a).status_code == 200
    assert state_of("1") == "exhausted" and sent_sids() == ["sid1", "sid2"]


def test_http_402_means_exhausted(ready, fake):
    rig, a = ready
    fake.replies = {"sid1": [(402, "<Response><Message>Payment required</Message></Response>")]}
    assert call(rig, a).status_code == 200
    assert state_of("1") == "exhausted" and sent_sids() == ["sid1", "sid2"]


@pytest.mark.parametrize("status", [401, 403])
def test_refused_credentials_mark_auth_failed_and_move_on(ready, fake, status, caplog):
    rig, a = ready
    fake.replies = {"sid1": [(status, {"RestException": {"Message": "Authentication failed"}})]}
    with caplog.at_level(logging.WARNING, logger="bonclay.phone.exotel"):
        assert call(rig, a).status_code == 200
    assert state_of("1") == "auth_failed" and sent_sids() == ["sid1", "sid2"]
    assert "token1" not in "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("message", ["Number is not whitelisted on this account", "The number is not verified"])
def test_a_number_not_allowed_on_an_account_moves_on_without_marking_it(ready, fake, message):
    rig, a = ready
    fake.replies = {"sid1": [(400, {"Message": message})]}
    assert call(rig, a).status_code == 200
    assert sent_sids() == ["sid1", "sid2"]
    assert state_of("1") == "active"                              # the account itself is fine
    fake.sent.clear()
    fake.replies = {}
    call(rig, a)
    assert sent_sids() == ["sid1"]                                # and is asked first again


def test_a_403_that_says_whitelist_is_not_an_auth_failure(ready, fake):
    rig, a = ready
    fake.replies = {"sid1": [(403, {"Message": "Destination not whitelisted"})]}
    assert call(rig, a).status_code == 200
    assert state_of("1") == "active"


def test_a_kyc_refusal_moves_to_the_next_account_without_marking_it(ready, fake):
    """KYC is per account: one that has not finished it may place no outbound call, another may. The KYC wording must
    not read as a credentials failure (it is a 403), and exotel_reason's whitelisting hint must not matter."""
    rig, a = ready
    kyc = "Your account is not yet KYC compliant. This is mandatory before making outbound calls."
    fake.replies = {"sid1": [(403, {"Message": kyc})]}
    assert call(rig, a).status_code == 200
    assert sent_sids() == ["sid1", "sid2"]
    assert state_of("1") == "active"


def test_every_account_refusing_for_kyc_says_so(ready, fake):
    rig, a = ready
    kyc = "Your account is not yet KYC compliant. This is mandatory before making outbound calls."
    fake.replies = {"sid1": [(403, {"Message": kyc})], "sid2": [(403, {"Message": kyc})]}
    r = call(rig, a)
    assert r.status_code == 403 and r.json()["code"] == "EXOTEL_KYC_REQUIRED"
    assert "KYC" in r.json()["error"] and "credentials" not in r.json()["error"]
    assert state_of("1") == "active" and state_of("2") == "active"


def test_any_other_refusal_stops_without_trying_the_next_account(ready, fake):
    rig, a = ready
    fake.replies = {"sid1": [(400, {"Message": "Invalid number"})]}
    r = call(rig, a)
    assert r.status_code == 502 and r.json()["code"] == "EXOTEL_REFUSED" and "Invalid number" in r.json()["error"]
    assert sent_sids() == ["sid1"] and state_of("1") == "active"
    fake.sent.clear()
    fake.replies = {"sid1": [(503, "Service unavailable")]}
    assert call(rig, a).status_code == 502 and sent_sids() == ["sid1"]


def test_a_network_error_is_not_retried_on_another_account(ready, fake):
    rig, a = ready
    fake.error_for = {"sid1"}
    r = call(rig, a)
    assert r.status_code == 502 and r.json()["code"] == "EXOTEL_UNREACHABLE"
    assert sent_sids() == ["sid1"]


def test_when_every_account_is_out_the_error_says_so_and_names_no_secrets(ready, fake):
    rig, a = ready
    fake.replies = {**balance("sid1"), "sid2": [(401, {"Message": "Authentication failed"})]}
    r = call(rig, a)
    assert r.status_code == 503 and r.json()["code"] == "EXOTEL_ALL_ACCOUNTS_EXHAUSTED"
    msg = r.json()["error"]
    assert msg.startswith("All 2 Exotel accounts are out of credit or refused (1 out of credit, 2 credentials refused)")
    assert "Recharge one or add another account in backend/.env." in msg
    assert "key1" not in msg and "token" not in msg and "sid" not in msg
    assert sent_sids() == ["sid1", "sid2"]                        # one try per account


def test_with_every_account_marked_each_is_still_tried_once_per_call(ready, fake):
    """A recharge cannot be seen from here, so when all are marked the next call asks each once instead of giving up."""
    rig, a = ready
    fake.replies = {**balance("sid1"), **balance("sid2")}
    assert call(rig, a).status_code == 503
    fake.sent.clear()
    fake.replies = {"sid1": [(200, {"Call": {"Sid": "CA-back"}})]}      # account 1 was recharged
    r = call(rig, a)
    assert r.status_code == 200 and sent_sids() == ["sid1"]
    assert state_of("1") == "active"
    fake.sent.clear()
    call(rig, a)
    assert sent_sids() == ["sid1"]


def test_a_single_account_keeps_the_old_refusal_error(rig, clean, fake, monkeypatch):
    monkeypatch.setenv("PHONE_ALLOWED_NUMBERS", "+919876543210")
    set_account(monkeypatch, 1)
    a = make_agent(rig.client)
    fake.replies = balance("sid1")
    r = call(rig, a)
    assert r.status_code == 502 and r.json()["code"] == "EXOTEL_REFUSED" and "Insufficient balance" in r.json()["error"]
    assert sent_sids() == ["sid1"]


def test_marked_accounts_are_tried_last_never_left_out(ready, fake, monkeypatch):
    """Accounts 1 and 2 ran out earlier; 3 is active but now refuses for credit: account 1 may have been recharged."""
    rig, a = ready
    set_account(monkeypatch, 3)
    accounts = px.exotel_accounts()
    px._set_account_state(accounts[0], "exhausted")
    px._set_account_state(accounts[1], "exhausted")
    fake.replies = {**balance("sid3")}
    r = call(rig, a)
    assert r.status_code == 200, r.text
    assert sent_sids() == ["sid3", "sid1"]
    assert state_of("1") == "active" and state_of("2") == "exhausted" and state_of("3") == "exhausted"
    assert px._known_calls[r.json()["call_sid"]]["account"] == "1"


def test_an_exhausted_second_account_is_the_last_resort_when_the_first_refuses(ready, fake):
    rig, a = ready
    px._set_account_state(px.exotel_accounts()[1], "exhausted")
    fake.replies = balance("sid1")
    r = call(rig, a)
    assert r.status_code == 200, r.text
    assert sent_sids() == ["sid1", "sid2"]
    assert state_of("1") == "exhausted" and state_of("2") == "active"


def test_one_configured_account_refusing_for_credit_gives_the_plain_refusal_and_is_marked(rig, clean, fake, monkeypatch):
    monkeypatch.setenv("PHONE_ALLOWED_NUMBERS", "+919876543210")
    set_account(monkeypatch, 1)
    a = make_agent(rig.client)
    fake.replies = balance("sid1")
    r = call(rig, a)
    assert r.status_code == 502 and r.json()["code"] == "EXOTEL_REFUSED"
    assert state_of("1") == "exhausted" and sent_sids() == ["sid1"]


# ---------------------------------------------------------------------------------------------- rotation and state

def test_round_robin_alternates_between_accounts(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("EXOTEL_ROTATION", "round_robin")
    set_account(monkeypatch, 3)
    monkeypatch.setattr(px, "CALLS_PER_HOUR", 100)
    for _ in range(6):
        assert call(rig, a).status_code == 200
    assert sent_sids() == ["sid1", "sid2", "sid3", "sid1", "sid2", "sid3"]


def test_round_robin_still_fails_over_and_skips_exhausted_accounts(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setenv("EXOTEL_ROTATION", "round_robin")
    fake.replies = balance("sid1")
    assert call(rig, a).status_code == 200 and sent_sids() == ["sid1", "sid2"]
    fake.sent.clear()
    assert call(rig, a).status_code == 200 and call(rig, a).status_code == 200
    assert sent_sids() == ["sid2", "sid2"]


def test_failover_always_starts_with_the_first_account(ready, fake, monkeypatch):
    rig, a = ready
    monkeypatch.setattr(px, "CALLS_PER_HOUR", 100)
    for _ in range(3):
        call(rig, a)
    assert sent_sids() == ["sid1", "sid1", "sid1"]


def test_an_exhausted_account_comes_back_after_the_reset_window(ready, fake, monkeypatch):
    rig, a = ready
    clock = [1_000_000.0]
    monkeypatch.setattr(px, "_now_ts", lambda: clock[0])
    monkeypatch.setenv("EXOTEL_EXHAUSTED_RESET_HOURS", "2")
    fake.replies = balance("sid1")
    assert call(rig, a).status_code == 200
    assert state_of("1") == "exhausted"
    clock[0] += 3600
    assert state_of("1") == "exhausted"
    fake.sent.clear()
    fake.replies = {}
    call(rig, a)
    assert sent_sids() == ["sid2"]
    clock[0] += 3601                                               # 2 h and 1 s after it ran out
    assert state_of("1") == "active"
    fake.sent.clear()
    assert call(rig, a).status_code == 200 and sent_sids() == ["sid1"]


def test_a_successful_call_makes_the_account_active_again(ready, fake):
    rig, a = ready
    accounts = px.exotel_accounts()
    px._set_account_state(accounts[0], "exhausted")
    px._set_account_state(accounts[1], "auth_failed")              # every account marked: each is tried, 1 first
    assert call(rig, a).status_code == 200 and sent_sids() == ["sid1"]
    assert state_of("1") == "active"
    assert state_of("2") == "auth_failed"


def test_the_state_file_round_trips_is_private_and_holds_no_secrets(ready, fake, monkeypatch, tmp_path):
    rig, a = ready
    clock = [2_000_000.0]
    monkeypatch.setattr(px, "_now_ts", lambda: clock[0])
    fake.replies = balance("sid1")
    assert call(rig, a).status_code == 200
    path = tmp_path / "exotel_accounts.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    raw = path.read_text()
    key = px._account_key(px.exotel_accounts()[0])
    assert json.loads(raw) == {key: {"state": "exhausted", "since": 2_000_000.0}}
    assert "sid1" not in raw and "key1" not in raw and "token1" not in raw
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    # a restart: the memory is empty, the file is read again
    px._account_state.clear()
    monkeypatch.setattr(px, "_state_loaded_from", None)
    assert state_of("1") == "exhausted" and state_of("2") == "active"


def test_a_broken_state_file_means_every_account_is_active(ready, tmp_path, monkeypatch):
    (tmp_path / "exotel_accounts.json").write_text("{not json")
    monkeypatch.setattr(px, "_state_loaded_from", None)
    assert state_of("1") == "active"


# ---------------------------------------------------------------------------------------------- status of calls and the route

def test_call_status_uses_the_credentials_of_the_account_that_placed_the_call(ready, fake):
    rig, a = ready
    fake.replies = {**balance("sid1"), "sid2": [(200, {"Call": {"Sid": "CAsecond"}})]}
    sid = call(rig, a).json()["call_sid"]
    assert sid == "CAsecond"
    assert px._known_calls[sid]["account"] == "2"
    assert px.call_records.get(sid)["exotel_account"] == "2"
    r = rig.client.get(f"/api/phone/exotel/call/{sid}", headers=LOCAL)
    assert r.status_code == 200 and r.json()["status"] == "ringing"
    assert fake.gets[0]["sid"] == "sid2" and fake.gets[0]["auth"] == ("key2", "token2")


def test_call_status_falls_back_to_the_first_account_when_the_placing_one_is_gone(ready, fake, monkeypatch):
    rig, a = ready
    fake.replies = {"sid1": [(200, {"Call": {"Sid": "CAfirst"}})]}
    px._known_calls["CAold"] = {"owner": "local-user", "agent_id": a["id"], "account": "7"}
    rig.client.get("/api/phone/exotel/call/CAold", headers=LOCAL)
    assert fake.gets[0]["sid"] == "sid1"


def test_the_status_route_lists_accounts_without_any_secret(ready, fake):
    rig, a = ready
    fake.replies = balance("sid1")
    call(rig, a)
    body = rig.client.get("/api/phone/exotel/status", headers=LOCAL).json()
    assert body["configured"] is True and body["subdomain"] == "api.in.exotel.com"   # existing fields are unchanged
    assert [x["label"] for x in body["accounts"]] == ["1", "2"]
    assert [x["state"] for x in body["accounts"]] == ["exhausted", "active"]
    assert body["accounts"][0]["since"] and body["accounts"][1]["since"] is None
    assert set(body["accounts"][0]) == {"label", "caller_id_masked", "state", "since"}
    raw = json.dumps(body)
    for secret in ("key1", "key2", "token1", "token2", "sid1", "sid2", "08047491891", "08047491892"):
        assert secret not in raw
    assert body["accounts"][0]["caller_id_masked"].endswith("1891")
