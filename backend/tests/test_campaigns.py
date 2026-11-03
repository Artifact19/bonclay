"""Outbound campaigns: CSV import, calling windows, retries, concurrency, the do-not-call list and the circuit breaker
(pure logic with a fake dialer), plus the API and the scheduler pass against the fake Exotel of the phone tests."""
import asyncio
from datetime import datetime, timezone

import pytest

from routers import campaigns as camp_router
from routers import phone_exotel as px
from services import call_records, campaigns
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401
from tests.test_phone_exotel import configure, exotel, rig  # noqa: F401  (fixtures)

IST_MONDAY_11 = datetime(2026, 10, 19, 5, 30, tzinfo=timezone.utc).timestamp()   # Monday 11:00 in Asia/Kolkata
IST_MONDAY_20 = IST_MONDAY_11 + 9 * 3600                                          # 20:00, after the window
IST_SUNDAY_11 = IST_MONDAY_11 - 24 * 3600


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("BONCLAY_CAMPAIGNS_DIR", str(tmp_path / "campaigns"))
    monkeypatch.setenv("BONCLAY_CALL_RECORDS_FILE", str(tmp_path / "records.jsonl"))
    monkeypatch.setenv("CAMPAIGN_SCHEDULER", "off")


def _contacts(*numbers):
    return [{"number": n, "name": ""} for n in numbers]


def _camp(numbers=("+919000000001", "+919000000002", "+919000000003"), **kw):
    c = campaigns.create("owner", name="t", agent_id="a1", mode="local", contacts=_contacts(*numbers), **kw)
    campaigns.set_status(c["id"], "owner", "running")
    return c["id"]


class Phone:
    """A fake dialer and call-record book."""
    def __init__(self):
        self.dialled, self.records, self.fail = [], {}, False

    async def dial(self, camp, contact):
        if self.fail:
            raise RuntimeError("exotel refused")
        sid = f"CA{len(self.dialled) + 1}"
        self.dialled.append(contact["number"])
        self.records[sid] = {"status": "queued"}
        return sid

    def record_of(self, sid):
        return self.records.get(sid)


def tick(cid, now, phone):
    return asyncio.run(campaigns.tick(cid, now, phone.dial, phone.record_of))


def test_csv_with_a_header_a_name_column_bad_rows_and_duplicates():
    text = "name,phone\nAsha,98765 43210\nRavi,+919876543211\nbad,12\nDup,9876543210\n,\n"
    out = campaigns.parse_csv(text, px.normalise_number)
    assert [c["number"] for c in out["contacts"]] == ["+919876543210", "+919876543211"]
    assert out["contacts"][0]["name"] == "Asha"
    assert out["invalid"] == 1 and out["duplicates"] == 1


def test_csv_of_bare_numbers_and_name_first_columns_and_the_size_limit():
    assert len(campaigns.parse_csv("9876543210\n9876543211\n", px.normalise_number)["contacts"]) == 2
    out = campaigns.parse_csv("Asha,9876543210\n", px.normalise_number)
    assert out["contacts"][0]["name"] == "Asha"
    many = "\n".join(f"+9190000{n:05d}" for n in range(campaigns.MAX_CONTACTS + 5))
    big = campaigns.parse_csv(many, px.normalise_number)
    assert len(big["contacts"]) == campaigns.MAX_CONTACTS and big["over_limit"] == 5


def test_the_calling_window_checks_day_hours_and_time_zone():
    w = campaigns.clean_window(None)
    assert campaigns.in_window(IST_MONDAY_11, w) is True
    assert campaigns.in_window(IST_MONDAY_20, w) is False
    assert campaigns.in_window(IST_SUNDAY_11, w) is False
    assert campaigns.clean_window({"timezone": "Not/AZone", "start": "9:00", "days": [9]}) == campaigns.DEFAULT_WINDOW


def test_nothing_is_dialled_outside_the_window():
    phone, cid = Phone(), _camp()
    assert tick(cid, IST_MONDAY_20, phone)["state"] == "outside_window" and phone.dialled == []
    assert tick(cid, IST_SUNDAY_11, phone)["state"] == "outside_window"


def test_the_concurrency_cap_limits_calls_in_flight():
    phone, cid = Phone(), _camp(concurrency=2)
    assert tick(cid, IST_MONDAY_11, phone)["dialled"] == 2
    assert tick(cid, IST_MONDAY_11 + 10, phone)["dialled"] == 0          # both still calling
    phone.records["CA1"] = {"status": "completed"}
    assert tick(cid, IST_MONDAY_11 + 20, phone)["dialled"] == 1          # a slot freed up
    assert campaigns.counts(campaigns.get(cid)) == {"pending": 0, "calling": 2, "done": 1, "failed": 0, "dnc": 0}


def test_a_connected_call_counts_as_done_even_without_a_callback():
    phone, cid = Phone(), _camp(numbers=("+919000000001",))
    tick(cid, IST_MONDAY_11, phone)
    phone.records["CA1"] = {"status": "in-progress", "started_at": "x", "hangup_reason": "caller_hung_up"}
    out = tick(cid, IST_MONDAY_11 + 60, phone)
    assert out["state"] == "done" and campaigns.get(cid)["contacts"][0]["status"] == "done"


def test_busy_and_no_answer_calls_are_retried_after_the_delay_then_fail():
    phone, cid = Phone(), _camp(numbers=("+919000000001",), max_attempts=2, retry_minutes=30)
    tick(cid, IST_MONDAY_11, phone)
    phone.records["CA1"] = {"status": "no-answer"}
    tick(cid, IST_MONDAY_11 + 60, phone)
    c = campaigns.get(cid)["contacts"][0]
    assert c["status"] == "pending" and c["attempts"] == 1
    assert tick(cid, IST_MONDAY_11 + 600, phone)["dialled"] == 0         # 30 minutes have not passed
    assert tick(cid, IST_MONDAY_11 + 31 * 60, phone)["dialled"] == 1     # second attempt
    phone.records["CA2"] = {"status": "busy"}
    out = tick(cid, IST_MONDAY_11 + 32 * 60, phone)
    assert campaigns.get(cid)["contacts"][0]["status"] == "failed" and out["state"] == "done"


def test_a_call_that_cannot_be_placed_uses_up_an_attempt():
    phone, cid = Phone(), _camp(numbers=("+919000000001",), max_attempts=1)
    phone.fail = True
    tick(cid, IST_MONDAY_11, phone)
    assert campaigns.get(cid)["contacts"][0]["status"] == "failed"


def test_numbers_on_the_do_not_call_list_are_never_dialled():
    campaigns.dnc_add("owner", ["+919000000002"])
    phone, cid = Phone(), _camp()
    assert [c["status"] for c in campaigns.get(cid)["contacts"]] == ["pending", "dnc", "pending"]
    tick(cid, IST_MONDAY_11, phone)
    assert "+919000000002" not in phone.dialled
    campaigns.dnc_add("owner", ["+919000000003"])                          # added after the campaign was made
    tick(cid, IST_MONDAY_11 + 5000, phone)
    assert "+919000000003" not in phone.dialled
    assert campaigns.dnc_remove("owner", "+919000000002") is True and campaigns.dnc_remove("owner", "+91nope") is False


def test_the_hourly_cap_limits_dialling():
    phone, cid = Phone(), _camp(numbers=tuple(f"+91900000{n:04d}" for n in range(6)), concurrency=3, max_per_hour=2)
    assert tick(cid, IST_MONDAY_11, phone)["dialled"] == 2
    for sid in list(phone.records):
        phone.records[sid] = {"status": "completed"}
    assert tick(cid, IST_MONDAY_11 + 600, phone)["dialled"] == 0
    assert tick(cid, IST_MONDAY_11 + 3700, phone)["dialled"] == 2


def test_the_circuit_breaker_pauses_when_half_of_the_recent_calls_fail():
    numbers = tuple(f"+91900000{n:04d}" for n in range(20))
    phone, cid = Phone(), _camp(numbers=numbers, max_attempts=1, concurrency=3, max_per_hour=120)
    t = IST_MONDAY_11
    state = ""
    for _ in range(12):
        tick(cid, t, phone)
        for sid, rec in phone.records.items():
            if rec["status"] == "queued":
                rec["status"] = "failed"
        state = tick(cid, t + 5, phone)["state"]
        t += 60
        if state == "paused_breaker":
            break
    camp = campaigns.get(cid)
    assert state == "paused_breaker" and camp["status"] == "paused" and "failed to connect" in camp["paused_reason"]


def test_a_stale_call_with_no_answer_from_the_network_counts_as_failed():
    phone, cid = Phone(), _camp(numbers=("+919000000001",), max_attempts=1)
    tick(cid, IST_MONDAY_11, phone)
    tick(cid, IST_MONDAY_11 + campaigns.STALE_CALL_S + 60, phone)
    assert campaigns.get(cid)["contacts"][0]["status"] == "failed"


def test_finished_campaigns_and_their_numbers_are_purged_after_the_keep_days(monkeypatch):
    monkeypatch.setenv("CAMPAIGN_KEEP_DAYS", "1")
    cid = _camp()
    campaigns.set_status(cid, "owner", "stopped")
    assert campaigns.purge_finished(now=campaigns.get(cid)["finished_at"] + 3600) == 0
    assert campaigns.purge_finished(now=campaigns.get(cid)["finished_at"] + 2 * 86400) == 1
    assert campaigns.get(cid) is None


# ---- API and scheduler --------------------------------------------------------------------------------------------

def _create(rig, agent, **over):
    body = {"name": "Reminders", "agent_id": agent["id"], "csv": "name,phone\nAsha,9876543210\nRavi,9876543211\n",
            "caller_agreed": True, **over}
    return rig.client.post("/api/campaigns", headers=LOCAL, json=body)


def test_creating_a_campaign_needs_consent_and_valid_numbers(rig, exotel, monkeypatch):
    a = make_agent(rig.client)
    assert _create(rig, a, caller_agreed=False).json()["code"] == "CONSENT_REQUIRED"
    assert _create(rig, a, csv="not,a,number\nx,y,z\n").json()["code"] == "NO_CONTACTS"
    r = _create(rig, a)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2 and body["status"] == "draft" and body["counts"]["pending"] == 2
    assert "98765" not in str(rig.client.get(f"/api/campaigns/{body['id']}", headers=LOCAL).json())   # numbers masked


def test_starting_needs_exotel_and_other_owners_cannot_see_a_campaign(rig, exotel, monkeypatch):
    a = make_agent(rig.client)
    cid = _create(rig, a).json()["id"]
    assert rig.client.post(f"/api/campaigns/{cid}/start", headers=LOCAL).json()["code"] == "PHONE_NOT_CONFIGURED"
    configure(monkeypatch)
    assert rig.client.post(f"/api/campaigns/{cid}/start", headers=LOCAL).json()["status"] == "running"
    assert rig.client.post(f"/api/campaigns/{cid}/explode", headers=LOCAL).json()["code"] == "BAD_ACTION"
    assert rig.client.get("/api/campaigns/nope", headers=LOCAL).status_code == 404
    assert rig.client.post(f"/api/campaigns/{cid}/stop", headers=LOCAL).json()["status"] == "stopped"
    assert rig.client.post(f"/api/campaigns/{cid}/start", headers=LOCAL).json()["status"] == "stopped"   # final
    assert rig.client.delete(f"/api/campaigns/{cid}", headers=LOCAL).json() == {"status": "deleted"}


def test_the_scheduler_pass_places_calls_through_exotel_and_records_them(rig, exotel, monkeypatch):
    configure(monkeypatch)
    a = make_agent(rig.client)
    rig.client.post("/api/do-not-call", headers=LOCAL, json={"numbers": ["9876543211"]})
    cid = _create(rig, a, concurrency=2).json()["id"]
    rig.client.post(f"/api/campaigns/{cid}/start", headers=LOCAL)
    assert asyncio.run(camp_router.run_once(IST_MONDAY_20)) == 0         # outside the window: nothing
    assert asyncio.run(camp_router.run_once(IST_MONDAY_11)) == 1         # one contact is on the do-not-call list
    assert [s["data"]["From"] for s in exotel.sent] == ["+919876543210"]
    assert call_records.get("CAabc123")["direction"] == "outbound"
    view = rig.client.get(f"/api/campaigns/{cid}", headers=LOCAL).json()
    assert view["counts"]["calling"] == 1 and view["counts"]["dnc"] == 1


def test_the_do_not_call_api_adds_lists_and_removes(rig):
    r = rig.client.post("/api/do-not-call", headers=LOCAL, json={"numbers": ["9876543210", "bad"]})
    assert r.json() == {"added": 1, "total": 1}
    assert rig.client.get("/api/do-not-call", headers=LOCAL).json()["numbers"] == ["+919876543210"]
    assert rig.client.delete("/api/do-not-call/9876543210", headers=LOCAL).json() == {"status": "removed"}
    assert rig.client.delete("/api/do-not-call/9876543210", headers=LOCAL).status_code == 404


def test_a_voicemail_call_is_retried_later_not_counted_as_answered():
    phone, cid = Phone(), _camp(numbers=("+919000000001",), max_attempts=2, retry_minutes=30)
    tick(cid, IST_MONDAY_11, phone)
    phone.records["CA1"] = {"status": "in-progress", "started_at": "x", "hangup_reason": "voicemail",
                            "outcome": "voicemail"}
    tick(cid, IST_MONDAY_11 + 60, phone)
    c = campaigns.get(cid)["contacts"][0]
    assert c["status"] == "pending" and c["last_status"] == "voicemail" and c["attempts"] == 1
