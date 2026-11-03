"""Caller details in a Google Sheet (services/caller_sheet.py, routers/integrations.py): one row per conversation, written
when it ends, updated in place, queued and retried, never in the way of a call or a reply.

Google is replaced at its single HTTP function (services.integrations._google_http) by FakeSheets, a small stateful
spreadsheet: the real Google API is never called and no key file of the owner is read (GOOGLE_SERVICE_ACCOUNT_FILE
points into tmp_path)."""
import json
import re
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import unquote

import jwt
import pytest

from services import caller_memory as cm
from services import caller_sheet as cs
from services import integrations as ig
from tests.test_integrations import SHEET_ID, SHEET_URL, sa_key  # noqa: F401  (sa_key is a fixture)
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_phone_exotel import rig as phone_rig  # noqa: F401  (a fixture: the phone line, with fakes)
from tests.test_turn_stream import rig  # noqa: F401  (rig is a fixture)

DETAILS = {"name": "Priya Sharma", "phone": "9876543210", "date": "2026-10-28", "time": "10:30",
           "reason": "Fever and cough", "doctor": "Dr. Meera Kulkarni", "age": "34", "email": "priya@example.com",
           "other": "Prefers mornings"}


# ------------------------------------------------------------------------------------------------ fixtures

@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("BONCLAY_INTEGRATIONS_DIR", str(tmp_path / "integrations"))
    monkeypatch.setenv("BONCLAY_ALLOW_LOCAL_WEBHOOKS", "on")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_FILE", str(tmp_path / "no_key.json"))   # never the owner's real file
    monkeypatch.setenv("CALLER_SHEET_END_DELAY_S", "0")
    monkeypatch.delenv("BONCLAY_SECRETS_KEY", raising=False)
    monkeypatch.delenv("BONCLAY_OFFLINE", raising=False)
    monkeypatch.setattr(ig, "BACKOFF_S", (0.02, 0.02, 0.02))
    ig.dispatcher.clear()
    ig._log_cache.clear()
    ig._token_cache.clear()
    ig._sheets_times.clear()
    cs.reset_state()
    yield
    ig.dispatcher.wait_idle(5)
    ig.dispatcher.clear()
    cs.reset_state()


@pytest.fixture()
def server_key(sa_key, tmp_path, monkeypatch):  # noqa: F811
    """The server's service-account file (the one Google Calendar uses), written under tmp_path."""
    path = tmp_path / "sa.json"
    path.write_text(sa_key.json)
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_FILE", str(path))
    return sa_key


class FakeSheets:
    """A spreadsheet behind `_google_http`: tabs with rows, the token endpoint, values get / update / append, addSheet.
    `script` lists answers for the next calls to Sheets (not the token): (status, body), consumed in order."""

    def __init__(self, sa, monkeypatch, tabs=("Sheet1",)):
        self.sa, self.tabs, self.script, self.log, self.tokens = sa, {t: [] for t in tabs}, [], [], 0
        monkeypatch.setattr(ig, "_google_http", self)

    # -- the grid
    @staticmethod
    def _col(letters):
        n = 0
        for ch in letters:
            n = n * 26 + ord(ch) - 64
        return n - 1

    def rows(self, tab="BonClay"):
        return self.tabs[tab]

    def writes(self):
        return [c for c in self.log if c["method"] in ("PUT", "POST") and "/values/" in c["url"]]

    def _get_range(self, grid, a1):
        m = re.fullmatch(r"([A-Z]+)(\d*)(?::([A-Z]+)(\d*))?", a1)
        c1, r1, c2, r2 = m.groups()
        c1i, c2i = self._col(c1), self._col(c2 or c1)
        if r1 == "" and r2 in ("", None):              # a whole column
            out = [[(row + [""] * 30)[c1i]] if len(row) > c1i and row[c1i] != "" else [] for row in grid]
            while out and not out[-1]:
                out.pop()
            return out
        first, last = int(r1) - 1, int(r2 or r1) - 1
        out = []
        for row in grid[first:last + 1]:
            cells = list(row[c1i:c2i + 1])
            while cells and cells[-1] == "":
                cells.pop()
            out.append(cells)
        while out and not out[-1]:
            out.pop()
        return out

    def _put(self, grid, a1, values):
        m = re.fullmatch(r"([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?", a1)
        c1, r1 = m.group(1), int(m.group(2))
        for i, vals in enumerate(values):
            while len(grid) < r1 + i:
                grid.append([])
            row = grid[r1 + i - 1]
            c = self._col(c1)
            while len(row) < c + len(vals):
                row.append("")
            row[c:c + len(vals)] = vals

    # -- the HTTP function
    def __call__(self, method, url, *, headers=None, json_body=None, data=None, timeout=10):
        if url == ig.GOOGLE_TOKEN_URL:
            claims = jwt.decode(data["assertion"], self.sa.public, algorithms=["RS256"], audience=ig.GOOGLE_TOKEN_URL)
            assert claims["iss"] == self.sa.info["client_email"] and claims["scope"] == ig.SHEETS_SCOPE
            self.tokens += 1
            return 200, {"access_token": f"tok{self.tokens}", "expires_in": 3600}
        assert headers["Authorization"].startswith("Bearer tok")
        self.log.append({"method": method, "url": unquote(url), "body": json_body})
        if self.script:
            return self.script.pop(0)
        path = unquote(url.split(ig.SHEETS_API + "/", 1)[1])
        query = path.split("?", 1)[1] if "?" in path else ""
        path = path.split("?", 1)[0]
        if "/values/" in path:
            sheet_id, rng = path.split("/values/", 1)
            assert sheet_id == SHEET_ID
            append = rng.endswith(":append")
            rng = rng[:-len(":append")] if append else rng
            tab, a1 = rng.rsplit("!", 1)
            tab = tab.strip("'").replace("''", "'")
            if tab not in self.tabs:
                return 400, {"error": {"message": f"Unable to parse range: {rng}"}}
            grid = self.tabs[tab]
            if method == "GET":
                got = self._get_range(grid, a1)
                return 200, ({"values": got} if got else {})
            assert "valueInputOption=RAW" in query            # nothing a caller says can become a formula
            if append:
                grid.append(list(json_body["values"][0]))
                return 200, {"updates": {"updatedRange": f"'{tab}'!A{len(grid)}:N{len(grid)}", "updatedRows": 1}}
            self._put(grid, a1, json_body["values"])
            return 200, {"updatedRows": 1}
        if path.endswith(":batchUpdate"):
            title = json_body["requests"][0]["addSheet"]["properties"]["title"]
            self.tabs[title] = []
            return 200, {"replies": [{}]}
        assert path == SHEET_ID
        return 200, {"sheets": [{"properties": {"title": t}} for t in self.tabs]}


def make_conversation(env, agent_id, *, form=None, minutes_ago=10, source=None, lines=None, is_test=False, **extra):
    """A stored conversation (local store) with turns and, when given, the caller details form."""
    started = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    row = {"agent_id": agent_id, "user_id": "local-user", "created_at": started,
           **({"source": source} if source else {}), **({"is_test": True} if is_test else {}), **extra}
    cid = env.local_db.insert("conversations", row)["id"]
    for i, (role, text, *more) in enumerate([("user", "I need an appointment"), ("agent", "Sure, when?")] if lines is None else lines):
        env.local_db.insert("conversation_turns", {"conversation_id": cid, "role": role, "text_content": text,
                                                   "created_at": started, **(more[0] if more else {})})
    if form:
        cm.save(cid, form, env.local_db, "local-user")
    return cid


def args_for(agent_id, name="Reception", **over):
    base = {"agent_id": agent_id, "user_id": "local-user", "local": True, "enc": None, "agent_name": name,
            "spreadsheet_id": SHEET_ID, "sheet_name": "BonClay", "include_phone": True, "include_test": False}
    return {**base, **over}


def settle():
    assert ig.dispatcher.wait_idle(10)


def sheet_cfg(**over):
    cfg = ig.default_cfg()
    cfg["caller_sheet"].update({"enabled": True, "spreadsheet_id": SHEET_ID, **over})
    return cfg


# ------------------------------------------------------------------------------------- the row's content

def test_the_columns_and_their_order():
    assert cs.HEADER == ["Time (IST)", "Agent", "Channel", "Caller name", "Phone", "Date requested", "Time requested",
                         "Reason", "Other details", "Outcome", "Call length", "Conversation link", "Receipt id",
                         "Conversation id"]
    assert cs.HEADER[-1] == "Conversation id" and cs.ID_COLUMN == "N" and len(cs.HEADER) == 14


def test_a_row_from_sample_details_and_an_outcome():
    row = cs.build_row(form=DETAILS, agent_name="Reception", channel="phone", outcome_key="booked",
                       started_at="2026-10-27T05:00:00+00:00", duration_s=125, conversation_id="c-1",
                       receipt_id="rcpt-9")
    assert row[0] == "2026-10-27 10:30"                      # IST, not UTC
    assert row[1:10] == ["Reception", "phone", "Priya Sharma", "9876543210", "2026-10-28", "10:30", "Fever and cough",
                         "Doctor: Dr. Meera Kulkarni; Age: 34; Email: priya@example.com; Prefers mornings", "Booked"]
    assert row[10] == "2 min 05 s" and row[11].endswith("/dashboard/calls?call=c-1")
    assert row[12:] == ["rcpt-9", "c-1"]


@pytest.mark.parametrize("key, label", [("booked", "Booked"), ("needs_approval", "Meeting requested"),
                                        ("answered", "Answered"), ("couldnt_answer", "Could not answer"),
                                        ("handed_off", "Handed to a person")])
def test_outcomes_are_written_in_plain_words(key, label):
    assert cs.build_row(form={}, agent_name="A", channel="web", outcome_key=key, started_at=None, duration_s=None,
                        conversation_id="c")[9] == label


def test_numbers_and_emails_can_be_masked():
    row = cs.build_row(form=DETAILS, agent_name="A", channel="phone", outcome_key="answered", started_at=None,
                       duration_s=None, conversation_id="c", include_phone=False)
    assert row[4] == "******3210" and "9876543210" not in json.dumps(row)
    assert "priya@example.com" not in json.dumps(row) and "p***@example.com" in row[8]
    full = cs.build_row(form=DETAILS, agent_name="A", channel="phone", outcome_key="answered", started_at=None,
                        duration_s=None, conversation_id="c", include_phone=True)
    assert full[4] == "9876543210"


def test_the_dashboard_link_follows_the_setting(monkeypatch):
    monkeypatch.setenv("BONCLAY_DASHBOARD_URL", "https://app.example.com/")
    assert cs.conversation_link("c 1") == "https://app.example.com/dashboard/calls?call=c%201"


# -------------------------------------------------------------------------------- writing to the sheet

def test_a_missing_tab_and_header_are_created_and_the_row_follows(sa_key, monkeypatch):
    g = FakeSheets(sa_key, monkeypatch, tabs=("Sheet1",))
    creds = ig.parse_service_account(sa_key.json)
    row = ["x"] * 13 + ["c-1"]
    at = cs.write_row("k", creds, SHEET_ID, "BonClay", "c-1", row)
    assert "BonClay" in g.tabs and g.rows()[0] == cs.HEADER and g.rows()[1] == row and at == 2
    # a second conversation lands under it; the header is not written again
    cs.write_row("k", creds, SHEET_ID, "BonClay", "c-2", ["y"] * 13 + ["c-2"])
    assert [r[-1] for r in g.rows()[1:]] == ["c-1", "c-2"] and g.rows()[0] == cs.HEADER
    assert sum(1 for c in g.log if c["method"] == "PUT" and c["body"]["values"] == [cs.HEADER]) == 1


def test_an_existing_header_is_left_alone(sa_key, monkeypatch):
    g = FakeSheets(sa_key, monkeypatch, tabs=("BonClay",))
    g.tabs["BonClay"].append(["my own header"])
    cs.write_row("k", ig.parse_service_account(sa_key.json), SHEET_ID, "BonClay", "c-1", ["x"] * 13 + ["c-1"])
    assert g.rows()[0] == ["my own header"] and len(g.rows()) == 2


def test_the_same_conversation_updates_its_row_and_never_adds_another(sa_key, monkeypatch):
    g = FakeSheets(sa_key, monkeypatch, tabs=("BonClay",))
    creds = ig.parse_service_account(sa_key.json)
    first = cs.write_row("k", creds, SHEET_ID, "BonClay", "c-1", ["a"] * 13 + ["c-1"])
    cs.write_row("k", creds, SHEET_ID, "BonClay", "c-2", ["b"] * 13 + ["c-2"])
    again = cs.write_row("k", creds, SHEET_ID, "BonClay", "c-1", ["new"] * 13 + ["c-1"], known_row=first)
    assert again == first == 2 and len(g.rows()) == 3 and g.rows()[1][0] == "new" and g.rows()[2][0] == "b"


def test_a_row_the_owner_moved_is_found_by_its_id_and_a_deleted_one_is_not_overwritten(sa_key, monkeypatch):
    g = FakeSheets(sa_key, monkeypatch, tabs=("BonClay",))
    creds = ig.parse_service_account(sa_key.json)
    at = cs.write_row("k", creds, SHEET_ID, "BonClay", "c-1", ["a"] * 13 + ["c-1"])
    other = cs.write_row("k", creds, SHEET_ID, "BonClay", "c-2", ["b"] * 13 + ["c-2"])
    g.rows().insert(0, ["a row the owner added at the top"])       # everything moved down one row
    moved = cs.write_row("k", creds, SHEET_ID, "BonClay", "c-1", ["a2"] * 13 + ["c-1"], known_row=at)
    assert moved == at + 1 and g.rows()[moved - 1][0] == "a2" and g.rows()[other][0] == "b"
    del g.rows()[moved - 1]                                        # the owner deleted c-1's row
    fresh = cs.write_row("k", creds, SHEET_ID, "BonClay", "c-1", ["a3"] * 13 + ["c-1"], known_row=moved)
    assert [r[0] for r in g.rows() if r and r[-1] in ("c-1", "c-2")] == ["b", "a3"] and fresh == len(g.rows())


def test_values_are_sent_raw(sa_key, monkeypatch):
    g = FakeSheets(sa_key, monkeypatch, tabs=("BonClay",))
    cs.write_row("k", ig.parse_service_account(sa_key.json), SHEET_ID, "BonClay", "c", ["=HYPERLINK(\"x\")"] + ["a"] * 12 + ["c"])
    assert all("valueInputOption=RAW" in c["url"] for c in g.writes())     # the fake asserts it too


# ------------------------------------------------------------------------- a conversation, queued and written

def test_a_queued_conversation_is_written_with_its_details_and_outcome(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env)
    book = {"tool_calls": [{"name": "book_appointment", "arguments": {}, "ok": True, "result": "booking A12 done"}]}
    cid = make_conversation(env, a["id"], form=DETAILS, source="whatsapp", external_id="919876543210",
                            lines=[("user", "Book me with Dr Meera"), ("agent", "Booked.", book)])
    assert cs.request_write(args_for(a["id"]), cid)
    settle()
    header, row = g.rows()
    assert header == cs.HEADER
    assert row[1:4] == ["Reception", "WhatsApp", "Priya Sharma"] and row[4] == "9876543210"
    assert row[5:8] == ["2026-10-28", "10:30", "Fever and cough"] and row[9] == "Booked" and row[13] == cid
    assert cs.sheet_status(cid) == {"state": "sent"}
    assert cs.ledger_entry(cid)["row"] == 2


def make_agent_row(env, **fields):
    return env.local_db.insert("agents", {"user_id": "local-user", "project_id": "local_project", "name": "Reception",
                                          "system_instructions": "x", "greeting": "", "supported_languages": ["en"],
                                          **fields})


def test_details_that_arrive_later_update_the_same_row(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], form={"phone": "9876543210"})
    cs.request_write(args_for(a["id"]), cid)
    settle()
    assert len(g.rows()) == 2 and g.rows()[1][3] == ""
    cm.save(cid, {"phone": "9876543210", "name": "Priya"}, env.local_db, "local-user")   # the small model's late reading
    settle()
    assert len(g.rows()) == 2 and g.rows()[1][3] == "Priya"


def test_an_update_that_changes_nothing_writes_nothing(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid)
    settle()
    writes = len(g.writes())
    cs.request_write(args_for(a["id"]), cid)
    settle()
    assert len(g.writes()) == writes and cs.sheet_status(cid) == {"state": "sent"}


def test_requests_for_one_conversation_while_one_is_waiting_are_one_write(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], form=DETAILS)
    assert cs.request_write(args_for(a["id"]), cid, delay=0.2) is True
    assert cs.request_write(args_for(a["id"]), cid) is False and cs.request_write(args_for(a["id"]), cid) is False
    assert cs.sheet_status(cid) == {"state": "waiting"}
    settle()
    assert len(g.rows()) == 2 and cs.sheet_status(cid) == {"state": "sent"}


def test_phone_numbers_are_masked_in_the_sheet_when_the_toggle_is_off(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"], include_phone=False), cid)
    settle()
    assert g.rows()[1][4] == "******3210" and "9876543210" not in json.dumps(g.rows())


def test_the_agents_own_key_is_used_when_there_is_one(env, sa_key, monkeypatch):  # noqa: F811
    g = FakeSheets(sa_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env)
    ig.save_secrets(a["id"], {"service_account": ig.parse_service_account(sa_key.json)}, local=True)
    cid = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid)
    settle()
    assert len(g.rows()) == 2 and g.tokens == 1


def test_no_key_anywhere_is_a_clear_failure(env, monkeypatch, sa_key):  # noqa: F811
    FakeSheets(sa_key, monkeypatch)
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid)
    settle()
    st = cs.sheet_status(cid)
    assert st["state"] == "failed" and "service-account key" in st["error"]


# -------------------------------------------------------------------------------------- retries and failures

def test_a_server_error_is_retried_and_then_succeeds(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    g.script = [(503, {"error": {"message": "backend error"}}), (500, {"error": {"message": "again"}})]
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid)
    settle()
    assert len(g.rows()) == 2 and cs.sheet_status(cid) == {"state": "sent"}
    log = ig.get_log(a["id"], local=True)
    statuses = [e["status"] for e in log]
    assert statuses == ["delivered"] and log[0]["attempts"] == 3 and log[0]["target"] == "caller_sheet"   # one entry per job


def test_a_429_waits_and_is_not_given_up_on(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    g.script = [(429, {"error": {"message": "Quota exceeded"}})]
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid)
    time.sleep(0.4)
    entries = ig.get_log(a["id"], local=True)
    assert entries[0]["status"] == "retrying" and "429" in entries[0]["error"] and "30 s" in entries[0]["error"]
    assert cs.sheet_status(cid) == {"state": "waiting"}           # it is still going to be written
    ig.dispatcher.clear()


def test_after_the_last_retry_it_is_failed_with_googles_reason(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    g.script = [(500, {"error": {"message": "boom"}})] * 12
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid)
    settle()
    st = cs.sheet_status(cid)
    assert st["state"] == "failed" and "boom" in st["error"]
    assert cs.last_write(a["id"])["status"] == "failed"


def test_not_shared_is_failed_at_once_with_what_to_do(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    g.script = [(403, {"error": {"message": "The caller does not have permission"}})]
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid)
    settle()
    st = cs.sheet_status(cid)
    assert st["state"] == "failed" and "does not have permission" in st["error"] and "logger@demo" in st["error"]
    assert len(g.log) == 1                                         # a 403 is not retried


def test_offline_mode_refuses_and_calls_nothing(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    monkeypatch.setenv("BONCLAY_OFFLINE", "1")
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid)
    settle()
    st = cs.sheet_status(cid)
    assert st["state"] == "failed" and "Offline mode" in st["error"] and g.log == [] and g.tokens == 0


def test_a_conversation_that_is_gone_fails_cleanly(env, server_key, monkeypatch):
    FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env)
    cs.request_write(args_for(a["id"]), "no-such-conversation")
    settle()
    assert cs.sheet_status("no-such-conversation") == {"state": "failed", "error": "the conversation was not found"}


def test_an_empty_conversation_is_not_written(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env)
    cid = make_conversation(env, a["id"], lines=[])
    cs.request_write(args_for(a["id"]), cid)
    settle()
    assert g.log == [] and cs.sheet_status(cid) is None


def test_a_broken_queue_never_raises_into_the_caller(env, monkeypatch):
    monkeypatch.setattr(ig.dispatcher, "submit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cs.request_write(args_for("a"), "c") is False
    cs.on_phone_call_end({"id": "a", "integrations": sheet_cfg(), "name": "x"}, user_id="u", local=True,
                         conversation_id="c")                      # does not raise
    cs.touch(SimpleNamespace())                                    # nor does a malformed run
    cs.on_details_saved(None)


# ----------------------------------------------------------------------------------------- phone call end

def test_the_end_of_a_phone_call_queues_exactly_one_write(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env, integrations=sheet_cfg())
    cid = make_conversation(env, a["id"], form=DETAILS, source="phone",
                            lines=[("user", "[call connected]"), ("agent", "Hello"), ("user", "Book me"), ("agent", "Done")])
    agent = env.local_db.get("agents", a["id"])
    cs.on_phone_call_end(agent, user_id="local-user", local=True, conversation_id=cid, duration_s=95)
    cs.on_phone_call_end(agent, user_id="local-user", local=True, conversation_id=cid, duration_s=95)   # a repeated end
    settle()
    assert len(g.writes()) >= 1 and len([r for r in g.rows()[1:]]) == 1
    row = g.rows()[1]
    assert row[2] == "phone" and row[3] == "Priya Sharma" and row[10] == "1 min 35 s" and row[13] == cid
    n = len(g.writes())
    cs.sweep_once(time.time() + 10 ** 6)                             # nothing else is pending for it
    settle()
    assert len(g.writes()) == n


def test_an_agent_without_the_setting_costs_nothing(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    for integrations in (None, ig.default_cfg(), sheet_cfg(enabled=False), sheet_cfg(spreadsheet_id="")):
        a = make_agent_row(env, **({"integrations": integrations} if integrations else {}))
        cid = make_conversation(env, a["id"], form=DETAILS, source="phone")
        cs.on_phone_call_end(env.local_db.get("agents", a["id"]), user_id="local-user", local=True, conversation_id=cid)
    settle()
    assert g.log == [] and cs._state == {}



def test_a_phone_call_end_hook_in_the_runner(phone_rig, env, server_key, monkeypatch):
    """The call runner's own end (routers/phone_exotel.py): a call is made and hung up, one row is written."""
    from tests import test_phone_exotel as tp
    prig = phone_rig
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent(prig.client)
    env.local_db.update("agents", a["id"], {"integrations": sheet_cfg()}, "local-user")
    with prig.client.websocket_connect(f"{tp.px.STREAM_PATH}/{tp.token_for(a)}") as ws:
        ws.send_text(tp.start_msg())
        tp.read_until(ws, tp.is_mark)
        ws.send_text(json.dumps({"event": "stop", "stream_sid": tp.STREAM_SID, "stop": {"reason": "callended"}}))
    deadline = time.time() + 10
    while time.time() < deadline and len(g.rows()) < 2:
        time.sleep(0.1)
    settle()
    assert len(g.rows()) == 2 and g.rows()[0] == cs.HEADER                        # one header, one conversation
    row = g.rows()[1]
    assert row[1] == a["name"] and row[2] == "phone" and row[9] == "Ended early" and row[13]
    cid = row[13]
    time.sleep(0.3)
    settle()
    assert len([r for r in g.rows()[1:] if r[13] == cid]) == 1 and cs.sheet_status(cid) == {"state": "sent"}


# ---------------------------------------------------------------------------------- web: after idle time

def fake_run(agent, cid, *, source=None, test=False):
    return SimpleNamespace(agent=agent, agent_id=agent["id"], user_id="local-user", local=True, is_test=test,
                           conversation_id=cid, source=source)


def test_a_web_conversation_is_written_once_after_it_has_been_idle(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    monkeypatch.setenv("CALLER_SHEET_IDLE_MIN", "5")
    a = make_agent_row(env, integrations=sheet_cfg())
    agent = env.local_db.get("agents", a["id"])
    cid = make_conversation(env, a["id"], form=DETAILS, source="widget")
    t0 = time.time()
    cs.touch(fake_run(agent, cid, source="widget"))
    cs.touch(fake_run(agent, cid, source="widget"))                  # a second turn of the same conversation
    assert cs.sweep_once(t0 + 60) == 0 and g.log == []                # still talking: nothing yet
    assert cs.sweep_once(t0 + 6 * 60) == 1                            # quiet for more than five minutes
    settle()
    assert len(g.rows()) == 2 and g.rows()[1][2] == "widget"
    assert cs.sweep_once(t0 + 12 * 60) == 0                           # it is not written again by the sweep
    n = len(g.writes())
    cs.touch(fake_run(agent, cid, source="widget"))                  # the visitor comes back: nothing changed ...
    cs.sweep_once(time.time() + 10 * 60)
    settle()
    assert len(g.rows()) == 2 and len(g.writes()) == n                # ... so the unchanged row is not rewritten


def test_a_returning_visitor_with_new_details_updates_the_same_row(env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env, integrations=sheet_cfg())
    agent = env.local_db.get("agents", a["id"])
    cid = make_conversation(env, a["id"], form={"name": "Priya"}, source="widget")
    cs.touch(fake_run(agent, cid, source="widget"))
    cs.sweep_once(time.time() + 600)
    settle()
    cs.touch(fake_run(agent, cid, source="widget"))                  # open again: a later save does not write mid-chat
    cm.save(cid, {"name": "Priya", "phone": "9876543210"}, env.local_db, "local-user")
    settle()
    assert g.rows()[1][4] == ""
    cs.sweep_once(time.time() + 600)
    settle()
    assert len(g.rows()) == 2 and g.rows()[1][4] == "9876543210"


def test_phone_calls_and_test_calls_are_not_tracked_for_idle(env, server_key, monkeypatch):
    FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent_row(env, integrations=sheet_cfg())
    agent = env.local_db.get("agents", a["id"])
    cs.touch(fake_run(agent, "c-phone", source="phone"))
    cs.touch(fake_run(agent, "c-test", test=True))
    assert cs._idle == {}
    a2 = make_agent_row(env, integrations=sheet_cfg(include_test=True))
    cs.touch(fake_run(env.local_db.get("agents", a2["id"]), "c-test", test=True))
    assert list(cs._idle) == ["c-test"]


# -------------------------------------------------------------------------------------- a real turn

def put_caller_sheet(client, agent_id, **body):
    return client.put(f"/api/agents/{agent_id}/integrations/caller-sheet", headers=LOCAL, json=body)


def test_a_failing_sheet_never_breaks_or_delays_a_turn(rig, server_key, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Google is down")
    monkeypatch.setattr(ig, "_google_http", boom)
    a = make_agent(rig.client, disclose_ai=False)
    assert put_caller_sheet(rig.client, a["id"], spreadsheet=SHEET_URL, enabled=True).status_code == 200
    started = time.time()
    r = rig.post(a["id"])
    assert r.status_code == 200 and time.time() - started < 5
    [cid] = list(cs._idle)                                            # the turn was tracked for the idle sweep
    assert cs.sweep_once(time.time() + 10 * 60) == 1
    settle()
    entries = ig.get_log(a["id"], local=True)
    assert entries and entries[0]["target"] == "caller_sheet" and entries[0]["status"] == "failed"
    assert cs.sheet_status(cid)["state"] == "failed" and "Could not reach Google" in cs.sheet_status(cid)["error"]


def test_a_real_turn_ends_up_as_a_row(rig, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent(rig.client, disclose_ai=False)
    assert put_caller_sheet(rig.client, a["id"], spreadsheet=SHEET_URL, enabled=True).status_code == 200
    assert rig.post(a["id"]).status_code == 200
    assert cs.sweep_once(time.time() + 10 * 60) == 1
    settle()
    [row] = g.rows()[1:]
    assert row[1] == "Ria" and row[2] == "web" and row[9] == "Answered" and row[13]


# --------------------------------------------------------------------------------------- the settings API

def test_the_settings_are_saved_validated_and_the_view_says_who_to_share_with(app_client, env, server_key):
    a = make_agent(app_client)
    base = f"/api/agents/{a['id']}/integrations"
    first = app_client.get(base, headers=LOCAL).json()["caller_sheet"]
    assert first["enabled"] is False and first["sheet_name"] == "BonClay" and first["include_phone"] is True
    assert first["share_with"] == "logger@demo.iam.gserviceaccount.com" and first["key_source"] == "server"
    assert first["last_write"] is None and "Caller name" in first["columns"] and "Conversation id" not in first["columns"]
    assert "private_key" not in json.dumps(first) and "BEGIN" not in json.dumps(first)

    r = put_caller_sheet(app_client, a["id"], spreadsheet=SHEET_URL, sheet_name="Bookings", include_phone=False, enabled=True)
    assert r.status_code == 200, r.text
    got = r.json()["caller_sheet"]
    assert got["enabled"] and got["spreadsheet_id"] == SHEET_ID and got["sheet_name"] == "Bookings"
    assert got["include_phone"] is False and got["include_test"] is False
    assert put_caller_sheet(app_client, a["id"], sheet_name="").json()["caller_sheet"]["sheet_name"] == "BonClay"
    for body, code in (({"spreadsheet": "nope"}, "INVALID_SPREADSHEET"), ({"sheet_name": "a!b"}, "INVALID_SHEET_NAME")):
        r = put_caller_sheet(app_client, a["id"], **body)
        assert r.status_code == 422 and r.json()["code"] == code, (body, r.text)
    cleared = put_caller_sheet(app_client, a["id"], spreadsheet="").json()["caller_sheet"]
    assert cleared["spreadsheet_id"] == "" and cleared["enabled"] is False
    # the other settings are untouched by this one, and the webhook / per-turn Sheets sections keep their shape
    full = app_client.get(base, headers=LOCAL).json()
    assert full["webhook"]["enabled"] is False and full["sheets"]["enabled"] is False


def test_it_cannot_be_turned_on_without_a_sheet_or_a_key(app_client, env, sa_key):  # noqa: F811
    a = make_agent(app_client)
    r = put_caller_sheet(app_client, a["id"], enabled=True)
    assert r.status_code == 422 and r.json()["code"] == "CALLER_SHEET_INCOMPLETE"
    r = put_caller_sheet(app_client, a["id"], spreadsheet=SHEET_URL, enabled=True)
    assert r.status_code == 422 and r.json()["code"] == "CALLER_SHEET_INCOMPLETE" and "GOOGLE_SERVICE_ACCOUNT_FILE" in r.json()["error"]
    key = app_client.put(f"/api/agents/{a['id']}/integrations/sheets", headers=LOCAL, json={"service_account_json": sa_key.json})
    assert key.status_code == 200                                    # a key pasted for this agent is enough
    assert put_caller_sheet(app_client, a["id"], spreadsheet=SHEET_URL, enabled=True).status_code == 200
    view = app_client.get(f"/api/agents/{a['id']}/integrations", headers=LOCAL).json()["caller_sheet"]
    assert view["key_source"] == "agent" and view["share_with"] == "logger@demo.iam.gserviceaccount.com"


def test_other_users_agents_are_not_reachable(app_client, env):
    for method, path in (("put", "caller-sheet"), ("post", "caller-sheet/test")):
        kw = {"json": {}} if method == "put" else {}
        r = getattr(app_client, method)(f"/api/agents/nope/integrations/{path}", headers=LOCAL, **kw)
        assert r.status_code == 404 and r.json()["code"] == "AGENT_NOT_FOUND", path


def test_the_test_button_writes_one_marked_row_and_updates_it_next_time(app_client, env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("Sheet1",))
    a = make_agent(app_client)
    base = f"/api/agents/{a['id']}/integrations/caller-sheet/test"
    assert app_client.post(base, headers=LOCAL).json()["ok"] is False                   # no sheet yet
    put_caller_sheet(app_client, a["id"], spreadsheet=SHEET_URL)
    out = app_client.post(base, headers=LOCAL).json()
    assert out["ok"] is True and out["row"] == 2 and out["error"] is None
    assert g.rows()[0] == cs.HEADER and "Test row from BonClay" in g.rows()[1][7] and g.rows()[1][13] == cs.TEST_CONVERSATION
    assert app_client.post(base, headers=LOCAL).json()["row"] == 2 and len(g.rows()) == 2   # updated, not added again
    view = app_client.get(f"/api/agents/{a['id']}/integrations", headers=LOCAL).json()["caller_sheet"]
    assert view["last_write"]["status"] == "sent" and view["last_write"]["test"] is True


def test_the_test_button_reports_googles_exact_error(app_client, env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent(app_client)
    put_caller_sheet(app_client, a["id"], spreadsheet=SHEET_URL)
    g.script = [(403, {"error": {"message": "The caller does not have permission"}})]
    out = app_client.post(f"/api/agents/{a['id']}/integrations/caller-sheet/test", headers=LOCAL).json()
    assert out["ok"] is False and out["status"] == 403 and "does not have permission" in out["error"]
    assert "Share the spreadsheet with logger@demo.iam.gserviceaccount.com" in out["hint"]
    view = app_client.get(f"/api/agents/{a['id']}/integrations", headers=LOCAL).json()["caller_sheet"]
    assert view["last_write"]["status"] == "failed"


def test_the_test_button_refuses_offline(app_client, env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch)
    a = make_agent(app_client)
    put_caller_sheet(app_client, a["id"], spreadsheet=SHEET_URL)
    monkeypatch.setenv("BONCLAY_OFFLINE", "1")
    out = app_client.post(f"/api/agents/{a['id']}/integrations/caller-sheet/test", headers=LOCAL).json()
    assert out["ok"] is False and "Offline mode" in out["error"] and g.log == []


# ------------------------------------------------------------------------------------------- Calls page

def test_the_calls_page_shows_what_happened_to_the_sheet(app_client, env, server_key, monkeypatch):
    g = FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent(app_client)
    cid = make_conversation(env, a["id"], form=DETAILS)
    plain = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid, delay=0.3)
    by = {c["id"]: c for c in app_client.get("/api/calls", headers=LOCAL).json()["calls"]}
    assert by[cid]["sheet"] == {"state": "waiting"} and "sheet" not in by[plain]
    settle()
    by = {c["id"]: c for c in app_client.get("/api/calls", headers=LOCAL).json()["calls"]}
    assert by[cid]["sheet"] == {"state": "sent"}
    assert app_client.get(f"/api/calls/{cid}", headers=LOCAL).json()["sheet"] == {"state": "sent"}
    assert "sheet" not in app_client.get(f"/api/calls/{plain}", headers=LOCAL).json()
    g.script = [(403, {"error": {"message": "nope"}})]
    cs._ready.clear()
    cid3 = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid3)
    settle()
    got = app_client.get(f"/api/calls/{cid3}", headers=LOCAL).json()["sheet"]
    assert got["state"] == "failed" and "nope" in got["error"]


def test_the_ledger_is_a_private_file_and_goes_with_the_agent(app_client, env, server_key, monkeypatch):
    import os
    import stat
    FakeSheets(server_key, monkeypatch, tabs=("BonClay",))
    a = make_agent(app_client)
    cid = make_conversation(env, a["id"], form=DETAILS)
    cs.request_write(args_for(a["id"]), cid)
    settle()
    path = ig.secrets_dir() / "caller_sheet.ledger.json"
    assert path.is_file() and (os.name == "nt" or stat.S_IMODE(os.stat(path).st_mode) == 0o600)
    assert "9876543210" not in path.read_text() and "private_key" not in path.read_text()
    cs.reset_state()                                                   # a restart: the ledger is read again
    assert cs.sheet_status(cid) == {"state": "sent"}
    assert app_client.delete(f"/api/agents/{a['id']}", headers=LOCAL).status_code == 200
    assert cs.sheet_status(cid) is None
