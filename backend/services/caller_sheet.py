"""Caller details in a Google Sheet: one row per conversation, so a clinic or an office owner sees bookings and requests in
a spreadsheet.

What the caller told the agent (services/caller_memory.py: name, phone, date, time, reason, doctor, age, e-mail, other) is
written once when the conversation ends, and the same row is updated in place if the details change afterwards.

When a conversation is "over"
    phone call            the call's WebSocket closed (routers/phone_exotel.py calls `on_phone_call_end`);
    web, widget, WhatsApp, the app's own chat
                          `touch` (called for every stored turn) remembers the conversation; once no turn has come for
                          CALLER_SHEET_IDLE_MIN minutes (default 5) a small background sweep queues the write. A later
                          turn simply makes the conversation "open" again; its next idle moment updates the same row.
    details found later   the small model that reads details in the background (caller_memory.llm_update) may add a name
                          or reason a few seconds after the last turn: `caller_memory.save` tells `on_details_saved`,
                          which queues an update of a conversation that was already written. An update that changes
                          nothing is skipped (a hash of the row is kept).

How it is written (Sheets API v4, the service-account sign-in of services/integrations.py)
    * The sheet (tab) is created when it is missing and a header row is written when the tab is empty.
    * The row of a conversation is found by a last column, "Conversation id" (the number of the row is remembered too, and
      checked against that column before a row is overwritten, so a row the owner sorted or deleted is never clobbered).
      Not found: the row is appended. Found: it is replaced with `values.update`. One row per conversation, always.
    * Values go in with valueInputOption=RAW, so nothing a caller says can become a formula.
    * Writes are jobs of the integrations dispatcher (kind "caller_sheet"): queued, retried with the same backoff on 429
      and 5xx, written to the same delivery log and never run on a call's or a reply's path. Offline mode refuses.
    * The state of every conversation (waiting / sent / failed, the row, the hash) is kept in
      data/integrations/caller_sheet.ledger.json (0600) and shown on the Calls page.

Which key signs in: the one the owner pasted under Google Sheets for this agent when there is one, else the server's
service-account file (GOOGLE_SERVICE_ACCOUNT_FILE, the one Google Calendar uses). The owner shares the sheet with that
service account's e-mail as Editor. The key itself is never shown or logged.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo

try:
    from core import offline
    from services import caller_memory
    from services import integrations as ig
except ImportError:  # pragma: no cover - imported as backend.services.caller_sheet
    from backend.core import offline
    from backend.services import caller_memory
    from backend.services import integrations as ig

log = logging.getLogger("bonclay.caller_sheet")

KIND = "caller_sheet"
EVENT = "conversation.ended"
DEFAULT_TAB = "BonClay"
HEADER = ["Time (IST)", "Agent", "Channel", "Caller name", "Phone", "Date requested", "Time requested", "Reason",
          "Other details", "Outcome", "Call length", "Conversation link", "Receipt id", "Conversation id"]
ID_COLUMN = "N"                      # the 14th column: where the conversation's id is, to find its row again
LAST_COLUMN = ID_COLUMN
IST = ZoneInfo("Asia/Kolkata")
OUTCOMES = {"booked": "Booked", "needs_approval": "Meeting requested", "declined": "Meeting declined",
            "handed_off": "Handed to a person", "couldnt_answer": "Could not answer", "ended_early": "Ended early",
            "answered": "Answered"}
CHANNELS = {"phone": "phone", "widget": "widget", "whatsapp": "WhatsApp"}
LEDGER_MAX = 2000
IDLE_MAX = 2000
TEST_CONVERSATION = "bonclay-test-row"


# ------------------------------------------------------------------------------------------------------ settings

def _flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def idle_seconds() -> float:
    return max(0.0, _float_env("CALLER_SHEET_IDLE_MIN", 5.0) * 60.0)


def end_delay_seconds() -> float:
    """After a call ends, wait this long before writing: the background reading of the last words may still add details."""
    return max(0.0, _float_env("CALLER_SHEET_END_DELAY_S", 4.0))


def dashboard_url() -> str:
    return (os.getenv("BONCLAY_DASHBOARD_URL", "").strip() or "http://localhost:3000").rstrip("/")


def conversation_link(conversation_id: str) -> str:
    return f"{dashboard_url()}/dashboard/calls?call={quote(str(conversation_id), safe='')}"


def clean_tab(value: Any) -> str:
    """The tab name; empty means the default "BonClay"."""
    return ig.clean_sheet_name(value) or DEFAULT_TAB


def settings_of(raw: Any) -> dict:
    return ig.normalize_cfg(raw)["caller_sheet"]


# ------------------------------------------------------------------------------------------------- credentials

def _server_key_path() -> str:
    try:
        from services import gcal
    except ImportError:  # pragma: no cover
        from backend.services import gcal
    return gcal.key_path()


def server_credentials() -> Optional[dict]:
    """The service account of GOOGLE_SERVICE_ACCOUNT_FILE (the one Google Calendar uses), validated; None when absent."""
    try:
        with open(_server_key_path(), "r", encoding="utf-8") as f:
            return ig.parse_service_account(json.load(f))
    except (OSError, ValueError, ig.IntegrationError):
        return None


def credentials_for(agent_id: str, local: bool, enc: Optional[str]) -> Tuple[Optional[dict], str]:
    """(credentials, "agent" | "server"): the key pasted for this agent, else the server's file. (None, "") when neither."""
    own = ig.load_secrets(agent_id, local=local, enc=enc).get("service_account")
    if own:
        return own, "agent"
    server = server_credentials()
    return (server, "server") if server else (None, "")


def share_with(agent_id: str, cfg: dict, local: bool, enc: Optional[str] = None) -> Tuple[str, str]:
    """(the e-mail to share the sheet with, where the key comes from). Never the key."""
    sh = cfg["sheets"]
    if sh.get("credentials_set") and sh.get("service_account_email"):
        return sh["service_account_email"], "agent"
    server = server_credentials()
    return (server["client_email"], "server") if server else ("", "")


# --------------------------------------------------------------------------------------------------- the ledger

_ledger_lock = threading.RLock()
_ledger: Optional[dict] = None


def _ledger_path():
    return ig.secrets_dir() / "caller_sheet.ledger.json"


def _ledger_get() -> dict:
    global _ledger
    if _ledger is None:
        data: dict = {}
        try:
            path = _ledger_path()
            if path.is_file():
                data = json.loads(path.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
        _ledger = {"rows": dict(data.get("rows") or {}), "last": dict(data.get("last") or {})}
    return _ledger


def _ledger_save() -> None:
    try:
        led = _ledger_get()
        if len(led["rows"]) > LEDGER_MAX:
            for key in sorted(led["rows"], key=lambda k: led["rows"][k].get("at", ""))[:len(led["rows"]) - LEDGER_MAX]:
                led["rows"].pop(key, None)
        ig._write_private(_ledger_path(), json.dumps(led).encode("utf-8"))
    except Exception as e:  # noqa: BLE001 - the ledger must never break a write
        log.warning("could not store the caller sheet ledger: %s", e.__class__.__name__)


def reset_state() -> None:
    """Tests: forget everything held in memory (the ledger file is read again on next use)."""
    global _ledger
    with _ledger_lock:
        _ledger = None
    with _state_lock:
        _state.clear()
        _idle.clear()
        _known_args.clear()
    _ready.clear()


def ledger_entry(conversation_id: str) -> Optional[dict]:
    with _ledger_lock:
        e = _ledger_get()["rows"].get(conversation_id)
        return dict(e) if e else None


def _ledger_set(conversation_id: str, agent_id: str, **fields: Any) -> None:
    with _ledger_lock:
        led = _ledger_get()
        entry = {**(led["rows"].get(conversation_id) or {}), "agent_id": agent_id, **fields, "at": ig._iso()}
        led["rows"][conversation_id] = entry
        if fields.get("status") in ("sent", "failed"):
            led["last"][agent_id] = {"status": entry["status"], "error": entry.get("error"), "at": entry["at"],
                                     "test": conversation_id == TEST_CONVERSATION}
        _ledger_save()


def sheet_status(conversation_id: str) -> Optional[dict]:
    """For the Calls page: {"state": "sent" | "waiting" | "failed", "error"?} or None when this conversation was never
    queued for a sheet."""
    e = ledger_entry(conversation_id)
    if not e:
        return None
    state = {"sent": "sent", "waiting": "waiting", "failed": "failed"}.get(e.get("status"), "waiting")
    out = {"state": state}
    if state == "failed" and e.get("error"):
        out["error"] = str(e["error"])[:200]
    return out


def last_write(agent_id: str) -> Optional[dict]:
    with _ledger_lock:
        e = _ledger_get()["last"].get(agent_id)
        return dict(e) if e else None


def forget_agent(agent_id: str) -> None:
    with _ledger_lock:
        led = _ledger_get()
        for k in [k for k, v in led["rows"].items() if v.get("agent_id") == agent_id]:
            led["rows"].pop(k, None)
        led["last"].pop(agent_id, None)
        _ledger_save()


def describe(agent_id: str, cfg: dict, local: bool, enc: Optional[str] = None) -> dict:
    """What the settings card shows besides the saved settings: who to share with, and how the last write went."""
    email, source = share_with(agent_id, cfg, local, enc)
    return {"share_with": email, "key_source": source or None, "last_write": last_write(agent_id),
            "columns": list(HEADER[:-1])}


# ------------------------------------------------------------------------------------------------------- the row

def _duration_label(seconds: Optional[int]) -> str:
    if not seconds or seconds < 0:
        return ""
    m, s = divmod(int(seconds), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def _ist(value: Any) -> str:
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        d = datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(IST).strftime("%Y-%m-%d %H:%M")


def _mask_email(email: str) -> str:
    user, _, domain = email.partition("@")
    return (user[:1] + "***@" + domain) if domain else "***"


def other_details(form: Dict[str, str], include_contact: bool) -> str:
    """Everything the form holds that has no column of its own, as `Label: value; ...`."""
    parts: List[str] = []
    for key, label in (("doctor", "Doctor"), ("role", "Role"), ("age", "Age")):
        if form.get(key):
            parts.append(f"{label}: {form[key]}")
    if form.get("email"):
        parts.append("Email: " + (form["email"] if include_contact else _mask_email(form["email"])))
    if form.get("other"):
        parts.append(str(form["other"]))
    return "; ".join(parts)


def build_row(*, form: Dict[str, str], agent_name: str, channel: str, outcome_key: str, started_at: Any,
              duration_s: Optional[int], conversation_id: str, receipt_id: str = "", include_phone: bool = True) -> List[str]:
    """One row, in the order of HEADER. The caller's number is written in full only when `include_phone` is on."""
    phone = form.get("phone") or ""
    if phone and not include_phone:
        phone = caller_memory.mask_phone(phone) or ""
    return [_ist(started_at), agent_name, channel, form.get("name") or "", phone, form.get("date") or "",
            form.get("time") or "", form.get("reason") or "", other_details(form, include_phone),
            OUTCOMES.get(outcome_key, outcome_key), _duration_label(duration_s), conversation_link(conversation_id),
            receipt_id, conversation_id]


def row_hash(row: List[Any]) -> str:
    return hashlib.sha256(json.dumps(row, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]


# ------------------------------------------------------------------------------------------------- Sheets API

_ready: set = set()            # (spreadsheet id, tab) known to exist and to have a header row
_ready_lock = threading.Lock()


def _api(key: str, creds: dict, method: str, url: str, body: Any = None) -> Any:
    """One Sheets call with the service account's token; refreshes an expired token once. Raises ig.SheetsError."""
    for attempt in (0, 1):
        token = ig._access_token(key, creds)
        try:
            status, data = ig._google_http(method, url, headers={"Authorization": f"Bearer {token}"}, json_body=body)
        except Exception as e:  # noqa: BLE001
            raise ig.SheetsError(ig._clean_error(f"Could not reach Google Sheets: {e.__class__.__name__}"),
                                 retryable=True) from e
        if status == 401 and attempt == 0:
            ig._token_cache.pop(key, None)
            continue
        if 200 <= status < 300:
            return data
        raise ig._sheets_error(status, data, creds["client_email"], 30.0 if status == 429 else None)
    raise ig.SheetsError("Google Sheets did not accept the sign-in.", status=401, hint="Check the service-account key.")


def _quote_tab(tab: str) -> str:
    return "'" + tab.replace("'", "''") + "'"


def _values_url(spreadsheet_id: str, a1: str, suffix: str = "") -> str:
    return f"{ig.SHEETS_API}/{quote(spreadsheet_id, safe='')}/values/{quote(a1, safe='')}{suffix}"


def _ensure_tab(key: str, creds: dict, spreadsheet_id: str, tab: str) -> None:
    """The tab exists (created when it does not) and has a header row (written when row 1 is empty)."""
    cache_key = (spreadsheet_id, tab)
    with _ready_lock:
        if cache_key in _ready:
            return
    meta = _api(key, creds, "GET", f"{ig.SHEETS_API}/{quote(spreadsheet_id, safe='')}?fields=sheets.properties.title")
    titles = [((s or {}).get("properties") or {}).get("title") for s in (meta or {}).get("sheets") or []]
    if tab not in titles:
        _api(key, creds, "POST", f"{ig.SHEETS_API}/{quote(spreadsheet_id, safe='')}:batchUpdate",
             {"requests": [{"addSheet": {"properties": {"title": tab}}}]})
    first = _api(key, creds, "GET", _values_url(spreadsheet_id, f"{_quote_tab(tab)}!A1:{LAST_COLUMN}1"))
    if not (first or {}).get("values"):
        _api(key, creds, "PUT", _values_url(spreadsheet_id, f"{_quote_tab(tab)}!A1:{LAST_COLUMN}1",
                                            "?valueInputOption=RAW"),
             {"majorDimension": "ROWS", "values": [HEADER]})
    with _ready_lock:
        _ready.add(cache_key)


def _find_row(key: str, creds: dict, spreadsheet_id: str, tab: str, conversation_id: str,
              known_row: Optional[int]) -> Optional[int]:
    """The row that holds this conversation: the remembered one when its id column still matches, else a search."""
    if known_row:
        got = _api(key, creds, "GET", _values_url(spreadsheet_id, f"{_quote_tab(tab)}!{ID_COLUMN}{known_row}"))
        cells = (got or {}).get("values") or []
        if cells and cells[0] and str(cells[0][0]) == conversation_id:
            return known_row
    got = _api(key, creds, "GET", _values_url(spreadsheet_id, f"{_quote_tab(tab)}!{ID_COLUMN}:{ID_COLUMN}"))
    for i, cells in enumerate((got or {}).get("values") or [], start=1):
        if cells and str(cells[0]) == conversation_id:
            return i
    return None


def _row_from_range(updated_range: Any) -> Optional[int]:
    """`'BonClay'!A5:N5` -> 5."""
    import re
    m = re.search(r"!\$?[A-Z]+(\d+)", str(updated_range or ""))
    return int(m.group(1)) if m else None


def write_row(key: str, creds: dict, spreadsheet_id: str, tab: str, conversation_id: str, row: List[Any],
              known_row: Optional[int] = None) -> int:
    """Put the conversation's row in the sheet: update its row when there is one, else append. Returns the row number.
    Raises ig.SheetsError."""
    try:
        _ensure_tab(key, creds, spreadsheet_id, tab)
        at = _find_row(key, creds, spreadsheet_id, tab, conversation_id, known_row)
        if at:
            _api(key, creds, "PUT", _values_url(spreadsheet_id, f"{_quote_tab(tab)}!A{at}:{LAST_COLUMN}{at}",
                                                "?valueInputOption=RAW"), {"majorDimension": "ROWS", "values": [row]})
            return at
        out = _api(key, creds, "POST", _values_url(spreadsheet_id, f"{_quote_tab(tab)}!A1", ":append"
                                                   "?valueInputOption=RAW&insertDataOption=INSERT_ROWS"),
                   {"majorDimension": "ROWS", "values": [row]})
        return _row_from_range(((out or {}).get("updates") or {}).get("updatedRange")) or 0
    except ig.SheetsError as e:
        if e.status in (400, 404):   # the tab or the sheet changed under us: check it again next time
            with _ready_lock:
                _ready.discard((spreadsheet_id, tab))
        raise


# --------------------------------------------------------------------------------------------- the conversation

def _backend_module():
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "local_db"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "local_db"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


class ContextError(Exception):
    pass


def load_context(job: "ig.Job") -> dict:
    """The facts of a conversation, read when the job runs (so a retry or a late update has the newest): its row, the
    caller details form, the outcome. Blocking. Raises ContextError. Tests replace this."""
    try:
        from routers import conversations as conv
    except ImportError:  # pragma: no cover
        from backend.routers import conversations as conv
    cid = job.payload["conversation_id"]
    b = _backend_module()
    if job.local:
        convs, by_conv, _names = conv._load_local(b, job.user_id, None, "", cid)
    else:
        convs, by_conv, _names = conv._load_cloud(b, job.user_id, None, "", cid)
    row = next((c for c in convs if c["id"] == cid), None)
    if row is None:
        raise ContextError("the conversation was not found")
    turns = by_conv.get(cid, [])
    form = caller_memory.load(cid, b.local_db if job.local else None, job.user_id)
    outcome =conv.derive_outcome(row.get("source"), bool(row.get("needs_human")), turns, conv._meeting_lookup())
    duration = job.payload.get("duration_s") or conv._duration_s(row, turns)
    receipt = ""
    agent_turns = [t for t in turns if t.get("role") == "agent" and t.get("id")]
    if agent_turns:
        receipt = ig._find_receipt_id(str(agent_turns[-1]["id"])) or ""
    return {"form": form, "outcome_key": outcome["key"], "started_at": row.get("created_at") or ig._iso(),
            "channel": CHANNELS.get(row.get("source") or "", "web"), "duration_s": duration, "receipt_id": receipt,
            "has_turns": bool(turns), "is_test": bool(row.get("is_test"))}


# ----------------------------------------------------------------------------------------------------- queueing

_state_lock = threading.Lock()
_state: Dict[str, str] = {}          # conversation id -> "queued" | "running" | "dirty" (changed while it was running)
_known_args: "collections.OrderedDict[str, dict]" = collections.OrderedDict()   # how a written conversation is written again


def _make_job(args: dict, conversation_id: str, duration_s: Optional[int] = None) -> "ig.Job":
    payload = {"conversation_id": conversation_id}
    if duration_s:
        payload["duration_s"] = int(duration_s)
    return ig.Job(KIND, args["agent_id"], args["user_id"], bool(args["local"]), args.get("enc"), EVENT, payload,
                  args.get("agent_name", ""),
                  sheet={"spreadsheet_id": args["spreadsheet_id"], "sheet_name": args["sheet_name"],
                         "include_phone": "1" if args["include_phone"] else "0"})


def _args_from_agent(agent: dict, *, user_id: str, local: bool, agent_id: Optional[str] = None) -> Optional[dict]:
    """The write's settings from an agent row; None when this agent does not send caller details to a sheet."""
    raw = (agent or {}).get("integrations")
    if not raw:
        return None
    cs = settings_of(raw)
    if not (cs["enabled"] and cs["spreadsheet_id"]):
        return None
    return {"agent_id": str(agent_id or agent.get("id") or ""), "user_id": str(user_id), "local": local,
            "enc": raw.get("secrets_enc") if isinstance(raw, dict) else None, "agent_name": str(agent.get("name") or ""),
            "spreadsheet_id": cs["spreadsheet_id"], "sheet_name": cs["sheet_name"] or DEFAULT_TAB,
            "include_phone": bool(cs["include_phone"]), "include_test": bool(cs["include_test"])}


def request_write(args: dict, conversation_id: str, *, delay: float = 0.0, duration_s: Optional[int] = None) -> bool:
    """Queue the write of one conversation. Many requests for the same conversation while one is waiting are one write
    (it reads the newest facts when it runs); one made while it is running makes it run once more afterwards. Never
    raises. Returns whether a job was queued."""
    try:
        with _state_lock:
            st = _state.get(conversation_id)
            if st == "queued":
                return False
            if st in ("running", "dirty"):
                _state[conversation_id] = "dirty"
                return False
            _state[conversation_id] = "queued"
        with _state_lock:
            _known_args[conversation_id] = args
            _known_args.move_to_end(conversation_id)
            while len(_known_args) > LEDGER_MAX:
                _known_args.popitem(last=False)
        _ledger_set(conversation_id, args["agent_id"], status="waiting", error=None)
        if not ig.dispatcher.submit(_make_job(args, conversation_id, duration_s), delay):
            with _state_lock:
                _state.pop(conversation_id, None)
            _ledger_set(conversation_id, args["agent_id"], status="failed", error="the queue is full")
            log.warning("integrations queue is full: dropped a caller sheet row")
            return False
        return True
    except Exception as e:  # noqa: BLE001 - never break a call or a turn
        log.warning("could not queue a caller sheet row: %s: %s", e.__class__.__name__, e)
        with _state_lock:
            _state.pop(conversation_id, None)
        return False


def on_phone_call_end(agent: dict, *, user_id: str, local: bool, conversation_id: Optional[str],
                      duration_s: Optional[int] = None) -> None:
    """A phone call is over (routers/phone_exotel.py, `finally` of the call). Queues exactly one write, a few seconds
    later so details the small model is still reading are in it. Never blocks, never raises."""
    try:
        if not conversation_id:
            return
        args = _args_from_agent(agent, user_id=user_id, local=local)
        if args is None:
            return
        request_write(args, conversation_id, delay=end_delay_seconds(), duration_s=duration_s)
    except Exception as e:  # noqa: BLE001
        log.warning("caller sheet: could not handle the end of a call: %s", e.__class__.__name__)


# ---- conversations that are not phone calls: written after they have been idle

_idle: Dict[str, dict] = {}          # conversation id -> {"last": time, "args": {...}}
_sweeper: Optional[threading.Thread] = None
SWEEP_S = 20.0


def touch(run: Any) -> None:
    """Called for every stored turn (backend._turn_store). A conversation of a web visitor, WhatsApp or the app is
    remembered with the time of its last turn; the sweep writes it once it has been quiet for CALLER_SHEET_IDLE_MIN
    minutes. A phone call is written when it ends, so it is not tracked here. One dict lookup for an agent without the
    setting. Never raises."""
    try:
        agent = getattr(run, "agent", None) or {}
        if not agent.get("integrations") or getattr(run, "source", None) == "phone":
            return
        cid = getattr(run, "conversation_id", None)
        if not cid:
            return
        args = _args_from_agent(agent, user_id=str(run.user_id), local=bool(getattr(run, "local", True)),
                                agent_id=str(run.agent_id))
        if args is None or (getattr(run, "is_test", False) and not args["include_test"]):
            return
        with _state_lock:
            _idle[cid] = {"last": time.time(), "args": args}
            if len(_idle) > IDLE_MAX:
                for old in sorted(_idle, key=lambda k: _idle[k]["last"])[:len(_idle) - IDLE_MAX]:
                    _idle.pop(old, None)
        _ensure_sweeper()
    except Exception as e:  # noqa: BLE001
        log.warning("caller sheet: could not track a conversation: %s", e.__class__.__name__)


def sweep_once(now: Optional[float] = None) -> int:
    """Queue the conversations that have been idle long enough. Returns how many were queued."""
    now = time.time() if now is None else now
    limit = idle_seconds()
    with _state_lock:
        due = [(cid, e["args"]) for cid, e in _idle.items() if now - e["last"] >= limit]
        for cid, _ in due:
            _idle.pop(cid, None)
    return sum(1 for cid, args in due if request_write(args, cid))


def _ensure_sweeper() -> None:
    global _sweeper
    if _sweeper is not None and _sweeper.is_alive():
        return
    with _state_lock:
        if _sweeper is not None and _sweeper.is_alive():
            return
        _sweeper = threading.Thread(target=_sweep_loop, name="bonclay-caller-sheet-sweep", daemon=True)
        _sweeper.start()


def _sweep_loop() -> None:
    while True:
        time.sleep(SWEEP_S)
        try:
            sweep_once()
        except Exception as e:  # noqa: BLE001
            log.warning("caller sheet sweep failed: %s", e.__class__.__name__)


def on_details_saved(conversation_id: Optional[str]) -> None:
    """`caller_memory.save` stored a conversation's form. When that conversation was written to a sheet already and is no
    longer open (a web visitor still chatting is written once it goes quiet), queue an update: the write is skipped if
    the row would not change. Never blocks, never raises."""
    try:
        if not conversation_id:
            return
        entry = ledger_entry(conversation_id)
        if not entry or entry.get("status") not in ("sent", "failed"):
            return
        with _state_lock:
            if conversation_id in _idle:
                return
        with _state_lock:
            args = _known_args.get(conversation_id)
        if args is None:      # after a restart: the update is skipped
            return
        request_write(args, conversation_id, delay=end_delay_seconds())
    except Exception as e:  # noqa: BLE001
        log.debug("caller sheet: details listener skipped: %s", e.__class__.__name__)


# ---------------------------------------------------------------------------------------------- the job itself

def _finish(conversation_id: str, requeued: bool) -> None:
    """Release the conversation's slot, or keep it when the job was queued again (a retry, a rate-limit wait); run once
    more when the conversation changed while it was being written."""
    with _state_lock:
        st = _state.get(conversation_id)
        if requeued:
            _state[conversation_id] = "queued"
            return
        _state.pop(conversation_id, None)
        args = _known_args.get(conversation_id)
    if st == "dirty" and args is not None:
        request_write(args, conversation_id, delay=1.0)


def process(d: "ig.Dispatcher", job: "ig.Job") -> None:
    """Run one caller-sheet job (called by the dispatcher worker). Never raises."""
    cid = job.payload["conversation_id"]
    with _state_lock:
        if _state.get(cid) != "dirty":
            _state[cid] = "running"
    requeued = False
    try:
        requeued = _process(d, job, cid)
    except Exception as e:  # noqa: BLE001
        log.exception("caller sheet write crashed")
        _fail(job, cid, f"internal error: {e.__class__.__name__}")
    finally:
        _finish(cid, requeued)


def _fail(job: "ig.Job", cid: str, error: str, *, code: Optional[int] = None, ms: Optional[int] = None) -> None:
    _ledger_set(cid, job.agent_id, status="failed", error=error)
    ig.log_update(job.agent_id, job.user_id, job.local, ig._entry(job, "failed", code=code, latency_ms=ms, error=error))


def _process(d: "ig.Dispatcher", job: "ig.Job", cid: str) -> bool:
    """True when the job was queued again."""
    if offline.enabled():
        _fail(job, cid, offline.reason("Google Sheets"))
        return False
    creds, source = credentials_for(job.agent_id, job.local, job.enc)
    if not creds:
        _fail(job, cid, "no Google service-account key: paste one under Google Sheets, or set "
                        "GOOGLE_SERVICE_ACCOUNT_FILE on the server")
        return False
    try:
        ctx = load_context(job)
    except ContextError as e:
        _fail(job, cid, str(e))
        return False
    if not ctx["has_turns"] and not ctx["form"] and ctx["channel"] != "phone":   # a call that connected is always a row
        with _ledger_lock:
            _ledger_get()["rows"].pop(cid, None)
            _ledger_save()
        return False
    sheet = job.sheet
    row = build_row(form=ctx["form"], agent_name=job.agent_name, channel=ctx["channel"], outcome_key=ctx["outcome_key"],
                    started_at=ctx["started_at"], duration_s=ctx["duration_s"], conversation_id=cid,
                    receipt_id=ctx["receipt_id"], include_phone=sheet.get("include_phone") == "1")
    digest = row_hash(row)
    entry = ledger_entry(cid) or {}
    same_place = entry.get("spreadsheet_id") == sheet["spreadsheet_id"] and entry.get("tab") == sheet["sheet_name"]
    if same_place and entry.get("hash") == digest and entry.get("row"):
        _ledger_set(cid, job.agent_id, status="sent", error=None)   # nothing changed: nothing to write
        return False
    wait = ig._sheets_pace(creds["client_email"])
    if wait:
        if job.rate_waits >= 20 or not d.submit(job, wait):
            _fail(job, cid, "Sheets write limit reached")
            return False
        job.rate_waits += 1
        return True
    started = time.time()
    key = "caller-sheet:" + (job.agent_id if source == "agent" else "server")
    try:
        at = write_row(key, creds, sheet["spreadsheet_id"], sheet["sheet_name"], cid, row,
                       entry.get("row") if same_place else None)
    except ig.SheetsError as e:
        ms = int((time.time() - started) * 1000)
        before = job.attempt
        ig._retry_or_fail(d, job, code=e.status, latency_ms=ms, error=e.message + (f" {e.hint}" if e.hint else ""),
                          retryable=e.retryable, wait=e.wait)
        if job.attempt == before:   # no retry left or not retryable: it failed
            _ledger_set(cid, job.agent_id, status="failed", error=(e.message + (f" {e.hint}" if e.hint else ""))[:300])
            return False
        return True
    ms = int((time.time() - started) * 1000)
    _ledger_set(cid, job.agent_id, status="sent", error=None, row=at, hash=digest, spreadsheet_id=sheet["spreadsheet_id"],
                tab=sheet["sheet_name"])
    ig.log_update(job.agent_id, job.user_id, job.local, ig._entry(job, "delivered", code=200, latency_ms=ms))
    return False


# ---------------------------------------------------------------------------------------------- the Test button

def check_connection(agent_id: str, user_id: str, local: bool, enc: Optional[str], cfg: dict, agent_name: str) -> dict:
    """The "Test" button: write (or update) one clearly marked row and say exactly what Google answered. Blocking."""
    cs = cfg["caller_sheet"]
    if offline.enabled():
        return {"ok": False, "error": offline.reason("Google Sheets"), "hint": "", "status": None, "row": None}
    creds, src = credentials_for(agent_id, local, enc)
    if not creds or not cs["spreadsheet_id"]:
        return {"ok": False, "error": "Add the spreadsheet link first (and a service-account key, here or on the "
                                      "server).", "hint": "", "status": None, "row": None}
    tab = cs["sheet_name"] or DEFAULT_TAB
    row = [_ist(ig._iso()), agent_name, "test", "BonClay test", "", "", "", "Test row from BonClay. You can delete it.",
           "", "Answered", "", conversation_link("test"), "", TEST_CONVERSATION]
    job = ig.Job(KIND, agent_id, user_id, local, enc, "test.ping", {"conversation_id": TEST_CONVERSATION}, agent_name)
    entry = ledger_entry(TEST_CONVERSATION) or {}
    key = "caller-sheet:" + (agent_id if src == "agent" else "server")
    started = time.time()
    try:
        at = write_row(key, creds, cs["spreadsheet_id"], tab, TEST_CONVERSATION, row,
                       entry.get("row") if entry.get("tab") == tab else None)
    except ig.SheetsError as e:
        ig.log_update(agent_id, user_id, local, ig._entry(job, "failed", code=e.status,
                                                          latency_ms=int((time.time() - started) * 1000), error=e.message))
        _ledger_set(TEST_CONVERSATION, agent_id, status="failed", error=e.message[:300])
        return {"ok": False, "error": e.message, "hint": e.hint, "status": e.status, "row": None}
    ig.log_update(agent_id, user_id, local, ig._entry(job, "delivered", code=200,
                                                      latency_ms=int((time.time() - started) * 1000)))
    _ledger_set(TEST_CONVERSATION, agent_id, status="sent", error=None, row=at, tab=tab)
    return {"ok": True, "error": None, "hint": "", "status": 200, "row": at}


caller_memory.add_save_listener(on_details_saved)
