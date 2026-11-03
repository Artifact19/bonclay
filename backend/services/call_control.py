"""What an agent may do to its own phone call: end it, or hand the caller to a person.

Two tools of the built-in "call" tool server, offered only on a phone call and run in this process (the other tool
servers are separate processes that cannot touch a live call):

* end_call: hang up after the agent has said goodbye.
* transfer_call: hand over to the agent's transfer number. Offered only when the owner set one for the agent. The model
  never chooses the number: it can only ask for "a person". What happens next is PHONE_TRANSFER_MODE:
    callback (default): the caller is told someone will call back, and the conversation is flagged for a person
                        (`needs_human`, shown on the Calls page) with the transfer number noted on the call record.
    flow:               the audio stream is ended so the Exotel call flow carries on to its next step. Add a Connect
                        step that dials the transfer number after the stream step in the Exotel flow. Not verified
                        against a live Exotel account.

The per-agent transfer number lives in data/agent_call_settings.json (BONCLAY_CALL_SETTINGS_FILE)."""
from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

TOOL_TIMEOUT_S = 6.0
SERVER = "call"
LABEL = "Phone call controls (end the call, hand the caller to a person)"

END_CALL = {
    "tool": "end_call",
    "description": ("Hang up the phone call after your reply. Use it only when the caller has said goodbye or has "
                    "confirmed they need nothing else. Say a short goodbye in the same reply."),
    "schema": {"type": "object", "properties": {"reason": {"type": "string", "maxLength": 120}}, "additionalProperties": False},
}
TRANSFER_CALL = {
    "tool": "transfer_call",
    "description": ("Hand the caller to a person when they ask for one, or when you cannot help. You cannot choose who: "
                    "the number is set by the owner. Tell the caller what will happen next, in the same reply."),
    "schema": {"type": "object", "properties": {"reason": {"type": "string", "maxLength": 200}}, "additionalProperties": False},
}

_lock = threading.RLock()
_sessions: Dict[str, Any] = {}


# ------------------------------------------------------------------------------------------ per-agent settings

def _path() -> str:
    env = os.getenv("BONCLAY_CALL_SETTINGS_FILE", "").strip()
    return env or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                               "agent_call_settings.json")


def _read() -> Dict[str, dict]:
    try:
        with open(_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def transfer_number(agent_id: str) -> Optional[str]:
    n = (_read().get(str(agent_id)) or {}).get("transfer_number")
    return n if isinstance(n, str) and n else None


def set_transfer_number(agent_id: str, number: Optional[str]) -> None:
    with _lock:
        data = _read()
        row = dict(data.get(str(agent_id)) or {})
        if number:
            row["transfer_number"] = number
        else:
            row.pop("transfer_number", None)
        if row:
            data[str(agent_id)] = row
        else:
            data.pop(str(agent_id), None)
        os.makedirs(os.path.dirname(_path()), exist_ok=True)
        tmp = _path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.chmod(tmp, 0o600)
        os.replace(tmp, _path())


def transfer_mode() -> str:
    return "flow" if os.getenv("PHONE_TRANSFER_MODE", "callback").strip().lower() == "flow" else "callback"


# ------------------------------------------------------------------------------------------ live sessions

def register(conversation_id: str, session: Any) -> None:
    with _lock:
        _sessions[str(conversation_id)] = session


def unregister(conversation_id: Optional[str]) -> None:
    if conversation_id:
        with _lock:
            _sessions.pop(str(conversation_id), None)


def session_by_conversation(conversation_id: str) -> Optional[Any]:
    with _lock:
        return _sessions.get(str(conversation_id))


def live_sessions() -> List[Any]:
    with _lock:
        return list(_sessions.values())


def _session_for(run: Any) -> Optional[Any]:
    with _lock:
        cid = getattr(run, "conversation_id", None)
        if cid and str(cid) in _sessions:
            return _sessions[str(cid)]
        agent_id = getattr(run, "agent_id", None)
        live = [s for s in _sessions.values() if getattr(s, "agent", {}).get("id") == agent_id]
        return live[0] if len(live) == 1 else None


def run(tool: str, arguments: dict, run_obj: Any) -> Tuple[bool, Any]:
    """Blocking (called from the tool thread): do `tool` on the live call of this turn. Returns (ok, result)."""
    session = _session_for(run_obj)
    if session is None or getattr(session, "loop", None) is None:
        return False, "this is not a live phone call, so there is nothing to end or transfer"
    try:
        fut = asyncio.run_coroutine_threadsafe(session.control(tool, arguments or {}), session.loop)
        return fut.result(TOOL_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        return False, f"the call control failed ({type(e).__name__})"


def specs_for_run(run_obj: Any, tools_mod: Any) -> List[Any]:
    """The tool specs for a phone turn, bound to `run_obj` so a call reaches the right live call. transfer_call only
    when the agent has a transfer number."""
    defs = [END_CALL]
    if transfer_number(getattr(run_obj, "agent_id", "")):
        defs.append(TRANSFER_CALL)
    out = []
    for d in defs:
        spec = tools_mod.ToolSpec(SERVER, d["tool"], d["description"], d["schema"])
        spec.runner = lambda tool, args, _run=run_obj: run(tool, args, _run)
        out.append(spec)
    return out
