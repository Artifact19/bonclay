"""A local fallback model for background language-model jobs (the question index, the key-fact sheet).

Those jobs normally use the hosted model (Groq) and share its free quota with live replies. When Groq answers 429, has
used its daily quota, or is not configured, they fall back to a small model on this Mac: BACKGROUND_LOCAL_MODEL (default
mlx-community/gemma-4-e4b-it-4bit, which writes far better Hindi than the Llama model used for offline live replies).
Live replies are never touched: they keep LOCAL_LLM_MODEL and the provider chain in services/llm.py.

The local model runs only when all of these hold; otherwise the job waits and asks again (`run_local`):
  - the app is idle: no conversation turn started in the last BACKGROUND_LLM_IDLE_S (60) seconds and no phone call is
    live (core/activity.py; the same rule as the answer audit's live-traffic check),
  - memory allows it (core/memory_guard.fits("background_llm"): about 6.5 GB free; MEMORY_NEED_BACKGROUND_LLM_MB),
  - the model is downloaded and the MLX environment exists.
The model is a separate process: it is stopped BACKGROUND_LLM_UNLOAD_S (300) seconds after its last use, as soon as a
conversation turn starts, and by `release()` at the end of a job, so it never holds memory during a call.

Settings (backend/.env.example):
    BACKGROUND_LLM_FALLBACK=local|off   default local
    BACKGROUND_LOCAL_MODEL=...          default mlx-community/gemma-4-e4b-it-4bit
    BACKGROUND_LLM_IDLE_S=60            quiet time required before the model starts
    BACKGROUND_LLM_UNLOAD_S=300         idle time after which the model is unloaded
    BACKGROUND_LLM_WAIT_S=1800          longest a job waits for idle + memory before it gives up (it is retried later)
    BACKGROUND_LLM_POLL_S=15            seconds between two checks while waiting
    BACKGROUND_LLM_FORCE=on             use the local model first (scripts/rechunk_knowledge.py --local sets it)
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

try:
    from core import activity, memory_guard
except ImportError:  # pragma: no cover - imported as backend.services.background_llm
    from backend.core import activity, memory_guard

from . import llm

log = logging.getLogger("bonclay.background_llm")

DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
ENGINE = "background_llm"          # the memory guard's name for it

STRICT_JSON_NOTE = ("Reply with exactly one JSON object and nothing else: no code fences, no comments, no text before or "
                    "after it. Use double quotes for every key and string. Write Hindi in Devanagari script.")


class Busy(RuntimeError):
    """The local model could not run within the wait: the app stayed busy or memory stayed short."""


# ----------------------------------------------------------------------------------------------------- settings

def _env_float(name: str, default: float, low: float = 0.0) -> float:
    try:
        return max(low, float(os.getenv(name, "") or default))
    except ValueError:
        return default


def mode() -> str:
    return "off" if os.getenv("BACKGROUND_LLM_FALLBACK", "local").strip().lower() in ("off", "0", "false", "no") else "local"


def model_name() -> str:
    return (os.getenv("BACKGROUND_LOCAL_MODEL") or "").strip() or DEFAULT_MODEL


def forced() -> bool:
    return os.getenv("BACKGROUND_LLM_FORCE", "").strip().lower() in ("on", "1", "true", "yes")


def unload_s() -> float:
    return _env_float("BACKGROUND_LLM_UNLOAD_S", 300.0, 1.0)


def wait_s() -> float:
    return _env_float("BACKGROUND_LLM_WAIT_S", 1800.0)


def poll_s() -> float:
    return _env_float("BACKGROUND_LLM_POLL_S", 15.0, 0.01)


def unavailable_reason() -> Optional[str]:
    """Why the local fallback cannot be used at all (None when it can, busy or not)."""
    if mode() == "off":
        return "BACKGROUND_LLM_FALLBACK=off"
    return llm.background_unavailable_reason(model_name())


def local_ready() -> bool:
    return unavailable_reason() is None


def conditions() -> Tuple[bool, str]:
    """(may the model run now, why not). Idle app, no live call, enough free memory."""
    if activity.call_live():
        return False, "a phone call is live"
    if not activity.is_idle():
        return False, "a conversation turn happened in the last %d s" % int(activity.idle_s())
    if llm.background_running():
        return True, ""      # already loaded: its memory is already counted
    ok, avail, required = memory_guard.fits(ENGINE)
    if not ok:
        return False, "only %.1f GB of memory is free, %.1f GB needed" % ((avail or 0) / 1024, required / 1024)
    return True, ""


# ------------------------------------------------------------------------------------------------ unloading

_lock = threading.Lock()
_last_used = 0.0
_timer: Optional[threading.Timer] = None


def release() -> bool:
    """Unload the model now (end of a job). True when it was loaded."""
    global _timer
    with _lock:
        if _timer is not None:
            _timer.cancel()
            _timer = None
    return llm.unload_background()


def _idle_check() -> None:
    global _timer
    with _lock:
        _timer = None
        left = unload_s() - (time.monotonic() - _last_used)
    if left > 0.5:
        _arm(left)
    else:
        llm.unload_background()


def _arm(delay: float) -> None:
    global _timer
    with _lock:
        if _timer is not None:
            _timer.cancel()
        _timer = threading.Timer(max(0.1, delay), _idle_check)
        _timer.daemon = True
        _timer.start()


def _touch() -> None:
    global _last_used
    _last_used = time.monotonic()
    _arm(unload_s())


def _on_turn() -> None:
    """A conversation turn started: free the memory now, off the caller's thread (stopping a process takes a moment)."""
    if llm.background_running():
        threading.Thread(target=release, name="background-llm-unload", daemon=True).start()


activity.on_turn(_on_turn)


# ------------------------------------------------------------------------------------------------ the call

def strict_messages(messages: List[dict]) -> List[dict]:
    """The same messages with the strict-JSON note added to the last system message."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "system":
            m["content"] = (m.get("content") or "") + "\n\n" + STRICT_JSON_NOTE
            return out
    return [{"role": "system", "content": STRICT_JSON_NOTE}, *out]


def run_local(messages: List[dict], *, max_tokens: int = 800, temperature: float = 0.2, json_mode: bool = True,
              sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
              chat: Optional[Callable[..., Any]] = None, wait: Optional[float] = None, on_wait: Optional[Callable[[str], None]] = None):
    """Run `messages` on the local model once the app is idle and memory allows it. Blocks (a background thread's job):
    polls every BACKGROUND_LLM_POLL_S seconds for at most `wait` (default BACKGROUND_LLM_WAIT_S) and raises Busy when the
    conditions never held. A request cut short because a turn started meanwhile (the model was unloaded under it) is waited
    out and made again. `chat` replaces the model call (tests): chat(messages, model=, json_mode=, temperature=, max_tokens=).
    Returns a services.llm.ChatResult."""
    reason = unavailable_reason()
    if reason:
        raise Busy("the local background model is unavailable: " + reason)
    call = chat or llm.background_chat
    msgs = strict_messages(messages) if json_mode else messages
    deadline = clock() + (wait_s() if wait is None else wait)
    why = ""
    while True:
        ok, why = conditions()
        if ok:
            try:
                res = call(msgs, model=model_name(), json_mode=json_mode, temperature=temperature, max_tokens=max_tokens)
                _touch()
                return res
            except Exception as e:  # noqa: BLE001
                if conditions()[0]:
                    raise
                why = "interrupted: " + why_not_now()
                log.info("background model request interrupted (%s); waiting", str(e)[:80])
        if clock() >= deadline:
            raise Busy("the local model could not run: " + (why or "busy"))
        if on_wait:
            on_wait(why)
        log.info("background model waits: %s", why)
        sleep(poll_s())


def why_not_now() -> str:
    return conditions()[1] or "ready"


# ------------------------------------------------------------------------------------------ JSON repair

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def _close_truncated(text: str) -> str:
    """Close what a cut-off reply left open: an unfinished string, then every open bracket (innermost first)."""
    stack: List[str] = []
    in_str = esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    out = text + ('"' if in_str else "")
    out = re.sub(r"[,:\s]+$", "", out) if not in_str else out
    return out + "".join(reversed(stack))


def parse_json(text: Any) -> Any:
    """A JSON object or list from a small model's reply, repairing the usual slips: code fences, text around the JSON,
    trailing commas, curly quotes as delimiters, control characters, a reply cut off before its closing brackets.
    Raises ValueError when nothing usable is left."""
    t = _FENCE.sub("", str(text or "")).strip()
    t = "".join(c for c in t if c >= " " or c in "\n\t")
    if not t:
        raise ValueError("empty reply")
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("model reply contained no JSON")
    t = t[min(starts):]
    candidates = [t]
    end = max(t.rfind("}"), t.rfind("]"))
    if end > 0:
        candidates.append(t[:end + 1])
    for base in list(candidates):
        candidates.append(re.sub(r",\s*([}\]])", r"\1", base))
        candidates.append(re.sub(r",\s*([}\]])", r"\1", base.replace("“", '"').replace("”", '"')))
    candidates.append(re.sub(r",\s*([}\]])", r"\1", _close_truncated(t)))
    last: Optional[Exception] = None
    for c in candidates:
        try:
            out = json.loads(c)
        except ValueError as e:
            last = e
            continue
        if isinstance(out, (dict, list)):
            return out
    raise ValueError("unreadable JSON: %s" % last)
