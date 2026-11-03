"""One chat-completion client for every LLM call in BonClay, with a fallback chain.

Providers are tried in order (LLM_PROVIDERS, default "groq,openai,mlx") and a provider that is not configured
or fails is skipped, so the app keeps working offline:

- groq    GROQ_API_KEY set (and optionally GROQ_API_KEY_2..9, more accounts: see services/groq_keys.py). Model GROQ_CHAT_MODEL (default qwen/qwen3.8-27b).
- openai  Any OpenAI-compatible server: LLM_BASE_URL (e.g. http://localhost:11434/v1 for Ollama, or LM Studio),
          LLM_MODEL, optional LLM_API_KEY.
- mlx     A local model on the Apple GPU: the backend starts `mlx_lm server` from backend/.venv-mlx (created by
          scripts/setup_mlx.sh) with LOCAL_LLM_MODEL (default mlx-community/Llama-3.2-3B-Instruct-4bit).

Requests go through one pooled keep-alive HTTP client (`_urlopen`, httpx), so the TCP and TLS handshake to the provider
is paid once, not on every reply; `prewarm()` opens that connection ahead of time (a phone starts ringing).

Callers use `chat()` (blocking; run it with asyncio.to_thread) and get the text plus which provider answered.
`chat_stream()` is the same chain over the OpenAI-compatible streaming API (SSE): it yields text deltas as they are
generated so a caller can start speaking the first sentence while the rest is still being written.
"""
import atexit
import contextlib
import io
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import httpx
except ImportError:  # the pooled client needs httpx (in requirements.txt); without it every call opens its own connection
    httpx = None

try:
    from core import offline
    from services import groq_keys
except ImportError:  # pragma: no cover - running from a different cwd
    from backend.core import offline
    from backend.services import groq_keys

log = logging.getLogger("bonclay.llm")

DEFAULT_LOCAL_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


class LLMUnavailable(RuntimeError):
    """No provider could answer. `reasons` maps provider -> why it was skipped or failed."""

    def __init__(self, reasons: Dict[str, str]):
        self.reasons = reasons
        super().__init__("No language model is available: " + "; ".join(f"{k}: {v}" for k, v in reasons.items()))


@dataclass
class ChatResult:
    text: str
    provider: str
    model: str
    latency_ms: int
    usage: Dict[str, int] = field(default_factory=dict)
    tool_calls: List[dict] = field(default_factory=list, kw_only=True)  # [{"id", "name", "arguments": dict}] when the model asked for tools


@dataclass
class StreamResult(ChatResult):
    """What a finished stream summarises: the full text, which provider answered and how long it took."""
    first_token_ms: Optional[int] = None


class LLMStreamInterrupted(RuntimeError):
    """A provider failed after it had already produced text. `partial` is what arrived before the failure; the chain
    does not try another provider then (the caller has already seen, and maybe spoken, part of the reply)."""

    def __init__(self, provider: str, model: str, partial: str, cause: BaseException):
        self.provider, self.model, self.partial, self.cause = provider, model, partial, cause
        super().__init__(f"{provider} stopped mid-reply after {len(partial)} characters: {type(cause).__name__}: "
                         f"{str(cause)[:200]}")


# ---------------------------------------------------------------------------------------------- usage meter

_usage_lock = threading.Lock()
_usage: Dict[str, Dict[str, int]] = {}
_usage_day = date.today()


def _record_usage(provider: str, usage: Dict[str, int]) -> None:
    global _usage_day
    with _usage_lock:
        if date.today() != _usage_day:
            _usage.clear()
            _usage_day = date.today()
        row = _usage.setdefault(provider, {"requests": 0, "tokens": 0})
        row["requests"] += 1
        row["tokens"] += int(usage.get("total_tokens") or 0)


def usage_today() -> Dict[str, Dict[str, int]]:
    with _usage_lock:
        return {k: dict(v) for k, v in _usage.items()}


# ---------------------------------------------------------------------------------------------- HTTP helper

# Groq sits behind Cloudflare, which refuses Python's default "Python-urllib" User-Agent (HTTP 403, error 1010).
USER_AGENT = "BonClay/1.0 (+https://github.com/Artifact19/bonclay)"

# One keep-alive connection pool for every provider request. urllib opens a new connection (TCP + TLS, about 70-100 ms
# to Groq) for each call and closes it; httpx keeps idle connections for reuse. A stale one (closed by the server while
# idle) is detected before it is reused, and a request that dies on a reused connection before the server answered is
# sent once more on a fresh one.
POOL_KEEPALIVE_S = 120.0
_pool_lock = threading.Lock()
_pool = None


def _client():
    """The shared httpx client (created on first use), or None when httpx is not installed."""
    global _pool
    if httpx is None:
        return None
    with _pool_lock:
        if _pool is None:
            _pool = httpx.Client(limits=httpx.Limits(max_connections=32, max_keepalive_connections=8,
                                                     keepalive_expiry=POOL_KEEPALIVE_S),
                                 follow_redirects=False)
        return _pool


def close_pool() -> None:
    """Drop every pooled connection (the next request opens a new one)."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        try:
            pool.close()
        except Exception:  # noqa: BLE001
            pass


atexit.register(close_pool)


def _translate(e: BaseException, timeout: Optional[float]) -> BaseException:
    """An httpx failure as the exception urlopen would have raised."""
    if isinstance(e, httpx.TimeoutException):
        return TimeoutError(f"{type(e).__name__}: timed out after {timeout} s")
    return urllib.error.URLError(e)


class _Response:
    """What a request returns, shaped like urllib's response: a context manager, `.read()`, iterable by raw lines."""

    def __init__(self, resp, timeout: Optional[float]):
        self._resp, self._timeout = resp, timeout
        self._lines = None
        self.status = resp.status_code
        self.headers = resp.headers

    def read(self) -> bytes:
        try:
            return self._resp.read()
        except httpx.HTTPError as e:
            raise _translate(e, self._timeout) from e

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        if self._lines is None:
            self._lines = self._resp.iter_lines()
        try:
            return next(self._lines).encode("utf-8") + b"\n"
        except httpx.HTTPError as e:
            raise _translate(e, self._timeout) from e

    def close(self) -> None:
        self._resp.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _urlopen(req: "urllib.request.Request", timeout: Optional[float] = None):
    """Send a urllib Request through the pooled client. Raises urllib.error.HTTPError for an HTTP error status (its
    body readable, like urlopen's), TimeoutError for a timeout and urllib.error.URLError for other connection problems,
    so callers handle it exactly as they handled urlopen. The response body is read lazily: `.read()` reads all of it,
    iterating yields the lines as they arrive (an event stream)."""
    client = _client()
    if client is None:
        return urllib.request.urlopen(req, timeout=timeout)
    headers = {k: v for k, v in req.header_items()}
    headers.setdefault("Accept-Encoding", "identity")  # a compressed event stream may be buffered on its way
    limit = httpx.Timeout(timeout) if timeout else httpx.Timeout(None)
    request = client.build_request(req.get_method(), req.full_url, content=req.data, headers=headers, timeout=limit)
    resp = None
    for attempt in (0, 1):
        try:
            resp = client.send(request, stream=True)
            break
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError) as e:
            if attempt == 0:  # a pooled connection that the server had closed: once more on a new one
                log.info("pooled connection was stale (%s); retrying on a new one", type(e).__name__)
                continue
            raise _translate(e, timeout) from e
        except httpx.HTTPError as e:
            raise _translate(e, timeout) from e
    if resp.status_code >= 400:
        try:
            body = resp.read()
        except httpx.HTTPError as e:
            raise _translate(e, timeout) from e
        finally:
            resp.close()
        raise urllib.error.HTTPError(req.full_url, resp.status_code, resp.reason_phrase, dict(resp.headers),
                                     io.BytesIO(body))
    return _Response(resp, timeout)


def _flatten_tool_messages(messages: List[dict]) -> List[dict]:
    """The same conversation with tool calls and results written as plain text, for a request that carries no tools."""
    out: List[dict] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = "; ".join(f"{(c.get('function') or {}).get('name')}({(c.get('function') or {}).get('arguments')})"
                              for c in m["tool_calls"])
            text = ((m.get("content") or "") + f" [called: {calls}]").strip()
            out.append({"role": "assistant", "content": text})
        elif m.get("role") == "tool":
            out.append({"role": "user", "content": f"Tool result: {m.get('content')}"})
        else:
            out.append(m)
    return out


class ProviderHTTPError(RuntimeError):
    """An HTTP error status from a provider. `str()` is "HTTP <code>: <first 300 characters of the body>" (what the
    fallback chain matches on); the whole body and the response headers are kept for reading a rate limit."""

    def __init__(self, code: int, body: str, headers: Any = None):
        self.code, self.body = int(code), body or ""
        try:
            self.headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        except AttributeError:  # pragma: no cover - an unusual headers object
            self.headers = {}
        super().__init__(f"HTTP {self.code}: {self.body[:300]}")


def _send(url: str, body: dict, headers: dict, timeout: Optional[float]) -> dict:
    """POST a (non-streaming) JSON request; an HTTP error status raises ProviderHTTPError."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with _urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise ProviderHTTPError(e.code, e.read().decode(errors="replace"), e.headers) from e


# ---------------------------------------------------------------------------------------------- tool-use failures
#
# Groq's gpt-oss models are trained to call tools natively. Given the JSON tool protocol of services/tools.py (tools
# described in the prompt, no `tools` in the request, so tool_choice is "none"), they emit a native call anyway and
# Groq answers HTTP 400 {"error": {"code": "tool_use_failed", "message": "Tool choice is none, but model called a
# tool", "failed_generation": "{\"name\": \"book_appointment\", \"arguments\": {...}}"}}. A streamed request gets HTTP
# 200 and then the same error object as an `event: error`. The call the model attempted is in `failed_generation`:
# when the caller offers that tool through the JSON protocol (`json_tools`) it becomes the reply, written in that
# protocol ({"tool": ..., "arguments": ...}), and the tool turn carries on. With native `tools` and tool_choice
# "auto" it becomes the result's tool_calls. Otherwise the model is asked once more, with no tools and an instruction
# to answer in words.

_TOOL_USE_FAILED = ("tool_use_failed", "tool choice is none", "model called a tool", "failed to call a function")
PLAIN_TEXT_NOTE = ("(Answer the caller now in plain spoken text. Do not call a tool or function and do not write "
                   "JSON.)")


def _tool_use_failed(code: int, body: str) -> bool:
    return code == 400 and any(s in (body or "").lower() for s in _TOOL_USE_FAILED)


def _error_object(body: Any) -> dict:
    """The `error` object of a provider's error body (a JSON string or an already parsed dict); {} when there is none."""
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (TypeError, ValueError):
            return {}
    if isinstance(body, dict):
        err = body.get("error", body)
        return err if isinstance(err, dict) else {}
    return {}


def _json_value(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except (TypeError, ValueError):
                return None
    return None


def failed_tool_call(body: Any) -> Optional[dict]:
    """The tool call a model attempted, from the `failed_generation` of a tool_use_failed error: {"name", "arguments"}
    (arguments a dict), or None when it cannot be read. Accepts the error body as text or parsed. Handles the shapes
    seen from Groq: {"name", "arguments"} with arguments as an object or a JSON string, a list of such calls (the first
    is used), {"function": {...}}, the JSON protocol's own {"tool", "arguments"} (also nested inside "arguments"), and
    the raw "to=functions.NAME ... {json}" text of the model's tool channel."""
    raw = _error_object(body).get("failed_generation")
    if not raw:
        return None
    name, args = None, None
    channel = re.search(r"to=(?:functions\.)?([A-Za-z0-9_\-]+)", raw) if isinstance(raw, str) else None
    if channel:  # the raw text of the model's tool channel
        name, args = channel.group(1), _json_value(raw[channel.end():]) or {}
    else:
        obj = _json_value(raw)
        if isinstance(obj, list):
            obj = next((o for o in obj if isinstance(o, dict)), None)
        if isinstance(obj, dict):
            fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
            name = fn.get("name") or fn.get("tool") or fn.get("recipient_name")
            args = fn.get("arguments", fn.get("parameters", {}))
    args = _json_value(args) if isinstance(args, str) else args
    # {"name": "book_appointment", "arguments": {"tool": "book_appointment", "arguments": {...}}}: the model wrapped
    # the JSON protocol inside its native call
    if isinstance(args, dict) and isinstance(args.get("arguments"), dict) and set(args) <= {"tool", "name", "arguments"}:
        name = args.get("tool") or args.get("name") or name
        args = args["arguments"]
    if not isinstance(name, str) or not name.strip() or not isinstance(args, dict):
        return None
    name = name.strip()
    if name.startswith("functions."):
        name = name[len("functions."):]
    return {"name": name, "arguments": args}


def _plain_text_messages(messages: List[dict]) -> List[dict]:
    """`messages` with the tool history as plain text and PLAIN_TEXT_NOTE added to the caller's last message."""
    out = [dict(m) for m in _flatten_tool_messages(messages)]
    if out and out[-1].get("role") == "user":
        out[-1]["content"] = f"{out[-1].get('content') or ''}\n\n{PLAIN_TEXT_NOTE}"
    else:
        out.append({"role": "user", "content": PLAIN_TEXT_NOTE})
    return out


def _protocol_text(call: dict) -> str:
    """An attempted tool call written as a reply of the JSON tool protocol (services/tools.py parses it)."""
    return json.dumps({"tool": call["name"], "arguments": call["arguments"]}, ensure_ascii=False)


def _recover_tool_use(send, body: dict, messages: List[dict], error_body: str, *, model: str,
                      tools: Optional[List[dict]], tool_choice: Optional[str], json_tools) -> dict:
    """A blocking request the server rejected as tool_use_failed: the attempted call as the reply when the caller can
    use it, else one more request with no tools and PLAIN_TEXT_NOTE. Returns the response (or the retry's)."""
    call = failed_tool_call(error_body)
    native = {(t.get("function") or {}).get("name") for t in tools or []}
    if call and tools and tool_choice != "none" and call["name"] in native:
        log.warning("model %s sent a malformed native tool call; using the call it attempted (%s)", model, call["name"])
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_failed_generation", "type": "function",
             "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}]}}],
                "usage": {}}
    if call and not tools and json_tools and call["name"] in json_tools:
        log.warning("model %s called %s natively under the JSON tool protocol; using that call", model, call["name"])
        return {"choices": [{"message": {"role": "assistant", "content": _protocol_text(call)}}], "usage": {}}
    log.warning("model %s called a tool where none was allowed (%s); asking once more for a plain answer", model,
                (call or {}).get("name") or "unreadable call")
    retry = {k: v for k, v in body.items() if k not in ("tools", "tool_choice")}
    retry["messages"] = _plain_text_messages(messages)
    return send(retry)


def _openai_chat(base_url: str, model: str, messages: List[dict], *, api_key: Optional[str], json_mode: bool,
                 temperature: float, max_tokens: int, timeout: float, tools: Optional[List[dict]] = None,
                 tool_choice: Optional[str] = None, extra: Optional[dict] = None,
                 json_tools: Optional[Iterable[str]] = None) -> dict:
    """One chat completion. `json_tools`: names of the tools the prompt offers through the JSON tool protocol (see
    _recover_tool_use)."""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "stream": False, **(extra or {})}
    body["max_tokens"] = _with_reasoning_room(max_tokens, extra)
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/chat/completions"
    send = lambda b: _send(url, b, headers, timeout)  # noqa: E731
    try:
        return send(body)
    except ProviderHTTPError as e:
        detail = e.body[:300]
        if _tool_use_failed(e.code, e.body):
            return _recover_tool_use(send, body, messages, e.body, model=model, tools=tools, tool_choice=tool_choice,
                                     json_tools=set(json_tools or ()))
        if e.code in (400, 422) and "reasoning_effort" in body and "reasoning" in detail.lower():
            body.pop("reasoning_effort")  # a model that does not take the switch: ask again without it
            return send(body)
        if json_mode and e.code in (400, 422) and "response_format" in detail:
            body.pop("response_format")  # server does not support JSON mode; the prompt still asks for JSON
            return send(body)
        raise


def _post_stream(url: str, body: dict, headers: dict, timeout: float):
    """Open a streaming POST; a server that rejects `stream_options` (or `reasoning_effort`) gets a retry without it.
    Any other HTTP error status raises ProviderHTTPError."""
    for attempt in (0, 1, 2):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            return _urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            full = e.read().decode(errors="replace")
            detail = full[:300]
            if attempt < 2 and e.code in (400, 422) and "stream_options" in detail and "stream_options" in body:
                body = {k: v for k, v in body.items() if k != "stream_options"}
                continue
            if (attempt < 2 and e.code in (400, 422) and "reasoning_effort" in body
                    and "reasoning" in detail.lower()):
                body = {k: v for k, v in body.items() if k != "reasoning_effort"}
                continue
            raise ProviderHTTPError(e.code, full, e.headers) from e
    raise AssertionError("unreachable")


def _openai_stream(base_url: str, model: str, messages: List[dict], *, api_key: Optional[str], temperature: float,
                   max_tokens: int, timeout: float, info: dict, extra: Optional[dict] = None,
                   json_tools: Optional[Iterable[str]] = None) -> Iterator[str]:
    """Yield the text deltas of a streamed chat completion (`data:` server-sent-event lines).

    `timeout` bounds each wait on the socket (connect and every read), so a stalled server is noticed. Token usage,
    when the server reports it, lands in info["usage"]. A tool_use_failed error before any text (the model called a
    tool natively, see _recover_tool_use) yields the attempted call in the JSON tool protocol when it names one of
    `json_tools`, else the request is made once more with PLAIN_TEXT_NOTE."""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, **(extra or {})}
    body["max_tokens"] = _with_reasoning_room(max_tokens, extra)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/chat/completions"
    names = set(json_tools or ())

    def recover(error_body: Any):
        """The text to yield (the attempted call) or the body of the one retry."""
        call = failed_tool_call(error_body)
        if call and names and call["name"] in names:
            log.warning("model %s called %s natively under the JSON tool protocol; using that call", model,
                        call["name"])
            return _protocol_text(call)
        log.warning("model %s called a tool where none was allowed (%s); asking once more for a plain answer", model,
                    (call or {}).get("name") or "unreadable call")
        return {**body, "messages": _plain_text_messages(messages)}

    for attempt in (0, 1):
        try:
            resp = _post_stream(url, body, headers, timeout)
        except ProviderHTTPError as e:
            if attempt == 0 and _tool_use_failed(e.code, e.body):
                nxt = recover(e.body)
                if isinstance(nxt, str):
                    yield nxt
                    return
                body = nxt
                continue
            raise
        failure: Optional[dict] = None
        produced = False
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):  # blank separators, ": keep-alive" comments, "event:" lines
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    # Read to the end of the body (only its closing bytes are left) so the connection goes back to
                    # the pool: a response closed unfinished costs the next reply a new TCP + TLS handshake.
                    try:
                        for _ in resp:
                            pass
                    except Exception:  # noqa: BLE001 - the reply is complete whatever happens to its closing bytes
                        pass
                    return
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("error"):
                    err = obj["error"]
                    err_text = json.dumps(err, ensure_ascii=False) if isinstance(err, dict) else str(err)
                    if _tool_use_failed(400, err_text):
                        if attempt == 0 and not produced:
                            failure = err if isinstance(err, dict) else {"message": err_text}
                            break
                        raise ProviderHTTPError(400, json.dumps({"error": err}, ensure_ascii=False))
                    raise RuntimeError(str(err.get("message") if isinstance(err, dict) else err)[:300])
                usage = obj.get("usage") or (obj.get("x_groq") or {}).get("usage")
                if isinstance(usage, dict) and usage:
                    info["usage"] = usage
                for choice in obj.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        produced = True
                        yield piece
        if failure is None:
            return
        nxt = recover({"error": failure})
        if isinstance(nxt, str):
            yield nxt
            return
        body = nxt


def _tool_calls(message: dict) -> List[dict]:
    """The tool calls of an OpenAI-style assistant message, with the arguments parsed ({} when they are not JSON)."""
    out = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            args = {}
        out.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name") or "",
                    "arguments": args if isinstance(args, dict) else {}})
    return out


def _chat_result(data: dict, provider: str, model: str, started: float) -> ChatResult:
    message = data["choices"][0]["message"]
    return ChatResult(message.get("content") or "", provider, model, int((time.time() - started) * 1000),
                      data.get("usage") or {}, tool_calls=_tool_calls(message))


# ---------------------------------------------------------------------------------------------- providers

# Qwen 3.8 27B was the fastest of Groq's current models in our Hindi, Marathi and tool-call checks (first token ~0.12 s
# vs ~0.4 s), but it is a Preview model with a low per-minute token limit on the free tier, so a rate-limited or
# unavailable model falls back to the next one in GROQ_FALLBACK_MODELS before the local model.
#
# Fallback order: gpt-oss-20b (fast; not on Vectara's hallucination leaderboard) before gpt-oss-120b (14.2 % there,
# 2026-10-16). Llama 3.3 70B has the lowest rate of the models Groq has served (4.1 %) but Groq shut it down for free and
# developer accounts on 2026-08-16, so it is not a default; an enterprise account can put it first with
# GROQ_FALLBACK_MODELS=llama-3.3-70b-versatile,openai/gpt-oss-20b,openai/gpt-oss-120b. A model that answers
# 404/decommissioned is skipped for GROQ_DEAD_MODEL_S (an hour). GROQ_FALLBACK_MODELS replaces the whole list ("" = none).
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
DEFAULT_GROQ_FALLBACKS = "openai/gpt-oss-20b,openai/gpt-oss-120b"
# The small, fast model of side jobs that must never slow a reply (groq_one_shot: the caller-details reading in
# services/caller_memory.py, CALLER_MEMORY_MODEL). Not used by the answer chain, and not run anywhere in this repo's
# tests. gpt-oss-20b has its own per-model quota, so these jobs do not eat the reply model's budget; it reasons before
# it answers, so groq_one_shot asks for reasoning_effort "low" and leaves room for it (see _groq_extra and
# _with_reasoning_room). If Groq stops serving it the side job fails and is skipped.
DEFAULT_SMALL_GROQ_MODEL = "openai/gpt-oss-20b"
_GROQ_RETRYABLE = ("HTTP 429", "HTTP 404", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "model_not_found",
                   "model_decommissioned", "decommissioned", "tool_use_failed", "json_validate_failed")
_GROQ_GONE = ("HTTP 404", "model_not_found", "model_decommissioned", "decommissioned")
_dead_models: Dict[str, float] = {}  # model -> monotonic time it answered "no such model"
# (key label, model) -> (monotonic time that account's daily quota of the model resets, the same moment as a UTC
# datetime for messages). The state lives in services/groq_keys.py (several Groq accounts, see there); this is the same dict.
_quota_until = groq_keys._daily_paused


def _dead_for() -> float:
    try:
        return float(os.getenv("GROQ_DEAD_MODEL_S", "3600"))
    except ValueError:
        return 3600.0


# ---- daily quotas. A 429 for the per-MINUTE limit passes in seconds: the next model answers this turn and the model
# is tried again on the next one. A 429 for the per-DAY limit ("... on tokens per day (TPD): Limit 200000, Used ...
# Please try again in 7m12.5s", or requests per day, RPD) does not: the model is skipped until the quota resets
# instead of costing every turn a failed request. The reset is read from the message's "try again in", else the
# retry-after header, else (requests per day) x-ratelimit-reset-requests, else it is the next midnight UTC.
_DAILY_RE = re.compile(r"\b(tokens|requests|audio\s+seconds)\s+per\s+day\b|\((TPD|RPD|ASD)\)|\b(TPD|RPD|ASD)\b",
                       re.IGNORECASE)
_TRY_AGAIN_RE = re.compile(r"try again in\s+((?:\d+(?:\.\d+)?\s*(?:ms|h|m|s)\s*)+)", re.IGNORECASE)
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", re.IGNORECASE)
QUOTA_MIN_PAUSE_S = 30.0
QUOTA_MAX_PAUSE_S = 24 * 3600.0


def _duration_s(text: Optional[str]) -> Optional[float]:
    """Seconds in a Groq duration ("7m12.5s", "1h2m", "45.2s", "250ms") or a plain number of seconds; None if neither."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    parts = _DURATION_PART.findall(text)
    if not parts:
        return None
    scale = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    return sum(float(n) * scale[u.lower()] for n, u in parts)


def _until_midnight_utc(now: Optional[float] = None) -> float:
    from datetime import datetime, timedelta, timezone
    t = datetime.fromtimestamp(now if now is not None else time.time(), timezone.utc)
    midnight = (t + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (midnight - t).total_seconds()


def daily_quota_pause(e: BaseException) -> Optional[tuple]:
    """(seconds until the quota resets, "tokens" | "requests") when `e` is a 429 for a per-DAY limit, else None."""
    e = _as_http_error(e)  # a ProviderHTTPError, or the Groq SDK's RateLimitError (speech recognition)
    if e is None or e.code != 429:
        return None
    m = _DAILY_RE.search(e.body)
    if not m:
        return None
    word = re.sub(r"\s+", " ", (m.group(1) or m.group(2) or m.group(3) or "").lower())
    kind = "requests" if word in ("requests", "rpd") else "audio seconds" if word in ("audio seconds", "asd") else "tokens"
    wait = None
    found = _TRY_AGAIN_RE.search(e.body)
    if found:
        wait = _duration_s(found.group(1))
    if wait is None:
        wait = _duration_s(e.headers.get("retry-after"))
    if wait is None and kind == "requests":
        wait = _duration_s(e.headers.get("x-ratelimit-reset-requests"))
    if wait is None:
        wait = _until_midnight_utc()
    return min(QUOTA_MAX_PAUSE_S, max(QUOTA_MIN_PAUSE_S, wait)), kind


def _quota_paused(model: str, now: Optional[float] = None) -> bool:
    """True when every configured Groq key has used its daily quota of `model` (with one key: that key has)."""
    ks = groq_keys.keys()
    return bool(ks) and all(groq_keys.is_daily_paused(k["label"], model, now) for k in ks)


def paused_models() -> Dict[str, str]:
    """Models skipped because every key's daily quota is used up -> when the first key can use it again (UTC, ISO 8601)."""
    out: Dict[str, str] = {}
    for model in groq_keys.paused_model_names():
        wall = groq_keys.model_resets(model)
        if wall is not None:
            out[model] = wall.isoformat(timespec="seconds")
    return out


def _groq_configured() -> List[str]:
    models = [os.getenv("GROQ_CHAT_MODEL", DEFAULT_GROQ_MODEL)]
    for m in os.getenv("GROQ_FALLBACK_MODELS", DEFAULT_GROQ_FALLBACKS).split(","):
        if m.strip() and m.strip() not in models:
            models.append(m.strip())
    return models


def _groq_models() -> List[str]:
    """The configured Groq model first, then GROQ_FALLBACK_MODELS (comma-separated; empty turns fallback off). A model
    that recently answered "no such model" is left out (unless nothing else is left), and so is a model whose daily
    quota is used up (always: the list may then be empty)."""
    models = _groq_configured()
    now = time.monotonic()
    alive = [m for m in models if now - _dead_models.get(m, -1e18) >= _dead_for()]
    return [m for m in (alive or models) if not _quota_paused(m, now)]


def _mark_gone(model: str, e: Exception, label: Optional[str] = None) -> bool:
    """Remember a model that is gone (for every key) or whose daily quota is used up on key `label`. True when the
    daily quota is the reason."""
    if any(code in str(e) for code in _GROQ_GONE):
        _dead_models[model] = time.monotonic()
    pause = daily_quota_pause(e)
    if pause is None:
        return False
    seconds, kind = pause
    label = label or (groq_keys.keys() or [{"label": "1"}])[0]["label"]
    wall, newly = groq_keys.pause_daily(label, model, seconds)
    if not newly:
        return True  # another thread got there first: one log line per pause
    if len(groq_keys.keys()) > 1:
        log.warning("Groq model %s on key %s has used its daily %s quota; skipping it until %s UTC (%.0f min)", model,
                    label, kind, wall.strftime("%Y-%m-%d %H:%M"), seconds / 60)
    else:
        log.warning("Groq model %s has used its daily %s quota; skipping it until %s UTC (%.0f min)", model, kind,
                    wall.strftime("%Y-%m-%d %H:%M"), seconds / 60)
    return True


def thinking_enabled() -> bool:
    """LLM_THINKING=off (default): Qwen3 models answer without a reasoning pass (Groq reasoning_effort "none")."""
    return (os.getenv("LLM_THINKING", "off") or "off").strip().lower() in ("on", "1", "true", "yes")


def answer_temperature() -> float:
    """LLM_ANSWER_TEMPERATURE (default 0.2, clamped to 0..1): sampling temperature of a grounded spoken answer. Low, so
    the model copies a time or a fee from the passages instead of paraphrasing it."""
    try:
        return min(1.0, max(0.0, float(os.getenv("LLM_ANSWER_TEMPERATURE", "0.2"))))
    except ValueError:
        return 0.2


def _with_reasoning_room(max_tokens: int, extra: Optional[dict]) -> int:
    """A model that reasons before answering (gpt-oss at reasoning_effort "low") spends part of max_tokens on hidden
    reasoning, so a long JSON answer could be cut off ("max completion tokens reached before generating a valid
    document"). Such a request gets REASONING_TOKEN_ROOM (default 1024) on top of what the answer needs."""
    effort = (extra or {}).get("reasoning_effort")
    if not effort or effort == "none":
        return max_tokens
    try:
        room = int(os.getenv("REASONING_TOKEN_ROOM", "1024"))
    except ValueError:
        room = 1024
    return max_tokens + max(0, room)


def _groq_extra(model: str, tools: bool = False) -> dict:
    """gpt-oss models reason before answering; keep it short and out of the reply. Qwen3 models think by default: an
    answer without tools turns that off (reasoning_effort "none", Groq's instruct mode; faster, and reasoning models
    hallucinate more on grounded summaries). A request with tools is left as it was."""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "include_reasoning": False}
    if "qwen3" in model and not tools and not thinking_enabled():
        return {"reasoning_effort": "none"}
    return {}


def _groq_retryable(e: Exception) -> bool:
    return any(code in str(e) for code in _GROQ_RETRYABLE)


def _groq_reason() -> Optional[str]:
    if offline.enabled():
        return offline.reason("Groq")
    if not groq_keys.has_any():
        return "GROQ_API_KEY is not set"
    if not _groq_models() and paused_models():
        return _all_used_up(" UTC")
    return None


def _all_used_up(utc: str = "") -> str:
    """Why no Groq model can be used: every key has used the daily quota of every model, with when each model is back."""
    resets = ", ".join(f"{m} {t}{utc}" for m, t in paused_models().items())
    n = len(groq_keys.keys())
    if n > 1:
        return f"all {n} Groq keys have used the daily quota of every model (resets: {resets})"
    return f"every Groq model has used its daily quota (resets: {resets})"


def _no_groq_model() -> RuntimeError:
    return RuntimeError("HTTP 429: " + _all_used_up())


# ---- which key a failure belongs to. A 429 or a 401/403 is about the ACCOUNT behind the key (its per-minute limit, its
# daily quota, a refused key), so the same request goes out on the next account's key. Anything else retryable (a model
# that is gone, a 5xx, a failed tool call) is about the model and moves to the next model, as before.
_KEY_FAILURE_WORDS = {"daily": "daily", "minute": "per-minute", "auth": "auth"}


def _as_http_error(e: BaseException) -> Optional[ProviderHTTPError]:
    """`e` as a ProviderHTTPError: itself, or the Groq SDK's APIStatusError (RateLimitError, AuthenticationError, ...)
    which carries status_code, message and response."""
    if isinstance(e, ProviderHTTPError):
        return e
    code = getattr(e, "status_code", None)
    if not isinstance(code, int):
        code = {"RateLimitError": 429, "AuthenticationError": 401, "PermissionDeniedError": 403}.get(type(e).__name__)
    if code is None:
        return None
    headers = getattr(getattr(e, "response", None), "headers", None)
    return ProviderHTTPError(code, str(getattr(e, "message", None) or e), headers)


def key_failure(e: BaseException) -> Optional[tuple]:
    """(kind, seconds) when `e` is a failure of the key's ACCOUNT, else None. "daily": a per-day 429, seconds until the
    quota resets. "minute": any other 429, seconds to keep the model off this key (Groq's "try again in", else the
    retry-after header, else None for the default). "auth": 401 or 403, seconds is None."""
    he = _as_http_error(e)
    if he is None:
        return None
    if he.code in (401, 403):
        return "auth", None
    if he.code != 429:
        return None
    pause = daily_quota_pause(he)
    if pause is not None:
        return "daily", pause[0]
    m = _TRY_AGAIN_RE.search(he.body or "")
    wait = _duration_s(m.group(1)) if m else None
    if wait is None:
        wait = _duration_s(he.headers.get("retry-after"))
    return "minute", wait


def _next_attempt(plan: list, i: int, e: BaseException, fail: Optional[tuple], bad: set) -> Optional[int]:
    """Index in `plan` of the attempt to make after plan[i] failed with `e`, None when the error ends the request. An
    account failure goes to the next attempt on a key that was not refused (the same model on the next account, else
    the next model). Another retryable error skips to the next model."""
    model = plan[i][0]
    if fail is not None:
        return next((j for j in range(i + 1, len(plan)) if plan[j][1]["label"] not in bad), None)
    if _groq_retryable(e):
        return next((j for j in range(i + 1, len(plan)) if plan[j][0] != model and plan[j][1]["label"] not in bad), None)
    return None


def _groq_failed(model: str, label: str, e: Exception, nxt: Optional[tuple], fail: Optional[tuple] = None) -> None:
    """Note a failed Groq attempt (model gone, daily quota, per-minute cool-down, refused key) and log the switch to the
    next attempt `nxt` = (model, key)."""
    daily = _mark_gone(model, e, label)
    if fail is not None and fail[0] == "auth":
        groq_keys.mark_auth_failed(label)
    elif fail is not None and fail[0] == "minute" and len(groq_keys.keys()) > 1:
        groq_keys.cool_down(label, model, fail[1])  # one key: the minute-wait below handles it, as it always did
    if not nxt:
        return
    if fail is not None and nxt[1]["label"] != label:
        log.warning("Groq key %s hit the %s limit for %s; trying key %s", label, _KEY_FAILURE_WORDS[fail[0]], model,
                    nxt[1]["label"])
    elif not daily:
        log.warning("Groq model %s unavailable (%s); trying %s", model, str(e)[:80], nxt[0])


# ---- the per-minute limit. On the free tier every model shares 8,000 tokens a minute; a burst of turns puts all three
# at the limit at once, and Groq says "Please try again in 1.6s". Dropping to the local 3B model then costs 8-10 s and a
# worse answer (measured 22 Oct 2026: one web turn out of ten took 10.5 s that way). When the asked-for wait is short,
# waiting it out and asking Groq once more is the faster, better path. GROQ_MINUTE_WAIT_S (default 3) is the longest
# wait worth taking; 0 turns this off.
def minute_limit_wait(e: BaseException) -> Optional[float]:
    """Seconds Groq asks us to wait for a per-MINUTE limit, when that is no longer than GROQ_MINUTE_WAIT_S; None for
    any other error, for a daily quota, or for a longer wait."""
    if not isinstance(e, ProviderHTTPError) or e.code != 429 or daily_quota_pause(e) is not None:
        return None
    try:
        cap = float(os.getenv("GROQ_MINUTE_WAIT_S", "3"))
    except ValueError:
        cap = 3.0
    if cap <= 0:
        return None
    m = _TRY_AGAIN_RE.search(e.body or str(e))
    wait = _duration_s(m.group(1)) if m else None
    return wait if wait is not None and 0.0 <= wait <= cap else None


def _groq_chat_once(models: List[str], messages, kw) -> ChatResult:
    """One pass: the best model on every usable key before the next model (see groq_keys.plan)."""
    plan = groq_keys.plan(models)
    if not plan:
        raise RuntimeError("GROQ_API_KEY is not set")
    bad: set = set()
    i = 0
    while i < len(plan):
        model, k = plan[i]
        t = time.time()
        try:
            data = _openai_chat("https://api.groq.com/openai/v1", model, messages,
                                api_key=k["key"], extra=_groq_extra(model, bool(kw.get("tools"))), **kw)
            return _chat_result(data, "groq", model, t)
        except Exception as e:  # noqa: BLE001
            fail = key_failure(e)
            if fail is not None and fail[0] == "auth":
                bad.add(k["label"])
            j = _next_attempt(plan, i, e, fail, bad)
            _groq_failed(model, k["label"], e, plan[j] if j is not None else None, fail)
            if j is None:
                raise
            i = j
    raise AssertionError("unreachable")


def _groq_chat(messages, **kw) -> ChatResult:
    models = _groq_models()
    if not models:
        raise _no_groq_model()
    try:
        return _groq_chat_once(models, messages, kw)
    except Exception as e:  # noqa: BLE001
        wait = minute_limit_wait(e)
        if wait is None:
            raise
        log.warning("every Groq model is at its per-minute limit; waiting %.1f s and asking once more instead of "
                    "falling back to the local model", wait)
        time.sleep(wait + 0.2)
        return _groq_chat_once(_groq_models() or models, messages, kw)


def groq_one_shot(messages: List[dict], *, model: str, json_mode: bool = False, temperature: float = 0.0,
                  max_tokens: int = 200, timeout: float = 4.0) -> ChatResult:
    """One request to one named Groq model, with no fallback chain and no waiting out of a rate limit: for small side jobs
    that must never slow a reply (the caller-details reading, services/caller_memory.py). Raises LLMUnavailable when
    Groq is off (no key, offline mode, the model's daily quota used up), and whatever the request raised when it failed
    (a 429 included): the caller skips the job. gpt-oss models get reasoning_effort "low" and REASONING_TOKEN_ROOM extra
    tokens (the same helpers every Groq call here uses), so a JSON answer is not cut off by the hidden reasoning. The key
    is the first account (services/groq_keys.py) that still has this model's quota; a side job tries only that one."""
    reason = _groq_reason()
    if reason:
        raise LLMUnavailable({"groq": reason})
    usable = groq_keys.ordered_keys(model) or groq_keys.ordered_keys(model, soft=False)
    if not usable:
        raise LLMUnavailable({"groq": f"{model} has used its daily quota on every key"})
    started = time.time()
    try:
        data = _openai_chat("https://api.groq.com/openai/v1", model, messages, api_key=usable[0]["key"],
                            json_mode=json_mode, temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                            extra=_groq_extra(model))
    except Exception as e:  # noqa: BLE001
        _mark_gone(model, e)
        raise
    res = _chat_result(data, "groq", model, started)
    _record_usage("groq", res.usage)
    return res


_OLLAMA_AUTO_PROBED = False
_OLLAMA_AUTO_URL: Optional[str] = None
_OLLAMA_AUTO_MODEL: Optional[str] = None


def _auto_ollama_info() -> Tuple[Optional[str], Optional[str]]:
    global _OLLAMA_AUTO_PROBED, _OLLAMA_AUTO_URL, _OLLAMA_AUTO_MODEL
    if _OLLAMA_AUTO_PROBED:
        return _OLLAMA_AUTO_URL, _OLLAMA_AUTO_MODEL
    _OLLAMA_AUTO_PROBED = True
    if os.getenv("LLM_BASE_URL"):
        return None, None
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "BonClay"})
        with urllib.request.urlopen(req, timeout=0.3) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                if models:
                    _OLLAMA_AUTO_URL = "http://127.0.0.1:11434/v1"
                    _OLLAMA_AUTO_MODEL = models[0]
                    log.info("Auto-detected local Ollama at %s with models: %s", _OLLAMA_AUTO_URL, models)
    except Exception:
        pass
    return _OLLAMA_AUTO_URL, _OLLAMA_AUTO_MODEL


def _openai_base_url() -> Optional[str]:
    return os.getenv("LLM_BASE_URL") or _auto_ollama_info()[0]


def _openai_model() -> str:
    return os.getenv("LLM_MODEL") or _auto_ollama_info()[1] or "llama3.2"


def _openai_reason() -> Optional[str]:
    url = _openai_base_url()
    if url and offline.enabled() and not offline.is_local_url(url):
        return offline.reason("a language-model server that is not on this computer")
    return None if url else "LLM_BASE_URL is not set"


def _openai_compat_chat(messages, **kw) -> ChatResult:
    url = _openai_base_url()
    if not url:
        raise AppError("MODEL_UNAVAILABLE", "LLM_BASE_URL is not set", 503)
    model = _openai_model()
    t = time.time()
    data = _openai_chat(url, model, messages, api_key=os.getenv("LLM_API_KEY"), **kw)
    return _chat_result(data, "openai", model, t)


def prompt_cache_args() -> List[str]:
    """Limits for the local model server's prompt cache, which otherwise keeps up to 10 conversations' attention caches with
    no byte limit (several GB with long prompts, on a 16 GB Mac that means swapping). MLX_PROMPT_CACHE_MB (default 600,
    about 5,000 tokens for the 3B model) and MLX_PROMPT_CACHE_SIZE (default 4 entries); 0 for either leaves it unlimited."""
    args: List[str] = []
    for env, flag, default, scale in (("MLX_PROMPT_CACHE_MB", "--prompt-cache-bytes", 600, 1_000_000),
                                      ("MLX_PROMPT_CACHE_SIZE", "--prompt-cache-size", 4, 1)):
        try:
            value = int(float(os.getenv(env, "") or default))
        except ValueError:
            value = default
        if value > 0:
            args += [flag, str(value * scale)]
    return args


class _MlxServer:
    """`mlx_lm server` in the MLX virtualenv, started on first use and kept warm."""

    def __init__(self, model: Optional[str] = None, log_name: str = "local_llm.log"):
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()  # the local server answers one request at a time; queue the rest
        self.proc: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.model = model or os.getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL)
        # The server's warnings go to a log file: a pipe nobody reads would block the server once it fills up.
        self._log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs",
                                      log_name)
        self._log = None

    def python(self) -> Optional[str]:
        try:
            from services.mlx_clone import mlx_python
        except ImportError:
            from backend.services.mlx_clone import mlx_python
        return mlx_python()

    def unavailable_reason(self) -> Optional[str]:
        if os.getenv("LOCAL_LLM", "").lower() == "off":
            return "LOCAL_LLM=off"
        if not self.python():
            return "no MLX environment (run: bash backend/scripts/setup_mlx.sh) or not an Apple Silicon Mac"
        if not _hf_cached(self.model):
            return f"{self.model} is not downloaded (run: bash backend/scripts/setup_mlx.sh)"
        return None

    def memory_mb(self) -> Optional[float]:
        """Resident memory of the server process in MB (None when it is not running or cannot be read)."""
        if not self.running:
            return None
        try:
            import psutil
            return psutil.Process(self.proc.pid).memory_info().rss / 2 ** 20
        except Exception:  # noqa: BLE001
            return None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure(self, timeout: float = 180.0) -> str:
        with self.lock:
            if self.running:
                return f"http://127.0.0.1:{self.port}/v1"
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                self.port = s.getsockname()[1]
            env = {**os.environ, "HF_HUB_OFFLINE": "1"}
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            if self._log is None or self._log.closed:
                self._log = open(self._log_path, "ab")
            self.proc = subprocess.Popen(
                [self.python(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "parent_watch.py"),
                 "--run-module", "mlx_lm", "server", "--model", self.model, "--host", "127.0.0.1",
                 "--port", str(self.port), "--log-level", "WARNING", *prompt_cache_args()],
                env=env, stdout=subprocess.DEVNULL, stderr=self._log)
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    err = _tail(self._log_path)
                    self.proc = None
                    raise RuntimeError(f"mlx_lm server exited: {err}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=2):
                        log.info("local LLM ready: %s", self.model)
                        return f"http://127.0.0.1:{self.port}/v1"
                except (urllib.error.URLError, ConnectionError, OSError):
                    time.sleep(0.5)
            self.stop()
            raise RuntimeError("local LLM did not start in time")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


_mlx = _MlxServer()
atexit.register(_mlx.stop)


def _tail(path: str, n: int = 300) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - n))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def _hf_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


def _hf_complete(repo_id: str) -> bool:
    """config.json and every weight file of the model are in the Hugging Face cache (a download still running is not)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = lambda name: try_to_load_from_cache(repo_id, name)  # noqa: E731
        if not isinstance(cached("config.json"), str):
            return False
        index = cached("model.safetensors.index.json")
        if isinstance(index, str):
            with open(index, "r", encoding="utf-8") as f:
                shards = set((json.load(f).get("weight_map") or {}).values())
            return bool(shards) and all(isinstance(cached(n), str) for n in shards)
        return isinstance(cached("model.safetensors"), str)
    except Exception:  # noqa: BLE001
        return False


LOCAL_REQUEST_TIMEOUT = 180  # seconds the local server may stay silent before a request is given up

# Per-thread request profile. A phone turn runs its model calls in one worker thread and cannot wait 90 s behind another
# local request (the caller hears silence), so it sets a short queue timeout for that thread only.
_profile = threading.local()


@contextlib.contextmanager
def local_queue_timeout(seconds: Optional[float]):
    """Within the block (this thread only), a local-model request waits at most `seconds` for the local server's slot
    instead of LOCAL_LLM_QUEUE_S. None restores the default."""
    previous = getattr(_profile, "queue_timeout", None)
    _profile.queue_timeout = seconds
    try:
        yield
    finally:
        _profile.queue_timeout = previous


def queue_timeout_s() -> float:
    """How long a local request may wait for the slot: the thread's profile, else LOCAL_LLM_QUEUE_S (default 90)."""
    override = getattr(_profile, "queue_timeout", None)
    if override is not None:
        return max(0.0, float(override))
    try:
        return float(os.getenv("LOCAL_LLM_QUEUE_S", "90"))
    except ValueError:
        return 90.0


@contextlib.contextmanager
def _local_slot():
    """Take the local server's one request slot and always give it back (timeout, error, or a consumer that stops
    early and closes the stream). A request queued behind a stuck one waits at most LOCAL_LLM_QUEUE_S (or the thread's
    `local_queue_timeout`, 5 s on a phone turn), then fails so the provider chain moves on instead of every later local
    call hanging."""
    if not _mlx.request_lock.acquire(timeout=queue_timeout_s()):
        raise TimeoutError("the local model is busy with another request")
    try:
        yield
    finally:
        _mlx.request_lock.release()


def _mlx_chat(messages, **kw) -> ChatResult:
    base = _mlx.ensure()
    kw.pop("timeout", None)
    with _local_slot():
        t = time.time()
        data = _openai_chat(base, _mlx.model, messages, api_key=None, timeout=LOCAL_REQUEST_TIMEOUT, **kw)
    return ChatResult(data["choices"][0]["message"]["content"] or "", "mlx", _mlx.model,
                      int((time.time() - t) * 1000), data.get("usage") or {})


def _groq_stream_once(models: List[str], messages, info, kw) -> Iterator[str]:
    """One pass, as _groq_chat_once. Once a piece has been yielded the caller has heard it: a failure then ends the
    request (the failed model or key is still noted) instead of starting again on another key."""
    plan = groq_keys.plan(models)
    if not plan:
        raise RuntimeError("GROQ_API_KEY is not set")
    bad: set = set()
    i = 0
    while i < len(plan):
        model, k = plan[i]
        info["model"] = model
        produced = False
        try:
            for piece in _openai_stream("https://api.groq.com/openai/v1", model, messages,
                                        api_key=k["key"], info=info, extra=_groq_extra(model), **kw):
                produced = True
                info["produced"] = True
                yield piece
            return
        except Exception as e:  # noqa: BLE001
            fail = key_failure(e)
            if fail is not None and fail[0] == "auth":
                bad.add(k["label"])
            j = None if produced else _next_attempt(plan, i, e, fail, bad)
            _groq_failed(model, k["label"], e, plan[j] if j is not None else None, fail)
            if j is None:
                raise
            i = j


def _groq_stream(messages, info, **kw) -> Iterator[str]:
    models = _groq_models()
    if not models:
        raise _no_groq_model()
    try:
        yield from _groq_stream_once(models, messages, info, kw)
        return
    except Exception as e:  # noqa: BLE001
        wait = None if info.get("produced") else minute_limit_wait(e)
        if wait is None:
            raise
        log.warning("every Groq model is at its per-minute limit; waiting %.1f s and asking once more instead of "
                    "falling back to the local model", wait)
        time.sleep(wait + 0.2)
    yield from _groq_stream_once(_groq_models() or models, messages, info, kw)


def _openai_compat_stream(messages, info, **kw) -> Iterator[str]:
    url = _openai_base_url()
    if not url:
        raise AppError("MODEL_UNAVAILABLE", "LLM_BASE_URL is not set", 503)
    info["model"] = _openai_model()
    yield from _openai_stream(url, info["model"], messages,
                              api_key=os.getenv("LLM_API_KEY"), info=info, **kw)


def _mlx_stream(messages, info, **kw) -> Iterator[str]:
    base = _mlx.ensure()
    kw.pop("timeout", None)
    info["model"] = _mlx.model
    with _local_slot():  # held for the whole reply: the local server answers one request at a time
        info["lock_held"] = True
        try:
            yield from _openai_stream(base, _mlx.model, messages, api_key=None, timeout=LOCAL_REQUEST_TIMEOUT, info=info, **kw)
        finally:
            info["lock_held"] = False


# ---- the background model: a second local server for jobs that nobody waits for (question index, fact sheet).
# It is a separate `mlx_lm server` process with its own model (BACKGROUND_LOCAL_MODEL), started on demand and stopped by
# services/background_llm.py when the job is over, so the model used for live replies (LOCAL_LLM_MODEL) is never changed
# and no gigabytes stay resident during calls. When BACKGROUND_LOCAL_MODEL is the live model, that server is used as it is.
_bg_lock = threading.Lock()
_bg: Optional[_MlxServer] = None


def _bg_server(model: str) -> _MlxServer:
    global _bg
    if model == _mlx.model:
        return _mlx
    with _bg_lock:
        if _bg is None or _bg.model != model:
            if _bg is not None:
                _bg.stop()
            _bg = _MlxServer(model, "background_llm.log")
            atexit.register(_bg.stop)
        return _bg


def background_unavailable_reason(model: str) -> Optional[str]:
    """Why the background model cannot run here (None when it can)."""
    srv = _bg_server(model)
    if os.getenv("LOCAL_LLM", "").lower() == "off":
        return "LOCAL_LLM=off"
    if not srv.python():
        return "no MLX environment (run: bash backend/scripts/setup_mlx.sh) or not an Apple Silicon Mac"
    if not _hf_complete(model):
        return f"{model} is not downloaded (huggingface-cli download {model})"
    return None


def background_running() -> bool:
    return _bg is not None and _bg.model != _mlx.model and _bg.running


def background_memory_mb() -> Optional[float]:
    return _bg.memory_mb() if background_running() else None


def background_chat(messages: List[dict], *, model: str, json_mode: bool = False, temperature: float = 0.2,
                    max_tokens: int = 800, timeout: float = LOCAL_REQUEST_TIMEOUT) -> ChatResult:
    """One chat completion on the background model (starts its server when needed). Raises on any failure."""
    srv = _bg_server(model)
    base = srv.ensure()
    with srv.request_lock:   # the server answers one request at a time
        t = time.time()
        data = _openai_chat(base, srv.model, messages, api_key=None, json_mode=json_mode, temperature=temperature,
                            max_tokens=max_tokens, timeout=timeout)
    return ChatResult(data["choices"][0]["message"]["content"] or "", "mlx-background", srv.model,
                      int((time.time() - t) * 1000), data.get("usage") or {})


def unload_background() -> bool:
    """Stop the background model's server (frees its memory). True when one was running. The live model's server is
    never stopped here."""
    srv = _bg
    if srv is None or srv.model == _mlx.model or not srv.running:
        return False
    srv.stop()
    log.info("background model unloaded: %s", srv.model)
    return True


PROVIDERS = {
    "groq": (_groq_reason, _groq_chat),
    "openai": (_openai_reason, _openai_compat_chat),
    "mlx": (_mlx.unavailable_reason, _mlx_chat),
}
STREAM_PROVIDERS = {"groq": _groq_stream, "openai": _openai_compat_stream, "mlx": _mlx_stream}


# The latest error of each provider that failed and was skipped, cleared when it answers again. A failing first choice
# is otherwise invisible: the next provider answers and everything looks fine, only slower.
last_failures: Dict[str, str] = {}


def _order() -> List[str]:
    return [p.strip() for p in os.getenv("LLM_PROVIDERS", "groq,openai,mlx").split(",") if p.strip() in PROVIDERS]


# ---------------------------------------------------------------------------------------------- public API

# Providers whose native tool calling (OpenAI `tools` / `tool_calls`) is used. The local model is left out: a 3B model
# does not call tools reliably, so services/tools.py gives it a strict JSON protocol in the prompt instead.
TOOL_PROVIDERS = ("groq", "openai")


def chat(messages: List[dict], *, json_mode: bool = False, temperature: float = 0.3, max_tokens: int = 400,
         timeout: float = 30.0, tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None,
         providers: Optional[List[str]] = None, json_tools: Optional[Iterable[str]] = None) -> ChatResult:
    """Ask the first available provider. In JSON mode the reply text is a JSON object string.
    `providers` limits the choice (still in the configured order), e.g. ["groq", "openai"] to never use the local model.

    With `tools` (OpenAI function definitions) only providers with native tool calling are tried, and the result's
    `tool_calls` lists what the model asked for (`tool_choice` "none" forces a plain answer).

    `json_tools`: the names of the tools the prompt offers through the JSON tool protocol (services/tools.py). A model
    that calls one of them natively instead (Groq's gpt-oss, which the server rejects with tool_use_failed) gets that
    call back as its reply, written in the protocol. Without it, such a call is answered by asking once more for words."""
    if json_mode and not any("json" in (m.get("content") or "").lower() for m in messages):
        messages = [*messages, {"role": "system", "content": "Respond with a single JSON object only."}]
    reasons: Dict[str, str] = {}
    for name in _order():
        if providers is not None and name not in providers:
            reasons[name] = "not allowed for this call"
            continue
        reason_fn, call = PROVIDERS[name]
        reason = reason_fn()
        if reason:
            reasons[name] = reason
            continue
        if tools and name not in TOOL_PROVIDERS:
            reasons[name] = "no native tool calling"
            continue
        try:
            extra = {"tools": tools, "tool_choice": tool_choice} if tools else {}
            if json_tools:
                extra["json_tools"] = tuple(json_tools)
            res = call(messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                       **extra)
            if json_mode:
                res.text = json.dumps(extract_json(res.text))
            _record_usage(name, res.usage)
            last_failures.pop(name, None)
            return res
        except Exception as e:  # noqa: BLE001  try the next provider
            log.warning("LLM provider %s failed: %s", name, e)
            reasons[name] = f"failed: {type(e).__name__}: {str(e)[:200]}"
            last_failures[name] = reasons[name]
    raise LLMUnavailable(reasons)


class ChatStream:
    """Iterate to receive text deltas; afterwards `.result` is the summary (StreamResult).

    The provider chain is walked lazily on the first `next()`: a provider that is not configured, cannot connect or
    fails before producing any text is skipped and the next one is tried; one that fails after producing text raises
    LLMStreamInterrupted (with the partial text) and the chain stops. LLMUnavailable is raised when nothing answered.
    `provider` / `model` are set as soon as the first delta arrives. `info` is the live provider state: the local
    server's `lock_held` flag says it is busy with this reply, so a second local-model call would queue behind it."""

    def __init__(self, messages: List[dict], *, temperature: float, max_tokens: int, timeout: float,
                 total_timeout: Optional[float], json_tools: Optional[Iterable[str]] = None):
        self.messages, self.temperature, self.max_tokens = messages, temperature, max_tokens
        self.json_tools = tuple(json_tools or ())
        self.timeout, self.total_timeout = timeout, total_timeout
        self.provider: Optional[str] = None
        self.model: Optional[str] = None
        self.first_token_ms: Optional[int] = None
        self.text = ""
        self.result: Optional[StreamResult] = None
        self.info: dict = {}
        self._gen: Optional[Iterator[str]] = None

    def __iter__(self) -> Iterator[str]:
        if self._gen is None:
            self._gen = self._run()
        return self._gen

    def close(self) -> None:
        """Stop early (the consumer went away); releases the connection and the local model."""
        if self._gen is not None:
            self._gen.close()

    def _run(self) -> Iterator[str]:
        reasons: Dict[str, str] = {}
        for name in _order():
            reason = PROVIDERS[name][0]()
            if reason:
                reasons[name] = reason
                continue
            info: dict = {}
            self.info = info
            t0 = time.time()
            deadline = t0 + self.total_timeout if self.total_timeout else None
            produced = ""
            stream = STREAM_PROVIDERS[name](self.messages, info, temperature=self.temperature,
                                            max_tokens=self.max_tokens, timeout=self.timeout,
                                            **({"json_tools": self.json_tools} if self.json_tools else {}))
            try:
                for piece in stream:
                    if not produced:
                        self.provider, self.model = name, info.get("model") or ""
                        self.first_token_ms = int((time.time() - t0) * 1000)
                    produced += piece
                    self.text = produced
                    yield piece
                    if deadline and time.time() > deadline:
                        raise TimeoutError(f"reply not finished within {self.total_timeout:g} s")
            except GeneratorExit:  # the consumer stopped listening
                stream.close()
                raise
            except Exception as e:  # noqa: BLE001
                stream.close()
                if produced:
                    log.warning("LLM provider %s stopped mid-reply: %s", name, e)
                    raise LLMStreamInterrupted(name, info.get("model") or "", produced, e) from e
                log.warning("LLM provider %s failed to stream: %s", name, e)
                reasons[name] = f"failed: {type(e).__name__}: {str(e)[:200]}"
                last_failures[name] = reasons[name]
                continue
            usage = info.get("usage") or {}
            if produced:
                self.result = StreamResult(produced, name, info.get("model") or "", int((time.time() - t0) * 1000),
                                           usage, self.first_token_ms)
                _record_usage(name, usage)
                last_failures.pop(name, None)
                return
            log.warning("LLM provider %s streamed an empty reply", name)
            reasons[name] = "failed: empty reply"
        raise LLMUnavailable(reasons)


def chat_stream(messages: List[dict], *, temperature: float = 0.3, max_tokens: int = 400, timeout: float = 30.0,
                total_timeout: Optional[float] = None, json_tools: Optional[Iterable[str]] = None) -> ChatStream:
    """Like `chat()`, but streamed: iterate the returned ChatStream for text deltas (blocking; run it in a thread).

    `timeout` is the longest wait for the next bytes from the provider; `total_timeout` (optional) caps the whole
    reply. After the iteration `.result` holds provider, model, latency, usage and the time to the first token.
    `json_tools` as for chat(): a native call to one of them arrives as one delta, the call in the JSON protocol."""
    return ChatStream(messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                      total_timeout=total_timeout, json_tools=json_tools)


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating code fences or prose around it."""
    text = (text or "").strip()
    try:
        out = json.loads(text)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("model reply contained no JSON object")
    return json.loads(m.group(0))


def status() -> dict:
    """Which provider would answer now, and why the others would not."""
    providers = []
    active = None
    for name in _order():
        reason = PROVIDERS[name][0]()
        providers.append({"name": name, "available": reason is None, "reason": reason,
                          "last_error": last_failures.get(name)})
        if reason is None and active is None:
            active = name
    model = {"groq": os.getenv("GROQ_CHAT_MODEL", DEFAULT_GROQ_MODEL),
             "openai": os.getenv("LLM_MODEL", "llama3.2"), "mlx": _mlx.model}.get(active)
    return {"active": active, "model": model, "local_running": _mlx.running, "providers": providers,
            "usage_today": usage_today(), "groq_paused": paused_models(), "groq_keys": groq_keys.status()}


def warm_local() -> None:
    """Start the local model ahead of the first request (used by the Overview warm-up)."""
    if _mlx.unavailable_reason() is None:
        _mlx.ensure()


# ---------------------------------------------------------------------------------------------- connection prewarm

PREWARM_MIN_INTERVAL_S = 60.0
_prewarm_lock = threading.Lock()
_last_prewarm: Optional[float] = None


def _origin(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}/"


def _prewarm_targets() -> List[str]:
    """The hosted providers a reply may use right now (configured, not the local model), as origins."""
    out: List[str] = []
    for name in _order():
        if name == "mlx" or PROVIDERS[name][0]():
            continue
        base_url = "https://api.groq.com/" if name == "groq" else (_openai_base_url() or "")
        if not base_url:
            continue
        url = _origin(base_url)
        if url not in out:
            out.append(url)
    return out


def _prewarm_now() -> None:
    client = _client()
    if client is None:
        return
    for url in _prewarm_targets():
        started = time.time()
        try:
            # A HEAD to the origin: no key is sent and no token is spent, the answer (usually a 404) does not matter. It
            # leaves a TCP + TLS connection in the pool, which the first real request of a reply then reuses.
            client.head(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
                        timeout=httpx.Timeout(5.0))
            log.info("LLM connection to %s warm (%d ms)", url, int((time.time() - started) * 1000))
        except Exception as e:  # noqa: BLE001 - a warm-up must never matter
            log.info("LLM connection prewarm of %s failed: %s", url, type(e).__name__)


def prewarm(*, wait: bool = False) -> None:
    """Open the connection to the hosted language model ahead of the first reply (a phone starts ringing, the backend
    starts), so the reply does not pay the TCP + TLS handshake (about 70-100 ms to Groq).

    Cheap (one HEAD request, no key, no tokens), never raises, and at most once per PREWARM_MIN_INTERVAL_S (a call
    inside that window does nothing). It returns at once: the request is made on a short-lived daemon thread, so it is
    safe to call straight from async code (or through asyncio.to_thread). `wait=True` makes the request on the calling
    thread and returns when it is done (the startup warm-up uses it to time the step)."""
    global _last_prewarm
    try:
        with _prewarm_lock:
            now = time.monotonic()
            if _last_prewarm is not None and now - _last_prewarm < PREWARM_MIN_INTERVAL_S:
                return
            _last_prewarm = now
        if wait:
            _prewarm_now()
        else:
            threading.Thread(target=_prewarm_now, name="llm-prewarm", daemon=True).start()
    except Exception as e:  # noqa: BLE001
        log.info("LLM prewarm skipped: %s", e)
