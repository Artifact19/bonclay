"""A Groq model that has used its DAILY quota (429 "tokens per day (TPD)" / "requests per day (RPD)") is skipped until
the quota resets instead of costing every turn a failed request; a per-MINUTE 429 only moves this turn to the next
model."""
import io
import json
import logging
import time
import urllib.error

import pytest

from services import llm

TPD = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on tokens "
       "per day (TPD): Limit 200000, Used 199514, Requested 2210. Please try again in 12m35.5s. Need more tokens? "
       "Upgrade to Dev Tier today at https://console.groq.com/settings/billing")
TPM = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on tokens "
       "per minute (TPM): Limit 8000, Used 7000, Requested 2210. Please try again in 9.6s.")


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def err(req, message, headers=None, code=429):
    body = {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}
    return urllib.error.HTTPError(req.full_url, code, "Too Many Requests", headers or {},
                                  io.BytesIO(json.dumps(body).encode()))


def ok():
    return Resp(json.dumps({"choices": [{"message": {"content": "OK"}}], "usage": {}}).encode())


@pytest.fixture()
def groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    monkeypatch.delenv("GROQ_CHAT_MODEL", raising=False)
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b")
    llm._dead_models.clear()
    llm._quota_until.clear()
    yield
    llm._dead_models.clear()
    llm._quota_until.clear()


def script(monkeypatch, plan):
    sent = []

    def fake(req, timeout=None):
        body = json.loads(req.data)
        sent.append(body["model"])
        out = plan(req, body)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(llm, "_urlopen", fake)
    return sent


def pause_of(message, headers=None):
    return llm.daily_quota_pause(llm.ProviderHTTPError(429, json.dumps({"error": {"message": message}}), headers))


def test_reading_the_reset_time():
    assert pause_of(TPD) == (pytest.approx(755.5), "tokens")
    assert pause_of(TPM) is None                                                  # per minute: not a daily limit
    assert pause_of("Rate limit reached on requests per day (RPD): Limit 1000, Used 1000. Please try again in "
                    "1h2m3s.") == (3723.0, "requests")
    # no "try again in": the retry-after header, then (requests per day) x-ratelimit-reset-requests
    assert pause_of("on tokens per day (TPD): Limit 200000", {"Retry-After": "600"}) == (600.0, "tokens")
    assert pause_of("on requests per day (RPD): Limit 1000", {"x-ratelimit-reset-requests": "4h3m21.6s"}) == (
        pytest.approx(14601.6), "requests")
    # nothing at all: until the next midnight UTC
    seconds, kind = pause_of("on tokens per day (TPD): Limit 200000")
    assert kind == "tokens" and abs(seconds - llm._until_midnight_utc()) < 5
    assert llm.daily_quota_pause(RuntimeError("HTTP 429: tokens per day")) is None  # no status or body to read
    assert llm._duration_s("250ms") == 0.25 and llm._duration_s("7m12.5s") == 432.5 and llm._duration_s("x") is None


def test_daily_quota_skips_the_model_until_it_resets(groq, monkeypatch, caplog):
    sent = script(monkeypatch, lambda req, body: err(req, TPD) if body["model"] == llm.DEFAULT_GROQ_MODEL else ok())
    with caplog.at_level(logging.WARNING, logger="bonclay.llm"):
        assert llm.chat([{"role": "user", "content": "hi"}]).model == "openai/gpt-oss-20b"
        assert llm.chat([{"role": "user", "content": "hi"}]).model == "openai/gpt-oss-20b"
    assert sent[:3] == [llm.DEFAULT_GROQ_MODEL, "openai/gpt-oss-20b", "openai/gpt-oss-20b"]  # qwen asked once
    assert llm.DEFAULT_GROQ_MODEL not in sent[1:]
    daily = [r for r in caplog.records if "daily tokens quota" in r.getMessage()]
    assert len(daily) == 1 and "skipping it until" in daily[0].getMessage()      # logged once, clearly
    assert llm.DEFAULT_GROQ_MODEL in llm.status()["groq_paused"]
    # once the reset time has passed the model is used again
    until, wall = llm._quota_until[("1", llm.DEFAULT_GROQ_MODEL)]            # keyed by (key label, model)
    llm._quota_until[("1", llm.DEFAULT_GROQ_MODEL)] = (time.monotonic() - 1, wall)
    assert llm._groq_models()[0] == llm.DEFAULT_GROQ_MODEL


def test_per_minute_429_keeps_the_short_behaviour(groq, monkeypatch):
    sent = script(monkeypatch, lambda req, body: err(req, TPM, {"retry-after": "10"})
                  if body["model"] == llm.DEFAULT_GROQ_MODEL else ok())
    llm.chat([{"role": "user", "content": "hi"}])
    llm.chat([{"role": "user", "content": "hi"}])
    assert sent == [llm.DEFAULT_GROQ_MODEL, "openai/gpt-oss-20b"] * 2            # qwen is tried on every turn
    assert llm._quota_until == {}


def test_a_streamed_daily_429_pauses_the_model_too(groq, monkeypatch):
    def plan(req, body):
        if body["model"] == llm.DEFAULT_GROQ_MODEL:
            return err(req, TPD)
        return Resp(b'data: {"choices":[{"delta":{"content":"Hi."}}]}\n\ndata: [DONE]\n\n')

    sent = script(monkeypatch, plan)
    assert "".join(llm.chat_stream([{"role": "user", "content": "hi"}])) == "Hi."
    assert "".join(llm.chat_stream([{"role": "user", "content": "hi"}])) == "Hi."
    assert sent == [llm.DEFAULT_GROQ_MODEL, "openai/gpt-oss-20b", "openai/gpt-oss-20b"]


def test_every_model_out_for_the_day_skips_groq_without_a_request(groq, monkeypatch):
    sent = script(monkeypatch, lambda req, body: err(req, TPD))
    with pytest.raises(llm.LLMUnavailable):
        llm.chat([{"role": "user", "content": "hi"}])
    assert sent == [llm.DEFAULT_GROQ_MODEL, "openai/gpt-oss-20b"]
    sent.clear()
    with pytest.raises(llm.LLMUnavailable) as e:
        llm.chat([{"role": "user", "content": "hi"}])
    assert sent == [] and "daily quota" in e.value.reasons["groq"]
    with pytest.raises(llm.LLMUnavailable):
        list(llm.chat_stream([{"role": "user", "content": "hi"}]))
    assert sent == []
    groq_row = next(p for p in llm.status()["providers"] if p["name"] == "groq")
    assert groq_row["available"] is False and "daily quota" in groq_row["reason"]
