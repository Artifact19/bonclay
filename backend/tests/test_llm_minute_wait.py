"""When every Groq model is at its per-minute limit and Groq says "try again in 1.6s", wait that long and ask Groq once
more instead of dropping to the slow local model (services/llm.py minute_limit_wait). A daily quota or a long wait
still falls through as before."""
import pytest

from services import llm

MINUTE_429 = ('{"error":{"message":"Rate limit reached for model `x` in organization `o` service tier `on_demand` on '
              'tokens per minute (TPM): Limit 8000, Used 6918, Requested 2401. Please try again in 1.2s. Need more '
              'tokens? Upgrade to Dev Tier","type":"tokens","code":"rate_limit_exceeded"}}')
DAILY_429 = ('{"error":{"message":"Rate limit reached for model `x` on tokens per day (TPD): Limit 200000, Used 199000, '
             'Requested 2000. Please try again in 7m12.5s."}}')
LONG_429 = MINUTE_429.replace("try again in 1.2s", "try again in 25s")


@pytest.fixture(autouse=True)
def groq_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GROQ_CHAT_MODEL", "m1")
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "m2,m3")
    monkeypatch.delenv("GROQ_MINUTE_WAIT_S", raising=False)
    llm._quota_until.clear()                                  # the pause state lives in services/groq_keys.py
    monkeypatch.setattr(llm, "_dead_models", {})
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(llm, "_chat_result", lambda data, provider, model, started: llm.ChatResult(data["text"], provider, model, 1, {}))
    return slept


def test_minute_limit_wait_reads_the_short_wait_only():
    assert llm.minute_limit_wait(llm.ProviderHTTPError(429, MINUTE_429)) == pytest.approx(1.2)
    assert llm.minute_limit_wait(llm.ProviderHTTPError(429, DAILY_429)) is None      # daily quota: pause the model
    assert llm.minute_limit_wait(llm.ProviderHTTPError(429, LONG_429)) is None       # 25 s: not worth waiting
    assert llm.minute_limit_wait(llm.ProviderHTTPError(500, MINUTE_429)) is None
    assert llm.minute_limit_wait(RuntimeError("x")) is None


def test_minute_wait_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("GROQ_MINUTE_WAIT_S", "0")
    assert llm.minute_limit_wait(llm.ProviderHTTPError(429, MINUTE_429)) is None


def test_chat_waits_once_and_asks_groq_again(groq_env, monkeypatch):
    calls = []

    def fake_chat(base, model, messages, **kw):
        calls.append(model)
        if len(calls) <= 3:                      # every model at the minute limit on the first pass
            raise llm.ProviderHTTPError(429, MINUTE_429)
        return {"text": "namaste"}

    monkeypatch.setattr(llm, "_openai_chat", fake_chat)
    out = llm._groq_chat([{"role": "user", "content": "hi"}], json_mode=False, temperature=0.2, max_tokens=50, timeout=5)
    assert out.text == "namaste" and out.model == "m1"
    assert calls == ["m1", "m2", "m3", "m1"]
    assert groq_env == [pytest.approx(1.4)]     # the asked-for 1.2 s plus a small margin


def test_chat_does_not_wait_for_a_daily_quota_or_a_long_wait(groq_env, monkeypatch):
    for body in (DAILY_429, LONG_429):
        calls = []

        def fake_chat(base, model, messages, **kw):
            calls.append(model)
            raise llm.ProviderHTTPError(429, body)

        monkeypatch.setattr(llm, "_openai_chat", fake_chat)
        llm._quota_until.clear()
        with pytest.raises(llm.ProviderHTTPError):
            llm._groq_chat([{"role": "user", "content": "hi"}], json_mode=False, temperature=0.2, max_tokens=50, timeout=5)
        assert len(calls) == 3 and groq_env == []


def test_stream_waits_once_too_but_never_after_text_was_produced(groq_env, monkeypatch):
    calls = []

    def fake_stream(base, model, messages, **kw):
        calls.append(model)
        if len(calls) <= 3:
            raise llm.ProviderHTTPError(429, MINUTE_429)
        yield "Hello "
        yield "there."

    monkeypatch.setattr(llm, "_openai_stream", fake_stream)
    info = {}
    text = "".join(llm._groq_stream([{"role": "user", "content": "hi"}], info, temperature=0.2, max_tokens=50, timeout=5))
    assert text == "Hello there." and info["model"] == "m1" and calls == ["m1", "m2", "m3", "m1"]
    assert groq_env == [pytest.approx(1.4)]

    # a stream that fails after producing text is not retried: the caller already heard part of it
    calls.clear()
    groq_env.clear()

    def half_stream(base, model, messages, **kw):
        calls.append(model)
        yield "Half"
        raise llm.ProviderHTTPError(429, MINUTE_429)

    monkeypatch.setattr(llm, "_openai_stream", half_stream)
    info = {}
    got = []
    with pytest.raises(llm.ProviderHTTPError):
        for piece in llm._groq_stream([{"role": "user", "content": "hi"}], info, temperature=0.2, max_tokens=50, timeout=5):
            got.append(piece)
    assert got == ["Half"] and calls == ["m1"] and groq_env == []
