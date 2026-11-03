"""Several Groq accounts (services/groq_keys.py): GROQ_API_KEY is account 1, GROQ_API_KEY_2 .. _9 are the others. Groq's
limits are per account and per model, so a request that one account cannot take goes out on the next account's key
(best model on every account before a weaker model). Chat, streaming and Whisper are faked; no key value is ever logged."""
import asyncio
import io
import json
import logging
import urllib.error

import httpx
import pytest

from services import groq_keys, llm

KEYS = {"GROQ_API_KEY": "key-one-aaaa", "GROQ_API_KEY_2": "key-two-bbbb", "GROQ_API_KEY_3": "key-three-cccc"}
LABEL_OF = {v: k.rsplit("_", 1)[-1] if k != "GROQ_API_KEY" else "1" for k, v in KEYS.items()}
M1 = llm.DEFAULT_GROQ_MODEL
M2 = "openai/gpt-oss-20b"

TPD = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on tokens "
       "per day (TPD): Limit 200000, Used 199514, Requested 2210. Please try again in 12m35.5s.")
TPM = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on tokens "
       "per minute (TPM): Limit 8000, Used 7000, Requested 2210. Please try again in 9.6s.")
ASD = ("Rate limit reached for model `whisper-large-v3-turbo` in organization `org_x` on audio seconds per day (ASD): "
       "Limit 28800, Used 28790, Requested 30. Please try again in 2h3m.")
HI = [{"role": "user", "content": "hi"}]


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_err(req, message, code=429, headers=None):
    body = {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}
    return urllib.error.HTTPError(req.full_url, code, "x", headers or {}, io.BytesIO(json.dumps(body).encode()))


def ok():
    return Resp(json.dumps({"choices": [{"message": {"content": "OK"}}], "usage": {}}).encode())


def stream_ok(text="Hi."):
    return Resp(f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n\ndata: [DONE]\n\n'.encode())


@pytest.fixture()
def groq(monkeypatch):
    """Three accounts, two models, chat calls recorded as (model, key label)."""
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    monkeypatch.delenv("GROQ_CHAT_MODEL", raising=False)
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", M2)
    monkeypatch.setenv("GROQ_MINUTE_WAIT_S", "0")
    llm._dead_models.clear()
    groq_keys.reset()
    yield
    llm._dead_models.clear()
    groq_keys.reset()


def script(monkeypatch, plan):
    """plan(model, label, req) -> response or exception. Returns the list of (model, label) attempts."""
    sent = []

    def fake(req, timeout=None):
        body = json.loads(req.data)
        label = LABEL_OF[req.get_header("Authorization").split(" ", 1)[1]]
        sent.append((body["model"], label))
        out = plan(body["model"], label, req)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(llm, "_urlopen", fake)
    return sent


# ---------------------------------------------------------------------------------------------- configuration

def test_keys_are_read_in_account_order_with_gaps(monkeypatch):
    for n in range(2, 10):
        monkeypatch.delenv(f"GROQ_API_KEY_{n}", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEY_2", "  ")          # empty: ignored
    monkeypatch.setenv("GROQ_API_KEY_3", "k3")
    monkeypatch.setenv("GROQ_API_KEY_5", "k3")          # the same account twice: once
    monkeypatch.setenv("GROQ_API_KEY_9", " k9 ")
    assert groq_keys.keys() == [{"label": "1", "key": "k1"}, {"label": "3", "key": "k3"}, {"label": "9", "key": "k9"}]


def test_has_any_is_true_with_only_a_suffixed_key(monkeypatch):
    assert groq_keys.has_any() is False                  # conftest blanks GROQ_API_KEY and _2.._9
    monkeypatch.setenv("GROQ_API_KEY_2", "only-second")
    assert groq_keys.has_any() is True
    assert llm._groq_reason() is None
    monkeypatch.setenv("GROQ_API_KEY_2", "")
    assert llm._groq_reason() == "GROQ_API_KEY is not set"


def test_conftest_blanks_every_account():
    import os
    assert all(not os.environ.get("GROQ_API_KEY" + ("" if n == 1 else f"_{n}")) for n in range(1, 10))


def test_the_other_checks_count_a_suffixed_key(monkeypatch):
    import backend
    from routers import say_it_once
    assert say_it_once._has_groq() is False
    assert backend._llm_timeout(5.0) == 90.0
    monkeypatch.setenv("GROQ_API_KEY_4", "only-fourth")
    assert say_it_once._has_groq() is True
    assert backend._llm_timeout(5.0) == 5.0


# ---------------------------------------------------------------------------------------------- chat rotation

def test_a_daily_429_moves_the_same_model_to_the_next_key(groq, monkeypatch, caplog):
    sent = script(monkeypatch, lambda m, label, req: http_err(req, TPD) if label == "1" and m == M1 else ok())
    with caplog.at_level(logging.WARNING, logger="bonclay.llm"):
        out = llm.chat(HI)
    assert out.model == M1 and sent == [(M1, "1"), (M1, "2")]                 # best model on account 2, not the weaker one
    assert groq_keys.is_daily_paused("1", M1) and not groq_keys.is_daily_paused("1", M2)   # only that pair
    assert not groq_keys.is_daily_paused("2", M1)
    assert llm.paused_models() == {}                                          # the model is still usable on keys 2 and 3
    assert "Groq key 1 hit the daily limit for %s; trying key 2" % M1 in caplog.text
    sent.clear()
    assert llm.chat(HI).model == M1 and sent == [(M1, "2")]                   # key 1 is skipped for M1 until the reset
    assert groq_keys.status()[0] == {"label": "1", "state": "paused",
                                     "paused_models": [{"model": M1, "until": groq_keys.daily_reset("1", M1).isoformat(timespec="seconds")}]}


def test_a_per_minute_429_tries_the_next_key_at_once(groq, monkeypatch, caplog):
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    sent = script(monkeypatch, lambda m, label, req: http_err(req, TPM) if label == "1" else ok())
    with caplog.at_level(logging.WARNING, logger="bonclay.llm"):
        assert llm.chat(HI).model == M1
    assert sent == [(M1, "1"), (M1, "2")] and slept == []
    assert "Groq key 1 hit the per-minute limit for %s; trying key 2" % M1 in caplog.text
    assert not groq_keys.is_daily_paused("1", M1)                             # a cool-down, not a daily pause
    assert groq_keys.is_cooling("1", M1)
    assert 8 < groq_keys._cooldown[("1", M1)] - llm.time.monotonic() <= 10    # Groq's "try again in 9.6s"
    sent.clear()
    llm.chat(HI)
    assert sent == [(M1, "2")]                                                # key 1 is left alone while it cools down
    groq_keys._cooldown[("1", M1)] = llm.time.monotonic() - 1
    sent.clear()
    llm.chat(HI)
    assert sent[0] == (M1, "1")                                               # and used again after


def test_a_per_minute_429_without_a_hint_cools_down_for_20_seconds(groq, monkeypatch):
    script(monkeypatch, lambda m, label, req: http_err(req, "Rate limit reached on tokens per minute (TPM)")
           if label == "1" else ok())
    llm.chat(HI)
    assert 19 < groq_keys._cooldown[("1", M1)] - llm.time.monotonic() <= 20


def test_a_refused_key_is_marked_and_the_next_key_answers(groq, monkeypatch, caplog):
    sent = script(monkeypatch, lambda m, label, req: http_err(req, "Invalid API Key", code=401) if label == "1" else ok())
    with caplog.at_level(logging.WARNING, logger="bonclay.llm"):
        assert llm.chat(HI).model == M1
    assert sent == [(M1, "1"), (M1, "2")]
    assert groq_keys.is_auth_failed("1") and groq_keys.status()[0]["state"] == "auth_failed"
    assert "Groq key 1 hit the auth limit for %s; trying key 2" % M1 in caplog.text
    assert "Groq key 1 was refused" in caplog.text
    sent.clear()
    llm.chat(HI)
    assert sent == [(M1, "2")]                                                # not asked again for an hour
    for value in KEYS.values():
        assert value not in caplog.text                                       # no key material in any log line
    # after the hour (or with a replaced key) it is tried again
    groq_keys._auth_failed["1"] = (llm.time.monotonic() - 1, groq_keys._auth_failed["1"][1])
    assert not groq_keys.is_auth_failed("1")


def test_a_replaced_key_is_not_held_against_the_new_one(groq, monkeypatch):
    script(monkeypatch, lambda m, label, req: http_err(req, "Invalid API Key", code=403) if label == "1" else ok())
    llm.chat(HI)
    assert groq_keys.is_auth_failed("1")
    monkeypatch.setenv("GROQ_API_KEY", "a-new-key-for-account-one")
    assert not groq_keys.is_auth_failed("1")


def test_every_key_tried_on_a_model_before_the_next_model(groq, monkeypatch):
    def plan(m, label, req):
        if m == M1:
            return http_err(req, TPD if label != "3" else TPM)               # all three accounts are out of M1
        return ok()

    sent = script(monkeypatch, plan)
    assert llm.chat(HI).model == M2
    assert sent == [(M1, "1"), (M1, "2"), (M1, "3"), (M2, "1")]               # the weaker model only after all keys
    assert llm.paused_models() == {}                                          # key 3 only cools down: M1 is not "paused"
    sent.clear()
    assert llm.chat(HI).model == M2
    assert sent == [(M2, "1")]                                                # keys 1 and 2 paused, 3 cooling: straight to M2


def test_all_keys_paused_for_one_model_goes_to_the_next_model_only_then(groq, monkeypatch):
    for label in ("1", "2", "3"):
        groq_keys.pause_daily(label, M1, 600)
    sent = script(monkeypatch, lambda m, label, req: ok())
    assert llm.chat(HI).model == M2 and sent == [(M2, "1")]                  # M1 is never tried
    assert list(llm.paused_models()) == [M1]
    assert M1 not in llm._groq_models()


def test_other_retryable_errors_still_move_to_the_next_model(groq, monkeypatch):
    sent = script(monkeypatch, lambda m, label, req: http_err(req, "boom", code=503) if m == M1 else ok())
    assert llm.chat(HI).model == M2
    assert sent == [(M1, "1"), (M2, "1")]                                     # about the model, not the key: keys 2, 3 skipped
    assert not any(groq_keys.is_cooling(k, M1) for k in "123") and not groq_keys.is_auth_failed("1")


def test_a_gone_model_is_dead_for_every_key(groq, monkeypatch):
    sent = script(monkeypatch, lambda m, label, req: http_err(req, "model_not_found", code=404) if m == M1 else ok())
    assert llm.chat(HI).model == M2
    sent.clear()
    llm.chat(HI)
    assert sent == [(M2, "1")]


def test_a_non_retryable_error_is_raised_on_the_first_key(groq, monkeypatch):
    sent = script(monkeypatch, lambda m, label, req: http_err(req, "bad request", code=400))
    with pytest.raises(llm.LLMUnavailable):
        llm.chat(HI)
    assert sent == [(M1, "1")]


def test_the_last_key_failure_is_raised_and_the_chain_moves_on(groq, monkeypatch):
    sent = script(monkeypatch, lambda m, label, req: http_err(req, "Invalid API Key", code=401))
    with pytest.raises(llm.LLMUnavailable):
        llm.chat(HI)
    assert sent == [(M1, "1"), (M1, "2"), (M1, "3")]                          # a refused key is not tried on the next model
    sent.clear()
    with pytest.raises(llm.LLMUnavailable):
        llm.chat(HI)
    assert len(sent) == 3                                                     # all refused: tried anyway, as a single key always was


def test_all_keys_out_for_the_day_names_the_key_count(groq, monkeypatch):
    sent = script(monkeypatch, lambda m, label, req: http_err(req, TPD))
    with pytest.raises(llm.LLMUnavailable) as e:
        llm.chat(HI)
    assert len(sent) == 6                                                     # 2 models x 3 keys
    sent.clear()
    with pytest.raises(llm.LLMUnavailable) as e:
        llm.chat(HI)
    assert sent == []
    assert "all 3 Groq keys have used the daily quota of every model (resets: " in e.value.reasons["groq"]
    assert str(llm._no_groq_model()).startswith("HTTP 429: all 3 Groq keys have used the daily quota of every model")
    row = next(p for p in llm.status()["providers"] if p["name"] == "groq")
    assert row["available"] is False and "all 3 Groq keys" in row["reason"]
    assert set(llm.status()["groq_paused"]) == {M1, M2}


def test_one_key_still_says_every_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", KEYS["GROQ_API_KEY"])
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", M2)
    groq_keys.reset()
    script(monkeypatch, lambda m, label, req: http_err(req, TPD))
    with pytest.raises(llm.LLMUnavailable):
        llm.chat(HI)
    assert str(llm._no_groq_model()).startswith("HTTP 429: every Groq model has used its daily quota")


def test_one_key_keeps_the_old_behaviour(monkeypatch):
    """No cool-down on a single key (the per-minute wait handles it), a refused key is raised and tried again."""
    monkeypatch.setenv("GROQ_API_KEY", KEYS["GROQ_API_KEY"])
    monkeypatch.setenv("LLM_PROVIDERS", "groq")
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", M2)
    monkeypatch.setenv("GROQ_MINUTE_WAIT_S", "0")
    groq_keys.reset()
    sent = script(monkeypatch, lambda m, label, req: http_err(req, TPM) if m == M1 else ok())
    llm.chat(HI)
    llm.chat(HI)
    assert sent == [(M1, "1"), (M2, "1")] * 2 and not groq_keys._cooldown
    sent2 = script(monkeypatch, lambda m, label, req: http_err(req, "Invalid API Key", code=401))
    for _ in range(2):
        with pytest.raises(llm.LLMUnavailable):
            llm.chat(HI)
    assert sent2 == [(M1, "1")] * 2


def test_the_minute_wait_still_applies_when_every_key_is_at_the_limit(groq, monkeypatch):
    monkeypatch.setenv("GROQ_MINUTE_WAIT_S", "3")
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    calls = []

    def plan(m, label, req):
        calls.append((m, label))
        return http_err(req, TPM.replace("9.6s", "1.2s")) if len(calls) <= 6 else ok()

    script(monkeypatch, plan)
    assert llm.chat(HI).model == M1
    assert calls[:6] == [(M1, "1"), (M1, "2"), (M1, "3"), (M2, "1"), (M2, "2"), (M2, "3")] and len(calls) == 7
    assert slept == [pytest.approx(1.4)]


# ---------------------------------------------------------------------------------------------- streaming

def test_a_stream_switches_keys_before_the_first_token(groq, monkeypatch):
    sent = script(monkeypatch, lambda m, label, req: http_err(req, TPD) if label == "1" else stream_ok("Hello."))
    assert "".join(llm.chat_stream(HI)) == "Hello."
    assert sent == [(M1, "1"), (M1, "2")] and groq_keys.is_daily_paused("1", M1)


def test_a_stream_switches_on_a_refused_key_too(groq, monkeypatch):
    sent = script(monkeypatch, lambda m, label, req: http_err(req, "Invalid API Key", code=401) if label != "3" else stream_ok())
    assert "".join(llm.chat_stream(HI)) == "Hi."
    assert sent == [(M1, "1"), (M1, "2"), (M1, "3")] and groq_keys.is_auth_failed("2")


def test_a_stream_that_failed_midway_does_not_start_over_on_another_key(groq, monkeypatch):
    calls = []

    def half(base, model, messages, **kw):
        calls.append(kw["api_key"])
        yield "Half"
        raise llm.ProviderHTTPError(429, TPD)

    monkeypatch.setattr(llm, "_openai_stream", half)
    got = []
    with pytest.raises(llm.ProviderHTTPError):
        for piece in llm._groq_stream(HI, {}, temperature=0.2, max_tokens=50, timeout=5):
            got.append(piece)
    assert got == ["Half"] and calls == [KEYS["GROQ_API_KEY"]]
    assert groq_keys.is_daily_paused("1", M1)                                 # still noted, for the next request


# ---------------------------------------------------------------------------------------------- status

def test_status_lists_every_key_by_label_and_no_key_material(groq, monkeypatch):
    script(monkeypatch, lambda m, label, req: http_err(req, TPD) if label == "1" else
           (http_err(req, "Invalid API Key", code=401) if label == "2" else ok()))
    llm.chat(HI)
    status = llm.status()
    text = json.dumps(status)
    for value in KEYS.values():
        assert value not in text
    assert "key-" not in text and "gsk_" not in text
    rows = {r["label"]: r for r in status["groq_keys"]}
    assert set(rows) == {"1", "2", "3"}
    assert rows["1"]["state"] == "paused" and rows["1"]["paused_models"][0]["model"] == M1
    assert rows["1"]["paused_models"][0]["until"].endswith("+00:00")
    assert rows["2"] == {"label": "2", "state": "auth_failed", "paused_models": []}
    assert rows["3"] == {"label": "3", "state": "active", "paused_models": []}


def test_status_is_empty_without_keys():
    assert llm.status()["groq_keys"] == []


def test_offline_mode_still_blocks_groq_with_keys(groq, monkeypatch):
    monkeypatch.setenv("BONCLAY_OFFLINE", "1")
    assert "Offline mode" in llm._groq_reason()
    with pytest.raises(llm.LLMUnavailable) as e:
        llm.chat(HI)
    assert "Offline mode" in e.value.reasons["groq"]


# ---------------------------------------------------------------------------------------------- speech recognition

def sdk_error(cls, status, message):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions")
    return cls(message, response=httpx.Response(status, request=request), body=None)


class FakeAsr:
    """A groq.Groq stand-in: transcriptions.create raises `error` (or returns the text) and records the call."""

    def __init__(self, label, error=None, text="namaste"):
        self.label, self.error, self.text, self.calls = label, error, text, []
        outer = self

        class Transcriptions:
            def create(self, **kw):
                outer.calls.append(kw["model"])
                if outer.error is not None:
                    raise outer.error
                return type("R", (), {"text": outer.text})()

        self.audio = type("A", (), {"transcriptions": Transcriptions()})()


@pytest.fixture()
def asr(app_client, monkeypatch):
    import backend
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    groq_keys.reset()
    clients = {label: FakeAsr(label) for label in "123"}
    monkeypatch.setattr(backend, "groq_client", clients["1"])               # what the SDK client of account 1 is
    monkeypatch.setattr(groq_keys, "sdk_client", lambda label: clients.get(label))
    yield backend, clients
    groq_keys.reset()


def transcribe(backend):
    return asyncio.run(backend._speech_to_text(b"RIFF....WAVEfmt ", "turn", "en"))


def test_whisper_retries_on_the_next_key_after_a_rate_limit(asr, caplog):
    import groq
    backend, clients = asr
    clients["1"].error = sdk_error(groq.RateLimitError, 429, "Rate limit reached for model `whisper-large-v3-turbo`: " + ASD)
    with caplog.at_level(logging.WARNING):
        assert transcribe(backend) == "namaste"
    assert clients["1"].calls == ["whisper-large-v3-turbo"] and clients["2"].calls == ["whisper-large-v3-turbo"]
    assert clients["3"].calls == []
    assert groq_keys.is_daily_paused("1", "whisper-large-v3-turbo")           # audio seconds per day: paused for the day
    assert "Groq key 1 hit the daily limit for whisper-large-v3-turbo; trying key 2" in caplog.text
    clients["1"].calls.clear()
    assert transcribe(backend) == "namaste" and clients["1"].calls == []       # key 1 is skipped until its reset
    assert not groq_keys.is_daily_paused("1", M1)                              # and only for Whisper


def test_whisper_retries_on_the_next_key_after_a_refused_key(asr):
    import groq
    backend, clients = asr
    clients["1"].error = sdk_error(groq.AuthenticationError, 401, "Invalid API Key")
    assert transcribe(backend) == "namaste" and clients["2"].calls
    assert groq_keys.is_auth_failed("1")


def test_whisper_rotates_on_a_plain_status_code_error(asr):
    import groq
    backend, clients = asr
    clients["1"].error = sdk_error(groq.APIStatusError, 429, "Rate limit reached on audio seconds per hour (ASH)")
    clients["2"].error = sdk_error(groq.PermissionDeniedError, 403, "forbidden")
    assert transcribe(backend) == "namaste" and clients["3"].calls
    assert groq_keys.is_cooling("1", "whisper-large-v3-turbo") and groq_keys.is_auth_failed("2")


def test_whisper_does_not_rotate_on_other_errors_or_after_the_last_key(asr):
    import groq
    backend, clients = asr
    clients["1"].error = sdk_error(groq.InternalServerError, 500, "oops")
    assert transcribe(backend) is None and clients["2"].calls == []           # the transcript is lost, as before
    clients["1"].error = None
    for label in "123":
        clients[label].error = sdk_error(groq.RateLimitError, 429, "Rate limit reached on audio seconds per hour (ASH)")
    assert transcribe(backend) is None and all(clients[x].calls for x in "123")


def test_whisper_with_one_key_uses_the_groq_client_only(app_client, monkeypatch):
    import backend
    monkeypatch.setenv("GROQ_API_KEY", KEYS["GROQ_API_KEY"])
    groq_keys.reset()
    only = FakeAsr("1")
    monkeypatch.setattr(backend, "groq_client", only)
    monkeypatch.setattr(groq_keys, "sdk_client", lambda label: pytest.fail("no other account to ask"))
    assert transcribe(backend) == "namaste" and only.calls


def test_sdk_clients_are_cached_one_per_key(monkeypatch):
    import groq
    monkeypatch.setenv("GROQ_API_KEY_2", "key-two-bbbb")
    groq_keys.reset()
    a = groq_keys.sdk_client("2")
    assert isinstance(a, groq.Groq) and groq_keys.sdk_client("2") is a
    assert groq_keys.sdk_client("3") is None
    monkeypatch.setenv("GROQ_API_KEY_2", "key-two-changed")
    assert groq_keys.sdk_client("2") is not a


# ---------------------------------------------------------------------------------------------- document Q&A

def test_document_qa_stream_rotates_on_a_rate_limit(monkeypatch):
    from routers import rag
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    groq_keys.reset()
    used = []

    async def fake_with(httpx_mod, api_key, question, context):
        used.append(api_key)
        if api_key == KEYS["GROQ_API_KEY"]:
            raise rag._KeyRefused(429, TPD, {})
        if api_key == KEYS["GROQ_API_KEY_2"]:
            raise rag._KeyRefused(401, "Invalid API Key", {})
        yield "from "
        yield "three"

    monkeypatch.setattr(rag, "_groq_stream_with", fake_with)

    async def run():
        return [t async for t in rag._groq_stream("q", "ctx")]

    assert asyncio.run(run()) == ["from ", "three"]
    assert used == list(KEYS.values())
    assert groq_keys.is_daily_paused("1", rag.GROQ_MODEL) and groq_keys.is_auth_failed("2")


def test_document_qa_works_with_only_a_suffixed_key(monkeypatch):
    from routers import rag
    monkeypatch.setenv("GROQ_API_KEY_2", "key-two-bbbb")
    seen = []

    async def fake_with(httpx_mod, api_key, question, context):
        seen.append(api_key)
        yield "ok"

    monkeypatch.setattr(rag, "_groq_stream_with", fake_with)

    async def run():
        return [t async for t in rag._groq_stream("q", "ctx")]

    assert asyncio.run(run()) == ["ok"] and seen == ["key-two-bbbb"]
