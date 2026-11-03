"""Latency of a live reply: the warm-up that runs when the backend starts (and that a phone agent may call when a call
starts ringing), the pre-rendered disclosure and filler clips, the phone filler delay and its guards, and tool discovery
that no longer waits for speech recognition. Everything heavy is faked."""
import asyncio
import json
import threading
import time

import numpy as np
import pytest

from tests.test_offline_agents import LOCAL, env, make_agent, upload_txt  # noqa: F401  (env is a fixture)
from tests.test_turn_filler import sentences  # noqa: F401
from tests.test_turn_stream import REPLY, FakeStream, Rig, parse  # noqa: F401


# ------------------------------------------------------------------------------------------------ WARM_ON_START

@pytest.fixture()
def gate(env, monkeypatch):
    """The startup gate with a clean environment (outside pytest's own guard unless a test puts it back)."""
    for name in ("WARM_ON_START", "EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_ACCOUNT_SID", "EXOTEL_CALLER_ID"):
        monkeypatch.delenv(name, raising=False)
    return env


def outside_pytest(monkeypatch):
    """pytest sets PYTEST_CURRENT_TEST again at the start of every phase, so it is removed in the test body."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)


def test_warm_on_start_off_and_on(gate, monkeypatch):
    outside_pytest(monkeypatch)
    monkeypatch.setenv("WARM_ON_START", "off")
    assert gate._warm_on_start() is False
    monkeypatch.setenv("WARM_ON_START", "on")
    assert gate._warm_on_start() is True            # on even with no agent and no phone


def test_warm_on_start_auto_needs_exotel_or_an_agent(gate, monkeypatch):
    outside_pytest(monkeypatch)
    assert gate._warm_on_start() is False           # auto, nothing configured, no agent
    monkeypatch.setenv("EXOTEL_API_KEY", "k")
    monkeypatch.setenv("EXOTEL_API_TOKEN", "t")
    monkeypatch.setenv("EXOTEL_ACCOUNT_SID", "s")
    assert gate._warm_on_start() is False           # one credential missing: not configured
    monkeypatch.setenv("EXOTEL_CALLER_ID", "0800")
    assert gate._warm_on_start() is True
    for name in ("EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_ACCOUNT_SID", "EXOTEL_CALLER_ID"):
        monkeypatch.delenv(name)
    assert gate._warm_on_start() is False
    gate.local_db.insert("agents", {"user_id": "u", "name": "A"})
    assert gate._warm_on_start() is True            # any agent exists


def test_warm_on_start_auto_never_runs_under_pytest(gate, monkeypatch):
    gate.local_db.insert("agents", {"user_id": "u", "name": "A"})
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    assert gate._warm_on_start() is False
    monkeypatch.setenv("WARM_ON_START", "on")
    assert gate._warm_on_start() is True


def test_the_startup_hook_does_not_wait_for_the_warm_up(gate, monkeypatch):
    monkeypatch.setenv("WARM_ON_START", "on")
    done = []

    async def slow_warm(reason="manual", agent_id=None):
        await asyncio.sleep(0.5)
        done.append(reason)
        return {}

    monkeypatch.setattr(gate, "warm_for_live_turns", slow_warm)

    async def main():
        t = time.time()
        gate._start_background_warmup()
        elapsed = time.time() - t
        assert elapsed < 0.05 and not done            # returned at once, the warm-up is still going
        assert len(gate._BACKGROUND_TASKS) == 1
        await asyncio.sleep(0.7)
        assert done == ["startup"] and not gate._BACKGROUND_TASKS

    asyncio.run(main())


def test_a_crashing_warm_up_never_breaks_startup(gate, monkeypatch):
    monkeypatch.setenv("WARM_ON_START", "on")

    async def boom(reason="manual", agent_id=None):
        raise RuntimeError("no")

    monkeypatch.setattr(gate, "warm_for_live_turns", boom)

    async def main():
        gate._start_background_warmup()
        await asyncio.sleep(0.05)

    asyncio.run(main())            # the task's error is only logged, nothing escapes


def test_startup_hook_is_registered_and_the_app_starts_with_warm_up_off(gate, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("WARM_ON_START", "off")
    monkeypatch.setattr(gate, "warm_for_live_turns", lambda *a, **k: pytest.fail("warm-up must be off"))
    with TestClient(gate.app) as c:
        assert c.get("/api/health").status_code in (200, 404)


# ------------------------------------------------------------------------------------------------ warm_for_live_turns

@pytest.fixture()
def steps(monkeypatch):
    """Every warm-up step replaced by a fake that records when it ran."""
    from routers import warmup

    log = []
    fail = set()
    delays = {}

    def fake(name):
        async def run():
            log.append(("start", name, time.time()))
            await asyncio.sleep(delays.get(name, 0))
            if name in fail:
                raise RuntimeError(f"{name} broke")
            log.append(("end", name, time.time()))
            return f"{name} ok"
        return run

    for fn, name in (("_warm_llm_connection", "llm"), ("_warm_asr_connection", "asr"), ("_warm_kokoro_en", "kokoro_en"),
                     ("_warm_kokoro_hi", "kokoro_hi"), ("_warm_clips", "clips"), ("_warm_embedding", "embedding"),
                     ("_warm_watermark", "watermark"), ("_warm_clone", "clone")):
        monkeypatch.setattr(warmup, fn, fake(name))

    async def knowledge(agent_id):
        log.append(("start", "knowledge", agent_id))
        return "1 agent(s)"

    monkeypatch.setattr(warmup, "_warm_knowledge", knowledge)
    monkeypatch.setattr(warmup, "_live_task", None)
    return warmup, log, fail, delays


def test_every_step_runs_once_and_is_reported_with_its_time(env, steps):
    warmup, log, fail, delays = steps
    delays["kokoro_en"] = 0.05
    out = asyncio.run(env.warm_for_live_turns("test"))
    assert list(out) == ["llm", "asr", "kokoro_en", "kokoro_hi", "clips", "embedding", "knowledge", "watermark", "clone"]
    assert all(v["state"] == "ready" for v in out.values())
    assert out["kokoro_en"]["seconds"] >= 0.05 and out["llm"]["detail"] == "llm ok"
    started = [n for k, n, _ in log if k == "start"]
    assert sorted(started) == sorted(out) and len(started) == len(set(started))


def test_the_steps_run_in_parallel_and_the_chains_keep_their_order(env, steps):
    warmup, log, fail, delays = steps
    for name in ("llm", "asr", "embedding", "watermark", "kokoro_en"):
        delays[name] = 0.2
    t = time.time()
    asyncio.run(env.warm_for_live_turns("test"))
    assert time.time() - t < 0.7                                    # five 0.2 s steps in well under 5 x 0.2 s
    times = {(k, n): ts for k, n, ts in log if isinstance(ts, float)}
    assert times[("start", "clips")] >= times[("end", "kokoro_hi")] >= times[("end", "kokoro_en")]
    order = [(k, n) for k, n, _ in log]
    assert order.index(("start", "knowledge")) > order.index(("end", "embedding"))


def test_a_failing_step_is_reported_and_the_others_still_run(env, steps):
    warmup, log, fail, delays = steps
    fail.update({"embedding", "llm"})
    out = asyncio.run(env.warm_for_live_turns("test"))
    assert out["embedding"]["state"] == "failed" and "embedding broke" in out["embedding"]["detail"]
    assert out["llm"]["state"] == "failed"
    assert out["knowledge"]["state"] == "skipped"                   # needs the embedding model
    assert out["kokoro_en"]["state"] == out["watermark"]["state"] == out["clips"]["state"] == "ready"


def test_without_a_kokoro_voice_the_clips_are_skipped(env, steps):
    warmup, log, fail, delays = steps
    fail.update({"kokoro_en", "kokoro_hi"})
    out = asyncio.run(env.warm_for_live_turns("test"))
    assert out["clips"]["state"] == "skipped" and "clips" not in [n for k, n, _ in log if k == "start"]


def test_a_call_while_it_runs_joins_it_and_the_agent_is_passed_on(env, steps):
    warmup, log, fail, delays = steps
    delays["llm"] = 0.2

    async def main():
        a = asyncio.ensure_future(env.warm_for_live_turns("startup", "agent-1"))
        await asyncio.sleep(0.05)
        b = asyncio.ensure_future(env.warm_for_live_turns("ringing", "agent-2"))
        return await asyncio.gather(a, b)

    a, b = asyncio.run(main())
    assert a == b
    assert [n for k, n, _ in log if k == "start" and n == "llm"] == ["llm"]       # one warm-up, not two
    assert ("start", "knowledge", "agent-1") in [(k, n, x) for k, n, x in log]


def test_a_second_call_after_it_finished_warms_again(env, steps):
    warmup, log, fail, delays = steps
    asyncio.run(env.warm_for_live_turns("a"))
    asyncio.run(env.warm_for_live_turns("b"))
    assert len([1 for k, n, _ in log if k == "start" and n == "llm"]) == 2


def test_the_warm_up_loads_nothing_heavy(env, monkeypatch):
    """The real steps with the models faked: only Kokoro is loaded, the embedder is encoded once, the model connection
    is opened with the hosted-provider prewarm."""
    from routers import warmup
    from services import llm, provenance

    loaded, calls = [], []

    async def load_model(key):
        loaded.append(key)
        return object()

    def kokoro(model, text, lang="en"):
        calls.append(("kokoro", lang))
        return np.full(2400, 0.05, dtype=np.float32)

    class Embed:
        loaded = True

        def encode(self, texts, *a, **k):
            calls.append(("embed", len(texts)))
            return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(env.advanced_manager, "load_model", load_model)
    monkeypatch.setattr(env, "_kokoro_generate", kokoro)
    monkeypatch.setattr(env, "kokoro_unavailable_languages", lambda: set())
    monkeypatch.setattr(env, "embedding_model", Embed())
    monkeypatch.setattr(env, "groq_client", None)
    monkeypatch.setitem(env._kokoro_lang_pipelines, "h", object())
    monkeypatch.setattr(warmup, "_has", lambda module: True)
    monkeypatch.setattr(warmup, "_live_task", None)
    monkeypatch.setattr(provenance, "unavailable_reason", lambda: None)
    monkeypatch.setattr(provenance, "watermark", lambda wav, sr, bits: wav)
    monkeypatch.setattr(provenance, "detect", lambda wav, sr: {})
    monkeypatch.setattr(llm, "_prewarm_targets", lambda: ["https://example.test/"])
    monkeypatch.setattr(llm, "prewarm", lambda **kw: calls.append(("llm.prewarm", kw)))
    monkeypatch.setattr(llm, "warm_local", lambda: pytest.fail("the local model must not start"))
    monkeypatch.setattr(warmup, "_warm_asr", lambda: asyncio.sleep(0, "asr"))
    monkeypatch.setenv("WARM_CLONE", "off")        # the clone step has its own tests (test_warm_clone.py)
    out = asyncio.run(env.warm_for_live_turns("test"))
    assert {k: v["state"] for k, v in out.items()} == {"llm": "ready", "asr": "ready", "kokoro_en": "ready",
                                                       "kokoro_hi": "ready", "clips": "ready", "embedding": "ready",
                                                       "knowledge": "skipped", "watermark": "ready", "clone": "skipped"}
    assert set(loaded) == {"kokoro"}
    assert ("llm.prewarm", {"wait": True}) in calls
    assert ("embed", 2) in calls
    assert "clips rendered" in out["clips"]["detail"] and not out["clips"]["detail"].startswith("0 ")


# ------------------------------------------------------------------------------------------------ pre-rendered clips

@pytest.fixture()
def rig(app_client, env, monkeypatch, ref_wav):
    env._FILLER_CLIPS.clear()
    env._FILLER_LAST.clear()
    monkeypatch.setenv("FILLER_AFTER_MS", "150")
    monkeypatch.delenv("FILLER_AUDIO", raising=False)
    monkeypatch.setattr(env, "kokoro_unavailable_languages", lambda: set())
    return Rig(env, monkeypatch, app_client, ref_wav)


def test_prerendered_clips_fill_both_caches_and_are_not_rendered_twice(rig):
    n = asyncio.run(rig.env._prerender_turn_clips())
    assert n == len(rig.spoken) and n > 0
    texts = {t for t, _ in rig.spoken}
    assert {"This is an AI assistant.", "One moment.", "Let me check."} <= texts
    assert any(k[3] == "यह एक AI सहायक है।" for k in rig.env._DISCLOSURE_CLIPS)
    assert ("kokoro", "hi", "एक सेकंड।") in rig.env._FILLER_CLIPS
    again = asyncio.run(rig.env._prerender_turn_clips())
    assert again == 0 and n == len(rig.spoken)


def test_a_turn_after_the_pre_render_does_not_synthesize_the_disclosure_or_the_filler(rig):
    asyncio.run(rig.env._prerender_turn_clips())
    rig.spoken.clear()
    rig.stream = lambda: FakeStream(REPLY, first_delay=0.6)
    a = make_agent(rig.client, disclose_ai=False)
    s = sentences(parse(rig.post(a["id"])))
    assert s[0]["filler"] is True and s[0]["text"] in ("One moment.", "Let me check.")
    assert all(t not in ("One moment.", "Let me check.") for t, _ in rig.spoken)       # played from the cache

    rig.spoken.clear()
    b = make_agent(rig.client)                                                          # discloses
    s = sentences(parse(rig.post(b["id"])))
    assert s[0]["text"] == "This is an AI assistant."
    assert "This is an AI assistant." not in [t for t, _ in rig.spoken]


def test_the_hinglish_and_hindi_disclosures_are_prerendered_under_the_keys_a_turn_uses(rig):
    asyncio.run(rig.env._prerender_turn_clips())
    rig.spoken.clear()
    rig.asr_text = "mera refund kab milega"
    rig.stream = lambda: FakeStream(["Aapka refund ", "tees din mein aayega. "])
    a = make_agent(rig.client)
    s = sentences(parse(rig.post(a["id"])))
    assert s[0]["text"] == "Yeh ek AI assistant hai."
    assert all("AI" not in t for t, _ in rig.spoken)


def test_a_missing_hindi_voice_skips_only_the_hindi_clips(rig, monkeypatch):
    monkeypatch.setattr(rig.env, "kokoro_unavailable_languages", lambda: {"hi"})
    asyncio.run(rig.env._prerender_turn_clips())
    assert {t for t, lang in rig.spoken} == {"This is an AI assistant.", "One moment.", "Let me check."}


def test_indic_speak_for_turns_means_only_english_is_prerendered(rig, monkeypatch):
    monkeypatch.setenv("TURN_INDIC_ENGINE", "indic_speak")
    asyncio.run(rig.env._prerender_turn_clips())
    assert {lang for _, lang in rig.spoken} == {"en"}


# ------------------------------------------------------------------------------------------------ phone filler

def test_phone_calls_get_the_filler_sooner(env, monkeypatch):
    monkeypatch.delenv("FILLER_AFTER_MS", raising=False)
    monkeypatch.delenv("PHONE_FILLER_AFTER_MS", raising=False)
    assert env._filler_after_s() == 0.9 and env._filler_after_s("widget") == 0.9 and env._filler_after_s(None) == 0.9
    assert env._filler_after_s("phone") == 0.5
    monkeypatch.setenv("PHONE_FILLER_AFTER_MS", "350")
    assert env._filler_after_s("phone") == 0.35 and env._filler_after_s("widget") == 0.9
    monkeypatch.setenv("FILLER_AFTER_MS", "1200")
    assert env._filler_after_s("phone") == 0.35 and env._filler_after_s() == 1.2
    monkeypatch.setenv("PHONE_FILLER_AFTER_MS", "soon")
    assert env._filler_after_s("phone") == 0.5                      # a bad value falls back to the default
    monkeypatch.setenv("PHONE_FILLER_AFTER_MS", "-5")
    assert env._filler_after_s("phone") == 0.0


def run_turn(env, agent, *, source, conversation_id=None, text_input=None):
    """The streaming turn as the phone agent drives it (no HTTP): returns [(event, data)]."""
    async def go():
        row = env.local_db.get("agents", agent["id"])
        run = await env._turn_begin(None, row["id"], row["user_id"], storage_mode="local", agent=row, source=source)
        events = []
        async for chunk in env._turn_event_stream(run, b"RIFF", conversation_id, "00000000-0000-0000-0000-000000000001"):
            lines = chunk.decode().strip().split("\n")
            events.append((lines[0][7:], json.loads(lines[1][6:])))
        return events
    return asyncio.run(go())


def test_a_phone_turn_uses_the_phone_delay_and_a_widget_turn_does_not(rig, monkeypatch):
    monkeypatch.setenv("FILLER_AFTER_MS", "5000")           # a normal turn would wait five seconds for its filler
    monkeypatch.setenv("PHONE_FILLER_AFTER_MS", "100")
    rig.stream = lambda: FakeStream(REPLY, first_delay=0.7)
    a = make_agent(rig.client, disclose_ai=False)
    phone = [d for e, d in run_turn(rig.env, a, source="phone") if e == "sentence"]
    assert phone[0].get("filler") is True
    rig.env._FILLER_LAST.clear()
    other = [d for e, d in run_turn(rig.env, a, source="widget") if e == "sentence"]
    assert not any(x.get("filler") for x in other)


def test_no_filler_once_the_model_has_started_answering(rig, monkeypatch):
    """The first words are out before the deadline, the first clause is not yet: its audio is moments away and a
    filler (a second long) would only delay it."""
    class Stalled(FakeStream):
        def __iter__(self):
            yield "Thanks for calling "
            time.sleep(0.5)                                  # the rest of the first clause is slow
            yield "the clinic, we open at nine. "
            self.result = None
    monkeypatch.setenv("FILLER_AFTER_MS", "150")
    rig.stream = lambda: Stalled([])
    a = make_agent(rig.client, disclose_ai=False)
    s = sentences(parse(rig.post(a["id"])))
    assert not any(x.get("filler") for x in s) and s[0]["text"].startswith("Thanks for calling")


def test_a_filler_still_covers_a_model_that_has_not_started(rig):
    rig.stream = lambda: FakeStream(REPLY, first_delay=0.6)
    a = make_agent(rig.client, disclose_ai=False)
    s = sentences(parse(rig.post(a["id"])))
    assert s[0]["filler"] is True and s[1]["text"] == "The clinic opens at 9 a.m. on Monday."


def test_a_filler_is_dropped_when_the_answer_gets_there_while_it_is_prepared(rig, monkeypatch):
    """The filler is prepared before its deadline; the answer's first words arriving in between cancel it."""
    monkeypatch.setenv("FILLER_AFTER_MS", "400")
    rig.stream = lambda: FakeStream(REPLY, first_delay=0.36)     # first words after the preparation started (0.28 s)
    a = make_agent(rig.client, disclose_ai=False)
    s = sentences(parse(rig.post(a["id"])))
    assert not any(x.get("filler") for x in s)
    assert [x["index"] for x in s] == [0, 1]


# ------------------------------------------------------------------------------------------------ tool discovery

def test_tool_discovery_starts_before_speech_recognition_finishes(rig, monkeypatch):
    a = make_agent(rig.client, disclose_ai=False)
    rig.env.local_db.update("agents", a["id"], {"tools": ["fake"]})
    marks = {}

    async def slow_asr(audio_bytes, name, language=None):
        marks["asr_start"] = time.time()
        await asyncio.sleep(0.3)
        marks["asr_end"] = time.time()
        return rig.asr_text

    def specs_for(names):
        marks["specs"] = time.time()
        time.sleep(0.2)
        marks["specs_end"] = time.time()
        return []

    monkeypatch.setattr(rig.env, "_speech_to_text", slow_asr)
    monkeypatch.setattr(rig.env.tools_mod.hub, "specs_for", specs_for)
    t = time.time()
    events = parse(rig.post(a["id"]))
    assert events[-1][0] == "done"
    assert marks["specs"] < marks["asr_end"]                        # started while the recording was recognised
    assert marks["specs"] - marks["asr_start"] < 0.1
    # the 0.2 s of discovery hid inside the 0.3 s of recognition: it added (almost) nothing to the reply
    assert events[-1][1]["timings"].get("tools_ms", 0) < 100


def test_slow_tool_discovery_still_counts_as_tools_time(rig, monkeypatch):
    a = make_agent(rig.client, disclose_ai=False)
    rig.env.local_db.update("agents", a["id"], {"tools": ["fake"]})

    def specs_for(names):
        time.sleep(0.5)
        return []

    monkeypatch.setattr(rig.env.tools_mod.hub, "specs_for", specs_for)
    events = parse(rig.post(a["id"]))
    assert 300 <= events[-1][1]["timings"]["tools_ms"] <= 600


def test_a_failed_recording_cancels_the_tool_discovery_quietly(rig, monkeypatch):
    a = make_agent(rig.client, disclose_ai=False)
    rig.env.local_db.update("agents", a["id"], {"tools": ["fake"]})

    async def silence(audio_bytes, name, language=None):
        return ""

    monkeypatch.setattr(rig.env, "_speech_to_text", silence)
    monkeypatch.setattr(rig.env.tools_mod.hub, "specs_for", lambda names: time.sleep(0.2) or [])
    events = parse(rig.post(a["id"]))
    assert [e for e, _ in events] == ["error"] and events[0][1]["code"] == "NO_SPEECH"


# ------------------------------------------------------------------------------------------------ retrieval table checks

def test_a_clean_knowledge_table_is_not_read_again_on_every_search(env, app_client, monkeypatch):
    a = make_agent(app_client)
    assert upload_txt(app_client, a["id"]).status_code == 200
    reads = []
    orig = env.local_db.select

    def counting(table, *args, **kw):
        reads.append(table)
        return orig(table, *args, **kw)

    env._LOCAL_EMBEDDINGS_CLEAN.clear()
    monkeypatch.setattr(env.local_db, "select", counting)
    for _ in range(3):
        assert env._refresh_local_embeddings(a["id"], 64) == 0
    assert reads == ["documents", "document_chunks"]                 # read once, not three times


def test_a_changed_knowledge_table_is_checked_again(env, app_client, monkeypatch):
    a = make_agent(app_client)
    upload_txt(app_client, a["id"])
    env._LOCAL_EMBEDDINGS_CLEAN.clear()
    assert env._refresh_local_embeddings(a["id"], 64) == 0
    # a chunk written by an older embedding model appears: the next search must find and redo it
    chunk = env.local_db.select("document_chunks")[0]
    env.local_db.update("document_chunks", chunk["id"], {"embed_model": "an-older-model"})
    assert env._refresh_local_embeddings(a["id"], 64) >= 1
    assert env._refresh_local_embeddings(a["id"], 64) == 0


# ------------------------------------------------------------------------------------------------ the real small steps

def test_the_knowledge_step_runs_the_real_search_for_the_agents(env, app_client):
    from routers import warmup
    a = make_agent(app_client)
    upload_txt(app_client, a["id"])
    env._LOCAL_EMBEDDINGS_CLEAN.clear()
    assert asyncio.run(warmup._warm_knowledge(None)) == "1 agent(s)"
    assert a["id"] in env._LOCAL_EMBEDDINGS_CLEAN                      # the table check was done ahead of the first call
    assert asyncio.run(warmup._warm_knowledge(a["id"])) == "1 agent(s)"
    with pytest.raises(warmup.Skipped):
        asyncio.run(warmup._warm_knowledge("someone-else"))


def test_the_knowledge_step_is_skipped_when_there_are_no_agents(env):
    from routers import warmup
    with pytest.raises(warmup.Skipped):
        asyncio.run(warmup._warm_knowledge(None))


def test_the_speech_step_opens_the_groq_connection_without_sending_audio(env, monkeypatch):
    from routers import warmup
    calls = []

    class Models:
        def list(self):
            calls.append("models.list")

    class Groq:
        models = Models()

        class audio:  # noqa: N801
            class transcriptions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    raise AssertionError("no audio may be sent")

    monkeypatch.setattr(env, "groq_client", Groq())
    assert "Groq" in asyncio.run(warmup._warm_asr_connection()) and calls == ["models.list"]


def test_the_model_step_is_skipped_without_a_hosted_model(env, monkeypatch):
    from routers import warmup
    from services import llm
    monkeypatch.setattr(llm, "_prewarm_targets", lambda: [])
    with pytest.raises(warmup.Skipped):
        asyncio.run(warmup._warm_llm_connection())
