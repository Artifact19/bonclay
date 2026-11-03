"""Nothing the API keeps per turn or per call may grow without bound.

bench/leak_hunt.py ran the real app (language model, speech recognition and Kokoro faked) for 12 minutes and 2,200 requests:
the Python heap did not grow (tracemalloc: about 1.4 MB over 650 requests, all of it stored conversation rows), so what is
checked here is that a run of fake turns keeps no audio, text or conversation state per turn, and that the caches keyed
by something a user can vary have a cap."""
import gc
import threading
import tracemalloc

from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import rig  # noqa: F401  (rig is a fixture)

QUESTIONS = ["When does the clinic open on {d}?", "How much is a consultation, {n} people?", "Is there parking for {n} cars?"]


def _run_turns(rig, agent_id, start, count):
    for i in range(start, start + count):
        rig.asr_text = QUESTIONS[i % len(QUESTIONS)].format(d=("Monday", "Tuesday", "Friday")[i % 3], n=i)
        if i % 2:
            assert rig.post(agent_id).status_code == 200
        else:
            r = rig.client.post(f"/api/conversations/{agent_id}/turn", headers=LOCAL,
                                files={"audio": ("rec.wav", rig.wav, "audio/wav")})
            assert r.status_code == 200, r.text


def test_a_hundred_turns_keep_no_audio_or_text_per_turn(rig):
    from services import caller_memory
    a = make_agent(rig.client, disclose_ai=False)
    _run_turns(rig, a["id"], 0, 30)                       # warm-up: caches fill, models load, threads start
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.take_snapshot()
        threads_before = threading.active_count()
        _run_turns(rig, a["id"], 30, 100)
        gc.collect()
        grown = sum(s.size_diff for s in tracemalloc.take_snapshot().compare_to(before, "filename") if s.size_diff > 0)
        threads_after = threading.active_count()
    finally:
        tracemalloc.stop()
    # a retained 2400-sample clip or WAV per turn would be 10-50 KB x 100 turns = 1-5 MB; the stored conversation rows
    # (a few hundred bytes a turn) are the only legitimate growth
    assert grown < 1.5 * 2 ** 20, f"the Python heap grew {grown / 2 ** 20:.1f} MB over 100 turns"
    assert threads_after - threads_before <= 4, "threads pile up per turn"
    assert len(caller_memory._CACHE) <= caller_memory._CACHE_MAX


def test_the_clip_caches_and_turn_owner_maps_have_caps(rig):
    import backend
    assert backend._CLIP_CACHE_MAX == 64
    for i in range(1000):
        backend._remember_stream_turn(f"t{i}", "u")
    assert len(backend._STREAM_TURN_OWNERS) <= 512


def test_the_suggested_questions_cache_is_capped():
    from routers import templates
    templates._questions_cache.clear()
    top = templates.QUESTIONS_CACHE_MAX * 3
    for i in range(top):
        templates._cache_questions((f"agent{i}", ("doc",)), ["q"])
    assert len(templates._questions_cache) == templates.QUESTIONS_CACHE_MAX
    assert (f"agent{top - 1}", ("doc",)) in templates._questions_cache      # newest kept
    assert ("agent0", ("doc",)) not in templates._questions_cache
    templates._questions_cache.clear()


def test_caller_memory_forgets_the_oldest_conversation_first():
    from services import caller_memory
    caller_memory.forget()
    for i in range(caller_memory._CACHE_MAX + 50):
        caller_memory.remember(f"c{i}", {"name": "Asha"})
    assert len(caller_memory._CACHE) == caller_memory._CACHE_MAX
    assert caller_memory.cached("c0") is None
    assert caller_memory.cached(f"c{caller_memory._CACHE_MAX + 49}") is not None
    caller_memory.forget()
