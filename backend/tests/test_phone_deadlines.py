"""In-call deadlines: a phone turn caps the wait for the model's next bytes (PHONE_LLM_STALL_S) and the wait for the
local model's slot (PHONE_LOCAL_QUEUE_S, through llm.local_queue_timeout), so a stall fails over instead of going silent."""
import threading
import time

import pytest

import backend
from services import llm


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("PHONE_LLM_STALL_S", "PHONE_LOCAL_LLM_TIMEOUT_S", "PHONE_LOCAL_QUEUE_S", "LOCAL_LLM_QUEUE_S"):
        monkeypatch.delenv(name, raising=False)


def test_phone_stall_timeout_defaults(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert backend._phone_stall_timeout() == 4.0
    monkeypatch.setenv("PHONE_LLM_STALL_S", "2.5")
    assert backend._phone_stall_timeout() == 2.5
    monkeypatch.delenv("GROQ_API_KEY")
    assert backend._phone_stall_timeout() == 20.0            # the local model may be loading
    monkeypatch.setenv("PHONE_LOCAL_LLM_TIMEOUT_S", "nonsense")
    assert backend._phone_stall_timeout() == 20.0
    assert backend._phone_local_queue_s() == 5.0
    monkeypatch.setenv("PHONE_LOCAL_QUEUE_S", "0.1")
    assert backend._phone_local_queue_s() == 0.5             # never below half a second


def test_queue_timeout_is_per_thread_and_defaults_to_the_env(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_QUEUE_S", "7")
    assert llm.queue_timeout_s() == 7.0
    seen = {}

    def other_thread():
        seen["other"] = llm.queue_timeout_s()

    with llm.local_queue_timeout(0.25):
        assert llm.queue_timeout_s() == 0.25
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
    assert seen["other"] == 7.0                              # the profile never leaks to another thread
    assert llm.queue_timeout_s() == 7.0                      # and is restored afterwards


def test_a_busy_local_model_fails_fast_on_the_phone_profile(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_QUEUE_S", "30")
    acquired = llm._mlx.request_lock.acquire(timeout=1.0)   # another request holds the local model
    assert acquired
    try:
        started = time.monotonic()
        with llm.local_queue_timeout(0.2):
            with pytest.raises(TimeoutError):
                with llm._local_slot():
                    pass
        assert time.monotonic() - started < 2.0              # not the 30 s default
    finally:
        llm._mlx.request_lock.release()
    with llm._local_slot():                                  # free again: the slot is taken and given back
        pass
