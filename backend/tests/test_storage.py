"""Supabase storage access with the core.retry helpers, using a mocked client."""
import asyncio
from unittest.mock import MagicMock

import pytest

retry = pytest.importorskip("core.retry")


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda s: None)

    async def _nosleep(s):
        return None

    monkeypatch.setattr(retry.asyncio, "sleep", _nosleep)


def mock_supabase(upload_side_effect):
    sb = MagicMock()
    bucket = sb.storage.from_.return_value
    bucket.upload.side_effect = upload_side_effect
    return sb, bucket


def test_upload_retries_transient_failures_then_succeeds():
    sb, bucket = mock_supabase([ConnectionError("reset"), TimeoutError("slow"), {"path": "u/v.wav"}])
    out = retry.retry_sync(lambda: sb.storage.from_("audio-assets").upload("u/v.wav", b"data"), attempts=3)
    assert out == {"path": "u/v.wav"}
    assert bucket.upload.call_count == 3


def test_upload_gives_up_after_attempts():
    sb, bucket = mock_supabase(ConnectionError("down"))
    with pytest.raises(ConnectionError):
        retry.retry_sync(lambda: bucket.upload("p", b"x"), attempts=3)
    assert bucket.upload.call_count == 3


def test_permanent_errors_not_retried():
    sb, bucket = mock_supabase(Exception("Object not found"))
    with pytest.raises(Exception, match="not found"):
        retry.retry_sync(lambda: bucket.upload("missing", b"x"), attempts=5)
    assert bucket.upload.call_count == 1
    assert retry.is_transient(ConnectionError("reset")) is True
    assert retry.is_transient(Exception("Object not found")) is False


def test_custom_should_retry_predicate():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        retry.retry_sync(fn, attempts=4, should_retry=lambda e: False)
    assert len(calls) == 1


def test_async_retry_runs_sync_callable_in_thread():
    sb, bucket = mock_supabase([OSError("blip"), b"audio-bytes"])
    out = asyncio.run(retry.retry_async(lambda: bucket.upload("p", b"x"), attempts=3))
    assert out == b"audio-bytes"
    assert bucket.upload.call_count == 2


def test_async_retry_coroutine_function():
    state = {"n": 0}

    async def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise ConnectionError("x")
        return "ok"

    assert asyncio.run(retry.retry_async(flaky, attempts=3)) == "ok"
    assert state["n"] == 3
