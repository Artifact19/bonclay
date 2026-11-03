"""services/heartbeat.py: a ping a minute to an uptime checker, gap detection, and the offline refusal."""
import asyncio
import json

import pytest

from services import heartbeat as hb


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("BONCLAY_HEARTBEAT_URL", raising=False)
    monkeypatch.delenv("BONCLAY_HEARTBEAT_S", raising=False)
    monkeypatch.delenv("BONCLAY_OFFLINE", raising=False)


def test_unset_means_disabled_with_a_reason():
    assert hb.disabled_reason() == "BONCLAY_HEARTBEAT_URL is not set"
    h = hb.Heartbeat(sender=lambda u, b: 200)
    assert h.beat() is False and h.sent == 0
    assert h.status()["enabled"] is False and h.status()["reason"]


def test_a_beat_posts_a_small_json_body(monkeypatch):
    monkeypatch.setenv("BONCLAY_HEARTBEAT_URL", "https://hc-ping.example/abc")
    seen = {}

    def sender(url, body):
        seen["url"], seen["body"] = url, json.loads(body)
        return 200

    clock = [1000.0]
    h = hb.Heartbeat(sender=sender, active_calls=lambda: 1, version="2.0.0", clock=lambda: clock[0])
    h.started_at = 940.0
    assert h.beat() is True
    assert seen["url"] == "https://hc-ping.example/abc"
    assert set(seen["body"]) == {"sent_at", "uptime_s", "active_calls", "version"}
    assert seen["body"]["uptime_s"] == 60 and seen["body"]["active_calls"] == 1 and seen["body"]["version"] == "2.0.0"
    s = h.status()
    assert s["enabled"] is True and s["sent"] == 1 and s["last_status"] == 200 and s["target"] == "hc-ping.example"


def test_failures_are_counted_and_never_raise(monkeypatch):
    monkeypatch.setenv("BONCLAY_HEARTBEAT_URL", "https://hc-ping.example/abc")

    def sender(url, body):
        raise OSError("network down")

    h = hb.Heartbeat(sender=sender)
    assert h.beat() is False and h.failures == 1 and "network down" in h.last_error


def test_offline_mode_refuses_the_heartbeat(monkeypatch):
    monkeypatch.setenv("BONCLAY_HEARTBEAT_URL", "https://hc-ping.example/abc")
    monkeypatch.setenv("BONCLAY_OFFLINE", "1")
    assert "offline" in hb.disabled_reason()
    calls = []
    h = hb.Heartbeat(sender=lambda u, b: calls.append(u) or 200)
    assert h.beat() is False and calls == []


def test_interval_has_a_floor_and_a_default(monkeypatch):
    assert hb.interval_s() == 60.0
    monkeypatch.setenv("BONCLAY_HEARTBEAT_S", "3")
    assert hb.interval_s() == hb.MIN_INTERVAL_S
    monkeypatch.setenv("BONCLAY_HEARTBEAT_S", "nonsense")
    assert hb.interval_s() == 60.0


def test_a_late_tick_is_recorded_as_a_gap(monkeypatch):
    monkeypatch.setenv("BONCLAY_HEARTBEAT_S", "60")
    clock = [0.0]
    h = hb.Heartbeat(sender=lambda u, b: 200, clock=lambda: clock[0])
    assert h.note_tick() is None
    clock[0] = 61.0
    assert h.note_tick() is None            # on time
    clock[0] = 61.0 + 540.0                 # the laptop slept for nine minutes
    assert h.note_tick() == 540.0
    assert h.status()["last_gap_s"] == 540.0 and h.status()["gaps"] == 1


@pytest.mark.asyncio
async def test_the_loop_starts_beats_and_stops(monkeypatch):
    monkeypatch.setenv("BONCLAY_HEARTBEAT_URL", "http://127.0.0.1:1/ping")
    monkeypatch.setenv("BONCLAY_HEARTBEAT_S", "10")
    beats = []
    h = hb.Heartbeat(sender=lambda u, b: beats.append(u) or 200)
    assert h.start() is None
    for _ in range(50):
        if beats:
            break
        await asyncio.sleep(0.02)
    assert beats and h.status()["running"] is True
    await h.stop()
    assert h.status()["running"] is False


def test_start_without_a_url_reports_the_reason():
    async def go():
        h = hb.Heartbeat(sender=lambda u, b: 200)
        return h.start(), h.status()["running"]
    reason, running = asyncio.run(go())
    assert reason and running is False
