"""GET /api/system/resources and POST /api/system/models/{key}/unload: the "Models and memory" indicator.
psutil, vm_stat and sysctl are faked, so the answers do not depend on this machine; no model is ever loaded."""
from core import memory_guard
import time
import types

import pytest

from routers import system_resources as sr
from tests.test_offline_agents import LOCAL

MB = 2 ** 20
GB = 1024 * MB

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12345.
Pages active:                            200000.
Pages occupied by compressor:            100000.
Swapouts:                                 55555.
"""


class FakeProc:
    def __init__(self, pid, rss_mb, children=()):
        self.pid, self._rss, self._children = pid, rss_mb, list(children)

    def memory_info(self):
        return types.SimpleNamespace(rss=int(self._rss * MB))

    def children(self, recursive=False):
        return self._children


class FakePopen:
    def __init__(self, pid, alive=True):
        self.pid, self._alive = pid, alive

    def poll(self):
        return None if self._alive else 0


class Worker:
    """A model that runs as a worker process (like MlxKokoro)."""
    def __init__(self, pid, busy=False):
        self.proc = FakePopen(pid)
        self.busy = busy


class InApp:
    """A model that lives inside the API process (no worker handle)."""
    busy = False


@pytest.fixture()
def rig(monkeypatch):
    import backend
    import psutil

    procs = {100: FakeProc(100, 700), 200: FakeProc(200, 1200, [FakeProc(201, 300), FakeProc(202, 100)]),
             1: FakeProc(1, 2500)}
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: types.SimpleNamespace(total=16 * GB, available=int(5.5 * GB)))
    monkeypatch.setattr(psutil, "swap_memory", lambda: types.SimpleNamespace(total=4 * GB, used=2 * GB))
    monkeypatch.setattr(psutil, "Process", lambda pid=None: procs[pid if pid is not None else 1])
    monkeypatch.setattr("os.getpid", lambda: 1)
    monkeypatch.setattr(sr._platform, "system", lambda: "Darwin")
    calls = []

    def run(cmd):
        calls.append(cmd[0])
        return VM_STAT if cmd[0] == "vm_stat" else "2\n"

    monkeypatch.setattr(sr, "_run", run)
    mgr = backend.advanced_manager
    monkeypatch.setattr(mgr, "loaded_models", {})
    monkeypatch.setattr(mgr, "pinned", {})

    async def boom(key):
        raise AssertionError("reading the indicator must not load a model")

    monkeypatch.setattr(mgr, "load_model", boom)
    # nothing else may show up in the list from services other tests have imported
    from services import answer_audit, llm, mlx_asr, provenance, retrieval, speaker
    monkeypatch.setattr(retrieval, "_embedders", {})
    monkeypatch.setattr(retrieval, "_reranker", None)
    monkeypatch.setattr(retrieval, "_reranker_loading", False)
    monkeypatch.setattr(provenance, "_models", None)
    monkeypatch.setattr(speaker, "_classifier", None)
    monkeypatch.setattr(answer_audit, "_verifier", None)
    monkeypatch.setattr(backend.whisper_model, "_model", None)
    monkeypatch.setattr(mlx_asr._worker, "proc", None)
    monkeypatch.setattr(llm._mlx, "proc", None)
    monkeypatch.setattr(llm, "_bg", None)
    sr.clear_caches()
    yield types.SimpleNamespace(mgr=mgr, calls=calls, procs=procs)
    sr.clear_caches()


def load(rig, key, model, last_used_ago=5.0):
    now = time.time()
    rig.mgr.loaded_models[key] = (model, now - 600, now - last_used_ago)


def get(app_client):
    sr.clear_caches()
    r = app_client.get("/api/system/resources", headers=LOCAL)
    assert r.status_code == 200, r.text
    return r.json()


def test_response_shape(app_client, rig):
    load(rig, "kokoro", Worker(100))
    body = get(app_client)
    assert set(body) == {"time", "system", "app", "models", "guard"}
    assert set(body["system"]) == {"total_mb", "used_mb", "available_mb", "compressed_mb", "swap_used_mb",
                                   "swap_total_mb", "pressure", "platform", "device", "gpu"}
    s = body["system"]
    assert s["total_mb"] == 16384 and s["available_mb"] == 5632 and s["used_mb"] == 16384 - 5632
    assert s["swap_used_mb"] == 2048 and s["swap_total_mb"] == 4096
    assert s["platform"] == "darwin" and s["pressure"] == "warn"
    assert s["compressed_mb"] == round(100000 * 16384 / MB)
    assert set(body["app"]) == {"api_rss_mb", "workers_rss_mb", "total_rss_mb"}
    assert set(body["guard"]) == {"enabled", "reserve_mb", "estimates_mb"}
    assert body["guard"]["estimates_mb"]["kokoro"] == memory_guard.need_mb("kokoro")   # was 600 before the 27 Oct measurements
    (m,) = body["models"]
    assert {"key", "label", "kind", "where", "device", "state", "memory_mb", "memory_source", "idle_s", "pid"} <= set(m)
    assert m["key"] == "kokoro" and m["label"] == "Standard voice (Kokoro)" and m["kind"] == "voice"
    assert m["where"] == "worker process" and m["pid"] == 100


def test_parse_vm_stat_sample():
    assert sr.parse_vm_stat(VM_STAT) == pytest.approx(100000 * 16384 / MB)
    assert sr.parse_vm_stat("Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
                            "Pages occupied by compressor: 256.\n") == pytest.approx(1.0)
    assert sr.parse_vm_stat("nothing useful") is None and sr.parse_vm_stat("") is None


@pytest.mark.parametrize("level,name", [("1", "normal"), ("2", "warn"), ("4", "critical"), ("3", None), ("x", None)])
def test_pressure_mapping(monkeypatch, level, name):
    monkeypatch.setattr(sr._platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sr, "_run", lambda cmd: level + "\n")
    assert sr.memory_pressure() == name


def test_macos_figures_are_null_when_the_commands_fail(app_client, rig, monkeypatch):
    monkeypatch.setattr(sr, "_run", lambda cmd: None)
    s = get(app_client)["system"]
    assert s["compressed_mb"] is None and s["pressure"] is None


def test_other_platforms_have_no_compressed_or_pressure(app_client, rig, monkeypatch):
    monkeypatch.setattr(sr._platform, "system", lambda: "Linux")
    s = get(app_client)["system"]
    assert s["compressed_mb"] is None and s["pressure"] is None and s["platform"] == "linux"


def test_worker_memory_is_the_process_plus_its_children(app_client, rig):
    assert sr.tree_rss_mb(200) == pytest.approx(1600)
    assert sr.tree_rss_mb(None) is None and sr.tree_rss_mb(999) is None
    load(rig, "qwen3_0_6b", Worker(200))
    body = get(app_client)
    (m,) = body["models"]
    assert m["memory_mb"] == 1600 and m["memory_source"] == "measured"
    assert body["app"] == {"api_rss_mb": 2500, "workers_rss_mb": 1600, "total_rss_mb": 4100}


def test_in_app_models_are_estimates(app_client, rig):
    load(rig, "indic_f5", InApp())
    (m,) = get(app_client)["models"]
    assert m["where"] == "in app" and m["memory_source"] == "estimated" and m["pid"] is None
    assert m["memory_mb"] == 3000          # the memory guard's figure for IndicF5
    assert get(app_client)["app"]["workers_rss_mb"] == 0


def test_a_dead_worker_falls_back_to_an_estimate(app_client, rig):
    w = Worker(100)
    w.proc._alive = False
    load(rig, "kokoro", w)
    (m,) = get(app_client)["models"]
    assert m["memory_source"] == "estimated" and m["memory_mb"] == memory_guard.need_mb("kokoro") and m["pid"] is None


def test_states(app_client, rig):
    load(rig, "kokoro", Worker(100), last_used_ago=3)
    load(rig, "qwen3_0_6b", Worker(200, busy=True), last_used_ago=3)
    load(rig, "dhvaani", Worker(100), last_used_ago=200)
    load(rig, "indic_speak", Worker(100), last_used_ago=3)
    rig.mgr.pinned["indic_speak"] = 1
    by = {m["key"]: m for m in get(app_client)["models"]}
    assert by["kokoro"]["state"] == "loaded" and by["kokoro"]["can_unload"]
    assert by["qwen3_0_6b"]["state"] == "busy" and not by["qwen3_0_6b"]["can_unload"]
    assert by["dhvaani"]["state"] == "idle" and by["dhvaani"]["idle_s"] >= 200
    assert by["indic_speak"]["state"] == "pinned (live call)" and not by["indic_speak"]["can_unload"]


def test_unload_an_idle_model(app_client, rig, monkeypatch):
    load(rig, "dhvaani", Worker(100), last_used_ago=300)
    unloaded = []
    monkeypatch.setattr(rig.mgr, "_unload_model", lambda key: (unloaded.append(key), rig.mgr.loaded_models.pop(key)))
    r = app_client.post("/api/system/models/dhvaani/unload", headers=LOCAL)
    assert r.status_code == 200 and r.json() == {"key": "dhvaani", "unloaded": True}
    assert unloaded == ["dhvaani"]


def test_pinned_and_busy_models_refuse_unload(app_client, rig, monkeypatch):
    load(rig, "kokoro", Worker(100))
    load(rig, "qwen3_0_6b", Worker(200, busy=True))
    rig.mgr.pinned["kokoro"] = 2
    monkeypatch.setattr(rig.mgr, "_unload_model", lambda key: pytest.fail("must not unload"))
    r = app_client.post("/api/system/models/kokoro/unload", headers=LOCAL)
    assert r.status_code == 409 and "live call" in r.json()["error"]
    r = app_client.post("/api/system/models/qwen3_0_6b/unload", headers=LOCAL)
    assert r.status_code == 409 and "working on a request" in r.json()["error"]
    assert set(rig.mgr.loaded_models) == {"kokoro", "qwen3_0_6b"}


def test_a_model_without_a_safe_unload_says_so(app_client, rig):
    r = app_client.post("/api/system/models/embedder:bge-m3/unload", headers=LOCAL)
    assert r.status_code == 409 and r.json()["error"] == sr.CANNOT_UNLOAD
    r = app_client.post("/api/system/models/nonsense/unload", headers=LOCAL)
    assert r.status_code == 409


def test_in_process_singletons_are_listed_with_estimates(app_client, rig, monkeypatch):
    from services import retrieval
    monkeypatch.setattr(retrieval, "_embedders", {"BAAI/bge-m3": object()})
    monkeypatch.setattr(retrieval, "_half_models", {"BAAI/bge-m3"})
    monkeypatch.setattr(retrieval, "_reranker_loading", True)
    by = {m["key"]: m for m in get(app_client)["models"]}
    emb = by["embedder:bge-m3"]
    assert emb["kind"] == "search" and emb["where"] == "in app" and emb["memory_source"] == "estimated"
    assert emb["memory_mb"] == 1100 and emb["label"] == "Search model (bge-m3)"
    assert by["reranker"]["state"] == "loading" and by["reranker"]["kind"] == "reranker"


def test_reading_loads_nothing_and_is_cached_for_a_second(app_client, rig):
    sr.clear_caches()
    first = app_client.get("/api/system/resources", headers=LOCAL).json()
    assert rig.calls.count("vm_stat") == 1
    n = len(rig.calls)
    second = app_client.get("/api/system/resources", headers=LOCAL).json()
    assert second == first and len(rig.calls) == n       # whole response from the cache: no command run again
    sr._snapshot_cache = (sr._snapshot_cache[0] - 2.0, sr._snapshot_cache[1])   # 1 s later: rebuilt...
    app_client.get("/api/system/resources", headers=LOCAL)
    assert rig.calls.count("vm_stat") == 1                # ...but vm_stat is cached for 2 s


def test_vm_stat_is_cached_for_two_seconds(rig):
    sr.clear_caches()
    assert sr.compressed_mb(now=100.0) is not None
    assert sr.compressed_mb(now=101.9) is not None and rig.calls.count("vm_stat") == 1
    sr.compressed_mb(now=102.1)
    assert rig.calls.count("vm_stat") == 2
