#!/usr/bin/env python3
"""Memory each BonClay engine really uses, measured the way the app starts it.

    HF_HUB_OFFLINE=1 backend/.venv/bin/python backend/bench/model_memory.py                 # every engine installed here
    HF_HUB_OFFLINE=1 backend/.venv/bin/python backend/bench/model_memory.py --engines kokoro,asr_large_only
    backend/.venv/bin/python backend/bench/model_memory.py --list                           # what is installed, what is not
    backend/.venv/bin/python backend/bench/model_memory.py --markdown backend/bench/results/model_memory-XXXX.json

For each engine: start it with the app's own loader or worker class, run one tiny real inference, then record
  * worker engines (a separate process): the resident set (RSS) and the macOS physical footprint of the worker process
    and its children, after the inference has settled and at its peak (sampled every 0.25 s from before the start);
  * in-process engines (embedder, reranker, AudioSeal, speaker model, answer audit): the same for a fresh Python process
    that imports only the app's service module, as a delta against the process after its imports, and after unloading;
  * the whole machine before the start, with the engine running, and after it is stopped: psutil "available", swap used,
    compressed memory (vm_stat "Pages occupied by compressor"), and on macOS the memory-pressure level.
The machine is not quiet (other work runs on it), so the system columns move by a few hundred MB on their own: read the
process figures (footprint above all) as the engine's size, and the system figures as context. Footprint is what Activity
Monitor calls Memory (dirty + compressed + swapped pages, GPU buffers included); RSS counts only pages in RAM at that moment, so
under pressure it can read low. Engines are run one at a time and stopped (with a wait) between runs. The memory guard is
switched off for the run (it is the thing being calibrated).

The results go to backend/bench/results/model_memory-<stamp>.json (git-ignored). `--markdown FILE` prints the table.
No model is downloaded (HF_HUB_OFFLINE is forced on) and nothing under backend/data is read.
"""
import argparse
import datetime as dt
import gc
import io
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MEMORY_GUARD"] = "off"
os.environ["LLM_PROVIDERS"] = "mlx"          # never a hosted model: the local server is what is measured

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
REPO = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
RESULTS_DIR = Path(HERE) / "results"
MB = 2 ** 20

import numpy as np  # noqa: E402
import psutil  # noqa: E402

EN_TEXT = "Good morning, you have reached the front desk. How can I help you today?"
HI_TEXT = "नमस्ते, आप रिसेप्शन पर पहुंचे हैं, कृपया अपना नाम और मिलने का दिन बताइए।"
REF_EN = ("a", "am_michael", "Good morning, you have reached the front desk, please tell me your name and the day you would like to visit.")
REF_HI = ("h", "hf_alpha", "नमस्ते, आप रिसेप्शन पर पहुंचे हैं, कृपया अपना नाम और मिलने का दिन बताइए।")


# ------------------------------------------------------------------------------------------------ measuring

def footprint_mb(pid):
    """macOS physical footprint of a process in MB (what Activity Monitor calls Memory); None elsewhere."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("proc"))
        names = ("ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins ri_wired_size "
                 "ri_resident_size ri_phys_footprint").split()

        class Rusage(ctypes.Structure):
            _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in names] + \
                       [(f"pad{i}", ctypes.c_uint64) for i in range(40)]
        ru = Rusage()
        if lib.proc_pid_rusage(int(pid), 4, ctypes.byref(ru)) != 0:
            return None
        return ru.ri_phys_footprint / MB
    except Exception:  # noqa: BLE001
        return None


def tree_memory(pid):
    """{"rss_mb", "footprint_mb", "pids"} summed over a process and all its descendants (None when it is gone)."""
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.Error:
        return None
    rss = fp = 0.0
    pids = []
    for p in procs:
        try:
            rss += p.memory_info().rss / MB
            f = footprint_mb(p.pid)
            fp += f if f is not None else 0.0
            pids.append(p.pid)
        except psutil.Error:
            pass
    return {"rss_mb": rss, "footprint_mb": fp if sys.platform == "darwin" else None, "pids": pids}


class PeakSampler(threading.Thread):
    """Samples a process tree until stopped; keeps the maximum RSS and footprint (the process may not exist yet)."""

    def __init__(self, get_pid, interval=0.25):
        super().__init__(daemon=True)
        self.get_pid, self.interval = get_pid, interval
        self.peak_rss = self.peak_fp = 0.0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            pid = self.get_pid()
            m = tree_memory(pid) if pid else None
            if m:
                self.peak_rss = max(self.peak_rss, m["rss_mb"])
                self.peak_fp = max(self.peak_fp, m["footprint_mb"] or 0.0)
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)


def vm_stat():
    """Pages from `vm_stat` in MB (macOS), {} elsewhere."""
    if sys.platform != "darwin":
        return {}
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = int(re.search(r"page size of (\d+) bytes", out).group(1))
        pages = {}
        for line in out.splitlines()[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip().rstrip(".")
                if v.isdigit():
                    pages[k.strip()] = int(v)
        mb = lambda key: round(pages.get(key, 0) * page / MB)  # noqa: E731
        return {"compressor_occupied_mb": mb("Pages occupied by compressor"),
                "compressor_stored_mb": mb("Pages stored in compressor"), "free_mb": mb("Pages free"),
                "inactive_mb": mb("Pages inactive"), "wired_mb": mb("Pages wired down"),
                "purgeable_mb": mb("Pages purgeable")}
    except Exception:  # noqa: BLE001
        return {}


def system_snapshot():
    from core import memory_guard
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    snap = {"available_mb": round(vm.available / MB), "used_mb": round((vm.total - vm.available) / MB),
            "swap_used_mb": round(sw.used / MB), "pressure": memory_guard.pressure_name(),
            "pressure_level": memory_guard.pressure_level(), "load_avg_1m": round(os.getloadavg()[0], 2)}
    snap.update(vm_stat())
    return snap


def machine_info():
    top = []
    for p in sorted(psutil.process_iter(["name", "memory_info"]), key=lambda p: -(p.info["memory_info"].rss if p.info["memory_info"] else 0))[:6]:
        top.append({"name": p.info["name"], "rss_mb": round(p.info["memory_info"].rss / MB)})
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"date": dt.datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "total_ram_mb": round(psutil.virtual_memory().total / MB),
            "commit": commit, "python": sys.version.split()[0], "top_processes_before": top,
            "system_before": system_snapshot()}


def wait_gone(pids, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(psutil.pid_exists(p) for p in pids):
            return True
        time.sleep(0.25)
    return False


# ------------------------------------------------------------------------------------------------ audio helpers

_REFS = {}


def synth_refs():
    """The two reference voices (English, Hindi) spoken by the app's own MLX Kokoro worker: (audio, text, 24000)."""
    if _REFS:
        return _REFS
    from services import mlx_kokoro
    h = mlx_kokoro.MlxKokoro()
    try:
        for key, (code, voice, text) in (("en", REF_EN), ("hi", REF_HI)):
            _REFS[key] = (h.synthesize(text, voice, code), text, 24000)
    finally:
        h.close()
    return _REFS


def speech_wav_16k(path):
    import soundfile as sf
    audio, text, sr = synth_refs()["en"]
    n = int(len(audio) * 16000 / sr)
    x = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    sf.write(path, x, 16000)
    return path


# ------------------------------------------------------------------------------------------------ worker engines
# Each worker engine: available() -> reason or None; start() -> handle; infer(handle) -> detail; pid(handle); stop(handle).

class Worker:
    name = ""
    label = ""
    guard_key = None
    device = ""

    def available(self):
        return None

    def start(self):
        raise NotImplementedError

    def infer(self, h):
        return ""

    def pid(self, h):
        raise NotImplementedError

    def stop(self, h):
        h.close()


class KokoroMlx(Worker):
    name, label, guard_key, device = "kokoro", "Kokoro 82M (MLX worker; English and Hindi)", "kokoro", "Apple GPU (MLX)"

    def available(self):
        from services import mlx_kokoro
        return mlx_kokoro.unavailable_reason()

    def start(self):
        from services import mlx_kokoro
        return mlx_kokoro.MlxKokoro()

    def infer(self, h):
        a = h.synthesize(EN_TEXT, "af_heart", "a")
        b = h.synthesize(HI_TEXT, "hf_alpha", "h")
        return f"English {len(a) / 24000:.1f} s + Hindi {len(b) / 24000:.1f} s of audio"

    def pid(self, h):
        return h.proc.pid


class AsrWorker(Worker):
    """The MLX Whisper worker started the way services/mlx_asr.py starts it, with a strategy / residency setting."""
    device = "Apple GPU (MLX)"

    def __init__(self, name, label, env):
        self.name, self.label, self.env = name, label, env

    def available(self):
        from services import mlx_asr
        return mlx_asr.unavailable_reason()

    def start(self):
        from services import mlx_asr
        for k in ("MLX_ASR_RESIDENT", "MLX_ASR_STRATEGY", "ASR_STRATEGY"):
            os.environ.pop(k, None)
        os.environ.update(self.env)
        mlx_asr._worker.stop()
        mlx_asr._worker.ensure()
        return mlx_asr

    def infer(self, h):
        path = os.path.join(tempfile_dir(), "asr_probe.wav")
        speech_wav_16k(path)
        out = h.transcribe(path, "en")
        return f"model {out.get('model')}, {out.get('seconds')} s: {out.get('text', '')[:40]!r}"

    def pid(self, h):
        return h._worker.proc.pid if h._worker.proc else None

    def stop(self, h):
        h._worker.stop()
        for k in ("MLX_ASR_RESIDENT", "MLX_ASR_STRATEGY"):
            os.environ.pop(k, None)


class CloneWorker(Worker):
    """A voice-cloning worker class of the app (create a prompt from a reference voice, generate one sentence)."""

    def __init__(self, name, label, guard_key, device, lang, text):
        self.name, self.label, self.guard_key, self.device, self.lang, self.text = name, label, guard_key, device, lang, text

    def pid(self, h):
        return h.proc.pid

    def infer(self, h):
        audio, ref_text, sr = synth_refs()["hi" if self.lang == "hi" else "en"]
        try:
            prompt = h.create_voice_clone_prompt([(audio, sr)], ref_text=ref_text)
        except TypeError:
            prompt = h.create_voice_clone_prompt([(audio, sr)])
        wavs, out_sr = h.generate_voice_clone(self.text, self.lang, prompt)
        return f"{len(wavs[0]) / out_sr:.1f} s of cloned speech"


class Qwen3Clone(CloneWorker):
    def available(self):
        from services.mlx_clone import mlx_python
        return None if mlx_python() else "no MLX environment (backend/.venv-mlx) or not an Apple Silicon Mac"

    def start(self):
        from services.mlx_clone import MlxQwen3Clone, mlx_python
        return MlxQwen3Clone(mlx_python())


class DhVaani(CloneWorker):
    def available(self):
        from services import dhvaani_clone as dh
        return dh.unavailable_reason()

    def start(self):
        from services import dhvaani_clone as dh
        return dh.DhVaaniClone(dh.dhvaani_python())


class Chatterbox(CloneWorker):
    def available(self):
        from services import chatterbox_clone as cb
        return cb.unavailable_reason()

    def start(self):
        from services import chatterbox_clone as cb
        return cb.ChatterboxClone(cb.chatterbox_python())


class IndicSpeak(Worker):
    name, label, guard_key, device = "indic_speak", "Indic-Speak 3.8B (full-precision worker, Hindi preset voice)", "indic_speak", None

    def available(self):
        from services import indic_speak
        if os.getenv("INDIC_SPEAK_MLX_MODEL", "").strip():
            return "INDIC_SPEAK_MLX_MODEL is set: this run measures the default full-precision engine; unset it"
        return indic_speak.indic_speak_unavailable_reason()

    def start(self):
        from services import indic_speak
        self.device = "Apple GPU / CPU (PyTorch worker)"
        return indic_speak.IndicSpeakModel(indic_speak.indic_speak_python())

    def infer(self, h):
        audio, sr, truncated = h.generate("नमस्ते, आप कैसे हैं?", "Kavya")
        return f"{len(audio) / sr:.1f} s of speech"

    def pid(self, h):
        return h.proc.pid


class LocalLlm(Worker):
    name, label, guard_key, device = "local_llm", "Local language model server (live replies, mlx_lm)", None, "Apple GPU (MLX)"

    def available(self):
        from services import llm
        return llm._mlx.unavailable_reason()

    def start(self):
        from services import llm
        llm._mlx.ensure()
        return llm

    def infer(self, h):
        res = h.chat([{"role": "user", "content": "Reply with the single word OK."}], temperature=0.0, max_tokens=4,
                     timeout=120.0)
        return f"{res.provider} {res.model}: {res.text[:20]!r}"

    def pid(self, h):
        return h._mlx.proc.pid if h._mlx.proc else None

    def stop(self, h):
        h._mlx.stop()


class BackgroundLlm(Worker):
    name, label, guard_key, device = "background_llm", "Background language model (Gemma 4 E4B 4-bit, mlx_lm)", "background_llm", "Apple GPU (MLX)"

    def available(self):
        from services import background_llm
        return background_llm.unavailable_reason()

    def start(self):
        from services import background_llm, llm
        llm.background_chat([{"role": "user", "content": "Say OK."}], model=background_llm.model_name(), max_tokens=2)
        return llm

    def infer(self, h):
        from services import background_llm
        res = h.background_chat([{"role": "user", "content": "Reply with the single word OK."}],
                                model=background_llm.model_name(), temperature=0.0, max_tokens=4, timeout=120.0)
        return f"{res.model}: {res.text[:20]!r}"

    def pid(self, h):
        return h._bg.proc.pid if h._bg is not None and h._bg.proc else None

    def stop(self, h):
        h.unload_background()


_TMP = []


def tempfile_dir():
    if not _TMP:
        import tempfile
        _TMP.append(tempfile.mkdtemp(prefix="bonclay_memory_bench_"))
    return _TMP[0]


WORKERS = [
    KokoroMlx(),
    AsrWorker("asr_large_only", "Speech recognition, Whisper large-v3-turbo only (new default: strategy single)",
              {"MLX_ASR_STRATEGY": "single"}),
    AsrWorker("asr_small_plus_large", "Speech recognition, Whisper small + large-v3-turbo (before 27 Oct 2026)",
              {"MLX_ASR_STRATEGY": "single", "MLX_ASR_RESIDENT": "both"}),
    Qwen3Clone("qwen3_0_6b", "Qwen3-TTS 0.6B voice clone (MLX worker)", "qwen3_0_6b", "Apple GPU (MLX)", "en", EN_TEXT),
    DhVaani("dhvaani", "DhVaani 0.5 Indian-language clone worker", "dhvaani", "Apple GPU / CPU (PyTorch worker)", "hi", HI_TEXT),
    Chatterbox("chatterbox_ml", "Chatterbox Multilingual Hindi clone worker", "chatterbox_ml", "CPU (PyTorch worker)", "hi", HI_TEXT),
    IndicSpeak(),
    LocalLlm(),
    BackgroundLlm(),
]


def run_worker(w):
    reason = w.available()
    if reason:
        return {"status": "skipped", "reason": reason}
    before = system_snapshot()
    holder = {"pid": None}
    sampler = PeakSampler(lambda: holder["pid"])
    sampler.start()
    t0 = time.time()
    h = None
    try:
        if w.name in ("qwen3_0_6b", "dhvaani", "chatterbox_ml"):
            synth_refs()        # the reference voices are made first (their own worker is stopped again) so they are not counted
            before = system_snapshot()
        # start() blocks until the worker answers /health, so the process id is read while it starts
        starter = {}

        def go():
            try:
                starter["h"] = w.start()
            except Exception as e:  # noqa: BLE001
                starter["err"] = e

        th = threading.Thread(target=go, daemon=True)
        th.start()
        while th.is_alive():
            holder["pid"] = holder["pid"] or _find_new_worker_pid()
            time.sleep(0.2)
        if "err" in starter:
            raise starter["err"]
        h = starter["h"]
        start_s = time.time() - t0
        holder["pid"] = w.pid(h)
        t1 = time.time()
        detail = w.infer(h)
        infer_s = time.time() - t1
        time.sleep(4)  # let the allocator settle before the steady reading
        steady_sys = system_snapshot()
        m = tree_memory(w.pid(h))
        res = {"status": "measured", "label": w.label, "device": w.device, "guard_key": w.guard_key,
               "start_s": round(start_s, 1), "infer_s": round(infer_s, 1), "inference": detail,
               "rss_mb": round(m["rss_mb"]), "footprint_mb": None if m["footprint_mb"] is None else round(m["footprint_mb"]),
               "processes": len(m["pids"]), "system_before": before, "system_running": steady_sys}
        pids = m["pids"]
        sampler.stop()
        res["peak_rss_mb"], res["peak_footprint_mb"] = round(sampler.peak_rss), round(sampler.peak_fp)
        w.stop(h)
        h = None
        res["stopped_cleanly"] = wait_gone(pids)
        time.sleep(3)
        res["system_after_stop"] = system_snapshot()
        return res
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "reason": f"{type(e).__name__}: {e}"[:300]}
    finally:
        sampler.stop()
        if h is not None:
            try:
                w.stop(h)
            except Exception:  # noqa: BLE001
                pass


def _find_new_worker_pid():
    """While a worker starts, the newest direct child of this process that runs a python script is the worker."""
    try:
        kids = psutil.Process().children()
    except psutil.Error:
        return None
    kids = [k for k in kids if k.is_running()]
    return max(kids, key=lambda k: k.create_time()).pid if kids else None


# ------------------------------------------------------------------------------------------------ in-process engines
# Run in a fresh child process (`--child NAME`) so the figures are against a clean baseline.

def child_embedder():
    from services import retrieval
    name = retrieval.embedding_model_name()
    try:     # the hub's newest snapshot of a model can hold only the weights; then use the complete one, same files
        import glob
        from huggingface_hub import constants
        complete = [d for d in glob.glob(os.path.join(constants.HF_HUB_CACHE, "models--" + name.replace("/", "--"), "snapshots", "*"))
                    if os.path.exists(os.path.join(d, "tokenizer.json"))]
        if complete and not os.path.exists(os.path.join(retrieval.fp16_cache_dir(name), ".complete")):
            name = complete[0]
    except Exception:  # noqa: BLE001
        pass
    texts = ["What is the refund window?", "रिफंड कितने दिन में मिलता है"]
    return (lambda: retrieval.encode_texts(name, texts), lambda: retrieval._embedders.clear(),
            f"{name} ({retrieval.embedding_dtype()}, {retrieval.embedding_device()})")


def child_reranker():
    from services import retrieval
    name = retrieval.reranker_model_name()
    reason = None if retrieval._reranker_on_disk(name) else f"{name} is not in the local cache"
    if reason:
        raise Skip(reason)

    def unload():
        retrieval._reranker = None
    return (lambda: retrieval.rerank_scores("opening hours", ["We open at nine.", "Parking is free."]), unload,
            f"{name} ({retrieval.reranker_device()}, {retrieval.reranker_dtype()})")


def child_audioseal():
    from services import provenance
    reason = provenance.unavailable_reason()
    if reason:
        raise Skip(reason)

    def infer():
        sr = 24000
        t = np.arange(sr * 2, dtype=np.float32) / sr
        clip = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        provenance.detect(provenance.watermark(clip, sr, 0x5A5A), sr)

    def unload():
        provenance._models = None
    return infer, unload, "AudioSeal 16-bit generator + detector (CPU)"


def child_speaker():
    from services import speaker
    if not speaker.available():
        raise Skip(getattr(speaker, "_failed", None) or "speaker model unavailable")

    def infer():
        t = np.arange(speaker.SAMPLE_RATE * 2, dtype=np.float32) / speaker.SAMPLE_RATE
        speaker.embed((0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))

    def unload():
        speaker._classifier = None
    return infer, unload, "ECAPA-TDNN speaker model (CPU)"


def child_audit():
    from services import answer_audit
    if not answer_audit.model_cached():
        raise Skip("the answer-audit model is not in the local cache")
    v = answer_audit.HHEMVerifier(idle_s=3600)

    def infer():
        v.score([("The clinic opens at nine in the morning.", "The clinic opens at 9 am.")])
    return infer, v.unload, f"{answer_audit.model_spec()['repo']} (CPU)"


def child_api_baseline():
    import backend  # noqa: F401  the whole API module with its routers, nothing loaded on top
    return (lambda: None), (lambda: None), "import backend (the API process with no model loaded)"


class Skip(Exception):
    pass


INPROC = {
    "api_baseline": ("API process itself, nothing loaded (import backend)", child_api_baseline),
    "embedder_bge_m3": ("BGE-M3 embeddings (knowledge search), in the API process", child_embedder),
    "reranker": ("Reranker (RERANKER=on or auto-loaded), in the API process", child_reranker),
    "audioseal": ("AudioSeal watermark generator + detector, in the API process", child_audioseal),
    "ecapa_speaker": ("ECAPA speaker check, in the API process", child_speaker),
    "answer_audit": ("MiniCheck answer audit model, in the API process", child_audit),
}


def run_child(name):
    """Runs in the child process: baseline, load + one inference, steady, after unloading. Prints one JSON line."""
    pid = os.getpid()
    out = {"status": "measured", "label": INPROC[name][0]}
    try:
        if name == "api_baseline":
            before_import = tree_memory(pid)
        infer, unload, detail = INPROC[name][1]()
        gc.collect()
        time.sleep(1)
        base = tree_memory(pid)
        sampler = PeakSampler(lambda: pid)
        sampler.start()
        t = time.time()
        infer()
        out["load_and_infer_s"] = round(time.time() - t, 1)
        infer()   # a warm call
        time.sleep(4)
        steady = tree_memory(pid)
        sampler.stop()
        out.update({"inference": detail, "baseline_rss_mb": round(base["rss_mb"]),
                    "baseline_footprint_mb": None if base["footprint_mb"] is None else round(base["footprint_mb"]),
                    "rss_delta_mb": round(steady["rss_mb"] - base["rss_mb"]),
                    "footprint_delta_mb": None if steady["footprint_mb"] is None else round(steady["footprint_mb"] - base["footprint_mb"]),
                    "rss_mb": round(steady["rss_mb"]),
                    "footprint_mb": None if steady["footprint_mb"] is None else round(steady["footprint_mb"]),
                    "peak_rss_mb": round(sampler.peak_rss), "peak_footprint_mb": round(sampler.peak_fp)})
        if name == "api_baseline":
            out["rss_delta_mb"], out["footprint_delta_mb"] = out["rss_mb"], out["footprint_mb"]
            out["before_import_rss_mb"] = round(before_import["rss_mb"])
        unload()
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
        after = tree_memory(pid)
        out["rss_after_unload_mb"] = round(after["rss_mb"] - base["rss_mb"])
        out["footprint_after_unload_mb"] = None if after["footprint_mb"] is None else round(after["footprint_mb"] - base["footprint_mb"])
    except Skip as e:
        out = {"status": "skipped", "reason": str(e)}
    except Exception as e:  # noqa: BLE001
        out = {"status": "failed", "reason": f"{type(e).__name__}: {e}"[:300]}
    print("RESULT_JSON " + json.dumps(out), flush=True)


def run_inproc(name):
    before = system_snapshot()
    env = {**os.environ, "EMBEDDING_FP16_DIR": os.path.join(tempfile_dir(), "fp16")}
    for attempt in range(2 if name == "embedder_bge_m3" else 1):
        # the embedder makes its float16 copy on first use (a one-off, as in the app): measure the second start
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", name], capture_output=True, text=True,
                           env=env, timeout=1800)
    line = next((ln for ln in p.stdout.splitlines()[::-1] if ln.startswith("RESULT_JSON ")), None)
    if line is None:
        return {"status": "failed", "reason": ("child gave no result: " + p.stderr[-300:]).strip()}
    res = json.loads(line[len("RESULT_JSON "):])
    if res.get("status") == "measured":
        res["system_before"] = before
        res["system_after_exit"] = system_snapshot()
    return res


# ------------------------------------------------------------------------------------------------ report

def markdown(data):
    L = ["| Engine | Device | Process memory (footprint / RSS, MB) | Peak (footprint / RSS) | First request | System effect while running (available, swap, compressed) |",
         "| --- | --- | --- | --- | --- | --- |"]
    for key, r in data["engines"].items():
        if r["status"] != "measured":
            L.append(f"| {key} | | {r['status']}: {r.get('reason', '')[:110]} | | | |")
            continue
        sb, sr = r.get("system_before") or {}, r.get("system_running") or r.get("system_after_exit") or {}
        eff = ""
        if sb and sr:
            eff = (f"avail {sb['available_mb']} -> {sr['available_mb']}, swap {sb['swap_used_mb']} -> {sr['swap_used_mb']}, "
                   f"compressed {sb.get('compressor_occupied_mb', '?')} -> {sr.get('compressor_occupied_mb', '?')}")
        if "footprint_delta_mb" in r:
            mem = f"+{r['footprint_delta_mb']} / +{r['rss_delta_mb']} (delta over {r['baseline_rss_mb']} MB baseline)"
        else:
            mem = f"{r['footprint_mb']} / {r['rss_mb']}"
        L.append(f"| {r.get('label', key)} | {r.get('device') or 'CPU/GPU in the API process'} | {mem} | "
                 f"{r.get('peak_footprint_mb')} / {r.get('peak_rss_mb')} | {r.get('start_s', r.get('load_and_infer_s'))} s | {eff} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", help="comma separated; default: all installed. See --list")
    ap.add_argument("--list", action="store_true", help="show which engines are installed here and exit")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--markdown", metavar="JSON", help="print the table of a results file and exit")
    ap.add_argument("--out", help="results file (default backend/bench/results/model_memory-<stamp>.json)")
    args = ap.parse_args()
    if args.child:
        run_child(args.child)
        return
    if args.markdown:
        print(markdown(json.load(open(args.markdown))))
        return
    names = {w.name: w for w in WORKERS}
    allnames = list(names) + list(INPROC)
    if args.list:
        for n, w in names.items():
            print(f"{n:24s} {'installed' if not w.available() else 'missing: ' + str(w.available())}")
        for n in INPROC:
            print(f"{n:24s} in-process (checked when run)")
        return
    wanted = [e.strip() for e in args.engines.split(",")] if args.engines else allnames
    unknown = [e for e in wanted if e not in allnames]
    if unknown:
        sys.exit(f"unknown engine(s): {unknown}; choose from {allnames}")
    data = {"machine": machine_info(), "engines": {}}
    RESULTS_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / f"model_memory-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    for n in wanted:
        print(f"== {n} ...", flush=True)
        t = time.time()
        res = run_worker(names[n]) if n in names else run_inproc(n)
        data["engines"][n] = res
        print(f"   {res['status']}" + (f": {res.get('reason')}" if res["status"] != "measured" else
              f"  footprint {res.get('footprint_mb', res.get('footprint_delta_mb'))} MB, rss {res.get('rss_mb', res.get('rss_delta_mb'))} MB"
              f"  ({time.time() - t:.0f} s)"), flush=True)
        out.write_text(json.dumps(data, indent=1))
        time.sleep(3)   # let the machine settle between engines
    data["machine"]["system_after"] = system_snapshot()
    out.write_text(json.dumps(data, indent=1))
    print(f"\nwritten {out}\n")
    print(markdown(data))


if __name__ == "__main__":
    main()
