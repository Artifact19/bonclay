"""Warm up everything before a live demo.

After a restart the models load on first use, so the first spoken turn takes ~18 s while a warm one takes ~3 s.
POST /api/warmup loads the models a live voice conversation needs, in the background, in this order: the language
model, speech recognition, Kokoro English and Hindi (on MLX when the Kokoro worker is in use, KOKORO_ENGINE), the embedding model (and the reranker when RERANKER=on), the watermark model and the speaker-check
model, and the Qwen3 voice-clone engine (WARM_CLONE=on, the default; see `_warm_clone`). GET /api/warmup reports progress.
`?include=clone` loads the clone engine even with WARM_CLONE=off, and on a machine without MLX.

Not warmed (too heavy to keep resident, used for Studio renders only): Indic-Speak and Chatterbox.

The clone step starts the clone worker and, when a cloned voice exists, renders one short sentence with the newest one
and throws the audio away, so the first clone and the first cloned sentence after a restart cost what a warm one does
(bench/clone_time.py). Nothing is stored: no generation row, no receipt, no file.

`warm_live_turn_path()` (also `backend.warm_for_live_turns()`) is the light, automatic version of this: only what a live
reply needs, in parallel, each step timed and logged, never raising. It runs in the background when the backend starts
(WARM_ON_START=auto|on|off, see backend.py) and may be called again whenever a call is about to start (a phone ringing),
when it also brings back a model the idle clean-up unloaded.

The backend module is imported lazily inside the workers so this router never creates an import cycle."""
import asyncio
import importlib.util
import io
import logging
import os
import sys
import tempfile
import threading
import time
import wave
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, Query

try:
    from services import llm, mlx_asr, provenance, speaker
    from services.auth import get_current_user_id
    from core.errors import AppError
except ImportError:  # imported as backend.routers.warmup
    from backend.services import llm, mlx_asr, provenance, speaker
    from backend.services.auth import get_current_user_id
    from backend.core.errors import AppError

log = logging.getLogger("bonclay.warmup")

router = APIRouter(prefix="/api/warmup", tags=["warmup"])

NOT_WARMED = "Indic-Speak and Chatterbox are not warmed: used for Studio renders only."


class Skipped(Exception):
    """A component that cannot run on this machine (carries the reason)."""


def _b():
    """The backend module (imported lazily; also works when backend.py runs as __main__)."""
    mod = sys.modules.get("backend")
    if mod is None or not hasattr(mod, "advanced_manager"):
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "advanced_manager"):
            return main
        import backend as mod  # noqa: PLC0415
    return mod


def _silent_wav(seconds: float = 1.0, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))
    return buf.getvalue()


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------ components
# Each runner returns a short detail string, raises Skipped(reason) when the component does not apply here, and
# raises anything else when it fails. They run on the event loop and push blocking work into threads.

def warm_local_llm_mode() -> str:
    """WARM_LOCAL_LLM=auto|on|off (default auto). auto: the local language model server (about 2.2 GB) is started by a warm-up
    only when it is the model that will answer, that is when no hosted model (Groq, an OpenAI-compatible server) is
    configured; with one it starts on first use if the hosted model fails (3.7 s measured). on: always; off: never."""
    raw = (os.getenv("WARM_LOCAL_LLM", "auto") or "auto").strip().lower()
    if raw in ("on", "1", "true", "yes"):
        return "on"
    if raw in ("off", "0", "false", "no"):
        return "off"
    return "auto"


async def _warm_llm() -> str:
    def work() -> str:
        local_reason = llm._mlx.unavailable_reason()
        mode = warm_local_llm_mode()
        if local_reason is None and (mode == "on" or (mode == "auto" and llm.status()["active"] in (None, "mlx"))):
            llm.warm_local()
        try:
            res = llm.chat([{"role": "user", "content": "Reply with the single word OK."}],
                           temperature=0.0, max_tokens=4, timeout=120.0)
        except llm.LLMUnavailable as e:
            raise Skipped("; ".join(f"{k}: {v}" for k, v in e.reasons.items()) or "no language model available") from e
        detail = f"{res.provider} · {res.model}"
        first = llm.status()["active"]
        if first and first != res.provider and first in llm.last_failures:  # the preferred provider failed
            detail += f"  ({first} failed, using {res.provider}: {llm.last_failures[first][:120]})"
        return detail

    return await asyncio.to_thread(work)


def local_asr_warm_mode() -> str:
    """WARM_LOCAL_ASR=auto|on|off (default auto). auto: with a Groq key (and not in offline mode) the live call path
    recognises speech on Groq, so the local Whisper (the offline / rate-limit fallback) is not loaded by a warm-up; it
    starts on its first real use. on: always load it (the models the ASR strategy uses, one size by default). off:
    never load it in a warm-up."""
    raw = (os.getenv("WARM_LOCAL_ASR", "auto") or "auto").strip().lower()
    if raw in ("on", "1", "true", "yes"):
        return "on"
    if raw in ("off", "0", "false", "no"):
        return "off"
    return "auto"


def _local_asr_label() -> str:
    names = {"fast": "small", "accurate": "large-v3-turbo"}
    return "MLX Whisper on the Apple GPU (" + " + ".join(names[r] for r in mlx_asr.resident_roles()) + ")"


async def _warm_asr() -> str:
    mode = local_asr_warm_mode()
    b = _b()
    if mode != "on" and b.groq_client is not None:
        await asyncio.to_thread(b.groq_client.models.list)   # lists models: no audio, no tokens; opens the connection
        return "Groq speech recognition (local Whisper not loaded: it starts on first use; WARM_LOCAL_ASR=on loads it now)"
    if mode == "off":
        raise Skipped("WARM_LOCAL_ASR=off")

    def work() -> str:
        wav = _silent_wav()
        reason = mlx_asr.unavailable_reason()
        # delete=False and closed before use: Windows cannot reopen an open NamedTemporaryFile (ffmpeg gets
        # "Permission denied"), the same pattern the real ASR paths in backend.py use
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav)
            path = tmp.name
        try:
            if reason is None:
                mlx_asr.warm()  # loads the Whisper models the strategy uses (the large one only, by default)
                mlx_asr.transcribe(path, "en")
                return _local_asr_label()
            whisper = _b().whisper_model
            if not whisper.available:
                raise Skipped(f"{reason}; openai-whisper is not installed either")
            whisper.transcribe(path, fp16=False, language="en")
            return f"Whisper '{whisper.size}' on the CPU"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    return await asyncio.to_thread(work)


async def _kokoro(language: str, text: str) -> str:
    b = _b()
    mk = b.mlx_kokoro_service()
    use_mlx = mk.wants_mlx()  # Apple Silicon with the MLX environment: the worker speaks, PyTorch is only a fallback
    if not use_mlx and not _has("kokoro"):
        raise Skipped("the kokoro package is not installed")
    pipeline = await b.advanced_manager.load_model("kokoro")  # the MLX worker handle, or the PyTorch pipeline
    if mk.is_handle(pipeline):
        audio = await asyncio.to_thread(b._kokoro_generate, pipeline, text, language)  # first call per language compiles
        if pipeline.last_engine == "mlx":
            return f"MLX on the Apple GPU · {len(audio) / 24000:.1f} s of audio"
        # the worker could not speak this language: PyTorch did (logged once); report it as the old path would
        if language == "hi" and "h" not in b._kokoro_lang_pipelines:
            raise Skipped("Hindi pipeline could not be built (is espeak-ng installed?); replies fall back to English")
        return f"PyTorch fallback ({pipeline.last_error}) · {len(audio) / 24000:.1f} s of audio"
    if language == "hi":
        if "hi" in b.kokoro_unavailable_languages():
            raise Skipped("Hindi voice needs extra packages (espeak-ng); replies fall back to the English voice")
    audio = await asyncio.to_thread(b._kokoro_generate, pipeline, text, language)
    if language == "hi" and "h" not in b._kokoro_lang_pipelines:
        raise Skipped("Hindi pipeline could not be built (is espeak-ng installed?); replies fall back to English")
    return f"{len(audio) / 24000:.1f} s of audio"


async def _warm_kokoro_en() -> str:
    return await _kokoro("en", "Hello, how can I help you today?")


async def _warm_kokoro_hi() -> str:
    return await _kokoro("hi", "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?")


async def _warm_embedding() -> str:
    if not _has("sentence_transformers"):
        raise Skipped("sentence-transformers is not installed")
    b = _b()

    def work() -> str:
        # one document-sized text and one query, so the first real search does not pay the model's warm-up
        vectors = b.embedding_model.encode(["What is the refund window?", "रिफंड कितने दिन में मिलता है"])
        detail = f"{b.EMBEDDING_MODEL} · {len(vectors[0])}-dimensional vectors"
        if b.retrieval.warm_reranker():          # RERANKER=on: the cross-encoder is part of every search
            detail += f" + reranker {b.retrieval.reranker_model_name()}"
        return detail

    return await asyncio.to_thread(work)


async def _warm_watermark() -> str:
    def work() -> str:
        reason = provenance.unavailable_reason()  # loads the AudioSeal models
        if reason:
            raise Skipped(reason)
        sr = 24000
        t = np.arange(sr, dtype=np.float32) / sr
        clip = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        marked = provenance.watermark(clip, sr, 0x5A5A)
        provenance.detect(marked, sr)
        return "AudioSeal 16-bit"

    return await asyncio.to_thread(work)


async def _warm_speaker() -> str:
    def work() -> str:
        if not speaker.enabled():
            raise Skipped("speaker check is turned off (CONSENT_SPEAKER_CHECK=off)")
        if not speaker.available():  # loads the model
            raise Skipped(getattr(speaker, "_failed", None) or "SpeechBrain or the speaker model is unavailable")
        t = np.arange(speaker.SAMPLE_RATE, dtype=np.float32) / speaker.SAMPLE_RATE
        speaker.embed((0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
        return "ECAPA-TDNN"

    return await asyncio.to_thread(work)


CLONE_WARM_TEXT = "Hello, how can I help you today?"
_clone_rendered: Optional[tuple] = None  # (id of the clone engine instance, voice id) of the last warm-up render


def clone_warm_enabled() -> bool:
    """WARM_CLONE=off turns the clone step off (default on)."""
    return (os.getenv("WARM_CLONE", "on") or "on").strip().lower() not in ("off", "0", "false", "no")


def _newest_clone_voice(b) -> Optional[dict]:
    """The newest locally stored voice made for the current clone engine, or None. Voices of other engines are left
    out: using one would rebuild and re-save its prompt, and a warm-up writes nothing."""
    rows = [v for v in b.get_local_voices()
            if v.get("engine") == b.CLONE_ENGINE and v.get("consent_verified") is not False and v.get("id")]
    return max(rows, key=lambda v: v.get("created_at") or "", default=None)


async def _warm_clone(forced: bool = False) -> str:
    """Start the voice-clone engine (the MLX worker process, ~7 s) and, when a cloned voice is stored, render one short
    sentence with the newest one through the same `_clone_chunks` path a real clone uses, to compile the kernels. The
    audio is discarded: no receipt, no file, no generation row. Without a cloned voice only the engine is started.
    Skipped (with the reason) when the memory guard says the engine does not fit, or, unless `forced`, when it would
    run on PyTorch (too slow to be worth loading at startup). Raises on a failure; the callers report it."""
    global _clone_rendered
    from core import memory_guard  # noqa: PLC0415 - same lazy import style as the rest of this module
    b = _b()
    engine, manager = b.CLONE_ENGINE, b.advanced_manager
    if not forced and not b.mlx_python():
        raise Skipped("voice cloning runs on PyTorch here, which is slow to load; it starts on first use "
                      "(./bonclay warm --clone loads it anyway)")
    if engine not in manager.loaded_models:
        ok, avail, required = memory_guard.fits(engine)  # asks only: unloads nothing, raises nothing
        if not ok:
            have = "?" if avail is None else f"{avail:.0f}"
            raise Skipped(f"not enough free memory for the clone engine ({have} MB free, {required} MB needed"
                          f"{memory_guard.pressure_note()})")
    model = await manager.load_model(engine)
    row = await asyncio.to_thread(_newest_clone_voice, b)
    if row is None:
        return "engine started (no cloned voice to render with yet)"
    key = (id(model), row["id"])
    if _clone_rendered == key:
        return "engine running, already rendered once"
    prompt = await b.load_voice_embedding(row["id"], row.get("user_id") or b.LOCAL_USER, "local", row)
    if not b._is_clone_prompt(prompt, model):
        return f"engine started (the newest voice, {row.get('display_name') or row['id']}, needs a rebuild: not rendered)"
    wav, sr = await b._clone_chunks(model, CLONE_WARM_TEXT, prompt, "en")
    _clone_rendered = key
    return f"engine started, rendered {len(wav) / sr:.1f} s with the voice {row.get('display_name') or row['id']}"


# id, label, runner name (looked up at run time), clone_only: left out with WARM_CLONE=off unless ?include=clone
_COMPONENTS = [
    ("llm", "Language model", "_warm_llm", False),
    ("asr", "Speech recognition", "_warm_asr", False),
    ("kokoro_en", "English voice (Kokoro)", "_warm_kokoro_en", False),
    ("kokoro_hi", "Hindi voice (Kokoro)", "_warm_kokoro_hi", False),
    ("embedding", "Knowledge search (embeddings)", "_warm_embedding", False),
    ("watermark", "Watermark model", "_warm_watermark", False),
    ("speaker", "Speaker check", "_warm_speaker", False),
    ("clone", "Voice cloning (Qwen3)", "_warm_clone", True),
]


# ------------------------------------------------------------------------------------------------ state

_lock = threading.Lock()
_state: Dict[str, Any] = {"state": "idle", "started_at": None, "finished_at": None, "components": []}
_task: Optional["asyncio.Task"] = None


def _fresh_components(include_clone: bool) -> List[dict]:
    return [{"id": cid, "label": label, "state": "pending", "seconds": None, "detail": None}
            for cid, label, _fn, clone_only in _COMPONENTS if include_clone or not clone_only]


def _snapshot() -> dict:
    with _lock:
        comps = _state["components"] or _fresh_components(clone_warm_enabled())
        return {"state": _state["state"], "started_at": _state["started_at"], "finished_at": _state["finished_at"],
                "components": [dict(c) for c in comps], "not_warmed": NOT_WARMED}


def _finish_state(components: List[dict]) -> str:
    return "partial" if any(c["state"] == "failed" for c in components) else "ready"


async def _run(include_clone: bool, forced: bool = False) -> None:
    with _lock:
        components = _state["components"]
    runners: Dict[str, Callable[[], Awaitable[str]]] = {
        cid: globals()[fn] for cid, _l, fn, clone_only in _COMPONENTS if include_clone or not clone_only}
    if "clone" in runners:
        runners["clone"] = lambda: _warm_clone(forced)
    try:
        for comp in components:
            with _lock:
                comp["state"] = "warming"
            started = time.time()
            state, detail = "ready", None
            try:
                detail = await runners[comp["id"]]()
            except Skipped as e:
                state, detail = "skipped", str(e)
            except AppError as e:
                state, detail = "failed", f"{e.code}: {e.message}"[:300]
                log.warning("warm-up of %s failed: %s", comp["id"], detail)
            except Exception as e:  # noqa: BLE001 - one bad component must not stop the others
                state, detail = "failed", f"{type(e).__name__}: {e}"[:300]
                log.warning("warm-up of %s failed: %s", comp["id"], detail, exc_info=True)
            with _lock:
                comp.update(state=state, detail=detail, seconds=round(time.time() - started, 2))
    finally:
        with _lock:
            for comp in components:  # only reachable on cancellation: do not leave anything "warming"
                if comp["state"] in ("warming", "pending"):
                    comp["state"] = "failed"
                    comp["detail"] = comp["detail"] or "interrupted"
            _state["state"] = _finish_state(components)
            _state["finished_at"] = time.time()
        log.info("warm-up finished: %s", _state["state"])


@router.post("")
async def start_warmup(include: str = Query("", max_length=64), user_id: str = Depends(get_current_user_id)):
    """Start warming in the background; when a warm-up is already running, return its current state instead. The
    clone step is part of it unless WARM_CLONE=off; `?include=clone` adds it regardless."""
    global _task
    forced = "clone" in [p.strip() for p in include.split(",")]
    include_clone = forced or clone_warm_enabled()
    with _lock:
        if _state["state"] != "running":
            _state.update(state="running", started_at=time.time(), finished_at=None,
                          components=_fresh_components(include_clone))
            _task = asyncio.create_task(_run(include_clone, forced))
    return _snapshot()


@router.get("")
async def get_warmup(user_id: str = Depends(get_current_user_id)):
    return _snapshot()


# ------------------------------------------------------------------------------------------------ live-turn warm-up

STEP_IDS = ("llm", "asr", "kokoro_en", "kokoro_hi", "clips", "embedding", "knowledge", "watermark", "clone")
KNOWLEDGE_WARM_AGENTS = 5   # agents whose knowledge search is run once (their chunks are read, checked and tokenized)
_live_task: Optional["asyncio.Future"] = None


async def _step(results: Dict[str, dict], name: str, fn: Callable[[], Awaitable[str]]) -> bool:
    """Run one warm-up step: time it, log it, record it in `results`, never raise. True when it worked."""
    started = time.time()
    state, detail = "ready", None
    try:
        detail = await fn()
    except Skipped as e:
        state, detail = "skipped", str(e)
    except asyncio.CancelledError:
        raise
    except AppError as e:
        state, detail = "failed", f"{e.code}: {e.message}"[:200]
    except Exception as e:  # noqa: BLE001 - one step failing must not stop the others
        state, detail = "failed", f"{type(e).__name__}: {e}"[:200]
    seconds = round(time.time() - started, 2)
    results[name] = {"state": state, "seconds": seconds, "detail": detail}
    (log.warning if state == "failed" else log.info)("warm-up step %s: %s in %.2f s%s", name, state, seconds,
                                                      f" ({detail})" if detail else "")
    return state == "ready"


async def _warm_llm_connection() -> str:
    if not llm._prewarm_targets():
        raise Skipped("no hosted language model is configured")
    await asyncio.to_thread(llm.prewarm, wait=True)
    return "connection to the hosted model opened"


async def _warm_asr_connection() -> str:
    b = _b()
    if b.groq_client is None:  # local speech recognition: a live call needs it loaded
        return await _warm_asr()
    await asyncio.to_thread(b.groq_client.models.list)  # lists models: no audio, no tokens; opens the connection
    if local_asr_warm_mode() == "on":   # the fallback too, for a machine that must not wait when Groq refuses
        await _warm_asr()
    return "connection to Groq speech recognition opened"


async def _warm_clips() -> str:
    count = await _b()._prerender_turn_clips()
    return f"{count} clips rendered"


def _warm_agent_extras(agent: dict) -> None:
    """Start what the agent's first turn would otherwise lack: its fact sheet (Key facts in the prompt) and its
    speech-recognition vocabulary and name list. Both build in the background; a failure only costs the extra."""
    try:
        from services import fact_sheet, query_understanding  # noqa: PLC0415
        fact_sheet.prefetch(agent["id"], agent.get("user_id"), "local")
        query_understanding.get_profile(agent, agent.get("user_id"), "local")
    except Exception as e:  # noqa: BLE001
        log.debug("agent warm-up extras skipped: %s", e)


async def _warm_knowledge(agent_id: Optional[str]) -> str:
    """Run the knowledge search once for the agents a live call may reach, so its first real search finds the
    tokenizer caches filled and the table checks done (see backend._refresh_local_embeddings)."""
    b = _b()
    agents = await asyncio.to_thread(b.local_db.select, "agents")
    if agent_id:
        agents = [a for a in agents if a.get("id") == agent_id]
    done = 0
    for agent in agents[:KNOWLEDGE_WARM_AGENTS]:
        await b.retrieve_context(agent["id"], "opening hours", user_id=agent.get("user_id"), storage_mode="local")
        _warm_agent_extras(agent)
        done += 1
    if not done:
        raise Skipped("no agents yet")
    return f"{done} agent(s)"


async def _live_warm(reason: str, agent_id: Optional[str]) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    started = time.time()

    async def voice_chain() -> None:
        en = await _step(results, "kokoro_en", _warm_kokoro_en)
        hi = await _step(results, "kokoro_hi", _warm_kokoro_hi)
        if en or hi:  # the clips are spoken by Kokoro
            await _step(results, "clips", _warm_clips)
        else:
            results["clips"] = {"state": "skipped", "seconds": 0.0, "detail": "no Kokoro voice is available"}

    async def knowledge_chain() -> None:
        if await _step(results, "embedding", _warm_embedding):
            await _step(results, "knowledge", lambda: _warm_knowledge(agent_id))
        else:
            results["knowledge"] = {"state": "skipped", "seconds": 0.0, "detail": "no embedding model"}

    await asyncio.gather(
        _step(results, "llm", _warm_llm_connection),
        _step(results, "asr", _warm_asr_connection),
        voice_chain(),
        knowledge_chain(),
        _step(results, "watermark", _warm_watermark),
    )
    # The clone engine goes last, after every live-path step has finished: on a 16 GB Mac its 3 GB must not compete with
    # Kokoro (the live path), which has to win the memory. Not for a ringing phone: it must not wait for a worker
    # process to start (its own warm-up renders the agent's voice), and a render on every ring would take the GPU
    # from the call before it.
    if reason != "phone":
        if clone_warm_enabled():
            await _step(results, "clone", _warm_clone)
        else:
            results["clone"] = {"state": "skipped", "seconds": 0.0, "detail": "WARM_CLONE=off"}
    ordered = {k: results[k] for k in STEP_IDS if k in results}
    log.info("warm-up for live turns (%s) finished in %.2f s: %s", reason, time.time() - started,
             ", ".join(f"{k} {v['state']} {v['seconds']:.2f}s" for k, v in ordered.items()))
    return ordered


async def warm_live_turn_path(reason: str = "manual", agent_id: Optional[str] = None) -> Dict[str, dict]:
    """Warm what a live spoken reply needs, now: the hosted model's connection (llm.prewarm), speech recognition (the
    Groq connection, or the local model when there is no key), Kokoro English and Hindi plus the pre-rendered disclosure
    and filler clips, the embedding model with one search, and the watermark model. The Qwen3-TTS clone engine is started and run once with the newest cloned voice (WARM_CLONE=off
    skips that; a phone about to ring skips it too). Nothing else heavy is loaded (no DhVaani, Indic-Speak, Chatterbox or
    local language model). Steps run in parallel and are timed and logged;
    a failing step is reported, never raised. A call made while a warm-up is already running joins it. Returns
    {step: {"state": ready|skipped|failed, "seconds", "detail"}}.

    `agent_id` limits the knowledge step to one agent (a phone about to ring for it)."""
    global _live_task
    task = _live_task
    if task is None or task.done():
        task = _live_task = asyncio.ensure_future(_live_warm(reason, agent_id))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - belt and braces: a warm-up must never raise into its caller
        log.warning("warm-up for live turns failed: %s", e)
        return {}
