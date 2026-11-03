"""The phone agent's opening line, ready before the call is picked up, and the warm-up that runs while it rings.

On pickup the agent used to run a whole turn on the word "Hello" (knowledge search, the language model writing a
greeting, speech synthesis, watermark): 1-1.4 s of silence on a warm backend and 7.5 s after a restart. The opening of
a call never needs any of that. It is always the same words for an agent: the AI disclosure (the sentence the streaming
turn speaks, `backend.ai_disclosure`) followed by the agent's `greeting` field (a short generic greeting when it is
empty). So it is rendered once and kept:

  * rendered with the voice a live phone reply of this agent would use (`backend._turn_voice` with source "phone":
    Kokoro unless PHONE_CLONE_VOICE=1 keeps a cloned voice), the disclosure and the greeting as two sentences joined
    by the turn's SENTENCE_GAP_S, exactly like a streamed reply;
  * watermarked, labelled and receipted as one clip (`provenance.apply_provenance`, kind "agent_reply", one receipt
    per rendered clip; every call logs which clip and receipt it played);
  * converted to PCM16 at the call's rate (8 kHz unless Exotel announces 16 or 24 kHz);
  * kept in memory and on disk under backend/data/phone_greetings/ (PHONE_GREETING_DIR overrides): `<key>.wav` (the
    receipted clip), `<key>.json` (what it says, engine, receipt) and `<key>.<rate>.pcm`. The key is a hash of the agent
    id, the words, the language, the voice settings (voice id, PHONE_CLONE_VOICE, TURN_INDIC_ENGINE, KOKORO_ENGINE and
    the Kokoro voice table) and the watermark version, so a changed greeting or voice simply misses the cache (the old
    files of that agent are removed when the new clip is written). A restart reads the disk instead of rendering.

The call (routers/phone_exotel.py) sends it the moment Exotel's `start` event arrives: no speech recognition, no
knowledge search, no language model. When the cache is cold it renders just this text (still no model call).

Renders are triggered ahead of time: when an outbound call is placed (with a warm-up of everything the first turn needs,
while the phone rings), when an inbound link is created, and at backend startup for agents that have been on a call
(`prewarm_all`). The agents are remembered in `<dir>/agents.json` (agent id, owner id, storage mode; no numbers).

Settings: PHONE_PREWARM=off turns the background renders and warm-ups off (a call still renders its greeting when it
needs one), PHONE_PREWARM_MAX (default 8) agents at startup, PHONE_PREWARM_DELAY_S (default 5) after startup,
PHONE_WARM_TTL_S (default 60): a warm-up for the same agent within this time is skipped.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import inspect
import io
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from services import phone_audio as pa
    from services import provenance
except ImportError:  # pragma: no cover - imported as backend.services.phone_greeting
    from backend.services import phone_audio as pa
    from backend.services import provenance

log = logging.getLogger("bonclay.phone.greeting")

CACHE_VERSION = 1
MEMORY_MAX = 64
REGISTRY_MAX = 50
GREETING_MAX_CHARS = 400        # a greeting is one or two sentences; a pasted essay would make the caller wait
CALL_MARKER = "[call connected]"  # the caller's side of the greeting turn in the stored conversation
GENERIC_GREETINGS = {
    "en": "Hello! How can I help you today?",
    "hi": "नमस्ते! मैं आपकी कैसे मदद कर सकती हूँ?",
    "mr": "नमस्कार! मी तुमची कशी मदत करू शकते?",
    "hinglish": "Namaste! Main aapki kaise madad kar sakti hoon?",
}


@dataclass
class Greeting:
    key: str
    agent_id: str
    text: str                   # what is stored as the agent's first turn: "<disclosure> <greeting>"
    language: str
    discloses: bool
    disclosure_s: float         # seconds of the clip that are the AI disclosure (0 when the agent does not disclose)
    engine: str
    voice_id: Optional[str]     # the cloned voice that was used, else None (a preset voice)
    clip_id: str                # the receipt's subject: one receipt per rendered clip
    receipt_id: Optional[str]
    watermarked: bool
    sr: int
    wav: bytes                  # the receipted clip (watermarked, labelled) at the engine's rate
    pcm: Dict[int, bytes] = field(default_factory=dict)   # rate -> PCM16 for the phone

    def meta(self) -> dict:
        return {"version": CACHE_VERSION, "key": self.key, "agent_id": self.agent_id, "text": self.text,
                "language": self.language, "discloses": self.discloses, "disclosure_s": self.disclosure_s,
                "engine": self.engine, "voice_id": self.voice_id, "clip_id": self.clip_id,
                "receipt_id": self.receipt_id, "watermarked": self.watermarked, "sr": self.sr,
                "created_at": time.time()}


_mem: "collections.OrderedDict[str, Greeting]" = collections.OrderedDict()
_mem_lock = threading.Lock()
_inflight: Dict[str, Tuple[Any, "asyncio.Future"]] = {}   # key -> (loop, render future)
_warm_until: Dict[Tuple[str, str], float] = {}           # (agent id, mode) -> monotonic time the warm-up stays fresh
_tts_warm: set = set()                                   # (engine, tts language) spoken once in this process
_bg: set = set()                                         # background tasks (a reference, so they are not collected)
_registry_lock = threading.Lock()
_disk_lock = threading.Lock()


def _flag(name: str, default: str = "on") -> bool:
    return os.getenv(name, default).strip().lower() not in ("off", "0", "false", "no")


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


# ------------------------------------------------------------------------------------------------- the words

def opening(b: Any, agent: dict) -> Tuple[str, Optional[str], str]:
    """(language, AI disclosure or None, greeting) of an agent's call. The language is the greeting's own (the
    language policy's detection, the agent's first language for a greeting with no letters); an empty greeting is a
    short generic one in the agent's first language (English when there is none for it)."""
    greeting = " ".join(str(agent.get("greeting") or "").split())[:GREETING_MAX_CHARS]
    supported = agent.get("supported_languages") or ["en"]
    if greeting:
        language = b._reply_language(greeting, agent)
    else:
        language = str(supported[0] or "en").split("-")[0].lower()
        if language not in GENERIC_GREETINGS:
            language = "en"
        greeting = GENERIC_GREETINGS[language]
    disclosure = b.ai_disclosure(language) if agent.get("disclose_ai") is not False else None
    return language, disclosure, greeting


def watermark_version() -> str:
    if not provenance.enabled():
        return "off"
    return f"audioseal16-snr{provenance.target_snr_db():g}-receipt{provenance.RECEIPT_VERSION}"


def greeting_key(b: Any, agent: dict) -> str:
    """The cache key of an agent's greeting: everything that changes the words, the voice or the watermark."""
    language, disclosure, greeting = opening(b, agent)
    parts = {
        "v": CACHE_VERSION, "agent": str(agent.get("id")), "words": [disclosure, greeting], "lang": language,
        "voice": agent.get("voice_id") or None,
        "phone_clone_voice": os.getenv("PHONE_CLONE_VOICE", "0") == "1",
        "turn_indic_engine": os.getenv("TURN_INDIC_ENGINE", "kokoro"),
        "kokoro_engine": os.getenv("KOKORO_ENGINE", ""),
        "kokoro_voices": sorted((k, list(v)) for k, v in (getattr(b, "KOKORO_VOICES", None) or {}).items()),
        "watermark": watermark_version(),
    }
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


# -------------------------------------------------------------------------------------------------- the disk

def cache_dir(b: Any = None) -> str:
    env = os.getenv("PHONE_GREETING_DIR", "").strip()
    if env:
        return env
    base = getattr(b, "LOCAL_DATA_DIR", None) or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    return os.path.join(base, "phone_greetings")


def _paths(directory: str, key: str, rate: Optional[int] = None) -> str:
    return os.path.join(directory, f"{key}.{rate}.pcm" if rate else f"{key}.wav")


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _save_disk(directory: str, g: Greeting) -> None:
    """Write the clip, its PCM and its metadata (metadata last: a reader that finds it finds the rest), then remove
    the older clips of the same agent (a changed greeting or voice leaves its old files behind otherwise)."""
    with _disk_lock:
        _write(_paths(directory, g.key), g.wav)
        for rate, pcm in list(g.pcm.items()):
            _write(_paths(directory, g.key, rate), pcm)
        _write(os.path.join(directory, f"{g.key}.json"), json.dumps(g.meta(), ensure_ascii=False).encode("utf-8"))
        for name in os.listdir(directory):
            if not name.endswith(".json") or name == "agents.json" or name.startswith(g.key):
                continue
            try:
                with open(os.path.join(directory, name), "r", encoding="utf-8") as f:
                    other = json.load(f)
            except (OSError, ValueError):
                continue
            if other.get("agent_id") != g.agent_id:
                continue
            old = name[:-len(".json")]
            for leftover in os.listdir(directory):
                if leftover.startswith(old + "."):
                    try:
                        os.unlink(os.path.join(directory, leftover))
                    except OSError:
                        pass


def _load_disk(directory: str, key: str) -> Optional[Greeting]:
    try:
        with open(os.path.join(directory, f"{key}.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(_paths(directory, key), "rb") as f:
            wav = f.read()
    except (OSError, ValueError):
        return None
    if meta.get("version") != CACHE_VERSION or meta.get("key") != key or not wav:
        return None
    g = Greeting(key=key, agent_id=str(meta.get("agent_id")), text=str(meta.get("text") or ""),
                 language=str(meta.get("language") or "en"), discloses=bool(meta.get("discloses")),
                 disclosure_s=float(meta.get("disclosure_s") or 0.0), engine=str(meta.get("engine") or "kokoro"),
                 voice_id=meta.get("voice_id"), clip_id=str(meta.get("clip_id") or ""),
                 receipt_id=meta.get("receipt_id"), watermarked=bool(meta.get("watermarked")),
                 sr=int(meta.get("sr") or 24000), wav=wav)
    for rate in pa.SUPPORTED_RATES:
        try:
            with open(_paths(directory, key, rate), "rb") as f:
                g.pcm[rate] = f.read()
        except OSError:
            continue
    return g


def _remember(g: Greeting) -> None:
    with _mem_lock:
        _mem[g.key] = g
        _mem.move_to_end(g.key)
        while len(_mem) > MEMORY_MAX:
            _mem.popitem(last=False)


def _phone_pcm(g: Greeting, rate: int) -> bytes:
    """Blocking: the clip as PCM16 at `rate` (decoded from the receipted WAV: the watermark is in the samples)."""
    import soundfile as sf  # noqa: PLC0415
    x, sr = sf.read(io.BytesIO(g.wav), dtype="float32", always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return pa.to_phone_pcm(x, int(sr), rate)


async def pcm_for(b: Any, g: Greeting, rate: int) -> bytes:
    """The greeting at `rate` Hz, converted (and written to disk) the first time a rate is asked for."""
    pcm = g.pcm.get(rate)
    if pcm is None:
        pcm = await asyncio.to_thread(_phone_pcm, g, rate)
        g.pcm[rate] = pcm
        directory = cache_dir(b)
        try:
            await asyncio.to_thread(_write, _paths(directory, g.key, rate), pcm)
        except OSError as e:
            log.warning("could not keep the %d Hz greeting on disk: %s", rate, e)
    return pcm


def cached(b: Any, agent: dict) -> Optional[Greeting]:
    """The greeting in memory, else None (no I/O)."""
    with _mem_lock:
        return _mem.get(greeting_key(b, agent))


async def _lookup(b: Any, key: str) -> Tuple[Optional[Greeting], str]:
    with _mem_lock:
        g = _mem.get(key)
    if g is not None:
        return g, "memory"
    g = await asyncio.to_thread(_load_disk, cache_dir(b), key)
    if g is not None:
        _remember(g)
        return g, "disk"
    return None, ""


# ------------------------------------------------------------------------------------------------ rendering

async def _render(b: Any, agent: dict, user_id: str, mode: str, key: str) -> Greeting:
    """Speak the opening with the live phone voice, watermark and receipt it. No language model, no knowledge search."""
    language, disclosure, greeting = opening(b, agent)
    run = await b._turn_begin(None, agent["id"], user_id, storage_mode=mode, agent=agent, source="phone")
    run.language = language
    segments = ([(disclosure, True)] if disclosure else []) + [(greeting, False)]
    plan = None
    clips: List[np.ndarray] = []
    sr0: Optional[int] = None
    disclosure_s = 0.0
    try:
        for text, is_disclosure in segments:
            # the same steps as one sentence of the streaming turn (`speak` in backend._turn_event_stream)
            spoken, tts_language = await b._turn_spoken(language, text, False, use_llm=False)
            if plan is None:
                plan = await b._turn_voice(run, tts_language, False)
            clip_cache = getattr(b, "_DISCLOSURE_CLIPS", None)
            ck = (plan.engine_key, plan.tts_language, plan.voice_id if plan.clone_required else None, spoken)
            hit = clip_cache.get(ck) if (is_disclosure and isinstance(clip_cache, dict)) else None
            if hit is not None:
                wav, sr = hit
            else:
                wav, sr = await b._turn_synth(plan, spoken)
                wav = b._to_numpy(wav)
                if is_disclosure and isinstance(clip_cache, dict):
                    clip_cache[ck] = (wav, sr)
            if sr0 is None:
                sr0 = int(sr)
            clip = b._resample_linear(np.asarray(wav, dtype=np.float32), int(sr), sr0)
            clips.append(clip)
            if is_disclosure:
                disclosure_s = clip.size / float(sr0)
        _tts_warm.add((plan.engine_key, plan.tts_language))
        engine_key, used_voice = plan.engine_key, plan.used_voice
    finally:
        if plan is not None:
            plan.cleanup()
    gap = np.zeros(int(sr0 * float(getattr(b, "SENTENCE_GAP_S", 0.08))), dtype=np.float32)
    pieces: List[np.ndarray] = []
    for i, clip in enumerate(clips):
        if i:
            pieces.append(gap)
        pieces.append(clip)
    full = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    text = " ".join(t for t, _ in segments)
    consent_id, consent_verified, voice_name = await b._voice_consent(used_voice, mode, user_id)
    clip_id = str(uuid.uuid4())
    raw = await asyncio.to_thread(b._wav_bytes, full, sr0)
    wav_out, receipt = await asyncio.to_thread(
        lambda: provenance.apply_provenance(
            kind="agent_reply", subject_id=clip_id, wav_bytes=raw, text=text, engine=engine_key, voice_id=used_voice,
            voice_name=voice_name, consent_id=consent_id, consent_verified=consent_verified, user_id=user_id,
            cloud=mode != "local"))
    return Greeting(key=key, agent_id=str(agent["id"]), text=text, language=language, discloses=bool(disclosure),
                    disclosure_s=round(disclosure_s, 3), engine=engine_key, voice_id=used_voice, clip_id=clip_id,
                    receipt_id=receipt.get("receipt_id"), watermarked=receipt.get("watermark_message") is not None,
                    sr=sr0, wav=wav_out)


async def _render_and_store(b: Any, agent: dict, user_id: str, mode: str, key: str, rates: Iterable[int]) -> Greeting:
    started = time.monotonic()
    g = await _render(b, agent, user_id, mode, key)
    for rate in rates:
        g.pcm[rate] = await asyncio.to_thread(_phone_pcm, g, rate)
    _remember(g)
    try:
        await asyncio.to_thread(_save_disk, cache_dir(b), g)
    except OSError as e:
        log.warning("could not keep the phone greeting on disk: %s", e)
    log.info("phone greeting rendered for agent %s in %d ms: %s, %.1f s, clip %s, receipt %s%s", g.agent_id,
             int((time.monotonic() - started) * 1000), g.engine, _clip_seconds(g), g.clip_id, g.receipt_id,
             "" if g.watermarked else " (not watermarked)")
    return g


def _clip_seconds(g: Greeting) -> float:
    try:
        import soundfile as sf  # noqa: PLC0415
        info = sf.info(io.BytesIO(g.wav))
        return info.frames / float(info.samplerate)
    except Exception:  # noqa: BLE001
        return 0.0


async def _still_valid(b: Any, agent: dict, user_id: str, mode: str, g: Greeting) -> bool:
    """A cached clip is stale when today's routing would speak it with another engine or voice (an engine became
    available or went away), or it was made while watermarking was unavailable and it is available now. Loads the
    reply voice (which the warm-up wants anyway)."""
    language, _d, _g = opening(b, agent)
    run = await b._turn_begin(None, agent["id"], user_id, storage_mode=mode, agent=agent, source="phone")
    _spoken, tts_language = await b._turn_spoken(language, g.text, False, use_llm=False)
    plan = await b._turn_voice(run, tts_language, False)
    try:
        if plan.engine_key != g.engine or plan.used_voice != g.voice_id:
            log.info("phone greeting of agent %s was made with %s; the call voice is now %s: rendering again",
                     g.agent_id, g.engine, plan.engine_key)
            return False
    finally:
        plan.cleanup()
    if not g.watermarked and provenance.enabled():
        if await asyncio.to_thread(provenance.unavailable_reason) is None:
            return False
    return True


async def ensure(b: Any, agent: dict, user_id: str, mode: str, *, rates: Iterable[int] = (pa.PHONE_RATE,),
                 validate: bool = False) -> Greeting:
    """The agent's greeting (memory, else disk, else rendered now), with PCM for `rates`. Concurrent callers for the
    same greeting share one render. `validate` re-checks a cached clip against today's voice routing."""
    key = greeting_key(b, agent)
    g, _source = await _lookup(b, key)
    if g is not None and validate and not await _still_valid(b, agent, user_id, mode, g):
        with _mem_lock:
            _mem.pop(key, None)
        g = None
    if g is None:
        loop = asyncio.get_running_loop()
        held = _inflight.get(key)
        if held is None or held[0] is not loop or held[1].done():
            fut = asyncio.ensure_future(_render_and_store(b, agent, user_id, mode, key, tuple(rates)))
            _inflight[key] = (loop, fut)

            def _forget(f, key=key):
                if _inflight.get(key, (None, None))[1] is f:
                    _inflight.pop(key, None)
                if not f.cancelled() and f.exception() is not None:
                    log.warning("phone greeting render failed for agent %s: %r", agent.get("id"), f.exception())
            fut.add_done_callback(_forget)
            held = _inflight[key]
        g = await asyncio.shield(held[1])   # a call that gives up (barge-in, hang-up) does not stop the render
    for rate in rates:
        await pcm_for(b, g, rate)
    return g


async def for_call(b: Any, agent: dict, user_id: str, mode: str, rate: int) -> Tuple[Greeting, bytes, str]:
    """(greeting, PCM at `rate`, where it came from: "memory", "disk" or "rendered") for a call that just started.
    A memory hit does no I/O at all."""
    key = greeting_key(b, agent)
    g, source = await _lookup(b, key)
    if g is None:
        g, source = await ensure(b, agent, user_id, mode, rates=(rate,)), "rendered"
    pcm = g.pcm.get(rate)
    if pcm is None:
        pcm = await pcm_for(b, g, rate)
    return g, pcm, source


# -------------------------------------------------------------------------------------- the stored conversation

async def create_conversation(b: Any, agent: dict, user_id: str, mode: str) -> str:
    """A new conversation for the call (source "phone"), the same row `_turn_context` would insert for a first turn."""
    if mode == "local":
        conv = await asyncio.to_thread(b.local_db.insert, "conversations",
                                       {"agent_id": agent["id"], "user_id": user_id, "source": "phone"})
        return conv["id"]
    sb = b._require_supabase()
    row = {"agent_id": agent["id"], "user_id": user_id}
    try:
        res = await asyncio.to_thread(lambda: sb.table("conversations").insert({**row, "source": "phone"}).execute())
    except Exception as e:  # noqa: BLE001 - migration 013 (source) may not be applied yet
        log.warning("could not flag the conversation as a phone call (apply migration 013): %s", e)
        res = await asyncio.to_thread(lambda: sb.table("conversations").insert(row).execute())
    return res.data[0]["id"]


async def record_turn(b: Any, g: Greeting, agent: dict, user_id: str, mode: str, conversation_id: str,
                      first_audio_ms: int) -> str:
    """Store the greeting as the conversation's first turn (CALL_MARKER for the caller, the greeting for the agent,
    the receipted clip as its audio) through the turn's own store, so the language model's history has it and the
    next reply does not disclose again. Returns the turn id."""
    run = await b._turn_begin(None, agent["id"], user_id, storage_mode=mode, agent=agent, source="phone")
    run.conversation_id = conversation_id
    run.user_text = CALL_MARKER
    run.language = g.language
    run.timings = {"first_audio_ms": int(first_audio_ms), "greeting_ms": int(first_audio_ms)}
    run.start_time = time.time() - first_audio_ms / 1000.0
    turn_id = str(uuid.uuid4())
    await b._turn_store(run, turn_id, g.wav, g.text, engine_key=g.engine, llm_provider=None, llm_model=None,
                        fallback=False)
    return turn_id


# ------------------------------------------------------------------------------------------------- warm-up

def _silent_wav(seconds: float = 0.5, sr: int = 16000) -> bytes:
    import wave  # noqa: PLC0415
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))
    return buf.getvalue()


async def _warm_llm(b: Any) -> str:
    llm_mod = getattr(b, "llm", None)
    fn = getattr(llm_mod, "prewarm", None)
    if fn is None:
        return "skipped (services.llm.prewarm is not there)"
    if inspect.iscoroutinefunction(fn):
        await fn()
    else:
        out = await asyncio.to_thread(fn)
        if inspect.isawaitable(out):
            await out
    return "ok"


async def _warm_shared(shared: Any, agent: dict) -> str:
    out = await shared("phone", str(agent.get("id")))
    if isinstance(out, dict):
        return " ".join(f"{k}:{(v or {}).get('state', v) if isinstance(v, dict) else v}" for k, v in out.items())
    return "ok"


async def _warm_retrieval(b: Any, agent: dict, user_id: str, mode: str) -> str:
    """One tiny search for the agent: loads the embedding model (BGE-M3) and its index (and the reranker when on)."""
    await b.retrieve_context(agent["id"], "Hello", user_id=user_id, storage_mode=mode)
    return "ok"


async def _warm_asr(b: Any) -> str:
    """Local speech recognition only (Groq needs no warm-up): the MLX worker, else openai-whisper."""
    if getattr(b, "groq_client", None) is not None:
        return "skipped (Groq)"
    from services import mlx_asr  # noqa: PLC0415
    whisper = getattr(b, "whisper_model", None)

    def work() -> str:
        import tempfile  # noqa: PLC0415
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(_silent_wav())
            path = tmp.name
        try:
            if mlx_asr.unavailable_reason() is None:
                mlx_asr.warm()
                mlx_asr.transcribe(path, "en")
                return "mlx"
            if whisper is not None and whisper.available:
                if not whisper.loaded:
                    whisper.transcribe(path, fp16=False, language="en")
                return "whisper"
            return "skipped (no local ASR)"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    return await asyncio.to_thread(work)


async def _warm_vad(b: Any) -> str:
    if os.getenv("PHONE_VAD", "silero").strip().lower() == "energy":
        return "skipped (energy)"
    await asyncio.to_thread(pa._silero_session)
    return "ok"


async def _warm_watermark(b: Any) -> str:
    reason = await asyncio.to_thread(provenance.unavailable_reason)   # loads AudioSeal: every reply sentence is marked
    return "ok" if reason is None else f"skipped ({reason[:60]})"


async def _warm_tts(b: Any, agent: dict, user_id: str, mode: str) -> str:
    """The reply voice, loaded and run once per language (Kokoro on MLX compiles per language on its first call).
    Skipped when the greeting was rendered in this process (it did exactly this)."""
    language, _d, _g = opening(b, agent)
    run = await b._turn_begin(None, agent["id"], user_id, storage_mode=mode, agent=agent, source="phone")
    spoken, tts_language = await b._turn_spoken(language, "Okay.", False, use_llm=False)
    plan = await b._turn_voice(run, tts_language, False)
    try:
        if (plan.engine_key, plan.tts_language) in _tts_warm:
            return "already warm"
        await b._turn_synth(plan, spoken)
        _tts_warm.add((plan.engine_key, plan.tts_language))
        return plan.engine_key
    finally:
        plan.cleanup()


async def warm_up(b: Any, agent: dict, user_id: str, mode: str, *, force: bool = False,
                  tts: bool = True) -> Dict[str, Any]:
    """Load what the first reply of a call needs, all off the event loop and side by side: the language model's
    connection (services.llm.prewarm, when it exists), the embedding model and the agent's index (one tiny search),
    local speech recognition (only when Groq is not used), the Silero VAD session, the watermark model and (with
    `tts`) the reply voice. When the backend has its shared live-turn warm-up (`backend.warm_for_live_turns`) that
    does the generic part, and this adds the VAD and the voice this agent's calls use (a cloned voice with
    PHONE_CLONE_VOICE=1, another language). Heavy engines the live reply does not use are never loaded (the voice is whatever
    `_turn_voice` routes a phone reply to). Within PHONE_WARM_TTL_S seconds of the last warm-up of the same agent only
    the language model's connection is refreshed (it is cheap, and a connection can go idle while the phone rings)."""
    k = (str(agent.get("id")), mode)
    now = time.monotonic()
    fresh = not force and _warm_until.get(k, 0.0) > now
    if not fresh:
        _warm_until[k] = now + _num("PHONE_WARM_TTL_S", 60.0)
    results: Dict[str, Any] = {}

    async def step(name: str, coro) -> None:
        t = time.monotonic()
        try:
            detail = await coro
            results[name] = f"{detail} {int((time.monotonic() - t) * 1000)}ms"
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - one part failing must not stop the others (or the call)
            results[name] = f"failed: {type(e).__name__}"
            log.warning("phone warm-up of %s failed: %r", name, e)

    started = time.monotonic()
    shared = getattr(b, "warm_for_live_turns", None)
    if fresh:
        steps = [step("llm", _warm_llm(b))]
    elif shared is not None:
        # the backend's shared live-turn warm-up (LLM connection, speech recognition, Kokoro and its clips, embedding
        # model + a search of this agent, watermark), joined when it is already running; then what is phone-specific
        steps = [step("live", _warm_shared(shared, agent)), step("vad", _warm_vad(b))]
    else:
        steps = [step("llm", _warm_llm(b)), step("retrieval", _warm_retrieval(b, agent, user_id, mode)),
                 step("asr", _warm_asr(b)), step("vad", _warm_vad(b)), step("watermark", _warm_watermark(b))]
    if tts and not fresh:
        steps.append(step("tts", _warm_tts(b, agent, user_id, mode)))   # the voice this agent's calls are spoken in
    await asyncio.gather(*steps)
    if fresh:
        results["skipped"] = "warm"
    log.info("phone warm-up for agent %s in %d ms: %s", agent.get("id"), int((time.monotonic() - started) * 1000),
             ", ".join(f"{k}={v}" for k, v in results.items()))
    return results


async def prepare(b: Any, agent: dict, user_id: str, mode: str, *, warm: bool = True,
                  rates: Iterable[int] = (pa.PHONE_RATE,)) -> None:
    """Render (or check) the greeting and, with `warm`, warm up the first reply alongside it; the reply voice is
    warmed after the greeting (a fresh render already did that). Never raises."""
    try:
        await asyncio.to_thread(remember_agent, b, str(agent["id"]), user_id, mode)
    except Exception as e:  # noqa: BLE001
        log.warning("could not remember the phone agent: %s", e)
    jobs = [ensure(b, agent, user_id, mode, rates=rates, validate=True)]
    if warm:
        jobs.append(warm_up(b, agent, user_id, mode, tts=False))
    for res in await asyncio.gather(*jobs, return_exceptions=True):
        if isinstance(res, BaseException) and not isinstance(res, asyncio.CancelledError):
            log.warning("phone prepare for agent %s: %r", agent.get("id"), res)
    if warm:
        try:
            await _warm_tts(b, agent, user_id, mode)
        except Exception as e:  # noqa: BLE001
            log.warning("phone warm-up of the reply voice failed: %r", e)


def _spawn(coro) -> Optional["asyncio.Task"]:
    try:
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:  # no running loop (called from sync code): nothing to schedule on
        coro.close()
        return None
    _bg.add(task)
    task.add_done_callback(_bg.discard)
    return task


def schedule_prepare(b: Any, agent: dict, user_id: str, mode: str, *, warm: bool = True) -> Optional["asyncio.Task"]:
    """Start `prepare` in the background and return at once (None when PHONE_PREWARM=off)."""
    if not _flag("PHONE_PREWARM"):
        return None
    return _spawn(prepare(b, agent, user_id, mode, warm=warm))


def schedule_warm(b: Any, agent: dict, user_id: str, mode: str) -> Optional["asyncio.Task"]:
    """Start `warm_up` in the background (a call that was not announced: an inbound call). None when switched off."""
    if not _flag("PHONE_PREWARM"):
        return None

    async def run() -> None:
        try:
            await warm_up(b, agent, user_id, mode)
        except Exception as e:  # noqa: BLE001
            log.warning("phone warm-up failed: %r", e)
    return _spawn(run())


async def warm_call_models(b: Any) -> Dict[str, str]:
    """A call has started (pickup, or the stream of an inbound call): the embedding model is run once on a tiny text,
    and with RERANKER=on the reranker is loaded and scores one tiny pair (retrieval.warm_reranker), so the call's first
    search is not a cold start. Each runs in a thread. Returns {"embedding" | "reranker": "ok" | "failed: ..."}; never
    raises."""
    out: Dict[str, str] = {}
    try:
        await asyncio.to_thread(b.embedding_model.encode, ["warm-up"])
        out["embedding"] = "ok"
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - the search loads it on demand if this failed
        out["embedding"] = f"failed: {type(e).__name__}"
        log.warning("call warm-up of the embedding model failed: %r", e)
    if b.retrieval.reranker_enabled():
        try:
            await asyncio.to_thread(b.retrieval.warm_reranker)
            out["reranker"] = "ok"
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - searches go without the reranker when it cannot load
            out["reranker"] = f"failed: {type(e).__name__}"
            log.warning("call warm-up of the reranker failed: %r", e)
    return out


def schedule_call_warm(b: Any) -> Optional["asyncio.Task"]:
    """Start `warm_call_models` in the background when a call starts. None when PHONE_PREWARM is off (tests, or a
    machine that must not warm anything on its own)."""
    if not _flag("PHONE_PREWARM"):
        return None
    return _spawn(warm_call_models(b))


# ------------------------------------------------------------------------------------------ startup and registry

def _registry_path(b: Any) -> str:
    return os.path.join(cache_dir(b), "agents.json")


def _read_registry(b: Any) -> Dict[str, dict]:
    try:
        with open(_registry_path(b), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def remember_agent(b: Any, agent_id: str, user_id: str, mode: str) -> None:
    """Blocking: note that this agent takes phone calls, so startup renders its greeting (newest REGISTRY_MAX kept)."""
    with _registry_lock:
        reg = _read_registry(b)
        reg[agent_id] = {"user_id": user_id, "mode": mode, "last": time.time()}
        if len(reg) > REGISTRY_MAX:
            reg = dict(sorted(reg.items(), key=lambda kv: kv[1].get("last", 0), reverse=True)[:REGISTRY_MAX])
        _write(_registry_path(b), json.dumps(reg, ensure_ascii=False, indent=1).encode("utf-8"))


def _candidates(b: Any) -> List[Tuple[str, str, str]]:
    """(agent id, owner id, mode) of agents that take calls: the registry, then local agents with phone conversations
    (calls made before the registry existed), newest first, without duplicates."""
    out: List[Tuple[str, str, str]] = []
    seen = set()
    for agent_id, e in sorted(_read_registry(b).items(), key=lambda kv: kv[1].get("last", 0), reverse=True):
        if isinstance(e, dict) and e.get("user_id") and e.get("mode") in ("local", "cloud"):
            seen.add((agent_id, e["mode"]))
            out.append((agent_id, str(e["user_id"]), e["mode"]))
    try:
        rows = b.local_db.select("conversations", lambda r: r.get("source") == "phone")
    except Exception:  # noqa: BLE001
        rows = []
    for r in sorted(rows, key=lambda r: str(r.get("created_at") or ""), reverse=True):
        k = (str(r.get("agent_id")), "local")
        if r.get("agent_id") and r.get("user_id") and k not in seen:
            seen.add(k)
            out.append((k[0], str(r["user_id"]), "local"))
    return out


async def _agent_row(b: Any, mode: str, agent_id: str) -> Optional[dict]:
    if mode == "local":
        return await asyncio.to_thread(b.local_db.get, "agents", agent_id)
    sb = getattr(b, "supabase", None)
    if sb is None:
        return None
    res = await asyncio.to_thread(lambda: sb.table("agents").select("*").eq("id", agent_id).limit(1).execute())
    return res.data[0] if res.data else None


async def prewarm_all(b: Any, *, limit: Optional[int] = None) -> List[str]:
    """Render (or check) the greetings of the agents that take phone calls, one after another (Kokoro is ~0.3 s a clip;
    a clip already on disk only costs a routing check). Returns the agent ids that are ready. Never raises."""
    if not _flag("PHONE_PREWARM"):
        return []
    limit = int(limit if limit is not None else _num("PHONE_PREWARM_MAX", 8))
    ready: List[str] = []
    try:
        candidates = await asyncio.to_thread(_candidates, b)
    except Exception as e:  # noqa: BLE001
        log.warning("phone greeting prewarm: could not list the agents: %s", e)
        return ready
    for agent_id, user_id, mode in candidates[:max(0, limit)]:
        try:
            agent = await _agent_row(b, mode, agent_id)
            if agent is None or agent.get("user_id") != user_id:
                continue
            await ensure(b, agent, user_id, mode, rates=(pa.PHONE_RATE,), validate=True)
            ready.append(agent_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("phone greeting prewarm failed for agent %s: %r", agent_id, e)
    if candidates:
        log.info("phone greetings ready for %d of %d agents", len(ready), min(len(candidates), limit))
    return ready


async def prewarm_all_later(b: Any) -> None:
    """`prewarm_all` after PHONE_PREWARM_DELAY_S (default 5 s), so it does not compete with the rest of startup."""
    try:
        await asyncio.sleep(max(0.0, _num("PHONE_PREWARM_DELAY_S", 5.0)))
        await prewarm_all(b)
    except asyncio.CancelledError:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning("phone greeting prewarm failed: %r", e)
