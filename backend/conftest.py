"""Shared pytest fixtures. Everything here must work on CPU without models."""
import ast
import math
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent
# Tests never read the developer's real backend/.env or frontend/.env.local: backend.py and core/offline.py skip their
# load_dotenv calls when this flag is set, so the suite gives the same answers on every machine. Set before any import
# of the app.
os.environ["BONCLAY_SKIP_DOTENV"] = "1"
FIXTURES_DIR = BACKEND_DIR / "fixtures"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(autouse=True)
def _kokoro_on_pytorch_by_default(monkeypatch):
    """Tests never start the MLX Kokoro worker (a real process and model download); the tests of that engine opt in."""
    monkeypatch.setenv("KOKORO_ENGINE", "torch")


@pytest.fixture(autouse=True)
def _exotel_account_state_in_tmp(monkeypatch, tmp_path):
    """The Exotel account state (which account is out of credit) is written to a file; tests never touch backend/data."""
    monkeypatch.setenv("BONCLAY_EXOTEL_STATE_FILE", str(tmp_path / "exotel_accounts.json"))


@pytest.fixture(autouse=True)
def _memory_guard_off_by_default(monkeypatch):
    """The guard reads this machine's real free memory, so model-manager tests would pass or fail with whatever else is
    open. Tests of the guard itself (test_memory_guard.py) set their own environment."""
    monkeypatch.setenv("MEMORY_GUARD", "off")


@pytest.fixture(autouse=True)
def _memory_pressure_and_live_engines_neutral_by_default(monkeypatch):
    """macOS memory pressure (core/memory_guard.pressure_level) and whether the phone channel is configured (live-call
    engines kept warm) depend on the machine and its environment; tests of those set their own."""
    from core import memory_guard
    monkeypatch.setattr(memory_guard, "pressure_level", lambda: None)
    monkeypatch.setenv("KEEP_LIVE_ENGINES_WARM", "off")
    for name in ("ENGINE_IDLE_UNLOAD_MIN", "MLX_ASR_IDLE_UNLOAD_S", "WARM_LOCAL_ASR", "MLX_ASR_RESIDENT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _retrieval_models_off_by_default(monkeypatch):
    """The suite must behave the same whether or not this machine has downloaded the reranker or BGE-M3's head files:
    RERANKER=auto would load the real cross-encoder (and call it from several test threads), M3_MODES=on would load the
    heads and start a background passage worker. Tests of those features set their own environment or fakes."""
    monkeypatch.setenv("RERANKER", "off")
    monkeypatch.setenv("M3_MODES", "off")


@pytest.fixture(autouse=True)
def _grounding_extras_off_by_default(monkeypatch, tmp_path):
    """Grounded-answer extras that would change what other tests' faked models say (services/grounding.py,
    services/fact_sheet.py) are off here; tests/test_grounding.py and tests/test_fact_sheet.py turn them on. The fact
    sheet would also call the faked model from a background thread and write under backend/data; the number guard would
    replace a faked reply whose numbers are in no document; the confidence gate would skip the faked model."""
    monkeypatch.setenv("FACT_SHEET", "off")
    monkeypatch.setenv("FACT_SHEET_DIR", str(tmp_path / "fact_sheets"))
    monkeypatch.setenv("ANSWER_GUARD", "off")
    monkeypatch.setenv("CONFIDENCE_GATE", "off")


@pytest.fixture(autouse=True)
def _caller_memory_private_and_offline(monkeypatch, tmp_path):
    """The caller details (services/caller_memory.py) go to a file under tmp_path, never backend/data, and the small model
    that reads them is never called: tests of it (tests/test_caller_memory.py) fake the call and turn it on."""
    from services import caller_memory
    monkeypatch.setenv("BONCLAY_CALLER_DETAILS_FILE", str(tmp_path / "caller_details.json"))
    monkeypatch.setenv("CALLER_MEMORY_LLM", "off")
    caller_memory.forget()
    yield
    caller_memory.forget()


@pytest.fixture(autouse=True)
def _tool_readback_off_by_default(monkeypatch):
    """Committing tools (book_appointment, request_meeting) answer NOT DONE until the agent has read the slot back and
    asked (services/tools.py READBACK_TOOLS). The loop and endpoint tests script a model that books in one step, so the
    rule is off here; tests/test_tools_readback.py turns it on."""
    monkeypatch.setenv("TOOLS_READBACK", "off")


@pytest.fixture(autouse=True)
def _background_local_model_off_by_default(monkeypatch):
    """Background jobs (question index, fact sheet) fall back to a local Gemma model when the hosted one is unavailable.
    Unit tests never start it, whether or not this machine has downloaded it; tests of the fallback set
    BACKGROUND_LLM_FALLBACK=local and inject a fake model call."""
    monkeypatch.setenv("BACKGROUND_LLM_FALLBACK", "off")
    monkeypatch.delenv("BACKGROUND_LLM_FORCE", raising=False)


@pytest.fixture(autouse=True)
def _voice_vault_off_by_default(monkeypatch, tmp_path):
    """Tests never touch the real Keychain or write keys under backend/data. The vault is off, so tests that write
    plain voice files and read them back keep working; tests/test_voice_vault.py turns it on, with a fixed key."""
    import base64
    from services import voice_vault
    monkeypatch.setenv("VOICE_VAULT", "off")
    monkeypatch.setenv("BONCLAY_VAULT_KEY", base64.b64encode(b"k" * 32).decode("ascii"))
    monkeypatch.setenv("VOICE_VAULT_DIR", str(tmp_path / "voice_keys"))
    monkeypatch.setenv("VOICE_VAULT_KEYCHAIN", "off")
    voice_vault.reset()
    yield
    voice_vault.reset()


def write_sine_wav(path: Path, seconds: float = 5.0, sr: int = 24000, freq: float = 220.0):
    n = int(seconds * sr)
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / sr) * (0.6 + 0.4 * math.sin(2 * math.pi * 3 * i / sr))))
        for i in range(n)
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)
    return path


@pytest.fixture(scope="session")
def ref_wav() -> Path:
    """5 second, 24 kHz mono 16-bit WAV reference clip, generated on demand."""
    FIXTURES_DIR.mkdir(exist_ok=True)
    p = FIXTURES_DIR / "ref_5s_24k.wav"
    if not p.exists():
        write_sine_wav(p)
    return p


@pytest.fixture(scope="session")
def rag_txt() -> Path:
    FIXTURES_DIR.mkdir(exist_ok=True)
    p = FIXTURES_DIR / "rag_sample.txt"
    if not p.exists():
        p.write_text(
            "BonClay is a voice intelligence platform. " * 40
            + "The refund window is thirty days from purchase. " * 40,
            encoding="utf-8",
        )
    return p


def _extract_class(source_path: Path, class_name: str, namespace: dict):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, str(source_path), "exec"), namespace)
            return namespace[class_name]
    return None


@pytest.fixture(scope="session")
def model_manager_cls():
    """AdvancedModelManager without importing backend.py's heavy deps (whisper, qwen_tts...).

    Prefers an extracted module (core.model_manager) if the refactor created one,
    otherwise extracts the class definition from backend.py via ast.
    """
    torch = pytest.importorskip("torch")
    from core.errors import AppError

    class _AnyKeys(dict):  # every engine key counts as registered
        def __contains__(self, key):
            return True
    import asyncio
    import gc
    import time
    from datetime import datetime
    from typing import Any, Dict, Tuple

    import psutil

    try:
        from core.model_manager import AdvancedModelManager  # type: ignore

        return AdvancedModelManager
    except Exception:
        pass
    ns = dict(torch=torch, asyncio=asyncio, gc=gc, time=time, datetime=datetime, psutil=psutil,
              Dict=Dict, Tuple=Tuple, Any=Any, ENGINE_REGISTRY=_AnyKeys(), AppError=AppError, DEVICE="cpu", DTYPE=torch.float32)
    cls = _extract_class(BACKEND_DIR / "backend.py", "AdvancedModelManager", ns)
    if cls is None:
        pytest.skip("AdvancedModelManager not found")
    return cls


def models_present() -> bool:
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return (hf / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base").exists()


@pytest.fixture(scope="session")
def app_client():
    """FastAPI TestClient for the real app, or skip if the app cannot import on this machine."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    try:
        import backend  # noqa: WPS433
    except BaseException as exc:  # heavy deps / model downloads unavailable
        pytest.skip(f"backend app not importable here: {type(exc).__name__}: {exc}")
    return TestClient(backend.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _no_gpu_workers(monkeypatch):
    """Tests must not depend on (or start) the Apple-GPU speech-recognition worker this machine may have."""
    monkeypatch.setenv("ASR_BACKEND", "whisper")


@pytest.fixture(autouse=True)
def _no_chatterbox_worker(monkeypatch):
    """Tests must not depend on (or start) the Chatterbox worker this machine may have installed: it is off unless a
    test turns it on by faking services.chatterbox_clone.unavailable_reason."""
    monkeypatch.setenv("CHATTERBOX", "off")


@pytest.fixture(autouse=True)
def _no_dhvaani_worker(monkeypatch):
    """Same for DhVaani (Indian-language cloning): off unless a test turns it on by faking
    services.dhvaani_clone.unavailable_reason. INDIC_CLONE_ENGINE is cleared so the developer's .env cannot steer routing."""
    monkeypatch.setenv("DHVAANI", "off")
    monkeypatch.delenv("INDIC_CLONE_ENGINE", raising=False)


@pytest.fixture(autouse=True)
def _no_cloud_keys(monkeypatch):
    """Tests never call Groq with the developer's real key from backend/.env; a test that needs it sets its own."""
    import sys

    # Empty rather than unset: load_dotenv() in backend.py never overrides a variable that exists.
    monkeypatch.setenv("GROQ_API_KEY", "")
    for n in range(2, 10):  # accounts 2..9 (services/groq_keys.py)
        monkeypatch.setenv(f"GROQ_API_KEY_{n}", "")
    monkeypatch.delenv("GROQ_CHAT_MODEL", raising=False)
    for name in ("backend", "backend.backend"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "groq_client"):
            monkeypatch.setattr(mod, "groq_client", None)
    try:
        from services import llm
        llm.last_failures.clear()
    except Exception:  # noqa: BLE001
        pass
    try:
        from services import groq_keys
        groq_keys.reset()  # no pause, cool-down or refused key carries over from another test
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(autouse=True)
def _isolated_receipts(monkeypatch, tmp_path_factory):
    """Receipts (and their signing secret) written during a test go to a temp dir, never to backend/data."""
    from services import provenance

    monkeypatch.setattr(provenance, "_data_dir", str(tmp_path_factory.mktemp("provenance")))
    monkeypatch.setattr(provenance, "_secret_cache", None)
    provenance._store.reset()
    yield
    provenance._store.reset()


@pytest.fixture(autouse=True)
def _isolated_instant_answers(monkeypatch, tmp_path_factory):
    """Every turn looks for an instant answer: tests read and write their own (empty) instant answers file, never
    backend/data/instant_answers.json, with the settings at their defaults and no answer or audio cached from another test."""
    monkeypatch.setenv("BONCLAY_INSTANT_ANSWERS_FILE", str(tmp_path_factory.mktemp("instant") / "instant_answers.json"))
    for name in ("INSTANT_ANSWERS", "INSTANT_ANSWER_THRESHOLD", "INSTANT_ANSWER_MIN_SUPPORT", "INSTANT_ANSWER_MIN_ASKS",
                 "INSTANT_ANSWER_BUILD_MAX"):
        monkeypatch.delenv(name, raising=False)
    try:
        from services import instant_answers as ia
        ia.clear_cache()
        ia.audio_cache.clear()
    except Exception:  # noqa: BLE001
        pass
