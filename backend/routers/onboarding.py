"""First-run guidance: the legacy five-step checklist, the six-step business setup checklist, what needs attention, and
setup hints, all from the user's real data and environment.

`backend` (backend.py) is imported lazily inside handlers: it loads this module while it is still importing.
"""
import asyncio
import logging
import os
import platform
import shutil
import sys
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request

try:
    from services.auth import get_current_user_id
    from services import groq_keys
except ImportError:
    from backend.services.auth import get_current_user_id
    from backend.services import groq_keys

log = logging.getLogger("bonclay.onboarding")

router = APIRouter(tags=["onboarding"])

INDIAN_LANGUAGES = {"hi", "mr", "bn", "gu", "kn", "ml", "or", "pa", "ta", "te", "as", "ur", "ne", "sa", "sd", "kok",
                    "mai", "doi", "brx", "sat", "ks", "mni"}


def _backend():
    mod = sys.modules.get("backend") or sys.modules.get("__main__")
    if mod is None or not hasattr(mod, "local_db"):
        import backend as mod  # noqa: WPS433 - lazy: backend imports this module
    return mod


def _is_indian(code: Any) -> bool:
    return str(code or "").strip().lower().replace("_", "-").split("-")[0] in INDIAN_LANGUAGES


# -- checklist ---------------------------------------------------------------------------------------------------

def _local_facts(B, user_id: str) -> Dict[str, Any]:
    voices = [v for v in B.get_local_voices() if B._owns_local(v, user_id)]
    agents = B.local_db.select("agents", lambda r: r.get("user_id") == user_id)
    agent_ids = {a["id"] for a in agents}
    docs = B.local_db.select("documents", lambda r: r.get("agent_id") in agent_ids)
    runs = B.local_db.select("evaluation_runs", lambda r: r.get("agent_id") in agent_ids)
    convs = B.local_db.select("conversations", lambda r: r.get("agent_id") in agent_ids)
    return {"voice": bool(voices), "agent": bool(agents), "doc": bool(docs), "eval": bool(runs),
            **_conversation_facts(convs), "agent_voice": any(a.get("voice_id") for a in agents),
            "agent_id": _first_agent_id(agents), "agent_with_docs": _first_agent_id(
                [a for a in agents if any(d.get("agent_id") == a["id"] for d in docs)])}


def _first_agent_id(agents: List[dict]) -> Optional[str]:
    """The agent a deep link should open: the oldest one (the user's first), by creation time when it is known."""
    rows = sorted(agents, key=lambda a: str(a.get("created_at") or ""))
    return str(rows[0]["id"]) if rows else None


def _conversation_facts(convs: List[dict]) -> Dict[str, bool]:
    """`talked`: the owner held a conversation with an agent. `practice`: a Test call, or a real phone call, was made."""
    return {"talked": bool(convs),
            "practice": any(c.get("is_test") or c.get("source") == "phone" for c in convs)}


def _cloud_facts(B, user_id: str) -> Dict[str, Any]:
    sb = B.supabase
    projs = sb.table("projects").select("id").eq("owner_user_id", user_id).execute().data or []
    pids = [p["id"] for p in projs]
    voice = False
    if pids:
        voice = bool(sb.table("voices").select("id").in_("project_id", pids).limit(1).execute().data)
    agents = sb.table("agents").select("*").eq("user_id", user_id).limit(1000).execute().data or []
    aids = [a["id"] for a in agents]
    doc = ev = False
    convs: List[dict] = []
    docs: List[dict] = []
    if aids:
        docs = sb.table("documents").select("agent_id").in_("agent_id", aids).limit(1000).execute().data or []
        doc = bool(docs)
        ev = bool(sb.table("evaluation_runs").select("id").in_("agent_id", aids).limit(1).execute().data)
        for cols in ("id, is_test, source", "id, is_test", "id"):  # migrations 010 and 013 may not be applied yet
            try:
                convs = sb.table("conversations").select(cols).in_("agent_id", aids).limit(1000).execute().data or []
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("onboarding: conversations query (%s) failed: %s", cols, exc)
    return {"voice": bool(voice), "agent": bool(agents), "doc": doc, "eval": ev,
            **_conversation_facts(convs), "agent_voice": any(a.get("voice_id") for a in agents),
            "agent_id": _first_agent_id(agents), "agent_with_docs": _first_agent_id(
                [a for a in agents if any(d.get("agent_id") == a["id"] for d in docs)])}


def _spoke_indian(B, user_id: str, cloud: bool) -> bool:
    try:
        from services.auth import LOCAL_USER
        sb = B.supabase if cloud else None
        if sb is not None:
            try:
                rows = sb.table("generations").select("language").eq("user_id", user_id).limit(10000).execute().data
                return any(_is_indian(r.get("language")) for r in rows or [])
            except Exception as exc:  # noqa: BLE001
                log.warning("onboarding: generations query failed, using the local file: %s", exc)
        gs = B.generations_store
        for r in gs._read_local():
            mine = r.get("user_id") == user_id or (not r.get("user_id") and user_id == LOCAL_USER)
            if mine and _is_indian(r.get("language")):
                return True
    except Exception as exc:  # noqa: BLE001
        log.warning("onboarding: could not read generations: %s", exc)
    return False


def _phone_connected() -> bool:
    """What GET /api/phone/exotel/status reports as `configured`: every Exotel setting is present."""
    try:
        try:
            from routers import phone_exotel
        except ImportError:  # pragma: no cover - imported as backend.routers.onboarding
            from backend.routers import phone_exotel
        return not phone_exotel.missing_settings()
    except Exception as exc:  # noqa: BLE001
        log.warning("onboarding: could not read the phone status: %s", exc)
        return False


def _approvals_waiting() -> int:
    """Meeting requests the agent has put in that nobody has decided yet."""
    try:
        try:
            from mcp_servers import meetings
        except ImportError:  # pragma: no cover
            from backend.mcp_servers import meetings
        return len(meetings.list_requests("pending"))
    except Exception as exc:  # noqa: BLE001
        log.warning("onboarding: could not count pending approvals: %s", exc)
        return 0


def _agent_link(agent_id: Optional[str], tab: Optional[str] = None) -> str:
    base = "/dashboard/studio/agents"
    if not agent_id:
        return base
    return f"{base}?agent={agent_id}" + (f"&tab={tab}" if tab else "")


def _setup_steps(facts: Dict[str, Any], phone: bool) -> List[Dict[str, Any]]:
    """The six things a business does to get an agent answering its phone. Each: why, minutes, one button."""
    agent_id = facts.get("agent_id")
    return [
        {"id": "create_agent", "title": "Create an agent", "done": bool(facts["agent"]), "minutes": 2,
         "why": "Your agent answers questions in your organisation's name.",
         "cta": "Create agent", "href": "/dashboard/studio/agents"},
        {"id": "add_documents", "title": "Add your organisation's documents", "done": bool(facts["doc"]), "minutes": 3,
         "why": "Timings, fees and rules, so it answers from your own facts instead of guessing.",
         "cta": "Add documents", "href": _agent_link(facts.get("agent_with_docs") or agent_id, "knowledge")},
        {"id": "talk_to_it", "title": "Talk to it", "done": bool(facts["talked"]), "minutes": 1,
         "why": "Ask it one question you expect callers to ask and hear the reply.",
         "cta": "Talk to your agent", "href": "/dashboard/playground"},
        {"id": "pick_voice", "title": "Pick or clone a voice",
         "done": bool(facts["agent_voice"] or facts["voice"]), "minutes": 2,
         "why": "Choose how your agent sounds, or record your own voice.",
         "cta": "Choose a voice", "href": "/dashboard/voices?new=1"},
        {"id": "connect_phone", "title": "Connect your phone line", "done": bool(phone), "minutes": 5,
         "why": "Lets people call your number and reach your agent.",
         "cta": "Connect phone", "href": _agent_link(agent_id, "phone")},
        {"id": "practice_call", "title": "Make a practice call", "done": bool(facts["practice"]), "minutes": 2,
         "why": "Hear a pretend caller ask your agent for help, before real callers do.",
         "cta": "Practice call", "href": _agent_link(agent_id)},
    ]


@router.get("/api/onboarding")
async def onboarding(request: Request, user_id: str = Depends(get_current_user_id)):
    B = _backend()
    cloud = B._storage_mode(request) == "cloud"
    facts: Dict[str, Any] = {"voice": False, "agent": False, "doc": False, "eval": False, "talked": False,
                             "practice": False, "agent_voice": False, "agent_id": None, "agent_with_docs": None}
    try:
        facts = await asyncio.to_thread(_cloud_facts if cloud else _local_facts, B, user_id)
    except Exception as exc:  # noqa: BLE001 - a failing store must not break the Overview page
        log.warning("onboarding: could not read user data: %s", exc)
    indian = await asyncio.to_thread(_spoke_indian, B, user_id, cloud)
    phone = await asyncio.to_thread(_phone_connected)
    approvals = await asyncio.to_thread(_approvals_waiting)

    steps = [
        ("clone_voice", "Clone a voice", facts["voice"], "/dashboard/voices?new=1",
         "A cloned voice is what makes every agent and generation sound like someone."),
        ("speak_indian", "Speak a line in an Indian language", indian, "/dashboard/studio",
         "Hear your voice in Hindi, Tamil or another Indian language, the thing BonClay is built for."),
        ("create_agent", "Create an agent", facts["agent"], "/dashboard/studio/agents",
         "An agent turns a voice and a role description into something you can hold a conversation with."),
        ("add_knowledge", "Give an agent a document", facts["doc"], "/dashboard/studio/agents",
         "Documents let the agent answer from your own material instead of guessing."),
        ("run_attack_test", "Run the attack test", facts["eval"], "/dashboard/eval",
         "Adversarial personas show where the agent invents facts or breaks its rules before a customer does."),
    ]
    out = [{"id": i, "title": t, "done": bool(d), "href": h, "hint": hint} for i, t, d, h, hint in steps]
    setup = _setup_steps(facts, phone)
    return {"steps": out, "completed": sum(1 for s in out if s["done"]), "total": 5,
            "setup": setup, "setup_completed": sum(1 for s in setup if s["done"]), "setup_total": len(setup),
            "agent_id": facts.get("agent_id"), "has_agent": bool(facts["agent"]),
            "phone_connected": bool(phone), "approvals_waiting": approvals}


# -- setup hints -------------------------------------------------------------------------------------------------

def _hint(id_: str, severity: str, title: str, detail: str, command: Optional[str] = None) -> Dict[str, Any]:
    return {"id": id_, "severity": severity, "title": title, "detail": detail, "command": command}


def _install_cmd(pkg: str, winget_id: str) -> str:
    system = platform.system()
    if system == "Darwin":
        return f"brew install {pkg}"
    if system == "Windows":
        return f"winget install {winget_id}"
    return f"sudo apt install {pkg}"


def _apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _llm_active(B) -> Optional[str]:
    try:
        return B.llm.status().get("active")
    except Exception as exc:  # noqa: BLE001
        log.warning("onboarding: llm.status failed: %s", exc)
        return None


@router.get("/api/setup-hints")
async def setup_hints(request: Request, user_id: str = Depends(get_current_user_id)):
    B = _backend()
    hints: List[Dict[str, Any]] = []
    apple = _apple_silicon()
    has_groq = groq_keys.has_any()

    llm_active = _llm_active(B)
    if llm_active is None:
        hints.append(_hint(
            "no_llm", "error", "No language model is available",
            "Agents cannot reply and Agent Studio cannot draft personas until a language model is set up.",
            "bash backend/scripts/setup_mlx.sh" if apple else "add GROQ_API_KEY to backend/.env"))

    try:
        mlx_ok = bool(B.mlx_python())
    except Exception:  # noqa: BLE001
        mlx_ok = False
    if apple and not mlx_ok:
        hints.append(_hint(
            "no_mlx", "warning", "Voice cloning runs on the CPU (slow)",
            "This Mac has an Apple GPU but the MLX environment is not installed, so cloning falls back to the CPU.",
            "bash backend/scripts/setup_mlx.sh"))

    try:
        indic_reason = B.indic_speak_service().indic_speak_unavailable_reason()
    except Exception as exc:  # noqa: BLE001
        indic_reason = f"Indic-Speak could not be checked: {exc}"
    if indic_reason and os.getenv("INDIC_SPEAK", "").strip().lower() not in ("0", "off", "false", "disabled"):
        hints.append(_hint(
            "no_indic_speak", "info", "Indian-language preset voices use Kokoro (Hindi only)",
            "Indic-Speak adds preset voices for 20+ Indian languages. It needs an HF_TOKEN with access to the model.",
            "./bonclay setup --indic"))

    try:
        cb_reason = B.chatterbox_service().unavailable_reason()
    except Exception as exc:  # noqa: BLE001
        cb_reason = f"Chatterbox could not be checked: {exc}"
    if cb_reason and os.getenv("CHATTERBOX", "").strip().lower() not in ("0", "off", "false", "disabled"):
        hints.append(_hint(
            "no_chatterbox", "info", "Hindi voice cloning needs Chatterbox",
            "Chatterbox Multilingual (MIT licence, about 3 GB, no token needed) lets a cloned voice speak Hindi. "
            "It renders about 5x slower than real time, so it is for Studio clips, not live calls.",
            "./bonclay setup --chatterbox"))

    try:
        dh_reason = B.dhvaani_service().unavailable_reason()
    except Exception as exc:  # noqa: BLE001
        dh_reason = f"DhVaani could not be checked: {exc}"
    if dh_reason and os.getenv("DHVAANI", "").strip().lower() not in ("0", "off", "false", "disabled"):
        hints.append(_hint(
            "no_dhvaani", "info", "Voice cloning in Indian languages needs DhVaani",
            "DhVaani (ARTPARK-IISc, about 0.5 GB, needs an HF_TOKEN with access granted) lets your cloned voice speak "
            "Marathi, Tamil, Bengali, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Odia, Urdu and Hindi. Without it "
            "only Hindi can be cloned (Chatterbox) and the other languages use preset voices.",
            "./bonclay setup --dhvaani"))

    try:
        whisper_ok = bool(B.whisper_model.available)
    except Exception:  # noqa: BLE001
        whisper_ok = False
    if not whisper_ok and not has_groq:
        hints.append(_hint(
            "no_whisper", "error", "Speech recognition is off",
            "Transcripts and consent checks need Whisper (local) or a GROQ_API_KEY.",
            "backend/.venv/bin/pip install openai-whisper"))

    if not shutil.which("ffmpeg"):
        hints.append(_hint("no_ffmpeg", "error", "ffmpeg is not installed",
                           "Audio decoding and conversion need ffmpeg; uploads and generation will fail without it.",
                           _install_cmd("ffmpeg", "Gyan.FFmpeg")))
    if not shutil.which("espeak-ng"):
        hints.append(_hint("no_espeak", "warning", "espeak-ng is not installed",
                           "Kokoro's non-English preset voices need espeak-ng for pronunciation.",
                           _install_cmd("espeak-ng", "eSpeak-NG.eSpeak-NG")))

    if B.supabase is not None:
        try:
            reachable = await B._supabase_reachable()
        except Exception:  # noqa: BLE001
            reachable = False
        if not reachable:
            hints.append(_hint("supabase_unreachable", "warning", "Supabase is configured but unreachable",
                               "Cloud storage and sign-in will not work. Check your connection and the "
                               "NEXT_PUBLIC_SUPABASE_URL value, or switch the sidebar to local storage.", None))

    if not has_groq and llm_active is not None:
        hints.append(_hint("no_groq", "info", "Using the local language model",
                           "Add GROQ_API_KEY for faster, stronger answers.", "add GROQ_API_KEY to backend/.env"))

    return {"hints": hints}
