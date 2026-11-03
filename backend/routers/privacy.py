"""GET /api/privacy/attestation: a signed statement of how this server is configured to handle data.

It attests CONFIGURATION (is offline mode on, which model and speech-recognition providers can run, which outbound
integrations are switched on), not network traffic: it cannot prove that no packet left the machine, only what the
software is set to allow. It is signed with the same Ed25519 key as receipts (see docs/RECEIPT_SPEC.md), so anyone can
check it with the key at /.well-known/bonclay-receipt-key."""
import json
import os
import platform
from datetime import datetime, timezone

from fastapi import APIRouter, Request

try:
    from core import offline
    from core.errors import AppError
    from services import provenance
except ImportError:  # pragma: no cover - imported as backend.routers.privacy
    from backend.core import offline
    from backend.core.errors import AppError
    from backend.services import provenance

router = APIRouter()

STATEMENT_VERSION = 1


def _b():
    import sys
    mod = sys.modules.get("backend") or sys.modules.get("__main__")
    if mod is None or not hasattr(mod, "app"):
        import backend as mod  # noqa: PLC0415
    return mod


def _llm_state() -> dict:
    try:
        from services import llm
    except ImportError:  # pragma: no cover
        from backend.services import llm
    usable, skipped = [], {}
    for name in llm._order():
        reason = llm.PROVIDERS[name][0]()
        if reason:
            skipped[name] = reason
        else:
            usable.append(name)
    remote = [n for n in usable if n == "groq" or (n == "openai" and not offline.is_local_url(os.getenv("LLM_BASE_URL")))]
    return {"usable": usable, "remote_usable": remote, "unavailable": skipped}


def build_statement() -> dict:
    b = _b()
    llm_state = _llm_state()
    asr_remote = getattr(b, "groq_client", None) is not None
    gcal_on = False
    try:
        from services import gcal
        gcal_on = bool(gcal.configured())
    except Exception:  # noqa: BLE001
        pass
    storage_remote = getattr(b, "supabase", None) is not None
    flags = {
        "offline_mode": offline.enabled(),
        "huggingface_downloads_blocked": os.getenv("HF_HUB_OFFLINE", "") in ("1", "true", "True"),
        "language_model_remote_possible": bool(llm_state["remote_usable"]),
        "speech_recognition_remote_possible": asr_remote,
        "cloud_storage_possible": storage_remote,
        "calendar_possible": gcal_on,
    }
    # "No audio or text leaves this machine" is claimed only when every remote path above is off.
    local_only = (flags["offline_mode"] and not flags["language_model_remote_possible"]
                  and not flags["speech_recognition_remote_possible"] and not flags["cloud_storage_possible"]
                  and not flags["calendar_possible"])
    return {
        "statement_version": STATEMENT_VERSION,
        "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim": "configured_to_keep_data_on_this_machine" if local_only else "not_offline",
        "scope": "configuration only: this states what the software is set to allow, it does not inspect network packets",
        "settings": flags,
        "language_model": llm_state,
        "host": {"platform": platform.system(), "machine": platform.machine()},
        "blocked_when_offline": ["Groq (language model and speech recognition)", "language-model servers not on this machine",
                                 "Supabase", "outbound webhooks and Google Sheets", "Google Calendar",
                                 "Exotel phone calls", "WhatsApp", "Hugging Face downloads"],
    }


@router.get("/api/privacy/attestation")
async def attestation(request: Request):
    """The statement, an Ed25519 signature over its canonical JSON (sorted keys, no spaces, UTF-8) and the key id."""
    statement = build_statement()
    body = json.dumps(statement, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    signature = provenance._sign_bytes(body)
    if signature is None:
        raise AppError("NOT_AVAILABLE", "This server cannot sign statements (no Ed25519 key)", 404)
    return {"statement": statement, "signature_ed25519": signature, "key_id": provenance.key_id(),
            "public_key": provenance.public_key_b64()}
