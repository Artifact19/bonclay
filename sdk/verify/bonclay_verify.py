"""Verify BonClay receipts and the withdrawal log without an BonClay server.

Needs only `cryptography`. See docs/RECEIPT_SPEC.md for the format this checks.

    from bonclay_verify import verify_receipt, verify_audio_sha256, verify_withdrawal_log

    key = {"public_key": "<base64 from /.well-known/bonclay-receipt-key>"}
    verify_receipt(receipt, key["public_key"])          # True / False
    verify_audio_sha256(receipt, open("clip.wav", "rb").read())
    verify_withdrawal_log(rows, key["public_key"])      # {"ok": ..., "broken_at": ..., ...}

What this proves: the receipt was signed by the holder of that key, and the file is the one it describes. It does not
prove the key belongs to who you think (get the key from the operator over a channel you trust) and it cannot tell you
that a voice was withdrawn after the log snapshot you hold.
"""
import base64
import hashlib
import json
from typing import Any, Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

SUPPORTED_VERSIONS = (1,)
LOG_GENESIS = "0" * 64
SIGNED_FIELDS = ("version", "receipt_id", "kind", "generation_id", "turn_id", "created_at", "engine", "voice_id",
                 "voice_name", "consent_id", "consent_verified", "text_sha256", "audio_sha256", "watermark_message")


def canonical(receipt: Dict[str, Any], fields=SIGNED_FIELDS) -> bytes:
    """The bytes that were signed: the signed fields (missing ones as null), keys sorted, no spaces, UTF-8."""
    payload = {k: receipt.get(k) for k in fields}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _check(public_b64: str, signature_b64: Any, data: bytes) -> bool:
    if not isinstance(signature_b64, str):
        return False
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(public_b64)).verify(base64.b64decode(signature_b64), data)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def key_id(public_b64: str) -> str:
    """First 16 hex characters of the sha256 of the raw public key (what `key_id` fields in receipts carry)."""
    return hashlib.sha256(base64.b64decode(public_b64)).hexdigest()[:16]


def verify_receipt(receipt: Dict[str, Any], public_b64: str, fields=SIGNED_FIELDS) -> bool:
    """True when the receipt's Ed25519 signature is valid for `public_b64` and its version is one this SDK knows."""
    if not receipt or receipt.get("version") not in SUPPORTED_VERSIONS:
        return False
    return _check(public_b64, receipt.get("signature_ed25519"), canonical(receipt, fields))


def verify_audio_sha256(receipt: Dict[str, Any], audio_bytes: bytes) -> bool:
    """True when these exact bytes are the audio the receipt describes (a re-encoded copy will not match; use the
    watermark check on the server for that)."""
    expected = receipt.get("audio_sha256")
    return bool(expected) and hashlib.sha256(audio_bytes).hexdigest() == expected


def _row_hash(row: Dict[str, Any], prev: str) -> str:
    body = {"voice_id": row.get("voice_id"), "withdrawn_at": row.get("withdrawn_at"), "reason": row.get("reason"),
            "prev": prev}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_withdrawal_row(row: Dict[str, Any], public_b64: str) -> bool:
    if not row.get("row_hash") or not row.get("sig"):
        return False
    if _row_hash(row, str(row.get("prev"))) != row["row_hash"]:
        return False
    return _check(public_b64, row["sig"], str(row["row_hash"]).encode("ascii"))


def verify_withdrawal_log(rows: List[Dict[str, Any]], public_b64: str) -> Dict[str, Any]:
    """Walk the published log (`rows` from /api/withdrawals). Each signed row must hash and verify and its `prev` must be
    the hash of the signed row before it. Unsigned legacy rows are counted and skipped."""
    prev, signed, legacy, broken = LOG_GENESIS, 0, 0, None
    for i, row in enumerate(rows):
        if not row.get("row_hash"):
            legacy += 1
            continue
        if row.get("prev") != prev or not verify_withdrawal_row(row, public_b64):
            broken = i
            break
        prev = row["row_hash"]
        signed += 1
    return {"ok": broken is None, "rows": len(rows), "signed_rows": signed, "legacy_rows": legacy,
            "broken_at": broken, "head": prev}


def voice_withdrawn(receipt: Dict[str, Any], rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The log row that withdrew the receipt's voice in `rows`, or None. Only as current as the rows you hold."""
    vid = receipt.get("voice_id")
    for row in rows:
        if vid and row.get("voice_id") == vid:
            return row
    return None
