"""C2PA manifests on exported clips: a second, standard layer of provenance beside the AudioSeal watermark.

A C2PA manifest is a signed statement embedded in the file (WAV and MP3 are supported by c2pa-python, Apache-2.0 OR
MIT) that any C2PA reader (c2patool, Content Credentials) can show: who made the file, that it is AI-generated
("trainedAlgorithmicMedia"), and, in BonClay's own assertion `com.bonclay.receipt`, the receipt id, the signing key id
and where to verify it. The EU Code of Practice of June 2026 expects at least two layers (signed metadata plus a
watermark); India's IT Rules amendment asks for a unique identifier in metadata. Both are counsel questions; this is
the mechanism.

Behind a flag: C2PA_MANIFEST=on (default off). Needs the c2pa-python package in backend/.venv; without it the flag
does nothing and `unavailable_reason()` says so. The manifest is embedded after the watermark and the RIFF label and
before the receipt is written, so the receipt's audio_sha256 is the hash of the file as delivered and the exact match on
/verify still works. The samples are untouched: the manifest is metadata.

Certificates: C2PA_CERT_FILE and C2PA_KEY_FILE (PEM: the signing certificate followed by its chain, and a PKCS#8 key)
for a real signing certificate; without them a self-signed test chain is made once in data/c2pa/ (key mode 600). A
C2PA reader shows the test chain as an untrusted signer, which is honest: it proves integrity, not identity, until a
certificate from a recognised authority replaces it."""
from __future__ import annotations

import datetime
import io
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("bonclay.c2pa")

_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GENERATOR = {"name": "BonClay", "version": "2.0.0"}
RECEIPT_LABEL = "com.bonclay.receipt"
SOURCE_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


def enabled() -> bool:
    return os.getenv("C2PA_MANIFEST", "off").strip().lower() in ("on", "1", "true", "yes")


def _import_c2pa():
    import c2pa  # noqa: PLC0415  (optional dependency)
    return c2pa


def unavailable_reason() -> Optional[str]:
    """None when manifests can be written here, else why not."""
    try:
        _import_c2pa()
    except ImportError:
        return "c2pa-python is not installed (backend/.venv/bin/pip install c2pa-python)"
    return None


def active() -> bool:
    """The flag is on and the library is present."""
    return enabled() and unavailable_reason() is None


def status() -> Dict[str, Any]:
    reason = unavailable_reason()
    return {"enabled": enabled(), "available": reason is None, "reason": reason,
            "signer": "file" if os.getenv("C2PA_CERT_FILE") else "self-signed test chain"}


# ---- signer ------------------------------------------------------------------------------------------------------

def _cert_paths() -> Tuple[str, str]:
    cert = os.getenv("C2PA_CERT_FILE", "").strip() or os.path.join(_data_dir, "c2pa", "signer.pem")
    key = os.getenv("C2PA_KEY_FILE", "").strip() or os.path.join(_data_dir, "c2pa", "signer.key")
    return cert, key


def _test_chain() -> Tuple[bytes, bytes]:
    """A self-signed CA plus a signing certificate (P-256, ES256), like bench/c2pa_spike.py: integrity, not identity."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.timezone.utc)

    def name(cn: str):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn),
                          x509.NameAttribute(NameOID.ORGANIZATION_NAME, "BonClay (self-signed)")])

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder().subject_name(name("BonClay Self-Signed CA")).issuer_name(name("BonClay Self-Signed CA"))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=3650))
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
          .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), False)
          .sign(ca_key, hashes.SHA256()))
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (x509.CertificateBuilder().subject_name(name("BonClay Signer")).issuer_name(ca.subject)
            .public_key(leaf_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
            .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), False)
            .sign(ca_key, hashes.SHA256()))
    chain = leaf.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM)
    key = leaf_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption())
    return chain, key


def _leaf_expired(chain_pem: bytes) -> bool:
    try:
        from cryptography import x509
        leaf = x509.load_pem_x509_certificates(chain_pem)[0]
        return leaf.not_valid_after_utc <= datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    except Exception:  # noqa: BLE001
        return True


def signer_material() -> Tuple[bytes, bytes]:
    """(certificate chain PEM, private key PEM): the configured files, else the self-signed test chain, made on first
    use (and again when its signing certificate is within a day of expiry) under data/c2pa/."""
    cert_path, key_path = _cert_paths()
    try:
        with open(cert_path, "rb") as f:
            chain = f.read()
        with open(key_path, "rb") as f:
            key = f.read()
        if os.getenv("C2PA_CERT_FILE") or not _leaf_expired(chain):
            return chain, key
    except OSError:
        if os.getenv("C2PA_CERT_FILE") or os.getenv("C2PA_KEY_FILE"):
            raise
    chain, key = _test_chain()
    os.makedirs(os.path.dirname(cert_path), exist_ok=True)
    with open(cert_path, "wb") as f:
        f.write(chain)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    log.info("made a self-signed C2PA test signing chain in %s (set C2PA_CERT_FILE for a real certificate)",
             os.path.dirname(cert_path))
    return chain, key


# ---- the manifest --------------------------------------------------------------------------------------------------

def manifest(receipt_id: str, *, engine: Optional[str], voice_id: Optional[str], key_id: Optional[str],
             watermark: Optional[str], verify_base: Optional[str] = None) -> Dict[str, Any]:
    """The manifest JSON: created by a trained algorithm (BonClay), plus the receipt assertion a verifier can follow."""
    base = (verify_base or os.getenv("BONCLAY_PUBLIC_URL", "")).rstrip("/")
    receipt = {"receipt_id": receipt_id, "engine": engine, "voice_id": voice_id, "receipt_key_id": key_id,
               "watermark": watermark}
    if base:
        receipt["verify_url"] = f"{base}/api/receipts/{receipt_id}"
        receipt["keys_url"] = f"{base}/.well-known/bonclay-receipt-key"
    return {
        "claim_generator_info": [dict(GENERATOR)],
        "title": "BonClay synthetic speech",
        "assertions": [
            {"label": "c2pa.actions", "data": {"actions": [{
                "action": "c2pa.created", "digitalSourceType": SOURCE_TYPE, "softwareAgent": dict(GENERATOR)}]}},
            {"label": RECEIPT_LABEL, "data": {k: v for k, v in receipt.items() if v is not None}},
        ],
    }


def sniff_mime(data: bytes) -> Optional[str]:
    head = bytes(data[:12])
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio/wav"
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    return None


def embed(data: bytes, receipt_id: str, *, engine: Optional[str] = None, voice_id: Optional[str] = None,
          key_id: Optional[str] = None, watermark: Optional[str] = None, mime: Optional[str] = None,
          verify_base: Optional[str] = None) -> bytes:
    """The file with a signed manifest inside. Raises when the library is missing, the type is unsupported or signing
    fails; callers keep the original bytes then. The audio samples are byte for byte the same."""
    c2pa = _import_c2pa()
    mime = mime or sniff_mime(data)
    if mime is None:
        raise ValueError("not a WAV or MP3 file")
    chain, key = signer_material()
    signer = c2pa.Signer.from_info(c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, chain, key, None))
    body = manifest(receipt_id, engine=engine, voice_id=voice_id, key_id=key_id, watermark=watermark,
                    verify_base=verify_base)
    src, dst = io.BytesIO(bytes(data)), io.BytesIO()
    with c2pa.Builder(json.dumps(body)) as builder:
        builder.sign(signer, mime, src, dst)
    return dst.getvalue()


def read_summary(data: bytes, mime: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """What /verify says about a manifest in an uploaded file: None when the library is missing or the type is not
    one C2PA covers; {"present": False} when the file carries none; else the generator, the validation state, the
    assertion labels and the receipt assertion's fields."""
    try:
        c2pa = _import_c2pa()
    except ImportError:
        return None
    mime = mime or sniff_mime(data)
    if mime is None:
        return None
    try:
        with c2pa.Reader(mime, io.BytesIO(bytes(data))) as reader:
            active_manifest = reader.get_active_manifest()
            if not active_manifest:
                return {"present": False}
            assertions = active_manifest.get("assertions") or []
            receipt = next((a.get("data") for a in assertions if a.get("label") == RECEIPT_LABEL), None)
            state = reader.get_validation_state()
            failures = []
            try:
                results = reader.get_validation_results() or {}
                failures = [s.get("code") for s in (results.get("activeManifest") or {}).get("failure", []) or []]
            except Exception:  # noqa: BLE001
                pass
            generator = (active_manifest.get("claim_generator_info") or [{}])[0].get("name")
            return {"present": True, "validation_state": str(state), "generator": generator,
                    "assertions": [a.get("label") for a in assertions], "receipt": receipt,
                    "failures": failures or None}
    except Exception as e:  # noqa: BLE001  - no manifest, or a file the reader cannot parse
        text = str(e).lower()
        if "no manifest" in text or "jumbf" in text or "not found" in text:
            return {"present": False}
        return {"present": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
