#!/usr/bin/env python3
"""C2PA spike: sign a WAV and an MP3 with a manifest (c2pa.actions + a custom BonClay receipt assertion), read it back,
check it survives re-encode, and compare PCM + size. Run in a scratch venv that has c2pa-python, numpy, soundfile,
cryptography (NOT backend/.venv).  usage: c2pa_spike.py <marked_clip.wav> <outdir>"""
import datetime, io, json, os, subprocess, sys, uuid
import numpy as np, soundfile as sf
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
import c2pa

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
now = datetime.datetime.now(datetime.timezone.utc)


def name(cn): return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn), x509.NameAttribute(NameOID.ORGANIZATION_NAME, "BonClay Test")])


# test CA + leaf (P-256). Self-signed *test* chain; untrusted by any real C2PA trust list.
ca_key = ec.generate_private_key(ec.SECP256R1())
ca = (x509.CertificateBuilder().subject_name(name("BonClay Test CA")).issuer_name(name("BonClay Test CA")).public_key(ca_key.public_key())
      .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=365))
      .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
      .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), True)
      .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), False)
      .sign(ca_key, hashes.SHA256()))
leaf_key = ec.generate_private_key(ec.SECP256R1())
leaf = (x509.CertificateBuilder().subject_name(name("BonClay Test Signer")).issuer_name(ca.subject).public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=90))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), False)
        .sign(ca_key, hashes.SHA256()))
chain = (leaf.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM)).decode()
key_pem = leaf_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()

receipt_id = str(uuid.uuid4())
manifest = {
    "claim_generator_info": [{"name": "BonClay", "version": "0.1-spike"}],
    "title": "BonClay synthetic speech",
    "assertions": [
        {"label": "c2pa.actions", "data": {"actions": [{
            "action": "c2pa.created",
            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
            "softwareAgent": {"name": "BonClay", "version": "0.1-spike"}}]}},
        {"label": "com.bonclay.receipt", "data": {"receipt_id": receipt_id, "engine": "kokoro", "voice_id": "af_heart",
                                                    "watermark": "audioseal_16bits", "verify_url": "https://example.invalid/api/receipts/" + receipt_id}},
    ],
}


def sign(path_in, path_out, mime):
    signer = c2pa.Signer.from_info(c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, chain.encode(), key_pem.encode(), None))
    with c2pa.Builder(manifest) as b, open(path_in, "rb") as s, open(path_out, "w+b") as d:
        b.sign(signer, mime, s, d)


def read(path, mime):
    try:
        with open(path, "rb") as f, c2pa.Reader(mime, f) as r:
            m = r.get_active_manifest()
            return {"state": r.get_validation_state(), "assertions": [a["label"] for a in m["assertions"]],
                    "generator": m.get("claim_generator_info"), "receipt": next(a["data"] for a in m["assertions"] if a["label"] == "com.bonclay.receipt"),
                    "actions": next(a["data"] for a in m["assertions"] if a["label"].startswith("c2pa.actions")),
                    "validation_status": [s.get("code") for s in (r.get_validation_results() or {}).get("activeManifest", {}).get("failure", [])] or None}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def pcm(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "24000", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


res = {"c2pa_python": c2pa.__name__, "sdk": c2pa.sdk_version(), "receipt_id": receipt_id,
       "wav_supported": "audio/wav" in c2pa.Builder.get_supported_mime_types(), "mp3_supported": "audio/mpeg" in c2pa.Builder.get_supported_mime_types(),
       "ogg/opus/webm_supported": [m for m in ("audio/ogg", "audio/opus", "audio/webm", "ogg", "opus", "webm") if m in c2pa.Builder.get_supported_mime_types()]}
wav_in = os.path.join(out, "in.wav"); mp3_in = os.path.join(out, "in.mp3")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", wav_in], check=True)  # 16-bit PCM like the app serves
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-c:a", "libmp3lame", "-b:a", "128k", "-write_xing", "0", mp3_in], check=True)
for tag, p_in, mime in (("wav", wav_in, "audio/wav"), ("mp3", mp3_in, "audio/mpeg")):
    p_out = os.path.join(out, "signed." + tag)
    sign(p_in, p_out, mime)
    a, b = pcm(p_in), pcm(p_out)
    n = min(len(a), len(b))
    res[tag] = {"size_in": os.path.getsize(p_in), "size_signed": os.path.getsize(p_out), "overhead_bytes": os.path.getsize(p_out) - os.path.getsize(p_in),
                "pcm_identical": bool(len(a) == len(b) and np.array_equal(a, b)), "pcm_len": [len(a), len(b)], "max_abs_diff": float(np.max(np.abs(a[:n] - b[:n]))),
                "readback": read(p_out, mime)}
# survives re-encode?
re_tests = {"wav->mp3_128k": ("signed.wav", ["-c:a", "libmp3lame", "-b:a", "128k"], "mp3", "audio/mpeg"),
            "mp3->mp3_64k": ("signed.mp3", ["-c:a", "libmp3lame", "-b:a", "64k"], "mp3", "audio/mpeg"),
            "wav->wav_ffmpeg_copy_codec": ("signed.wav", ["-c:a", "pcm_s16le"], "wav", "audio/wav"),
            "mp3->mp3_ffmpeg_stream_copy": ("signed.mp3", ["-c:a", "copy"], "mp3", "audio/mpeg"),
            "wav->opus_32k": ("signed.wav", ["-c:a", "libopus", "-b:a", "32k"], "ogg", "audio/ogg"),
            "mp3->mp3_128k_copy_with_-map_metadata_-1": ("signed.mp3", ["-c:a", "copy", "-map_metadata", "-1"], "mp3", "audio/mpeg")}
res["reencode"] = {}
for k, (f, args, ext, mime) in re_tests.items():
    o = os.path.join(out, "re_" + k.replace(">", "_").replace("/", "_") + "." + ext)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(out, f)] + args + [o], check=True)
    r = read(o, mime)
    res["reencode"][k] = "manifest survives" if "assertions" in r else r.get("error", "none")
# tamper: flip one PCM byte in signed wav
w = bytearray(open(os.path.join(out, "signed.wav"), "rb").read()); w[100] ^= 0x01
open(os.path.join(out, "tampered.wav"), "wb").write(w)
res["tampered_wav"] = read(os.path.join(out, "tampered.wav"), "audio/wav")
json.dump(res, open(os.path.join(out, "c2pa_spike.json"), "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
