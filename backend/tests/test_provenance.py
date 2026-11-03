"""Provenance: watermark + signed receipt for every generated clip, and the public verify endpoint.

AudioSeal is replaced by deterministic fakes (fast, no weights); one test at the bottom runs the real model."""
import asyncio
import base64
import io
import os

import numpy as np
import pytest
import soundfile as sf

from services import provenance
from test_offline_agents import _turn, _turn_setup, env, make_agent  # noqa: F401  (env is a fixture)

SR = 24000
PILOT_HZ = 7000.0
BIT_HZ = [3000.0 + 250.0 * k for k in range(16)]
AMP = 0.004


def _tone(seconds=2.0, sr=SR, freq=440.0, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _wav(x, sr=SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


def _fake_embed16(x16, message):
    """Pilot tone plus one tone per set bit, far below the speech level (stand-in for AudioSeal's generator). Its level
    is deliberately NOT proportional to the input (it grows as level^0.6 like the real generator), so the tests exercise
    the per-clip scaling in watermark()."""
    t = np.arange(x16.size) / provenance.MODEL_SR
    amp = AMP * (max(float(np.sqrt(np.mean(np.square(x16)))), 1e-6) / 0.2) ** 0.6
    wm = amp * np.sin(2 * np.pi * PILOT_HZ * t)
    for k, bit in enumerate(provenance._bits(message)):
        if bit:
            wm = wm + amp * np.sin(2 * np.pi * BIT_HZ[k] * t)
    return wm.astype(np.float32)


def _fake_detect16(x16):
    """Tones above 0.6% of the clip's own level count (the scaled watermark is about 4-9% of it per tone)."""
    t = np.arange(x16.size) / provenance.MODEL_SR
    level = lambda hz: 2 * abs(np.mean(x16 * np.exp(-2j * np.pi * hz * t)))  # noqa: E731
    floor = 0.006 * float(np.sqrt(np.mean(np.square(x16))))
    if level(PILOT_HZ) < floor:
        return 0.0, None
    return 1.0, provenance._from_bits([1 if level(hz) > floor else 0 for hz in BIT_HZ])


@pytest.fixture()
def fake_seal(monkeypatch):
    monkeypatch.delenv("WATERMARK", raising=False)
    monkeypatch.setenv("BONCLAY_RECEIPT_SECRET", "unit-test-secret")
    monkeypatch.setattr(provenance, "_engine_reason", lambda: None)
    monkeypatch.setattr(provenance, "_embed16", _fake_embed16)
    monkeypatch.setattr(provenance, "_detect16", _fake_detect16)
    monkeypatch.setattr(provenance, "_get_supabase", lambda: None)


@pytest.fixture()
def backend_mod(app_client, monkeypatch, tmp_path, fake_seal):
    import backend
    from core import generations

    monkeypatch.setattr(backend, "LOCAL_AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(backend, "supabase", None)
    monkeypatch.setattr(generations, "_data_dir", str(tmp_path))
    backend.limiter.reset()
    return backend


def _generate(backend, x=None, **kw):
    x = _tone() if x is None else x
    kw.setdefault("text", "hello there")
    return asyncio.run(backend.persist_generation(
        "local-user", "local", _wav(x), len(x) / SR, engine="kokoro", language="en", voice_name="af_heart", **kw))


def _verify(client, data: bytes, name="clip.wav"):
    return client.post("/api/verify", files={"file": (name, data, "audio/wav")})


# ---- receipts ---------------------------------------------------------------------------------------------------

def test_a_generation_gets_a_watermark_and_a_signed_receipt(backend_mod, tmp_path):
    original = _wav(_tone())
    gen = _generate(backend_mod)
    marked = gen.wav_bytes
    assert marked != original

    receipt = provenance.get_receipt(gen["receipt_id"])
    assert receipt["kind"] == "generation" and receipt["generation_id"] == gen["id"]
    assert receipt["engine"] == "kokoro" and receipt["voice_name"] == "af_heart"
    assert receipt["text_sha256"] == provenance.sha256_hex(b"hello there")
    assert receipt["audio_sha256"] == provenance.sha256_hex(marked)          # hash of the watermarked file
    assert receipt["watermark_message"] == provenance.message_for(gen["id"])
    assert provenance.signature_valid(receipt)
    assert (tmp_path / f"gen_{gen['id']}.wav").read_bytes() == marked        # what is stored is what was marked

    audio, sr = sf.read(io.BytesIO(marked), dtype="float32")
    assert provenance.detect(audio, sr)["message"] == receipt["watermark_message"]


def test_a_cloned_voice_receipt_carries_its_consent(backend_mod, monkeypatch):
    async def lookup(voice_id, storage_mode, user_id=None):
        return {"id": voice_id, "display_name": "Asha", "consent_id": "consent-1", "consent_verified": True}

    monkeypatch.setattr(backend_mod, "_lookup_voice", lookup)
    gen = _generate(backend_mod, voice_id="voice-1")
    receipt = provenance.get_receipt(gen["receipt_id"])
    assert receipt["voice_id"] == "voice-1"
    assert receipt["consent_id"] == "consent-1" and receipt["consent_verified"] is True


def test_wav_responses_carry_the_receipt_id_and_the_watermarked_audio(backend_mod):
    gen = _generate(backend_mod)
    resp = backend_mod._wav_response(_wav(_tone()), "x.wav", gen)
    assert resp.headers["x-receipt-id"] == gen["receipt_id"]
    body = b"".join(asyncio.run(_collect(resp.body_iterator)))
    assert body == gen.wav_bytes


async def _collect(it):
    return [chunk async for chunk in it]


def test_a_tampered_receipt_fails_its_signature(backend_mod):
    receipt = provenance.get_receipt(_generate(backend_mod)["receipt_id"])
    assert provenance.signature_valid(receipt)
    for field, value in (("consent_verified", True), ("voice_name", "Someone Else"), ("engine", "other"),
                         ("audio_sha256", "0" * 64)):
        assert not provenance.signature_valid({**receipt, field: value}), field
    assert not provenance.signature_valid({**receipt, "signature": "0" * 64})
    assert not provenance.signature_valid({k: v for k, v in receipt.items() if k != "signature"})


def test_a_different_secret_does_not_validate_old_receipts(backend_mod, monkeypatch):
    receipt = provenance.get_receipt(_generate(backend_mod)["receipt_id"])
    monkeypatch.setenv("BONCLAY_RECEIPT_SECRET", "another-secret")
    assert not provenance.signature_valid(receipt)


def test_the_secret_is_generated_once_and_kept(monkeypatch, tmp_path):
    monkeypatch.delenv("BONCLAY_RECEIPT_SECRET", raising=False)
    monkeypatch.setattr(provenance, "_data_dir", str(tmp_path))
    monkeypatch.setattr(provenance, "_secret_cache", None)
    first = provenance._secret()
    assert (tmp_path / "receipt_secret").read_text().strip() == first.decode()
    monkeypatch.setattr(provenance, "_secret_cache", None)
    assert provenance._secret() == first
    if os.name != "nt":
        assert oct((tmp_path / "receipt_secret").stat().st_mode & 0o777) == "0o600"


def test_watermark_off_skips_the_watermark_but_still_issues_a_receipt(backend_mod, monkeypatch):
    monkeypatch.setenv("WATERMARK", "off")
    original = _wav(_tone())
    gen = _generate(backend_mod)
    # unmarked audio is delivered sample for sample, plus the AI label in the file's metadata
    assert gen.wav_bytes == provenance.label_wav(original, gen["receipt_id"]) and gen.wav_bytes != original
    receipt = provenance.get_receipt(gen["receipt_id"])
    assert receipt["watermark_message"] is None and provenance.signature_valid(receipt)
    out = provenance.detect(_tone(), SR)
    assert out["detected"] is False and "off" in out["reason"]
    assert np.array_equal(provenance.watermark(_tone(), SR, 5), _tone())


def test_a_missing_audioseal_never_breaks_generation(backend_mod, monkeypatch):
    monkeypatch.setattr(provenance, "_engine_reason", lambda: "AudioSeal unavailable: not installed")
    gen = _generate(backend_mod)
    assert gen.wav_bytes == provenance.label_wav(_wav(_tone()), gen["receipt_id"])
    assert provenance.detect(_tone(), SR)["detected"] is False


def test_the_watermark_is_quiet(fake_seal):
    x = _tone()
    y = provenance.watermark(x, SR, 0xBEEF)
    snr = 10 * np.log10(np.sum(x ** 2) / np.sum((y - x) ** 2))
    assert snr > 25 and y.shape == x.shape and y.dtype == np.float32


def _speechlike(seconds=3.0, level=0.1, seed=0):
    """Noise bursts shaped like syllables (about 4 per second) with gaps, at a given active-frame RMS."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    env = np.clip(np.sin(2 * np.pi * 4 * np.arange(n) / SR), 0, None) ** 2
    x = rng.standard_normal(n) * env
    return (x * level / provenance._speech_rms(x.astype(np.float32))).astype(np.float32)


def _snr16(x, y):
    """Active-speech to watermark ratio in dB, measured where the models work (16 kHz), as watermark() does."""
    x16, w16 = (provenance._resample(a, SR, provenance.MODEL_SR) for a in (x, y - x))
    return float(20 * np.log10(provenance._speech_rms(x16) / np.sqrt(np.mean(w16 ** 2))))


@pytest.mark.parametrize("level", [0.01, 0.05, 0.1, 0.15])
def test_the_watermark_to_speech_ratio_is_the_same_at_every_level(fake_seal, level):
    x = _speechlike(level=level)
    y = provenance.watermark(x, SR, 0xBEEF)
    assert abs(_snr16(x, y) - provenance.TARGET_WM_SNR_DB) < 1.0


def test_a_louder_voice_is_not_marked_any_weaker(fake_seal):
    quiet, loud = _speechlike(level=0.05), _speechlike(level=0.14)   # the English / Hindi Kokoro levels
    assert abs(_snr16(quiet, provenance.watermark(quiet, SR, 7)) - _snr16(loud, provenance.watermark(loud, SR, 7))) < 0.5


def test_silence_and_the_scale_cap_are_safe(fake_seal):
    z = np.zeros(SR * 2, dtype=np.float32)
    assert np.max(np.abs(provenance.watermark(z, SR, 5))) < 1e-4             # nothing to scale against: the raw mark, tiny
    raw = np.ones(1000, dtype=np.float32) * 1e-4
    assert provenance._scaled_watermark(np.ones(5000, np.float32) * 1e-9, raw, np.ones(5000, np.float32) * 1e-9) is raw
    big = np.ones(1000, dtype=np.float32) * 1e-9                              # a near-zero mark is capped, not blown up
    out = provenance._scaled_watermark(_speechlike(), big, _speechlike())
    assert np.max(np.abs(out)) <= 1e-9 * provenance.MAX_SCALE * 1.001


def test_the_snr_can_be_tuned_by_env(fake_seal, monkeypatch):
    x = _speechlike()
    monkeypatch.setenv("WATERMARK_SNR_DB", "20")
    d20 = np.std(provenance.watermark(x, SR, 7) - x)
    monkeypatch.setenv("WATERMARK_SNR_DB", "junk")
    assert provenance.target_snr_db() == provenance.TARGET_WM_SNR_DB
    monkeypatch.delenv("WATERMARK_SNR_DB")
    assert d20 > 1.5 * np.std(provenance.watermark(x, SR, 7) - x)


def test_detection_rescales_when_the_raw_score_is_not_convincing(monkeypatch):
    """A piece is scored as it is, and again at the reference level only when the first score is under 0.9; the higher wins."""
    seen = []

    def fake(x16):
        seen.append(float(provenance._speech_rms(x16)))
        return (0.3, 1) if len(seen) == 1 else (0.7, 2)

    monkeypatch.setattr(provenance, "_detect16", fake)
    loud = _speechlike(level=0.3)
    prob, msg = provenance._detect_piece(provenance._resample(loud, SR, provenance.MODEL_SR))
    assert (prob, msg) == (0.7, 2) and len(seen) == 2
    assert abs(seen[1] - provenance.DETECT_SPEECH_RMS) < 0.002

    seen.clear()
    monkeypatch.setattr(provenance, "_detect16", lambda x: (seen.append(1) or (0.95, 9)))   # convincing: scored once
    assert provenance._detect_piece(provenance._resample(loud, SR, provenance.MODEL_SR)) == (0.95, 9) and len(seen) == 1
    seen.clear()
    monkeypatch.setattr(provenance, "_detect16", lambda x: (seen.append(1) or (0.1, None)))  # silence: scored once
    assert provenance._detect_piece(np.zeros(32000, np.float32)) == (0.1, None) and len(seen) == 1


def test_message_round_trips_at_every_level(fake_seal):
    for level in (0.01, 0.1, 0.4):
        for message in (0x0000, 0xBEEF, 0xFFFF, 0x1234):
            y = provenance.watermark(_tone(amp=level), SR, message)
            out = provenance.detect(y, SR)
            assert out["detected"] and out["message"] == message


def test_an_agent_reply_is_watermarked_and_gets_a_receipt(app_client, env, fake_seal, monkeypatch, ref_wav):
    _turn_setup(env, monkeypatch)
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": _tone(1.0))
    a = make_agent(app_client)
    body = _turn(app_client, a["id"], ref_wav.read_bytes())
    receipt = provenance.get_receipt(body["receipt_id"])
    assert receipt["kind"] == "agent_reply" and receipt["turn_id"] and receipt["generation_id"] is None
    assert receipt["engine"] == "kokoro" and receipt["consent_id"] is None
    assert receipt["text_sha256"] == provenance.sha256_hex(body["agent_text"].encode("utf-8"))
    assert provenance.signature_valid(receipt)

    served = app_client.get(body["agent_audio_url"], headers={"storage-mode": "local"}).content
    assert provenance.sha256_hex(served) == receipt["audio_sha256"]           # the stored reply is the marked one
    audio, sr = sf.read(io.BytesIO(served), dtype="float32")
    assert provenance.detect(audio, sr)["message"] == provenance.message_for(receipt["turn_id"])


# ---- /api/verify ------------------------------------------------------------------------------------------------

def test_verify_an_exact_copy_is_made_with_bonclay(app_client, backend_mod):
    gen = _generate(backend_mod)
    r = _verify(app_client, gen.wav_bytes)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "made_with_bonclay"
    assert body["watermark"]["detected"] is True
    assert body["match"]["exact"] is True and body["match"]["signature_valid"] is True
    receipt = body["match"]["receipt"]
    assert receipt["receipt_id"] == gen["receipt_id"] and receipt["voice_name"] == "af_heart"
    assert "user_id" not in receipt and receipt["signature_valid"] is True


def test_verify_a_resampled_copy_is_found_by_its_watermark(app_client, backend_mod):
    gen = _generate(backend_mod)
    audio, sr = sf.read(io.BytesIO(gen.wav_bytes), dtype="float32")
    copy = _wav(provenance._resample(audio, sr, 16000), 16000)      # different bytes, same watermark
    body = _verify(app_client, copy).json()
    assert body["match"]["exact"] is False
    assert body["verdict"] == "made_with_bonclay"
    assert body["match"]["receipt"]["receipt_id"] == gen["receipt_id"]


def test_verify_unknown_audio_is_not_detected(app_client, backend_mod):
    body = _verify(app_client, _wav(_tone(freq=330.0))).json()
    assert body["verdict"] == "not_detected"
    assert body["watermark"]["detected"] is False
    assert body["match"] == {"exact": False, "receipt": None, "signature_valid": False, "candidates": []}


def test_verify_watermark_without_a_receipt_is_watermark_only(app_client, backend_mod):
    marked = provenance.watermark(_tone(), SR, 4242)                # watermarked elsewhere: no receipt on this server
    body = _verify(app_client, _wav(marked)).json()
    assert body["verdict"] == "watermark_only"
    assert body["match"]["receipt"] is None


def test_verify_with_a_colliding_message_lists_candidates(app_client, backend_mod, monkeypatch):
    monkeypatch.setattr(provenance, "message_for", lambda _id: 777)  # every generation gets the same 16 bits
    a, b = _generate(backend_mod), _generate(backend_mod)
    audio, sr = sf.read(io.BytesIO(a.wav_bytes), dtype="float32")
    body = _verify(app_client, _wav(provenance._resample(audio, sr, 16000), 16000)).json()
    assert body["match"]["receipt"] is None and body["verdict"] == "made_with_bonclay"
    assert {c["receipt_id"] for c in body["match"]["candidates"]} == {a["receipt_id"], b["receipt_id"]}


def test_verify_rejects_bad_uploads(app_client, backend_mod):
    r = app_client.post("/api/verify", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = app_client.post("/api/verify", files={"file": ("clip.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 415
    r = app_client.post("/api/verify", files={"file": ("clip.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_verify_needs_no_sign_in(app_client, backend_mod, monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "secret")
    r = _verify(app_client, _wav(_tone()))                           # no Authorization header at all
    assert r.status_code == 200


def test_get_receipt_is_public_and_reports_the_signature(app_client, backend_mod):
    gen = _generate(backend_mod)
    r = app_client.get(f"/api/receipts/{gen['receipt_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["signature_valid"] is True and body["generation_id"] == gen["id"] and "user_id" not in body
    assert app_client.get("/api/receipts/00000000-0000-0000-0000-000000000000").status_code == 404
    assert app_client.get("/api/receipts/bad%20id").status_code == 422


# ---- the real model ---------------------------------------------------------------------------------------------

def test_real_audioseal_round_trip(monkeypatch):
    """A 3 s tone watermarked at 24 kHz is detected with high confidence and the 16 message bits come back."""
    pytest.importorskip("audioseal")
    monkeypatch.delenv("WATERMARK", raising=False)
    reason = provenance._engine_reason()
    if reason:
        pytest.skip(reason)
    x = _tone(seconds=3.0)
    message = provenance.message_for("round-trip-test-id")
    y = provenance.watermark(x, SR, message)
    assert not np.array_equal(x, y)
    snr = 10 * np.log10(np.sum(x ** 2) / np.sum((y - x) ** 2))
    assert snr > 25
    out = provenance.detect(y, SR)
    assert out["detected"] is True and out["probability"] > 0.9
    assert out["message"] == message
    assert provenance.detect(x, SR)["detected"] is False


def _legacy_watermark(x, sr, message):
    """The embed path before per-clip scaling: the generator's raw output added to the audio, as shipped until 2026-10-15.
    Clips made that way are still in the wild (and in receipts), so detection must keep verifying them."""
    x16 = provenance._resample(x.astype(np.float32), sr, provenance.MODEL_SR)
    wm = provenance._resample(provenance._embed16(x16, message), provenance.MODEL_SR, sr)
    return np.clip(x + wm[:x.size], -1.0, 1.0).astype(np.float32)


def test_a_clip_marked_by_the_old_path_still_verifies(fake_seal):
    x = _tone(amp=0.12)
    old = _legacy_watermark(x, SR, 0xA5A5)
    assert not np.array_equal(old, provenance.watermark(x, SR, 0xA5A5))     # the new path really differs
    out = provenance.detect(old, SR)
    assert out["detected"] is True and out["message"] == 0xA5A5


def test_real_audioseal_old_clips_verify_and_new_clips_are_level_independent(monkeypatch):
    """Real model: speech-like audio at the English and Hindi Kokoro levels. The old path leaves the louder clip's mark
    ~8 dB weaker; the new path puts both at the target; both old and new clips are detected with the right message."""
    pytest.importorskip("audioseal")
    monkeypatch.delenv("WATERMARK", raising=False)
    reason = provenance._engine_reason()
    if reason:
        pytest.skip(reason)
    snrs = {}
    for name, level in (("quiet", 0.05), ("loud", 0.14)):
        x = _speechlike(seconds=4.0, level=level, seed=3) + _tone(seconds=4.0, freq=180.0, amp=level * 0.5)
        message = provenance.message_for(f"level-{name}")
        new = provenance.watermark(x, SR, message)
        old = _legacy_watermark(x, SR, message)
        snrs[name] = _snr16(x, new)
        assert abs(snrs[name] - provenance.TARGET_WM_SNR_DB) < 1.5
        for clip in (old, new):
            out = provenance.detect(clip, SR)
            assert out["detected"] is True and out["message"] == message
        assert provenance.detect(x, SR)["detected"] is False
    assert abs(snrs["quiet"] - snrs["loud"]) < 1.5


# ---- Ed25519 receipts and the signed withdrawal log -----------------------------------------------------------------

def test_a_receipt_has_an_ed25519_signature_anyone_can_check(app_client, backend_mod):
    """Verified from the public key alone: no HMAC secret, no server call."""
    import base64
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    gen = _generate(backend_mod)
    body = app_client.get(f"/api/receipts/{gen['receipt_id']}").json()
    key = app_client.get("/.well-known/bonclay-receipt-key").json()
    assert key["alg"] == "Ed25519" and key["key_id"] == body["key_id"]
    assert body["signature_ed25519_valid"] is True
    import json
    canonical = json.dumps({k: body.get(k) for k in key["signed_fields"]}, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    Ed25519PublicKey.from_public_bytes(base64.b64decode(key["public_key"])).verify(
        base64.b64decode(body["signature_ed25519"]), canonical)  # raises if invalid


def test_a_tampered_receipt_fails_its_ed25519_signature(backend_mod):
    receipt = provenance.get_receipt(_generate(backend_mod)["receipt_id"])
    assert provenance.signature_ed25519_valid(receipt) is True
    for field, value in (("consent_verified", True), ("voice_name", "Someone Else"), ("audio_sha256", "0" * 64)):
        assert provenance.signature_ed25519_valid({**receipt, field: value}) is False, field
    assert provenance.signature_ed25519_valid({k: v for k, v in receipt.items() if k != "signature_ed25519"}) is None


def test_another_key_does_not_validate_a_receipt(backend_mod, monkeypatch):
    import base64
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    receipt = provenance.get_receipt(_generate(backend_mod)["receipt_id"])
    other = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    assert provenance.signature_ed25519_valid(receipt, base64.b64encode(other).decode()) is False


def test_the_signing_key_is_generated_once_and_kept(monkeypatch, tmp_path):
    monkeypatch.delenv("BONCLAY_RECEIPT_SIGNING_KEY", raising=False)
    monkeypatch.setattr(provenance, "_data_dir", str(tmp_path))
    monkeypatch.setattr(provenance, "_signing_cache", None)
    first = provenance.public_key_b64()
    monkeypatch.setattr(provenance, "_signing_cache", None)
    assert provenance.public_key_b64() == first
    if os.name != "nt":
        assert oct((tmp_path / "receipt_ed25519.key").stat().st_mode & 0o777) == "0o600"


def test_a_receipt_made_before_ed25519_still_verifies_by_hmac(backend_mod):
    receipt = provenance.get_receipt(_generate(backend_mod)["receipt_id"])
    old = {k: v for k, v in receipt.items() if k not in ("signature_ed25519", "key_id")}
    pub = provenance.public_receipt(old)
    assert pub["signature_valid"] is True and pub["signature_ed25519_valid"] is None


def _withdraw(voice_id, reason="withdrawn"):
    return provenance.record_withdrawal(voice_id, reason)


def test_withdrawals_are_hash_chained_and_signed(app_client, backend_mod):
    first, second = _withdraw("voice-a"), _withdraw("voice-b", "retention_expired")
    assert first["prev"] == provenance.LOG_GENESIS and second["prev"] == first["row_hash"]
    assert provenance.withdrawal_row_valid(first) and provenance.withdrawal_row_valid(second)
    log = app_client.get("/api/withdrawals").json()
    assert log["check"]["ok"] is True and log["check"]["signed_rows"] == 2 and log["check"]["head"] == second["row_hash"]
    assert [r["voice_id"] for r in log["rows"]] == ["voice-a", "voice-b"]


def test_an_edited_deleted_or_reordered_withdrawal_row_breaks_the_chain(backend_mod):
    rows = [_withdraw("v1"), _withdraw("v2"), _withdraw("v3")]
    assert provenance.verify_withdrawal_log(rows)["ok"] is True
    edited = [rows[0], {**rows[1], "withdrawn_at": "2020-01-01T00:00:00"}, rows[2]]
    assert provenance.verify_withdrawal_log(edited)["broken_at"] == 1
    assert provenance.verify_withdrawal_log([rows[0], rows[2]])["broken_at"] == 1       # a deleted row
    assert provenance.verify_withdrawal_log([rows[1], rows[0], rows[2]])["broken_at"] == 0  # reordered
    assert provenance.verify_withdrawal_log([rows[0], {**rows[1], "sig": rows[2]["sig"]}, rows[2]])["broken_at"] == 1


def test_legacy_unsigned_withdrawal_rows_still_work_and_are_flagged(backend_mod, tmp_path):
    import json
    (tmp_path / "voice_withdrawals.jsonl").write_text(
        json.dumps({"voice_id": "old", "withdrawn_at": "2026-10-01T00:00:00", "reason": "withdrawn"}) + "\n")
    provenance.configure(data_dir=str(tmp_path))
    assert provenance.withdrawal_for("old")["withdrawn_at"] == "2026-10-01T00:00:00"
    new = _withdraw("new")
    assert new["prev"] == provenance.LOG_GENESIS  # the chain starts at the first signed row
    check = provenance.verify_withdrawal_log()
    assert check["ok"] is True and check["legacy_rows"] == 1 and check["signed_rows"] == 1


def test_a_receipt_of_a_withdrawn_voice_reports_a_signed_withdrawal(backend_mod):
    receipt = provenance.get_receipt(_generate(backend_mod)["receipt_id"])
    assert provenance.public_receipt(receipt)["voice_withdrawal_signed"] is None  # no voice, not withdrawn
    receipt = {**receipt, "voice_id": "voice-x"}
    row = _withdraw("voice-x")
    pub = provenance.public_receipt(receipt)
    assert pub["voice_withdrawn_at"] == row["withdrawn_at"]
    assert pub["voice_withdrawal_signed"] is True and pub["voice_withdrawal_row_hash"] == row["row_hash"]


# ---- receipt spec, SDK and consent status -----------------------------------------------------------------------

def _sdk():
    import os
    import sys
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sdk", "verify")
    if path not in sys.path:
        sys.path.insert(0, path)
    import bonclay_verify
    return bonclay_verify


def test_the_sdk_verifies_a_receipt_made_by_the_backend_without_the_server(backend_mod):
    sdk = _sdk()
    gen = _generate(backend_mod)
    receipt = provenance.get_receipt(gen["receipt_id"])
    pub = provenance.public_key_b64()
    assert sdk.verify_receipt(receipt, pub) is True
    assert sdk.key_id(pub) == provenance.key_id(pub)
    assert sdk.verify_audio_sha256(receipt, gen.wav_bytes) is True
    assert sdk.verify_audio_sha256(receipt, gen.wav_bytes + b"x") is False
    assert sdk.verify_receipt({**receipt, "voice_name": "someone else"}, pub) is False   # a field was edited
    assert sdk.verify_receipt({**receipt, "version": 2}, pub) is False                    # unknown version
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    other = base64.b64encode(Ed25519PrivateKey.generate().public_key().public_bytes_raw()).decode()
    assert sdk.verify_receipt(receipt, other) is False


def test_the_sdk_checks_the_withdrawal_log_like_the_server(backend_mod):
    sdk = _sdk()
    rows = [_withdraw("v1"), _withdraw("v2"), _withdraw("v3")]
    pub = provenance.public_key_b64()
    assert sdk.verify_withdrawal_log(rows, pub)["ok"] is True
    assert sdk.verify_withdrawal_log([rows[0], rows[2]], pub)["broken_at"] == 1
    assert sdk.verify_withdrawal_log([rows[0], {**rows[1], "reason": "retention_expired"}, rows[2]], pub)["broken_at"] == 1
    assert sdk.voice_withdrawn({"voice_id": "v2"}, rows)["voice_id"] == "v2"
    assert sdk.voice_withdrawn({"voice_id": "nobody"}, rows) is None


def test_the_javascript_sdk_agrees_with_python(backend_mod, tmp_path):
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    gen = _generate(backend_mod)
    receipt = provenance.get_receipt(gen["receipt_id"])
    rows = [_withdraw("v1"), _withdraw("v2")]
    pub = provenance.public_key_b64()
    case = {"receipt": receipt, "bad": {**receipt, "engine": "edited"}, "rows": rows,
            "badrows": [rows[1], rows[0]], "pub": pub}
    (tmp_path / "case.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    sdk_file = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / "sdk" / "verify" / "verify.mjs"
    case_path = (tmp_path / "case.json").as_posix()
    script = (
        "import {verifyReceipt,verifyWithdrawalLog} from '%s';"
        "import fs from 'node:fs';const c=JSON.parse(fs.readFileSync('%s','utf8'));"
        "console.log(JSON.stringify([verifyReceipt(c.receipt,c.pub),verifyReceipt(c.bad,c.pub),"
        "verifyWithdrawalLog(c.rows,c.pub).ok,verifyWithdrawalLog(c.badrows,c.pub).ok]));"
    ) % (sdk_file.as_uri(), case_path)
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([node, str(tmp_path / "run.mjs")], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [True, False, True, False]


def test_consent_status_says_what_the_consent_behind_a_receipt_is(backend_mod):
    preset = provenance.get_receipt(_generate(backend_mod)["receipt_id"])
    assert provenance.public_receipt(preset)["voice_consent_status"] == "preset_voice"
    cloned = {**preset, "voice_id": "vc-1", "consent_id": "c-1", "consent_verified": True}
    assert provenance.public_receipt(cloned)["voice_consent_status"] == "active"
    assert provenance.public_receipt({**cloned, "consent_verified": False})["voice_consent_status"] == "unverified"
    _withdraw("vc-1")
    assert provenance.public_receipt(cloned)["voice_consent_status"] == "withdrawn"
    _withdraw("vc-2", "retention_expired")
    assert provenance.public_receipt({**cloned, "voice_id": "vc-2"})["voice_consent_status"] == "retention_ended"


def test_the_public_key_endpoint_lists_keys_and_the_receipt_version(app_client):
    body = app_client.get("/.well-known/bonclay-receipt-key").json()
    assert body["receipt_version"] == 1 and body["alg"] == "Ed25519"
    assert body["keys"] == [{"key_id": body["key_id"], "public_key": body["public_key"], "status": "current"}]
