"""Compliance pack: itemised consent record, notice version, retention, withdrawal (receipts marked, /verify says so)
and the AI label written into exported files. Models, Groq and SpeechBrain are faked so this runs on CPU.

Not legal advice: these tests check that BonClay records and shows what it says it does."""
import asyncio
import glob
import io
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from services import ai_label, consent_record, provenance
from tests.test_consent import LOCAL, backend_mod, fake_groq  # noqa: F401  (backend_mod is a fixture)
from tests.test_provenance import SR, _tone, _verify, _wav, fake_seal  # noqa: F401  (fake_seal is a fixture)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    from services import speaker

    monkeypatch.setattr(speaker, "similarity", lambda a, b: 0.83)
    monkeypatch.delenv("VOICE_RETENTION_DAYS", raising=False)
    monkeypatch.setenv("VOICE_RETENTION_ENFORCE", "off")  # these tests exercise the report-only path; the default is on
    monkeypatch.delenv("AUDIBLE_AI_LABEL", raising=False)
    monkeypatch.delenv("BONCLAY_OPERATOR_NAME", raising=False)
    import backend
    backend.consent_challenges._challenges.clear()
    yield
    backend.consent_challenges._challenges.clear()


@pytest.fixture()
def env(backend_mod, fake_seal, tmp_path, monkeypatch):  # noqa: F811
    """Backend with temp voice/audio/receipt/generation stores (never the owner's real data)."""
    from core import generations
    from services.local_store import LocalStore

    monkeypatch.setattr(backend_mod, "supabase", None)
    monkeypatch.setattr(generations, "_data_dir", str(tmp_path))
    monkeypatch.setattr(backend_mod, "local_db", LocalStore(str(tmp_path)))
    return backend_mod


def clone(client, ref_wav, name="Priya Sharma", heard=None, monkeypatch=None, backend=None):
    ch = client.get("/api/consent/challenge", params={"name": name}, headers=LOCAL).json()
    if backend is not None:
        monkeypatch.setattr(backend, "groq_client", fake_groq(heard or ch["text"]))
    wav = ref_wav.read_bytes()
    r = client.post("/api/voices/clone", headers=LOCAL, data={
        "display_name": name, "language": "en", "consent_confirmed": "true", "challenge_id": ch["challenge_id"]},
        files={"reference_audio": ("ref.wav", wav, "audio/wav"), "consent_audio": ("consent.wav", wav, "audio/wav")})
    return ch, r


@pytest.fixture()
def cloned(app_client, env, ref_wav, monkeypatch):
    ch, r = clone(app_client, ref_wav, monkeypatch=monkeypatch, backend=env)
    assert r.status_code == 200, r.text
    return {"challenge": ch, **r.json()}


# ---- 1. consent record --------------------------------------------------------------------------------------------

def test_clone_stores_an_itemised_consent_record(app_client, cloned):
    rec = app_client.get(f"/api/voices/{cloned['id']}/consent-record", headers=LOCAL)
    assert rec.status_code == 200, rec.text
    rec = rec.json()
    ch = cloned["challenge"]
    assert rec["voice_id"] == cloned["id"] and rec["voice_name"] == "Priya Sharma"
    assert rec["consent_id"] == cloned["consent_id"]
    assert rec["purpose"] == "Create an AI version of this voice for use in BonClay by the owner of this BonClay studio."
    assert rec["data_stored"] == list(consent_record.DATA_STORED)
    assert any("sample" in x for x in rec["data_stored"]) and any("embedding" in x for x in rec["data_stored"])
    assert any("consent recording" in x for x in rec["data_stored"]) and any("transcript" in x for x in rec["data_stored"])
    assert rec["challenge_sentence"] == ch["text"] and rec["challenge_id"] == ch["challenge_id"]
    assert rec["challenge_code"] == re.search(r"My code is (.+)\.$", ch["text"]).group(1)
    assert rec["transcript_match_score"] == 1.0 and rec["transcript_verified"] is True
    assert rec["speaker_match_score"] == 0.83 and rec["speaker_checked"] is True
    assert rec["notice_version"] == consent_record.NOTICE_VERSION
    assert rec["retention_days"] == 365
    consented = datetime.fromisoformat(rec["consented_at"])
    assert abs((datetime.now(timezone.utc) - consented).total_seconds()) < 60
    assert datetime.fromisoformat(rec["retain_until"]) - consented == timedelta(days=365)
    assert rec["legacy"] is False and rec["how_to_withdraw"] and rec["contact"]
    # the clone response carries it too, and so does the stored voice row
    assert cloned["consent_record"]["voice_id"] == cloned["id"]
    assert [v for v in app_client.get("/api/voices", headers=LOCAL).json() if v["id"] == cloned["id"]][0][
        "consent_record"]["notice_version"] == consent_record.NOTICE_VERSION


def test_the_purpose_names_the_operator_when_configured(app_client, env, ref_wav, monkeypatch):
    monkeypatch.setenv("BONCLAY_OPERATOR_NAME", "Acme Clinic")
    _, r = clone(app_client, ref_wav, monkeypatch=monkeypatch, backend=env)
    assert r.json()["consent_record"]["purpose"].endswith("for use in BonClay by Acme Clinic.")


def test_only_the_owner_can_read_a_consent_record(app_client, env, cloned):
    env.save_local_voice({"id": "11111111-1111-4111-8111-111111111111", "user_id": "someone-else",
                          "display_name": "Theirs", "consent_id": "c", "created_at": datetime.now().isoformat()})
    assert app_client.get("/api/voices/11111111-1111-4111-8111-111111111111/consent-record",
                          headers=LOCAL).status_code == 404
    assert app_client.get("/api/voices/00000000-0000-4000-8000-000000000000/consent-record",
                          headers=LOCAL).status_code == 404
    assert app_client.get("/api/voices/..%2Fx/consent-record", headers=LOCAL).status_code in (404, 422)
    assert app_client.get(f"/api/voices/{cloned['id']}/consent-record", headers=LOCAL).status_code == 200


def test_a_voice_cloned_before_records_existed_gets_a_partial_legacy_record(app_client, env):
    env.save_local_voice({"id": "22222222-2222-4222-8222-222222222222", "user_id": "local-user",
                          "display_name": "Old", "consent_id": "c1", "consent_verified": True,
                          "speaker_score": 0.7, "language": "en", "engine": "qwen3_0_6b",
                          "created_at": "2026-01-02T10:00:00"})
    rec = app_client.get("/api/voices/22222222-2222-4222-8222-222222222222/consent-record", headers=LOCAL).json()
    assert rec["legacy"] is True and rec["voice_name"] == "Old" and rec["speaker_match_score"] == 0.7
    assert rec["challenge_sentence"] is None and rec["notice_version"] is None
    assert rec["retain_until"].startswith("2027-01-02")


def test_the_challenge_carries_the_notice_version_retention_and_contact(app_client, monkeypatch):
    monkeypatch.setenv("VOICE_RETENTION_DAYS", "90")
    monkeypatch.setenv("BONCLAY_PRIVACY_CONTACT", "privacy@example.org")
    c = app_client.get("/api/consent/challenge", params={"name": "A"}, headers=LOCAL).json()
    assert c["notice_version"] == consent_record.NOTICE_VERSION
    assert c["retention_days"] == 90 and c["contact"] == "privacy@example.org"


# ---- 2. notice ----------------------------------------------------------------------------------------------------

def test_the_wizard_and_the_server_agree_on_the_notice_version():
    src = (ROOT / "frontend/src/components/voices/noticeText.ts").read_text()
    assert re.search(rf'NOTICE_VERSION\s*=\s*"{re.escape(consent_record.NOTICE_VERSION)}"', src)


# ---- 3. retention -------------------------------------------------------------------------------------------------

def test_retention_days_default_override_and_bad_values(monkeypatch):
    assert consent_record.retention_days() == 365
    monkeypatch.setenv("VOICE_RETENTION_DAYS", "30")
    assert consent_record.retention_days() == 30
    for bad in ("abc", "0", "-5", ""):
        monkeypatch.setenv("VOICE_RETENTION_DAYS", bad)
        assert consent_record.retention_days() == 365


def test_retain_until_and_expiry_calculation():
    assert consent_record.retain_until("2026-01-01T00:00:00+00:00", 10) == "2026-01-11T00:00:00+00:00"
    assert consent_record.retain_until("not a date") is None and consent_record.retain_until(None) is None
    now = datetime(2026, 1, 11, 0, 0, 1, tzinfo=timezone.utc)
    assert consent_record.is_expired("2026-01-01T00:00:00+00:00", now, 10)
    assert not consent_record.is_expired("2026-01-02T00:00:00+00:00", now, 10)
    assert not consent_record.is_expired(None, now, 10)  # unknown age is never "expired"
    voices = [{"id": "a", "display_name": "Old", "created_at": "2025-01-01T00:00:00+00:00"},
              {"id": "b", "display_name": "New", "created_at": datetime.now(timezone.utc).isoformat()},
              {"id": "c", "display_name": "Own period", "created_at": "2026-01-01T00:00:00+00:00",
               "consent_record": {"retention_days": 5, "consented_at": "2026-01-01T00:00:00+00:00"}}]
    out = consent_record.expired_voices(voices, now=datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert [v["id"] for v in out] == ["a", "c"]
    assert consent_record.expired_voices(voices[:1], now=datetime(2025, 6, 1, tzinfo=timezone.utc)) == []


def _age_voice(env, voice_id, days):
    voices = env.get_local_voices()
    for v in voices:
        if v["id"] == voice_id:
            when = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
            v["created_at"] = when
            v["consent_record"]["consented_at"] = when
    Path(env.LOCAL_VOICES_FILE).write_text(json.dumps(voices))


def test_retention_enforcement_is_on_by_default(monkeypatch):
    from services import consent_record

    monkeypatch.delenv("VOICE_RETENTION_ENFORCE", raising=False)
    assert consent_record.enforce_enabled() is True
    monkeypatch.setenv("VOICE_RETENTION_ENFORCE", "off")
    assert consent_record.enforce_enabled() is False


def test_retention_report_lists_expired_voices_and_deletes_nothing(app_client, env, cloned):
    r = app_client.get("/api/voices/retention", headers=LOCAL).json()
    assert r == {"retention_days": 365, "enforce": False, "expired": []}
    _age_voice(env, cloned["id"], 400)
    r = app_client.get("/api/voices/retention", headers=LOCAL).json()
    assert [v["id"] for v in r["expired"]] == [cloned["id"]] and r["enforce"] is False
    assert r["expired"][0]["name"] == "Priya Sharma" and r["expired"][0]["retain_until"]
    # report only: the voice, its files and its record are all still there
    assert any(v["id"] == cloned["id"] for v in app_client.get("/api/voices", headers=LOCAL).json())
    assert (Path(env.LOCAL_AUDIO_DIR) / f"{cloned['id']}.wav").exists()
    # and enforcing is refused while the flag is off
    r = app_client.post("/api/voices/retention/enforce", headers=LOCAL)
    assert r.status_code == 409 and r.json()["code"] == "RETENTION_ENFORCE_OFF"
    assert (Path(env.LOCAL_AUDIO_DIR) / f"{cloned['id']}.wav").exists()


def test_enforce_flag_deletes_only_expired_voices_and_marks_them(app_client, env, cloned, ref_wav, monkeypatch):
    _, second = clone(app_client, ref_wav, name="Fresh Voice", monkeypatch=monkeypatch, backend=env)
    fresh = second.json()["id"]
    _age_voice(env, cloned["id"], 400)
    monkeypatch.setenv("VOICE_RETENTION_ENFORCE", "on")
    assert app_client.get("/api/voices/retention", headers=LOCAL).json()["enforce"] is True
    r = app_client.post("/api/voices/retention/enforce", headers=LOCAL)
    assert r.status_code == 200, r.text
    assert [v["id"] for v in r.json()["deleted"]] == [cloned["id"]]
    ids = [v["id"] for v in app_client.get("/api/voices", headers=LOCAL).json()]
    assert fresh in ids and cloned["id"] not in ids
    assert not (Path(env.LOCAL_AUDIO_DIR) / f"{cloned['id']}.wav").exists()
    assert provenance.withdrawal_for(cloned["id"])["reason"] == "retention_expired"
    assert provenance.withdrawal_for(fresh) is None


def test_startup_sweep_reports_by_default_and_deletes_with_the_flag(app_client, env, cloned, monkeypatch, caplog):
    _age_voice(env, cloned["id"], 400)
    with caplog.at_level("WARNING"):
        asyncio.run(env._voice_retention_startup())
    assert any(v["id"] == cloned["id"] for v in env.get_local_voices())
    assert "past their 365-day retention period" in caplog.text
    monkeypatch.setenv("VOICE_RETENTION_ENFORCE", "on")
    asyncio.run(env._voice_retention_startup())
    assert env.get_local_voices() == []
    assert provenance.withdrawal_for(cloned["id"]) is not None


# ---- 4. withdrawal ------------------------------------------------------------------------------------------------

def _generate_with(env, voice_id, name="Priya Sharma", x=None):
    x = _tone() if x is None else x
    return asyncio.run(env.persist_generation(
        "local-user", "local", _wav(x), len(x) / SR, text="hello", engine="qwen3_0_6b", language="en",
        voice_id=voice_id, voice_name=name))


def test_withdrawing_deletes_the_voice_but_keeps_and_marks_its_receipts(app_client, env, cloned):
    vid = cloned["id"]
    gen = _generate_with(env, vid)
    before = _verify(app_client, gen.wav_bytes).json()
    assert before["verdict"] == "made_with_bonclay"
    assert before["match"]["receipt"]["voice_withdrawn_at"] is None

    r = app_client.delete(f"/api/voices/{vid}", headers=LOCAL)
    assert r.status_code == 200 and r.json()["removed"]["consents"] == 1
    withdrawn_at = r.json()["withdrawn_at"]
    assert withdrawn_at
    # sample, embedding, consent recording and record are gone
    assert not glob.glob(str(Path(env.LOCAL_AUDIO_DIR) / f"{vid}.*"))
    assert not glob.glob(str(Path(env.LOCAL_AUDIO_DIR) / f"consent_{cloned['consent_id']}.*"))
    assert app_client.get(f"/api/voices/{vid}/consent-record", headers=LOCAL).status_code == 404

    # the receipt is still there, still validly signed, and unchanged
    receipt = provenance.get_receipt(gen["receipt_id"])
    assert receipt is not None and provenance.signature_valid(receipt) and "voice_withdrawn_at" not in receipt
    after = _verify(app_client, gen.wav_bytes).json()
    assert after["verdict"] == "made_with_bonclay" and after["match"]["signature_valid"] is True
    shown = after["match"]["receipt"]
    assert shown["voice_withdrawn_at"] == withdrawn_at and shown["voice_removed_reason"] == "withdrawn"
    assert shown["signature_valid"] is True
    # the public receipt endpoint says the same
    pub = app_client.get(f"/api/receipts/{gen['receipt_id']}").json()
    assert pub["voice_withdrawn_at"] == withdrawn_at


def test_receipts_of_other_voices_and_preset_voices_are_not_marked(app_client, env, cloned):
    other = _generate_with(env, None, name="af_heart")
    gen = _generate_with(env, cloned["id"])
    app_client.delete(f"/api/voices/{cloned['id']}", headers=LOCAL)
    assert app_client.get(f"/api/receipts/{other['receipt_id']}").json()["voice_withdrawn_at"] is None
    assert app_client.get(f"/api/receipts/{gen['receipt_id']}").json()["voice_withdrawn_at"]


def test_a_second_withdrawal_keeps_the_first_date(env):
    first = provenance.record_withdrawal("v-1")
    again = provenance.record_withdrawal("v-1", "retention_expired")
    assert again["withdrawn_at"] == first["withdrawn_at"] and again["reason"] == "withdrawn"
    assert provenance.record_withdrawal("") is None and provenance.withdrawal_for(None) is None


# ---- 5. labels in files -------------------------------------------------------------------------------------------

def _ffprobe_tags(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(path)],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return json.loads(out).get("format", {}).get("tags", {})


def test_label_wav_keeps_every_sample_byte_for_byte():
    raw = _wav(_tone())
    out = provenance.label_wav(raw, "rid-1")
    assert out != raw and out.startswith(b"RIFF")
    assert provenance.read_wav_label(out) == {"ICMT": "AI-generated voice · BonClay · receipt rid-1",
                                              "ISFT": "BonClay"}
    a, sr_a = sf.read(io.BytesIO(raw), dtype="int16")
    b, sr_b = sf.read(io.BytesIO(out), dtype="int16")
    assert sr_a == sr_b == SR and np.array_equal(a, b)
    assert sf.info(io.BytesIO(out)).frames == len(_tone())
    assert out[8:12] == b"WAVE" and int.from_bytes(out[4:8], "little") == len(out) - 8
    assert provenance.label_wav(out, "rid-2") == out  # never stacked twice


def test_label_wav_leaves_non_wav_data_alone():
    for junk in (b"", b"not a wav", b"RIFF" + b"\x00" * 20):
        assert provenance.label_wav(junk, "x") == junk


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
def test_generated_audio_carries_the_label_ffprobe_reads_it(app_client, env, tmp_path):
    gen = _generate_with(env, None, name="af_heart")
    rid = gen["receipt_id"]
    path = tmp_path / "download.wav"
    path.write_bytes(Path(env.LOCAL_AUDIO_DIR, f"gen_{gen['id']}.wav").read_bytes())
    tags = _ffprobe_tags(path)
    assert tags["comment"] == f"AI-generated voice · BonClay · receipt {rid}"
    assert tags["encoder"] == "BonClay"
    # the served bytes are what was stored, and the receipt hash covers the labelled file as delivered
    assert path.read_bytes() == gen.wav_bytes
    assert provenance.get_receipt(rid)["audio_sha256"] == provenance.sha256_hex(gen.wav_bytes)
    # the label did not disturb the watermark: the downloaded file is still detected and matched
    body = _verify(app_client, path.read_bytes()).json()
    assert body["verdict"] == "made_with_bonclay" and body["match"]["exact"] is True
    assert body["watermark"]["detected"] is True


def test_the_label_does_not_change_the_watermarked_samples(env):
    x, _ = sf.read(io.BytesIO(_wav(_tone())), dtype="float32")  # what apply_provenance reads (16-bit quantised)
    wm = provenance.watermark(x, SR, provenance.message_for("g1"))
    out, receipt = provenance.apply_provenance(
        kind="generation", subject_id="g1", wav_bytes=_wav(x), text="t", engine="kokoro", voice_id=None,
        voice_name="af_heart", consent_id=None, consent_verified=None, user_id="u")
    got, _ = sf.read(io.BytesIO(out), dtype="float32")
    ref, _ = sf.read(io.BytesIO(_wav(wm)), dtype="float32")
    assert np.array_equal(got, ref)
    assert receipt["receipt_id"] in provenance.read_wav_label(out)["ICMT"]


def test_audible_label_is_off_by_default_and_prepends_when_on(env, monkeypatch):
    calls = []
    monkeypatch.setattr(ai_label, "_clip", None)
    monkeypatch.setattr(ai_label, "_synthesize", lambda: calls.append(1) or (_tone(0.8, 24000, 880.0), 24000))
    plain = _generate_with(env, None, name="af_heart")
    assert not calls
    n_plain = sf.info(io.BytesIO(plain.wav_bytes)).frames
    assert n_plain == len(_tone())

    monkeypatch.setenv("AUDIBLE_AI_LABEL", "on")
    labelled = _generate_with(env, None, name="af_heart")
    info = sf.info(io.BytesIO(labelled.wav_bytes))
    extra = int((0.8 + ai_label.GAP_SECONDS) * 24000)
    assert abs(info.frames - (n_plain + extra)) <= 2 and len(calls) == 1
    assert abs(labelled["duration_s"] - (len(_tone()) / SR + 0.8 + ai_label.GAP_SECONDS)) < 0.01
    # metadata label and receipt still apply
    assert labelled["receipt_id"] in provenance.read_wav_label(labelled.wav_bytes)["ICMT"]


def test_audible_label_failure_never_blocks_generation(env, monkeypatch):
    def boom():
        raise RuntimeError("kokoro missing")

    monkeypatch.setattr(ai_label, "_clip", None)
    monkeypatch.setattr(ai_label, "_synthesize", boom)
    monkeypatch.setenv("AUDIBLE_AI_LABEL", "on")
    gen = _generate_with(env, None, name="af_heart")
    assert sf.info(io.BytesIO(gen.wav_bytes)).frames == len(_tone())
