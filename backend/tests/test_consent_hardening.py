"""Consent hardening: fresh one-time challenges (GET /api/consent/challenge) and the speaker match between the
consent recording and the voice sample. Models, Groq and SpeechBrain are faked so this runs on CPU."""
import time

import pytest

from tests.test_consent import LOCAL, backend_mod, fake_groq  # noqa: F401  (backend_mod is a fixture)


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    """No real speaker model in these tests unless a test asks for one; start with an empty challenge store."""
    import backend
    from services import speaker

    monkeypatch.setattr(speaker, "similarity", lambda a, b: 0.9)
    backend.consent_challenges._challenges.clear()
    yield
    backend.consent_challenges._challenges.clear()


def get_challenge(client, name="Priya Sharma"):
    r = client.get("/api/consent/challenge", params={"name": name}, headers=LOCAL)
    assert r.status_code == 200, r.text
    return r.json()


def clone(client, ref_wav, challenge_id=None, consent_text="ignored by the server"):
    wav = ref_wav.read_bytes()
    data = {"display_name": "Test", "language": "en", "consent_confirmed": "true", "consent_text": consent_text}
    if challenge_id is not None:
        data["challenge_id"] = challenge_id
    return client.post("/api/voices/clone", headers=LOCAL, data=data,
                       files={"reference_audio": ("ref.wav", wav, "audio/wav"),
                              "consent_audio": ("consent.wav", wav, "audio/wav")})


# ---- challenge ---------------------------------------------------------------------------------------------

def test_challenge_shape_and_text(app_client):
    c = get_challenge(app_client)
    assert set(c) == {"challenge_id", "text", "expires_at", "notice_version", "retention_days", "contact"}
    import re
    assert re.fullmatch(r"I, Priya Sharma, consent to creating an AI version of my voice with BonClay on "
                        r"\d{1,2} [A-Z][a-z]+ \d{4}\. My code is [a-z]+ [1-9]\d\.", c["text"]), c["text"]


def test_challenges_are_unique_and_name_is_cleaned(app_client):
    a, b = get_challenge(app_client), get_challenge(app_client, name="<b>Ann</b>; DROP")
    assert a["challenge_id"] != b["challenge_id"]
    assert "<" not in b["text"] and ";" not in b["text"]
    assert get_challenge(app_client, name="")["text"].startswith("I, the speaker, consent")


def test_challenge_expires_and_is_one_time(backend_mod):
    ch = backend_mod.consent_challenges
    issued = ch.issue("u1", "Priya")
    text = ch.claim(issued["challenge_id"], "u1")
    assert text == issued["text"]
    ch.finish(issued["challenge_id"], False)  # a failed attempt can be retried
    assert ch.claim(issued["challenge_id"], "u1") == text
    ch.finish(issued["challenge_id"], True)  # a successful clone uses it up
    with pytest.raises(backend_mod.AppError) as e:
        ch.claim(issued["challenge_id"], "u1")
    assert e.value.code == "CONSENT_CHALLENGE_INVALID" and e.value.status == 422

    old = ch.issue("u1", "Priya")
    ch._challenges[old["challenge_id"]]["expires_at"] = time.time() - 1
    with pytest.raises(backend_mod.AppError) as e:
        ch.claim(old["challenge_id"], "u1")
    assert e.value.code == "CONSENT_CHALLENGE_INVALID"


def test_challenge_belongs_to_one_user_and_cannot_run_twice_at_once(backend_mod):
    ch = backend_mod.consent_challenges
    issued = ch.issue("u1", "Priya")
    with pytest.raises(backend_mod.AppError):
        ch.claim(issued["challenge_id"], "someone-else")
    ch.claim(issued["challenge_id"], "u1")
    with pytest.raises(backend_mod.AppError):  # still running: a parallel submit is refused
        ch.claim(issued["challenge_id"], "u1")


def test_store_is_bounded(backend_mod, monkeypatch):
    ch = backend_mod.consent_challenges
    monkeypatch.setattr(ch, "MAX_CHALLENGES", 5)
    ids = [ch.issue("u", "n")["challenge_id"] for _ in range(8)]
    assert len(ch._challenges) <= 5
    assert ids[-1] in ch._challenges and ids[0] not in ch._challenges


# ---- clone with a challenge --------------------------------------------------------------------------------

def test_clone_with_challenge_uses_the_stored_text(app_client, backend_mod, ref_wav, monkeypatch):
    c = get_challenge(app_client)
    seen = {}

    async def fake_stt(audio, name, language=None):
        if name == "consent":
            return c["text"]
        return "hello this is my reference voice"

    monkeypatch.setattr(backend_mod, "_speech_to_text", fake_stt)
    real = backend_mod._verify_consent

    async def spy(consent_bytes, text):
        seen["text"] = text
        return await real(consent_bytes, text)

    monkeypatch.setattr(backend_mod, "_verify_consent", spy)
    # the client text is not even a consent statement: it is ignored when a challenge id is sent
    r = clone(app_client, ref_wav, c["challenge_id"], consent_text="the weather is sunny")
    assert r.status_code == 200, r.text
    assert seen["text"] == c["text"]
    body = r.json()["consent"]
    assert body["verified"] is True and body["score"] == 1.0
    assert body["speaker_score"] == 0.9 and body["challenge_id"] == c["challenge_id"]
    # stored with the local voice row
    row = backend_mod.get_local_voices()[-1]
    assert row["speaker_score"] == 0.9


def test_challenge_cannot_be_reused_after_success(app_client, backend_mod, ref_wav, monkeypatch):
    c = get_challenge(app_client)
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq(c["text"]))
    assert clone(app_client, ref_wav, c["challenge_id"]).status_code == 200
    r = clone(app_client, ref_wav, c["challenge_id"])
    assert (r.status_code, r.json()["code"]) == (422, "CONSENT_CHALLENGE_INVALID")


def test_challenge_survives_a_failed_attempt(app_client, backend_mod, ref_wav, monkeypatch):
    c = get_challenge(app_client)
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq("something else entirely"))
    r = clone(app_client, ref_wav, c["challenge_id"])
    assert (r.status_code, r.json()["code"]) == (422, "CONSENT_MISMATCH")
    backend_mod.limiter.reset()
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq(c["text"]))
    assert clone(app_client, ref_wav, c["challenge_id"]).status_code == 200


@pytest.mark.parametrize("which", ["unknown", "expired"])
def test_unknown_or_expired_challenge_is_rejected(app_client, backend_mod, ref_wav, which, tmp_path):
    cid = "nope"
    if which == "expired":
        c = get_challenge(app_client)
        cid = c["challenge_id"]
        backend_mod.consent_challenges._challenges[cid]["expires_at"] = time.time() - 1
    r = clone(app_client, ref_wav, cid)
    assert (r.status_code, r.json()["code"]) == (422, "CONSENT_CHALLENGE_INVALID")
    assert not (tmp_path / "voices.json").exists()


def test_without_challenge_id_the_old_behaviour_remains(app_client, backend_mod, ref_wav, monkeypatch):
    text = "I consent to creating an AI version of my voice for the BonClay demonstration on 15 October 2026."
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq(text))
    r = clone(app_client, ref_wav, consent_text=text)
    assert r.status_code == 200, r.text
    assert "challenge_id" not in r.json()["consent"]


# ---- speaker match -----------------------------------------------------------------------------------------

def test_speaker_mismatch_is_refused(app_client, backend_mod, ref_wav, monkeypatch, tmp_path):
    from services import speaker

    c = get_challenge(app_client)
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq(c["text"]))
    monkeypatch.setattr(speaker, "similarity", lambda a, b: 0.1)
    r = clone(app_client, ref_wav, c["challenge_id"])
    body = r.json()
    assert (r.status_code, body["code"]) == (422, "CONSENT_SPEAKER_MISMATCH")
    assert body["details"]["score"] == 0.1 and body["details"]["threshold"] == speaker.threshold()
    assert "different people" in body["error"]
    assert not (tmp_path / "voices.json").exists() and not list(tmp_path.glob("*.wav"))
    # the challenge is still usable once the right person records
    backend_mod.limiter.reset()
    monkeypatch.setattr(speaker, "similarity", lambda a, b: 0.8)
    assert clone(app_client, ref_wav, c["challenge_id"]).status_code == 200


def test_speaker_check_skipped_when_unavailable(app_client, backend_mod, ref_wav, monkeypatch):
    from services import speaker

    c = get_challenge(app_client)
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq(c["text"]))
    monkeypatch.setattr(speaker, "similarity", lambda a, b: None)
    r = clone(app_client, ref_wav, c["challenge_id"])
    assert r.status_code == 200, r.text
    consent = r.json()["consent"]
    assert consent["speaker_score"] is None and consent["speaker_checked"] is False


def test_speaker_check_can_be_switched_off(app_client, backend_mod, ref_wav, monkeypatch):
    from services import speaker

    c = get_challenge(app_client)
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq(c["text"]))
    monkeypatch.setattr(speaker, "similarity", lambda a, b: 0.0)
    monkeypatch.setenv("CONSENT_SPEAKER_CHECK", "off")
    r = clone(app_client, ref_wav, c["challenge_id"])
    assert r.status_code == 200 and r.json()["consent"]["speaker_checked"] is False


def test_speaker_threshold_env(monkeypatch):
    from services import speaker

    assert speaker.threshold() == speaker.DEFAULT_THRESHOLD
    monkeypatch.setenv("CONSENT_SPEAKER_THRESHOLD", "0.7")
    assert speaker.threshold() == 0.7
    monkeypatch.setenv("CONSENT_SPEAKER_THRESHOLD", "garbage")
    assert speaker.threshold() == speaker.DEFAULT_THRESHOLD


def test_similarity_without_speechbrain_is_none(monkeypatch, ref_wav):
    from services import speaker

    monkeypatch.undo()  # undo the autouse fake: exercise the real function with no model
    monkeypatch.setattr(speaker, "_load", lambda: None)
    wav = ref_wav.read_bytes()
    assert speaker.similarity(wav, wav) is None
