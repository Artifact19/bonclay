"""Consent gate on POST /api/voices/clone: a spoken consent recording is required and, when ASR is
available, must match the challenge phrase. Models and Groq are faked so this runs on CPU."""
import types

import pytest

CHALLENGE = "I consent to creating an AI version of my voice for the BonClay demonstration on October 14, 2026."
LOCAL = {"storage-mode": "local"}


@pytest.fixture()
def backend_mod(app_client, monkeypatch, tmp_path):
    import backend

    class FakeModel:
        last = None

        def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
            FakeModel.last = {"ref_text": ref_text, "x_vector": x_vector_only_mode}
            return {"emb": [0.0]}

    async def fake_load(key):
        return FakeModel()

    monkeypatch.setattr(backend.advanced_manager, "load_model", fake_load)
    monkeypatch.setattr(backend, "_serialize_prompt", lambda prompt: b"pt")
    monkeypatch.setattr(backend, "LOCAL_AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(backend, "LOCAL_VOICES_FILE", str(tmp_path / "voices.json"))
    backend.FakeModel = FakeModel
    backend.limiter.reset()  # clone is limited to 5/minute; each test starts with a fresh window
    return backend


def fake_groq(heard_consent: str, heard_reference: str = "hello this is my reference voice"):
    def create(**kw):
        name = kw["file"][0]
        return types.SimpleNamespace(text=heard_consent if name.startswith("consent") else heard_reference)

    return types.SimpleNamespace(audio=types.SimpleNamespace(transcriptions=types.SimpleNamespace(create=create)))


def clone(client, ref_wav, consent=True, consent_audio=True):
    wav = ref_wav.read_bytes()
    files = {"reference_audio": ("ref.wav", wav, "audio/wav")}
    if consent_audio:
        files["consent_audio"] = ("consent.wav", wav, "audio/wav")
    data = {"display_name": "Test", "language": "en", "consent_text": CHALLENGE,
            "consent_confirmed": "true" if consent else "false"}
    return client.post("/api/voices/clone", headers=LOCAL, data=data, files=files)


def test_clone_without_consent_recording_is_refused(app_client, backend_mod, ref_wav):
    r = clone(app_client, ref_wav, consent_audio=False)
    assert r.status_code == 400
    assert r.json()["code"] == "CONSENT_REQUIRED"


def test_clone_with_checkbox_false_is_refused(app_client, backend_mod, ref_wav):
    r = clone(app_client, ref_wav, consent=False)
    assert r.status_code == 400
    assert r.json()["code"] == "CONSENT_REQUIRED"


def test_mismatched_consent_phrase_is_refused(app_client, backend_mod, ref_wav, monkeypatch):
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq("what is the weather today"))
    r = clone(app_client, ref_wav)
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "CONSENT_MISMATCH"
    assert body["details"]["heard"] == "what is the weather today"
    assert body["details"]["score"] < backend_mod.CONSENT_MATCH_THRESHOLD


def test_matching_consent_clones_and_auto_transcribes(app_client, backend_mod, ref_wav, monkeypatch):
    heard = CHALLENGE.replace("14,", "14th,")  # typical ASR variation
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq(heard))
    r = clone(app_client, ref_wav)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["consent"]["verified"] is True
    assert body["consent"]["score"] >= backend_mod.CONSENT_MATCH_THRESHOLD
    assert body["consent_id"]
    assert body["mode"] == "transcript"
    assert backend_mod.FakeModel.last == {"ref_text": "hello this is my reference voice", "x_vector": False}


def test_consent_without_asr_is_kept_but_unverified(app_client, backend_mod, ref_wav, monkeypatch):
    monkeypatch.setattr(backend_mod, "groq_client", None)
    monkeypatch.setattr(type(backend_mod.whisper_model), "available", property(lambda self: False))
    r = clone(app_client, ref_wav)
    assert r.status_code == 200, r.text
    assert r.json()["consent"]["verified"] is False


def test_local_whisper_verifies_consent_without_groq(app_client, backend_mod, ref_wav, monkeypatch):
    monkeypatch.setattr(backend_mod, "groq_client", None)
    monkeypatch.setattr(type(backend_mod.whisper_model), "available", property(lambda self: True))
    heard = {"consent": CHALLENGE.lower().replace(",", ""), "reference": "hello this is my reference voice"}

    async def fake_local(audio_bytes, ext, language=None):
        return {"text": heard["consent"] if language == "en" else heard["reference"], "language": language or "en"}

    monkeypatch.setattr(backend_mod, "_local_transcribe", fake_local)
    r = clone(app_client, ref_wav)
    assert r.status_code == 200, r.text
    assert r.json()["consent"]["verified"] is True


def test_consent_match_score():
    backend = pytest.importorskip("backend")
    assert backend._consent_match(CHALLENGE, CHALLENGE) == 1.0
    assert backend._consent_match(CHALLENGE, "") == 0.0
    assert backend._consent_match(CHALLENGE, "what is the weather today") < 0.2


# ---- the consent text itself is enforced -------------------------------------------------------------------

def clone_with_text(client, ref_wav, consent_text):
    """A clone request whose consent recording is paired with the given consent_text (None = field left out)."""
    wav = ref_wav.read_bytes()
    data = {"display_name": "Test", "language": "en", "consent_confirmed": "true"}
    if consent_text is not None:
        data["consent_text"] = consent_text
    return client.post("/api/voices/clone", headers=LOCAL, data=data,
                       files={"reference_audio": ("ref.wav", wav, "audio/wav"),
                              "consent_audio": ("consent.wav", wav, "audio/wav")})


def nothing_saved(backend_mod, tmp_path):
    return not (tmp_path / "voices.json").exists() and not list(tmp_path.glob("*.wav"))


def test_leaving_out_the_consent_text_does_not_skip_the_check(app_client, backend_mod, ref_wav, tmp_path, monkeypatch):
    """Regression: with no consent_text any audio was accepted as 'consent' and the voice was created unverified."""
    asr_calls = []
    monkeypatch.setattr(backend_mod, "groq_client", types.SimpleNamespace(audio=types.SimpleNamespace(
        transcriptions=types.SimpleNamespace(create=lambda **kw: asr_calls.append(kw)))))
    for missing in (None, "", "   "):
        r = clone_with_text(app_client, ref_wav, missing)
        assert (r.status_code, r.json()["code"]) == (400, "CONSENT_REQUIRED"), missing
    assert asr_calls == [] and nothing_saved(backend_mod, tmp_path)


def test_the_text_must_be_a_consent_statement_not_any_matching_sentence(app_client, backend_mod, ref_wav, tmp_path, monkeypatch):
    """Regression: the server verified the recording against client-supplied text, so audio saying 'the weather is
    sunny' paired with the text 'the weather is sunny' counted as consent."""
    weather = "The weather today is sunny with a light breeze."
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq(weather))  # the recording does say exactly this text
    r = clone_with_text(app_client, ref_wav, weather)
    assert (r.status_code, r.json()["code"]) == (422, "CONSENT_INVALID")
    assert "I consent to creating an AI version of my voice" in r.json()["details"]["expected"]
    # the words in the wrong order are not the statement either
    r = clone_with_text(app_client, ref_wav, "voice my of version AI an creating to consent I")
    assert (r.status_code, r.json()["code"]) == (422, "CONSENT_INVALID")
    assert nothing_saved(backend_mod, tmp_path)


@pytest.mark.parametrize("text", [
    CHALLENGE,                                                                    # what the wizard sends (US date)
    "I consent to creating an AI version of my voice for the BonClay demonstration on 14 October 2026.",  # UK date
    "i consent to creating an ai version of my voice for the bonclay demo",      # case and punctuation do not matter
    "Today, I consent to creating an AI version of my voice, for any use I approve.",   # extra words around it
])
def test_real_consent_sentences_are_accepted_whatever_the_date_format(app_client, backend_mod, ref_wav, monkeypatch, text):
    monkeypatch.setattr(backend_mod, "groq_client", fake_groq(text))
    r = clone_with_text(app_client, ref_wav, text)
    assert r.status_code == 200, r.text
    assert r.json()["consent"]["verified"] is True
