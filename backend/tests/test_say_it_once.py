"""Say it once, hear it everywhere: routing per language, consent guard, translation validation, ownership, response
shapes, progress/ETA, cancel and the fast (preset voices) speed. Translation (the LLM), synthesis, the clock and the
job runner are faked, so this runs on CPU without models."""
import asyncio
from types import SimpleNamespace

import pytest

LOCAL = {"storage-mode": "local"}
SENTENCE = "Your appointment is confirmed for Thursday at four."
TRANSLATIONS = {
    "hi": "आपकी अपॉइंटमेंट गुरुवार को चार बजे की पक्की है।",
    "mr": "तुमची अपॉइंटमेंट गुरुवारी चार वाजता निश्चित आहे.",
    "ta": "உங்கள் சந்திப்பு வியாழக்கிழமை நான்கு மணிக்கு உறுதிசெய்யப்பட்டுள்ளது.",
    "fr": "Votre rendez-vous est confirmé pour jeudi à seize heures.",
    "es": "Su cita está confirmada para el jueves a las cuatro.",
}
VOICE = "voice-consented"


class Setup:
    """Everything a test needs to steer: which engines are 'installed', what the model says, what was spoken."""

    def __init__(self):
        self.chatterbox = True
        self.dhvaani = False  # DhVaani (Indian-language cloning) is an optional install; tests below switch it on
        self.indic = True
        self.kokoro = True
        self.llm_answers = dict(TRANSLATIONS)  # target code -> what the fake model answers (or an Exception)
        self.llm_calls = []
        self.spoken = []  # (kind, engine, language, text)
        self.speak_fail = set()
        self.jobs = []  # (factory) captured from job_runner.submit
        self.clock = 1_000.0  # the fake clock (seconds); translating and speaking advance it
        self.translate_seconds = 1.0
        self.speak_seconds = None  # None: 8 s per clip; or a callable (route, text) -> seconds
        self.snaps = []  # (label, snapshot) taken while the work is in flight
        self.active = 0
        self.max_active = 0
        self.during_speak = None  # optional hook(language) called in the middle of a clip
        self.audio_seconds = lambda text: 2.5

    def run_jobs(self):
        while self.jobs:
            asyncio.run(self.jobs.pop(0)())

    def snap(self, label):
        from routers import say_it_once as sio
        for run in list(sio._runs.values()):
            if run["state"] == "running":
                self.snaps.append((label, sio._snapshot(run)))


@pytest.fixture()
def env(app_client, monkeypatch, tmp_path):
    import backend
    from core.jobs import job_runner
    from routers import say_it_once as sio
    from services import chatterbox_clone, dhvaani_clone, indic_speak, llm
    from services.speech_estimate import Estimator

    s = Setup()
    monkeypatch.setattr(sio, "ESTIMATOR", Estimator(path=str(tmp_path / "rtf.json")))  # never the real data file
    monkeypatch.setattr(sio, "_now", lambda: s.clock)
    voices = [
        {"id": VOICE, "user_id": "local-user", "display_name": "Speaker", "engine": "qwen3_0_6b", "language": "en",
         "consent_id": "consent-1", "consent_verified": True},
        {"id": "voice-no-consent", "user_id": "local-user", "display_name": "Old clone", "engine": "qwen3_0_6b",
         "language": "en"},
        {"id": "voice-of-someone-else", "user_id": "another-user", "display_name": "Not mine", "engine": "qwen3_0_6b",
         "language": "en", "consent_id": "consent-2", "consent_verified": True},
    ]
    monkeypatch.setattr(backend, "get_local_voices", lambda: voices)
    monkeypatch.setattr(chatterbox_clone, "unavailable_reason", lambda: None if s.chatterbox else "not set up")
    monkeypatch.setattr(dhvaani_clone, "unavailable_reason", lambda: None if s.dhvaani else "DhVaani is not set up")
    monkeypatch.setattr(indic_speak, "indic_speak_unavailable_reason", lambda: None if s.indic else "not set up")
    monkeypatch.setattr(sio, "_kokoro_available", lambda lang: s.kokoro and lang in backend.ENGINE_REGISTRY["kokoro"]["languages"])
    monkeypatch.setattr(backend, "indic_f5_unavailable_reason", lambda: "no f5_tts")
    sio._runs.clear()

    def fake_chat(messages, **kw):
        target = next(c for c, (_, desc, _s) in sio.LANGUAGES.items() if f"into {desc}." in messages[0]["content"])
        s.llm_calls.append((target, messages, kw))
        s.clock += s.translate_seconds
        s.snap(f"translated:{target}")
        ans = s.llm_answers[target]
        if isinstance(ans, Exception):
            raise ans
        return SimpleNamespace(text=ans)

    monkeypatch.setattr(llm, "chat", fake_chat)

    async def fake_speak(route, text, language, voice_id, user_id, storage_mode, on_stage=None):
        if language in s.speak_fail:
            raise backend.AppError("GENERATION_FAILED", "synthesis blew up", 500)
        s.active += 1
        s.max_active = max(s.max_active, s.active)
        try:
            total = 8.0 if s.speak_seconds is None else (s.speak_seconds(route, text) if callable(s.speak_seconds) else s.speak_seconds)
            s.snap(f"speak-start:{language}")
            for _ in range(4):
                await asyncio.sleep(0)
                s.clock += total / 4
                s.snap(f"speaking:{language}")
                if s.during_speak:
                    s.during_speak(language)
            if on_stage:
                on_stage("finishing")
                s.snap(f"finishing:{language}")
                s.clock += 0.5
                s.snap(f"finishing:{language}")
        finally:
            s.active -= 1
        s.spoken.append((route["voice_kind"], route["engine"], language, text))
        n = len(s.spoken)
        return {"id": f"gen-{n}", "audio_url": f"/api/generations/gen-{n}/audio", "duration_s": s.audio_seconds(text),
                "receipt_id": f"rcpt-{n}"}

    monkeypatch.setattr(sio, "speak", fake_speak)
    monkeypatch.setattr(job_runner, "submit", lambda user_id, factory, kind="generate": (s.jobs.append(factory), f"job-{len(s.jobs)}")[1])
    if hasattr(backend.limiter, "reset"):
        backend.limiter.reset()
    yield s
    sio._runs.clear()


def post(client, languages, voice_id=VOICE, text=SENTENCE, **extra):
    return client.post("/api/say-it-once", headers=LOCAL,
                       json={"voice_id": voice_id, "text": text, "languages": languages, **extra})


def run_to_end(client, env, languages, **kw):
    r = post(client, languages, **kw)
    assert r.status_code == 202, r.text
    env.run_jobs()
    out = client.get(f"/api/say-it-once/{r.json()['job_id']}", headers=LOCAL)
    assert out.status_code == 200, out.text
    return {it["language"]: it for it in out.json()["items"]}, out.json()


# ---- routing per language -------------------------------------------------------------------------------------

def test_routing_with_every_engine_on(app_client, env):
    items, body = run_to_end(app_client, env, ["en", "hi", "mr", "fr"])
    assert body["state"] == "done"
    assert {l: (i["voice_kind"], i["engine"]) for l, i in items.items()} == {
        "en": ("your_voice", "qwen3_0_6b"),
        "fr": ("your_voice", "qwen3_0_6b"),
        "hi": ("your_voice", "chatterbox_ml"),   # Hindi cloning needs Chatterbox
        "mr": ("preset_voice", "indic_speak"),   # nothing clones Marathi: a preset voice, clearly labelled
    }
    assert items["mr"]["attribution"] == "Built with Indic-Speak from Bodhan AI / AI4Bharat."
    assert items["hi"]["attribution"] is None and items["fr"]["attribution"] is None
    assert items["hi"]["engine_name"].startswith("Chatterbox")
    assert all(i["status"] == "done" for i in items.values())


def test_hindi_falls_back_to_kokoro_preset_without_chatterbox(app_client, env):
    env.chatterbox = False
    items, _ = run_to_end(app_client, env, ["hi"])
    assert (items["hi"]["voice_kind"], items["hi"]["engine"]) == ("preset_voice", "kokoro")
    assert items["hi"]["attribution"] is None


def test_marathi_and_tamil_are_unavailable_without_indic_speak(app_client, env):
    env.indic = False
    items, body = run_to_end(app_client, env, ["mr", "ta", "fr"])
    for code in ("mr", "ta"):
        assert items[code]["status"] == "unavailable"
        assert "not set up" in items[code]["error"]
        assert items[code]["audio_url"] is None and items[code]["receipt_id"] is None
    assert items["fr"]["status"] == "done"  # the others still run
    assert body["state"] == "done"
    assert [s[2] for s in env.spoken] == ["fr"]
    assert all(c[0] == "fr" for c in env.llm_calls)  # no translation wasted on unavailable languages


def test_everything_off_keeps_your_voice_where_qwen_speaks_the_language(app_client, env):
    env.chatterbox = env.indic = env.kokoro = False
    items, _ = run_to_end(app_client, env, ["en", "fr", "hi", "ta"])
    assert items["en"]["voice_kind"] == "your_voice" and items["fr"]["voice_kind"] == "your_voice"
    assert items["hi"]["status"] == "unavailable" and items["ta"]["status"] == "unavailable"


def test_cheapest_first_and_sequential(app_client, env):
    items, body = run_to_end(app_client, env, ["mr", "hi", "fr", "es"])
    # job items come back in processing order: Qwen3 (fast) before Chatterbox (30 s) before Indic-Speak (60 s)
    r = post(app_client, ["mr", "hi", "fr"])
    assert [i["language"] for i in r.json()["items"]] == ["fr", "hi", "mr"]
    assert all(i["status"] == "queued" for i in r.json()["items"])
    env.run_jobs()
    assert [s[2] for s in env.spoken[-3:]] == ["fr", "hi", "mr"]


def test_same_language_is_not_translated(app_client, env):
    items, _ = run_to_end(app_client, env, ["en", "fr"])
    assert items["en"]["text"] == SENTENCE
    assert [c[0] for c in env.llm_calls] == ["fr"]


# ---- capabilities ---------------------------------------------------------------------------------------------

def test_capabilities_shape_and_values(app_client, env):
    r = app_client.get("/api/say-it-once/capabilities", params={"voice_id": VOICE}, headers=LOCAL)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["voice_id"] == VOICE
    caps = {c["language"]: c for c in body["languages"]}
    assert len(caps) == 20
    assert set(caps["hi"]) == {"language", "language_name", "voice_kind", "engine", "engine_name", "available",
                               "reason", "attribution", "est_seconds", "fast"}
    assert set(caps["hi"]["fast"]) == {"voice_kind", "engine", "engine_name", "available", "reason", "attribution",
                                       "est_seconds"}
    assert caps["hi"]["voice_kind"] == "your_voice" and caps["hi"]["engine"] == "chatterbox_ml"
    assert caps["hi"]["fast"]["engine"] == "kokoro" and caps["hi"]["fast"]["voice_kind"] == "preset_voice"
    assert caps["mr"]["fast"]["engine"] == "indic_speak" and caps["de"]["fast"]["available"] is False
    assert caps["hi"]["est_seconds"] > caps["hi"]["fast"]["est_seconds"] > 0  # your Hindi voice is slower than Kokoro
    assert caps["mr"]["voice_kind"] == "preset_voice" and caps["mr"]["attribution"]
    assert caps["fr"]["voice_kind"] == "your_voice" and caps["fr"]["language_name"] == "French"
    assert all(c["available"] for c in caps.values())


def test_capabilities_report_what_is_missing(app_client, env):
    env.indic = env.chatterbox = False
    caps = {c["language"]: c for c in app_client.get(
        "/api/say-it-once/capabilities", params={"voice_id": VOICE}, headers=LOCAL).json()["languages"]}
    assert caps["mr"]["available"] is False and "not set up" in caps["mr"]["reason"]
    assert caps["hi"]["voice_kind"] == "preset_voice" and caps["hi"]["engine"] == "kokoro" and caps["hi"]["available"]
    assert caps["fr"]["available"] is True


def test_capabilities_guarded_like_the_job(app_client, env):
    for vid, code in (("voice-no-consent", 403), ("voice-of-someone-else", 404), ("nope", 404)):
        r = app_client.get("/api/say-it-once/capabilities", params={"voice_id": vid}, headers=LOCAL)
        assert r.status_code == code, (vid, r.text)


# ---- consent guard and ownership ------------------------------------------------------------------------------

def test_voice_without_consent_is_refused(app_client, env):
    r = post(app_client, ["fr"], voice_id="voice-no-consent")
    assert r.status_code == 403
    assert r.json()["code"] == "CONSENT_REQUIRED"
    assert "consent" in r.json()["error"].lower()
    assert env.jobs == []  # nothing was queued


def test_someone_elses_or_unknown_voice_is_404(app_client, env):
    for vid in ("voice-of-someone-else", "does-not-exist"):
        r = post(app_client, ["fr"], voice_id=vid)
        assert r.status_code == 404, r.text
        assert r.json()["code"] == "VOICE_NOT_FOUND"
    assert env.jobs == []


def test_job_belongs_to_its_owner(app_client, env):
    from routers import say_it_once as sio
    r = post(app_client, ["fr"])
    job_id = r.json()["job_id"]
    env.run_jobs()
    assert app_client.get(f"/api/say-it-once/{job_id}", headers=LOCAL).status_code == 200
    sio._runs[job_id]["user_id"] = "someone-else"
    miss = app_client.get(f"/api/say-it-once/{job_id}", headers=LOCAL)
    assert miss.status_code == 404 and miss.json()["code"] == "JOB_NOT_FOUND"
    assert app_client.get("/api/say-it-once/not-a-job", headers=LOCAL).status_code == 404


# ---- translation validation -----------------------------------------------------------------------------------

def test_a_refusal_fails_that_language_and_the_rest_continue(app_client, env):
    env.llm_answers["hi"] = "I'm sorry, but I can't help with that request."
    items, body = run_to_end(app_client, env, ["fr", "hi", "es"])
    assert items["hi"]["status"] == "failed"
    assert "did not translate" in items["hi"]["error"] and "Hindi" in items["hi"]["error"]
    assert items["hi"]["audio_url"] is None
    assert items["fr"]["status"] == "done" and items["es"]["status"] == "done"
    assert body["state"] == "done"
    assert [s[2] for s in env.spoken] == ["fr", "es"]  # a refused translation is never spoken


def test_wrong_script_is_rejected(app_client, env):
    env.llm_answers["hi"] = "Aapki appointment Guruvaar ko chaar baje confirm hai"  # romanised, not Devanagari
    env.llm_answers["mr"] = "Your appointment is confirmed for Thursday at four."  # not translated
    items, _ = run_to_end(app_client, env, ["hi", "mr"])
    assert items["hi"]["status"] == "failed"
    assert "script" in items["hi"]["error"]
    assert items["mr"]["status"] == "failed"


def test_untranslated_latin_and_wrapped_answers(app_client, env):
    env.llm_answers["fr"] = SENTENCE  # echoed back
    env.llm_answers["es"] = '"Su cita está confirmada para el jueves a las cuatro."\n(Note: formal register)'
    items, _ = run_to_end(app_client, env, ["fr", "es"])
    assert items["fr"]["status"] == "failed" and "untranslated" in items["fr"]["error"]
    assert items["es"]["status"] == "done"
    assert items["es"]["text"] == "Su cita está confirmada para el jueves a las cuatro."  # quotes and note dropped


def test_no_language_model_is_a_clear_error_per_language(app_client, env):
    from services import llm
    env.llm_answers["fr"] = llm.LLMUnavailable({"groq": "no key"})
    items, _ = run_to_end(app_client, env, ["fr", "es"])
    assert items["fr"]["status"] == "failed" and "No language model" in items["fr"]["error"]
    assert items["es"]["status"] == "done"


def test_synthesis_failure_is_isolated(app_client, env):
    env.speak_fail = {"hi"}
    items, body = run_to_end(app_client, env, ["hi", "fr"])
    assert items["hi"]["status"] == "failed" and items["hi"]["error"] == "synthesis blew up"
    assert items["hi"]["text"] == TRANSLATIONS["hi"]  # the translation is still shown
    assert items["fr"]["status"] == "done" and body["state"] == "done"


def test_translation_prompt_is_strict_and_deterministic(app_client, env):
    run_to_end(app_client, env, ["hi"])
    target, messages, kw = env.llm_calls[0]
    assert kw["temperature"] == 0.0
    system = messages[0]["content"]
    assert "Hindi, written in Devanagari script" in system and "Output ONLY the translation" in system
    assert messages[1] == {"role": "user", "content": SENTENCE}


def test_validate_translation_unit():
    from routers.say_it_once import validate_translation as v
    assert v("Translation: Bonjour tout le monde ici", "Hello everyone out there", "en", "fr") == "Bonjour tout le monde ici"
    with pytest.raises(ValueError):
        v("", SENTENCE, "en", "fr")
    with pytest.raises(ValueError):
        v("Sure! Here is the translation: Votre rendez-vous", SENTENCE, "en", "fr")
    with pytest.raises(ValueError):
        v("x" * 900, SENTENCE, "en", "fr")
    with pytest.raises(ValueError):
        v("こんにちは", SENTENCE, "en", "ko")


# ---- response shapes and request validation ------------------------------------------------------------------

def test_start_response_shape(app_client, env):
    r = post(app_client, ["fr", "mr"])
    assert r.status_code == 202
    body = r.json()
    assert set(body) == {"job_id", "items", "speed", "est_total_s"}
    assert body["speed"] == "own" and body["est_total_s"] > 0
    assert all(set(i) == {"language", "status"} for i in body["items"])


def test_status_item_shape_while_queued_and_when_done(app_client, env):
    job_id = post(app_client, ["fr"]).json()["job_id"]
    queued = app_client.get(f"/api/say-it-once/{job_id}", headers=LOCAL).json()
    assert queued["state"] == "running" and queued["items"][0]["status"] == "queued"
    assert queued["items"][0]["stage"] == "waiting" and queued["items"][0]["progress"] == 0
    assert queued["progress"] == 0 and queued["eta_s"] > 0 and (queued["done"], queued["total"]) == (0, 1)
    env.run_jobs()
    done = app_client.get(f"/api/say-it-once/{job_id}", headers=LOCAL).json()
    assert done["state"] == "done"
    item = done["items"][0]
    assert set(item) == {"language", "language_name", "status", "text", "voice_kind", "engine", "engine_name",
                         "attribution", "audio_url", "receipt_id", "audio_seconds", "generation_id", "seconds",
                         "error", "stage", "progress", "eta_s", "started_at", "queue_position", "est_s", "slow"}
    assert item["language_name"] == "French" and item["text"] == TRANSLATIONS["fr"]
    assert item["audio_url"] == "/api/generations/gen-1/audio" and item["receipt_id"] == "rcpt-1"
    assert isinstance(item["seconds"], float) and item["error"] is None
    assert (item["stage"], item["progress"], item["eta_s"]) == ("done", 100, None)
    assert done["progress"] == 100 and done["eta_s"] == 0 and (done["done"], done["total"]) == (1, 1)
    assert not any(k.startswith("_") for k in item)


@pytest.mark.parametrize("payload", [
    {"languages": []},
    {"languages": ["fr", "es", "de", "it", "pt"]},   # more than four
    {"languages": ["xx"]},
    {"languages": ["fr"], "text": "a" * 201},
    {"languages": ["fr"], "text": ""},
    {"languages": ["fr"], "source_language": "xx"},
])
def test_bad_requests_are_rejected(app_client, env, payload):
    body = {"voice_id": VOICE, "text": SENTENCE, **payload}
    r = app_client.post("/api/say-it-once", headers=LOCAL, json=body)
    assert r.status_code in (400, 422), r.text
    assert env.jobs == []


def test_duplicate_languages_are_spoken_once(app_client, env):
    r = post(app_client, ["fr", "FR", "fr"])
    assert [i["language"] for i in r.json()["items"]] == ["fr"]


def test_a_user_cannot_pile_up_jobs(app_client, env):
    assert post(app_client, ["fr"]).status_code == 202
    assert post(app_client, ["fr"]).status_code == 202
    r = post(app_client, ["fr"])
    assert r.status_code == 429 and r.json()["code"] == "TOO_MANY_JOBS"


# ---- weekday guard --------------------------------------------------------------------------------------------

def test_wrong_weekday_is_refused_and_the_prompt_carries_the_right_word(app_client, env):
    env.llm_answers["hi"] = "आपकी अपॉइंटमेंट शनिवार को चार बजे की पक्की है।"  # Saturday, not Thursday
    env.llm_answers["mr"] = TRANSLATIONS["mr"]  # says गुरुवारी: Thursday, correct
    items, _ = run_to_end(app_client, env, ["hi", "mr"])
    assert items["hi"]["status"] == "failed" and "different day" in items["hi"]["error"]
    assert items["mr"]["status"] == "done"
    prompt = next(m for c, m, _ in env.llm_calls if c == "hi")[0]["content"]
    assert "Thursday = गुरुवार" in prompt and "Friday" not in prompt


def test_the_right_day_is_accepted(app_client, env):
    env.llm_answers["hi"] = "आपकी अपॉइंटमेंट गुरुवार को चार बजे की पक्की है।"
    items, _ = run_to_end(app_client, env, ["hi"])
    assert items["hi"]["status"] == "done"


# ---- review step: translate only, then speak approved text -----------------------------------------------------

def translate_post(client, languages, voice_id=VOICE, text=SENTENCE, **extra):
    return client.post("/api/say-it-once/translate", headers=LOCAL,
                       json={"voice_id": voice_id, "text": text, "languages": languages, **extra})


def test_translate_returns_per_language_review_and_speaks_nothing(app_client, env):
    r = translate_post(app_client, ["fr", "mr", "hi", "en"])
    assert r.status_code == 200, r.text
    items = {i["language"]: i for i in r.json()["items"]}
    assert list(items) == ["fr", "mr", "hi", "en"]  # the order the user picked
    assert set(items["fr"]) == {"language", "language_name", "translation", "warnings", "voice", "engine",
                                "engine_name", "available", "reason", "attribution"}
    assert items["fr"]["translation"] == TRANSLATIONS["fr"] and items["fr"]["warnings"] == []
    assert (items["fr"]["voice"], items["fr"]["engine"]) == ("yours", "qwen3_0_6b")
    assert (items["hi"]["voice"], items["hi"]["engine"]) == ("yours", "chatterbox_ml")
    assert (items["mr"]["voice"], items["mr"]["engine"]) == ("preset", "indic_speak")
    assert items["mr"]["attribution"]
    assert items["en"]["translation"] == SENTENCE  # same language: no model call
    assert [c[0] for c in env.llm_calls] == ["fr", "mr", "hi"]
    assert env.spoken == [] and env.jobs == []  # nothing was spoken or queued


def test_translate_surfaces_the_weekday_guard_as_a_warning_and_keeps_the_text(app_client, env):
    env.llm_answers["hi"] = "आपकी अपॉइंटमेंट शनिवार को चार बजे की पक्की है।"  # Saturday, not Thursday
    r = translate_post(app_client, ["hi", "mr"])
    assert r.status_code == 200
    items = {i["language"]: i for i in r.json()["items"]}
    assert items["hi"]["translation"] == env.llm_answers["hi"]  # kept, so the user can fix it by editing
    assert any("different day" in w for w in items["hi"]["warnings"])
    assert items["mr"]["warnings"] == []


def test_translate_turns_bad_answers_into_warnings(app_client, env):
    from services import llm
    env.llm_answers["hi"] = "I'm sorry, but I can't help with that request."
    env.llm_answers["mr"] = "Your appointment is confirmed for Thursday at four."  # wrong script
    env.llm_answers["fr"] = llm.LLMUnavailable({"groq": "no key"})
    env.llm_answers["es"] = TRANSLATIONS["es"]
    items = {i["language"]: i for i in translate_post(app_client, ["hi", "mr", "fr", "es"]).json()["items"]}
    assert items["hi"]["translation"] == "" and "did not translate" in items["hi"]["warnings"][0]
    assert items["mr"]["translation"] and any("script" in w for w in items["mr"]["warnings"])
    assert items["fr"]["translation"] == "" and "No language model" in items["fr"]["warnings"][0]
    assert items["es"]["warnings"] == [] and items["es"]["translation"] == TRANSLATIONS["es"]


def test_translate_skips_languages_that_cannot_be_spoken(app_client, env):
    env.indic = False
    items = {i["language"]: i for i in translate_post(app_client, ["mr", "fr"]).json()["items"]}
    assert items["mr"]["available"] is False and "not set up" in items["mr"]["reason"]
    assert items["mr"]["translation"] == "" and items["mr"]["voice"] == "preset"
    assert [c[0] for c in env.llm_calls] == ["fr"]


def test_translate_keeps_the_consent_guard_and_validates_input(app_client, env):
    for vid, code in (("voice-no-consent", 403), ("voice-of-someone-else", 404), ("nope", 404)):
        assert translate_post(app_client, ["fr"], voice_id=vid).status_code == code
    assert translate_post(app_client, ["fr"], voice_id="voice-no-consent").json()["code"] == "CONSENT_REQUIRED"
    assert translate_post(app_client, ["xx"]).status_code == 422
    assert translate_post(app_client, []).status_code == 422
    assert translate_post(app_client, ["fr"], text="a" * 201).status_code == 422
    assert env.llm_calls == []


def test_approved_translations_bypass_the_model(app_client, env):
    edited = "Votre rendez-vous est bien confirmé pour jeudi à quatre heures."
    items, body = run_to_end(app_client, env, ["fr", "es"], translations={"fr": edited})
    assert body["state"] == "done"
    assert items["fr"]["status"] == "done" and items["fr"]["text"] == edited
    assert [c[0] for c in env.llm_calls] == ["es"]  # only the language without an approved text hits the model
    assert items["es"]["text"] == TRANSLATIONS["es"]
    assert ("your_voice", "qwen3_0_6b", "fr", edited) in env.spoken


def test_approved_translations_for_every_language_never_call_the_model(app_client, env):
    approved = {"hi": TRANSLATIONS["hi"], "fr": TRANSLATIONS["fr"], "mr": TRANSLATIONS["mr"]}
    items, _ = run_to_end(app_client, env, ["hi", "fr", "mr"], translations=approved)
    assert env.llm_calls == []
    assert all(i["status"] == "done" for i in items.values())
    assert {s[2]: s[3] for s in env.spoken} == approved


def test_approved_translation_is_tidied_but_the_weekday_guard_is_the_users_call(app_client, env):
    saturday = "  आपकी अपॉइंटमेंट  शनिवार को चार बजे की पक्की है।\n"
    items, _ = run_to_end(app_client, env, ["hi"], translations={"hi": saturday})
    assert items["hi"]["status"] == "done"
    assert items["hi"]["text"] == "आपकी अपॉइंटमेंट शनिवार को चार बजे की पक्की है।"


@pytest.mark.parametrize("approved", [
    {"fr": "   "},                                # empty
    {"hi": "Aapki appointment Guruvaar ko"},      # wrong script
    {"fr": "x" * 900},                            # absurd length
    {"es": "Su cita está confirmada."},           # not one of the chosen languages
])
def test_bad_approved_translations_are_refused_before_anything_is_queued(app_client, env, approved):
    r = post(app_client, ["fr", "hi"], translations=approved)
    assert r.status_code == 422, r.text
    assert r.json()["code"] in ("INVALID_TRANSLATION", "INVALID_LANGUAGE")
    assert env.jobs == [] and env.llm_calls == []


def test_approved_translations_keep_the_consent_guard(app_client, env):
    r = post(app_client, ["fr"], voice_id="voice-no-consent", translations={"fr": TRANSLATIONS["fr"]})
    assert r.status_code == 403 and r.json()["code"] == "CONSENT_REQUIRED"
    assert env.jobs == []


def test_clean_approved_unit():
    from routers.say_it_once import clean_approved as c
    assert c(" Bonjour   tout le monde ", "Hello everyone", "en", "fr") == "Bonjour tout le monde"
    assert c("Hello again", "Hello again", "en", "en") == "Hello again"
    with pytest.raises(ValueError):
        c("", SENTENCE, "en", "fr")
    with pytest.raises(ValueError):
        c("こんにちは", SENTENCE, "en", "ko")


# ---- progress: stages, percentages, ETA -----------------------------------------------------------------------

def _per_language(snaps, key):
    out = {}
    for _, snap in snaps:
        for it in snap["items"]:
            out.setdefault(it["language"], []).append(it[key])
    return out


def _dedupe(seq):
    return [x for i, x in enumerate(seq) if i == 0 or seq[i - 1] != x]


def test_progress_never_goes_backwards_and_finishes_at_100(app_client, env):
    r = post(app_client, ["mr", "hi", "fr"])
    job_id = r.json()["job_id"]
    env.snap("start")
    env.run_jobs()
    final = app_client.get(f"/api/say-it-once/{job_id}", headers=LOCAL).json()
    snaps = env.snaps + [("end", final)]
    for lang, values in _per_language(snaps, "progress").items():
        assert values == sorted(values), (lang, values)
        assert values[0] == 0 and values[-1] == 100
        first = values.index(100)  # 100 only once the clip is really saved, and it stays there
        assert all(v == 100 for v in values[first:]) and all(v < 100 for v in values[:first]), (lang, values)
    overall = [snap["progress"] for _, snap in snaps]
    assert overall == sorted(overall) and overall[0] == 0 and overall[-1] == 100
    assert all(v <= 99 for v in overall[:-1])
    assert final["state"] == "done" and final["done"] == 3 and final["eta_s"] == 0


def test_stages_come_in_order_with_the_documented_bands(app_client, env):
    post(app_client, ["fr"])
    env.snap("start")
    env.run_jobs()
    stages = _dedupe(_per_language(env.snaps, "stage")["fr"])
    assert stages == ["waiting", "translating", "speaking", "finishing"]
    by_label = {}
    for label, snap in env.snaps:
        by_label.setdefault(label, []).append(snap["items"][0])
    assert 0 < by_label["translated:fr"][0]["progress"] <= 15          # translating: 0..15
    for it in by_label["speaking:fr"]:
        assert 15 <= it["progress"] <= 95 and it["stage"] == "speaking"  # generating the voice: 15..95
    for it in by_label["finishing:fr"]:
        assert 95 <= it["progress"] < 100 and it["stage"] == "finishing"  # watermark and save: 95..100
    speaking = [it["progress"] for it in by_label["speaking:fr"]]
    assert len(set(speaking)) == len(speaking)  # it actually moves between polls


def test_approved_text_skips_the_translating_stage(app_client, env):
    post(app_client, ["fr"], translations={"fr": TRANSLATIONS["fr"]})
    env.snap("start")
    env.run_jobs()
    assert _dedupe(_per_language(env.snaps, "stage")["fr"]) == ["waiting", "speaking", "finishing"]
    assert env.llm_calls == []
    assert min(it["progress"] for _, snap in env.snaps for it in snap["items"] if it["stage"] == "speaking") >= 15


def test_an_overrunning_clip_creeps_but_stays_below_95_until_done(app_client, env):
    env.speak_seconds = lambda route, text: 20.0  # about three times the estimate for Qwen3 French
    post(app_client, ["fr"])
    env.run_jobs()
    speaking = [it["progress"] for label, snap in env.snaps for it in snap["items"] if label.startswith("speaking:")]
    assert speaking == sorted(speaking) and 90 < speaking[-1] <= 95
    assert all(a < b for a, b in zip(speaking, speaking[1:]))  # still moving: it does not freeze at one number
    assert all(snap["items"][0]["eta_s"] >= 2 for label, snap in env.snaps if label.startswith("speaking:"))  # no "0 s left" while working


def test_a_huge_overrun_never_shows_more_than_95_until_the_clip_is_saved(app_client, env):
    env.speak_seconds = lambda route, text: 4000.0
    post(app_client, ["fr"])
    env.run_jobs()
    speaking = [it["progress"] for label, snap in env.snaps for it in snap["items"] if label.startswith("speaking:")]
    assert max(speaking) <= 95 and speaking == sorted(speaking)


def test_queued_cards_show_their_position_and_a_growing_eta(app_client, env):
    job_id = post(app_client, ["mr", "hi", "fr", "es"]).json()["job_id"]
    snap = app_client.get(f"/api/say-it-once/{job_id}", headers=LOCAL).json()
    assert [it["language"] for it in snap["items"]][0] in ("fr", "es")  # fast engine first
    assert [it["queue_position"] for it in snap["items"]] == [1, 2, 3, 4]
    etas = [it["eta_s"] for it in snap["items"]]
    assert etas == sorted(etas) and etas[-1] <= snap["eta_s"] + 1  # each card's ETA is when it would be done
    env.run_jobs()
    done = app_client.get(f"/api/say-it-once/{job_id}", headers=LOCAL).json()
    assert all(it["queue_position"] is None for it in done["items"])


def test_the_jobs_bar_weights_slow_languages_more(app_client, env):
    # Marathi (Indic-Speak, ~7x real time) is most of the work: finishing Qwen3 French alone is well under half
    post(app_client, ["fr", "mr"])
    env.snap("start")
    env.run_jobs()
    after_fr = next(snap for label, snap in env.snaps if label == "finishing:fr")
    assert 0 < after_fr["progress"] < 50


def test_unavailable_languages_do_not_count_toward_the_total(app_client, env):
    env.indic = False
    items, body = run_to_end(app_client, env, ["mr", "fr"])
    assert (body["done"], body["total"], body["progress"]) == (1, 1, 100)
    assert items["mr"]["stage"] == "unavailable"


def test_a_failed_language_still_lets_the_job_reach_100(app_client, env):
    env.speak_fail = {"hi"}
    items, body = run_to_end(app_client, env, ["hi", "fr"])
    assert items["hi"]["stage"] == "failed" and items["hi"]["status"] == "failed"
    assert (body["done"], body["failed"], body["total"], body["progress"]) == (1, 1, 2, 100)


# ---- ETA math and the running averages ------------------------------------------------------------------------

def test_estimator_uses_measured_rtf_and_speaking_rate(tmp_path):
    from services.speech_estimate import DEFAULT_RTF, Estimator
    e = Estimator(path=str(tmp_path / "e.json"))
    text = "x" * 120  # Latin: 14 chars per second -> 8.57 s of speech
    assert e.chars_per_second("fr") == 14 and e.chars_per_second("hi") == 12 and e.chars_per_second("zh") < 6
    assert e.audio_seconds(text, "fr") == pytest.approx(120 / 14)
    assert e.voice_seconds("kokoro", text, "fr") == pytest.approx(DEFAULT_RTF["kokoro"] * 120 / 14)
    assert e.voice_seconds("indic_speak", text, "hi") == pytest.approx(7.0 * 10)
    assert e.voice_seconds("kokoro", "Hi", "en") >= 1.0  # the floor: start-up and save are never free
    assert e.total_seconds("qwen3_0_6b", text, "fr", translate=True) == pytest.approx(
        e.translate_seconds() + 1.7 * 120 / 14)
    assert e.total_seconds(None, text, "fr", True) == 0


def test_running_average_moves_toward_what_this_machine_does_and_is_persisted(tmp_path):
    from services.speech_estimate import Estimator
    path = str(tmp_path / "e.json")
    e = Estimator(path=path)
    assert e.rtf("indic_speak") == 7.0 and e.samples("indic_speak") == 0
    for _ in range(6):  # six clips of 10 s of speech that took 40 s each: RTF 4
        e.record_voice("indic_speak", "hi", "x" * 120, voice_s=40.0, audio_s=10.0)
    assert 3.9 < e.rtf("indic_speak") < 4.5 and e.samples("indic_speak") == 6
    assert e.chars_per_second("hi") == pytest.approx(12.0, rel=0.01)  # 120 chars in 10 s
    again = Estimator(path=path)  # a new process reads the same numbers
    assert again.rtf("indic_speak") == pytest.approx(e.rtf("indic_speak")) and again.samples("indic_speak") == 6
    assert again.rtf("kokoro") == 0.18  # engines with no history keep the measured default


def test_estimator_ignores_junk_samples_and_a_damaged_file(tmp_path):
    from services.speech_estimate import Estimator
    path = tmp_path / "e.json"
    path.write_text("{not json")
    e = Estimator(path=str(path))
    e.record_voice("kokoro", "en", "hello there", voice_s=0.0, audio_s=3.0)
    e.record_voice("kokoro", "en", "hello there", voice_s=3.0, audio_s=0.0)
    assert e.samples("kokoro") == 0 and e.rtf("kokoro") == 0.18


def test_stage_fraction_is_monotonic_smooth_and_never_reaches_one():
    from services.speech_estimate import Estimator as E
    xs = [E.stage_fraction(t, 20.0) for t in range(0, 400, 2)]
    assert xs == sorted(xs) and xs[0] == 0 and max(xs) < 1.0
    assert E.stage_fraction(10, 20) == pytest.approx(0.5)
    assert E.stage_fraction(18, 20) == pytest.approx(0.9) and E.stage_fraction(19, 20) > 0.9
    assert E.remaining(5, 20) == 15 and E.remaining(30, 20) >= 2.0  # an overrun never reports "0 s left"


def test_estimates_converge_on_the_real_speed_after_a_few_clips(app_client, env):
    # the fake engine is 1.6x slower than the built-in Chatterbox figure; each finished clip teaches the estimator
    real_rtf = 8.0
    env.audio_seconds = lambda text: len(text) / 12.0
    env.speak_seconds = lambda route, text: real_rtf * len(text) / 12.0
    env.translate_seconds = 4.0
    from routers import say_it_once as sio
    sio.ESTIMATOR.touch("chatterbox_ml", env.clock)  # the model is loaded already: only the speed is unknown
    errors = []
    for _ in range(7):
        r = post(app_client, ["hi"])
        estimate, t0 = r.json()["est_total_s"], env.clock
        env.run_jobs()
        errors.append(abs(estimate - (env.clock - t0)) / (env.clock - t0))
    assert errors[0] > 0.2 and errors[-1] < 0.1, errors   # starts off by ~30 %, ends within 10 %
    assert sio.ESTIMATOR.rtf("chatterbox_ml") > 7
    assert errors[-1] < errors[0]


def test_estimate_endpoint_lists_languages_in_run_order_with_totals(app_client, env):
    r = app_client.post("/api/say-it-once/estimate", headers=LOCAL, json={
        "voice_id": VOICE, "text": SENTENCE, "languages": ["mr", "hi", "fr", "en"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["speed"] == "own"
    assert [i["language"] for i in body["items"]] == ["en", "fr", "hi", "mr"] or \
        [i["language"] for i in body["items"]] == ["fr", "en", "hi", "mr"]
    est = {i["language"]: i["est_s"] for i in body["items"]}
    assert est["en"] < est["hi"] < est["mr"]  # same language: no translation; Chatterbox < Indic-Speak
    assert body["total_s"] == sum(est.values()) or abs(body["total_s"] - sum(est.values())) <= 2
    assert env.llm_calls == [] and env.jobs == []  # nothing translated, nothing queued


def test_estimate_follows_the_learned_speed(app_client, env):
    from routers import say_it_once as sio
    long = " ".join([SENTENCE] * 3)
    ask = lambda: app_client.post("/api/say-it-once/estimate", headers=LOCAL, json={
        "voice_id": VOICE, "text": long, "languages": ["mr"]}).json()["items"][0]["est_s"]
    before = ask()
    for _ in range(6):
        sio.ESTIMATOR.record_voice("indic_speak", "mr", long, voice_s=30.0, audio_s=15.0)  # a warm clip at RTF 2
    after = ask()
    assert after < before * 0.6


def test_estimate_keeps_the_consent_guard(app_client, env):
    for vid, code in (("voice-no-consent", 403), ("voice-of-someone-else", 404)):
        r = app_client.post("/api/say-it-once/estimate", headers=LOCAL, json={
            "voice_id": vid, "text": SENTENCE, "languages": ["fr"]})
        assert r.status_code == code


# ---- sequential, fast engines first ---------------------------------------------------------------------------

def test_languages_run_one_at_a_time_fastest_engine_first(app_client, env):
    items, body = run_to_end(app_client, env, ["mr", "ta", "hi", "fr"])
    assert env.max_active == 1  # two heavy engines never run at once
    assert [s[2] for s in env.spoken] == ["fr", "hi", "mr", "ta"]  # Qwen3, Chatterbox, Indic-Speak, Indic-Speak
    starts = [items[l]["started_at"] for l in ("fr", "hi", "mr", "ta")]
    assert starts == sorted(starts) and len(set(starts)) == 4  # each starts after the one before it


def test_kokoro_goes_before_the_cloning_engines(app_client, env):
    items, _ = run_to_end(app_client, env, ["hi", "fr", "mr"], speed="fast")
    assert [s[1] for s in env.spoken] == ["kokoro", "kokoro", "indic_speak"]


# ---- cancel ---------------------------------------------------------------------------------------------------

def test_cancel_stops_the_remaining_languages_and_keeps_the_finished_one(app_client, env):
    job_id = post(app_client, ["mr", "hi", "fr"]).json()["job_id"]
    seen = {}

    def hook(language):
        if language == "fr" and "cancel" not in seen:  # in the middle of the first clip
            r = app_client.post(f"/api/say-it-once/{job_id}/cancel", headers=LOCAL)
            seen["cancel"] = r.json()
    env.during_speak = hook
    env.run_jobs()
    assert seen["cancel"]["cancelling"] is True and seen["cancel"]["state"] == "running"
    final = app_client.get(f"/api/say-it-once/{job_id}", headers=LOCAL).json()
    by = {i["language"]: i for i in final["items"]}
    assert final["state"] == "cancelled" and final["cancelling"] is False
    assert by["fr"]["status"] == "done" and by["fr"]["audio_url"]  # the clip being spoken finishes and is kept
    assert by["hi"]["status"] == by["mr"]["status"] == "cancelled" and by["hi"]["stage"] == "cancelled"
    assert by["hi"]["audio_url"] is None and by["mr"]["queue_position"] is None
    assert [s[2] for s in env.spoken] == ["fr"]
    assert final["eta_s"] == 0 and final["done"] == 1


def test_cancel_before_anything_started(app_client, env):
    job_id = post(app_client, ["fr", "hi"]).json()["job_id"]
    assert app_client.post(f"/api/say-it-once/{job_id}/cancel", headers=LOCAL).status_code == 200
    env.run_jobs()
    final = app_client.get(f"/api/say-it-once/{job_id}", headers=LOCAL).json()
    assert final["state"] == "cancelled" and env.spoken == [] and env.llm_calls == []
    assert [i["status"] for i in final["items"]] == ["cancelled", "cancelled"]


def test_cancel_while_translating_drops_that_language_without_speaking_it(app_client, env):
    job_id = post(app_client, ["fr", "es"]).json()["job_id"]
    orig = env.llm_answers["fr"]

    from routers import say_it_once as sio

    real_chat = sio.llm.chat

    def chat_then_cancel(messages, **kw):
        out = real_chat(messages, **kw)
        sio._runs[job_id]["cancel"] = True  # cancelled while the model was answering
        return out
    sio.llm.chat = chat_then_cancel
    try:
        env.run_jobs()
    finally:
        sio.llm.chat = real_chat
    final = app_client.get(f"/api/say-it-once/{job_id}", headers=LOCAL).json()
    assert final["state"] == "cancelled" and env.spoken == []
    assert {i["status"] for i in final["items"]} == {"cancelled"}
    assert orig  # (the translation existed; it was simply never spoken)


def test_cancel_a_finished_job_changes_nothing_and_is_owner_only(app_client, env):
    from routers import say_it_once as sio
    job_id = post(app_client, ["fr"]).json()["job_id"]
    env.run_jobs()
    r = app_client.post(f"/api/say-it-once/{job_id}/cancel", headers=LOCAL)
    assert r.status_code == 200 and r.json()["state"] == "done"
    sio._runs[job_id]["user_id"] = "someone-else"
    assert app_client.post(f"/api/say-it-once/{job_id}/cancel", headers=LOCAL).status_code == 404
    assert app_client.post("/api/say-it-once/nope/cancel", headers=LOCAL).status_code == 404


# ---- re-speak one language from an edited translation ----------------------------------------------------------

def test_respeak_one_language_uses_the_approved_text_and_nothing_else(app_client, env):
    edited = "Votre rendez-vous est bien confirmé pour jeudi à quatre heures."
    items, body = run_to_end(app_client, env, ["fr"], translations={"fr": edited})
    assert list(items) == ["fr"] and body["total"] == 1
    assert env.llm_calls == []  # no model involved: the user's text is spoken as written
    assert env.spoken == [("your_voice", "qwen3_0_6b", "fr", edited)]
    assert items["fr"]["text"] == edited and items["fr"]["receipt_id"]  # new clip, new receipt


def test_respeak_keeps_the_guards_and_the_speed(app_client, env):
    r = post(app_client, ["hi"], speed="fast", translations={"hi": "Aapki appointment Guruvaar ko"})
    assert r.status_code == 422 and r.json()["code"] == "INVALID_TRANSLATION"  # script check still applies
    assert post(app_client, ["hi"], voice_id="voice-no-consent", translations={"hi": TRANSLATIONS["hi"]}).status_code == 403
    items, _ = run_to_end(app_client, env, ["hi"], speed="fast", translations={"hi": TRANSLATIONS["hi"]})
    assert (items["hi"]["engine"], items["hi"]["voice_kind"]) == ("kokoro", "preset_voice")


# ---- fast mode: preset voices only ---------------------------------------------------------------------------

def test_fast_mode_uses_kokoro_where_it_speaks_and_indic_speak_elsewhere(app_client, env):
    langs = ["en", "hi", "fr", "es", "it", "pt", "ja", "zh", "mr", "ta"]
    for chunk in (langs[:4], langs[4:8], langs[8:]):
        items, _ = run_to_end(app_client, env, chunk, speed="fast", **{})
        assert all(i["voice_kind"] == "preset_voice" for i in items.values())
        for code, it in items.items():
            expected = "kokoro" if code in ("en", "hi", "fr", "es", "it", "pt", "ja", "zh") else "indic_speak"
            assert it["engine"] == expected, (code, it["engine"])
    assert not any(kind == "your_voice" for kind, *_ in env.spoken)


def test_fast_mode_languages_no_preset_voice_speaks_are_unavailable_with_a_reason(app_client, env):
    items, body = run_to_end(app_client, env, ["de", "ko", "fr"], speed="fast")
    for code in ("de", "ko"):
        assert items[code]["status"] == "unavailable" and "Your voice" in items[code]["error"]
    assert items["fr"]["status"] == "done" and body["total"] == 1


def test_fast_mode_without_indic_speak_marks_those_languages_unavailable(app_client, env):
    env.indic = False
    items, _ = run_to_end(app_client, env, ["mr", "hi"], speed="fast")
    assert items["mr"]["status"] == "unavailable" and items["hi"]["engine"] == "kokoro"


def test_own_mode_is_unchanged_by_the_speed_option(app_client, env):
    items, _ = run_to_end(app_client, env, ["fr", "hi"], speed="own")
    assert (items["fr"]["engine"], items["hi"]["engine"]) == ("qwen3_0_6b", "chatterbox_ml")
    items, _ = run_to_end(app_client, env, ["fr"])  # the default is "own"
    assert items["fr"]["engine"] == "qwen3_0_6b"


def test_fast_is_quicker_than_your_voice_in_the_estimate(app_client, env):
    def total(speed):
        return app_client.post("/api/say-it-once/estimate", headers=LOCAL, json={
            "voice_id": VOICE, "text": SENTENCE, "languages": ["hi", "fr", "es"], "speed": speed}).json()["total_s"]
    assert total("fast") < total("own") / 2


def test_translate_for_review_follows_the_speed(app_client, env):
    r = app_client.post("/api/say-it-once/translate", headers=LOCAL, json={
        "voice_id": VOICE, "text": SENTENCE, "languages": ["hi", "fr"], "speed": "fast"})
    items = {i["language"]: i for i in r.json()["items"]}
    assert items["hi"]["engine"] == "kokoro" and items["fr"]["voice"] == "preset"


@pytest.mark.parametrize("route", ["/api/say-it-once", "/api/say-it-once/translate", "/api/say-it-once/estimate"])
def test_an_unknown_speed_is_refused(app_client, env, route):
    r = app_client.post(route, headers=LOCAL, json={"voice_id": VOICE, "text": SENTENCE, "languages": ["fr"],
                                                    "speed": "turbo"})
    assert r.status_code == 422 and r.json()["code"] == "INVALID_SPEED"
    assert env.jobs == []


# ---- cold starts: the first clip of an engine also pays for loading its model --------------------------------

def test_cold_engine_costs_its_load_time_and_the_cost_is_learned_separately(tmp_path):
    from services.speech_estimate import DEFAULT_LOAD_S, WARM_WINDOW_S, Estimator
    e = Estimator(path=str(tmp_path / "e.json"))
    text = "x" * 120
    warm = e.voice_seconds("indic_speak", text, "hi", warm=True)
    assert e.voice_seconds("indic_speak", text, "hi", warm=False) == pytest.approx(warm + DEFAULT_LOAD_S["indic_speak"])
    assert not e.is_warm("indic_speak", 1000.0)
    e.touch("indic_speak", 1000.0)
    assert e.is_warm("indic_speak", 1000.0 + WARM_WINDOW_S - 1) and not e.is_warm("indic_speak", 1000.0 + WARM_WINDOW_S + 1)
    # a cold clip: 10 s of speech took 70 + 40 s to make -> the extra 40 s is the load, the speed is untouched
    e.record_voice("indic_speak", "hi", text, voice_s=110.0, audio_s=10.0, cold=True)
    assert e.rtf("indic_speak") == 7.0 and e.samples("indic_speak") == 0
    assert 30 < e.load_seconds("indic_speak") < 40
    # a warm clip teaches the speed and leaves the load alone
    load = e.load_seconds("indic_speak")
    e.record_voice("indic_speak", "hi", text, voice_s=50.0, audio_s=10.0, cold=False)
    assert e.rtf("indic_speak") < 7.0 and e.load_seconds("indic_speak") == load
    assert Estimator(path=str(tmp_path / "e.json")).load_seconds("indic_speak") == pytest.approx(load)


def test_the_estimate_includes_loading_only_for_engines_that_are_not_loaded_yet(app_client, env):
    from routers import say_it_once as sio
    ask = lambda langs: {i["language"]: i["est_s"] for i in app_client.post(
        "/api/say-it-once/estimate", headers=LOCAL, json={"voice_id": VOICE, "text": SENTENCE, "languages": langs}
    ).json()["items"]}
    cold = ask(["mr", "ta"])
    assert cold["mr"] - cold["ta"] >= 20  # both use Indic-Speak: only the first one in line pays for loading it
    sio.ESTIMATOR.touch("indic_speak", env.clock)
    warm = ask(["mr"])
    assert warm["mr"] <= cold["mr"] - 20
    env.clock += 400  # idle for longer than the model manager keeps a model
    assert ask(["mr"])["mr"] >= warm["mr"] + 20


def test_a_clip_is_flagged_slow_only_once_it_overruns_its_own_estimate(app_client, env):
    env.speak_seconds = lambda route, text: 60.0  # Qwen3 French, cold, is expected to take about 12 s
    post(app_client, ["fr"])
    env.run_jobs()
    flags = [snap["items"][0]["slow"] for label, snap in env.snaps if label.startswith("speaking:")]
    assert flags[0] is False and flags[-1] is True  # at first on schedule, later "taking longer than expected"
    assert any(snap["slow"] for label, snap in env.snaps if label.startswith("speaking:"))
    env.snaps.clear()
    env.speak_seconds = lambda route, text: 6.0
    post(app_client, ["fr"])
    env.run_jobs()
    assert not any(snap["items"][0]["slow"] for _, snap in env.snaps)


# ---- DhVaani: your own voice in the Indian languages -------------------------------------------------------------

CREDIT = "Voice cloning in Indian languages by DhVaani from ARTPARK-IISc."


def test_capabilities_report_your_voice_for_the_indian_languages_when_dhvaani_is_available(app_client, env):
    env.dhvaani = True
    caps = {c["language"]: c for c in app_client.get(
        "/api/say-it-once/capabilities", params={"voice_id": VOICE}, headers=LOCAL).json()["languages"]}
    for code in ("hi", "mr", "ta", "bn", "te", "gu", "kn", "ml", "pa", "or"):
        c = caps[code]
        assert (c["voice_kind"], c["engine"], c["available"]) == ("your_voice", "dhvaani", True), code
        assert c["engine_name"] == "DhVaani-0.5 (ARTPARK-IISc)" and c["attribution"] == CREDIT, code
        assert c["est_seconds"] > 0 and c["fast"]["voice_kind"] == "preset_voice", code  # fast stays on presets
    assert caps["fr"]["engine"] == "qwen3_0_6b" and caps["fr"]["attribution"] is None  # Qwen3 keeps its languages
    assert caps["en"]["engine"] == "qwen3_0_6b"


def test_capabilities_without_dhvaani_are_unchanged(app_client, env):
    caps = {c["language"]: c for c in app_client.get(
        "/api/say-it-once/capabilities", params={"voice_id": VOICE}, headers=LOCAL).json()["languages"]}
    assert caps["hi"]["engine"] == "chatterbox_ml"  # the Hindi fallback
    for code in ("mr", "ta", "bn", "te"):
        assert (caps[code]["voice_kind"], caps[code]["engine"]) == ("preset_voice", "indic_speak"), code


def test_own_mode_speaks_in_your_voice_with_dhvaani_and_credits_it(app_client, env):
    env.dhvaani = True
    items, body = run_to_end(app_client, env, ["en", "hi", "mr", "ta"], speed="own")
    assert body["state"] == "done"
    assert {l: (i["voice_kind"], i["engine"]) for l, i in items.items()} == {
        "en": ("your_voice", "qwen3_0_6b"),
        "hi": ("your_voice", "dhvaani"), "mr": ("your_voice", "dhvaani"), "ta": ("your_voice", "dhvaani")}
    assert items["mr"]["attribution"] == CREDIT and items["hi"]["attribution"] == CREDIT
    assert items["en"]["attribution"] is None


def test_own_mode_falls_back_to_presets_when_dhvaani_is_missing(app_client, env):
    items, _ = run_to_end(app_client, env, ["mr", "ta"], speed="own")
    assert {i["engine"] for i in items.values()} == {"indic_speak"}
    assert all(i["voice_kind"] == "preset_voice" for i in items.values())


def test_fast_mode_never_loads_dhvaani_even_when_it_is_available(app_client, env):
    env.dhvaani = True
    items, _ = run_to_end(app_client, env, ["hi", "mr"], speed="fast")
    assert (items["hi"]["voice_kind"], items["hi"]["engine"]) == ("preset_voice", "kokoro")
    assert (items["mr"]["voice_kind"], items["mr"]["engine"]) == ("preset_voice", "indic_speak")
    assert all(s[1] != "dhvaani" for s in env.spoken)


def test_dhvaani_runs_after_qwen3_and_before_indic_speak(app_client, env):
    env.dhvaani = True
    env.indic = True
    r = post(app_client, ["mr", "fr", "hi"], speed="own")
    assert [i["language"] for i in r.json()["items"]][0] == "fr"  # Qwen3 (fastest) first, then DhVaani
    env.run_jobs()
    assert [s[2] for s in env.spoken][0] == "fr"


def test_estimate_uses_the_dhvaani_prior(app_client, env):
    env.dhvaani = True
    r = app_client.post("/api/say-it-once/estimate", headers=LOCAL, json={
        "voice_id": VOICE, "text": SENTENCE, "languages": ["mr"], "speed": "own"})
    assert r.status_code == 200, r.text
    item = r.json()["items"][0]
    assert item["engine"] == "dhvaani" and item["available"] is True and item["est_s"] > 0
