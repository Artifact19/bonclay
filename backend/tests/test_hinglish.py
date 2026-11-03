"""Hinglish mode: detection, Roman-Hindi to Devanagari conversion (model + dictionary), the endpoints, and a
Hinglish agent turn that is answered in Hinglish but spoken through the Hindi voice. The language model is faked."""
import json

import numpy as np
import pytest

from services import hinglish, llm
from test_offline_agents import LOCAL, PERSONA, env, make_agent  # noqa: F401  (env is a fixture)


@pytest.fixture(autouse=True)
def _fresh_cache():
    hinglish._cache.clear()
    yield
    hinglish._cache.clear()


def fake_llm(monkeypatch, answer=None, calls=None, exc=None):
    """Replace llm.chat: `answer` is the reply text, or `exc` is raised."""
    def chat(messages, **kw):
        if calls is not None:
            calls.append(messages)
        if exc is not None:
            raise exc
        return llm.ChatResult(answer, "fake", "fake-model", 3, {})
    monkeypatch.setattr(llm, "chat", chat)


# ------------------------------------------------------------------------------------------------ detection

@pytest.mark.parametrize("text", [
    "kal subah aana",
    "Bhai mujhe jaldi batao, price kitna hai?",
    "mera order abhi tak deliver nahi hua",
    "aap kaise ho",
    "theek hai, kal milte hain",
    "mujhe appointment chahiye",
    "Aapka refund 3 din mein aa jayega",
    "kya hoga",
    "namaste",
])
def test_looks_hinglish_true(text):
    assert hinglish.looks_hinglish(text) is True


@pytest.mark.parametrize("text", [
    "I will call you tomorrow",
    "Please call me back about my order",
    "What is the refund window for this product?",
    "Hi, how are you doing today?",
    "नमस्ते, आप कैसे हैं?",
    "कल सुबह आना",
    "",
    "   ",
    "12345",
    "main street is closed",
])
def test_looks_hinglish_false(text):
    assert hinglish.looks_hinglish(text) is False


# ------------------------------------------------------------------------------------------------ dictionary

def test_dictionary_is_big_and_correct():
    assert len(hinglish.DICTIONARY) >= 300
    d = hinglish.DICTIONARY
    assert d["theek"] == "ठीक" and d["appointment"] == "अपॉइंटमेंट" and d["refund"] == "रिफंड"
    assert d["subsidy"] == "सब्सिडी" and d["price"] == "प्राइस" and d["doctor"] == "डॉक्टर"


def test_dictionary_fallback_converts_casual_hinglish():
    f = hinglish.dictionary_to_devanagari
    assert f("kal subah aana") == "कल सुबह आना"
    assert f("Bhai mujhe jaldi batao, price kitna hai?") == "भाई मुझे जल्दी बताओ, प्राइस कितना है?"
    assert f("mera appointment cancel kar do") == "मेरा अपॉइंटमेंट कैंसल कर दो"
    assert f("doctors ki fees 500 hai") == "डॉक्टर्स की फीस 500 है"  # plural loanword, digits untouched


def test_unknown_words_use_the_rules_without_virama_habits():
    w = hinglish.convert_word
    assert w("kitnaa") == "कितना"
    for word in ("mehnat", "rahul", "pyaar", "sundar", "samosa", "paneer", "hamara"):
        out = w(word)
        assert not out.endswith("्"), (word, out)
        assert not any(c.isascii() and c.isalpha() for c in out)
    assert w("kunal") == "कुनल"  # inherent schwa dropped at the end, no halant
    assert w("pyaar") == "प्यार" and w("sundar") == "सुंदर" and w("mehnat") == "मेहनत"
    assert w("badla") == "बदला"  # final "a" is a long vowel


# ------------------------------------------------------------------------------------------------ to_devanagari

def test_good_llm_answer_is_used(monkeypatch):
    calls = []
    fake_llm(monkeypatch, "कल सुबह आना", calls)
    r = hinglish.to_devanagari("kal subah aana")
    assert r == {"text": "कल सुबह आना", "method": "llm", "provider": "fake"}
    msgs = calls[0]
    assert msgs[0]["role"] == "system" and "transliteration tool" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "kal subah aana"}
    assert len(msgs) == 1 + 2 * len(hinglish._SHOTS) + 1  # system + few-shot pairs + the text


def test_known_words_are_pinned_to_the_dictionary(monkeypatch):
    fake_llm(monkeypatch, "भाई थीक है मेरा फॉर्म भर दो")  # the model wrote थीक
    r = hinglish.to_devanagari("bhai theek hai mera form bhar do")
    assert r["method"] == "llm" and r["text"] == "भाई ठीक है मेरा फॉर्म भर दो"


def test_leftover_latin_is_fixed(monkeypatch):
    fake_llm(monkeypatch, "भाई मेरा फॉर्म भarna है और कल सुबह मेरा ऑर्डर deliver नहीं हुआ बिल्कुल पक्का")
    r = hinglish.to_devanagari("bhai mera form bharna hai aur kal subah mera order deliver nahi hua bilkul pakka")
    assert r["method"] == "llm"
    assert not any(c.isascii() and c.isalpha() for c in r["text"])
    assert "भरना" in r["text"] and "डिलीवर" in r["text"]


def test_numbers_stay_digits(monkeypatch):
    fake_llm(monkeypatch, "रिफंड ३ से ५ दिन में आएगा")
    r = hinglish.to_devanagari("refund 3 se 5 din mein aayega")
    assert r["method"] == "llm" and r["text"] == "रिफंड 3 से 5 दिन में आएगा"


def test_english_words_do_not_override_the_model(monkeypatch):
    fake_llm(monkeypatch, "कृपया मुझे कॉल मी करें")  # "me" stays whatever the model decided
    r = hinglish.to_devanagari("kripya mujhe call me karen")
    assert "मी" in r["text"] and "में" not in r["text"]


def test_chatbot_answer_falls_back_to_dictionary(monkeypatch):
    fake_llm(monkeypatch, "मुझे खेद है, मैं इसमें आपकी मदद नहीं कर सकता।")
    r = hinglish.to_devanagari("kal subah aana")
    assert r == {"text": "कल सुबह आना", "method": "dictionary", "provider": None}


@pytest.mark.parametrize("answer", ["Sorry, I cannot help with that.", "", "kal subah aana",
                                    "कल सुबह आना जी मैं आपको सुबह बुलाऊँगा और फिर बाद में कुछ बताऊँगा"])
def test_invalid_answers_fall_back(monkeypatch, answer):
    fake_llm(monkeypatch, answer)
    r = hinglish.to_devanagari("kal subah aana")
    assert r["method"] == "dictionary" and r["text"] == "कल सुबह आना"


def test_no_llm_uses_the_dictionary(monkeypatch):
    fake_llm(monkeypatch, exc=llm.LLMUnavailable({"mlx": "no MLX environment"}))
    r = hinglish.to_devanagari("Bhai mujhe jaldi batao, price kitna hai?")
    assert r == {"text": "भाई मुझे जल्दी बताओ, प्राइस कितना है?", "method": "dictionary", "provider": None}


def test_llm_crash_uses_the_dictionary(monkeypatch):
    fake_llm(monkeypatch, exc=RuntimeError("boom"))
    assert hinglish.to_devanagari("kal aana")["method"] == "dictionary"


def test_use_llm_false_never_calls_the_model(monkeypatch):
    calls = []
    fake_llm(monkeypatch, "कल आना", calls)
    assert hinglish.to_devanagari("kal aana", use_llm=False)["method"] == "dictionary"
    assert calls == []


def test_results_are_cached(monkeypatch):
    calls = []
    fake_llm(monkeypatch, "कल सुबह आना", calls)
    a = hinglish.to_devanagari("kal subah aana")
    b = hinglish.to_devanagari("kal subah aana")
    assert a == b and len(calls) == 1
    b["text"] = "changed"  # callers get copies
    assert hinglish.to_devanagari("kal subah aana")["text"] == "कल सुबह आना"


def test_long_text_is_converted_in_chunks(monkeypatch):
    calls = []

    def chat(messages, **kw):
        calls.append(1)
        return llm.ChatResult(hinglish.dictionary_to_devanagari(messages[-1]["content"]), "fake", "m", 1, {})
    monkeypatch.setattr(llm, "chat", chat)
    text = " ".join(["kal subah aana hai mujhe bhi batao kitna price hai."] * 20)
    r = hinglish.to_devanagari(text)
    assert r["method"] == "llm" and len(calls) > 1
    assert r["text"].count("कल सुबह आना") == 20


# ------------------------------------------------------------------------------------------------ endpoints

def test_transliterate_endpoint_reports_the_method(app_client, monkeypatch):
    fake_llm(monkeypatch, "कल सुबह आना")
    r = app_client.post("/api/indic/transliterate", headers=LOCAL, json={"text": "kal subah aana"})
    assert r.status_code == 200
    assert r.json()["text"] == "कल सुबह आना" and r.json()["method"] == "llm" and r.json()["provider"] == "fake"

    fake_llm(monkeypatch, exc=llm.LLMUnavailable({"mlx": "off"}))
    r = app_client.post("/api/indic/transliterate", headers=LOCAL, json={"text": "mera order kab aayega"})
    assert r.json()["method"] == "dictionary" and r.json()["text"] == "मेरा ऑर्डर कब आएगा"


def test_transliterate_endpoint_keeps_formal_itrans(app_client, monkeypatch):
    calls = []
    fake_llm(monkeypatch, "x", calls)
    r = app_client.post("/api/indic/transliterate", headers=LOCAL, json={"text": "bhaarata"})
    assert r.status_code == 200 and r.json()["text"] == "भारत" and r.json()["method"] == "rules"
    assert r.json()["scheme"] == "itrans" and calls == []  # no model for formal input
    assert app_client.post("/api/indic/transliterate", headers=LOCAL, json={"text": " "}).status_code == 400


def test_detect_endpoint_flags_hinglish(app_client):
    r = app_client.post("/api/indic/detect", headers=LOCAL, json={"text": "kal subah aana"})
    assert r.status_code == 200 and r.json()["hinglish"] is True
    assert {"language", "confidence", "method", "is_indic"} <= set(r.json())
    for text in ("I will call you tomorrow", "नमस्ते आप कैसे हैं"):
        assert app_client.post("/api/indic/detect", headers=LOCAL, json={"text": text}).json()["hinglish"] is False


# ------------------------------------------------------------------------------------------------ agent turn

def test_reply_language_is_hinglish_for_roman_hindi(env):
    agent = {"supported_languages": ["en", "hi"]}
    assert env._reply_language("kal subah aana", agent) == "hinglish"
    assert env._reply_language("What is the refund window?", agent) == "en"
    assert env._reply_language("कल सुबह आना", agent) == "hi"
    assert "Roman letters" in env.LANGUAGE_NAMES["hinglish"]
    assert "Hinglish" in env._agent_system_prompt(agent, "ctx", "hinglish")
    assert env.ai_disclosure("hinglish") == "Yeh ek AI assistant hai."


def test_hinglish_turn_is_spoken_in_devanagari_with_the_hindi_voice(app_client, env, monkeypatch, ref_wav):
    spoken, routed, kokoro_langs = [], [], []

    def chat(messages, *, json_mode=False, **kw):
        if json_mode:
            return llm.ChatResult(json.dumps(PERSONA), "fake", "m", 1, {})
        if messages[0]["content"].startswith("You are a transliteration tool"):
            return llm.ChatResult("आपका रिफंड तीस दिन में आएगा।", "fake", "m", 1, {})
        return llm.ChatResult("Aapka refund tees din mein aayega.", "fake", "m", 1, {})
    monkeypatch.setattr(llm, "chat", chat)

    async def fake_asr(audio_bytes, name, language=None):
        return "mera refund kab milega"

    async def fake_load(key):
        return object()

    real_route = env.route_engine

    def spy_route(language, mode, clone_required):
        routed.append(language)
        return real_route(language, mode, clone_required)

    def fake_kokoro(model, text, lang="en"):
        spoken.append(text)
        kokoro_langs.append(lang)
        return np.zeros(2400, dtype=np.float32)

    monkeypatch.setattr(env, "_speech_to_text", fake_asr)
    monkeypatch.setattr(env, "_kokoro_generate", fake_kokoro)
    monkeypatch.setattr(env, "route_engine", spy_route)
    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
    monkeypatch.setattr(env, "indic_speak_service",
                        lambda: type("S", (), {"indic_speak_unavailable_reason": staticmethod(lambda: "off"),
                                               "MAX_TEXT_CHARS": 400})())

    a = make_agent(app_client, disclose_ai=False)
    r = app_client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL,
                        files={"audio": ("rec.wav", ref_wav.read_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["language"] == "hinglish"
    assert body["agent_text"] == "Aapka refund tees din mein aayega."  # shown as the agent wrote it
    assert spoken == ["आपका रिफंड तीस दिन में आएगा।"] and kokoro_langs == ["hi"]
    assert routed == ["hi"]
    assert body["engine"] == "kokoro" and body["fallback"] is False
    stored = env.local_db.select("conversation_turns", lambda row: row["role"] == "agent")
    assert stored[-1]["text_content"] == "Aapka refund tees din mein aayega." and stored[-1]["language"] == "hinglish"


def test_hinglish_turn_prompt_asks_for_hinglish(app_client, env, monkeypatch, ref_wav):
    seen = []

    def chat(messages, *, json_mode=False, **kw):
        if json_mode:
            return llm.ChatResult(json.dumps(PERSONA), "fake", "m", 1, {})
        seen.append(messages[-1]["content"])   # the caller's message: this turn's language lines are there
        return llm.ChatResult("Theek hai, kal aa jaiye.", "fake", "m", 1, {})
    monkeypatch.setattr(llm, "chat", chat)

    async def fake_asr(audio_bytes, name, language=None):
        return "kal aana hai kya"

    async def fake_load(key):
        return object()

    monkeypatch.setattr(env, "_speech_to_text", fake_asr)
    monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": np.zeros(2400, dtype=np.float32))
    monkeypatch.setattr(env.advanced_manager, "load_model", fake_load)
    a = make_agent(app_client)
    r = app_client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL,
                        files={"audio": ("rec.wav", ref_wav.read_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert any("Reply in Hinglish: Hindi written in Roman letters" in p for p in seen)
    assert r.json()["language"] == "hinglish"
