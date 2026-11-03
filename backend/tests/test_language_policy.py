"""The reply-language policy (services/language_policy.py): detection of the caller's language and script, explicit
switch requests, the remembered choice, the reply check and its one regeneration, and the policy wired into the spoken
turn (blocking and streaming) with the language model faked."""
import json

import pytest

from services import language_policy as lp
from services import llm
from tests.test_offline_agents import LOCAL, PERSONA, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import FakeStream, names, parse, rig  # noqa: F401  (rig is a fixture)

EN_HI = ["en", "hi"]


# ------------------------------------------------------------------------------------------------ detection

@pytest.mark.parametrize("text,expected", [
    ("What are the clinic timings on Sunday?", "en"),
    ("क्लिनिक कितने बजे खुलता है?", "hi"),
    ("kal subah kitne baje aana hai", "hinglish"),
    ("Bhai fees kitni hai?", "hinglish"),
    ("கிளினிக் எத்தனை மணிக்கு திறக்கும்?", "ta"),
    ("క్లినిక్ ఎప్పుడు తెరుస్తారు", "te"),
    ("ক্লিনিক কখন খোলে", "bn"),
    ("Hello, I need an appointment", "en"),
])
def test_detect_language_of_a_caller_turn(text, expected):
    assert lp.detect_language(text, EN_HI) == expected


def test_devanagari_is_marathi_only_for_a_marathi_agent():
    assert lp.detect_language("आज किती वाजता उघडते", ["mr", "en"]) == "mr"
    assert lp.detect_language("आज किती वाजता उघडते", ["mr", "hi"]) == "hi"


def test_other_scripts_only_count_when_the_agent_supports_them():
    assert lp.detect_language("こんにちは、予約をしたいです", ["en"]) == "en"
    assert lp.detect_language("こんにちは、予約をしたいです", ["en", "ja"]) == "ja"
    assert lp.detect_language("Привет, мне нужна запись", ["en", "ru"]) == "ru"


def test_no_letters_falls_back_to_the_agents_first_language():
    assert lp.detect_language("", ["hi", "en"]) == "hi"
    assert lp.detect_language("9876543210", ["hi"]) == "hi"


# ------------------------------------------------------------------------------------------------ explicit requests

@pytest.mark.parametrize("text,code", [
    ("Tamil mein bolo", "ta"),
    ("Hindi mein baat karo please", "hi"),
    ("please speak English", "en"),
    ("Could you please speak in English?", "en"),
    ("Switch to Hindi now", "hi"),
    ("reply in Tamil", "ta"),
    ("answer in English please", "en"),
    ("In Hindi please.", "hi"),
    ("go back to English", "en"),
    ("हिंदी में बताइए", "hi"),
    ("अंग्रेज़ी में बोलो", "en"),
    ("आप Marathi बोलोगे?", "mr"),
    ("English bolo", "en"),
    ("aap English me baat kar sakte ho?", "en"),
    ("Bhai English mein baat karo, mujhe Hindi samajh nahi aati, ek dum bolo sab kuch English mein.", "en"),
    ("Teacher ji, aap Tamil mein bolo na, mujhe Tamil seekhni hai, Hindi band karo.", "ta"),
    ("डॉक्टर से बंगाली में बात करनी है, क्या आपका सिस्टम बंगाली समझता है?", "bn"),
    ("Hindi mein nahi, English mein baat karo", "en"),
    ("Wait, switch to Hindi now, I did not understand what you said in English just now.", "hi"),
])
def test_explicit_switch_requests(text, code):
    req = lp.find_switch_request(text, EN_HI)
    assert req is not None and req.code == code, req


@pytest.mark.parametrize("text", [
    "What are the clinic timings?",
    "Does the doctor speak Marathi?",
    "do you have a Hindi speaking doctor",
    "How do you say thanks in Hindi?",
    "don't speak Hindi",
    "mujhe Hindi mein appointment chahiye",
    "mujhe Hindi samajh nahi aati",
    "My English is not good",
    "मुझे कुछ समझ नहीं आ रहा, आप जो बोल रहे हो वो किस भाषा में है?",
])
def test_ordinary_mentions_are_not_requests(text):
    assert lp.find_switch_request(text, EN_HI) is None


def test_request_support_follows_the_agents_languages():
    assert lp.find_switch_request("Tamil mein bolo", EN_HI).supported is False
    assert lp.find_switch_request("Tamil mein bolo", ["en", "ta"]).supported is True
    assert lp.find_switch_request("speak English", ["hi"]).supported is False
    assert lp.find_switch_request("Hinglish mein bolo", EN_HI).supported is True  # Hindi covers Hinglish
    assert lp.find_switch_request("speak Hindi", ["en", "hinglish"]).supported is True


# ------------------------------------------------------------------------------------------------ the decision

def test_mirrors_the_latest_turn_without_a_choice():
    d = lp.decide("kal subah aana hai", EN_HI)
    assert (d.target, d.choice, d.reason) == ("hinglish", None, "script")
    assert lp.decide("When do you open?", EN_HI).target == "en"
    assert lp.decide("कल सुबह आना है", EN_HI).target == "hi"
    assert lp.decide("வணக்கம், நேரம் என்ன", ["en", "ta"]).target == "ta"


def test_a_supported_request_switches_and_is_remembered():
    d = lp.decide("English mein baat karo", EN_HI)
    assert d.target == "en" and d.choice == "en" and d.choice_changed and d.switched and d.reason == "explicit"
    # the next turn is typed in Hinglish: the stored choice wins until the caller asks for something else
    d2 = lp.decide("haan theek hai, kal subah aana hai", EN_HI, sticky="en")
    assert d2.target == "en" and d2.choice == "en" and not d2.choice_changed and d2.reason == "sticky"
    d3 = lp.decide("Tamil mein bolo", ["en", "hi", "ta"], sticky="en")
    assert d3.target == "ta" and d3.choice == "ta" and d3.choice_changed


def test_a_hindi_choice_keeps_the_callers_script():
    d = lp.decide("Switch to Hindi now", EN_HI)
    assert d.target == "hi" and d.choice == "hi"
    assert lp.decide("haan kal subah aana hai", EN_HI, sticky="hi").target == "hinglish"
    assert lp.decide("हाँ कल सुबह आना है", EN_HI, sticky="hi").target == "hi"
    assert lp.decide("Hindi mein baat karo", EN_HI).target == "hinglish"  # typed in Roman letters
    assert lp.decide("Hinglish mein bolo", EN_HI).target == "hinglish"


def test_an_unsupported_request_is_not_followed_and_not_stored():
    d = lp.decide("Tamil mein bolo", EN_HI)
    assert d.unsupported == "Tamil" and d.choice is None and not d.switched
    assert d.target == "hinglish"  # answered in the language it was asked in
    d = lp.decide("डॉक्टर से बंगाली में बात करनी है", EN_HI)
    assert d.unsupported == "Bengali" and d.target == "hi"
    # an earlier choice survives an unsupported request
    d = lp.decide("Tamil mein bolo", EN_HI, sticky="en")
    assert d.unsupported == "Tamil" and d.target == "en" and d.choice == "en" and not d.choice_changed


def test_ultra_short_turns_keep_the_language_of_the_last_reply():
    assert lp.decide("ok", EN_HI, previous_agent_text="Ji, clinic subah 9 baje khulta hai.").target == "hinglish"
    assert lp.decide("9876543210", EN_HI, previous_agent_text="कृपया अपना नंबर बताइए।").target == "hi"
    assert lp.decide("thanks", EN_HI, previous_agent_text="The clinic opens at nine.").target == "en"
    # a real sentence is never overridden by the history
    assert lp.decide("What time do you open on Monday?", EN_HI, previous_agent_text="कृपया बताइए।").target == "en"
    assert lp.decide("ok", EN_HI).target == "en"


# ------------------------------------------------------------------------------------------------ the reply check

@pytest.mark.parametrize("target,reply,ok", [
    ("en", "The clinic opens at nine in the morning.", True),
    ("en", "Clinic subah 9 baje khulta hai, aap aa sakte hain.", False),
    ("en", "क्लिनिक सुबह नौ बजे खुलता है।", False),
    ("hi", "क्लिनिक सुबह नौ बजे खुलता है।", True),
    ("hi", "Sunrise Clinic सुबह नौ बजे खुलता है।", True),
    ("hi", "The clinic opens at nine.", False),
    ("hinglish", "Clinic subah 9 baje khulta hai, aap aa sakte hain.", True),
    ("hinglish", "Ji, Dr. Meera Monday ko milengi.", True),
    ("hinglish", "क्लिनिक सुबह नौ बजे खुलता है।", False),
    ("hinglish", "The clinic opens at nine in the morning.", False),
    ("ta", "கிளினிக் காலை ஒன்பது மணிக்கு திறக்கும்.", True),
    ("ta", "The clinic opens at nine.", False),
    ("ta", "क्लिनिक सुबह नौ बजे खुलता है।", False),
    ("es", "La clínica abre a las nueve.", True),
    ("es", "क्लिनिक सुबह नौ बजे खुलता है।", False),
    ("en", "108", True),  # nothing to read a language from
])
def test_reply_matches(target, reply, ok):
    assert lp.reply_matches(target, reply) is ok


def test_enforce_regenerates_once_and_never_loses_the_reply():
    calls = []

    def regen_ok():
        calls.append(1)
        return "Clinic subah 9 baje khulta hai, aap aa sakte hain."

    first = "The clinic opens at nine."
    text, info = lp.enforce("hinglish", first, regen_ok)
    assert text.startswith("Clinic subah") and info == {"target": "hinglish", "matched": False, "retried": True,
                                                       "retry_matched": True} and len(calls) == 1
    text, info = lp.enforce("hinglish", "Clinic subah 9 baje khulta hai, aap aa sakte hain.", regen_ok)
    assert not info["retried"] and len(calls) == 1  # a matching reply costs nothing
    text, info = lp.enforce("hinglish", first, lambda: "Still English words only.")
    assert text == first and info["retried"] and info["retry_matched"] is False
    assert lp.enforce("hinglish", first, lambda: 1 / 0)[0] == first  # a failing retry keeps the first reply
    assert lp.enforce("hinglish", first, lambda: None)[0] == first


def test_prompt_text_states_the_policy():
    plain = lp.policy_prompt(None, EN_HI, "English")
    assert "You can speak English, Hindi and Hinglish" in plain
    assert "never switch language or script on your own" in plain
    assert "do not answer in the language they asked for" in plain
    d = lp.decide("Tamil mein bolo", EN_HI)
    note = lp.policy_prompt(d, EN_HI, "Hinglish")
    assert "asked for Tamil, which you cannot speak" in note
    d = lp.decide("please speak English", EN_HI)
    assert "just asked you to speak English" in lp.policy_prompt(d, EN_HI, "English")
    assert "only in Tamil" in lp.strong_instruction("Tamil")


# ------------------------------------------------------------------------------------------------ the spoken turn

class Script:
    """A scripted caller (speech recognition) and model for the blocking turn."""

    def __init__(self, env, monkeypatch, replies):
        self.lines: list = []
        self.replies = list(replies)
        self.prompts: list = []        # the system prompt of each reply
        self.user_msgs: list = []      # the caller's message of each reply (this turn's language lines are here)
        self.calls: list = []

        async def asr(audio_bytes, name, language=None):
            return self.lines.pop(0)

        async def load(key):
            return object()

        def chat(messages, *, json_mode=False, **kw):
            if json_mode:
                return llm.ChatResult(json.dumps(PERSONA), "fake", "m", 1, {})
            if "transliteration tool" in messages[0]["content"]:  # Hinglish spoken through the Hindi voice
                return llm.ChatResult("कल सुबह आइए", "fake", "m", 1, {})
            self.prompts.append(messages[0]["content"])
            self.user_msgs.append(messages[-1]["content"])
            self.calls.append(messages)
            return llm.ChatResult(self.replies.pop(0), "fake", "m", 1, {})

        monkeypatch.setattr(env, "_speech_to_text", asr)
        monkeypatch.setattr(env, "_kokoro_generate", lambda model, text, lang="en": __import__("numpy").zeros(2400, dtype="float32"))
        monkeypatch.setattr(env.advanced_manager, "load_model", load)
        monkeypatch.setattr(llm, "chat", chat)


def post_turn(client, agent_id, wav, conversation_id=None):
    data = {"conversation_id": conversation_id} if conversation_id else None
    r = client.post(f"/api/conversations/{agent_id}/turn", headers=LOCAL, data=data,
                    files={"audio": ("rec.wav", wav, "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_language_request_is_remembered_on_the_conversation(app_client, env, monkeypatch, ref_wav):
    s = Script(env, monkeypatch, ["Sure, I will speak English now. How can I help?", "The clinic opens at nine."])
    a = make_agent(app_client, disclose_ai=False)
    wav = ref_wav.read_bytes()
    s.lines = ["Bhai English mein baat karo please", "haan theek hai, subah kitne baje khulta hai"]
    first = post_turn(app_client, a["id"], wav)
    assert first["language"] == "en" and "just asked you to speak English" in s.user_msgs[0]
    conv = env.local_db.get("conversations", first["conversation_id"])
    assert conv["reply_language"] == "en"
    # turn 2 is typed in Hinglish, but the caller asked for English and has not asked for anything else
    second = post_turn(app_client, a["id"], wav, first["conversation_id"])
    assert second["language"] == "en" and "just asked you" not in s.user_msgs[1]
    assert second["language_check"]["matched"] is True


def test_an_unsupported_language_is_declined_in_the_callers_language(app_client, env, monkeypatch, ref_wav):
    s = Script(env, monkeypatch, ["Main Tamil nahi bol sakta, lekin Hindi ya English mein madad kar sakta hoon."])
    a = make_agent(app_client, disclose_ai=False)
    s.lines = ["Tamil mein bolo"]
    body = post_turn(app_client, a["id"], ref_wav.read_bytes())
    assert body["language"] == "hinglish"
    assert "asked for Tamil, which you cannot speak" in s.user_msgs[0]
    assert env.local_db.get("conversations", body["conversation_id"]).get("reply_language") is None


def test_a_reply_in_the_wrong_language_is_regenerated_once(app_client, env, monkeypatch, ref_wav):
    s = Script(env, monkeypatch, ["The clinic opens at nine in the morning.",
                                  "Clinic subah 9 baje khulta hai, aap aa sakte hain."])
    a = make_agent(app_client, disclose_ai=False)
    s.lines = ["clinic kitne baje khulta hai"]
    body = post_turn(app_client, a["id"], ref_wav.read_bytes())
    assert body["agent_text"].startswith("Clinic subah")
    assert body["language_check"] == {"target": "hinglish", "matched": False, "retried": True, "retry_matched": True}
    assert len(s.calls) == 2
    retry = s.calls[1]
    assert "not in the required language" in retry[0]["content"]
    assert retry[-2] == {"role": "assistant", "content": "The clinic opens at nine in the morning."}


def test_the_retry_is_capped_at_one_and_the_first_reply_is_kept(app_client, env, monkeypatch, ref_wav):
    s = Script(env, monkeypatch, ["The clinic opens at nine.", "It opens at nine o'clock.", "never asked for"])
    a = make_agent(app_client, disclose_ai=False)
    s.lines = ["clinic kitne baje khulta hai"]
    body = post_turn(app_client, a["id"], ref_wav.read_bytes())
    assert body["agent_text"] == "The clinic opens at nine."
    assert body["language_check"]["retried"] is True and body["language_check"]["retry_matched"] is False
    assert len(s.calls) == 2


def test_streaming_turn_holds_a_wrong_language_reply_and_regenerates(rig, monkeypatch):
    a = make_agent(rig.client, disclose_ai=False)
    rig.asr_text = "clinic kitne baje khulta hai"
    rig.stream = lambda: FakeStream(["The clinic opens at nine ", "in the morning on Monday. ", "See you then."])
    regen = []

    def chat(messages, **kw):
        if "transliteration tool" in messages[0]["content"]:
            return llm.ChatResult("क्लिनिक सुबह 9 बजे खुलता है", "fake", "m", 1, {})
        regen.append(messages)
        return llm.ChatResult("Clinic subah 9 baje khulta hai, aap aa sakte hain.", "fake", "m", 1, {})

    monkeypatch.setattr(llm, "chat", chat)
    events = parse(rig.post(a["id"]))
    spoken = [d["text"] for e, d in events if e == "sentence"]
    assert spoken == ["Clinic subah 9 baje khulta hai, aap aa sakte hain."]  # the English sentences were never spoken
    done = events[-1][1]
    assert done["language"] == "hinglish" and done["language_check"]["retry_matched"] is True
    assert len(regen) == 1 and "not in the required language" in regen[0][0]["content"]


def test_streaming_turn_in_the_right_language_is_untouched(rig, monkeypatch):
    a = make_agent(rig.client, disclose_ai=False)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("no regeneration expected"))
    events = parse(rig.post(a["id"]))
    assert names(events)[-1] == "done" and events[-1][1]["language_check"] == {
        "target": "en", "matched": True, "retried": False, "retry_matched": None}


# The doc QA evaluation (docs/DOC_QA_EVAL.md): college-18's correct Hindi reply was flagged because English programme
# names pushed the Devanagari share under 50%, and the reply was regenerated for nothing.
COLLEGE_18 = "B.Tech Mechanical Engineering में कुल 60 सीटें उपलब्ध हैं।"


@pytest.mark.parametrize("reply,ok", [
    (COLLEGE_18, True),
    ("Dr. Meera Kulkarni सोमवार, बुधवार और शुक्रवार को OPD में बैठती हैं।", True),
    ("MRI और CT scan की सुविधा Radiology में है।", True),            # "scan" is lower case: still well under half
    ("Dr. Meera Kulkarni Monday, Wednesday aur Friday ko milengi.", False),  # Hinglish, not Devanagari
    ("Dr. Meera Kulkarni is available on Monday, Wednesday and Friday.", False),
    ("जी, Dr. Meera Kulkarni is available on Monday, Wednesday and Friday from 9 to 1.", False),
])
def test_hindi_reply_with_english_names_and_terms(reply, ok):
    assert lp.reply_matches("hi", reply) is ok


def test_latin_words_the_caller_used_count_as_terms():
    reply = "हाँ, fees online भी जमा हो सकती है।"
    assert lp.reply_matches("hi", reply, known_text="fees online jama ho sakti hai kya?") is True
    assert lp.reply_matches("hi", "हाँ, you can pay the fees online here.", known_text="fees online?") is False


def test_enforce_keeps_college_18_without_a_regeneration():
    calls = []
    text, info = lp.enforce("hi", COLLEGE_18, lambda: calls.append(1) or "दोबारा")
    assert text == COLLEGE_18 and info == {"target": "hi", "matched": True, "retried": False, "retry_matched": None}
    assert calls == []


def test_backend_passes_only_what_the_caller_said_as_known_text(env):
    msgs = [{"role": "system", "content": "x"},
            {"role": "user", "content": "Context from the documents:\nThe fees are payable online.\n\nCaller says: MRI ka rate?"}]
    assert env._language_known_text(msgs).strip() == "MRI ka rate?"
    assert env._language_known_text([{"role": "assistant", "content": "hi"}]) == ""
