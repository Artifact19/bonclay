"""Answering-machine detection: clear greetings are caught, people are not."""
import pytest

from services import voicemail


@pytest.mark.parametrize("text", [
    "The person you are calling is not available. Please leave a message after the tone.",
    "Hi, you have reached the voicemail of Asha. Leave your message.",
    "The number you have dialled is currently switched off",
    "आप जिनसे संपर्क करना चाहते हैं उनका फोन स्विच ऑफ है",
    "कृपया बीप के बाद अपना संदेश छोड़ें",
])
def test_machine_greetings_are_detected(text):
    out = voicemail.classify(text, 3000)
    assert out["voicemail"] is True and out["reason"] == "phrase"


@pytest.mark.parametrize("text", ["Hello?", "Yes, who is this?", "हैलो, कौन बोल रहा है", "I'm not available right now, call later",
                                  "Haan boliye"])
def test_short_human_answers_are_not(text):
    assert voicemail.classify(text, 1500)["voicemail"] is False


def test_a_phrase_a_person_could_say_counts_only_after_a_long_first_utterance():
    text = "sorry the owner is not available at the moment"
    assert voicemail.classify(text, 2000)["voicemail"] is False
    assert voicemail.classify(text, 6000)["voicemail"] is True


def test_very_long_speech_alone_is_trusted_only_past_the_limit(monkeypatch):
    assert voicemail.classify("lorem ipsum dolor", 10000)["voicemail"] is False
    assert voicemail.classify("lorem ipsum dolor", 16000) == {"voicemail": True, "reason": "long_speech", "phrase": None}
    monkeypatch.setenv("VOICEMAIL_LONG_MS", "8000")
    assert voicemail.classify("lorem ipsum dolor", 10000)["voicemail"] is True


def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("VOICEMAIL_DETECT", "off")
    assert voicemail.enabled() is False
