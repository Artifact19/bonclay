"""Hinglish in the streaming turn: no model call sits in front of the first audio. The first sentence is converted to
Devanagari with the dictionary at once; later sentences may use a model rendering that was started when the sentence
was written, when it is ready within a short grace; the text shown stays Roman."""
import threading
import time

from services import hinglish
from tests.test_offline_agents import make_agent  # noqa: F401
from tests.test_offline_agents import env  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import FakeStream, names, parse, rig  # noqa: F401  (rig is a fixture)

REPLY = ["Aapka refund tees din mein aayega. ", "Aap kal subah aa jaiye. ", "Dhanyavaad."]


def setup_turn(rig, monkeypatch, convert):
    rig.asr_text = "mera refund kab milega"
    rig.stream = lambda: FakeStream(REPLY)
    calls = []

    def to_devanagari(text, *, use_llm=True, timeout=15.0):
        calls.append((text, threading.current_thread().name))
        return convert(text)

    monkeypatch.setattr(hinglish, "to_devanagari", to_devanagari)
    return calls


def run(rig):
    a = make_agent(rig.client, disclose_ai=False)
    events = parse(rig.post(a["id"]))
    return [d for e, d in events if e == "sentence"], events[-1][1]


def test_the_first_sentence_never_waits_for_a_model(rig, monkeypatch):
    calls = setup_turn(rig, monkeypatch, lambda t: {"text": "MODEL:" + t, "method": "llm", "provider": "fake"})
    sentences, done = run(rig)
    assert rig.spoken[0] == (hinglish.dictionary_to_devanagari("Aapka refund tees din mein aayega."), "hi")
    assert "Aapka refund tees din mein aayega." not in [c[0] for c in calls]
    assert sentences[0]["text"] == "Aapka refund tees din mein aayega."          # shown in Roman
    assert done["agent_text"] == "Aapka refund tees din mein aayega. Aap kal subah aa jaiye. Dhanyavaad."


def test_later_sentences_use_the_background_rendering_when_it_is_ready(rig, monkeypatch):
    calls = setup_turn(rig, monkeypatch, lambda t: {"text": "MODEL:" + t, "method": "llm", "provider": "fake"})
    run(rig)
    assert [lang for _, lang in rig.spoken] == ["hi", "hi", "hi"]
    assert [s for s, _ in rig.spoken[1:]] == ["MODEL:Aap kal subah aa jaiye.", "MODEL:Dhanyavaad."]
    assert all(thread.startswith("hinglish") for _, thread in calls)           # off the event loop and the model thread


def test_a_slow_rendering_is_not_waited_for(rig, monkeypatch):
    def slow(t):
        time.sleep(1.5)
        return {"text": "MODEL:" + t, "method": "llm", "provider": "fake"}

    setup_turn(rig, monkeypatch, slow)
    monkeypatch.setattr(rig.env, "HINGLISH_GRACE_S", 0.05)
    started = time.time()
    run(rig)
    assert [s for s, _ in rig.spoken[1:]] == [hinglish.dictionary_to_devanagari("Aap kal subah aa jaiye."),
                                              hinglish.dictionary_to_devanagari("Dhanyavaad.")]
    assert time.time() - started < 1.4            # the turn did not sit through the slow conversions one by one


def test_a_failing_rendering_falls_back_to_the_dictionary(rig, monkeypatch):
    def boom(t):
        raise RuntimeError("model down")

    setup_turn(rig, monkeypatch, boom)
    run(rig)
    assert rig.spoken[1][0] == hinglish.dictionary_to_devanagari("Aap kal subah aa jaiye.")


def test_no_second_model_call_while_the_local_model_holds_its_lock(rig, monkeypatch):
    calls = setup_turn(rig, monkeypatch, lambda t: {"text": "MODEL:" + t, "method": "llm", "provider": "fake"})

    def locked():
        s = FakeStream(REPLY)
        s.info = {"lock_held": True}
        return s

    rig.stream = locked
    run(rig)
    assert calls == []
    assert rig.spoken[1][0] == hinglish.dictionary_to_devanagari("Aap kal subah aa jaiye.")


def test_english_turns_start_no_conversion(rig, monkeypatch):
    calls = setup_turn(rig, monkeypatch, lambda t: {"text": t, "method": "llm", "provider": "fake"})
    rig.asr_text = "When does the clinic open?"
    rig.stream = lambda: FakeStream(["The clinic opens at nine. ", "It closes at eight."])
    run(rig)
    assert calls == []


def test_the_blocking_turn_still_converts_the_whole_reply_with_the_model(rig, monkeypatch):
    calls = setup_turn(rig, monkeypatch, lambda t: {"text": "MODEL:" + t, "method": "llm", "provider": "fake"})
    from tests.test_offline_agents import LOCAL
    from services import llm
    a = make_agent(rig.client, disclose_ai=False)
    monkeypatch.setattr(llm, "chat", lambda messages, **kw: llm.ChatResult("Aapka refund tees din mein aayega.",
                                                                          "fake", "m", 1, {}))
    r = rig.client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL,
                        files={"audio": ("rec.wav", rig.wav, "audio/wav")})
    assert r.status_code == 200, r.text
    assert rig.spoken[-1][0] == "MODEL:Aapka refund tees din mein aayega." and len(calls) == 1
