"""A typed question on POST /api/conversations/{agent}/turn (the Playground's text box): no recording, no speech
recognition, the same agent turn."""
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import rig  # noqa: F401  (rig is a fixture)


def test_a_typed_question_gets_a_reply_without_speech_recognition(rig):
    a = make_agent(rig.client)
    rig.asr_text = "THIS MUST NOT BE HEARD"
    r = rig.client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL,
                        data={"text": "  When are you open?  "})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_text"] == "When are you open?"
    turns = rig.env.local_db.select("conversation_turns", lambda t: t["conversation_id"] == body["conversation_id"])
    assert [t["text_content"] for t in turns if t["role"] == "user"] == ["When are you open?"]


def test_neither_text_nor_audio_is_refused(rig):
    a = make_agent(rig.client)
    r = rig.client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL, data={"text": "   "})
    assert r.status_code == 422 and r.json()["code"] == "NO_INPUT"


def test_a_very_long_typed_question_is_refused(rig):
    a = make_agent(rig.client)
    r = rig.client.post(f"/api/conversations/{a['id']}/turn", headers=LOCAL, data={"text": "x" * 1001})
    assert r.status_code == 422 and r.json()["code"] == "TEXT_TOO_LONG"
