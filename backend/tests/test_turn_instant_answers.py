"""Instant answers inside the turn pipeline (backend._turn_event_stream and backend._turn_blocking): an approved answer is
spoken without retrieval, tools or the language model, through the normal sentence, voice and provenance path, and stored
like any other turn. Models are faked as in test_turn_stream.py; embeddings are the bag of words of test_offline_agents."""
import pytest

from services import instant_answers as ia
from tests.test_offline_agents import LOCAL, env, fake_embed, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import names, parse, rig  # noqa: F401  (rig is a fixture)

USER = "local-user"
FP = "fp-test"
QUESTION = "When does the clinic open?"
ANSWER = "We open at nine in the morning."
DISCLOSURE = "This is an AI assistant."


@pytest.fixture()
def answers(monkeypatch, env):
    """Bag-of-words embeddings, a fixed document fingerprint, and spies on retrieval and the matcher."""
    ia.set_embedder(lambda texts: fake_embed(list(texts)))
    ia.set_fingerprinter(lambda a, u, m: FP)
    seen = {"retrieval": 0, "match": 0}
    real_retrieve, real_match = env.retrieve_context, ia.match

    async def retrieve(*a, **k):
        seen["retrieval"] += 1
        return await real_retrieve(*a, **k)

    def match(*a, **k):
        seen["match"] += 1
        return real_match(*a, **k)
    monkeypatch.setattr(env, "retrieve_context", retrieve)
    monkeypatch.setattr(ia, "match", match)
    yield seen
    ia.set_embedder(None)
    ia.set_fingerprinter(None)


def approve(agent, question=QUESTION, answer=ANSWER, language="en"):
    row = ia.new_row(agent_id=agent["id"], user_id=USER, question=question, answer=answer, language=language,
                     source="owner", status=ia.STATUS_APPROVED, fingerprint=FP, embed=lambda t: fake_embed(list(t)))
    return ia.get_store("local").insert(row)


def agent_rows(env, conversation_id):
    return [r for r in env.local_db.select("conversation_turns", lambda r: r.get("conversation_id") == conversation_id)
            if r["role"] == "agent"]


def turn_chats(env):
    """Language model calls of turns (creating the agent drafts its persona with the model too)."""
    return [c for c in env.chat_calls if not c["json_mode"]]


def typed(rig, agent_id, text, conversation_id=None):
    data = {"text": text, **({"conversation_id": conversation_id} if conversation_id else {})}
    r = rig.client.post(f"/api/conversations/{agent_id}/turn", headers=LOCAL, data=data)
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------------------------- streaming turn

def test_a_hit_is_spoken_without_retrieval_or_the_model_and_stored_like_a_turn(rig, env, answers):
    a = make_agent(rig.client)
    row = approve(a)
    events = parse(rig.post(a["id"]))
    assert names(events) == ["transcript", "sentence", "sentence", "done"]
    assert [d["text"] for e, d in events if e == "sentence"] == [DISCLOSURE, ANSWER]   # the disclosure, as normal
    assert rig.asked == [] and answers["retrieval"] == 0                                # no model, no search
    done = events[-1][1]
    assert done["agent_text"] == f"{DISCLOSURE} {ANSWER}" and done["fallback"] is False
    assert done["llm_provider"] is None and done["retrieval"] is None and done["receipt_id"]
    assert done["instant_answer"]["id"] == row["id"] and done["instant_answer"]["audio_cached"] is False
    t = done["timings"]
    assert t["retrieval_ms"] == 0 and t["llm_ms"] == 0 and t["llm_first_token_ms"] == 0
    assert t["instant_answer_id"] == row["id"]
    [stored] = agent_rows(env, done["conversation_id"])
    assert stored["id"] == done["turn_id"] and stored["text_content"] == done["agent_text"]
    assert stored["fallback"] is False and "retrieval" not in stored
    assert stored["timings"]["instant_answer_id"] == row["id"] and stored["timings"]["llm_ms"] == 0
    from services import provenance
    assert provenance.get_receipt(done["receipt_id"])["turn_id"] == done["turn_id"]


def test_the_disclosure_is_only_on_the_first_reply_and_the_audio_is_rendered_once(rig, env, answers, monkeypatch):
    marked = []
    real_mark = env._watermark_clip

    def mark(wav, sr, turn_id):
        marked.append(turn_id)
        return real_mark(wav, sr, turn_id)
    monkeypatch.setattr(env, "_watermark_clip", mark)
    a = make_agent(rig.client)
    approve(a)
    first = parse(rig.post(a["id"]))[-1][1]
    second = parse(rig.post(a["id"], conversation_id=first["conversation_id"]))
    done = second[-1][1]
    assert [d["text"] for e, d in second if e == "sentence"] == [ANSWER]       # no second disclosure
    assert done["agent_text"] == ANSWER and done["instant_answer"]["audio_cached"] is True
    assert [t for t, _ in rig.spoken].count(ANSWER) == 1                       # synthesized once, for both turns
    assert marked.count(done["turn_id"]) == 1 and marked.count(first["turn_id"]) == 2   # every clip marked per turn
    assert len(agent_rows(env, first["conversation_id"])) == 2


def test_a_miss_takes_the_normal_path(rig, env, answers):
    a = make_agent(rig.client)
    approve(a)
    rig.asr_text = "Do you have parking near the hospital?"
    done = parse(rig.post(a["id"]))[-1][1]
    assert answers["match"] == 1 and len(rig.asked) == 1 and answers["retrieval"] == 1
    assert "instant_answer" not in done and "instant_answer_id" not in done["timings"]
    assert done["llm_provider"] == "fake"


def test_switched_off_the_old_path_runs_and_nothing_is_matched(rig, env, answers, monkeypatch):
    monkeypatch.setenv("INSTANT_ANSWERS", "off")
    a = make_agent(rig.client)
    approve(a)
    done = parse(rig.post(a["id"]))[-1][1]
    assert answers["match"] == 0 and len(rig.asked) == 1 and answers["retrieval"] == 1
    assert "instant_answer" not in done and done["llm_provider"] == "fake"


def seed_conversation(env, agent, agent_text, **agent_fields):
    conv = env.local_db.insert("conversations", {"agent_id": agent["id"], "user_id": USER})
    env.local_db.insert("conversation_turns", {"conversation_id": conv["id"], "role": "user", "text_content": "hi"})
    env.local_db.insert("conversation_turns", {"conversation_id": conv["id"], "role": "agent", "text_content": agent_text,
                                               **agent_fields})
    return conv["id"]


def test_no_instant_answer_while_the_agent_waits_for_a_detail(rig, env, answers):
    a = make_agent(rig.client)
    approve(a)
    conv = seed_conversation(env, a, "Sure, I can help with that. May I have your name, please?")
    done = parse(rig.post(a["id"], conversation_id=conv))[-1][1]
    assert answers["match"] == 0 and len(rig.asked) == 1 and "instant_answer" not in done


def test_no_instant_answer_once_a_tool_was_called_in_the_conversation(rig, env, answers):
    a = make_agent(rig.client)
    approve(a)
    conv = seed_conversation(env, a, "Done, your appointment is booked.",
                             tool_calls=[{"name": "book_appointment", "ok": True}])
    done = parse(rig.post(a["id"], conversation_id=conv))[-1][1]
    assert answers["match"] == 0 and len(rig.asked) == 1 and "instant_answer" not in done


def test_an_ordinary_earlier_reply_does_not_block_a_hit(rig, env, answers):
    a = make_agent(rig.client)
    approve(a)
    conv = seed_conversation(env, a, "Hello, this is the clinic. How can I help you today?")
    done = parse(rig.post(a["id"], conversation_id=conv))[-1][1]
    assert rig.asked == [] and done["agent_text"] == ANSWER                     # disclosed earlier: no disclosure now


@pytest.mark.parametrize("text,asks", [
    ("May I have your name, please?", True),
    ("Which doctor would you like to see?", True),
    ("What date works for you?", True),
    ("Please tell me your phone number.", True),
    ("Thank you. Could you confirm your mobile number?", True),
    ("कृपया अपना नाम बताइए।", True),
    ("आप किस तारीख को आना चाहेंगे?", True),
    ("Aapka naam kya hai?", True),
    ("How can I help you today?", False),
    ("The clinic opens at nine. Is there anything else?", False),
    ("We are open from 9 to 5.", False),
    ("", False),
    (None, False),
])
def test_the_detail_request_heuristic(env, text, asks):
    assert env._asks_for_detail(text) is asks


# ---------------------------------------------------------------------------------------------- blocking turn

def test_the_blocking_turn_uses_the_instant_answer_and_its_cached_audio(rig, env, answers):
    a = make_agent(rig.client)
    row = approve(a)
    body = typed(rig, a["id"], QUESTION)
    assert turn_chats(env) == [] and answers["retrieval"] == 0
    assert body["agent_text"] == f"{DISCLOSURE} {ANSWER}" and body["fallback"] is False
    assert body["instant_answer"]["id"] == row["id"] and body["instant_answer"]["audio_cached"] is False
    assert body["timings"]["llm_ms"] == 0 and body["timings"]["retrieval_ms"] == 0
    assert body["timings"]["instant_answer_id"] == row["id"] and body["receipt_id"]
    [stored] = agent_rows(env, body["conversation_id"])
    assert stored["timings"]["instant_answer_id"] == row["id"] and "retrieval" not in stored
    again = typed(rig, a["id"], QUESTION)                                       # a new conversation: same words
    assert again["instant_answer"]["audio_cached"] is True and again["receipt_id"] != body["receipt_id"]
    later = typed(rig, a["id"], QUESTION, conversation_id=body["conversation_id"])
    assert later["agent_text"] == ANSWER and turn_chats(env) == []


def test_the_blocking_turn_misses_like_before(rig, env, answers):
    a = make_agent(rig.client)
    approve(a)
    body = typed(rig, a["id"], "Do you have parking near the hospital?")
    assert len(turn_chats(env)) == 1 and "instant_answer" not in body and body["llm_provider"] == "fake"
