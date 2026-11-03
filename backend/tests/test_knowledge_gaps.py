"""Knowledge-gap report: gap detection (English, Hindi, Hinglish), redaction, clustering with a fake embedder, and
GET /api/analytics/knowledge-gaps (local mode, temp store, owner scoping)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from services import knowledge_gaps as kg
from tests.test_offline_agents import LOCAL, env  # noqa: F401  (env is a fixture)

NO_INFO = "I'm sorry, I don't have that information. I can connect you with a member of staff."


@pytest.fixture(autouse=True)
def fresh_cache():
    kg.clear_cache()
    yield
    kg.clear_cache()


# -- detection ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply", [
    NO_INFO,
    "Sorry, I do not have that information right now.",
    "I’m not sure about that, let me get a member of staff.",  # curly apostrophe
    "माफ कीजिए, मेरे पास यह जानकारी नहीं है।",  # Hindi
    "मुझे नहीं पता, मैं स्टाफ से पूछता हूं।",
    "Maaf kijiye, mere paas yeh jaankari nahi hai.",  # Hinglish
    "Mujhe nahi pata, main staff se baat karwata hoon.",
    "Lo siento, no tengo esa información.",
])
def test_gap_replies_are_detected(reply):
    assert kg.is_gap_reply(reply)


@pytest.mark.parametrize("reply", [
    "", "Refunds are possible within thirty days.", "We are open on Sundays from 10 to 1.",
    "रिफंड तीस दिन में मिल जाएगा।",
    "Aapka refund teen din mein aa jayega.",
])
def test_answers_are_not_gaps(reply):
    assert not kg.is_gap_reply(reply)


def pair(q, reply=NO_INFO, agent="a1", at="2026-10-10T10:00:00+00:00", fallback=False, language="en"):
    return {"agent_id": agent, "created_at": at, "question": q, "reply": reply, "fallback": fallback,
            "language": language}


def test_llm_outage_fallback_is_not_a_gap_and_not_counted():
    gaps, total = kg.find_gaps([pair("Do you do root canals?"),
                                pair("Are you open?", "I'm having trouble responding right now.", fallback=True),
                                pair("What are your hours?", "Nine to five.")])
    assert [g["question"] for g in gaps] == ["Do you do root canals?"] and total == 2


# -- redaction ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,gone", [
    ("call me on 9876543210 about fees", "9876543210"),
    ("my number is +91 98765 43210 please", "98765"),
    ("+91-9876543210 is mine", "9876543210"),
    ("Aadhaar 1234 5678 9012 is linked", "5678"),
    ("aadhaar 123456789012 hai", "123456789012"),
    ("mail ayush.c@example.co.in now", "example"),
    ("card 4111 1111 1111 1111 declined", "4111"),
    ("order 55512345 status", "55512345"),
])
def test_redact_removes_identifiers(raw, gone):
    out = kg.redact(raw)
    assert gone not in out and "[" in out


def test_redact_keeps_ordinary_numbers_and_truncates():
    assert kg.redact("Is it open on 15 August 2026 at 10:30?") == "Is it open on 15 August 2026 at 10:30?"
    assert len(kg.redact("word " * 500)) <= kg.MAX_TEXT_CHARS


def test_gap_questions_are_redacted_before_clustering():
    gaps, _ = kg.find_gaps([pair("Is 9876543210 registered? Also ayush@x.com")])
    assert "9876543210" not in gaps[0]["question"] and "@" not in gaps[0]["question"]


# -- clustering --------------------------------------------------------------------------------------------------

VEC = {"fee a": [1, 0, 0], "fee b": [0.95, 0.2, 0], "fee c": [0.9, 0.3, 0.1], "parking": [0, 1, 0], "x": [0, 0, 1]}


def vec_embed(texts):
    return np.array([VEC[t.lower()] for t in texts], dtype=np.float32)


def test_greedy_clusters_group_by_cosine_threshold():
    m = np.array([VEC["fee a"], VEC["parking"], VEC["fee b"], VEC["x"], VEC["fee c"]])
    assert kg.greedy_clusters(m, 0.75) == [[0, 2, 4], [1], [3]]


def test_clusters_count_representative_examples_language_and_order():
    gaps = [pair("fee a", at="2026-10-01T00:00:00+00:00", language="hi"),
            pair("fee b", at="2026-10-05T00:00:00+00:00", language="en"),
            pair("fee b", at="2026-10-06T00:00:00+00:00", language="en"),
            pair("fee c", at="2026-10-02T00:00:00+00:00", language="en"),
            pair("parking", at="2026-10-09T00:00:00+00:00"),
            pair("parking", at="2026-10-08T00:00:00+00:00", agent="a2")]  # another agent: never merged
    report = kg.build_report(gaps, {"a1": "Ria", "a2": "Dev"}, vec_embed, 30)
    assert report["total_gap_turns"] == 6 and report["window_days"] == 30
    top = report["clusters"][0]
    assert top["count"] == 4 and top["agent_id"] == "a1" and top["agent_name"] == "Ria"
    assert top["language"] == "en" and top["last_asked"] == "2026-10-06T00:00:00+00:00"
    assert top["question"] == "fee b"  # closest to the centre of fee a / b / c
    assert top["examples"][0] == "fee b" and len(top["examples"]) == 3 and len(set(top["examples"])) == 3
    assert sorted((c["agent_id"], c["count"]) for c in report["clusters"][1:]) == [("a1", 1), ("a2", 1)]


def test_no_embedder_falls_back_to_exact_text_groups():
    def broken(_):
        raise RuntimeError("model missing")

    for embed in (None, broken):
        out = kg.build_clusters([pair("Fee?"), pair("fee?  "), pair("Parking?")], {}, embed)
        assert sorted(c["count"] for c in out) == [1, 2]


def test_pairing_uses_the_user_turn_before_each_agent_turn():
    rows = [
        {"conversation_id": "c1", "role": "user", "text_content": "q1", "created_at": "2026-10-10T10:00:00+00:00"},
        {"conversation_id": "c1", "role": "agent", "text_content": "r1", "created_at": "2026-10-10T10:00:01+00:00"},
        {"conversation_id": "c1", "role": "user", "text_content": "q2", "created_at": "2026-10-10T10:01:00+00:00"},
        {"conversation_id": "c1", "role": "agent", "text_content": "r2", "created_at": "2026-10-10T10:01:01+00:00"},
        {"conversation_id": "other", "role": "user", "text_content": "q", "created_at": "2026-10-10T10:00:00+00:00"},
        {"conversation_id": "other", "role": "agent", "text_content": "r", "created_at": "2026-10-10T10:00:01+00:00"},
    ]
    pairs = kg.pair_turns(rows, {"c1": "a1"})
    assert [(p["question"], p["reply"]) for p in pairs] == [("q2", "r2"), ("q1", "r1")]


# -- endpoint ----------------------------------------------------------------------------------------------------

def seed(env, user, agent_id, qa, days_ago=1, language="en", fallback=False):
    """One conversation holding (question, reply) turns; returns the conversation id."""
    db = env.local_db
    conv = db.insert("conversations", {"agent_id": agent_id, "user_id": user})
    base = datetime.now(timezone.utc) - timedelta(days=days_ago)
    for i, (q, r) in enumerate(qa):
        t = (base + timedelta(minutes=i)).isoformat()
        db.insert("conversation_turns", {"conversation_id": conv["id"], "role": "user", "text_content": q,
                                         "created_at": t})
        db.insert("conversation_turns", {"conversation_id": conv["id"], "role": "agent", "text_content": r,
                                         "created_at": (base + timedelta(minutes=i, seconds=1)).isoformat(),
                                         "language": language, "fallback": fallback, "audio_storage_path": "local:x.wav"})
    return conv["id"]


def test_endpoint_empty_state(app_client, env):
    body = app_client.get("/api/analytics/knowledge-gaps", headers=LOCAL).json()
    assert body == {"clusters": [], "total_gap_turns": 0, "total_turns": 0, "window_days": 30}


def test_endpoint_shape_grouping_and_window(app_client, env):
    a = env.local_db.insert("agents", {"user_id": "local-user", "name": "Ria", "supported_languages": ["en"]})
    seed(env, "local-user", a["id"], [
        ("What is the dental whitening consultation fee?", NO_INFO),
        ("what is the dental whitening consultation fee please", NO_INFO),
        ("Do you have parking space?", NO_INFO),
        ("What are your hours?", "We are open nine to five."),
        ("Call me on 9876543210 about braces cost", NO_INFO),
    ])
    seed(env, "local-user", a["id"], [("Do you offer implants?", NO_INFO)], days_ago=45)  # outside 30 days
    body = app_client.get("/api/analytics/knowledge-gaps", headers=LOCAL).json()
    assert set(body) == {"clusters", "total_gap_turns", "total_turns", "window_days"}
    assert body["total_turns"] == 5 and body["total_gap_turns"] == 4 and body["window_days"] == 30
    assert sum(c["count"] for c in body["clusters"]) == 4
    first = body["clusters"][0]
    assert set(first) == {"question", "examples", "count", "last_asked", "agent_id", "agent_name", "language", "reason",
                         "reasons"}
    assert first["agent_id"] == a["id"] and first["agent_name"] == "Ria" and first["language"] == "en"
    assert first["count"] == 2 and 1 <= len(first["examples"]) <= 3  # the two whitening-fee questions
    assert "whitening" in first["question"].lower()
    blob = str(body)
    assert "9876543210" not in blob and "implants" not in blob and "audio" not in blob.lower()
    wide = app_client.get("/api/analytics/knowledge-gaps?days=60", headers=LOCAL).json()
    assert wide["total_gap_turns"] == 5


def test_endpoint_agent_filter_and_owner_scoping(app_client, env):
    mine = env.local_db.insert("agents", {"user_id": "local-user", "name": "Mine"})
    mine2 = env.local_db.insert("agents", {"user_id": "local-user", "name": "Mine2"})
    theirs = env.local_db.insert("agents", {"user_id": "someone-else", "name": "Theirs"})
    seed(env, "local-user", mine["id"], [("Do you sell gift cards?", NO_INFO)])
    seed(env, "local-user", mine2["id"], [("Is there wheelchair access?", NO_INFO)])
    seed(env, "someone-else", theirs["id"], [("Secret competitor question", NO_INFO)])
    body = app_client.get("/api/analytics/knowledge-gaps", headers=LOCAL).json()
    assert body["total_gap_turns"] == 2 and "Secret" not in str(body) and "Theirs" not in str(body)
    one = app_client.get(f"/api/analytics/knowledge-gaps?agent_id={mine['id']}", headers=LOCAL).json()
    assert [c["agent_name"] for c in one["clusters"]] == ["Mine"]
    foreign = app_client.get(f"/api/analytics/knowledge-gaps?agent_id={theirs['id']}", headers=LOCAL).json()
    assert foreign["clusters"] == [] and foreign["total_turns"] == 0


def test_endpoint_validates_days_and_caches(app_client, env, monkeypatch):
    assert app_client.get("/api/analytics/knowledge-gaps?days=0", headers=LOCAL).status_code == 422
    assert app_client.get("/api/analytics/knowledge-gaps?days=500", headers=LOCAL).status_code == 422
    a = env.local_db.insert("agents", {"user_id": "local-user", "name": "Ria"})
    seed(env, "local-user", a["id"], [("Do you sell gift cards?", NO_INFO)])
    first = app_client.get("/api/analytics/knowledge-gaps", headers=LOCAL).json()
    seed(env, "local-user", a["id"], [("Is there wheelchair access?", NO_INFO)])
    assert app_client.get("/api/analytics/knowledge-gaps", headers=LOCAL).json() == first  # served from the cache
    kg.clear_cache()
    assert app_client.get("/api/analytics/knowledge-gaps", headers=LOCAL).json()["total_gap_turns"] == 2


# -- retrieval signals: reasons and threshold ---------------------------------------------------------------------

ANSWER = "We are open nine to five."


def sig(n=3, top=0.6, **extra):
    return {"n_chunks": n, "top_score": top, "top_dense": top, "doc_ids": ["d1"], "model": "m", **extra}


def rpair(q, retrieval, reply=ANSWER, **kw):
    return {**pair(q, reply, **kw), "retrieval": retrieval}


def reasons(pairs):
    gaps, _ = kg.find_gaps(pairs)
    return [g["reason"] for g in gaps]


def test_reason_said_it_didnt_know_wins_over_the_scores():
    assert reasons([rpair("What is the whitening fee here", sig(top=0.1), reply=NO_INFO)]) == [kg.REASON_SAID]
    assert reasons([pair("What is the whitening fee here")]) == [kg.REASON_SAID]  # an old turn: no signals stored


def test_reason_no_matching_document_needs_an_agent_that_has_a_knowledge_base():
    pairs = [rpair("Do you offer teeth whitening", sig(n=0, top=None)),
             rpair("What time do you open", sig())]                      # the same agent found chunks on another turn
    assert reasons(pairs) == [kg.REASON_NONE]
    assert reasons([rpair("Do you offer teeth whitening", sig(n=0, top=None))]) == []  # no knowledge base at all


def test_reason_weak_match_below_threshold_only():
    assert reasons([rpair("Who won the cricket match", sig(top=0.30))]) == [kg.REASON_WEAK]
    assert reasons([rpair("Who won the cricket match", sig(top=0.35))]) == []     # at the threshold is a match
    assert reasons([rpair("What time do you open", sig(top=0.62))]) == []


def test_signals_ignore_failed_retrieval_short_questions_and_missing_scores():
    assert reasons([rpair("Who won the cricket match", sig(n=0, top=None, error=True)),
                    rpair("Who won the cricket match", sig())]) == []
    assert reasons([rpair("hello", sig(top=0.1)), rpair("thank you", sig(top=0.1))]) == []
    assert reasons([rpair("Who won the cricket match", {"n_chunks": 2, "top_score": None})]) == []
    assert reasons([rpair("Who won the cricket match", "garbage")]) == []


def test_llm_outage_is_not_a_gap_even_with_a_weak_match():
    assert reasons([rpair("Who won the cricket match", sig(top=0.1), fallback=True)]) == []


def test_threshold_comes_from_the_environment(monkeypatch):
    p = [rpair("Who won the cricket match", sig(top=0.45))]
    assert kg.min_score() == kg.DEFAULT_MIN_SCORE == 0.35 and reasons(p) == []
    monkeypatch.setenv("KNOWLEDGE_GAP_MIN_SCORE", "0.5")
    assert kg.min_score() == 0.5 and reasons(p) == [kg.REASON_WEAK]
    monkeypatch.setenv("KNOWLEDGE_GAP_MIN_SCORE", "not-a-number")
    assert kg.min_score() == kg.DEFAULT_MIN_SCORE
    monkeypatch.setenv("KNOWLEDGE_GAP_MIN_SCORE", "")
    assert kg.min_score() == kg.DEFAULT_MIN_SCORE


def test_cluster_reports_the_dominant_reason_and_counts():
    gaps = [{**pair("Who won the cricket match"), "reason": r}
            for r in (kg.REASON_WEAK, kg.REASON_WEAK, kg.REASON_SAID)]
    c = kg.build_clusters(gaps, {"a1": "Ria"}, None)[0]
    assert c["reason"] == kg.REASON_WEAK and c["reasons"] == {kg.REASON_WEAK: 2, kg.REASON_SAID: 1}
    tie = [{**pair("Who won the cricket match"), "reason": r} for r in (kg.REASON_WEAK, kg.REASON_SAID)]
    assert kg.build_clusters(tie, {}, None)[0]["reason"] == kg.REASON_SAID          # the strongest signal wins a tie


def test_pair_turns_carries_the_retrieval_of_the_agent_turn():
    rows = [{"conversation_id": "c", "role": "user", "text_content": "q q q", "created_at": "2026-10-10T10:00:00"},
            {"conversation_id": "c", "role": "agent", "text_content": "a", "created_at": "2026-10-10T10:00:01",
             "retrieval": sig(top=0.2)},
            {"conversation_id": "c", "role": "user", "text_content": "w w w", "created_at": "2026-10-10T10:01:00"},
            {"conversation_id": "c", "role": "agent", "text_content": "a", "created_at": "2026-10-10T10:01:01"}]
    out = kg.pair_turns(rows, {"c": "a1"})
    assert [p["retrieval"] is not None for p in out] == [False, True]   # newest first; old turns have none


def test_endpoint_flags_by_retrieval_and_reports_the_reason(app_client, env):
    a = env.local_db.insert("agents", {"user_id": "local-user", "name": "Ria", "supported_languages": ["en"]})
    conv = seed(env, "local-user", a["id"], [("Who won the cricket match yesterday", ANSWER),
                                             ("What time do you open on sunday", ANSWER)])
    for t, r in zip(env.local_db.select("conversation_turns", lambda x: x["role"] == "agent"
                                        and x["conversation_id"] == conv),
                    (sig(top=0.2), sig(top=0.7))):
        env.local_db.update("conversation_turns", t["id"], {"retrieval": r})
    body = app_client.get("/api/analytics/knowledge-gaps", headers=LOCAL).json()
    assert body["total_gap_turns"] == 1 and body["total_turns"] == 2
    c = body["clusters"][0]
    assert "cricket" in c["question"] and c["reason"] == "weak match" and c["reasons"] == {"weak match": 1}
