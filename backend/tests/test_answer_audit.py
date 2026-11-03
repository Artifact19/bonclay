"""Post-call answer audit (services/answer_audit.py, routers/answer_audit.py) with a fake verifier: no model, no download,
no network. Turn selection, skipping rules, threshold, storage, owner scoping, rate limit, job scheduling and the
verifier's lazy-load / idle-unload lifecycle."""
import asyncio
import re
import sys
from datetime import datetime, timedelta, timezone

import pytest

from services import answer_audit as aa
from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)

PASSAGE = "The refund window is thirty days from the date of purchase. Items must be unused."
GOOD = "The refund window is thirty days from purchase."
BAD = "We also offer free delivery across the whole country."
NO_INFO = "I'm sorry, I don't have that information. I can connect you with a member of staff."


class FakeVerifier:
    """Share of the hypothesis's words (3+ letters) found in the premise: 1.0 supported, near 0 unsupported."""
    name = "fake-hhem"

    def __init__(self):
        self.calls = []

    def score(self, pairs):
        self.calls.append(list(pairs))
        out = []
        for premise, hyp in pairs:
            words = {w for w in re.findall(r"[a-z0-9]+", hyp.lower()) if len(w) >= 3}
            have = set(re.findall(r"[a-z0-9]+", premise.lower()))
            out.append(len(words & have) / max(1, len(words)))
        return out


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setenv("BONCLAY_ANSWER_AUDIT_FILE", str(tmp_path / "answer_audit.json"))
    for name in ("AUDIT_THRESHOLD", "ANSWER_AUDIT", "ANSWER_AUDIT_AUTO", "ANSWER_AUDIT_RUN_MAX", "ANSWER_AUDIT_AUTO_DOWNLOAD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(aa, "_local_store", None)
    monkeypatch.setattr(aa, "model_cached", lambda: False)
    aa._due_now.clear()
    yield
    aa.set_verifier(None)
    aa._due_now.clear()
    from routers import answer_audit as router
    router._runs.clear()


@pytest.fixture()
def fake():
    v = FakeVerifier()
    aa.set_verifier(v)
    return v


def iso(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


class Chunks:
    def __init__(self, rows):
        self.rows = rows

    def by_ids(self, ids):
        return [r for r in self.rows if r["id"] in ids]

    def by_docs(self, doc_ids):
        return [r for r in self.rows if r["document_id"] in doc_ids]


CHUNKS = Chunks([{"id": "c1", "document_id": "d1", "chunk_text": PASSAGE},
                 {"id": "c2", "document_id": "d1", "chunk_text": "Parking is free for patients."}])


def turn(text=GOOD, *, tid="t1", retrieval=None, language="en", **extra):
    r = {"n_chunks": 1, "chunk_ids": ["c1"], "doc_ids": ["d1"]} if retrieval is None else retrieval
    return {"id": tid, "role": "agent", "text_content": text, "created_at": iso(600), "language": language,
            **({"retrieval": r} if r else {}), **extra}


def cand(t=None, question="What is the refund window?"):
    t = t or turn()
    return aa.Candidate("u1", "a1", "conv1", "widget", t, question)


# ---------------------------------------------------------------------------------------------------- sentence rules

def test_checkable_sentences_leave_out_disclosure_pleasantries_questions_and_refusals():
    reply = ("This is an AI assistant. Hello there! The refund window is thirty days from purchase. "
             "Items must be unused to qualify. Is there anything else I can help with? Sure.")
    assert aa.checkable_sentences(reply) == ["The refund window is thirty days from purchase.",
                                             "Items must be unused to qualify."]
    assert aa.checkable_sentences("I'm sorry, I don't have that information. The fee is five hundred rupees.") == \
        ["The fee is five hundred rupees."]
    assert aa.checkable_sentences("Could you tell me your name?") == []


def test_tool_confirmations_are_not_checked_but_document_facts_in_the_same_reply_are():
    t = {"tool_calls": [{"name": "book_appointment", "ok": True,
                         "result": "booking A12 with Dr. Meera Kulkarni on Saturday 2026-10-17 12:30"}]}
    reply = "Your booking A12 with Dr. Meera is confirmed for Saturday. The refund window is thirty days from purchase."
    assert aa.checkable_sentences(reply, t) == ["The refund window is thirty days from purchase."]
    failed = {"tool_calls": [{"name": "book_appointment", "ok": False, "result": "failed"}]}
    assert len(aa.checkable_sentences("The booking is confirmed for Saturday morning.", failed)) == 1  # nothing was done


def test_english_only():
    assert aa.reply_is_english("The refund window is thirty days.", "en")
    assert aa.reply_is_english("The refund window is thirty days.", None)
    assert not aa.reply_is_english("The refund window is thirty days.", "hi")
    assert not aa.reply_is_english("रिफंड की अवधि तीस दिन है।", "en")
    assert not aa.reply_is_english("Aapka refund tees din mein mil jayega, aap chinta mat kijiye.", "en")  # Hinglish


# ---------------------------------------------------------------------------------------------------- skipping rules

@pytest.mark.parametrize("t,question,why", [
    (turn(NO_INFO), "Do you treat dogs?", "refusal"),
    (turn(GOOD), "", "greeting"),
    (turn(GOOD, timings={"instant_answer_id": "ia1"}), "What is the refund window?", "instant answer"),
    (turn(GOOD, fallback=True), "What is the refund window?", "model failure"),
    (turn(GOOD, retrieval={"n_chunks": 0}), "What is the refund window?", "not answered from documents"),
    (turn(GOOD, retrieval={"n_chunks": 2, "error": True}), "What is the refund window?", "not answered from documents"),
    ({"id": "t9", "role": "agent", "text_content": GOOD}, "What is the refund window?", "not answered from documents"),
    (turn(""), "What is the refund window?", "empty reply"),
])
def test_turns_that_are_never_checked(t, question, why, fake):
    row = aa.evaluate(cand(t, question), CHUNKS, fake)
    assert row["status"] == aa.STATUS_SKIPPED and why in row["skip_reason"] and row["score"] is None
    assert fake.calls == []                 # no model call, so no model load either
    assert row["question"] is None and row["reply"] is None  # nothing but the verdict is kept for a skipped turn


def test_other_languages_are_marked_not_checked_not_scored(fake):
    row = aa.evaluate(cand(turn("रिफंड की अवधि तीस दिन है।", language="hi"), "रिफंड कितने दिन में मिलता है"), CHUNKS, fake)
    assert row["status"] == aa.STATUS_LANGUAGE and row["flagged"] is False and fake.calls == []
    hinglish = aa.evaluate(cand(turn("Aapka refund tees din mein mil jayega, aap chinta mat kijiye.", language="hinglish"),
                                "refund kab milega"), CHUNKS, fake)
    assert hinglish["status"] == aa.STATUS_LANGUAGE
    hindi_docs = Chunks([{"id": "c1", "document_id": "d1", "chunk_text": "रिफंड की अवधि खरीद से तीस दिन है। सामान उपयोग में नहीं होना चाहिए।"}])
    row = aa.evaluate(cand(), hindi_docs, fake)
    assert row["status"] == aa.STATUS_LANGUAGE and "documents" in row["skip_reason"] and fake.calls == []


# ---------------------------------------------------------------------------------------------------- scoring

def test_supported_reply_is_not_flagged_and_unsupported_is(fake):
    ok = aa.evaluate(cand(turn(GOOD)), CHUNKS, fake)
    assert ok["status"] == aa.STATUS_CHECKED and ok["flagged"] is False and ok["score"] == 1.0
    assert ok["passage_source"] == "stored" and "thirty days" in ok["passage"] and ok["model"] == "fake-hhem"
    bad = aa.evaluate(cand(turn(BAD, tid="t2")), CHUNKS, fake)
    assert bad["flagged"] is True and bad["score"] < 0.5 and bad["weakest_sentence"] == BAD
    assert bad["question"] == "What is the refund window?" and bad["reply"] == BAD and bad["threshold"] == 0.5


def test_a_reply_is_as_good_as_its_weakest_sentence(fake):
    row = aa.evaluate(cand(turn(f"{GOOD} {BAD}")), CHUNKS, fake)
    assert [s["text"] for s in row["sentences"]] == [GOOD, BAD]
    assert row["flagged"] and row["weakest_sentence"] == BAD and row["score"] == min(s["score"] for s in row["sentences"])
    assert len(fake.calls) == 1 and len(fake.calls[0]) == 2   # one model call for the whole reply


def test_threshold_is_configurable(fake, monkeypatch):
    partial = "The refund window is thirty days but returns need a receipt and a photo."
    assert 0.3 < aa.evaluate(cand(turn(partial)), CHUNKS, fake)["score"] < 0.9
    monkeypatch.setenv("AUDIT_THRESHOLD", "0.99")
    assert aa.evaluate(cand(turn(partial)), CHUNKS, fake)["flagged"] is True
    monkeypatch.setenv("AUDIT_THRESHOLD", "0.05")
    assert aa.evaluate(cand(turn(partial)), CHUNKS, fake)["flagged"] is False
    monkeypatch.setenv("AUDIT_THRESHOLD", "banana")
    assert aa.threshold() == 0.5


def test_turns_without_chunk_ids_use_the_best_passages_of_their_documents(fake):
    old = turn(GOOD, retrieval={"n_chunks": 1, "doc_ids": ["d1"]})   # stored before chunk ids were kept
    row = aa.evaluate(cand(old), CHUNKS, fake)
    assert row["passage_source"] == "approx" and "thirty days" in row["passage"] and row["flagged"] is False
    assert "Parking" not in fake.calls[0][0][0] or fake.calls[0][0][0].index("refund") < fake.calls[0][0][0].index("Parking")


def test_no_passages_means_skipped_not_flagged(fake):
    row = aa.evaluate(cand(turn(GOOD, retrieval={"n_chunks": 1, "chunk_ids": ["gone"], "doc_ids": ["nope"]})), CHUNKS, fake)
    assert row["status"] == aa.STATUS_SKIPPED and row["skip_reason"] == "passages not available"


def test_a_verifier_failure_stops_the_run_and_stores_nothing_for_that_turn(tmp_path):
    class Boom(FakeVerifier):
        def score(self, pairs):
            raise RuntimeError("model could not load")

    store = aa.LocalFileStore(str(tmp_path / "x.json"))
    cands = [cand(turn(NO_INFO, tid="t1"), "dogs?"), cand(turn(GOOD, tid="t2")), cand(turn(GOOD, tid="t3"))]
    with pytest.raises(RuntimeError):
        aa.run_candidates(cands, lambda c: CHUNKS, Boom(), lambda c: store)
    assert {r["turn_id"] for r in store.list("u1")} == {"t1"}   # the refusal needed no model; t2 failed and is retried later


# ---------------------------------------------------------------------------------------------------- selection

def conv(cid="conv1", user="u1", agent="a1", **kw):
    return {"id": cid, "user_id": user, "agent_id": agent, "created_at": iso(900), **kw}


def pair(cid, *turns, **kw):
    return conv(cid, **kw), [{"role": r, "text_content": t, "id": f"{cid}-{i}", "created_at": iso(800 - i)}
                             for i, (r, t) in enumerate(turns)]


def test_collect_picks_new_agent_turns_with_their_question():
    c1 = pair("c1", ("agent", "Hello!"), ("user", "Refund window?"), ("agent", GOOD), ("agent", "Anything else?"))
    got = aa.collect([c1], set())
    assert [(c.turn["id"], c.question) for c in sorted(got, key=lambda c: c.turn["id"])] == \
        [("c1-0", ""), ("c1-2", "Refund window?"), ("c1-3", "")]   # the greeting and a second reply have no question
    assert aa.collect([c1], {("u1", "c1-2")}) and len(aa.collect([c1], {("u1", "c1-2"), ("u1", "c1-0"), ("u1", "c1-3")})) == 0


def test_collect_leaves_out_practice_calls_other_agents_and_caps_newest_first():
    pairs = [pair(f"c{i}", ("user", "q"), ("agent", GOOD)) for i in range(5)]
    for i, (_c, ts) in enumerate(pairs):
        ts[1]["created_at"] = iso(1000 - i * 10)   # c4 is the newest
    assert [c.conversation_id for c in aa.collect(pairs, set(), limit=2)] == ["c4", "c3"]
    practice = pair("p", ("user", "q"), ("agent", GOOD), is_test=True)
    other = pair("o", ("user", "q"), ("agent", GOOD), agent="a2")
    assert [c.conversation_id for c in aa.collect([practice, other], set())] == ["o"]
    assert aa.collect([practice, other], set(), agent_id="a1") == []


def test_a_conversation_is_over_after_45_seconds_for_phone_and_two_minutes_otherwise():
    def stamped(source, age):
        c, ts = pair("x", ("user", "q"), ("agent", GOOD), **({"source": source} if source else {}))
        for t in ts:
            t["created_at"] = iso(age)
        return c, ts
    assert aa.conversation_over(*stamped("phone", 60)) and not aa.conversation_over(*stamped("phone", 20))
    assert not aa.conversation_over(*stamped("widget", 100)) and aa.conversation_over(*stamped("widget", 130))
    assert not aa.conversation_over(*stamped(None, 100)) and aa.conversation_over(*stamped("whatsapp", 600))
    c, ts = stamped("phone", 5)
    assert aa.conversation_over(c, ts, due_now={c["id"]})      # the phone channel said the call hung up
    assert not aa.conversation_over({"id": "e"}, [])


def test_live_traffic_pauses_the_job():
    c, ts = pair("x", ("user", "q"), ("agent", GOOD))
    ts[-1]["created_at"] = iso(3)
    assert aa.live_traffic([(c, ts)])
    ts[-1]["created_at"] = iso(60)
    assert not aa.live_traffic([(c, ts)])


# ---------------------------------------------------------------------------------------------------- storage

def test_store_keeps_the_owners_mark_when_a_turn_is_checked_again_and_scopes_by_user(tmp_path):
    s = aa.LocalFileStore(str(tmp_path / "s.json"))
    row = {**aa._base_row(cand()), "status": aa.STATUS_CHECKED, "flagged": True, "score": 0.1, "question": "q", "reply": "r"}
    s.upsert(row)
    assert s.set_feedback("u1", "t1", "wrong")["feedback"] == "wrong"
    s.upsert({**row, "score": 0.2})
    got = s.get("u1", "t1")
    assert got["score"] == 0.2 and got["feedback"] == "wrong" and got["feedback_at"]
    assert s.list("u2") == [] and s.get("u2", "t1") is None and s.set_feedback("u2", "t1", "fine") is None
    assert s.known("u1") == {("u1", "t1")} and s.known("u2") == set()
    assert s.set_feedback("u1", "t1", None)["feedback"] is None
    assert s.purge_agent("u1", "a1") == 1 and s.list("u1") == []
    assert aa.LocalFileStore(str(tmp_path / "missing.json")).list("u1") == []


def test_the_results_file_is_git_ignored_and_has_a_migration():
    import os
    backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert "data/answer_audit.json" in open(os.path.join(backend, ".gitignore")).read()
    sql = open(os.path.join(backend, "supabase", "migrations", "019_answer_audit.sql")).read()
    assert "enable row level security" in sql and "auth.uid()" in sql and "answer_audit" in sql


# ---------------------------------------------------------------------------------------------------- the model's lifecycle

def test_importing_or_asking_for_the_verifier_loads_nothing():
    aa.set_verifier(None)
    v = aa.get_verifier()
    assert isinstance(v, aa.HHEMVerifier) and not v.loaded
    import inspect
    assert "import transformers" not in inspect.getsource(aa).split("def _default_loader")[0]


def test_model_loads_on_first_score_once_and_unloads_when_idle():
    loads, now = [], [100.0]

    class Model:
        def predict(self, pairs):
            return [0.9] * len(pairs)

    def loader():
        loads.append(1)
        return Model()

    v = aa.HHEMVerifier(loader=loader, idle_s=60, clock=lambda: now[0])
    assert not v.loaded and loads == []
    assert v.score([]) == [] and loads == []                       # nothing to score: nothing loaded
    assert v.score([("p", "h"), ("p", "h2")]) == [0.9, 0.9] and v.loaded and loads == [1]
    v.score([("p", "h")])
    assert loads == [1]
    now[0] += 30
    assert v.unload_if_idle() is False and v.loaded
    now[0] += 40
    assert v.unload_if_idle() is True and not v.loaded
    v.score([("p", "h")])
    assert loads == [1, 1]
    v.unload()
    assert not v.loaded


def test_default_loader_asks_the_memory_guard_before_loading(monkeypatch):
    from core import memory_guard
    from core.errors import AppError
    seen = {}

    def refuse(engine, candidates, unload, **kw):
        seen["engine"] = engine
        raise AppError("INSUFFICIENT_MEMORY", "no room", 503)

    monkeypatch.setattr(memory_guard, "ensure_headroom", refuse)
    with pytest.raises(AppError):
        aa._default_loader()
    assert seen["engine"] == aa.ENGINE and memory_guard.need_mb(aa.ENGINE) == aa.MODEL_NEED_MB


# ---------------------------------------------------------------------------------------------------- API and job

@pytest.fixture()
def world(app_client, env, fake):
    """An agent with one document chunk and conversations that finished a while ago."""
    agent = make_agent(app_client)
    db = env.local_db
    chunk = db.insert("document_chunks", {"document_id": "doc1", "chunk_text": PASSAGE, "chunk_index": 0})

    def talk(question, reply, *, user="local-user", source=None, seconds_ago=900, agent_id=agent["id"], **turn_extra):
        row = {"agent_id": agent_id, "user_id": user, "created_at": iso(seconds_ago), **({"source": source} if source else {})}
        cid = db.insert("conversations", row)["id"]
        db.insert("conversation_turns", {"conversation_id": cid, "role": "user", "text_content": question,
                                         "created_at": iso(seconds_ago - 1)})
        t = db.insert("conversation_turns", {
            "conversation_id": cid, "role": "agent", "text_content": reply, "created_at": iso(seconds_ago - 2),
            "language": "en", "retrieval": {"n_chunks": 1, "chunk_ids": [chunk["id"]], "doc_ids": ["doc1"]}, **turn_extra})
        return cid, t["id"]

    return type("World", (), {"agent": agent, "db": db, "talk": staticmethod(talk), "chunk": chunk})


def run(client, wait=True, **body):
    return client.post("/api/answer-audit/run" + ("?wait=true" if wait else ""), headers=LOCAL, json=body)


def flagged(client, **params):
    r = client.get("/api/answer-audit", headers=LOCAL, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_run_checks_recent_answers_and_lists_only_the_flagged_with_the_passage(app_client, world, fake):
    _, ok_turn = world.talk("What is the refund window?", GOOD)
    bad_conv, bad_turn = world.talk("Do you deliver?", BAD)
    world.talk("Do you treat dogs?", NO_INFO)
    world.talk("hi, refund?", GOOD, timings={"instant_answer_id": "ia"})
    r = run(app_client)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["run"]["state"] == "done" and body["run"]["total"] == 4 and body["run"]["checked"] == 2
    assert body["run"]["flagged"] == 1 and body["run"]["skipped"] == 2 and body["first_run_downloads_model"] is False
    got = flagged(app_client)
    assert [f["turn_id"] for f in got["flagged"]] == [bad_turn]
    f = got["flagged"][0]
    assert f["question"] == "Do you deliver?" and f["reply"] == BAD and f["conversation_id"] == bad_conv
    assert "thirty days" in f["passage"] and f["passage_source"] == "stored" and f["score"] < 0.5
    assert f["agent_name"] == world.agent["name"]
    assert got["counts"] == {"checked": 2, "flagged": 1, "not_checked_language": 0, "marked_fine": 0, "marked_wrong": 0}
    assert got["threshold"] == 0.5 and "Hindi and Hinglish" in got["note"] and got["run"]["state"] == "done"
    assert flagged(app_client, agent_id=world.agent["id"])["flagged"][0]["turn_id"] == bad_turn
    n = len(fake.calls)
    assert run(app_client).json()["run"]["total"] == 0 and len(fake.calls) == n   # nothing is checked twice


def test_hindi_and_hinglish_answers_are_counted_as_not_checked(app_client, world, fake):
    world.talk("refund kab milega", "Aapka refund tees din mein mil jayega, aap chinta mat kijiye.", language="hinglish")
    world.talk("रिफंड", "रिफंड की अवधि तीस दिन है।", language="hi")
    run(app_client)
    got = flagged(app_client)
    assert got["flagged"] == [] and got["counts"]["not_checked_language"] == 2 and fake.calls == []


def test_the_owner_marks_an_answer_fine_or_wrong(app_client, world):
    _, tid = world.talk("Do you deliver?", BAD)
    run(app_client)
    r = app_client.patch(f"/api/answer-audit/{tid}", headers=LOCAL, json={"feedback": "wrong"})
    assert r.status_code == 200 and r.json()["feedback"] == "wrong"
    got = flagged(app_client)
    assert got["flagged"][0]["feedback"] == "wrong" and got["counts"]["marked_wrong"] == 1
    r = app_client.put(f"/api/answer-audit/{tid}", headers=LOCAL, json={"feedback": "fine"})
    assert r.status_code == 200
    got = flagged(app_client)
    assert got["flagged"] == [] and got["counts"]["marked_fine"] == 1 and got["counts"]["flagged"] == 0
    assert app_client.patch(f"/api/answer-audit/{tid}", headers=LOCAL, json={"feedback": None}).json()["feedback"] is None
    assert len(flagged(app_client)["flagged"]) == 1
    assert app_client.patch(f"/api/answer-audit/{tid}", headers=LOCAL, json={"feedback": "maybe"}).status_code == 422
    assert app_client.patch("/api/answer-audit/not-a-turn", headers=LOCAL, json={"feedback": "fine"}).status_code == 404
    assert app_client.patch("/api/answer-audit/bad%20id!", headers=LOCAL, json={"feedback": "fine"}).status_code == 422


def test_owner_scoping(app_client, world, fake):
    _, mine = world.talk("Do you deliver?", BAD)
    world.talk("Do you deliver?", BAD, user="someone-else")
    run(app_client)
    assert [f["turn_id"] for f in flagged(app_client)["flagged"]] == [mine]
    # a row that belongs to someone else is invisible and cannot be marked
    store = aa.get_store("local")
    theirs = {**aa._base_row(aa.Candidate("someone-else", "ax", "cx", None, {"id": "their-turn", "created_at": iso(5)}, "q")),
              "status": aa.STATUS_CHECKED, "flagged": True, "score": 0.0, "question": "secret", "reply": "secret"}
    store.upsert(theirs)
    assert "secret" not in str(flagged(app_client))
    assert app_client.patch("/api/answer-audit/their-turn", headers=LOCAL, json={"feedback": "fine"}).status_code == 404
    assert store.get("someone-else", "their-turn")["feedback"] is None
    # an agent that is not the caller's is a 404, for the list and for a run
    assert app_client.get("/api/answer-audit", headers=LOCAL, params={"agent_id": "00000000-0000-0000-0000-000000000000"}).status_code == 404
    assert run(app_client, agent_id="00000000-0000-0000-0000-000000000000").status_code == 404


def test_run_is_rate_limited_one_at_a_time_and_can_be_turned_off(app_client, world, monkeypatch):
    from routers import answer_audit as router
    router._runs["local-user"] = {"state": "running"}
    r = run(app_client)
    assert r.status_code == 409 and r.json()["code"] == "AUDIT_RUNNING"
    router._runs.clear()
    monkeypatch.setenv("ANSWER_AUDIT", "off")
    r = run(app_client)
    assert r.status_code == 409 and r.json()["code"] == "ANSWER_AUDIT_OFF"
    monkeypatch.delenv("ANSWER_AUDIT")
    app_client.app.state.limiter.reset()   # the two refusals above also count towards the limit
    codes = [run(app_client).status_code for _ in range(7)]
    assert codes[:6] == [202] * 6 and codes[6] == 429


def test_a_failing_model_reports_an_error_and_keeps_the_turn_for_later(app_client, world):
    class Boom(FakeVerifier):
        def score(self, pairs):
            raise RuntimeError("could not load")
    aa.set_verifier(Boom())
    world.talk("Do you deliver?", BAD)
    body = run(app_client).json()
    assert body["run"]["state"] == "error" and "could not finish" in body["run"]["error"]
    aa.set_verifier(FakeVerifier())
    assert run(app_client).json()["run"]["checked"] == 1


def test_background_sweep_checks_only_conversations_that_ended(app_client, world, fake):
    _, phone_done = world.talk("Do you deliver?", BAD, source="phone", seconds_ago=90)
    _, widget_done = world.talk("Do you deliver?", BAD, source="widget", seconds_ago=400)
    live_cid, phone_live = world.talk("Do you deliver?", BAD, source="phone", seconds_ago=20)
    _, widget_live = world.talk("Do you deliver?", BAD, source="widget", seconds_ago=60)
    from routers import answer_audit as router
    stats = asyncio.run(router.sweep_once())
    assert stats["checked"] == 2, stats
    store = aa.get_store("local")
    assert {r["turn_id"] for r in store.list("local-user")} == {phone_done, widget_done}
    # the call that hung up is checked at once when its channel reports the end
    aa.notify_conversation_end(live_cid)
    assert asyncio.run(router.sweep_once())["checked"] == 1
    assert {r["turn_id"] for r in store.list("local-user")} == {phone_done, widget_done, phone_live}
    assert asyncio.run(router.sweep_once())["checked"] == 0   # nothing is checked twice


def test_background_sweep_waits_for_the_model_pauses_for_live_calls_and_obeys_the_switches(app_client, world, monkeypatch):
    from routers import answer_audit as router
    world.talk("Do you deliver?", BAD, source="widget", seconds_ago=400)
    aa.set_verifier(None)                                    # no fake: the real model is not downloaded (model_cached False)
    assert "skipped" in asyncio.run(router.sweep_once())
    assert aa.get_store("local").list("local-user") == []
    v = FakeVerifier()
    aa.set_verifier(v)
    monkeypatch.setenv("ANSWER_AUDIT_AUTO", "off")
    assert "skipped" in asyncio.run(router.sweep_once()) and v.calls == []
    monkeypatch.delenv("ANSWER_AUDIT_AUTO")
    world.talk("Fees?", GOOD, source="widget", seconds_ago=5)    # a conversation is happening right now
    out = asyncio.run(router.sweep_once())
    assert out["skipped"] == "a live conversation is in progress" and v.calls == []


def test_the_job_is_not_started_under_pytest_and_is_registered_with_the_app(app_client):
    from routers import answer_audit as router
    asyncio.run(router._start_job())
    assert router._loop_task is None
    import backend
    assert any(getattr(r, "path", "").startswith("/api/answer-audit") or "answer-audit" in str(r) for r in backend.app.routes)
    assert aa.auto_ready() is False   # model not downloaded in tests (model_cached is False) and no verifier forced


def test_retrieval_signals_keep_the_chunk_ids_the_audit_reads_back(app_client):
    import backend
    sig = backend._retrieval_signals([{"id": "c1", "document_id": "d1", "similarity": 0.7},
                                      {"id": "c2", "document_id": "d1", "similarity": 0.6}])
    assert sig["chunk_ids"] == ["c1", "c2"] and sig["doc_ids"] == ["d1"]
    assert "chunk_ids" not in backend._retrieval_signals([], error=True)
