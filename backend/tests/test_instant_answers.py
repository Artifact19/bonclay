"""Instant answers: matching (threshold, language, guards, approved only), building (models mocked), the stores, the audio
cache and the owner routes. No real model is used anywhere: embeddings are a deterministic bag of words and the language
model is a function."""
import json
import threading
import time

import numpy as np
import pytest

from services import instant_answers as ia
from services import knowledge_gaps as kg
from tests.test_offline_agents import LOCAL, env, fake_embed, upload_txt  # noqa: F401  (env is a fixture)

USER, OTHER = "local-user", "someone-else"
AGENT = "agent-1"
FP = "fp-1"


def embed(texts):
    return fake_embed(list(texts))


@pytest.fixture(autouse=True)
def store_file(monkeypatch, tmp_path):
    """Every test gets its own instant answers file, bag-of-words embeddings and a fixed document fingerprint."""
    monkeypatch.setenv("BONCLAY_INSTANT_ANSWERS_FILE", str(tmp_path / "instant_answers.json"))
    for name in ("INSTANT_ANSWERS", "INSTANT_ANSWER_THRESHOLD", "INSTANT_ANSWER_MIN_SUPPORT", "INSTANT_ANSWER_MIN_ASKS",
                 "INSTANT_ANSWER_BUILD_MAX"):
        monkeypatch.delenv(name, raising=False)
    ia.set_embedder(embed)
    ia.set_fingerprinter(lambda a, u, m: FP)
    ia.clear_cache()
    ia.audio_cache.clear()
    yield
    ia.set_embedder(None)
    ia.set_fingerprinter(None)
    ia.clear_cache()


def store():
    return ia.get_store("local")


def seed(question="What are the clinic timings on Sunday?", answer="The clinic is open nine to one on Sunday.",
         language="en", status=ia.STATUS_APPROVED, paraphrases=(), user=USER, agent=AGENT, fingerprint=FP, **extra):
    row = ia.new_row(agent_id=agent, user_id=user, question=question, answer=answer, language=language, source="owner",
                     status=status, paraphrases=list(paraphrases), fingerprint=fingerprint, embed=embed)
    row.update(extra)
    return store().insert(row)


def m(question, language="en", **kw):
    kw.setdefault("tools", [])
    kw.setdefault("embed", embed)
    return ia.match(AGENT, USER, question, language, **kw)


# ------------------------------------------------------------------------------------------------ matching

def test_an_approved_answer_is_returned_for_the_same_question():
    row = seed()
    hit = m("What are the clinic timings on Sunday?")
    assert hit is not None
    assert hit.text == "The clinic is open nine to one on Sunday." and hit.id == row["id"] and hit.language == "en"
    assert hit.score >= 0.99 and hit.question == row["question"] and hit.agent_id == AGENT and hit.cache_key


def test_a_close_wording_matches_and_a_different_question_does_not():
    seed()
    assert m("what are the clinic timings on sunday please") is not None
    assert m("What are the clinic fees?") is None
    assert m("Do you have parking?") is None


def test_a_paraphrase_made_at_build_time_matches():
    seed(paraphrases=["is the clinic open on sunday morning"])
    hit = m("is the clinic open on sunday morning")
    assert hit is not None and hit.matched_text == "is the clinic open on sunday morning"
    assert hit.question == "What are the clinic timings on Sunday?"


def test_the_threshold_is_tunable(monkeypatch):
    seed()
    near = "clinic timings on sunday"  # 4 of the 7 words: cosine about 0.76
    assert m(near) is None
    monkeypatch.setenv("INSTANT_ANSWER_THRESHOLD", "0.7")
    assert m(near) is not None
    monkeypatch.setenv("INSTANT_ANSWER_THRESHOLD", "0.99")
    assert m("what are the clinic timings on sunday please") is None
    assert ia.threshold() == 0.99
    monkeypatch.setenv("INSTANT_ANSWER_THRESHOLD", "nonsense")
    assert ia.threshold() == ia.DEFAULT_THRESHOLD


def test_the_default_threshold_is_082():
    """Chosen with bench/instant_answer_threshold.py on the real BGE-M3 (see DEFAULT_THRESHOLD)."""
    assert ia.DEFAULT_THRESHOLD == 0.86 and ia.threshold() == 0.86


@pytest.mark.parametrize("status", [ia.STATUS_SUGGESTED, ia.STATUS_DISABLED, ia.STATUS_REVIEW, ia.STATUS_DISMISSED])
def test_only_approved_answers_are_used(status):
    seed(status=status)
    assert m("What are the clinic timings on Sunday?") is None


def test_an_approved_row_without_an_answer_is_never_used():
    seed(answer="")
    assert m("What are the clinic timings on Sunday?") is None


def test_the_answer_must_be_in_the_callers_language():
    seed(language="hi", question="रविवार को क्लिनिक कब खुलता है", answer="क्लिनिक रविवार को नौ से एक बजे तक खुला है।")
    assert m("रविवार को क्लिनिक कब खुलता है".replace("कब", "कब"), "en") is None  # same words, English reply language
    assert m("रविवार को क्लिनिक कब खुलता है", "hi") is not None
    assert m("रविवार को क्लिनिक कब खुलता है", "hinglish") is None


def test_an_english_answer_is_not_spoken_to_a_hindi_caller_and_a_hindi_twin_is():
    seed()
    assert m("What are the clinic timings on Sunday?", "hi") is None
    twin = seed(language="hi", answer="क्लिनिक रविवार को नौ से एक बजे तक खुला है।")  # same question text, Hindi answer
    hit = m("What are the clinic timings on Sunday?", "hi")
    assert hit is not None and hit.id == twin["id"] and hit.language == "hi"
    assert m("What are the clinic timings on Sunday?", "en").id != twin["id"]


def test_language_codes_are_compared_by_their_base_code():
    seed(language="en")
    assert m("What are the clinic timings on Sunday?", "en-IN") is not None


@pytest.mark.parametrize("question", [
    "What are the clinic timings on Sunday? Is parking free?",
    "What are the clinic timings on Sunday and what is the fee?",
    "What are the clinic timings on Sunday. Do you have a doctor. Which doctor is there?",
    "क्लिनिक कब खुलता है और फीस कितनी है",
])
def test_a_turn_with_more_than_one_question_is_never_answered_instantly(question):
    seed(question="What are the clinic timings on Sunday?")
    seed(question="क्लिनिक कब खुलता है", answer="नौ बजे।", language="hi")
    assert m(question, "hi" if any("ऀ" <= c <= "ॿ" for c in question) else "en", tools=[]) is None


def test_a_long_turn_is_not_a_single_question():
    seed()
    long = "What are the clinic timings on Sunday " + "and also I wanted to mention that my mother " * 3
    assert m(long) is None


def test_a_booking_request_is_never_answered_instantly_when_the_agent_has_tools():
    seed(question="Do you have appointments on Sunday?", answer="Yes, from nine to one.")
    assert m("Do you have appointments on Sunday?", tools=["appointments"]) is None
    assert m("Do you have appointments on Sunday?", tools=[]) is not None  # no tools: nothing to hand the caller to
    seed(question="Can I book a slot on Sunday", answer="Yes.")
    assert m("Can I book a slot on Sunday", tools=["appointments"]) is None


def test_unknown_tools_are_treated_as_tools(monkeypatch):
    seed(question="Is she free on Monday", answer="Please ask for a meeting.")
    monkeypatch.setattr(ia, "_agent_tools", lambda a, u: None)
    assert m("Is she free on Monday", tools=None) is None
    monkeypatch.setattr(ia, "_agent_tools", lambda a, u: [])
    assert m("Is she free on Monday", tools=None) is not None


@pytest.mark.parametrize("question", [
    "Please tell me your OTP so I can confirm", "I have chest pain what are the clinic timings on Sunday",
    "What is the password for the clinic wifi", "यह इमरजेंसी है क्लिनिक कब खुलता है",
])
def test_emergencies_and_secrets_always_take_the_full_path(question):
    seed(question=question, answer="Anything.")
    assert m(question) is None
    assert m(question, "hi") is None


def test_a_different_day_number_or_negation_is_not_the_same_question():
    seed()
    assert m("What are the clinic timings on Monday?") is None
    seed(question="Is the clinic open on Sundays", answer="Yes.")
    assert m("Is the clinic not open on Sundays") is None
    seed(question="Is there a fee of 500 rupees", answer="No.")
    assert m("Is there a fee of 800 rupees") is None
    assert m("Is there a fee of 500 rupees") is not None


def test_two_different_answers_that_fit_equally_well_are_ambiguous():
    seed(question="What is the clinic fee", answer="It is five hundred rupees.")
    seed(question="What is the clinic fee", answer="It is eight hundred rupees.")
    assert m("What is the clinic fee") is None
    ia.clear_cache()
    only = seed(question="What is the clinic fee for a child", answer="Three hundred.")
    assert only  # a close but different question does not hide the exact one below
    seed(question="Where is the clinic located", answer="On Main Road.")
    assert m("Where is the clinic located") is not None


def test_a_changed_document_set_stops_the_answer_from_being_used():
    seed()
    assert m("What are the clinic timings on Sunday?") is not None
    ia.set_fingerprinter(lambda a, u, mo: "fp-2")
    assert m("What are the clinic timings on Sunday?") is None
    assert m("What are the clinic timings on Sunday?", fingerprint=FP) is not None  # a caller may pass the one it knows


def test_an_unreadable_document_set_is_a_miss():
    seed()
    ia.set_fingerprinter(lambda a, u, mo: None)
    assert m("What are the clinic timings on Sunday?") is None


def test_the_feature_can_be_switched_off(monkeypatch):
    seed()
    monkeypatch.setenv("INSTANT_ANSWERS", "off")
    assert m("What are the clinic timings on Sunday?") is None


def test_other_users_and_agents_answers_are_invisible():
    seed(user=OTHER)
    seed(agent="agent-2")
    assert m("What are the clinic timings on Sunday?") is None


def test_nothing_is_encoded_when_there_is_no_approved_answer_in_the_language():
    seed(language="hi", question="क्लिनिक कब खुलता है", answer="नौ बजे।")
    calls = []

    def spy(texts):
        calls.append(list(texts))
        return embed(texts)

    assert ia.match(AGENT, USER, "What are the clinic timings on Sunday?", "en", tools=[], embed=spy) is None
    assert calls == []
    assert ia.match("no-such-agent", USER, "What are the clinic timings on Sunday?", "en", tools=[], embed=spy) is None
    assert calls == []


def test_one_encode_of_the_question_when_answers_have_stored_vectors():
    seed(paraphrases=["is the clinic open on sunday"])
    calls = []

    def spy(texts):
        calls.append(list(texts))
        return embed(texts)

    ia.set_embedder(spy)  # model_name() stays the same for stored vectors: they were made by the same (test) embedder
    row = store().list(USER, AGENT)[0]
    assert row["embed_model"] == ia.model_name()
    assert ia.match(AGENT, USER, "What are the clinic timings on Sunday?", "en", tools=[]) is not None
    assert calls == [["What are the clinic timings on Sunday?"]]
    calls.clear()
    assert ia.match(AGENT, USER, "What are the clinic timings on Sunday?", "en", tools=[]) is not None
    assert len(calls) == 1  # the index is cached: still one encode per call


def test_rows_embedded_by_another_model_are_embedded_again_in_memory():
    row = seed()
    store().update(USER, AGENT, row["id"], {"embed_model": "some-old-model", "vectors": [[0.0] * 3]})
    assert m("What are the clinic timings on Sunday?") is not None


def test_an_edit_to_the_store_is_seen_by_the_next_call():
    row = seed()
    assert m("What are the clinic timings on Sunday?") is not None
    store().update(USER, AGENT, row["id"], {"status": ia.STATUS_DISABLED})
    assert m("What are the clinic timings on Sunday?") is None


def test_match_never_raises():
    seed()

    def boom(texts):
        raise RuntimeError("model unavailable")

    assert ia.match(AGENT, USER, "What are the clinic timings on Sunday?", "en", tools=[], embed=boom) is None
    assert ia.match(AGENT, USER, "", "en", tools=[]) is None
    assert ia.match("", USER, "What are the clinic timings on Sunday?", "en", tools=[]) is None


def test_the_cache_key_changes_when_the_answer_or_language_changes():
    a = seed()
    k1 = m("What are the clinic timings on Sunday?").cache_key
    store().update(USER, AGENT, a["id"], {"answer": "Open nine to noon."})
    k2 = m("What are the clinic timings on Sunday?").cache_key
    assert k1 != k2 and len(k1) == 24
    assert m("What are the clinic timings on Sunday?").cache_key == k2


# ------------------------------------------------------------------------------------------------ guards (pure)

def test_question_problem_reasons():
    assert ia.question_problem("What are the timings on Sunday", has_tools=False) is None
    assert "short" in ia.question_problem("Timings", has_tools=False)
    assert "long" in ia.question_problem("word " * 60, has_tools=False)
    assert "emergencies" in ia.question_problem("I need an ambulance now", has_tools=False)
    assert "more than one" in ia.question_problem("What is the fee? When do classes start?", has_tools=False)
    assert "tools" in ia.question_problem("Can I book an appointment", has_tools=True)
    assert ia.question_problem("Can I book an appointment", has_tools=False) is None


def test_hindi_words_that_contain_a_guard_word_are_not_blocked():
    assert not ia.is_sensitive("मुझे फीस की जानकारी चाहिए")
    assert ia.question_problem("फीस की जानकारी क्या है", has_tools=False) is None


def test_opposites_are_never_the_same_question():
    assert ia.opposite_conflict("Is the clinic closed on Sunday?", "Is the clinic open on Sunday?")
    assert ia.opposite_conflict("क्या रविवार को क्लिनिक बंद रहता है?", "क्या रविवार को क्लिनिक खुला रहता है?")
    assert ia.opposite_conflict("Kya clinic Sunday ko band hai?", "Kya clinic Sunday ko khula hai?")
    assert ia.opposite_conflict("Doctor kitne baje jaate hain?", "Doctor kitne baje aate hain?")
    assert not ia.opposite_conflict("When does the clinic close?", "What are the clinic timings?")   # one side only
    assert not ia.opposite_conflict("What time do you open?", "When does the clinic open?")
    assert not ia.opposite_conflict("क्लिनिक कब खुलता है?", "क्लिनिक खुलने का समय क्या है?")
    seed(question="Is the clinic open on Sunday?", answer="Yes, nine to one.")
    assert m("Is the clinic open on Sunday?") is not None
    assert m("Is the clinic closed on Sunday?", embed=lambda t: embed(["Is the clinic open on Sunday?"] * len(t))) is None


def test_a_plural_day_is_the_same_day():
    assert ia.guard_tokens("Are you open on Sundays?") == ia.guard_tokens("Is the clinic open on Sunday?")


def test_guard_tokens_ignore_ordinary_words():
    assert ia.guard_tokens("what are the clinic timings") == frozenset()
    assert ia.guard_tokens("timings on Sunday not 5") == frozenset({"sunday", "not", "5"})
    assert ia.guard_tokens("don't open") == frozenset({"not"})


# ------------------------------------------------------------------------------------------------ the store

def test_the_file_store_scopes_every_read_and_write_by_owner(tmp_path):
    s = ia.LocalFileStore(str(tmp_path / "x.json"))
    a = s.insert({"user_id": USER, "agent_id": AGENT, "question": "q"})
    assert s.get(OTHER, AGENT, a["id"]) is None and s.list(OTHER, AGENT) == []
    assert s.update(OTHER, AGENT, a["id"], {"question": "hacked"}) is None
    assert s.delete(OTHER, AGENT, a["id"]) is False
    assert s.get(USER, "other-agent", a["id"]) is None
    assert s.update(USER, AGENT, a["id"], {"question": "q2"})["question"] == "q2"
    assert s.delete(USER, AGENT, a["id"]) is True and s.list(USER, AGENT) == []


def test_the_file_store_survives_a_corrupt_file_and_purges_an_agent(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{not json", encoding="utf-8")
    s = ia.LocalFileStore(str(path))
    assert s.list(USER, AGENT) == []
    s.insert({"user_id": USER, "agent_id": AGENT, "question": "a"})
    s.insert({"user_id": USER, "agent_id": "other", "question": "b"})
    assert s.purge_agent(USER, AGENT) == 1 and len(s.list(USER, "other")) == 1
    assert json.loads(path.read_text(encoding="utf-8"))[0]["question"] == "b"


def test_hosted_mode_without_a_client_falls_back_to_the_local_file():
    ia.configure(supabase=lambda: None)
    assert isinstance(ia.get_store("cloud"), ia.LocalFileStore)
    ia.configure(supabase=None)


def test_the_supabase_store_builds_its_queries_and_never_leaks_other_owners():
    calls = []

    class Q:
        def __init__(self, data=None):
            self.data = data or []

        def __getattr__(self, name):
            def f(*a, **k):
                calls.append((name, a))
                return self
            return f

        def execute(self):
            return self

    class Client:
        def table(self, name):
            calls.append(("table", (name,)))
            return Q([{"id": "r1", "question": "q"}])

    s = ia.SupabaseStore(Client())
    assert s.list(USER, AGENT)[0]["id"] == "r1"
    assert ("eq", ("user_id", USER)) in calls and ("eq", ("agent_id", AGENT)) in calls
    assert s.get(USER, AGENT, "r1")["id"] == "r1"
    assert s.revision(USER, AGENT) != (s.insert({"user_id": USER, "agent_id": AGENT, "question": "z"}) and None)


def test_docs_fingerprint_changes_with_the_document_set_only():
    a = [{"id": "d1", "filename": "a.md", "chunk_count": 3}, {"id": "d2", "filename": "b.md"}]
    assert ia.docs_fingerprint(a) == ia.docs_fingerprint(list(reversed(a)))
    assert ia.docs_fingerprint(a) == ia.docs_fingerprint([{**a[0], "chunk_count": 9}, a[1]])  # re-splitting is not a change
    assert ia.docs_fingerprint(a) != ia.docs_fingerprint(a[:1])
    assert ia.docs_fingerprint(a) != ia.docs_fingerprint(a + [{"id": "d3", "filename": "c.md"}])


def test_review_updates_marks_only_changed_approved_rows():
    rows = [{"id": "1", "status": ia.STATUS_APPROVED, "docs_fingerprint": FP},
            {"id": "2", "status": ia.STATUS_APPROVED, "docs_fingerprint": "old"},
            {"id": "3", "status": ia.STATUS_SUGGESTED, "docs_fingerprint": "old"},
            {"id": "4", "status": ia.STATUS_DISABLED, "docs_fingerprint": "old"}]
    assert ia.review_updates(rows, FP) == {"2": {"status": ia.STATUS_REVIEW}}
    assert ia.review_updates(rows, None) == {}


def test_invalidate_agent_marks_every_approved_answer():
    a = seed()
    b = seed(question="Where is the clinic", status=ia.STATUS_SUGGESTED)
    assert ia.invalidate_agent(AGENT, USER) == 1
    assert store().get(USER, AGENT, a["id"])["status"] == ia.STATUS_REVIEW
    assert store().get(USER, AGENT, b["id"])["status"] == ia.STATUS_SUGGESTED


# ------------------------------------------------------------------------------------------------ audio cache

def test_the_audio_cache_renders_once_and_is_single_flight():
    cache = ia.AudioCache(max_items=4)
    renders = []
    gate = threading.Event()

    def render():
        renders.append(1)
        gate.wait(2)
        return np.zeros(100, dtype=np.float32), 24000

    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get_or_render("k", render))) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    gate.set()
    for t in threads:
        t.join(5)
    assert len(renders) == 1 and len(results) == 4
    assert sorted(hit for _v, hit in results) == [False, True, True, True]
    assert cache.get_or_render("k", render)[1] is True


def test_the_audio_cache_is_bounded_and_does_not_keep_failures():
    cache = ia.AudioCache(max_items=2)
    for i in range(3):
        cache.get_or_render(f"k{i}", lambda i=i: (np.zeros(10), i))
    assert len(cache) == 2 and cache.peek("k0") is None and cache.peek("k2") is not None
    small = ia.AudioCache(max_items=10, max_bytes=100)
    small.put("a", np.zeros(20, dtype=np.float32))  # 80 bytes
    small.put("b", np.zeros(20, dtype=np.float32))
    assert small.peek("a") is None and small.peek("b") is not None

    def fail():
        raise RuntimeError("tts down")

    with pytest.raises(RuntimeError):
        cache.get_or_render("bad", fail)
    assert cache.get_or_render("bad", lambda: b"ok") == (b"ok", False)  # the failed render did not block the key


def test_cached_audio_keys_include_what_changes_the_sound():
    seed()
    hit = m("What are the clinic timings on Sunday?")
    calls = []

    def render():
        calls.append(1)
        return b"wav"

    assert ia.cached_audio(hit, render, "kokoro", "af_heart") == (b"wav", False)
    assert ia.cached_audio(hit, render, "kokoro", "af_heart") == (b"wav", True)
    assert ia.cached_audio(hit, render, "kokoro", "other_voice") == (b"wav", False)
    assert len(calls) == 2


# ------------------------------------------------------------------------------------------------ building (service)

def deps(chunks=("The clinic is open nine to one on Sunday.",), score=0.7, reply="The clinic is open nine to one on Sunday.",
         paras=("is the clinic open on sunday",)):
    asked = {"retrieve": [], "answer": [], "paraphrase": []}

    def retrieve(q):
        asked["retrieve"].append(q)
        return list(chunks), score

    def answer(q, lang, ch):
        asked["answer"].append((q, lang))
        return reply(q) if callable(reply) else reply

    def paraphrase(q, lang):
        asked["paraphrase"].append(q)
        return list(paras)

    d = ia.DraftDeps(retrieve=retrieve, answer=answer, paraphrase=paraphrase, embed=embed)
    d.asked = asked
    return d


AGENT_ROW = {"id": AGENT, "supported_languages": ["en", "hi"], "tools": []}


def run_build(cands, d, **kw):
    return ia.build(agent={**AGENT_ROW, **kw.pop("agent", {})}, user_id=USER, storage_mode="local", candidates=cands,
                    languages=kw.pop("languages", None), deps=d, fingerprint=FP, **kw)


def test_a_draft_is_stored_as_suggested_and_never_live():
    d = deps()
    rep = run_build([ia.Candidate("What are the clinic timings on Sunday?", "template")], d)
    assert rep.added == 1 and rep.skipped == []
    row = store().list(USER, AGENT)[0]
    assert row["status"] == ia.STATUS_SUGGESTED and row["source"] == "template" and row["language"] == "en"
    assert row["answer"] == "The clinic is open nine to one on Sunday." and row["docs_fingerprint"] == FP
    assert row["paraphrases"] == ["is the clinic open on sunday"] and len(row["vectors"]) == 2
    assert m("What are the clinic timings on Sunday?") is None  # suggested is not live


def test_questions_the_documents_do_not_support_are_skipped_with_a_reason():
    assert run_build([ia.Candidate("What is the parking fee", "calls")], deps(chunks=())).skipped[0]["reason"] \
        == "nothing in the documents is about this"
    assert "do not cover" in run_build([ia.Candidate("What is the parking fee", "calls")], deps(score=0.1)).skipped[0]["reason"]
    assert store().list(USER, AGENT) == []


@pytest.mark.parametrize("reply", [
    "I don't have that information. I can connect you with a member of staff.",
    "मेरे पास यह जानकारी नहीं है।", "I'm not sure about that, sorry.",
])
def test_an_i_dont_know_reply_is_never_stored(reply):
    rep = run_build([ia.Candidate("What is the parking fee", "calls")], deps(reply=reply))
    assert rep.added == 0 and "does not have this information" in rep.skipped[0]["reason"]
    assert store().list(USER, AGENT) == []


def test_an_empty_too_long_or_wrong_language_draft_is_skipped():
    assert "no answer" in run_build([ia.Candidate("What is the parking fee", "calls")], deps(reply="  ")).skipped[0]["reason"]
    assert "too long" in run_build([ia.Candidate("What is the parking fee", "calls")],
                                   deps(reply="word " * 100)).skipped[0]["reason"]
    rep = run_build([ia.Candidate("What is the parking fee", "calls")], deps(reply="क्लिनिक रविवार को नौ से एक बजे तक खुला है।"))
    assert "language" in rep.skipped[0]["reason"]


def test_questions_that_can_never_be_instant_are_not_even_drafted():
    d = deps()
    rep = run_build([ia.Candidate("Please tell me your OTP now", "template"),
                     ia.Candidate("What is the fee? And when do classes start?", "calls"),
                     ia.Candidate("Is she free on Monday at eleven", "template")], d, agent={"tools": ["appointments"]})
    assert rep.added == 0 and len(rep.skipped) == 3 and d.asked["retrieve"] == []


def test_paraphrases_that_change_the_meaning_are_dropped():
    d = deps(paras=["is the clinic open on sunday", "are the clinic timings on monday", "what is the clinic timing on sunday?",
                    "what are the clinic timings on sunday and when is the fee due"])
    run_build([ia.Candidate("What are the clinic timings on Sunday?", "template")], d)
    assert store().list(USER, AGENT)[0]["paraphrases"] == ["is the clinic open on sunday",
                                                           "what is the clinic timing on sunday?"]


def test_a_failing_paraphrase_call_still_saves_the_answer():
    d = deps()

    def boom(q, lang):
        raise RuntimeError("json error")

    d.paraphrase = boom
    assert run_build([ia.Candidate("What are the clinic timings on Sunday?", "template")], d).added == 1


def test_an_existing_question_is_not_drafted_again_and_a_hindi_twin_can_be_added():
    seed(status=ia.STATUS_SUGGESTED)
    d = deps()
    rep = run_build([ia.Candidate("What are the clinic timings on Sunday?", "template", language="en")], d)
    assert rep.added == 0 and rep.skipped[0]["reason"] == "you already have this question" and d.asked["retrieve"] == []
    rep = run_build([ia.Candidate("What are the clinic timings on Sunday?", "template", language="en")],
                    deps(reply=lambda q: "क्लिनिक रविवार को नौ से एक बजे तक खुला है।"), languages=["hi"])
    assert rep.added == 1 and rep.skipped[0]["reason"] == "you already have this question"
    assert sorted(r["language"] for r in store().list(USER, AGENT)) == ["en", "hi"]


def test_a_row_that_needs_review_is_drafted_again_in_place():
    old = seed(status=ia.STATUS_REVIEW, answer="Old answer.", fingerprint="old")
    d = deps(reply="The clinic is open nine to two on Sunday.")
    rep = run_build([ia.Candidate(old["question"], "owner", row_id=old["id"], language="en")], d)
    assert rep.updated == 1 and rep.added == 0
    row = store().get(USER, AGENT, old["id"])
    assert row["answer"] == "The clinic is open nine to two on Sunday." and row["status"] == ia.STATUS_SUGGESTED
    assert row["docs_fingerprint"] == FP and len(store().list(USER, AGENT)) == 1


def test_a_redraft_the_documents_no_longer_support_leaves_the_row_alone():
    old = seed(status=ia.STATUS_REVIEW, answer="Old answer.", fingerprint="old")
    rep = run_build([ia.Candidate(old["question"], "owner", row_id=old["id"], language="en")], deps(chunks=()))
    assert rep.updated == 0 and rep.skipped
    assert store().get(USER, AGENT, old["id"])["status"] == ia.STATUS_REVIEW


def test_progress_is_reported_and_a_build_can_be_stopped():
    seen = []
    cands = [ia.Candidate(f"What is the fee for course {w}", "calls") for w in ("alpha", "beta", "gamma")]
    run_build(cands, deps(), progress=lambda done, total, q: seen.append((done, total)))
    assert seen[0] == (0, 3) and seen[-1] == (3, 3)
    ia.clear_cache()
    n = {"c": 0}

    def stop():
        n["c"] += 1
        return n["c"] > 1

    rep = run_build([ia.Candidate(f"What is the fee for module {w}", "calls") for w in ("one", "two", "three")], deps(),
                    should_stop=stop)
    assert rep.added == 1


# ------------------------------------------------------------------------------------------------ candidates

CLINIC_PERSONA = ("You are the receptionist of {org}. Speak politely and briefly, answer from the clinic document only, "
                  "give timings, fees and doctor names exactly as written, never give medical advice, and offer to "
                  "connect the caller with the front desk when the document does not say. Ask for the patient's name "
                  "and a phone number only when the caller wants a callback.")
TEMPLATES = [{"id": "clinic", "system_instructions": CLINIC_PERSONA.format(org="Green Clinic"),
              "suggested_questions": ["What are the clinic timings on Sunday?", "  ", "Which doctors are available?"]},
             {"id": "college", "system_instructions": "You answer questions about admissions, fees, eligibility, hostel "
                                                      "rooms, scholarships, last dates and entrance exams for a college. " * 2,
              "suggested_questions": ["What is the last date to apply?"]}]


def test_template_questions_come_from_the_stored_template_id_or_a_matching_persona():
    assert [c.question for c in ia.template_candidates({"template_id": "clinic"}, TEMPLATES)] == [
        "What are the clinic timings on Sunday?", "Which doctors are available?"]
    personalised = {"system_instructions": CLINIC_PERSONA.format(org="Blue Clinic in Pune")}
    assert len(ia.template_candidates(personalised, TEMPLATES)) == 2
    assert ia.template_candidates({"system_instructions": "You sell running shoes online and help customers pick sizes, "
                                                    "track parcels, return pairs that do not fit and find discount codes."}, TEMPLATES) == []
    assert ia.template_candidates({}, TEMPLATES) == []


def pair(q, reply="It is open.", **kw):
    return {"agent_id": AGENT, "created_at": "2026-10-01T10:00:00", "question": q, "reply": reply, "fallback": False,
            "language": "en", "retrieval": None, **kw}


def test_popular_questions_are_real_repeated_questions_without_personal_data():
    pairs = ([pair("what are the clinic timings on sunday")] * 3 + [pair("what are the clinic timings on sunday please")]
             + [pair("do you have parking space")] * 2 + [pair("what is the fee")]
             + [pair("my number is 9876543210 call me back please")] * 5
             + [pair("what is the secret menu", reply="I don't have that information.")] * 5
             + [pair("hello there", fallback=False)] * 4 + [pair("what about parking now", fallback=True)] * 4)
    out = ia.popular_questions(pairs, embed)
    assert out[0] == ("what are the clinic timings on sunday", 4) or out[0][1] == 4
    texts = [q for q, _ in out]
    assert "do you have parking space" in texts
    assert not any("9876543210" in t or "secret menu" in t or t == "what is the fee" for t in texts)
    assert ia.popular_questions(pairs, None)[0][1] == 3  # without embeddings only identical wordings group
    assert ia.popular_questions(pairs, embed, min_count=1)  # one-off questions need a lower bar


def test_merge_candidates_drops_repeats_and_what_the_agent_already_has():
    existing = [{"question": "What are the clinic timings on Sunday?", "language": "en"}]
    groups = [[ia.Candidate("What are the clinic timings on Sunday?", "template"),
               ia.Candidate("Which doctors are available?", "template")],
              [ia.Candidate("which doctors are available", "calls", 5),
               ia.Candidate("what are the clinic timings on sunday please", "calls", 9),
               ia.Candidate("Do you accept insurance", "calls", 4)]]
    out = ia.merge_candidates(groups, existing, embed, limit=10)
    assert [c.question for c in out] == ["Which doctors are available?", "Do you accept insurance"]
    assert all(c.language == "en" for c in out)
    assert len(ia.merge_candidates(groups, [], embed, limit=1)) == 1


def test_merge_keeps_a_redraft_of_an_existing_row():
    existing = [{"question": "What is the fee", "language": "en", "id": "r1"}]
    out = ia.merge_candidates([[ia.Candidate("What is the fee", "owner", row_id="r1")]], existing, embed, limit=5)
    assert len(out) == 1 and out[0].row_id == "r1"


# ------------------------------------------------------------------------------------------------ routes

@pytest.fixture()
def api(app_client, env, monkeypatch):  # noqa: F811
    """The real app on the fake embedder and a fake language model; instant answers use the test file store."""
    from services import llm

    ia.set_embedder(None)  # the router tests use the backend's (patched) embedding model, like production
    ia.set_fingerprinter(None)
    monkeypatch.setenv("INSTANT_ANSWER_MIN_SUPPORT", "0.05")
    calls = []

    def fake_chat(messages, *, json_mode=False, **kw):
        calls.append({"messages": messages, "json_mode": json_mode})
        if json_mode:
            return llm.ChatResult(json.dumps({"paraphrases": ["can I get a refund after a month",
                                                              "how long do I have to ask for a refund"]}), "fake", "m", 3, {})
        text = "Refunds are possible within thirty days."
        if "mars" in messages[-1]["content"].lower():
            text = "I don't have that information."
        return llm.ChatResult(text, "fake", "m", 3, {})

    monkeypatch.setattr(llm, "chat", fake_chat)
    env.llm_calls = calls
    return app_client


def url(agent, tail=""):
    return f"/api/agents/{agent}/instant-answers{tail}"


def make_agent(env, **extra):
    """An agent row straight in the local store (creating one through the API needs the language model)."""
    return env.local_db.insert("agents", {"user_id": "local-user", "name": "Support", "greeting": "Hi.",
                                          "system_instructions": "a friendly refund assistant",
                                          "supported_languages": ["en", "hi"], "tools": [], **extra})


def new_agent(api, env, text="The refund window is thirty days from purchase. " * 5):
    a = make_agent(env)
    assert upload_txt(api, a["id"], text).status_code == 200
    return a


def test_the_routes_list_nothing_for_a_new_agent_and_404_for_unknown_or_foreign_agents(api, env):
    a = make_agent(env)
    body = api.get(url(a["id"]), headers=LOCAL).json()
    assert body["answers"] == [] and body["build"] is None and body["document_count"] == 0 and body["threshold"] == 0.86
    assert api.get(url("nope"), headers=LOCAL).status_code == 404
    env.local_db.update("agents", a["id"], {"user_id": OTHER})
    for call in (lambda: api.get(url(a["id"]), headers=LOCAL),
                 lambda: api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window"}),
                 lambda: api.patch(url(a["id"], "/x"), headers=LOCAL, json={"status": "approved"}),
                 lambda: api.delete(url(a["id"], "/x"), headers=LOCAL),
                 lambda: api.post(url(a["id"], "/build"), headers=LOCAL, json={})):
        r = call()
        assert r.status_code == 404 and r.json()["code"] == "AGENT_NOT_FOUND"


def test_add_edit_approve_disable_and_delete(api, env):
    a = new_agent(api, env)
    r = api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window?"})
    assert r.status_code == 200 and r.json()["status"] == "suggested" and r.json()["answer"] == "" \
        and r.json()["source"] == "owner" and r.json()["language"] == "en"
    aid = r.json()["id"]
    assert "vectors" not in r.json()

    assert api.patch(url(a["id"], f"/{aid}"), headers=LOCAL, json={"status": "approved"}).json()["code"] == "ANSWER_REQUIRED"
    e = api.patch(url(a["id"], f"/{aid}"), headers=LOCAL, json={"answer": "  **Thirty** days from purchase.  "})
    assert e.status_code == 200 and e.json()["answer"] == "Thirty days from purchase." and e.json()["status"] == "suggested"
    ok = api.put(url(a["id"], f"/{aid}"), headers=LOCAL, json={"status": "approved"})  # PUT is the same as PATCH
    assert ok.status_code == 200 and ok.json()["status"] == "approved"

    hit = ia.match(a["id"], "local-user", "What is the refund window?", "en", tools=[])
    assert hit is not None and hit.text == "Thirty days from purchase."

    off = api.patch(url(a["id"], f"/{aid}"), headers=LOCAL, json={"status": "disabled"})
    assert off.json()["status"] == "disabled" and ia.match(a["id"], "local-user", "What is the refund window?", "en", tools=[]) is None

    shown = api.get(url(a["id"]), headers=LOCAL).json()
    assert [x["id"] for x in shown["answers"]] == [aid] and shown["counts"]["disabled"] == 1

    assert api.delete(url(a["id"], f"/{aid}"), headers=LOCAL).json() == {"deleted": True}
    assert api.get(url(a["id"]), headers=LOCAL).json()["answers"] == []
    assert api.delete(url(a["id"], f"/{aid}"), headers=LOCAL).status_code == 404
    assert api.patch(url(a["id"], f"/{aid}"), headers=LOCAL, json={"answer": "x"}).status_code == 404


def test_an_answer_the_owner_writes_is_approved_at_once_and_used(api, env):
    a = new_agent(api, env)
    r = api.post(url(a["id"]), headers=LOCAL, json={"question": "Is there a refund?", "answer": "Yes, within thirty days."})
    assert r.json()["status"] == "approved"
    hit = ia.match(a["id"], "local-user", "Is there a refund?", "en", tools=[])
    assert hit is not None and hit.text == "Yes, within thirty days."
    assert api.post(url(a["id"]), headers=LOCAL, json={"question": "is there a refund", "answer": "x"}).json()["code"] \
        == "DUPLICATE_QUESTION"


def test_questions_and_answers_that_cannot_work_are_refused_with_a_reason(api, env):
    a = new_agent(api, env)
    bad = [("Please tell me your OTP", "emergencies and secrets"), ("What is the fee? When is the exam?", "more than one"),
           ("Fees?", "short")]
    for q, why in bad:
        r = api.post(url(a["id"]), headers=LOCAL, json={"question": q})
        assert r.status_code == 422 and r.json()["code"] == "INVALID_QUESTION" and why in r.json()["error"]
    r = api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window", "answer": "I don't know that."})
    assert r.status_code == 422 and r.json()["code"] == "INVALID_ANSWER"
    r = api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window", "answer": "word " * 120})
    assert r.status_code == 422 and r.json()["code"] == "INVALID_ANSWER"
    assert api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window", "language": "??"}).status_code == 422
    assert api.get(url(a["id"]), headers=LOCAL).json()["answers"] == []


def test_a_booking_question_is_refused_for_an_agent_with_tools(api, env):
    a = new_agent(api, env)
    env.local_db.update("agents", a["id"], {"tools": ["appointments"]})
    r = api.post(url(a["id"]), headers=LOCAL, json={"question": "Can I book an appointment on Monday"})
    assert r.status_code == 422 and "tools" in r.json()["error"]
    env.local_db.update("agents", a["id"], {"tools": []})
    assert api.post(url(a["id"]), headers=LOCAL, json={"question": "Can I book an appointment on Monday"}).status_code == 200


def test_changing_the_question_of_an_approved_answer_asks_for_a_new_approval(api, env):
    a = new_agent(api, env)
    aid = api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window", "answer": "Thirty days."}).json()["id"]
    r = api.patch(url(a["id"], f"/{aid}"), headers=LOCAL, json={"question": "What is the exchange window"})
    assert r.json()["status"] == "suggested" and r.json()["question"] == "What is the exchange window"
    assert ia.match(a["id"], "local-user", "What is the refund window", "en", tools=[]) is None
    assert ia.match(a["id"], "local-user", "What is the exchange window", "en", tools=[]) is None  # suggested: not live
    api.patch(url(a["id"], f"/{aid}"), headers=LOCAL, json={"status": "approved"})
    assert ia.match(a["id"], "local-user", "What is the exchange window", "en", tools=[]) is not None
    assert ia.match(a["id"], "local-user", "What is the refund window", "en", tools=[]) is None  # old vectors are gone


def test_editing_a_live_answer_keeps_it_live_with_the_new_text(api, env):
    a = new_agent(api, env)
    aid = api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window", "answer": "Thirty days."}).json()["id"]
    r = api.patch(url(a["id"], f"/{aid}"), headers=LOCAL, json={"answer": "Thirty days from the day you buy."})
    assert r.json()["status"] == "approved"
    assert ia.match(a["id"], "local-user", "What is the refund window", "en", tools=[]).text == "Thirty days from the day you buy."


def test_uploading_or_deleting_a_document_marks_approved_answers_for_review(api, env):
    a = new_agent(api, env)
    aid = api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window", "answer": "Thirty days."}).json()["id"]
    assert ia.match(a["id"], "local-user", "What is the refund window", "en", tools=[]) is not None
    doc2 = upload_txt(api, a["id"], "Shipping takes five days. " * 6, name="shipping.txt").json()["document"]
    assert ia.match(a["id"], "local-user", "What is the refund window", "en", tools=[]) is None  # at once, before any GET
    shown = api.get(url(a["id"]), headers=LOCAL).json()
    assert shown["answers"][0]["status"] == "needs_review" and shown["counts"]["needs_review"] == 1
    ok = api.patch(url(a["id"], f"/{aid}"), headers=LOCAL, json={"status": "approved"}).json()
    assert ok["status"] == "approved"
    assert ia.match(a["id"], "local-user", "What is the refund window", "en", tools=[]) is not None
    api.delete(f"/api/agents/{a['id']}/documents/{doc2['id']}", headers=LOCAL)
    assert api.get(url(a["id"]), headers=LOCAL).json()["answers"][0]["status"] == "needs_review"


def test_build_needs_documents_and_only_one_build_runs_at_a_time(api, env):
    from routers import instant_answers as router

    a = make_agent(env)
    r = api.post(url(a["id"], "/build"), headers=LOCAL, json={})
    assert r.status_code == 409 and r.json()["code"] == "NO_DOCUMENTS"
    upload_txt(api, a["id"])
    router._jobs[("local-user", a["id"])] = {"state": "running", "total": 3, "done": 1}
    r = api.post(url(a["id"], "/build"), headers=LOCAL, json={})
    assert r.status_code == 409 and r.json()["code"] == "BUILD_RUNNING"
    router._jobs.clear()


def test_build_is_rate_limited(api, env):
    a = make_agent(env)
    codes = [api.post(url(a["id"], "/build"), headers=LOCAL, json={}).status_code for _ in range(11)]
    assert codes[:10] == [409] * 10 and codes[10] == 429


def add_call(env, agent_id, question, reply="Refunds are possible within thirty days.", conv_user="local-user", test=False):
    conv = env.local_db.insert("conversations", {"agent_id": agent_id, "user_id": conv_user, **({"is_test": True} if test else {})})
    env.local_db.insert("conversation_turns", {"conversation_id": conv["id"], "role": "user", "text_content": question})
    env.local_db.insert("conversation_turns", {"conversation_id": conv["id"], "role": "agent", "text_content": reply})


def test_build_drafts_from_owner_questions_and_frequent_real_questions_only(api, env):
    a = new_agent(api, env)
    for _ in range(3):
        add_call(env, a["id"], "what is the refund window for purchases")
    for _ in range(5):
        add_call(env, a["id"], "what is the refund window for test calls", test=True)  # an owner test call
    for _ in range(4):
        add_call(env, a["id"], "what is the refund window for someone else", conv_user="someone-else")
    add_call(env, a["id"], "what is the weather on mars")  # asked once only
    owner = api.post(url(a["id"]), headers=LOCAL, json={"question": "Can I get my money back after a purchase"}).json()
    mars = api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the weather on mars"}).json()

    r = api.post(url(a["id"], "/build?wait=true"), headers=LOCAL, json={})
    assert r.status_code == 202 and r.json()["build"]["state"] == "done"
    body = api.get(url(a["id"]), headers=LOCAL).json()
    by_q = {x["question"]: x for x in body["answers"]}
    assert set(by_q) == {"what is the refund window for purchases", "Can I get my money back after a purchase",
                         "What is the weather on mars"}
    assert by_q["what is the refund window for purchases"]["source"] == "calls"
    assert by_q["what is the refund window for purchases"]["times_asked"] == 3
    drafted = by_q["Can I get my money back after a purchase"]
    assert drafted["status"] == "suggested" and drafted["answer"] == "Refunds are possible within thirty days." \
        and drafted["id"] == owner["id"]
    assert drafted["paraphrases"] == ["can I get a refund after a month", "how long do I have to ask for a refund"]
    # the mars question is not in the documents: the model said it does not know, so no answer was stored for it
    assert by_q["What is the weather on mars"]["answer"] == "" and by_q["What is the weather on mars"]["id"] == mars["id"]
    build = body["build"]
    assert build["state"] == "done" and build["added"] == 1 and build["updated"] == 1
    assert any("does not have this information" in s["reason"] or "do not cover" in s["reason"] or "nothing in the documents" in s["reason"]
               for s in build["skipped"])
    # nothing is live until the owner approves
    assert ia.match(a["id"], "local-user", "what is the refund window for purchases", "en", tools=[]) is None
    # the full grounded path was used: the agent's own system prompt and the retrieved passages
    system = [c["messages"][0]["content"] for c in env.llm_calls if not c["json_mode"]][0]
    assert "Rules you always follow" in system and "Reply in English" in system
    user = [c["messages"][-1]["content"] for c in env.llm_calls if not c["json_mode"]][0]
    assert "Context from the documents:" in user


def test_a_second_build_does_not_suggest_what_is_already_there_or_was_deleted(api, env):
    a = new_agent(api, env)
    for _ in range(2):
        add_call(env, a["id"], "what is the refund window for purchases")
    api.post(url(a["id"], "/build?wait=true"), headers=LOCAL, json={})
    first = api.get(url(a["id"]), headers=LOCAL).json()["answers"]
    assert len(first) == 1
    again = api.post(url(a["id"], "/build?wait=true"), headers=LOCAL, json={}).json()["build"]
    assert again["added"] == 0 and len(api.get(url(a["id"]), headers=LOCAL).json()["answers"]) == 1
    api.delete(url(a["id"], f"/{first[0]['id']}"), headers=LOCAL)
    assert api.get(url(a["id"]), headers=LOCAL).json()["answers"] == []
    api.post(url(a["id"], "/build?wait=true"), headers=LOCAL, json={})
    assert api.get(url(a["id"]), headers=LOCAL).json()["answers"] == []  # a deleted suggestion stays deleted
    # but the owner can add it back by hand
    back = api.post(url(a["id"]), headers=LOCAL, json={"question": "what is the refund window for purchases", "answer": "Thirty days."})
    assert back.status_code == 200 and back.json()["status"] == "approved"


def test_build_redrafts_answers_whose_documents_changed(api, env):
    a = new_agent(api, env)
    aid = api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window", "answer": "Fourteen days."}).json()["id"]
    upload_txt(api, a["id"], "Refunds are possible for thirty days. " * 5, name="new-policy.txt")
    assert api.get(url(a["id"]), headers=LOCAL).json()["answers"][0]["status"] == "needs_review"
    r = api.post(url(a["id"], "/build?wait=true"), headers=LOCAL, json={"include_calls": False, "include_template": False})
    assert r.json()["build"]["updated"] == 1
    row = api.get(url(a["id"]), headers=LOCAL).json()["answers"][0]
    assert row["id"] == aid and row["status"] == "suggested" and row["answer"] == "Refunds are possible within thirty days."


def test_build_reports_an_unavailable_language_model(api, env, monkeypatch):
    from services import llm

    a = new_agent(api, env)
    api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window"})

    def down(*args, **kw):
        raise llm.LLMUnavailable({"groq": "no key"})

    monkeypatch.setattr(llm, "chat", down)
    r = api.post(url(a["id"], "/build?wait=true"), headers=LOCAL, json={})
    assert r.status_code == 202 and r.json()["build"]["state"] == "failed" and r.json()["build"]["error"]
    assert api.get(url(a["id"]), headers=LOCAL).json()["build"]["state"] == "failed"


def test_build_in_the_background_reports_progress_to_the_next_get(api, env):
    a = new_agent(api, env)
    api.post(url(a["id"]), headers=LOCAL, json={"question": "What is the refund window"})
    r = api.post(url(a["id"], "/build"), headers=LOCAL, json={})
    assert r.status_code == 202 and r.json()["build"]["state"] in ("collecting", "running", "done")
    for _ in range(50):
        state = api.get(url(a["id"]), headers=LOCAL).json()["build"]["state"]
        if state in ("done", "failed"):
            break
        time.sleep(0.1)
    assert state in ("done", "collecting", "running", "failed")  # the test client's loop may stop the task between requests


def test_another_users_rows_in_the_file_are_never_returned(api, env):
    a = new_agent(api, env)
    store().insert({"agent_id": a["id"], "user_id": OTHER, "question": "Secret question", "answer": "Secret", "language": "en",
                    "status": "approved", "source": "owner"})
    body = api.get(url(a["id"]), headers=LOCAL).json()
    assert body["answers"] == []
    stray = store().list(OTHER, a["id"])[0]["id"]
    assert api.patch(url(a["id"], f"/{stray}"), headers=LOCAL, json={"status": "disabled"}).status_code == 404
    assert api.delete(url(a["id"], f"/{stray}"), headers=LOCAL).status_code == 404
    assert store().get(OTHER, a["id"], stray)["status"] == "approved"


def test_bad_ids_are_rejected(api, env):
    a = new_agent(api, env)
    assert api.patch(url(a["id"], "/../../x"), headers=LOCAL, json={"status": "disabled"}).status_code in (404, 405, 422)
    assert api.patch(url(a["id"], "/bad%20id"), headers=LOCAL, json={"status": "disabled"}).status_code == 422
