"""services/query_understanding.py: the agent's Whisper vocabulary prompt, entity snapping, ASR confidence and the
clarification it can trigger, follow-up condensation, and how a turn hands the search query to retrieval.
Everything is faked: no Groq, no Whisper, no network."""
import asyncio
import os
import time
import types

import pytest

from services import query_understanding as qu

HERE = os.path.dirname(os.path.abspath(__file__))
CLINIC_DOC = open(os.path.join(HERE, "..", "templates", "agents", "clinic_receptionist.md"), encoding="utf-8").read()
AGENT = {"id": "clinic-1", "name": "Sunrise Family Clinic Receptionist",
         "greeting": "Hello, thank you for calling Sunrise Family Clinic. How can I help you today?",
         "supported_languages": ["en", "hi"]}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("ASR_VOCAB_PROMPT", "ASR_VOCAB_PROMPT_TOKENS", "ASR_LANGUAGE_HINT", "ASR_VERBOSE_JSON", "ASR_CLARIFY",
              "QUERY_ENTITY_SNAP", "QUERY_CONDENSE", "ASR_LOW_LOGPROB", "ASR_HIGH_LOGPROB"):
        monkeypatch.delenv(k, raising=False)
    qu.set_document_loader(None)
    qu._CLARIFIED.clear()
    yield
    qu.set_document_loader(None)
    qu._CLARIFIED.clear()


@pytest.fixture()
def lex():
    return qu.build_profile(AGENT, [CLINIC_DOC], "fp").lexicon


def H(*pairs):
    return [{"role": r, "text_content": t} for r, t in pairs]


# ---------------------------------------------------------------------------------------------- vocabulary prompt

def test_entities_are_extracted_deterministically():
    ents = qu.extract_entities([CLINIC_DOC])
    people = [e.text for e in ents if e.kind == "person"]
    assert people == sorted(people) and set(people) == {
        "Dr. Meera Kulkarni", "Dr. Rohan Deshmukh", "Dr. Anjali Rao", "Dr. Imran Shaikh", "Dr. Sunita Patil"}
    texts = {e.text for e in ents}
    assert {"Sunrise Family Clinic", "Lake View Road", "Shivajinagar", "Paediatrician", "OPD", "UPI"} <= texts
    assert "Monday" not in texts and "The" not in texts and "AM" not in texts
    assert qu.extract_entities([CLINIC_DOC]) == ents  # same input, same output


def test_devanagari_titled_names_are_entities():
    ents = qu.extract_entities(["डॉ. मीरा कुलकर्णी सोमवार को बैठती हैं। डॉक्टर रोहन कब आते हैं?"])
    people = {e.text for e in ents if e.kind == "person"}
    assert "डॉ. मीरा कुलकर्णी" in people and "डॉक्टर रोहन" in people


def test_vocab_prompt_is_plain_terms_with_people_last_and_within_the_token_limit():
    prompt = qu.build_profile(AGENT, [CLINIC_DOC], "fp").prompt
    assert prompt.endswith("Dr. Sunita Patil.") and "Dr. Meera Kulkarni" in prompt
    assert "Paediatrician" in prompt and "Gynaecologist" in prompt
    assert prompt.index("Paediatrician") < prompt.index("Dr. Anjali Rao")
    for instruction in ("transcribe", "please", "spell", "correct"):
        assert instruction not in prompt.lower()
    assert qu.estimate_tokens(prompt) <= 100           # default budget
    assert ", Kulkarni," not in prompt                  # a word of a longer name is not repeated on its own


def test_prompt_budget_is_capped_at_200_and_the_estimate_over_counts_whisper():
    big = CLINIC_DOC + "\n" + "\n".join(f"- Dr. Name{i} Surname{i}, Specialist{i}: Monday." for i in range(80))
    os.environ["ASR_VOCAB_PROMPT_TOKENS"] = "5000"
    try:
        prompt = qu.build_profile(AGENT, [big], "fp").prompt
    finally:
        del os.environ["ASR_VOCAB_PROMPT_TOKENS"]
    assert 150 < qu.estimate_tokens(prompt) <= qu.PROMPT_TOKEN_LIMIT == 200
    try:
        from whisper.tokenizer import get_tokenizer
    except Exception:  # noqa: BLE001
        pytest.skip("openai-whisper not installed")
    tok = get_tokenizer(multilingual=True)
    assert len(tok.encode(prompt)) <= 224
    for s in (prompt, "डॉ. मीरा कुलकर्णी, शिवाजीनगर, पुणे, ₹400 फीस", "Dr. Meera Kulkarni, Dr. Rohan Deshmukh."):
        assert qu.estimate_tokens(s) >= len(tok.encode(s))


def test_profile_is_cached_per_document_fingerprint_and_rebuilt_when_documents_or_agent_change(monkeypatch):
    calls = []
    state = {"fp": "f1", "texts": [CLINIC_DOC]}

    def loader(agent_id, user_id, mode):
        calls.append(agent_id)
        return state["fp"], state["texts"]

    qu.set_document_loader(loader)
    monkeypatch.setattr(qu, "CLOUD_CHECK_S", 0.0)   # every call checks the fingerprint (in the background)
    first = qu.get_profile(AGENT, "u", "cloud", wait=True)
    assert first is not None and "Dr. Meera Kulkarni" in first.prompt
    again = qu.get_profile(AGENT, "u", "cloud", wait=True)
    assert again is first and len(calls) == 2         # checked, same fingerprint: not rebuilt
    state.update(fp="f2", texts=[CLINIC_DOC.replace("Meera Kulkarni", "Kavita Joshi")])
    changed = qu.get_profile(AGENT, "u", "cloud", wait=True)
    assert changed is not first and "Dr. Kavita Joshi" in changed.prompt
    renamed = qu.get_profile({**AGENT, "greeting": "Namaste, Sunrise Clinic."}, "u", "cloud", wait=True)
    assert renamed is not changed


def test_profile_lookup_never_blocks_a_turn(monkeypatch):
    def slow_loader(agent_id, user_id, mode):
        time.sleep(0.3)
        return "f", [CLINIC_DOC]

    qu.set_document_loader(slow_loader)
    started = time.perf_counter()
    assert qu.get_profile(AGENT, "u", "local") is None                 # not built yet: the turn goes on without
    assert time.perf_counter() - started < 0.05
    deadline = time.time() + 3
    while qu.get_profile(AGENT, "u", "local") is None and time.time() < deadline:
        time.sleep(0.02)
    assert qu.get_profile(AGENT, "u", "local") is not None


def test_asr_hints_follow_the_switches(monkeypatch):
    qu.set_document_loader(lambda *a: ("f", [CLINIC_DOC]))
    qu.get_profile(AGENT, "u", "local", wait=True)
    hints = qu.turn_asr_hints(AGENT, "u", "local")
    assert hints["prompt"] and hints["verbose"] is True
    monkeypatch.setenv("ASR_VOCAB_PROMPT", "off")
    monkeypatch.setenv("ASR_VERBOSE_JSON", "off")
    assert qu.turn_asr_hints(AGENT, "u", "local") == {"prompt": None, "verbose": False}


# ---------------------------------------------------------------------------------------------- language passing

def test_language_is_passed_only_for_a_single_language_agent(monkeypatch):
    assert qu.asr_language({"supported_languages": ["en"]}) == "en"
    assert qu.asr_language({"supported_languages": ["hi"]}) == "hi"
    assert qu.asr_language({"supported_languages": ["hi-IN"]}) == "hi"
    assert qu.asr_language({}) == "en"                                  # the default agent language
    assert qu.asr_language({"supported_languages": ["en", "hi"]}) is None   # Hinglish must keep working
    assert qu.asr_language({"supported_languages": ["hinglish"]}) is None   # Whisper has no Hinglish code
    assert qu.asr_language({"supported_languages": ["hi", "hinglish"]}) is None
    monkeypatch.setenv("ASR_LANGUAGE_HINT", "off")
    assert qu.asr_language({"supported_languages": ["en"]}) is None


# ---------------------------------------------------------------------------------------------- snapping

@pytest.mark.parametrize("heard, expected", [
    ("dr meera kulkarany ki fees kitni hai", "dr meera Kulkarni ki fees kitni hai"),
    ("Dr. Mira kab baithti hain", "Dr. Meera kab baithti hain"),
    ("pediatrician kab milte hain", "Paediatrician kab milte hain"),
    ("what is the gynecologist fee", "what is the Gynaecologist fee"),
    ("Shivaji nagar mein clinic kahan hai", "Shivajinagar mein clinic kahan hai"),
    ("Dr Rohan Desmukh timing", "Dr Rohan Deshmukh timing"),
    ("is Dr Imran Shaik available on Friday", "is Dr Imran Shaikh available on Friday"),
    ("dermatologyst fees", "Dermatologist fees"),
])
def test_snapping_fixes_near_miss_names_and_terms(lex, heard, expected):
    out, changes = qu.snap_entities(heard, lex)
    assert out == expected and changes and all({"from", "to", "score"} <= set(c) for c in changes)


@pytest.mark.parametrize("heard", [
    "mera appointment book karo",              # "mera" (my) is not "Meera"
    "patients waiting time kitna hai",
    "please book the appointment for tomorrow",
    "my number is 9876543210",
    "fees ₹400 hai ya 500",
    "डॉ. मीरा कब बैठती हैं",                    # Devanagari is left alone
    "मेरा अपॉइंटमेंट बुक कर दो",
    "What are the timings at 9 am",
    "Dr. Meera Kulkarni",                      # already right
])
def test_snapping_leaves_common_words_numbers_and_hindi_alone(lex, heard):
    assert qu.snap_entities(heard, lex) == (heard, [])


def test_devanagari_changes_only_through_an_exact_alias(lex):
    lex.add_alias("मीरा", "Meera")
    assert qu.snap_entities("डॉ. मीरा कब बैठती हैं", lex)[0] == "डॉ. Meera कब बैठती हैं"
    assert qu.snap_entities("डॉ. मिरा कब बैठती हैं", lex)[1] == []      # a near miss in Devanagari: untouched


def test_an_ambiguous_near_miss_is_not_snapped():
    lexicon = qu.build_lexicon([qu.Entity("Dr. Rohan Sharma", "person"), qu.Entity("Dr. Rohin Verma", "person")])
    assert qu.snap_entities("doctor Rohen", lexicon)[1] == []
    assert qu.snap_entities("anything", None) == ("anything", [])


def test_snapping_switch(monkeypatch, lex):
    monkeypatch.setenv("QUERY_ENTITY_SNAP", "off")
    q, info = asyncio.run(qu.search_query("dr meera kulkarany", [], lex))
    assert q == "dr meera kulkarany" and info["snap"] == []


# ---------------------------------------------------------------------------------------------- confidence

def seg(lp, nsp=0.01, cr=1.2, text="x", start=0.0, end=1.0):
    return {"avg_logprob": lp, "no_speech_prob": nsp, "compression_ratio": cr, "text": text, "start": start, "end": end}


def test_confidence_levels_from_verbose_json_segments():
    assert qu.asr_confidence(None) == {"level": "unknown"}
    assert qu.asr_confidence([seg(-0.2)])["level"] == "high"
    assert qu.asr_confidence([seg(-0.6)])["level"] == "medium"
    low = qu.asr_confidence([seg(-1.2, text="kulkarni")])
    assert low["level"] == "low" and low["weak_segments"] == ["kulkarni"]
    assert qu.asr_confidence([seg(-0.6, nsp=0.8)])["level"] == "low"          # probably not speech
    assert qu.asr_confidence([seg(-0.1, cr=3.0)])["level"] == "low"           # repetitive hallucination
    # duration-weighted: a long good segment outweighs a short bad one
    mixed = qu.asr_confidence([seg(-0.2, start=0, end=4), seg(-1.5, text="98900", start=4, end=4.5)])
    assert mixed["level"] == "high" and mixed["min_logprob"] == -1.5 and mixed["weak_segments"] == ["98900"]


def test_segments_are_read_from_groq_results_and_dicts():
    class Transcription:   # like the Groq SDK's pydantic model: verbose_json fields are extras
        text = "hello"
        model_extra = {"segments": [seg(-0.3)]}

    assert qu.segments_of(Transcription())[0]["avg_logprob"] == -0.3
    assert qu.segments_of({"segments": [seg(-0.4)]})[0]["avg_logprob"] == -0.4
    assert qu.segments_of(types.SimpleNamespace(text="x")) is None


# ---------------------------------------------------------------------------------------------- clarification

LOW = {"level": "low", "weak_segments": []}


def test_clarification_only_on_low_confidence_phone_turns(monkeypatch, lex):
    assert qu.should_clarify(LOW, "Monday ko?", lex, source="phone") == "generic"
    assert qu.should_clarify({"level": "medium"}, "Monday ko?", lex, source="phone") is None
    assert qu.should_clarify({"level": "unknown"}, "Monday ko?", lex, source="phone") is None
    assert qu.should_clarify(LOW, "Monday ko?", lex, source="widget") is None      # phone only by default
    monkeypatch.setenv("ASR_CLARIFY", "on")
    assert qu.should_clarify(LOW, "Monday ko?", lex, source="widget") == "generic"
    monkeypatch.setenv("ASR_CLARIFY", "off")
    assert qu.should_clarify(LOW, "Monday ko?", lex, source="phone") is None


def test_clarification_needs_a_short_turn_or_a_doubtful_name_or_number(lex):
    long_plain = "I wanted to know whether the clinic is open on public holidays as well"
    assert qu.should_clarify(LOW, long_plain, lex, source="phone") is None
    weak_doctor = {"level": "low", "weak_segments": ["doctor kulkarany"]}
    assert qu.should_clarify(weak_doctor, "I want to book with doctor kulkarany next week please if possible",
                             lex, source="phone") == "doctor"
    weak_number = {"level": "low", "weak_segments": ["98900 555"]}
    assert qu.should_clarify(weak_number, "my number is 98900 555 and please call me back today",
                             lex, source="phone") == "number"
    assert qu.should_clarify(LOW, "Dr Rohan?", lex, source="phone") == "doctor"


def test_never_clarify_twice_in_a_row(lex):
    asked = qu.clarification_text("doctor", "hinglish")
    assert qu.should_clarify(LOW, "Monday ko?", lex, source="phone", last_agent_text=asked) is None
    disclosed = "This is an AI assistant. " + qu.clarification_text("generic", "en")
    assert qu.is_clarification(disclosed)
    assert qu.clarification_answer("a", "generic", "en", disclosed, "c1") is None
    first = qu.clarification_answer("a", "generic", "en", "Hello!", "c2")
    assert first is not None and first.id == "asr_clarify" and first.text == qu.CLARIFY_TEXTS["en"]["generic"]
    assert qu.clarification_answer("a", "generic", "en", None, "c2") is None      # history unknown: memory says no
    qu.note_answered("c2")
    assert qu.clarification_answer("a", "generic", "en", None, "c2") is not None


def test_clarification_speaks_the_callers_language():
    assert qu.clarification_text("doctor", "en") == "Sorry, I didn't catch that. Which doctor did you mean?"
    assert "डॉक्टर" in qu.clarification_text("doctor", "hi")
    assert "kis doctor" in qu.clarification_text("doctor", "hinglish")
    assert qu.clarification_text("number", "ta") == qu.CLARIFY_TEXTS["en"]["number"]


# ---------------------------------------------------------------------------------------------- condensation

def run_condense(text, history, lexicon=None, llm_call=None, timeout_s=None):
    return asyncio.run(qu.condense(text, history, lexicon, llm_call=llm_call or no_llm, timeout_s=timeout_s))


def no_llm(text, history, timeout):
    raise AssertionError("the LLM must not be called")


@pytest.mark.parametrize("text, history, expected", [
    # Hinglish: the slot goes before the verb
    ("aur Monday ko?", H(("user", "Dr. Meera kab baithti hain?"), ("agent", "Somvar, budhvar aur shukravar.")),
     "Dr. Meera Monday ko kab baithti hain"),
    ("kya wo Sunday ko bhi baithti hain?", H(("user", "Dr. Meera kab baithti hain?"), ("agent", "...")),
     "Dr. Meera Sunday ko bhi baithti hain"),
    ("uski fees kitni hai?", H(("user", "Bachon ke doctor kaun hain?"),
                               ("agent", "Dr. Rohan Deshmukh hamare Paediatrician hain.")),
     "Dr. Rohan Deshmukh fees kitni hai"),
    ("aur Dr. Rohan?", H(("user", "Dr. Meera kab baithti hain?"), ("agent", "...")), "Dr. Rohan kab baithti hain"),
    # English: the slot goes at the end
    ("and on Saturday?", H(("user", "When does Dr. Rohan sit?"), ("agent", "Tuesday, Thursday and Saturday.")),
     "When does Dr. Rohan sit on Saturday"),
    ("what about her fees?", H(("user", "Is Dr. Anjali Rao available on Monday?"), ("agent", "Yes, from 4 PM.")),
     "Dr. Anjali Rao fees"),
    # Hindi
    ("और सोमवार को?", H(("user", "डॉ. मीरा कब बैठती हैं?"), ("agent", "बुधवार को।")), "डॉ. मीरा सोमवार को कब बैठती हैं"),
    ("और उनकी फीस?", H(("user", "डॉ. मीरा कब बैठती हैं?"), ("agent", "सोमवार को।")), "डॉ. मीरा फीस"),
])
def test_follow_ups_are_condensed_by_rules(lex, text, history, expected):
    q, info = run_condense(text, history, lex)
    assert q == expected and info["path"] == "rules"


@pytest.mark.parametrize("text", [
    "What are the clinic timings?",
    "How much does the dentist charge for a cleaning?",
    "ok thank you",
    "Thank you so much",
    "हाँ जी",
])
def test_non_follow_ups_are_never_rewritten(lex, text):
    history = H(("user", "Dr. Meera kab baithti hain?"), ("agent", "Monday ko."))
    assert run_condense(text, history, lex) == (text, {"path": "none", "ms": pytest.approx(0, abs=50)})


def test_no_history_means_no_rewrite(lex):
    assert run_condense("aur Monday ko?", [], lex)[0] == "aur Monday ko?"


def test_llm_is_used_only_when_the_rules_cannot(lex):
    seen = []

    def fake_llm(text, history, timeout):
        seen.append((text, timeout))
        return "Sunrise clinic phone number"

    history = H(("user", "clinic kab khulta hai?"), ("agent", "9 baje."))
    q, info = run_condense("uska number kya hai?", history, lex, llm_call=fake_llm)
    assert q == "Sunrise clinic phone number" and info["path"] == "llm" and seen[0][1] == pytest.approx(0.4)
    # QUERY_CONDENSE=rules never calls the model
    os.environ["QUERY_CONDENSE"] = "rules"
    try:
        assert run_condense("uska number kya hai?", history, lex)[1]["path"] == "none"
    finally:
        del os.environ["QUERY_CONDENSE"]


def test_llm_timeout_and_failure_fall_back_to_the_raw_text(lex):
    history = H(("user", "clinic kab khulta hai?"), ("agent", "9 baje."))

    def slow(text, history, timeout):
        time.sleep(0.5)
        return "too late"

    q, info = run_condense("aur?", history, lex, llm_call=slow, timeout_s=0.1)
    assert q == "aur?" and info["path"] == "llm_timeout" and info["ms"] < 300   # the turn waited 0.1 s, not 0.5

    def broken(text, history, timeout):
        raise RuntimeError("HTTP 503")

    q, info = run_condense("aur?", history, lex, llm_call=broken)
    assert q == "aur?" and info["path"] == "llm_failed"


def test_real_llm_call_is_tiny_and_groq_only(monkeypatch):
    from services import llm
    calls = []

    def fake_chat(messages, **kw):
        calls.append((messages, kw))
        return llm.ChatResult('"Dr. Meera fees"\n', "groq", "m", 5, {})

    monkeypatch.setattr(llm, "chat", fake_chat)
    out = qu._llm_condense("uski fees?", H(("user", "Dr. Meera kab?"), ("agent", "Monday.")), 0.4)
    assert out == "Dr. Meera fees"
    kw = calls[0][1]
    assert kw["max_tokens"] == 40 and kw["timeout"] == 0.4 and kw["providers"] == ["groq"]


def test_condense_off_and_answers_to_the_agent_are_left_alone(monkeypatch, lex):
    history = H(("user", "Dr. Meera kab baithti hain?"), ("agent", "Aapka naam bataiye?"))
    q, info = asyncio.run(qu.search_query("Monday ko?", history, lex, answering_agent=True))
    assert q == "Monday ko?" and info["condense"]["path"] == "none"
    monkeypatch.setenv("QUERY_CONDENSE", "off")
    assert asyncio.run(qu.condense("aur Monday ko?", history, lex))[0] == "aur Monday ko?"


def test_search_query_snaps_then_condenses(lex):
    history = H(("user", "Dr. Meera kab baithti hain?"), ("agent", "Monday ko."))
    q, info = asyncio.run(qu.search_query("aur Dr. Rohan Desmukh?", history, lex, llm_call=no_llm))
    assert q == "Dr. Rohan Deshmukh kab baithti hain"
    assert info["snap"][0]["to"] == "Deshmukh" and info["condense"]["path"] == "rules"


# ---------------------------------------------------------------------------------------------- latency

def test_added_work_per_turn_is_well_under_ten_milliseconds(lex):
    history = H(("user", "Dr. Meera kab baithti hain?"), ("agent", "Dr. Meera Kulkarni Monday ko baithti hain."))
    texts = ["dr meera kulkarany ki fees kitni hai", "aur Monday ko?", "What are the clinic timings on Sunday?",
             "और सोमवार को?", "pediatrician kab milte hain"]
    segments = [seg(-0.4, start=0, end=2), seg(-0.9, start=2, end=3)]
    n = 200
    started = time.perf_counter()
    for i in range(n):
        t = texts[i % len(texts)]
        conf = qu.asr_confidence(segments, t)
        _, changes = qu.snap_entities(t, lex)
        qu.should_clarify(conf, t, lex, changes, source="phone")
        asyncio.run(qu.search_query(t, history, lex, llm_call=no_llm))
    per_turn_ms = (time.perf_counter() - started) * 1000 / n
    print(f"query understanding per turn: {per_turn_ms:.3f} ms (incl. an asyncio.run per call)")
    assert per_turn_ms < 10


# ---------------------------------------------------------------------------------------------- backend plumbing

@pytest.fixture()
def B():
    import backend
    return backend


def make_run(B, history_rows, *, agent=None, source=None):
    """A cloud-mode turn whose conversation history comes from a fake Supabase (newest first, like the real query)."""

    class Query:
        def __init__(self, rows):
            self.rows = rows

        def __getattr__(self, name):
            return lambda *a, **k: self

        def execute(self):
            return types.SimpleNamespace(data=self.rows)

    class FakeSb:
        def table(self, name):
            return Query([{"id": "conv", "reply_language": None}] if name == "conversations"
                         else list(reversed(history_rows)))

    run = B._TurnRun.__new__(B._TurnRun)
    run.local, run.sb, run.agent_id, run.user_id, run.storage_mode = False, FakeSb(), "clinic-1", "u", "cloud"
    run.timings, run.is_test, run.source, run.text_input = {}, False, source, None
    run.agent = agent or dict(AGENT)
    run.history = []
    return run


def test_turn_context_searches_the_condensed_query_but_the_model_sees_the_transcript(B, monkeypatch):
    qu.set_document_loader(lambda *a: ("f", [CLINIC_DOC]))
    qu.get_profile(AGENT, "u", "cloud", wait=True)
    asked = []

    async def fake_retrieve(agent_id, query, *a, **k):
        asked.append(query)
        return ["Dr. Meera Kulkarni, General Physician: Monday, Wednesday and Friday."]

    monkeypatch.setattr(B, "retrieve_context", fake_retrieve)
    run = make_run(B, H(("user", "Dr. Meera kab baithti hain?"), ("agent", "Budhvar aur shukravar ko.")))
    run.user_text = "aur Monday ko?"
    asyncio.run(B._turn_context(run, "conv", time.time()))
    assert asked == ["Dr. Meera Monday ko kab baithti hain"]
    assert run.search_query == asked[0] and run.user_text == "aur Monday ko?"


def test_turn_context_without_a_follow_up_searches_the_transcript(B, monkeypatch):
    asked = []

    async def fake_retrieve(agent_id, query, *a, **k):
        asked.append(query)
        return []

    monkeypatch.setattr(B, "retrieve_context", fake_retrieve)
    run = make_run(B, [])
    run.user_text = "When does the clinic open?"
    asyncio.run(B._turn_context(run, "conv", time.time()))
    assert asked == ["When does the clinic open?"]


def test_a_clarification_replaces_retrieval_and_the_answer_once(B, monkeypatch):
    asked = []

    async def fake_retrieve(agent_id, query, *a, **k):
        asked.append(query)
        return []

    monkeypatch.setattr(B, "retrieve_context", fake_retrieve)
    run = make_run(B, H(("user", "Hello"), ("agent", "Hi, how can I help?")), source="phone")
    run.user_text, run.clarify_text, run.clarify_kind = "Dr Rohan?", "x", "doctor"
    asyncio.run(B._turn_context(run, "conv", time.time()))
    assert asked == [] and run.instant is not None and run.instant.id == "asr_clarify"
    assert run.clarify_text == run.instant.text == "Sorry, I didn't catch that. Which doctor did you mean?"
    assert run.chunks == [] and run.retrieval is None
    # the agent just asked: this time the turn is answered
    qu.note_answered("conv")
    run2 = make_run(B, H(("user", "Dr Rohan?"), ("agent", run.instant.text)), source="phone")
    run2.user_text, run2.clarify_text, run2.clarify_kind = "Dr Rohan?", "x", "doctor"
    asyncio.run(B._turn_context(run2, "conv", time.time()))
    assert run2.clarify_text is None and len(asked) == 1


def test_turn_asr_passes_prompt_language_and_reads_confidence(B, monkeypatch):
    qu.set_document_loader(lambda *a: ("f", [CLINIC_DOC]))
    agent = {**AGENT, "supported_languages": ["en"]}
    qu.get_profile(agent, "u", "cloud", wait=True)
    got = {}

    async def fake_asr(audio_bytes, name, language=None):
        hints = qu.asr_hints()
        got.update(language=language, prompt=hints["prompt"], verbose=hints["verbose"])
        hints["segments"] = [seg(-1.3, text="doctor rohan")]
        return "doctor Rohan?"

    monkeypatch.setattr(B, "_speech_to_text", fake_asr)
    run = make_run(B, [], agent=agent, source="phone")
    asyncio.run(B._turn_asr(run, b"wav"))
    assert got["language"] == "en" and "Dr. Rohan Deshmukh" in got["prompt"] and got["verbose"] is True
    assert run.asr_confidence["level"] == "low" and run.clarify_text.endswith("Which doctor did you mean?")
    assert run.search_query == run.user_text == "doctor Rohan?"
    assert qu.asr_hints() is None                      # the context is reset after the call

    # a multilingual agent: no language forced; a confident turn: no clarification
    async def confident(audio_bytes, name, language=None):
        got["language"] = language
        qu.asr_hints()["segments"] = [seg(-0.1)]
        return "aur Monday ko?"

    monkeypatch.setattr(B, "_speech_to_text", confident)
    run = make_run(B, [], source="phone")
    asyncio.run(B._turn_asr(run, b"wav"))
    assert got["language"] is None and run.clarify_text is None and run.asr_confidence["level"] == "high"


def test_speech_to_text_sends_prompt_and_verbose_json_to_groq_only_inside_a_turn(B, monkeypatch):
    calls = []

    class Transcriptions:
        def create(self, **kw):
            calls.append(kw)
            if kw.get("prompt") == "boom":
                raise RuntimeError("HTTP 400 prompt rejected")
            return types.SimpleNamespace(text=" hello ", model_extra={"segments": [seg(-0.2)]})

    monkeypatch.setattr(B, "groq_client", types.SimpleNamespace(audio=types.SimpleNamespace(
        transcriptions=Transcriptions())))
    assert asyncio.run(B._speech_to_text(b"RIFF....WAVE", "consent", "en")) == "hello"
    assert "prompt" not in calls[-1] and "response_format" not in calls[-1]       # consent etc.: unchanged

    async def in_turn(hints):
        token = qu.ASR_CTX.set(hints)
        try:
            return await B._speech_to_text(b"RIFF....WAVE", "turn", None)
        finally:
            qu.ASR_CTX.reset(token)

    hints = {"prompt": "Dr. Meera Kulkarni.", "verbose": True}
    assert asyncio.run(in_turn(hints)) == "hello"
    assert calls[-1]["prompt"] == "Dr. Meera Kulkarni." and calls[-1]["response_format"] == "verbose_json"
    assert hints["segments"][0]["avg_logprob"] == -0.2 and hints["prompt_used"] is True

    bad = {"prompt": "boom", "verbose": True}   # a rejected prompt never costs the transcript: one plain retry
    assert asyncio.run(in_turn(bad)) == "hello"
    assert "prompt" not in calls[-1] and bad["prompt_used"] is False and bad["segments"] is None


def test_local_lexicon_reads_passages_with_headers_and_skips_generated_questions(B, monkeypatch, tmp_path):
    from services.local_store import LocalStore
    db = LocalStore(str(tmp_path))
    doc = db.insert("documents", {"agent_id": "ag", "user_id": "u", "filename": "a.md"})
    p = db.insert("document_chunks", {"document_id": doc["id"], "chunk_text": "Timings: 9 to 5.",
                                      "header": "Sunrise Clinic › Dr. Kavita Joshi"})
    db.insert("document_chunks", {"document_id": doc["id"], "chunk_text": "Is Dr. Fakename Wrongly available?",
                                  "kind": "question", "parent_chunk_id": p["id"]})
    db.insert("document_chunks", {"document_id": doc["id"], "chunk_text": "Dr. Old Passage works Monday.",
                                  "kind": None})
    monkeypatch.setattr(B, "local_db", db)
    fp, texts = qu._load_documents("ag", "u", "local")
    assert fp and any("Dr. Kavita Joshi" in t for t in texts) and any("Old Passage" in t for t in texts)
    assert not any("Fakename" in t for t in texts)
