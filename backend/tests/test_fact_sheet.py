"""Per-agent fact sheet (services/fact_sheet.py): extraction with a faked model, verbatim validation, the on-disk cache
keyed by the documents' fingerprint, the offline skip, relevance per question and the non-blocking turn hook."""
import json
import time
from types import SimpleNamespace

import pytest

from services import fact_sheet as fs
from services import llm

DOC = """Sahyadri Clinic
OPD timings: Monday to Saturday, 9:30 AM to 1:00 PM. Sunday closed.
Consultation fee: Rs. 500 (general physician), Rs. 800 (cardiology).
Helpline: +91 98765 43210. Address: 12 FC Road, Pune 411004.
Dr. Mehta (Cardiology) sits on Tuesday and Friday."""

MODEL_FACTS = {"facts": [
    {"key": "OPD timings, Monday to Saturday", "value": "9:30 AM to 1:00 PM"},
    {"key": "Consultation fee, general physician", "value": "Rs. 500"},
    {"key": "Consultation fee, cardiology", "value": "Rs. 800"},
    {"key": "Helpline phone number", "value": "+91 98765 43210"},
    {"key": "Address", "value": "12 FC Road, Pune 411004"},
    {"key": "Dr. Mehta (Cardiology) days", "value": "Tuesday and Friday"},
    {"key": "MRI scan fee", "value": "Rs. 4500"},                         # not in the text: dropped
    {"key": "OPD timings", "value": "9:30 am - 1 pm"},                   # reformatted: dropped
    {"key": "Emergency ward (open 24 hours)", "value": "Sunday closed"},  # the key names a number the text lacks
    {"key": "Consultation fee, cardiology", "value": "Rs. 800"},          # duplicate
    "not a dict",
]}


@pytest.fixture()
def sheet(monkeypatch, tmp_path):
    monkeypatch.setenv("FACT_SHEET", "on")
    monkeypatch.setenv("FACT_SHEET_DIR", str(tmp_path / "sheets"))
    docs = {"fp": "fp-1", "texts": [DOC]}
    loads = []

    def loader(agent_id, user_id, storage_mode):
        loads.append(agent_id)
        return docs["fp"], (lambda: list(docs["texts"]))
    fs.set_loader(loader)
    fs.clear_memory()
    asked = []

    def chat(messages):
        asked.append(messages)
        return llm.ChatResult(json.dumps(MODEL_FACTS), "fake", "fake-model", 1, {})
    yield SimpleNamespace(docs=docs, asked=asked, chat=chat, loads=loads, dir=tmp_path / "sheets")
    fs.set_loader(None)
    fs.clear_memory()


def test_validation_keeps_only_verbatim_values():
    kept = fs.validate(MODEL_FACTS["facts"], DOC)
    assert [f["value"] for f in kept] == ["9:30 AM to 1:00 PM", "Rs. 500", "Rs. 800", "+91 98765 43210",
                                         "12 FC Road, Pune 411004", "Tuesday and Friday"]
    assert fs.validate("nope", DOC) == [] and fs.validate([{"key": "", "value": "Rs. 500"}], DOC) == []


def test_build_asks_for_json_and_copy_verbatim(sheet):
    facts, model = fs.build_facts([DOC], sheet.chat)
    assert len(facts) == 6 and model == "fake-model"
    system = sheet.asked[0][0]["content"]
    assert "JSON" in system and "VERBATIM" in system and DOC in sheet.asked[0][1]["content"]


def test_refresh_builds_once_then_reads_the_disk_cache(sheet):
    facts = fs.refresh("agent1", "u", "local", chat=sheet.chat)
    assert len(facts) == 6 and len(sheet.asked) == 1
    saved = json.loads((sheet.dir / "agent1.json").read_text(encoding="utf-8"))
    assert saved["fingerprint"] == "fp-1" and len(saved["facts"]) == 6
    fs.clear_memory()                                                    # a restart: memory gone, disk kept
    assert fs.refresh("agent1", "u", "local", chat=sheet.chat) == facts and len(sheet.asked) == 1
    sheet.docs.update(fp="fp-2", texts=[DOC.replace("Rs. 500", "Rs. 550")])  # a document changed: built again
    facts2 = fs.refresh("agent1", "u", "local", chat=sheet.chat)
    assert len(sheet.asked) == 2 and "Rs. 500" not in [f["value"] for f in facts2]


def test_no_documents_no_facts(sheet):
    sheet.docs.update(fp=None, texts=[])
    assert fs.refresh("agent1", "u", "local", chat=sheet.chat) == [] and sheet.asked == []


def test_offline_skips_the_build(sheet, monkeypatch):
    monkeypatch.setattr(fs, "hosted_llm_available", lambda: False)
    assert fs.refresh("agent1", "u", "local") == []
    assert not (sheet.dir / "agent1.json").exists()


def test_hosted_llm_available_reads_the_provider_chain(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "groq,mlx")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert fs.hosted_llm_available() is False
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert fs.hosted_llm_available() is True


def test_a_failed_build_is_not_retried_at_once(sheet):
    def broken(messages):
        raise llm.LLMUnavailable({"groq": "down"})
    calls = []

    def counting(messages):
        calls.append(1)
        return broken(messages)
    assert fs.refresh("agent1", "u", "local", chat=counting) == []
    assert fs.refresh("agent1", "u", "local", chat=counting) == [] and len(calls) == 1


def _entry(sheet):
    fs.refresh("agent1", "u", "local", chat=sheet.chat)
    return fs._mem[("agent1", "u", "local")]


@pytest.mark.parametrize("question,expect", [
    ("What is the consultation fee for cardiology?", "Consultation fee, cardiology: Rs. 800"),
    ("Fees kitni hai?", "Consultation fee, general physician: Rs. 500"),
    ("OPD kab khulti hai?", "OPD timings, Monday to Saturday: 9:30 AM to 1:00 PM"),
    ("ओपीडी का समय क्या है?", "OPD timings, Monday to Saturday: 9:30 AM to 1:00 PM"),
    ("Helpline number kya hai?", "Helpline phone number: +91 98765 43210"),
    ("Dr Mehta kab aate hain?", "Dr. Mehta (Cardiology) days: Tuesday and Friday"),
])
def test_relevant_facts_follow_the_question(sheet, question, expect):
    lines = fs.relevant(_entry(sheet), question, [])
    assert expect in lines and len(lines) <= fs.MAX_PROMPT_FACTS


def test_relevant_facts_also_follow_the_passages(sheet):
    lines = fs.relevant(_entry(sheet), "xyz", ["Address: 12 FC Road, Pune 411004."])
    assert lines == ["Address: 12 FC Road, Pune 411004"]


def test_turn_facts_never_blocks_and_fills_in_the_background(sheet, monkeypatch):
    monkeypatch.setattr(fs, "hosted_llm_available", lambda: True)
    monkeypatch.setattr(fs, "build_facts", lambda texts, chat=None: (fs.validate(MODEL_FACTS["facts"], DOC), "m"))
    run = SimpleNamespace(agent_id="agent1", user_id="u", storage_mode="local", user_text="Fees kitni hai?", chunks=[])
    started = time.perf_counter()
    assert fs.turn_facts(run) == []                                       # nothing yet: the turn does not wait
    assert (time.perf_counter() - started) * 1000 < 10
    deadline = time.time() + 5
    while not fs.turn_facts(run) and time.time() < deadline:
        time.sleep(0.02)
    assert "Consultation fee, general physician: Rs. 500" in fs.turn_facts(run)


def test_switched_off_means_no_facts_and_no_work(sheet, monkeypatch):
    monkeypatch.setenv("FACT_SHEET", "off")
    run = SimpleNamespace(agent_id="agent1", user_id="u", storage_mode="local", user_text="fee?", chunks=[])
    assert fs.turn_facts(run) == [] and sheet.loads == []


def test_cache_dir_is_git_ignored():
    from pathlib import Path
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "data/fact_sheets/" in ignore


def test_key_facts_reach_the_prompt(monkeypatch):
    import backend
    monkeypatch.setattr(fs, "turn_facts", lambda run: ["Consultation fee, general physician: Rs. 500"])
    run = SimpleNamespace(agent={"name": "Ria"}, language="en", language_decision=None, tool_specs=[], history=[],
                          user_text="Fees?", chunks=["passage"])
    msgs = backend._turn_messages(run)
    assert "Key facts (copied from the documents):\n- Consultation fee, general physician: Rs. 500" in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("Caller says: Fees?")
