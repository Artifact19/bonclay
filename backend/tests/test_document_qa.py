"""The Document Q&A template: a sample PDF to download, example questions for the agent's page, and an agent that
answers only from the document."""
import io

import pypdf

from services import llm
from tests.test_offline_agents import LOCAL, upload_txt
from tests.test_templates import env  # noqa: F401  (fixture: fake embeddings, no language model)


def direct_agent(env):
    return env.local_db.insert("agents", {"user_id": "local-user", "project_id": "local_project", "name": "Q",
                                          "system_instructions": "x", "greeting": "", "supported_languages": ["en"]})


def make_qa(client):
    r = client.post("/api/agents/from-template", headers=LOCAL, json={"template_id": "document_qa"})
    assert r.status_code == 200, r.text
    return r.json()


def test_the_template_is_listed_with_a_sample_pdf_and_questions(app_client, env):
    [t] = [x for x in app_client.get("/api/agent-templates", headers=LOCAL).json()["templates"]
           if x["id"] == "document_qa"]
    assert t["sample_pdf"] == "/api/agent-templates/document_qa/sample.pdf" and len(t["suggested_questions"]) >= 5
    pdf = app_client.get(t["sample_pdf"], headers=LOCAL)
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    text = " ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf.content)).pages)
    assert "Student membership" in text and "late fee" in text.lower()
    other = app_client.get("/api/agent-templates/clinic_receptionist/sample.pdf", headers=LOCAL)
    assert other.status_code == 404


def test_the_agent_is_told_to_answer_only_from_the_document(app_client, env):
    a = make_qa(app_client)
    assert a["tools"] == [] and a["document_count"] == 1
    ins = a["system_instructions"].lower()
    assert "only from the document" in ins and "cannot find it in the document" in ins


def test_a_fresh_template_agent_gets_the_template_questions(app_client, env):
    a = make_qa(app_client)
    r = app_client.get(f"/api/agents/{a['id']}/example-questions", headers=LOCAL).json()
    assert r["source"] == "template" and "How much is a student membership?" in r["questions"]


def test_questions_for_an_uploaded_pdf_come_from_the_model(app_client, env, monkeypatch):
    a = direct_agent(env)
    upload_txt(app_client, a["id"], "## Refunds\nThe refund window is thirty days from purchase.\n" * 3)

    def chat(messages, **kw):
        assert "refund window" in messages[0]["content"]  # the model reads the document's text
        return llm.ChatResult('["How long is the refund window?", "Is there a refund for a late return?", '
                              '"Not a question", "Who pays for shipping?"]', "fake", "fake-model", 1, {})

    monkeypatch.setattr(llm, "chat", chat)
    r = app_client.get(f"/api/agents/{a['id']}/example-questions", headers=LOCAL).json()
    assert r["source"] == "model"
    assert r["questions"] == ["How long is the refund window?", "Is there a refund for a late return?",
                              "Who pays for shipping?"]
    assert app_client.get(f"/api/agents/{a['id']}/example-questions", headers=LOCAL).json()["source"] == "cache"


def test_headings_are_the_fallback_when_no_model_answers(app_client, env, monkeypatch):
    a = direct_agent(env)
    upload_txt(app_client, a["id"], "## Late fees\nTwo rupees a day.\n\n## Opening hours\nNine to seven.\n" * 2)

    def broken(*_a, **_k):
        raise RuntimeError("no model")

    monkeypatch.setattr(llm, "chat", broken)
    r = app_client.get(f"/api/agents/{a['id']}/example-questions", headers=LOCAL).json()
    assert r["source"] == "headings"
    assert "What does the document say about late fees?" in r["questions"]


def test_an_agent_with_no_documents_has_no_questions(app_client, env):
    a = direct_agent(env)
    assert app_client.get(f"/api/agents/{a['id']}/example-questions", headers=LOCAL).json()["questions"] == []
