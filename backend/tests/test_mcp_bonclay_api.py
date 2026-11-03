"""The BonClay API MCP server (mcp_servers/bonclay_api.py) against a fake API (no model loading): auth header, read
tools, write tools gated by BONCLAY_MCP_WRITE, multipart upload, start-then-poll evaluation, and error messages."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import pytest

from mcp_servers import bonclay_api as api

KEY = "aur_abc123_secret"
SEEN = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        u = urlparse(self.path)
        SEEN.append((self.command, u.path, u.query, raw, self.headers.get("Authorization"),
                     self.headers.get("Content-Type", "")))
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            return self._send(401, {"message": "bad key"})
        p, m = u.path, self.command
        if (m, p) == ("GET", "/api/agent-templates"):
            return self._send(200, {"templates": [{"id": "clinic_receptionist", "name": "Clinic"}]})
        if (m, p) == ("GET", "/api/agents"):
            return self._send(200, [{"id": "a1", "name": "Bot", "system_prompt": "SECRET PROMPT"}])
        if (m, p) == ("POST", "/api/agents/from-template"):
            return self._send(200, {"agent": {"id": "a2"}})
        if (m, p) == ("POST", "/api/knowledge/upload"):
            return self._send(200, {"document_id": "d1", "chunks_created": 3})
        if (m, p) == ("POST", "/api/agents/a1/knowledge/rechunk"):
            return self._send(200, {"chunks_before": 3, "chunks_after": 9})
        if (m, p) == ("POST", "/api/evaluations/extended"):
            return self._send(200, {"job_id": "j1", "total": 5, "poll": "/api/evaluations/extended/j1"})
        if (m, p) == ("GET", "/api/evaluations/extended/j1"):
            return self._send(200, {"status": "running", "progress": {"done": 2, "total": 5}})
        if (m, p) == ("GET", "/api/calls"):
            return self._send(200, {"calls": [], "query": u.query})
        if (m, p) == ("GET", "/api/receipts/r1"):
            return self._send(200, {"id": "r1", "signature": "x"})
        if (m, p) == ("DELETE", "/api/agents/a1"):
            return self._send(200, {"ok": True})
        self._send(404, {"message": "no such thing"})

    do_GET = do_POST = do_DELETE = _handle


@pytest.fixture(autouse=True)
def fake_api(monkeypatch):
    SEEN.clear()
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("BONCLAY_API", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("BONCLAY_API_KEY", KEY)
    monkeypatch.delenv("BONCLAY_MCP_WRITE", raising=False)
    yield srv
    srv.shutdown()


def err(fn, *a, **kw):
    with pytest.raises(api.BonClayApiError) as e:
        fn(*a, **kw)
    return str(e.value)


def tool_names(server):
    return {t.name for t in asyncio.run(server.list_tools())}


def test_read_tools_send_the_bearer_key():
    assert api.list_templates()["templates"][0]["id"] == "clinic_receptionist"
    assert api.get_receipt("r1")["id"] == "r1"
    assert all(s[4] == f"Bearer {KEY}" for s in SEEN)


def test_agents_are_summarised_without_the_prompt():
    out = api.list_agents()
    assert out == {"agents": [{"id": "a1", "name": "Bot"}]}


def test_conversations_pass_filters():
    api.list_conversations(agent_id="a1", days=7, limit=5)
    assert SEEN[-1][2] in ("agent_id=a1&days=7&limit=5", "days=7&limit=5&agent_id=a1") or "agent_id=a1" in SEEN[-1][2]
    assert "days=7" in SEEN[-1][2]


def test_write_tools_are_hidden_unless_enabled(monkeypatch):
    assert "create_agent_from_template" not in tool_names(api.build_server())
    assert {"list_agents", "get_evaluation"} <= tool_names(api.build_server())
    monkeypatch.setenv("BONCLAY_MCP_WRITE", "1")
    names = tool_names(api.build_server())
    assert {"create_agent_from_template", "upload_document", "rechunk_documents", "start_evaluation",
            "delete_agent"} <= names


def test_create_from_template_omits_unset_fields():
    api.create_agent_from_template("clinic_receptionist", org_name="Acme")
    body = json.loads(SEEN[-1][3])
    assert body == {"template_id": "clinic_receptionist", "org_name": "Acme"}


def test_upload_is_multipart(tmp_path):
    f = tmp_path / "faq.txt"
    f.write_text("Opening hours are 9 to 5.")
    assert api.upload_document("a1", str(f))["chunks_created"] == 3
    m, p, _, raw, _, ctype = SEEN[-1]
    assert ctype.startswith("multipart/form-data") and b'name="agent_id"' in raw and b'filename="faq.txt"' in raw
    assert b"Opening hours" in raw


def test_upload_rejects_missing_and_empty_files(tmp_path):
    assert "No file" in err(api.upload_document, "a1", str(tmp_path / "nope.txt"))
    (tmp_path / "e.txt").write_text("")
    assert "between 1 byte" in err(api.upload_document, "a1", str(tmp_path / "e.txt"))
    assert not SEEN


def test_rechunk_and_delete():
    assert api.rechunk_documents("a1")["chunks_after"] == 9
    assert api.delete_agent("a1") == {"ok": True}


def test_evaluation_is_start_then_poll():
    job = api.start_evaluation("a1", max_cases=5)
    assert job["job_id"] == "j1"
    assert json.loads(SEEN[-1][3]) == {"agent_id": "a1", "max_cases": 5, "use_tools": False}
    assert api.get_evaluation("j1")["status"] == "running"


def test_errors_are_readable(monkeypatch):
    monkeypatch.setenv("BONCLAY_API_KEY", "aur_wrong_key")
    msg = err(api.list_agents)
    assert "401" in msg and "BONCLAY_API_KEY" in msg
    monkeypatch.setenv("BONCLAY_API_KEY", KEY)
    assert "404" in err(api.get_receipt, "missing")
    monkeypatch.delenv("BONCLAY_API_KEY")
    assert "Set BONCLAY_API_KEY" in err(api.list_agents)
    monkeypatch.setenv("BONCLAY_API_KEY", KEY)
    monkeypatch.setenv("BONCLAY_API", "http://127.0.0.1:1")
    assert "Cannot reach" in err(api.list_agents)
