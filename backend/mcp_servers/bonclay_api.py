"""An MCP server that manages a running BonClay API from a coding tool (Claude Code, Cursor, ...).

It is an HTTP client: it holds no data of its own and calls the BonClay API with a personal API key
(`aur_<id>_<secret>`, created in the app or with POST /api/keys):

    BONCLAY_API=http://localhost:8000 BONCLAY_API_KEY=aur_... .venv/bin/python -m mcp_servers.bonclay_api

Read-only tools are always available. Tools that change things (create an agent, upload a document, rechunk, delete,
start an evaluation) exist only when BONCLAY_MCP_WRITE=1, so a client cannot even see them otherwise. Long jobs are
start-then-poll: start_evaluation returns a job id at once and get_evaluation reports progress.

Every failure raises BonClayApiError with a message the caller can act on (MCP reports it as a tool error)."""
import mimetypes
import os
import sys
from typing import Annotated, List, Optional

import httpx
from pydantic import Field

try:  # the official MCP SDK: v2 calls the server class MCPServer, v1 called it FastMCP
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # pragma: no cover - depends on the installed SDK major version
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError

DEFAULT_API = "http://localhost:8000"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
TIMEOUT = 60.0
UPLOAD_TIMEOUT = 300.0  # embedding a large document can take a while


class BonClayApiError(ToolError):
    """The API call failed (not configured, unreachable, refused, or a bad request)."""


def write_enabled() -> bool:
    return os.environ.get("BONCLAY_MCP_WRITE", "").strip().lower() in ("1", "true", "yes")


def _base() -> str:
    return (os.environ.get("BONCLAY_API") or DEFAULT_API).rstrip("/")


def _request(method: str, path: str, *, timeout: float = TIMEOUT, **kw):
    key = (os.environ.get("BONCLAY_API_KEY") or "").strip()
    if not key.startswith("aur_"):
        raise BonClayApiError("Set BONCLAY_API_KEY to an BonClay API key (it starts with aur_; create one in the app).")
    try:
        r = httpx.request(method, _base() + path, headers={"Authorization": f"Bearer {key}"}, timeout=timeout, **kw)
    except httpx.HTTPError as e:
        raise BonClayApiError(f"Cannot reach the BonClay API at {_base()} ({type(e).__name__}). Is it running? "
                              "Set BONCLAY_API to its address.") from e
    if r.status_code >= 400:
        try:
            body = r.json()
            detail = body.get("message") or body.get("detail") or body.get("error") or body
        except ValueError:
            detail = r.text[:300]
        hint = " The API key was refused: check BONCLAY_API_KEY." if r.status_code in (401, 403) else ""
        raise BonClayApiError(f"BonClay API {method} {path} failed with {r.status_code}: {detail}.{hint}")
    try:
        return r.json()
    except ValueError:
        return {}


def _agent_summary(a: dict) -> dict:
    keys = ("id", "name", "template_id", "language", "voice_id", "created_at")
    return {k: a[k] for k in keys if k in a}


# ---------------------------------------------------------------------------------------------- read tools

def list_templates() -> dict:
    """The agent templates that can be turned into an agent (id, name, description)."""
    return _request("GET", "/api/agent-templates")


def list_agents() -> dict:
    """The agents of this account."""
    data = _request("GET", "/api/agents")
    rows = data if isinstance(data, list) else data.get("agents", [])
    return {"agents": [_agent_summary(a) for a in rows]}


def list_documents(agent_id: Annotated[str, Field(description="Agent id")]) -> dict:
    """The knowledge documents of an agent."""
    return _request("GET", f"/api/agents/{agent_id}/documents")


def get_evaluation(job_id: Annotated[str, Field(description="Job id from start_evaluation")]) -> dict:
    """Progress of an evaluation started with start_evaluation: status (running/done/...), progress counts and, when
    done, the result. Poll every few seconds until the status is no longer running."""
    return _request("GET", f"/api/evaluations/extended/{job_id}")


def list_conversations(agent_id: Annotated[Optional[str], Field(description="Only this agent")] = None,
                       days: Annotated[int, Field(ge=1, le=365)] = 30,
                       limit: Annotated[int, Field(ge=1, le=200)] = 20) -> dict:
    """Recent conversations (calls and chats), newest first."""
    params = {"days": days, "limit": limit}
    if agent_id:
        params["agent_id"] = agent_id
    return _request("GET", "/api/calls", params=params)


def get_receipt(receipt_id: Annotated[str, Field(description="Receipt id, from a conversation or generated audio")]) -> dict:
    """A signed provenance receipt (what produced a piece of audio, with consent)."""
    return _request("GET", f"/api/receipts/{receipt_id}")


# --------------------------------------------------------------------------------------------- write tools

def create_agent_from_template(
        template_id: Annotated[str, Field(description="Id from list_templates")],
        org_name: Annotated[Optional[str], Field(description="Replaces the sample organisation everywhere")] = None,
        city: Optional[str] = None,
        details: Annotated[Optional[str], Field(description="Opening hours or what callers usually ask")] = None,
        languages: Annotated[Optional[List[str]], Field(description="ISO 639-1 codes, e.g. ['hi','en']")] = None,
        voice_id: Optional[str] = None) -> dict:
    """Create an agent (with its sample knowledge document) from a template."""
    body = {"template_id": template_id, "org_name": org_name, "city": city, "details": details,
            "languages": languages, "voice_id": voice_id}
    return _request("POST", "/api/agents/from-template", json={k: v for k, v in body.items() if v is not None})


def upload_document(agent_id: Annotated[str, Field(description="Agent id")],
                    file_path: Annotated[str, Field(description="Path of a file on this machine (txt, md, pdf, ...)")]) -> dict:
    """Add a document to an agent's knowledge base. The API reads, splits and embeds it before answering."""
    path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.isfile(path):
        raise BonClayApiError(f"No file at {path}.")
    size = os.path.getsize(path)
    if size == 0 or size > MAX_UPLOAD_BYTES:
        raise BonClayApiError(f"The file must be between 1 byte and {MAX_UPLOAD_BYTES // (1024 * 1024)} MB (it is {size} bytes).")
    name = os.path.basename(path)
    with open(path, "rb") as f:
        files = {"file": (name, f, mimetypes.guess_type(name)[0] or "application/octet-stream")}
        return _request("POST", "/api/knowledge/upload", timeout=UPLOAD_TIMEOUT, data={"agent_id": agent_id}, files=files)


def rechunk_documents(agent_id: Annotated[str, Field(description="Agent id")]) -> dict:
    """Split an agent's existing documents into smaller chunks (returns chunks_before and chunks_after)."""
    return _request("POST", f"/api/agents/{agent_id}/knowledge/rechunk", timeout=UPLOAD_TIMEOUT)


def start_evaluation(agent_id: Annotated[str, Field(description="Agent id")],
                     max_cases: Annotated[Optional[int], Field(ge=1, le=500, description="Small (5-10) for a quick run")] = 10,
                     categories: Annotated[Optional[List[str]], Field(description="Only these attack categories")] = None,
                     use_tools: bool = False) -> dict:
    """Start the Hindi/Hinglish attack evaluation in the background. Returns a job_id at once; poll get_evaluation."""
    body = {"agent_id": agent_id, "max_cases": max_cases, "use_tools": use_tools, "categories": categories}
    return _request("POST", "/api/evaluations/extended", json={k: v for k, v in body.items() if v is not None})


def delete_agent(agent_id: Annotated[str, Field(description="Agent id")]) -> dict:
    """Delete an agent and its documents. This cannot be undone."""
    return _request("DELETE", f"/api/agents/{agent_id}")


def delete_document(agent_id: Annotated[str, Field(description="Agent id")],
                    document_id: Annotated[str, Field(description="Document id from list_documents")]) -> dict:
    """Delete one knowledge document."""
    return _request("DELETE", f"/api/agents/{agent_id}/documents/{document_id}")


READ_TOOLS = [list_templates, list_agents, list_documents, get_evaluation, list_conversations, get_receipt]
WRITE_TOOLS = [create_agent_from_template, upload_document, rechunk_documents, start_evaluation, delete_agent,
               delete_document]


def build_server(write: Optional[bool] = None):
    """The MCP server; write tools are registered only when `write` (default: BONCLAY_MCP_WRITE=1)."""
    server = _Server("BonClay API")
    for fn in READ_TOOLS + (WRITE_TOOLS if (write_enabled() if write is None else write) else []):
        server.tool()(fn)
    return server


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from services.parent_watch import watch_parent
        watch_parent()
    except ImportError:  # pragma: no cover
        pass
    build_server().run()


if __name__ == "__main__":
    main()
