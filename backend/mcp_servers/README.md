# Local MCP servers

Servers for the [Model Context Protocol](https://modelcontextprotocol.io) that BonClay agents use as tools. Each runs
over stdio, so any MCP client can use it too: Claude Desktop, the MCP Inspector, or your own code.

## appointments

A booking desk for the fictional Sunrise Family Clinic (the `clinic_receptionist` agent template). Its doctors and OPD
hours are the ones in `templates/agents/clinic_receptionist.md`; the clinic takes no appointments on Sundays.

| Tool | What it does |
| --- | --- |
| `list_free_slots(date, doctor?)` | Free 30-minute slots on a date (`YYYY-MM-DD`), for one doctor (name or specialty) or all |
| `book_appointment(name, phone, date, time, doctor?, reason?)` | Books a slot (`time` is `HH:MM`, 24-hour); without a doctor it is the General Physician. Returns the booking with its `booking_id` (for example `A12`) |
| `find_booking(phone)` | The upcoming bookings made with a phone number |
| `cancel_appointment(booking_id)` | Cancels an upcoming booking and frees the slot |

Inputs are validated: the date must be today or later (within 60 days), not a Sunday, and the time must be a free slot
inside the doctor's OPD hours; the phone number needs 10 digits (`+91` and spaces are fine). A request that cannot be done
comes back as a tool error with a message that can be said out loud ("The 10:30 slot on 2026-10-19 is already booked").

Data lives in one JSON file, `backend/data/mcp/appointments.json` (override with `BONCLAY_APPOINTMENTS_FILE`), created
on first use. Edit it to change doctors or hours; delete it to start over. Several clients can use it at once (a file lock
guards every change). Times are the machine's local time.

### Use it in BonClay

Open an agent in Agent Studio, tick **Appointments (local MCP server)** under Tools, and save. The clinic template has it
on already. BonClay starts the server when the first turn of such an agent needs it, keeps it running, and stops it when
the API exits. In the Playground a small chip under the reply shows what the agent did, for example
`Used tool: book_appointment -> booking A12 with Dr. Meera Kulkarni on Monday 2026-10-19 10:30`.

Cancelling asks first: the agent must say "Shall I cancel booking A12?" in one reply, and the tool only runs on the
caller's yes in the next one.

### Use it in Claude Desktop (or any MCP client)

Run it from the `backend` directory with the project's Python:

```bash
cd backend
.venv/bin/python -m mcp_servers.appointments
```

Claude Desktop (`claude_desktop_config.json`, use absolute paths):

```json
{
  "mcpServers": {
    "sunrise-appointments": {
      "command": "/path/to/BonClay/backend/.venv/bin/python",
      "args": ["/path/to/BonClay/backend/mcp_servers/appointments.py"]
    }
  }
}
```

To try the tools without a model: `npx @modelcontextprotocol/inspector backend/.venv/bin/python -m mcp_servers.appointments`
(from `backend/`).

## Adding another server

Write a stdio MCP server (any language), then register it in `backend/data/mcp/servers.json`
(or the file named by `BONCLAY_MCP_SERVERS`):

```json
{
  "orders": {"label": "Orders", "command": "python", "args": ["/path/to/orders_server.py"], "env": {"KEY": "value"}}
}
```

An agent can then list `"orders"` in its `tools` (`PUT /api/agents/{id}` with `{"tools": ["orders"]}`). The Studio
checkbox list is in `frontend/src/components/agents/AgentEditor.tsx` (`TOOL_OPTIONS`). A tool that changes or removes
something should be added to `CONFIRM_BEFORE` in `backend/services/tools.py` so it needs the same ask-first step as
`cancel_appointment`.

## How a turn uses tools

`services/tools.py` is the client. For an agent with tools the model gets the tool schemas (OpenAI-style `tools` on
Groq and OpenAI-compatible servers; a strict JSON protocol in the prompt on the local model), may make up to 3 tool calls,
and then answers. Tool results reach the model labelled as data, never as instructions. A booking id the model says that no
tool returned is replaced by a safe line. The calls are stored with the turn (`tool_calls`) and their time is reported as
`tools_ms`.

## bonclay_api

Manage a running BonClay from a coding tool: build an agent from a template, give it documents, test it, read its
conversations and receipts. Unlike the servers above it is an HTTP client over the BonClay API (it holds no data), signed
in with a personal API key (`aur_<id>_<secret>`; create one in the app or with `POST /api/keys`).

| Setting | Meaning |
| --- | --- |
| `BONCLAY_API` | Address of the API (default `http://localhost:8000`) |
| `BONCLAY_API_KEY` | Your `aur_...` key (required) |
| `BONCLAY_MCP_WRITE` | `1` to expose the tools that change things. Unset, they are not even listed |

Read-only (always): `list_templates`, `list_agents`, `list_documents(agent_id)`, `list_conversations(agent_id?, days?, limit?)`,
`get_receipt(receipt_id)`, `get_evaluation(job_id)`.
Write (`BONCLAY_MCP_WRITE=1`): `create_agent_from_template(template_id, org_name?, city?, details?, languages?, voice_id?)`,
`upload_document(agent_id, file_path)`, `rechunk_documents(agent_id)`, `start_evaluation(agent_id, max_cases?, categories?, use_tools?)`,
`delete_agent`, `delete_document`.

An evaluation is start-then-poll: `start_evaluation` returns a `job_id` straight away (use `max_cases` of 5 to 10 for a
quick run); call `get_evaluation(job_id)` every few seconds until the status is no longer `running`. Failures come back as
tool errors saying what to fix (API not running, key refused, file missing).

Claude Code:

```bash
claude mcp add bonclay -e BONCLAY_API=http://localhost:8000 -e BONCLAY_API_KEY=aur_... -e BONCLAY_MCP_WRITE=1 \
  -- /path/to/BonClay/backend/.venv/bin/python -m mcp_servers.bonclay_api
```

(run from `backend/`, or use `-m` with `cwd` set; or put the same under `mcpServers` in `.mcp.json`). Cursor
(`~/.cursor/mcp.json` or `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "bonclay": {
      "command": "/path/to/BonClay/backend/.venv/bin/python",
      "args": ["/path/to/BonClay/backend/mcp_servers/bonclay_api.py"],
      "env": {"BONCLAY_API": "http://localhost:8000", "BONCLAY_API_KEY": "aur_...", "BONCLAY_MCP_WRITE": "0"}
    }
  }
}
```

Leave `BONCLAY_MCP_WRITE` at `0` (or remove it) to keep the client read-only. A key grants the same access as its owner,
so keep it out of version control.
