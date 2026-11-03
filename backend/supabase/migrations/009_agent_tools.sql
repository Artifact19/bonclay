-- 009_agent_tools.sql: MCP tool servers an agent may use (see services/tools.py), and the tool calls of a turn.
-- Idempotent. `tools` holds server names such as 'appointments'; existing agents have none. `tool_calls` records what
-- the tools did on an agent turn: [{"name", "arguments", "ok", "result"}].

alter table public.agents add column if not exists tools text[] not null default '{}';
alter table public.conversation_turns add column if not exists tool_calls jsonb;
