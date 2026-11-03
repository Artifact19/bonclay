-- 007_agent_disclose_ai.sql: agents say they are an AI at the start of each conversation (IT Rules on synthetic media).
-- Idempotent. Existing agents default to true.

alter table public.agents add column if not exists disclose_ai boolean not null default true;
