-- 010_conversation_is_test.sql: conversations started by the owner's simulated "Test call" (Agent Studio) are flagged so
-- Analytics and the knowledge-gap report can leave them out. Idempotent; existing conversations are real calls.

alter table public.conversations add column if not exists is_test boolean not null default false;
