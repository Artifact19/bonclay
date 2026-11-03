-- 014_conversation_reply_language.sql: the language a caller explicitly asked the agent to speak ("Tamil mein bolo",
-- "please speak English") is kept on the conversation so the agent stays in it for the rest of the call. Null means
-- the caller has not asked: the agent mirrors the language of each turn. Idempotent.

alter table public.conversations add column if not exists reply_language text;
