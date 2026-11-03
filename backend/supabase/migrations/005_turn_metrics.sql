-- 005_turn_metrics.sql: per-turn performance written by POST /api/conversations/{agent_id}/turn and read by
-- GET /api/perf/turns. Idempotent; all columns are nullable so existing rows and older backends keep working.

alter table public.conversation_turns add column if not exists timings jsonb;      -- {asr_ms, retrieval_ms, llm_ms, tts_ms}
alter table public.conversation_turns add column if not exists total_ms integer;   -- server-side, speech in to speech out
alter table public.conversation_turns add column if not exists engine text;
alter table public.conversation_turns add column if not exists llm_provider text;
alter table public.conversation_turns add column if not exists llm_model text;
alter table public.conversation_turns add column if not exists fallback boolean;
alter table public.conversation_turns add column if not exists language text;

create index if not exists conversation_turns_created_at_idx on public.conversation_turns (created_at desc);
