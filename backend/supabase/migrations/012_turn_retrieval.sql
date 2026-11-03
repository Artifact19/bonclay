-- 012_turn_retrieval.sql: what retrieval found for each agent turn, {n_chunks, top_score, top_dense, doc_ids, model, ...}.
-- Read by the knowledge-gap report (no matching document / weak match). Idempotent; older turns keep null.

alter table public.conversation_turns add column if not exists retrieval jsonb;
