-- 018_chunk_structure.sql: structure-aware passages and the question index (backend/services/chunking.py,
-- backend/services/question_index.py). Idempotent; nothing existing is changed or dropped.
--
--   kind             'passage' (default when null) or 'question'. A question row holds a question a caller might ask in
--                    `chunk_text`, with its own embedding; retrieval scores it like a passage and returns its parent passage.
--   parent_chunk_id  for a question row, the passage it belongs to (deleting the passage deletes its questions).
--   header           for a passage, "document title > section > FAQ question"; the text that is embedded and searched is
--                    header + newline + chunk_text.
--
-- All three are nullable, so rows written before this migration stay valid and the backend works without it (it then stores
-- passages as before and writes no question rows). Deleting a document already deletes all its rows (document_id cascades).

alter table public.document_chunks add column if not exists kind text;
alter table public.document_chunks add column if not exists parent_chunk_id uuid
  references public.document_chunks (id) on delete cascade;
alter table public.document_chunks add column if not exists header text;

do $$
begin
  if not exists (select 1 from pg_constraint where conname = 'document_chunks_kind_check') then
    alter table public.document_chunks
      add constraint document_chunks_kind_check check (kind is null or kind in ('passage', 'question'));
  end if;
end $$;

create index if not exists document_chunks_parent_idx on public.document_chunks (parent_chunk_id)
  where parent_chunk_id is not null;
