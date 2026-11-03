-- 011_embeddings_1024.sql: 1024-dimensional embeddings (BAAI/bge-m3, the default EMBEDDING_MODEL) next to the existing
-- 384-dimensional ones (paraphrase-multilingual-MiniLM-L12-v2). Idempotent; nothing existing is changed or dropped.
--
-- A pgvector column has a fixed dimension, and Postgres cannot overload a function on it, so each dimension gets its
-- own column and its own match function. The backend picks them from the length of the query vector
-- (backend.py CLOUD_VECTOR_TARGETS): 384 -> `embedding` + match_document_chunks, 1024 -> `embedding_1024` +
-- match_document_chunks_1024. Vectors of different sizes are therefore never compared.
--
-- Existing chunks keep their 384-d vector and have no 1024-d one: the backend embeds them on the first search of
-- each agent after the switch (once per process), so no manual backfill is needed. Roll back by setting
-- EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (the 384-d column is untouched).

alter table public.document_chunks add column if not exists embedding_1024 vector(1024);
-- which model produced the vector in `embedding_1024` (the legacy `embedding` column does not record it)
alter table public.document_chunks add column if not exists embed_model text;

-- HNSW, like the 384-d index (000): no training data needed, correct on an empty table.
create index if not exists document_chunks_embedding_1024_idx
  on public.document_chunks using hnsw (embedding_1024 vector_cosine_ops);

-- Same contract as match_document_chunks, for 1024-d query vectors. SECURITY INVOKER (default): RLS applies to
-- non-service callers; the backend's service-role key bypasses it and checks ownership itself before calling.
create or replace function public.match_document_chunks_1024(
  query_embedding vector(1024),
  match_count integer,
  target_agent_id uuid
)
returns table (
  id uuid,
  document_id uuid,
  chunk_text text,
  chunk_index integer,
  similarity double precision
)
language sql
stable
as $$
  select c.id, c.document_id, c.chunk_text, c.chunk_index,
         (1 - (c.embedding_1024 <=> query_embedding))::double precision as similarity
  from public.document_chunks c
  join public.documents d on d.id = c.document_id
  where d.agent_id = target_agent_id
    and c.embedding_1024 is not null
  order by c.embedding_1024 <=> query_embedding
  limit match_count;
$$;
