-- 020_match_question_rows.sql: the two match functions also return `kind`, `parent_chunk_id` and `header` (columns added by
-- 018_chunk_structure.sql, which must run first), and the parent passage of every question row they return.
--
-- Why: the question index (018) stores caller-style questions as rows of document_chunks with their own vector. The
-- backend scores a question row like a passage but hands the model its PARENT passage (best score wins) and never the
-- question itself, so it needs to know which rows are questions and which passage they belong to, and the parent must be
-- in the result even when its own vector is not among the closest. The header ("document > section") is matched by BM25
-- and shown to the model with the passage.
--
-- Contract (both functions): the `match_count` closest rows (passages and questions), plus the parents of the questions
-- among them that are not already there (so up to 2 x match_count rows). `kind` is 'passage' when null (rows written
-- before 018). The backend tolerates the old functions (no such columns: every row is a passage), so it can be deployed
-- before or after this file. A function's return type cannot be changed in place, hence drop + create. Idempotent.

drop function if exists public.match_document_chunks(vector, integer, uuid);
create function public.match_document_chunks(
  query_embedding vector(384),
  match_count integer,
  target_agent_id uuid
)
returns table (
  id uuid,
  document_id uuid,
  chunk_text text,
  chunk_index integer,
  similarity double precision,
  kind text,
  parent_chunk_id uuid,
  header text
)
language sql
stable
as $$
  with scored as (
    select c.id, c.document_id, c.chunk_text, c.chunk_index,
           (1 - (c.embedding <=> query_embedding))::double precision as similarity,
           coalesce(c.kind, 'passage') as kind, c.parent_chunk_id, c.header
    from public.document_chunks c
    join public.documents d on d.id = c.document_id
    where d.agent_id = target_agent_id
      and c.embedding is not null
    order by c.embedding <=> query_embedding
    limit match_count
  )
  select * from scored
  union all
  select p.id, p.document_id, p.chunk_text, p.chunk_index,
         (1 - (p.embedding <=> query_embedding))::double precision,
         coalesce(p.kind, 'passage'), p.parent_chunk_id, p.header
  from public.document_chunks p
  where p.id in (select s.parent_chunk_id from scored s where s.parent_chunk_id is not null)
    and p.id not in (select s.id from scored s);
$$;

drop function if exists public.match_document_chunks_1024(vector, integer, uuid);
create function public.match_document_chunks_1024(
  query_embedding vector(1024),
  match_count integer,
  target_agent_id uuid
)
returns table (
  id uuid,
  document_id uuid,
  chunk_text text,
  chunk_index integer,
  similarity double precision,
  kind text,
  parent_chunk_id uuid,
  header text
)
language sql
stable
as $$
  with scored as (
    select c.id, c.document_id, c.chunk_text, c.chunk_index,
           (1 - (c.embedding_1024 <=> query_embedding))::double precision as similarity,
           coalesce(c.kind, 'passage') as kind, c.parent_chunk_id, c.header
    from public.document_chunks c
    join public.documents d on d.id = c.document_id
    where d.agent_id = target_agent_id
      and c.embedding_1024 is not null
    order by c.embedding_1024 <=> query_embedding
    limit match_count
  )
  select * from scored
  union all
  select p.id, p.document_id, p.chunk_text, p.chunk_index,
         (1 - (p.embedding_1024 <=> query_embedding))::double precision,
         coalesce(p.kind, 'passage'), p.parent_chunk_id, p.header
  from public.document_chunks p
  where p.id in (select s.parent_chunk_id from scored s where s.parent_chunk_id is not null)
    and p.id not in (select s.id from scored s);
$$;
