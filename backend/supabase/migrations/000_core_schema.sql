-- 000_core_schema.sql: core tables, pgvector search function and storage bucket.
-- Run FIRST (000 -> 004). Everything the backend reads/writes (backend.py, core/, services/) is created here;
-- 001-004 then add RLS policies and ALTER on top. Fully idempotent: safe on a project that already has some tables.
--
-- Reconciled against the code, not the PRD SQL (BONCLAY_PRD_TDD.md 5.1):
--   * no app-level `users` table (code never uses it; every owner column points at auth.users)
--   * agents/documents/conversations carry user_id from the start (002 would add it anyway)
--   * generations has the columns persist_generation() actually writes (voice_name, text, duration_s, audio_url)
--   * evaluation_runs stores persona + three verdict flags + reasoning (no NOT NULL transcript)
--   * voices.consent_id is added by 004, consents are created by 004

create extension if not exists vector;
create extension if not exists pgcrypto;  -- gen_random_uuid() on older Postgres; built in from PG13

-- ---------------------------------------------------------------------------
-- projects / voices (same shape as 003; created here because agents reference voices)
-- ---------------------------------------------------------------------------
create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  owner_user_id uuid not null references auth.users (id) on delete cascade,
  name text not null default 'Default Project',
  created_at timestamptz not null default now()
);

create table if not exists public.voices (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects (id) on delete cascade,
  engine text,
  language text,
  reference_audio_path text,
  display_name text,
  transcript text,  -- read by the conversation turn path (row.get("transcript")) for F5 reference text
  created_at timestamptz not null default now()
);
alter table public.voices add column if not exists transcript text;

create index if not exists projects_owner_user_id_idx on public.projects (owner_user_id);
create index if not exists voices_project_id_idx on public.voices (project_id);

-- ---------------------------------------------------------------------------
-- agents / knowledge base / conversations
-- ---------------------------------------------------------------------------
create table if not exists public.agents (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  voice_id uuid references public.voices (id) on delete set null,
  name text not null,
  system_instructions text not null default '',
  greeting text not null default '',
  supported_languages text[] not null default array['en'],
  created_at timestamptz not null default now()
);

create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  agent_id uuid not null references public.agents (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  filename text not null,
  storage_path text,  -- PRD column; the backend extracts text and does not keep the PDF
  created_at timestamptz not null default now()
);

-- all-MiniLM-L6-v2 (sentence-transformers) produces 384-dimensional vectors.
create table if not exists public.document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  chunk_text text not null,
  embedding vector(384),
  chunk_index integer not null default 0,
  created_at timestamptz not null default now()
);

create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  agent_id uuid not null references public.agents (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  started_at timestamptz not null default now()
);

create table if not exists public.conversation_turns (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations (id) on delete cascade,
  role text not null check (role in ('user', 'agent')),
  text_content text not null,
  audio_storage_path text,
  latency_ms integer,
  created_at timestamptz not null default now()
);

create table if not exists public.evaluation_runs (
  id uuid primary key default gen_random_uuid(),
  agent_id uuid not null references public.agents (id) on delete cascade,
  persona text not null,
  task_completion boolean,
  hallucination_flag boolean,
  policy_compliance boolean,
  reasoning text,
  transcript jsonb,  -- PRD column; not written by the current backend
  created_at timestamptz not null default now()
);

create index if not exists agents_project_id_idx on public.agents (project_id);
create index if not exists agents_user_id_idx on public.agents (user_id);
create index if not exists documents_agent_id_idx on public.documents (agent_id);
create index if not exists document_chunks_document_id_idx on public.document_chunks (document_id);
create index if not exists conversations_agent_id_idx on public.conversations (agent_id);
create index if not exists conversation_turns_conversation_id_idx
  on public.conversation_turns (conversation_id, created_at);
create index if not exists evaluation_runs_agent_id_idx on public.evaluation_runs (agent_id);

-- HNSW rather than ivfflat: it needs no training data, so it is correct when built on an empty table.
create index if not exists document_chunks_embedding_idx
  on public.document_chunks using hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- generations / api_keys (001 re-creates them with `if not exists`, so these definitions win on a fresh project)
-- ---------------------------------------------------------------------------
create table if not exists public.generations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  voice_id text,  -- saved-voice uuid or a preset voice name; deliberately no FK so history survives voice deletion
  voice_name text,
  text text,
  engine text not null,
  language text,
  duration_s double precision,
  audio_url text,
  text_length integer,  -- legacy 001 columns, unused by the backend
  duration_ms integer,
  created_at timestamptz not null default now()
);
-- A project that ran the original 001 first has the old shape: bring it up to what the code writes.
alter table public.generations add column if not exists voice_name text;
alter table public.generations add column if not exists text text;
alter table public.generations add column if not exists duration_s double precision;
alter table public.generations add column if not exists audio_url text;

create table if not exists public.api_keys (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  name text not null,
  key_hash text not null,
  prefix text,
  last_used_at timestamptz,
  created_at timestamptz not null default now()
);
create index if not exists api_keys_prefix_idx on public.api_keys (prefix);  -- looked up on every API-key request

-- ---------------------------------------------------------------------------
-- RLS for evaluation_runs (001-004 do not cover it; the other new tables are covered by 001/002)
-- ---------------------------------------------------------------------------
alter table public.evaluation_runs enable row level security;
drop policy if exists evaluation_runs_select_own on public.evaluation_runs;
create policy evaluation_runs_select_own on public.evaluation_runs for select to authenticated
  using (exists (select 1 from public.agents a where a.id = agent_id and a.user_id = (select auth.uid())));

-- ---------------------------------------------------------------------------
-- RAG retrieval: called as supabase.rpc("match_document_chunks", {query_embedding, match_count, target_agent_id})
-- SECURITY INVOKER (default): RLS applies to non-service callers; the backend's service-role key bypasses it
-- and enforces ownership itself (_owned_agent) before calling.
-- ---------------------------------------------------------------------------
create or replace function public.match_document_chunks(
  query_embedding vector(384),
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
         (1 - (c.embedding <=> query_embedding))::double precision as similarity
  from public.document_chunks c
  join public.documents d on d.id = c.document_id
  where d.agent_id = target_agent_id
    and c.embedding is not null
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

-- ---------------------------------------------------------------------------
-- Storage bucket used by AUDIO_BUCKET in backend.py. Private (holds consent recordings and voice samples);
-- the backend uses the service-role key, which bypasses storage policies.
-- backend.py hands clients time-limited signed links (create_signed_url, SIGNED_URL_EXPIRES_S, default 1 hour),
-- so the bucket stays private.
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('audio-assets', 'audio-assets', false)
on conflict (id) do nothing;
