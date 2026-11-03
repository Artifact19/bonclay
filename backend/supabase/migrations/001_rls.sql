-- 001_rls.sql: owner-only RLS for generations, embeddings, api_keys.
-- projects/voices are scoped through project ownership in 003_projects_voices_rls.sql.
-- Idempotent where possible. Existing tables are altered to ensure user_id + RLS.

create extension if not exists vector;

create table if not exists public.generations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  engine text not null,
  language text,
  text_length integer,
  duration_ms integer,
  voice_id uuid,
  created_at timestamptz not null default now()
);

create table if not exists public.embeddings (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  document_id uuid,
  chunk_index integer,
  content text not null,
  embedding vector(384),
  created_at timestamptz not null default now()
);

create table if not exists public.api_keys (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  name text not null,
  key_hash text not null,
  prefix text,
  last_used_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists generations_user_id_idx on public.generations (user_id);
create index if not exists generations_created_at_idx on public.generations (created_at);
create index if not exists embeddings_user_id_idx on public.embeddings (user_id);
create index if not exists api_keys_user_id_idx on public.api_keys (user_id);

do $$
declare
  t text;
begin
  foreach t in array array['generations', 'embeddings', 'api_keys'] loop
    execute format('alter table public.%I enable row level security', t);

    execute format('drop policy if exists "%1$s_select_own" on public.%1$I', t);
    execute format('create policy "%1$s_select_own" on public.%1$I for select to authenticated using ((select auth.uid()) = user_id)', t);

    execute format('drop policy if exists "%1$s_insert_own" on public.%1$I', t);
    execute format('create policy "%1$s_insert_own" on public.%1$I for insert to authenticated with check ((select auth.uid()) = user_id)', t);

    execute format('drop policy if exists "%1$s_update_own" on public.%1$I', t);
    execute format('create policy "%1$s_update_own" on public.%1$I for update to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id)', t);

    execute format('drop policy if exists "%1$s_delete_own" on public.%1$I', t);
    execute format('create policy "%1$s_delete_own" on public.%1$I for delete to authenticated using ((select auth.uid()) = user_id)', t);
  end loop;
end $$;
