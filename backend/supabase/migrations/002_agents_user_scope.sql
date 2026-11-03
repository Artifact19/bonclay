-- 002_agents_user_scope.sql: per-user ownership for agents, conversations and knowledge documents.
-- Idempotent. Existing rows without an owner become invisible to everyone but the service role
-- (backfill them manually if needed).

do $$
declare
  t text;
begin
  foreach t in array array['agents', 'conversations', 'documents'] loop
    if to_regclass('public.' || t) is not null then
      execute format('alter table public.%I add column if not exists user_id uuid references auth.users (id) on delete cascade', t);
      execute format('create index if not exists %I on public.%I (user_id)', t || '_user_id_idx', t);
      execute format('alter table public.%I enable row level security', t);
      execute format('drop policy if exists %I on public.%I', t || '_owner_all', t);
      execute format(
        'create policy %I on public.%I for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid())',
        t || '_owner_all', t);
    end if;
  end loop;
end $$;

-- Child tables inherit ownership through their parent.
do $$
begin
  if to_regclass('public.document_chunks') is not null and to_regclass('public.documents') is not null then
    alter table public.document_chunks enable row level security;
    drop policy if exists document_chunks_owner_all on public.document_chunks;
    create policy document_chunks_owner_all on public.document_chunks for all to authenticated
      using (exists (select 1 from public.documents d where d.id = document_id and d.user_id = auth.uid()))
      with check (exists (select 1 from public.documents d where d.id = document_id and d.user_id = auth.uid()));
  end if;
  if to_regclass('public.conversation_turns') is not null and to_regclass('public.conversations') is not null then
    alter table public.conversation_turns enable row level security;
    drop policy if exists conversation_turns_owner_all on public.conversation_turns;
    create policy conversation_turns_owner_all on public.conversation_turns for all to authenticated
      using (exists (select 1 from public.conversations c where c.id = conversation_id and c.user_id = auth.uid()))
      with check (exists (select 1 from public.conversations c where c.id = conversation_id and c.user_id = auth.uid()));
  end if;
end $$;
