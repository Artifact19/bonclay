-- 004_consents.sql: spoken-consent records written by POST /api/voices/clone.
-- Idempotent. The backend writes with the service-role key; RLS lets users read only their own records.

create table if not exists public.consents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  challenge_text text not null,
  audio_storage_path text not null,
  permitted_project_id uuid references public.projects (id) on delete set null,
  expiry_date timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists consents_user_id_idx on public.consents (user_id);

alter table public.voices add column if not exists consent_id uuid references public.consents (id) on delete set null;

alter table public.consents enable row level security;
drop policy if exists consents_select_own on public.consents;
create policy consents_select_own on public.consents for select to authenticated
  using ((select auth.uid()) = user_id);
