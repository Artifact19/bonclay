-- 003_projects_voices_rls.sql: owner-only RLS for projects and voices.
-- The app stores voices under a project (voices.project_id -> projects.id, projects.owner_user_id),
-- so voice access is scoped through project ownership rather than a user_id column.
-- The frontend reads projects/voices directly with the anon key, so these policies are what
-- keeps one user's rows invisible to another. The backend's service-role key bypasses RLS.

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
  created_at timestamptz not null default now()
);

create index if not exists projects_owner_user_id_idx on public.projects (owner_user_id);
create index if not exists voices_project_id_idx on public.voices (project_id);

alter table public.projects enable row level security;
alter table public.voices enable row level security;

drop policy if exists "projects_select_own" on public.projects;
create policy "projects_select_own" on public.projects for select to authenticated
  using ((select auth.uid()) = owner_user_id);
drop policy if exists "projects_insert_own" on public.projects;
create policy "projects_insert_own" on public.projects for insert to authenticated
  with check ((select auth.uid()) = owner_user_id);
drop policy if exists "projects_update_own" on public.projects;
create policy "projects_update_own" on public.projects for update to authenticated
  using ((select auth.uid()) = owner_user_id) with check ((select auth.uid()) = owner_user_id);
drop policy if exists "projects_delete_own" on public.projects;
create policy "projects_delete_own" on public.projects for delete to authenticated
  using ((select auth.uid()) = owner_user_id);

-- Remove any user_id-based voice policies an earlier draft of 001 may have created.
drop policy if exists "voices_select_own" on public.voices;
drop policy if exists "voices_insert_own" on public.voices;
drop policy if exists "voices_update_own" on public.voices;
drop policy if exists "voices_delete_own" on public.voices;

create policy "voices_select_own" on public.voices for select to authenticated
  using (project_id in (select id from public.projects where owner_user_id = (select auth.uid())));
create policy "voices_insert_own" on public.voices for insert to authenticated
  with check (project_id in (select id from public.projects where owner_user_id = (select auth.uid())));
create policy "voices_update_own" on public.voices for update to authenticated
  using (project_id in (select id from public.projects where owner_user_id = (select auth.uid())))
  with check (project_id in (select id from public.projects where owner_user_id = (select auth.uid())));
create policy "voices_delete_own" on public.voices for delete to authenticated
  using (project_id in (select id from public.projects where owner_user_id = (select auth.uid())));
