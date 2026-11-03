-- 019_answer_audit.sql: the post-call answer audit (services/answer_audit.py, routers/answer_audit.py). One row per audited
-- agent turn: whether the reply was supported by the passages the agent was given (HHEM score 0..1), the weakest sentence,
-- the passage it was compared with, and the owner's verdict ('fine' / 'wrong'). New table only: no existing table is touched.
-- The backend uses the service-role key and always filters by user_id; the policy below keeps direct client access
-- owner-only too. `id` is the audited turn's id. Idempotent.

create table if not exists public.answer_audit (
  id uuid primary key,
  turn_id uuid not null,
  user_id uuid not null references auth.users (id) on delete cascade,
  agent_id uuid references public.agents(id) on delete cascade,
  conversation_id uuid,
  channel text,
  status text not null default 'checked' check (status in ('checked', 'skipped', 'not_checked_language')),
  skip_reason text,
  score double precision,
  flagged boolean not null default false,
  question text,
  reply text,
  weakest_sentence text,
  passage text,
  passage_source text,
  sentences jsonb not null default '[]'::jsonb,
  threshold double precision,
  model text,
  feedback text check (feedback in ('fine', 'wrong')),
  feedback_at timestamptz,
  turn_at timestamptz,
  checked_at timestamptz not null default now()
);

create unique index if not exists answer_audit_user_turn_idx on public.answer_audit (user_id, turn_id);
create index if not exists answer_audit_agent_idx on public.answer_audit (agent_id, flagged, turn_at desc);

alter table public.answer_audit enable row level security;
drop policy if exists answer_audit_owner on public.answer_audit;
create policy answer_audit_owner on public.answer_audit
  for all using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
