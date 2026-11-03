-- 017_instant_answers.sql: reviewed, pre-written answers to the questions callers ask most
-- (services/instant_answers.py, routers/instant_answers.py). Only status = 'approved' rows are used on calls.
-- `vectors` holds the BGE-M3 embeddings of the question and its paraphrases (jsonb list of float lists) and
-- `embed_model` the model that made them; `docs_fingerprint` is the agent's document set when the owner approved the
-- answer (a different set marks the answer 'needs_review'). Idempotent. The backend uses the service-role key and
-- always filters by user_id; the policy below keeps direct client access owner-only too.

create table if not exists public.instant_answers (
  id uuid primary key,
  agent_id uuid not null references public.agents(id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  question text not null,
  answer text not null default '',
  language text not null default 'en',
  paraphrases jsonb not null default '[]'::jsonb,
  vectors jsonb,
  embed_model text,
  variants jsonb,
  source text not null default 'owner' check (source in ('template', 'calls', 'owner')),
  status text not null default 'suggested'
    check (status in ('suggested', 'approved', 'disabled', 'needs_review', 'dismissed')),
  docs_fingerprint text,
  times_asked integer not null default 0,
  approved_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists instant_answers_agent_idx on public.instant_answers (agent_id, status);

alter table public.instant_answers enable row level security;
drop policy if exists instant_answers_owner on public.instant_answers;
create policy instant_answers_owner on public.instant_answers
  for all using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
