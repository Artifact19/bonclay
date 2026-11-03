-- 015_agent_integrations.sql: outbound integrations of an agent (services/integrations.py, routers/integrations.py):
-- a signed webhook and Google Sheets logging. Idempotent.
-- `agents.integrations` holds the settings {webhook: {...}, sheets: {...}} and, in cloud mode, `secrets_enc`: the signing
-- secret and the Google service-account key, encrypted with BONCLAY_SECRETS_KEY (the backend refuses to store them when
-- that key is not set). The API never returns `secrets_enc`. Treat the column as sensitive: it is not covered by any
-- client-facing select, and the backend reads it with the service-role key only.
-- `integration_deliveries` is the delivery log (the backend keeps the last 50 per agent): one row per delivery.

alter table public.agents add column if not exists integrations jsonb;

create table if not exists public.integration_deliveries (
  delivery_id text primary key,
  agent_id uuid not null references public.agents (id) on delete cascade,
  user_id uuid references auth.users (id) on delete cascade,
  target text not null check (target in ('webhook', 'sheets')),
  event text not null,
  status text not null,
  code integer,
  latency_ms integer,
  error text,
  attempts integer not null default 1,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists integration_deliveries_agent_idx on public.integration_deliveries (agent_id, created_at desc);

alter table public.integration_deliveries enable row level security;
drop policy if exists integration_deliveries_select_own on public.integration_deliveries;
create policy integration_deliveries_select_own on public.integration_deliveries for select to authenticated
  using ((select auth.uid()) = user_id);
