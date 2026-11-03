-- 008_receipts.sql: signed provenance receipts, one per generation and per agent reply (see services/provenance.py).
-- Idempotent. Written by the backend with the service-role key; the public /api/verify endpoint reads them through
-- the backend too, so there is no anonymous access to this table. RLS lets signed-in users read only their own.
-- Deliberately no foreign keys to voices, generations or conversation_turns: a receipt must outlive the thing it
-- describes (deleting a voice or a clip never erases the record that it was synthetic).

create table if not exists public.receipts (
  receipt_id uuid primary key,
  user_id uuid references auth.users (id) on delete set null,
  version integer not null default 1,
  kind text not null check (kind in ('generation', 'agent_reply')),
  generation_id text,
  turn_id text,
  created_at timestamptz not null default now(),
  engine text,
  voice_id text,
  voice_name text,
  consent_id text,
  consent_verified boolean,
  text_sha256 text not null,
  audio_sha256 text not null,
  watermark_message integer check (watermark_message between 0 and 65535),
  signature text not null
);

create index if not exists receipts_audio_sha256_idx on public.receipts (audio_sha256);
create index if not exists receipts_watermark_message_idx on public.receipts (watermark_message);
create index if not exists receipts_user_id_idx on public.receipts (user_id);

alter table public.receipts enable row level security;
drop policy if exists receipts_select_own on public.receipts;
create policy receipts_select_own on public.receipts for select to authenticated
  using ((select auth.uid()) = user_id);
