-- 016_whatsapp.sql: the WhatsApp voice-note channel (routers/whatsapp.py, services/whatsapp.py). Idempotent.
-- `agents.whatsapp` holds {enabled, phone_number_id, handoff_text, voice_for_text, verify_token, access_token_set,
-- access_token_hint, app_secret_set, created_at} and, in cloud mode, `secrets_enc`: the Meta access token and app secret,
-- encrypted with BONCLAY_SECRETS_KEY (the backend refuses to store them when that key is not set). The API never returns
-- `secrets_enc`, the token or the secret; treat the column as sensitive and read it with the service-role key only.
-- `conversations.external_id` is the caller's WhatsApp number (digits), `last_inbound_at` the time of their latest
-- message (a conversation continues while it is inside Meta's 24 hour window), and `needs_human` / `handoff_at` mark a
-- caller who asked for a person (the owner lists them and marks them followed up).

alter table public.agents add column if not exists whatsapp jsonb;
-- one business phone number id belongs to one agent
create unique index if not exists agents_whatsapp_phone_number_id_idx
  on public.agents ((whatsapp ->> 'phone_number_id')) where whatsapp ->> 'phone_number_id' is not null
  and whatsapp ->> 'phone_number_id' <> '';

alter table public.conversations add column if not exists external_id text;
alter table public.conversations add column if not exists last_inbound_at timestamptz;
alter table public.conversations add column if not exists needs_human boolean not null default false;
alter table public.conversations add column if not exists handoff_at timestamptz;

create index if not exists conversations_external_idx on public.conversations (agent_id, external_id)
  where external_id is not null;
create index if not exists conversations_needs_human_idx on public.conversations (agent_id, handoff_at desc)
  where needs_human;
