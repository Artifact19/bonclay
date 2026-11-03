-- 013_agent_widget.sql: the website widget (routers/widget.py). An agent can be published to a business's own site with a
-- public, revocable key. `widget` holds {enabled, origins[], per_visitor_hour, per_key_day, business_name, created_at};
-- `widget_key` is the key the page's <bonclay-agent agent="..."> tag carries (unique; null when never published).
-- `conversations.source` says where a conversation came from ('widget' for website visitors; null for the app).
-- Idempotent.

alter table public.agents add column if not exists widget jsonb;
alter table public.agents add column if not exists widget_key text;
create unique index if not exists agents_widget_key_idx on public.agents (widget_key) where widget_key is not null;

alter table public.conversations add column if not exists source text;
