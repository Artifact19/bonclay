-- 006_consent_speaker.sql: consent hardening. Idempotent.
--   speaker_score : cosine similarity between the consent recording and the voice sample (null = check not run)
--   challenge_id  : the server-issued consent challenge the recording answered (null = legacy fixed sentence)
alter table public.consents add column if not exists speaker_score real;
alter table public.consents add column if not exists challenge_id text;
