# Phone calls and HOD meeting requests

## Phone calls (Exotel)

The agent can place a call to a phone and talk, or answer a call to an Exotel number. It speaks first (the AI disclosure), and the
caller can talk over it.

**Set up (once)**
1. `backend/.env`: `EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, `EXOTEL_ACCOUNT_SID`, `EXOTEL_CALLER_ID` (your ExoPhone), and
   `EXOTEL_SUBDOMAIN` (`api.in.exotel.com` for India accounts, `api.exotel.com` for Singapore: if you get a 401, try the other).
2. `PUBLIC_BASE_URL`: the https address Exotel can reach. `./bonclay start` sets it from the tunnel it starts (ngrok if
   installed, else cloudflared; `BONCLAY_TUNNEL=off` to skip). The tunnel points at the **phone-only port** (`PHONE_PUBLIC_PORT`,
   default 8010; `backend/phone_public.py`), which serves only the call stream and the status callback: the rest of the API
   (voices, cloning, keys, the call endpoint) stays on the loopback-only main port and is never reachable from the internet.
   By hand: `ngrok http 8010` (not 8000) and paste the address; it changes when ngrok restarts. A request from another
   device on your network needs `BONCLAY_LOCAL_TOKEN` (see `backend/.env.example`).
   **Your own named tunnel** (for example a Cloudflare tunnel with a fixed address): put `BONCLAY_TUNNEL=off` and
   `PUBLIC_BASE_URL=https://phone.example.in` in `backend/.env`. `./bonclay` reads both from the shell first, else from
   `backend/.env`, starts no tunnel of its own, and `start`, `status` and `doctor` show your address (doctor also checks it
   answers). The tunnel must point only at the phone-only port (`PHONE_PUBLIC_PORT`, default 8010), never at 8000.
3. `PHONE_ALLOWED_NUMBERS`: the numbers that may be called, for example `+919876543210`. A trial Exotel account also only rings
   numbers you whitelisted in the Exotel dashboard.
4. `./bonclay stop` and `./bonclay start`. Placing a call warms everything the call needs while the phone rings; `./bonclay warm`
   still warms everything up front. With `PHONE_CLONE_VOICE=1` and a cloned voice, run one test turn first, or the first sentence
   waits for the voice engine to load.

**Call from the app:** Agents, press **Call** on an agent, pick the number, tick that the person agreed, press Call.

**Call from the terminal**
```bash
curl -X POST localhost:8000/api/phone/exotel/call -H 'storage-mode: local' -H 'content-type: application/json' \
  -d '{"agent_id":"<agent id>","to":"+91XXXXXXXXXX","caller_agreed":true}'
```

**Inbound:** the Call dialog has "Let people call this agent": copy the wss address into the Voicebot applet of your ExoPhone.

**If the agent cannot take the call (fall back to a person).** In the Exotel flow, put a **Connect** applet that dials your
mobile (or the agent's transfer number) **after** the Voicebot applet. The flow moves on to it when the agent refuses the
stream because it is already on `PHONE_MAX_CALLS` calls (default 2; the stream is closed with code 4429 before it is
accepted), when the stream cannot connect at all (the computer is asleep, the tunnel is down, the backend is stopped), and,
with `PHONE_TRANSFER_MODE=flow`, when the agent hands the caller over. `GET /api/phone/exotel/status` shows `max_calls`,
`active_calls` and this fallback; `POST /api/phone/exotel/inbound-link` returns the same under `flow`. That the Voicebot
applet continues to the next applet on a refused or failed stream is how Exotel documents applet chaining, but it has not
been verified on a real account yet: the week-2 gate is five forced failures that all reach the mobile on a real call.

Limits and notes: calls end after `PHONE_MAX_CALL_S` (default 600 s); at most 10 calls an hour; cloned voices pause a few seconds
between sentences, preset voices (Kokoro) answer in about a second; the code is `backend/routers/phone_exotel.py`,
`backend/services/phone_audio.py` and `backend/services/phone_greeting.py`. Call only people who agreed.

**Instant greeting.** The agent's opening is always the same words: the AI disclosure ("This is an AI assistant.") and the
agent's **Greeting** field (a short "Hello! How can I help you today?" in the agent's first language when it is empty). It is
rendered once, ahead of the call, with the voice the agent's phone replies use (Kokoro, or the cloned voice with
`PHONE_CLONE_VOICE=1`), watermarked and given one receipt per rendered clip, converted to 8 kHz (16 or 24 kHz when Exotel asks
for it), and kept in memory and in `backend/data/phone_greetings/` (so a restart does not render it again). When Exotel's `start`
arrives it is sent at once: no speech recognition, knowledge search or language model. It is stored as the conversation's first
turn (`[call connected]` for the caller, the greeting for the agent), so the model knows it was said and the first reply does not
repeat the disclosure. If the caller talks over the greeting before the disclosure has finished, nothing is stored and the first
reply says the disclosure. The clip is rendered when you place a call (with a warm-up of the language model connection, the
knowledge search, speech recognition, the watermark model and the voice while the phone rings), when you create an inbound link,
and a few seconds after the backend starts for agents that have taken calls. Changing the greeting, the voice or the watermark
settings gives a new clip (the cache key covers them); the old one is removed. The log shows per call `accept->start` and
`start->first byte` in ms and which clip and receipt were played.

**Answering sooner.** The caller's sentence ends after `PHONE_END_SILENCE_MS` of silence. After `PHONE_EARLY_SILENCE_MS` (default
250) the reply already starts, speculatively: its audio is held until the full silence confirms the end of the sentence, and if
the caller goes on talking it is cancelled (nothing is sent or stored) and a new reply starts when they finish. Agents with
tools never start speculatively (a tool may book something). The log line `phone turn N latency` shows each stage in ms after the
end-of-speech event (a speculative start is negative) and the first audio after the caller's last word.

| Setting (backend/.env) | Default | What it does |
|---|---|---|
| `PHONE_END_SILENCE_MS` | 500 | silence that ends the caller's sentence |
| `PHONE_EARLY_SILENCE_MS` | 250 | silence after which the reply starts speculatively (0 or >= the end silence: never) |
| `PHONE_SPECULATIVE` | on | `on`: the whole reply runs during the pause and only its audio waits; `asr`: only speech recognition runs during the pause (a cancelled start costs no language-model call); `off` |
| `PHONE_PREWARM` | on | render greetings and warm up in the background (a call still renders its greeting when it needs one) |
| `PHONE_PREWARM_MAX` / `PHONE_PREWARM_DELAY_S` | 8 / 5 | how many agents' greetings are checked after startup, and how long after startup |
| `PHONE_WARM_TTL_S` | 60 | a warm-up of the same agent within this time only refreshes the language-model connection |
| `PHONE_GREETING_DIR` | `backend/data/phone_greetings` | where the rendered greetings are kept |

**Measuring it without a phone.** `backend/.venv/bin/python backend/bench/phone_latency.py --agent <agent id> --runs 5` plays
Exotel against the running backend's phone WebSocket (no call is placed, Exotel is not contacted): it streams a spoken question
(Kokoro's preset voice) in real time and prints the median start -> first greeting byte and end of speech -> first reply byte.
Each run is stored as a phone conversation of that agent.

## Memory and turn-taking on calls (added 23 October 2026)

On a test call the agent asked "What are the patient details?"; the caller said a name and a number, paused, then said the day
and the time. The agent cut in during the pause, asked "Can you tell me the patient's name?" and kept asking for what it had
been told. Three causes, three fixes. All of it is tested against fakes only (no real call yet): try it on your own phone first.

1. **The caller's words were thrown away.** A turn is stored (caller's words, reply, receipt) only at the end, once the end of the
   caller's speech is confirmed. When the caller spoke again while a turn was still thinking or speaking, that turn was cancelled
   before it was stored and the next turn only had the new audio. Now the audio of a turn cancelled before it was stored is kept
   and put in front of the next utterance (a 300 ms gap between the pieces), so the next turn hears and answers the whole thing.
2. **A fixed half second ends the turn.** People pause 0.7 to 1.5 s between the items of a name, a number and a date. After each
   reply the call now sets the silence that ends the caller's next utterance from what the reply asked for, and waits once more
   when the answer looks unfinished.
3. **Only a booking tool remembered anything.** The model had the dialogue and nothing else, so a name given before any tool ran
   was gone for an agent without tools. A per-conversation "caller details" form now holds what the caller said, and every turn
   of every agent (phone and website, with or without tools) is told what is in it.

| Setting (backend/.env) | Default | What it does |
|---|---|---|
| `PHONE_CARRY_UNANSWERED` | on | carry the audio of a reply cancelled before it was stored into the next turn (at most 3 pieces, under 25 s in all; a stored turn is never carried; a speculative start cancelled because the caller went on is not carried either, the detector already holds the whole utterance). The log says `phone turn N: carried unanswered caller audio (X ms) into the next turn` |
| `PHONE_ADAPTIVE_END` | on | set the end silence from the agent's last reply: `PHONE_DETAILS_END_SILENCE_MS` when it asked for several details ("details", "jaankari", two or more of name / number / date / time / e-mail / address), `PHONE_DETAIL_END_SILENCE_MS` when it asked for one of name, number, date, time, e-mail or address, else `PHONE_END_SILENCE_MS`. English, Hindi and Hinglish. A short piece (under 1.5 s) while the agent thinks is no longer ignored as "hello?" after such a request |
| `PHONE_DETAIL_END_SILENCE_MS` | 1100 | the wait after one detail was asked for |
| `PHONE_DETAILS_END_SILENCE_MS` | 1500 | the wait after several were |
| `PHONE_INCOMPLETE_EXTRA_MS` | 1200 | at the end silence, when the speculative turn's transcript ends in "and", "or", "aur", "toh", "phir", "uske baad", "matlab", "um", "uh", "haan wo" or a comma, or the agent asked for a phone number and the transcript holds 1 to 9 digits (digits said as words, in Devanagari, "double five" included), the reply is not confirmed yet: wait this long once more. If the caller goes on, the speculative reply is dropped and the whole of what they said is answered. 0 turns it off. An agent that does not reply early (it has tools, or `PHONE_SPECULATIVE=off`) gets the same check only after it asked for details (the wait above is 1100 or 1500 ms): its recording goes to speech recognition at the caller's pause (or at the end silence when there is no pause) with nothing said or stored; when the caller has finished, that result is the turn's, so there is no second recognition; when the answer looks unfinished the turn starts only after the extra wait, and the caller going on in that time gets one answer to everything. A speculative turn whose speech recognition has not finished at the end silence is confirmed as before. "The patient details" counts as a request that may include a number: 3 or more digits and fewer than 10 are an unfinished number |
| `CALLER_MEMORY` | on | keep the caller details form and tell the model on every turn: `(Details the caller already gave in this conversation: name ..., phone ..., date 2026-10-25, time 17:00. Do not ask for these again. If you still need other details, ask for all the missing ones together in one short question.)`, after the caller's words (the system prompt does not change, so the model's prompt cache keeps working). The arguments of earlier tool calls are merged in and win for the keys they share. Rules read the phone (ten digits, +91 or 0 prefix, numerals or words), the date and time (the resolver of `services/when.py`, relative to the call's local day), an e-mail, an age, and a name only with a clear cue ("my name is", "mera naam ... hai", "मेरा नाम ... है") or as the whole answer to a question about the name |
| `CALLER_MEMORY_LLM` | on | one small JSON-mode Groq call per caller turn, in the background once the reply is stored, for what the rules cannot read (a name with no cue, a doctor, a reason). Never when `GROQ_API_KEY` is missing or in offline mode; any error or rate limit skips it. It only fills what is still empty, and its phone, date and time are checked like the rules' |
| `CALLER_MEMORY_MODEL` | `openai/gpt-oss-20b` | that model (the owner's Groq account serves it; it has its own quota, apart from the reply model). A gpt-oss model is asked for `reasoning_effort` "low" with room for it in `max_tokens`. If the call fails, is cut off or is rate limited it is skipped silently, so the form is then filled by the rules alone |
| `ASR_FORMAT_EXAMPLE` | on | after the agent asks for a name, a number, a date or a time, the next caller turn's Whisper prompt ends with an example of how it is written ("Name: Priya Sharma. Phone: 98765 43210. Date: 25 October, 5 baje."); no language is forced. Phone calls only for now (the call hands the turn what the agent said last) |

The form is stored on the conversation (`caller_details`, local store) or, in cloud storage mode, in `backend/data/caller_details.json`
(mode 0600; `BONCLAY_CALLER_DETAILS_FILE` moves it). Only a turn that was stored changes it, so a speculative turn that is
cancelled leaves it alone. The Calls page API (`GET /api/calls/{id}`) returns it as `details`, with the number and the e-mail masked.
The values are the caller's personal data: the log names which keys were read, never the values.

The `phone turn N latency` log line now ends with `end_wait=1500(details)`, or `400(normal)+1200(incomplete:trailing)` when the
extra wait was used, next to the first audio after the caller's last word.

## Calls, campaigns and live control (added 19 October 2026)

Everything in this section is tested against a fake Exotel and a fake phone only. Nothing here has been tried on a real call, so
check each item with your own phone first. Branch: `m5-call-records-campaigns`.

**Call records.** Every call is kept in `backend/data/call_records.jsonl`: status, direction, agent, how it ended
(`hangup_reason`), the conversation with its transcript, and the outcome when there was one (`voicemail`, `human_requested`,
`ended_by_agent`, `transfer_requested`, `human_took_over`). The callee's number is stored masked only. Exotel is asked to report status
to `POST /api/phone/exotel/status-callback/{token}` (a signed address; it needs `PUBLIC_BASE_URL`). Its field names (`CallSid`,
`Status`, `Duration`) are read leniently and have not been checked against a live account. `GET /api/phone/exotel/calls` lists
your calls.

**Campaigns** (Agents, Campaigns). Upload a CSV (a number column, optionally a name column, up to 500 people), choose the days,
hours and time zone, how many calls at once (1 to 3), tries per person and the wait between tries, and tick that everyone agreed
to be called. A background loop (`CAMPAIGN_TICK_S`, default 10 s) places the calls through the same Exotel path as a single call.
It tries again on busy, no-answer and failed calls, never goes over the hourly cap, and pauses the campaign when half of the
last ten calls failed to connect. The **do-not-call list** applies to every campaign, including people added later.
Numbers are stored in full in `backend/data/campaigns.json` (file mode 0600) until you delete the campaign or it is purged
`CAMPAIGN_KEEP_DAYS` (30) after it finished. BonClay does not check India's do-not-disturb register or any other register: check
your list yourself. The agent still says it is an AI at the start of every call. `CAMPAIGN_SCHEDULER=off` turns the loop off.

**Voicemail and key presses.** On an outbound call, the callee's first sentence is checked for an answering machine
(`VOICEMAIL_DETECT`, `VOICEMAIL_LONG_MS`): clear phrases ("leave a message after the beep", "the number you have dialled", Hindi
and Hinglish equivalents) end the call without the agent answering, the outcome is `voicemail`, and a campaign tries that person
again later. A phrase a person could say ("not available") only counts after a long first sentence. A machine that plays only a beep
is not detected. Key presses are counted, never stored (they may be a card number or a code); pressing 0 flags the conversation as
needing a person. Exotel's `dtmf` event shape is not verified.

**Ending and handing over** (tools the agent can call during a phone call). Turn on the tool server **call** for the agent through the API (the
Tools switches in the agent screen do not list it yet): `PUT /api/agents/{id}` with `{"tools": ["appointments", "call"]}`, keeping the
ones it already has. `end_call` hangs up after the agent's goodbye. `transfer_call` exists only when you set a transfer
number: `PUT /api/agents/{id}/call-settings` with `{"transfer_number": "+91..."}`. The model cannot choose the number.
`PHONE_TRANSFER_MODE=callback` (default): the caller is told someone will call back, the conversation is flagged for a person and the
call record notes the masked number. `PHONE_TRANSFER_MODE=flow`: the audio stream is ended so the Exotel call flow can continue;
add a Connect step that dials your number after the stream step in the Exotel flow. That behaviour depends on how your Exotel flow is
built and has not been verified. There is no screen for the transfer number or the tools yet.

**Live calls** (Calls and bookings, "Live now"). A call in progress is listed with **Listen in**: you hear both sides, read the
transcript, and can **Take over**: the agent stops talking and listening, you speak into the call with your microphone, and **Hand
back** (or closing the panel, or losing the connection) returns it to the agent. One person can have the call at a time and up to three
can listen. Your voice is not transcribed or saved; the conversation is flagged for a person. The browser audio and the
microphone path have only been tested with mocks, so expect to fix rough edges on a first real call.

## Several Exotel accounts

Trial accounts run out of credit, so `backend/.env` can hold up to nine. Account 1 is the names you already have
(`EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, `EXOTEL_ACCOUNT_SID`, `EXOTEL_CALLER_ID`, optional `EXOTEL_SUBDOMAIN`). Accounts 2 to 9 use the
same names ending `_2` ... `_9` (a gap is fine). An account counts only when its key, token, SID and caller id are all set; a
half-filled one is ignored and the log names the missing variables (never their values).

- `EXOTEL_ROTATION=failover` (default) always starts with the first usable account in order. `round_robin` takes the usable
  accounts in turn. Both move on when an account refuses.
- Out of credit (HTTP 402, or Exotel's message mentions balance, credit, insufficient, recharge, quota or limit exceeded): the
  account is marked `exhausted` and the call goes to the next one. Credentials refused (HTTP 401 or 403): marked `auth_failed`, next
  account. A message saying the number is not whitelisted or not verified moves to the next account without marking this one
  (it is the number, not the account). Any other refusal (bad number, KYC, a server error) stops the call as before, and a network
  error is never retried on another account, because the call may already have been placed. Each account is tried at most once per call.
- When every account is out, the call fails with `EXOTEL_ALL_ACCOUNTS_EXHAUSTED` and the message lists each account's state. A marked
  account is tried again after `EXOTEL_EXHAUSTED_RESET_HOURS` (default 24); a successful call on it makes it `active` at once. A marked
  account is never left out: active accounts are tried first, then the marked ones in order (each at most once per call), so a
  recharge is noticed and a call stops only when no account works.
- The state is kept in `backend/data/exotel_accounts.json` (mode 0600, keyed by a hash of the SID, no keys or tokens) and survives a
  restart. `GET /api/phone/exotel/status` lists `accounts` with label, masked caller id, state and since. Each call record notes
  the account that placed it (`exotel_account`), and the Call dialog's live status is read with that account's credentials.
- **Inbound calls stay per ExoPhone.** This change is for outbound calls only: each ExoPhone's own flow needs the stream link from
  the Call dialog, one flow per account you want to answer.
- The person called sees the caller id of whichever account placed the call, so it can change from one call to the next.
- A trial account only rings numbers that are whitelisted on that same account: add the test phone in every account's dashboard,
  or the call moves on to the next account and may fail on all of them.

## Phone providers (Exotel, Twilio, Plivo, Vobiz, Telnyx; added 27 October 2026)

The same agent can place and answer calls through any of five carriers, chosen by configuration, with failover across providers
and accounts. The call itself (listening, turns, barge-in, greeting, live monitor, campaigns, call records) is one piece of
code; a provider contributes only its **accounts**, the **HTTP call** that places a call, its **refusal rules** and its
**WebSocket dialect** (`backend/services/telephony/`: `base.py`, `exotel.py`, `twilio.py`, `plivo.py` for Plivo and Vobiz,
`telnyx.py`, `registry.py`, `state.py`). With nothing set, only Exotel is used and everything behaves as before.

### Choosing and ordering providers

- `PHONE_PROVIDERS=exotel` (default). A comma list, in order of preference: `PHONE_PROVIDERS=twilio,exotel` tries Twilio first.
  A name that is not one of `exotel, twilio, plivo, vobiz, telnyx` is ignored with one warning.
- `PHONE_ROUTES=+919876543210:exotel#2,+919812345678:twilio` pins numbers (any format; they are normalised to E.164):
  that number goes to Exotel account 2 first, the other to Twilio (any account, in its own order) first. Whatever fails
  falls through to the providers in `PHONE_PROVIDERS` order. A route to a provider that has no complete account is skipped.
- Accounts use the scheme Exotel always had: account 1 is the plain names, accounts 2 to 9 end `_2` ... `_9`; a half-filled
  account is ignored and the log names the missing variables. Per provider, `<PREFIX>_ROTATION=failover|round_robin`
  and `<PREFIX>_EXHAUSTED_RESET_HOURS=24` (prefix `EXOTEL`, `TWILIO`, `PLIVO`, `VOBIZ` or `TELNYX`).

| Provider | Settings (account 1; `_2`..`_9` for more) |
| --- | --- |
| Exotel | `EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, `EXOTEL_ACCOUNT_SID`, `EXOTEL_CALLER_ID`, optional `EXOTEL_SUBDOMAIN` |
| Twilio | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_CALLER_ID` |
| Plivo | `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_CALLER_ID` |
| Vobiz | `VOBIZ_AUTH_ID`, `VOBIZ_AUTH_TOKEN`, `VOBIZ_CALLER_ID`, optional `VOBIZ_BASE_URL` (default `https://api.vobiz.ai/api/v1`) |
| Telnyx | `TELNYX_API_KEY`, `TELNYX_CONNECTION_ID`, `TELNYX_CALLER_ID` |

`PUBLIC_BASE_URL` (the tunnel) is shared by all of them. `GET /api/phone/exotel/status` keeps every field it had and adds
`providers: [{name, display, configured, enabled, verified, accounts: [{label, caller_id_masked, state, since}]}]` and
`provider_order`; never a key, token or SID.

### Failover

For one call the order of attempts is: the routes for that number, then every **active** account of the providers in
`PHONE_PROVIDERS` order (each provider's accounts in its own order, or rotated with `round_robin`), then the accounts marked
`exhausted` or `auth_failed`, in the same order. Each (provider, account) is tried at most once. The rules are Exotel's, per
provider:

- Account-level refusal (out of credit, credentials refused, this number not allowed on the account, KYC not finished) moves to the
  next. Out of credit and refused credentials mark the account (kept in `backend/data/exotel_accounts.json` for Exotel and
  `phone_accounts_<provider>.json` for the others, mode 0600, keyed by a hash, no secrets; forgotten after the reset hours; a
  successful call makes it active at once). A number-not-allowed or KYC refusal marks nothing.
- Any other refusal (bad number, a server error), and any network error, stops the call: the next account would refuse it the same
  way, or the call may already have been placed.
- One (provider, account) configured: the plain refusal, `<PROVIDER>_REFUSED` (for example `TWILIO_REFUSED`), as Exotel always gave.
- All failed: one provider with several accounts gives `<PROVIDER>_ALL_ACCOUNTS_EXHAUSTED` (Exotel: unchanged, plus
  `EXOTEL_KYC_REQUIRED` when every account lacks KYC); several providers give `PHONE_ALL_ACCOUNTS_FAILED`: "Every phone account
  refused the call (exotel 1 not KYC compliant, twilio 1 credentials refused). ...".
- The dial answer adds `provider` and `account` (`{"status": "calling", "call_sid": ..., "provider": "twilio", "account": "2"}`); the
  Call dialog shows "via Twilio 2"; the call record has `provider` and `account` (and, for Exotel, still `exotel_account`).

### Routes

Every provider but Exotel has `/api/phone/{provider}/...`; Exotel keeps its own paths. On the phone-only public port (`phone_public.py`,
the only thing the tunnel exposes) a provider adds exactly three routes, all authenticated by a signed token in the URL, and nothing else:

| Route | What it is |
| --- | --- |
| `WS /api/phone/{provider}/stream/{token}` | the call's audio WebSocket (token names the agent and the provider) |
| `POST /api/phone/{provider}/status-callback/{token}` | the provider's call-status callback (form fields; Telnyx posts JSON) |
| `GET, POST /api/phone/{provider}/answer/{token}` | the answer document (TwiML for Twilio, Plivo XML for Plivo and Vobiz), or Telnyx's Call Control webhook |

A provider's routes are **inert unless it is in use** (named in `PHONE_PROVIDERS` or a `PHONE_ROUTES` entry): the stream closes with
4404 and the answer is a 404, and `GET /api/phone/public/health` lists only the routes that work. An answer token (`k: "ans"`)
opens no stream and a stream token fetches no answer. Why an answer route at all: Plivo and Vobiz take an Answer URL, not the stream address, when
a call is placed, and every provider's inbound call needs a public address to ask what to do.

### Inbound calls

`POST /api/phone/{provider}/inbound-link` with `{"agent_id": ...}` (owner; Exotel's `inbound-link` is unchanged) returns
`answer_url` (valid 30 days) and `how`. Each call to the number fetches it and is sent to a fresh stream address for the agent.

- **Twilio**: Console, Phone Numbers, the number, Voice Configuration, "A call comes in": Webhook, HTTP POST, the `answer_url`.
- **Plivo / Vobiz**: create an Application whose Answer URL (method POST) is the `answer_url`, attach it to the number.
- **Telnyx**: create a Voice API (Call Control) application whose webhook URL is the `answer_url` and assign the number to it. The
  URL answers the `call.initiated` webhook through the API with the stream attached (account chosen by the webhook's `connection_id`).
- **Exotel**: as before, the Voicebot applet with the `stream_url`.

### Ending a call

Closing the stream ends the call on Exotel, Twilio (nothing follows the `<Connect>`) and Plivo/Vobiz (`keepCallAlive`, nothing
follows). A **Telnyx** call outlives its stream, so when the agent ends the call (goodbye, answering machine, time limit, promised
callback) it is hung up through the API.

### What each provider's trial allows, and what is verified

| Provider | Trial / free tier (check the provider's current terms) | Status |
| --- | --- | --- |
| Exotel | Whitelisted numbers only, KYC before outbound; credit runs out | **Verified** on real accounts (22 to 23 October 2026) |
| Twilio | Trial accounts may only call numbers verified in the console, and geo permissions can block a destination (error 21215, treated as "number not allowed") | **Unverified**: from Twilio's docs, never run on a real account |
| Plivo | Sandbox: only numbers verified in the console | **Unverified**: from Plivo's docs |
| Vobiz | Not checked: read Vobiz's current terms | **Partly documented, unverified**: call API, headers, `<Stream>`, `start`, `playAudio`, `clearAudio`, `checkpoint`, hangup are documented; `media`, `dtmf` and the call-detail shapes are assumed to be Plivo's |
| Telnyx | Not checked: read Telnyx's current terms for a new account (a refusal worded "not verified" is treated as "number not allowed") | **Unverified**: from Telnyx's docs; the bare-mu-law RTP payload and webhook cause names are assumed |

Audio is mu-law 8 kHz on Twilio, Plivo, Vobiz and Telnyx (converted to 16-bit PCM at the edge of the codec); Exotel stays raw PCM
at the rate its applet names. Sources (read 27 October 2026): Twilio
`twilio.com/docs/voice/media-streams`, `/media-streams/websocket-messages`, `/twiml/stream`, `/api/call-resource`; Plivo
`plivo.com/docs/voice/api/call/make-a-call` and `docs.plivo.com/docs/voice-agents/audio-streaming/concepts/audio-streaming-reference`;
Vobiz `vobiz.ai/docs/call/make-call`, `/call/hangup-call`, `/call/retrieve-live-call`, `/xml/stream`, `/integrations/websockets`;
Telnyx `developers.telnyx.com/docs/voice/programmable-voice/media-streaming` and the Dial, Answer, Hangup and Retrieve-a-call-status
API references. The module docstrings in `backend/services/telephony/` list the exact URLs and what is assumed.

## HOD meeting requests (template "HOD Office Assistant")

A student, parent or faculty member talks to the agent. It checks the HOD's free time, offers other times when the wanted one is
taken or the day is closed, and puts in a **request** that holds the slot. The HOD approves or declines it on the **Approvals** page
(Agents mode). The agent never confirms a meeting itself; the caller can ask about a request later with its number or phone number,
and hears the HOD's note and any other time she suggested.

- Rules (office hours, slot length, blocked times, daily limit, notice, who may book) are in `backend/data/mcp/meetings.json` under
  `settings`, created with defaults on first use (`backend/mcp_servers/meetings.py`, `DEFAULT_SETTINGS`). Edit the file and restart.
- Google Calendar (optional): set the variables in `backend/.env.example`. Then busy times on the calendar are never offered, and an
  approved meeting is added to it (and removed when cancelled). If the calendar cannot be read, the agent says so and carries on from
  its own list.
- The sample document `backend/templates/agents/hod_office.md` holds placeholder rules: replace it with the department's real ones.
- API: `GET /api/meetings`, `POST /api/meetings/{id}/approve` and `/decline`, `GET /api/meetings/week`, `PUT /api/meetings/settings`,
  `PUT /api/meetings/slots` (`backend/routers/approvals.py`).

**This week (Approvals page, 23 October 2026).** A Monday-to-Friday grid of the slot times: Busy (hatched, the reason in the tooltip),
free, Asked for (a pending request), Booked (approved). Click a free or busy cell to flip it (your mark is kept in the data file under
`settings.overrides`); click Asked for or Booked to approve, decline or open the full request. Two switches sit under it, both stored
in the meetings data file, so the agent's tool process reads them on every call:

| Switch | Default | On | Off |
|---|---|---|---|
| **Agent checks this schedule during calls** (`availability_check`) | on | The agent offers only free times and suggests others; a taken or busy slot is refused. | `check_availability` never says a time is free or busy: it says the schedule is not checked on calls, and the agent takes the caller's preferred day and time. `request_meeting` accepts any time inside office hours (not a closed day, not outside the hours, not in the past) as a **pending** request, and the agent tells the caller the office will confirm. You still approve or decline each one here. |
| **Fill the week with a sample schedule** (`sample_schedule`) | on, for demos | Each ISO week gets the same made-up busy slots every time (about a third of the slots: lectures, a department meeting, exam duty, visitors). They show as "Sample schedule" and count as busy for the agent while it checks the schedule. A "free" mark on a cell clears one. | Only your blocked times, Google Calendar and your own marks are busy. |

Old data files without these keys load with both on. The sample week is not real: turn it off before the agent talks to real callers.

