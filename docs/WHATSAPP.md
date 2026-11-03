# WhatsApp voice notes

A customer sends a voice note (or text) to your business's WhatsApp number. The BonClay agent answers with a **voice
note** and the **same text** as a second message, in the customer's language, from the agent's documents and
guardrails. It is the same turn as the Playground and the website widget (retrieval, language policy, tools, AI
disclosure on the first message, watermarked audio and a signed receipt), delivered over Meta's WhatsApp Cloud API.

```
customer ──voice note──▶ WhatsApp ──POST──▶  /api/whatsapp/webhook   (signature checked, 200 at once)
                                                  │ background
                                                  ▼
            download the OGG/Opus ▶ speech to text ▶ person asked for? ──yes──▶ handoff text + flag
                                                  │ no
                                                  ▼
            the agent's turn (documents, guardrails, language) ▶ reply audio (watermarked, receipt)
                                                  │
                                                  ▼
            ffmpeg ▶ OGG/Opus mono 48 kHz ~24 kbps ▶ upload ▶ voice note  +  the text as a second message
```

## What you need

* A Meta developer account and a WhatsApp Business app (free; the test number below needs no business verification).
* `ffmpeg` on the server (`brew install ffmpeg`): it decodes the customer's voice note and encodes the reply.
* A public HTTPS address for the webhook. On your laptop that is a tunnel (step 5).

## Setup, step by step (Meta's test number)

1. **Create the app.** developers.facebook.com, My Apps, Create App, type **Business**, then add the **WhatsApp**
   product.
2. **Get the three values.** In the app, **WhatsApp > API setup**:
   * the **Phone number ID** of the test number (a long number, not the phone number itself),
   * a **temporary access token** (valid 24 hours; step 8 makes a permanent one),
   * under **To**, add up to five phone numbers that may talk to the test number (each confirms by a code).

   Then **App settings > Basic > App secret > Show**. That is the **app secret**.
3. **Configure the agent.** Agent Studio, open an agent, scroll to **WhatsApp voice notes**. Paste the phone number ID,
   the access token and the app secret. Optionally write the message people get when they ask for a person. **Save
   settings**, then switch on **Answer on WhatsApp**. **Check connection** asks Meta which number the token belongs to.
   The token and secret are stored on the server only and are never shown again (only the last four characters of
   the token).
4. **Try it without Meta.** **Simulate a message** runs a typed message through the real flow and shows the voice note
   (playable) and the text that would be sent. Nothing is sent to Meta, no Meta settings are needed, and the
   conversation is an owner test (kept out of Analytics and the handoff list).
5. **Expose the webhook.** Meta needs a public HTTPS URL. The phone tunnel that `./bonclay start` opens points at the
   phone-only port (8010), which does **not** carry the WhatsApp webhook (never connected to a real Meta number yet; the
   webhook will join the phone-only app when it is). Until then, a tunnel to the main port exposes the whole local API,
   so only do this on a machine nobody else can reach and stop it afterwards:

   ```sh
   cloudflared tunnel --url http://localhost:8000      # prints https://something.trycloudflare.com
   # or:  ngrok http 8000
   ```

   Type that address into **Public address of this server** in the panel; the **Webhook URL** below it updates
   (`https://something.trycloudflare.com/api/whatsapp/webhook`). Or set `PUBLIC_BASE_URL` on the server.
6. **Register the webhook in Meta.** **WhatsApp > Configuration > Webhook > Edit**: paste the **Webhook URL** and the
   **Verify token** from the panel, **Verify and save**. Meta calls `GET /api/whatsapp/webhook` and expects the
   challenge back. Then **Manage** webhook fields and subscribe to **messages**.
7. **Talk to it.** From one of the numbers you added in step 2, message the test number with a voice note. The reply
   arrives as a voice note, then the text.
8. **Production.** Add your own number (WhatsApp > API setup > Add phone number, needs business verification), and
   replace the 24-hour token with a permanent one: Business settings > Users > **System users** > add one, assign the
   app, generate a token with `whatsapp_business_messaging` and `whatsapp_business_management`. Paste it in the panel.

## Settings and environment

| Where | What |
| --- | --- |
| Panel | Phone number ID (one number belongs to one agent), access token, app secret (both write-only), handoff message, "also reply to typed messages with a voice note" (on by default), enable switch |
| `PUBLIC_BASE_URL` | public address shown as the Webhook URL (otherwise the address the dashboard talks to) |
| `WHATSAPP_APP_SECRET` | one app secret for the whole server (several numbers of one Meta app); an agent's own secret wins |
| `WHATSAPP_ACCESS_TOKEN` | one access token for the whole server; an agent's own token wins |
| `WHATSAPP_VERIFY_TOKEN` | an extra verify token accepted by the handshake (the per-agent tokens work too) |
| `WHATSAPP_GRAPH_URL` | Graph API base (default `https://graph.facebook.com/v23.0`); tests point it at a fake server |
| `BONCLAY_SECRETS_KEY` | cloud mode only: encrypts the stored token and secret (the backend refuses to store them without it) |
| `BONCLAY_WHATSAPP_DIR` | local mode: where the 0600 secrets files live (default `backend/data/whatsapp/`) |

Cloud (Supabase) mode needs `backend/supabase/migrations/016_whatsapp.sql` (`agents.whatsapp`, and on
`conversations`: `external_id`, `last_inbound_at`, `needs_human`, `handoff_at`). Local mode keeps the same fields in
`backend/data/local_db`.

## How a conversation behaves

* **Text** gets a text reply (plus a voice note unless you turned that off). **Voice notes** always get a voice note
  and the text. Other types (images, stickers, locations) get one short "I can read text and voice notes" message.
* **Language.** The caller's language decides the reply language (Hindi, Hinglish, English and the others the agent
  supports), including "answer in Tamil" requests, exactly like the other channels.
* **AI disclosure.** The first reply of a conversation says it is an AI assistant (the agent's "Say it is an AI"
  setting, on by default), in the reply and in the voice note.
* **Conversations.** A caller's messages continue one conversation while they keep writing within 24 hours; after that
  a new conversation starts (and the AI disclosure is said again). Each conversation is stored with source
  `whatsapp` and the caller's number.
* **Human handoff.** If the caller asks for a person ("I want to talk to a human", "manager please", "मुझे मैनेजर से
  बात करनी है", "kisi insaan se baat karao", also in a voice note), the model is not asked. They get your handoff
  message (a default in their language if you left it empty), and the conversation is flagged `needs_human`. The
  panel lists these callers under **Waiting for a person** with their number; **Mark followed up** clears the flag.
  The agent keeps answering other questions from that caller. Following up is a human job done in the WhatsApp
  Business app or your own system, within 24 hours of the caller's last message (see policy).
* **Safety nets.** A reply that fails to encode or upload still goes out as text. A turn that fails sends a short
  apology. A voice note over 3 minutes or 16 MB gets a "send a shorter one" message.

## API

Meta calls (public, authenticated by Meta's signature):

* `GET /api/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=...&hub.challenge=...` echoes the challenge when the
  token is an agent's verify token or `WHATSAPP_VERIFY_TOKEN`, else 403.
* `POST /api/whatsapp/webhook` needs `X-Hub-Signature-256: sha256=<HMAC-SHA256 of the raw body with the app secret>`.
  Missing or wrong: 403 and nothing runs. Valid: `200 {"status":"ok"}` immediately; messages are processed in the
  background, status updates (sent, delivered, read) are ignored, repeated message ids are answered once.

Owner (normal login, the agent must be yours):

| Call | |
| --- | --- |
| `GET /api/agents/{id}/whatsapp` | settings, flags, the masked token hint, Webhook URL, verify token |
| `PUT /api/agents/{id}/whatsapp` | `phone_number_id`, `access_token`, `app_secret`, `handoff_text`, `voice_for_text`, `enabled` (all optional; empty secrets keep the stored ones) |
| `DELETE /api/agents/{id}/whatsapp` | remove the settings and the stored secrets |
| `POST /api/agents/{id}/whatsapp/verify-token` | new verify token |
| `POST /api/agents/{id}/whatsapp/check` | asks Meta which number the token and phone number ID belong to |
| `POST /api/agents/{id}/whatsapp/simulate` | `{"text": "..."}` dry run, returns the messages that would be sent |
| `GET /api/agents/{id}/whatsapp/handoffs` | callers waiting for a person |
| `POST /api/agents/{id}/whatsapp/handoffs/{conversation_id}/resolve` | mark followed up |

## Security

* Every webhook call is verified with HMAC-SHA256 over the raw body (constant-time compare) against the app secret of
  the agent that owns the receiving phone number ID, else `WHATSAPP_APP_SECRET`. With no secret configured the call is
  rejected: there is no switch to skip the check.
* The access token and app secret are **write-only**: the API returns only "set" flags and the token's last four
  characters, and nothing logs them. Local mode: a 0600 file under a 0700 directory; cloud mode: Fernet-encrypted with
  `BONCLAY_SECRETS_KEY`, inside `agents.whatsapp.secrets_enc`, which no API returns.
* The media URL Meta returns is only fetched when it is on a Meta host over HTTPS (the token is sent to it), redirects
  are not followed, and downloads stop at 16 MB.
* A sender can cause at most 40 turns an hour, and at most 3 turns run at once, so a stranger cannot spend your model
  quota.
* Caller phone numbers are stored as the conversation's `external_id` (you need them to follow up) and are masked in
  logs. Voice notes are decoded in memory and not stored; transcripts and the agent's reply audio are stored like any
  other turn.

## Costs, policy and compliance

Check Meta's current pricing and policy pages before you go live: both change.

* **24-hour window.** BonClay only ever replies to a message the customer just sent; it never starts a conversation,
  and there are no templates. Messages older than 23 hours (for example a delayed webhook retry) are not answered.
  A human follow-up after the window closes needs an approved template message, sent from your own tooling.
* **Cost.** Replies inside the customer service window are "service" messages, which Meta does not charge for at the
  time of writing; marketing and utility templates are charged per message. Your costs are the model, speech and
  hosting running BonClay (the test number is free).
* **Bot disclosure.** The first reply says it is an AI assistant, and the voice is AI-generated and watermarked. Keep
  "Say it is an AI" on. Some regions also require telling people when conversations are stored (see Privacy).
* **Human escalation.** Meta expects a business bot to give people a way to reach a person. The handoff detection and
  message do that; staff must actually answer the flagged conversations.
* **Meta's 2026 policy on general-purpose AI assistants.** Meta restricts businesses from offering general-purpose AI
  chatbots (an assistant whose main function is open-ended Q&A, such as a model provider's own assistant) through the
  WhatsApp Business Platform. An BonClay agent is a business agent answering questions about its own company from
  its own documents, so it is the allowed kind. Keep it that way: give the agent a concrete business role and
  documents, do not offer it as a "chat with an AI about anything" line, and re-read the current terms.
* **Privacy.** Tell customers in your privacy notice that AI answers their WhatsApp messages and that messages and
  transcripts are stored. Honour deletion requests by deleting the conversation in BonClay.

## Limits

* One webhook URL per Meta app. Several numbers in one app share `WHATSAPP_APP_SECRET`.
* Duplicate detection, the per-sender limit and the in-flight turn limit are in memory: a restart forgets them, and
  several server processes would each have their own. (Meta retries only deliveries that did not get a 200.)
* The reply voice is the agent's normal voice, so a cloned voice makes replies slower. A first message after a cold
  start can take 30 seconds while the models load; warm replies take a few seconds plus the model time.
* No outbound-initiated messages, templates, groups, images, documents or calls.
* The simulator returns the reply as WAV for preview (the real voice note is OGG/Opus, shown by size).
* The watermark survives the 24 kbps Opus encoding (checked on a real reply: detected, same watermark message
  recovered). The signed receipt always refers to the stored WAV, not to the Opus file.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Meta says "callback URL or verify token couldn't be validated" | tunnel not running, wrong URL, or the verify token differs (New token changes it) |
| Webhook saved, no reply, the log says "invalid signature" | wrong app secret in the panel (use App settings > Basic, not the token) |
| No reply, nothing in the log | the **messages** webhook field is not subscribed, or the agent switch is off |
| Meta error 131030 in Meta's dashboard | the sender is not in the test number's **To** list |
| Meta error 190 / "Invalid OAuth access token" | the temporary token expired (24 h): paste a new one, or use a system-user token |
| Text arrives but no voice note | `ffmpeg` missing, or the upload failed (see the server log); the text still goes out |
