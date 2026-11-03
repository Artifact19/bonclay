# Website widget

(Looking for a link anyone can open to talk to an agent, without embedding anything? See [WEB_CALL.md](WEB_CALL.md).)

One script tag puts an BonClay agent on a business's own website. The visitor taps a microphone button, talks, and
hears the agent answer (with captions). Everything is served from the BonClay API host: there is nothing else to host.

```html
<script src="https://YOUR-BONCLAY-HOST/widget/v1/bonclay-agent.js" defer></script>
<bonclay-agent agent="PUBLIC_WIDGET_KEY"></bonclay-agent>
```

Optional attributes on `<bonclay-agent>`: `label="Talk to us"` (text next to the button), `position="left"`,
`color="#0b7a75"` (accent), `api="https://other-host"` (defaults to the host the script came from).

## Owner setup

1. Agent Studio, open an agent, scroll to **Embed on your website**.
2. List the sites that may use it, one per line, exactly as visitors see them: `https://www.example.com`. No paths, no
   wildcards. `http://localhost:3000` only if you list it (for testing).
3. Optional: business name (shown in the visitor notice), turns per visitor per hour (default 30) and per day for the
   whole key (default 300).
4. Switch on **Publish to website**. The key and the two-line snippet appear, with a Copy button.
5. **Replace key** makes a new key and kills the old one at once. **Remove widget** deletes the key. Switching the
   toggle off keeps the key but every request is refused until you switch it on again.

Cloud mode needs `backend/supabase/migrations/013_agent_widget.sql` (agents.widget, agents.widget_key,
conversations.source). Local mode stores the same fields in the agent's row in `backend/data/local_db/agents.json`.

## What the visitor sees

* A floating microphone button. First open: "You're talking to an AI assistant. Audio is processed to answer you;
  conversations may be stored by <business>." and a **Start** button (acceptance is remembered in the browser).
* Hold the button to talk and release to send; or tap once to start and tap again to send. Space/Enter work on the
  focused button. Pressing while the agent speaks interrupts it.
* Captions of both sides in an `aria-live` log, a status line, and the label **AI-generated voice**.
* Plain-language errors: microphone blocked, no speech heard, too many people right now, cannot reach the assistant.

## API

Owner (normal auth, the agent must be yours):

| Method | Path | |
| --- | --- | --- |
| GET | `/api/agents/{id}/widget` | `{enabled, key, origins, per_visitor_hour, per_key_day, business_name, script_path}` |
| PUT | `/api/agents/{id}/widget` | any of `enabled`, `origins`, `per_visitor_hour`, `per_key_day`, `business_name`; creates the key on first publish; publishing needs at least one origin |
| POST | `/api/agents/{id}/widget/rotate` | new key, old key dead |
| DELETE | `/api/agents/{id}/widget` | removes settings and key |

Public (no owner auth):

| Method | Path | |
| --- | --- | --- |
| GET | `/widget/v1/bonclay-agent.js` | the bundle (about 28 KB, no dependencies, cacheable 5 minutes) |
| POST | `/api/widget/session` | body `{key, previous?}`; needs an `Origin` header on the key's list; returns `{token, expires_in: 600, agent: {name, business_name, greeting}}` |
| POST | `/api/widget/turn/stream` | multipart `audio` + `session` (or `Authorization: Bearer <token>`); the same SSE events as `/api/conversations/{agent}/turn/stream` (`transcript`, `sentence`, `done`, `error`), trimmed for visitors |
| GET | `/api/widget/clip/{turn}/{index}` | a sentence clip of a turn that session started (`Authorization: Bearer <token>`) |

The turn runs the owner's streaming-turn function (`_stream_turn_response` in `backend.py`), as the agent's owner, so
the language, retrieval, tools, disclosure, watermark and receipt behave exactly as in the Playground. Conversations
are stored with `source = "widget"`.

## Security model

* The key is public (it is in the page HTML). It names one agent and nothing else; it is random (`awk_` + 144 bits),
  revocable, and re-checked on every turn, so revoking takes effect immediately, also for open sessions.
* **Origin allowlist**: the browser's `Origin` header must equal one of the listed origins exactly (scheme, host,
  port). Missing or `null` origins are refused. Localhost works only when listed. Non-browser clients can fake
  `Origin`, so the rate limits, not the allowlist, protect the owner's quota.
* **Session token**: HMAC-SHA256 signed (`WIDGET_TOKEN_SECRET`, else a random secret kept in
  `backend/data/widget_secret`, mode 600), valid 10 minutes, bound to key + origin + random visitor id. A turn must
  arrive from the same origin. The widget renews it before it expires (passing the old token keeps the visitor and
  their conversation). The token allows turns for that one agent; no route accepts an agent or conversation id from a
  visitor (the server keeps one conversation per visitor).
* **CORS**: only the validated origin is echoed on `/api/widget/*`; the app-wide CORS list does not apply there.
  Preflights are answered for any origin (they carry no key) but grant nothing by themselves.
* **Limits** (in memory, per process; use one worker or a shared store when scaling out): per visitor IP and per visitor
  id per hour, per key per day, sessions per IP per hour (60 per key, 120 overall). A refused request answers 429
  with `Retry-After` and is not charged. Uploads over 6 MB are refused before they are read.
* **Data minimisation**: visitors get transcript, sentence text and clips, the final text, language and flags. No
  receipts, timings, models, retrieval, tool calls, agent id, owner id or error internals.
* The visitor notice and the "AI-generated voice" label are part of the bundle and cannot be turned off by the
  embedding page's markup (shadow DOM). The agent's own spoken AI disclosure still plays on the first turn when enabled.

## Test it locally

1. Start the API as usual (port 8000) and the web app (3000).
2. In Agent Studio publish an agent with the allowed site `http://localhost:3000` (the demo page is served by Next).
3. Open `http://localhost:3000/widget-demo.html?agent=PASTE_KEY&api=http://localhost:8000`, click the microphone
   button, press Start, allow the microphone, hold the button and ask a question.
4. To serve the page from another port, list that origin too, for example `python3 -m http.server 8013 -d frontend/public`
   and allow `http://localhost:8013`.
5. Without a microphone: Chrome with `--use-fake-device-for-media-stream --use-fake-ui-for-media-stream
   --use-file-for-fake-audio-capture=question.wav`.

Tests: `backend/tests/test_widget.py` (origins, tokens, limits, revocation, CORS, the turn with `source="widget"`),
`frontend/src/components/__tests__/EmbedPanel.test.tsx`, `frontend/src/lib/__tests__/widgetBundle.test.ts` (SSE parser,
clip queue, helpers, the custom element).
