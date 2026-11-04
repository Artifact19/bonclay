# Web call link

A link anyone can open to talk to one BonClay agent by voice, in their browser. No phone network, no KYC, no login,
nothing to install. Share it on WhatsApp, in an email, or on a slide: `https://phone.example.in/call/awc_...`.

The page is served by the phone-only public app (`backend/phone_public.py`, the port your tunnel already points at),
so the main API (voices, cloning, keys, calls, system) stays unreachable from the internet.

## Share a link

1. Agent Studio, open an agent, **Channels**, **Web call link**.
2. Choose how long the link lives (1 day, 7 days or never; the default is 7 days), the minutes per call (default 10, at
   most 30), the minutes per day for everyone together (default 60) and, optionally, the minutes one visitor may use per
   day. Add a note for yourself if you like (visitors never see it).
3. **Create link**, then **Copy link** and send it. The card shows sessions, turns and the minutes used today for each link.
4. **Revoke** switches a link off at once, also for a call in progress (it ends on the visitor's next turn). A revoked or
   expired link can be removed from the list.

The address comes from `PUBLIC_BASE_URL` in `backend/.env` (the same https address the phone line uses). Without it the
card shows `http://localhost:8411`-style links that only open on this computer, and says so.

## What the visitor sees

* A page branded "BonClay" with the agent's name, a plain statement that they are talking to an AI (their voice is
  recorded while they speak, sent to be turned into text and answered, and the conversation may be saved), and a big
  **Start talking** button.
* After the browser's microphone prompt the call is hands-free: speak, pause, the agent answers out loud. Speaking over
  the agent cuts it off. The microphone has echo cancellation on; headphones help.
* Live captions of both sides (one line per finished sentence, in an `aria-live` log so a screen reader reads each line
  once), a status line, **Mute microphone**, **End call** and a countdown. One minute before the limit a line says so.
* A footer: "Voice is AI-generated, watermarked and signed; check a clip at bonclay.app/verify".
* It works on phones (no horizontal scroll, large touch targets), follows the light or dark setting, and stops
  decorative motion when the visitor prefers reduced motion.
* Friendly messages for a blocked or missing microphone, a link that is switched off or expired, "too many people right
  now", and the day's minutes being used up.

## Limits (defaults)

| Limit | Default | Where it is enforced |
| --- | --- | --- |
| Call length | 10 min (1 to 30 per link) | inside the signed session token; the next turn after it is refused |
| Minutes per day, per link | 60 (1 to 1440) | server side, kept with the link; new sessions and turns are refused when used up. Resets at local midnight |
| Minutes per visitor per day | off (1 to 1440 when set) | server side, per visitor IP, in memory |
| Active links per agent | 20 | owner route |
| Page loads per IP | 90 per minute | in memory |
| Session starts | 10 per hour per IP per link, 30 per hour per IP, 300 per day per link | in memory |
| Turns | 20 per minute per visitor, 300 per hour per IP; each turn costs 5 to 60 s of the day's minutes | in memory |
| Upload | 6 MB per recording, refused before it is read | middleware |

A refused request answers 429 with `Retry-After` (rate limits) and is not charged. "Minutes" are server time between a
visitor's turns (at least 5 s and at most 60 s per turn) plus the time to the End button, not something the browser
reports. In-memory limits reset when the app restarts; the link's day total survives restarts.

The visitor's IP is the connection's address, or, when the connection comes from this computer (the tunnel), the address
the tunnel reports (`CF-Connecting-IP`, else the last `X-Forwarded-For` entry). Behind a tunnel that reports no address,
every visitor shares one bucket, so the per-IP limits become overall limits (safe, but stricter).

## Public routes

Only these exist on the public app for the web call, and each is useless without a valid link id:

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/call/{link}` | the page (HTML; 404 for an id that was never a link, 410 for a revoked or expired one) |
| GET | `/call/assets/call.js`, `/call/assets/call.css` | the page's script and stylesheet (fixed names, cacheable 5 minutes) |
| POST | `/call/api/{link}/session` | a signed session token for this page; the request's `Origin` must be the page's own host |
| POST | `/call/api/{link}/turn` | one spoken turn (multipart `audio`, `Authorization: Bearer <token>`); returns the same trimmed server-sent events as the website widget |
| GET | `/call/api/{link}/clip/{turn}/{n}` | one sentence clip of a turn that this session started |
| POST | `/call/api/{link}/end` | the visitor hung up (settles the minutes) |

Owner routes (main API, normal auth, never on the public app): `GET/POST /api/agents/{id}/web-call-links`,
`POST .../{link}/revoke`, `DELETE .../{link}`.

## Security model

* The link id (`awc_` plus 192 random bits) is a bearer capability for one agent's conversation. It names the link, the
  link names the agent. No route takes an agent, conversation, owner or turn id from a visitor, and the page contains none.
* Everything is re-read on each request: revoking, expiry, a deleted agent or a changed owner stops the next turn.
* Session tokens are the widget's HMAC tokens (`WIDGET_TOKEN_SECRET`, else a random secret in `backend/data/widget_secret`),
  bound to link, page origin and a random visitor id, with a `call` kind. A token made for one link fails on any other
  link, and widget tokens and call tokens are refused by each other's routes.
* POSTs must come from the page's own origin (the `Origin` host equals the `Host` the request arrived for, the
  `X-Forwarded-Host`, or the `PUBLIC_BASE_URL` host). No CORS header is ever sent, so a page on another site cannot read
  an answer. A script outside a browser can fake `Origin`; the limits above are what protect your minutes.
* The page is sent with `Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; connect-src
  'self'; media-src 'self' blob: data:; frame-ancestors 'none'` and no inline script or style, plus `no-referrer`,
  `nosniff`, `X-Frame-Options: DENY`, `Permissions-Policy: microphone=(self)`. Everything is same-origin, so it works
  over the tunnel's https with no mixed content. The agent name is HTML-escaped; captions are written as text.
* Visitors get what the website widget's visitors get: the transcript, sentence text and clips, the final text and
  language. No receipts, timings, models, retrieval, tool calls, agent id or owner id.
* The turn runs the owner's streaming-turn pipeline as the agent's owner, so the agent's language rules, documents, tools
  (including booking), AI disclosure, watermark and receipts behave exactly as in the Playground. Conversations are
  stored with `source = "widget"` and show up under Calls as website conversations. If the agent can book meetings or
  call tools, a visitor can trigger them, as with the widget; share the link accordingly.
* Links are stored in `backend/data/web_call_links.json` (mode 0600): ids, agent id, limits and counters only.

## Test it

1. Start the app as usual; set `PUBLIC_BASE_URL` to your tunnel address (or leave it and open the `localhost` link on the
   same computer; the public port is `PHONE_PUBLIC_PORT`, default 8010).
2. Create a link on an agent, open it in a browser, press **Start talking**, allow the microphone and speak.
3. Without a microphone, Chrome with `--use-fake-device-for-media-stream --use-fake-ui-for-media-stream
   --use-file-for-fake-audio-capture=question.wav`.

Tests: `backend/tests/test_web_call.py` (owner routes, the page and its headers, sessions, turns, clips, limits,
revocation and expiry, cross-link and cross-widget refusal, the public app exposing nothing else),
`frontend/src/components/__tests__/WebCallPanel.test.tsx`, `frontend/src/lib/__tests__/webCallBundle.test.ts` (parser,
clip queue, voice detection compared with `src/lib/vad.ts`, the page driven with a fake microphone).
