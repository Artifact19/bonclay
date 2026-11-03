# Integrations: signed webhook and Google Sheets

Each agent can send what happened on a call to your own systems: a webhook, a Google Sheet of every turn, and a Google Sheet with one row of caller details per conversation. All are set up in **Agent Studio, open the agent, Integrations**.

**Privacy:** these integrations send what callers say, and what the agent answers, to systems you choose. Turn on **Redact phone numbers** to replace phone numbers (Indian mobile numbers, +country-code numbers, bare 10 to 15 digit runs, and the `phone` of a booking) with `[phone]` before anything leaves BonClay. Nothing else is redacted, so tell callers if your policy requires it. Calls from the editor's **Test call** are not sent unless you tick **Include test calls**.

Nothing here can slow a call down: events are queued and delivered by a background worker after the reply has been stored.

## Webhook

Set an `https://` URL (for local testing `http://localhost` or `http://127.0.0.1` is accepted when `BONCLAY_ALLOW_LOCAL_WEBHOOKS=on`, which is the default on a local single-user install). BonClay creates a **signing secret** (`whsec_...`) and shows it **once**; rotate it any time (the old one stops working at once).

### Events

| `X-BonClay-Event` | When |
| --- | --- |
| `turn.completed` | after every finished turn |
| `tool.called` | after every successful tool call of a turn (for example a booking); sent before `turn.completed` is queued |
| `test.ping` | the **Send test event** button |

You can choose which of the first two you want.

### Request

`POST` with `Content-Type: application/json` and these headers:

| Header | Value |
| --- | --- |
| `X-BonClay-Signature` | `sha256=<hex>`, the HMAC-SHA256 of `"<timestamp>." + raw body`, keyed with your signing secret |
| `X-BonClay-Timestamp` | Unix seconds of this attempt |
| `X-BonClay-Event` | the event name |
| `X-BonClay-Delivery` | a UUID, the same on every retry of one delivery (use it to de-duplicate) |

Body:

```json
{
  "event": "turn.completed",
  "agent_id": "…", "conversation_id": "…", "turn_id": "…",
  "created_at": "2026-10-19T10:31:02+00:00",
  "language": "en",
  "caller_text": "Book me for Monday at 10:30, I am Priya, 9876543210",
  "agent_text": "Done. Booking A10 with Dr. Meera Kulkarni on Monday at 10:30.",
  "tool_calls": [{"name": "book_appointment", "arguments": {"…": "…"}, "ok": true, "result": "booking A10 …"}],
  "booking": {"booking_id": "A10", "name": "Priya", "phone": "9876543210", "doctor": "Dr. Meera Kulkarni", "date": "2026-10-19", "time": "10:30", "status": "booked"},
  "receipt_id": "…",
  "is_test": false,
  "timings": {"asr_ms": 310, "llm_ms": 820, "tts_ms": 400, "total_ms": 1710}
}
```

`booking` is `null` when the turn made no booking. `receipt_id` is the provenance receipt of the spoken reply (check it at `/api/verify`).

### Verify the signature

Sign the **raw bytes** you received (not a re-serialised copy), and reject a timestamp older than five minutes: that stops someone who captured a request from replaying it later.

Python:

```python
import hashlib, hmac, time

def verify(secret: str, raw_body: bytes, timestamp: str, signature: str, tolerance=300) -> bool:
    if abs(time.time() - int(timestamp)) > tolerance:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

# Flask: verify(SECRET, request.get_data(), request.headers["X-BonClay-Timestamp"], request.headers["X-BonClay-Signature"])
```

Node (Express; use `express.raw({ type: "application/json" })` so `req.body` is the raw Buffer):

```js
const crypto = require("crypto");

function verify(secret, rawBody, timestamp, signature, tolerance = 300) {
  if (Math.abs(Date.now() / 1000 - Number(timestamp)) > tolerance) return false;
  const expected = "sha256=" + crypto.createHmac("sha256", secret)
    .update(`${timestamp}.`).update(rawBody).digest("hex");
  const a = Buffer.from(expected), b = Buffer.from(signature || "");
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}
// verify(SECRET, req.body, req.get("X-BonClay-Timestamp"), req.get("X-BonClay-Signature"))
```

### Receiving it in n8n, Make or Zapier

The simplest receiver is a **Webhook** trigger: copy its production URL into BonClay, click **Send test event**, then map the fields.

1. n8n: add a **Webhook** node (method POST, "Respond: Immediately"). Make: **Webhooks, Custom webhook**. Zapier: **Webhooks by Zapier, Catch Hook**.
2. Paste its URL (it must be `https://`) into **Webhook URL** in BonClay and save. Copy the signing secret if you want to verify.
3. Click **Send test event**. The trigger now has a sample with `caller_text`, `agent_text`, `booking.*`.
4. Add your action, for example: if `{{booking.booking_id}}` exists, create a calendar event or send a WhatsApp confirmation to `{{booking.phone}}`.
5. n8n only: to verify the signature, add a **Code** node that runs the Node snippet above on `$json.headers` and the raw body (enable "Raw Body" on the Webhook node).
6. Answer with any `2xx` within five seconds. Anything else is retried.

Hosted tools do not all expose the raw body; if yours does not, rely on the URL being secret and the delivery id, or put a tiny verifying proxy (the Python snippet above) in front of it.

### Delivery, retries and safety

- 5 second timeout. A network error, timeout, `408`, `425`, `429` or `5xx` is retried three times, after 2 s, 8 s and 30 s (four tries in all). Other `4xx` and redirects are not retried.
- The last 50 deliveries per agent (status, HTTP code, latency, error, tries) are listed under **Delivery log**. The log never contains the payload or your URL.
- **SSRF protection:** the URL's host is resolved on every attempt and every address must be public. Private ranges, loopback, link-local (including the cloud metadata address 169.254.169.254), CGNAT, multicast and unique-local IPv6 are refused, and the connection goes to the vetted address (no second lookup). Redirects are never followed and proxy environment variables are ignored. Loopback is only reachable for the literal names `localhost`, `127.x` and `::1`, and only when `BONCLAY_ALLOW_LOCAL_WEBHOOKS=on` (default: on in local mode, off in a Supabase deployment).
- The signing secret is stored on the server only: a `0600` file under `backend/data/integrations/` in local mode; in cloud mode it is encrypted (Fernet) with `BONCLAY_SECRETS_KEY`, and BonClay **refuses to store it** when that variable is not set. The API never returns it again after it was shown.

## Google Sheets

Each finished turn adds one row. Columns, in order:

`timestamp, agent, language, caller_text, agent_text, booking_id, booking_date, booking_time, receipt_id`

(Put those names in row 1 yourself if you want headers; BonClay only appends.) Values are written as plain text, so a caller cannot make a cell run a formula.

### Setup

1. In [Google Cloud Console](https://console.cloud.google.com/) create or pick a project and **enable the Google Sheets API** (APIs and services, Library).
2. **IAM and admin, Service accounts, Create service account.** No roles are needed. Open it, **Keys, Add key, Create new key, JSON**. A `.json` file downloads.
3. Open your Google Sheet, **Share**, and add the service account's e-mail (`something@your-project.iam.gserviceaccount.com`) as **Editor**.
4. In BonClay, Integrations, Google Sheets: paste the whole JSON file into **Service-account key**, paste the sheet's link into **Spreadsheet link or id**, optionally a **Tab name** (empty means the first tab), tick **Add a row for every finished turn**, **Save Google Sheets**.
5. Click **Test connection**. A marked test row is appended, or you see Google's exact error. The usual ones: `403 ... does not have permission` (the sheet is not shared with the service account's e-mail, shown in the hint), `404 ... not found` (wrong link or id), `400 Unable to parse range` (the tab name does not match), `403 ... has not been used ... or it is disabled` (enable the Sheets API, step 1).

The key is write-only: BonClay stores it on the server (a `0600` file locally; encrypted with `BONCLAY_SECRETS_KEY` in cloud mode, otherwise refused) and never sends it back to the browser. Rows are sent in batches, at most 50 requests a minute per service account (Google allows 60), and retried like webhooks. Failed rows show up in the same delivery log.

## Caller details in a Google Sheet (bookings and requests in a spreadsheet)

For a clinic or an office: every conversation becomes **one row** in your spreadsheet with what the caller told the agent, so you can read bookings and requests without opening the dashboard. This is separate from the per-turn **Google Sheets** log above (that one adds a row for every turn with the full text); you can use either or both.

Columns, in order: `Time (IST)`, `Agent`, `Channel` (phone, web, widget, WhatsApp), `Caller name`, `Phone`, `Date requested`, `Time requested`, `Reason`, `Other details` (doctor, age, e-mail and anything else the caller said), `Outcome` (Booked, Meeting requested, Meeting declined, Answered, Could not answer, Handed to a person, Ended early), `Call length`, `Conversation link` (opens that conversation on the Calls page), `Receipt id` and a last column, `Conversation id`, which BonClay uses to find the row again. Do not delete or edit that last column; you can add your own columns to the right of it and sort or filter freely.

### Set it up

1. **Create a Google Sheet** (or pick one). Leave it empty: BonClay creates the tab and the header row.
2. **Share it with the service account** as **Editor**. The e-mail to share with is shown in Agent Studio, open the agent, **Integrations, Caller details in a Google Sheet** (`something@your-project.iam.gserviceaccount.com`). It is the same service account Google Calendar uses (`GOOGLE_SERVICE_ACCOUNT_FILE` on the server), or the key you pasted under **Google Sheets** for this agent; that one wins when both exist. If the card says no key is set up, follow steps 1 and 2 of the Google Sheets setup above (enable the Sheets API, create a key) and put the file in `GOOGLE_SERVICE_ACCOUNT_FILE`.
3. **Paste the sheet's link** into **Sheet for caller details**, optionally change the **Tab name** (default `BonClay`; it is created when missing), and tick **Send caller details to this sheet**. Click **Save caller sheet**.
4. Click **Test caller sheet**. A marked row ("Test row from BonClay. You can delete it.") is added, or you see Google's exact error. Clicking it again updates the same row. The usual errors: `403 ... does not have permission` (not shared with that e-mail, step 2), `404 ... not found` (wrong link), `403 ... has not been used ... or it is disabled` (enable the Sheets API for the Google Cloud project of that service account).

**Include phone numbers** (on by default) writes the number the caller gave in full. Turn it off and it is written as `******3210`, and e-mail addresses as `p***@example.com`. **Include practice calls** adds the calls you make yourself from the editor (off by default).

### When a row is written

- **Phone call:** a few seconds after the call ends (the short wait lets the background reading of the caller's last words finish first).
- **Web chat, website widget, WhatsApp, the app's own chat:** once the conversation has been quiet for 5 minutes. A visitor who comes back and adds details updates the same row after the next quiet period.
- **Later changes:** if details are added after the row was written, the same row is updated. If nothing in the row changed, nothing is sent.

There is always one row per conversation. If you sort the sheet, move rows or insert rows above, BonClay still finds the row by its conversation id; if you delete a row, the next update adds a new one.

### Reliability and where to look

- Writes are queued: they never slow a call or a reply, and a Google outage never breaks a turn. A `429` or `5xx` is retried with the same backoff as webhooks (2 s, 8 s, 30 s; a `429` waits 30 s). Other errors (not shared, wrong link) are not retried; fix the cause and the next update of that conversation writes the row.
- The Calls page shows **Sent to Google Sheets ✓**, **Waiting to send to Google Sheets** or **Google Sheets failed: reason** on each conversation that went to a sheet. The delivery log in the Integrations section lists every write as `Caller sheet`, and the card shows how the last write went.
- Offline mode (`BONCLAY_OFFLINE=1`) refuses to write, like every other Google call, and says so.
- The status of every conversation (sent, waiting, failed, the row number) is kept in `backend/data/integrations/caller_sheet.ledger.json` (`0600`, git-ignored). A conversation that was still "quiet for 5 minutes" when the server restarted is not written (the server forgets it); the next turn of it queues it again.
- Values are written as plain text (`RAW`), so nothing a caller says can become a formula.

Cloud mode needs no new migration: the settings are saved in the agent's existing `integrations` column.

## Environment

| Variable | Meaning |
| --- | --- |
| `BONCLAY_ALLOW_LOCAL_WEBHOOKS` | `on` allows `http://localhost` webhook targets. Default: on without Supabase auth, off with it. |
| `BONCLAY_SECRETS_KEY` | Required in cloud mode to store the signing secret and the Google key (any long random string). |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | The service-account key file the caller-details sheet signs in with when the agent has no key of its own (the same file Google Calendar uses, default `backend/data/google_calendar.json`). |
| `CALLER_SHEET_IDLE_MIN` | Minutes of quiet after which a web, widget or WhatsApp conversation is written to the caller-details sheet. Default 5. |
| `CALLER_SHEET_END_DELAY_S` | Seconds to wait after a phone call ends before its row is written. Default 4. |
| `BONCLAY_DASHBOARD_URL` | The address of the dashboard, used for the `Conversation link` column. Default `http://localhost:3000`. |
| `BONCLAY_INTEGRATIONS_DIR` | Where local secrets and logs go (default `backend/data/integrations/`, git-ignored). |

Cloud mode also needs migration `backend/supabase/migrations/015_agent_integrations.sql`.

## API

All under `/api/agents/{id}/integrations` (the owner's normal auth): `GET` (settings, never a secret), `PUT /webhook`, `POST /webhook/rotate-secret`, `POST /webhook/test`, `DELETE /webhook`, `PUT /sheets`, `POST /sheets/test`, `DELETE /sheets`, `PUT /caller-sheet`, `POST /caller-sheet/test`, `GET /deliveries`. `GET /api/calls` and `GET /api/calls/{id}` carry `sheet` (`{state: sent | waiting | failed, error?}`) for a conversation that went to a caller-details sheet.
