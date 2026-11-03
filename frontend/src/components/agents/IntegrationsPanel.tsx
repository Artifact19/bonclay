"use client";

import { useCallback, useState } from "react";
import { ChevronDown, KeyRound, RefreshCw, Send } from "lucide-react";
import { toast } from "sonner";
import { Badge, Button, Card } from "@/components/ui";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";

export interface WebhookSettings {
  enabled: boolean;
  url: string;
  url_host?: string;
  include_test: boolean;
  redact_phones: boolean;
  events: string[];
  secret_set: boolean;
  secret_created_at: string | null;
}
export interface SheetsSettings {
  enabled: boolean;
  spreadsheet_id: string;
  sheet_name: string;
  include_test: boolean;
  redact_phones: boolean;
  service_account_email: string;
  credentials_set: boolean;
}
/** One row per conversation with what the caller told the agent (backend/services/caller_sheet.py). */
export interface CallerSheetSettings {
  enabled: boolean;
  spreadsheet_id: string;
  sheet_name: string;
  include_phone: boolean;
  include_test: boolean;
  /** The e-mail of the service account to share the sheet with (the key itself is never sent). Empty: no key yet. */
  share_with: string;
  key_source: "agent" | "server" | null;
  last_write: { status: "sent" | "failed"; error: string | null; at: string; test?: boolean } | null;
  columns: string[];
}
export interface IntegrationsView {
  webhook: WebhookSettings;
  sheets: SheetsSettings;
  caller_sheet: CallerSheetSettings;
  privacy_note: string;
  secrets_storage: { available: boolean; reason: string | null };
  local_webhooks_allowed: boolean;
  sheet_columns: string[];
  /** Present only in the response that created or rotated the signing secret. */
  secret?: string;
}
export interface Delivery {
  delivery_id: string;
  target: "webhook" | "sheets" | "caller_sheet";
  event: string;
  status: "delivered" | "retrying" | "failed";
  code: number | null;
  latency_ms: number | null;
  error: string | null;
  attempts: number;
  at: string;
}

// Same look as the editor's fields (not imported from AgentEditor, which renders this panel).
const fieldClass =
  "w-full rounded-lg border border-border bg-bg-primary px-3 py-2.5 text-sm text-text-primary focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent-primary";
const labelClass = "mb-1.5 block text-sm font-medium text-text-secondary";

const EVENT_LABELS: Record<string, string> = {
  "turn.completed": "After every finished turn",
  "tool.called": "After every successful tool call (such as a booking)",
};
const STATUS_TONE = { delivered: "success", retrying: "warning", failed: "error" } as const;
const boxClass =
  "mt-0.5 size-4 shrink-0 accent-[var(--color-accent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary";

function Check({ id, checked, onChange, title, hint }: { id: string; checked: boolean; onChange: (v: boolean) => void; title: string; hint?: string }) {
  return (
    <label htmlFor={id} className="flex cursor-pointer items-start gap-3">
      <input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className={boxClass} />
      <span>
        <span className="block text-sm font-medium text-text-primary">{title}</span>
        {hint && <span className="mt-0.5 block text-xs text-text-secondary">{hint}</span>}
      </span>
    </label>
  );
}

/** Webhook and Google Sheets settings of one agent. Secrets are write-only: the server never sends them back (the
 *  signing secret is shown once, right after it is created or rotated). Loads when the section is opened. */
export function IntegrationsPanel({ agentId }: { agentId: string }) {
  const base = `/api/agents/${agentId}/integrations`;
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<IntegrationsView | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [deliveries, setDeliveries] = useState<Delivery[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [secret, setSecret] = useState<string | null>(null);

  const [url, setUrl] = useState("");
  const [whEnabled, setWhEnabled] = useState(false);
  const [events, setEvents] = useState<string[]>([]);
  const [whTest, setWhTest] = useState(false);
  const [whRedact, setWhRedact] = useState(false);

  const [sheet, setSheet] = useState("");
  const [tab, setTab] = useState("");
  const [key, setKey] = useState("");
  const [shEnabled, setShEnabled] = useState(false);
  const [shTest, setShTest] = useState(false);
  const [shRedact, setShRedact] = useState(false);
  const [sheetsResult, setSheetsResult] = useState<{ ok: boolean; error: string | null; hint: string } | null>(null);

  const [csSheet, setCsSheet] = useState("");
  const [csTab, setCsTab] = useState("");
  const [csEnabled, setCsEnabled] = useState(false);
  const [csPhone, setCsPhone] = useState(true);
  const [csTest, setCsTest] = useState(false);
  const [csResult, setCsResult] = useState<{ ok: boolean; error: string | null; hint: string; row: number | null } | null>(null);

  const apply = useCallback((v: IntegrationsView) => {
    setView(v);
    setUrl(v.webhook.url);
    setWhEnabled(v.webhook.enabled);
    setEvents(v.webhook.events);
    setWhTest(v.webhook.include_test);
    setWhRedact(v.webhook.redact_phones);
    setSheet(v.sheets.spreadsheet_id ? `https://docs.google.com/spreadsheets/d/${v.sheets.spreadsheet_id}` : "");
    setTab(v.sheets.sheet_name);
    setShEnabled(v.sheets.enabled);
    setShTest(v.sheets.include_test);
    setShRedact(v.sheets.redact_phones);
    const cs = v.caller_sheet;
    if (cs) {
      setCsSheet(cs.spreadsheet_id ? `https://docs.google.com/spreadsheets/d/${cs.spreadsheet_id}` : "");
      setCsTab(cs.sheet_name);
      setCsEnabled(cs.enabled);
      setCsPhone(cs.include_phone);
      setCsTest(cs.include_test);
    }
    setKey("");
    if (v.secret) setSecret(v.secret);
  }, []);

  const refreshLog = useCallback(async () => {
    try {
      const r = await api.get<{ deliveries: Delivery[] }>(`${base}/deliveries`);
      setDeliveries(r.deliveries);
    } catch {
      /* the log is optional */
    }
  }, [base]);

  const load = useCallback(async () => {
    setLoadError(null);
    try {
      apply(await api.get<IntegrationsView>(base));
      await refreshLog();
    } catch (e) {
      setLoadError(errMessage(e, "Could not load the integrations."));
    }
  }, [apply, base, refreshLog]);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (next && !view) void load();
  };

  const run = async (name: string, fn: () => Promise<void>) => {
    setBusy(name);
    try {
      await fn();
    } catch (e) {
      toast.error(errMessage(e, "That did not work. Try again."));
    } finally {
      setBusy(null);
    }
  };

  const saveWebhook = () =>
    run("webhook", async () => {
      apply(await api.put<IntegrationsView>(`${base}/webhook`, { url, enabled: whEnabled && !!url.trim(), events, include_test: whTest, redact_phones: whRedact }));
      toast.success("Webhook saved.");
    });
  const rotate = () =>
    run("rotate", async () => {
      apply(await api.post<IntegrationsView>(`${base}/webhook/rotate-secret`));
      toast.success("New signing secret created. Copy it now.");
    });
  const sendTest = () =>
    run("test-webhook", async () => {
      const r = await api.post<{ ok: boolean; delivery: Delivery }>(`${base}/webhook/test`);
      if (r.ok) toast.success(`Test event delivered (HTTP ${r.delivery.code}).`);
      else toast.error(`Test event failed: ${r.delivery.error ?? "no answer"}`);
      await refreshLog();
    });
  const saveSheets = () =>
    run("sheets", async () => {
      const body: Record<string, unknown> = { spreadsheet: sheet, sheet_name: tab, enabled: shEnabled, include_test: shTest, redact_phones: shRedact };
      if (key.trim()) body.service_account_json = key;
      apply(await api.put<IntegrationsView>(`${base}/sheets`, body));
      toast.success("Google Sheets settings saved.");
    });
  const saveCallerSheet = () =>
    run("caller-sheet", async () => {
      apply(
        await api.put<IntegrationsView>(`${base}/caller-sheet`, {
          spreadsheet: csSheet,
          sheet_name: csTab,
          enabled: csEnabled && !!csSheet.trim(),
          include_phone: csPhone,
          include_test: csTest,
        })
      );
      toast.success("Caller details settings saved.");
    });
  const testCallerSheet = () =>
    run("test-caller-sheet", async () => {
      setCsResult(await api.post(`${base}/caller-sheet/test`));
      await load();
    });
  const testSheets = () =>
    run("test-sheets", async () => {
      setSheetsResult(await api.post(`${base}/sheets/test`));
      await refreshLog();
    });

  const s = view?.secrets_storage;
  return (
    <Card className="space-y-5" data-testid="integrations-panel">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        aria-controls="agent-integrations-body"
        className="flex w-full items-center justify-between gap-3 rounded text-left focus-visible:outline-2 focus-visible:outline-accent-primary"
      >
        <span>
          <span className="block font-display text-lg font-semibold text-text-primary">Integrations</span>
          <span className="mt-1 block text-sm text-text-secondary">Send a copy of finished calls to your own systems, such as a Google Sheet or a webhook. Optional.</span>
        </span>
        <ChevronDown className={`size-5 shrink-0 text-text-secondary transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
      </button>

      {open && (
        <div id="agent-integrations-body" className="space-y-6">
          <p role="note" className="rounded-lg border border-border bg-bg-elevated px-3 py-2 text-xs text-text-secondary">
            {view?.privacy_note ??
              "These integrations send what callers say, and what the agent answers, to systems you choose. Turn on Hide phone numbers to replace phone numbers with [phone] before anything is sent."}
          </p>
          {loadError && <p role="alert" className="text-sm text-accent-error">{loadError}</p>}
          {!view && !loadError && <p className="text-sm text-text-secondary">Loading...</p>}
          {view && s && !s.available && <p role="alert" className="text-sm text-accent-warning">{s.reason}</p>}

          {view && (
            <>
              <section aria-labelledby="int-webhook" className="space-y-4">
                <h3 id="int-webhook" className="text-sm font-semibold text-text-primary">Webhook (send to your own system)</h3>
                <div>
                  <label htmlFor="int-webhook-url" className={labelClass}>Webhook URL</label>
                  <input
                    id="int-webhook-url"
                    className={fieldClass}
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://hooks.example.com/bonclay"
                    inputMode="url"
                    autoComplete="off"
                  />
                  <p className="mt-1.5 text-xs text-text-secondary">
                    https only{view.local_webhooks_allowed ? " (http://localhost is allowed here for testing)" : ""}. Addresses inside private networks are blocked and redirects are not followed.
                  </p>
                </div>
                <Check id="int-webhook-on" checked={whEnabled} onChange={setWhEnabled} title="Send events to this URL" />
                <fieldset className="space-y-2">
                  <legend className={labelClass}>Events</legend>
                  {Object.keys(EVENT_LABELS).map((ev) => (
                    <Check
                      key={ev}
                      id={`int-event-${ev}`}
                      checked={events.includes(ev)}
                      onChange={(on) => setEvents((prev) => (on ? [...prev, ev] : prev.filter((x) => x !== ev)))}
                      title={EVENT_LABELS[ev]}
                    />
                  ))}
                </fieldset>
                <Check id="int-webhook-test" checked={whTest} onChange={setWhTest} title="Include test calls" hint="Test calls from the editor are skipped unless this is on." />
                <Check id="int-webhook-redact" checked={whRedact} onChange={setWhRedact} title="Hide phone numbers" hint="Phone numbers in the caller and agent text, tool arguments and bookings become [phone]." />

                {secret && (
                  <div role="status" className="rounded-lg border border-accent-warning/40 bg-bg-elevated p-3" data-testid="signing-secret">
                    <p className="text-sm font-medium text-text-primary">Signing secret (shown once)</p>
                    <code className="mt-1 block break-all font-mono text-xs text-text-primary">{secret}</code>
                    <p className="mt-1 text-xs text-text-secondary">Copy it into your receiver now. It is not shown again; rotate it to get a new one.</p>
                    <Button variant="ghost" size="sm" onClick={() => setSecret(null)} className="mt-2">I copied it</Button>
                  </div>
                )}
                {view.webhook.secret_set && !secret && (
                  <p className="text-xs text-text-secondary">A signing secret is set{view.webhook.secret_created_at ? ` (created ${view.webhook.secret_created_at.slice(0, 10)})` : ""}. Each request carries X-BonClay-Signature, a timestamp and a delivery id.</p>
                )}

                <div className="flex flex-wrap gap-2">
                  <Button onClick={saveWebhook} loading={busy === "webhook"} size="sm">Save webhook</Button>
                  <Button variant="outline" size="sm" onClick={sendTest} loading={busy === "test-webhook"} disabled={!view.webhook.secret_set} leftIcon={<Send className="size-3.5" />}>Send test event</Button>
                  <Button variant="outline" size="sm" onClick={rotate} loading={busy === "rotate"} disabled={!view.webhook.secret_set} leftIcon={<KeyRound className="size-3.5" />}>Rotate secret</Button>
                </div>
              </section>

              {view.caller_sheet && (
                <section aria-labelledby="int-caller-sheet" className="space-y-4 border-t border-border pt-5" data-testid="caller-sheet-card">
                  <h3 id="int-caller-sheet" className="text-sm font-semibold text-text-primary">Caller details in a Google Sheet</h3>
                  <p className="text-xs text-text-secondary">
                    One row for every conversation: {view.caller_sheet.columns.join(", ")}. It is added when a call ends (a chat, after five quiet minutes) and updated if more details arrive. Bookings and requests then show up in your spreadsheet.
                  </p>
                  {view.caller_sheet.share_with ? (
                    <p className="text-sm text-text-primary" data-testid="share-with">
                      Share the sheet with <code className="break-all font-mono text-xs">{view.caller_sheet.share_with}</code> as <strong>Editor</strong>.
                    </p>
                  ) : (
                    <p role="note" className="text-sm text-accent-warning">
                      No Google service-account key is set up yet. Ask whoever runs the server to set GOOGLE_SERVICE_ACCOUNT_FILE, or paste a key under Google Sheets below.
                    </p>
                  )}
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <label htmlFor="int-cs-url" className={labelClass}>Sheet for caller details (link or id)</label>
                      <input id="int-cs-url" className={fieldClass} value={csSheet} onChange={(e) => setCsSheet(e.target.value)} placeholder="https://docs.google.com/spreadsheets/d/..." autoComplete="off" />
                    </div>
                    <div>
                      <label htmlFor="int-cs-tab" className={labelClass}>Tab name</label>
                      <input id="int-cs-tab" className={fieldClass} value={csTab} onChange={(e) => setCsTab(e.target.value)} placeholder="BonClay" autoComplete="off" />
                      <p className="mt-1.5 text-xs text-text-secondary">Created for you if the sheet does not have it. Empty means BonClay.</p>
                    </div>
                  </div>
                  <Check id="int-cs-on" checked={csEnabled} onChange={setCsEnabled} title="Send caller details to this sheet" />
                  <Check id="int-cs-phone" checked={csPhone} onChange={setCsPhone} title="Include phone numbers" hint="Off: the number the caller gave is written as ******3210 and e-mail addresses are shortened." />
                  <Check id="int-cs-test" checked={csTest} onChange={setCsTest} title="Include practice calls" hint="Calls you make yourself from the editor are skipped unless this is on." />
                  <div className="flex flex-wrap gap-2">
                    <Button onClick={saveCallerSheet} loading={busy === "caller-sheet"} size="sm">Save caller sheet</Button>
                    <Button variant="outline" size="sm" onClick={testCallerSheet} loading={busy === "test-caller-sheet"} disabled={!view.caller_sheet.spreadsheet_id}>Test caller sheet</Button>
                  </div>
                  {csResult && (
                    <p role="status" data-testid="caller-sheet-result" className={`text-sm ${csResult.ok ? "text-accent-success" : "text-accent-error"}`}>
                      {csResult.ok ? `Connected: a test row is in the sheet (row ${csResult.row}).` : csResult.error}
                      {!csResult.ok && csResult.hint ? ` ${csResult.hint}` : ""}
                    </p>
                  )}
                  {view.caller_sheet.last_write && (
                    <p data-testid="caller-sheet-last" className={`text-xs ${view.caller_sheet.last_write.status === "sent" ? "text-text-secondary" : "text-accent-error"}`}>
                      Last write{view.caller_sheet.last_write.test ? " (test)" : ""}: {view.caller_sheet.last_write.status === "sent" ? "sent" : `failed, ${view.caller_sheet.last_write.error ?? "no reason given"}`} · {view.caller_sheet.last_write.at.replace("T", " ").slice(0, 19)}
                    </p>
                  )}
                </section>
              )}

              <section aria-labelledby="int-sheets" className="space-y-4 border-t border-border pt-5">
                <h3 id="int-sheets" className="text-sm font-semibold text-text-primary">Google Sheets</h3>
                <div>
                  <label htmlFor="int-sheets-key" className={labelClass}>Service-account key (JSON)</label>
                  <textarea
                    id="int-sheets-key"
                    className={`${fieldClass} h-24 resize-y font-mono text-xs`}
                    value={key}
                    onChange={(e) => setKey(e.target.value)}
                    placeholder={view.sheets.credentials_set ? "A key is stored. Paste a new one to replace it." : '{"type": "service_account", ...}'}
                    spellCheck={false}
                    autoComplete="off"
                  />
                  <p className="mt-1.5 text-xs text-text-secondary">
                    Stored on the server only and never shown again.
                    {view.sheets.credentials_set && view.sheets.service_account_email ? ` Share your spreadsheet with ${view.sheets.service_account_email} (Editor).` : ""}
                  </p>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label htmlFor="int-sheets-url" className={labelClass}>Spreadsheet link or id</label>
                    <input id="int-sheets-url" className={fieldClass} value={sheet} onChange={(e) => setSheet(e.target.value)} placeholder="https://docs.google.com/spreadsheets/d/..." autoComplete="off" />
                  </div>
                  <div>
                    <label htmlFor="int-sheets-tab" className={labelClass}>Tab name (optional)</label>
                    <input id="int-sheets-tab" className={fieldClass} value={tab} onChange={(e) => setTab(e.target.value)} placeholder="First tab" autoComplete="off" />
                  </div>
                </div>
                <p className="text-xs text-text-secondary">One row per finished turn: {view.sheet_columns.join(", ")}.</p>
                <Check id="int-sheets-on" checked={shEnabled} onChange={setShEnabled} title="Add a row for every finished turn" />
                <Check id="int-sheets-test" checked={shTest} onChange={setShTest} title="Include test calls (Google Sheets)" />
                <Check id="int-sheets-redact" checked={shRedact} onChange={setShRedact} title="Hide phone numbers (Google Sheets)" />
                <div className="flex flex-wrap gap-2">
                  <Button onClick={saveSheets} loading={busy === "sheets"} size="sm">Save Google Sheets</Button>
                  <Button variant="outline" size="sm" onClick={testSheets} loading={busy === "test-sheets"} disabled={!view.sheets.credentials_set || !view.sheets.spreadsheet_id}>Test connection</Button>
                </div>
                {sheetsResult && (
                  <p role="status" data-testid="sheets-result" className={`text-sm ${sheetsResult.ok ? "text-accent-success" : "text-accent-error"}`}>
                    {sheetsResult.ok ? "Connected: a test row was added to the sheet." : sheetsResult.error}
                    {!sheetsResult.ok && sheetsResult.hint ? ` ${sheetsResult.hint}` : ""}
                  </p>
                )}
              </section>

              <section aria-labelledby="int-log" className="space-y-3 border-t border-border pt-5">
                <div className="flex items-center justify-between gap-3">
                  <h3 id="int-log" className="text-sm font-semibold text-text-primary">Delivery log</h3>
                  <Button variant="ghost" size="sm" onClick={() => void refreshLog()} leftIcon={<RefreshCw className="size-3.5" />}>Refresh</Button>
                </div>
                {deliveries.length === 0 ? (
                  <p className="text-sm text-text-secondary">No deliveries yet. Send a test event to try it.</p>
                ) : (
                  <ul className="divide-y divide-border rounded-lg border border-border text-xs" data-testid="delivery-log">
                    {deliveries.map((d) => (
                      <li key={d.delivery_id} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2">
                        <Badge tone={STATUS_TONE[d.status] ?? "neutral"}>{d.status}</Badge>
                        <span className="font-medium text-text-primary">{d.target === "sheets" ? "Sheets" : d.target === "caller_sheet" ? "Caller sheet" : "Webhook"} · {d.event}</span>
                        <span className="text-text-secondary">{d.code ? `HTTP ${d.code}` : "no response"}</span>
                        {d.latency_ms != null && <span className="text-text-secondary">{d.latency_ms} ms</span>}
                        {d.attempts > 1 && <span className="text-text-secondary">{d.attempts} tries</span>}
                        <span className="text-text-secondary">{d.at.replace("T", " ").slice(0, 19)}</span>
                        {d.error && <span className="w-full break-words text-accent-error">{d.error}</span>}
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
