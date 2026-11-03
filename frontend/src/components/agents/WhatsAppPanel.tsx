"use client";

import { useCallback, useEffect, useState } from "react";
import { Check, ChevronDown, Copy, MessageCircle, PlugZap, RefreshCw, Send, UserRound } from "lucide-react";
import { toast } from "sonner";
import { Button, Card, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";

/** The WhatsApp settings of one agent (GET/PUT /api/agents/{id}/whatsapp). The access token and app secret are
 * write-only: the API only says whether they are stored and shows the last four characters of the token. */
export interface WhatsAppSettings {
  enabled: boolean;
  phone_number_id: string;
  handoff_text: string;
  voice_for_text: boolean;
  verify_token: string;
  access_token_set: boolean;
  access_token_hint: string;
  app_secret_set: boolean;
  webhook_url: string;
  webhook_path: string;
  ready: boolean;
  secrets_storage: { available: boolean; reason: string | null };
  default_handoff_text: string;
}

export interface SimulatedMessage {
  type: "audio" | "text";
  body?: string;
  mime?: string;
  bytes?: number;
  audio_url?: string;
}
export interface Simulation {
  dry_run: boolean;
  handoff: boolean;
  agent_text: string;
  language?: string;
  messages: SimulatedMessage[];
}
export interface Handoff {
  conversation_id: string;
  contact: string;
  requested_at: string | null;
}

const field =
  "w-full rounded-lg border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent-primary";
const label = "mb-1.5 block text-sm font-medium text-text-secondary";

/** The address to paste into Meta: the tunnel/public address typed by the owner, else the one the server reports. */
export const webhookAddress = (publicBase: string, cfg: Pick<WhatsAppSettings, "webhook_url" | "webhook_path">) => {
  const base = publicBase.trim().replace(/\/+$/, "");
  return /^https?:\/\/[^\s/]+$/.test(base) ? base + cfg.webhook_path : cfg.webhook_url;
};

function CopyField({ id, text, ariaLabel }: { id: string; text: string; ariaLabel: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Could not copy. Select the text and copy it by hand.");
    }
  };
  return (
    <div className="flex gap-2">
      <input id={id} readOnly value={text} aria-label={ariaLabel} className={`${field} font-mono text-xs`} onFocus={(e) => e.target.select()} />
      <Button variant="outline" size="sm" onClick={copy} leftIcon={copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}>
        {copied ? "Copied" : "Copy"}
      </Button>
    </div>
  );
}

/** "WhatsApp voice notes": connect the agent to a WhatsApp Business number (Meta's Cloud API). */
export function WhatsAppPanel({ agentId }: { agentId: string }) {
  const [cfg, setCfg] = useState<WhatsAppSettings | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [phoneId, setPhoneId] = useState("");
  const [token, setToken] = useState("");
  const [secret, setSecret] = useState("");
  const [handoff, setHandoff] = useState("");
  const [voiceForText, setVoiceForText] = useState(true);
  const [publicBase, setPublicBase] = useState("");
  const [busy, setBusy] = useState(false);
  const [check, setCheck] = useState<string | null>(null);
  const [simText, setSimText] = useState("");
  const [simBusy, setSimBusy] = useState(false);
  const [sim, setSim] = useState<Simulation | null>(null);
  const [handoffs, setHandoffs] = useState<Handoff[]>([]);
  const path = `/api/agents/${agentId}/whatsapp`;

  const apply = useCallback((s: WhatsAppSettings) => {
    setCfg(s);
    setPhoneId(s.phone_number_id);
    setHandoff(s.handoff_text);
    setVoiceForText(s.voice_for_text);
    setToken("");
    setSecret("");
  }, []);

  const loadHandoffs = useCallback(() => {
    api
      .get<{ handoffs: Handoff[] }>(`${path}/handoffs`)
      .then((r) => setHandoffs(r.handoffs ?? []))
      .catch(() => setHandoffs([]));
  }, [path]);

  useEffect(() => {
    let alive = true;
    api
      .get<WhatsAppSettings>(path)
      .then((s) => alive && apply(s))
      .catch((e) => alive && setLoadError(errMessage(e, "Could not load the WhatsApp settings.")));
    loadHandoffs();
    return () => {
      alive = false;
    };
  }, [path, apply, loadHandoffs]);

  const save = async (extra: { enabled?: boolean } = {}, done?: string) => {
    setBusy(true);
    try {
      const body: Record<string, unknown> = {
        phone_number_id: phoneId.trim(),
        handoff_text: handoff.trim(),
        voice_for_text: voiceForText,
        ...extra,
      };
      if (token.trim()) body.access_token = token.trim();
      if (secret.trim()) body.app_secret = secret.trim();
      apply(await api.put<WhatsAppSettings>(path, body));
      if (done) toast.success(done);
    } catch (e) {
      toast.error(errMessage(e, "Could not save the WhatsApp settings."));
    } finally {
      setBusy(false);
    }
  };

  const toggle = (next: boolean) =>
    save({ enabled: next }, next ? "WhatsApp is on. Add the webhook address in Meta's dashboard." : "WhatsApp is off. The agent no longer answers there.");

  const newVerifyToken = async () => {
    try {
      apply(await api.post<WhatsAppSettings>(`${path}/verify-token`));
      toast.success("New verify token. Paste it into Meta's webhook settings.");
    } catch (e) {
      toast.error(errMessage(e, "Could not create a new verify token."));
    }
  };

  const checkConnection = async () => {
    setCheck(null);
    try {
      const r = await api.post<{ ok: boolean; display_phone_number?: string; verified_name?: string; error?: string }>(`${path}/check`);
      setCheck(r.ok ? `Connected: ${r.verified_name || "WhatsApp number"} ${r.display_phone_number ?? ""}`.trim() : (r.error ?? "Meta did not accept these settings."));
    } catch (e) {
      setCheck(errMessage(e, "Could not check the connection."));
    }
  };

  const simulate = async () => {
    if (!simText.trim()) return;
    setSimBusy(true);
    try {
      setSim(await api.post<Simulation>(`${path}/simulate`, { text: simText.trim() }));
      loadHandoffs();
    } catch (e) {
      toast.error(errMessage(e, "The simulation failed."));
    } finally {
      setSimBusy(false);
    }
  };

  const resolve = async (id: string) => {
    try {
      await api.post(`${path}/handoffs/${id}/resolve`);
      setHandoffs((h) => h.filter((x) => x.conversation_id !== id));
    } catch (e) {
      toast.error(errMessage(e, "Could not mark it followed up."));
    }
  };

  return (
    <Card className="space-y-5" aria-labelledby="wa-title">
      <div className="flex items-start justify-between gap-4 border-b border-border pb-4">
        <div>
          <h2 id="wa-title" className="flex items-center gap-2 font-display text-lg font-semibold text-text-primary">
            <MessageCircle className="size-4 text-accent-primary" aria-hidden />
            WhatsApp voice notes
          </h2>
          <p className="mt-1 text-sm text-text-secondary">
            Customers message your WhatsApp Business number with a voice note or text. The agent answers with an
            AI-generated voice note and the same text, in their language, from your documents. You need a WhatsApp
            Business account from Meta first.
          </p>
        </div>
        {cfg && (
          <label className="flex shrink-0 cursor-pointer items-center gap-2 text-sm font-medium text-text-primary">
            <input
              type="checkbox"
              role="switch"
              aria-label="Answer on WhatsApp"
              checked={cfg.enabled}
              disabled={busy}
              onChange={(e) => toggle(e.target.checked)}
              className="size-4 accent-[var(--color-accent-primary)]"
            />
            Answer on WhatsApp
          </label>
        )}
      </div>

      {loadError && (
        <p role="alert" className="text-sm text-accent-error">
          {loadError}
        </p>
      )}
      {!cfg && !loadError && <Skeleton className="h-24 w-full" />}

      {cfg && (
        <>
          {!cfg.secrets_storage.available && (
            <p role="alert" className="text-sm text-accent-error">
              {cfg.secrets_storage.reason}
            </p>
          )}
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label htmlFor="wa-phone" className={label}>
                Phone number ID
              </label>
              <input
                id="wa-phone"
                className={`${field} font-mono`}
                value={phoneId}
                inputMode="numeric"
                autoComplete="off"
                onChange={(e) => setPhoneId(e.target.value)}
                placeholder="From Meta: WhatsApp > API setup"
              />
            </div>
            <div>
              <label htmlFor="wa-token" className={label}>
                Access token
              </label>
              <input
                id="wa-token"
                type="password"
                autoComplete="off"
                className={`${field} font-mono`}
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder={cfg.access_token_set ? `Saved ${cfg.access_token_hint}. Paste to replace.` : "Paste the access token"}
              />
            </div>
            <div>
              <label htmlFor="wa-secret" className={label}>
                App secret
              </label>
              <input
                id="wa-secret"
                type="password"
                autoComplete="off"
                className={`${field} font-mono`}
                value={secret}
                onChange={(e) => setSecret(e.target.value)}
                placeholder={cfg.app_secret_set ? "Saved. Paste to replace." : "Meta app settings > Basic"}
                aria-describedby="wa-secret-hint"
              />
              <p id="wa-secret-hint" className="mt-1 text-xs text-text-secondary">
                Used to check that each message really came from Meta. The token and the secret are stored on the
                server only and are never shown again.
              </p>
            </div>
            <div>
              <label htmlFor="wa-handoff" className={label}>
                Message when someone asks for a person
              </label>
              <textarea
                id="wa-handoff"
                rows={3}
                maxLength={600}
                className={`${field} resize-none`}
                value={handoff}
                onChange={(e) => setHandoff(e.target.value)}
                placeholder={cfg.default_handoff_text}
              />
            </div>
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-sm text-text-primary">
            <input
              type="checkbox"
              checked={voiceForText}
              onChange={(e) => setVoiceForText(e.target.checked)}
              className="size-4 accent-[var(--color-accent-primary)]"
            />
            Also reply to typed messages with a voice note
          </label>
          <div className="flex flex-wrap items-center justify-end gap-2">
            {check && (
              <p role="status" className="mr-auto text-sm text-text-secondary">
                {check}
              </p>
            )}
            <Button variant="ghost" size="sm" onClick={checkConnection} disabled={!cfg.ready} leftIcon={<PlugZap className="size-3.5" />}>
              Check connection
            </Button>
            <Button variant="outline" size="sm" loading={busy} onClick={() => save({}, "WhatsApp settings saved.")}>
              Save settings
            </Button>
          </div>

          <details className="group space-y-3 border-t border-border pt-4">
            <summary className="flex min-h-10 cursor-pointer list-none items-center gap-1.5 text-sm font-medium text-text-primary">
              <ChevronDown className="size-4 transition-transform group-open:rotate-180" aria-hidden /> Details for your IT person: webhook address
            </summary>
            <p className="text-sm font-medium text-text-primary">Paste these into Meta (WhatsApp &gt; Configuration &gt; Webhook)</p>
            <div>
              <label htmlFor="wa-public" className={label}>
                Public address of this server (your tunnel, for example from cloudflared)
              </label>
              <input
                id="wa-public"
                className={`${field} font-mono`}
                value={publicBase}
                onChange={(e) => setPublicBase(e.target.value)}
                placeholder="https://your-tunnel.trycloudflare.com"
              />
            </div>
            <div>
              <label htmlFor="wa-url" className={label}>
                Webhook URL
              </label>
              <CopyField id="wa-url" text={webhookAddress(publicBase, cfg)} ariaLabel="Webhook URL" />
            </div>
            <div>
              <label htmlFor="wa-verify" className={label}>
                Verify token
              </label>
              <div className="flex gap-2">
                <div className="grow">
                  <CopyField id="wa-verify" text={cfg.verify_token} ariaLabel="Verify token" />
                </div>
                <Button variant="ghost" size="sm" onClick={newVerifyToken} leftIcon={<RefreshCw className="size-3.5" />}>
                  New token
                </Button>
              </div>
            </div>
            <p className="text-xs text-text-secondary">
              Then subscribe to the <span className="font-mono">messages</span> field. The agent only replies to people
              who write first (WhatsApp&apos;s 24-hour window) and says it is an AI at the start of a conversation.
            </p>
          </details>

          <div className="space-y-3 border-t border-border pt-4">
            <p className="text-sm font-medium text-text-primary">Simulate a message</p>
            <p className="text-xs text-text-secondary">
              Runs a typed message through the same flow without calling Meta, and shows what would be sent.
            </p>
            <div className="flex gap-2">
              <input
                aria-label="Message to simulate"
                className={field}
                value={simText}
                onChange={(e) => setSimText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && simulate()}
                placeholder="What is your refund policy?"
              />
              <Button variant="outline" size="sm" loading={simBusy} onClick={simulate} leftIcon={<Send className="size-3.5" />}>
                Simulate a message
              </Button>
            </div>
            {sim && (
              <ul className="space-y-2" aria-label="Simulated replies">
                {sim.messages.map((m, i) => (
                  <li key={i} className="rounded-lg border border-border bg-bg-elevated p-3 text-sm text-text-primary">
                    {m.type === "audio" ? (
                      <div className="space-y-1">
                        <p className="text-xs text-text-secondary">
                          Voice note ({Math.max(1, Math.round((m.bytes ?? 0) / 1024))} KB, OGG/Opus)
                        </p>
                        {m.audio_url && <audio controls src={m.audio_url} aria-label="Voice note preview" className="w-full" />}
                      </div>
                    ) : (
                      <p>{m.body}</p>
                    )}
                  </li>
                ))}
                {sim.handoff && <li className="text-xs text-text-secondary">Asked for a person: the conversation would be flagged for follow-up.</li>}
              </ul>
            )}
          </div>

          {handoffs.length > 0 && (
            <div className="space-y-2 border-t border-border pt-4">
              <p className="flex items-center gap-2 text-sm font-medium text-text-primary">
                <UserRound className="size-4 text-accent-primary" aria-hidden />
                Waiting for a person ({handoffs.length})
              </p>
              <ul className="space-y-2">
                {handoffs.map((h) => (
                  <li key={h.conversation_id} className="flex items-center justify-between gap-2 rounded-lg border border-border p-3 text-sm">
                    <span className="font-mono text-text-primary">+{h.contact}</span>
                    <span className="text-xs text-text-secondary">{h.requested_at ? new Date(h.requested_at).toLocaleString() : ""}</span>
                    <Button variant="ghost" size="sm" onClick={() => resolve(h.conversation_id)} aria-label={`Mark +${h.contact} followed up`}>
                      Mark followed up
                    </Button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </Card>
  );
}
