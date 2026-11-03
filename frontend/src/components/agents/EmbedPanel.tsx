"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Copy, Globe, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button, Card, Skeleton } from "@/components/ui";
import { API_URL, api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";
import { ConfirmDialog } from "./ConfirmDialog";

/** The widget settings of one agent (GET/PUT /api/agents/{id}/widget). */
export interface WidgetSettings {
  enabled: boolean;
  key: string | null;
  origins: string[];
  per_visitor_hour: number;
  per_key_day: number;
  business_name: string;
  script_path: string;
}

const field =
  "w-full rounded-lg border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent-primary";
const label = "mb-1.5 block text-sm font-medium text-text-secondary";

export const embedSnippet = (key: string, scriptPath: string, apiUrl: string = API_URL) =>
  `<script src="${apiUrl.replace(/\/$/, "")}${scriptPath}" defer></script>\n<bonclay-agent agent="${key}"></bonclay-agent>`;

const parseOrigins = (text: string) =>
  text
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean);

/** "Put your agent on your website": publish the agent as a voice widget with a public, revocable key. */
export function EmbedPanel({ agentId }: { agentId: string }) {
  const [cfg, setCfg] = useState<WidgetSettings | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [origins, setOrigins] = useState("");
  const [business, setBusiness] = useState("");
  const [perVisitor, setPerVisitor] = useState("30");
  const [perDay, setPerDay] = useState("300");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [confirm, setConfirm] = useState<"rotate" | "remove" | null>(null);
  const originsRef = useRef<HTMLTextAreaElement>(null);
  const path = `/api/agents/${agentId}/widget`;

  const apply = useCallback((s: WidgetSettings) => {
    setCfg(s);
    setOrigins(s.origins.join("\n"));
    setBusiness(s.business_name);
    setPerVisitor(String(s.per_visitor_hour));
    setPerDay(String(s.per_key_day));
  }, []);

  useEffect(() => {
    let alive = true;
    api
      .get<WidgetSettings>(path)
      .then((s) => alive && apply(s))
      .catch((e) => alive && setLoadError(errMessage(e, "Could not load the website settings.")));
    return () => {
      alive = false;
    };
  }, [path, apply]);

  const draft = () => ({
    origins: parseOrigins(origins),
    business_name: business.trim(),
    per_visitor_hour: Number(perVisitor) || 30,
    per_key_day: Number(perDay) || 300,
  });

  const save = async (extra: { enabled?: boolean } = {}, done?: string) => {
    setBusy(true);
    try {
      apply(await api.put<WidgetSettings>(path, { ...draft(), ...extra }));
      if (done) toast.success(done);
    } catch (e) {
      const code = (e as { code?: string }).code;
      toast.error(errMessage(e, "Could not save the website settings."));
      if (code === "ORIGINS_REQUIRED" || code === "INVALID_ORIGIN") originsRef.current?.focus();
    } finally {
      setBusy(false);
    }
  };

  const toggle = (next: boolean) =>
    save({ enabled: next }, next ? "Published. Copy the code below into your site." : "Unpublished. The widget no longer answers.");

  const copy = async () => {
    if (!cfg?.key) return;
    try {
      await navigator.clipboard.writeText(embedSnippet(cfg.key, cfg.script_path));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Could not copy. Select the code and copy it by hand.");
    }
  };

  const rotate = async () => {
    try {
      apply(await api.post<WidgetSettings>(`${path}/rotate`));
      toast.success("New key created. Pages with the old code stop working.");
    } catch (e) {
      toast.error(errMessage(e, "Could not replace the key."));
    }
    setConfirm(null);
  };

  const remove = async () => {
    try {
      apply(await api.del<WidgetSettings>(path));
      toast.success("Widget removed.");
    } catch (e) {
      toast.error(errMessage(e, "Could not remove the widget."));
    }
    setConfirm(null);
  };

  return (
    <Card className="space-y-5" aria-labelledby="embed-title">
      <div className="flex items-start justify-between gap-4 border-b border-border pb-4">
        <div>
          <h2 id="embed-title" className="flex items-center gap-2 font-display text-lg font-semibold text-text-primary">
            <Globe className="size-4 text-accent-primary" aria-hidden />
            Embed on your website
          </h2>
          <p className="mt-1 text-sm text-text-secondary">
            Visitors to your website can talk to this agent with their microphone and hear AI-generated voice replies.
            Add your site, switch on Publish, then paste the code into your page.
          </p>
        </div>
        {cfg && (
          <label className="flex shrink-0 cursor-pointer items-center gap-2 text-sm font-medium text-text-primary">
            <input
              type="checkbox"
              role="switch"
              aria-label="Publish to website"
              checked={cfg.enabled}
              disabled={busy}
              onChange={(e) => toggle(e.target.checked)}
              className="size-4 accent-[var(--color-accent-primary)]"
            />
            Publish to website
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
          <div>
            <label htmlFor="embed-origins" className={label}>
              Allowed websites
            </label>
            <textarea
              id="embed-origins"
              ref={originsRef}
              rows={3}
              className={`${field} resize-none font-mono`}
              value={origins}
              onChange={(e) => setOrigins(e.target.value)}
              placeholder={"https://www.example.com\nhttps://shop.example.com"}
              aria-describedby="embed-origins-hint"
            />
            <p id="embed-origins-hint" className="mt-1 text-xs text-text-secondary">
              One address per line, exactly as visitors see it (https://example.com, no path or wildcard). The widget
              only works on these sites. Add http://localhost:3000 only while you test.
            </p>
          </div>

          <div>
            <label htmlFor="embed-business" className={label}>
              Business name
            </label>
            <input
              id="embed-business"
              className={field}
              value={business}
              maxLength={80}
              onChange={(e) => setBusiness(e.target.value)}
              placeholder="Shown in the privacy notice"
            />
          </div>

          <details className="group rounded-lg border border-border px-4 py-1">
            <summary className="flex min-h-10 cursor-pointer list-none items-center gap-1.5 text-sm font-medium text-text-secondary hover:text-text-primary">
              <ChevronDown className="size-4 transition-transform group-open:rotate-180" aria-hidden /> Details: limits per visitor
            </summary>
            <div className="grid gap-4 pb-3 pt-2 sm:grid-cols-2">
            <div>
              <label htmlFor="embed-visitor" className={label}>
                Turns per visitor per hour
              </label>
              <input
                id="embed-visitor"
                type="number"
                min={1}
                max={600}
                className={field}
                value={perVisitor}
                onChange={(e) => setPerVisitor(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="embed-day" className={label}>
                Turns per day, all visitors
              </label>
              <input
                id="embed-day"
                type="number"
                min={1}
                max={10000}
                className={field}
                value={perDay}
                onChange={(e) => setPerDay(e.target.value)}
              />
            </div>
            </div>
          </details>

          <div className="flex justify-end">
            <Button variant="outline" size="sm" loading={busy} onClick={() => save({}, "Website settings saved.")}>
              Save settings
            </Button>
          </div>

          {cfg.enabled && cfg.key ? (
            <div className="space-y-2 border-t border-border pt-4">
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm font-medium text-text-primary">Paste this before the closing &lt;/body&gt; tag</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={copy}
                  leftIcon={copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                >
                  {copied ? "Copied" : "Copy code"}
                </Button>
              </div>
              <pre
                tabIndex={0}
                aria-label="Embed code"
                className="overflow-x-auto rounded-lg border border-border bg-bg-elevated p-3 font-mono text-xs text-text-primary"
              >
                {embedSnippet(cfg.key, cfg.script_path)}
              </pre>
              <p className="text-xs text-text-secondary">
                Visitors see a notice that they are talking to an AI assistant before they start. Conversations are
                saved to this agent as website conversations.
              </p>
              <div className="flex flex-wrap justify-end gap-2 pt-1">
                <Button variant="ghost" size="sm" onClick={() => setConfirm("rotate")} leftIcon={<RefreshCw className="size-3.5" />}>
                  Replace key
                </Button>
                <Button variant="ghost" size="sm" onClick={() => setConfirm("remove")} leftIcon={<Trash2 className="size-3.5" />}>
                  Remove widget
                </Button>
              </div>
            </div>
          ) : (
            <p className="text-sm text-text-secondary">
              {cfg.key
                ? "Unpublished: the code on your site shows nothing until you publish again."
                : "Not published. Add your site above, then switch on Publish to website."}
            </p>
          )}
        </>
      )}

      <ConfirmDialog
        open={confirm === "rotate"}
        title="Replace the widget key?"
        description="Your site stops working until you paste the new code. Use this if the key was copied somewhere it should not be."
        confirmLabel="Replace key"
        keepLabel="Keep key"
        onCancel={() => setConfirm(null)}
        onConfirm={rotate}
      />
      <ConfirmDialog
        open={confirm === "remove"}
        title="Remove the widget?"
        description="The key is deleted and every site using it stops working. Past website conversations are kept."
        confirmLabel="Remove widget"
        keepLabel="Keep widget"
        onCancel={() => setConfirm(null)}
        onConfirm={remove}
      />
    </Card>
  );
}
