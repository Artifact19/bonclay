"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Check, Copy, Link2, Trash2, XCircle } from "lucide-react";
import { toast } from "sonner";
import { Badge, Button, Card, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";
import { ConfirmDialog } from "./ConfirmDialog";

export type WebCallExpiry = "1d" | "7d" | "never";
export type WebCallStatus = "active" | "expired" | "revoked";

/** One web call link of an agent (GET/POST /api/agents/{id}/web-call-links). */
export interface WebCallLink {
  id: string;
  url: string;
  label: string;
  status: WebCallStatus;
  created_at: string | null;
  expires_at: string | null;
  revoked_at: string | null;
  last_used_at: string | null;
  max_session_minutes: number;
  max_minutes_per_day: number;
  max_minutes_per_visitor: number | null;
  sessions: number;
  turns: number;
  minutes_today: number;
  minutes_total: number;
}

export interface WebCallLinks {
  links: WebCallLink[];
  base_url: string;
  /** The address is https (the tunnel), so people outside this computer can open it. */
  public: boolean;
  defaults: { expires_in: WebCallExpiry; max_session_minutes: number; max_minutes_per_day: number };
}

const field =
  "w-full rounded-lg border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent-primary";
const label = "mb-1.5 block text-sm font-medium text-text-secondary";

const dateText = (iso: string | null) =>
  iso ? new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "";

const STATUS_TONE: Record<WebCallStatus, "success" | "neutral" | "error"> = { active: "success", expired: "neutral", revoked: "error" };
const STATUS_TEXT: Record<WebCallStatus, string> = { active: "Active", expired: "Expired", revoked: "Revoked" };

export const expiryText = (l: Pick<WebCallLink, "status" | "expires_at" | "revoked_at">) =>
  l.status === "revoked"
    ? `Revoked ${dateText(l.revoked_at)}`.trim()
    : l.expires_at
      ? `${l.status === "expired" ? "Expired" : "Expires"} ${dateText(l.expires_at)}`
      : "Never expires";

/** "Web call link": a shareable page where anyone talks to this agent in their browser, no phone and no login. */
export function WebCallPanel({ agentId }: { agentId: string }) {
  const [data, setData] = useState<WebCallLinks | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [expires, setExpires] = useState<WebCallExpiry>("7d");
  const [session, setSession] = useState("10");
  const [perDay, setPerDay] = useState("60");
  const [perVisitor, setPerVisitor] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<{ kind: "revoke" | "remove"; link: WebCallLink } | null>(null);
  const path = `/api/agents/${agentId}/web-call-links`;

  const load = useCallback(async () => {
    try {
      const d = await api.get<WebCallLinks>(path);
      setData(d);
      setLoadError(null);
    } catch (e) {
      setLoadError(errMessage(e, "Could not load the web call links."));
    }
  }, [path]);

  useEffect(() => {
    let alive = true;
    api
      .get<WebCallLinks>(path)
      .then((d) => {
        if (!alive) return;
        setData(d);
        setExpires(d.defaults.expires_in);
        setSession(String(d.defaults.max_session_minutes));
        setPerDay(String(d.defaults.max_minutes_per_day));
      })
      .catch((e) => alive && setLoadError(errMessage(e, "Could not load the web call links.")));
    return () => {
      alive = false;
    };
  }, [path]);

  const create = async () => {
    setBusy(true);
    try {
      await api.post<WebCallLink>(path, {
        label: name.trim() || undefined,
        expires_in: expires,
        max_session_minutes: Number(session) || 10,
        max_minutes_per_day: Number(perDay) || 60,
        max_minutes_per_visitor: perVisitor.trim() ? Number(perVisitor) || null : null,
      });
      setName("");
      await load();
      toast.success("Link created. Copy it and share it.");
    } catch (e) {
      toast.error(errMessage(e, "Could not create the link."));
    } finally {
      setBusy(false);
    }
  };

  const copy = async (l: WebCallLink) => {
    try {
      await navigator.clipboard.writeText(l.url);
      setCopied(l.id);
      setTimeout(() => setCopied((c) => (c === l.id ? null : c)), 2000);
    } catch {
      toast.error("Could not copy. Select the link and copy it by hand.");
    }
  };

  const act = async () => {
    if (!confirm) return;
    const { kind, link } = confirm;
    try {
      if (kind === "revoke") {
        await api.post(`${path}/${link.id}/revoke`);
        toast.success("Link switched off. Anyone using it is cut off on their next turn.");
      } else {
        await api.del(`${path}/${link.id}`);
        toast.success("Link removed.");
      }
      await load();
    } catch (e) {
      toast.error(errMessage(e, kind === "revoke" ? "Could not revoke the link." : "Could not remove the link."));
    }
    setConfirm(null);
  };

  return (
    <Card className="space-y-5" aria-labelledby="webcall-title">
      <div className="border-b border-border pb-4">
        <h2 id="webcall-title" className="flex items-center gap-2 font-display text-lg font-semibold text-text-primary">
          <Link2 className="size-4 text-accent-primary" aria-hidden />
          Web call link
        </h2>
        <p className="mt-1 text-sm text-text-secondary">
          Share a link and anyone can talk to this agent by voice in their browser. No phone number, no sign-in. The
          link opens one agent&apos;s conversation and nothing else, and you can switch it off at any time.
        </p>
      </div>

      {loadError && (
        <p role="alert" className="text-sm text-accent-error">
          {loadError}
        </p>
      )}
      {!data && !loadError && <Skeleton className="h-24 w-full" />}

      {data && !data.public && (
        <p role="status" className="flex items-start gap-2 rounded-lg border border-border bg-bg-elevated px-3 py-2 text-sm text-text-secondary">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-accent-warning" aria-hidden />
          <span>
            These links only open on this computer. To share them, set <span className="font-mono">PUBLIC_BASE_URL</span> in
            backend/.env to your public https address (the tunnel) and restart.
          </span>
        </p>
      )}

      {data && (
        <>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="sm:col-span-2">
              <label htmlFor="webcall-label" className={label}>
                Note for yourself (optional)
              </label>
              <input
                id="webcall-label"
                className={field}
                value={name}
                maxLength={80}
                onChange={(e) => setName(e.target.value)}
                placeholder="For example: Clinic demo for Dr. Mehta"
              />
            </div>
            <div>
              <label htmlFor="webcall-expires" className={label}>
                Link expires
              </label>
              <select id="webcall-expires" className={field} value={expires} onChange={(e) => setExpires(e.target.value as WebCallExpiry)}>
                <option value="1d">After 1 day</option>
                <option value="7d">After 7 days</option>
                <option value="never">Never</option>
              </select>
            </div>
            <div>
              <label htmlFor="webcall-session" className={label}>
                Minutes per call
              </label>
              <input
                id="webcall-session"
                type="number"
                min={1}
                max={30}
                className={field}
                value={session}
                onChange={(e) => setSession(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="webcall-day" className={label}>
                Minutes per day, everyone together
              </label>
              <input
                id="webcall-day"
                type="number"
                min={1}
                max={1440}
                className={field}
                value={perDay}
                onChange={(e) => setPerDay(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="webcall-visitor" className={label}>
                Minutes per visitor per day (optional)
              </label>
              <input
                id="webcall-visitor"
                type="number"
                min={1}
                max={1440}
                className={field}
                value={perVisitor}
                onChange={(e) => setPerVisitor(e.target.value)}
                placeholder="No limit"
              />
            </div>
          </div>
          <div className="flex justify-end">
            <Button size="sm" loading={busy} onClick={create} leftIcon={<Link2 className="size-3.5" />}>
              Create link
            </Button>
          </div>

          {data.links.length === 0 ? (
            <p className="text-sm text-text-secondary">No links yet. Create one above and share it.</p>
          ) : (
            <ul className="space-y-3 border-t border-border pt-4" aria-label="Web call links">
              {data.links.map((l) => (
                <li key={l.id} className="space-y-2 rounded-lg border border-border p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="flex min-w-0 items-center gap-2 text-sm font-medium text-text-primary">
                      <span className="truncate">{l.label || "Web call link"}</span>
                      <Badge tone={STATUS_TONE[l.status]}>{STATUS_TEXT[l.status]}</Badge>
                    </p>
                    <p className="text-xs text-text-secondary">{expiryText(l)}</p>
                  </div>
                  <input
                    readOnly
                    aria-label={`Link${l.label ? ` ${l.label}` : ""}`}
                    value={l.url}
                    onFocus={(e) => e.currentTarget.select()}
                    className={`${field} font-mono text-xs`}
                  />
                  <p className="text-xs text-text-secondary">
                    {l.sessions} {l.sessions === 1 ? "session" : "sessions"} · {l.turns} {l.turns === 1 ? "turn" : "turns"} ·{" "}
                    {l.minutes_today} of {l.max_minutes_per_day} min used today · up to {l.max_session_minutes} min per call
                    {l.max_minutes_per_visitor ? ` · ${l.max_minutes_per_visitor} min per visitor per day` : ""}
                    {l.last_used_at ? ` · last used ${dateText(l.last_used_at)}` : ""}
                  </p>
                  <div className="flex flex-wrap justify-end gap-2">
                    {l.status === "active" && (
                      <>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => copy(l)}
                          leftIcon={copied === l.id ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                        >
                          {copied === l.id ? "Copied" : "Copy link"}
                        </Button>
                        <Button variant="ghost" size="sm" onClick={() => setConfirm({ kind: "revoke", link: l })} leftIcon={<XCircle className="size-3.5" />}>
                          Revoke
                        </Button>
                      </>
                    )}
                    {l.status !== "active" && (
                      <Button variant="ghost" size="sm" onClick={() => setConfirm({ kind: "remove", link: l })} leftIcon={<Trash2 className="size-3.5" />}>
                        Remove
                      </Button>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
          <p className="text-xs text-text-secondary">
            Visitors are told they are talking to an AI before they start. Conversations are saved to this agent as
            website conversations. Anyone who has the link can use your minutes, so share it only with people you trust.
          </p>
        </>
      )}

      <ConfirmDialog
        open={confirm?.kind === "revoke"}
        title="Revoke this link?"
        description="The link stops working at once, also for a call that is in progress. You cannot switch it back on; create a new link instead."
        confirmLabel="Revoke link"
        keepLabel="Keep link"
        onCancel={() => setConfirm(null)}
        onConfirm={act}
      />
      <ConfirmDialog
        open={confirm?.kind === "remove"}
        title="Remove this link from the list?"
        description="Its counters are deleted. The link is already off."
        confirmLabel="Remove link"
        keepLabel="Keep"
        onCancel={() => setConfirm(null)}
        onConfirm={act}
      />
    </Card>
  );
}
