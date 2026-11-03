"use client";

import { useState } from "react";
import useSWR from "swr";
import { Pause, Play, Square, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Badge, Button, Card } from "@/components/ui";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";
import { CONTACT_LABEL, STATUS_LABEL, STATUS_TONE, progressPercent, windowLabel } from "./format";
import type { Campaign } from "./types";

export function CampaignCard({ campaign: c, agentName, onChanged }: { campaign: Campaign; agentName: string; onChanged: () => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<"stop" | "delete" | null>(null);
  const [showPeople, setShowPeople] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const people = useSWR<Campaign>(showPeople ? `/api/campaigns/${c.id}` : null, (p: string) => api.get<Campaign>(p), {
    refreshInterval: c.status === "running" ? 8000 : 0, shouldRetryOnError: false,
  });

  const act = async (action: "start" | "pause" | "stop") => {
    setBusy(action);
    setError(null);
    try {
      await api.post(`/api/campaigns/${c.id}/${action}`, {});
      setConfirm(null);
      onChanged();
    } catch (e) {
      setError(errMessage(e, "Could not do that."));
    } finally {
      setBusy(null);
    }
  };
  const remove = async () => {
    setBusy("delete");
    setError(null);
    try {
      await api.del(`/api/campaigns/${c.id}`);
      toast.success("Campaign deleted, with its phone numbers");
      onChanged();
    } catch (e) {
      setError(errMessage(e, "Could not delete the campaign."));
    } finally {
      setBusy(null);
    }
  };

  const pct = progressPercent(c);
  const final = c.status === "done" || c.status === "stopped";
  return (
    <Card className="space-y-3 p-5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="truncate text-[15px] font-semibold text-text-primary">{c.name}</h3>
          <p className="mt-0.5 text-sm text-text-secondary">{agentName} · {c.total} {c.total === 1 ? "person" : "people"}</p>
        </div>
        <Badge tone={STATUS_TONE[c.status]} dot>{STATUS_LABEL[c.status]}</Badge>
      </div>

      <div>
        <div role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label={`${c.name} progress`} className="h-2 overflow-hidden rounded-full bg-bg-elevated">
          <div className="h-full rounded-full bg-accent-success transition-all" style={{ width: `${pct}%` }} />
        </div>
        <p className="mt-1.5 text-sm text-text-secondary">
          {c.counts.done} reached · {c.counts.pending + c.counts.calling} still to call · {c.counts.failed} not reached
          {c.counts.dnc > 0 && ` · ${c.counts.dnc} on the do-not-call list`}
          {c.counts.calling > 0 && ` · ${c.counts.calling} on a call now`}
        </p>
      </div>

      <p className="text-sm text-text-secondary">
        Calls {windowLabel(c.window)} · up to {c.max_attempts} {c.max_attempts === 1 ? "try" : "tries"} each, {c.retry_minutes} min apart
      </p>
      {c.paused_reason && c.status === "paused" && <p role="status" className="text-sm text-accent-warning">{c.paused_reason}</p>}
      {error && <p role="alert" className="text-sm text-accent-error">{error}</p>}

      <div className="flex flex-wrap items-center gap-2">
        {(c.status === "draft" || c.status === "paused") && (
          <Button size="sm" onClick={() => act("start")} loading={busy === "start"}><Play className="size-4" />{c.status === "paused" ? "Resume" : "Start calling"}</Button>
        )}
        {c.status === "running" && (
          <Button size="sm" variant="outline" onClick={() => act("pause")} loading={busy === "pause"}><Pause className="size-4" />Pause</Button>
        )}
        {!final && confirm !== "stop" && c.status !== "draft" && (
          <Button size="sm" variant="ghost" onClick={() => setConfirm("stop")}><Square className="size-4" />Stop</Button>
        )}
        {confirm === "stop" && (
          <span className="flex items-center gap-2 text-sm text-text-secondary">
            Stopping is final.
            <Button size="sm" onClick={() => act("stop")} loading={busy === "stop"}>Stop campaign</Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirm(null)}>Keep going</Button>
          </span>
        )}
        {confirm !== "delete" ? (
          <Button size="sm" variant="ghost" onClick={() => setConfirm("delete")}><Trash2 className="size-4" />Delete</Button>
        ) : (
          <span className="flex items-center gap-2 text-sm text-text-secondary">
            This removes the phone numbers.
            <Button size="sm" onClick={remove} loading={busy === "delete"}>Delete campaign</Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirm(null)}>Cancel</Button>
          </span>
        )}
        <Button size="sm" variant="ghost" onClick={() => setShowPeople((v) => !v)} aria-expanded={showPeople}>{showPeople ? "Hide people" : "Show people"}</Button>
      </div>

      {showPeople && (
        <ul className="divide-y divide-border rounded-lg border border-border text-sm" aria-label="People in this campaign">
          {(people.data?.contacts ?? []).map((p, i) => (
            <li key={i} className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
              <span className="text-text-primary">{p.name || "No name"} <span className="text-text-secondary">{p.number}</span></span>
              <span className="text-text-secondary">{CONTACT_LABEL[p.status]}{p.attempts ? ` · ${p.attempts} ${p.attempts === 1 ? "try" : "tries"}` : ""}</span>
            </li>
          ))}
          {people.isLoading && <li className="px-3 py-2 text-text-secondary">Loading…</li>}
        </ul>
      )}
    </Card>
  );
}
