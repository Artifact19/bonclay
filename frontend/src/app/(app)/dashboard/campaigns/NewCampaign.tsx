"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Button, Card } from "@/components/ui";
import { useAgents } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";
import { cn } from "@/lib/utils";
import { DAY_LABELS } from "./format";
import type { Campaign, NewCampaignBody } from "./types";

const inputCls = "h-10 min-w-0 rounded-lg border border-border bg-bg-surface px-3 text-sm text-text-primary";
const ZONES = ["Asia/Kolkata", "Asia/Dubai", "Asia/Singapore", "Europe/London", "America/New_York", "America/Los_Angeles"];

/** The form that makes a campaign from a CSV of people who agreed to be called. */
export function NewCampaign({ onCreated, onCancel }: { onCreated: (c: Campaign) => void; onCancel: () => void }) {
  const agents = useAgents();
  const [name, setName] = useState("");
  const [agentId, setAgentId] = useState("");
  const [csv, setCsv] = useState("");
  const [days, setDays] = useState<number[]>([0, 1, 2, 3, 4, 5]);
  const [start, setStart] = useState("10:00");
  const [end, setEnd] = useState("18:00");
  const [zone, setZone] = useState("Asia/Kolkata");
  const [concurrency, setConcurrency] = useState(1);
  const [attempts, setAttempts] = useState(2);
  const [retry, setRetry] = useState(30);
  const [agreed, setAgreed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reason = !name.trim() ? "Give the campaign a name" : !agentId ? "Choose an agent" : !csv.trim() ? "Add the phone numbers"
    : !days.length ? "Choose at least one day" : start >= end ? "The end time must be after the start" : !agreed ? "Confirm that everyone agreed to be called" : "";

  const readFile = async (file: File | undefined) => {
    if (!file) return;
    if (file.size > 200_000) { setError("That file is too large. Use a list of up to 500 numbers."); return; }
    setError(null);
    setCsv(await file.text());
  };

  const toggleDay = (d: number) => setDays((cur) => (cur.includes(d) ? cur.filter((x) => x !== d) : [...cur, d].sort()));

  const submit = async () => {
    setBusy(true);
    setError(null);
    const body: NewCampaignBody = {
      name: name.trim(), agent_id: agentId, csv, caller_agreed: agreed,
      window: { days, start, end, timezone: zone }, max_attempts: attempts, retry_minutes: retry, concurrency,
    };
    try {
      const made = await api.post<Campaign>("/api/campaigns", body);
      const skipped = (made.import?.invalid ?? 0) + (made.import?.duplicates ?? 0) + (made.import?.over_limit ?? 0);
      toast.success(`Campaign ready with ${made.total} ${made.total === 1 ? "person" : "people"}${skipped ? ` (${skipped} lines skipped)` : ""}`);
      onCreated(made);
    } catch (e) {
      setError(errMessage(e, "Could not create the campaign."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="space-y-4 p-5">
      <h2 className="text-base font-semibold text-text-primary">New campaign</h2>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm text-text-secondary">
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} maxLength={80} placeholder="October fee reminders" className={inputCls} />
        </label>
        <label className="flex flex-col gap-1 text-sm text-text-secondary">
          Agent that makes the calls
          <select value={agentId} onChange={(e) => setAgentId(e.target.value)} className={inputCls}>
            <option value="">Choose an agent</option>
            {(agents.data ?? []).map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>
        </label>
      </div>

      <div className="space-y-1.5">
        <label htmlFor="campaign-csv" className="text-sm text-text-secondary">
          Phone numbers (CSV: a number column, and optionally a name column. Up to 500.)
        </label>
        <textarea
          id="campaign-csv" value={csv} onChange={(e) => setCsv(e.target.value)} rows={5} spellCheck={false}
          placeholder={"name,phone\nAsha,98765 43210\nRavi,+919876543211"}
          className="w-full rounded-lg border border-border bg-bg-surface p-3 font-mono text-sm text-text-primary"
        />
        <input type="file" accept=".csv,text/csv,text/plain" aria-label="Upload a CSV file" onChange={(e) => readFile(e.target.files?.[0])} className="text-sm text-text-secondary" />
      </div>

      <fieldset className="space-y-2">
        <legend className="text-sm text-text-secondary">When BonClay may call</legend>
        <div role="group" aria-label="Days" className="flex flex-wrap gap-1.5">
          {DAY_LABELS.map((label, d) => (
            <button
              key={label} type="button" aria-pressed={days.includes(d)} onClick={() => toggleDay(d)}
              className={cn("min-h-9 rounded-full border px-3 text-sm font-medium transition-colors",
                days.includes(d) ? "border-accent-primary/50 bg-accent-primary/15 text-text-primary" : "border-border text-text-secondary hover:bg-bg-elevated")}
            >{label}</button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm text-text-secondary">
          <label className="flex items-center gap-2">From <input type="time" value={start} onChange={(e) => setStart(e.target.value)} className={inputCls} aria-label="Start time" /></label>
          <label className="flex items-center gap-2">to <input type="time" value={end} onChange={(e) => setEnd(e.target.value)} className={inputCls} aria-label="End time" /></label>
          <select value={zone} onChange={(e) => setZone(e.target.value)} className={inputCls} aria-label="Time zone">
            {ZONES.map((z) => <option key={z} value={z}>{z}</option>)}
          </select>
        </div>
      </fieldset>

      <div className="grid gap-3 sm:grid-cols-3">
        <label className="flex flex-col gap-1 text-sm text-text-secondary">
          Calls at the same time
          <select value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value))} className={inputCls}>
            {[1, 2, 3].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm text-text-secondary">
          Tries per person
          <select value={attempts} onChange={(e) => setAttempts(Number(e.target.value))} className={inputCls}>
            {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm text-text-secondary">
          Wait before trying again
          <select value={retry} onChange={(e) => setRetry(Number(e.target.value))} className={inputCls}>
            {[10, 30, 60, 120, 240].map((n) => <option key={n} value={n}>{n >= 60 ? `${n / 60} h` : `${n} min`}</option>)}
          </select>
        </label>
      </div>

      <label className="flex items-start gap-2 text-sm text-text-primary">
        <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} className="mt-0.5 size-4" />
        <span>Everyone on this list agreed to receive this call. The agent still says it is an AI at the start of every call.</span>
      </label>

      {error && <p role="alert" className="text-sm text-accent-error">{error}</p>}
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={submit} loading={busy} disabled={!!reason} disabledReason={reason}>Create campaign</Button>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
    </Card>
  );
}
