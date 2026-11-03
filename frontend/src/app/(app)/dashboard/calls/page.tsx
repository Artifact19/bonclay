"use client";

import { Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { PhoneCall } from "lucide-react";
import { Button, EmptyState, ErrorBoundary, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { useAgents } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";
import { cn } from "@/lib/utils";
import { CallRow } from "./CallRow";
import { LiveCalls } from "./LiveCalls";
import { CHANNEL_LABEL, OUTCOME_LABEL, OUTCOME_ORDER } from "./format";
import type { CallChannel, CallOutcomeKey, CallsResponse } from "./types";

const DAYS = 30;
const CHANNELS = Object.keys(CHANNEL_LABEL) as CallChannel[];
const selectCls = "h-10 min-w-0 rounded-lg border border-border bg-bg-surface px-3 text-sm text-text-primary";
const primaryLink =
  "inline-flex h-10 items-center justify-center rounded-full bg-brand px-4 text-sm font-medium text-[var(--color-on-accent)] transition-opacity hover:opacity-90";

function CallsContent() {
  const [agent, setAgent] = useState("");
  const [channel, setChannel] = useState<"" | CallChannel>("");
  const [outcome, setOutcome] = useState<"" | CallOutcomeKey>("");
  const agents = useAgents();
  const focus = useSearchParams()?.get("call") ?? null; // the link in the Google Sheet row opens that conversation

  const key = `/api/calls?days=${DAYS}&limit=100${agent ? `&agent_id=${encodeURIComponent(agent)}` : ""}`;
  const { data, error, isLoading, mutate } = useSWR<CallsResponse>(key, (p: string) => api.get<CallsResponse>(p), {
    refreshInterval: 30000, // new conversations appear without a reload (paused while the tab is hidden)
    revalidateOnFocus: true,
    shouldRetryOnError: false,
  });

  const calls = useMemo(() => data?.calls ?? [], [data]);
  const inChannel = useMemo(() => calls.filter((c) => !channel || c.channel === channel), [calls, channel]);
  const counts = useMemo(() => {
    const n: Partial<Record<CallOutcomeKey, number>> = {};
    for (const c of inChannel) n[c.outcome.key] = (n[c.outcome.key] ?? 0) + 1;
    return n;
  }, [inChannel]);
  const shown = inChannel.filter((c) => !outcome || c.outcome.key === outcome);
  const filtered = !!(agent || channel || outcome);
  const clear = () => { setAgent(""); setChannel(""); setOutcome(""); };

  return (
    <>
      <PageHeader
        width="default"
        title="Calls and bookings"
        description="Every conversation your agents had in the last 30 days, on the phone, on your website, on WhatsApp or here in the app. Open one to read what was said."
      >
        <LiveCalls />
        {error && <ErrorState title="Could not load the conversations" message={errMessage(error, "Check that the API is running.")} onRetry={() => mutate()} />}
        {isLoading && (
          <div className="space-y-3" aria-busy="true" aria-label="Loading conversations">
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-20 w-full" />
          </div>
        )}

        {data && calls.length === 0 && !agent && (
          <EmptyState
            icon={<PhoneCall />}
            title="No conversations yet"
            description="When someone calls your agent, writes to it on WhatsApp or uses the chat on your website, or when you talk to it here, it appears on this page: when it happened, what was asked, and whether it ended in a booking."
            action={
              <div className="flex flex-wrap justify-center gap-2">
                <Link href="/dashboard/playground" className={primaryLink}>Make a practice call</Link>
                <Link href="/dashboard/studio/agents" className="inline-flex h-10 items-center justify-center rounded-full border border-border bg-bg-surface px-4 text-sm font-medium text-text-primary hover:border-text-primary">
                  Go to agents
                </Link>
              </div>
            }
          />
        )}

        {data && (calls.length > 0 || !!agent) && (
          <div className="space-y-5">
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-sm text-text-secondary">
                <span>Agent</span>
                <select value={agent} onChange={(e) => setAgent(e.target.value)} className={selectCls}>
                  <option value="">All agents</option>
                  {(agents.data ?? []).map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
                </select>
              </label>
              <label className="flex items-center gap-2 text-sm text-text-secondary">
                <span>Where</span>
                <select value={channel} onChange={(e) => setChannel(e.target.value as "" | CallChannel)} className={selectCls}>
                  <option value="">Everywhere</option>
                  {CHANNELS.map((c) => <option key={c} value={c}>{CHANNEL_LABEL[c]}</option>)}
                </select>
              </label>
              {filtered && <Button variant="ghost" size="sm" onClick={clear}>Clear filters</Button>}
            </div>

            <div role="group" aria-label="Filter by what happened" className="flex flex-wrap gap-1.5">
              {([["", "All", inChannel.length] as const, ...OUTCOME_ORDER.filter((k) => counts[k]).map((k) => [k, OUTCOME_LABEL[k], counts[k] ?? 0] as const)]).map(([k, label, n]) => (
                <button
                  key={k || "all"}
                  type="button"
                  aria-pressed={outcome === k}
                  onClick={() => setOutcome(k as "" | CallOutcomeKey)}
                  className={cn(
                    "min-h-10 rounded-full border px-3.5 text-sm font-medium transition-colors",
                    outcome === k ? "border-accent-primary/50 bg-accent-primary/15 text-text-primary" : "border-border text-text-secondary hover:bg-bg-elevated hover:text-text-primary"
                  )}
                >
                  {label} <span className="text-text-secondary">{n}</span>
                </button>
              ))}
            </div>

            {shown.length === 0 ? (
              <EmptyState
                icon={<PhoneCall />}
                title="Nothing matches these filters"
                description="Try another agent, place or outcome."
                action={<Button variant="outline" onClick={clear}>Clear filters</Button>}
              />
            ) : (
              <ul className="space-y-3">
                {shown.map((c) => <li key={c.id}><CallRow call={c} defaultOpen={c.id === focus} /></li>)}
              </ul>
            )}
            {calls.length >= 100 && <p className="text-center text-xs text-text-secondary">Showing the 100 most recent conversations.</p>}
          </div>
        )}
      </PageHeader>
    </>
  );
}

export default function CallsPage() {
  return (
    <ErrorBoundary>
      {/* useSearchParams needs a Suspense boundary for static rendering */}
      <Suspense fallback={null}>
        <CallsContent />
      </Suspense>
    </ErrorBoundary>
  );
}
