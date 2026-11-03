"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { CalendarCheck, Check, X } from "lucide-react";
import { toast } from "sonner";
import { Badge, Button, Card, EmptyState, ErrorBoundary, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import type { BadgeTone } from "@/components/ui/Badge";
import { PageShell } from "@/components/ui/PageHeader";
import { useMeetingSettings, useMeetings } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";
import { WeekSchedule } from "@/components/dashboard/WeekSchedule";
import { maskNumber } from "@/app/(app)/dashboard/calls/format";
import type { MeetingRequest, MeetingStatus } from "@/types/api";

const TABS: { id: MeetingStatus; label: string }[] = [
  { id: "pending", label: "Waiting for you" },
  { id: "approved", label: "Approved" },
  { id: "declined", label: "Declined" },
];
const TONE: Record<MeetingStatus, BadgeTone> = { pending: "warning", approved: "success", declined: "error", cancelled: "neutral" };
const STATUS_LABEL: Record<MeetingStatus, string> = { pending: "Waiting", approved: "Approved", declined: "Declined", cancelled: "Withdrawn" };
const ROLE_LABEL: Record<string, string> = { student: "Student", parent: "Parent", faculty: "Staff", other: "" };

/** "Mon 19 Oct, 10:30" from the request's ISO date and HH:MM time. */
function when(date: string, time: string): string {
  const d = new Date(`${date}T00:00:00`);
  const day = Number.isNaN(d.getTime()) ? date : d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
  return `${day}, ${time}`;
}

function RequestCard({ r, onDone, highlight }: { r: MeetingRequest; onDone: () => void; highlight?: boolean }) {
  const [mode, setMode] = useState<"approve" | "decline" | null>(null);
  const [note, setNote] = useState("");
  const [date, setDate] = useState("");
  const [time, setTime] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const decide = async () => {
    if (!mode) return;
    setBusy(true);
    setError(null);
    try {
      const body = mode === "approve" ? { note, date: date || undefined, time: time || undefined }
        : { note, suggest_date: date || undefined, suggest_time: time || undefined };
      await api.post(`/api/meetings/${encodeURIComponent(r.request_id)}/${mode}`, body);
      toast.success(mode === "approve" ? `Approved the request from ${r.name}` : `Declined the request from ${r.name}`);
      onDone();
    } catch (e) {
      setError(errMessage(e, "Could not save the decision."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card id={`meeting-${r.request_id}`} className={`space-y-3 p-5 ${highlight ? "ring-2 ring-accent-primary" : ""}`}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-[15px] font-semibold text-text-primary">{r.name}</p>
          <p className="mt-0.5 text-sm text-text-secondary">
            {[ROLE_LABEL[r.role] ?? r.role, r.class_or_ref, maskNumber(r.phone)].filter(Boolean).join(" · ")}
          </p>
        </div>
        <Badge tone={TONE[r.status]} dot>{STATUS_LABEL[r.status] ?? r.status}</Badge>
      </div>
      <p className="text-sm font-medium text-text-primary">
        <span className="text-text-secondary">Wants </span>
        {when(r.date, r.time)} <span className="font-normal text-text-secondary">({r.minutes} min)</span>
        {r.on_calendar && <span className="ml-2 text-xs font-normal text-accent-success">on Google Calendar</span>}
      </p>
      {r.reason && <p className="text-sm text-text-primary"><span className="text-text-secondary">About: </span>{r.reason}</p>}
      {r.decision_note && <p className="text-sm text-text-secondary">Your note: {r.decision_note}</p>}
      {r.suggested && <p className="text-sm text-text-secondary">You suggested {when(r.suggested.date, r.suggested.time)}.</p>}

      {r.status === "pending" && mode === null && (
        <div className="flex flex-wrap gap-2 pt-1">
          <Button size="sm" onClick={() => setMode("approve")}><Check className="size-4" />Approve</Button>
          <Button size="sm" variant="outline" onClick={() => setMode("decline")}><X className="size-4" />Decline</Button>
        </div>
      )}

      {mode && (
        <div className="space-y-3 rounded-xl border border-border bg-bg-elevated p-3">
          <label className="block text-sm">
            <span className="text-text-secondary">{mode === "approve" ? "Note for the caller (optional)" : "Reason for the caller (optional)"}</span>
            <input value={note} onChange={(e) => setNote(e.target.value)} maxLength={300}
              className="mt-1 w-full rounded-lg border border-border bg-bg-surface px-3 py-2 text-sm text-text-primary" />
          </label>
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block text-sm">
              <span className="text-text-secondary">{mode === "approve" ? "Approve on another day (optional)" : "Suggest another day (optional)"}</span>
              <input type="date" value={date} onChange={(e) => setDate(e.target.value)}
                className="mt-1 w-full rounded-lg border border-border bg-bg-surface px-3 py-2 text-sm text-text-primary" />
            </label>
            <label className="block text-sm">
              <span className="text-text-secondary">{mode === "approve" ? "Another time (optional)" : "Suggested time (optional)"}</span>
              <input type="time" value={time} onChange={(e) => setTime(e.target.value)}
                className="mt-1 w-full rounded-lg border border-border bg-bg-surface px-3 py-2 text-sm text-text-primary" />
            </label>
          </div>
          {error && <p role="alert" className="text-sm text-accent-error">{error}</p>}
          <div className="flex gap-2">
            <Button size="sm" onClick={decide} disabled={busy}>{busy ? "Saving..." : mode === "approve" ? "Confirm approval" : "Confirm decline"}</Button>
            <Button size="sm" variant="ghost" onClick={() => { setMode(null); setError(null); }} disabled={busy}>Back</Button>
          </div>
        </div>
      )}
    </Card>
  );
}

function ApprovalsContent() {
  const [tab, setTab] = useState<MeetingStatus>("pending");
  const { data, error, isLoading, mutate } = useMeetings(tab);
  const { data: info } = useMeetingSettings();
  const [focusId, setFocusId] = useState<string | null>(null);
  const counts = data?.counts;
  const calendar = info?.calendar;
  const focusShown = !!focusId && !!data?.requests.some((r) => r.request_id === focusId);

  // "Show the full request" in the week grid: switch to the tab that holds it, scroll to its card and ring it for a few seconds.
  useEffect(() => {
    if (!focusId || !focusShown) return;
    const el = document.getElementById(`meeting-${focusId}`);
    const calm = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el?.scrollIntoView?.({ behavior: calm ? "auto" : "smooth", block: "center" });
    const t = setTimeout(() => setFocusId(null), 4000);
    return () => clearTimeout(t);
  }, [focusId, focusShown]);

  return (
    <PageShell width="default" className="space-y-6">
      <PageHeader
        title="Approvals"
        description="Requests your agents have taken for a meeting or an appointment. Nothing is confirmed until you approve it here."
      />
      <details className="group rounded-xl border border-border bg-bg-surface px-4 py-3 text-sm text-text-secondary">
        <summary className="min-h-6 cursor-pointer font-medium text-text-primary">How requests get here</summary>
        <p className="mt-2 leading-relaxed">
          When someone asks your agent for a meeting or an appointment, in a phone call, on WhatsApp, in the chat on your website or in a practice call,
          the agent checks which times are free and puts in a request. It never confirms anything itself. You approve or decline here, and the
          caller can ask the agent what you decided.
        </p>
      </details>
      <WeekSchedule onDecision={() => mutate()}
        onShowRequest={(id, status) => { setTab(status); setFocusId(id); }} />
      {calendar && (
        <div className="text-sm text-text-secondary">
          <p>
            {calendar.configured
              ? "Google Calendar is connected: approved requests are added to it, and busy times are not offered."
              : "Google Calendar is not connected. Approved requests stay in BonClay only."}
          </p>
          {!calendar.configured && (
            <details className="mt-1">
              <summary className="min-h-6 cursor-pointer">Details for whoever sets this up</summary>
              <p className="mt-1">Add GOOGLE_CALENDAR_ID and the service account key to the backend settings to connect it.</p>
            </details>
          )}
        </div>
      )}
      <div role="tablist" aria-label="Request status" className="flex flex-wrap gap-1">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${tab === t.id ? "bg-accent-primary/15 text-text-primary" : "text-text-secondary hover:bg-bg-elevated"}`}>
            {t.label}{counts ? ` (${counts[t.id]})` : ""}
          </button>
        ))}
      </div>
      {isLoading && <Skeleton className="h-32 w-full" />}
      {error && <ErrorState title="Could not load the requests" message={errMessage(error, "Check that the API is running.")} onRetry={() => mutate()} />}
      {data && data.requests.length === 0 && (
        <EmptyState icon={<CalendarCheck />} title={tab === "pending" ? "Nothing is waiting for you" : `No ${tab} requests`}
          description={tab === "pending" ? "When someone asks your agent for a meeting or an appointment, the request appears here for you to approve or decline." : undefined}
          action={tab === "pending" ? (
            <Link href="/dashboard/calls" className="inline-flex h-10 items-center justify-center rounded-full border border-border bg-bg-surface px-4 text-sm font-medium text-text-primary hover:border-text-primary">
              See recent calls
            </Link>
          ) : undefined} />
      )}
      <div className="space-y-4">
        {data?.requests.map((r) => <RequestCard key={r.request_id} r={r} onDone={() => mutate()} highlight={r.request_id === focusId} />)}
      </div>
    </PageShell>
  );
}

export default function ApprovalsPage() {
  return (
    <ErrorBoundary>
      <ApprovalsContent />
    </ErrorBoundary>
  );
}
