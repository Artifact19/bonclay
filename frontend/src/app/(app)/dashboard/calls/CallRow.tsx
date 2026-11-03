"use client";

import { useEffect, useId, useRef, useState } from "react";
import Link from "next/link";
import useSWR from "swr";
import { ChevronDown, Globe, MessageCircle, Mic, Phone, Wrench } from "lucide-react";
import { Badge, Card, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useFlaggedByCall } from "@/components/analytics/answerAudit";
import { CHANNEL_LABEL, OUTCOME_LABEL, OUTCOME_TONE, callerLabel, durationLabel, whenLabel } from "./format";
import type { CallChannel, CallDetail, CallSheet, CallSummary } from "./types";

const CHANNEL_ICON: Record<CallChannel, typeof Phone> = { phone: Phone, website: Globe, whatsapp: MessageCircle, playground: Mic };

/** What happened to this conversation's row in the owner's Google Sheet (backend/services/caller_sheet.py). */
export function SheetBadge({ sheet }: { sheet: CallSheet }) {
  if (sheet.state === "sent") return <Badge tone="success" data-testid="sheet-status">Sent to Google Sheets ✓</Badge>;
  if (sheet.state === "waiting") return <Badge tone="neutral" data-testid="sheet-status">Waiting to send to Google Sheets</Badge>;
  return <Badge tone="error" data-testid="sheet-status">Google Sheets failed{sheet.error ? `: ${sheet.error}` : ""}</Badge>;
}

function Transcript({ id }: { id: string }) {
  const { data, error, isLoading } = useSWR<CallDetail>(`/api/calls/${encodeURIComponent(id)}`, (p: string) => api.get<CallDetail>(p), {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
  });
  if (isLoading) return <Skeleton className="h-24 w-full" />;
  if (error || !data) return <p role="alert" className="text-sm text-accent-error">Could not load this transcript. Try again in a moment.</p>;
  if (data.transcript.length === 0) return <p className="text-sm text-text-secondary">Nothing was said in this conversation, so there is no transcript.</p>;
  return (
    <ol className="space-y-2" aria-label="Transcript">
      {data.transcript.map((line, i) => {
        const agent = line.role === "agent";
        return (
          <li key={i} className={cn("flex", agent ? "justify-start" : "justify-end")}>
            <div className={cn("max-w-[85%] rounded-2xl px-3.5 py-2 text-sm", agent ? "bg-bg-elevated text-text-primary" : "bg-accent-primary/10 text-text-primary")}>
              <p className="text-[11px] font-semibold uppercase tracking-wide text-text-secondary">{agent ? data.agent_name : "Caller"}</p>
              <p className="mt-0.5 whitespace-pre-wrap break-words">{line.text}</p>
              {line.actions.map((a, j) => (
                <p key={j} className={cn("mt-1.5 flex items-start gap-1.5 text-xs", a.ok ? "text-text-secondary" : "text-accent-error")}>
                  <Wrench aria-hidden className="mt-0.5 size-3 shrink-0" />
                  <span>{a.ok ? a.summary : "The agent tried an action and it did not work."}</span>
                </p>
              ))}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

export function CallRow({ call, defaultOpen = false }: { call: CallSummary; defaultOpen?: boolean }) {
  // A link from a spreadsheet row (?call=<id>) opens this conversation; the owner's own click wins afterwards.
  const [toggled, setToggled] = useState<boolean | null>(null);
  const open = toggled ?? defaultOpen;
  const card = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (defaultOpen) card.current?.scrollIntoView?.({ block: "center" });
  }, [defaultOpen]);
  const panel = useId();
  const Icon = CHANNEL_ICON[call.channel];
  const duration = durationLabel(call.duration_s);
  const { outcome } = call;
  const toCheck = useFlaggedByCall()[call.id] ?? 0;
  return (
    <Card ref={card} variant="solid" padded={false} className="overflow-hidden">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panel}
        onClick={() => setToggled(!open)}
        className="flex min-h-[44px] w-full items-start gap-3 px-4 py-3.5 text-left transition-colors hover:bg-bg-elevated/60 sm:px-5"
      >
        <span aria-hidden className="mt-0.5 grid size-9 shrink-0 place-items-center rounded-lg bg-bg-elevated text-text-secondary">
          <Icon className="size-[18px]" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="text-[15px] font-semibold text-text-primary">{whenLabel(call.when)}</span>
            <Badge tone={OUTCOME_TONE[outcome.key]} dot>{OUTCOME_LABEL[outcome.key] ?? outcome.label}</Badge>
            {call.practice && <Badge>Practice call</Badge>}
            {toCheck > 0 && <Badge tone="warning">{toCheck === 1 ? "Check this answer" : `Check ${toCheck} answers`}</Badge>}
            {call.sheet && <SheetBadge sheet={call.sheet} />}
          </span>
          <span className="mt-0.5 block text-sm text-text-secondary">
            {CHANNEL_LABEL[call.channel]} · {call.agent_name} · {callerLabel(call)}
            {duration ? ` · ${duration}` : ""}
          </span>
          {call.preview && <span className="mt-1.5 block truncate text-sm text-text-primary">&ldquo;{call.preview}&rdquo;</span>}
          {outcome.detail && <span className="mt-1 block text-xs text-text-secondary">{outcome.detail}</span>}
        </span>
        <ChevronDown aria-hidden className={cn("mt-2 size-4 shrink-0 text-text-secondary transition-transform motion-reduce:transition-none", open && "rotate-180")} />
        <span className="sr-only">{open ? "Hide transcript" : "Show transcript"}</span>
      </button>
      {open && (
        <div id={panel} className="space-y-3 border-t border-border bg-bg-base/40 px-4 py-4 sm:px-5">
          <Transcript id={call.id} />
          {outcome.request_id && (
            <Link href="/dashboard/approvals" className="inline-flex min-h-10 items-center text-sm font-medium text-accent-secondary underline-offset-2 hover:underline">
              {outcome.key === "needs_approval" ? "Review this request in Approvals" : "See this request in Approvals"}
            </Link>
          )}
        </div>
      )}
    </Card>
  );
}
