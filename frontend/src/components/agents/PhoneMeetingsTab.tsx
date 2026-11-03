"use client";

import Link from "next/link";
import { CalendarCheck, ChevronDown, Phone, PhoneIncoming } from "lucide-react";
import { Badge, Button } from "@/components/ui";
import type { Agent } from "@/types/api";
import { usePendingRequests, usePhoneConnected } from "./agentStatus";
import { InstantAnswersPanel } from "./InstantAnswersPanel";

/** Tool servers an agent can use (the backend knows them by these names). */
export const TOOL_OPTIONS: { id: string; label: string; hint: string; details: string }[] = [
  {
    id: "appointments",
    label: "Let the agent book appointments",
    hint: "The agent can check free times, book, find and cancel appointments while it talks to someone.",
    details: "This turns on the \"appointments\" tool server (MCP), which runs on this computer. Bookings it makes appear on the Approvals page.",
  },
];

interface Props {
  agent: Agent;
  tools: string[];
  onToolsChange: (next: string[]) => void;
  onPhone?: (agent: Agent) => void;
}

const box =
  "mt-0.5 size-5 shrink-0 accent-[var(--color-accent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary";

/** The "Phone and meetings" tab: calls in and out, booking appointments, and where requests are approved. */
export function PhoneMeetingsTab({ agent, tools, onToolsChange, onPhone }: Props) {
  const phone = usePhoneConnected();
  const pending = usePendingRequests();

  return (
    <div className="space-y-6">
      <section aria-labelledby="phone-title" className="space-y-4 rounded-2xl border border-border bg-bg-surface p-5">
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border pb-4">
          <div>
            <h2 id="phone-title" className="font-display text-lg font-semibold text-text-primary">Phone calls</h2>
            <p className="mt-1 text-sm text-text-secondary">
              Let people ring your number and talk to this agent, or have the agent phone someone for you.
            </p>
          </div>
          {phone === true && <Badge tone="success" dot>Phone line connected</Badge>}
          {phone === false && <Badge tone="warning" dot>Phone line not connected</Badge>}
        </div>
        <div className="flex flex-wrap gap-2">
          {onPhone && (
            <Button onClick={() => onPhone(agent)} leftIcon={<PhoneIncoming className="size-4" />}>
              Receive calls on your number
            </Button>
          )}
          {onPhone && (
            <Button variant="outline" onClick={() => onPhone(agent)} leftIcon={<Phone className="size-4" />}>
              Phone someone
            </Button>
          )}
        </div>
        {phone === false && (
          <p className="text-sm text-text-secondary">
            Your phone line is not connected yet. Open the setup above for the steps, or ask the person who installed BonClay.
          </p>
        )}
      </section>

      <section aria-labelledby="meet-title" className="space-y-4 rounded-2xl border border-border bg-bg-surface p-5">
        <div className="border-b border-border pb-4">
          <h2 id="meet-title" className="font-display text-lg font-semibold text-text-primary">Appointments and meetings</h2>
          <p className="mt-1 text-sm text-text-secondary">Nothing is confirmed until you approve it.</p>
        </div>
        {TOOL_OPTIONS.map((t) => (
          <div key={t.id}>
            <label htmlFor={`agent-tool-${t.id}`} className="flex cursor-pointer items-start gap-3">
              <input
                id={`agent-tool-${t.id}`}
                type="checkbox"
                checked={tools.includes(t.id)}
                onChange={(e) => onToolsChange(e.target.checked ? [...tools, t.id] : tools.filter((x) => x !== t.id))}
                className={box}
              />
              <span>
                <span className="block text-sm font-medium text-text-primary">{t.label}</span>
                <span className="mt-0.5 block text-xs text-text-secondary">{t.hint}</span>
              </span>
            </label>
            <details className="group mt-2 pl-8">
              <summary className="inline-flex min-h-10 cursor-pointer list-none items-center gap-1 text-xs font-medium text-text-secondary hover:text-text-primary">
                <ChevronDown className="size-3.5 transition-transform group-open:rotate-180" aria-hidden /> Details
              </summary>
              <p className="mt-1 text-xs text-text-secondary">{t.details}</p>
            </details>
          </div>
        ))}
        <div className="flex flex-wrap items-center gap-3 border-t border-border pt-4">
          <Link
            href="/dashboard/approvals"
            className="inline-flex min-h-10 items-center gap-2 rounded-full border border-border bg-bg-surface px-4 text-sm font-medium text-text-primary transition-colors hover:border-text-primary focus-visible:outline-2 focus-visible:outline-accent-primary"
          >
            <CalendarCheck className="size-4" aria-hidden /> Review requests on Approvals
          </Link>
          {!!pending && <Badge tone="warning">{pending} waiting for you</Badge>}
        </div>
      </section>

      <InstantAnswersPanel agent={agent} />
    </div>
  );
}
