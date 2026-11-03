"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card } from "@/components/ui";
import { languageName, seconds } from "@/components/dashboard/format";
import { shortDate, tooltipProps } from "./AdvancedSection";
import type { Results } from "./results";

const AXIS = { fill: "var(--color-text-secondary)", fontSize: 12 };
const pct = (a: number, b: number) => (b > 0 ? Math.round((100 * a) / b) : 0);
const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`;

function Stat({
  label,
  value,
  note,
  tone = "default",
  testId,
}: {
  label: string;
  value: ReactNode;
  note?: ReactNode;
  tone?: "default" | "warn";
  testId?: string;
}) {
  return (
    <Card variant="solid" className="flex flex-col" data-testid={testId}>
      <p className="text-xs font-medium text-text-secondary">{label}</p>
      <p
        className={`mt-1 font-display text-3xl font-semibold tabular-nums tracking-tight ${
          tone === "warn" ? "text-accent-warning" : "text-text-primary"
        }`}
      >
        {value}
      </p>
      {note && <p className="mt-1.5 text-xs leading-relaxed text-text-secondary">{note}</p>}
    </Card>
  );
}

const NOT_TRACKED = <span className="text-xl font-medium text-text-muted">Not tracked yet</span>;

/** One sentence that answers "is my agent working?" from the counts. */
export function verdict(r: Results): string {
  const days = r.days;
  const who = r.agent_name ?? "Your agents";
  if (!r.questions_asked) return `${who} handled ${plural(r.conversations, "conversation")} in the last ${days} days.`;
  const share = pct(r.questions_answered, r.questions_asked);
  return `${who} answered ${r.questions_answered.toLocaleString()} of ${r.questions_asked.toLocaleString()} questions (${share}%) in the last ${days} days.`;
}

/** The headline numbers, in plain words. Anything the backend cannot derive says "Not tracked yet". */
export function HeadlineCards({ r }: { r: Results }) {
  const rt = r.reply_time;
  const waiting = r.bookings?.waiting ?? 0;
  return (
    <section aria-label="Headline numbers">
      <p className="mb-3 text-sm text-text-primary" data-testid="verdict">
        {verdict(r)}
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Stat label="Conversations handled" value={r.conversations.toLocaleString()} note="Real callers and chats. Your own practice calls are not counted." />
        <Stat
          label="Questions answered"
          value={r.questions_answered.toLocaleString()}
          note={r.questions_asked ? `Out of ${r.questions_asked.toLocaleString()} questions asked.` : "No questions asked yet."}
        />
        <Stat
          label="Questions it couldn't answer"
          value={r.questions_unanswered.toLocaleString()}
          tone={r.questions_unanswered > 0 ? "warn" : "default"}
          note={
            r.questions_unanswered > 0
              ? `${pct(r.questions_unanswered, r.questions_asked)}% of questions. See the list below to fix them.`
              : "Nothing it was unable to answer."
          }
        />
        {r.bookings ? (
          <Stat
            label="Meetings booked"
            value={r.bookings.booked.toLocaleString()}
            testId="stat-bookings"
            note={
              waiting > 0 ? (
                <>
                  <Link href="/dashboard/approvals" className="font-medium text-text-primary underline underline-offset-2">
                    {plural(waiting, "request")} waiting
                  </Link>{" "}
                  for your approval.
                </>
              ) : (
                `${plural(r.bookings.requested, "request")} made in this period. None waiting.`
              )
            }
          />
        ) : (
          <Stat
            label="Meetings booked"
            value={NOT_TRACKED}
            testId="stat-bookings"
            note="Meeting requests are not split by agent. Choose All agents to see them."
          />
        )}
        <Stat
          label="Usual reply time"
          value={rt.p50_ms == null ? NOT_TRACKED : seconds(rt.p50_ms)}
          testId="stat-p50"
          note={
            rt.p50_ms == null
              ? "Shows up after your agent has replied to someone."
              : `Half of replies are faster than this${rt.within_target_pct != null ? `; ${Math.round(rt.within_target_pct)}% come within ${seconds(rt.target_ms)}` : ""}.`
          }
        />
        <Stat
          label="Slowest replies"
          value={rt.p95_ms == null ? NOT_TRACKED : seconds(rt.p95_ms)}
          testId="stat-p95"
          tone={rt.p95_ms != null && rt.p95_ms > rt.target_ms ? "warn" : "default"}
          note={rt.p95_ms == null ? undefined : "Only about 1 reply in 20 takes longer than this."}
        />
      </div>
    </section>
  );
}

/** Questions per day (answered and not), and the languages callers used. */
export function ActivityCard({ r }: { r: Results }) {
  const data = r.daily.map((d) => ({ date: d.date, answered: Math.max(0, d.asked - d.unanswered), unanswered: d.unanswered }));
  const langs = r.by_language;
  const total = langs.reduce((n, l) => n + l.count, 0);
  return (
    <Card variant="solid" role="region" aria-labelledby="activity-title">
      <h2 id="activity-title" className="font-display text-base font-semibold text-text-primary">
        Questions per day
      </h2>
      <p className="mt-0.5 text-xs text-text-secondary">Questions callers asked each day, and how many your agent could not answer.</p>
      <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-[2fr_1fr]">
        <div className="h-52" role="img" aria-label={`${r.questions_asked} questions in ${r.days} days, ${r.questions_unanswered} not answered`}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="date" tickFormatter={shortDate} tick={AXIS} tickLine={false} axisLine={{ stroke: "var(--color-border)" }} minTickGap={24} />
              <YAxis tick={AXIS} tickLine={false} axisLine={false} allowDecimals={false} />
              <Tooltip {...tooltipProps} labelFormatter={(l) => shortDate(String(l))} />
              <Bar dataKey="answered" name="Answered" stackId="q" fill="var(--color-accent-success)" isAnimationActive={false} />
              <Bar dataKey="unanswered" name="Couldn't answer" stackId="q" fill="var(--color-accent-warning)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h3 className="text-sm font-semibold text-text-primary">Languages callers used</h3>
          {langs.length === 0 ? (
            <p className="mt-2 text-xs text-text-secondary">Not recorded for these conversations.</p>
          ) : (
            <ul className="mt-2 space-y-2">
              {langs.slice(0, 6).map((l) => (
                <li key={l.language} className="text-sm">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-text-primary">{languageName(l.language)}</span>
                    <span className="tabular-nums text-text-secondary">{pct(l.count, total)}%</span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-bg-elevated" aria-hidden="true">
                    <div className="h-full rounded-full bg-accent-primary" style={{ width: `${pct(l.count, total)}%` }} />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </Card>
  );
}
