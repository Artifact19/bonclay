"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowRight, Check, Clock } from "lucide-react";
import { Button, Skeleton, TaskProgressBar } from "@/components/ui";
import { useOnboarding } from "@/hooks/useApi";
import { cn } from "@/lib/utils";
import type { HomeOnboarding, SetupStep } from "./homeTypes";

/** The old "Hide" button wrote a different key; a new key means nobody starts out with this list already hidden. */
const DISMISS_KEY = "bonclay_setup_done_dismissed";

function readDismissed(): boolean {
  try {
    return localStorage.getItem(DISMISS_KEY) === "1";
  } catch {
    return false;
  }
}

function ProgressRing({ completed, total }: { completed: number; total: number }) {
  const r = 16;
  const c = 2 * Math.PI * r;
  const frac = total > 0 ? completed / total : 0;
  return (
    <div className="relative size-11 shrink-0" role="img" aria-label={`${completed} of ${total} steps done`}>
      <svg viewBox="0 0 40 40" className="size-full -rotate-90">
        <circle cx="20" cy="20" r={r} fill="none" strokeWidth="4" className="stroke-border" />
        <circle
          cx="20"
          cy="20"
          r={r}
          fill="none"
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - frac)}
          className="stroke-accent-success transition-[stroke-dashoffset] duration-500 motion-reduce:transition-none"
        />
      </svg>
      <span className="absolute inset-0 grid place-items-center text-[11px] font-semibold text-text-primary">
        {completed}/{total}
      </span>
    </div>
  );
}

function Row({ step, next }: { step: SetupStep; next: boolean }) {
  return (
    <li className="flex items-center gap-3 py-3">
      <span
        aria-hidden
        className={cn(
          "grid size-6 shrink-0 place-items-center rounded-full border",
          step.done ? "border-accent-success bg-accent-success/15 text-accent-success" : "border-border text-transparent"
        )}
      >
        <Check className="size-3.5" />
      </span>
      <div className="min-w-0 flex-1">
        <p className={cn("text-sm font-medium", step.done ? "text-text-secondary line-through" : "text-text-primary")}>
          {step.title}
          {step.done && <span className="sr-only"> (done)</span>}
        </p>
        {/* On a phone only the next step explains itself, so the whole list stays short. */}
        {!step.done && (
          <p className={cn("text-xs text-text-secondary", next ? "mt-0.5" : "mt-0.5 hidden sm:block")}>{step.why}</p>
        )}
      </div>
      {!step.done && (
        <>
          <span className="hidden shrink-0 items-center gap-1 text-xs text-text-muted sm:inline-flex">
            <Clock className="size-3.5" aria-hidden />
            {step.minutes} min
          </span>
          <Link
            href={step.href}
            className={cn(
              "inline-flex min-h-11 shrink-0 items-center gap-1.5 rounded-full border px-4 text-sm font-medium transition-colors sm:min-h-10",
              next
                ? "border-transparent bg-brand text-[var(--color-on-accent)] hover:opacity-90"
                : "border-border bg-bg-surface text-text-primary hover:border-text-primary"
            )}
          >
            {step.cta}
            <span className="sr-only"> ({step.minutes} minutes)</span>
            <ArrowRight className="size-3.5" aria-hidden />
          </Link>
        </>
      )}
    </li>
  );
}

/** The six things a business does to get an agent answering its phone. Stays until all are done, then can be dismissed. */
export function FirstRunChecklist() {
  const { data: raw, isLoading, error } = useOnboarding();
  const data = raw as HomeOnboarding | undefined;
  const [dismissed, setDismissed] = useState(readDismissed);

  const hide = () => {
    setDismissed(true);
    try {
      localStorage.setItem(DISMISS_KEY, "1");
    } catch {
      /* dismissal just will not persist */
    }
  };

  if (error) return null;
  if (isLoading || !data) return <Skeleton className="h-56 w-full rounded-xl" />;
  const setup = data.setup;
  if (!setup?.length) return null;

  const completed = setup.filter((s) => s.done).length;
  const total = setup.length;
  const allDone = completed >= total;
  if (allDone && dismissed) return null;

  if (allDone) {
    return (
      <section aria-labelledby="setup-title" className="flex items-center gap-3 rounded-xl border border-border bg-bg-surface p-4">
        <span className="grid size-9 shrink-0 place-items-center rounded-full bg-accent-success/15 text-accent-success">
          <Check className="size-5" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <h2 id="setup-title" className="text-sm font-semibold text-text-primary">
            You are all set
          </h2>
          <p className="text-xs text-text-secondary">Your agent is ready for callers. All {total} setup steps are done.</p>
        </div>
        <Button variant="outline" size="md" onClick={hide}>
          Dismiss
        </Button>
      </section>
    );
  }

  const nextId = setup.find((s) => !s.done)?.id;
  const minutesLeft = setup.filter((s) => !s.done).reduce((n, s) => n + s.minutes, 0);
  const pct = Math.round((completed / total) * 100);

  return (
    <section aria-labelledby="setup-title" className="rounded-xl border border-border bg-bg-surface p-4 sm:p-5">
      <div className="flex items-center gap-3">
        <ProgressRing completed={completed} total={total} />
        <div className="min-w-0 flex-1">
          <h2 id="setup-title" className="text-sm font-semibold text-text-primary">
            Set up your agent
          </h2>
          <p className="text-xs text-text-secondary">
            {completed} of {total} done. About {minutesLeft} minutes to go.
          </p>
        </div>
      </div>
      <TaskProgressBar
        value={pct}
        label="Checklist completion"
        detail={`${completed}/${total} steps`}
        tone="success"
        size="sm"
        className="mt-3"
        testId="checklist-progress"
      />
      <ol className="mt-2 divide-y divide-border">
        {setup.map((step) => (
          <Row key={step.id} step={step} next={step.id === nextId} />
        ))}
      </ol>
    </section>
  );
}
