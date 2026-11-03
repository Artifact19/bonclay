"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

export interface TaskProgressBarProps {
  /** Percentage from 0 to 100. */
  value: number;
  /** Accessible and visible label describing the current task or stage. */
  label?: string;
  /** Secondary text shown beside the percentage (e.g., "2 of 5 ready" or "about 10 s left"). */
  detail?: string | null;
  /** Visual tone of the bar. */
  tone?: "accent" | "success" | "warning" | "error";
  /** Size variant. */
  size?: "sm" | "md" | "lg";
  /** Show the numeric percentage pill badge. Defaults to true. */
  showPercent?: boolean;
  /** Animate a subtle active pulse while work is in progress. */
  active?: boolean;
  className?: string;
  testId?: string;
}

/**
 * Aesthetic, UI-friendly percentage progress bar with a pill badge (`XX%`), stage label, and smooth fill.
 * Built for the BonClay editorial bento theme and accessible to screen readers via `role="progressbar"`.
 */
export function TaskProgressBar({
  value,
  label,
  detail,
  tone = "accent",
  size = "md",
  showPercent = true,
  active = false,
  className,
  testId = "task-progress-bar",
}: TaskProgressBarProps) {
  const pct = Math.max(0, Math.min(100, Math.round(Number.isFinite(value) ? value : 0)));

  const fillTone =
    tone === "error"
      ? "bg-[var(--color-accent-error,#E8632B)]"
      : tone === "success"
        ? "bg-[var(--color-accent-success,#2E7D5B)]"
        : tone === "warning"
          ? "bg-[var(--color-accent-warning,#E8632B)]"
          : "bg-[var(--color-accent-primary,#E8632B)]";

  const barHeight = size === "sm" ? "h-2" : size === "lg" ? "h-3.5" : "h-2.5";

  return (
    <div className={cn("w-full select-none", className)} data-testid={testId}>
      {(label || detail || showPercent) && (
        <div className="mb-1.5 flex items-center justify-between gap-2 text-xs">
          <div className="flex min-w-0 items-center gap-2">
            {active && (
              <span
                aria-hidden
                className="inline-block size-2 shrink-0 rounded-full bg-[var(--color-accent-primary,#E8632B)] motion-safe:animate-ping"
              />
            )}
            {label && (
              <span className="truncate font-medium text-[var(--color-text-primary,#141414)]">
                {label}
              </span>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {detail && (
              <span className="text-[11px] text-[var(--color-text-secondary,#52504C)] tabular-nums">
                {detail}
              </span>
            )}
            {showPercent && (
              <span
                data-testid={`${testId}-pct`}
                className="inline-flex items-center rounded-full border border-[var(--color-border,#141414)] bg-[#F3F59E] px-2 py-0.5 font-mono text-[11px] font-semibold leading-none text-[#141414] tabular-nums"
              >
                {pct}%
              </span>
            )}
          </div>
        </div>
      )}
      <div
        role="progressbar"
        aria-label={label || "Task progress"}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        className={cn(
          "w-full overflow-hidden rounded-full border border-[var(--color-border,#141414)] bg-[var(--color-bg-elevated,#E2DDD3)] p-[1px]",
          barHeight
        )}
      >
        <div
          className={cn(
            "h-full rounded-full transition-[width] duration-300 ease-out motion-reduce:transition-none",
            fillTone
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Smoothly estimates percentage progress (0 -> 94% while active, 100% when done) for single-request tasks
 * such as speech generation, audio verification, or document indexing where the server returns at completion.
 */
export function useEstimatedProgress(
  active: boolean,
  expectedSeconds: number = 18,
  completed: boolean = false
): number {
  const [pct, setPct] = useState(0);

  useEffect(() => {
    if (!active) return;
    const tau = Math.max(4, expectedSeconds * 0.65) * 1000;
    const started = Date.now();
    const id = setInterval(() => {
      const elapsed = Date.now() - started;
      const next = Math.min(94, Math.round(8 + 86 * (1 - Math.exp(-elapsed / tau))));
      setPct(next);
    }, 400);
    return () => clearInterval(id);
  }, [active, expectedSeconds]);

  return active ? Math.max(8, pct) : completed ? 100 : 0;
}

