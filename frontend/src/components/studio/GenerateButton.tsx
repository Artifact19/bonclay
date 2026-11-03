"use client";

import { useEffect, useId, useState } from "react";
import { Loader2, Play, RotateCcw, Sparkles, Square } from "lucide-react";
import { TaskProgressBar, useEstimatedProgress } from "@/components/ui/TaskProgressBar";

export type GenerateState = "idle" | "working" | "done";

/** What the working state says. */
export const WORKING_HINT = "Making audio. Usually 10 to 40 seconds.";

interface Props {
  state: GenerateState;
  disabled?: boolean;
  /** Why the button is disabled, in plain words. Shown under the button while it is disabled. */
  disabledReason?: string;
  idleLabel?: string;
  /** Replaces the default time estimate while working (a slower engine says so). */
  workingHint?: string;
  /** Optional explicit percentage (0-100) if caller streams progress; otherwise smoothly estimated. */
  progress?: number | null;
  /** The last attempt failed: shown beside the button with a Retry. Cleared by the parent on the next attempt. */
  error?: string | null;
  onGenerate: () => void;
  onPlay: () => void;
  /** Cancels the request in flight. Without it the working state has no Stop button. */
  onStop?: () => void;
  /** Retry after a failure (defaults to onGenerate). */
  onRetry?: () => void;
  /** "Make again" after success: the same text and voice (defaults to onGenerate). */
  onRegenerate?: () => void;
}

/** Whole seconds since it appeared (it is mounted only while working, so it starts at 0 each time). Not a live region. */
function Elapsed() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <span data-testid="generate-elapsed" className="tabular-nums opacity-80">
      {seconds} s
    </span>
  );
}

const primary =
  "relative flex h-12 min-w-0 flex-1 items-center justify-center gap-2 overflow-hidden rounded-xl px-5 text-sm font-semibold text-[var(--color-on-accent)] transition-[transform,opacity] duration-200 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent-cyan)]";
const secondary =
  "inline-flex h-12 shrink-0 items-center justify-center gap-2 rounded-xl border border-[var(--color-border)] px-4 text-sm font-medium text-[var(--color-text-primary)] transition-colors hover:bg-[hsl(var(--au-text-primary)/0.06)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent-cyan)]";

/** Generate -> making audio (with Stop & percentage progress bar) -> play, and Make again. Failures show here with a Retry. */
export function GenerateButton({
  state,
  disabled,
  disabledReason,
  idleLabel = "Generate speech",
  workingHint = WORKING_HINT,
  progress,
  error,
  onGenerate,
  onPlay,
  onStop,
  onRetry,
  onRegenerate,
}: Props) {
  const working = state === "working";
  const blocked = state === "idle" && !!disabled;
  const reasonId = useId();
  const estimatedPct = useEstimatedProgress(working, 20, state === "done");
  const shownPct = progress != null ? progress : estimatedPct;

  // Announced once per change: it says only that work started, finished or failed, never a tick.
  const announcement = error ? `Failed: ${error}` : working ? "Making audio" : state === "done" ? "Audio ready" : "";

  return (
    <div>
      <div className="flex gap-2">
        {state === "done" ? (
          <>
            <button type="button" onClick={onPlay} className={primary} style={{ background: "var(--gradient-brand)" }}>
              <Play className="h-4 w-4" aria-hidden="true" />
              Play result
            </button>
            <button type="button" onClick={onRegenerate ?? onGenerate} className={secondary}>
              <RotateCcw className="h-4 w-4" aria-hidden="true" />
              Make again
            </button>
          </>
        ) : (
          <>
            <button
              type="button"
              onClick={onGenerate}
              disabled={working || blocked}
              aria-busy={working || undefined}
              aria-describedby={blocked && disabledReason ? reasonId : undefined}
              className={primary}
              style={{ background: "var(--gradient-brand)" }}
            >
              {working ? (
                <>
                  <Loader2 className="h-4 w-4 motion-safe:animate-spin" aria-hidden="true" />
                  <span>Making audio</span>
                  <Elapsed />
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" aria-hidden="true" />
                  {idleLabel}
                </>
              )}
            </button>
            {working && onStop && (
              <button type="button" onClick={onStop} data-testid="generate-stop" className={secondary}>
                <Square className="h-3.5 w-3.5" aria-hidden="true" />
                Stop
              </button>
            )}
          </>
        )}
      </div>

      {working && (
        <div className="mt-3 space-y-1.5">
          <TaskProgressBar
            value={shownPct}
            label="Synthesizing voice audio"
            active
            testId="generate-progress"
          />
          <p data-testid="generate-hint" className="text-center text-sm text-[var(--color-text-secondary)]">
            {workingHint}
          </p>
        </div>
      )}
      {blocked && disabledReason && (
        <p id={reasonId} data-testid="generate-reason" className="mt-2 text-center text-sm text-[var(--color-text-secondary)]">
          {disabledReason}
        </p>
      )}
      {error && !working && (
        <div
          data-testid="generate-error"
          className="mt-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-2 rounded-lg border border-accent-error/40 bg-accent-error/10 px-3 py-2 text-sm text-[var(--color-text-primary)]"
        >
          <span className="min-w-0 break-words">{error}</span>
          <button
            type="button"
            onClick={onRetry ?? onGenerate}
            className="inline-flex min-h-10 items-center gap-1.5 rounded-lg border border-[var(--color-border)] px-3 text-sm font-medium hover:bg-[hsl(var(--au-text-primary)/0.06)] focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Retry
          </button>
        </div>
      )}

      {/* The one live region: changes only when work starts, ends or fails. */}
      <p role="status" aria-live="polite" data-testid="generate-status" className="sr-only">
        {announcement}
      </p>
    </div>
  );
}
