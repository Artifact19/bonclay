import { Zap } from "lucide-react";
import { llmLabel } from "@/lib/llm";

export interface ReplyTimingInfo {
  latencyMs?: number;
  firstAudioMs?: number;
  engineName?: string;
  llmProvider?: string | null;
  llmModel?: string | null;
  routeReason?: string;
  timings?: Record<string, number>;
  grounded?: boolean;
  sourcesUsed?: number;
}

/** What each stage of a reply is called to a person (the keys are the server's timing fields). */
export const STAGES: [string, string][] = [
  ["asr_ms", "Listening"],
  ["retrieval_ms", "Looking up your documents"],
  ["llm_ms", "Thinking"],
  ["tools_ms", "Tools"],
  ["tts_ms", "Speaking"],
];

/** "1.2 s": one decimal, which is as exact as a person needs. */
export const seconds = (ms: number) => `${(ms / 1000).toFixed(1)} s`;

/** Replies slower than this get a gentle note. */
const SLOW_MS = 4000;

export function hasTimingInfo(i: ReplyTimingInfo): boolean {
  return i.latencyMs != null || !!i.engineName || !!i.llmProvider || i.grounded != null;
}

/**
 * "Replied in 1.2 s" in plain words. Engine names, model names and the per-stage milliseconds sit behind
 * a Details disclosure, never as the first thing a person reads.
 */
export function ReplyTiming({ info }: { info: ReplyTimingInfo }) {
  const stages = STAGES.filter(([k]) => info.timings?.[k] != null);
  const hasDetails = info.firstAudioMs != null || !!info.engineName || !!info.llmProvider || info.grounded != null || stages.length > 0;
  return (
    <div className="flex flex-col gap-1 text-xs text-[var(--color-text-secondary)]">
      {info.latencyMs != null && (
        <p className="inline-flex items-center gap-1 tabular-nums">
          <Zap className="h-3 w-3" aria-hidden="true" />
          Replied in {seconds(info.latencyMs)}
          {info.latencyMs > SLOW_MS ? " (slower than usual)" : ""}
        </p>
      )}
      {hasDetails && (
        <details className="group">
          <summary className="inline-flex min-h-6 cursor-pointer list-none items-center gap-1 text-[var(--color-text-secondary)] underline-offset-2 hover:text-[var(--color-text-primary)] hover:underline focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)] [&::-webkit-details-marker]:hidden">
            Details
          </summary>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5 tabular-nums">
            {info.firstAudioMs != null && (
              <span title="From the end of your question to the first sentence being ready to play" className="rounded bg-[var(--color-bg-elevated)] px-1.5 py-0.5">
                First sentence ready after {seconds(info.firstAudioMs)}
              </span>
            )}
            {info.latencyMs != null && info.firstAudioMs != null && (
              <span className="rounded bg-[var(--color-bg-elevated)] px-1.5 py-0.5">Whole reply {seconds(info.latencyMs)}</span>
            )}
            {info.engineName && (
              <span title={info.routeReason} className="rounded bg-[hsl(var(--au-accent)/0.15)] px-1.5 py-0.5 text-[var(--color-accent-secondary)]">
                {info.engineName}
              </span>
            )}
            {info.llmProvider && (
              <span title={info.llmModel ? `Answered by ${info.llmModel}` : undefined} className="rounded bg-[var(--color-bg-elevated)] px-1.5 py-0.5">
                LLM · {llmLabel(info.llmProvider, info.llmModel)}
              </span>
            )}
            {info.grounded != null && (
              <span className="rounded bg-[var(--color-bg-elevated)] px-1.5 py-0.5">
                {info.grounded
                  ? `Answered from your documents (${info.sourcesUsed ?? 0} ${info.sourcesUsed === 1 ? "section" : "sections"})`
                  : "Nothing in your documents matched"}
              </span>
            )}
            {stages.map(([k, label]) => (
              <span key={k} className="rounded bg-[var(--color-bg-elevated)] px-1.5 py-0.5">
                {label} {info.timings![k]} ms
              </span>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
