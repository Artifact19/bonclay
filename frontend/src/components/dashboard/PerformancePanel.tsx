"use client";

import Link from "next/link";
import { usePerfTurns } from "@/hooks/useApi";
import { llmLabel } from "@/lib/llm";
import { Card, Skeleton } from "@/components/ui";
import type { PerfStageTimings, PerfTurn } from "@/types/api";

export const PERF_SUBTITLE = "Your last 20 spoken replies, from the caller finishing to your agent answering";

const MAX_TURNS = 20;

/** Marigold for the language model; the other stages are ink tints that differ by lightness in both themes. */
const SEGMENTS: { key: keyof PerfStageTimings; label: string; color: string }[] = [
  { key: "asr_ms", label: "Speech-to-text", color: "color-mix(in srgb, var(--color-text-primary) 78%, transparent)" },
  { key: "retrieval_ms", label: "Document search", color: "color-mix(in srgb, var(--color-text-primary) 48%, transparent)" },
  { key: "llm_ms", label: "Language model", color: "var(--color-accent-primary)" },
  { key: "tts_ms", label: "Voice", color: "color-mix(in srgb, var(--color-text-primary) 24%, transparent)" },
];

export const fmtSeconds = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
const fmtPct = (n: number) => `${Number.isInteger(n) ? n : n.toFixed(1)}%`;

interface Props {
  /** Limit to one agent (the Playground drawer passes the selected agent). */
  agentId?: string | null;
  /** Hide the title and subtitle when a surrounding dialog already shows them. */
  hideHeading?: boolean;
  className?: string;
}

function Stat({ label, ms, target, small = false }: { label: string; ms: number | null; target: number; small?: boolean }) {
  const over = ms != null && ms > target;
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-text-secondary">{label}</p>
      <p
        className={`font-display ${small ? "text-3xl" : "text-4xl"} font-semibold tabular-nums tracking-tight ${over ? "text-accent-error" : "text-text-primary"}`}
      >
        {ms == null ? "—" : fmtSeconds(ms)}
      </p>
    </div>
  );
}

/** Usual (p50) and slowest (p95) reply time on a 0..scale track with the target marked, so "under 3 s" is visible at a glance. */
function TargetTrack({ p50, p95, target }: { p50: number; p95: number; target: number }) {
  const scale = Math.max(p95, target) * 1.15;
  const at = (ms: number) => `${Math.min(100, (ms / scale) * 100)}%`;
  return (
    <div
      className="relative mt-4 h-2 rounded-full bg-bg-elevated"
      role="img"
      aria-label={`Usual reply time ${fmtSeconds(p50)}, slowest replies ${fmtSeconds(p95)}, target ${fmtSeconds(target)}`}
    >
      <div className="absolute inset-y-0 left-0 rounded-full bg-accent-primary/30" style={{ width: at(p95) }} />
      <div className="absolute inset-y-0 left-0 rounded-full bg-accent-primary" style={{ width: at(p50) }} />
      <div className="absolute -inset-y-1.5 w-px bg-text-primary" style={{ left: at(target) }} aria-hidden="true" />
      <span
        className="absolute top-3 -translate-x-1/2 text-[11px] tabular-nums text-text-secondary"
        style={{ left: at(target) }}
        aria-hidden="true"
      >
        {fmtSeconds(target)} target
      </span>
    </div>
  );
}

function TurnBar({ turn, scale, target }: { turn: PerfTurn; scale: number; target: number }) {
  const parts = SEGMENTS.map((s) => ({ ...s, ms: turn.timings[s.key] ?? 0 })).filter((s) => s.ms > 0);
  const summary = parts.map((s) => `${s.label} ${s.ms} ms`).join(", ");
  const first = turn.first_audio_ms ?? null;
  const provider = turn.llm_provider ? llmLabel(turn.llm_provider) : null;
  return (
    <li className="grid grid-cols-[3.5rem_1fr] items-center gap-x-3 gap-y-0.5 sm:grid-cols-[3.5rem_1fr_9rem]">
      <span
        className={`text-xs font-medium tabular-nums ${turn.total_ms > target ? "text-accent-error" : "text-text-primary"}`}
      >
        {fmtSeconds(turn.total_ms)}
      </span>
      <div className="relative h-4" role="img" aria-label={`${fmtSeconds(turn.total_ms)} total. ${first != null ? `First sound ${fmtSeconds(first)}. ` : ""}${summary}`}>
        <div className="flex h-full gap-px overflow-hidden rounded-sm" style={{ width: `${(turn.total_ms / scale) * 100}%` }}>
          {parts.map((s) => (
            <div
              key={s.key}
              title={`${s.label}: ${s.ms} ms`}
              className="h-full min-w-px"
              style={{ flexGrow: s.ms, flexBasis: 0, background: s.color }}
            />
          ))}
        </div>
        {first != null && (
          <div
            title={`First sound: ${first} ms`}
            className="pointer-events-none absolute -inset-y-1 w-0.5 rounded-full bg-text-primary"
            style={{ left: `${Math.min(100, (first / scale) * 100)}%` }}
            aria-hidden="true"
          />
        )}
        <div
          className="pointer-events-none absolute -inset-y-0.5 border-l border-dashed border-text-secondary/60"
          style={{ left: `${(target / scale) * 100}%` }}
          aria-hidden="true"
        />
      </div>
      <span className="col-start-2 truncate text-[11px] text-text-secondary sm:col-start-3">
        {[turn.fallback ? "fallback reply" : null, turn.engine, provider].filter(Boolean).join(" · ") || "—"}
      </span>
    </li>
  );
}

/** "Reply under 3 seconds" as a number: p50 / p95 and a per-turn stage breakdown from GET /api/perf/turns. */
export function PerformancePanel({ agentId, hideHeading = false, className }: Props) {
  const { data, isLoading, error } = usePerfTurns(MAX_TURNS, agentId);
  const turns = (data?.turns ?? []).slice(0, MAX_TURNS);
  const target = data?.target_ms ?? 3000;
  // Streamed turns also measure the time until the caller hears the first sentence.
  const hasFirstAudio = data?.first_audio_p50_ms != null && data?.first_audio_p95_ms != null;

  return (
    <Card variant="solid" className={className} aria-labelledby={hideHeading ? undefined : "perf-heading"}>
      {!hideHeading && (
        <div className="mb-4">
          <h2 id="perf-heading" className="text-sm font-semibold text-text-primary">
            Reply time
          </h2>
          <p className="mt-0.5 text-xs text-text-secondary">{PERF_SUBTITLE}</p>
        </div>
      )}

      {isLoading ? (
        <Skeleton className="h-40 w-full rounded-xl" />
      ) : error ? (
        <p className="text-sm text-text-secondary" role="alert">
          Could not load reply times. {error.message}
        </p>
      ) : !turns.length || data?.p50_ms == null || data.p95_ms == null ? (
        <div className="rounded-xl border border-dashed border-border px-4 py-8 text-center">
          <p className="text-sm text-text-secondary">
            No conversations yet —{" "}
            <Link href="/dashboard/playground" className="font-medium text-text-primary underline underline-offset-2">
              talk to an agent in the Playground
            </Link>
          </p>
        </div>
      ) : (
        <div className="space-y-5">
          <div>
            <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-2">
              <div className="flex flex-wrap gap-x-8 gap-y-3">
                {hasFirstAudio && (
                  <>
                    <Stat label="First sound (usual)" ms={data.first_audio_p50_ms ?? null} target={target} />
                    <Stat label="First sound (slowest)" ms={data.first_audio_p95_ms ?? null} target={target} />
                  </>
                )}
                <Stat label={hasFirstAudio ? "Full reply (usual)" : "Usual reply time"} ms={data.p50_ms} target={target} small={hasFirstAudio} />
                <Stat label={hasFirstAudio ? "Full reply (slowest)" : "Slowest replies"} ms={data.p95_ms} target={target} small={hasFirstAudio} />
              </div>
              <div className="space-y-0.5 text-sm text-text-secondary">
                {hasFirstAudio && data.first_audio_within_target_pct != null && (
                  <p>
                    <span className="font-semibold tabular-nums text-text-primary">
                      {fmtPct(data.first_audio_within_target_pct)}
                    </span>{" "}
                    first sound within {fmtSeconds(target)}
                  </p>
                )}
                {data.within_target_pct != null && (
                  <p>
                    <span className="font-semibold tabular-nums text-text-primary">{fmtPct(data.within_target_pct)}</span>{" "}
                    {hasFirstAudio ? "full reply " : ""}within {fmtSeconds(target)}
                  </p>
                )}
              </div>
            </div>
            <TargetTrack p50={data.p50_ms} p95={data.p95_ms} target={target} />
          </div>

          <div className="pt-3">
            <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-secondary" aria-label="Legend">
              {hasFirstAudio && (
                <li className="inline-flex items-center gap-1.5">
                  <span className="h-3 w-0.5 rounded-full bg-text-primary" aria-hidden="true" />
                  First sound
                </li>
              )}
              {SEGMENTS.map((s) => (
                <li key={s.key} className="inline-flex items-center gap-1.5">
                  <span className="size-2.5 rounded-sm" style={{ background: s.color }} aria-hidden="true" />
                  {s.label}
                  {data.stage_p50[s.key] != null && (
                    <span className="tabular-nums text-text-primary/80">{data.stage_p50[s.key]} ms</span>
                  )}
                </li>
              ))}
            </ul>
            <ul className="mt-3 space-y-1.5" aria-label="Recent turns, newest first">
              {(() => {
                const scale = Math.max(target * 1.1, ...turns.map((t) => t.total_ms));
                return turns.map((t) => <TurnBar key={t.id} turn={t} scale={scale} target={target} />);
              })()}
            </ul>
          </div>
        </div>
      )}
    </Card>
  );
}
