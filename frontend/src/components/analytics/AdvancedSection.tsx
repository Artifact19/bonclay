"use client";

import { useMemo, useState, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChevronDown } from "lucide-react";
import { Card, ErrorState, Skeleton } from "@/components/ui";
import { PerformancePanel } from "@/components/dashboard/PerformancePanel";
import { languageName, titleCase } from "@/components/dashboard/format";
import { useAnalytics } from "@/hooks/useApi";
import type { Range } from "./results";

type Row = { name: string; value: number };
type Day = { date: string; value: number };

const PALETTE = [
  "var(--color-accent-primary)",
  "var(--color-accent-secondary)",
  "var(--color-accent-success)",
  "var(--color-accent-warning)",
  "var(--color-accent-error)",
];

const AXIS = { fill: "var(--color-text-secondary)", fontSize: 12 };

export const tooltipProps = {
  contentStyle: {
    background: "var(--color-bg-elevated)",
    border: "1px solid var(--color-border)",
    borderRadius: 8,
    color: "var(--color-text-primary)",
    fontSize: 12,
  },
  itemStyle: { color: "var(--color-text-primary)" },
  labelStyle: { color: "var(--color-text-secondary)" },
  cursor: { fill: "var(--color-bg-elevated)", opacity: 0.6 },
} as const;

const num = (v: unknown): number => {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : 0;
};

/** Accept either [{engine|language|name, count|value|...}] or {name: value}. */
function toRows(input: unknown): Row[] {
  if (!input) return [];
  if (Array.isArray(input)) {
    return input
      .map((it) => {
        if (!it || typeof it !== "object") return null;
        const o = it as Record<string, unknown>;
        const name = String(o.engine ?? o.language ?? o.name ?? o.label ?? o.key ?? "Unknown");
        const value = num(o.count ?? o.value ?? o.generations ?? o.avg_ms ?? o.avg ?? o.latency_ms ?? o.latency ?? o.p50);
        return { name, value };
      })
      .filter((r): r is Row => r !== null);
  }
  if (typeof input === "object") {
    return Object.entries(input as Record<string, unknown>).map(([k, v]) => ({
      name: k,
      value: typeof v === "object" && v ? num((v as Record<string, unknown>).avg_ms ?? (v as Record<string, unknown>).avg ?? (v as Record<string, unknown>).count) : num(v),
    }));
  }
  return [];
}

function toDaily(input: unknown): Day[] {
  if (!Array.isArray(input)) return [];
  return input.map((it) => {
    const o = (it ?? {}) as Record<string, unknown>;
    return { date: String(o.date ?? o.day ?? ""), value: num(o.count ?? o.generations ?? o.value) };
  });
}

export function shortDate(d: string) {
  const t = new Date(d);
  return Number.isNaN(t.getTime()) ? d : t.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** A raw engine id as a plain name, with the id kept beside it (this section is the technical one). */
export function engineLabel(id: string): string {
  const k = id.toLowerCase();
  if (k === "kokoro") return "Standard voice (Kokoro)";
  if (k.startsWith("indic")) return "Indian-language voice (Indic-Speak)";
  if (k === "dialogue") return "Two-person dialogue";
  if (k.startsWith("dhvaani")) return "Indian-language voice (DhVaani)";
  if (k.startsWith("chatterbox")) return "My voice (Chatterbox)";
  if (k.startsWith("qwen3") || k.startsWith("audio8") || k.includes("clone")) return `My voice (${titleCase(id)})`;
  return titleCase(id);
}

function ChartCard({
  title,
  description,
  loading,
  empty,
  emptyText,
  children,
  className,
}: {
  title: string;
  description?: string;
  loading: boolean;
  empty: boolean;
  emptyText: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <Card variant="solid" className={className}>
      <div className="mb-4">
        <h3 className="font-display text-base font-semibold text-text-primary">{title}</h3>
        {description && <p className="mt-0.5 text-xs text-text-secondary">{description}</p>}
      </div>
      <div className="h-64">
        {loading ? (
          <Skeleton className="h-full w-full rounded-xl" />
        ) : empty ? (
          <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-border px-4 text-center text-sm text-text-secondary">
            {emptyText}
          </div>
        ) : (
          children
        )}
      </div>
    </Card>
  );
}

function AdvancedBody({ range, agentId }: { range: Range; agentId: string | null }) {
  const { data, error, isLoading, mutate } = useAnalytics(range);
  const model = useMemo(() => {
    const d = (data ?? {}) as Record<string, unknown>;
    return {
      engine: toRows(d.by_engine).map((r) => ({ ...r, name: engineLabel(r.name) })),
      language: toRows(d.by_language).map((r) => ({ ...r, name: languageName(r.name) })),
      daily: toDaily(d.daily ?? d.series),
      latency: toRows(d.latency).map((r) => ({ ...r, name: engineLabel(r.name) })),
    };
  }, [data]);
  const loading = isLoading && !data;
  const emptyText = "No audio clips in this period yet. Make one in Say it once or the Studio to see it here.";

  return (
    <div className="mt-4 space-y-4">
      <p className="text-xs text-text-secondary">
        The three charts below count audio clips made in Say it once and the Studio. They are not split by agent
        {agentId ? ", so they show every agent" : ""}.
      </p>
      {error && !data ? (
        <ErrorState title="Could not load audio clip details" message={error.message} onRetry={() => mutate()} />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <ChartCard title="Audio clips made, by voice engine" loading={loading} empty={!model.engine.length} emptyText={emptyText}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={model.engine} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
                <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={AXIS} tickLine={false} axisLine={false} allowDecimals={false} />
                <YAxis type="category" dataKey="name" tick={AXIS} tickLine={false} axisLine={false} width={150} />
                <Tooltip {...tooltipProps} />
                <Bar dataKey="value" name="Audio clips" radius={[0, 6, 6, 0]} isAnimationActive={false}>
                  {model.engine.map((_, i) => (
                    <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Audio clips made, by language" loading={loading} empty={!model.language.length} emptyText={emptyText}>
            <div className="flex h-full flex-col sm:flex-row sm:items-center">
              <div className="min-h-0 flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Tooltip {...tooltipProps} />
                    <Pie data={model.language} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="86%" paddingAngle={2} stroke="var(--color-bg-surface)" isAnimationActive={false}>
                      {model.language.map((_, i) => (
                        <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <ul className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-text-secondary sm:w-48 sm:flex-col">
                {model.language.map((l, i) => (
                  <li key={l.name} className="flex items-center gap-2">
                    <span aria-hidden className="size-2.5 rounded-sm" style={{ background: PALETTE[i % PALETTE.length] }} />
                    <span className="truncate">{l.name}</span>
                    <span className="tabular-nums text-text-muted">{l.value}</span>
                  </li>
                ))}
              </ul>
            </div>
          </ChartCard>

          <ChartCard title="Audio clips made per day" description={range === "7d" ? "Last 7 days" : "Last 30 days"} loading={loading} empty={!model.daily.length} emptyText={emptyText} className="lg:col-span-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={model.daily} margin={{ top: 4, right: 12, left: -16, bottom: 0 }}>
                <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tickFormatter={shortDate} tick={AXIS} tickLine={false} axisLine={{ stroke: "var(--color-border)" }} minTickGap={24} />
                <YAxis tick={AXIS} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip {...tooltipProps} labelFormatter={(l) => shortDate(String(l))} />
                <Line type="monotone" dataKey="value" name="Audio clips" stroke="var(--color-accent-primary)" strokeWidth={2.5} dot={{ r: 3, fill: "var(--color-accent-primary)", strokeWidth: 0 }} activeDot={{ r: 5 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="Time to make an audio clip, by voice engine"
            description="Milliseconds until the audio is ready. This is about audio clips; your agent's conversation reply times are at the top of the page."
            loading={loading}
            empty={!model.latency.length}
            emptyText="Not measured yet. This fills in when you make audio clips."
            className="lg:col-span-2"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={model.latency} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
                <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={AXIS} tickLine={false} axisLine={false} unit=" ms" />
                <YAxis type="category" dataKey="name" tick={AXIS} tickLine={false} axisLine={false} width={150} />
                <Tooltip {...tooltipProps} formatter={(v) => [`${Math.round(num(v))} ms`, "Time"]} />
                <Bar dataKey="value" name="Time" fill="var(--color-accent-secondary)" radius={[0, 6, 6, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}

      <div>
        <h3 className="mb-1 font-display text-base font-semibold text-text-primary">Where reply time goes</h3>
        <p className="mb-3 text-xs text-text-secondary">
          The same reply times the Overview page shows, split into hearing, looking up your documents, thinking and speaking.
        </p>
        <PerformancePanel agentId={agentId} hideHeading />
      </div>
    </div>
  );
}

/** Collapsed by default: audio clip charts by engine and the technical reply-time breakdown. Loads only when opened. */
export function AdvancedSection({ range, agentId }: { range: Range; agentId: string | null }) {
  const [open, setOpen] = useState(false);
  return (
    <section aria-labelledby="advanced-title" className="mt-8">
      <button
        type="button"
        id="advanced-title"
        aria-expanded={open}
        aria-controls="advanced-body"
        onClick={() => setOpen((o) => !o)}
        className="flex min-h-11 w-full items-center justify-between gap-3 rounded-xl border border-border bg-bg-surface px-4 py-2 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary"
      >
        <span>
          <span className="block font-display text-base font-semibold text-text-primary">Advanced</span>
          <span className="block text-xs text-text-secondary">Audio clips made, voice engines and a detailed reply-time breakdown</span>
        </span>
        <ChevronDown className={`size-5 shrink-0 text-text-secondary transition-transform ${open ? "rotate-180" : ""}`} aria-hidden="true" />
      </button>
      <div id="advanced-body" hidden={!open}>
        {open && <AdvancedBody range={range} agentId={agentId} />}
      </div>
    </section>
  );
}
