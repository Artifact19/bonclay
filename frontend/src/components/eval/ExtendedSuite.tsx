"use client";

import { CheckCircle2, TriangleAlert, XCircle } from "lucide-react";
import { Badge, Card } from "@/components/ui";
import { titleCase, relativeTime } from "@/components/dashboard/format";

export const QUICK_LABEL = "Quick check: 6 tricky callers";
export const EXTENDED_LABEL = "Longer check: 60 tricky calls, including Hindi and Hinglish";
export const TOOLS_LABEL = "Let the agent use its tools (for example booking)";

/** A tool call on one line, e.g. `book_appointment(name: "Asha", date: "2026-10-19") → done`. */
export function toolCallLine(c: ToolCallRecord): string {
  const args = Object.entries(c.arguments ?? {})
    .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
    .join(", ");
  return `${c.name}(${args}) → ${c.ok ? "done" : "refused"}`;
}

/** One tool call the agent made in a case (tools-on run): name, arguments, ok = it really went through. */
export interface ToolCallRecord {
  name: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  result: string;
}

/** The language check of a reply: what the policy asked for and whether the final reply was in it. */
export interface LanguageCheck {
  target: string;
  matched: boolean | null;
  retried: boolean;
  retry_matched: boolean | null;
  request?: string | null;
  unsupported?: string | null;
}

export interface ExtendedCase {
  id: string;
  category: string;
  script: string;
  caller_text: string;
  expected_behaviour: string;
  agent_reply: string;
  status: "judged" | "error";
  passed: boolean;
  invented_facts: boolean;
  broke_rules: boolean;
  followed_expected_behaviour: boolean;
  second_pass: boolean;
  agreed: boolean | null;
  disputed: boolean;
  reason: string;
  error?: string | null;
  /** null: the agent had no tools for this case */
  tool_calls?: ToolCallRecord[] | null;
  tool_violations?: string[];
  language?: LanguageCheck | null;
}

interface Agreement {
  checked: number;
  agreed: number;
  pct: number | null;
}

export interface ExtendedResult {
  run_id: string;
  total: number;
  judged: number;
  errors: number;
  passed: number;
  failed: number;
  overall_score: number;
  hallucinations: number;
  per_category: { category: string; total: number; passed: number; errors: number; pass_rate: number | null }[];
  judge_agreement: Agreement & { on_failed: Agreement; on_passed_sample: Agreement };
  failures: ExtendedCase[];
  error_cases: ExtendedCase[];
  models_used?: { agent?: Record<string, number>; judge?: Record<string, number> };
  with_tools?: boolean;
  tools?: { cases_with_tools: number; calls: number; violation_cases: number };
  language_retries?: number;
  results: ExtendedCase[];
}

export interface ExtendedProgressInfo {
  done: number;
  total: number;
  passed: number;
  failed: number;
  errors: number;
}

export interface ExtendedJob {
  status: "queued" | "running" | "done" | "failed";
  progress: ExtendedProgressInfo | null;
  result: ExtendedResult | null;
  error: { code: string; message: string } | null;
}

const scoreColor = (n: number) =>
  n >= 80 ? "text-[var(--color-accent-success)]" : n >= 50 ? "text-amber-400" : "text-red-400";

export function ExtendedProgress({ progress, total }: { progress: ExtendedProgressInfo | null; total: number }) {
  const done = progress?.done ?? 0;
  const all = progress?.total || total || 0;
  const pct = all ? Math.round((100 * done) / all) : 0;
  return (
    <div className="space-y-2">
      <p className="text-sm text-text-secondary">
        Tricky call {Math.min(done + 1, all)} of {all}: {done} done
        {progress ? ` · ${progress.passed} handled · ${progress.failed} slipped` : ""}
        {progress?.errors ? ` · ${progress.errors} could not run` : ""}. Two callers at a time; this can take several
        minutes.
      </p>
      <div
        role="progressbar"
        aria-label="Longer safety check progress"
        aria-valuemin={0}
        aria-valuemax={all}
        aria-valuenow={done}
        className="h-2 w-full overflow-hidden rounded-full bg-bg-elevated"
      >
        <div className="h-full rounded-full bg-accent-primary transition-[width] duration-500" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function ExtendedReport({ result }: { result: ExtendedResult }) {
  const ag = result.judge_agreement;
  const models = (Object.entries(result.models_used ?? {}) as [string, Record<string, number> | undefined][]).filter(
    ([, m]) => m && Object.keys(m).length
  );
  return (
    <div className="space-y-6" data-testid="extended-report">
      <Card variant="solid" className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-text-primary">
              Your agent handled {result.passed} of {result.judged} tricky calls
            </p>
            <p className="mt-1 text-xs text-text-secondary">
              {result.hallucinations
                ? `It made up facts in ${result.hallucinations} repl${result.hallucinations === 1 ? "y" : "ies"}.`
                : "It did not make up any facts."}
              {result.errors ? ` ${result.errors} could not run.` : ""}
            </p>
          </div>
          <div className="text-right">
            <p className={`text-4xl font-bold tabular-nums ${scoreColor(result.overall_score)}`}>{result.overall_score}%</p>
            <p className="text-xs text-text-secondary">safety score</p>
          </div>
        </div>
        <details className="text-xs text-text-secondary">
          <summary className="min-h-6 cursor-pointer hover:text-text-primary focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]">
            Details
          </summary>
          <div className="mt-2 space-y-1.5">
            <p>
              Second opinions:{" "}
              {ag.checked
                ? `${ag.agreed} of ${ag.checked} matched the first grader (${ag.pct === null ? "n/a" : `${ag.pct}%`})`
                : "no call needed one"}
            </p>
            {models.length > 0 && (
              <p>
                {models
                  .map(([role, m]) => `${role === "judge" ? "Grader" : "Agent replies"}: ${Object.entries(m ?? {}).map(([name, n]) => `${name} (${n})`).join(", ")}`)
                  .join(" · ")}
              </p>
            )}
          </div>
        </details>
      </Card>

      <section aria-labelledby="ext-cat-heading" className="space-y-3">
        <h2 id="ext-cat-heading" className="text-sm font-semibold text-text-primary">
          How it did with each kind of caller
        </h2>
        <Card variant="solid" padded={false} className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] text-left text-sm" aria-label="How it did with each kind of caller">
              <thead className="border-b border-border bg-bg-elevated text-xs text-text-secondary">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium">Kind of caller</th>
                  <th scope="col" className="px-4 py-3 text-right font-medium">Handled</th>
                  <th scope="col" className="w-1/3 px-4 py-3 font-medium">Success rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {result.per_category.map((c) => (
                  <tr key={c.category}>
                    <td className="px-4 py-3 font-medium text-text-primary">{titleCase(c.category)}</td>
                    <td className="px-4 py-3 text-right tabular-nums text-text-secondary">
                      {c.passed}/{c.total}
                      {c.errors ? ` (+${c.errors} not run)` : ""}
                    </td>
                    <td className="px-4 py-3">
                      {c.pass_rate === null ? (
                        <span className="text-xs text-text-secondary">n/a</span>
                      ) : (
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-bg-elevated">
                            <div
                              className={`h-full rounded-full ${c.pass_rate >= 80 ? "bg-accent-success" : c.pass_rate >= 50 ? "bg-accent-warning" : "bg-accent-error"}`}
                              style={{ width: `${c.pass_rate}%` }}
                            />
                          </div>
                          <span className="w-10 text-right text-xs tabular-nums text-text-primary">{c.pass_rate}%</span>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      <section aria-labelledby="ext-fail-heading" className="space-y-3">
        <h2 id="ext-fail-heading" className="text-sm font-semibold text-text-primary">
          Where it slipped ({result.failures.length})
        </h2>
        {result.failures.length === 0 ? (
          <Card variant="solid" className="flex items-center gap-2 text-sm text-text-secondary">
            <CheckCircle2 className="size-4 text-accent-success" aria-hidden /> No slips in this run.
          </Card>
        ) : (
          <ul className="space-y-3">
            {result.failures.map((f) => (
              <li key={f.id}>
                <Card variant="solid" className="space-y-2 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone="error">
                      <XCircle className="size-3.5" aria-hidden /> {titleCase(f.category)}
                    </Badge>
                    <span className="font-mono text-[11px] text-text-secondary">{f.id}</span>
                    {f.invented_facts && <Badge tone="warning">Made up a fact</Badge>}
                    {f.broke_rules && <Badge tone="warning">Broke a rule</Badge>}
                    {f.disputed && (
                      <Badge tone="neutral">
                        <TriangleAlert className="size-3.5" aria-hidden /> The graders disagree: read it yourself
                      </Badge>
                    )}
                  </div>
                  <p>
                    <span className="text-xs text-text-secondary">Caller said: </span>
                    <span className="text-text-primary">{f.caller_text}</span>
                  </p>
                  <p>
                    <span className="text-xs text-text-secondary">Agent replied: </span>
                    <span className="text-text-primary">{f.agent_reply}</span>
                  </p>
                  <p className="text-xs text-text-secondary">What it should have done: {f.expected_behaviour}</p>
                  {f.reason && <p className="text-xs text-text-secondary">Why: {f.reason}</p>}
                  {f.tool_calls != null && (
                    <div className="text-xs text-text-secondary" data-testid={`tool-calls-${f.id}`}>
                      <span>Tool calls: </span>
                      {f.tool_calls.length === 0 ? (
                        <span>none</span>
                      ) : (
                        <ul className="mt-1 space-y-0.5 font-mono text-[11px] text-text-primary">
                          {f.tool_calls.map((c, i) => (
                            <li key={i} title={c.result}>
                              {toolCallLine(c)}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                  {!!f.tool_violations?.length && (
                    <p className="text-xs text-red-400">Tool check: {f.tool_violations.join("; ")}</p>
                  )}
                  {f.language && f.language.matched === false && (
                    <p className="text-xs text-red-400">
                      Language check: the reply was not in {f.language.target}
                      {f.language.retried ? " (regenerated once, still wrong)" : ""}.
                    </p>
                  )}
                </Card>
              </li>
            ))}
          </ul>
        )}
      </section>

      {result.error_cases.length > 0 && (
        <p className="text-xs text-text-secondary">
          {result.error_cases.length} call{result.error_cases.length === 1 ? "" : "s"} could not run (no answer or grade came
          back) and are left out of the score.
        </p>
      )}
      {result.with_tools && result.tools && (
        <p className="text-xs text-text-secondary" data-testid="tools-summary">
          Run with the agent&apos;s tools on a private copy of the appointment data: {result.tools.calls} tool call
          {result.tools.calls === 1 ? "" : "s"}, {result.tools.violation_cases} case
          {result.tools.violation_cases === 1 ? "" : "s"} with a tool violation.
        </p>
      )}
      {!!result.language_retries && (
        <p className="text-xs text-text-secondary">
          {result.language_retries} repl{result.language_retries === 1 ? "y was" : "ies were"} regenerated once because{" "}
          {result.language_retries === 1 ? "it was" : "they were"} not in the caller&apos;s language.
        </p>
      )}
    </div>
  );
}

interface HistoryRow {
  run_id?: string;
  suite?: string;
  passed?: boolean | null;
  created_at?: string;
}

/** One line per past extended run (the stored rows are one per case). */
export function ExtendedHistory({ rows }: { rows: HistoryRow[] }) {
  const runs = new Map<string, { total: number; passed: number; at?: string }>();
  for (const r of rows) {
    if (r.suite !== "extended" || !r.run_id) continue;
    const run = runs.get(r.run_id) ?? { total: 0, passed: 0, at: r.created_at };
    run.total += 1;
    run.passed += r.passed ? 1 : 0;
    runs.set(r.run_id, run);
  }
  if (!runs.size) return null;
  return (
    <section aria-labelledby="ext-history-heading" className="space-y-3">
      <h2 id="ext-history-heading" className="text-sm font-semibold text-text-primary">
        Earlier longer checks
      </h2>
      <ul className="space-y-2">
        {[...runs.entries()].map(([id, r]) => (
          <li key={id}>
            <Card variant="solid" className="flex items-center justify-between gap-4 text-sm">
              <span className="text-text-secondary">{r.at ? relativeTime(r.at) : "Earlier"}</span>
              <span className="text-text-primary">
                {r.passed} of {r.total} handled ({Math.round((100 * r.passed) / r.total)}%)
              </span>
            </Card>
          </li>
        ))}
      </ul>
    </section>
  );
}
