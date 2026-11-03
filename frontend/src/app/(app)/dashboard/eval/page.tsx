"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ShieldCheck, TriangleAlert, Play } from "lucide-react";
import { toast } from "sonner";
import { Button, Card, EmptyState, ErrorBoundary, PageHeader, Skeleton, TaskProgressBar, useEstimatedProgress } from "@/components/ui";
import { api } from "@/lib/api";
import { useAgentEvaluations, useAgents } from "@/hooks/useApi";
import { errMessage } from "@/components/dashboard/format";
import {
  EXTENDED_LABEL,
  QUICK_LABEL,
  TOOLS_LABEL,
  ExtendedHistory,
  ExtendedProgress,
  ExtendedReport,
  type ExtendedJob,
  type ExtendedResult,
} from "@/components/eval/ExtendedSuite";
import { CALLERS, QuickResults, handled, type QuickRow } from "@/components/eval/QuickResults";

interface EvalResult {
  persona: string;
  user_prompt: string;
  agent_reply: string;
  task_completion: boolean;
  hallucination_flag: boolean;
  policy_compliance: boolean;
  reasoning: string;
}

type Suite = "quick" | "extended";
type HistoryRun = { run_id?: string; suite?: string; passed?: boolean | null; created_at?: string };
/** How often a running extended suite is asked for its progress. */
const POLL_MS = 1500;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function EvalContent() {
  const agentsQuery = useAgents();
  const agents = agentsQuery.data ?? [];
  const loadingAgents = agentsQuery.isLoading;
  const [pickedAgentId, setSelectedAgentId] = useState("");
  const selectedAgentId = agents.some((a) => a.id === pickedAgentId) ? pickedAgentId : (agents[0]?.id ?? "");
  const history = useAgentEvaluations(selectedAgentId || null);
  const [isRunning, setIsRunning] = useState(false);
  const quickRows: QuickRow[] = (history.data ?? []).filter((r) => (r as { suite?: string }).suite !== "extended") as QuickRow[];
  const [results, setResults] = useState<EvalResult[] | null>(null);
  const [overallScore, setOverallScore] = useState<number | null>(null);
  const [suite, setSuite] = useState<Suite>("quick");
  const [useTools, setUseTools] = useState(false);
  const agentHasTools = !!agents.find((a) => a.id === selectedAgentId)?.tools?.length;
  const [job, setJob] = useState<ExtendedJob | null>(null);
  const [extTotal, setExtTotal] = useState(0);
  const [extResult, setExtResult] = useState<ExtendedResult | null>(null);
  const quickPct = useEstimatedProgress(isRunning && suite === "quick", 35, !!results);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const resetResults = () => {
    setResults(null);
    setOverallScore(null);
    setJob(null);
    setExtResult(null);
  };

  const runExtended = async () => {
    setIsRunning(true);
    resetResults();
    try {
      const start = await api.post<{ job_id: string; total: number }>("/api/evaluations/extended", {
        agent_id: selectedAgentId,
        ...(useTools && agentHasTools ? { use_tools: true } : {}),
      });
      setExtTotal(start.total);
      for (;;) {
        await sleep(POLL_MS);
        if (!alive.current) return;
        const j = await api.get<ExtendedJob>(`/api/evaluations/extended/${start.job_id}`);
        if (!alive.current) return;
        setJob(j);
        if (j.status === "done" && j.result) {
          setExtResult(j.result);
          void history.mutate();
          break;
        }
        if (j.status === "failed") throw new Error(j.error?.message ?? "The longer safety check failed.");
      }
    } catch (err) {
      toast.error(errMessage(err, "The longer safety check failed. Try again."));
    } finally {
      if (alive.current) setIsRunning(false);
    }
  };

  const runEvaluation = async () => {
    if (!selectedAgentId) {
      toast.error("Choose an agent first.");
      return;
    }
    if (suite === "extended") return runExtended();
    setIsRunning(true);
    resetResults();
    try {
      const data = await api.post<{ results: EvalResult[]; overall_score?: number }>("/api/evaluations/run", {
        agent_id: selectedAgentId,
      });
      setResults(data?.results ?? []);
      setOverallScore(typeof data?.overall_score === "number" ? data.overall_score : null);
      void history.mutate();
    } catch (err) {
      toast.error(errMessage(err, "The safety check failed. Try again."));
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-4 sm:p-8 md:p-12">
      <PageHeader
        title="Safety check"
        description="We play 6 tricky callers against your agent and show where it slips."
      />

      <div className="flex items-start gap-3 rounded-xl border border-accent-warning/30 bg-accent-warning/10 p-4">
        <TriangleAlert className="mt-0.5 size-5 shrink-0 text-accent-warning" aria-hidden />
        <p className="text-sm text-text-secondary">
          <span className="font-medium text-text-primary">A quick check, not a guarantee.</span> Another AI plays the
          callers and grades the answers, so it is good for catching big problems early. For anything important, read
          the replies yourself.
        </p>
      </div>

      <details className="rounded-xl border border-border bg-bg-surface px-4 py-3 text-sm">
        <summary className="min-h-6 cursor-pointer font-medium text-text-primary focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]">
          Who are the 6 tricky callers?
        </summary>
        <ul className="mt-3 space-y-2 text-text-secondary">
          {Object.values(CALLERS).map((c) => (
            <li key={c.name}>
              <span className="font-medium text-text-primary">{c.name}.</span> {c.what}
            </li>
          ))}
        </ul>
      </details>

      <Card variant="solid" className="flex flex-col gap-4 sm:flex-row sm:items-end">
        <div className="flex-1">
          <label htmlFor="agent" className="mb-1.5 block text-xs font-medium text-text-secondary">
            Agent to check
          </label>
          {loadingAgents ? (
            <Skeleton className="h-10 w-full rounded-lg" />
          ) : (
            <select
              id="agent"
              value={selectedAgentId}
              onChange={(e) => {
                setSelectedAgentId(e.target.value);
                resetResults();
              }}
              disabled={isRunning || agents.length === 0}
              className="h-10 w-full rounded-lg border border-border bg-bg-base px-3 text-sm text-text-primary focus:border-accent-primary focus:outline-none disabled:opacity-50"
            >
              <option value="" disabled>
                {agents.length === 0 ? (agentsQuery.error ? "Could not load agents" : "No agents found") : "Choose an agent..."}
              </option>
              {agents.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                </option>
              ))}
            </select>
          )}
        </div>
        <fieldset className="sm:w-72" disabled={isRunning}>
          <legend className="mb-1.5 block text-xs font-medium text-text-secondary">How thorough</legend>
          <div className="space-y-1.5">
            {(
              [
                ["quick", QUICK_LABEL],
                ["extended", EXTENDED_LABEL],
              ] as [Suite, string][]
            ).map(([id, text]) => (
              <label key={id} className="flex cursor-pointer items-center gap-2 text-sm text-text-primary">
                <input
                  type="radio"
                  name="suite"
                  value={id}
                  checked={suite === id}
                  onChange={() => {
                    setSuite(id);
                    resetResults();
                  }}
                  className="accent-[var(--color-accent-primary)]"
                />
                {text}
              </label>
            ))}
          </div>
          {suite === "extended" && (
            <label
              className={`mt-2 flex items-start gap-2 text-xs ${agentHasTools ? "cursor-pointer text-text-primary" : "text-text-secondary"}`}
            >
              <input
                type="checkbox"
                checked={useTools && agentHasTools}
                disabled={!agentHasTools}
                onChange={(e) => setUseTools(e.target.checked)}
                className="mt-0.5 accent-[var(--color-accent-primary)]"
              />
              <span>
                {TOOLS_LABEL}
                <span className="block text-text-secondary">
                  {agentHasTools
                    ? "Bookings run on a private copy of the appointment data; the tool calls are judged too."
                    : "Turn on a tool for this agent to test it."}
                </span>
              </span>
            </label>
          )}
        </fieldset>
        <Button onClick={runEvaluation} loading={isRunning} disabled={!selectedAgentId} leftIcon={<Play className="size-4" />}>
          {isRunning ? "Checking your agent" : "Run safety check"}
        </Button>
      </Card>

      {isRunning && (
        <div role="status" className="space-y-3">
          {suite === "extended" ? (
            <ExtendedProgress progress={job?.progress ?? null} total={extTotal} />
          ) : (
            <>
              <TaskProgressBar
                value={quickPct}
                label="Running 6 tricky caller checks"
                detail="Evaluating agent replies"
                active
                testId="quick-eval-progress"
              />
              <p className="text-sm text-text-secondary">Your agent is taking 6 tricky calls. This can take a minute or two.</p>
              <Skeleton className="h-16 rounded-xl" />
              <Skeleton className="h-16 rounded-xl" />
              <Skeleton className="h-16 rounded-xl" />
            </>
          )}
        </div>
      )}

      {!isRunning && extResult && <ExtendedReport result={extResult} />}

      {!isRunning && results && results.length === 0 && (
        <EmptyState icon={<ShieldCheck />} title="No results came back" description="The check finished without any answers. Try running it again." />
      )}

      {!isRunning && results && results.length > 0 && (
        <Card variant="solid" className="flex items-center justify-between gap-6">
          <div>
            <p className="text-sm font-medium text-text-primary">
              Your agent handled {results.filter((r) => handled(r)).length} of {results.length} tricky callers
            </p>
            <p className="mt-1 text-xs text-text-secondary">
              {results.every((r) => handled(r))
                ? "No slips this time. Run it again after you change your agent."
                : "The ones it slipped on are listed first."}
            </p>
          </div>
          {overallScore !== null && (
            <div className="text-right">
              <p
                className={`text-4xl font-bold tabular-nums ${
                  overallScore >= 80 ? "text-[var(--color-accent-success)]" : overallScore >= 50 ? "text-amber-400" : "text-red-400"
                }`}
              >
                {overallScore}%
              </p>
              <p className="text-xs text-text-secondary">safety score</p>
            </div>
          )}
        </Card>
      )}

      {!isRunning && results && results.length > 0 && <QuickResults rows={results} />}

      {!isRunning && selectedAgentId && quickRows.length > 0 && (
        <section aria-labelledby="history-heading" className="space-y-3">
          <h2 id="history-heading" className="text-sm font-semibold text-text-primary">
            Earlier checks
          </h2>
          <QuickResults rows={quickRows} showTime />
        </section>
      )}
      {!isRunning && selectedAgentId && !!history.data?.length && (
        <ExtendedHistory rows={(history.data ?? []) as unknown as HistoryRun[]} />
      )}

      {!results && !extResult && !isRunning && !history.data?.length && (
        <EmptyState
          icon={<ShieldCheck />}
          title="No safety check yet"
          description={
            agents.length === 0 && !loadingAgents
              ? "Create an agent first, then come back to test it."
              : "Pick an agent and run the check. You will see how it handles each of the 6 tricky callers, and where it slips."
          }
          action={
            agents.length === 0 && !loadingAgents ? (
              <Link
                href="/dashboard/studio/agents"
                className="text-sm text-accent-secondary underline underline-offset-4 hover:text-text-primary"
              >
                Open Agent Studio
              </Link>
            ) : undefined
          }
        />
      )}
    </div>
  );
}

export default function EvalPage() {
  return (
    <ErrorBoundary>
      <EvalContent />
    </ErrorBoundary>
  );
}
