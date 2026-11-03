"use client";

import Link from "next/link";
import { useState } from "react";
import { BarChart3 } from "lucide-react";
import { Button, EmptyState, ErrorBoundary, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { useAgents } from "@/hooks/useApi";
import { AdvancedSection } from "@/components/analytics/AdvancedSection";
import { KnowledgeGapsCard } from "@/components/analytics/KnowledgeGapsCard";
import { PossibleMistakes } from "@/components/analytics/PossibleMistakes";
import { ActivityCard, HeadlineCards } from "@/components/analytics/ResultsOverview";
import { RANGE_DAYS, useResults, type Range } from "@/components/analytics/results";

const linkButton =
  "inline-flex min-h-11 items-center justify-center rounded-lg bg-accent-primary px-5 text-sm font-medium text-[var(--color-on-accent)] transition hover:brightness-110 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary";

function RangeToggle({ value, onChange }: { value: Range; onChange: (r: Range) => void }) {
  return (
    <div role="group" aria-label="Date range" className="inline-flex rounded-lg border border-border bg-bg-base p-0.5">
      {(["7d", "30d"] as const).map((r) => (
        <button
          key={r}
          type="button"
          aria-pressed={value === r}
          onClick={() => onChange(r)}
          className={`h-10 rounded-md px-4 text-sm font-medium transition-colors duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary ${
            value === r ? "bg-accent-primary text-[var(--color-on-accent)]" : "text-text-secondary hover:text-text-primary"
          }`}
        >
          {r === "7d" ? "7 days" : "30 days"}
        </button>
      ))}
    </div>
  );
}

function ResultsContent() {
  const [range, setRange] = useState<Range>("7d");
  const [agentId, setAgentId] = useState("");
  const agentsQuery = useAgents();
  const agents = agentsQuery.data ?? [];
  const results = useResults(range, agentId || null);
  const { data, error, isLoading, mutate } = results;
  const loading = isLoading && !data;
  const noAgents = !agentsQuery.isLoading && !agentsQuery.error && agents.length === 0;
  const quiet = !!data && data.conversations === 0 && data.questions_asked === 0;

  const controls = (
    <>
      <div className="min-w-0">
        <label htmlFor="results-agent" className="sr-only">
          Agent
        </label>
        <select
          id="results-agent"
          value={agentId}
          onChange={(e) => setAgentId(e.target.value)}
          className="h-10 w-full max-w-[14rem] rounded-lg border border-border bg-bg-base px-3 text-sm text-text-primary focus:border-accent-primary focus:outline-none"
        >
          <option value="">All agents</option>
          {agents.map((a) => (
            <option key={a.id} value={a.id}>
              {a.name}
            </option>
          ))}
        </select>
      </div>
      <RangeToggle value={range} onChange={setRange} />
    </>
  );

  return (
    <div className="mx-auto max-w-6xl p-4 sm:p-8 md:p-12">
      <PageHeader
        title="Results"
        description="Is your agent working? How many people it helped, and what it could not answer."
        actions={noAgents ? undefined : controls}
        className="[&>div:last-child]:min-w-0 [&>div:last-child]:shrink [&>div:last-child]:flex-wrap"
      />

      {noAgents ? (
        <EmptyState
          icon={<BarChart3 />}
          title="Results appear once an agent takes its first conversation"
          description="Create an agent, add your documents, then try it out. You will see how many people it helped and what it could not answer."
          action={
            <Link href="/dashboard/studio/agents" className={linkButton}>
              Create your first agent
            </Link>
          }
        />
      ) : error && !data ? (
        <ErrorState title="Could not load results" message={error.message} onRetry={() => mutate()} />
      ) : loading ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3" aria-busy="true">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-2xl" />
          ))}
        </div>
      ) : quiet ? (
        <EmptyState
          icon={<BarChart3 />}
          title={`No conversations in the last ${RANGE_DAYS[range]} days`}
          description="Talk to your agent yourself, or share its phone number or website chat. Practice calls are not counted here, so real conversations show up on this page."
          action={
            <div className="flex flex-wrap justify-center gap-2">
              <Link href="/dashboard/playground" className={linkButton}>
                Talk to your agent
              </Link>
              {range === "7d" && (
                <Button variant="outline" size="lg" onClick={() => setRange("30d")}>
                  Look back 30 days
                </Button>
              )}
            </div>
          }
        />
      ) : data ? (
        <div className="space-y-6">
          <HeadlineCards r={data} />
          <KnowledgeGapsCard agentId={agentId || null} days={RANGE_DAYS[range]} />
          <PossibleMistakes agentId={agentId || null} days={RANGE_DAYS[range]} />
          <ActivityCard r={data} />
        </div>
      ) : null}

      {!noAgents && <AdvancedSection range={range} agentId={agentId || null} />}
    </div>
  );
}

export default function ResultsPage() {
  return (
    <ErrorBoundary>
      <ResultsContent />
    </ErrorBoundary>
  );
}
