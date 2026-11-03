"use client";

import Link from "next/link";
import { useCallback, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { Card, ErrorState, Skeleton } from "@/components/ui";
import { useGaps } from "./results";
import { languageName, relativeTime } from "@/components/dashboard/format";
import type { KnowledgeGapCluster, KnowledgeGapReason } from "@/types/api";

const DAYS = 30;
const FINE_KEY = "bonclay.analytics.fine.v1";

/** Where an owner adds documents for an agent: the Knowledge tab of the agent's page in Studio. */
export const knowledgeHref = (agentId: string) =>
  `/dashboard/studio/agents?agent=${encodeURIComponent(agentId)}&tab=knowledge`;

const fineKey = (c: KnowledgeGapCluster) => `${c.agent_id}|${c.question.trim().toLowerCase()}`;

/** Questions the owner marked as fine ("this is not a gap"). Saved on this device only; the backend keeps no such list. */
function useMarkedFine() {
  const [fine, setFine] = useState<string[]>(() => {
    try {
      const raw = JSON.parse(localStorage.getItem(FINE_KEY) || "[]");
      return Array.isArray(raw) ? raw.filter((x): x is string => typeof x === "string") : [];
    } catch {
      return []; // storage blocked, unreadable, or not in a browser: start empty
    }
  });
  const save = useCallback((next: string[]) => {
    setFine(next);
    try {
      localStorage.setItem(FINE_KEY, JSON.stringify(next.slice(-500)));
    } catch {
      /* the choice still holds for this visit */
    }
  }, []);
  return { fine, mark: (k: string) => save([...fine.filter((x) => x !== k), k]), undo: (k: string) => save(fine.filter((x) => x !== k)) };
}

const REASON_HINT: Record<KnowledgeGapReason, string> = {
  "said it didn't know": "The agent told the caller it did not have the answer.",
  "no matching document": "No document in the knowledge base matched this question.",
  "weak match": "The closest document was only a weak match, so the answer may not be grounded.",
};

function GapRow({ cluster, id, onFine }: { cluster: KnowledgeGapCluster; id: string; onFine?: () => void }) {
  const [open, setOpen] = useState(false);
  const more = cluster.examples.filter((e) => e !== cluster.question);
  const when = relativeTime(cluster.last_asked);
  return (
    <li className="py-3 first:pt-0 last:pb-0">
      <div className="flex flex-col gap-x-4 gap-y-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 flex-1">
          <p className="break-words text-sm font-medium text-text-primary">
            {cluster.question}
            {cluster.reason && (
              <span
                data-testid="gap-reason"
                title={REASON_HINT[cluster.reason]}
                className="ml-2 inline-block rounded-full border border-border px-2 py-0.5 align-middle text-[11px] font-normal text-text-secondary"
              >
                {cluster.reason}
              </span>
            )}
          </p>
          <p className="mt-0.5 text-xs text-text-secondary">
            Asked {cluster.count} {cluster.count === 1 ? "time" : "times"}
            {cluster.agent_name ? ` · ${cluster.agent_name}` : ""}
            {cluster.language ? ` · ${languageName(cluster.language)}` : ""}
            {when ? ` · last ${when}` : ""}
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          <Link
            href={knowledgeHref(cluster.agent_id)}
            className="inline-flex min-h-10 items-center rounded-md border border-border px-3 text-xs font-medium text-text-primary transition-colors hover:bg-bg-elevated focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary"
            aria-label={`Add a document: ${cluster.question}`}
          >
            Add a document
          </Link>
          {onFine && (
            <button
              type="button"
              onClick={onFine}
              title="Hide this question. It stays hidden on this device."
              aria-label={`Mark as fine: ${cluster.question}`}
              className="inline-flex min-h-10 items-center gap-1 rounded-md px-3 text-xs font-medium text-text-secondary transition-colors hover:bg-bg-elevated hover:text-text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary"
            >
              <Check className="size-3.5" aria-hidden="true" />
              Mark as fine
            </button>
          )}
        </div>
      </div>
      {more.length > 0 && (
        <>
          <button
            type="button"
            aria-expanded={open}
            aria-controls={`${id}-examples`}
            onClick={() => setOpen((o) => !o)}
            className="mt-1.5 inline-flex items-center gap-1 rounded text-xs text-text-secondary hover:text-text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary"
          >
            <ChevronDown className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`} aria-hidden="true" />
            {open ? "Hide" : "Show"} other ways it was asked ({more.length})
          </button>
          {open && (
            <ul id={`${id}-examples`} className="mt-1.5 list-disc space-y-1 pl-5 text-xs text-text-secondary">
              {more.map((e) => (
                <li key={e} className="break-words">
                  {e}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </li>
  );
}

/** The questions the agent could not answer, each with "Add a document" and "Mark as fine". */
export function KnowledgeGapsCard({ agentId, days = DAYS }: { agentId?: string | null; days?: number }) {
  const { data, error, isLoading, mutate } = useGaps(days, agentId);
  const { fine, mark, undo } = useMarkedFine();
  const [showFine, setShowFine] = useState(false);
  const all = data?.clusters ?? [];
  const clusters = all.filter((c) => !fine.includes(fineKey(c)));
  const hidden = all.filter((c) => fine.includes(fineKey(c)));
  const windowDays = data?.window_days ?? days;
  return (
    <Card variant="solid" role="region" aria-labelledby="knowledge-gaps-title">
      <div className="mb-4">
        <h2 id="knowledge-gaps-title" className="font-display text-base font-semibold text-text-primary">
          Questions your agent couldn&apos;t answer
        </h2>
        <p className="mt-0.5 text-xs text-text-secondary">
          Add these answers to your documents so your agent can handle them next time. Phone numbers and emails are removed.
        </p>
      </div>
      {isLoading && !data ? (
        <Skeleton className="h-24 w-full rounded-xl" />
      ) : error && !data ? (
        <ErrorState title="Could not load unanswered questions" message={error.message} onRetry={() => mutate()} />
      ) : clusters.length === 0 ? (
        <p className="rounded-xl border border-dashed border-border px-4 py-6 text-center text-sm text-text-secondary">
          {all.length > 0
            ? "You marked every unanswered question as fine."
            : `No unanswered questions in the last ${windowDays} days`}
        </p>
      ) : (
        <>
          <p className="mb-3 text-xs text-text-secondary">
            {data?.total_gap_turns} of {data?.total_turns} replies in the last {windowDays} days
          </p>
          <ul aria-label="Unanswered question groups" className="divide-y divide-border">
            {clusters.map((c, i) => (
              <GapRow key={`${c.agent_id}-${i}-${c.question}`} cluster={c} id={`gap-${i}`} onFine={() => mark(fineKey(c))} />
            ))}
          </ul>
        </>
      )}
      {hidden.length > 0 && (
        <div className="mt-3 border-t border-border pt-3">
          <button
            type="button"
            aria-expanded={showFine}
            onClick={() => setShowFine((o) => !o)}
            className="inline-flex min-h-10 items-center gap-1 rounded text-xs text-text-secondary hover:text-text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary"
          >
            <ChevronDown className={`size-3.5 transition-transform ${showFine ? "rotate-180" : ""}`} aria-hidden="true" />
            Marked as fine ({hidden.length})
          </button>
          {showFine && (
            <ul className="mt-1 space-y-1">
              {hidden.map((c) => (
                <li key={fineKey(c)} className="flex flex-wrap items-center justify-between gap-2 text-xs text-text-secondary">
                  <span className="min-w-0 break-words">{c.question}</span>
                  <button
                    type="button"
                    onClick={() => undo(fineKey(c))}
                    className="min-h-10 rounded px-2 font-medium text-text-primary underline underline-offset-2 focus-visible:outline-2 focus-visible:outline-accent-primary"
                  >
                    Show again
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </Card>
  );
}
