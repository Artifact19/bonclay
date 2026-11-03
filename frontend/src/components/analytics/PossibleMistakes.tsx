"use client";

import Link from "next/link";
import { useState } from "react";
import { Check, Zap } from "lucide-react";
import { Button, Card, ErrorState, Skeleton } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { relativeTime } from "@/components/dashboard/format";
import { knowledgeHref } from "./KnowledgeGapsCard";
import { useAnswerAudit, type FlaggedAnswer } from "./answerAudit";

const DAYS = 30;
const NOTE = "Checked automatically in English; Hindi and Hinglish answers are not checked yet.";

const phoneHref = (agentId: string) => `/dashboard/studio/agents?agent=${encodeURIComponent(agentId)}&tab=phone`;
const errText = (e: unknown, fallback: string) => (e instanceof ApiError && e.message ? e.message : fallback);
const btn =
  "inline-flex min-h-10 items-center gap-1 rounded-md border border-border px-3 text-xs font-medium text-text-primary transition-colors hover:bg-bg-elevated focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary disabled:opacity-50";

function Row({ item, onChanged }: { item: FlaggedAnswer; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const mark = async (feedback: "fine" | "wrong") => {
    setBusy(true);
    try {
      await api.put(`/api/answer-audit/${encodeURIComponent(item.turn_id)}`, { feedback });
      onChanged();
    } catch (e) {
      setMessage(errText(e, "Could not save that. Try again."));
    } finally {
      setBusy(false);
    }
  };
  const instant = async () => {
    if (!item.agent_id) return;
    setBusy(true);
    try {
      await api.post(`/api/agents/${encodeURIComponent(item.agent_id)}/instant-answers`, { question: item.question });
      setMessage("Added to this agent's instant answers. Write the answer and approve it on the agent's Phone tab.");
    } catch (e) {
      setMessage(e instanceof ApiError && e.status === 409 ? "That question is already in this agent's instant answers." : errText(e, "Could not add it. Try again."));
    } finally {
      setBusy(false);
    }
  };
  const when = relativeTime(item.at);
  return (
    <li className="py-4 first:pt-0 last:pb-0" data-testid="possible-mistake">
      <p className="text-xs text-text-secondary">
        {item.agent_name ? `${item.agent_name} · ` : ""}
        {when || "Recently"}
        {item.feedback === "wrong" ? " · you marked it wrong" : ""}
      </p>
      <dl className="mt-1.5 space-y-2 text-sm">
        <div>
          <dt className="text-[11px] font-semibold uppercase tracking-wide text-text-secondary">Caller asked</dt>
          <dd className="break-words text-text-primary">{item.question}</dd>
        </div>
        <div>
          <dt className="text-[11px] font-semibold uppercase tracking-wide text-text-secondary">Agent said</dt>
          <dd className="break-words text-text-primary">{item.reply}</dd>
        </div>
        <div>
          <dt className="text-[11px] font-semibold uppercase tracking-wide text-text-secondary">
            Your documents say{item.passage_source === "approx" ? " (closest passage, found again afterwards)" : ""}
          </dt>
          <dd className="break-words text-text-secondary">{item.passage || "No matching passage was found."}</dd>
        </div>
      </dl>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button type="button" className={btn} disabled={busy} onClick={() => mark("fine")} aria-label={`It was fine: ${item.question}`}>
          <Check className="size-3.5" aria-hidden="true" />
          It was fine
        </button>
        {item.agent_id && (
          <Link
            href={knowledgeHref(item.agent_id)}
            onClick={() => void mark("wrong")}
            className={btn}
            aria-label={`It was wrong, add a document: ${item.question}`}
          >
            It was wrong — add a document
          </Link>
        )}
        {item.agent_id && (
          <button type="button" className={btn} disabled={busy} onClick={instant} aria-label={`Make an instant answer: ${item.question}`}>
            <Zap className="size-3.5" aria-hidden="true" />
            Make an instant answer
          </button>
        )}
      </div>
      {message && (
        <p role="status" className="mt-2 text-xs text-text-secondary">
          {message}{" "}
          {item.agent_id && message.startsWith("Added") && (
            <Link href={phoneHref(item.agent_id)} className="font-medium text-text-primary underline underline-offset-2">
              Open Phone tab
            </Link>
          )}
        </p>
      )}
    </li>
  );
}

/** Replies the agent gave from the documents that the documents do not seem to support, found by a check that runs after the
 *  conversation. Each can be marked fine, wrong (add a document) or turned into an instant answer. */
export function PossibleMistakes({ agentId, days = DAYS }: { agentId?: string | null; days?: number }) {
  const { data, error, isLoading, mutate } = useAnswerAudit(days, agentId);
  const [starting, setStarting] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const running = starting || data?.run?.state === "running";
  const start = async () => {
    setStarting(true);
    setRunError(null);
    try {
      await api.post("/api/answer-audit/run", agentId ? { agent_id: agentId } : {});
      await mutate();
    } catch (e) {
      setRunError(errText(e, "Could not start the check. Try again in a moment."));
    } finally {
      setStarting(false);
    }
  };
  const flagged = data?.flagged ?? [];
  const counts = data?.counts;
  const neverRun = !!data && data.enabled && !data.model_ready && counts?.checked === 0;
  return (
    <Card variant="solid" role="region" aria-labelledby="possible-mistakes-title">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 id="possible-mistakes-title" className="font-display text-base font-semibold text-text-primary">
            Possible mistakes
          </h2>
          <p className="mt-0.5 text-xs text-text-secondary">
            Answers that your documents do not seem to back up. Look at each one and tell us if it was fine. {NOTE}
          </p>
        </div>
        {data?.enabled !== false && (
          <Button variant="outline" size="sm" loading={running} onClick={start} disabled={running}>
            Check recent answers
          </Button>
        )}
      </div>
      {isLoading && !data ? (
        <Skeleton className="h-24 w-full rounded-xl" />
      ) : error && !data ? (
        <ErrorState title="Could not load the answer check" message={error.message} onRetry={() => mutate()} />
      ) : (
        <>
          {runError && (
            <p role="alert" className="mb-3 text-sm text-accent-error">
              {runError}
            </p>
          )}
          {data?.run?.state === "running" && (
            <p role="status" className="mb-3 text-xs text-text-secondary">
              Checking answers… {data.run.done} of {data.run.total}
            </p>
          )}
          {data?.run?.state === "error" && (
            <p role="alert" className="mb-3 text-sm text-accent-error">
              {data.run.error || "The last check could not finish."}
            </p>
          )}
          {data?.enabled === false ? (
            <p className="rounded-xl border border-dashed border-border px-4 py-6 text-center text-sm text-text-secondary">
              The answer check is turned off on this server.
            </p>
          ) : flagged.length === 0 ? (
            <p className="rounded-xl border border-dashed border-border px-4 py-6 text-center text-sm text-text-secondary">
              {neverRun
                ? "Nothing has been checked yet. The first check downloads a small checking model (about 1.4 GB), once, so it can take a few minutes."
                : (counts?.checked ?? 0) > 0
                  ? "No possible mistakes found in the answers checked."
                  : "No answers have been checked yet. Press Check recent answers."}
            </p>
          ) : (
            <ul aria-label="Possible mistakes" className="divide-y divide-border">
              {flagged.map((f) => (
                <Row key={f.turn_id} item={f} onChanged={() => void mutate()} />
              ))}
            </ul>
          )}
          {counts && (counts.checked > 0 || counts.not_checked_language > 0) && (
            <p className="mt-3 border-t border-border pt-3 text-xs text-text-secondary">
              {counts.checked} {counts.checked === 1 ? "answer" : "answers"} checked, {counts.flagged} flagged
              {counts.not_checked_language > 0 ? `, ${counts.not_checked_language} not checked (language)` : ""}.
            </p>
          )}
        </>
      )}
    </Card>
  );
}
