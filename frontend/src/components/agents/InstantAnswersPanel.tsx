"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronDown, Plus, Power, Sparkles, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Badge, Button } from "@/components/ui";
import type { BadgeTone } from "@/components/ui/Badge";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";
import type { Agent } from "@/types/api";
import { ConfirmDialog } from "./ConfirmDialog";
import { languageName } from "./languages";

export type InstantStatus = "suggested" | "approved" | "disabled" | "needs_review";

export interface InstantAnswer {
  id: string;
  question: string;
  answer: string;
  language: string;
  status: InstantStatus;
  source: "template" | "calls" | "owner";
  paraphrases: string[];
  times_asked: number;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface BuildState {
  state: "collecting" | "running" | "done" | "failed";
  total: number;
  done: number;
  current: string;
  added: number;
  updated: number;
  skipped: { question: string; reason: string }[];
  error: string | null;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface InstantAnswersView {
  answers: InstantAnswer[];
  counts: Partial<Record<InstantStatus, number>>;
  build: BuildState | null;
  document_count: number;
  has_tools: boolean;
  threshold: number;
  enabled: boolean;
}

const POLL_MS = 2000;
const fieldClass =
  "w-full rounded-lg border border-border bg-bg-primary px-3 py-2.5 text-sm text-text-primary focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent-primary";
const labelClass = "mb-1.5 block text-sm font-medium text-text-secondary";

const STATUS: Record<InstantStatus, { label: string; tone: BadgeTone }> = {
  approved: { label: "Live on calls", tone: "success" },
  suggested: { label: "Waiting for your review", tone: "warning" },
  needs_review: { label: "Documents changed: check again", tone: "warning" },
  disabled: { label: "Turned off", tone: "neutral" },
};
// What the owner should look at first comes first.
const ORDER: Record<InstantStatus, number> = { needs_review: 0, suggested: 1, approved: 2, disabled: 3 };

const lang = (code: string) => (code === "hinglish" ? "Hinglish" : languageName(code));

function sourceText(a: InstantAnswer): string {
  if (a.source === "template") return "Suggested by your template";
  if (a.source === "calls") return a.times_asked > 1 ? `Asked by callers ${a.times_asked} times` : "Asked by a caller";
  return "Added by you";
}

const NO_ANSWER = "Write an answer first, or use Build suggestions to draft one from your documents.";

/** "Instant answers" of one agent: the questions callers ask most, answered the moment they are asked. The owner builds
 *  suggestions from the agent's documents, reads each one and approves it; only approved answers are used on calls. */
export function InstantAnswersPanel({ agent }: { agent: Agent }) {
  const base = `/api/agents/${agent.id}/instant-answers`;
  const [view, setView] = useState<InstantAnswersView | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [alsoOther, setAlsoOther] = useState(false);
  const [newQuestion, setNewQuestion] = useState("");
  const [newAnswer, setNewAnswer] = useState("");
  const [newLang, setNewLang] = useState("");
  const [toDelete, setToDelete] = useState<InstantAnswer | null>(null);
  const mounted = useRef(true);

  const otherLangs = useMemo(() => agent.supported_languages.slice(1), [agent.supported_languages]);

  const load = useCallback(async () => {
    try {
      const v = await api.get<InstantAnswersView>(base);
      if (!mounted.current) return;
      setView(v);
      setLoadError(null);
    } catch (e) {
      if (mounted.current) setLoadError(errMessage(e, "Could not load the instant answers."));
    }
  }, [base]);

  useEffect(() => {
    mounted.current = true;
    void load();
    return () => {
      mounted.current = false;
    };
  }, [load]);

  const building = view?.build?.state === "collecting" || view?.build?.state === "running";
  useEffect(() => {
    if (!building) return;
    const t = setInterval(() => void load(), POLL_MS);
    return () => clearInterval(t);
  }, [building, load]);

  const run = async (name: string, fn: () => Promise<void>) => {
    setBusy(name);
    try {
      await fn();
    } catch (e) {
      toast.error(errMessage(e, "That did not work. Try again."));
    } finally {
      if (mounted.current) setBusy(null);
    }
  };

  const textOf = (a: InstantAnswer) => drafts[a.id] ?? a.answer;
  const dirty = (a: InstantAnswer) => drafts[a.id] !== undefined && drafts[a.id].trim() !== a.answer.trim();
  const clearDraft = (id: string) =>
    setDrafts((d) => Object.fromEntries(Object.entries(d).filter(([k]) => k !== id)));

  const change = (a: InstantAnswer, body: Record<string, unknown>, done: string) =>
    run(a.id, async () => {
      await api.put(`${base}/${a.id}`, body);
      clearDraft(a.id);
      toast.success(done);
      await load();
    });

  const approve = (a: InstantAnswer) =>
    change(a, { status: "approved", ...(dirty(a) ? { answer: textOf(a) } : {}) }, "Approved. The agent will say this straight away on calls.");

  const startBuild = async () => {
    setStarting(true);
    try {
      await api.post(`${base}/build`, alsoOther && otherLangs.length ? { languages: otherLangs } : {});
      await load();
    } catch (e) {
      toast.error(errMessage(e, "Could not start building suggestions."));
    } finally {
      if (mounted.current) setStarting(false);
    }
  };

  const add = () =>
    run("add", async () => {
      const body: Record<string, unknown> = { question: newQuestion.trim() };
      if (newAnswer.trim()) body.answer = newAnswer.trim();
      if (newLang) body.language = newLang;
      await api.post(base, body);
      setNewQuestion("");
      setNewAnswer("");
      toast.success(newAnswer.trim() ? "Added and approved." : "Question added. Use Build suggestions to draft its answer.");
      await load();
    });

  const documents = view?.document_count ?? agent.document_count;
  const buildDisabledReason = building
    ? "Suggestions are being written. This can take a minute."
    : documents === 0
      ? "Upload a document first: answers are written from your documents."
      : view && !view.enabled
        ? "Instant answers are switched off on this server."
        : undefined;

  const sorted = useMemo(
    () => [...(view?.answers ?? [])].sort((a, b) => ORDER[a.status] - ORDER[b.status]),
    [view?.answers]
  );
  const build = view?.build ?? null;
  const live = view?.counts.approved ?? 0;
  const waiting = (view?.counts.suggested ?? 0) + (view?.counts.needs_review ?? 0);

  return (
    <section aria-labelledby="instant-title" className="space-y-5 rounded-2xl border border-border bg-bg-surface p-5" data-testid="instant-answers-panel">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border pb-4">
        <div className="min-w-0 max-w-xl">
          <h2 id="instant-title" className="font-display text-lg font-semibold text-text-primary">Instant answers</h2>
          <p className="mt-1 text-sm text-text-secondary">
            Common questions your agent answers immediately on calls. Only answers you approve are used; everything else is answered the usual way.
          </p>
          {view && (
            <p className="mt-2 text-xs text-text-secondary" data-testid="instant-summary">
              {live} live{waiting ? `, ${waiting} waiting for you` : ""}
            </p>
          )}
        </div>
        <div className="flex flex-col items-start gap-2 sm:items-end">
          <Button
            onClick={startBuild}
            loading={starting}
            disabled={!!buildDisabledReason}
            disabledReason={buildDisabledReason}
            leftIcon={<Sparkles className="size-4" />}
          >
            Build suggestions
          </Button>
          {otherLangs.length > 0 && (
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
              <input
                type="checkbox"
                checked={alsoOther}
                onChange={(e) => setAlsoOther(e.target.checked)}
                className="size-4 accent-[var(--color-accent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary"
              />
              Also write each answer in {otherLangs.map(lang).join(", ")}
            </label>
          )}
        </div>
      </div>

      {view && !view.enabled && (
        <p role="note" className="rounded-lg border border-border bg-bg-elevated px-3 py-2 text-sm text-text-secondary">
          Instant answers are switched off on this server, so the agent answers every question the usual way.
        </p>
      )}

      {building && build && (
        <div role="status" className="rounded-lg border border-border bg-bg-elevated px-3 py-3 text-sm text-text-primary" data-testid="build-progress">
          <p className="font-medium">
            {build.state === "collecting"
              ? "Finding the questions callers ask most..."
              : `Writing suggestions: ${build.done} of ${build.total} done`}
          </p>
          {build.state === "running" && build.total > 0 && (
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-border" aria-hidden>
              <div className="h-full rounded-full bg-[var(--color-accent)] transition-all" style={{ width: `${Math.round((100 * build.done) / build.total)}%` }} />
            </div>
          )}
          {build.current && <p className="mt-2 truncate text-xs text-text-secondary">Now: {build.current}</p>}
        </div>
      )}
      {build?.state === "failed" && (
        <p role="alert" className="text-sm text-accent-error">{build.error || "The build stopped. Try again."}</p>
      )}
      {build?.state === "done" && (
        <div role="status" className="text-sm text-text-secondary" data-testid="build-result">
          {build.added + build.updated === 0
            ? "No new suggestions this time."
            : `Added ${build.added} new suggestion${build.added === 1 ? "" : "s"}${build.updated ? ` and refreshed ${build.updated}` : ""}. Read each one, then approve it.`}
          {build.skipped.length > 0 && (
            <details className="group mt-1">
              <summary className="inline-flex min-h-10 cursor-pointer list-none items-center gap-1 text-xs font-medium hover:text-text-primary">
                <ChevronDown className="size-3.5 transition-transform group-open:rotate-180" aria-hidden />
                {build.skipped.length} question{build.skipped.length === 1 ? " was" : "s were"} skipped
              </summary>
              <ul className="mt-1 space-y-1 text-xs">
                {build.skipped.map((s, i) => (
                  <li key={`${s.question}-${i}`}>
                    <span className="text-text-primary">{s.question}</span>: {s.reason}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}

      {loadError && (
        <div role="alert" className="flex flex-wrap items-center gap-3 text-sm text-accent-error">
          {loadError}
          <Button variant="outline" size="sm" onClick={() => void load()}>Try again</Button>
        </div>
      )}
      {!view && !loadError && <p className="text-sm text-text-secondary">Loading...</p>}

      {view && sorted.length === 0 && (
        <p className="text-sm text-text-secondary" data-testid="instant-empty">
          No instant answers yet. Choose Build suggestions and BonClay will draft answers from your documents for the questions
          callers ask most, or add a question yourself below.
        </p>
      )}

      {sorted.length > 0 && (
        <ul className="space-y-3" data-testid="instant-list">
          {sorted.map((a) => {
            const text = textOf(a);
            const isBusy = busy === a.id;
            return (
              <li key={a.id} className="space-y-3 rounded-xl border border-border bg-bg-primary p-4" data-testid={`instant-${a.id}`}>
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <p className="min-w-0 break-words text-sm font-medium text-text-primary">{a.question}</p>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone={STATUS[a.status].tone} dot>{STATUS[a.status].label}</Badge>
                    <Badge>{lang(a.language)}</Badge>
                  </div>
                </div>
                <p className="text-xs text-text-secondary">{sourceText(a)}</p>
                {a.status === "needs_review" && (
                  <p role="note" className="text-xs text-accent-warning">
                    Your documents changed after you approved this, so it is not used on calls. Read it again, then approve it.
                  </p>
                )}
                <div>
                  <label htmlFor={`ia-answer-${a.id}`} className={labelClass}>What the agent says</label>
                  <textarea
                    id={`ia-answer-${a.id}`}
                    className={`${fieldClass} min-h-20 resize-y`}
                    value={text}
                    maxLength={700}
                    onChange={(e) => setDrafts((d) => ({ ...d, [a.id]: e.target.value }))}
                    placeholder="No answer yet"
                  />
                </div>
                {a.paraphrases.length > 0 && (
                  <details className="group">
                    <summary className="inline-flex min-h-10 cursor-pointer list-none items-center gap-1 text-xs font-medium text-text-secondary hover:text-text-primary">
                      <ChevronDown className="size-3.5 transition-transform group-open:rotate-180" aria-hidden />
                      Other ways of asking this ({a.paraphrases.length})
                    </summary>
                    <ul className="mt-1 list-disc space-y-0.5 pl-8 text-xs text-text-secondary">
                      {a.paraphrases.map((p) => <li key={p}>{p}</li>)}
                    </ul>
                  </details>
                )}
                <div className="flex flex-wrap gap-2">
                  {a.status === "approved" ? (
                    <Button variant="outline" size="sm" onClick={() => change(a, { status: "disabled" }, "Turned off. The agent answers this the usual way.")} loading={isBusy} leftIcon={<Power className="size-3.5" />}>
                      Turn off
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      onClick={() => approve(a)}
                      loading={isBusy}
                      disabled={!text.trim()}
                      disabledReason={NO_ANSWER}
                      leftIcon={<Check className="size-3.5" />}
                    >
                      {a.status === "disabled" ? "Turn on" : "Approve"}
                    </Button>
                  )}
                  {dirty(a) && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => change(a, { answer: text }, "Answer saved.")}
                      loading={isBusy}
                      disabled={!text.trim() && a.status === "approved"}
                      disabledReason="An approved answer cannot be empty. Turn it off instead."
                    >
                      Save answer
                    </Button>
                  )}
                  <Button variant="ghost" size="sm" onClick={() => setToDelete(a)} disabled={isBusy} leftIcon={<Trash2 className="size-3.5" />}>
                    Delete
                  </Button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <form
        className="space-y-3 border-t border-border pt-5"
        onSubmit={(e) => {
          e.preventDefault();
          if (newQuestion.trim()) void add();
        }}
      >
        <h3 className="text-sm font-semibold text-text-primary">Add a question</h3>
        <div>
          <label htmlFor="ia-new-question" className={labelClass}>A question callers ask</label>
          <input
            id="ia-new-question"
            className={fieldClass}
            value={newQuestion}
            maxLength={300}
            onChange={(e) => setNewQuestion(e.target.value)}
            placeholder="What are your opening hours on Sunday?"
            autoComplete="off"
          />
        </div>
        <div>
          <label htmlFor="ia-new-answer" className={labelClass}>Answer (optional)</label>
          <textarea
            id="ia-new-answer"
            className={`${fieldClass} min-h-16 resize-y`}
            value={newAnswer}
            maxLength={700}
            onChange={(e) => setNewAnswer(e.target.value)}
          />
          <p className="mt-1.5 text-xs text-text-secondary">
            Leave this empty and Build suggestions will write it from your documents. An answer you write here is approved straight away.
          </p>
        </div>
        {agent.supported_languages.length > 1 && (
          <div>
            <label htmlFor="ia-new-language" className={labelClass}>Language of the question and answer</label>
            <select id="ia-new-language" className={fieldClass} value={newLang} onChange={(e) => setNewLang(e.target.value)}>
              <option value="">Detect from the question</option>
              {agent.supported_languages.map((c) => (
                <option key={c} value={c}>{lang(c)}</option>
              ))}
            </select>
          </div>
        )}
        <Button
          type="submit"
          variant="outline"
          loading={busy === "add"}
          disabled={!newQuestion.trim()}
          disabledReason="Type the question first."
          leftIcon={<Plus className="size-4" />}
        >
          Add question
        </Button>
      </form>

      <ConfirmDialog
        open={!!toDelete}
        title="Delete this instant answer?"
        description={toDelete ? `The agent will answer "${toDelete.question}" the usual way again.` : ""}
        confirmLabel="Delete"
        keepLabel="Keep it"
        onCancel={() => setToDelete(null)}
        onConfirm={async () => {
          const a = toDelete;
          if (!a) return;
          try {
            await api.del(`${base}/${a.id}`);
            toast.success("Deleted.");
            clearDraft(a.id);
            await load();
          } catch (e) {
            toast.error(errMessage(e, "Could not delete it. Try again."));
          } finally {
            setToDelete(null);
          }
        }}
      />
    </section>
  );
}
