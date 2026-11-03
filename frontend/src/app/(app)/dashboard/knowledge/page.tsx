"use client";

import Link from "next/link";
import { Suspense, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Bot, Mic, Pencil } from "lucide-react";
import { useAgents } from "@/hooks/useApi";
import { Card, EmptyState, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { KnowledgePanel } from "@/components/agents/KnowledgePanel";
import { talkHref } from "@/components/agents/agentStatus";
import { cn } from "@/lib/utils";
import type { Agent } from "@/types/api";

/** The agent picked last time, so the page opens on it when the link names none. */
const LAST_AGENT_KEY = "bonclay.documents.agent";
const AGENT_PARAM = "agent";
const AGENTS_PAGE = "/dashboard/studio/agents";
const TRY_PAGE = "/dashboard/knowledge/try";

const fieldClass =
  "w-full rounded-lg border border-border bg-bg-primary px-3 py-2.5 text-sm text-text-primary focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent-primary";
const labelClass = "mb-1.5 block text-sm font-medium text-text-secondary";

const linkButton =
  "inline-flex h-10 items-center justify-center gap-2 rounded-full px-4 text-sm font-medium select-none transition-[filter,background-color,border-color] duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary";

/** The agent editor, opened on one agent. */
const editHref = (agentId: string) => `${AGENTS_PAGE}?agent=${encodeURIComponent(agentId)}&tab=basics`;

/** Newest first. Agents carry no "updated" time, so the day they were made is the closest thing. */
function newestFirst(agents: Agent[]): Agent[] {
  const time = (a: Agent) => {
    const t = Date.parse(a.created_at);
    return Number.isNaN(t) ? 0 : t;
  };
  return [...agents].sort((a, b) => time(b) - time(a));
}

function readLastAgent(): string {
  try {
    return localStorage.getItem(LAST_AGENT_KEY) ?? "";
  } catch {
    return "";
  }
}

function writeLastAgent(id: string) {
  try {
    localStorage.setItem(LAST_AGENT_KEY, id);
  } catch {
    /* storage blocked: the choice still lives in the address */
  }
}

export default function DocumentsPage() {
  // useSearchParams needs a Suspense boundary for static rendering
  return (
    <Suspense fallback={null}>
      <Documents />
    </Suspense>
  );
}

function Documents() {
  const router = useRouter();
  const pathname = usePathname() ?? "/dashboard/knowledge";
  const linkedId = useSearchParams()?.get(AGENT_PARAM) ?? "";
  const agentsQuery = useAgents();
  const agents = useMemo(() => newestFirst(agentsQuery.data ?? []), [agentsQuery.data]);

  // Only used once the agents have loaded (until then the page shows a skeleton), so reading it on first render is safe.
  const [rememberedId, setRememberedId] = useState(() => (typeof window === "undefined" ? "" : readLastAgent()));
  const [pickedId, setPickedId] = useState("");

  // What the person just picked; else the agent in the link (?agent=<id>); else the last one used; else the newest.
  const has = (id: string) => !!id && agents.some((a) => a.id === id);
  const selectedId = has(pickedId) ? pickedId : has(linkedId) ? linkedId : has(rememberedId) ? rememberedId : (agents[0]?.id ?? "");
  const selected = agents.find((a) => a.id === selectedId) ?? null;

  // Keep the chosen agent in the address without adding history entries, so the link can be shared or reloaded.
  useEffect(() => {
    if (!selectedId || selectedId === linkedId) return;
    router.replace(`${pathname}?${AGENT_PARAM}=${encodeURIComponent(selectedId)}`, { scroll: false });
  }, [selectedId, linkedId, pathname, router]);

  const pick = (id: string) => {
    setPickedId(id);
    setRememberedId(id);
    writeLastAgent(id);
  };

  // Two agents can share a name (a template used twice): add the day it was made so the list tells them apart.
  const optionLabel = (a: Agent) => {
    if (agents.filter((o) => o.name === a.name).length < 2) return a.name;
    const made = new Date(a.created_at);
    return Number.isNaN(made.getTime()) ? a.name : `${a.name} (made ${made.toLocaleDateString(undefined, { day: "numeric", month: "short" })})`;
  };

  let body: React.ReactNode;
  if (agentsQuery.isLoading) {
    body = (
      <div className="space-y-4" aria-busy="true">
        <Skeleton className="h-10 w-full max-w-sm rounded-lg" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
  } else if (agentsQuery.error) {
    body = (
      <ErrorState
        title="Could not load your agents"
        message="Your documents belong to an agent, so we need the list first. Check your connection and try again."
        onRetry={() => void agentsQuery.mutate()}
      />
    );
  } else if (!agents.length) {
    body = (
      <EmptyState
        icon={<Bot />}
        title="Create an agent first"
        description="Documents belong to an agent: it answers callers from the files you add. Make an agent, then come back to add its documents."
        action={
          <Link href={AGENTS_PAGE} className={cn(linkButton, "bg-brand text-[var(--color-on-accent)] hover:opacity-90")}>
            Go to Agents
          </Link>
        }
      />
    );
  } else {
    body = (
      <div className="space-y-6">
        <div className="max-w-sm">
          <label htmlFor="documents-agent" className={labelClass}>
            Agent
          </label>
          <select id="documents-agent" value={selectedId} onChange={(e) => pick(e.target.value)} className={fieldClass}>
            {agents.map((a) => (
              <option key={a.id} value={a.id}>
                {optionLabel(a)}
              </option>
            ))}
          </select>
        </div>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem] lg:items-start">
          <div className="min-w-0">
            {/* Keyed so switching agents starts a fresh panel (no upload progress or open dialog carried over). */}
            {selected && <KnowledgePanel key={selected.id} agentId={selected.id} />}
          </div>
          {selected && (
            <Card variant="solid" className="space-y-3" role="region" aria-labelledby="documents-next">
              <h2 id="documents-next" className="font-display text-base font-semibold text-text-primary">
                Next step
              </h2>
              <p className="text-sm text-text-secondary">
                After adding files, ask {selected.name || "your agent"} a question to see what it says.
              </p>
              <div className="flex flex-wrap gap-2 lg:flex-col">
                <Link href={talkHref(selected.id)} className={cn(linkButton, "bg-brand text-[var(--color-on-accent)] hover:opacity-90")}>
                  <Mic className="size-4" aria-hidden /> Try it
                </Link>
                <Link
                  href={editHref(selected.id)}
                  className={cn(linkButton, "border border-border bg-bg-surface text-text-primary hover:border-text-primary")}
                >
                  <Pencil className="size-4" aria-hidden /> Edit agent
                </Link>
              </div>
            </Card>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-8 sm:py-10">
      <PageHeader title="Documents" description="The files your agent answers from." className="mb-3" />
      <p className="mb-8 text-sm text-text-secondary">
        <Link
          href={TRY_PAGE}
          className="inline-flex min-h-10 items-center rounded text-accent-primary underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-accent-primary"
        >
          Just trying a PDF?
        </Link>
      </p>
      {body}
    </div>
  );
}
