"use client";

import Link from "next/link";
import { ArrowRight, CheckCircle2, CircleAlert, Loader2 } from "lucide-react";
import type { ReactNode } from "react";
import { Skeleton } from "@/components/ui";
import { useHealth, useKnowledgeGaps, useOnboarding, useWarmup } from "@/hooks/useApi";
import { phoneHref, type HomeOnboarding } from "./homeTypes";

interface Item {
  id: string;
  text: ReactNode;
  href?: string;
  cta?: string;
  busy?: boolean;
}

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

/** Things that need the owner now, each from real data: nothing is listed that is not true. */
export function NeedsAttention() {
  const onboarding = useOnboarding();
  const gaps = useKnowledgeGaps();
  const warm = useWarmup();
  const health = useHealth();
  const home = onboarding.data as HomeOnboarding | undefined;

  if (onboarding.isLoading && !home) return <Skeleton className="h-24 w-full rounded-xl" />;

  const items: Item[] = [];
  const waiting = home?.approvals_waiting ?? 0;
  if (waiting > 0) {
    items.push({
      id: "approvals",
      text: `${plural(waiting, "meeting request is", "meeting requests are")} waiting for your decision`,
      href: "/dashboard/approvals",
      cta: "Review",
    });
  }
  const unanswered = gaps.data?.total_gap_turns ?? 0;
  if (unanswered > 0) {
    items.push({
      id: "unanswered",
      text: `Your agent could not answer ${plural(unanswered, "question", "questions")} in the last 30 days`,
      href: "/dashboard/analytics",
      cta: "See them",
    });
  }
  if (home?.has_agent && home.phone_connected === false) {
    items.push({
      id: "phone",
      text: "Your phone line is not connected, so nobody can call your agent yet",
      href: phoneHref(home),
      cta: "Connect phone",
    });
  }
  if (warm.data?.state === "running") {
    items.push({
      id: "loading",
      text: "Your agent's voice and language models are still loading. Replies may be slow for about 20 seconds.",
      busy: true,
    });
  }
  if (health.data && health.data.llm && !health.data.llm.active) {
    items.push({
      id: "no-llm",
      text: "Your agent cannot reply yet because no language model is set up",
      href: "#system-status",
      cta: "See how to fix",
    });
  }
  if (health.error && !health.data) {
    items.push({
      id: "server",
      text: "BonClay cannot reach its server right now",
      href: "#system-status",
      cta: "Check",
    });
  }

  return (
    <section aria-labelledby="attention-title">
      <h2 id="attention-title" className="mb-2 text-sm font-semibold text-text-primary">
        Needs your attention
        {items.length > 0 && (
          <span className="ml-2 rounded-full bg-accent-warning/15 px-2 py-0.5 text-xs text-accent-warning">{items.length}</span>
        )}
      </h2>
      {items.length === 0 ? (
        <p className="flex items-center gap-2 rounded-xl border border-border bg-bg-surface px-4 py-3 text-sm text-text-secondary">
          <CheckCircle2 className="size-4 shrink-0 text-accent-success" aria-hidden />
          Nothing needs your attention right now.
        </p>
      ) : (
        <ul className="divide-y divide-border rounded-xl border border-border bg-bg-surface">
          {items.map((it) => (
            <li key={it.id} className="flex items-center gap-3 px-4 py-3">
              {it.busy ? (
                <Loader2 className="size-4 shrink-0 animate-spin text-accent-primary motion-reduce:animate-none" aria-hidden />
              ) : (
                <CircleAlert className="size-4 shrink-0 text-accent-warning" aria-hidden />
              )}
              <p className="min-w-0 flex-1 text-sm text-text-primary">{it.text}</p>
              {it.href && (
                <Link
                  href={it.href}
                  className="inline-flex min-h-11 shrink-0 items-center gap-1 rounded-full px-2 text-sm font-medium text-accent-primary hover:underline sm:min-h-10"
                >
                  {it.cta}
                  <ArrowRight className="size-3.5" aria-hidden />
                </Link>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
