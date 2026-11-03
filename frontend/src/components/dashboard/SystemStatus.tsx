"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { useHealth, useSetupHints, useWarmup } from "@/hooks/useApi";
import { cn } from "@/lib/utils";
import { DemoModeCard } from "./DemoModeCard";
import { DemoReadinessCard } from "./DemoReadinessCard";
import { EnginePanel } from "./EnginePanel";
import { HealthStrip } from "./HealthStrip";
import { PerformancePanel } from "./PerformancePanel";
import { SetupHints } from "./SetupHints";
import { StatCards } from "./StatCards";

/** What is wrong right now, in plain words (empty when everything works). */
export function useSystemProblems(): string[] {
  const health = useHealth();
  const hints = useSetupHints();
  const warm = useWarmup();
  const problems: string[] = [];
  if (health.error && !health.data) problems.push("The server is not responding");
  else if (health.data && health.data.status !== "ok") problems.push("The server is running below full strength");
  if (health.data?.llm && !health.data.llm.active) problems.push("No language model is set up");
  if (health.data?.supabase?.configured && health.data.supabase.reachable === false) problems.push("Online storage is unreachable");
  const setup = (hints.data?.hints ?? []).filter((h) => h.severity === "error" || h.severity === "warning");
  if (setup.length > 0) problems.push(`${setup.length} setup ${setup.length === 1 ? "problem" : "problems"} on this computer`);
  if (warm.data?.state === "partial") problems.push("Some models failed to load");
  return problems;
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">{title}</h3>
      {children}
    </section>
  );
}

/** Engine, speed, sample data and model status: for the person who runs the computer, folded away unless something is wrong. */
export function SystemStatus() {
  const problems = useSystemProblems();
  // Folded away unless something is wrong; once the person opens or closes it, their choice wins.
  const [chosen, setChosen] = useState<boolean | null>(null);
  const hasProblems = problems.length > 0;
  const open = chosen ?? hasProblems;

  return (
    <section id="system-status" aria-labelledby="system-status-title" className="scroll-mt-20 rounded-xl border border-border bg-bg-surface">
      <h2 id="system-status-title" className="m-0">
        <button
          type="button"
          aria-expanded={open}
          aria-controls="system-status-body"
          onClick={() => setChosen(!open)}
          className="flex min-h-14 w-full items-center gap-3 rounded-xl px-4 py-3 text-left"
        >
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-semibold text-text-primary">System status</span>
            <span className={cn("block text-xs", hasProblems ? "text-accent-warning" : "text-text-secondary")}>
              {hasProblems ? problems.join(". ") : "Everything is working. Open for models, speed and sample data."}
            </span>
          </span>
          <ChevronDown
            className={cn("size-4 shrink-0 text-text-secondary transition-transform motion-reduce:transition-none", open && "rotate-180")}
            aria-hidden
          />
        </button>
      </h2>
      {open && (
        <div id="system-status-body" className="space-y-6 border-t border-border p-4 sm:p-5">
          <HealthStrip />
          <SetupHints />
          <Group title="Usage">
            <StatCards />
          </Group>
          <Group title="Speed">
            <PerformancePanel />
          </Group>
          <EnginePanel />
          <Group title="Get ready and sample data">
            <DemoReadinessCard />
            <DemoModeCard />
          </Group>
        </div>
      )}
    </section>
  );
}
