"use client";

import { useState } from "react";
import { CheckCircle2, Circle, Flame, Loader2, MinusCircle, XCircle } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { useWarmup } from "@/hooks/useApi";
import { Button, Card, TaskProgressBar } from "@/components/ui";
import { errMessage } from "@/components/dashboard/format";
import type { WarmupComponent, WarmupComponentState, WarmupStatus } from "@/types/api";

const STATE_LABEL: Record<WarmupComponentState, string> = {
  pending: "waiting",
  warming: "loading",
  ready: "ready",
  skipped: "skipped",
  failed: "failed",
};

function StateIcon({ state }: { state: WarmupComponentState }) {
  const cls = "mt-0.5 size-4 shrink-0";
  if (state === "ready") return <CheckCircle2 className={`${cls} text-accent-success`} aria-hidden />;
  if (state === "warming") return <Loader2 className={`${cls} animate-spin text-accent-primary`} aria-hidden />;
  if (state === "failed") return <XCircle className={`${cls} text-accent-error`} aria-hidden />;
  if (state === "skipped") return <MinusCircle className={`${cls} text-accent-warning`} aria-hidden />;
  return <Circle className={`${cls} text-text-muted`} aria-hidden />;
}

function Row({ c }: { c: WarmupComponent }) {
  return (
    <li className="flex items-start gap-2.5 py-1.5">
      <StateIcon state={c.state} />
      <div className="min-w-0 flex-1">
        <p className="text-sm text-text-primary">
          {c.label}
          <span className="sr-only">: {STATE_LABEL[c.state]}</span>
        </p>
        {c.detail && (c.state === "failed" || c.state === "skipped") && (
          <p className="break-words text-xs text-text-muted">{c.detail}</p>
        )}
      </div>
      {c.seconds != null && c.state !== "pending" && c.state !== "warming" && (
        <span className="shrink-0 font-mono text-xs tabular-nums text-text-muted">{c.seconds.toFixed(1)} s</span>
      )}
    </li>
  );
}

/** Loads every model a live demo needs, so the first spoken reply is as fast as the rest. */
export function DemoReadinessCard() {
  const warm = useWarmup();
  const [starting, setStarting] = useState(false);
  const data = warm.data;
  const running = data?.state === "running";
  const ready = data?.state === "ready";
  const partial = data?.state === "partial";

  const start = async () => {
    setStarting(true);
    try {
      const next = await api.post<WarmupStatus>("/api/warmup");
      await warm.mutate(next, { revalidate: false });
    } catch (e) {
      toast.error(errMessage(e, "Could not start loading."));
    } finally {
      setStarting(false);
    }
  };

  let line: React.ReactNode = "Voices and models load the first time they are used, so the first spoken reply can take about 18 seconds.";
  if (warm.error && !data) line = "Status is unavailable right now.";
  else if (running) line = "Getting ready. This takes about 20 seconds.";
  else if (ready) line = "Ready. The first replies will be fast.";
  else if (partial) line = "Partly ready: something failed to load. The rest is ready.";

  const totalComps = data?.components?.length ?? 0;
  const doneComps =
    data?.components?.filter((c) => c.state === "ready" || c.state === "skipped" || c.state === "failed").length ?? 0;
  const warmingComps = data?.components?.filter((c) => c.state === "warming").length ?? 0;
  const pct =
    totalComps > 0
      ? Math.round(((doneComps + warmingComps * 0.4) / totalComps) * 100)
      : ready
        ? 100
        : running || starting
          ? 15
          : 0;

  return (
    <Card variant="solid">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <span className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent-primary/10 text-accent-primary">
            <Flame className="size-4" aria-hidden />
          </span>
          <div className="min-w-0">
            <h2 className="font-display text-sm font-semibold text-text-primary">Make the first reply fast</h2>
            <p role="status" className="mt-0.5 text-sm text-text-secondary">{line}</p>
          </div>
        </div>
        <Button size="sm" onClick={start} loading={starting || running} disabled={starting || running}
                leftIcon={<Flame className="size-3.5" />}>
          {ready ? "Load again" : "Get everything ready"}
        </Button>
      </div>
      {(running || starting || ready || partial) && totalComps > 0 && (
        <TaskProgressBar
          value={ready ? 100 : pct}
          label={running || starting ? "Warming up AI engines" : "Engine warm-up complete"}
          detail={`${doneComps} of ${totalComps} models checked`}
          active={running || starting}
          tone={ready ? "success" : partial ? "warning" : "accent"}
          className="mt-3"
          testId="warmup-progress"
        />
      )}
      {data && data.components.length > 0 && (
        <ul className="mt-3 divide-y divide-border border-t border-border pt-2 sm:columns-2 sm:gap-8 sm:divide-y-0">
          {data.components.map((c) => (
            <Row key={c.id} c={c} />
          ))}
        </ul>
      )}
      {data?.not_warmed && (
        <p className="mt-2 border-t border-border pt-2 text-xs text-text-muted">{data.not_warmed}</p>
      )}
    </Card>
  );
}
