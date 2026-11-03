"use client";

import { useState } from "react";
import Link from "next/link";
import { FlaskConical, Loader2, Play, RotateCcw } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { AGENTS_KEY, DEMO_STATUS_KEY, useDemoStatus, useHealth } from "@/hooks/useApi";
import { useSWRConfig } from "swr";
import { Button, Card, TaskProgressBar, useEstimatedProgress } from "@/components/ui";
import { errMessage } from "@/components/dashboard/format";
import type { DemoLoadResult } from "@/types/api";

const linkClass = "font-medium text-accent-primary underline-offset-2 hover:underline";

/** "Sample data": two ready agents with documents and a finished safety check, in one click. Local storage only. */
export function DemoModeCard() {
  const health = useHealth();
  const cloud = health.data?.supabase?.configured === true;
  const status = useDemoStatus(!cloud);
  const { mutate } = useSWRConfig();
  const [busy, setBusy] = useState<"load" | "reset" | null>(null);

  const loaded = status.data?.loaded === true;
  const running = loaded && status.data?.evaluation_running === true && !status.data.evaluation_ready;
  const evalPct = useEstimatedProgress(busy === "load" || running, busy === "load" ? 6 : 35, loaded && status.data?.evaluation_ready === true);

  if (cloud) return null;

  const refresh = async () => {
    await Promise.all([status.mutate(), mutate(AGENTS_KEY), mutate(DEMO_STATUS_KEY)]);
  };

  const load = async () => {
    setBusy("load");
    try {
      await api.post<DemoLoadResult>("/api/demo/load");
      await refresh();
      toast.success("Sample data loaded. The safety check is running.");
    } catch (e) {
      toast.error(errMessage(e, "Could not load the sample data."));
    } finally {
      setBusy(null);
    }
  };

  const reset = async () => {
    setBusy("reset");
    try {
      await api.post<{ reset: boolean; removed: number }>("/api/demo/reset");
      await refresh();
      toast.success("Sample data removed.");
    } catch (e) {
      toast.error(errMessage(e, "Could not remove the sample data."));
    } finally {
      setBusy(null);
    }
  };

  let line: React.ReactNode = "Not loaded yet.";
  if (status.error) line = "Sample data status is unavailable right now.";
  else if (loaded && running) {
    line = (
      <span className="inline-flex items-center gap-1.5">
        <Loader2 className="size-3.5 animate-spin" aria-hidden /> Sample data loaded. The safety check is running, results in about a minute.
      </span>
    );
  } else if (loaded && status.data?.evaluation_ready) {
    line = (
      <>
        Sample data is ready. Open <Link href="/dashboard/studio/agents" className={linkClass}>Agents</Link> or see the
        finished check in <Link href="/dashboard/eval" className={linkClass}>Safety check</Link>.
      </>
    );
  } else if (loaded) {
    line = (
      <>
        Sample agents loaded. Open <Link href="/dashboard/studio/agents" className={linkClass}>Agents</Link>, or run
        the check from <Link href="/dashboard/eval" className={linkClass}>Safety check</Link>.
      </>
    );
  }

  return (
    <Card variant="solid" className="flex flex-col gap-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <span className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent-primary/10 text-accent-primary">
            <FlaskConical className="size-4" aria-hidden />
          </span>
          <div className="min-w-0">
            <h2 className="font-display text-sm font-semibold text-text-primary">Sample data</h2>
            <p className="text-sm text-text-secondary">
              Try BonClay with two ready-made agents, their documents and a finished safety check. Remove them any time.
            </p>
            <p role="status" className="mt-1 text-xs text-text-muted">{line}</p>
          </div>
        </div>
        <div className="flex shrink-0 gap-2">
          {loaded && (
            <Button variant="outline" size="sm" onClick={reset} loading={busy === "reset"} disabled={busy !== null}
                    leftIcon={<RotateCcw className="size-3.5" />}>
              Remove sample data
            </Button>
          )}
          <Button size="sm" onClick={load} loading={busy === "load"} disabled={busy !== null || loaded}
                  leftIcon={<Play className="size-3.5" />}>
            Load sample data
          </Button>
        </div>
      </div>
      {(busy === "load" || running) && (
        <TaskProgressBar
          value={evalPct}
          label={busy === "load" ? "Loading sample agents and documents..." : "Running background safety checks..."}
          active
          testId="demo-mode-progress"
        />
      )}
    </Card>
  );
}
