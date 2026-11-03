"use client";

import { Badge, Skeleton } from "@/components/ui";
import { useHealth, useModelStatus } from "@/hooks/useApi";
import { LLM_FIX_TEXT, llmLabel } from "@/lib/llm";

/** Short display names; the backend's full model ids are too long for a status strip. */
const ENGINE_LABEL: Record<string, string> = {
  kokoro: "Kokoro",
  qwen3_0_6b: "Qwen3",
  indic_f5: "IndicF5",
  indic_speak: "Indic-Speak",
  chatterbox_ml: "Chatterbox",
  dhvaani: "DhVaani",
};

export function HealthStrip() {
  const health = useHealth();
  const models = useModelStatus();
  const loading = health.isLoading || models.isLoading;
  const down = !!health.error || !health.data;

  return (
    <div
      className="flex flex-wrap items-center gap-2 rounded-xl border border-border bg-bg-surface px-4 py-3"
      aria-live="polite"
    >
      <span className="mr-1 text-xs font-medium text-text-secondary">Right now</span>
      {loading ? (
        <>
          <Skeleton className="h-5 w-20" />
          <Skeleton className="h-5 w-24" />
          <Skeleton className="h-5 w-16" />
        </>
      ) : down ? (
        <>
          <Badge tone="error" dot>
            Cannot reach the server
          </Badge>
          <button
            type="button"
            onClick={() => {
              health.mutate();
              models.mutate();
            }}
            className="rounded text-xs text-accent-secondary underline underline-offset-4 hover:text-text-primary"
          >
            Check again
          </button>
        </>
      ) : (
        <>
          <Badge tone={health.data!.status === "ok" ? "success" : "warning"} dot>
            {health.data!.status === "ok" ? "Server online" : "Server slow or partly down"}
          </Badge>
          {health.data!.platform ? (
            <Badge
              tone={health.data!.platform.accelerator === "apple" || health.data!.platform.accelerator === "nvidia" ? "success" : "neutral"}
              dot
            >
              {`${health.data!.platform.os} · ${health.data!.platform.accelerator_name}`}
            </Badge>
          ) : (
            <Badge tone={health.data!.gpu?.available ? "success" : "neutral"} dot>
              {health.data!.gpu?.available ? `GPU · ${health.data!.gpu.name ?? "CUDA"}` : "CPU mode"}
            </Badge>
          )}
          <Badge tone={health.data!.supabase?.configured ? (health.data!.supabase.reachable ? "success" : "error") : "neutral"} dot>
            {health.data!.supabase?.configured
              ? health.data!.supabase.reachable
                ? "Cloud storage online"
                : "Cloud storage unreachable"
              : "Local storage"}
          </Badge>
          {health.data!.llm &&
            (health.data!.llm.active ? (
              <Badge
                tone="success"
                dot
                title={health.data!.llm.model ? `Model: ${health.data!.llm.model}` : undefined}
              >
                Language model · {llmLabel(health.data!.llm.active, health.data!.llm.model)}
              </Badge>
            ) : (
              <Badge tone="warning" dot title={LLM_FIX_TEXT}>
                No language model
              </Badge>
            ))}
          {models.data?.engines.map((e) => (
            <Badge key={e.id} tone={e.loaded ? "success" : "neutral"} dot>
              {ENGINE_LABEL[e.id] ?? e.name} · {e.loaded ? "ready" : "loads when first used"}
            </Badge>
          ))}
        </>
      )}
    </div>
  );
}
