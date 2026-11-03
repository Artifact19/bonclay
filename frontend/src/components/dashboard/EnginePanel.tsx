"use client";

import { useState } from "react";
import { Flame, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Badge, Button, Card, Skeleton } from "@/components/ui";
import { useHealth, useModelStatus } from "@/hooks/useApi";
import { LLM_FIX_TEXT, llmLabel } from "@/lib/llm";
import type { LlmStatus } from "@/types/api";
import { api } from "@/lib/api";
import { errMessage } from "./format";

const ENGINE_ROLE: Record<string, string> = {
  kokoro: "Fast path · preset voices",
  qwen3_0_6b: "Zero-shot voice cloning",
  indic_f5: "Indian languages · cloning",
  indic_speak: "Indian languages · preset voices",
  chatterbox_ml: "Hindi · cloning",
  dhvaani: "Indian languages · cloning",
};

/** Live model-manager state with one-click warm-up, so engines can be loaded before a demo or a burst of traffic. */
export function EnginePanel() {
  const models = useModelStatus();
  const health = useHealth();
  const llm = health.data?.llm;
  const [warmingLlm, setWarmingLlm] = useState(false);
  const [warming, setWarming] = useState<string | null>(null);

  const warmUp = async (engine: string) => {
    setWarming(engine);
    try {
      const res = await api.post<{ seconds: number }>("/api/models/preload", { engine });
      toast.success(`Engine ready in ${res?.seconds ?? "?"} s`);
      await models.mutate();
    } catch (e) {
      toast.error(errMessage(e, "Warm-up failed."));
    } finally {
      setWarming(null);
    }
  };

  const warmLlm = async () => {
    setWarmingLlm(true);
    try {
      await api.post<{ ok: boolean; llm: LlmStatus }>("/api/llm/warm");
      toast.success("Language model ready.");
      await health.mutate();
    } catch (e) {
      toast.error(errMessage(e, "Warm-up failed."));
    } finally {
      setWarmingLlm(false);
    }
  };

  return (
    <section aria-labelledby="engines-heading" className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h2 id="engines-heading" className="text-sm font-semibold text-text-primary">
          Voice and language models
        </h2>
        <span className="text-xs text-text-secondary">
          {models.data ? `${models.data.device.toUpperCase()} · ${models.data.dtype} · ` : ""}unused models unload after 5 minutes
        </span>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {models.isLoading &&
          [0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-28 w-full rounded-xl" />)}
        {llm && (
          <Card variant="solid" className="flex flex-col gap-3 md:col-span-2">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate font-medium text-text-primary">Language model</p>
                <p className="text-xs text-text-secondary">
                  {llm.active
                    ? `${llmLabel(llm.active, llm.model)}${llm.model ? ` · ${llm.model}` : ""}`
                    : LLM_FIX_TEXT}
                </p>
              </div>
              <Badge tone={llm.active ? (llm.active === "mlx" && !llm.local_running ? "neutral" : "success") : "warning"} dot>
                {!llm.active ? "Not configured" : llm.active === "mlx" ? (llm.local_running ? "Warm" : "Cold") : "Cloud"}
              </Badge>
            </div>
            <div className="flex flex-wrap gap-1">
              {llm.providers.map((p) => (
                <span
                  key={p.name}
                  title={p.reason ?? undefined}
                  className="rounded bg-bg-elevated px-1.5 py-0.5 font-mono text-[10px] uppercase text-text-secondary"
                >
                  {p.name} · {p.available ? "available" : "unavailable"}
                </span>
              ))}
              {Object.entries(llm.usage_today ?? {}).map(([name, u]) => (
                <span key={name} className="rounded bg-bg-elevated px-1.5 py-0.5 text-[10px] tabular-nums text-text-secondary">
                  {name} today: {u.requests} req · {u.tokens} tok
                </span>
              ))}
            </div>
            <div className="flex justify-end">
              <Button variant="outline" onClick={warmLlm} disabled={warmingLlm || llm.local_running || !llm.active}>
                {warmingLlm ? <Loader2 className="h-4 w-4 animate-spin" /> : <Flame className="h-4 w-4" />}
                {warmingLlm ? "Loading" : llm.local_running ? "Ready" : "Warm up"}
              </Button>
            </div>
          </Card>
        )}
        {models.data?.engines.map((e) => (
          <Card key={e.id} variant="solid" className="flex flex-col gap-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate font-medium text-text-primary">{e.name}</p>
                <p className="text-xs text-text-secondary">{ENGINE_ROLE[e.id] ?? e.type}</p>
              </div>
              <Badge tone={e.loaded ? "success" : "neutral"} dot>
                {e.loaded ? `Warm · idle ${Math.round(e.idle_s ?? 0)}s` : "Cold"}
              </Badge>
            </div>
            <div className="flex flex-wrap gap-1">
              {e.languages.map((l) => (
                <span key={l} className="rounded bg-bg-elevated px-1.5 py-0.5 font-mono text-[10px] uppercase text-text-secondary">
                  {l}
                </span>
              ))}
            </div>
            <div className="flex justify-end">
              <Button variant="outline" onClick={() => warmUp(e.id)} disabled={warming !== null || e.loaded}>
                {warming === e.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Flame className="h-4 w-4" />}
                {warming === e.id ? "Loading" : e.loaded ? "Ready" : "Warm up"}
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </section>
  );
}
