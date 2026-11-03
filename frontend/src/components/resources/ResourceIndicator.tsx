"use client";

import { useState } from "react";
import { FOOTER_POLL_MS, isSnapshot, useSystemResources } from "@/hooks/useSystemResources";
import { cn } from "@/lib/utils";
import { LEVEL_DOT, LEVEL_LABEL, gbNumber, levelExplanation, levelOf, modelCount, summarise, type Level } from "./format";
import { MemoryBar } from "./MemoryBar";
import { ModelsMemoryPanel } from "./ModelsMemoryPanel";

/**
 * Footer of the dashboard sidebar: a memory bar, "Memory 9.8 / 16 GB · 4 models" and a status dot. A button that opens
 * the "Models and memory" panel. `compact` (the collapsed sidebar) shows only the dot and the bar, with a tooltip.
 */
export function ResourceIndicator({ compact = false }: { compact?: boolean }) {
  const [open, setOpen] = useState(false);
  const { data, error } = useSystemResources(FOOTER_POLL_MS);
  const snap = isSnapshot(data) ? data : undefined;
  const level: Level | null = snap ? levelOf(snap.system) : null;

  // A screen-reader announcement only when the pressure level changes, never on an ordinary refresh. The message is
  // derived while rendering (React's "adjust state during render" pattern) so no effect is needed.
  const [seen, setSeen] = useState<Level | null>(null);
  const [announcement, setAnnouncement] = useState("");
  if (level !== seen) {
    setSeen(level);
    if (level && seen && snap) setAnnouncement(`Memory pressure is now ${LEVEL_LABEL[level].toLowerCase()}. ${levelExplanation(level, snap.system.platform)}`);
  }

  const label = snap
    ? summarise(snap)
    : error
      ? "Memory use is unavailable. Open the models and memory details."
      : "Reading memory use. Open the models and memory details.";
  const tooltip = snap
    ? `Memory ${gbNumber(snap.system.used_mb)} / ${gbNumber(snap.system.total_mb)} GB, ${modelCount(snap.models.length)}. Pressure: ${LEVEL_LABEL[level!]}`
    : error
      ? "Memory use is unavailable"
      : "Reading memory use";

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label={label}
        aria-haspopup="dialog"
        title={tooltip}
        data-testid="resource-indicator"
        className={cn(
          "w-full rounded-lg border border-transparent text-left text-xs text-text-secondary transition-colors hover:border-border hover:bg-bg-elevated hover:text-text-primary",
          "focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-accent-secondary)]",
          compact ? "grid place-items-center gap-1.5 px-1 py-2" : "px-2 py-1.5"
        )}
      >
        <span className={cn("flex items-center gap-2", compact && "justify-center")}>
          <span aria-hidden data-testid="resource-dot" className={cn("size-2 shrink-0 rounded-full", level ? LEVEL_DOT[level] : "bg-text-muted")} />
          {!compact && (
            <span className="min-w-0 truncate tabular-nums">
              {snap ? (
                <>
                  Memory {gbNumber(snap.system.used_mb)} / {gbNumber(snap.system.total_mb)} GB · {modelCount(snap.models.length)}
                </>
              ) : error ? (
                "Memory unavailable"
              ) : (
                "Memory…"
              )}
            </span>
          )}
        </span>
        {snap ? (
          <MemoryBar system={snap.system} level={level!} className={compact ? "w-8" : "mt-1.5"} />
        ) : (
          <span aria-hidden className={cn("block h-1.5 rounded-full bg-bg-elevated", compact ? "w-8" : "mt-1.5 w-full")} />
        )}
      </button>
      <span aria-live="polite" aria-atomic="true" data-testid="pressure-announcer" className="sr-only">
        {announcement}
      </span>
      <ModelsMemoryPanel open={open} onClose={() => setOpen(false)} />
    </>
  );
}
