"use client";

import { useState, type ComponentType, type ReactNode } from "react";
import {
  ClipboardCheck,
  Copy,
  Loader2,
  MessageSquare,
  Mic,
  Search,
  ListOrdered,
  ShieldCheck,
  Stamp,
  Volume2,
} from "lucide-react";
import { Modal } from "@/components/voices/Modal";
import { Button } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { PANEL_POLL_MS, isSnapshot, useSystemResources } from "@/hooks/useSystemResources";
import { cn } from "@/lib/utils";
import type { ResourceModel, ResourceModelKind, SystemResources } from "@/types/api";
import {
  DEVICE_LABEL,
  LEVEL_DOT,
  LEVEL_FILL,
  LEVEL_LABEL,
  formatMb,
  levelExplanation,
  levelOf,
  stateText,
  type Level,
} from "./format";
import { MemoryBar } from "./MemoryBar";

const KIND_ICON: Record<ResourceModelKind, ComponentType<{ className?: string }>> = {
  voice: Volume2,
  clone: Copy,
  speech_recognition: Mic,
  search: Search,
  reranker: ListOrdered,
  watermark: Stamp,
  speaker_check: ShieldCheck,
  language_model: MessageSquare,
  audit: ClipboardCheck,
};

const KIND_NAME: Record<ResourceModelKind, string> = {
  voice: "Voice",
  clone: "Voice cloning",
  speech_recognition: "Speech recognition",
  search: "Search",
  reranker: "Re-ranking",
  watermark: "Watermark",
  speaker_check: "Speaker check",
  language_model: "Language model",
  audit: "Answer check",
};

const STATE_TONE: Record<ResourceModel["state"], string> = {
  "pinned (live call)": "border-accent-primary/40 bg-accent-primary/15 text-accent-primary",
  busy: "border-accent-warning/40 bg-accent-warning/15 text-accent-warning",
  loading: "border-accent-warning/40 bg-accent-warning/15 text-accent-warning",
  idle: "border-border bg-bg-elevated text-text-secondary",
  loaded: "border-accent-success/40 bg-accent-success/15 text-accent-success",
};

interface PanelProps {
  open: boolean;
  onClose: () => void;
}

/** The "Models and memory" dialog. Its contents mount only while it is open, so it polls (every 3 s) only then. */
export function ModelsMemoryPanel({ open, onClose }: PanelProps) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      size="lg"
      title="Models and memory"
      description="What is loaded on this computer right now, and how much room is left. Refreshes every few seconds."
    >
      <PanelBody />
    </Modal>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-6 first:mt-0">
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary">{title}</h3>
      {children}
    </section>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="min-w-0 rounded-lg border border-border bg-bg-base px-3 py-2">
      <dt className="text-[11px] text-text-secondary">{label}</dt>
      <dd className="mt-0.5 truncate text-sm font-medium tabular-nums text-text-primary" title={hint}>
        {value}
      </dd>
    </div>
  );
}

function PanelBody() {
  const { data, error, mutate } = useSystemResources(PANEL_POLL_MS);
  const snap = isSnapshot(data) ? data : undefined;

  if (!snap) {
    return error ? (
      <p role="alert" className="text-sm text-accent-error">
        Could not read memory use from the server{error.message ? `: ${error.message}` : "."}
      </p>
    ) : (
      <p className="text-sm text-text-secondary">Reading memory use…</p>
    );
  }
  const { system, app, models, guard } = snap;
  const level = levelOf(system);
  const sorted = [...models].sort((a, b) => b.memory_mb - a.memory_mb);

  return (
    <div>
      <Section title="This computer">
        <PressureBanner level={level} platform={system.platform} />
        <div className="mt-3">
          <MemoryBar system={system} level={level} className="h-2.5" />
          <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-text-secondary" aria-label="Legend">
            <li className="flex items-center gap-1.5">
              <span aria-hidden className={cn("size-2 rounded-sm", LEVEL_FILL[level])} /> In use
            </li>
            {system.compressed_mb != null && (
              <li className="flex items-center gap-1.5">
                <span aria-hidden className="size-2 rounded-sm bg-accent-secondary/60" /> Compressed
              </li>
            )}
            <li className="flex items-center gap-1.5">
              <span aria-hidden className="size-2 rounded-sm border border-border bg-bg-elevated" /> Available
            </li>
          </ul>
        </div>
        <dl className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
          <Stat label="Total memory" value={formatMb(system.total_mb)} />
          <Stat label="In use" value={formatMb(system.used_mb)} />
          <Stat label="Available" value={formatMb(system.available_mb)} />
          {system.compressed_mb != null && (
            <Stat label="Compressed" value={formatMb(system.compressed_mb)} hint="Memory macOS has squeezed to make room" />
          )}
          <Stat
            label="Swap"
            value={system.swap_total_mb > 0 ? `${formatMb(system.swap_used_mb)} of ${formatMb(system.swap_total_mb)}` : "None"}
            hint="Memory pushed out to the disk, which is slow"
          />
          <Stat label="Pressure" value={LEVEL_LABEL[level]} />
          {system.gpu && (
            <Stat
              label="Graphics memory"
              value={
                system.gpu.used_mb != null ? `${formatMb(system.gpu.used_mb)} of ${formatMb(system.gpu.total_mb)}` : formatMb(system.gpu.total_mb)
              }
              hint={system.gpu.name}
            />
          )}
        </dl>
      </Section>

      <Section title="BonClay">
        <dl className="grid grid-cols-3 gap-2">
          <Stat label="The app (API)" value={formatMb(app.api_rss_mb)} />
          <Stat label="Model workers" value={formatMb(app.workers_rss_mb)} />
          <Stat label="Total" value={formatMb(app.total_rss_mb)} />
        </dl>
      </Section>

      <Section title={`Models loaded (${models.length})`}>
        {sorted.length === 0 ? (
          <p className="rounded-lg border border-dashed border-border px-3 py-4 text-sm text-text-secondary">
            No models are loaded. They load the first time they are used.
          </p>
        ) : (
          <ul className="space-y-2">
            {sorted.map((m) => (
              <ModelRow key={m.key} model={m} totalMb={system.total_mb} onUnloaded={() => mutate()} />
            ))}
          </ul>
        )}
        <p className="mt-3 text-xs leading-relaxed text-text-secondary">
          <span className="font-medium text-text-primary">What uses memory.</span> A model that runs as its own worker process is{" "}
          <span className="font-medium">measured</span>: the memory of that process. A model inside the app is an{" "}
          <span className="font-medium">estimate</span>
          {" "}(the size of its weights), because the app&apos;s memory cannot be split up by model.
          {guard.enabled
            ? ` BonClay keeps about ${formatMb(guard.reserve_mb)} free and unloads idle models, or says so, before it loads a big one.`
            : " The memory guard is switched off."}
        </p>
      </Section>
    </div>
  );
}

function PressureBanner({ level, platform }: { level: Level; platform: SystemResources["system"]["platform"] }) {
  const tone =
    level === "critical"
      ? "border-accent-error/40 bg-accent-error/10"
      : level === "warn"
        ? "border-accent-warning/40 bg-accent-warning/10"
        : "border-border bg-bg-base";
  return (
    <div data-testid="pressure-banner" data-level={level} className={cn("flex items-start gap-2.5 rounded-lg border px-3 py-2.5", tone)}>
      <span aria-hidden className={cn("mt-1 size-2.5 shrink-0 rounded-full", LEVEL_DOT[level])} />
      <div className="min-w-0">
        <p className="text-[11px] text-text-secondary">Memory pressure</p>
        <p className="text-sm font-medium text-text-primary">{levelExplanation(level, platform)}</p>
      </div>
    </div>
  );
}

function ModelRow({ model: m, totalMb, onUnloaded }: { model: ResourceModel; totalMb: number; onUnloaded: () => void }) {
  const Icon = KIND_ICON[m.kind] ?? Volume2;
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const canUnload = m.can_unload ?? (m.state === "idle" || m.state === "loaded");
  const share = Math.min(100, Math.max(0, (m.memory_mb / Math.max(1, totalMb)) * 100));
  const estimated = m.memory_source === "estimated";

  const unload = async () => {
    setBusy(true);
    setMessage(null);
    try {
      await api.post(`/api/system/models/${encodeURIComponent(m.key)}/unload`);
      setConfirming(false);
      onUnloaded();
    } catch (e) {
      setMessage(e instanceof ApiError && e.message ? e.message : "Could not unload this model.");
      setConfirming(false);
      onUnloaded();
    } finally {
      setBusy(false);
    }
  };

  return (
    <li data-testid="model-row" data-key={m.key} className="rounded-lg border border-border bg-bg-base p-3">
      <div className="flex items-start gap-3">
        <span aria-hidden className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-md bg-bg-elevated text-text-secondary">
          <Icon className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <p className="min-w-0 text-sm font-medium text-text-primary">{m.label}</p>
            <span className="rounded-md border border-border bg-bg-elevated px-1.5 py-0.5 text-[11px] font-medium text-text-secondary">
              {DEVICE_LABEL[m.device] ?? m.device}
            </span>
            <span className={cn("rounded-md border px-1.5 py-0.5 text-[11px] font-semibold", STATE_TONE[m.state] ?? STATE_TONE.loaded)}>
              {stateText(m)}
            </span>
          </div>
          <p className="mt-0.5 text-xs text-text-secondary">
            {KIND_NAME[m.kind] ?? m.kind} · {m.where === "worker process" ? "own process" : "inside the app"}
          </p>
          <div className="mt-2 flex items-center gap-3">
            <div
              aria-hidden
              className="h-1.5 flex-1 overflow-hidden rounded-full bg-bg-elevated"
              title={`${share.toFixed(1)}% of this computer's memory`}
            >
              <div
                className={cn("h-full rounded-full", estimated ? "" : "bg-accent-primary")}
                style={{
                  width: `${Math.max(share, 1.5)}%`,
                  ...(estimated
                    ? {
                        backgroundImage:
                          "repeating-linear-gradient(135deg, var(--color-accent-primary) 0 4px, transparent 4px 7px)",
                      }
                    : {}),
                }}
              />
            </div>
            <p className="shrink-0 text-xs tabular-nums text-text-primary">
              <span className="font-semibold">{estimated ? "about " : ""}{formatMb(m.memory_mb)}</span>{" "}
              <span className="text-text-secondary">{estimated ? "estimated" : "measured"}</span>
            </p>
          </div>
        </div>
        {canUnload && !confirming && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              setMessage(null);
              setConfirming(true);
            }}
            aria-label={`Unload ${m.label}`}
          >
            Unload
          </Button>
        )}
      </div>

      {confirming && (
        <div role="group" aria-label={`Confirm unloading ${m.label}`} className="mt-3 rounded-md border border-border bg-bg-elevated p-3">
          <p className="text-sm text-text-primary">
            Unload {m.label}? It frees about {formatMb(m.memory_mb)} and loads again, a few seconds slower, the next time it is used.
          </p>
          <div className="mt-2 flex flex-wrap justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => setConfirming(false)} disabled={busy} data-autofocus>
              Keep it loaded
            </Button>
            <Button size="sm" variant="outline" onClick={unload} disabled={busy} aria-label={`Confirm unload ${m.label}`}>
              {busy && <Loader2 className="size-3.5 animate-spin" aria-hidden />}
              Unload now
            </Button>
          </div>
        </div>
      )}
      {message && (
        <p role="alert" className="mt-2 text-xs text-accent-error">
          {message}
        </p>
      )}
    </li>
  );
}
