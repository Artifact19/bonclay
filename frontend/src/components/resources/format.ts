import type { MemoryPressure, ResourceModel, SystemResources } from "@/types/api";

/** 9830 -> "9.6 GB", 600 -> "600 MB". One decimal, no trailing ".0". */
export function formatMb(mb: number): string {
  if (!Number.isFinite(mb) || mb <= 0) return "0 MB";
  if (mb < 1024) return `${Math.round(mb)} MB`;
  const gb = mb / 1024;
  return `${gb >= 100 ? Math.round(gb) : gb.toFixed(1).replace(/\.0$/, "")} GB`;
}

/** GB number only, for "9.8 / 16 GB". */
export function gbNumber(mb: number): string {
  const gb = mb / 1024;
  return gb >= 100 ? String(Math.round(gb)) : gb.toFixed(1).replace(/\.0$/, "");
}

export function modelCount(n: number): string {
  return `${n} ${n === 1 ? "model" : "models"}`;
}

/** Idle time in plain words: "45 s", "3 min", "2 h". */
export function formatIdle(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))} s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  return `${Math.round(seconds / 3600)} h`;
}

export type Level = MemoryPressure | "unknown";

/**
 * How worried to look. macOS reports a pressure level; elsewhere (or when it cannot be read) the amount of memory still
 * available stands in for it.
 */
export function levelOf(system: SystemResources["system"]): Level {
  if (system.pressure) return system.pressure;
  if (!system.total_mb) return "unknown";
  const free = system.available_mb / system.total_mb;
  if (free < 0.06) return "critical";
  if (free < 0.12) return "warn";
  return "normal";
}

export const LEVEL_LABEL: Record<Level, string> = {
  normal: "Normal",
  warn: "Busy",
  critical: "Critical",
  unknown: "Unknown",
};

/** The plain-words explanation of the pressure level. */
export function levelExplanation(level: Level, platform: SystemResources["system"]["platform"]): string {
  const machine = platform === "darwin" ? "the Mac" : "this computer";
  if (level === "normal") return "Normal: plenty of room.";
  if (level === "warn") {
    return platform === "darwin"
      ? "Busy: macOS is compressing memory."
      : "Busy: memory is getting tight and the system may start swapping.";
  }
  if (level === "critical") return `Critical: ${machine} is short of memory; unload something.`;
  return "Memory pressure is not reported on this system.";
}

export const LEVEL_DOT: Record<Level, string> = {
  normal: "bg-accent-success",
  warn: "bg-accent-warning",
  critical: "bg-accent-error",
  unknown: "bg-text-muted",
};

/** Fill colour of the "used" part of a bar. */
export const LEVEL_FILL: Record<Level, string> = {
  normal: "bg-accent-primary",
  warn: "bg-accent-warning",
  critical: "bg-accent-error",
  unknown: "bg-text-muted",
};

export interface Split {
  /** Percent of total memory: in use (not counting compressed), compressed, available. */
  used: number;
  compressed: number;
  available: number;
}

/** The three parts of the memory bar. Compressed memory is part of "used" on macOS, so it is split out of it. */
export function splitMemory(system: SystemResources["system"]): Split {
  const total = Math.max(1, system.total_mb);
  const compressed = Math.min(system.compressed_mb ?? 0, system.used_mb);
  const used = Math.max(0, system.used_mb - compressed);
  const pct = (n: number) => Math.min(100, Math.max(0, (n / total) * 100));
  const u = pct(used);
  const c = pct(compressed);
  return { used: u, compressed: c, available: Math.max(0, 100 - u - c) };
}

export const DEVICE_LABEL: Record<ResourceModel["device"], string> = {
  mlx: "Apple GPU",
  mps: "Apple GPU",
  cuda: "NVIDIA",
  cpu: "CPU",
};

/** What the state chip says: "Pinned during a call", "Busy", "Idle 3 min", "Loaded", "Loading". */
export function stateText(m: ResourceModel): string {
  switch (m.state) {
    case "pinned (live call)":
      return "Pinned during a call";
    case "busy":
      return "Busy";
    case "loading":
      return "Loading";
    case "idle":
      return m.idle_s != null ? `Idle ${formatIdle(m.idle_s)}` : "Idle";
    default:
      return "Loaded";
  }
}

/** The one-sentence summary used as the footer button's accessible name. */
export function summarise(s: SystemResources): string {
  const level = levelOf(s.system);
  const parts = [
    `Memory: ${gbNumber(s.system.used_mb)} of ${gbNumber(s.system.total_mb)} GB used`,
    modelCount(s.models.length) + " loaded",
  ];
  if (s.system.compressed_mb) parts.push(`${formatMb(s.system.compressed_mb)} compressed`);
  if (s.system.swap_used_mb > 0) parts.push(`${formatMb(s.system.swap_used_mb)} swapped`);
  parts.push(`pressure ${LEVEL_LABEL[level].toLowerCase()}`);
  return `${parts.join(", ")}. Open the models and memory details.`;
}
