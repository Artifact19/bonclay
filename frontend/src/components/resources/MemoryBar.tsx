import type { SystemResources } from "@/types/api";
import { cn } from "@/lib/utils";
import { LEVEL_FILL, splitMemory, type Level } from "./format";

interface Props {
  system: SystemResources["system"];
  level: Level;
  className?: string;
}

/** Used (colour follows memory pressure), compressed (hatched) and available (the empty track) of the machine's memory. */
export function MemoryBar({ system, level, className }: Props) {
  const s = splitMemory(system);
  return (
    <div aria-hidden data-testid="memory-bar" className={cn("flex h-1.5 w-full overflow-hidden rounded-full bg-bg-elevated", className)}>
      <span data-part="used" className={cn("h-full transition-[width] duration-500", LEVEL_FILL[level])} style={{ width: `${s.used}%` }} />
      <span
        data-part="compressed"
        className="h-full bg-accent-secondary/60 transition-[width] duration-500"
        style={{ width: `${s.compressed}%` }}
      />
    </div>
  );
}
