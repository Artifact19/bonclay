"use client";

import { useRef, type KeyboardEvent } from "react";
import { useRouter } from "next/navigation";
import { AgentsIcon, StudioIcon } from "./icons/NavIcons";
import { AREAS, AREA_HOME, type Area } from "@/lib/areas";
import { cn } from "@/lib/utils";

const ICONS = { studio: StudioIcon, agents: AgentsIcon } as const;

/**
 * Studio | Agents: a two-option radio group. Arrow keys (and Home/End) move to the other option and select it, as a
 * native radio group does; Tab leaves the group. Each option is 44 px tall and carries its own area accent.
 * `compact` (the collapsed sidebar) stacks the two options as icon-only buttons.
 */
export function AreaSwitch({ area, onChange, compact = false }: { area: Area; onChange: (next: Area) => void; compact?: boolean }) {
  const router = useRouter();
  const refs = useRef<Partial<Record<Area, HTMLButtonElement | null>>>({});

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = AREAS.findIndex((a) => a.id === area);
    let next = -1;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (i + 1) % AREAS.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (i - 1 + AREAS.length) % AREAS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = AREAS.length - 1;
    if (next < 0) return;
    e.preventDefault();
    const target = AREAS[next].id;
    onChange(target);
    refs.current[target]?.focus();
  };

  return (
    <div
      role="radiogroup"
      aria-label="Area"
      onKeyDown={onKeyDown}
      className={cn("rounded-full border-[1.5px] border-[#141414] bg-[#E2DDD3] p-1 shadow-[0_2px_0_#141414]", compact ? "flex flex-col gap-1 rounded-2xl" : "grid grid-cols-2 gap-1")}
    >
      {AREAS.map(({ id, label, hint }) => {
        const checked = id === area;
        const Icon = ICONS[id];
        return (
          <button
            key={id}
            ref={(el) => {
              refs.current[id] = el;
            }}
            type="button"
            role="radio"
            aria-checked={checked}
            aria-label={compact ? `${label}: ${hint}` : undefined}
            title={compact ? `${label}: ${hint}` : hint}
            tabIndex={checked ? 0 : -1}
            data-area={id}
            onMouseEnter={() => {
              try { router.prefetch(AREA_HOME[id]); } catch {}
            }}
            onFocus={() => {
              try { router.prefetch(AREA_HOME[id]); } catch {}
            }}
            onClick={() => onChange(id)}
            className={cn(
              "flex h-11 min-w-0 items-center justify-center gap-2 rounded-full text-sm font-semibold outline-none transition-all duration-150",
              checked
                ? "bg-[#141414] text-[#FAF9F5] shadow-sm"
                : "text-[#141414]/75 hover:bg-[#FAF9F5]/80 hover:text-[#141414]"
            )}
          >
            <Icon className="size-4 shrink-0" active={checked} />
            {!compact && <span className="truncate">{label}</span>}
          </button>
        );
      })}
    </div>
  );
}
