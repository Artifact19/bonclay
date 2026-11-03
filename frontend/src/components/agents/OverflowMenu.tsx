"use client";

import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { MoreHorizontal } from "lucide-react";
import { cn } from "@/lib/utils";

export interface MenuItem {
  label: string;
  icon?: ReactNode;
  onSelect: () => void;
  danger?: boolean;
}

/** A "…" button with a small menu. Closes on Escape, on an outside click and after a choice. */
export function OverflowMenu({ label, items, className }: { label: string; items: MenuItem[]; className?: string }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const menuId = useId();

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        trigger.current?.focus();
      }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={root} className={cn("relative", className)}>
      <button
        ref={trigger}
        type="button"
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => setOpen((o) => !o)}
        className="grid size-10 place-items-center rounded-full border border-border bg-bg-surface text-text-secondary transition-colors hover:border-text-primary hover:text-text-primary focus-visible:outline-2 focus-visible:outline-accent-primary"
      >
        <MoreHorizontal className="size-4" aria-hidden />
      </button>
      {open && (
        <ul
          id={menuId}
          role="menu"
          aria-label={label}
          className="absolute right-0 z-30 mt-1.5 min-w-48 overflow-hidden rounded-xl border border-border bg-bg-surface py-1 shadow-xl"
        >
          {items.map((item) => (
            <li key={item.label} role="none">
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setOpen(false);
                  item.onSelect();
                }}
                className={cn(
                  "flex min-h-11 w-full items-center gap-2.5 px-3.5 py-2 text-left text-sm transition-colors hover:bg-bg-elevated focus-visible:bg-bg-elevated focus-visible:outline-none",
                  item.danger ? "text-accent-error" : "text-text-primary"
                )}
              >
                {item.icon}
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
