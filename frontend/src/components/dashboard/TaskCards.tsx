"use client";

import Link from "next/link";
import type { LucideIcon } from "lucide-react";

export interface Task {
  id: string;
  href: string;
  label: string;
  hint: string;
  Icon: LucideIcon;
  /** A second, separate link on the card. */
  more?: { href: string; label: string };
}

/** A grid of big "do this" cards: the whole card is one link, with an optional second link above it. */
export function TaskCards({ tasks, label }: { tasks: Task[]; label: string }) {
  return (
    <ul className="grid grid-cols-2 gap-3 lg:grid-cols-4" aria-label={label}>
      {tasks.map(({ id, href, label: title, hint, Icon, more }) => (
        <li
          key={id}
          className="group relative flex min-h-[112px] flex-col gap-2 rounded-xl border border-border bg-bg-surface p-4 transition-[border-color,background-color] duration-200 focus-within:border-[var(--area-accent)] hover:border-[var(--area-accent)] hover:bg-bg-elevated motion-reduce:transition-none"
        >
          <span className="flex size-9 items-center justify-center rounded-lg bg-[color-mix(in_srgb,var(--area-accent)_16%,transparent)] text-[var(--area-accent)]">
            <Icon className="size-5" aria-hidden />
          </span>
          <span className="min-w-0">
            <Link
              href={href}
              className="block text-sm font-semibold leading-snug text-text-primary after:absolute after:inset-0 after:rounded-xl focus-visible:outline-none focus-visible:after:ring-2 focus-visible:after:ring-[var(--area-accent)]"
            >
              {title}
            </Link>
            <span className="mt-0.5 block text-xs text-text-secondary">{hint}</span>
          </span>
          {more && (
            <Link
              href={more.href}
              className="relative z-10 -mb-1 mt-auto inline-flex min-h-8 items-center text-xs font-medium text-[var(--area-accent)] underline-offset-2 hover:underline"
            >
              {more.label}
            </Link>
          )}
        </li>
      ))}
    </ul>
  );
}
