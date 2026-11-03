"use client";

import Link from "next/link";
import { Mic } from "lucide-react";
import { Button, EmptyState, ErrorState, Skeleton } from "@/components/ui";
import { useVoices } from "@/hooks/useApi";
import { languageName } from "./format";

/** "Your voices": a compact row of the voices you have made, newest first, with a way in to clone another. */
export function VoiceStrip({ limit = 8 }: { limit?: number }) {
  const { data, error, isLoading, mutate } = useVoices();
  const voices = [...(data ?? [])].sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? "")).slice(0, limit);

  return (
    <section aria-labelledby="voices-heading">
      <div className="mb-2 flex items-center justify-between gap-3">
        <h2 id="voices-heading" className="text-sm font-semibold text-text-primary">
          Your voices
        </h2>
        {voices.length > 0 && (
          <Link href="/dashboard/voices" className="inline-flex min-h-8 items-center text-xs font-medium text-[var(--area-accent)] underline-offset-2 hover:underline">
            See all voices
          </Link>
        )}
      </div>
      {error && !data ? (
        <ErrorState title="Could not load your voices" onRetry={() => mutate()} />
      ) : isLoading ? (
        <div className="flex gap-3 overflow-hidden">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-44 shrink-0 rounded-xl" />
          ))}
        </div>
      ) : voices.length === 0 ? (
        <EmptyState
          icon={<Mic />}
          title="No voices yet"
          description="Record about a minute of your own voice and BonClay can speak in it. You stay in control of who may use it."
          action={
            <Link href="/dashboard/voices?new=1">
              <Button size="sm">Clone my voice</Button>
            </Link>
          }
        />
      ) : (
        <ul className="flex gap-3 overflow-x-auto pb-1" aria-label="Your voices">
          {voices.map((v) => (
            <li key={v.id} className="shrink-0">
              <Link
                href="/dashboard/voices"
                className="flex h-16 w-48 items-center gap-3 rounded-xl border border-border bg-bg-surface px-3 transition-colors hover:border-[var(--area-accent)] hover:bg-bg-elevated"
              >
                <span aria-hidden className="grid size-9 shrink-0 place-items-center rounded-full bg-[color-mix(in_srgb,var(--area-accent)_16%,transparent)] text-sm font-semibold text-[var(--area-accent)]">
                  {(v.name || "?").charAt(0).toUpperCase()}
                </span>
                <span className="min-w-0">
                  <span className="block truncate text-sm font-medium text-text-primary">{v.name}</span>
                  <span className="block truncate text-xs text-text-secondary">{languageName(v.language)}</span>
                </span>
              </Link>
            </li>
          ))}
          <li className="shrink-0">
            <Link
              href="/dashboard/voices?new=1"
              className="flex h-16 w-44 items-center justify-center gap-2 rounded-xl border border-dashed border-border text-sm font-medium text-text-secondary transition-colors hover:border-[var(--area-accent)] hover:text-text-primary"
            >
              <Mic className="size-4" aria-hidden />
              Clone another voice
            </Link>
          </li>
        </ul>
      )}
    </section>
  );
}
