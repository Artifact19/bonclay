"use client";

import { Clock } from "lucide-react";
import { StudioIcon, VoicesIcon } from "@/components/icons/NavIcons";
import { AnimatedNumber, Card, ErrorState, Skeleton } from "@/components/ui";
import { useStats } from "@/hooks/useApi";

export function StatCards() {
  const { data, error, isLoading, mutate } = useStats();

  if (error && !data) {
    return (
      <ErrorState
        title="Could not load your usage"
        message={error.message || "The server did not respond."}
        onRetry={() => mutate()}
      />
    );
  }

  const items = [
    { label: "Voices", value: data?.voices ?? 0, decimals: 0, Icon: VoicesIcon },
    { label: "Audio clips made", value: data?.generations ?? 0, decimals: 0, Icon: StudioIcon },
    { label: "Minutes of audio", value: data?.minutes ?? 0, decimals: 1, Icon: Clock },
  ];

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      {items.map(({ label, value, decimals, Icon }) => (
        <Card key={label} variant="solid" className="flex items-center gap-4">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-accent-primary/15 text-accent-primary">
            <Icon className="size-5" aria-hidden />
          </div>
          <div className="min-w-0">
            <p className="text-xs text-text-secondary">{label}</p>
            {isLoading ? (
              <Skeleton className="mt-1.5 h-8 w-20" />
            ) : (
              <p className="font-display text-3xl font-semibold leading-tight text-text-primary">
                <AnimatedNumber value={value} decimals={decimals} />
              </p>
            )}
          </div>
        </Card>
      ))}
    </div>
  );
}
