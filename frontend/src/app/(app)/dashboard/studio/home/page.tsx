"use client";

import { useSyncExternalStore } from "react";
import { ErrorBoundary } from "@/components/ui";
import { RecentActivity } from "@/components/dashboard/RecentActivity";
import { StatCards } from "@/components/dashboard/StatCards";
import { StudioTasks } from "@/components/dashboard/StudioTasks";
import { VoiceStrip } from "@/components/dashboard/VoiceStrip";

function greeting(hour: number): string {
  if (hour < 5) return "Hello";
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

const subscribeNever = () => () => {};

/** Home of the Studio area: making audio. Nothing here is about agents. */
export default function StudioHome() {
  const hello = useSyncExternalStore(subscribeNever, () => greeting(new Date().getHours()), () => "Hello");

  return (
    <ErrorBoundary>
      <div data-area="studio" className="mx-auto max-w-5xl space-y-5 p-4 sm:space-y-6 sm:p-8 md:p-12">
        <header className="border-l-2 border-[var(--area-accent)] pl-4">
          <p className="text-sm font-medium text-[var(--area-accent)]">Studio · {hello}</p>
          <h1 className="font-display text-[clamp(1.5rem,2.6vw,2.25rem)] font-medium leading-[1.1] tracking-[-0.03em] text-text-primary">
            What do you want to make?
          </h1>
        </header>
        <StudioTasks />
        <StatCards />
        <RecentActivity limit={5} />
        <VoiceStrip />
      </div>
    </ErrorBoundary>
  );
}
