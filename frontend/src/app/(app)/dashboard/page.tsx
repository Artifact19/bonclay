"use client";

import { useEffect, useSyncExternalStore } from "react";
import { useRouter } from "next/navigation";
import { ErrorBoundary } from "@/components/ui";
import { FirstRunChecklist } from "@/components/dashboard/FirstRunChecklist";
import { NeedsAttention } from "@/components/dashboard/NeedsAttention";
import { QuickActions } from "@/components/dashboard/QuickActions";
import { RecentActivity } from "@/components/dashboard/RecentActivity";
import { SystemStatus } from "@/components/dashboard/SystemStatus";
import { useRememberedArea } from "@/hooks/useArea";
import { AREA_HOME } from "@/lib/areas";
import DashboardLoading from "./loading";

function greeting(hour: number): string {
  if (hour < 5) return "Hello";
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

const subscribeNever = () => () => {};

/** Home of the Agents area. Someone who last chose Studio is sent to Studio home instead (see useRememberedArea). */
export default function DashboardHome() {
  const router = useRouter();
  const remembered = useRememberedArea();
  const toStudio = remembered === "studio";
  useEffect(() => {
    if (toStudio) router.replace(AREA_HOME.studio);
  }, [toStudio, router]);

  // The server renders "Hello"; the browser swaps in the right greeting for its own clock without a mismatch.
  const hello = useSyncExternalStore(subscribeNever, () => greeting(new Date().getHours()), () => "Hello");

  // Not decided yet (the browser has not looked) or on the way to Studio home: show the loading skeleton, never the Agents page.
  if (remembered === undefined || toStudio) return <DashboardLoading />;

  return (
    <ErrorBoundary>
      <div data-area="agents" className="mx-auto max-w-5xl space-y-5 p-4 sm:space-y-6 sm:p-8 md:p-12">
        <header className="border-l-2 border-[var(--area-accent)] pl-4">
          <p className="text-sm font-medium text-[var(--area-accent)]">Agents · {hello}</p>
          <h1 className="font-display text-[clamp(1.5rem,2.6vw,2.25rem)] font-medium leading-[1.1] tracking-[-0.03em] text-text-primary">
            What do you want to do?
          </h1>
        </header>
        <QuickActions />
        <FirstRunChecklist />
        <NeedsAttention />
        <RecentActivity limit={4} hideWhenEmpty />
        <SystemStatus />
      </div>
    </ErrorBoundary>
  );
}
