"use client";

import { useEffect } from "react";
import Link from "next/link";
import { RotateCcw, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui";
import { PublicMain } from "@/components/PublicMain";
import { manrope } from "@/lib/brandFont";

// Rendered by the root layout, outside the app shell. It uses the brand face and the system UI font, not Geist: see
// not-found.tsx for why the root's special files must not import the app fonts. PublicMain supplies <main>.
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("BonClay route error:", error);
  }, [error]);

  return (
    <PublicMain>
      <div role="alert" className={`flex min-h-[70vh] flex-col items-center justify-center px-4 py-16 text-center ${manrope.variable}`}>
        <div className="mb-5 flex size-14 items-center justify-center rounded-2xl bg-accent-error/15 text-accent-error">
          <TriangleAlert className="size-7" aria-hidden />
        </div>
        <h1 className="text-2xl font-semibold text-text-primary">This page hit a snag</h1>
        <p className="mt-2 max-w-md text-sm text-text-secondary">
          Something failed while loading this screen. Your work elsewhere is safe. Retry, or head back to the dashboard.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <Button onClick={() => reset()} leftIcon={<RotateCcw className="size-4" />}>
            Retry
          </Button>
          <Link
            href="/dashboard"
            className="inline-flex h-10 items-center rounded-lg border border-border px-4 text-sm font-medium text-text-primary transition-colors duration-200 hover:border-accent-primary hover:bg-bg-elevated"
          >
            Go to dashboard
          </Link>
        </div>
        {process.env.NODE_ENV === "development" && (
          <pre className="mt-8 max-w-full overflow-auto rounded-xl border border-border bg-bg-surface p-4 text-left font-mono text-xs text-accent-error">
            {error.message}
          </pre>
        )}
      </div>
    </PublicMain>
  );
}
