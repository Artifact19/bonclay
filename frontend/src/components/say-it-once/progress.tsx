"use client";

import { cn } from "@/lib/utils";

/** Each language's name in its own script, shown next to the English name on its card. */
export const NATIVE_NAMES: Record<string, string> = {
  hi: "हिन्दी", mr: "मराठी", ta: "தமிழ்", bn: "বাংলা", te: "తెలుగు", gu: "ગુજરાતી", kn: "ಕನ್ನಡ", ml: "മലയാളം",
  pa: "ਪੰਜਾਬੀ", or: "ଓଡ଼ିଆ", en: "English", fr: "Français", es: "Español", de: "Deutsch", it: "Italiano",
  pt: "Português", ja: "日本語", zh: "中文", ko: "한국어", ru: "Русский",
};

/** "about 40 s left", "about 2 min left": rounded so it does not flicker, never "0 s left" while work remains. */
export function formatEta(seconds: number | null | undefined): string | null {
  if (seconds == null) return null;
  if (seconds <= 5) return "a few seconds left";
  if (seconds < 90) return `about ${Math.round(seconds / 5) * 5} s left`;
  const min = Math.round(seconds / 30) / 2; // half minutes
  return `about ${Number.isInteger(min) ? min : min.toFixed(1)} min left`;
}

/** The same figure without the "left": "about 40 s", for estimates made before anything starts. */
export function formatDuration(seconds: number): string {
  if (seconds < 8) return `about ${Math.max(1, Math.round(seconds))} s`;
  if (seconds < 90) return `about ${Math.round(seconds / 5) * 5} s`;
  const min = Math.round(seconds / 30) / 2;
  return `about ${Number.isInteger(min) ? min : min.toFixed(1)} min`;
}

export function ordinal(n: number): string {
  const v = n % 100;
  if (v >= 11 && v <= 13) return `${n}th`;
  return `${n}${["th", "st", "nd", "rd"][n % 10 > 3 ? 0 : n % 10]}`;
}

/** A labelled percentage bar. The width eases between the server's once-a-second values so it moves smoothly. */
export function ProgressBar({
  value,
  label,
  tone = "accent",
  className,
  testId,
}: {
  value: number;
  label: string;
  tone?: "accent" | "error" | "muted";
  className?: string;
  testId?: string;
}) {
  const pct = Math.max(0, Math.min(100, Math.round(value)));
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      data-testid={testId}
      className={cn("h-2 w-full overflow-hidden rounded-full bg-bg-elevated", className)}
    >
      <div
        className={cn(
          "h-full rounded-full transition-[width] duration-1000 ease-linear motion-reduce:transition-none",
          tone === "error" ? "bg-accent-error" : tone === "muted" ? "bg-text-secondary/50" : "bg-brand"
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
