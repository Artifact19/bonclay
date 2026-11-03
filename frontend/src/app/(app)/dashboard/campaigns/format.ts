import type { Campaign, CampaignStatus, ContactStatus } from "./types";
import type { BadgeTone } from "@/components/ui/Badge";

export const STATUS_LABEL: Record<CampaignStatus, string> = {
  draft: "Not started",
  running: "Calling",
  paused: "Paused",
  done: "Finished",
  stopped: "Stopped",
};
export const STATUS_TONE: Record<CampaignStatus, BadgeTone> = {
  draft: "neutral",
  running: "success",
  paused: "warning",
  done: "primary",
  stopped: "neutral",
};
export const CONTACT_LABEL: Record<ContactStatus, string> = {
  pending: "Waiting",
  calling: "On a call",
  done: "Reached",
  failed: "Not reached",
  dnc: "Do not call",
};
export const DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/** Share of the list that no longer needs a call (reached, given up on, or skipped), 0..100. */
export function progressPercent(c: Pick<Campaign, "counts" | "total">): number {
  if (!c.total) return 0;
  const finished = (c.counts.done ?? 0) + (c.counts.failed ?? 0) + (c.counts.dnc ?? 0);
  return Math.round((finished / c.total) * 100);
}

/** "Mon to Sat, 10:00 to 18:00 (Asia/Kolkata)" */
export function windowLabel(w: Campaign["window"]): string {
  const days = [...w.days].sort((a, b) => a - b);
  const consecutive = days.length > 2 && days.every((d, i) => i === 0 || d === days[i - 1] + 1);
  const dayText = consecutive ? `${DAY_LABELS[days[0]]} to ${DAY_LABELS[days[days.length - 1]]}` : days.map((d) => DAY_LABELS[d]).join(", ");
  return `${dayText}, ${w.start} to ${w.end} (${w.timezone})`;
}
