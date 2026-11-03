import type { CallChannel, CallOutcomeKey, CallSummary } from "./types";
import type { BadgeTone } from "@/components/ui/Badge";

export const CHANNEL_LABEL: Record<CallChannel, string> = {
  phone: "Phone",
  website: "Website",
  whatsapp: "WhatsApp",
  playground: "In the app",
};

export const OUTCOME_LABEL: Record<CallOutcomeKey, string> = {
  booked: "Booked",
  answered: "Answered",
  needs_approval: "Needs approval",
  couldnt_answer: "Couldn't answer",
  ended_early: "Ended early",
  handed_off: "Needs a person",
  declined: "Request declined",
};

export const OUTCOME_TONE: Record<CallOutcomeKey, BadgeTone> = {
  booked: "success",
  answered: "secondary",
  needs_approval: "warning",
  couldnt_answer: "error",
  ended_early: "neutral",
  handed_off: "warning",
  declined: "neutral",
};

/** Order of the filter chips: the ones that need the owner first. */
export const OUTCOME_ORDER: CallOutcomeKey[] = ["needs_approval", "handed_off", "booked", "answered", "couldnt_answer", "ended_early", "declined"];

/** "Today, 10:42 am", "Yesterday, 4:05 pm", "Mon 12 Oct, 10:42 am". */
export function whenLabel(iso: string | null, now: Date = new Date()): string {
  if (!iso) return "Unknown time";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Unknown time";
  const time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  const day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(now) - day(d)) / 86_400_000);
  if (diff === 0) return `Today, ${time}`;
  if (diff === 1) return `Yesterday, ${time}`;
  return `${d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" })}, ${time}`;
}

/** "45 s", "2 min 05 s". */
export function durationLabel(seconds: number | null): string | null {
  if (seconds == null || seconds <= 0) return null;
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const m = Math.floor(seconds / 60);
  return `${m} min ${String(Math.round(seconds % 60)).padStart(2, "0")} s`;
}

/** Who was on the other end, in words. Numbers arrive masked from the server and are never unmasked here. */
export function callerLabel(c: Pick<CallSummary, "channel" | "caller">): string {
  if (c.caller) return c.caller;
  if (c.channel === "phone") return "Number not recorded";
  if (c.channel === "website") return "Website visitor";
  if (c.channel === "playground") return "You";
  return "Unknown caller";
}

/** "••••3210": a phone number reduced to its last four digits, for screens that only need to tell callers apart. */
export function maskNumber(value: string | null | undefined): string {
  const digits = String(value ?? "").replace(/\D/g, "");
  return digits.length >= 5 ? `••••${digits.slice(-4)}` : digits ? "••••" : "Number not given";
}
