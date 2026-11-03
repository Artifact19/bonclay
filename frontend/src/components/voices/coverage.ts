import type { EngineStatus } from "@/types/api";
import { FALLBACK_SAMPLES, languageName } from "@/components/studio/studioApi";

/** The Indian languages a saved voice can speak, in the order people usually ask for them. */
const PREFERRED = ["hi", "mr", "ta", "te", "bn", "gu", "kn", "ml", "pa", "or", "as"];

export interface IndianCoverage {
  /** A few names to show, e.g. Hindi, Marathi, Tamil. */
  names: string[];
  /** How many more Indian languages there are beyond `names`. */
  more: number;
  /** All the language codes, for the Details view. */
  codes: string[];
}

/** DhVaani (set up on this server) lets a saved voice speak Indian languages; null when it is not set up. */
export function indianCoverage(engines: EngineStatus[] | undefined, show = 3): IndianCoverage | null {
  const dh = engines?.find((e) => e.id === "dhvaani" && e.available === true);
  const codes = (dh?.languages ?? []) as string[];
  if (!dh || !codes.length) return null;
  const ordered = [...PREFERRED.filter((c) => codes.includes(c)), ...codes.filter((c) => !PREFERRED.includes(c))];
  return { names: ordered.slice(0, show).map(languageName), more: Math.max(0, ordered.length - show), codes: ordered };
}

/** "Hindi, Marathi, Tamil and 19 more" */
export function coverageText(c: IndianCoverage): string {
  return c.more > 0 ? `${c.names.join(", ")} and ${c.more} more` : c.names.join(", ");
}

/** A short line to hear a voice say: in its own language when there is a sample for it, else English. */
export function previewLine(language: string | undefined | null): string {
  const hit = FALLBACK_SAMPLES.find((s) => s.language === (language ?? "").toLowerCase());
  return hit ? hit.text : "Hello, this is how my voice sounds. Your appointment is confirmed for tomorrow morning.";
}
