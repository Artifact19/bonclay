/** Languages BonClay shows text in, as BCP-47 primary subtags. A code outside this list gets no `lang` attribute. */
const KNOWN = new Set([
  "en", "hi", "mr", "bn", "ta", "te", "gu", "kn", "ml", "pa", "or", "as", "ur", "ne", "sa", "es", "fr", "de", "pt", "ja", "zh", "ar",
]);

/** Hinglish is Hindi written in Roman letters. */
const HINGLISH = new Set(["hinglish", "hi-en", "hi-latn", "hi-in-latn"]);

/**
 * The value for an HTML `lang` attribute from an BonClay language code ("hi", "en-IN", "hinglish").
 * Screen readers pick the right voice and browsers the right font features from it. Returns undefined for a code we
 * do not recognise ("auto", "unknown", nothing): leaving `lang` off is better than a wrong one.
 */
export function langAttr(code?: string | null): string | undefined {
  if (!code) return undefined;
  const c = code.trim().toLowerCase().replace(/_/g, "-");
  if (HINGLISH.has(c)) return "hi-Latn";
  const base = c.split("-")[0];
  return KNOWN.has(base) ? base : undefined;
}
