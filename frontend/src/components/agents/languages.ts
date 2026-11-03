/** Languages an agent can be set up for. The agent stores the ISO 639-1 code; people pick by name. */
export const LANGUAGE_OPTIONS: { code: string; name: string }[] = [
  { code: "en", name: "English" },
  { code: "hi", name: "Hindi" },
  { code: "mr", name: "Marathi" },
  { code: "ta", name: "Tamil" },
  { code: "te", name: "Telugu" },
  { code: "bn", name: "Bengali" },
  { code: "gu", name: "Gujarati" },
  { code: "kn", name: "Kannada" },
  { code: "ml", name: "Malayalam" },
  { code: "pa", name: "Punjabi" },
  { code: "ur", name: "Urdu" },
  { code: "or", name: "Odia" },
  { code: "as", name: "Assamese" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "ja", name: "Japanese" },
  { code: "zh", name: "Chinese" },
];

const NAMES: Record<string, string> = Object.fromEntries(LANGUAGE_OPTIONS.map((l) => [l.code, l.name]));

/** "hi" -> "Hindi"; a code we do not know stays as it is (uppercase) so nothing is lost. */
export const languageName = (code: string): string => NAMES[code.toLowerCase()] ?? code.toUpperCase();

/** ["hi", "en"] -> "Hindi, English". */
export const languageNames = (codes: string[]): string => codes.map(languageName).join(", ");
