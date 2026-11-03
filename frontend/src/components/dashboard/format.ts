export function relativeTime(iso?: string | null): string {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} hr ago`;
  const d = Math.round(h / 24);
  if (d < 30) return `${d} day${d === 1 ? "" : "s"} ago`;
  return new Date(t).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function clock(seconds?: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return "";
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function titleCase(s: string): string {
  return s.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function errMessage(e: unknown, fallback = "Something went wrong."): string {
  return e instanceof Error && e.message ? e.message : fallback;
}

const LANGUAGE_NAMES: Record<string, string> = {
  en: "English", hi: "Hindi", mr: "Marathi", ta: "Tamil", te: "Telugu", bn: "Bengali", gu: "Gujarati", kn: "Kannada",
  ml: "Malayalam", pa: "Punjabi", ur: "Urdu", or: "Odia", as: "Assamese", ne: "Nepali", sa: "Sanskrit",
  es: "Spanish", fr: "French", de: "German", pt: "Portuguese", ja: "Japanese", zh: "Chinese", ar: "Arabic",
  hinglish: "Hinglish", "hi-en": "Hinglish", auto: "Detected automatically", unknown: "Not recorded",
};

/** "hi" -> "Hindi", "en-IN" -> "English". A code we do not know is shown as it is rather than guessed. */
export function languageName(code?: string | null): string {
  const raw = (code ?? "").trim();
  if (!raw) return "Not recorded";
  const key = raw.toLowerCase();
  const known = LANGUAGE_NAMES[key] ?? LANGUAGE_NAMES[key.split(/[-_]/)[0]];
  if (known) return known;
  try {
    const shown = new Intl.DisplayNames(["en"], { type: "language" }).of(raw);
    if (shown && shown.toLowerCase() !== key) return shown;
  } catch {
    /* an unusual code: fall through */
  }
  return raw;
}

/** 850 -> "0.9 s"; 12400 -> "12 s". Empty string when there is no number. */
export function seconds(ms?: number | null): string {
  if (ms == null || !Number.isFinite(ms)) return "";
  const s = ms / 1000;
  return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
}
