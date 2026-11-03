export const ALLOWED_EXTENSIONS = [".wav", ".mp3", ".m4a", ".flac"] as const;
export const MAX_AUDIO_BYTES = 50 * 1024 * 1024;
export const MAX_GENERATE_CHARS = 2000;

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds) || seconds <= 0) return "";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Returns an error message, or null when the file is acceptable. */
export function validateAudioFile(file: File): string | null {
  const dot = file.name.lastIndexOf(".");
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  if (!(ALLOWED_EXTENSIONS as readonly string[]).includes(ext)) {
    return `${file.name} is not supported. Use a ${ALLOWED_EXTENSIONS.join(", ")} file.`;
  }
  if (file.size > MAX_AUDIO_BYTES) {
    return `${file.name} is ${formatBytes(file.size)}. The limit is 50 MB.`;
  }
  if (file.size === 0) return `${file.name} is empty.`;
  return null;
}

export const LANGUAGES: { code: string; label: string }[] = [
  { code: "en", label: "English" },
  { code: "hi", label: "Hindi" },
  { code: "te", label: "Telugu" },
  { code: "ta", label: "Tamil" },
  { code: "bn", label: "Bengali" },
  { code: "mr", label: "Marathi" },
  { code: "kn", label: "Kannada" },
  { code: "ml", label: "Malayalam" },
  { code: "gu", label: "Gujarati" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "ja", label: "Japanese" },
  { code: "zh", label: "Chinese" },
];

export function languageLabel(code: string | undefined | null): string {
  if (!code) return "Unknown";
  return LANGUAGES.find((l) => l.code === code.toLowerCase())?.label ?? code.toUpperCase();
}

export function engineLabel(engine: string | undefined | null): string {
  const e = (engine ?? "").toLowerCase();
  if (e.includes("qwen")) return "Qwen3-TTS";
  if (e.includes("kokoro")) return "Kokoro";
  if (e.includes("f5") || e.includes("indic")) return "IndicF5";
  return engine || "Unknown";
}
