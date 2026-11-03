import { API_URL, api, ApiError } from "@/lib/api";

/** POST a multipart form or JSON body to a route that returns a WAV blob. */
export function generateBlob(path: string, body: object | FormData, signal?: AbortSignal): Promise<Blob> {
  return api.blob(path, body, { signal, retries: 0 });
}

/** The `done` event of POST /api/voices/generate/stream: the generation of the whole clip. */
export interface StreamedGeneration {
  id?: string;
  audio_url?: string | null;
  receipt_id?: string | null;
  duration_s?: number;
  engine?: string;
}

/** A path the API serves ("/api/...") becomes a full URL for an Audio element; http, data and blob URLs stay as they are. */
export function absoluteAudioUrl(u: string): string {
  return u.startsWith("http") || u.startsWith("blob:") || u.startsWith("data:") ? u : `${API_URL}${u}`;
}

/** Download a finished clip: through the API client for our own paths (it adds the sign-in header), a plain fetch for
 *  signed storage links and data URIs (an Authorization header would be refused there). */
export async function fetchClipBlob(url: string, signal?: AbortSignal): Promise<Blob> {
  if (url.startsWith("/")) return api.blob(url, undefined, { signal, retries: 1 });
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`Could not load the clip (${res.status}).`);
  return res.blob();
}

export const MAX_CHARS = 2000;

/** Shown wherever DhVaani audio is offered (ARTPARK-IISc asks for the credit). */
export const DHVAANI_CREDIT = "Voice cloning in Indian languages by DhVaani from ARTPARK-IISc.";
/** How Studio names the engine that speaks a saved voice in an Indian language. */
export const DHVAANI_LABEL = "Your voice (DhVaani · ARTPARK-IISc)";

/** Indic language codes the backend routes to Indic-Speak (preset voices) or IndicF5 (cloning). */
export const INDIC_CODES = new Set(["hi", "mr", "bn", "gu", "kn", "ml", "or", "pa", "ta", "te", "as"]);

export const LANGUAGE_NAMES: Record<string, string> = {
  en: "English",
  hi: "Hindi",
  mr: "Marathi",
  bn: "Bengali",
  gu: "Gujarati",
  kn: "Kannada",
  ml: "Malayalam",
  or: "Odia",
  pa: "Punjabi",
  ta: "Tamil",
  te: "Telugu",
  as: "Assamese",
};

export function languageName(code: string): string {
  return LANGUAGE_NAMES[code] ?? code.toUpperCase();
}

export interface SamplePrompt {
  language: string;
  text: string;
}

export const FALLBACK_SAMPLES: SamplePrompt[] = [
  { language: "hi", text: "नमस्ते, आपका स्वागत है। आज हम एक नई कहानी सुनाने जा रहे हैं।" },
  { language: "mr", text: "नमस्कार, तुमचे स्वागत आहे. आज आपण एक नवीन गोष्ट ऐकूया." },
  { language: "ta", text: "வணக்கம், உங்களை வரவேற்கிறோம். இன்று ஒரு புதிய கதையைக் கேட்போம்." },
  { language: "te", text: "నమస్కారం, మీకు స్వాగతం. ఈ రోజు ఒక కొత్త కథ వినిపిస్తాను." },
  { language: "bn", text: "নমস্কার, আপনাকে স্বাগত। আজ আমরা একটি নতুন গল্প শুনব।" },
];

/** GET /api/indic/samples returns { samples: [{ language, text }] }. */
export async function fetchSamples(): Promise<SamplePrompt[]> {
  const r = await api.get<{ samples?: SamplePrompt[] }>("/api/indic/samples", { retries: 0 });
  return (r?.samples ?? []).filter((x) => x && typeof x.text === "string" && x.text.trim());
}

export function errorMessage(e: unknown, fallback = "Something went wrong. Try again."): string {
  if (e instanceof ApiError) return e.message || fallback;
  if (e instanceof Error) return e.message || fallback;
  return fallback;
}
