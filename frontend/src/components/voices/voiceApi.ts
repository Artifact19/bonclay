import { api, ApiError } from "@/lib/api";
import type { Voice } from "@/types/api";

export const VOICES_KEY = "/api/voices";

/** Voice as shown in the UI: backend rows use display_name, the shared type uses name. */
export interface VoiceView {
  id: string;
  name: string;
  language: string;
  engine: string;
  createdAt: string | null;
  durationSec: number | null;
  /** A consent recording exists for this voice. */
  consent: boolean;
  /** true: checked against the consent sentence; false: recorded but not checked (no speech recognition was
   *  available); undefined: the server did not say (cloud rows do not store it). */
  consentVerified?: boolean;
  raw: Voice;
}

const DURATION_KEY = "bonclay_voice_durations";

function readDurations(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem(DURATION_KEY) || "{}");
  } catch {
    return {};
  }
}

export function rememberDuration(id: string, seconds: number) {
  try {
    const d = readDurations();
    d[id] = seconds;
    localStorage.setItem(DURATION_KEY, JSON.stringify(d));
  } catch {
    /* storage unavailable */
  }
}

export function toView(v: Voice): VoiceView {
  const r = v as Record<string, unknown>;
  const num = (x: unknown) => (typeof x === "number" && isFinite(x) ? x : null);
  const stored = typeof window !== "undefined" ? readDurations()[v.id] : undefined;
  return {
    id: v.id,
    name: String(r.display_name ?? v.name ?? "Untitled voice"),
    language: String(v.language ?? "en"),
    engine: String(v.engine ?? ""),
    createdAt: v.created_at ? String(v.created_at) : null,
    durationSec: num(r.sample_duration) ?? num(r.duration) ?? num(r.duration_sec) ?? stored ?? null,
    consent: Boolean(r.consent_id),
    consentVerified: typeof r.consent_verified === "boolean" ? r.consent_verified : undefined,
    raw: v,
  };
}

export function storageMode(): "local" | "cloud" {
  try {
    return localStorage.getItem("bonclay_storage_mode") === "cloud" ? "cloud" : "local";
  } catch {
    return "local";
  }
}

/** "Withdraw consent and delete": removes the voice, its sample, embedding and consent recording. Receipts of clips
 *  made with it are kept and marked "voice withdrawn". */
export async function deleteVoice(id: string): Promise<void> {
  await api.del(`${VOICES_KEY}/${encodeURIComponent(id)}`);
}

/** Itemised consent record of one voice (GET /api/voices/{id}/consent-record, owner only). */
export interface ConsentRecord {
  record_version: number;
  /** null for voices cloned before notices were versioned. */
  notice_version: string | null;
  voice_id: string;
  voice_name: string | null;
  consent_id: string | null;
  purpose: string;
  data_stored: string[];
  challenge_sentence: string | null;
  challenge_code: string | null;
  challenge_id: string | null;
  transcript_match_score: number | null;
  transcript_verified: boolean | null;
  speaker_match_score: number | null;
  speaker_match_threshold: number | null;
  speaker_checked: boolean;
  language: string | null;
  engine: string | null;
  consented_at: string | null;
  retention_days: number;
  /** ISO date-time after which the voice is due for deletion. */
  retain_until: string | null;
  how_to_withdraw: string;
  contact: string;
  /** true when the voice was cloned before records existed: some fields were never stored. */
  legacy: boolean;
}

export function consentRecordKey(voiceId: string): string {
  return `${VOICES_KEY}/${encodeURIComponent(voiceId)}/consent-record`;
}

export async function fetchConsentRecord(voiceId: string, signal?: AbortSignal): Promise<ConsentRecord> {
  return api.get<ConsentRecord>(consentRecordKey(voiceId), { signal });
}

/** Save the record as a .json file in the browser (nothing is sent anywhere). */
export function downloadConsentRecord(record: ConsentRecord): void {
  const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const safe = (record.voice_name || "voice").replace(/[^\w.-]+/g, "_").slice(0, 40) || "voice";
  a.href = url;
  a.download = `consent-record-${safe}-${record.voice_id.slice(0, 8)}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function transcribe(file: Blob, filename: string, onProgress?: (p: number) => void) {
  const fd = new FormData();
  fd.append("audio_file", file, filename);
  return api.upload<{ text: string; language?: string }>("/api/transcribe", fd, { onProgress });
}

export interface CloneInput {
  file: Blob;
  filename: string;
  displayName: string;
  language: string;
  transcript?: string;
  /** Spoken consent recording of `consentText`; the backend transcribes and verifies it. */
  consentAudio: Blob;
  consentText: string;
  /** Id of the sentence from GET /api/consent/challenge; the server then verifies against its stored text. */
  challengeId?: string;
}

export interface ConsentChallenge {
  challenge_id: string;
  text: string;
  /** ISO timestamp after which the sentence can no longer be used. */
  expires_at: string;
  /** Version of the notice the server stores in the consent record (absent on an older backend). */
  notice_version?: string;
  /** How long the voice is kept. */
  retention_days?: number;
  contact?: string;
}

/** GET /api/consent/challenge: a fresh sentence (with the person's name, today's date and a code) to read aloud. */
export async function fetchConsentChallenge(name: string, signal?: AbortSignal): Promise<ConsentChallenge> {
  return api.get<ConsentChallenge>(`/api/consent/challenge?name=${encodeURIComponent(name)}`, { signal });
}

export interface CloneConsentResult {
  verified: boolean;
  score: number | null;
  heard: string | null;
  /** Voice match between the sample and the consent recording; null when the check could not run. */
  speaker_score?: number | null;
  speaker_checked?: boolean;
}

/** POST /api/voices/clone as multipart with upload progress. */
export async function cloneVoice(
  input: CloneInput,
  onProgress?: (pct: number) => void,
  signal?: AbortSignal
): Promise<{ id: string; engine: string; message?: string; consent?: CloneConsentResult }> {
  const fd = new FormData();
  fd.append("reference_audio", input.file, input.filename);
  fd.append("display_name", input.displayName);
  fd.append("language", input.language);
  fd.append("consent_confirmed", "true");
  fd.append("consent_audio", input.consentAudio, "consent.wav");
  fd.append("consent_text", input.consentText);
  if (input.challengeId) fd.append("challenge_id", input.challengeId);
  if (input.transcript) fd.append("transcript", input.transcript);
  return api.upload("/api/voices/clone", fd, { onProgress, signal });
}

/** POST /api/voices/generate, returns the synthesized audio. */
export async function generateSpeech(voice: VoiceView, text: string, signal?: AbortSignal): Promise<Blob> {
  const fd = new FormData();
  fd.append("text", text);
  fd.append("voice_id", voice.id);
  fd.append("language", voice.language);
  fd.append("mode", "premium");
  try {
    return await api.blob("/api/voices/generate", fd, { signal });
  } catch (e) {
    if (e instanceof ApiError && e.status === 429) {
      throw new ApiError("Too many requests. Wait a minute and try again.", 429, e.code);
    }
    if (e instanceof ApiError && e.code === "aborted") throw new ApiError("Generation cancelled.", 0, "aborted");
    throw e;
  }
}
