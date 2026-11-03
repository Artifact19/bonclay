export type ItemStatus = "queued" | "translating" | "speaking" | "done" | "failed" | "unavailable" | "cancelled";
/** Where a language is in its own pipeline (the status line of its card). */
export type ItemStage = "waiting" | "translating" | "speaking" | "finishing" | "done" | "failed" | "unavailable" | "cancelled";
export type VoiceKind = "your_voice" | "preset_voice";
/** "own": the user's cloned voice where an engine can speak the language. "fast": preset voices only. */
export type Speed = "own" | "fast";

/** One language of a GET /api/say-it-once/{job_id} response. */
export interface SayItem {
  language: string;
  language_name: string;
  status: ItemStatus;
  stage?: ItemStage;
  /** 0..100: waiting 0, translating 0-15, generating the voice 15-95, watermark and save 95-100, done 100. Never goes backwards. */
  progress?: number;
  /** Seconds until this language is done (an honest estimate), null once it has finished. */
  eta_s?: number | null;
  /** 1 = next to start; null once it has started. */
  queue_position?: number | null;
  /** Epoch seconds when this language started, null while it waits. */
  started_at?: number | null;
  /** The voice stage has run well past its estimate (a busy laptop): the time left is no longer a promise. */
  slow?: boolean;
  /** What the server expected this language to take when the job began. */
  est_s?: number | null;
  /** The sentence in this language (the translation). */
  text: string | null;
  voice_kind: VoiceKind | null;
  engine: string | null;
  engine_name: string | null;
  /** Licence notice to show next to the clip (Indic-Speak), when the engine requires one. */
  attribution?: string | null;
  audio_url: string | null;
  receipt_id: string | null;
  /** Seconds the language took, translation included. */
  seconds: number | null;
  error?: string | null;
}

export interface SayJob {
  state: "running" | "done" | "failed" | "cancelled";
  speed?: Speed;
  /** Overall 0..100, weighted by how long each language is expected to take. */
  progress?: number;
  eta_s?: number;
  /** A language is taking longer than estimated. */
  slow?: boolean;
  done?: number;
  failed?: number;
  /** Languages that can be spoken here (unavailable ones are not counted). */
  total?: number;
  elapsed_s?: number;
  /** Cancel was requested; the language being spoken finishes first. */
  cancelling?: boolean;
  items: SayItem[];
}

interface CapabilityRoute {
  voice_kind: VoiceKind | null;
  engine: string | null;
  engine_name: string | null;
  available: boolean;
  reason: string | null;
  attribution?: string | null;
  est_seconds: number;
}

/** One language of GET /api/say-it-once/capabilities?voice_id=. */
export interface Capability extends CapabilityRoute {
  language: string;
  language_name: string;
  /** The same language when only preset voices are used (Speed: Fast). */
  fast?: CapabilityRoute;
}

/** POST /api/say-it-once/estimate */
export interface Estimate {
  speed: Speed;
  total_s: number;
  items: {
    language: string;
    language_name: string;
    voice_kind: VoiceKind | null;
    engine: string | null;
    engine_name: string | null;
    available: boolean;
    reason: string | null;
    est_s: number;
  }[];
}

/** One language of POST /api/say-it-once/translate: a translation to read and edit before it is spoken. */
export interface ReviewItem {
  language: string;
  language_name: string;
  translation: string;
  /** Things the user can fix by editing (a changed weekday, a wrong script, a refused answer). */
  warnings: string[];
  voice: "yours" | "preset" | null;
  engine: string | null;
  engine_name: string | null;
  available: boolean;
  reason: string | null;
  attribution?: string | null;
}
