export interface Voice {
  id: string;
  name: string;
  description?: string | null;
  language?: string | null;
  engine?: string | null;
  audio_url?: string | null;
  sample_url?: string | null;
  transcript?: string | null;
  created_at: string;
  [key: string]: unknown;
}

export interface Generation {
  id: string;
  voice_id?: string | null;
  voice_name?: string | null;
  text: string;
  audio_url?: string | null;
  duration?: number | null;
  engine?: string | null;
  created_at: string;
  [key: string]: unknown;
}

export interface DashboardStats {
  voices: number;
  generations: number;
  minutes: number;
  recent: Generation[];
}

export interface KokoroVoice {
  id: string;
  name: string;
  language?: string;
  gender?: string;
  [key: string]: unknown;
}

/** One Indic-Speak preset voice (GET /api/indic/voices). */
export interface IndicSpeakVoice {
  id: string;
  name: string;
  language: string;
  language_code: string;
  gender: "female" | "male";
  tier: "production" | "preview";
}

export interface IndicSpeakInfo {
  available: boolean;
  reason: string | null;
  voices: IndicSpeakVoice[];
  /** Longest text the engine speaks in one request. */
  max_chars: number;
  /** Notice the Indic Open Model License requires wherever the output reaches a user interface. */
  attribution: string;
  model_url: string;
}

export interface AnalyticsSummary {
  range?: string;
  by_engine: { engine: string; count: number }[];
  by_language: { language: string; count: number }[];
  daily: { date: string; count: number }[];
  latency: { engine: string; avg_ms: number }[];
}

/** One entry of GET /api/rag/documents (`{ documents: RagDocument[] }`). */
export interface RagDocument {
  id: string;
  filename: string;
  chunks: number;
  embedder: "sentence-transformers" | "tfidf";
}

export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface Job {
  id: string;
  status: JobStatus;
  progress?: number;
  result?: unknown;
  error?: string | null;
  created_at?: string;
}

export interface HealthStatus {
  status: string;
  gpu?: { available: boolean; name: string | null; device: string; dtype: string };
  models_loaded?: string[];
  supabase?: { configured: boolean; reachable?: boolean };
  disk_free_gb?: number;
  llm?: LlmStatus;
  platform?: {
    os: "macOS" | "Windows" | "Linux";
    os_version: string;
    arch: string;
    accelerator: "apple" | "nvidia" | "cpu";
    accelerator_name: string;
    device: string;
    vram_gb: number | null;
    gpu_budget_gb: number | null;
  };
  [key: string]: unknown;
}

export type LlmProviderName = "groq" | "openai" | "mlx";

/** Language-model status inside GET /api/health and POST /api/llm/warm. */
export interface LlmStatus {
  active: LlmProviderName | null;
  model: string | null;
  local_running: boolean;
  providers: { name: string; available: boolean; reason?: string | null }[];
  usage_today: Record<string, { requests: number; tokens: number }>;
  /** Models whose daily quota is used up on every Groq key -> when the first key can use them again (UTC, ISO 8601). */
  groq_paused?: Record<string, string>;
  /** One row per configured Groq account (GROQ_API_KEY, GROQ_API_KEY_2 ..). `label` is "1".."9", never the key itself. */
  groq_keys?: {
    label: string;
    state: "active" | "paused" | "auth_failed";
    paused_models: { model: string; until: string }[];
  }[];
}

export interface EngineStatus {
  id: string;
  name: string;
  type: string;
  hf_id?: string | null;
  languages: string[];
  loaded: boolean;
  loaded_for_s?: number | null;
  idle_s?: number | null;
  /** Engines that need a separate setup step also report whether they can run here, and why not (e.g. DhVaani). */
  available?: boolean;
  unavailable_reason?: string | null;
  setup?: string | null;
}

export interface ModelStatus {
  device: string;
  dtype: string;
  engines: EngineStatus[];
  whisper_loaded?: boolean;
  embedding_model_loaded?: boolean;
  /** Which engine clones a saved voice in Hindi (INDIC_CLONE_ENGINE): "chatterbox_ml", "dhvaani" or null when none can. */
  indic_clone?: { preference: "auto" | "dhvaani" | "chatterbox"; hindi_engine: string | null };
}

/** Raw GET /api/stats response. */
export interface StatsResponse {
  voices_count: number;
  generations_count: number;
  minutes_synthesized: number;
  recent_generations: Generation[];
}

export interface ApiKey {
  id: string;
  name: string;
  prefix?: string;
  created_at: string;
  last_used_at?: string | null;
}

export type UserRole = "admin" | "user" | string;

/** GET /api/agents (`{ agents: Agent[] }`). */
export interface Agent {
  id: string;
  name: string;
  greeting: string;
  system_instructions: string;
  supported_languages: string[];
  voice_id: string | null;
  /** The agent says it is an AI at the start of each conversation (default true). */
  disclose_ai?: boolean;
  /** Names of the MCP tool servers the agent may use, e.g. "appointments". */
  tools?: string[];
  created_at: string;
  document_count: number;
}

/** One tool call of an agent turn (`tool_calls` of the turn response and of the stream's `done` event). */
export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  /** One line saying what happened, e.g. "booking A12 with Dr. Meera Kulkarni on Monday 2026-10-19 10:30". */
  result: string;
}

/** Fields PUT /api/agents/{id} accepts (all optional). */
export type AgentUpdate = Partial<
  Pick<Agent, "name" | "greeting" | "system_instructions" | "supported_languages" | "voice_id" | "disclose_ai" | "tools">
>;

export interface AgentDocument {
  id: string;
  filename: string;
  chunk_count: number;
  created_at: string;
}

export interface AgentEvaluationRun {
  id?: string;
  persona: string;
  user_prompt?: string;
  agent_reply: string;
  task_completion: boolean;
  hallucination_flag: boolean;
  policy_compliance: boolean;
  reasoning: string;
  created_at: string;
  [key: string]: unknown;
}

/** GET /api/onboarding. */
export interface OnboardingStep {
  id: string;
  title: string;
  done: boolean;
  href: string;
  hint: string;
}

export interface OnboardingStatus {
  steps: OnboardingStep[];
  completed: number;
  total: number;
}

/** GET /api/setup-hints. */
export type SetupHintSeverity = "error" | "warning" | "info";

export interface SetupHint {
  id: string;
  severity: SetupHintSeverity;
  title: string;
  detail: string;
  command: string | null;
}

export interface SetupHints {
  hints: SetupHint[];
}

/** GET /api/perf/turns: reply time of the most recent spoken agent turns. */
export interface PerfStageTimings {
  asr_ms: number;
  retrieval_ms: number;
  llm_ms: number;
  tts_ms: number;
}

export interface PerfTurn {
  id: string;
  created_at: string;
  agent_id: string | null;
  agent_name: string | null;
  total_ms: number;
  timings: PerfStageTimings;
  /** Streamed turns only: time until the caller could hear the first sentence. */
  first_audio_ms?: number | null;
  llm_first_token_ms?: number | null;
  engine: string | null;
  llm_provider: string | null;
  language: string | null;
  fallback: boolean;
}

export interface PerfTurns {
  turns: PerfTurn[];
  count: number;
  p50_ms: number | null;
  p95_ms: number | null;
  stage_p50: { [K in keyof PerfStageTimings]: number | null };
  target_ms: number;
  within_target_pct: number | null;
  /** Over the streamed turns among them; null/absent when none were streamed. */
  first_audio_count?: number;
  first_audio_p50_ms?: number | null;
  first_audio_p95_ms?: number | null;
  first_audio_within_target_pct?: number | null;
}

/** GET /api/agent-templates item. */
export interface AgentTemplate {
  id: string;
  name: string;
  role: string;
  languages: string[];
  greeting: string;
  document: string;
  suggested_questions: string[];
  /** Path of a sample PDF to download (templates that come with one). */
  sample_pdf?: string | null;
}

/** GET /api/demo/status */
export interface DemoStatus {
  loaded: boolean;
  agents: string[];
  evaluation_ready: boolean;
  evaluation_running?: boolean;
}

/** POST /api/demo/load */
export interface DemoLoadResult {
  loaded: boolean;
  agents: string[];
  evaluation: "running" | "ready" | "not_started";
}

/** GET /api/warmup component state. */
export type WarmupComponentState = "pending" | "warming" | "ready" | "skipped" | "failed";

export interface WarmupComponent {
  id: string;
  label: string;
  state: WarmupComponentState;
  seconds: number | null;
  detail: string | null;
}

/** GET /api/warmup, POST /api/warmup */
export interface WarmupStatus {
  state: "idle" | "running" | "ready" | "partial";
  started_at: number | null;
  finished_at: number | null;
  components: WarmupComponent[];
  not_warmed?: string;
}

/** One group of similar questions an agent could not answer, from GET /api/analytics/knowledge-gaps. */
export interface KnowledgeGapCluster {
  question: string;
  examples: string[];
  count: number;
  last_asked: string;
  agent_id: string;
  agent_name: string | null;
  language: string | null;
  /** Which signal flagged the group (dominant one): the reply, retrieval found nothing, or only a weak match. */
  reason?: KnowledgeGapReason;
  reasons?: Partial<Record<KnowledgeGapReason, number>>;
}

export type KnowledgeGapReason = "said it didn't know" | "no matching document" | "weak match";

export interface KnowledgeGaps {
  clusters: KnowledgeGapCluster[];
  total_gap_turns: number;
  total_turns: number;
  window_days: number;
}

/** One number the owner may call: GET /api/phone/exotel/status (`number` is E.164, `masked` is for display). */
export interface ExotelAllowedNumber {
  number: string;
  masked: string;
  label: string;
  /** "env": from PHONE_ALLOWED_NUMBERS in backend/.env (cannot be removed in the app); "saved": added in the app. */
  source: "env" | "saved";
  /** Set for source "saved" only. */
  id: string | null;
}

/** POST /api/phone/exotel/numbers */
export interface SavedPhoneNumber {
  id: string;
  label: string;
  number: string;
  masked: string;
  source: "saved";
}

/** One configured Exotel account in GET /api/phone/exotel/status (never the key, token or SID). */
export interface ExotelAccountStatus {
  /** "1" for the plain EXOTEL_* settings, "2".."9" for the ones ending _2 .. _9. */
  label: string;
  caller_id_masked: string | null;
  /** exhausted = out of credit, auth_failed = credentials refused; both are tried again after a while. */
  state: "active" | "exhausted" | "auth_failed";
  since: string | null;
}

/** The phone providers BonClay can call through, chosen by PHONE_PROVIDERS in backend/.env. */
export type PhoneProviderName = "exotel" | "twilio" | "plivo" | "vobiz" | "telnyx";

/** One provider in GET /api/phone/exotel/status: its accounts (never a key, token or SID) and whether it is in use. */
export interface PhoneProviderStatus {
  name: PhoneProviderName;
  /** How it reads to a person: "Exotel", "Twilio", ... */
  display: string;
  /** At least one complete account is set in backend/.env. */
  configured: boolean;
  /** Named in PHONE_PROVIDERS (in use for outbound calls). */
  enabled: boolean;
  /** false: written from the provider's docs, not yet run against a real account. */
  verified: boolean;
  accounts: ExotelAccountStatus[];
}

/** GET /api/phone/exotel/status (owner only). */
export interface ExotelStatus {
  configured: boolean;
  /** Names of the missing backend/.env settings, e.g. "EXOTEL_API_KEY". */
  missing: string[];
  allowed_numbers: ExotelAllowedNumber[];
  allowed_numbers_set: boolean;
  caller_id_masked: string | null;
  /** PUBLIC_BASE_URL is set, so the inbound link can be built. */
  inbound_ready: boolean;
  public_base_url: string | null;
  max_call_seconds: number;
  /** Every complete Exotel account, in order. */
  accounts?: ExotelAccountStatus[];
  /** Every provider with its accounts, and the order calls try them (PHONE_PROVIDERS). */
  providers?: PhoneProviderStatus[];
  provider_order?: PhoneProviderName[];
}

export type ExotelCallState =
  | "queued"
  | "ringing"
  | "in-progress"
  | "completed"
  | "failed"
  | "busy"
  | "no-answer"
  | "canceled"
  | "unknown";

/** POST /api/phone/exotel/call */
export interface ExotelCallPlaced {
  status: "calling";
  to: string;
  call_sid: string | null;
  agent_id: string;
  /** The provider and account that took the call after any failover, e.g. "twilio" and "2". */
  provider?: PhoneProviderName;
  account?: string;
}

/** GET /api/phone/exotel/call/{call_sid} */
export interface ExotelCallStatus {
  call_sid: string;
  status: ExotelCallState;
  duration_s: number | null;
  started_at: string | null;
  ended_at: string | null;
  /** Exotel's explanation when the call failed, was busy, unanswered or canceled. */
  message?: string | null;
}

/** POST /api/phone/exotel/inbound-link */
export interface ExotelInboundLink {
  stream_url: string;
  valid_days: number;
}

/** GET /api/meetings: requests for a meeting with the HOD, put in by the office agent and decided on the Approvals page. */
export type MeetingStatus = "pending" | "approved" | "declined" | "cancelled";
export interface MeetingRequest {
  request_id: string;
  name: string;
  role: "student" | "parent" | "faculty" | "other";
  phone: string;
  class_or_ref: string;
  reason: string;
  date: string;
  weekday: string;
  time: string;
  minutes: number;
  status: MeetingStatus;
  decision_note?: string | null;
  suggested?: { date: string; weekday: string; time: string } | null;
  created_at?: string;
  decided_at?: string | null;
  on_calendar: boolean;
}
export interface MeetingsResponse {
  requests: MeetingRequest[];
  counts: Record<MeetingStatus, number>;
}
/** GET /api/meetings/settings */
export interface MeetingSettings {
  settings: {
    owner: string;
    title: string;
    location: string;
    slot_minutes: number;
    availability_check?: boolean;
    sample_schedule?: boolean;
  };
  calendar: { configured: boolean; calendar_id_set: boolean; key_file_found: boolean; service_account_email: string | null; timezone: string };
}

/** GET /api/meetings/week: what one cell of the weekly schedule is. */
export type MeetingCellState = "closed" | "past" | "busy" | "asked_for" | "booked" | "free";
export type MeetingBusySource = "blocked" | "calendar" | "owner" | "sample";
export interface MeetingCell {
  state: MeetingCellState;
  /** Why it is busy ("Lecture", "Busy on Google Calendar"), or why it is closed. */
  reason?: string;
  /** Where a busy cell comes from. "sample" cells are made up. */
  source?: MeetingBusySource;
  /** The request in an asked_for (pending) or booked (approved) cell. */
  request?: { request_id: string; name: string; role: MeetingRequest["role"]; on_calendar: boolean };
  /** Other requests for the same cell. */
  more?: number;
  /** A free cell that is too close in time for the agent to offer. */
  too_soon?: boolean;
}
export interface MeetingWeekDay {
  date: string;
  weekday: string;
  open: boolean;
  full: boolean;
  /** Google Calendar could not be read for this day, so busy times may be missing. */
  calendar_error?: boolean;
}
export interface MeetingWeek {
  /** The Monday of the week. */
  start: string;
  today: string;
  max_days_ahead: number;
  slot_minutes: number;
  owner?: string | null;
  label?: string | null;
  days: MeetingWeekDay[];
  /** Every slot start in the week's office windows, "HH:MM". */
  times: string[];
  cells: Record<string, Record<string, MeetingCell>>;
  settings: { availability_check: boolean; sample_schedule: boolean };
  calendar: MeetingSettings["calendar"];
}

// ---- GET /api/system/resources: the "Models and memory" indicator ----------------------------------------------------

export type MemoryPressure = "normal" | "warn" | "critical";
export type ResourceModelKind =
  | "voice"
  | "clone"
  | "speech_recognition"
  | "search"
  | "reranker"
  | "watermark"
  | "speaker_check"
  | "language_model"
  | "audit";
export type ResourceModelState = "loaded" | "loading" | "pinned (live call)" | "busy" | "idle";

export interface ResourceModel {
  key: string;
  label: string;
  kind: ResourceModelKind;
  where: "worker process" | "in app";
  device: "mlx" | "mps" | "cuda" | "cpu";
  state: ResourceModelState;
  memory_mb: number;
  /** "measured": the worker process and its children (psutil). "estimated": a model inside the API process. */
  memory_source: "measured" | "estimated";
  idle_s: number | null;
  pid: number | null;
  /** The server will accept an unload request for this model now (not busy, not in a live call, has a safe unload). */
  can_unload?: boolean;
}

export interface SystemResources {
  time: string;
  system: {
    total_mb: number;
    used_mb: number;
    available_mb: number;
    compressed_mb: number | null;
    swap_used_mb: number;
    swap_total_mb: number;
    pressure: MemoryPressure | null;
    platform: "darwin" | "linux" | "windows";
    device: "mps" | "cuda" | "cpu";
    gpu: { name: string; total_mb: number; used_mb: number | null } | null;
  };
  app: { api_rss_mb: number; workers_rss_mb: number; total_rss_mb: number };
  models: ResourceModel[];
  guard: { enabled: boolean; reserve_mb: number; estimates_mb: Record<string, number> };
}
