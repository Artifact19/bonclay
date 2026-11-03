/** GET /api/calls (backend/routers/conversations.py). Everything here is read from what the agents already stored. */
export type CallChannel = "phone" | "website" | "whatsapp" | "playground";
export type CallOutcomeKey =
  | "answered"
  | "booked"
  | "needs_approval"
  | "declined"
  | "handed_off"
  | "couldnt_answer"
  | "ended_early";

export interface CallOutcome {
  key: CallOutcomeKey;
  label: string;
  /** One plain sentence about why, when the backend can say. */
  detail: string | null;
  /** The meeting request this conversation made, when it made one (see Approvals). */
  request_id: string | null;
}

/** What happened to this conversation's row in the owner's Google Sheet; absent when it was never queued for one. */
export interface CallSheet {
  state: "sent" | "waiting" | "failed";
  error?: string;
}

export interface CallSummary {
  id: string;
  agent_id: string | null;
  agent_name: string;
  channel: CallChannel;
  /** A practice conversation the owner started themselves (a Test call). */
  practice: boolean;
  when: string | null;
  /** Already masked by the server ("••••3210"); only WhatsApp keeps a caller number. */
  caller: string | null;
  /** Phone calls only, estimated from the stored turns. */
  duration_s: number | null;
  outcome: CallOutcome;
  replies: number;
  preview: string;
  sheet?: CallSheet;
}

export interface TranscriptLine {
  role: "caller" | "agent";
  text: string;
  at: string | null;
  actions: { name: string; ok: boolean; summary: string }[];
}

export interface CallsResponse {
  calls: CallSummary[];
  days: number;
}
export interface CallDetail extends CallSummary {
  transcript: TranscriptLine[];
}

/** A phone call happening right now (GET /api/phone/exotel/live). */
export interface LiveCall {
  conversation_id: string;
  call_sid: string | null;
  agent_id: string;
  agent_name: string;
  direction: "inbound" | "outbound";
  started_at: number;
  seconds: number;
  human_active: boolean;
  listeners: number;
}
