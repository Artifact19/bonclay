"use client";

import useSWR from "swr";
import { api, ApiError } from "@/lib/api";

/** One answer the post-call check flagged (GET /api/answer-audit, backend/routers/answer_audit.py). Text is already masked. */
export interface FlaggedAnswer {
  turn_id: string;
  agent_id: string | null;
  agent_name: string | null;
  conversation_id: string | null;
  channel: string | null;
  question: string;
  reply: string;
  /** The sentence of the reply the documents support least. */
  weakest_sentence: string;
  /** The part of the documents it was compared with. */
  passage: string;
  /** "stored": the passages the agent was given. "approx": the closest passages of the same documents (older replies). */
  passage_source: "stored" | "approx" | null;
  score: number | null;
  threshold: number | null;
  feedback: "fine" | "wrong" | null;
  at: string | null;
}

export interface AuditRun {
  id: string;
  state: "running" | "done" | "error";
  total: number;
  done: number;
  checked: number;
  flagged: number;
  skipped: number;
  language: number;
  error: string | null;
}

export interface AnswerAudit {
  flagged: FlaggedAnswer[];
  counts: { checked: number; flagged: number; not_checked_language: number; marked_fine: number; marked_wrong: number };
  threshold: number;
  days: number;
  enabled: boolean;
  /** The checking model is already on this machine (the first check downloads it, once). */
  model_ready: boolean;
  note: string;
  run: AuditRun | null;
}

const opts = { revalidateOnFocus: false, shouldRetryOnError: false } as const;

export const auditKey = (days: number, agentId?: string | null) =>
  `/api/answer-audit?days=${days}${agentId ? `&agent_id=${encodeURIComponent(agentId)}` : ""}`;

/** Flagged answers; polls every 2 seconds while a check is running. */
export const useAnswerAudit = (days: number, agentId?: string | null) =>
  useSWR<AnswerAudit, ApiError>(auditKey(days, agentId), (p: string) => api.get<AnswerAudit>(p), {
    ...opts,
    refreshInterval: (d) => (d?.run?.state === "running" ? 2000 : 0),
  });

/** How many flagged answers each conversation has (for the Calls page). Same request as the Results card for 30 days
 *  and all agents, so the two share one fetch. A missing or failed answer is simply no badge. */
export function useFlaggedByCall(): Record<string, number> {
  const { data } = useSWR<AnswerAudit, ApiError>(auditKey(30), (p: string) => api.get<AnswerAudit>(p), opts);
  const out: Record<string, number> = {};
  for (const f of data?.flagged ?? []) {
    if (f.conversation_id) out[f.conversation_id] = (out[f.conversation_id] ?? 0) + 1;
  }
  return out;
}
