"use client";

import useSWR from "swr";
import { api, ApiError } from "@/lib/api";
import type { KnowledgeGaps } from "@/types/api";

export type Range = "7d" | "30d";
export const RANGE_DAYS: Record<Range, number> = { "7d": 7, "30d": 30 };

/** GET /api/analytics/results. A figure the backend cannot derive is null, and the page says "Not tracked yet". */
export interface Results {
  range: Range;
  days: number;
  agent_id: string | null;
  agent_name: string | null;
  conversations: number;
  questions_asked: number;
  questions_answered: number;
  questions_unanswered: number;
  reply_time: {
    count: number;
    p50_ms: number | null;
    p95_ms: number | null;
    target_ms: number;
    within_target_pct: number | null;
  };
  daily: { date: string; asked: number; unanswered: number }[];
  by_language: { language: string; count: number }[];
  /** Null for one agent: meeting requests are not split by agent. */
  bookings: { requested: number; booked: number; waiting: number } | null;
}

const opts = { revalidateOnFocus: false, shouldRetryOnError: false } as const;

/** The headline numbers for one agent (or all of them) over 7 or 30 days. */
export const useResults = (range: Range, agentId?: string | null) =>
  useSWR<Results, ApiError>(
    `/api/analytics/results?range=${range}${agentId ? `&agent_id=${encodeURIComponent(agentId)}` : ""}`,
    (p: string) => api.get<Results>(p),
    opts
  );

/** Questions the agent could not answer. Without an agent it is the same request as useKnowledgeGaps. */
export const useGaps = (days: number, agentId?: string | null) =>
  useSWR<KnowledgeGaps, ApiError>(
    `/api/analytics/knowledge-gaps?days=${days}${agentId ? `&agent_id=${encodeURIComponent(agentId)}` : ""}`,
    (p: string) => api.get<KnowledgeGaps>(p),
    opts
  );
