"use client";

import { useMemo } from "react";
import useSWR, { SWRConfiguration } from "swr";
import { api, ApiError } from "@/lib/api";
import type {
  Voice,
  DashboardStats,
  HealthStatus,
  ModelStatus,
  KokoroVoice,
  IndicSpeakInfo,
  AnalyticsSummary,
  MeetingsResponse,
  MeetingSettings,
  MeetingWeek,
  KnowledgeGaps,
  StatsResponse,
  RagDocument,
  Job,
  ApiKey,
  Agent,
  AgentDocument,
  AgentEvaluationRun,
  AgentTemplate,
  DemoStatus,
  OnboardingStatus,
  SetupHints,
  PerfTurns,
  WarmupStatus,
} from "@/types/api";

function useGet<T>(path: string | null, config?: SWRConfiguration<T, ApiError>) {
  return useSWR<T, ApiError>(path, (p: string) => api.get<T>(p), {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
    ...config,
  });
}

/** GET a list endpoint whose body is either a bare array or `{ [key]: T[] }`. */
function useList<T>(path: string, key: string, config?: SWRConfiguration<T[], ApiError>) {
  return useSWR<T[], ApiError>(
    path,
    async (p: string) => {
      const body = await api.get<T[] | Record<string, T[]>>(p);
      if (Array.isArray(body)) return body;
      return (body?.[key] as T[] | undefined) ?? [];
    },
    { revalidateOnFocus: false, shouldRetryOnError: false, ...config }
  );
}

/** Backend voice rows carry `display_name`; the shared Voice type (and every picker) reads `name`. */
const withVoiceName = (v: Voice): Voice => {
  const displayName = (v as Record<string, unknown>).display_name;
  return v.name ? v : { ...v, name: typeof displayName === "string" && displayName ? displayName : "Untitled voice" };
};

export const useVoices = () => {
  const res = useList<Voice>("/api/voices", "voices");
  const raw = res.data;
  const data = useMemo(() => (raw?.some((v) => !v.name) ? raw.map(withVoiceName) : raw), [raw]);
  return { ...res, data };
};
export const useStats = () =>
  useSWR<DashboardStats, ApiError>(
    "/api/stats",
    async (p: string) => {
      const r = await api.get<StatsResponse>(p);
      return {
        voices: r?.voices_count ?? 0,
        generations: r?.generations_count ?? 0,
        minutes: r?.minutes_synthesized ?? 0,
        recent: r?.recent_generations ?? [],
      };
    },
    { revalidateOnFocus: false, shouldRetryOnError: false }
  );
export const useHealth = () => useGet<HealthStatus>("/api/health", { refreshInterval: 30000 });
export const useOnboarding = () => useGet<OnboardingStatus>("/api/onboarding");
export const useSetupHints = () => useGet<SetupHints>("/api/setup-hints", { refreshInterval: 60000 });
export const useModelStatus = () => useGet<ModelStatus>("/api/models/status");
/** Kokoro presets the server can actually speak (a voice whose language packages are missing is left out). */
export const useKokoroVoices = () =>
  useSWR<KokoroVoice[], ApiError>(
    "/api/kokoro/voices",
    async (p: string) => {
      const body = await api.get<{ voices?: KokoroVoice[] }>(p);
      return (body?.voices ?? []).filter((v) => v.available !== false);
    },
    { revalidateOnFocus: false, shouldRetryOnError: false }
  );
export const useIndicSpeak = () => useGet<IndicSpeakInfo>("/api/indic/voices");
export const useAnalytics = (range = "7d") =>
  useGet<AnalyticsSummary>(`/api/analytics/summary?range=${encodeURIComponent(range)}`);
/** Questions the agents could not answer well (grouped), over the last `days` days. */
export const useKnowledgeGaps = (days = 30) =>
  useGet<KnowledgeGaps>(`/api/analytics/knowledge-gaps?days=${days}`, { revalidateOnFocus: false });
export const useRagDocuments = () => useList<RagDocument>("/api/rag/documents", "documents");
export const useApiKeys = () => useList<ApiKey>("/api/keys", "keys");
/** Meeting requests for the HOD office agent, refreshed while the Approvals page is open. */
export const useMeetings = (status: string) =>
  useGet<MeetingsResponse>(`/api/meetings?status=${encodeURIComponent(status)}`, { refreshInterval: 10000 });
/** One week of the HOD's schedule (Monday of `start`'s week; this week when left out), refreshed while the page is open. */
export const useMeetingWeek = (start?: string) =>
  useGet<MeetingWeek>(`/api/meetings/week${start ? `?start=${encodeURIComponent(start)}` : ""}`, { refreshInterval: 15000, keepPreviousData: true });
export const useMeetingSettings = () => useGet<MeetingSettings>("/api/meetings/settings", { revalidateOnFocus: false });

// The backend job runner (core/jobs.py) finishes with "done" or "failed"; the others are kept for older servers.
const TERMINAL = new Set(["done", "completed", "failed", "cancelled"]);
export const useJob = (id: string | null | undefined) =>
  useGet<Job>(id ? `/api/jobs/${id}` : null, {
    refreshInterval: (job) => (job && TERMINAL.has(job.status) ? 0 : 1500),
  });

export const AGENTS_KEY = "/api/agents";
export const useAgents = () => useList<Agent>(AGENTS_KEY, "agents");
export const useAgentDocuments = (id: string | null | undefined) =>
  useSWR<AgentDocument[], ApiError>(
    id ? `/api/agents/${id}/documents` : null,
    async (p: string) => (await api.get<{ documents?: AgentDocument[] }>(p))?.documents ?? [],
    { revalidateOnFocus: false, shouldRetryOnError: false }
  );
/** Questions the user can ask this agent, written from its documents (re-read when the document count changes). */
export const useExampleQuestions = (id: string | null | undefined, documentCount = 0) =>
  useSWR<string[], ApiError>(
    id && documentCount > 0 ? [`/api/agents/${id}/example-questions`, documentCount] : null,
    async ([p]: [string, number]) => (await api.get<{ questions?: string[] }>(p))?.questions ?? [],
    { revalidateOnFocus: false, shouldRetryOnError: false }
  );
export const useAgentEvaluations = (id: string | null | undefined) =>
  useSWR<AgentEvaluationRun[], ApiError>(
    id ? `/api/agents/${id}/evaluations` : null,
    async (p: string) => (await api.get<{ runs?: AgentEvaluationRun[] }>(p))?.runs ?? [],
    { revalidateOnFocus: false, shouldRetryOnError: false }
  );

/** Reply time of the latest spoken turns (optionally one agent's), refreshed every 10 s. */
export const usePerfTurns = (limit = 20, agentId?: string | null) =>
  useGet<PerfTurns>(`/api/perf/turns?limit=${limit}${agentId ? `&agent_id=${encodeURIComponent(agentId)}` : ""}`, {
    refreshInterval: 10000,
  });

export const useAgentTemplates = () =>
  useList<AgentTemplate>("/api/agent-templates", "templates", { revalidateOnFocus: false, revalidateIfStale: false });

export const DEMO_STATUS_KEY = "/api/demo/status";
/** Demo mode status; re-checks every 5 s while the demo evaluation is still running. */
export const useDemoStatus = (enabled = true) =>
  useGet<DemoStatus>(enabled ? DEMO_STATUS_KEY : null, {
    refreshInterval: (d) => (d?.loaded && d.evaluation_running && !d.evaluation_ready ? 5000 : 0),
  });

export const WARMUP_KEY = "/api/warmup";
/** Model warm-up progress for the live-demo card; re-checks every 1.5 s while a warm-up is running. */
export const useWarmup = () =>
  useGet<WarmupStatus>(WARMUP_KEY, { refreshInterval: (d) => (d?.state === "running" ? 1500 : 0) });
