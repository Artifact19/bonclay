"use client";

import useSWR from "swr";
import { api } from "@/lib/api";
import type { Agent, ExotelStatus } from "@/types/api";

const quiet = { revalidateOnFocus: false, shouldRetryOnError: false } as const;

/**
 * Is a phone line set up on this computer? `undefined` while loading or when the answer is unknown.
 * It is account-wide (one Exotel line), not per agent.
 */
export function usePhoneConnected(): boolean | undefined {
  const { data } = useSWR<boolean>("/api/phone/exotel/status", async (p: string) => !!(await api.get<ExotelStatus>(p)).configured, quiet);
  return data;
}

/** Has anyone (other than an owner practice call) talked to this agent? `undefined` while unknown. */
export function useAgentTalkedTo(agentId: string | null | undefined): boolean | undefined {
  const { data } = useSWR<boolean>(
    agentId ? `/api/analytics/agents/${agentId}` : null,
    async (p: string) => ((await api.get<{ total_turns?: number }>(p))?.total_turns ?? 0) > 0,
    quiet
  );
  return data;
}

/** How many meeting or appointment requests wait for the owner. `undefined` while unknown. */
export function usePendingRequests(): number | undefined {
  const { data } = useSWR<number>(
    "/api/meetings?status=pending",
    async (p: string) => (await api.get<{ counts?: { pending?: number } }>(p))?.counts?.pending ?? 0,
    quiet
  );
  return data;
}

export type SetupTab = "basics" | "phone" | "channels" | "advanced";

export interface SetupStep {
  id: "instructions" | "documents" | "voice" | "talked" | "phone";
  label: string;
  done: boolean;
  /** Where to go to finish it. */
  tab: SetupTab;
}

/** The five real checks behind "Setup 3 of 5". Unknown answers (still loading, or the check failed) count as not done. */
export function setupSteps(agent: Agent, facts: { talkedTo?: boolean; phoneConnected?: boolean }): SetupStep[] {
  return [
    { id: "instructions", label: "Tell the agent how to behave", done: agent.system_instructions.trim().length > 0, tab: "advanced" },
    { id: "documents", label: "Add your documents", done: agent.document_count > 0, tab: "basics" },
    { id: "voice", label: "Choose a voice", done: !!agent.voice_id, tab: "basics" },
    { id: "talked", label: "Talk to your agent once", done: !!facts.talkedTo, tab: "basics" },
    { id: "phone", label: "Connect your phone line", done: !!facts.phoneConnected, tab: "phone" },
  ];
}

/** Ready to answer questions: it has documents to answer from. */
export const agentIsReady = (agent: Agent): boolean => agent.document_count > 0;

/** Link that opens the Playground with this agent selected (the Playground reads `?agent=`). */
export const talkHref = (agentId: string): string => `/dashboard/playground?agent=${encodeURIComponent(agentId)}`;
