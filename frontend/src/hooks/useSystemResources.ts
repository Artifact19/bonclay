"use client";

import useSWR from "swr";
import { api, ApiError } from "@/lib/api";
import type { SystemResources } from "@/types/api";

export const RESOURCES_KEY = "/api/system/resources";

/** How often the sidebar footer re-reads memory use, and how often the open "Models and memory" panel does. */
export const FOOTER_POLL_MS = 15000;
export const PANEL_POLL_MS = 3000;

/** True when the body is a usable snapshot (an older server, or a proxy error page, must not break the footer). */
export function isSnapshot(d: unknown): d is SystemResources {
  const s = (d as SystemResources | undefined)?.system;
  return !!s && typeof s.total_mb === "number" && typeof s.used_mb === "number" && Array.isArray((d as SystemResources).models);
}

/**
 * Memory and loaded models (GET /api/system/resources). SWR does not poll while the tab is hidden
 * (refreshWhenHidden is off) and re-reads when the tab comes back; the previous answer stays on screen while it does.
 */
export function useSystemResources(intervalMs: number = FOOTER_POLL_MS) {
  return useSWR<SystemResources, ApiError>(RESOURCES_KEY, (p: string) => api.get<SystemResources>(p), {
    refreshInterval: intervalMs,
    refreshWhenHidden: false,
    revalidateOnFocus: true,
    shouldRetryOnError: false,
    keepPreviousData: true,
    dedupingInterval: 1000,
  });
}
