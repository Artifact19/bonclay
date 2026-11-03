/** The two areas of the dashboard. Studio is for making audio; Agents is for voice agents that answer calls. */
export type Area = "studio" | "agents";

export const AREAS: { id: Area; label: string; hint: string }[] = [
  { id: "studio", label: "Studio", hint: "Make audio" },
  { id: "agents", label: "Agents", hint: "Voice agents" },
];

export const AREA_STORAGE_KEY = "bonclay.area";
export const AREA_EVENT = "bonclay:area-change";

/** Where each area starts. Agents keeps the original /dashboard so old links and bookmarks still work. */
export const AREA_HOME: Record<Area, string> = { studio: "/dashboard/studio/home", agents: "/dashboard" };

/**
 * The explicit map: which area each route belongs to. "shared" routes are useful in both areas, so they keep
 * whichever area the person is in (Voices: agents need a voice too; API keys: the same keys for everything).
 * Not a plain prefix test: /dashboard/studio/agents sits under /dashboard/studio but is an Agents page, so the
 * longest matching entry wins and each entry matches only at a path boundary.
 */
const ROUTES: { path: string; area: Area | "shared"; exact?: boolean }[] = [
  { path: "/dashboard", area: "agents", exact: true },
  { path: "/dashboard/studio", area: "studio" },
  { path: "/dashboard/studio/home", area: "studio" },
  { path: "/dashboard/studio/agents", area: "agents" },
  { path: "/dashboard/say-it-once", area: "studio" },
  { path: "/dashboard/calls", area: "agents" },
  { path: "/dashboard/campaigns", area: "agents" },
  { path: "/dashboard/approvals", area: "agents" },
  { path: "/dashboard/playground", area: "agents" },
  { path: "/dashboard/knowledge", area: "agents" },
  { path: "/dashboard/knowledge/try", area: "agents" },
  { path: "/dashboard/analytics", area: "agents" },
  { path: "/dashboard/eval", area: "agents" },
  { path: "/dashboard/voices", area: "shared" },
  { path: "/dashboard/keys", area: "shared" },
  { path: "/verify", area: "shared" },
];

/** The area a route belongs to, or null when it is shared (or unknown) and the current area should stay. */
export function areaForPath(pathname: string | null | undefined): Area | null {
  const p = (pathname ?? "").replace(/\/+$/, "") || "/";
  let best: (typeof ROUTES)[number] | null = null;
  for (const r of ROUTES) {
    const hit = r.exact ? p === r.path : p === r.path || p.startsWith(r.path + "/");
    if (hit && (!best || r.path.length > best.path.length)) best = r;
  }
  return best && best.area !== "shared" ? best.area : null;
}

export function isArea(v: unknown): v is Area {
  return v === "studio" || v === "agents";
}

/** The area remembered from last time (null: none yet, or storage is blocked). */
export function readRememberedArea(): Area | null {
  try {
    const v = localStorage.getItem(AREA_STORAGE_KEY);
    return isArea(v) ? v : null;
  } catch {
    return null;
  }
}

export function rememberArea(area: Area): void {
  try {
    localStorage.setItem(AREA_STORAGE_KEY, area);
  } catch {
    /* storage blocked: the area still follows the route */
  }
  try {
    window.dispatchEvent(new Event(AREA_EVENT));
  } catch {
    /* no window */
  }
}
