"use client";

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { usePathname, useRouter } from "next/navigation";
import { AREA_EVENT, AREA_HOME, AREA_STORAGE_KEY, areaForPath, readRememberedArea, rememberArea, type Area } from "@/lib/areas";

function subscribe(onChange: () => void) {
  const onStorage = (e: StorageEvent) => (e.key === null || e.key === AREA_STORAGE_KEY) && onChange();
  window.addEventListener(AREA_EVENT, onChange);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(AREA_EVENT, onChange);
    window.removeEventListener("storage", onStorage);
  };
}

/**
 * The remembered area. `undefined` while the server (and the first browser pass) renders, because only the browser
 * knows; `null` once the browser has looked and nothing is remembered.
 */
export function useRememberedArea(): Area | null | undefined {
  return useSyncExternalStore<Area | null | undefined>(subscribe, readRememberedArea, () => undefined);
}

/**
 * The area the person is in. A route that belongs to one area decides it (and is remembered); a shared route (Voices,
 * API keys) keeps the remembered one; with nothing remembered the answer is Agents, the original dashboard.
 */
export function useArea() {
  const pathname = usePathname();
  const router = useRouter();
  const remembered = useRememberedArea();
  const routeArea = areaForPath(pathname);

  // The choice just made in the switch shows at once, until the new page has loaded (the route then decides again).
  const [choice, setChoice] = useState<Area | null>(null);
  const [seenPath, setSeenPath] = useState(pathname);
  if (seenPath !== pathname) {
    setSeenPath(pathname);
    setChoice(null);
  }

  // /dashboard is the one ambiguous landing page: a Studio person arriving there is about to be sent to Studio home,
  // so it neither counts as "went to Agents" nor shows the Agents list for a moment.
  const atLanding = pathname === AREA_HOME.agents;
  const area: Area = choice ?? (atLanding && remembered === "studio" ? "studio" : (routeArea ?? remembered ?? "agents"));

  // Remember the area of each page you open (not on every render: the switch has already saved a newer choice).
  useEffect(() => {
    if (routeArea && !atLanding && routeArea !== readRememberedArea()) rememberArea(routeArea);
  }, [routeArea, atLanding, pathname]);

  /** Switch area: remember it first (so /dashboard does not bounce a Studio person back), then go to that area's home. */
  const switchTo = useCallback(
    (next: Area) => {
      setChoice(next);
      rememberArea(next);
      if (pathname !== AREA_HOME[next]) router.push(AREA_HOME[next]);
    },
    [pathname, router]
  );

  // False only during the server / first browser pass on a page that does not say which area it is in: the answer
  // may still change once the browser has read the remembered one, so nobody should start area-specific work yet.
  const known = (!!routeArea && !atLanding) || remembered !== undefined;

  return { area, known, switchTo };
}
