"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { mapLegacyHash } from "./routes";

/**
 * Old links such as /#faq or /#rig-check pointed at sections of the one long page. Those sections now live on their own
 * pages, so on Home this moves the visitor there with router.replace (no extra history entry, so Back still works).
 * Hashes that still belong to Home (#why, #products, #hear, #start ...) are left alone. Renders nothing.
 */
export function LegacyHashRedirect() {
  const router = useRouter();
  useEffect(() => {
    const go = () => {
      const target = mapLegacyHash(window.location.hash);
      if (target) router.replace(target);
    };
    go();
    window.addEventListener("hashchange", go);
    return () => window.removeEventListener("hashchange", go);
  }, [router]);
  return null;
}
