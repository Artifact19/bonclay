import type { MetadataRoute } from "next";
import { ALL_SITE_PAGES } from "@/components/site/routes";

/** Public marketing pages only. The base address comes from NEXT_PUBLIC_SITE_URL, the same value the root layout uses
 *  for metadataBase, and falls back to the local dev address so the file is always valid. */
const SITE_URL = (process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000").replace(/\/+$/, "");

export default function sitemap(): MetadataRoute.Sitemap {
  return ALL_SITE_PAGES.map((p) => ({
    url: p.href === "/" ? SITE_URL : `${SITE_URL}${p.href}`,
    changeFrequency: "monthly" as const,
    priority: p.href === "/" ? 1 : 0.7,
  }));
}
