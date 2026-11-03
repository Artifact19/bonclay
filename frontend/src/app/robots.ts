import type { MetadataRoute } from "next";

/** The marketing pages are public; the signed-in app is not worth indexing. */
export default function robots(): MetadataRoute.Robots {
  const base = (process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000").replace(/\/+$/, "");
  return {
    rules: [{ userAgent: "*", allow: "/", disallow: ["/dashboard", "/studio", "/voices", "/login", "/register"] }],
    sitemap: `${base}/sitemap.xml`,
  };
}
