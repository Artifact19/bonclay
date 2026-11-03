import type { Metadata } from "next";

const SITE_NAME = "BonClay";

/** Metadata for one marketing page: a unique title and description, a canonical path and matching social cards.
 *  The title is "<page> | BonClay"; Home keeps its own exact title in app/(site)/page.tsx. */
export function pageMetadata({ title, description, path }: { title: string; description: string; path: string }): Metadata {
  const full = `${title} | ${SITE_NAME}`;
  return {
    title: { absolute: full },
    description,
    alternates: { canonical: path },
    openGraph: {
      type: "website",
      siteName: SITE_NAME,
      title: full,
      description,
      url: path,
      images: [{ url: "/logo.png", width: 180, height: 180, alt: SITE_NAME }],
    },
    twitter: { card: "summary", title: full, description, images: ["/logo.png"] },
  };
}
