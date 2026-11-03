import type { Metadata } from "next";
import { Hero } from "@/components/site/Hero";
import { WhySection } from "@/components/site/WhySection";
import { HearIt } from "@/components/site/Proof";
import { Languages, Products, StackStrip, StatsBand } from "@/components/site/Sections";
import { FinalCta } from "@/components/site/Ending";
import { GoDeeper } from "@/components/site/GoDeeper";
import { LegacyHashRedirect } from "@/components/site/LegacyHashRedirect";

const TITLE = "BonClay: your voice, in Indian languages";
const DESCRIPTION =
  "Speech in 22 Indian languages, and your own voice cloned from a few seconds of consented speech. Runs on one laptop. Every clip is watermarked and comes with a signed receipt anyone can verify.";

// Server component so the page can export real metadata; the interactive pieces are client components below.
export const metadata: Metadata = {
  title: { absolute: TITLE },
  description: DESCRIPTION,
  openGraph: {
    type: "website",
    siteName: "BonClay",
    title: TITLE,
    description: DESCRIPTION,
    // The 1200x630 card rendered by app/opengraph-image.tsx. It is named here because a page-level openGraph object
    // replaces the file-based image that the root layout would otherwise have passed down.
    images: [{ url: "/opengraph-image", width: 1200, height: 630, alt: "BonClay: your voice, in Indian languages" }],
  },
  alternates: { canonical: "/" },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION, images: ["/opengraph-image"] },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "BonClay",
  applicationCategory: "MultimediaApplication",
  description: DESCRIPTION,
};

export default function HomePage() {
  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      {/* Old single-page links (/#faq, /#rig-check ...) move to the page that now holds that section. */}
      <LegacyHashRedirect />
      <Hero />
      <StatsBand />
      <StackStrip />
      <WhySection compact />
      <Products />
      <Languages />
      <HearIt />
      <GoDeeper />
      <FinalCta />
    </>
  );
}
