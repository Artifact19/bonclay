import type { Metadata, Viewport } from "next";
import "./globals.css";
import { ALL_SITE_PAGES } from "@/components/site/routes";

// This root layout is shared by every route, so it stays minimal: no fonts, auth, sidebar or toaster. Those live in
// the route groups: app/(app)/layout.tsx (dashboard, studio, voices, auth pages) and the public layouts (site, verify).

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#F7F4ED",
};

export const metadata: Metadata = {
  // Absolute URLs for Open Graph / Twitter images. Set NEXT_PUBLIC_SITE_URL in production.
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: "BonClay: your voice, in Indian languages",
  description:
    "Speech in 22 Indian languages. Clone your own voice from a few seconds of consented speech, then let it read, answer and talk. Every clip is watermarked and comes with a signed receipt anyone can check.",
  keywords: ["voice cloning", "text to speech", "Indian languages", "voice agents", "consent", "watermark", "BonClay"],
  openGraph: {
    type: "website",
    siteName: "BonClay",
    title: "BonClay: your voice, in Indian languages",
    description: "Consented voice cloning, voice agents that answer from your documents, and audio anyone can verify. Runs on one laptop.",
  },
  twitter: {
    card: "summary",
    title: "BonClay: your voice, in Indian languages",
    description: "Consented voice cloning, voice agents that answer from your documents, and audio anyone can verify.",
  },
  icons: {
    icon: "/favicon.ico",
    apple: "/logo.png",
    shortcut: "/favicon.ico",
  },
};

/** Runs before first paint so the saved theme never flashes. Light Musmentor theme is default everywhere. */
const themeScript = `(function(){try{document.documentElement.setAttribute('data-theme','light');localStorage.setItem('bonclay-theme','light');}catch(e){}})();`;

/** Before first paint on the marketing pages (the list in components/site/routes): hide the hero and [data-reveal] elements via CSS (html.reveal-on) so server-rendered text
 *  never flashes and then snaps away when GSAP starts. Skipped under reduced motion and without JS (everything shows).
 *  SmoothScroll sets window.__bonclayMotion; if hydration never gets there, the class is dropped after 6 s. */
const SITE_PATHS = JSON.stringify(ALL_SITE_PAGES.map((p) => p.href));
const revealScript = `(function(){try{var d=document.documentElement;var p=location.pathname;if(p.length>1&&p.slice(-1)==="/")p=p.slice(0,-1);if(${SITE_PATHS}.indexOf(p)<0||window.matchMedia("(prefers-reduced-motion: reduce)").matches)return;d.classList.add("reveal-on");setTimeout(function(){if(!window.__bonclayMotion)d.classList.remove("reveal-on");},6000);}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="light" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <script dangerouslySetInnerHTML={{ __html: revealScript }} />
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="apple-touch-icon" href="/logo.png" />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
