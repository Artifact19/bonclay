import { Geist, Geist_Mono } from "next/font/google";

// The app shell's faces: Geist (dashboard body and display) and Geist Mono. Manrope (the wordmark) is a separate module
// (lib/brandFont) and is added by the caller where the Logo renders.
// preload: false. The public site shares the CSS that declares these faces (Turbopack merges it with the Manrope CSS), and
// a preload would download them on every public page. Without a preload a face downloads only when text uses it, so the
// site, which never uses Geist, fetches nothing; the dashboard fetches them on first use.

/** Dashboard body face. Its variable is --nf-body. */
export const geist = Geist({
  subsets: ["latin"],
  variable: "--nf-body",
  display: "swap",
  preload: false,
});

export const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  display: "swap",
  preload: false,
});

/**
 * Classes for an element whose subtree needs Geist. The variables only resolve on the element that declares them and its
 * descendants, so put this on the wrapper of the subtree, and on any portal root (portals render into document.body,
 * outside the wrapper). `app-fonts` (globals.css) points --nf-display at that element's Geist.
 */
export const APP_FONT_CLASSES = `${geist.variable} ${geistMono.variable} app-fonts font-sans`;
