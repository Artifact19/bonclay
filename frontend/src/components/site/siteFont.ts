import { Instrument_Serif, Manrope } from "next/font/google";

/**
 * The marketing site's two Latin faces, both from Google Fonts under the SIL OFL 1.1.
 *
 * Instrument Serif (Rodrigo Fuenzalida, Jordan Egstad): the display face. Regular and italic, one weight. Every h1 and
 * h2 on the site is set in it at a tight negative tracking, with one phrase in the italic cut and the bronze accent
 * colour (see .site .accent in globals.css). It is a transitional serif with a small x-height, so it is used large
 * only; nothing under the h3 size is set in it.
 *
 * Manrope (Mikhail Sharanda): the text and interface face, variable in weight (200 to 800). Body copy at 400, labels
 * and buttons at 500 to 600, the big numerals at 500 with tight tracking. The brand wordmark (components/Logo) loads its
 * own Regular cut of the same family through lib/brandFont, so the wordmark and the site text agree.
 *
 * Both are applied only inside `.site` (the marketing pages and /verify) through CSS variables, so the dashboard keeps
 * Geist and does not download them. next/font self-hosts and preloads them for the routes that import this module.
 */
export const siteFont = Manrope({
  subsets: ["latin"],
  variable: "--s-font-latin",
  display: "swap",
  fallback: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
});

export const siteSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
  variable: "--s-font-serif",
  display: "swap",
  fallback: ["Georgia", "Times New Roman", "serif"],
  // Instrument Serif sits smaller than Georgia at the same size; these keep the fallback's line lengths close so the
  // swap does not reflow the hero.
  adjustFontFallback: true,
});
