/**
 * The public marketing pages, in reading order. One list feeds the nav, the footer, the "Go deeper" cards, the "Next"
 * links, the sitemap and the check that decides which routes get the marketing page treatment (see isSitePath).
 * `blurb` is one plain sentence that only restates what the page contains.
 */
export interface SitePage {
  href: string;
  /** Short label for the nav and footer. */
  nav: string;
  /** Heading of the card that links here. */
  title: string;
  blurb: string;
}

export const HOME: SitePage = {
  href: "/",
  nav: "Home",
  title: "BonClay",
  blurb: "Speech in 22 Indian languages, and your own voice from a few seconds of consented speech.",
};

/** Every page after Home, in the order a visitor would read them. */
export const SITE_PAGES: SitePage[] = [
  {
    href: "/how-it-works",
    nav: "How it works",
    title: "How it works",
    blurb: "The pipeline from words to a verifiable clip, and the spoken consent behind every clone.",
  },
  {
    href: "/channels",
    nav: "Channels",
    title: "Channels",
    blurb: "One agent on the phone, in a website widget, in meetings and on WhatsApp.",
  },
  {
    href: "/proof",
    nav: "Proof",
    title: "Proof",
    blurb: "What we measured, the attacks we tried, and how BonClay compares.",
  },
  {
    href: "/run-it",
    nav: "Can you run it?",
    title: "Can you run it?",
    blurb: "Pick your laptop and see what works well, what is slow and what will not fit.",
  },
  {
    href: "/faq",
    nav: "FAQ",
    title: "FAQ",
    blurb: "What it costs, what runs on your machine, and what is not ready yet.",
  },
];

/** Home first, then SITE_PAGES. */
export const ALL_SITE_PAGES: SitePage[] = [HOME, ...SITE_PAGES];

/** The page after this one in reading order, or undefined for the last. */
export function nextPage(href: string): SitePage | undefined {
  const i = ALL_SITE_PAGES.findIndex((p) => p.href === href);
  return i >= 0 ? ALL_SITE_PAGES[i + 1] : undefined;
}

/** True for the marketing pages (not for /dashboard, /studio, /verify, /voices or the auth routes). */
export function isSitePath(pathname: string | null | undefined): boolean {
  if (!pathname) return false;
  const p = pathname.length > 1 && pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
  return ALL_SITE_PAGES.some((page) => page.href === p);
}

const LEGACY_HASHES: Record<string, string> = {
  consent: "/how-it-works#consent",
  pipeline: "/how-it-works#how",
  how: "/how-it-works#how",
  channels: "/channels",
  proof: "/proof",
  numbers: "/proof#numbers",
  attacks: "/proof#attacks",
  compare: "/proof#compare",
  rig: "/run-it",
  "rig-check": "/run-it",
  faq: "/faq",
};

function decodeSafe(s: string): string {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

/** Map an old single-page anchor to where that content lives now. Returns null when the hash still belongs to Home
 *  (#why, #products, #hear, #start ...) or is unknown, so nothing is redirected. Case-insensitive; the leading "#" is optional. */
export function mapLegacyHash(hash: string): string | null {
  const id = decodeSafe(hash.replace(/^#/, "")).toLowerCase();
  return Object.prototype.hasOwnProperty.call(LEGACY_HASHES, id) ? LEGACY_HASHES[id] : null;
}
