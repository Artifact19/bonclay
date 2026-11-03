import { render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import SiteLayout from "@/app/(site)/layout";
import HomePage, { metadata as homeMeta } from "@/app/(site)/page";
import HowItWorksPage, { metadata as howMeta } from "@/app/(site)/how-it-works/page";
import ChannelsPage, { metadata as channelsMeta } from "@/app/(site)/channels/page";
import ProofPage, { metadata as proofMeta } from "@/app/(site)/proof/page";
import RunItPage, { metadata as runMeta } from "@/app/(site)/run-it/page";
import FaqPage, { metadata as faqMeta } from "@/app/(site)/faq/page";
import sitemap from "@/app/sitemap";
import robots from "@/app/robots";
import { ALL_SITE_PAGES, isSitePath, nextPage } from "@/components/site/routes";

let mockPath = "/";
jest.mock("next/navigation", () => ({ usePathname: () => mockPath, useRouter: () => ({ replace: jest.fn(), push: jest.fn() }) }));
jest.mock("@/components/Logo", () => ({ Logo: () => <span>BonClay</span> }));
jest.mock("@/components/site/VoiceReel", () => ({ VoiceReel: () => null }));
jest.mock("@/components/site/fonts", () => ({ SCRIPT_FONT: new Proxy({}, { get: () => "" }), useScriptFontsNear: () => true }));
// The real SmoothScroll drives Lenis and ScrollTrigger; here only its context and children matter.
jest.mock("@/components/site/SmoothScroll", () => ({
  useLenis: () => null,
  SmoothScroll: ({ children }: { children: ReactNode }) => <>{children}</>,
}));
jest.mock("gsap", () => {
  const chain: Record<string, unknown> = {};
  const tl = new Proxy(chain, { get: (_t, k) => (k === "then" ? undefined : () => tl) });
  const mm = { add: jest.fn(), revert: jest.fn() };
  const gsap = {
    registerPlugin: jest.fn(),
    matchMedia: () => mm,
    context: jest.fn(),
    fromTo: jest.fn(),
    to: jest.fn(),
    set: jest.fn(),
    timeline: () => tl,
    utils: { toArray: () => [], selector: () => () => [] },
  };
  return { gsap, default: gsap };
});
jest.mock("gsap/ScrollTrigger", () => ({ ScrollTrigger: { refresh: jest.fn(), create: jest.fn(), config: jest.fn() } }));

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (q: string) => ({ matches: q.includes("reduce"), media: q, addEventListener: jest.fn(), removeEventListener: jest.fn(), addListener: jest.fn(), removeListener: jest.fn() }),
  });
});

type Case = { path: string; page: () => ReactNode; h1: RegExp | string; navLabel: string | null; meta: { title?: unknown; description?: unknown } };

const CASES: Case[] = [
  { path: "/", page: () => <HomePage />, h1: /Your\s+voice,\s+in\s*Indian\s+languages\./, navLabel: null, meta: homeMeta },
  { path: "/how-it-works", page: () => <HowItWorksPage />, h1: "From a few words to a clip you can check.", navLabel: "How it works", meta: howMeta },
  { path: "/channels", page: () => <ChannelsPage />, h1: "Where people can reach your agent.", navLabel: "Channels", meta: channelsMeta },
  { path: "/proof", page: () => <ProofPage />, h1: "Numbers, attacks and comparisons.", navLabel: "Proof", meta: proofMeta },
  { path: "/run-it", page: () => <RunItPage />, h1: "Can you run it?", navLabel: "Can you run it?", meta: runMeta },
  { path: "/faq", page: () => <FaqPage />, h1: "Questions people ask.", navLabel: "FAQ", meta: faqMeta },
];

function renderAt(path: string, page: () => ReactNode) {
  mockPath = path;
  return render(<SiteLayout>{page()}</SiteLayout>);
}

describe.each(CASES)("page $path", ({ path, page, h1, navLabel }) => {
  it("has exactly one h1, the content landmark and the shared footer", () => {
    const { container } = renderAt(path, page);
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0].textContent).toMatch(h1);
    expect(container.querySelector("#content")).not.toBeNull();
    expect(screen.getByRole("contentinfo", { name: "Site footer" })).toBeInTheDocument();
    expect(screen.getAllByRole("navigation", { name: "Main" })).toHaveLength(1);
  });

  it("marks the right nav item (and only it) as the current page", () => {
    renderAt(path, page);
    const bar = screen.getByRole("navigation", { name: "Main" });
    const current = Array.from(bar.querySelectorAll("[aria-current='page']"));
    if (navLabel === null) {
      expect(current.map((e) => e.getAttribute("aria-label"))).toEqual(["BonClay home"]);
    } else {
      expect(current).toHaveLength(1);
      expect(current[0].textContent).toBe(navLabel);
    }
  });
});

describe("page content", () => {
  it("Home is the short version: hero, why, products, languages, hear, go-deeper cards, closing call to action", () => {
    const { container } = renderAt("/", () => <HomePage />);
    for (const id of ["top", "why", "products", "languages", "hear", "go-deeper", "start"]) expect(container.querySelector(`#${id}`)).not.toBeNull();
    // the languages carousel comes right before the voice samples
    expect(container.querySelector("#languages")!.compareDocumentPosition(container.querySelector("#hear")!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(container.querySelector("#products")!.compareDocumentPosition(container.querySelector("#languages")!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // the sections that moved are not on Home
    for (const id of ["consent", "how", "channels", "numbers", "attacks", "compare", "rig-check", "faq"]) expect(container.querySelector(`#${id}`)).toBeNull();
    const deeper = within(container.querySelector<HTMLElement>("#go-deeper")!);
    for (const p of ALL_SITE_PAGES.slice(1)) expect(deeper.getByRole("link", { name: new RegExp(p.title) })).toHaveAttribute("href", p.href);
    // the short Why has no statistics block and no outside links
    expect(container.querySelector("#why a[target=_blank]")).toBeNull();
  });

  it("/how-it-works holds the pipeline and consent (the languages moved to Home), then a Next link to Channels", () => {
    const { container } = renderAt("/how-it-works", () => <HowItWorksPage />);
    for (const id of ["how", "consent"]) expect(container.querySelector(`section#${id}`)).not.toBeNull();
    expect(container.querySelector("#languages")).toBeNull();
    expect(within(screen.getByRole("navigation", { name: "Next page" })).getByRole("link")).toHaveAttribute("href", "/channels");
  });

  it("/channels holds the channels section, then a Next link to Proof", () => {
    const { container } = renderAt("/channels", () => <ChannelsPage />);
    expect(container.querySelector("section#channels")).not.toBeNull();
    expect(within(screen.getByRole("navigation", { name: "Next page" })).getByRole("link")).toHaveAttribute("href", "/proof");
  });

  it("/proof holds measured numbers, attacks, the comparison and the sourced why, then a Next link to Can you run it?", () => {
    const { container } = renderAt("/proof", () => <ProofPage />);
    for (const id of ["numbers", "attacks", "compare", "why-evidence"]) expect(container.querySelector(`section#${id}`)).not.toBeNull();
    expect(within(screen.getByRole("navigation", { name: "Next page" })).getByRole("link")).toHaveAttribute("href", "/run-it");
  });

  it("/run-it shows the rig check as the main content with its picker and result, and a Next link to the FAQ", () => {
    const { container } = renderAt("/run-it", () => <RunItPage />);
    expect(container.querySelector("section#rig-check")).not.toBeNull();
    expect(screen.getByRole("group", { name: "Your machine" })).toBeInTheDocument();
    expect(screen.getByTestId("rig-result")).toBeInTheDocument();
    // the page heading replaces the section's own h2: there is no second "Will BonClay run" heading
    expect(screen.queryByRole("heading", { level: 2, name: "Will BonClay run on your laptop?" })).toBeNull();
    expect(within(screen.getByRole("navigation", { name: "Next page" })).getByRole("link")).toHaveAttribute("href", "/faq");
  });

  it("/faq holds the questions and the closing call to action, then a way back to Home", () => {
    const { container } = renderAt("/faq", () => <FaqPage />);
    expect(container.querySelectorAll("details.faq-item").length).toBeGreaterThanOrEqual(8);
    expect(container.querySelector("section#start")).not.toBeNull();
    expect(within(screen.getByRole("navigation", { name: "Next page" })).getByRole("link")).toHaveAttribute("href", "/");
  });
});

describe("metadata", () => {
  it("keeps the exact Home title and gives every other page its own title and description", () => {
    expect(homeMeta.title).toEqual({ absolute: "BonClay: your voice, in Indian languages" });
    const titles = CASES.map((c) => JSON.stringify(c.meta.title));
    const descriptions = CASES.map((c) => String(c.meta.description));
    expect(new Set(titles).size).toBe(CASES.length);
    expect(new Set(descriptions).size).toBe(CASES.length);
    for (const c of CASES.slice(1)) {
      expect(JSON.stringify(c.meta.title)).toMatch(/ \| BonClay/);
      expect(String(c.meta.description).length).toBeGreaterThan(40);
    }
  });
});

describe("routes, sitemap and robots", () => {
  it("lists the six public pages in the sitemap and points robots at it", () => {
    const urls = sitemap().map((e) => e.url);
    expect(urls).toHaveLength(6);
    for (const p of ["/how-it-works", "/channels", "/proof", "/run-it", "/faq"]) expect(urls.some((u) => u.endsWith(p))).toBe(true);
    expect(urls.some((u) => /\/(dashboard|studio|verify)/.test(u))).toBe(false);
    const r = robots();
    expect(String(r.sitemap)).toMatch(/\/sitemap\.xml$/);
    expect(JSON.stringify(r.rules)).toContain("/dashboard");
  });

  it("treats only the marketing pages as site paths and chains them in reading order", () => {
    for (const p of ALL_SITE_PAGES) expect(isSitePath(p.href)).toBe(true);
    expect(isSitePath("/faq/")).toBe(true);
    for (const p of ["/dashboard", "/studio", "/verify", "/voices", "/login", "/register", "/nope"]) expect(isSitePath(p)).toBe(false);
    expect(nextPage("/")?.href).toBe("/how-it-works");
    expect(nextPage("/run-it")?.href).toBe("/faq");
    expect(nextPage("/faq")).toBeUndefined();
  });
});
