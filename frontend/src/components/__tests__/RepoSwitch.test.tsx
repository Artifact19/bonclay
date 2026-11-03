/* eslint-disable @typescript-eslint/no-require-imports -- the components must be loaded again after the environment switch changes */
/**
 * The site sends nobody to the (private) GitHub repository unless NEXT_PUBLIC_REPO_PUBLIC=1 (components/site/repo.ts).
 * Off: no link on any landing section points to github.com and nothing claims "the source is on GitHub".
 * On: the links and sentences come back.
 */
import type { ReactElement } from "react";

jest.mock("@/components/Logo", () => ({ Logo: () => <span>BonClay</span> }));
jest.mock("@/components/site/VoiceReel", () => ({ VoiceReel: () => null }));
jest.mock("gsap", () => {
  const chain: Record<string, unknown> = {};
  const tl = new Proxy(chain, { get: (_t, k) => (k === "then" ? undefined : () => tl) });
  return { gsap: { registerPlugin: jest.fn(), matchMedia: () => ({ add: jest.fn(), revert: jest.fn() }), fromTo: jest.fn(), to: jest.fn(), set: jest.fn(), timeline: () => tl, utils: { toArray: () => [] } } };
});
jest.mock("@/components/site/SmoothScroll", () => ({ useLenis: () => null, SmoothScroll: () => null }));
jest.mock("@/components/site/fonts", () => ({ SCRIPT_FONT: new Proxy({}, { get: () => "" }), useScriptFontsNear: () => true }));
jest.mock("gsap/ScrollTrigger", () => ({ ScrollTrigger: { refresh: jest.fn(), create: jest.fn() } }));

function sections(): Record<string, ReactElement> {
  const { Hero } = require("@/components/site/Hero");
  const { WhySection } = require("@/components/site/WhySection");
  const { Comparison, HearIt, Measured } = require("@/components/site/Proof");
  const { RigCheck } = require("@/components/site/RigCheck");
  const { Attacks, Consent, Languages, Pipeline, Products, StackStrip, SiteFooter } = require("@/components/site/Sections");
  const { Faq, FinalCta, Footer } = require("@/components/site/Ending");
  const { Channels } = require("@/components/site/ChannelsSection");
  const all = { Hero, WhySection, StackStrip, Products, Languages, HearIt, Consent, Channels, Pipeline, Measured, Attacks, Comparison, RigCheck, Faq, FinalCta, Footer, SiteFooter };
  return Object.fromEntries(Object.entries(all).map(([name, C]) => { const Comp = C as () => ReactElement; return [name, <Comp key={name} />]; }));
}

function renderAll(env: "0" | "1") {
  process.env.NEXT_PUBLIC_REPO_PUBLIC = env;
  let markup = "";
  jest.isolateModules(() => {
    // React and its server renderer must come from the same fresh module registry as the components
    const { renderToStaticMarkup } = require("react-dom/server");
    for (const el of Object.values(sections())) markup += renderToStaticMarkup(el);
  });
  const hrefs = Array.from(markup.matchAll(/href="([^"]*)"/g), (m) => m[1].replace(/&amp;/g, "&")).filter((h) => h.includes("github.com"));
  const html = markup.replace(/<script[\s\S]*?<\/script>/g, " ").replace(/<[^>]+>/g, " ").replace(/&#x27;|&apos;/g, "'").replace(/&amp;/g, "&").replace(/\s+/g, " ");
  return { html, hrefs };
}

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (q: string) => ({ matches: false, media: q, addEventListener: jest.fn(), removeEventListener: jest.fn(), addListener: jest.fn(), removeListener: jest.fn() }),
  });
});

afterEach(() => { delete process.env.NEXT_PUBLIC_REPO_PUBLIC; });

test("with the switch off no section links to github.com or says the source is on GitHub", () => {
  const { html, hrefs } = renderAll("0");
  expect(hrefs).toEqual([]);
  expect(html).not.toMatch(/source is on GitHub|source on GitHub|Source on GitHub/);
  expect(html).toMatch(/MIT-licensed/);
  expect(html).toMatch(/shared on request/);
  expect(html).not.toMatch(/open[- ]source code|\bis open source\b/i);
});

test("with the switch on the GitHub links and sentences come back", () => {
  const { html, hrefs } = renderAll("1");
  expect(hrefs.length).toBeGreaterThan(8);
  expect(hrefs).toContain("https://github.com/Artifact19/bonclay");
  expect(hrefs).toContain("https://github.com/Artifact19/bonclay/blob/main/docs/RESULTS.md");
  expect(html).toMatch(/Source on GitHub/);
});
