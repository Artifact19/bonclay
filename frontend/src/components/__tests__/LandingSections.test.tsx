import { existsSync, statSync } from "fs";
import path from "path";
import { act, render, screen } from "@testing-library/react";
import { MONUMENTS, MONUMENT_CREDITS } from "@/components/site/monuments";
import { SCRIPT_FONT } from "@/components/site/fonts";
import { Attacks, Consent, Languages, Pipeline, Products, StackStrip } from "@/components/site/Sections";

jest.mock("@/components/Logo", () => ({ Logo: () => <span>BonClay</span> }));
jest.mock("@/components/site/SmoothScroll", () => ({ useLenis: () => null }));
jest.mock("gsap", () => {
  const mm = { add: jest.fn(), revert: jest.fn() };
  const gsap = { registerPlugin: jest.fn(), matchMedia: () => mm, utils: { toArray: () => [] } };
  return { gsap, default: gsap };
});
jest.mock("gsap/ScrollTrigger", () => ({ ScrollTrigger: { refresh: jest.fn() } }));

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (q: string) => ({ matches: false, media: q, addEventListener: jest.fn(), removeEventListener: jest.fn(), addListener: jest.fn(), removeListener: jest.fn() }),
  });
});

describe("landing sections", () => {
  it("points both product screenshots at files that exist", () => {
    const { container } = render(<Products />);
    const srcs = Array.from(container.querySelectorAll("img")).map((i) => decodeURIComponent(i.getAttribute("src") ?? ""));
    expect(srcs.some((s) => s.includes("/landing/shots/studio-v7.png"))).toBe(true);
    expect(srcs.some((s) => s.includes("/landing/shots/agent-studio-v7.png"))).toBe(true);
    expect(srcs.join()).not.toMatch(/-v6\.png/);
    expect(container.querySelectorAll("img[draggable=false]").length).toBe(2);
    expect(screen.getByText(/phone calls run through exotel/i)).toBeInTheDocument();
  });

  it("makes one language claim: 22 languages, and your own voice in them", () => {
    const { container } = render(<Languages />);
    expect(screen.getByRole("heading", { name: /22 Indian languages\. Your own voice in them\./ })).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/Hindi in your own voice|Preset or your voice|Eleven|Twelve/);
    // 11 cards, each rendered twice so the slow loop is seamless; the copies are hidden from assistive tech.
    expect(container.querySelectorAll("figure:not([aria-hidden])")).toHaveLength(11);
    expect(container.querySelectorAll("figure[aria-hidden='true']")).toHaveLength(11);
    // Motion drives the loop on its own, and the arrows are always there to nudge it.
    expect(screen.getByRole("button", { name: "Next language" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Previous language" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /slow carousel/ })).toBeInTheDocument();
  });

  it("fades a monument photo behind every language card, from a file that exists", () => {
    const { container } = render(<Languages />);
    const cards = Array.from(container.querySelectorAll("figure:not([aria-hidden])"));
    expect(cards).toHaveLength(11);
    const seen = new Set<string>();
    for (const card of cards) {
      const code = card.getAttribute("lang")!;
      const m = MONUMENTS[code];
      expect(m).toBeDefined();
      const img = card.querySelector("img")!;
      expect(img).not.toBeNull();
      // The photo has a real alt text that names the monument (the place is not repeated as visible text).
      expect(img.getAttribute("alt")).toBe(`${m.name}, ${m.city}`);
      expect(decodeURIComponent(img.getAttribute("src") ?? "")).toContain(m.image);
      const file = path.join(process.cwd(), "public", m.image);
      expect(existsSync(file)).toBe(true);
      expect(statSync(file).size).toBeLessThan(120 * 1024);
      seen.add(m.image);
      // The visible text is the greeting, the phrase and the language: no place line, no tag.
      expect(card).not.toHaveTextContent(m.name);
      expect(card).not.toHaveTextContent(m.city);
    }
    expect(seen.size).toBe(11);
  });

  it("shows no place line and no \"Ready voice or yours\" tag on the cards, only greeting, phrase and language", () => {
    const { container } = render(<Languages />);
    expect(container.textContent).not.toMatch(/Ready voice or yours/);
    expect(container.querySelector("figure svg")).toBeNull();
    const hindi = container.querySelector("figure[lang=hi]")!;
    expect(Array.from(hindi.querySelectorAll("figcaption > span")).map((x) => x.textContent)).toEqual(["Hindi", "हिन्दी"]);
    expect(hindi).toHaveTextContent("नमस्ते, आप कैसे हैं?");
    expect(hindi.querySelectorAll("p")).toHaveLength(1);
  });

  it("moves on its own and not with the page scroll, and falls back to a native scroller with arrows under reduced motion", () => {
    const { container, unmount } = render(<Languages />);
    const loop = container.querySelector(".s-loop")!;
    expect(loop.getAttribute("data-static")).toBeNull();
    expect(loop.hasAttribute("data-motion")).toBe(true);
    expect(loop.querySelector(".s-loop__track")).not.toBeNull();
    expect(container.querySelector("[data-pin]")).toBeNull();
    unmount();
    const prev = window.matchMedia;
    window.matchMedia = ((q: string) => ({ matches: /reduce/.test(q), media: q, addEventListener: jest.fn(), removeEventListener: jest.fn(), addListener: jest.fn(), removeListener: jest.fn() })) as unknown as typeof window.matchMedia;
    try {
      const r = render(<Languages />);
      expect(r.container.querySelector(".s-loop")!.hasAttribute("data-static")).toBe(true);
      expect(r.container.querySelectorAll("figure")).toHaveLength(11);
      expect(screen.getByRole("button", { name: "Next language" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Previous language" })).toBeInTheDocument();
    } finally {
      window.matchMedia = prev;
    }
  });

  it("credits every monument photo with an author and a reusable licence", () => {
    expect(MONUMENT_CREDITS).toHaveLength(11);
    for (const c of MONUMENT_CREDITS) {
      expect(c.author.trim().length).toBeGreaterThan(1);
      expect(c.author).not.toMatch(/[<>]/);
      expect(c.license).toMatch(/^(CC BY(-SA)? \d\.\d|CC0 1\.0|Public domain)$/);
      expect(c.licenseUrl).toMatch(/^https:\/\/(creativecommons\.org|en\.wikipedia\.org)\//);
      expect(c.source).toMatch(/^https:\/\/commons\.wikimedia\.org\/wiki\/File:/);
      expect(MONUMENTS[c.code].image).toBe(c.image);
    }
    // Every image on a card has a credit.
    const { container } = render(<Languages />);
    const imgs = Array.from(container.querySelectorAll("figure img")).map((i) => decodeURIComponent(i.getAttribute("src") ?? ""));
    for (const src of imgs) expect(MONUMENT_CREDITS.some((c) => src.includes(c.image))).toBe(true);
  });

  it("tells the consent story in four steps with the verify link and developer endpoints", () => {
    const { container } = render(<Consent />);
    expect(container.querySelector("section")).toHaveAttribute("id", "consent");
    expect(screen.getByRole("heading", { name: "Your voice. Your consent. Verifiable." })).toBeInTheDocument();
    expect(container.querySelectorAll("[data-cstep]")).toHaveLength(4);
    expect(screen.getByRole("link", { name: /open \/verify/i })).toHaveAttribute("href", "/verify");
    expect(screen.getByRole("link", { name: "/.well-known/bonclay-receipt-key" }).getAttribute("href")).toMatch(/\/\.well-known\/bonclay-receipt-key$/);
    expect(screen.getByRole("link", { name: "/api/withdrawals" }).getAttribute("href")).toMatch(/\/api\/withdrawals$/);
    // Without JS the final state is simply visible: nothing is hidden by the markup.
    expect(container.querySelector("[data-c-chip]")).not.toHaveAttribute("style");
  });

  it("covers the trust steps and both entry points in the pipeline", () => {
    const { container } = render(<Pipeline />);
    expect(container.querySelectorAll("ol > li")).toHaveLength(8);
    expect(screen.getByText("It says it is an AI")).toBeInTheDocument();
    expect(screen.getByText("Marked and signed")).toBeInTheDocument();
    expect(screen.getByText(/take a phone call/i)).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/\b(eleven|eight|six)\b/i);
  });

  it("marks the plain sections for the shared reveal", () => {
    // The stack strip is a marquee, painted on first paint: nothing in it waits for the reveal.
    expect(render(<StackStrip />).container.querySelector(".s-loop__track")).not.toBeNull();
    const { container } = render(<Attacks />);
    expect(container.querySelectorAll("li[data-reveal]")).toHaveLength(6);
  });
});

/** A stand-in IntersectionObserver that keeps its callbacks, so a test can say when a section comes near the screen. */
type FakeObserver = { cb: IntersectionObserverCallback; observed: Element[] };
const observers: FakeObserver[] = [];
class FakeIntersectionObserver {
  readonly root = null;
  readonly rootMargin = "";
  readonly thresholds = [];
  private record: FakeObserver;
  constructor(cb: IntersectionObserverCallback) {
    this.record = { cb, observed: [] };
    observers.push(this.record);
  }
  observe(el: Element) {
    this.record.observed.push(el);
  }
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}
/** Reports `el` as intersecting, to the observer that watches it. */
function nearScreen(el: Element) {
  const obs = observers.find((o) => o.observed.includes(el));
  if (!obs) throw new Error("no observer watches this element");
  act(() => obs.cb([{ isIntersecting: true, target: el } as IntersectionObserverEntry], {} as IntersectionObserver));
}

describe("language cards load their script faces near the screen", () => {
  const realIO = window.IntersectionObserver;
  beforeAll(() => {
    window.IntersectionObserver = FakeIntersectionObserver as unknown as typeof IntersectionObserver;
  });
  afterAll(() => {
    window.IntersectionObserver = realIO;
  });
  beforeEach(() => {
    observers.length = 0;
  });

  it("uses the page font for the greetings until the section nears the screen, then applies each script's face", () => {
    const { container } = render(<Languages />);
    const hi = SCRIPT_FONT.hi;
    const bn = SCRIPT_FONT.bn;
    expect(hi).toBeTruthy();
    const hindi = screen.getAllByText("नमस्ते")[0];
    const bengali = screen.getAllByText("নমস্কার")[0];
    expect(hindi.classList.contains(hi)).toBe(false);
    expect(bengali.classList.contains(bn)).toBe(false);

    nearScreen(container.querySelector("section#languages")!);

    expect(hindi.classList.contains(hi)).toBe(true);
    expect(bengali.classList.contains(bn)).toBe(true);
  });
});
