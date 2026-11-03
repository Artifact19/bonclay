import { act, render } from "@testing-library/react";
import { SmoothScroll } from "@/components/site/SmoothScroll";

/**
 * Regression: on a direct load of a page whose frames are not being rendered (a background or occluded tab) the GSAP
 * reveals never advanced and the page stayed blank. The frame watchdog in SmoothScroll must then show everything.
 * Real GSAP core runs here; Lenis (needs layout), ScrollTrigger (an ES module Jest cannot load) and the page's frame
 * source are replaced.
 */
jest.mock("next/navigation", () => ({ usePathname: () => "/run-it" }));
jest.mock("lenis", () => ({
  __esModule: true,
  default: class {
    on() {}
    raf() {}
    destroy() {}
    stop() {}
    start() {}
    scrollTo() {}
  },
}));

jest.mock("gsap/ScrollTrigger", () => ({
  ScrollTrigger: {
    config: jest.fn(),
    batch: jest.fn(),
    create: jest.fn(),
    refresh: jest.fn(),
    update: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
  },
}));

const root = document.documentElement;
const realRaf = window.requestAnimationFrame;
const realCaf = window.cancelAnimationFrame;

beforeEach(() => {
  jest.useFakeTimers();
  root.classList.remove("reveal-on");
  window.matchMedia = ((q: string) => ({ matches: q.includes("no-preference"), media: q, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent: () => false })) as unknown as typeof window.matchMedia;
  window.scrollTo = jest.fn() as unknown as typeof window.scrollTo;
});

afterEach(() => {
  jest.useRealTimers();
  window.requestAnimationFrame = realRaf;
  window.cancelAnimationFrame = realCaf;
  root.classList.remove("reveal-on");
});

function mount() {
  return render(
    <SmoothScroll>
      <h1 data-reveal>Can you run it?</h1>
      <p data-reveal>Intro</p>
    </SmoothScroll>,
  );
}

describe("SmoothScroll reveal watchdog", () => {
  it("shows the page when no animation frames are delivered (background or occluded tab)", () => {
    window.requestAnimationFrame = (() => 1) as unknown as typeof window.requestAnimationFrame; // frames never fire
    window.cancelAnimationFrame = jest.fn();
    const { getByRole } = mount();
    expect(root.classList.contains("reveal-on")).toBe(true); // hidden until revealed, as CSS expects
    act(() => {
      jest.advanceTimersByTime(1600);
    });
    expect(root.classList.contains("reveal-on")).toBe(false);
    const h1 = getByRole("heading", { level: 1 });
    expect(h1.style.opacity).not.toBe("0");
  });

  it("leaves the animated reveals alone when frames are arriving normally", () => {
    window.requestAnimationFrame = ((cb: FrameRequestCallback) => window.setTimeout(() => cb(performance.now()), 16)) as unknown as typeof window.requestAnimationFrame;
    window.cancelAnimationFrame = ((id: number) => window.clearTimeout(id)) as unknown as typeof window.cancelAnimationFrame;
    mount();
    act(() => {
      jest.advanceTimersByTime(1600);
    });
    expect(root.classList.contains("reveal-on")).toBe(true);
  });
});
