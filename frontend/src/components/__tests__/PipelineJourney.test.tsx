import { act, fireEvent, render, screen } from "@testing-library/react";
import { renderToString } from "react-dom/server";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { PipelineJourney } from "@/components/site/PipelineJourney";
import { Pipeline } from "@/components/site/Sections";
import { EXAMPLE, MARKERS, STEPS, TIMING, sec } from "@/components/site/pipelineData";
import { MEASURED_SPEED } from "@/components/site/proofData";

jest.mock("@/components/Logo", () => ({ Logo: () => <span>BonClay</span> }));
jest.mock("@/components/site/fonts", () => ({ SCRIPT_FONT: new Proxy({}, { get: () => "" }) }));
jest.mock("@/components/site/SmoothScroll", () => ({ useLenis: () => null }));

// A timeline that records what the component asks of it, so the tests do not depend on layout or a real ticker.
const mockCallbacks: Record<string, () => void> = {};
const mockTl: Record<string, jest.Mock> = {};
["fromTo", "to", "set", "pause"].forEach((m) => {
  mockTl[m] = jest.fn(() => mockTl);
});
mockTl.eventCallback = jest.fn((name: string, fn: () => void) => {
  mockCallbacks[name] = fn;
  return mockTl;
});
mockTl.play = jest.fn(() => mockTl);
mockTl.time = jest.fn((v?: number) => (v === undefined ? 0 : mockTl));
mockTl.duration = jest.fn(() => 5.7);

jest.mock("gsap", () => {
  const gsap = {
    registerPlugin: jest.fn(),
    context: (fn: () => void) => {
      fn();
      return { revert: jest.fn() };
    },
    timeline: () => mockTl,
    fromTo: jest.fn(),
    from: jest.fn(),
    set: jest.fn(),
  };
  return { gsap, default: gsap };
});
jest.mock("gsap/ScrollTrigger", () => ({
  ScrollTrigger: {
    create: jest.fn(() => ({ progress: 0, start: 0, end: 1000 })),
    batch: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
  },
}));

/** `motion`: prefers-reduced-motion is no-preference. `wide`: at least 64rem wide and 45rem tall. */
function media(motion: boolean, wide: boolean) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    configurable: true,
    value: (q: string) => ({
      matches: q.includes("min-width") ? motion && wide : q.includes("reduce)") ? !motion : motion,
      media: q,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      addListener: jest.fn(),
      removeListener: jest.fn(),
    }),
  });
}

beforeEach(() => {
  Object.keys(mockCallbacks).forEach((k) => delete mockCallbacks[k]);
  Object.values(mockTl).forEach((m) => m.mockClear());
  (ScrollTrigger.create as jest.Mock).mockClear();
  (ScrollTrigger.batch as jest.Mock).mockClear();
  media(true, true);
});

describe("PipelineJourney", () => {
  it("renders all eight step names on the server, with no script and no replay button", () => {
    const html = renderToString(<PipelineJourney />);
    for (const s of STEPS) expect(html).toContain(s.name);
    expect(STEPS).toHaveLength(8);
    expect(html).toContain('data-mode="static"');
    expect(html).not.toContain("Replay at real speed");
    expect(html).toContain("From your voice to a signed answer.");
    // The full text of every step is in the markup, not only the current one.
    for (const s of STEPS) expect(html).toContain(s.body.slice(0, 40).replace(/'/g, "&#x27;"));
    expect(html).not.toContain("—");
  });

  it("is the Pipeline section the site uses, as a labelled section with the #how anchor", () => {
    expect(Pipeline).toBe(PipelineJourney);
    const { container } = render(<PipelineJourney />);
    const section = container.querySelector("section#how");
    expect(section).toHaveAttribute("aria-labelledby", "how-title");
    expect(section?.className).toContain("s-container");
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent("From your voice to a signed answer.");
    // The heading and its first paragraph are the page's LCP on phones: they are never hidden for a scroll reveal.
    expect(screen.getByRole("heading", { level: 2 })).not.toHaveAttribute("data-reveal");
    expect(container.querySelector("section#how > p.lede")).not.toHaveAttribute("data-reveal");
    const list = screen.getByRole("list", { name: "The 8 steps" });
    expect(list.tagName).toBe("OL");
    expect(container.querySelectorAll("ol > li")).toHaveLength(8);
  });

  it("takes its numbers from the data module, which agrees with the Proof page's table", () => {
    const { container } = render(<PipelineJourney />);
    const text = container.textContent ?? "";
    for (const n of [TIMING.firstAudio, TIMING.firstAnswer, TIMING.asr, TIMING.search, TIMING.voice, TIMING.fullReply]) {
      expect(text).toContain(sec(n));
    }
    for (const m of MARKERS) expect(container.querySelector(`[data-m="${m.id}"]`)?.textContent).toContain(m.value);
    // proofData.ts quotes the same table of docs/RESULTS.md.
    const proof = MEASURED_SPEED.map((s) => s.value);
    expect(proof).toContain(sec(TIMING.firstAudio));
    expect(proof).toContain(sec(TIMING.firstAnswer));
    expect(proof).toContain(sec(TIMING.fullReply));
    expect(MEASURED_SPEED.find((s) => s.value === sec(TIMING.firstAnswer))?.note).toContain("3.58 s");
    expect(text).toContain("Stages overlap: speech starts while the reply is still being written.");
    expect(text).toContain("30 spoken questions");
  });

  it("describes the current reply stack and not the old Llama-on-Groq one", () => {
    const { container } = render(<PipelineJourney />);
    expect(container.textContent).toMatch(/Qwen3 model on Groq/);
    expect(container.textContent).not.toMatch(/Llama runs on Groq/);
    expect(container.textContent).toMatch(/Kokoro/);
    expect(container.textContent).toMatch(/Qwen3-TTS/);
    expect(container.textContent).toMatch(/Exotel/);
  });

  it("keeps the example consistent and readable to screen readers", () => {
    const { container } = render(<PipelineJourney />);
    const sr = container.querySelector("p.sr-only");
    expect(sr?.textContent).toContain(EXAMPLE.question);
    expect(sr?.textContent).toContain(EXAMPLE.reply);
    expect(sr?.textContent).toContain(EXAMPLE.passage);
    expect(sr).toHaveAttribute("lang", "hi-Latn");
    // The pictures are decoration.
    container.querySelectorAll(".pj-mini").forEach((m) => expect(m).toHaveAttribute("aria-hidden", "true"));
  });

  it("under reduced motion shows the final state: all steps listed, ruler drawn, nothing current, no replay", () => {
    media(false, true);
    const { container } = render(<PipelineJourney />);
    expect(container.querySelector(".pj")).toHaveAttribute("data-mode", "static");
    expect(container.querySelector(".pj-stage")).toBeNull();
    expect(screen.queryByRole("button", { name: /replay at real speed/i })).toBeNull();
    expect(container.querySelectorAll("[data-m]")).toHaveLength(MARKERS.length);
    container.querySelectorAll("[data-m]").forEach((m) => expect(m).toHaveAttribute("data-on", "true"));
    expect((container.querySelector("[data-fill]") as HTMLElement).style.transform).toBe(`scaleX(${TIMING.fullReply / 3.2})`);
    expect(container.querySelector("[data-clock]")).toHaveTextContent(sec(TIMING.fullReply));
    expect(container.querySelector("[aria-current]")).toBeNull();
    // Nothing was handed to GSAP.
    expect(mockTl.fromTo).not.toHaveBeenCalled();
    // The input toggle still works (it only changes the icon and the caption).
    fireEvent.click(screen.getByRole("button", { name: "Phone call" }));
    expect(screen.getAllByText(/8 kHz, through Exotel/).length).toBeGreaterThan(0);
  });

  it("marks a step as current when it is clicked", () => {
    media(false, true);
    render(<PipelineJourney />);
    const find = screen.getByRole("button", { name: /It finds the answer/ });
    expect(find).not.toHaveAttribute("aria-current");
    fireEvent.click(find);
    expect(find).toHaveAttribute("aria-current", "step");
    expect(find.closest("li")).toHaveAttribute("data-current", "true");
    fireEvent.click(screen.getByRole("button", { name: /It speaks/ }));
    expect(find).not.toHaveAttribute("aria-current");
    expect(find.closest("li")).toHaveAttribute("data-state", "done");
  });

  it("builds the stage and a scroll trigger when pinned, and hides the stage from assistive technology", () => {
    const { container } = render(<PipelineJourney />);
    expect(container.querySelector(".pj")).toHaveAttribute("data-mode", "pinned");
    expect(container.querySelector(".pj-stage")).toHaveAttribute("aria-hidden", "true");
    expect(mockTl.fromTo).toHaveBeenCalled();
    expect(ScrollTrigger.create).toHaveBeenCalledTimes(1);
  });

  it("toggles the replay button and its status text, then reports Done", () => {
    render(<PipelineJourney />);
    const btn = screen.getByRole("button", { name: /replay at real speed/i });
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("");
    expect(btn).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(btn);
    expect(btn).toHaveAttribute("aria-pressed", "true");
    expect(status).toHaveTextContent("Playing at real speed");
    expect(mockTl.play).toHaveBeenCalledWith(0);

    act(() => mockCallbacks.onComplete());
    expect(btn).toHaveAttribute("aria-pressed", "false");
    expect(status).toHaveTextContent("Done");

    // Pressing it again while it plays stops it and returns to scroll control.
    fireEvent.click(btn);
    expect(status).toHaveTextContent("Playing at real speed");
    fireEvent.click(btn);
    expect(status).toHaveTextContent("");
    expect(mockTl.pause).toHaveBeenCalled();
  });

  it("offers the same replay on phones, where the cards stack and nothing is pinned", () => {
    media(true, false);
    const { container } = render(<PipelineJourney />);
    expect(container.querySelector(".pj")).toHaveAttribute("data-mode", "stack");
    expect(container.querySelector(".pj-stage")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /replay at real speed/i }));
    expect(screen.getByRole("status")).toHaveTextContent("Playing at real speed");
    expect(ScrollTrigger.batch).toHaveBeenCalled();
    expect(ScrollTrigger.create).not.toHaveBeenCalled();
  });
});
