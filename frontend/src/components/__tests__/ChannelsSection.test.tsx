import { render, screen, within } from "@testing-library/react";
import { Channels } from "@/components/site/ChannelsSection";
import { CHANNELS, DOCS_BASE } from "@/components/site/channelsData";

jest.mock("@/components/site/fonts", () => ({ SCRIPT_FONT: new Proxy({}, { get: () => "" }) }));
jest.mock("@/components/site/SmoothScroll", () => ({ useLenis: () => null }));
jest.mock("gsap", () => {
  const mm = { add: jest.fn(), revert: jest.fn() };
  const gsap = { registerPlugin: jest.fn(), matchMedia: () => mm, context: jest.fn(), utils: { selector: () => () => [] } };
  return { gsap, default: gsap };
});
jest.mock("gsap/ScrollTrigger", () => ({ ScrollTrigger: { create: jest.fn() } }));

beforeAll(() => {
  // Reduced motion on: every mock must render complete, with no timeline.
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (q: string) => ({ matches: q.includes("reduce"), media: q, addEventListener: jest.fn(), removeEventListener: jest.fn(), addListener: jest.fn(), removeListener: jest.fn() }),
  });
});

describe("Channels section", () => {
  it("has an id, a labelled heading and four channels, each with a rail button", () => {
    const { container } = render(<Channels />);
    const section = container.querySelector("section#channels")!;
    expect(section.getAttribute("aria-labelledby")).toBe("channels-h");
    expect(screen.getByRole("heading", { level: 2, name: "One agent, every channel." })).toBeInTheDocument();
    const rail = container.querySelector<HTMLElement>("nav[aria-label=Channels]")!;
    expect(within(rail).getAllByRole("button", { hidden: true })).toHaveLength(4);
    expect(container.querySelectorAll("article")).toHaveLength(4);
  });

  it("labels every mock as an example and keeps names and numbers fictional", () => {
    const { container } = render(<Channels />);
    expect(screen.getByText(/example conversation, not a real caller/i)).toBeInTheDocument();
    expect(screen.getByText(/example, made-up names and times/i)).toBeInTheDocument();
    expect(screen.getByText(/example page/i)).toBeInTheDocument();
    expect(screen.getByText(/example conversation, not a real customer/i)).toBeInTheDocument();
    expect(container.textContent).toContain("••••4821");
    expect(container.textContent).not.toMatch(/\+91|\b\d{10}\b/);
  });

  it("states honestly how far each channel was tested, and does not link to the private repository", () => {
    const { container } = render(<Channels />);
    expect(screen.getAllByText(/Beta: not yet live-tested/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Tested on a live call\./)).toBeInTheDocument();
    expect(screen.getByText(/Tested against a fake WhatsApp API/)).toBeInTheDocument();
    const hrefs = Array.from(container.querySelectorAll("a")).map((a) => a.getAttribute("href"));
    expect(hrefs.filter((h) => h?.includes("github.com"))).toEqual([]);
    for (const c of CHANNELS) expect(screen.getByText(`${c.docLabel}: shared on request`)).toBeInTheDocument();
    expect(DOCS_BASE).toBe("https://github.com/Artifact19/bonclay/blob/main/docs/");
  });

  it("shows the AI disclosure and the barge-in in the phone example", () => {
    render(<Channels />);
    expect(screen.getByText("AI disclosure, spoken first")).toBeInTheDocument();
    expect(screen.getByText("Talked over: the agent stops")).toBeInTheDocument();
  });
});
