import { render, screen, within } from "@testing-library/react";
import { WhySection } from "@/components/site/WhySection";
import { BEATS, USE_CASES } from "@/components/site/whyData";

jest.mock("gsap", () => ({
  gsap: { registerPlugin: jest.fn(), matchMedia: () => ({ add: jest.fn(), revert: jest.fn() }), fromTo: jest.fn() },
}));
jest.mock("gsap/ScrollTrigger", () => ({ ScrollTrigger: {} }));

describe("WhySection", () => {
  it("is a labelled section with the #why anchor and three beats", () => {
    const { container } = render(<WhySection />);
    const section = container.querySelector("section#why");
    expect(section).toHaveAttribute("aria-labelledby", "why-heading");
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent("Built for how India actually talks.");
    expect(BEATS).toHaveLength(3);
  });

  it("gives every statistic a real, dated, external citation link", () => {
    render(<WhySection />);
    for (const beat of BEATS) {
      for (const stat of beat.stats) {
        expect(stat.cite).toMatch(/, (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) 20\d\d$/);
        expect(stat.href).toMatch(/^https:\/\//);
      }
    }
    const lists = screen.getAllByRole("list");
    expect(lists.length).toBeGreaterThanOrEqual(4);
    const cites = screen.getAllByRole("link", { name: /\b20\d\d\b/ });
    expect(cites.length).toBeGreaterThanOrEqual(8);
    for (const a of cites) {
      expect(a).toHaveAttribute("target", "_blank");
      expect(a.getAttribute("rel")).toContain("noopener");
    }
  });

  it("lists four use cases with working links", () => {
    render(<WhySection />);
    expect(USE_CASES).toHaveLength(4);
    const who = screen.getByRole("list", { name: "Who it is for" });
    expect(within(who).getAllByRole("link")).toHaveLength(4);
  });

  it("does not call the project open source", () => {
    const { container } = render(<WhySection />);
    expect(container.textContent?.toLowerCase()).not.toContain("open source");
    expect(container.textContent?.toLowerCase()).not.toContain("open-source");
  });

  it("has a short version for Home: the same three problems and answers, no statistics, and a pointer to the sourced numbers", () => {
    const { container } = render(<WhySection compact />);
    expect(container.querySelector("section#why")).toHaveAttribute("aria-labelledby", "why-heading");
    const list = screen.getByRole("list", { name: "Three problems BonClay answers" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(BEATS.length);
    for (const b of BEATS) {
      expect(container.textContent).toContain(b.title);
      expect(container.textContent).toContain(b.answer);
    }
    expect(container.querySelector("a[target=_blank]")).toBeNull();
    expect(container.querySelector("[data-countup]")).toBeNull();
    expect(screen.getByRole("link", { name: "Proof page" })).toHaveAttribute("href", "/proof#why-evidence");
    expect(screen.queryByRole("list", { name: "Who it is for" })).toBeNull();
  });

  it("takes another id for the full version on the Proof page", () => {
    const { container } = render(<WhySection id="why-evidence" />);
    expect(container.querySelector("section#why-evidence")).not.toBeNull();
    expect(container.querySelector("section#why")).toBeNull();
  });
});
