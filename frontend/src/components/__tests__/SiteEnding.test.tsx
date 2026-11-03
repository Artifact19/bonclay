import { render, screen, within } from "@testing-library/react";
import { Faq, FinalCta, Footer } from "@/components/site/Ending";
import { MONUMENT_CREDITS } from "@/components/site/monuments";

jest.mock("@/components/Logo", () => ({ Logo: () => <span>BonClay</span> }));

describe("landing ending", () => {
  it("renders the FAQ as native details with FAQPage JSON-LD that matches", () => {
    const { container } = render(<Faq />);
    const items = container.querySelectorAll("details");
    expect(items.length).toBeGreaterThanOrEqual(8);
    expect(screen.getByRole("heading", { name: /straight answers/i })).toBeInTheDocument();
    const ld = container.querySelector('script[type="application/ld+json"]');
    const data = JSON.parse(ld!.textContent!);
    expect(data["@type"]).toBe("FAQPage");
    expect(data.mainEntity).toHaveLength(items.length);
    expect(screen.getByRole("link", { name: "Can you run it?" })).toHaveAttribute("href", "/run-it");
  });

  it("offers the three closing actions", () => {
    render(<FinalCta />);
    expect(screen.getByRole("link", { name: /open studio/i })).toHaveAttribute("href", "/dashboard");
    expect(screen.getByRole("link", { name: /verify a clip/i })).toHaveAttribute("href", "/verify");
    expect(screen.getByRole("link", { name: /run it on your laptop/i })).toHaveAttribute("href", "/run-it");
  });

  it("links every page, Studio, Verify and Dashboard from the footer, and nothing to the private repository", () => {
    render(<Footer />);
    const nav = screen.getByRole("navigation", { name: "Footer" });
    const hrefs = within(nav).getAllByRole("link").map((a) => a.getAttribute("href"));
    for (const h of ["/", "/how-it-works", "/channels", "/proof", "/run-it", "/faq", "/dashboard/studio", "/verify", "/dashboard"]) expect(hrefs).toContain(h);
    expect(hrefs.filter((h) => h?.includes("github.com"))).toEqual([]);
    expect(within(nav).getByRole("link", { name: "Can you run it?" })).toHaveAttribute("href", "/run-it");
  });

  it("keeps the required model notices and the receipt key link in the footer", () => {
    render(<Footer />);
    expect(screen.getByText(/Voice cloning in Indian languages by DhVaani from ARTPARK-IISc/)).toBeInTheDocument();
    expect(screen.getByText(/Built with Indic-Speak from Bodhan AI/)).toBeInTheDocument();
    expect(screen.getByText(/Built with Llama/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /receipt public key/i }).getAttribute("href")).toMatch(/\/\.well-known\/bonclay-receipt-key$/);
  });

  it("credits every monument photo in the footer with links to the file page and the licence", () => {
    const { container } = render(<Footer />);
    const details = Array.from(container.querySelectorAll("details")).find((d) => /photo credits/i.test(d.querySelector("summary")?.textContent ?? ""));
    expect(details).toBeDefined();
    const items = details!.querySelectorAll("li");
    expect(items).toHaveLength(MONUMENT_CREDITS.length);
    MONUMENT_CREDITS.forEach((c, i) => {
      expect(items[i]).toHaveTextContent(`${c.name}`);
      expect(items[i]).toHaveTextContent(c.author);
      expect(items[i].querySelector(`a[href="${c.source}"]`)).not.toBeNull();
      const lic = items[i].querySelector(`a[href="${c.licenseUrl}"]`);
      expect(lic).not.toBeNull();
      expect(lic).toHaveTextContent(c.license);
    });
  });
});
