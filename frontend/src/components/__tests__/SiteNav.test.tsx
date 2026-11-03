import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { SiteNav } from "@/components/site/SiteNav";
import { scrollToHash } from "@/components/site/SmoothScroll";

let mockPath = "/";
jest.mock("next/navigation", () => ({ usePathname: () => mockPath, useRouter: () => ({ replace: jest.fn(), push: jest.fn() }) }));
jest.mock("gsap", () => ({ gsap: { registerPlugin() {}, matchMedia: () => ({ add() {}, revert() {} }) } }));
jest.mock("gsap/ScrollTrigger", () => ({ ScrollTrigger: { config() {} } }));
jest.mock("lenis", () => ({ __esModule: true, default: class {} }));
jest.mock("@/components/Logo", () => ({ Logo: () => <span>BonClay</span> }));

beforeAll(() => {
  window.matchMedia = ((q: string) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent: () => false })) as unknown as typeof window.matchMedia;
  window.scrollTo = jest.fn() as unknown as typeof window.scrollTo;
});

beforeEach(() => {
  mockPath = "/";
});

const ROUTES: Array<[string, string]> = [
  ["/how-it-works", "How it works"],
  ["/channels", "Channels"],
  ["/proof", "Proof"],
  ["/run-it", "Can you run it?"],
  ["/faq", "FAQ"],
];

describe("SiteNav", () => {
  it("links to the five pages by route, not by in-page anchor", () => {
    render(<SiteNav />);
    const bar = within(screen.getByRole("navigation", { name: "Main" }));
    for (const [href, label] of ROUTES) expect(bar.getByRole("link", { name: label })).toHaveAttribute("href", href);
    const hrefs = screen.getAllByRole("link").map((a) => a.getAttribute("href"));
    expect(hrefs.filter((h) => h?.startsWith("#"))).toEqual([]);
    expect(bar.getByRole("link", { name: /open studio/i })).toHaveAttribute("href", "/dashboard");
    expect(bar.getByRole("link", { name: "BonClay home" })).toHaveAttribute("href", "/");
  });

  it.each(ROUTES)("marks only %s as the current page", (href, label) => {
    mockPath = href;
    render(<SiteNav />);
    const bar = screen.getByRole("navigation", { name: "Main" });
    const current = Array.from(bar.querySelectorAll("[aria-current]"));
    expect(current).toHaveLength(1);
    expect(current[0]).toHaveAttribute("aria-current", "page");
    expect(current[0].textContent).toBe(label);
    // the mobile sheet agrees
    const sheet = document.querySelector("#site-menu")!;
    const sheetCurrent = Array.from(sheet.querySelectorAll("[aria-current]"));
    expect(sheetCurrent.map((e) => e.textContent)).toEqual([label]);
  });

  it("marks the logo, and no page link, on Home", () => {
    render(<SiteNav />);
    expect(screen.getByRole("link", { name: "BonClay home" })).toHaveAttribute("aria-current", "page");
    expect(document.querySelectorAll("ul [aria-current]")).toHaveLength(0);
  });

  it("opens and closes the mobile menu with aria-expanded, Escape and link click", () => {
    render(<SiteNav />);
    const btn = screen.getByRole("button", { name: /open menu/i });
    expect(btn).toHaveAttribute("aria-expanded", "false");
    expect(document.querySelector("#site-menu")).toHaveAttribute("inert");
    fireEvent.click(btn);
    expect(screen.getByRole("button", { name: /close menu/i })).toHaveAttribute("aria-expanded", "true");
    expect(document.querySelector("#site-menu")).not.toHaveAttribute("inert");
    act(() => {
      fireEvent.keyDown(document, { key: "Escape" });
    });
    expect(screen.getByRole("button", { name: /open menu/i })).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(screen.getByRole("button", { name: /open menu/i }));
    const sheetLink = document.querySelector<HTMLAnchorElement>('#site-menu a[href="/faq"]')!;
    fireEvent.click(sheetLink);
    expect(screen.getByRole("button", { name: /open menu/i })).toHaveAttribute("aria-expanded", "false");
  });

  it("lists every page in the mobile sheet and keeps Tab inside the header while it is open", () => {
    // jsdom has no layout, so every element reports no boxes; the trap only considers elements that have one.
    const rects = jest.spyOn(HTMLElement.prototype, "getClientRects").mockReturnValue([{}] as unknown as DOMRectList);
    render(<SiteNav />);
    fireEvent.click(screen.getByRole("button", { name: /open menu/i }));
    const sheet = document.querySelector<HTMLElement>("#site-menu")!;
    expect(within(sheet).getAllByRole("link").map((a) => a.getAttribute("href"))).toEqual(expect.arrayContaining(ROUTES.map(([h]) => h)));
    const links = Array.from(document.querySelectorAll<HTMLElement>("header a, header button"));
    const last = links[links.length - 1];
    last.focus();
    const ev = new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true });
    act(() => {
      document.dispatchEvent(ev);
    });
    expect(ev.defaultPrevented).toBe(true); // wrapped to the first focusable item instead of leaving the header
    expect(document.documentElement.style.overflow).toBe("hidden");
    act(() => {
      fireEvent.keyDown(document, { key: "Escape" });
    });
    expect(document.documentElement.style.overflow).not.toBe("hidden");
    rects.mockRestore();
  });

  it("closes the mobile sheet when the route changes (Back button)", () => {
    const { rerender } = render(<SiteNav />);
    fireEvent.click(screen.getByRole("button", { name: /open menu/i }));
    expect(screen.getByRole("button", { name: /close menu/i })).toBeInTheDocument();
    mockPath = "/proof";
    rerender(<SiteNav />);
    expect(screen.getByRole("button", { name: /open menu/i })).toHaveAttribute("aria-expanded", "false");
  });
});

describe("scrollToHash", () => {
  it("ignores missing ids without throwing and does not touch the URL", () => {
    expect(scrollToHash("#nope", null)).toBe(false);
    expect(window.location.hash).toBe("");
  });

  it("scrolls natively, updates the hash and focuses the section", () => {
    const sec = document.createElement("section");
    sec.id = "faq";
    document.body.appendChild(sec);
    expect(scrollToHash("#faq", null)).toBe(true);
    expect(window.scrollTo).toHaveBeenCalled();
    expect(window.location.hash).toBe("#faq");
    sec.remove();
  });
});
