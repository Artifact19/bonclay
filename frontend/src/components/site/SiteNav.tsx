"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUpRight, Menu, X } from "lucide-react";
import { LayoutGroup } from "motion/react";
import { Magnetic, motion } from "./motion";
import { useLenis } from "./SmoothScroll";
import { Logo } from "@/components/Logo";
import { SITE_PAGES } from "./routes";
import { TourGuide, TourTriggerButton } from "@/components/TourGuide";

/** One link per page after Home (Home is the logo). The active page gets aria-current="page". */
const LINKS = SITE_PAGES.map((p) => [p.href, p.nav] as const);

const FOCUSABLE = "a[href], button:not([disabled]), [tabindex]:not([tabindex='-1'])";

/* The bar sits flat on the paper at the top of the page. After the first scroll it pops into a floating pill: paper at
   85% with a blur behind it, a hairline, a soft drop, and a short overshoot on the way in (the one bounce on the site).
   The mobile sheet keeps the full-width bar so it has an edge to hang from. */
const NAV_CSS = `
.site .nav-pill {
  max-width: 79rem; margin-inline: auto; padding-inline: 1rem 0.65rem;
  border-radius: 999px; border: 1.5px solid #141414;
  background: #E2DDD3;
  box-shadow: 0 2px 0 rgba(20, 20, 20, 0.08);
  transition: background-color .25s, border-color .25s, box-shadow .25s, padding .3s var(--s-ease), max-width .3s var(--s-ease);
}
.site .nav-pill[data-floating] {
  max-width: 72rem; margin-inline: auto; padding-inline: 0.9rem 0.55rem;
  border-radius: 999px; border: 1.5px solid #141414;
  background: rgba(226, 221, 211, 0.95); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  box-shadow: 0 10px 24px -10px rgba(20, 20, 20, 0.22);
  animation: nav-pop .38s var(--s-ease) both;
}
@keyframes nav-pop { 0% { opacity: 0; transform: translateY(calc(-100% - 12px)) scale(.96); } 75% { opacity: 1; transform: translateY(2px) scale(1.005); } 100% { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .site .nav-pill, .site .nav-pill[data-floating] { animation: none; transition: none; } }
`;

export function SiteNav() {
  const lenis = useLenis();
  const pathname = usePathname();
  const isCurrent = (href: string) => pathname === href || pathname === `${href}/`;
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);
  // A navigation (including Back) always closes the mobile sheet. Adjusted during render, not in an effect.
  const [seenPath, setSeenPath] = useState(pathname);
  if (seenPath !== pathname) {
    setSeenPath(pathname);
    setOpen(false);
  }
  const header = useRef<HTMLElement>(null);
  const toggle = useRef<HTMLButtonElement>(null);

  const close = useCallback((restoreFocus = false) => {
    setOpen(false);
    if (restoreFocus) toggle.current?.focus();
  }, []);

  // Floating pill after the first scroll.
  useEffect(() => {
    let raf = 0;
    const compute = () => {
      raf = 0;
      setScrolled(window.scrollY > 24);
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(compute);
    };
    compute();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  // Mobile sheet: lock page scroll, Escape closes, Tab stays inside the header, first link gets focus.
  useEffect(() => {
    if (!open) return;
    const html = document.documentElement;
    const prev = html.style.overflow;
    html.style.overflow = "hidden";
    lenis?.stop();
    // After a tick: the sheet is still visibility:hidden on the first frame of its transition, which blocks focus().
    const focusTimer = window.setTimeout(() => header.current?.querySelector<HTMLElement>("#site-menu a")?.focus(), 60);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        close(true);
        return;
      }
      if (e.key !== "Tab" || !header.current) return;
      const items = Array.from(header.current.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => el.getClientRects().length > 0);
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      const current = document.activeElement;
      if (e.shiftKey && current === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && current === last) {
        e.preventDefault();
        first.focus();
      } else if (!header.current.contains(current)) {
        e.preventDefault();
        first.focus();
      }
    };
    const onResize = () => {
      if (window.innerWidth >= 1024) close();
    };
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", onResize);
    return () => {
      window.clearTimeout(focusTimer);
      html.style.overflow = prev;
      lenis?.start();
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onResize);
    };
  }, [open, lenis, close]);

  const floating = scrolled && !open;

  return (
    <>
    <header
      ref={header}
      className={`fixed inset-x-0 top-0 z-50 px-[var(--s-gutter)] pt-3 lg:pt-4 ${open ? "border-b border-[var(--s-line)] bg-[var(--s-ground)]" : ""}`}
    >
      <style>{NAV_CSS}</style>
      <nav
        data-floating={floating ? "" : undefined}
        className="nav-pill flex h-14 items-center gap-3 xl:gap-5 data-[floating]:h-13"
        aria-label="Main"
      >
        <Link href="/" className="flex items-center pl-1" aria-label="BonClay home" aria-current={pathname === "/" ? "page" : undefined}>
          <Logo size={28} />
        </Link>
        <span aria-hidden className="hidden h-[1.5px] w-12 shrink-0 bg-[#141414]/65 xl:block" />
        <LayoutGroup id="site-nav">
          <ul className="hidden items-center gap-0.5 lg:flex" onMouseLeave={() => setHovered(null)}>
            {LINKS.map(([href, label]) => {
              const current = isCurrent(href);
              const active = hovered === href || (hovered === null && current);
              return (
                <li key={href} className="relative">
                  {active && (
                    <motion.span
                      layoutId="nav-hover"
                      aria-hidden
                      className="absolute inset-0 rounded-full border border-[#141414]/25 bg-[#FAF9F5]"
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    />
                  )}
                  <Link
                    href={href}
                    aria-current={current ? "page" : undefined}
                    onMouseEnter={() => setHovered(href)}
                    onFocus={() => setHovered(href)}
                    onBlur={() => setHovered(null)}
                    className={`relative inline-flex min-h-9 items-center whitespace-nowrap rounded-full px-3 text-[0.875rem] font-medium transition-colors hover:text-[#141414] ${
                      current ? "font-semibold text-[#141414]" : "text-[#141414]/80"
                    }`}
                  >
                    {label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </LayoutGroup>
        <span aria-hidden className="ml-auto hidden h-[1.5px] w-12 shrink-0 bg-[#141414]/65 xl:block" />
        <div className="ml-auto flex items-center gap-2 xl:ml-0">
          <TourTriggerButton onClick={() => setTourOpen(true)} label="Tour guide" />
          {/* prefetch={false}: a prefetch would download the app's chunks (sign-in and Supabase client) to a visitor who never clicks. */}
          <Link
            href="/login"
            prefetch={false}
            className="hidden h-9 items-center whitespace-nowrap rounded-full px-3 text-[0.875rem] font-medium text-[#141414] transition-colors hover:bg-[#FAF9F5]/70 xl:inline-flex"
          >
            Sign in
          </Link>
          <Magnetic strength={0.2}>
            <Link
              href="/dashboard"
              prefetch={false}
              className="group inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full border-[1.5px] border-[#141414] bg-[#F3F59E] pl-4 pr-3.5 text-[0.875rem] font-semibold text-[#141414] transition-all hover:-translate-y-0.5 hover:bg-[#eef17a] hover:shadow-[0_3px_0_#141414]"
            >
              Open studio
              <ArrowUpRight className="h-4 w-4 transition-transform duration-200 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" aria-hidden />
            </Link>
          </Magnetic>
          <button
            ref={toggle}
            type="button"
            className="grid size-10 place-items-center rounded-full text-[#141414] transition-colors hover:bg-[#FAF9F5]/70 lg:hidden"
            aria-expanded={open}
            aria-controls="site-menu"
            aria-label={open ? "Close menu" : "Open menu"}
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X className="h-5 w-5" aria-hidden /> : <Menu className="h-5 w-5" aria-hidden />}
          </button>
        </div>
      </nav>

      {/* Mobile and tablet sheet. Absolute (not fixed) so the header's backdrop blur cannot become its containing block. */}
      <div
        id="site-menu"
        role="dialog"
        aria-modal="true"
        aria-label="Site menu"
        inert={!open}
        className={`absolute inset-x-0 top-full h-[calc(100dvh-4rem)] overflow-y-auto overscroll-contain border-t border-[var(--s-line)] bg-[var(--s-ground)] px-[var(--s-gutter)] pb-10 pt-4 transition-[opacity,visibility,transform] duration-300 motion-reduce:transition-none lg:hidden ${
          open ? "visible translate-y-0 opacity-100" : "invisible -translate-y-2 opacity-0"
        }`}
        data-lenis-prevent
      >
        <ul>
          {LINKS.map(([href, label]) => (
            <li key={href} className="border-b border-[var(--s-line)]">
              <Link
                href={href}
                onClick={() => close()}
                aria-current={isCurrent(href) ? "page" : undefined}
                className="flex min-h-14 items-center justify-between s-h3 py-2 text-[var(--s-ink)] aria-[current=page]:text-[var(--s-bronze)]"
              >
                {label}
                <ArrowUpRight className="h-5 w-5 text-[var(--s-mute)]" aria-hidden />
              </Link>
            </li>
          ))}
        </ul>
        <div className="mt-8 flex flex-col gap-3 sm:flex-row">
          <Link href="/dashboard" prefetch={false} onClick={() => close()} className="s-btn s-btn--primary">
            Open studio
          </Link>
          <Link href="/login" prefetch={false} onClick={() => close()} className="s-btn s-btn--secondary">
            Sign in
          </Link>
        </div>
      </div>
    </header>
    <TourGuide mode="site" open={tourOpen} onClose={() => setTourOpen(false)} />
    </>
  );
}
