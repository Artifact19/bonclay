"use client";

import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import Lenis from "lenis";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);
ScrollTrigger.config({ ignoreMobileResize: true });

/** Class on <html> that hides [data-reveal] / hero elements until GSAP shows them. It is added here (or by the inline
 *  script in the root layout, before first paint) and never under reduced motion, so no-JS visitors see everything. */
const REVEAL_CLASS = "reveal-on";
/** Fixed header height plus breathing room. */
const NAV_OFFSET = 72;
/** Frame watchdog (see SmoothScroll): fewer than MIN_FRAMES animation frames in WATCHDOG_MS means the page is not being
 *  rendered, so reveals are shown without animating. 1.5 s at a healthy 60 fps is about 90 frames; 12 is under 10 fps. */
const WATCHDOG_MS = 1500;
const MIN_FRAMES = 12;

const LenisContext = createContext<Lenis | null>(null);
export const useLenis = () => useContext(LenisContext);

const prefersReduced = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Scroll to `#id` (Lenis when running, native otherwise), update the URL hash and move focus there for screen readers.
 *  Returns false, without throwing, when the id does not exist (yet). */
export function scrollToHash(hash: string, lenis: Lenis | null, opts: { push?: boolean; immediate?: boolean } = {}): boolean {
  if (typeof document === "undefined" || hash.length < 2 || hash[0] !== "#") return false;
  let id = hash.slice(1);
  try {
    id = decodeURIComponent(id);
  } catch {
    /* keep raw */
  }
  const el = document.getElementById(id);
  if (!el) return false;
  const reduced = prefersReduced();
  if (lenis && !reduced) {
    lenis.scrollTo(el, { offset: -NAV_OFFSET, immediate: opts.immediate, force: true });
  } else {
    const top = el.getBoundingClientRect().top + window.scrollY - NAV_OFFSET;
    window.scrollTo({ top, behavior: reduced || opts.immediate ? "auto" : "smooth" });
  }
  if (opts.push !== false && window.location.hash !== hash) {
    try {
      window.history.pushState(null, "", hash);
    } catch {
      /* some embedded browsers forbid it */
    }
  }
  if (!el.hasAttribute("tabindex") && !/^(A|BUTTON|INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) el.setAttribute("tabindex", "-1");
  el.focus({ preventScroll: true });
  return true;
}

/** Lenis drives the page scroll and GSAP's ticker drives Lenis, so every ScrollTrigger reads the same smoothed
 *  position. This component also owns the page's single reveal vocabulary:
 *    [data-reveal]         y 24 px + fade, 0.8 s expo.out, 70 ms stagger, once
 *    [data-reveal="media"] scale .96 -> 1 + y 20 px + fade, 1.0 s, once
 *    [data-countup]        animates the first number in the element's text once, final text stays in the markup
 *  Everything lives in gsap.matchMedia, so it is torn down and shown in its final state the moment the visitor turns
 *  reduced motion on, and set up again if they turn it off.
 *
 *  It lives in the shared layout of the marketing pages, which stays mounted while the visitor moves between them, so the
 *  whole setup re-runs on every pathname change: it finds the new page's [data-reveal] elements, rebuilds the triggers,
 *  and the previous page's are killed first. A page change also resets the scroll to the top (unless the URL has a hash). */
export function SmoothScroll({ children }: { children: ReactNode }) {
  const [lenis, setLenis] = useState<Lenis | null>(null);
  const lenisRef = useRef<Lenis | null>(null);
  const bar = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const lastPath = useRef<string | null>(null);

  useLayoutEffect(() => {
    const root = document.documentElement;
    // A real page change (not the first mount, where the browser may be restoring a scroll position): start at the top,
    // before Lenis reads the position. A hash is honoured below once fonts have settled.
    const navigated = lastPath.current !== null && lastPath.current !== pathname;
    lastPath.current = pathname;
    if (navigated && !window.location.hash) window.scrollTo(0, 0);
    let first = true;
    const mm = gsap.matchMedia();
    mm.add({ motion: "(prefers-reduced-motion: no-preference)", reduce: "(prefers-reduced-motion: reduce)" }, (ctx) => {
      if ((ctx.conditions as { reduce?: boolean }).reduce) {
        // Reduced motion: native scrolling, nothing hidden, no count-up, no progress bar.
        root.classList.remove(REVEAL_CLASS);
        return;
      }
      root.classList.add(REVEAL_CLASS);
      (window as unknown as { __bonclayMotion?: boolean }).__bonclayMotion = true;

      const instance = new Lenis({ duration: 1.15, easing: (t) => 1 - Math.pow(1 - t, 4), smoothWheel: true });
      instance.on("scroll", ScrollTrigger.update);
      const tick = (time: number) => instance.raf(time * 1000);
      gsap.ticker.add(tick);
      gsap.ticker.lagSmoothing(0);
      lenisRef.current = instance;
      setLenis(instance);

      const tweens: gsap.core.Tween[] = [];
      let stalled = false;
      const run = (els: Element[], from: gsap.TweenVars, to: gsap.TweenVars) => {
        if (stalled) return;
        tweens.push(gsap.fromTo(els, from, { ...to, stagger: 0.07, overwrite: true, clearProps: "transform" }));
      };
      const plain = gsap.utils.toArray<HTMLElement>("[data-reveal]:not([data-reveal='media'])");
      const media = gsap.utils.toArray<HTMLElement>("[data-reveal='media']");
      const batch = (els: HTMLElement[], from: gsap.TweenVars, to: gsap.TweenVars) => {
        if (!els.length) return;
        const go = (b: Element[]) => run(b, from, to);
        ScrollTrigger.batch(els, { start: "top 88%", once: true, batchMax: 8, interval: 0.08, onEnter: go, onEnterBack: go });
      };
      batch(plain, { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.8, ease: "expo.out" });
      batch(media, { opacity: 0, scale: 0.96, y: 20 }, { opacity: 1, scale: 1, y: 0, duration: 1, ease: "expo.out" });

      // Anything already scrolled past (a deep link such as /#faq) would never fire onEnter: show it.
      const revealPast = () => {
        [...plain, ...media].forEach((el) => {
          if (el.style.opacity !== "1" && el.getBoundingClientRect().bottom < 0) gsap.set(el, { opacity: 1, clearProps: "transform" });
        });
      };
      ScrollTrigger.addEventListener("refresh", revealPast);

      // Count-ups: the markup holds the final text; the first number animates once on first view.
      const restore: Array<() => void> = [];
      gsap.utils.toArray<HTMLElement>("[data-countup]").forEach((el) => {
        const finalText = el.textContent ?? "";
        const m = finalText.match(/\d[\d,]*(\.\d+)?/);
        if (!m || m.index === undefined) return;
        const target = parseFloat(m[0].replace(/,/g, ""));
        const decimals = m[1] ? m[1].length - 1 : 0;
        const commas = m[0].includes(",");
        const before = finalText.slice(0, m.index);
        const after = finalText.slice(m.index + m[0].length);
        const fmt = (v: number) =>
          before + (commas ? v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : v.toFixed(decimals)) + after;
        // Lock the final width so counting never moves the text around it.
        el.style.minWidth = `${Math.ceil(el.getBoundingClientRect().width)}px`;
        el.textContent = fmt(0);
        const state = { v: 0 };
        ScrollTrigger.create({
          trigger: el,
          start: "top 92%",
          once: true,
          onEnter: () => {
            tweens.push(
              gsap.to(state, {
                v: target,
                duration: 1.4,
                ease: "power2.out",
                onUpdate: () => {
                  el.textContent = fmt(state.v);
                },
                onComplete: () => {
                  el.textContent = finalText;
                  el.style.minWidth = "";
                },
              }),
            );
          },
        });
        restore.push(() => {
          el.textContent = finalText;
          el.style.minWidth = "";
        });
      });

      // Thin scroll-progress bar (scaleX), purely decorative.
      const barEl = bar.current;
      if (barEl) {
        ScrollTrigger.create({
          start: 0,
          end: "max",
          onUpdate: (self) => {
            barEl.style.transform = `scaleX(${self.progress})`;
          },
        });
      }

      // Fonts and late layout move every trigger: refresh once they settle, then honour a deep link.
      let alive = true;
      const settle = () => {
        if (!alive) return;
        ScrollTrigger.refresh();
        if (first) {
          first = false;
          const hash = window.location.hash;
          if (hash) {
            scrollToHash(hash, instance, { push: false, immediate: true });
            // Late images can still shift the page a little: correct once more unless the visitor has started scrolling.
            let moved = false;
            const stop = () => {
              moved = true;
            };
            (["wheel", "touchstart", "keydown"] as const).forEach((t) => window.addEventListener(t, stop, { once: true, passive: true }));
            window.setTimeout(() => {
              if (!moved && alive) scrollToHash(hash, instance, { push: false, immediate: true });
            }, 1000);
          }
        }
      };
      if (document.fonts?.ready) document.fonts.ready.then(settle, () => undefined);
      else settle();
      if (document.readyState !== "complete") window.addEventListener("load", settle, { once: true });

      // Frame watchdog. Every reveal is a GSAP tween and GSAP advances on requestAnimationFrame, which a browser stops
      // (or nearly stops) for a page it is not rendering: a tab opened in the background, an occluded window, an embedded
      // preview. The tweens would then sit at their hidden first frame and the page would look blank. If fewer than
      // MIN_FRAMES frames arrive in the first WATCHDOG_MS, show everything in its final state instead; a page that is
      // rendering normally never gets here.
      let frames = 0;
      let rafId = 0;
      const count = () => {
        frames += 1;
        rafId = requestAnimationFrame(count);
      };
      rafId = requestAnimationFrame(count);
      const watchdog = window.setTimeout(() => {
        cancelAnimationFrame(rafId);
        if (!alive || frames >= MIN_FRAMES) return;
        stalled = true;
        tweens.forEach((t) => t.kill());
        restore.forEach((fn) => fn());
        root.classList.remove(REVEAL_CLASS);
        gsap.set(document.querySelectorAll("[data-reveal], [data-line], [data-fade], [data-reel]"), { clearProps: "opacity,transform,visibility" });
      }, WATCHDOG_MS);

      return () => {
        alive = false;
        window.clearTimeout(watchdog);
        cancelAnimationFrame(rafId);
        window.removeEventListener("load", settle);
        ScrollTrigger.removeEventListener("refresh", revealPast);
        tweens.forEach((t) => t.kill());
        restore.forEach((fn) => fn());
        gsap.set([...plain, ...media], { clearProps: "opacity,transform" });
        if (barEl) barEl.style.transform = "";
        gsap.ticker.remove(tick);
        instance.destroy();
        lenisRef.current = null;
        setLenis(null);
        (window as unknown as { __bonclayMotion?: boolean }).__bonclayMotion = false;
        root.classList.remove(REVEAL_CLASS);
      };
    });
    return () => mm.revert();
  }, [pathname]);

  // Every in-page anchor (nav, hero, final CTA, anything other agents add) glides and updates the hash.
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = (e.target as Element | null)?.closest?.("a[href^='#']") as HTMLAnchorElement | null;
      if (!a || (a.target && a.target !== "_self") || a.hasAttribute("download")) return;
      let hash = a.getAttribute("href") ?? "";
      if (hash.length < 2) return;
      // The shared "Skip to content" link targets <main>, which also contains the nav: land on #content instead.
      const skip = hash === "#main-content" && !!document.getElementById("content");
      if (skip) hash = "#content";
      if (scrollToHash(hash, lenisRef.current, { push: !skip })) e.preventDefault();
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  return (
    <LenisContext.Provider value={lenis}>
      <div ref={bar} className="s-progress" aria-hidden />
      {children}
    </LenisContext.Provider>
  );
}

/** In-page anchor that glides with Lenis when it is running and falls back to native scrolling. Updates the URL hash. */
export function useScrollTo() {
  const lenis = useLenis();
  return useCallback((hash: string) => scrollToHash(hash, lenis), [lenis]);
}
