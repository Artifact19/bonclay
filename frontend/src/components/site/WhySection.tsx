"use client";

import Link from "next/link";
import { useLayoutEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ArrowUpRight } from "lucide-react";
import { BEATS, CHECKED_ON, USE_CASES, type Beat, type Stat } from "./whyData";
import { Item, LIFT_SPRING, Reveal, SplitHeading, Stagger, motion, useStill } from "./motion";

gsap.registerPlugin(ScrollTrigger);

function Cite({ stat, night }: { stat: Stat; night: boolean }) {
  return (
    <p className={`mt-3 s-small leading-snug ${night ? "text-[var(--s-night-mute)]" : "text-[var(--s-mute)]"}`}>
      <a
        href={stat.href}
        target="_blank"
        rel="noopener noreferrer"
        className={`inline-block py-1.5 underline underline-offset-4 ${night ? "hover:text-[var(--s-night-text)]" : "hover:text-[var(--s-ink)]"}`}
      >
        {stat.cite}
        <span className="sr-only"> (opens in a new tab)</span>
      </a>
      {stat.note && <span className="mt-1 block">{stat.note}</span>}
    </p>
  );
}

function StatItem({ stat, night }: { stat: Stat; night: boolean }) {
  return (
    <li
      data-reveal
      className={`flex flex-col rounded-2xl border p-6 ${
        night ? "border-[var(--s-night-line)] bg-[var(--s-night-2)]" : "border-[var(--s-line)] bg-[var(--s-paper)]"
      }`}
    >
      <p
        className="s-stat"
        {...(stat.countup ? { "data-countup": "" } : {})}
      >
        {stat.numeral}
      </p>
      <p className={`mt-3 s-body leading-snug ${night ? "text-[var(--s-night-text)]" : "text-[var(--s-ink-2)]"}`}>{stat.meaning}</p>
      <div className="mt-auto pt-1">
        <Cite stat={stat} night={night} />
      </div>
    </li>
  );
}

function BeatLinks({ links, night }: { links?: Beat["links"]; night: boolean }) {
  if (!links?.length) return null;
  return (
    <p className="mt-4 flex flex-wrap gap-x-6 gap-y-2 s-body font-semibold">
      {links.map((l) => {
        const external = l.href.startsWith("http");
        const cls = `inline-flex min-h-[2.75rem] items-center gap-1 underline underline-offset-4 ${night ? "text-[var(--s-night-text)]" : "text-[var(--s-ink)]"}`;
        return external ? (
          <a key={l.href} href={l.href} target="_blank" rel="noopener noreferrer" className={cls}>
            {l.label}
            <ArrowUpRight className="h-4 w-4" aria-hidden />
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
        ) : l.href.startsWith("#") ? (
          <a key={l.href} href={l.href} className={cls}>
            {l.label}
          </a>
        ) : (
          <Link key={l.href} href={l.href} prefetch={false} className={cls}>
            {l.label}
          </Link>
        );
      })}
    </p>
  );
}

function BeatBlock({ beat }: { beat: Beat }) {
  const night = beat.id === "trust";
  const headingId = `why-${beat.id}`;
  const cols = "sm:grid-cols-2";
  const inner = (
    <div className="s-container py-16 md:py-24">
      <div
        data-why-rule
        aria-hidden
        className={`h-px origin-left ${night ? "bg-[var(--s-marigold)]" : "bg-[var(--s-ink)]"}`}
      />
      <div className="mt-8 grid gap-x-12 gap-y-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div data-reveal>
          <h3 id={headingId} className="h2 max-w-[16ch]">
            {beat.title}
          </h3>
          <p className={`mt-4 max-w-[44ch] s-body leading-relaxed ${night ? "text-[var(--s-night-mute)]" : "text-[var(--s-mute)]"}`}>{beat.story}</p>
        </div>
        <div>
          <ul role="list" aria-labelledby={headingId} className={`grid gap-4 ${cols}`}>
            {beat.stats.map((s) => (
              <StatItem key={s.id} stat={s} night={night} />
            ))}
          </ul>
          <div
            data-reveal
            className="mt-5 rounded-2xl bg-[var(--s-marigold)] p-6 text-[var(--s-marigold-ink)] selection:bg-[var(--s-marigold-ink)] selection:text-[var(--s-marigold)]"
          >
            <p className="s-label font-semibold text-[#3B2A10]">What BonClay does</p>
            <p className="mt-2 max-w-[60ch] s-body leading-snug">{beat.answer}</p>
            <BeatLinks links={beat.links} night={false} />
          </div>
        </div>
      </div>
    </div>
  );
  return night ? (
    <div className="on-night bg-[var(--s-night)] text-[var(--s-night-text)]">{inner}</div>
  ) : (
    inner
  );
}

/** The short version for Home: the three problems with what BonClay does about each, without the statistics.
 *  The full cited version is the same component without `compact` (it lives on the Proof page). */
function WhyCompact() {
  const still = useStill();
  return (
    <Stagger as="ol" role="list" aria-label="Three problems BonClay answers" className="s-container mt-10 grid gap-4 lg:grid-cols-3">
      {BEATS.map((beat) => (
        <Item key={beat.id} as="li" className="flex">
          <motion.div
            className="s-paper-card group flex w-full flex-col p-6"
            whileHover={still ? undefined : { y: -6 }}
            transition={LIFT_SPRING}
          >
            <h3 className="s-item text-[var(--s-ink)]">{beat.title}</h3>
            <p className="mt-3 s-body leading-relaxed text-[var(--s-mute)]">{beat.story}</p>
            <div className="mt-auto pt-5">
              <p className="rounded-[0.75rem] bg-[var(--s-marigold)] p-4 s-body leading-snug text-[var(--s-marigold-ink)] transition-transform duration-300 motion-safe:group-hover:-translate-y-1">
                {beat.answer}
              </p>
            </div>
          </motion.div>
        </Item>
      ))}
    </Stagger>
  );
}

export function WhySection({ compact = false, id = "why" }: { compact?: boolean; id?: string } = {}) {
  const still = useStill();
  const root = useRef<HTMLElement>(null);

  // One small touch: each beat's top rule draws in from the left. The CSS default is the final state, so it is fully
  // drawn if JS never runs or the visitor prefers reduced motion; only the no-preference branch animates it.
  useLayoutEffect(() => {
    const el = root.current;
    if (!el) return;
    const mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: no-preference)", () => {
      el.querySelectorAll<HTMLElement>("[data-why-rule]").forEach((rule) => {
        gsap.fromTo(
          rule,
          { scaleX: 0 },
          { scaleX: 1, duration: 1.2, ease: "expo.out", scrollTrigger: { trigger: rule, start: "top 88%", once: true } },
        );
      });
    });
    return () => mm.revert();
  }, []);

  return (
    <section id={id} ref={root} aria-labelledby="why-heading" className="mt-28 md:mt-36">
      <div className="s-container">
        <Reveal as="p" className="s-eyebrow">Why</Reveal>
        <SplitHeading id="why-heading" text="Built for how India" accent="actually talks." className="h2 mt-4 max-w-[20ch]" />
        <Reveal as="p" className="lede mt-5">
          {compact ? (
            <>
              Three problems kept coming up. The numbers behind them, each linked to where it came from, are on the{" "}
              <Link href="/proof#why-evidence" className="underline underline-offset-4 hover:text-[var(--s-ink)]">
                Proof page
              </Link>
              .
            </>
          ) : (
            <>Three problems kept coming up. Each comes with numbers, and every number links to where it came from. Prices and dates were checked on {CHECKED_ON}.</>
          )}
        </Reveal>
      </div>

      {compact ? (
        <WhyCompact />
      ) : (
        <>
          <BeatBlock beat={BEATS[0]} />
          <BeatBlock beat={BEATS[1]} />
          <BeatBlock beat={BEATS[2]} />
        </>
      )}

      {!compact && (
        <div className="s-container pb-4">
          <h3 id="why-who" data-reveal className="s-h3 text-[var(--s-ink)]">
            Who it is for
          </h3>
          <ul role="list" aria-labelledby="why-who" className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {USE_CASES.map((u) => {
              const body = (
                <>
                  <span className="flex items-start justify-between gap-3">
                    <span className="s-item font-semibold leading-snug">{u.title}</span>
                    <ArrowUpRight className="mt-1 h-4 w-4 shrink-0 transition-transform duration-200 group-hover:translate-x-0.5 group-hover:-translate-y-0.5" aria-hidden />
                  </span>
                  <span className="mt-2 block s-body leading-snug text-[var(--s-mute)]">{u.line}</span>
                  <span className="mt-4 block s-body font-semibold underline underline-offset-4">{u.cta}</span>
                </>
              );
              const cls =
                "group block min-h-[2.75rem] rounded-2xl border border-[var(--s-line)] bg-[var(--s-paper)] p-6 transition-colors hover:border-[var(--s-ink)]";
              return (
                <motion.li key={u.id} data-reveal whileHover={still ? undefined : { y: -4 }} transition={LIFT_SPRING}>
                  {u.href.startsWith("#") ? (
                    <a href={u.href} className={cls}>
                      {body}
                    </a>
                  ) : (
                    <Link href={u.href} prefetch={false} className={cls}>
                      {body}
                    </Link>
                  )}
                </motion.li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}
