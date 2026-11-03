"use client";

import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { SITE_PAGES } from "./routes";
import { Item, LIFT_SPRING, Reveal, SplitHeading, Stagger, motion, useStill } from "./motion";

const MotionLink = motion.create(Link);

/** Home's row of cards into the other pages. One card per page in SITE_PAGES, so adding a page adds a card. */
export function GoDeeper() {
  const still = useStill();
  return (
    <section id="go-deeper" aria-labelledby="go-deeper-title" className="s-container mt-28 md:mt-36">
      <Reveal as="p" className="s-eyebrow">Read on</Reveal>
      <SplitHeading id="go-deeper-title" text="Go" accent="deeper." className="h2 mt-4 max-w-[18ch]" />
      <Reveal as="p" className="lede mt-5">
        Each page answers one question, and you can read them in any order.
      </Reveal>
      <Stagger as="ul" role="list" className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {SITE_PAGES.map((p) => (
          <Item key={p.href} as="li" className="flex">
            <MotionLink
              href={p.href}
              className="s-paper-card group flex min-h-[2.75rem] w-full flex-col p-6"
              whileHover={still ? undefined : { y: -6 }}
              transition={LIFT_SPRING}
            >
              <span className="flex items-start justify-between gap-3">
                <span className="s-item font-semibold leading-snug">{p.title}</span>
                <ArrowUpRight className="mt-1 h-4 w-4 shrink-0 transition-transform duration-200 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" aria-hidden />
              </span>
              <span className="mt-2 block s-body leading-snug text-[var(--s-mute)]">{p.blurb}</span>
            </MotionLink>
          </Item>
        ))}
      </Stagger>
    </section>
  );
}
