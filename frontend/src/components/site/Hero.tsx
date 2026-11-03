"use client";

import Link from "next/link";
import { ArrowRight, ArrowUpRight, MoreHorizontal, Play, ShieldCheck, Laptop, Timer } from "lucide-react";
import { VoiceReel } from "./VoiceReel";
import { Magnetic, Reveal, SplitHeading } from "./motion";

const FACTS = [
  { icon: Laptop, href: "/run-it", text: "Runs on one laptop" },
  { icon: ShieldCheck, href: "/verify", text: "Every clip watermarked and signed" },
  { icon: Timer, href: "/proof", text: <>First audio in <span data-countup>0.84 s</span></> },
] as const;

const ENGINE_AVATARS = [
  { label: "KK", bg: "#E8632B", fg: "#FAF9F5", title: "Kokoro 82M" },
  { label: "QW", bg: "#FAF9F5", fg: "#141414", title: "Qwen3 0.6B" },
  { label: "F5", bg: "#F3F59E", fg: "#141414", title: "IndicF5" },
  { label: "DV", bg: "#141414", fg: "#FAF9F5", title: "DhVaani" },
] as const;

function StarDiamond({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden className={className}>
      <path
        d="M12 2L15.2 8.8L22 12L15.2 15.2L12 22L8.8 15.2L2 12L8.8 8.8L12 2Z"
        fill="#E8632B"
        stroke="#141414"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function Hero() {
  return (
    <section id="top" aria-labelledby="hero-title" className="pt-22 md:pt-26">
      <div className="s-container">
        <div className="grid gap-7 lg:grid-cols-12 lg:items-stretch">
          {/* LEFT COLUMN: 2x2 Musmentor Bento Grid */}
          <div className="grid gap-4 sm:grid-cols-2 lg:col-span-6">
            {/* LEFT SUB-COLUMN */}
            <div className="flex flex-col gap-4">
              {/* Top-Left Card: 22+ Indian languages + layered black squares */}
              <Reveal immediate delay={0.05} y={12} className="flex flex-col overflow-hidden rounded-[1.6rem] border-[1.5px] border-[#141414] bg-white">
                <div className="flex items-start justify-between p-5 pb-4">
                  <div>
                    <p className="font-sans text-[2.25rem] font-semibold leading-none tracking-tight text-[#141414]">
                      22+
                    </p>
                    <p className="mt-2 text-[0.875rem] font-medium text-[#141414]/80">
                      Indian languages
                    </p>
                  </div>
                  <Link
                    href="/how-it-works"
                    aria-label="Explore how 22 Indian languages work"
                    className="grid size-10 shrink-0 place-items-center rounded-full border-[1.5px] border-[#141414] bg-[#F3F59E] text-[#141414] transition-transform hover:-translate-y-0.5 hover:bg-[#eef17a]"
                  >
                    <ArrowUpRight className="size-4.5" aria-hidden />
                  </Link>
                </div>
                {/* Layered rotated geometric cards graphic */}
                <div className="relative mt-auto flex h-34 items-center justify-center overflow-hidden rounded-t-[1.35rem] border-t-[1.5px] border-[#141414] bg-gradient-to-b from-[#ECEAE4] to-[#DCD8CE]">
                  <svg viewBox="0 0 220 110" className="h-26 w-48" aria-hidden>
                    <rect x="118" y="25" width="54" height="54" rx="4" transform="rotate(-12 145 52)" fill="#141414" fillOpacity="0.14" />
                    <rect x="102" y="23" width="54" height="54" rx="4" transform="rotate(-18 129 50)" fill="#141414" fillOpacity="0.28" />
                    <rect x="86" y="21" width="54" height="54" rx="4" transform="rotate(-25 113 48)" fill="#141414" fillOpacity="0.48" />
                    <rect x="70" y="20" width="54" height="54" rx="4" transform="rotate(-33 97 47)" fill="#141414" fillOpacity="0.72" />
                    <rect x="52" y="20" width="54" height="54" rx="4" transform="rotate(-42 79 47)" fill="#141414" />
                  </svg>
                </div>
              </Reveal>

              {/* Bottom-Left Card: Notched Ticket Card with Barcode */}
              <Reveal immediate delay={0.15} y={14} className="flex flex-1 flex-col overflow-hidden rounded-[1.6rem] border-[1.5px] border-[#141414] bg-white">
                <div className="relative bg-[#E2DDD3] p-5 pb-6">
                  <div className="flex items-start justify-between gap-2">
                    <h2 className="font-sans text-[1.45rem] font-semibold leading-[1.15] tracking-tight text-[#141414]">
                      Secrets
                      <br />
                      of voice
                    </h2>
                    <span className="inline-flex items-center rounded-full border-[1.5px] border-[#141414] bg-[#F3F59E] px-2.5 py-0.5 text-[0.75rem] font-semibold text-[#141414]">
                      100% Local
                    </span>
                  </div>
                  <div className="mt-4 flex items-end justify-between gap-2">
                    <p className="max-w-[17ch] text-[0.78rem] leading-snug text-[#141414]/75">
                      Consented Cloning:
                      <br />
                      Unlocking Cross-Lingual
                      <br />
                      Speech &amp; Receipts.
                    </p>
                    <div className="text-right">
                      <p className="font-sans text-[1.2rem] font-semibold leading-none text-[#141414]">15s</p>
                      <p className="mt-0.5 text-[0.75rem] text-[#141414]/55 line-through">60s</p>
                    </div>
                  </div>
                </div>

                {/* Ticket notch divider */}
                <div className="relative border-t-[1.5px] border-[#141414] bg-white px-5 pb-5 pt-4">
                  <span aria-hidden className="absolute -left-2.5 -top-2.5 size-4.5 rounded-full border-[1.5px] border-[#141414] bg-[#FAF9F5]" />
                  <span aria-hidden className="absolute -right-2.5 -top-2.5 size-4.5 rounded-full border-[1.5px] border-[#141414] bg-[#FAF9F5]" />

                  <div className="flex items-center justify-between text-[0.75rem] font-medium text-[#141414]">
                    <span>AudioSeal</span>
                    <span>Ed25519</span>
                  </div>
                  <div className="my-2.5 border-t border-dashed border-[#141414]/45" />
                  <div className="flex items-end justify-between gap-3 pt-1">
                    <div>
                      <p className="font-sans text-[1.75rem] font-semibold leading-none tracking-tight text-[#141414]">
                        0.84s
                      </p>
                      <p className="mt-1 text-[0.75rem] text-[#141414]/70">First audio chunk</p>
                    </div>
                    {/* Crisp SVG Barcode */}
                    <svg viewBox="0 0 92 38" className="h-9.5 w-22 shrink-0 text-[#141414]" aria-hidden>
                      <rect x="0" y="0" width="3" height="38" fill="currentColor" />
                      <rect x="6" y="0" width="1.5" height="38" fill="currentColor" />
                      <rect x="10" y="0" width="4" height="38" fill="currentColor" />
                      <rect x="17" y="0" width="1.5" height="38" fill="currentColor" />
                      <rect x="21" y="0" width="2.5" height="38" fill="currentColor" />
                      <rect x="27" y="0" width="5" height="38" fill="currentColor" />
                      <rect x="35" y="0" width="1.5" height="38" fill="currentColor" />
                      <rect x="39" y="0" width="3" height="38" fill="currentColor" />
                      <rect x="45" y="0" width="2" height="38" fill="currentColor" />
                      <rect x="50" y="0" width="4.5" height="38" fill="currentColor" />
                      <rect x="58" y="0" width="1.5" height="38" fill="currentColor" />
                      <rect x="62" y="0" width="3" height="38" fill="currentColor" />
                      <rect x="68" y="0" width="5" height="38" fill="currentColor" />
                      <rect x="76" y="0" width="1.5" height="38" fill="currentColor" />
                      <rect x="80" y="0" width="3.5" height="38" fill="currentColor" />
                      <rect x="86" y="0" width="2" height="38" fill="currentColor" />
                      <rect x="90" y="0" width="2" height="38" fill="currentColor" />
                    </svg>
                  </div>
                </div>
              </Reveal>
            </div>

            {/* RIGHT SUB-COLUMN */}
            <div className="flex flex-col gap-4">
              {/* Top-Right Rating Pill */}
              <Reveal immediate delay={0.1} y={12} className="flex items-center justify-between rounded-full border-[1.5px] border-[#141414] bg-white px-4 py-2.5">
                <div className="flex items-center gap-1" aria-label="5 star architecture rating">
                  <StarDiamond className="size-4.5" />
                  <StarDiamond className="size-4.5" />
                  <StarDiamond className="size-4.5" />
                  <StarDiamond className="size-4.5" />
                  <StarDiamond className="size-4.5" />
                </div>
                <span className="text-[0.8125rem] font-semibold text-[#141414]">6 engines</span>
              </Reveal>

              {/* Middle-Right Team / Multi-Engine Card */}
              <Reveal immediate delay={0.14} y={12} className="rounded-[1.6rem] border-[1.5px] border-[#141414] bg-[#E2DDD3] p-5">
                <div className="flex items-center -space-x-2">
                  {ENGINE_AVATARS.map((a) => (
                    <span
                      key={a.label}
                      title={a.title}
                      style={{ backgroundColor: a.bg, color: a.fg }}
                      className="grid size-9.5 place-items-center rounded-full border-[1.5px] border-[#141414] font-mono text-[0.75rem] font-bold shadow-2xs"
                    >
                      {a.label}
                    </span>
                  ))}
                </div>
                <h2 className="mt-4 font-sans text-[1.35rem] font-semibold tracking-tight text-[#141414]">
                  Voice Engines
                </h2>
                <p className="mt-1.5 text-[0.8125rem] leading-snug text-[#141414]/75">
                  Kokoro, Qwen3, IndicF5, DhVaani, Chatterbox &amp; Indic-Speak ready on one machine
                </p>
              </Reveal>

              {/* Bottom-Right Voice Studio Card with Persimmon-Orange Circle */}
              <Reveal immediate delay={0.2} y={14} className="flex flex-1 flex-col justify-between rounded-[1.6rem] border-[1.5px] border-[#141414] bg-[#E2DDD3] p-5">
                <div className="flex items-center justify-between text-[0.8125rem] font-medium text-[#141414]">
                  <span>Voice studio</span>
                  <MoreHorizontal className="size-4.5 text-[#141414]" aria-hidden />
                </div>

                <div className="my-4 flex items-center justify-center">
                  <div className="relative grid size-36 place-items-center overflow-hidden rounded-full border-[1.5px] border-[#141414] bg-[#E8632B] shadow-inner">
                    {/* Artistic editorial voice profile & acoustic wave graphic */}
                    <svg viewBox="0 0 120 120" className="size-full" aria-hidden>
                      <circle cx="88" cy="34" r="36" fill="#141414" fillOpacity="0.18" />
                      <path
                        d="M36 96 C36 74, 84 74, 84 96 L84 120 L36 120 Z"
                        fill="#141414"
                      />
                      <circle cx="60" cy="48" r="19" fill="#141414" />
                      <path
                        d="M60 29 A19 19 0 0 0 60 67 Z"
                        fill="#FAF9F5"
                        fillOpacity="0.88"
                      />
                      <path
                        d="M20 56 L20 64 M26 48 L26 72 M94 48 L94 72 M100 56 L100 64"
                        stroke="#F3F59E"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                      />
                    </svg>
                  </div>
                </div>

                <div>
                  <h2 className="font-sans text-[1.4rem] font-semibold tracking-tight text-[#141414]">
                    Voice clones
                  </h2>
                  <p className="mt-1 text-[0.8125rem] leading-snug text-[#141414]/75">
                    Natural cross-lingual speech through consented voice cloning
                  </p>
                </div>
              </Reveal>
            </div>
          </div>

          {/* RIGHT COLUMN: Editorial Headline + Interlocked Pill CTAs + Twin Stat Cards */}
          <div className="flex flex-col justify-between lg:col-span-6 lg:pl-4">
            <div className="pt-2 lg:pt-4">
              <div className="relative">
                <SplitHeading
                  as="h1"
                  id="hero-title"
                  immediate
                  text="Your voice, in"
                  accent="Indian languages."
                  break
                  className="font-sans text-[clamp(2.65rem,4.3vw,4.35rem)] font-semibold leading-[1.06] tracking-[-0.035em] text-[#141414] [&_.accent]:not-italic [&_.accent]:text-[#141414]"
                />
                {/* Signature Musmentor overlapping circles + diamond star badge */}
                <div className="mt-3 inline-flex items-center -space-x-2.5 align-middle">
                  <span className="inline-block size-9 rounded-full border-[1.5px] border-[#141414] bg-white" aria-hidden />
                  <span className="inline-block size-9 rounded-full border-[1.5px] border-[#141414] bg-[#E8632B]" aria-hidden />
                  <span className="grid size-9 place-items-center rounded-full border-[1.5px] border-[#141414] bg-[#F3F59E]" aria-hidden>
                    <svg viewBox="0 0 24 24" className="size-4.5">
                      <path
                        d="M12 3L14.5 9.5L21 12L14.5 14.5L12 21L9.5 14.5L3 12L9.5 9.5L12 3Z"
                        fill="#FAF9F5"
                        stroke="#141414"
                        strokeWidth="1.5"
                      />
                    </svg>
                  </span>
                </div>
              </div>

              <Reveal immediate delay={0.3} y={14}>
                <p className="mt-6 max-w-[54ch] text-[0.95rem] leading-[1.65] text-[#141414]/80">
                  Speech in 22 Indian languages. Clone your own voice from a few seconds of consented speech, ground AI voice agents in your PDFs, and verify every audio clip with cryptographic proof.
                </p>
              </Reveal>

              {/* Interlocked Black + Sand CTA Pills (exact Musmentor button connector style) */}
              <Reveal immediate delay={0.45} y={14} className="mt-8 flex flex-wrap items-center">
                <div className="inline-flex flex-wrap items-center">
                  <Magnetic>
                    <Link
                      href="/dashboard"
                      prefetch={false}
                      className="group inline-flex h-12.5 items-center gap-3 rounded-full border-[1.5px] border-[#141414] bg-[#141414] px-7 text-[0.9375rem] font-medium text-[#FAF9F5] transition-transform hover:-translate-y-0.5"
                    >
                      <span>Open studio</span>
                      <ArrowRight className="size-4 transition-transform duration-200 group-hover:translate-x-1" aria-hidden />
                    </Link>
                  </Magnetic>
                  <span aria-hidden className="-mx-1.5 hidden h-4 w-4 bg-[#141414] sm:inline-block" />
                  <Magnetic>
                    <a
                      href="#hear"
                      className="group inline-flex h-12.5 items-center gap-3 rounded-full border-[1.5px] border-[#141414] bg-[#E2DDD3] pl-5 pr-2.5 text-[0.9375rem] font-medium text-[#141414] transition-transform hover:-translate-y-0.5 hover:bg-[#d8d2c5]"
                    >
                      <span>Hear it</span>
                      <span className="grid size-8 place-items-center rounded-full bg-[#141414] text-[#FAF9F5]">
                        <Play className="size-3.5 fill-current" aria-hidden />
                      </span>
                    </a>
                  </Magnetic>
                </div>
              </Reveal>

              {/* Evidence Pill Links */}
              <Reveal immediate delay={0.55} y={12} as="div" className="mt-6">
                <ul className="inline-flex flex-wrap items-center gap-2" aria-label="What you can check">
                  {FACTS.map((f) => (
                    <li key={f.href}>
                      <Link
                        href={f.href}
                        className="inline-flex min-h-8.5 items-center gap-1.5 rounded-full border border-[#141414]/35 bg-white px-3 py-1 text-[0.8125rem] font-medium text-[#141414]/85 transition-colors hover:border-[#141414] hover:bg-[#E2DDD3]"
                      >
                        <f.icon className="size-3.5 shrink-0 text-[#E8632B]" aria-hidden />
                        <span>{f.text}</span>
                      </Link>
                    </li>
                  ))}
                </ul>
              </Reveal>
            </div>

            {/* Bottom Twin Stat Cards with 4-point orange stars and horizontal dividers */}
            <Reveal immediate delay={0.65} y={16} className="mt-8 grid overflow-hidden rounded-[1.6rem] border-[1.5px] border-[#141414] bg-white sm:grid-cols-2">
              <div className="border-b-[1.5px] border-[#141414] p-6 sm:border-b-0 sm:border-r-[1.5px]">
                <div className="flex items-center gap-3">
                  <StarDiamond className="size-5 shrink-0" />
                  <span aria-hidden className="h-[1.5px] flex-1 bg-[#141414]/65" />
                </div>
                <p className="mt-4 font-sans text-[2.2rem] font-semibold leading-none tracking-tight text-[#141414]">
                  <span data-countup>22</span>+
                </p>
                <p className="mt-2.5 text-[0.8125rem] leading-relaxed text-[#141414]/75">
                  Our voice studio proudly supports 22 Indian languages with ready voices and consented cloning
                </p>
              </div>

              <div className="p-6">
                <div className="flex items-center gap-3">
                  <StarDiamond className="size-5 shrink-0" />
                  <span aria-hidden className="h-[1.5px] flex-1 bg-[#141414]/65" />
                  <span className="rounded-full border-[1.5px] border-[#141414] bg-white px-2.5 py-0.5 text-[0.75rem] font-medium text-[#141414]">
                    audio
                  </span>
                </div>
                <p className="mt-4 font-sans text-[2.2rem] font-semibold leading-none tracking-tight text-[#141414]">
                  <span data-countup>0.84</span>s
                </p>
                <p className="mt-2.5 text-[0.8125rem] leading-relaxed text-[#141414]/75">
                  Measured time to first audio chunk, with every clip watermarked by AudioSeal and signed
                </p>
              </div>
            </Reveal>
          </div>
        </div>
      </div>

      <div className="s-container mt-12 md:mt-16">
        <VoiceReel />
      </div>
    </section>
  );
}
