"use client";

import Link from "next/link";
import { useLayoutEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Check } from "lucide-react";

gsap.registerPlugin(ScrollTrigger);

const prefersReduced = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const API_ORIGIN = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const PIN_CSS = `
[data-pin="on"] .pin-wrap { height: calc(var(--stage-h) + var(--run)); }
[data-pin="on"] .pin-stage { position: sticky; top: 0; height: var(--stage-h); overflow: hidden; }
`;

const CONSENT_STEPS = [
  ["Say the sentence", "The speaker reads a fresh sentence and a code aloud. It is new every time, so an old recording cannot be replayed."],
  ["We check it is the same person", "The words and the code are checked, then the voice is matched with the sample. Two different people means no clone."],
  ["Every clip is marked and signed", "An inaudible watermark goes into the audio and a signed receipt is kept for it."],
  ["Anyone can verify it", "Drop a clip on the Verify page. It says whether BonClay made it. No login needed."],
] as const;

/** A deterministic voice-like line: a zigzag whose height follows a speech-shaped envelope. */
const WAVE_D = (() => {
  const n = 72;
  let d = "";
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const env = Math.pow(Math.sin(Math.PI * t), 0.7) * (0.45 + 0.55 * Math.abs(Math.sin(i * 0.9) * Math.cos(i * 0.37)));
    const y = 45 + (i % 2 ? -1 : 1) * env * 38;
    d += `${i === 0 ? "M" : "L"}${(t * 600).toFixed(1)} ${y.toFixed(1)} `;
  }
  return d.trim();
})();

const CONSENT_PIN_CSS = `
.consent-pin { --stage-h: 100svh; --run: min(130svh, 68.75rem); }
`;

export function Consent() {
  const root = useRef<HTMLElement>(null);

  useLayoutEffect(() => {
    if (prefersReduced()) return;
    const mm = gsap.matchMedia();
    mm.add("(min-width: 1024px) and (min-height: 820px) and (prefers-reduced-motion: no-preference)", () => {
      const sec = root.current!;
      sec.dataset.pin = "on";
      const q = <T extends Element>(s: string) => sec.querySelector<T>(s)!;
      const steps = gsap.utils.toArray<HTMLElement>("[data-cstep]", sec);
      const wrap = q<HTMLElement>(".pin-wrap");
      const stage = q<HTMLElement>(".pin-stage");
      const ctx = gsap.context(() => {
        const tl = gsap.timeline({
          defaults: { ease: "power2.out" },
          scrollTrigger: {
            trigger: wrap,
            start: "top top",
            end: () => `+=${Math.max(1, wrap.offsetHeight - stage.offsetHeight)}`,
            scrub: true,
            invalidateOnRefresh: true,
          },
        });
        const active = (i: number, at: number) =>
          steps.forEach((s, j) => tl.to(s, { opacity: j === i ? 1 : 0.38, duration: 0.3, ease: "none" }, at));
        // Start state: only the first step is lit and nothing of the card but its frame is drawn yet.
        gsap.set(steps, { opacity: 0.38 });
        gsap.set(steps[0], { opacity: 1 });
        gsap.set("[data-c-wave]", { strokeDashoffset: 1 });
        gsap.set("[data-c-text], [data-c-match], [data-c-stamp], [data-c-receipt], [data-c-chip], [data-c-verify]", { opacity: 0 });
        gsap.set("[data-c-stamp]", { scale: 1.6, transformOrigin: "50% 50%" });
        gsap.set("[data-c-chip]", { rotationX: -90, transformPerspective: 600, transformOrigin: "50% 100%" });
        gsap.set("[data-c-tick]", { strokeDashoffset: 1 });
        // 1: the sentence is said, the waveform is drawn.
        tl.to("[data-c-text]", { opacity: 1, duration: 1 }, 0);
        tl.to("[data-c-wave]", { strokeDashoffset: 0, duration: 3, ease: "none" }, 0);
        // 2: the speaker match ticks.
        active(1, 3);
        tl.to("[data-c-match]", { opacity: 1, duration: 0.8 }, 3);
        tl.to("[data-c-tick]", { strokeDashoffset: 0, duration: 1, ease: "none" }, 3.8);
        // 3: the clip is watermarked and signed.
        active(2, 5);
        tl.to("[data-c-stamp]", { opacity: 1, scale: 1, duration: 1, ease: "power4.out" }, 5);
        tl.to("[data-c-receipt]", { opacity: 1, duration: 1 }, 5.8);
        // 4: the Verified chip flips up.
        active(3, 7.6);
        tl.to("[data-c-chip]", { opacity: 1, rotationX: 0, duration: 1, ease: "expo.out" }, 7.6);
        tl.to("[data-c-verify]", { opacity: 1, duration: 0.8 }, 8.4);
        tl.to({}, { duration: 1 }, 9.2); // hold the finished state
      }, sec);
      ScrollTrigger.refresh();
      return () => {
        ctx.revert();
        delete sec.dataset.pin;
      };
    });
    return () => mm.revert();
  }, []);

  return (
    <section id="consent" aria-labelledby="consent-title" ref={root} className="consent-pin on-night mt-28 bg-[var(--s-night)] text-[var(--s-night-text)] md:mt-36">
      <style>{PIN_CSS + CONSENT_PIN_CSS}</style>
      <div className="pin-wrap">
        <div className="pin-stage">
          <div className="s-container grid h-full content-center gap-12 py-24 lg:grid-cols-[1fr_1.1fr] lg:items-center lg:gap-14 lg:py-20 [[data-pin=on]_&]:pt-24 [[data-pin=on]_&]:pb-10">
            <div>
              <h2 id="consent-title" data-reveal className="h2 max-w-[16ch]">
                Your voice. Your consent. Verifiable.
              </h2>
              <p data-reveal className="mt-4 max-w-[46ch] s-body leading-relaxed text-[var(--s-night-mute)]">
                Voice is personal data. Before BonClay clones anyone, the speaker has to say yes out loud, and BonClay has to hear it.
              </p>
              <ol className="mt-7 divide-y divide-[var(--s-night-line)] border-y border-[var(--s-night-line)]">
                {CONSENT_STEPS.map(([title, body], i) => (
                  <li key={title} data-cstep className="grid grid-cols-[2.25rem_1fr] gap-x-2 py-3.5">
                    <span className="s-body tabular-nums text-[var(--s-marigold)]">{i + 1}</span>
                    <div>
                      <p className="s-body font-medium">{title}</p>
                      <p className="mt-0.5 s-small leading-snug text-[var(--s-night-mute)]">
                        {body}
                        {i === 3 && (
                          <>
                            {" "}
                            <Link href="/verify" className="whitespace-nowrap text-[var(--s-night-text)] underline underline-offset-4">
                              Open /verify
                            </Link>
                          </>
                        )}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
              <p data-reveal className="mt-5 max-w-[52ch] s-small leading-relaxed text-[var(--s-night-mute)]">
                <span className="font-medium text-[var(--s-night-text)]">For developers.</span> Receipts are Ed25519 signatures. The public key is at{" "}
                <a href={`${API_ORIGIN}/.well-known/bonclay-receipt-key`} className="text-[var(--s-night-text)] underline underline-offset-4">
                  /.well-known/bonclay-receipt-key
                </a>{" "}
                and the signed withdrawal log, which lists voices whose consent was withdrawn, is at{" "}
                <a href={`${API_ORIGIN}/api/withdrawals`} className="text-[var(--s-night-text)] underline underline-offset-4">
                  /api/withdrawals
                </a>
                .
              </p>
            </div>

            <div
              data-reveal="media"
              role="img"
              aria-label="Illustration of the four steps: a consent sentence is said and drawn as a waveform, the speaker match is ticked, the clip is watermarked with a signed receipt, and a Verified label appears."
              className="mx-auto w-full max-w-[33.75rem] rounded-[var(--s-r-card)] border border-[var(--s-night-line)] bg-[var(--s-night-2)] p-5 sm:p-7"
            >
              <p className="s-label font-medium text-[var(--s-night-mute)]">Example, not a real recording</p>
              <p data-c-text className="mt-4 s-body leading-relaxed">
                &ldquo;I agree that BonClay may clone my voice. My code is{" "}
                <span className="font-semibold tabular-nums text-[var(--s-marigold)]">4 8 2 7</span>.&rdquo;
              </p>
              <svg viewBox="0 0 600 90" className="mt-3 h-[4.5rem] w-full" fill="none" aria-hidden>
                <path data-c-wave d={WAVE_D} pathLength={1} strokeDasharray={1} strokeDashoffset={0} stroke="var(--s-night-text)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
              </svg>

              <div data-c-match className="mt-4 flex items-center justify-between rounded-[0.75rem] border border-[var(--s-night-line)] px-4 py-3 s-small">
                <span className="text-[var(--s-night-mute)]">Speaker match: consent recording and voice sample</span>
                <span className="inline-flex shrink-0 items-center gap-1.5 font-medium">
                  <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" aria-hidden>
                    <path data-c-tick d="M5 12.5l4.5 4.5L19 7.5" pathLength={1} strokeDasharray={1} strokeDashoffset={0} stroke="var(--s-marigold)" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Same voice
                </span>
              </div>

              <div className="mt-4 grid grid-cols-[1fr_auto] items-center gap-5">
                <div className="relative">
                  <svg viewBox="0 0 600 90" className="h-[3.75rem] w-full" fill="none" aria-hidden>
                    <path d={WAVE_D} stroke="var(--s-marigold)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" opacity={0.9} />
                  </svg>
                  <svg data-c-stamp viewBox="0 0 64 64" className="absolute right-2 top-1/2 h-12 w-12 -translate-y-1/2 rounded-full bg-[var(--s-night-2)]" fill="none" aria-hidden>
                    <circle cx="32" cy="32" r="29" stroke="var(--s-marigold)" strokeWidth="2.5" strokeDasharray="4 4" />
                    <path d="M20 32v0M26 25v14M32 20v24M38 25v14M44 32v0" stroke="var(--s-marigold)" strokeWidth="3.5" strokeLinecap="round" />
                  </svg>
                </div>
                <div data-c-receipt className="min-w-[8.5rem] rounded-[var(--s-r-chip)] border border-[var(--s-night-line)] px-3 py-2 s-label leading-snug text-[var(--s-night-mute)]">
                  <p className="font-medium text-[var(--s-night-text)]">Receipt</p>
                  <p>Watermarked</p>
                  <p>Signed, Ed25519</p>
                </div>
              </div>

              <div className="mt-5 flex items-center justify-between gap-4">
                <span data-c-chip className="inline-flex h-10 items-center gap-2 rounded-full bg-[var(--s-marigold)] px-4 s-small font-semibold text-[var(--s-marigold-ink)]">
                  <Check className="h-4 w-4" aria-hidden /> Verified: made by BonClay
                </span>
                <span data-c-verify className="s-small text-[var(--s-night-mute)]">
                  Checked at /verify
                </span>
              </div>
              <p className="mt-5 border-t border-[var(--s-night-line)] pt-4 s-label leading-snug text-[var(--s-night-mute)]">
                In our tests the watermark was still found in 70 of 70 copies recompressed as MP3 (32 kbps), sent WhatsApp-style, or played over an 8 kHz phone line. It is evidence, not proof: speed or pitch changes and heavy noise can remove it.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
