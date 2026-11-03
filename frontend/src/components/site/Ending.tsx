"use client";

import Link from "next/link";
import { useRef, type ReactNode } from "react";
import { ArrowUpRight } from "lucide-react";
import { Logo } from "@/components/Logo";
import { MONUMENT_CREDITS } from "./monuments";
import { REPO_PUBLIC, REPO_URL } from "./repo";
import { ALL_SITE_PAGES } from "./routes";
import { Magnetic, Reveal, SplitHeading, motion, useMotionTemplate, useMotionValue, usePointerLean, useSpring, useStill, useTransform } from "./motion";

/**
 * The end of the landing page: Faq, FinalCta and Footer.
 * Everything is readable and operable without JS: the FAQ is native <details> and the CTA panel is static markup.
 */

const REPO = REPO_URL;
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ------------------------------------------------------------------ styles (scoped, no global edits) */

const ENDING_CSS = `
/* FAQ: native details; the height animates where ::details-content is supported, and simply snaps where it is not. */
.site .faq-item { interpolate-size: allow-keywords; }
.site .faq-item > summary { list-style: none; }
.site .faq-item > summary::-webkit-details-marker { display: none; }
.site .faq-item .faq-bar-v { transition: transform .3s cubic-bezier(.16,1,.3,1); transform-origin: center; }
.site .faq-item[open] .faq-bar-v { transform: scaleY(0); }
@supports selector(::details-content) {
  .site .faq-item::details-content {
    block-size: 0;
    overflow: hidden;
    transition: block-size .38s cubic-bezier(.16,1,.3,1), content-visibility .38s allow-discrete;
  }
  .site .faq-item[open]::details-content { block-size: auto; }
}
@media (prefers-reduced-motion: reduce) {
  .site .faq-item::details-content { transition: none !important; }
  .site .faq-item .faq-bar-v { transition: none; }
}

`;

/* ------------------------------------------------------------------ FAQ */

type Item = { q: string; a: ReactNode; text: string };

const link = "underline underline-offset-4 decoration-[var(--s-bar)] hover:decoration-[var(--s-ink)]";

const FAQ_ITEMS: Item[] = [
  {
    q: "What does it cost?",
    text: `Nothing to run it: there is no subscription and no per-minute fee from BonClay. ${REPO_PUBLIC ? "The source is on GitHub and you run it on your own computer." : "The code is MIT-licensed and shared on request, and you run it on your own computer."} A Groq key is optional (it makes replies faster; Groq sets its own limits and prices). Phone minutes are billed by your telephony provider, Exotel in our tests. The voice models keep their own licences, listed at the bottom of this page.`,
    a: (
      <>
        <p>
          Nothing to run it. There is no subscription and no per-minute fee from BonClay:{" "}
          {REPO_PUBLIC ? (
            <>
              the source is on{" "}
              <a className={link} href={REPO}>
                GitHub
              </a>{" "}
              and you run it on your own computer.
            </>
          ) : (
            "you run it on your own computer. The code is MIT-licensed and shared on request."
          )}
        </p>
        <p>Costs that can appear, and who charges them:</p>
        <ul>
          <li>A Groq key is optional. It makes replies faster. Groq sets its own limits and prices.</li>
          <li>Phone minutes are billed by your telephony provider (Exotel, in our tests), not by BonClay.</li>
          <li>The voice models keep their own licences, listed at the bottom of this page. One of them, DhVaani, needs its authors&apos; written approval for commercial use.</li>
        </ul>
      </>
    ),
  },
  {
    q: "Does it work offline?",
    text: "On an Apple Silicon Mac, yes: after the one-time model download, speaking, cloning, listening, answering from your documents, watermarking and receipts all run on the Mac with no internet and no keys. The network is used only for things you switch on: Groq, a hosted Supabase setup, Exotel phone calls, Google Calendar, WhatsApp. On Windows or Linux the agents need a Groq key or a local OpenAI-compatible server such as Ollama.",
    a: (
      <>
        <p>
          On an Apple Silicon Mac, yes. After the one-time model download, speaking, cloning, listening, answering from your documents,
          watermarking and receipts all run on the Mac, with no internet and no keys.
        </p>
        <p>The network is used only for things you switch on yourself:</p>
        <ul>
          <li>Groq, for faster listening and answers (your audio or text goes to Groq).</li>
          <li>A hosted Supabase setup, for sign-in and cloud storage.</li>
          <li>Exotel for phone calls, Google Calendar for meeting times, WhatsApp for voice notes.</li>
          <li>The first download of each model, from Hugging Face.</li>
        </ul>
        <p>On Windows or Linux the agents need a Groq key or a local OpenAI-compatible server such as Ollama.</p>
      </>
    ),
  },
  {
    q: "Can someone clone my voice without permission?",
    text: "Not through BonClay. Before a voice is cloned the speaker says a one-time sentence that contains a random code. BonClay checks the code was spoken, and checks the recording is the same person as the sample. Same-person recordings scored 0.88 and different people 0.14 in our test. The consent recording is stored with the voice, and one-click withdrawal deletes both. A clone made with some other tool is outside what BonClay can stop.",
    a: (
      <>
        <p>
          Not through BonClay. Before a voice is cloned, the speaker says a one-time sentence that contains a random code. BonClay checks
          the code was spoken, and checks the recording is the same person as the sample (in our test, the same person scored 0.88 and
          different people 0.14).
        </p>
        <p>
          The consent recording is stored with the voice. One-click withdrawal deletes the voice and its recording, and the withdrawal is
          written to a signed public log. A clone made with some other tool is outside what BonClay can stop.
        </p>
      </>
    ),
  },
  {
    q: "How can anyone tell a clip is AI-made?",
    text: "Every clip and agent reply gets an inaudible watermark, a signed receipt and an AI-generated label inside the WAV file. Anyone can drop a clip on the Verify page to check it. The watermark survived MP3 at 32 kbps, a WhatsApp-style chain and 8 kHz phone audio in 70 of 70 test cases. It is evidence, not proof: speed or pitch changes, heavy noise or AI re-encoding can remove it.",
    a: (
      <>
        <p>
          Every clip and agent reply gets an inaudible watermark, a signed receipt and an &ldquo;AI-generated&rdquo; label inside the WAV
          file. Anyone can drop a clip on the{" "}
          <Link className={link} href="/verify">
            Verify page
          </Link>{" "}
          to check it, and the receipt signature can be checked offline with the{" "}
          <a className={link} href={`${API_BASE}/.well-known/bonclay-receipt-key`}>
            public key
          </a>
          .
        </p>
        <p>
          The watermark was found in 70 of 70 test cases over MP3 at 32 kbps, a WhatsApp-style chain and 8 kHz phone audio. It is evidence,
          not proof: speed or pitch changes, heavy noise or re-encoding with an AI codec can remove it.
        </p>
      </>
    ),
  },
  {
    q: "Which languages does it speak?",
    text: "Speech in 22 Indian languages with Indic-Speak preset voices. Cloning your own voice in Indian languages with DhVaani-0.5 from ARTPARK-IISc, with Hindi also possible through Chatterbox. Qwen3-TTS clones in 10 other languages including English. Agents understand and answer in English, Hindi and Hinglish. Live phone and agent turns use a preset voice, because cloning is too slow on a laptop for a live call.",
    a: (
      <>
        <ul>
          <li>Speech in 22 Indian languages, with Indic-Speak preset voices.</li>
          <li>Your own consented voice speaking Indian languages, with DhVaani-0.5 from ARTPARK-IISc. Hindi can also use Chatterbox.</li>
          <li>Cloning in ten more languages, English among them, with Qwen3-TTS.</li>
          <li>Agents understand and answer in English, Hindi and Hinglish, and follow the caller if they switch.</li>
        </ul>
        <p>Live phone and agent turns use a preset voice, because cloned Indian-language speech is too slow on a laptop for a live call.</p>
      </>
    ),
  },
  {
    q: "What hardware do I need?",
    text: "It was measured on an Apple M4 with 16 GB. With every model loaded it uses 10 to 12 GB, so 16 GB works but swaps under heavy use. An 8 GB Mac and a Windows laptop with an RTX 3050 Ti are not tested yet. The Can you run it? page shows what to expect for your machine.",
    a: (
      <>
        <p>
          Our numbers come from an Apple M4 with 16 GB. With every model loaded BonClay uses 10 to 12 GB, so 16 GB works but swaps under
          heavy use. An 8 GB Mac and a Windows laptop with an RTX 3050 Ti have not been tested yet.
        </p>
        <p>
          The{" "}
          <Link className={link} href="/run-it">
            Can you run it?
          </Link>{" "}
          page lets you pick your machine and see what works, what to switch off and how sure we are.
        </p>
      </>
    ),
  },
  {
    q: "Is my data shared with anyone?",
    text: "By default no. In local mode your voices, documents and conversations stay in a folder on your own computer. Data leaves only when you switch a service on: Groq receives the audio or text of a turn it answers, Supabase stores what you put in a hosted setup, Exotel carries phone calls. This page has no trackers or analytics.",
    a: (
      <>
        <p>
          By default, no. In local mode your voices, documents and conversations stay in a folder on your own computer. This page has no
          trackers or analytics.
        </p>
        <p>
          Data leaves your machine only when you switch a service on: Groq receives the audio or text of the turns it handles, Supabase
          stores what you put in a hosted setup, and Exotel carries phone calls.
        </p>
      </>
    ),
  },
  {
    q: "Can it answer phone calls?",
    text: "Partly. An agent can place a call over Exotel from the Agents page, speaks first to say it is an AI, and stops when the person talks over it. One live call has been tested, on an Exotel trial account. Receiving calls on your own number is set up but not yet tested. Only call people who agreed.",
    a: (
      <>
        <p>
          Partly. An agent can place a call over Exotel from the Agents page. It speaks first to say it is an AI and stops when the person
          talks over it. One live call has been tested, on an Exotel trial account.
        </p>
        <p>Receiving calls on your own number is set up but not yet tested. Only call people who agreed to be called.</p>
      </>
    ),
  },
  {
    q: "What is not ready yet?",
    text: "WhatsApp voice notes and Google Sheets logging are built but only tested against fake services. Inbound phone calls and real Google Calendar booking are untested. Windows is untested. Hosted cloud mode is tested against fakes only. The Hindi watermark is weaker than English under heavy noise. A 16 GB Mac swaps 3 to 5 GB with every model loaded. Groq's free tier runs out after a day of heavy use.",
    a: (
      <>
        <p>
          The full list is in the{" "}
          {REPO_PUBLIC ? (
            <a className={link} href={`${REPO}/blob/main/STATUS.md`}>
              status file
            </a>
          ) : (
            "project's status file (shared on request)"
          )}
          . The short version:
        </p>
        <ul>
          <li>WhatsApp voice notes and Google Sheets logging are built but tested only against fake services.</li>
          <li>Inbound phone calls and real Google Calendar booking are untested.</li>
          <li>Windows is untested. Hosted cloud mode is tested against fakes only.</li>
          <li>The Hindi watermark is weaker than English under heavy noise (common channels are fine).</li>
          <li>A 16 GB Mac swaps 3 to 5 GB with every model loaded.</li>
          <li>Groq&apos;s free tier runs out after a day of heavy use; replies then come from the slower local model.</li>
        </ul>
      </>
    ),
  },
  {
    q: "Is this legal in India?",
    text: "BonClay builds in the things India's rules on synthetic media and personal data are concerned with: spoken consent before cloning, a retention period, one-click withdrawal, an AI-generated label and a watermark on everything it makes, and an AI disclosure at the start of every agent call. That is a design choice, not a legal opinion. We are a student project, not lawyers: what is lawful for your use, for example calling people, is for you to check.",
    a: (
      <>
        <p>
          BonClay builds in what India&apos;s rules on synthetic media and personal data are concerned with: spoken consent before cloning,
          a retention period, one-click withdrawal, an AI-generated label and watermark on everything it makes, and an AI disclosure at the
          start of every agent call.
        </p>
        <p>
          That is a design choice, not a legal opinion. We are a student project, not lawyers, so this is not legal advice. What is lawful
          for your use, for example calling people, is for you to check.
        </p>
      </>
    ),
  },
];

const FAQ_JSON_LD = JSON.stringify({
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: FAQ_ITEMS.map(({ q, text }) => ({
    "@type": "Question",
    name: q,
    acceptedAnswer: { "@type": "Answer", text },
  })),
}).replace(/</g, "\\u003c");

export function Faq() {
  return (
    <section id="faq" className="s-container mt-28 md:mt-36" aria-labelledby="faq-title">
      <style>{ENDING_CSS}</style>
      <div className="grid gap-10 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.4fr)] lg:gap-16">
        <div data-reveal>
          <h2 id="faq-title" className="h2 max-w-[14ch]">
            Straight answers.
          </h2>
          <p className="lede mt-5">What it costs, what runs on your machine, and what is not ready yet.</p>
        </div>
        <div className="border-t border-[var(--s-line)]">
          {FAQ_ITEMS.map(({ q, a }) => (
            <details key={q} className="faq-item group border-b border-[var(--s-line)]" data-reveal>
              <summary className="flex min-h-[4rem] cursor-pointer items-center justify-between gap-6 py-4 text-left s-item font-medium leading-snug text-[var(--s-ink)]">
                <span>{q}</span>
                <span
                  aria-hidden
                  className="relative grid h-9 w-9 shrink-0 place-items-center rounded-full border border-[var(--s-bar)] transition-colors duration-200 group-hover:border-[var(--s-ink)]"
                >
                  <span className="absolute h-[1.5px] w-3.5 rounded bg-[var(--s-ink)]" />
                  <span className="faq-bar-v absolute h-3.5 w-[1.5px] rounded bg-[var(--s-ink)]" />
                </span>
              </summary>
              <div className="faq-body max-w-[62ch] space-y-3 pb-7 pr-0 s-body leading-relaxed text-[var(--s-ink-2)] sm:pr-14 [&_li]:ml-5 [&_li]:list-disc [&_li]:pl-1 [&_ul]:space-y-1.5">
                {a}
              </div>
            </details>
          ))}
        </div>
      </div>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: FAQ_JSON_LD }} />
    </section>
  );
}

/* ------------------------------------------------------------------ final CTA */

const ctaBase =
  "group inline-flex min-h-[3rem] items-center justify-center gap-2 rounded-full px-6 s-body font-medium transition-[transform,background-color,border-color] duration-200 active:scale-[0.97]";

export function FinalCta() {
  const still = useStill();
  const panelRef = useRef<HTMLDivElement>(null);
  // A MotionValue, not React state: useSpring reads its source once, so a boolean would never turn the glow on.
  const hovered = useMotionValue(0);
  const lean = usePointerLean(panelRef);
  const glowX = useTransform(lean.x, [-1, 1], ["0%", "100%"]);
  const glowY = useTransform(lean.y, [-1, 1], ["0%", "100%"]);
  const glowBg = useMotionTemplate`radial-gradient(600px circle at ${glowX} ${glowY}, rgba(240,162,58,0.22), transparent 60%)`;
  const glowOpacity = useSpring(hovered, { stiffness: 120, damping: 20 });
  return (
    <section id="start" className="s-container mt-28 md:mt-36" aria-labelledby="start-title">
      <style>{ENDING_CSS}</style>
      <Reveal scale>
      <div
        ref={panelRef}
        onPointerEnter={(e) => {
          if (e.pointerType === "mouse") hovered.set(1);
        }}
        onPointerLeave={() => hovered.set(0)}
        className="s-night-panel on-night rounded-[1.75rem] px-7 py-16 sm:px-12 md:rounded-[2.25rem] md:py-24 lg:px-20"
      >
        <div aria-hidden className="pointer-events-none absolute inset-0 -z-10">
          {/* The warm glow is plain gradients painted once: a blur filter on moving layers made scrolling past this panel stutter. */}
          <div
            className="absolute inset-0"
            style={{
              background:
                "radial-gradient(60% 90% at 12% 0%, rgba(240,162,58,0.26), transparent 70%), radial-gradient(55% 80% at 92% 100%, rgba(226,115,58,0.14), transparent 70%)",
            }}
          />
          <div className="absolute inset-0" style={{ background: "linear-gradient(180deg, rgba(14,15,17,0) 40%, rgba(14,15,17,0.7))" }} />
          {!still && <motion.div className="absolute inset-0" style={{ background: glowBg, opacity: glowOpacity }} />}
        </div>

        <div className="max-w-[51.25rem]">
          <Reveal as="p" className="s-eyebrow">Start</Reveal>
          <SplitHeading id="start-title" text="Give your agent a voice" accent="people trust." className="s-title mt-4 max-w-[16ch]" />
          <Reveal as="p" delay={0.1} className="s-lede mt-6 max-w-[52ch] text-[var(--s-night-mute)]">
            Clone a voice only with a spoken yes, let it answer from your own documents, and let anyone check what it said.
            It all runs on your laptop.
          </Reveal>
          <Reveal delay={0.2} className="mt-10 flex flex-col gap-3 sm:flex-row sm:flex-wrap">
            <Magnetic><Link href="/dashboard" prefetch={true} className={`${ctaBase} bg-[var(--s-marigold)] text-[var(--s-marigold-ink)] hover:bg-[#F5B358]`}>
              Open studio
              <ArrowUpRight className="h-4 w-4 transition-transform duration-200 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" aria-hidden />
            </Link></Magnetic>
            <Magnetic><Link href="/verify" className={`${ctaBase} border border-[var(--s-night-line)] bg-[var(--s-night-2)] text-[var(--s-night-text)] hover:border-[var(--s-night-mute)]`}>
              Verify a clip
            </Link></Magnetic>
            <Magnetic>{REPO_PUBLIC ? (
              <a href={REPO} className={`${ctaBase} border border-[var(--s-night-line)] text-[var(--s-night-text)] hover:border-[var(--s-night-mute)]`}>
                Run it on your laptop
                <ArrowUpRight className="h-4 w-4 transition-transform duration-200 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" aria-hidden />
              </a>
            ) : (
              <Link href="/run-it" className={`${ctaBase} border border-[var(--s-night-line)] text-[var(--s-night-text)] hover:border-[var(--s-night-mute)]`}>
                Run it on your laptop
              </Link>
            )}</Magnetic>
          </Reveal>
        </div>
      </div>
      </Reveal>
    </section>
  );
}

/* ------------------------------------------------------------------ footer */

/** `repo`: points into the GitHub repository, so it is left out while the repository is private (see ./repo). */
type FooterLink = { label: string; href: string; external?: boolean; repo?: boolean };

const FOOTER_COLUMNS: { title: string; links: FooterLink[] }[] = [
  {
    title: "Explore",
    links: ALL_SITE_PAGES.map((p) => ({ label: p.nav, href: p.href })),
  },
  {
    title: "Product",
    links: [
      { label: "Studio", href: "/dashboard/studio" },
      { label: "Dashboard", href: "/dashboard" },
      { label: "Agents", href: "/dashboard/studio/agents" },
      { label: "Voices", href: "/dashboard/voices" },
      { label: "Verify a clip", href: "/verify" },
    ],
  },
  {
    title: "Developers",
    links: [
      { label: "Source on GitHub", href: REPO, external: true, repo: true },
      { label: "Website widget", href: `${REPO}/blob/main/docs/WIDGET.md`, external: true, repo: true },
      { label: "Phone and meetings", href: `${REPO}/blob/main/docs/PHONE_AND_MEETINGS.md`, external: true, repo: true },
      { label: "WhatsApp voice notes", href: `${REPO}/blob/main/docs/WHATSAPP.md`, external: true, repo: true },
      { label: "Receipt public key", href: `${API_BASE}/.well-known/bonclay-receipt-key`, external: true },
    ],
  },
  {
    title: "Project",
    links: [
      { label: "Status", href: `${REPO}/blob/main/STATUS.md`, external: true, repo: true },
      { label: "Measured results", href: `${REPO}/blob/main/docs/RESULTS.md`, external: true, repo: true },
      { label: "Licences", href: `${REPO}#-license--acknowledgments`, external: true, repo: true },
    ],
  },
];

const footLink =
  "inline-flex min-h-[2.75rem] items-center gap-1 s-body text-[var(--s-ink-2)] underline-offset-4 hover:text-[var(--s-ink)] hover:underline";

export function Footer() {
  return (
    <footer className="s-container mt-24 pb-12 md:mt-32" aria-label="Site footer">
      <div className="grid gap-12 border-t border-[var(--s-line)] pt-12 lg:grid-cols-[1fr_2.2fr]" data-reveal>
        <div>
          <Logo size={40} />
          <p className="mt-4 max-w-[36ch] s-body leading-relaxed text-[var(--s-mute)]">
            A voice studio for India. Final-year B.Tech project, MIT School of Computing, Pune.
          </p>
          <p className="mt-4 max-w-[36ch] s-small leading-relaxed text-[var(--s-mute)]">
            No trackers or analytics on this site. Cookies are used only to keep you signed in when a hosted setup is configured.
          </p>
        </div>
        <nav aria-label="Footer" className="grid grid-cols-2 gap-x-6 gap-y-8 sm:grid-cols-4">
          {FOOTER_COLUMNS.map((col) => ({ ...col, links: col.links.filter((l) => REPO_PUBLIC || !l.repo) })).filter((col) => col.links.length > 0).map((col) => (
            <div key={col.title} className={col.title === "Developers" ? "order-last col-span-2 sm:order-none sm:col-span-1" : ""}>
              <p className="s-small font-medium">{col.title}</p>
              <ul className="mt-2">
                {col.links.map((l) => (
                  <li key={l.label}>
                    {l.external ? (
                      <a href={l.href} className={footLink}>
                        {l.label}
                        <ArrowUpRight className="h-3.5 w-3.5 shrink-0" aria-hidden />
                      </a>
                    ) : l.href.startsWith("#") ? (
                      <a href={l.href} className={footLink}>
                        {l.label}
                      </a>
                    ) : (
                      <Link href={l.href} prefetch={true} className={footLink}>
                        {l.label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
      </div>

      <div className="mt-12 grid gap-10 border-t border-[var(--s-line)] pt-10 md:grid-cols-[1fr_1fr_1.4fr]" data-reveal>
        <div>
          <p className="s-small font-medium">Built by</p>
          <ul className="mt-3 space-y-1.5 s-body text-[var(--s-mute)]">
            <li>Ayush Chougula</li>
            <li>Atharva Gaikwad</li>
            <li>Arnav Raut</li>
            <li>Harpalsing Rajput</li>
          </ul>
          <p className="mt-3 s-small text-[var(--s-mute)]">Guided by Prof. Neha Chaube</p>
        </div>
        <div>
          <p className="s-small font-medium">Open models</p>
          <ul className="mt-3 space-y-1.5 s-body text-[var(--s-mute)]">
            <li>Kokoro-82M</li>
            <li>Qwen3-TTS 0.6B</li>
            <li>DhVaani by ARTPARK-IISc</li>
            <li>Chatterbox Multilingual</li>
            <li>Indic-Speak by Bodhan AI / AI4Bharat</li>
            <li>Whisper and AudioSeal</li>
            <li>Llama 3.3 70B and Llama 3.2 3B</li>
          </ul>
        </div>
        <div>
          <p className="s-small font-medium">Notices</p>
          <p className="mt-3 max-w-[40ch] s-small leading-relaxed text-[var(--s-mute)]">
            Built with Indic-Speak from Bodhan AI / AI4Bharat.
            <br />
            Voice cloning in Indian languages by DhVaani from ARTPARK-IISc.
            <br />
            Built with Llama.
          </p>
          <p className="mt-3 max-w-[40ch] s-small leading-relaxed text-[var(--s-mute)]">
            BonClay is an educational and research project. Each model keeps its own licence{REPO_PUBLIC ? "; see the project licences linked above." : ": the models and their authors are listed above."}
          </p>
          <details className="mt-3 max-w-[44ch] s-small leading-relaxed text-[var(--s-mute)]">
            <summary className="inline-flex min-h-[1.5rem] cursor-pointer items-center underline underline-offset-4 hover:text-[var(--s-ink)]">
              Photo credits
            </summary>
            <p className="mt-2">Monument photos on the language cards are from Wikimedia Commons, scaled and slightly desaturated.</p>
            <ul className="mt-2 space-y-1">
              {MONUMENT_CREDITS.map((c) => (
                <li key={c.code}>
                  <a href={c.source} className="underline underline-offset-2 hover:text-[var(--s-ink)]">
                    {c.name}
                  </a>
                  {", "}
                  {c.author},{" "}
                  <a href={c.licenseUrl} className="underline underline-offset-2 hover:text-[var(--s-ink)]">
                    {c.license}
                  </a>
                </li>
              ))}
            </ul>
          </details>
        </div>
      </div>
    </footer>
  );
}
