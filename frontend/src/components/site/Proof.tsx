"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUpRight, Loader2, Pause, Play } from "lucide-react";
import { claimAudio } from "@/components/site/audioBus";
import { SCRIPT_FONT, useScriptFontsNear } from "./fonts";
import {
  COMPARISON,
  COMPARISON_COLUMNS,
  COMPARISON_FOOTNOTE,
  ENGINE_CREDIT,
  INDIC_ATTRIBUTION,
  LANGUAGE_CLIPS,
  MEASURED_CAVEAT,
  MEASURED_CONTEXT,
  MEASURED_SPEED,
  MEASURED_SPEED_CONDITIONS,
  MEASURED_TRUST,
  MEASURED_TRUST_CONDITIONS,
  NOT_YET,
  RESULTS_URL,
  type Cell,
  type Stat,
} from "./proofData";
import { REPO_PUBLIC } from "./repo";
import { AnimatePresence } from "motion/react";
import { Item, Magnetic, Reveal, SplitHeading, Stagger, motion, useStill } from "./motion";

/* ------------------------------------------------------------------ playable language cards */

type Phase = "loading" | "playing";

/** Five bars that dance while a clip plays; static at mid height under reduced motion. */
function Equaliser({ still }: { still: boolean }) {
  return (
    <motion.span
      aria-hidden
      className="inline-flex h-4 items-center gap-[3px]"
      initial={still ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      {[0, 1, 2, 3, 4].map((i) => (
        <motion.span
          key={i}
          className="h-4 w-[3px] origin-center rounded-full bg-[var(--s-marigold)]"
          style={{ scaleY: 0.5 }}
          animate={still ? { scaleY: 0.5 } : { scaleY: [0.3, 1, 0.5, 0.9, 0.35] }}
          transition={still ? undefined : { duration: 0.9, repeat: Infinity, ease: "easeInOut", delay: i * 0.1 }}
        />
      ))}
    </motion.span>
  );
}

/** One short sentence about booking an appointment, pre-rendered once per language with preset voices.
 *  No cloned voice is ever used here. A single shared audio element means only one card plays at a time,
 *  and claimAudio() makes the hero reel (or any other player) stop when a card starts. */
export function HearIt() {
  const still = useStill();
  const audio = useRef<HTMLAudioElement | null>(null);
  const wanted = useRef<string | null>(null);
  const switching = useRef(false); // true while a card replaces another: the old clip's pause event must not clear the new state
  const [active, setActive] = useState<{ code: string; phase: Phase } | null>(null);
  const [failed, setFailed] = useState<string[]>([]);
  const sectionRef = useRef<HTMLElement>(null);
  // Hindi, Marathi, Tamil, Bengali and Telugu clips use script faces; they download only as this section nears the screen.
  const fontsNear = useScriptFontsNear(sectionRef);

  useEffect(() => {
    const a = new Audio();
    a.preload = "none";
    audio.current = a;
    const onPlaying = () => {
      const code = wanted.current;
      if (code) setActive({ code, phase: "playing" });
    };
    const onWaiting = () => {
      const code = wanted.current;
      if (code) setActive({ code, phase: "loading" });
    };
    const stop = () => setActive(null);
    const onPause = () => {
      if (switching.current) {
        switching.current = false;
        return;
      }
      stop();
    };
    const onError = () => {
      const code = wanted.current;
      setActive(null);
      if (code) setFailed((f) => (f.includes(code) ? f : [...f, code]));
    };
    a.addEventListener("playing", onPlaying);
    a.addEventListener("waiting", onWaiting);
    a.addEventListener("ended", stop);
    a.addEventListener("pause", onPause);
    a.addEventListener("error", onError);
    return () => {
      a.pause();
      a.removeEventListener("playing", onPlaying);
      a.removeEventListener("waiting", onWaiting);
      a.removeEventListener("ended", stop);
      a.removeEventListener("pause", onPause);
      a.removeEventListener("error", onError);
    };
  }, []);

  const toggle = useCallback(
    async (code: string, src: string) => {
      const a = audio.current;
      if (!a) return;
      if (wanted.current === code && !a.paused) {
        a.pause();
        return;
      }
      if (!a.paused) switching.current = true;
      a.pause();
      wanted.current = code;
      a.src = src;
      setActive({ code, phase: "loading" });
      claimAudio(a);
      try {
        await a.play();
      } catch (err) {
        // A newer click interrupts this one (AbortError): not a failure.
        if (wanted.current !== code || (err instanceof DOMException && err.name === "AbortError")) return;
        setActive(null);
        setFailed((f) => (f.includes(code) ? f : [...f, code]));
      }
    },
    []
  );

  return (
    <section id="hear" ref={sectionRef} className="s-container mt-28 md:mt-36" aria-labelledby="hear-title">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <Reveal as="p" className="s-eyebrow">Hear it</Reveal>
          <SplitHeading id="hear-title" text="22 Indian languages." accent="Hear five." className="h2 mt-4 max-w-[18ch]" />
        </div>
        <Reveal delay={0.1} className="max-w-[44ch]">
          <p className="lede">
            One sentence about booking an appointment, in Hindi, Marathi, Tamil, Bengali and Telugu, with English, French and Spanish for comparison. Each clip was rendered once with a preset voice and saved as a file, so what you hear is what the engines produce.
          </p>
          <p className="mt-3 s-small text-[var(--s-mute)]">
            Preset voices only. None of these is a cloned or real person&apos;s voice.
          </p>
        </Reveal>
      </div>
      <Stagger as="ul" className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {LANGUAGE_CLIPS.map((c) => {
          const phase = active?.code === c.code ? active.phase : null;
          const on = phase === "playing";
          const loading = phase === "loading";
          const bad = failed.includes(c.code);
          const font = fontsNear ? (SCRIPT_FONT[c.code] ?? "") : "";
          const verb = on ? "Pause" : loading ? "Loading" : "Play";
          return (
            <Item key={c.code} as="li" lang={c.lang} className="flex">
              <motion.div
                className="s-paper-card flex min-h-[18.75rem] w-full flex-col p-6"
                animate={{ boxShadow: on ? "0 0 0 2px var(--s-marigold), var(--s-shadow-paper)" : "var(--s-shadow-paper)" }}
                transition={{ duration: 0.3 }}
              >
              <div className="flex items-baseline justify-between gap-3">
                <p className="flex items-center gap-2.5 s-item font-medium text-[var(--s-ink)]">
                  {c.name}
                  <AnimatePresence initial={false}>{on && <Equaliser key="eq" still={still} />}</AnimatePresence>
                </p>
                {c.native !== c.name && <p className={`${font} s-body text-[var(--s-mute)]`}>{c.native}</p>}
              </div>
              <p className={`${font} mt-4 s-body leading-relaxed text-[var(--s-ink-2)]`}>{c.text}</p>
              <div className="mt-auto pt-6">
                <Magnetic strength={0.2}>
                <button
                  type="button"
                  onClick={() => toggle(c.code, c.src)}
                  disabled={bad}
                  aria-pressed={on}
                  aria-busy={loading}
                  aria-label={`${verb === "Loading" ? "Loading" : verb} the ${c.name} clip`}
                  className={`group inline-flex h-12 min-w-[9.5rem] items-center gap-2.5 rounded-full pl-5 pr-6 s-body font-medium transition-[transform,background-color] duration-200 active:scale-[0.97] disabled:opacity-40 ${
                    on ? "bg-[var(--s-marigold)] text-[var(--s-marigold-ink)] ring-2 ring-[var(--s-ink)]" : "bg-[image:var(--s-brand-gradient)] text-[#FBFAF7] shadow-[inset_0_1px_0_rgba(255,250,240,0.18),var(--s-shadow-float)]"
                  }`}
                >
                  {loading ? (
                    <Loader2 className="h-[1.125rem] w-[1.125rem] motion-safe:animate-spin" aria-hidden />
                  ) : on ? (
                    <Pause className="h-[1.125rem] w-[1.125rem]" aria-hidden />
                  ) : (
                    <Play className="h-[1.125rem] w-[1.125rem]" aria-hidden />
                  )}
                  {bad ? "Clip unavailable" : on ? "Playing, pause" : loading ? "Loading..." : `Play ${c.seconds.toFixed(1)} s`}
                </button>
                </Magnetic>
                <p className="mt-4 min-h-[6.25rem] border-t border-[var(--s-line)] pt-3 s-small leading-relaxed text-[var(--s-mute)]">
                  {ENGINE_CREDIT[c.engine]}, {c.voice}.
                  {c.engine === "indic_speak" && (
                    <>
                      <br />
                      {INDIC_ATTRIBUTION}
                    </>
                  )}
                </p>
              </div>
              </motion.div>
            </Item>
          );
        })}
      </Stagger>
    </section>
  );
}

/* ------------------------------------------------------------------ measured numbers */

/** Wrap the headline number in data-countup. For "733 / 733" or "1,390 + 404" only the first number counts up. */
function StatValue({ value }: { value: string }) {
  const m = value.match(/^([\d.,]+)(\s*[/+].*)$/);
  return (
    <p className="s-stat text-[var(--s-ink)]">
      {m ? (
        <>
          <span data-countup>{m[1]}</span>
          {m[2]}
        </>
      ) : (
        <span data-countup>{value}</span>
      )}
    </p>
  );
}

function StatGroup({ id, title, conditions, stats }: { id: string; title: string; conditions: string; stats: Stat[] }) {
  const cols = stats.length === 5 ? "lg:grid-cols-5" : "lg:grid-cols-4";
  return (
    <div className="mt-12 first-of-type:mt-10">
      <div data-reveal className="flex flex-col gap-1 md:flex-row md:items-baseline md:justify-between md:gap-6">
        <h3 id={id} className="s-item text-[var(--s-ink)]">{title}</h3>
        <p className="s-small text-[var(--s-mute)]">{conditions}</p>
      </div>
      <ul aria-labelledby={id} className={`mt-4 grid grid-cols-1 border-y border-[var(--s-line)] sm:grid-cols-2 ${cols}`}>
        {stats.map((m) => (
          <li key={m.label} data-reveal className="border-b border-[var(--s-line)] py-6 last:border-b-0 sm:pr-6 lg:border-b-0 lg:border-r lg:px-6 lg:first:pl-0 lg:last:border-r-0 lg:last:pr-0">
            <StatValue value={m.value} />
            <p className="mt-3 s-body font-medium leading-snug text-[var(--s-ink)]">{m.label}</p>
            <p className="mt-1 s-small leading-snug text-[var(--s-mute)]">{m.note}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Measured() {
  return (
    <section id="numbers" className="s-container mt-28 md:mt-36" aria-labelledby="numbers-title">
      <h2 id="numbers-title" data-reveal className="h2 max-w-[20ch]">Measured, not promised.</h2>
      <p data-reveal className="lede mt-5">{MEASURED_CONTEXT}</p>
      <StatGroup id="numbers-speed" title="Speed" conditions={MEASURED_SPEED_CONDITIONS} stats={MEASURED_SPEED} />
      <StatGroup id="numbers-trust" title="Trust" conditions={MEASURED_TRUST_CONDITIONS} stats={MEASURED_TRUST} />
      <p data-reveal className="mt-6 max-w-[78ch] s-small leading-relaxed text-[var(--s-mute)]">{MEASURED_CAVEAT}</p>
      {REPO_PUBLIC ? (
        <a data-reveal href={RESULTS_URL} className="mt-4 inline-flex min-h-[1.5rem] items-center gap-1 s-body underline underline-offset-4">
          Every number, method and the commands to rerun them <ArrowUpRight className="h-3.5 w-3.5" aria-hidden />
        </a>
      ) : (
        <p data-reveal className="mt-4 s-body text-[var(--s-ink-2)]">Every number, method and the commands to rerun them are in the project&apos;s results file, shared on request.</p>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ honest comparison */

function CompareCell({ cell, column, strong }: { cell: Cell; column: string; strong?: boolean }) {
  return (
    <div className={cell.unchecked ? "text-[var(--s-mute)]" : strong ? "text-[var(--s-ink)]" : "text-[var(--s-ink-2)]"}>
      <span className="mb-1 block s-label font-medium text-[var(--s-mute)] md:hidden">{column}</span>
      <p className={`s-body leading-relaxed ${cell.unchecked ? "italic" : ""}`}>{cell.text}</p>
      {cell.cites && (
        <p className="mt-1.5 s-small leading-snug text-[var(--s-mute)]">
          {cell.cites.map((c, i) => (
            <span key={c.href}>
              {i > 0 && "; "}
              <a href={c.href} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2">
                {c.label}
              </a>
            </span>
          ))}
        </p>
      )}
    </div>
  );
}

export function Comparison() {
  const grid = "md:grid-cols-[9.5rem_1fr_1fr_1.2fr] md:gap-6";
  return (
    <section id="compare" className="s-container mt-28 md:mt-36" aria-labelledby="compare-title">
      <h2 id="compare-title" data-reveal className="h2 max-w-[20ch]">How BonClay compares.</h2>
      <p data-reveal className="lede mt-5">
        A plain comparison with how hosted voice-agent platforms work. Our numbers are measured; theirs are quoted from their own public pages, with the month we read them.
      </p>
      <div data-reveal className="mt-10 overflow-hidden rounded-[var(--s-r-card)] border border-[var(--s-line)] bg-[var(--s-paper)]">
        <div className={`hidden gap-6 border-b border-[var(--s-line)] px-7 py-4 s-small font-medium text-[var(--s-mute)] md:grid ${grid}`}>
          <span />
          <span className="text-[var(--s-ink)]">{COMPARISON_COLUMNS[0]}</span>
          <span className="text-[var(--s-ink)]">{COMPARISON_COLUMNS[1]}</span>
          <span>{COMPARISON_COLUMNS[2]}</span>
        </div>
        <ul className="divide-y divide-[var(--s-line)]">
          {COMPARISON.map((r) => (
            <li key={r.topic} className={`grid gap-4 px-5 py-6 md:px-7 ${grid}`}>
              <p className="s-body font-medium text-[var(--s-ink)]">{r.topic}</p>
              <CompareCell cell={r.bonclay} column={COMPARISON_COLUMNS[0]} strong />
              <CompareCell cell={r.groq} column={COMPARISON_COLUMNS[1]} strong />
              <CompareCell cell={r.hosted} column={COMPARISON_COLUMNS[2]} />
            </li>
          ))}
        </ul>
      </div>
      <p data-reveal className="mt-3 max-w-[90ch] s-small leading-relaxed text-[var(--s-mute)]">{COMPARISON_FOOTNOTE}</p>
      <div data-reveal className="mt-5 rounded-[var(--s-r-card)] bg-[var(--s-marigold)] p-6 text-[var(--s-marigold-ink)] sm:p-8">
        <h3 className="s-h3">What BonClay does not do yet</h3>
        <ul className="mt-5 grid gap-5 md:grid-cols-3">
          {NOT_YET.map((n) => (
            <li key={n.topic}>
              <p className="s-body font-medium">{n.topic}</p>
              <p className="mt-1 s-body leading-relaxed text-[#3B2A10]">{n.body}</p>
            </li>
          ))}
        </ul>
      </div>
      <p className="mt-4 s-small text-[var(--s-mute)]">
        Method and raw numbers are in{" "}
        {REPO_PUBLIC ? (
          <a href={RESULTS_URL} className="underline underline-offset-4">
            docs/RESULTS.md
          </a>
        ) : (
          "the project's results file (shared on request)"
        )}
        . Try the{" "}
        <Link href="/verify" className="underline underline-offset-4">
          Verify page
        </Link>{" "}
        yourself.
      </p>
    </section>
  );
}
