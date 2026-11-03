"use client";

/* "From your voice to a signed answer": one example question followed through eight steps.
 *
 * Three modes, chosen in the browser with the same media queries the CSS uses (the server always renders "static"):
 *   static  no JavaScript, or prefers-reduced-motion: all eight steps as cards, each with its picture in its final state,
 *           and the time ruler fully drawn. Nothing moves.
 *   stack   phones, tablets and short windows: the same cards. Each card fades in as it scrolls into view, and the ruler
 *           can be replayed at real speed.
 *   pinned  wide, tall windows with motion allowed: the stage sticks (CSS position: sticky, so no pin spacer is inserted)
 *           while a ScrollTrigger scrubs one GSAP timeline through the eight steps. A step in the list jumps to it.
 * The timeline is written in clock seconds (0 = the moment the request is sent), so "Replay at real speed" is the same
 * timeline played at timeScale 1. The picture windows do not overlap; the markers on the ruler are the true times. */

import Link from "next/link";
import { Fragment, useCallback, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { AudioLines, FileText, Mic, Phone, Play, Square } from "lucide-react";
import { useLenis } from "./SmoothScroll";
import {
  CLOCK_NOTE,
  EXAMPLE,
  INPUTS,
  MARKERS,
  MEASURED_ON,
  OVERLAP_NOTE,
  RULER_MAX,
  STEPS,
  TIMING,
  WINDOWS,
  sec,
  type InputKind,
  type Step,
} from "./pipelineData";

gsap.registerPlugin(ScrollTrigger);

/* ------------------------------------------------------------------ mode */

type Mode = "static" | "stack" | "pinned";
const MQ_REDUCE = "(prefers-reduced-motion: reduce)";
const MQ_MOTION = "(prefers-reduced-motion: no-preference)";
const MQ_PIN = "(min-width: 64rem) and (min-height: 45rem) and (prefers-reduced-motion: no-preference)";

const hasMatchMedia = () => typeof window !== "undefined" && typeof window.matchMedia === "function";

function readMode(): Mode {
  // Motion needs a positive answer: reduce is off and no-preference is on (a browser that cannot say gets the static page).
  if (!hasMatchMedia() || window.matchMedia(MQ_REDUCE).matches || !window.matchMedia(MQ_MOTION).matches) return "static";
  return window.matchMedia(MQ_PIN).matches ? "pinned" : "stack";
}

function subscribeMode(onChange: () => void) {
  if (!hasMatchMedia()) return () => undefined;
  const lists = [window.matchMedia(MQ_REDUCE), window.matchMedia(MQ_MOTION), window.matchMedia(MQ_PIN)];
  lists.forEach((m) => m.addEventListener?.("change", onChange));
  return () => lists.forEach((m) => m.removeEventListener?.("change", onChange));
}

const useMode = (): Mode => useSyncExternalStore(subscribeMode, readMode, () => "static");

/* ------------------------------------------------------------------ small helpers */

const N = STEPS.length;
/** Timeline position of clock second 0. The first window starts at a negative clock time. */
const T0 = -WINDOWS.disclose[0];
const END = WINDOWS.answer[1];
const P = (clock: number) => clock + T0;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const ORDER = STEPS.map((s) => s.id);
const TYPE_FROM = 0.12;
const TYPE_TO = 0.68;

/** Scroll progress 0..1 to a clock second: each step gets an equal share of the scroll and plays its own window. */
function clockAt(progress: number): number {
  const p = clamp(progress, 0, 1) * N;
  const k = Math.min(N - 1, Math.floor(p));
  const [from, b] = WINDOWS[ORDER[k]];
  // The first step starts with its greeting already in place, so the stage is never empty when it first sticks.
  const a = k === 0 ? from + 0.55 : from;
  return a + (b - a) * (p - k);
}

/** Which step is on stage at a clock second. */
function stepAt(clock: number): number {
  let idx = 0;
  ORDER.forEach((id, i) => {
    if (clock >= WINDOWS[id][0] - 1e-6) idx = i;
  });
  return idx;
}

/** Deterministic pseudo-random bar heights (percent), loudest in the middle, the same on server and client. */
function heights(n: number, seed: number): number[] {
  return Array.from({ length: n }, (_, i) => {
    const body = Math.sin((Math.PI * (i + 0.5)) / n);
    const wobble = 0.5 + 0.5 * Math.abs(Math.sin(i * 1.9 + seed) * Math.cos(i * 0.7 + seed * 0.5));
    return Math.round(16 + 84 * body * wobble);
  });
}
const IN_BARS = heights(40, 1.3);
const OUT_BARS = heights(44, 4.1);
const MINI_IN_BARS = heights(26, 1.3);
const MINI_OUT_BARS = heights(26, 4.1);

const WORDS = EXAMPLE.reply.split(" ");
const HIT = EXAMPLE.pages.find((p) => p.hit)!;

/* ------------------------------------------------------------------ pictures (shared by the stage and the cards) */

function Seal() {
  return (
    <svg className="pj-seal" viewBox="0 0 24 24" aria-hidden focusable="false">
      <circle cx="12" cy="12" r="11" fill="var(--s-marigold)" />
      <path d="M7 12.5l3.2 3.2L17 8.8" fill="none" stroke="var(--s-marigold-ink)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Bars({ values, kind }: { values: number[]; kind: "in" | "out" }) {
  return (
    <span className="pj-bars" data-bars={kind}>
      {values.map((h, i) => (
        <span key={i} data-bar style={{ height: `${h}%` }} />
      ))}
    </span>
  );
}

function Ticks({ count }: { count: number }) {
  return (
    <span className="pj-ticks" aria-hidden>
      {Array.from({ length: count }, (_, i) => (
        <span key={i} data-tick data-long={i % 4 === 0} />
      ))}
    </span>
  );
}

function Greeting({ l }: { l?: boolean }) {
  return (
    <div className="pj-greet" {...(l ? { "data-l": "greet" } : {})}>
      <p className="pj-bubble" lang="hi-Latn">
        {EXAMPLE.greeting}
      </p>
      <span className="pj-tag" {...(l ? { "data-l": "seal" } : {})}>
        <Seal />
        AI disclosed
      </span>
    </div>
  );
}

function PageCards({ l }: { l?: boolean }) {
  return (
    <div className="pj-pages" {...(l ? { "data-l": "pages" } : {})}>
      {EXAMPLE.pages.map((p) => (
        <div key={p.id} className="pj-page" data-page data-hit={p.hit}>
          {p.hit ? (
            <span className="pj-passage" lang="en">
              <span className="pj-passage-text">
                <span>OPD · Dr. Mehta</span>
                <span>Mon-Sat 10:00-13:00</span>
              </span>
              <span className="pj-hl" aria-hidden {...(l ? { "data-l": "hl" } : {})}>
                <span>OPD · Dr. Mehta</span>
                <span>Mon-Sat 10:00-13:00</span>
              </span>
            </span>
          ) : (
            <span className="pj-skel" aria-hidden>
              <i />
              <i />
              <i />
            </span>
          )}
          <span className="pj-pname">
            {p.doc}
            <span> · p. {p.page}</span>
          </span>
          {p.hit && l && <span className="pj-conn" data-l="conn" aria-hidden />}
        </div>
      ))}
    </div>
  );
}

function Reply({ l }: { l?: boolean }) {
  return (
    <div className="pj-replywrap" {...(l ? { "data-l": "reply" } : {})}>
      <p className="pj-reply" lang="hi-Latn">
        {WORDS.map((w, i) => (
          <Fragment key={i}>
            <span data-w>{w}</span>
            {i < WORDS.length - 1 ? " " : ""}
          </Fragment>
        ))}
      </p>
      <p className="pj-src" {...(l ? { "data-l": "src" } : {})}>
        From {HIT.doc}, page {HIT.page}
      </p>
    </div>
  );
}

function Receipt({ l }: { l?: boolean }) {
  const r = EXAMPLE.receipt;
  return (
    <div className="pj-receipt" {...(l ? { "data-l": "receipt" } : {})}>
      <p className="pj-r-head">
        <Seal />
        <span>
          Receipt <code>{r.id}</code>
        </span>
      </p>
      <dl className="pj-r-rows">
        <div>
          <dt>Signature</dt>
          <dd>{r.signed}</dd>
        </div>
        <div>
          <dt>Engine</dt>
          <dd>{r.engine}</dd>
        </div>
        <div>
          <dt>Consent</dt>
          <dd>{r.consent}</dd>
        </div>
      </dl>
      <p className="pj-r-foot">{r.check}</p>
    </div>
  );
}

function InputIcon({ input }: { input: InputKind }) {
  return input === "phone" ? <Phone size={18} aria-hidden /> : <Mic size={18} aria-hidden />;
}

/* ------------------------------------------------------------------ the shared stage (wide screens, motion allowed) */

function Stage({ input, current }: { input: InputKind; current: number | null }) {
  return (
    <div className="pj-stage" aria-hidden="true">
      <p className="pj-stage-tag">
        Example: a clinic agent and a made-up caller <span className="pj-tag-n">Step {(current ?? 0) + 1} of {N}</span>
      </p>
      <div className="pj-lanes">
        <div className="pj-lane">
          <span className="pj-lane-icon" data-l="icon">
            <InputIcon input={input} />
          </span>
          <div className="pj-lane-body">
            <p className="pj-lane-title">Caller</p>
            <div className="pj-slot pj-slot-wave">
              <div className="pj-wave" data-l="wave">
                <Bars values={IN_BARS} kind="in" />
              </div>
              <div className="pj-ask-row" data-l="ask">
                <p className="pj-ask" lang="hi-Latn">
                  <span data-type>{EXAMPLE.question}</span>
                  <span className="pj-caret" data-caret />
                </p>
                <span className="pj-chip" data-l="chip-asr">
                  {sec(TIMING.asr)}
                </span>
              </div>
            </div>
            <p className="pj-cap" data-l="cap">
              {INPUTS[input].caption}
            </p>
          </div>
        </div>

        <div className="pj-lane pj-lane-docs">
          <span className="pj-lane-icon">
            <FileText size={18} aria-hidden />
          </span>
          <div className="pj-lane-body" data-l="docs">
            <p className="pj-lane-title">
              Your PDFs <span className="pj-chip" data-l="chip-find">{sec(TIMING.search)}</span>
            </p>
            <PageCards l />
          </div>
        </div>

        <div className="pj-lane">
          <span className="pj-lane-icon">
            <AudioLines size={18} aria-hidden />
          </span>
          <div className="pj-lane-body">
            <p className="pj-lane-title">BonClay</p>
            <div className="pj-slot pj-slot-agent">
              <Greeting l />
              <Reply l />
              <div className="pj-out" data-l="out">
                <div className="pj-outwave" data-l="outwave">
                  <Bars values={OUT_BARS} kind="out" />
                </div>
                <Ticks count={44} />
                <div className="pj-outcaps">
                  <p data-l="cap-a">Voice made in {sec(TIMING.voice)}</p>
                  <p data-l="cap-b">
                    <span className="pj-dot" aria-hidden />
                    Inaudible watermark added
                  </p>
                  <p data-l="cap-c">
                    Played to the caller. First sound {sec(TIMING.firstAudio)}, first answer sentence {sec(TIMING.firstAnswer)} (median).
                  </p>
                </div>
              </div>
              <Receipt l />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ the cards' pictures (final state) */

function Mini({ id, input }: { id: Step["id"]; input: InputKind }) {
  switch (id) {
    case "disclose":
      return <Greeting />;
    case "speak":
      return (
        <div className="pj-mini-speak">
          <span className="pj-lane-icon">
            <InputIcon input={input} />
          </span>
          <div>
            <div className="pj-wave">
              <Bars values={MINI_IN_BARS} kind="in" />
            </div>
            <p className="pj-cap">{INPUTS[input].caption}</p>
          </div>
        </div>
      );
    case "listen":
      return (
        <div className="pj-ask-row">
          <p className="pj-ask" lang="hi-Latn">
            {EXAMPLE.question}
          </p>
          <span className="pj-chip">{sec(TIMING.asr)}</span>
        </div>
      );
    case "find":
      return (
        <div>
          <PageCards />
          <p className="pj-cap">
            Search took <span className="pj-chip">{sec(TIMING.search)}</span>
          </p>
        </div>
      );
    case "write":
      return <Reply />;
    case "talk":
      return (
        <div>
          <div className="pj-outwave">
            <Bars values={MINI_OUT_BARS} kind="out" />
          </div>
          <p className="pj-cap">Voice made in {sec(TIMING.voice)}</p>
        </div>
      );
    case "sign":
      return (
        <div className="pj-mini-sign">
          <div>
            <div className="pj-outwave">
              <Bars values={MINI_OUT_BARS} kind="out" />
            </div>
            <Ticks count={26} />
            <p className="pj-cap">
              <span className="pj-dot" aria-hidden />
              Inaudible watermark added
            </p>
          </div>
          <Receipt />
        </div>
      );
    case "answer":
      return (
        <div>
          <div className="pj-outwave">
            <Bars values={MINI_OUT_BARS} kind="out" />
          </div>
          <p className="pj-cap">
            Played to the caller. First sound {sec(TIMING.firstAudio)}, first answer sentence {sec(TIMING.firstAnswer)} (median).
          </p>
        </div>
      );
  }
}

/* ------------------------------------------------------------------ the time ruler */

function Ruler({
  mode,
  status,
  onReplay,
}: {
  mode: Mode;
  status: "idle" | "playing" | "done";
  onReplay: () => void;
}) {
  const statusText = status === "playing" ? "Playing at real speed" : status === "done" ? "Done" : "";
  const side = (id: string) => (id === "asr" ? "left" : id === "first-sound" ? "right" : id === "first-answer" ? "mid" : "end");
  return (
    <div className="pj-ruler">
      <div className="pj-rhead">
        <p className="pj-clock">
          Time since the request was sent <span data-clock>{sec(TIMING.fullReply)}</span>
        </p>
        {mode !== "static" && (
          <div className="pj-replay">
            <button type="button" className="pj-btn" aria-pressed={status === "playing"} onClick={onReplay}>
              {status === "playing" ? <Square size={14} aria-hidden /> : <Play size={14} aria-hidden />}
              Replay at real speed
            </button>
            <span role="status" aria-live="polite" className="pj-status">
              {statusText}
            </span>
          </div>
        )}
      </div>
      <div
        className="pj-rtrack"
        role="img"
        aria-label={`Time ruler, 0 to 3 seconds. ${MARKERS.map((m) => `${m.long} at ${m.value}`).join(". ")}.`}
      >
        <span className="pj-rline" />
        <span className="pj-rfill" data-fill style={{ transform: `scaleX(${TIMING.fullReply / RULER_MAX})` }} />
        {[0, 1, 2].map((s) => (
          <span key={s} className="pj-raxis" style={{ left: `${(s / RULER_MAX) * 100}%` }}>
            {s} s
          </span>
        ))}
        {MARKERS.map((m) => (
          <span
            key={m.id}
            className="pj-mark"
            data-m={m.id}
            data-at={m.at}
            data-on="true"
            data-side={side(m.id)}
            data-trust={m.trust ? "true" : undefined}
            style={{ left: `${(m.at / RULER_MAX) * 100}%` }}
          >
            <span className="pj-mdot" />
            <span className="pj-mtext">
              <span>{m.label}</span>
              <span className="pj-mval">{m.value}</span>
            </span>
          </span>
        ))}
      </div>
      <p className="pj-note">{OVERLAP_NOTE}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ the section */

type Status = "idle" | "playing" | "done";

export function PipelineJourney() {
  const mode = useMode();
  const lenis = useLenis();
  const trackRef = useRef<HTMLDivElement>(null);
  const pinRef = useRef<HTMLDivElement>(null);
  const pjRef = useRef<HTMLDivElement>(null);
  const tlRef = useRef<gsap.core.Timeline | null>(null);
  const stRef = useRef<ScrollTrigger | null>(null);
  const replaying = useRef(false);
  const lock = useRef(false);
  const progress = useRef(0);
  const syncRef = useRef<() => void>(() => undefined);
  const [current, setCurrent] = useState<number | null>(null);
  const [input, setInput] = useState<InputKind>("browser");
  const [status, setStatus] = useState<Status>("idle");

  useLayoutEffect(() => {
    if (mode === "static") return;
    const root = pjRef.current;
    const track = trackRef.current;
    const pin = pinRef.current;
    if (!root || !track || !pin) return;
    const pinned = mode === "pinned";
    const q = <T extends HTMLElement>(sel: string) => root.querySelector<T>(sel);
    const qa = <T extends HTMLElement>(sel: string) => Array.from(root.querySelectorAll<T>(sel));
    const clockEl = q("[data-clock]");
    const fillEl = q("[data-fill]");
    const marks = qa("[data-m]");
    const typeEl = q("[data-type]");
    const caretEl = q("[data-caret]");
    let lastIdx = -1;
    let alive = true;
    let offRefresh: () => void = () => undefined;

    const finalRuler = () => {
      if (clockEl) clockEl.textContent = sec(TIMING.fullReply);
      if (fillEl) fillEl.style.transform = `scaleX(${TIMING.fullReply / RULER_MAX})`;
      marks.forEach((m) => m.setAttribute("data-on", "true"));
    };

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ paused: true, defaults: { ease: "expo.out" } });
      tlRef.current = tl;

      /** Everything that follows the clock and is not a tween: the readout, the ruler, the typed text, the current step. */
      const render = () => {
        const c = tl.time() - T0;
        const shown = clamp(c, 0, TIMING.fullReply);
        if (clockEl) clockEl.textContent = sec(shown);
        if (fillEl) fillEl.style.transform = `scaleX(${shown / RULER_MAX})`;
        marks.forEach((m) => m.setAttribute("data-on", String(c >= Number(m.dataset.at) - 1e-6)));
        if (typeEl) {
          const n = Math.round(clamp((c - TYPE_FROM) / (TYPE_TO - TYPE_FROM), 0, 1) * EXAMPLE.question.length);
          typeEl.textContent = EXAMPLE.question.slice(0, n);
          if (caretEl) caretEl.style.opacity = n > 0 && n < EXAMPLE.question.length ? "1" : "0";
        }
        const idx = stepAt(c);
        if (idx !== lastIdx) {
          lastIdx = idx;
          if (!lock.current && alive) setCurrent(idx);
        }
      };
      tl.eventCallback("onUpdate", render);

      if (pinned) {
        const stage = q(".pj-stage")!;
        const l = (name: string) => stage.querySelector<HTMLElement>(`[data-l="${name}"]`)!;
        const all = (sel: string) => Array.from(stage.querySelectorAll<HTMLElement>(sel));
        const inBars = all('[data-bars="in"] [data-bar]');
        const outBars = all('[data-bars="out"] [data-bar]');
        const ticks = all("[data-tick]");
        const pages = all("[data-page]");
        const otherPages = pages.filter((p) => p.dataset.hit !== "true");
        const words = all("[data-w]");

        // 1 It says it is an AI
        tl.fromTo(l("greet"), { opacity: 0, y: 12 }, { opacity: 1, y: 0, duration: 0.55 }, P(-1.5));
        tl.fromTo(l("seal"), { opacity: 0, scale: 0.7 }, { opacity: 1, scale: 1, duration: 0.5 }, P(-1.3));
        // 2 You speak
        tl.fromTo(l("icon"), { opacity: 0, scale: 0.8 }, { opacity: 1, scale: 1, duration: 0.45 }, P(-0.8));
        tl.fromTo(l("wave"), { opacity: 0 }, { opacity: 1, duration: 0.2 }, P(-0.8));
        tl.fromTo(inBars, { scaleY: 0.08 }, { scaleY: 1, duration: 0.5, stagger: { each: 0.01 } }, P(-0.78));
        tl.fromTo(l("cap"), { opacity: 0 }, { opacity: 1, duration: 0.3 }, P(-0.55));
        // 3 It listens: the bars collapse, the transcript is typed (see render)
        tl.to(l("cap"), { opacity: 0, duration: 0.2, ease: "power3.out" }, P(0.02));
        tl.to(inBars, { scaleY: 0.06, duration: 0.3, ease: "power3.out", stagger: { each: 0.006, from: "edges" } }, P(0));
        tl.to(l("wave"), { opacity: 0, duration: 0.2, ease: "power3.out" }, P(0.3));
        tl.fromTo(l("ask"), { opacity: 0 }, { opacity: 1, duration: 0.25 }, P(TYPE_FROM));
        tl.fromTo(l("chip-asr"), { opacity: 0, y: 6 }, { opacity: 1, y: 0, duration: 0.4 }, P(TIMING.asr - 0.35));
        // 4 It finds the answer: pages fan out, one passage is picked out
        tl.fromTo(
          pages,
          { opacity: 0, y: 14, scale: 0.94, rotate: (i: number) => (i - 1.5) * -5, transformOrigin: "50% 100%" },
          { opacity: 1, y: 0, scale: 1, rotate: 0, duration: 0.6, stagger: 0.06 },
          P(TIMING.asr),
        );
        tl.fromTo(l("chip-find"), { opacity: 0 }, { opacity: 1, duration: 0.3 }, P(0.95));
        tl.fromTo(l("hl"), { opacity: 0 }, { opacity: 1, duration: 0.3 }, P(1.05));
        tl.to(otherPages, { opacity: 0.45, duration: 0.3 }, P(1.05));
        // 5 It writes the reply: it streams in, tied back to the passage
        tl.to(l("greet"), { opacity: 0, y: -8, duration: 0.25, ease: "power3.out" }, P(WINDOWS.write[0]));
        tl.fromTo(words, { opacity: 0, y: 6 }, { opacity: 1, y: 0, duration: 0.3, stagger: 0.05 }, P(1.4));
        tl.fromTo(l("conn"), { scaleY: 0 }, { scaleY: 1, duration: 0.45, transformOrigin: "50% 100%" }, P(1.5));
        tl.fromTo(l("src"), { opacity: 0 }, { opacity: 1, duration: 0.3 }, P(1.85));
        // 6 It speaks: the pages leave, the reply text turns into the outgoing waveform
        tl.to(l("docs"), { opacity: 0, duration: 0.25, ease: "power3.out" }, P(TIMING.firstAnswer));
        tl.to(l("reply"), { opacity: 0, scaleY: 0.15, duration: 0.3, ease: "power3.out", transformOrigin: "50% 50%" }, P(TIMING.firstAnswer));
        tl.to(l("ask"), { opacity: 0.55, duration: 0.3 }, P(TIMING.firstAnswer));
        tl.fromTo(l("out"), { opacity: 0 }, { opacity: 1, duration: 0.15 }, P(2.3));
        tl.fromTo(outBars, { scaleY: 0.05, opacity: 0.4 }, { scaleY: 1, opacity: 0.4, duration: 0.5, stagger: { each: 0.008 } }, P(2.3));
        tl.fromTo(l("cap-a"), { opacity: 0 }, { opacity: 1, duration: 0.3 }, P(2.5));
        // 7 Marked and signed: the tick pattern goes over the waveform, the receipt slides out
        tl.to(l("cap-a"), { opacity: 0, duration: 0.2, ease: "power3.out" }, P(WINDOWS.sign[0]));
        tl.fromTo(ticks, { scaleY: 0 }, { scaleY: 1, duration: 0.35, stagger: { each: 0.006 } }, P(WINDOWS.sign[0]));
        tl.fromTo(l("cap-b"), { opacity: 0 }, { opacity: 1, duration: 0.3 }, P(2.9));
        tl.fromTo(l("receipt"), { opacity: 0, x: 24 }, { opacity: 1, x: 0, duration: 0.6 }, P(2.85));
        // 8 It answers: the caller hears it
        tl.to(l("cap-b"), { opacity: 0, duration: 0.2, ease: "power3.out" }, P(WINDOWS.answer[0]));
        tl.to(outBars, { opacity: 1, duration: 0.3, stagger: { each: 0.01 } }, P(WINDOWS.answer[0]));
        tl.fromTo(l("cap-c"), { opacity: 0 }, { opacity: 1, duration: 0.3 }, P(3.5));
      }
      // Hold the timeline open to the end of the last window.
      tl.to({}, { duration: 0.01 }, P(END) - 0.01);

      const sync = () => {
        if (replaying.current) return;
        if (pinned && stRef.current) tl.time(P(clockAt(progress.current)), false);
        else tl.time(tl.duration(), false);
      };
      syncRef.current = sync;

      if (pinned) {
        const top = () => parseFloat(getComputedStyle(pin).top) || 80;
        stRef.current = ScrollTrigger.create({
          trigger: track,
          start: () => `top top+=${top()}`,
          end: () => `bottom top+=${top() + pin.offsetHeight}`,
          invalidateOnRefresh: true,
          onUpdate: (self) => {
            progress.current = self.progress;
            sync();
          },
          onRefresh: (self) => {
            progress.current = self.progress;
            sync();
          },
        });
        progress.current = stRef.current.progress;
      }
      sync();
      render();

      if (!pinned) {
        // Phones, tablets, short windows: each card eases in as it reaches the viewport.
        const items = qa(".pj-item");
        const show = (els: Element[]) => {
          gsap.fromTo(els, { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.8, ease: "expo.out", stagger: 0.08, overwrite: true, clearProps: "transform" });
          els.forEach((el) => {
            const bars = Array.from(el.querySelectorAll("[data-bar]"));
            const ticks = Array.from(el.querySelectorAll("[data-tick]"));
            const words = Array.from(el.querySelectorAll("[data-w]"));
            if (bars.length) gsap.from(bars, { scaleY: 0.1, duration: 0.6, ease: "expo.out", stagger: { each: 0.012 }, delay: 0.15, clearProps: "transform" });
            if (ticks.length) gsap.from(ticks, { scaleY: 0, duration: 0.4, ease: "expo.out", stagger: { each: 0.01 }, delay: 0.4, clearProps: "transform" });
            if (words.length) gsap.from(words, { opacity: 0, y: 6, duration: 0.3, ease: "expo.out", stagger: 0.05, delay: 0.2, clearProps: "transform" });
          });
        };
        ScrollTrigger.batch(items, { start: "top 88%", once: true, batchMax: 4, interval: 0.08, onEnter: show, onEnterBack: show });
        // A deep link can land below the cards: anything already scrolled past is shown.
        const revealPast = () =>
          items.forEach((el) => {
            if (el.style.opacity !== "1" && el.getBoundingClientRect().bottom < 0) gsap.set(el, { opacity: 1 });
          });
        ScrollTrigger.addEventListener("refresh", revealPast);
        offRefresh = () => ScrollTrigger.removeEventListener("refresh", revealPast);
      }
    }, root);

    return () => {
      alive = false;
      offRefresh();
      ctx.revert();
      tlRef.current = null;
      stRef.current = null;
      replaying.current = false;
      lock.current = false;
      syncRef.current = () => undefined;
      finalRuler();
      if (typeEl) typeEl.textContent = EXAMPLE.question;
      setCurrent(null);
      setStatus("idle");
    };
  }, [mode]);

  const stopReplay = useCallback(() => {
    const tl = tlRef.current;
    replaying.current = false;
    if (tl) tl.pause();
    syncRef.current();
    setStatus("idle");
    if (mode !== "pinned") setCurrent(null);
  }, [mode]);

  const toggleReplay = useCallback(() => {
    const tl = tlRef.current;
    if (status === "playing") {
      stopReplay();
      return;
    }
    replaying.current = true;
    setStatus("playing");
    if (!tl) return;
    tl.eventCallback("onComplete", () => {
      replaying.current = false;
      setStatus("done");
      if (mode !== "pinned") setCurrent(null);
    });
    tl.play(0);
  }, [status, stopReplay, mode]);

  const goTo = useCallback(
    (i: number) => {
      if (replaying.current) stopReplay();
      setCurrent(i);
      const st = stRef.current;
      if (!st) return;
      // Land at the end of that step's share of the scroll, so its picture is complete.
      const y = st.start + (st.end - st.start) * Math.min(1, (i + 0.98) / N);
      lock.current = true;
      const unlock = () => {
        lock.current = false;
        syncRef.current();
      };
      window.setTimeout(unlock, 1100);
      if (lenis) lenis.scrollTo(y, { duration: 0.9, onComplete: unlock });
      else window.scrollTo({ top: y, behavior: "smooth" });
    },
    [lenis, stopReplay],
  );

  return (
    <section id="how" aria-labelledby="how-title" className="s-container mt-28 md:mt-36">
      <style>{CSS}</style>
      <h2 id="how-title" className="h2 max-w-[22ch]">
        From your voice to a signed answer.
      </h2>
      <p className="lede mt-5 max-w-[56ch]">
        Follow one question through 8 steps. The caller hears the first sound in {sec(TIMING.firstAudio)}, and the first sentence of the answer in {sec(TIMING.firstAnswer)}. The two marked trust steps are the AI disclosure, and the watermark with a signed receipt.
      </p>
      <p className="sr-only" lang="hi-Latn">
        Example, not a real caller: a clinic agent is asked, &ldquo;{EXAMPLE.question}&rdquo; It finds the passage &ldquo;{EXAMPLE.passage}&rdquo; in the clinic&apos;s PDF and answers, &ldquo;{EXAMPLE.reply}&rdquo;
      </p>

      <div ref={trackRef} className="pj-track mt-12 md:mt-14" data-mode={mode}>
        <div ref={pinRef} className="pj-pin">
          <div ref={pjRef} className="pj" data-mode={mode}>
            {mode === "pinned" && <Stage input={input} current={current} />}
            <Ruler mode={mode} status={status} onReplay={toggleReplay} />
            <ol className="pj-list" aria-label="The 8 steps">
              {STEPS.map((step, i) => {
                const state = current === null ? "todo" : i < current ? "done" : i === current ? "current" : "todo";
                return (
                  <li key={step.id} className="pj-item" data-step={step.id} data-trust={step.trust} data-state={state} data-current={state === "current"}>
                    <button type="button" className="pj-step" aria-current={state === "current" ? "step" : undefined} onClick={() => goTo(i)}>
                      <span className="pj-n" aria-hidden>
                        {i + 1}
                      </span>
                      <span className="pj-step-text">
                        <span className="pj-name">
                          {step.name}
                          {step.trust && <span className="pj-badge">Trust step</span>}
                        </span>
                        <span className="pj-line">{step.line}</span>
                      </span>
                      {step.time && <span className="pj-chip pj-step-chip">{step.time}</span>}
                    </button>
                    <p className="pj-body">
                      {step.body}
                      {step.id === "sign" && (
                        <>
                          {" "}
                          <Link href="/verify" className="underline underline-offset-4">
                            Check a clip at /verify
                          </Link>
                          .
                        </>
                      )}
                    </p>
                    {step.id === "speak" && (
                      <div className="pj-seg" role="group" aria-label="How the caller connects">
                        {(Object.keys(INPUTS) as InputKind[]).map((k) => (
                          <button key={k} type="button" aria-pressed={input === k} onClick={() => setInput(k)}>
                            <InputIcon input={k} />
                            {INPUTS[k].label}
                          </button>
                        ))}
                      </div>
                    )}
                    <div className="pj-mini" aria-hidden="true">
                      <Mini id={step.id} input={input} />
                    </div>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>
      </div>

      <p data-reveal className="mt-8 max-w-[70ch] text-[0.875rem] leading-snug text-[var(--s-mute)]">
        {MEASURED_ON} {CLOCK_NOTE} Details are on the{" "}
        <Link href="/proof" className="underline underline-offset-4">
          Proof page
        </Link>
        .
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------ styles */

const CSS = `
.pj-track{--pj-h:min(46rem,calc(100svh - 7rem))}
.pj{--pj-gap:1.25rem;--pj-title-h:1.25rem;display:grid;gap:1.75rem;grid-template-columns:minmax(0,1fr);grid-template-areas:"ruler" "list"}
.pj-stage{display:none;grid-area:stage}
.pj-ruler{grid-area:ruler}
.pj-list{grid-area:list;display:grid;grid-template-columns:minmax(0,1fr);gap:1rem;margin:0;padding:0;list-style:none}
@media (min-width:48rem){.pj-list{grid-template-columns:repeat(2,minmax(0,1fr))}}

/* ruler */
.pj-rhead{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.75rem 1.5rem}
.pj-clock{font-size:.9375rem;color:var(--s-mute)}
.pj-clock [data-clock]{display:inline-block;min-width:4.25rem;margin-left:.375rem;font-weight:500;color:var(--s-ink);font-variant-numeric:tabular-nums}
.pj-replay{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem 1rem}
.pj-btn{display:inline-flex;align-items:center;gap:.5rem;min-height:2.75rem;padding:0 1.125rem;border:1px solid var(--s-line);border-radius:var(--s-r-pill);background:var(--s-paper);color:var(--s-ink);font-size:.9375rem;font-weight:500;cursor:pointer;transition:border-color .2s,background-color .2s}
.pj-btn:hover{border-color:var(--s-ink)}
.pj-btn[aria-pressed="true"]{background:var(--s-ink);border-color:var(--s-ink);color:#fff}
.pj-status{min-width:9rem;font-size:.875rem;color:var(--s-mute)}
.pj-rtrack{position:relative;height:6rem;margin-top:.5rem}
.pj-rline,.pj-rfill{position:absolute;left:0;right:0;top:2.5rem;height:.25rem;border-radius:var(--s-r-pill)}
.pj-rline{background:var(--s-line)}
.pj-rfill{background:var(--s-ink);transform-origin:0 50%}
.pj-raxis{position:absolute;top:3.125rem;font-size:.75rem;line-height:1;color:var(--s-mute);font-variant-numeric:tabular-nums}
.pj-mark{position:absolute;top:2.5rem;width:0;height:.25rem}
.pj-mdot{position:absolute;left:-.4375rem;top:-.3125rem;width:.875rem;height:.875rem;border-radius:var(--s-r-pill);border:2px solid var(--s-bar);background:var(--s-paper);transition:background-color .2s,border-color .2s}
.pj-mark[data-on="true"] .pj-mdot{border-color:var(--s-ink);background:var(--s-ink)}
.pj-mark[data-trust="true"][data-on="true"] .pj-mdot{border-color:var(--s-ink);background:var(--s-marigold)}
.pj-mtext{position:absolute;display:flex;flex-direction:column;font-size:.75rem;line-height:1.2;color:var(--s-mute);white-space:nowrap;transition:color .2s}
.pj-mark[data-on="true"] .pj-mtext{color:var(--s-ink)}
.pj-mval{font-weight:600;font-variant-numeric:tabular-nums}
.pj-mark[data-side="left"] .pj-mtext{right:.5rem;bottom:.875rem;align-items:flex-end;text-align:right}
.pj-mark[data-side="right"] .pj-mtext{left:.5rem;bottom:.875rem}
.pj-mark[data-side="mid"] .pj-mtext{left:0;bottom:.875rem;transform:translateX(-50%);align-items:center;text-align:center}
.pj-mark[data-side="end"] .pj-mtext{right:-.25rem;top:1.375rem;align-items:flex-end;text-align:right}
.pj-note{margin-top:.25rem;font-size:.8125rem;line-height:1.4;color:var(--s-mute)}

/* steps (cards) */
.pj-item{position:relative;display:flex;flex-direction:column;gap:.75rem;padding:1.25rem;border:1px solid var(--s-line);border-radius:var(--s-r-card);background:var(--s-paper);transition:border-color .2s}
.pj-item[data-current="true"]{border-color:var(--s-ink)}
.pj-step{display:flex;align-items:flex-start;gap:.75rem;width:100%;min-height:2.75rem;padding:0;border:0;background:none;color:inherit;text-align:left;cursor:pointer}
.pj-n{display:grid;flex:none;place-items:center;width:1.75rem;height:1.75rem;border:1px solid var(--s-bar);border-radius:var(--s-r-pill);background:var(--s-paper);font-size:.8125rem;font-variant-numeric:tabular-nums;color:var(--s-ink-2);transition:background-color .2s,color .2s,border-color .2s}
.pj-item[data-trust="true"] .pj-n{border-color:var(--s-marigold);background:var(--s-marigold);color:var(--s-marigold-ink)}
.pj-item[data-state="current"] .pj-n,.pj-item[data-state="done"] .pj-n{border-color:var(--s-ink);background:var(--s-ink);color:#fff}
.pj-item[data-trust="true"][data-state="current"] .pj-n,.pj-item[data-trust="true"][data-state="done"] .pj-n{border-color:var(--s-ink);background:var(--s-marigold);color:var(--s-marigold-ink)}
.pj-step-text{display:flex;flex:1;flex-direction:column;min-width:0}
.pj-name{font-size:1.0625rem;font-weight:500;line-height:1.3;letter-spacing:-.01em;color:var(--s-ink)}
.pj-badge{display:inline-block;margin-left:.5rem;padding:.0625rem .5rem;border-radius:var(--s-r-pill);background:var(--s-marigold);color:var(--s-marigold-ink);font-size:.75rem;font-weight:600;letter-spacing:0;vertical-align:.0625rem}
.pj-line{display:none;font-size:.875rem;line-height:1.35;color:var(--s-mute)}
.pj-body{font-size:.9375rem;line-height:1.55;color:var(--s-mute)}
.pj-body a{color:var(--s-ink)}
.pj-mini{padding:1rem;border-radius:var(--s-r-inner);background:var(--s-ground);overflow:hidden}
.pj-step-chip{margin-left:auto}

/* pictures */
.pj-chip{display:inline-block;padding:.125rem .625rem;border-radius:var(--s-r-pill);background:var(--s-fill);font-size:.8125rem;font-weight:500;line-height:1.4;font-variant-numeric:tabular-nums;color:var(--s-ink);white-space:nowrap}
.pj-seal{flex:none;width:1.25rem;height:1.25rem}
.pj-greet{display:flex;flex-wrap:wrap;align-items:center;gap:.625rem .875rem}
.pj-bubble{padding:.625rem .875rem;border-radius:var(--s-r-inner);border-top-left-radius:.25rem;background:var(--s-paper);border:1px solid var(--s-line);font-size:1rem;line-height:1.35;color:var(--s-ink)}
.pj-tag{display:inline-flex;align-items:center;gap:.375rem;padding:.1875rem .625rem .1875rem .3125rem;border-radius:var(--s-r-pill);background:var(--s-marigold);color:var(--s-marigold-ink);font-size:.8125rem;font-weight:600;line-height:1.3}
.pj-wave,.pj-outwave{height:3.25rem}
.pj-bars{display:flex;align-items:center;gap:.1875rem;width:100%;height:100%}
.pj-bars>span{flex:1 1 0;min-width:.125rem;max-width:.375rem;border-radius:var(--s-r-pill);background:var(--s-ink);transform-origin:50% 50%}
.pj-ticks{display:flex;align-items:flex-end;gap:.1875rem;width:100%;height:.75rem;margin-top:.25rem}
.pj-ticks>span{flex:1 1 0;min-width:.0625rem;max-width:.375rem;height:55%;background:var(--s-marigold);transform-origin:50% 100%}
.pj-ticks>span[data-long="true"]{height:100%}
.pj-cap{margin-top:.5rem;font-size:.8125rem;line-height:1.4;color:var(--s-mute)}
.pj-dot{display:inline-block;width:.5rem;height:.5rem;margin-right:.375rem;border-radius:var(--s-r-pill);background:var(--s-marigold);box-shadow:0 0 0 1px var(--s-ink)}
.pj-ask-row{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem .75rem}
.pj-ask{font-size:1.0625rem;font-weight:500;line-height:1.35;letter-spacing:-.01em;color:var(--s-ink)}
.pj-caret{display:inline-block;width:.125rem;height:1.0625rem;margin-left:.125rem;vertical-align:-.1875rem;background:var(--s-ink);opacity:0}
.pj-pages{position:relative;display:grid;grid-template-columns:1fr 1.7fr 1fr 1fr;gap:.625rem}
.pj-page{position:relative;display:flex;flex-direction:column;justify-content:space-between;gap:.5rem;min-height:4.5rem;padding:.5rem .625rem;border:1px solid var(--s-line);border-radius:var(--s-r-chip);background:var(--s-paper)}
.pj-page[data-hit="true"]{border-color:var(--s-ink)}
.pj-skel i{display:block;height:.25rem;margin-bottom:.25rem;border-radius:var(--s-r-pill);background:var(--s-bar)}
.pj-skel i:nth-child(2){width:80%}
.pj-skel i:nth-child(3){width:55%}
.pj-pname{font-size:.75rem;line-height:1.25;color:var(--s-mute)}
.pj-pname span{white-space:nowrap}
.pj-passage{position:relative;display:block;font-size:.8125rem;line-height:1.3;color:var(--s-ink-2)}
.pj-passage-text>span,.pj-hl>span{display:block}
.pj-hl{position:absolute;inset:-.125rem -.3125rem;padding:.125rem .3125rem;border-radius:.25rem;background:var(--s-ink);color:#fff}
.pj-conn{position:absolute;left:50%;top:100%;width:1px;height:calc(var(--pj-gap) + var(--pj-title-h));background:var(--s-ink);pointer-events:none}
.pj-replywrap{padding-top:.125rem}
.pj-reply{font-size:1.125rem;font-weight:500;line-height:1.35;letter-spacing:-.01em;color:var(--s-ink)}
.pj-src{margin-top:.375rem;font-size:.8125rem;color:var(--s-mute)}
.pj-outcaps{position:relative;display:grid;margin-top:.375rem;font-size:.8125rem;line-height:1.4;color:var(--s-mute)}
.pj-outcaps>p{grid-area:1/1}
.pj-receipt{width:13.5rem;padding:.625rem .75rem .625rem 1rem;border:1px solid var(--s-line);border-left:.25rem solid var(--s-marigold);border-radius:var(--s-r-chip);background:var(--s-ground);font-size:.75rem;line-height:1.35;color:var(--s-ink-2)}
.pj-r-head{display:flex;align-items:center;gap:.5rem;font-weight:600;color:var(--s-ink)}
.pj-r-head code{font-family:inherit;font-weight:500}
.pj-r-rows{margin:.375rem 0 0}
.pj-r-rows>div{display:flex;justify-content:space-between;gap:.5rem}
.pj-r-rows dt{color:var(--s-mute)}
.pj-r-rows dd{margin:0;text-align:right;color:var(--s-ink)}
.pj-r-foot{margin-top:.375rem;font-weight:600;color:var(--s-ink)}
.pj-mini-speak{display:flex;align-items:flex-start;gap:.875rem}
.pj-mini-speak>div:last-child{flex:1;min-width:0}
.pj-mini-sign{display:grid;gap:.875rem}
.pj-lane-icon{display:grid;flex:none;place-items:center;width:2.5rem;height:2.5rem;border-radius:var(--s-r-pill);background:var(--s-fill);color:var(--s-ink)}
.pj-mini .pj-receipt{width:100%}
.pj-mini .pj-pname span{display:none}
.pj-mini .pj-pages{grid-template-columns:repeat(2,minmax(0,1fr))}
.pj-mini .pj-passage{font-size:.75rem}
.pj-seg{display:inline-flex;align-self:flex-start;padding:.1875rem;border:1px solid var(--s-line);border-radius:var(--s-r-pill);background:var(--s-paper)}
.pj-seg button{display:inline-flex;align-items:center;gap:.375rem;min-height:2.5rem;padding:0 .875rem;border:0;border-radius:var(--s-r-pill);background:none;color:var(--s-ink-2);font-size:.875rem;font-weight:500;cursor:pointer;transition:background-color .2s,color .2s}
.pj-seg button[aria-pressed="true"]{background:var(--s-ink);color:#fff}

/* the wide, tall, motion-allowed layout: the stage sticks while the timeline plays */
.pj-track[data-mode="pinned"]{height:calc(var(--pj-h) + 360svh)}
.pj-track[data-mode="pinned"] .pj-pin{position:sticky;top:max(5rem,calc((100svh - var(--pj-h)) / 2));height:var(--pj-h)}
.pj[data-mode="pinned"]{height:100%;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);grid-template-rows:minmax(0,1fr) auto;grid-template-areas:"stage list" "ruler list";column-gap:3rem;row-gap:1rem}
.pj[data-mode="pinned"] .pj-stage{display:grid;position:relative;padding:1.25rem 1.5rem;border:1px solid var(--s-line);border-radius:var(--s-r-card);background:var(--s-paper);overflow:hidden}
.pj-stage-tag{position:absolute;top:.875rem;right:1.25rem;display:flex;gap:.75rem;font-size:.75rem;line-height:1.25rem;color:var(--s-mute)}
.pj-tag-n{font-variant-numeric:tabular-nums;color:var(--s-ink)}
.pj-lanes{display:grid;grid-template-rows:auto auto minmax(0,1fr);row-gap:var(--pj-gap)}
.pj-lane{display:grid;grid-template-columns:2.5rem minmax(0,1fr);column-gap:1rem;align-items:start}
.pj-lane>.pj-lane-icon{margin-top:.125rem}
.pj-lane-title{display:flex;align-items:center;gap:.5rem;height:var(--pj-title-h);font-size:.75rem;color:var(--s-mute)}
.pj-lane-docs .pj-lane-body{align-self:end}
.pj-slot{position:relative;display:grid}
.pj-slot>*{grid-area:1/1}
.pj-slot-wave{height:3.25rem;align-items:center}
.pj-slot-agent{align-items:start}
.pj-out{width:56%;align-self:start}
.pj-slot-agent>.pj-receipt{justify-self:end}
.pj-lane-docs .pj-chip,.pj-lane-title .pj-chip{padding:0 .5rem;font-size:.75rem}
.pj-ruler{padding:0}
.pj[data-mode="pinned"] .pj-list{display:flex;flex-direction:column;gap:0;align-content:start}
.pj[data-mode="pinned"] .pj-item{display:block;padding:0 0 .25rem;border:0;border-radius:0;background:none}
.pj[data-mode="pinned"] .pj-item::before,.pj[data-mode="pinned"] .pj-item::after{content:"";position:absolute;left:.875rem;width:1px;background:var(--s-bar)}
.pj[data-mode="pinned"] .pj-item::before{top:0;height:.875rem}
.pj[data-mode="pinned"] .pj-item::after{top:.875rem;bottom:0}
.pj[data-mode="pinned"] .pj-item:first-child::before,.pj[data-mode="pinned"] .pj-item:last-child::after{display:none}
.pj[data-mode="pinned"] .pj-item[data-state="current"]::before,.pj[data-mode="pinned"] .pj-item[data-state="done"]::before,.pj[data-mode="pinned"] .pj-item[data-state="done"]::after{background:var(--s-ink)}
.pj[data-mode="pinned"] .pj-step{align-items:flex-start;padding:.1875rem 0 .1875rem 0}
.pj[data-mode="pinned"] .pj-n{position:relative;z-index:1;box-shadow:0 0 0 .25rem var(--s-ground)}
.pj[data-mode="pinned"] .pj-name{font-size:1rem;color:var(--s-mute);transition:color .2s}
.pj[data-mode="pinned"] .pj-item[data-state="current"] .pj-name,.pj[data-mode="pinned"] .pj-item[data-state="done"] .pj-name{color:var(--s-ink)}
.pj[data-mode="pinned"] .pj-line{display:block}
.pj[data-mode="pinned"] .pj-step-chip{display:none}
.pj[data-mode="pinned"] .pj-item[data-state="current"] .pj-step-chip{display:inline-block}
.pj[data-mode="pinned"] .pj-mini{display:none}
.pj[data-mode="pinned"] .pj-body,.pj[data-mode="pinned"] .pj-seg{margin-left:2.5rem}
.pj[data-mode="pinned"] .pj-item:not([data-current="true"]) .pj-body{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);clip-path:inset(50%);white-space:nowrap}
.pj[data-mode="pinned"] .pj-item:not([data-current="true"]) .pj-seg{display:none}
.pj[data-mode="pinned"] .pj-item[data-current="true"] .pj-body{margin-top:.25rem;animation:pj-in .3s cubic-bezier(.16,1,.3,1)}
.pj[data-mode="pinned"] .pj-item[data-current="true"] .pj-seg{margin-top:.625rem}
@keyframes pj-in{from{opacity:0}to{opacity:1}}

/* stack mode: the cards wait for their turn (the same hidden state the rest of the site uses, so it is never left hidden) */
html.reveal-on .pj[data-mode="stack"] .pj-item{opacity:0}
`;
