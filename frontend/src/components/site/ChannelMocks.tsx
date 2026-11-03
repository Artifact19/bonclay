"use client";

/* Animated HTML mocks for the Channels section. Every name, number and time here is made up.
 * Each mock renders in its FINAL state, so it is complete with no JS, under reduced motion and when stacked
 * on a phone. When motion is allowed a paused GSAP timeline is built on mount (which hides the pieces that
 * will animate in) and played when the panel becomes active. */

import { useEffect, useLayoutEffect, useRef, type ReactNode, type RefObject } from "react";
import { gsap } from "gsap";
import { Check, Mic, Play } from "lucide-react";
import { SCRIPT_FONT } from "./fonts";
import { WAVE_IN, WAVE_OUT } from "./channelsData";

const reduced = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Fade/slide in with explicit from and to values. (gsap.from reads the end state from the DOM, and that read
 *  proved unreliable for list items inside a panel that starts out hidden.) */
function appear(tl: gsap.core.Timeline, target: gsap.TweenTarget, vars: gsap.TweenVars, pos?: gsap.Position) {
  const { x, y, scale, ...rest } = vars;
  const from: gsap.TweenVars = { autoAlpha: 0 };
  const to: gsap.TweenVars = { autoAlpha: 1, ...rest };
  if (x !== undefined) { from.x = x; to.x = 0; }
  if (y !== undefined) { from.y = y; to.y = 0; }
  if (scale !== undefined) { from.scale = scale; to.scale = 1; }
  return tl.fromTo(target, from, to, pos);
}

type Build = (tl: gsap.core.Timeline, q: gsap.utils.SelectorFunc) => void;

function useMockTimeline(root: RefObject<HTMLElement | null>, play: boolean, build: Build) {
  const tlRef = useRef<gsap.core.Timeline | null>(null);
  const buildRef = useRef(build);
  useLayoutEffect(() => {
    if (reduced() || !root.current) return;
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ paused: true });
      buildRef.current(tl, gsap.utils.selector(root.current!));
      tlRef.current = tl;
    }, root);
    return () => {
      ctx.revert();
      tlRef.current = null;
    };
  }, [root]);

  useEffect(() => {
    const tl = tlRef.current;
    if (!tl) return;
    if (play) {
      tl.restart();
      return;
    }
    // Leave after the panel has faded out, so the reset is never seen.
    const id = window.setTimeout(() => tl.pause(0), 450);
    return () => window.clearTimeout(id);
  }, [play]);
}

/* ------------------------------------------------------------------ small parts */

function Words({ text }: { text: string }) {
  return (
    <>
      {text.split(" ").map((w, i) => (
        <span key={i}>
          <span data-w>{w}</span>{" "}
        </span>
      ))}
    </>
  );
}

function Tag({ children, dark = false }: { children: ReactNode; dark?: boolean }) {
  return (
    <span
      data-tag
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 s-label font-medium normal-case leading-4 tracking-normal ${
        dark ? "bg-[#2A2C31] text-[var(--s-night-text)]" : "bg-[var(--s-ground)] text-[var(--s-ink-2)] ring-1 ring-inset ring-[var(--s-line)]"
      }`}
    >
      {children}
    </span>
  );
}

function Wave({ bars, className }: { bars: number[]; className: string }) {
  return (
    <span className="flex h-7 w-max items-center gap-[3px]" aria-hidden>
      {bars.map((h, i) => (
        <span key={i} className={`w-[0.25rem] rounded-full ${className}`} style={{ height: `${h}%` }} />
      ))}
    </span>
  );
}

/* ------------------------------------------------------------------ phone */

const CALL = [
  { who: "AI agent", text: "नमस्ते! मैं डेमो क्लिनिक की AI सहायक हूँ। बताइए, कैसे मदद करूँ?", gloss: "Hello! I'm Demo Clinic's AI assistant. How can I help?", tag: "AI disclosure, spoken first" },
  { who: "Riya", text: "कल शाम को डॉक्टर से मिलना है, appointment मिलेगा?", gloss: "I'd like to see the doctor tomorrow evening. Is there a slot?" },
  { who: "AI agent", text: "जी, कल शाम 5:30 बजे का समय खाली है, और 6 बजे भी—", gloss: "Yes, 5:30 pm tomorrow is free, and 6 pm too—" },
  { who: "Riya", text: "5:30 ठीक है।", gloss: "5:30 is fine.", tag: "Talked over: the agent stops" },
  { who: "AI agent", text: "बुक कर दिया, कल शाम 5:30 बजे। और कुछ?", gloss: "Booked for 5:30 pm tomorrow. Anything else?", tag: "Appointment booked" },
] as const;

export function PhoneMock({ play }: { play: boolean }) {
  const root = useRef<HTMLDivElement>(null);
  useMockTimeline(root, play, (tl, q) => {
    const clock = { t: 19 };
    const timer = q("[data-timer]")[0] as HTMLElement | undefined;
    const show = () => {
      if (timer) timer.textContent = `0:${String(Math.round(clock.t)).padStart(2, "0")}`;
    };
    appear(tl, q("[data-head]"), { y: 10, duration: 0.4 });
    CALL.forEach((_, i) => {
      const line = q(`[data-line="${i}"]`)[0] as HTMLElement;
      const words = line.querySelectorAll("[data-w]");
      const gloss = line.querySelector("[data-gloss]");
      const tag = line.querySelector("[data-tag]");
      // Riya's reply cuts in while the agent is still mid-sentence.
      const at = i === 3 ? "<0.15" : ">0.2";
      tl.fromTo(line, { autoAlpha: 0, y: 10 }, { autoAlpha: 1, y: 0, duration: 0.35, ease: "power2.out" }, at);
      appear(tl, words, { duration: 0.2, stagger: 0.07 }, "<0.1");
      if (gloss) appear(tl, gloss, { duration: 0.4 }, ">-0.1");
      if (tag) appear(tl, tag, { scale: 0.9, duration: 0.3 }, ">");
    });
    // Added last, so the ">" positions above are not pushed out by the clock.
    tl.fromTo(clock, { t: 0 }, { t: 19, duration: tl.duration(), ease: "none", onUpdate: show }, 0);
  });

  return (
    <div
      ref={root}
      role="group"
      aria-label="Example phone conversation"
      className="on-night w-full max-w-[29.375rem] rounded-[1.125rem] bg-[var(--s-night)] p-4 text-[var(--s-night-text)] shadow-[0_18px_50px_-20px_rgba(20,16,8,0.55)] sm:p-5"
    >
      <div data-head className="flex items-center gap-3 border-b border-[var(--s-night-line)] pb-3">
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-[var(--s-marigold)] s-small font-semibold text-[var(--s-marigold-ink)]" aria-hidden>
          R
        </span>
        <div className="min-w-0 flex-1">
          <p className="s-small font-medium leading-5">Riya</p>
          <p className="s-label leading-4 text-[var(--s-night-mute)]">••••4821 · call to the clinic line</p>
        </div>
        <div className="text-right">
          <p data-timer className="s-small font-medium tabular-nums leading-5">0:19</p>
          <p className="flex items-center justify-end gap-1.5 s-label leading-4 text-[var(--s-night-mute)]">
            <span className="chn-eq flex h-3 items-center gap-[2px]" aria-hidden>
              {[0, 1, 2, 3].map((b) => (
                <span key={b} className="chn-eqbar w-[2px] rounded-full bg-[var(--s-marigold)]" style={{ height: "100%", animationDelay: `${b * 0.12}s` }} />
              ))}
            </span>
            On a call
          </p>
        </div>
      </div>
      <ol className="mt-3 space-y-2.5">
        {CALL.map((l, i) => {
          const agent = l.who === "AI agent";
          return (
            <li key={i} data-line={i} className={agent ? "" : "pl-7"}>
              <p className="flex flex-wrap items-center gap-x-2 gap-y-1 s-label font-semibold">
                <span className={agent ? "text-[var(--s-marigold)]" : "text-[var(--s-night-mute)]"}>{l.who}</span>
                {"tag" in l && l.tag ? <Tag dark>{l.tag}</Tag> : null}
              </p>
              <p lang="hi" className={`${SCRIPT_FONT.hi} mt-0.5 s-small leading-[1.5]`}>
                <Words text={l.text} />
              </p>
              <p data-gloss className="s-label leading-4 text-[var(--s-night-mute)]">
                {l.gloss}
              </p>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

/* ------------------------------------------------------------------ meetings */

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"] as const;
const TIMES = ["10:30", "11:00", "11:30", "14:00", "14:30"] as const;
/** [day, time] pairs that are already busy on the owner's calendar. */
const BUSY = new Set(["0-1", "1-0", "1-2", "2-1", "2-3", "3-1", "4-0", "4-2", "4-4", "0-4", "3-4"]);
const STEPS = ["Checks the calendar", "Offers other times", "Holds a pending request", "Owner approves", "Lands in Google Calendar"] as const;

const hatch = "bg-[repeating-linear-gradient(135deg,#ECECE7_0_5px,#F6F6F2_5px_10px)] text-[var(--s-mute)]";

export function MeetingsMock({ play }: { play: boolean }) {
  const root = useRef<HTMLDivElement>(null);
  useMockTimeline(root, play, (tl, q) => {
    const step = (i: number) => q(`[data-step="${i}"]`);
    const dimFrom = { opacity: 0.35 };
    const dimTo = { opacity: 1, duration: 0.3 };
    appear(tl, q("[data-day]"), { y: 6, duration: 0.3, stagger: 0.04 });
    // 1. The calendar is read: busy times appear, and the time Riya wanted is marked.
    tl.fromTo(step(0), dimFrom, dimTo, "<");
    appear(tl, q("[data-busy]"), { scale: 0.9, duration: 0.3, stagger: 0.04 }, ">-0.1");
    appear(tl, q("[data-asked]"), { scale: 0.85, duration: 0.35, ease: "back.out(2)" }, ">0.1");
    // 2. Other times are offered.
    tl.fromTo(step(1), dimFrom, dimTo, ">0.5");
    appear(tl, q("[data-free]"), { scale: 0.9, duration: 0.35 }, "<");
    tl.fromTo(q("[data-free-tmp]"), { autoAlpha: 0, scale: 0.9 }, { autoAlpha: 1, scale: 1, duration: 0.35 }, "<0.15");
    // 3. The agent holds one as a pending request; the owner's Approvals card appears.
    tl.fromTo(step(2), dimFrom, dimTo, ">0.7");
    tl.to(q("[data-free-tmp]"), { autoAlpha: 0, duration: 0.3 }, "<");
    tl.fromTo(q("[data-pending]"), { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.35 }, "<");
    appear(tl, q("[data-card]"), { y: 14, duration: 0.45, ease: "power2.out" }, "<0.1");
    tl.fromTo(q("[data-card-wait]"), { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.3 }, "<");
    // 4. The owner presses Approve.
    tl.fromTo(step(3), dimFrom, dimTo, ">0.9");
    tl.to(q("[data-approve]"), { scale: 0.92, duration: 0.12, yoyo: true, repeat: 1 }, "<");
    tl.to(q("[data-card-wait]"), { autoAlpha: 0, duration: 0.25 }, ">0.05");
    appear(tl, q("[data-card-ok]"), { duration: 0.3 }, "<");
    // 5. The slot turns solid and the event is in Google Calendar.
    tl.fromTo(step(4), dimFrom, dimTo, ">0.2");
    tl.to(q("[data-pending]"), { autoAlpha: 0, duration: 0.3 }, "<");
    appear(tl, q("[data-approved]"), { scale: 0.85, duration: 0.4, ease: "back.out(2)" }, "<");
    appear(tl, q("[data-gcal]"), { y: 6, duration: 0.35 }, ">-0.1");
  });

  return (
    <div ref={root} role="group" aria-label="Example meeting request" className="w-full max-w-[35rem] rounded-[1.125rem] bg-[var(--s-paper)] p-4 shadow-[0_18px_50px_-20px_rgba(20,16,8,0.45)] sm:p-5">
      <div className="flex items-baseline justify-between gap-3">
        <p className="s-small font-medium">Head of department, this week</p>
        <p className="s-label text-[var(--s-mute)]">20-minute slots</p>
      </div>
      <div className="mt-3 grid grid-cols-[2.6rem_repeat(5,minmax(0,1fr))] gap-1 s-label">
        <span aria-hidden />
        {DAYS.map((d) => (
          <span key={d} data-day className="pb-0.5 text-center font-medium text-[var(--s-mute)]">
            {d}
          </span>
        ))}
        {TIMES.map((t, r) => (
          <FragmentRow key={t}>
            <span className="grid h-9 place-items-center text-[var(--s-mute)] tabular-nums">{t}</span>
            {DAYS.map((d, c) => {
              const key = `${c}-${r}`;
              const thu = c === 3;
              if (thu && r === 1)
                return (
                  <span key={key} data-busy data-asked className={`relative grid h-9 place-items-center rounded-md ${hatch} ring-1 ring-[var(--s-ink)]`}>
                    <span className="s-label font-semibold leading-3 text-[var(--s-ink)]">Asked<span className="hidden sm:inline"> for</span></span>
                  </span>
                );
              if (thu && r === 2)
                return (
                  <span key={key} className="relative h-9">
                    <span data-free-tmp className="absolute inset-0 grid place-items-center rounded-md border border-dashed border-[var(--s-ink-2)] s-label font-medium opacity-0">
                      Free
                    </span>
                    <span data-pending className="absolute inset-0 grid place-items-center rounded-md border border-dashed border-[var(--s-ink)] bg-[#FBE3BE] s-label font-semibold text-[var(--s-ink)] opacity-0">
                      Pending
                    </span>
                    <span data-approved className="absolute inset-0 flex items-center justify-center gap-0.5 rounded-md bg-[var(--s-ink)] s-label font-semibold text-[var(--s-night-text)]">
                      <Check size={11} aria-hidden className="hidden sm:block" /> Booked
                    </span>
                  </span>
                );
              if (thu && r === 3)
                return (
                  <span key={key} data-free className="grid h-9 place-items-center rounded-md border border-dashed border-[var(--s-ink-2)] s-label font-medium">
                    Free
                  </span>
                );
              if (BUSY.has(key))
                return (
                  <span key={key} data-busy className={`grid h-9 place-items-center rounded-md s-label ${hatch}`}>
                    Busy
                  </span>
                );
              return <span key={key} className="h-9 rounded-md bg-[var(--s-ground)]" aria-hidden />;
            })}
          </FragmentRow>
        ))}
      </div>

      <div data-card className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-[var(--s-line)] bg-[var(--s-ground)] p-3">
        <div className="min-w-0 flex-1">
          <p className="s-label font-semibold text-[var(--s-mute)]">Approvals</p>
          <p className="s-small font-medium leading-5">Riya, a parent · Thu 11:30</p>
          <p className="grid s-label leading-4 text-[var(--s-mute)]">
            <span data-card-wait className="col-start-1 row-start-1 opacity-0">
              Waiting for the head of department
            </span>
            <span data-card-ok className="col-start-1 row-start-1">
              Approved. Riya can ask the agent about it.
            </span>
          </p>
        </div>
        <span data-approve className="inline-flex h-9 items-center gap-1 rounded-full bg-[var(--s-ink)] px-4 s-label font-medium text-[var(--s-night-text)]">
          <Check size={14} aria-hidden /> Approve
        </span>
      </div>

      <ol className="mt-4 grid grid-cols-1 gap-x-3 gap-y-1 s-label sm:grid-cols-5">
        {STEPS.map((s, i) => (
          <li key={s} data-step={i} className="flex items-start gap-1.5 leading-4">
            <span className="mt-px grid h-4 w-4 shrink-0 place-items-center rounded-full bg-[var(--s-marigold)] s-label font-semibold text-[var(--s-marigold-ink)]">{i + 1}</span>
            <span className="text-[var(--s-ink-2)]">{s}</span>
          </li>
        ))}
      </ol>
      <p data-gcal className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-[var(--s-ground)] px-3 py-1 s-label font-medium ring-1 ring-inset ring-[var(--s-line)]">
        <Check size={12} aria-hidden /> Event added to Google Calendar
      </p>
    </div>
  );
}

/** Rows of a CSS grid without a wrapper element, so every cell stays a direct grid child. */
function FragmentRow({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

/* ------------------------------------------------------------------ website */

export function WidgetMock({ play }: { play: boolean }) {
  const root = useRef<HTMLDivElement>(null);
  useMockTimeline(root, play, (tl, q) => {
    appear(tl, q("[data-code]"), { x: -10, duration: 0.45, stagger: 0.35 });
    appear(tl, q("[data-frame]"), { y: 10, duration: 0.45 }, ">0.1");
    // The collapsed bubble appears, then opens into the panel with the visitor notice.
    tl.fromTo(q("[data-bubble]"), { autoAlpha: 0, scale: 0.6 }, { autoAlpha: 1, scale: 1, duration: 0.4, ease: "back.out(2)" }, ">0.1");
    tl.to(q("[data-bubble]"), { autoAlpha: 0, scale: 0.8, duration: 0.25 }, ">0.8");
    appear(tl, q("[data-panel]"), { y: 14, scale: 0.96, duration: 0.4, ease: "power2.out" }, "<");
    tl.fromTo(q("[data-notice]"), { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.3 }, "<0.1");
    // The visitor presses Start, then talks.
    tl.to(q("[data-start]"), { scale: 0.92, duration: 0.12, yoyo: true, repeat: 1 }, ">1");
    tl.to(q("[data-notice]"), { autoAlpha: 0, duration: 0.25 }, ">0.05");
    appear(tl, q("[data-cap='0']"), { y: 8, duration: 0.35 }, ">");
    appear(tl, q("[data-cap='0'] [data-w]"), { duration: 0.2, stagger: 0.07 }, "<0.1");
    appear(tl, q("[data-cap='1']"), { y: 8, duration: 0.35 }, ">0.2");
    appear(tl, q("[data-cap='1'] [data-w]"), { duration: 0.2, stagger: 0.07 }, "<0.1");
    appear(tl, q("[data-chip]"), { y: 6, duration: 0.3, stagger: 0.12 }, ">0.2");
  });

  return (
    <div ref={root} role="group" aria-label="Example website with the agent widget" className="w-full max-w-[33.75rem]">
      <div className="on-night rounded-[var(--s-r-inner)] bg-[var(--s-night)] px-4 py-3 font-mono s-label leading-[1.7] text-[var(--s-night-text)] shadow-[0_18px_50px_-20px_rgba(20,16,8,0.55)]">
        <p data-code className="whitespace-pre-wrap break-all">
          <span className="text-[var(--s-night-mute)]">&lt;</span>script src=<span className="text-[var(--s-marigold)]">&quot;https://YOUR-HOST/widget/v1/bonclay-agent.js&quot;</span>
        </p>
        <p data-code className="whitespace-pre-wrap break-all">
          {"        "}defer<span className="text-[var(--s-night-mute)]">&gt;&lt;/script&gt;</span>
        </p>
        <p data-code className="whitespace-pre-wrap break-all">
          <span className="text-[var(--s-night-mute)]">&lt;</span>bonclay-agent agent=<span className="text-[var(--s-marigold)]">&quot;PUBLIC_WIDGET_KEY&quot;</span>
          <span className="text-[var(--s-night-mute)]">&gt;&lt;/bonclay-agent&gt;</span>
        </p>
      </div>

      <div data-frame className="mt-3 overflow-hidden rounded-[var(--s-r-inner)] bg-[var(--s-paper)] shadow-[0_18px_50px_-20px_rgba(20,16,8,0.45)]">
        <div className="flex items-center gap-2 border-b border-[var(--s-line)] bg-[var(--s-ground)] px-3 py-2" aria-hidden>
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--s-bar)]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--s-bar)]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--s-bar)]" />
          <span className="ml-2 flex-1 truncate rounded-md bg-[var(--s-paper)] px-2.5 py-0.5 s-label text-[var(--s-mute)]">www.example.com</span>
        </div>
        <div className="relative h-[15.625rem] sm:h-[16.125rem]">
          <div className="space-y-2 p-4" aria-hidden>
            <div className="h-3 w-1/3 rounded bg-[var(--s-line)]" />
            <div className="h-2 w-3/4 rounded bg-[var(--s-ground)]" />
            <div className="h-2 w-2/3 rounded bg-[var(--s-ground)]" />
            <div className="h-2 w-1/2 rounded bg-[var(--s-ground)]" />
          </div>
          <span data-bubble className="absolute bottom-3 right-3 inline-flex items-center gap-2 rounded-full bg-[var(--s-ink)] py-2 pl-3 pr-4 s-label font-medium text-[var(--s-night-text)] opacity-0">
            <Mic size={15} aria-hidden /> Talk to us
          </span>
          <div data-panel className="absolute bottom-3 right-3 w-[min(272px,calc(100%-1.5rem))] overflow-hidden rounded-xl border border-[var(--s-line)] bg-[var(--s-paper)] shadow-[0_10px_30px_-12px_rgba(20,16,8,0.4)]">
            <div className="flex items-center justify-between gap-2 bg-[var(--s-ink)] px-3 py-2 text-[var(--s-night-text)]">
              <span className="s-label font-medium">Demo Clinic</span>
              <span className="rounded-full bg-[#2A2C31] px-2 py-0.5 s-label font-medium">AI-generated voice</span>
            </div>
            <div className="relative h-[8.875rem]">
              <div data-notice className="absolute inset-0 flex flex-col justify-between p-3 opacity-0">
                <p className="s-label leading-[1.45] text-[var(--s-ink-2)]">You&apos;re talking to an AI assistant. Audio is processed to answer you; conversations may be stored by Demo Clinic.</p>
                <span data-start className="inline-flex h-8 items-center justify-center self-start rounded-full bg-[var(--s-marigold)] px-4 s-label font-semibold text-[var(--s-marigold-ink)]">
                  Start
                </span>
              </div>
              <div className="space-y-2 p-3" aria-live="off">
                <p data-cap="0" className="ml-auto w-fit max-w-[88%] rounded-xl rounded-br-sm bg-[var(--s-ground)] px-2.5 py-1.5 s-label leading-[1.4]">
                  <Words text="Saturday ko clinic kab tak khula hai?" />
                </p>
                <p data-cap="1" className="w-fit max-w-[88%] rounded-xl rounded-bl-sm bg-[#FBE3BE] px-2.5 py-1.5 s-label leading-[1.4]">
                  <Words text="Saturday ko hum subah 9 se 1 baje tak khule hain." />
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <ul className="mt-3 flex flex-wrap gap-2 s-label font-medium">
        {["Signed session, 10 minutes", "Only the sites you list", "Limits per visitor"].map((c) => (
          <li key={c} data-chip className="rounded-full bg-[var(--s-paper)] px-3 py-1 text-[var(--s-ink-2)]">
            {c}
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ whatsapp */

export function WhatsAppMock({ play }: { play: boolean }) {
  const root = useRef<HTMLDivElement>(null);
  useMockTimeline(root, play, (tl, q) => {
    appear(tl, q("[data-wa-head]"), { y: 8, duration: 0.4 });
    appear(tl, q("[data-in]"), { x: -14, duration: 0.4, ease: "power2.out" }, ">0.2");
    tl.fromTo(q("[data-in-fill]"), { width: 0 }, { width: "100%", duration: 1.6, ease: "none" }, ">0.1");
    appear(tl, q("[data-out]"), { x: 14, duration: 0.4, ease: "power2.out" }, ">0.3");
    tl.fromTo(q("[data-out-fill]"), { width: 0 }, { width: "100%", duration: 2, ease: "none" }, ">0.1");
    appear(tl, q("[data-text]"), { y: 8, duration: 0.4 }, ">0.2");
    appear(tl, q("[data-text] [data-w]"), { duration: 0.2, stagger: 0.06 }, "<0.1");
    appear(tl, q("[data-chip]"), { y: 6, duration: 0.3, stagger: 0.12 }, ">0.2");
  });

  return (
    <div ref={root} role="group" aria-label="Example WhatsApp conversation" className="w-full max-w-[27.5rem]">
      <div className="overflow-hidden rounded-[1.375rem] bg-[var(--s-paper)] shadow-[0_18px_50px_-20px_rgba(20,16,8,0.45)]">
        <div data-wa-head className="flex items-center justify-between gap-2 bg-[var(--s-ink)] px-4 py-3 text-[var(--s-night-text)]">
          <span className="s-small font-medium">Demo Clinic</span>
          <span className="rounded-full bg-[var(--s-marigold)] px-2.5 py-0.5 s-label font-semibold text-[var(--s-marigold-ink)]">Beta: not yet live-tested</span>
        </div>
        <div className="space-y-2.5 bg-[var(--s-ground)] p-4">
          <div data-in className="w-[82%] rounded-2xl rounded-bl-sm bg-[var(--s-paper)] p-2.5 shadow-sm">
            <p className="mb-1 s-label font-semibold text-[var(--s-mute)]">Riya · voice note</p>
            <VoiceNote bars={WAVE_IN} fill="data-in-fill" time="0:06" />
          </div>
          <div data-out className="ml-auto w-[88%] rounded-2xl rounded-br-sm bg-[#FBE3BE] p-2.5 shadow-sm">
            <p className="mb-1 flex items-center justify-between gap-2 s-label font-semibold text-[var(--s-ink-2)]">
              <span className="whitespace-nowrap">Agent</span>
              <span className="whitespace-nowrap rounded-full bg-[var(--s-ink)] px-2 py-px s-label font-medium text-[var(--s-night-text)]">AI-generated voice</span>
            </p>
            <VoiceNote bars={WAVE_OUT} fill="data-out-fill" time="0:09" />
          </div>
          <div data-text className="ml-auto w-[88%] rounded-2xl rounded-br-sm bg-[#FBE3BE] p-2.5 shadow-sm">
            <p lang="hi" className={`${SCRIPT_FONT.hi} s-small leading-[1.5]`}>
              <Words text="नमस्ते! मैं डेमो क्लिनिक की AI सहायक हूँ। कल शाम 5:30 बजे आपका appointment है।" />
            </p>
            <p className="mt-1 s-label leading-4 text-[var(--s-ink-2)]">Hello! I&apos;m Demo Clinic&apos;s AI assistant. Your appointment is tomorrow at 5:30 pm.</p>
          </div>
        </div>
      </div>
      <ul className="mt-3 flex flex-wrap gap-2 s-label font-medium">
        {["Voice note and the same text", "Never starts a chat", "Can hand over to staff"].map((c) => (
          <li key={c} data-chip className="rounded-full bg-[var(--s-paper)] px-3 py-1 text-[var(--s-ink-2)]">
            {c}
          </li>
        ))}
      </ul>
    </div>
  );
}

function VoiceNote({ bars, fill, time }: { bars: number[]; fill: string; time: string }) {
  const attr = { [fill]: "" } as Record<string, string>;
  return (
    <div className="flex items-center gap-2.5">
      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-[var(--s-ink)] text-[var(--s-night-text)]" aria-hidden>
        <Play size={13} fill="currentColor" />
      </span>
      <span className="relative block w-max max-w-full flex-1 overflow-hidden">
        <Wave bars={bars} className="bg-[var(--s-bar)]" />
        <span {...attr} className="absolute inset-y-0 left-0 block w-full overflow-hidden">
          <span className="block w-max">
            <Wave bars={bars} className="bg-[var(--s-ink)]" />
          </span>
        </span>
      </span>
      <span className="shrink-0 s-label tabular-nums text-[var(--s-mute)]">{time}</span>
    </div>
  );
}
