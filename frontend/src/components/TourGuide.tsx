"use client";

import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, Compass, Sparkles, X } from "lucide-react";
import { TaskProgressBar } from "@/components/ui/TaskProgressBar";

export interface TourStep {
  id: string;
  /** CSS selector(s) comma-separated; first visible element in the viewport/DOM is spotlighted. */
  selector: string;
  /** Short badge category. */
  category: string;
  /** Friendly title of the option. */
  title: string;
  /** Plain-English explanation for non-technical users. */
  description: string;
  /** Simple everyday example or beginner tip. */
  beginnerTip: string;
}

const SITE_STEPS: TourStep[] = [
  {
    id: "site-nav",
    selector: "header nav",
    category: "Navigation Bar",
    title: "Explore Every Part of BonClay",
    description:
      "Use this top bar to see how BonClay works, which phone and web channels it supports, real speed benchmarks, and answers to common questions.",
    beginnerTip:
      "You don't need any technical background—each page explains things in plain language with real audio examples you can click and play.",
  },
  {
    id: "site-hero",
    selector: "#hero-bento, #top .hero-island",
    category: "Overview",
    title: "What BonClay Does for You",
    description:
      "BonClay lets you turn written text into natural speech in 22 Indian languages, clone your own voice safely, and build AI receptionists that answer questions from your documents.",
    beginnerTip:
      "Everything can run on a single laptop without complicated cloud setup, and every audio clip is digitally signed so nobody can misuse a voice.",
  },
  {
    id: "site-voices",
    selector: "#voices",
    category: "Try Listening",
    title: "Hear Real Voices Right Now",
    description:
      "Pick any voice from the list on the left and press the Play button to hear how natural BonClay sounds in English and Indian languages.",
    beginnerTip:
      "Watch the words light up as the voice speaks—this is the exact speech engine you will use inside the Studio.",
  },
  {
    id: "site-products",
    selector: "#products",
    category: "Two Main Tools",
    title: "Studio vs. Voice Agents",
    description:
      "BonClay gives you two simple tools: 'Studio' (turn any script into downloadable audio) and 'Agents' (create a smart helper that reads your PDFs and talks to callers).",
    beginnerTip:
      "Use Studio for videos, announcements, or lessons. Use Agents for clinic front desks, college helpdesks, or office appointment booking.",
  },
  {
    id: "site-open",
    selector: 'header a[href="/dashboard"]',
    category: "Get Started",
    title: "Open the Studio Anytime",
    description:
      "Click 'Open studio' to enter the app workspace. Once inside, you can click the 'Tour Guide' button again for a step-by-step walkthrough of every tool.",
    beginnerTip:
      "No sign-up or credit card is required when running locally—your work is saved privately on your computer.",
  },
];

const DASHBOARD_STEPS: TourStep[] = [
  {
    id: "dash-switch",
    selector: 'aside [role="radiogroup"], aside',
    category: "1 · Workspace Switcher",
    title: "Switch Between 'Studio' and 'Agents'",
    description:
      "At the top of the left sidebar, you can switch between BonClay's two modes: 'Studio' is your recording booth for making audio files, and 'Agents' is your virtual receptionist builder.",
    beginnerTip:
      "If you only want to type text and download speech, choose 'Studio'. If you want an AI to answer questions from your PDFs or take phone calls, choose 'Agents'.",
  },
  {
    id: "dash-menu",
    selector: 'aside nav[aria-label="Main"]',
    category: "2 · Step-by-Step Menu",
    title: "Your Tools, Grouped in Order",
    description:
      "This menu shows only the tools you need for your current mode, arranged in the exact order you use them: first Set Up, then Try & Run, and finally Check Results.",
    beginnerTip:
      "Here is what each option does:\n• Agents / Voice Studio: Create your assistant or type your script.\n• Documents: Upload PDFs/brochures so your agent knows the facts.\n• Voices: Pick a ready voice or clone your own voice in 10 seconds.\n• Talk to your agent: Test your agent live using your mic or keyboard.",
  },
  {
    id: "dash-voices",
    selector: 'aside a[href="/dashboard/voices"]',
    category: "3 · Voices & Cloning",
    title: "Pick a Ready Voice or Clone Yours",
    description:
      "Open 'Voices' to browse ready-to-use voices or click 'Clone Voice' to record a short sample of yourself reading a consent sentence on screen.",
    beginnerTip:
      "BonClay checks that the person speaking the consent sentence matches the voice sample, keeping your voice safe from unauthorized use.",
  },
  {
    id: "dash-main",
    selector: "#main-content > div, #main-content",
    category: "4 · Main Workspace",
    title: "Where Your Task Happens",
    description:
      "The center of the screen is where you work on the option you selected—with clear percentage progress bars whenever BonClay is generating audio, uploading files, or checking safety.",
    beginnerTip:
      "Look for the 'Set up your agent' checklist or 'Sample data' card on the home screen if you want to load two ready-made example agents in one click!",
  },
  {
    id: "dash-search",
    selector: 'button[aria-label="Search and commands"]',
    category: "5 · Instant Finder",
    title: "Search Any Tool or Action (Ctrl + K)",
    description:
      "Click 'Search' in the top-right corner (or press Ctrl+K / ⌘K) to find any page or action by typing what you want in plain words.",
    beginnerTip:
      "For example, type 'upload', 'hindi', 'phone call', or 'verify' and jump straight there without hunting through menus.",
  },
];

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

function findVisibleTarget(selectorList: string): HTMLElement | null {
  if (typeof document === "undefined") return null;
  const selectors = selectorList.split(",").map((s) => s.trim()).filter(Boolean);
  for (const sel of selectors) {
    try {
      const els = Array.from(document.querySelectorAll<HTMLElement>(sel));
      const visible = els.find((el) => {
        const r = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return r.width > 16 && r.height > 16 && style.visibility !== "hidden" && style.display !== "none";
      });
      if (visible) return visible;
    } catch {
      /* ignore invalid selector */
    }
  }
  return null;
}

export interface TourGuideProps {
  mode: "site" | "dashboard";
  open: boolean;
  onClose: () => void;
}

/**
 * Interactive Spotlight Tour Guide that focuses on one UI option at a time, blurs and dims the rest of the screen,
 * and explains what each option does in clear, non-technical language with a percentage progress bar.
 */
export function TourGuide({ mode, open, onClose }: TourGuideProps) {
  const steps = useMemo(() => (mode === "dashboard" ? DASHBOARD_STEPS : SITE_STEPS), [mode]);
  const [idx, setIdx] = useState(0);
  const [prevOpen, setPrevOpen] = useState(open);
  if (prevOpen !== open) {
    setPrevOpen(open);
    if (!open) {
      setIdx(0);
    }
  }
  const [rect, setRect] = useState<Rect | null>(null);
  const [vw, setVw] = useState(1280);
  const [vh, setVh] = useState(800);
  const titleId = useId();
  const descId = useId();

  const activeIdx = Math.min(idx, steps.length - 1);
  const step = steps[activeIdx];
  const pct = Math.round(((activeIdx + 1) / steps.length) * 100);

  const measure = useCallback(() => {
    if (typeof window === "undefined" || !open || !step) return;
    setVw(window.innerWidth);
    setVh(window.innerHeight);
    const el = findVisibleTarget(step.selector);
    if (!el) {
      setRect(null);
      return;
    }
    const r = el.getBoundingClientRect();
    const pad = 8;
    const top = Math.max(4, r.top - pad);
    const left = Math.max(4, r.left - pad);
    const width = Math.min(window.innerWidth - left - 4, r.width + pad * 2);
    const height = Math.min(window.innerHeight - top - 4, r.height + pad * 2);
    setRect({ top, left, width, height });
  }, [open, step]);

  // Scroll the focused element into view when step changes, then measure its position.
  useEffect(() => {
    if (!open || !step) return;
    const el = findVisibleTarget(step.selector);
    if (el) {
      const r = el.getBoundingClientRect();
      const outOfView = r.top < 60 || r.bottom > window.innerHeight - 60;
      if (outOfView && typeof el.scrollIntoView === "function") {
        try {
          el.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
        } catch {
          /* ignore */
        }
      }
    }
    const t1 = window.setTimeout(measure, 60);
    const t2 = window.setTimeout(measure, 360);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [open, step, measure]);

  useEffect(() => {
    if (!open) return;
    const raf = requestAnimationFrame(() => measure());
    const onResizeOrScroll = () => measure();
    window.addEventListener("resize", onResizeOrScroll, { passive: true });
    window.addEventListener("scroll", onResizeOrScroll, { passive: true, capture: true });
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setIdx((i) => (i + 1 < steps.length ? i + 1 : i));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setIdx((i) => (i > 0 ? i - 1 : 0));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResizeOrScroll);
      window.removeEventListener("scroll", onResizeOrScroll, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [open, measure, onClose, steps.length]);

  if (!open || !step) return null;

  // Compute where to place the tour card so it sits beside or below/above the spotlight cutout without covering it.
  const cardWidth = Math.min(400, vw - 24);
  let cardStyle: React.CSSProperties = {
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: cardWidth,
  };

  if (rect) {
    const spaceRight = vw - (rect.left + rect.width);
    const spaceLeft = rect.left;
    const spaceBottom = vh - (rect.top + rect.height);
    const spaceTop = rect.top;

    if (spaceRight >= cardWidth + 28) {
      cardStyle = {
        top: Math.max(16, Math.min(vh - 380, rect.top)),
        left: rect.left + rect.width + 18,
        width: cardWidth,
      };
    } else if (spaceLeft >= cardWidth + 28) {
      cardStyle = {
        top: Math.max(16, Math.min(vh - 380, rect.top)),
        left: Math.max(12, rect.left - cardWidth - 18),
        width: cardWidth,
      };
    } else if (spaceBottom >= 300) {
      cardStyle = {
        top: Math.min(vh - 360, rect.top + rect.height + 16),
        left: Math.max(12, Math.min(vw - cardWidth - 12, rect.left)),
        width: cardWidth,
      };
    } else if (spaceTop >= 300) {
      cardStyle = {
        bottom: Math.max(16, vh - rect.top + 16),
        left: Math.max(12, Math.min(vw - cardWidth - 12, rect.left)),
        width: cardWidth,
      };
    } else {
      // Target fills most of the screen (like #main-content): dock card cleanly at bottom-right.
      cardStyle = {
        bottom: 20,
        right: 20,
        width: cardWidth,
      };
    }
  }

  const blurPanelClass =
    "fixed z-[90] bg-[#141414]/55 backdrop-blur-[7px] transition-all duration-300 ease-out pointer-events-auto";

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      aria-describedby={descId}
      data-testid="tour-guide-dialog"
      className="fixed inset-0 z-[90] pointer-events-none"
    >
      {/* 4-panel blur mask around the focused element so ONLY the rest of the screen is blurred out */}
      {rect ? (
        <>
          {/* Top blur panel */}
          <div
            onClick={onClose}
            className={blurPanelClass}
            style={{ top: 0, left: 0, width: "100vw", height: Math.max(0, rect.top) }}
          />
          {/* Bottom blur panel */}
          <div
            onClick={onClose}
            className={blurPanelClass}
            style={{
              top: rect.top + rect.height,
              left: 0,
              width: "100vw",
              height: Math.max(0, vh - (rect.top + rect.height)),
            }}
          />
          {/* Left blur panel */}
          <div
            onClick={onClose}
            className={blurPanelClass}
            style={{
              top: rect.top,
              left: 0,
              width: Math.max(0, rect.left),
              height: rect.height,
            }}
          />
          {/* Right blur panel */}
          <div
            onClick={onClose}
            className={blurPanelClass}
            style={{
              top: rect.top,
              left: rect.left + rect.width,
              width: Math.max(0, vw - (rect.left + rect.width)),
              height: rect.height,
            }}
          />
          {/* Crisp spotlight ring around the focused option */}
          <div
            aria-hidden
            className="fixed z-[91] rounded-2xl border-2 border-[#F3F59E] shadow-[0_0_0_4px_rgba(232,99,43,0.45),0_12px_32px_rgba(0,0,0,0.35)] transition-all duration-300 ease-out pointer-events-none"
            style={{
              top: rect.top,
              left: rect.left,
              width: rect.width,
              height: rect.height,
            }}
          >
            <span className="absolute -top-3 left-3 inline-flex items-center gap-1 rounded-full border border-[#141414] bg-[#F3F59E] px-2.5 py-0.5 text-[11px] font-bold text-[#141414] shadow-sm">
              <Sparkles className="size-3 text-[#E8632B]" />
              {step.category}
            </span>
          </div>
        </>
      ) : (
        <div onClick={onClose} className={`${blurPanelClass} inset-0`} />
      )}

      {/* Editorial Bento Tour Explanation Card */}
      <div
        style={cardStyle}
        className="fixed z-[92] pointer-events-auto rounded-[24px] border-[1.5px] border-[#141414] bg-[#FAF9F5] p-5 text-[#141414] shadow-[0_20px_50px_rgba(20,20,20,0.45)] transition-all duration-300"
      >
        <div className="flex items-center justify-between gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-[#141414] bg-[#E2DDD3] px-3 py-1 text-xs font-semibold text-[#141414]">
            <span aria-hidden className="text-[#E8632B]">✦</span>
            Step {activeIdx + 1} of {steps.length}
          </span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close tour guide"
            className="inline-flex size-8 items-center justify-center rounded-full border border-[#141414]/30 bg-white text-[#141414] transition-colors hover:border-[#141414] hover:bg-[#F3F59E]"
          >
            <X className="size-4" aria-hidden />
          </button>
        </div>

        <h3 id={titleId} className="mt-3 text-lg font-bold leading-snug tracking-tight text-[#141414]">
          {step.title}
        </h3>

        <p id={descId} className="mt-2 whitespace-pre-line text-sm leading-relaxed text-[#2C2B29]">
          {step.description}
        </p>

        <div className="mt-3 rounded-2xl border border-[#141414]/25 bg-[#E2DDD3] p-3 text-xs leading-relaxed text-[#141414]">
          <p className="font-bold uppercase tracking-wider text-[#E8632B]">Plain-English Tip</p>
          <p className="mt-1 whitespace-pre-line">{step.beginnerTip}</p>
        </div>

        <TaskProgressBar
          value={pct}
          label="Tour progress"
          detail={`${activeIdx + 1}/${steps.length}`}
          size="sm"
          className="mt-4"
          testId="tour-progress"
        />

        <div className="mt-4 flex items-center justify-between gap-2 pt-1">
          <button
            type="button"
            onClick={onClose}
            className="text-xs font-semibold text-[#52504C] underline underline-offset-4 hover:text-[#141414]"
          >
            Skip tour
          </button>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setIdx((i) => Math.max(0, i - 1))}
              disabled={activeIdx === 0}
              className="inline-flex h-9 items-center gap-1 rounded-full border-[1.5px] border-[#141414] bg-white px-3.5 text-xs font-semibold text-[#141414] transition-colors hover:bg-[#E2DDD3] disabled:opacity-40"
            >
              <ArrowLeft className="size-3.5" aria-hidden />
              Back
            </button>
            {activeIdx + 1 < steps.length ? (
              <button
                type="button"
                onClick={() => setIdx((i) => Math.min(steps.length - 1, i + 1))}
                className="inline-flex h-9 items-center gap-1.5 rounded-full border-[1.5px] border-[#141414] bg-[#F3F59E] px-4 text-xs font-bold text-[#141414] transition-transform hover:-translate-y-0.5 active:scale-95"
              >
                Next option
                <ArrowRight className="size-3.5" aria-hidden />
              </button>
            ) : (
              <button
                type="button"
                onClick={onClose}
                className="inline-flex h-9 items-center gap-1.5 rounded-full border-[1.5px] border-[#141414] bg-[#141414] px-4 text-xs font-bold text-[#F3F59E] transition-transform hover:-translate-y-0.5 active:scale-95"
              >
                Finish tour
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Button that opens the Interactive Spotlight Tour Guide from any navbar or header. */
export function TourTriggerButton({
  onClick,
  compact = false,
  label,
  className = "",
}: {
  onClick: () => void;
  compact?: boolean;
  label?: string;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      data-testid="tour-guide-trigger"
      aria-label="Start interactive tour guide"
      className={`inline-flex items-center gap-1.5 rounded-full border-[1.5px] border-[#141414] bg-[#F3F59E] px-3.5 py-1.5 text-xs font-bold text-[#141414] shadow-[0_2px_0_#141414] transition-all hover:-translate-y-0.5 hover:bg-[#eef17a] active:translate-y-0 ${className}`}
    >
      <Compass className="size-3.5 text-[#E8632B]" aria-hidden />
      <span>{label ?? (compact ? "Tour" : "Tour Guide")}</span>
    </button>
  );
}

