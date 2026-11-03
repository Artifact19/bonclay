"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Loader2, Pause, Play } from "lucide-react";
import { VOICE_CLIPS } from "./voices";
import { SCRIPT_FONT, useScriptFontsNear } from "./fonts";
import { claimAudio } from "./audioBus";

/** Index of the first Hindi voice: its row is the one the list watches to load the Devanagari face. */
const firstHindi = VOICE_CLIPS.findIndex((v) => v.lang === "hi");

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** Max-pool the peaks down to n bars so narrow screens still get legible bars. */
const pool = (peaks: number[], n: number) =>
  Array.from({ length: n }, (_, i) => Math.max(...peaks.slice(Math.floor((i * peaks.length) / n), Math.floor(((i + 1) * peaks.length) / n))));

function Bars({ peaks, tone }: { peaks: number[]; tone: string }) {
  const row = (values: number[], cls: string) => (
    <div className={`h-full items-center ${cls}`}>
      {values.map((p, i) => (
        <span key={i} className="min-w-0 flex-1 rounded-full" style={{ height: `${Math.max(6, p * 100)}%`, background: tone }} />
      ))}
    </div>
  );
  // CSS picks the density, so server and client render the same markup.
  return (
    <>
      {row(pool(peaks, 56), "flex gap-[3px] sm:hidden")}
      {row(peaks, "hidden gap-[2px] sm:flex")}
    </>
  );
}

/** The hero's proof: real clips rendered by the Kokoro engine, playable in place. Progress is written
 *  straight to the DOM from requestAnimationFrame so the 160-bar waveform never re-renders per frame. */
export function VoiceReel() {
  const reduce = useReducedMotion();
  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [wordIdx, setWordIdx] = useState(-1);
  const audio = useRef<HTMLAudioElement | null>(null);
  const playedLayer = useRef<HTMLDivElement>(null);
  const timeLabel = useRef<HTMLSpanElement>(null);
  const ring = useRef<HTMLSpanElement>(null);
  // The Hindi voice labels are script text. On a phone the first one sits just under the first screen, so the Devanagari
  // face loads only once that label itself is on screen, not on page load (watching the row or the list would not do:
  // their tops are above the fold). The player's Hindi line is rendered only after the reader picks that clip, so it
  // applies its face at once.
  const hindiLabel = useRef<HTMLSpanElement>(null);
  const hindiNear = useScriptFontsNear(hindiLabel, "0px");
  const raf = useRef<number>(0);
  // Web Audio is created lazily on the first click only. If the browser refuses, `analyser` stays null and the
  // ring falls back to its static "playing" look. `reduceRef` lets the long-lived callbacks see the latest preference.
  const ctxRef = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const freq = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const triedWebAudio = useRef(false);
  const level = useRef(0);
  const reduceRef = useRef(false);
  reduceRef.current = !!reduce;
  const wordCount = useRef(0);
  const clip = VOICE_CLIPS[active];
  const words = clip.text.split(/\s+/);
  wordCount.current = words.length;

  const resetRing = useCallback(() => {
    level.current = 0;
    const r = ring.current;
    if (r) {
      r.style.transform = "";
      r.style.opacity = "";
      r.style.boxShadow = "";
    }
  }, []);

  const paint = useCallback(() => {
    const a = audio.current;
    if (!a) return;
    const p = a.duration ? a.currentTime / a.duration : 0;
    if (playedLayer.current) playedLayer.current.style.clipPath = `inset(0 ${100 - p * 100}% 0 0)`;
    if (timeLabel.current) timeLabel.current.textContent = fmt(a.currentTime);
    const idx = Math.min(wordCount.current - 1, Math.floor(p * wordCount.current));
    setWordIdx((prev) => (prev === idx ? prev : idx));

    const an = analyser.current;
    const r = ring.current;
    if (an && r && freq.current && !reduceRef.current) {
      an.getByteFrequencyData(freq.current);
      // Voice energy sits in the low bins of a 256-point FFT (about 0 to 2.8 kHz at 44.1 kHz).
      let sum = 0;
      for (let i = 1; i < 24; i++) sum += freq.current[i];
      const raw = Math.min(1, (sum / 23 / 255) * 1.7);
      level.current = level.current * 0.6 + raw * 0.4;
      const l = level.current;
      r.style.transform = `scale(${(1 + l * 0.17).toFixed(3)})`;
      r.style.opacity = String(0.4 + l * 0.6);
      r.style.boxShadow = `0 0 ${Math.round(l * 12)}px var(--s-marigold)`;
    }
    if (!a.paused) raf.current = requestAnimationFrame(paint);
  }, []);

  useEffect(() => {
    const a = new Audio();
    a.preload = "auto";
    audio.current = a;
    // The element is the source of truth, so another sample (or the browser) pausing us updates the UI too.
    const onPlaying = () => {
      setLoading(false);
      setPlaying(true);
      cancelAnimationFrame(raf.current);
      raf.current = requestAnimationFrame(paint);
    };
    const onWaiting = () => !a.paused && setLoading(true);
    const onPause = () => {
      setPlaying(false);
      setLoading(false);
      cancelAnimationFrame(raf.current);
      resetRing();
    };
    const onEnd = () => {
      setPlaying(false);
      setLoading(false);
      setWordIdx(wordCount.current);
      resetRing();
    };
    const onError = () => {
      setPlaying(false);
      setLoading(false);
      resetRing();
    };
    a.addEventListener("playing", onPlaying);
    a.addEventListener("waiting", onWaiting);
    a.addEventListener("pause", onPause);
    a.addEventListener("ended", onEnd);
    a.addEventListener("error", onError);
    return () => {
      a.pause();
      a.removeEventListener("playing", onPlaying);
      a.removeEventListener("waiting", onWaiting);
      a.removeEventListener("pause", onPause);
      a.removeEventListener("ended", onEnd);
      a.removeEventListener("error", onError);
      cancelAnimationFrame(raf.current);
      ctxRef.current?.close().catch(() => {});
      ctxRef.current = null;
      analyser.current = null;
      triedWebAudio.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** First click only: build the AudioContext and analyser. Any failure leaves audio playing normally, without the ring motion. */
  const ensureAnalyser = useCallback(async (a: HTMLAudioElement) => {
    if (triedWebAudio.current || reduceRef.current) return;
    triedWebAudio.current = true;
    try {
      const AC: typeof AudioContext | undefined =
        window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AC) return;
      const ctx = new AC();
      if (ctx.state === "suspended") await ctx.resume();
      // Routing an element through a context that is not running would silence it, so bail out instead.
      if (ctx.state !== "running") {
        ctx.close().catch(() => {});
        return;
      }
      const src = ctx.createMediaElementSource(a);
      const an = ctx.createAnalyser();
      an.fftSize = 256;
      an.smoothingTimeConstant = 0.78;
      src.connect(an);
      an.connect(ctx.destination);
      ctxRef.current = ctx;
      analyser.current = an;
      freq.current = new Uint8Array(an.frequencyBinCount);
    } catch {
      /* silent fallback */
    }
  }, []);

  const start = useCallback(
    async (index: number) => {
      const a = audio.current;
      if (!a) return;
      const next = VOICE_CLIPS[index];
      if (!a.src.endsWith(next.src)) {
        a.src = next.src;
        a.currentTime = 0;
      }
      setActive(index);
      setWordIdx(-1);
      setLoading(true);
      await ensureAnalyser(a);
      if (ctxRef.current?.state === "suspended") ctxRef.current.resume().catch(() => {});
      claimAudio(a); // pauses any other sample on the page
      try {
        await a.play();
      } catch {
        setPlaying(false);
        setLoading(false);
      }
    },
    [ensureAnalyser]
  );

  const toggle = () => {
    const a = audio.current;
    if (!a) return;
    if (playing || loading) {
      a.pause();
    } else {
      if (a.ended || !a.src) a.currentTime = 0;
      start(active);
    }
  };

  const select = (i: number) => {
    if (i === active && (playing || loading)) return toggle();
    if (playedLayer.current) playedLayer.current.style.clipPath = "inset(0 100% 0 0)";
    if (timeLabel.current) timeLabel.current.textContent = "0:00";
    start(i);
  };

  const status = loading ? "Loading" : playing ? "Playing" : "Tap play";

  return (
    <div id="voices" className="grid overflow-hidden rounded-[var(--s-r-card)] border border-[var(--s-line)] bg-[var(--s-paper)] shadow-[var(--s-shadow-paper)] md:grid-cols-[18.75rem_1fr]">
      <ul className="border-b border-[var(--s-line)] p-2 md:border-b-0 md:border-r" aria-label="Voices">
        {VOICE_CLIPS.map((v, i) => {
          const isActive = i === active;
          return (
            <li key={v.id}>
              <button
                type="button"
                onClick={() => select(i)}
                aria-pressed={isActive}
                className="relative flex min-h-[3.25rem] w-full items-center gap-3 rounded-xl px-3.5 py-3 text-left transition-colors hover:bg-[var(--s-ground)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[var(--s-ink)]"
              >
                {isActive && (
                  <motion.span
                    layoutId="reel-active"
                    className="absolute inset-0 rounded-xl bg-[var(--s-ground)]"
                    transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 420, damping: 36 }}
                  />
                )}
                <span className="relative flex h-5 w-5 items-end justify-center gap-[2px]" aria-hidden>
                  {isActive && loading ? (
                    <Loader2 className={`h-4 w-4 self-center text-[var(--s-ink)] ${reduce ? "" : "animate-spin"}`} />
                  ) : (
                    [0.55, 1, 0.75].map((h, k) => (
                      <span
                        key={k}
                        className={`w-[3px] rounded-full ${isActive ? (playing ? "bg-[var(--s-ink)]" : "bg-[var(--s-marigold)]") : "bg-[var(--s-bar)]"} ${
                          isActive && playing && !reduce ? "animate-[eq_0.9s_ease-in-out_infinite]" : ""
                        }`}
                        style={{ height: `${h * 100}%`, animationDelay: `${k * 0.15}s` }}
                      />
                    ))
                  )}
                </span>
                <span className="relative min-w-0 flex-1">
                  <span className="block s-body font-medium text-[var(--s-ink)]">
                    {v.name}
                    {isActive && (playing || loading) && <span className="sr-only"> ({loading ? "loading" : "playing"})</span>}
                  </span>
                  <span ref={i === firstHindi ? hindiLabel : undefined} className={`block s-small text-[var(--s-mute)] ${hindiNear && v.lang === "hi" ? SCRIPT_FONT.hi : ""}`}>{v.language}</span>
                </span>
                <span className="relative s-label tabular-nums text-[var(--s-mute)]">{fmt(v.duration)}</span>
              </button>
            </li>
          );
        })}
      </ul>

      <div className="flex min-w-0 flex-col gap-6 p-5 sm:p-8">
        <div className="flex items-center gap-4">
          <div className="relative h-14 w-14 shrink-0">
            {/* Audio-reactive ring. JS drives scale and glow from the analyser; without it (reduced motion, blocked Web Audio) the same ring just sits there, solid, while playing. */}
            <span
              ref={ring}
              aria-hidden
              data-testid="reel-ring"
              data-playing={playing}
              className="pointer-events-none absolute -inset-1.5 rounded-full border-2 border-[var(--s-marigold)] opacity-0 transition-opacity duration-200 data-[playing=true]:opacity-100 motion-reduce:transition-none"
            />
            <button
              type="button"
              onClick={toggle}
              aria-label={loading ? `Loading ${clip.name}, press to cancel` : playing ? `Pause ${clip.name}` : `Play ${clip.name}`}
              aria-busy={loading}
              className="relative grid h-14 w-14 place-items-center rounded-full bg-[image:var(--s-brand-gradient)] text-[#FBFAF7] shadow-[inset_0_1px_0_rgba(255,250,240,0.18),var(--s-shadow-float)] transition-transform duration-200 hover:scale-[1.04] active:scale-95 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[var(--s-ink)] motion-reduce:transition-none"
            >
              {loading ? (
                <Loader2 className={`h-5 w-5 ${reduce ? "" : "animate-spin"}`} aria-hidden />
              ) : playing ? (
                <Pause className="h-5 w-5" aria-hidden />
              ) : (
                <Play className="h-5 w-5 translate-x-[1px]" aria-hidden />
              )}
            </button>
          </div>
          <div className="min-w-0">
            <p className="s-body font-medium text-[var(--s-ink)]">
              {clip.name} <span className={`font-normal text-[var(--s-mute)] ${clip.lang === "hi" ? SCRIPT_FONT.hi : ""}`}>· {clip.language}</span>
            </p>
            <p className="s-label tabular-nums text-[var(--s-mute)]">
              <span ref={timeLabel}>0:00</span> / {fmt(clip.duration)}
              <span role="status" data-testid="reel-status" className={`ml-2 inline-flex items-center gap-1.5 ${playing || loading ? "font-medium text-[var(--s-ink)]" : ""}`}>
                <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${playing ? "bg-[var(--s-marigold)]" : loading ? "bg-[var(--s-mute)]" : "bg-[var(--s-bar)]"}`} />
                {status}
              </span>
            </p>
          </div>
        </div>

        <div className="relative h-28 sm:h-36" aria-hidden>
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={clip.id}
              className="absolute inset-0"
              initial={reduce ? false : { opacity: 0, scaleY: 0.3 }}
              animate={{ opacity: 1, scaleY: 1 }}
              exit={reduce ? undefined : { opacity: 0, scaleY: 0.3 }}
              transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
            >
              <Bars peaks={clip.peaks} tone="var(--s-bar)" />
              <div ref={playedLayer} className="absolute inset-0" style={{ clipPath: "inset(0 100% 0 0)" }}>
                <Bars peaks={clip.peaks} tone="var(--s-marigold)" />
              </div>
            </motion.div>
          </AnimatePresence>
        </div>

        <p lang={clip.lang} className={`s-lede leading-snug text-[var(--s-ink)] ${clip.lang === "hi" ? SCRIPT_FONT.hi : ""}`}>
          {words.map((w, i) => (
            <span key={`${clip.id}-${i}`} className={`transition-colors duration-200 ${i <= wordIdx ? "text-[var(--s-ink)]" : "text-[var(--s-mute)]"}`}>
              {w}{" "}
            </span>
          ))}
        </p>

        <p className="mt-auto s-small text-[var(--s-mute)]">Rendered by BonClay on the open Kokoro-82M engine. Unedited audio.</p>
      </div>
    </div>
  );
}
