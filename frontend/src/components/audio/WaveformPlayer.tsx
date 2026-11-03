"use client";

import { useCallback, useEffect, useRef, useState, type MutableRefObject } from "react";
import { Download, Pause, Play } from "lucide-react";
import { AiGeneratedTag } from "./AiGeneratedTag";

export interface WaveformControls {
  play: () => void;
  pause: () => void;
  toggle: () => void;
}

export interface WaveformPlayerProps {
  /** Optional imperative handle so a parent can drive playback. */
  controlRef?: MutableRefObject<WaveformControls | null>;
  /** URL string or Blob/File of the audio to render. */
  src: string | Blob;
  /** Filename used for the download button. */
  filename?: string;
  /** Waveform height in px. */
  height?: number;
  autoPlay?: boolean;
  /** Hide the download button. */
  hideDownload?: boolean;
  /**
   * Show the "AI-generated" tag. Defaults to true for generated and agent audio. Pass false for the user's own
   * recordings (consent recording, voice sample preview). When left unset, a player with `hideDownload` (the
   * preview style used for those recordings) is not tagged.
   */
  aiGenerated?: boolean;
  className?: string;
  onReady?: (durationSeconds: number) => void;
  onError?: (message: string) => void;
}

const SPEEDS = [0.75, 1, 1.25, 1.5, 2];
const VIOLET = "#F0A23A";
const CYAN = "#E0782A";

function fmt(s: number) {
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${r.toString().padStart(2, "0")}`;
}

export function WaveformPlayer({
  src,
  filename = "bonclay-audio.wav",
  height = 72,
  autoPlay = false,
  hideDownload = false,
  aiGenerated,
  className = "",
  controlRef,
  onReady,
  onError,
}: WaveformPlayerProps) {
  const showAiTag = aiGenerated ?? !hideDownload;
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wsRef = useRef<any>(null);
  const cbRef = useRef({ onReady, onError });
  cbRef.current = { onReady, onError };

  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(1);

  useEffect(() => {
    let disposed = false;
    const el = containerRef.current;
    if (!el) return;
    setReady(false);
    setFailed(false);
    setPlaying(false);
    setTime(0);

    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    (async () => {
      const { default: WaveSurfer } = await import("wavesurfer.js");
      if (disposed) return;

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      const grad = ctx ? ctx.createLinearGradient(0, 0, 600, 0) : null;
      grad?.addColorStop(0, VIOLET);
      grad?.addColorStop(1, CYAN);

      const ws = WaveSurfer.create({
        container: el,
        height,
        waveColor: "rgba(148,163,184,0.38)",
        progressColor: grad ?? VIOLET,
        cursorColor: CYAN,
        cursorWidth: 2,
        barWidth: 3,
        barGap: 2,
        barRadius: 3,
        normalize: true,
        dragToSeek: true,
        // Reduced motion: no animated scrolling of the view while playing.
        autoScroll: !reduced,
      });
      wsRef.current = ws;

      ws.on("ready", (d: number) => {
        setReady(true);
        setDuration(d);
        cbRef.current.onReady?.(d);
        if (autoPlay) void ws.play().catch(() => undefined);
      });
      ws.on("timeupdate", (t: number) => setTime(t));
      ws.on("play", () => setPlaying(true));
      ws.on("pause", () => setPlaying(false));
      ws.on("finish", () => setPlaying(false));
      ws.on("error", (e: unknown) => {
        if (disposed) return;
        setFailed(true);
        cbRef.current.onError?.(String(e));
      });

      try {
        if (typeof src === "string") await ws.load(src);
        else await ws.loadBlob(src);
      } catch (e) {
        if (!disposed) {
          setFailed(true);
          cbRef.current.onError?.(String(e));
        }
      }
    })();

    return () => {
      disposed = true;
      try {
        wsRef.current?.destroy();
      } catch {
        /* already torn down */
      }
      wsRef.current = null;
    };
  }, [src, height, autoPlay]);

  const toggle = useCallback(() => {
    void wsRef.current?.playPause();
  }, []);

  useEffect(() => {
    if (!controlRef) return;
    controlRef.current = {
      play: () => void wsRef.current?.play(),
      pause: () => wsRef.current?.pause(),
      toggle: () => void wsRef.current?.playPause(),
    };
    return () => {
      controlRef.current = null;
    };
  }, [controlRef]);

  const cycleSpeed =useCallback(() => {
    const next = SPEEDS[(SPEEDS.indexOf(rate) + 1) % SPEEDS.length];
    setRate(next);
    wsRef.current?.setPlaybackRate(next, true);
  }, [rate]);

  const download = useCallback(() => {
    const url = typeof src === "string" ? src : URL.createObjectURL(src);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    if (typeof src !== "string") setTimeout(() => URL.revokeObjectURL(url), 2000);
  }, [src, filename]);

  const onKey = (e: React.KeyboardEvent) => {
    const ws = wsRef.current;
    if (!ws || !ready) return;
    if (e.key === "ArrowRight") {
      ws.setTime(Math.min(duration, ws.getCurrentTime() + 5));
      e.preventDefault();
    } else if (e.key === "ArrowLeft") {
      ws.setTime(Math.max(0, ws.getCurrentTime() - 5));
      e.preventDefault();
    }
  };

  return (
    <div
      className={`rounded-[var(--radius-card,16px)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3 sm:p-4 ${className}`}
      onKeyDown={onKey}
    >
      <div className="relative" style={{ minHeight: height }}>
        <div
          ref={containerRef}
          aria-label="Audio waveform. Click or drag to seek."
          className={ready ? "opacity-100" : "opacity-0"}
          style={{ transition: "opacity 300ms cubic-bezier(0.16,1,0.3,1)" }}
        />
        {!ready && !failed && (
          <div
            className="absolute inset-0 flex items-center gap-[3px] overflow-hidden"
            aria-hidden="true"
          >
            {Array.from({ length: 48 }).map((_, i) => (
              <span
                key={i}
                className="w-[3px] shrink-0 rounded-full bg-[hsl(var(--au-text-secondary)/0.2)] motion-safe:animate-pulse"
                style={{ height: 12 + ((i * 37) % 40) }}
              />
            ))}
          </div>
        )}
        {failed && (
          <p
            role="alert"
            className="absolute inset-0 flex items-center justify-center text-sm text-[var(--color-text-secondary)]"
          >
            This audio could not be loaded.
          </p>
        )}
      </div>

      <div className="mt-3 flex items-center gap-2 sm:gap-3">
        <button
          type="button"
          onClick={toggle}
          disabled={!ready}
          aria-label={playing ? "Pause" : "Play"}
          className="grid h-11 w-11 shrink-0 place-items-center rounded-full text-[var(--color-on-accent)] transition-transform active:scale-95 disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent-cyan)]"
          style={{ background: `linear-gradient(135deg, ${VIOLET}, ${CYAN})` }}
        >
          {playing ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5 translate-x-px" />}
        </button>

        <span className="text-xs tabular-nums text-[var(--color-text-secondary)]">
          {fmt(time)} / {fmt(duration)}
        </span>

        <div className="ml-auto flex items-center gap-2">
          {showAiTag && <AiGeneratedTag />}
          <button
            type="button"
            onClick={cycleSpeed}
            disabled={!ready}
            aria-label={`Playback speed ${rate}x. Click to change.`}
            className="h-9 min-w-[3.25rem] rounded-lg border border-[var(--color-border)] px-2 text-xs font-medium tabular-nums text-[var(--color-text-primary)] transition-colors hover:bg-[hsl(var(--au-text-primary)/0.06)] disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
          >
            {rate}x
          </button>
          {!hideDownload && (
            <button
              type="button"
              onClick={download}
              disabled={!ready}
              aria-label="Download audio"
              className="grid h-9 w-9 place-items-center rounded-lg border border-[var(--color-border)] text-[var(--color-text-primary)] transition-colors hover:bg-[hsl(var(--au-text-primary)/0.06)] disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
            >
              <Download className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default WaveformPlayer;
