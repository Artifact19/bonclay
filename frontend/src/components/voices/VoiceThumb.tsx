"use client";

import { useEffect, useRef } from "react";

/** Small deterministic PRNG so the same voice id always draws the same thumbnail. */
function hashSeed(input: string): number {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface VoiceThumbProps {
  seed: string;
  /** Real peaks (0..1) decoded from the sample, when available. */
  peaks?: number[] | null;
  className?: string;
}

/**
 * Canvas waveform / spectrogram thumbnail. Draws a faint spectrogram wash
 * behind mirrored bars. Derived from the voice id unless real peaks are given.
 */
export function VoiceThumb({ seed, peaks, className }: VoiceThumbProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (!w || !h) return;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const rand = mulberry32(hashSeed(seed));
      const hue = 255 + Math.floor(rand() * 40); // violet range
      const cols = Math.floor(w / 5);

      // spectrogram wash
      const cell = 5;
      const rows = Math.ceil(h / cell);
      for (let c = 0; c < cols; c++) {
        const energy = 0.25 + rand() * 0.75;
        for (let r = 0; r < rows; r++) {
          const f = 1 - r / rows;
          const a = Math.max(0, (f * energy - 0.25) * rand()) * 0.55;
          if (a < 0.03) continue;
          ctx.fillStyle = `hsla(${hue + f * 70}, 85%, ${50 + f * 15}%, ${a})`;
          ctx.fillRect(c * cell, r * cell, cell - 1, cell - 1);
        }
      }

      // mirrored waveform bars
      const bars = Math.floor(w / 4);
      const mid = h / 2;
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, "#F0A23A");
      grad.addColorStop(1, "#E0782A");
      ctx.fillStyle = grad;
      let prev = 0.5;
      for (let i = 0; i < bars; i++) {
        let v: number;
        if (peaks && peaks.length) {
          v = peaks[Math.min(peaks.length - 1, Math.floor((i / bars) * peaks.length))];
        } else {
          prev = prev * 0.6 + rand() * 0.4;
          const envelope = Math.sin((i / bars) * Math.PI) * 0.6 + 0.4;
          v = prev * envelope;
        }
        const bh = Math.max(2, v * h * 0.8);
        ctx.fillRect(i * 4, mid - bh / 2, 2.5, bh);
      }
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [seed, peaks]);

  return <canvas ref={ref} aria-hidden="true" className={className} style={{ width: "100%", height: "100%", display: "block" }} />;
}

export default VoiceThumb;
