"use client";

import { useEffect, useRef, useState } from "react";
import type { MicVAD } from "@ricky0123/vad-web";
import {
  DEFAULT_SILERO,
  DEFAULT_VAD,
  SILERO_ASSET_PATH,
  frameRms,
  initSileroGate,
  initVad,
  isEchoSegment,
  stepSileroGate,
  stepVad,
  type SileroConfig,
  type VadConfig,
  type VadState,
} from "@/lib/vad";

export type HandsFreeStatus = "off" | "starting" | "listening" | "hearing" | "error";

/** Which detector is running: the Silero neural model, or the built-in loudness detector it falls back to. */
export type HandsFreeDetector = "silero" | "rms";

export interface UseHandsFreeOptions {
  /** Keep the microphone open and listen. Turning this off (or unmounting) releases the microphone. */
  enabled: boolean;
  /** Whether the agent's voice is coming out of the speakers right now. Read on every tick, so it may change freely. */
  isPlaying: () => boolean;
  /** One finished utterance, recorded and ready to send as a turn. */
  onUtterance: (audio: Blob) => void;
  /** The user clearly spoke over the agent: stop its playback. */
  onBargeIn?: () => void;
  /** The microphone could not be opened. */
  onError?: (message: string) => void;
  /** Loudness 0..1 on every tick, for a level meter. Not state: it must not re-render the page 25 times a second. */
  onLevel?: (level: number) => void;
  /** Tuning of the built-in loudness detector (the fallback). */
  config?: Partial<VadConfig>;
  /** Tuning of the Silero detector. */
  silero?: Partial<SileroConfig>;
  /** "rms" skips Silero and uses the loudness detector (the default is Silero, with the loudness detector as the fallback). */
  detector?: "auto" | "rms";
}

/** What the browser's echo canceller needs so the agent's own voice mostly stays out of the recording. */
export const HANDS_FREE_AUDIO: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

interface Recording {
  rec: MediaRecorder;
  send: boolean;
}

const TICK_MS = 40;
const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
/** Silero keeps this much audio from before the speech was confirmed (the recording is made separately, see below). */
const PRE_SPEECH_PAD_MS = 400;

function pickMime(): string | undefined {
  const supported = typeof MediaRecorder !== "undefined" ? MediaRecorder.isTypeSupported : undefined;
  if (typeof supported !== "function") return undefined;
  return MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m));
}

let fallbackLogged = false;
/** The fallback is silent for the user; the developer console hears about it once. */
function logFallbackOnce(err: unknown) {
  if (fallbackLogged) return;
  fallbackLogged = true;
  console.warn("[hands-free] Silero VAD could not be loaded; using the built-in loudness detector instead.", err);
}

/**
 * Hands-free talk: opens the microphone while `enabled`, finds each utterance, records it with MediaRecorder and hands
 * it to `onUtterance`. Everything is released when `enabled` goes false or on unmount.
 *
 * Silero VAD (`@ricky0123/vad-web`, loaded lazily, served from /vad/) decides where speech starts and ends; the gate
 * in `@/lib/vad` adds what the model cannot know (the agent is talking: ignore its echo, and require sustained clear speech before cutting it off). If Silero cannot load, the loudness state machine in `@/lib/vad`
 * does the same job. The recording itself is always MediaRecorder, so the audio sent to the server is the same either way.
 */
export function useHandsFree(opts: UseHandsFreeOptions): { status: HandsFreeStatus; detector: HandsFreeDetector | null } {
  const [status, setStatus] = useState<HandsFreeStatus>("off");
  const [detector, setDetector] = useState<HandsFreeDetector | null>(null);
  // The latest callbacks, so the microphone is not reopened whenever the page re-renders.
  const optsRef = useRef(opts);
  useEffect(() => {
    optsRef.current = opts;
  });

  const { enabled } = opts;
  useEffect(() => {
    if (!enabled) return;
    let disposed = false;
    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;
    let timer: ReturnType<typeof setInterval> | null = null;
    let recorder: Recording | null = null;
    let vad: MicVAD | null = null;
    /** Silero failed to load: whatever it left behind must not act any more. */
    let sileroDead = false;
    let state: VadState = initVad();
    let shown: HandsFreeStatus = "off";
    const show = (next: HandsFreeStatus) => {
      if (disposed || shown === next) return;
      shown = next;
      setStatus(next);
    };
    show("starting");
    const cfg: VadConfig = { ...DEFAULT_VAD, ...optsRef.current.config };
    const scfg: SileroConfig = { ...DEFAULT_SILERO, ...optsRef.current.silero };

    /** Start recording the stream; `finishRecorder(true)` sends it on, `finishRecorder(false)` throws it away. */
    const startRecorder = () => {
      if (!stream || recorder) return;
      try {
        const mime = pickMime();
        const rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
        const handle: Recording = { rec, send: false };
        const chunks: Blob[] = [];
        rec.ondataavailable = (ev) => ev.data.size > 0 && chunks.push(ev.data);
        rec.onstop = () => {
          if (!handle.send) return;
          const blob = new Blob(chunks, { type: rec.mimeType || mime || "audio/webm" });
          if (blob.size > 0 && !disposed) optsRef.current.onUtterance(blob);
        };
        rec.start();
        recorder = handle;
      } catch {
        recorder = null;
      }
    };
    const finishRecorder = (send: boolean) => {
      const handle = recorder;
      recorder = null;
      if (!handle) return;
      handle.send = send;
      try {
        if (handle.rec.state !== "inactive") handle.rec.stop();
      } catch {
        /* already stopped */
      }
    };

    const newContext = (): AudioContext => {
      const Ctx: typeof AudioContext = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const c = new Ctx();
      if (c.state === "suspended") void c.resume().catch(() => undefined);
      return c;
    };
    const closeContext = () => {
      const c = ctx;
      ctx = null;
      if (c && c.state !== "closed") void c.close().catch(() => undefined);
    };

    const cleanup = () => {
      disposed = true;
      if (timer) clearInterval(timer);
      timer = null;
      finishRecorder(false);
      const v = vad;
      vad = null;
      if (v) void v.destroy().catch(() => undefined);
      stream?.getTracks().forEach((t) => t.stop());
      stream = null;
      closeContext();
      optsRef.current.onLevel?.(0);
    };

    /** The built-in loudness detector: one RMS reading every TICK_MS through the `stepVad` state machine. */
    const startRms = (s: MediaStream) => {
      ctx = newContext();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      ctx.createMediaStreamSource(s).connect(analyser); // not connected to the speakers: no feedback
      const buf = new Float32Array(analyser.fftSize);
      setDetector("rms");
      show("listening");

      timer = setInterval(() => {
        analyser.getFloatTimeDomainData(buf);
        const rms = frameRms(buf);
        optsRef.current.onLevel?.(Math.min(1, rms * 6));
        const step = stepVad(state, { rms, now: performance.now(), playing: optsRef.current.isPlaying() }, cfg);
        state = step.state;
        for (const ev of step.events) {
          if (ev === "candidate") startRecorder();
          else if (ev === "abort" || ev === "discard") {
            finishRecorder(false);
            show("listening");
          } else if (ev === "start") show("hearing");
          else if (ev === "barge-in") {
            show("hearing");
            optsRef.current.onBargeIn?.();
          } else if (ev === "end") {
            finishRecorder(true);
            show("listening");
          }
        }
      }, TICK_MS);
    };

    /** Silero VAD. Resolves once the model is running; rejects if it cannot load (the caller falls back). */
    const startSilero = async (s: MediaStream) => {
      const { MicVAD: Mic } = await import("@ricky0123/vad-web");
      if (disposed) return;
      ctx = ctx ?? newContext();
      let gate = initSileroGate();
      /** The segment Silero is currently inside, with what we know about the agent's voice during it. */
      let seg: { startedWhileAgentAudible: boolean; heardAgentDuring: boolean; bargedIn: boolean } | null = null;
      let lastCandidateAt = 0;

      const loading = Mic.new({
        model: "v5",
        baseAssetPath: SILERO_ASSET_PATH,
        onnxWASMBasePath: SILERO_ASSET_PATH,
        positiveSpeechThreshold: scfg.positiveSpeechThreshold,
        negativeSpeechThreshold: scfg.negativeSpeechThreshold,
        redemptionMs: scfg.redemptionMs,
        minSpeechMs: scfg.minSpeechMs,
        preSpeechPadMs: PRE_SPEECH_PAD_MS,
        audioContext: ctx,
        // The microphone is ours (opened above with the echo-cancelling constraints and released in cleanup).
        getStream: async () => s,
        pauseStream: async () => undefined,
        resumeStream: async () => s,
        startOnLoad: true,
        onFrameProcessed: (probs, frame) => {
          if (disposed || sileroDead) return;
          const now = performance.now();
          const playing = optsRef.current.isPlaying();
          optsRef.current.onLevel?.(Math.min(1, frameRms(frame) * 6));
          const g = stepSileroGate(gate, { prob: probs.isSpeech, now, playing }, scfg);
          gate = g.state;
          if (seg && playing) seg.heardAgentDuring = true;
          if (g.candidate) {
            lastCandidateAt = now;
            startRecorder(); // early, so the first syllable is in the recording; thrown away again if it was nothing
          }
          if (g.bargeIn) {
            if (seg) seg.bargedIn = true;
            show("hearing");
            optsRef.current.onBargeIn?.();
          }
          if (recorder && !seg && now - lastCandidateAt > scfg.candidateTimeoutMs) finishRecorder(false);
        },
        onSpeechStart: () => {
          if (disposed || sileroDead) return;
          const audible = optsRef.current.isPlaying() || performance.now() < gate.graceUntil;
          seg = { startedWhileAgentAudible: audible, heardAgentDuring: false, bargedIn: false };
          startRecorder();
          if (!audible) show("hearing");
        },
        onVADMisfire: () => {
          if (disposed || sileroDead) return;
          seg = null;
          finishRecorder(false);
          show("listening");
        },
        onSpeechEnd: () => {
          if (disposed || sileroDead) return;
          const done = seg;
          seg = null;
          // What Silero heard while the agent was talking, and the user never cut in over, is the agent itself.
          finishRecorder(!!done && !isEchoSegment(done));
          show("listening");
        },
      });

      let timeout: ReturnType<typeof setTimeout> | undefined;
      try {
        vad = await Promise.race([
          loading,
          new Promise<never>((_, reject) => {
            timeout = setTimeout(() => reject(new Error("Silero VAD took too long to load")), scfg.loadTimeoutMs);
          }),
        ]);
      } catch (err) {
        // A model that arrives after we gave up must not keep listening in the background.
        sileroDead = true;
        void loading.then((late) => late.destroy()).catch(() => undefined);
        throw err;
      } finally {
        clearTimeout(timeout);
      }
      if (disposed) {
        void vad.destroy().catch(() => undefined);
        vad = null;
        return;
      }
      setDetector("silero");
      show("listening");
    };

    (async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ audio: HANDS_FREE_AUDIO });
        if (disposed) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        if (optsRef.current.detector !== "rms") {
          try {
            await startSilero(s);
            return;
          } catch (err) {
            if (disposed) return;
            logFallbackOnce(err);
            vad = null;
            finishRecorder(false);
          }
        }
        if (!disposed) startRms(s);
      } catch {
        if (disposed) return;
        cleanup();
        setStatus("error");
        optsRef.current.onError?.("Microphone access was blocked. Allow it in your browser settings to talk hands-free.");
      }
    })();

    return () => {
      cleanup();
      setStatus("off");
      setDetector(null);
    };
  }, [enabled]);

  return { status: enabled ? status : "off", detector: enabled ? detector : null };
}
