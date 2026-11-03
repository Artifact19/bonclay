/**
 * Voice-activity detection for hands-free talk: a pure state machine fed with one loudness (RMS) reading per tick.
 * No Web Audio, no timers and no DOM in here, so it can be tested with synthetic sequences; the hook that owns the
 * microphone is `useHandsFree`.
 *
 * Events, in the order they happen for one utterance:
 *   candidate  loudness went above the threshold: start recording so the first syllable is not lost
 *   abort      it was only a blip (it dropped again before it counted as speech): throw that recording away
 *   start      speech is confirmed (above the threshold for `startMs`)
 *   barge-in   speech is confirmed while the agent is talking (clearly louder than its playback for `bargeMs`)
 *   end        speech stopped for `endSilenceMs` and it was long enough: send the recording
 *   discard    speech stopped but it was shorter than `minUtteranceMs`: throw the recording away
 */
export interface VadConfig {
  /** Measure the room's noise floor over this long before anything can be detected. */
  calibrationMs: number;
  /** Loudness must stay above the threshold this long to count as speech. */
  startMs: number;
  /** Silence this long after speech ends the utterance. */
  endSilenceMs: number;
  /** Utterances shorter than this (first loud sample to last loud sample) are ignored. */
  minUtteranceMs: number;
  /** A dip below the threshold shorter than this does not interrupt a candidate (speech is not a flat tone). */
  dipMs: number;
  /** Threshold = max(floor * floorRatio, floor + margin). RMS is 0..1. */
  floorRatio: number;
  margin: number;
  /** Time constant (ms) with which the noise floor follows slow changes in the room while idle. */
  floorAdaptMs: number;
  /** While the agent is talking: how long the user must clearly out-talk it to cut in. */
  bargeMs: number;
  /** ... and by how much: loudness must exceed the playback level seen so far times this. */
  bargeRatio: number;
  /** ... and at least this loud in absolute terms, and at least `bargeThresholdFactor` times the normal threshold. */
  bargeMin: number;
  bargeThresholdFactor: number;
  /** At the start of playback the speaker is only measured (echo before the echo canceller settles), not acted on. */
  echoGraceMs: number;
  /** After playback ends, ignore the room's reverb tail for this long. */
  tailGraceMs: number;
  /** How fast the remembered playback level decays (per second, 0..1). */
  echoDecayPerSec: number;
}

export const DEFAULT_VAD: VadConfig = {
  calibrationMs: 500,
  startMs: 150,
  endSilenceMs: 800,
  minUtteranceMs: 400,
  dipMs: 120,
  floorRatio: 3,
  margin: 0.012,
  floorAdaptMs: 4000,
  bargeMs: 300,
  bargeRatio: 2.5,
  bargeMin: 0.06,
  bargeThresholdFactor: 2,
  echoGraceMs: 500,
  tailGraceMs: 400,
  echoDecayPerSec: 0.5,
};

export type VadEvent = "candidate" | "abort" | "start" | "barge-in" | "end" | "discard";
export type VadPhase = "calibrating" | "idle" | "speech";

export interface VadState {
  phase: VadPhase;
  /** Time of the first reading. */
  t0: number | null;
  lastT: number | null;
  calSamples: number[];
  floor: number;
  /** When loudness first went above the threshold (a candidate is running), else null. */
  aboveSince: number | null;
  /** Last reading above the threshold. */
  lastAbove: number;
  /** Start of the confirmed utterance. */
  speechStart: number;
  wasPlaying: boolean;
  playStart: number;
  /** Readings are ignored until this time (reverb after playback). */
  graceUntil: number;
  /** Loudest recent playback level as heard by the mic. */
  echo: number;
}

export interface VadInput {
  /** Root-mean-square of the latest audio frame, 0..1. */
  rms: number;
  /** Milliseconds on any monotonic clock. */
  now: number;
  /** The agent's voice is coming out of the speakers. */
  playing: boolean;
}

export const initVad = (): VadState => ({
  phase: "calibrating",
  t0: null,
  lastT: null,
  calSamples: [],
  floor: 0,
  aboveSince: null,
  lastAbove: 0,
  speechStart: 0,
  wasPlaying: false,
  playStart: 0,
  graceUntil: 0,
  echo: 0,
});

/** Loudness above which a reading counts as speech while the agent is quiet. */
export const speechThreshold = (floor: number, cfg: VadConfig = DEFAULT_VAD): number =>
  Math.max(floor * cfg.floorRatio, floor + cfg.margin);

/** Loudness above which a reading counts as the user cutting in while the agent talks. */
export const bargeThreshold = (s: Pick<VadState, "floor" | "echo">, cfg: VadConfig = DEFAULT_VAD): number =>
  Math.max(speechThreshold(s.floor, cfg) * cfg.bargeThresholdFactor, cfg.bargeMin, s.echo * cfg.bargeRatio);

/** Mean of the quietest 70% of the samples: a few early words must not inflate the floor. */
function quietMean(samples: number[]): number {
  if (!samples.length) return 0;
  const sorted = [...samples].sort((a, b) => a - b);
  const keep = sorted.slice(0, Math.max(1, Math.ceil(sorted.length * 0.7)));
  return keep.reduce((a, b) => a + b, 0) / keep.length;
}

/** Advance the machine by one reading. Returns the new state (the old one is untouched) and what happened. */
export function stepVad(prev: VadState, input: VadInput, cfg: VadConfig = DEFAULT_VAD): { state: VadState; events: VadEvent[] } {
  const { rms, now, playing } = input;
  const s: VadState = { ...prev, calSamples: prev.calSamples };
  const events: VadEvent[] = [];
  if (s.t0 === null) s.t0 = now;
  const dt = s.lastT === null ? 0 : Math.max(0, now - s.lastT);
  s.lastT = now;

  if (playing && !prev.wasPlaying) {
    s.playStart = now;
    s.echo = 0;
  }
  if (!playing && prev.wasPlaying) s.graceUntil = now + cfg.tailGraceMs;
  s.wasPlaying = playing;

  if (s.phase === "calibrating") {
    s.calSamples = [...prev.calSamples, rms];
    s.floor = quietMean(s.calSamples);
    if (now - s.t0 >= cfg.calibrationMs) {
      s.phase = "idle";
      s.calSamples = [];
    }
    return { state: s, events };
  }

  // Speech in progress: only silence matters. (After a barge-in the hook has already stopped the agent.)
  if (s.phase === "speech") {
    if (rms > speechThreshold(s.floor, cfg)) {
      s.lastAbove = now;
    } else if (now - s.lastAbove >= cfg.endSilenceMs) {
      s.phase = "idle";
      s.aboveSince = null;
      events.push(s.lastAbove - s.speechStart >= cfg.minUtteranceMs ? "end" : "discard");
    }
    return { state: s, events };
  }

  // Idle. The agent talking, or its tail just fading out, changes what counts as the user speaking.
  const inTail = !playing && now < s.graceUntil;
  const inEchoGrace = playing && now - s.playStart < cfg.echoGraceMs;
  const threshold = playing ? bargeThreshold(s, cfg) : speechThreshold(s.floor, cfg);

  if (playing) {
    // Remember how loud the agent sounds in the mic, ignoring readings that look like the user.
    if (inEchoGrace || rms < threshold) {
      s.echo = Math.max(rms, s.echo * Math.pow(cfg.echoDecayPerSec, dt / 1000));
    } else {
      s.echo = s.echo * Math.pow(cfg.echoDecayPerSec, dt / 1000);
    }
  }

  const blocked = inTail || inEchoGrace;
  const loud = !blocked && rms > threshold;

  if (loud) {
    s.lastAbove = now;
    if (s.aboveSince === null) {
      s.aboveSince = now;
      events.push("candidate");
    }
    const hold = playing ? cfg.bargeMs : cfg.startMs;
    // The first loud reading counts from itself: `hold` must elapse between it and a reading that is still loud.
    if (now - s.aboveSince >= hold) {
      s.phase = "speech";
      s.speechStart = s.aboveSince;
      events.push(playing ? "barge-in" : "start");
    }
  } else {
    if (s.aboveSince !== null && (blocked || now - s.lastAbove > cfg.dipMs)) {
      s.aboveSince = null;
      events.push("abort");
    }
    // Follow slow changes in the room's noise, but only while it is not mixed with the agent or the user.
    if (s.aboveSince === null && !playing && !inTail && rms <= threshold && dt > 0) {
      s.floor += (rms - s.floor) * (1 - Math.exp(-dt / cfg.floorAdaptMs));
    }
  }
  return { state: s, events };
}

/* ---------------------------------------------------------------------------------------------------------------
 * Silero VAD (the neural detector, `@ricky0123/vad-web`) decides where speech starts and ends. It does not know the
 * agent is talking, so this small gate sits beside it, fed with Silero's speech probability (0..1, one per ~32 ms
 * frame) and answers the two questions the model cannot:
 *   candidate  the voice just got likely enough to start recording, so the first syllable is not lost
 *   barge-in   while the agent talks, the user has been clearly speaking for `bargeMs`: cut the agent off
 * It is pure for the same reason as `stepVad`: tested with synthetic sequences, no model and no timers.
 * ------------------------------------------------------------------------------------------------------------- */

/** Where the Silero model, its audio worklet and the ONNX runtime's wasm are served from: copied into /public by `npm run vad:assets`. */
export const SILERO_ASSET_PATH = "/vad/";

export interface SileroConfig {
  /** Probability at which a frame counts as speech and a segment starts. */
  positiveSpeechThreshold: number;
  /** Probability below which a frame counts as silence (hysteresis: between the two nothing changes). */
  negativeSpeechThreshold: number;
  /** Silence this long ends the utterance. */
  redemptionMs: number;
  /** Segments with less speech than this are dropped as noise. */
  minSpeechMs: number;
  /** Probability at which a frame makes us start recording early (before the segment is confirmed). */
  candidateProb: number;
  /** While the agent talks, a frame must be at least this likely to be speech to count towards a barge-in. */
  bargeProb: number;
  /** ... for this long without a break. */
  bargeMs: number;
  /** A break shorter than this does not restart the barge-in count (speech is not continuous). */
  bargeDipMs: number;
  /** The first moments of playback are only measured (the echo canceller has not settled). */
  echoGraceMs: number;
  /** After playback ends, ignore the room's reverb tail for this long. */
  tailGraceMs: number;
  /** A recording started early is dropped if Silero never confirmed speech within this long of the last likely frame. */
  candidateTimeoutMs: number;
  /** Give up loading the model after this long and use the built-in detector instead. */
  loadTimeoutMs: number;
}

export const DEFAULT_SILERO: SileroConfig = {
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.35,
  redemptionMs: 700,
  minSpeechMs: 400,
  candidateProb: 0.3,
  bargeProb: 0.85,
  bargeMs: 300,
  bargeDipMs: 120,
  echoGraceMs: 500,
  tailGraceMs: 400,
  candidateTimeoutMs: 600,
  loadTimeoutMs: 20000,
};

export interface SileroGateState {
  /** When the current run of likely frames began, else null. */
  loudSince: number | null;
  /** Last frame that was likely enough. */
  lastLoud: number;
  /** The barge-in for the current run has been reported. */
  fired: boolean;
  wasPlaying: boolean;
  playStart: number;
  /** Frames are ignored until this time (reverb after playback). */
  graceUntil: number;
}

export interface SileroGateInput {
  /** Silero's probability that the latest frame is speech, 0..1. */
  prob: number;
  /** Milliseconds on any monotonic clock. */
  now: number;
  /** The agent's voice is coming out of the speakers. */
  playing: boolean;
}

export const initSileroGate = (): SileroGateState => ({
  loudSince: null,
  lastLoud: 0,
  fired: false,
  wasPlaying: false,
  playStart: 0,
  graceUntil: 0,
});

/** Advance the gate by one frame. `blocked` means the frame is echo or reverb: it must not start or confirm anything. */
export function stepSileroGate(
  prev: SileroGateState,
  input: SileroGateInput,
  cfg: SileroConfig = DEFAULT_SILERO
): { state: SileroGateState; candidate: boolean; bargeIn: boolean; blocked: boolean } {
  const { prob, now, playing } = input;
  const s: SileroGateState = { ...prev };
  if (playing && !prev.wasPlaying) {
    s.playStart = now;
    s.loudSince = null;
    s.fired = false;
  }
  if (!playing && prev.wasPlaying) {
    s.graceUntil = now + cfg.tailGraceMs;
    s.loudSince = null;
    s.fired = false;
  }
  s.wasPlaying = playing;

  const inTail = !playing && now < s.graceUntil;
  const inEchoGrace = playing && now - s.playStart < cfg.echoGraceMs;
  const blocked = inTail || inEchoGrace;
  const loud = !blocked && prob >= (playing ? cfg.bargeProb : cfg.candidateProb);

  let bargeIn = false;
  if (loud) {
    s.lastLoud = now;
    if (s.loudSince === null) s.loudSince = now;
    if (playing && !s.fired && now - s.loudSince >= cfg.bargeMs) {
      s.fired = true;
      bargeIn = true;
    }
  } else if (s.loudSince !== null && (blocked || now - s.lastLoud > cfg.bargeDipMs)) {
    s.loudSince = null;
    s.fired = false;
  }
  return { state: s, candidate: loud, bargeIn, blocked };
}

/** Whether a segment that Silero found must be thrown away because it is the agent's own voice (or its reverb). */
export const isEchoSegment = (s: { startedWhileAgentAudible: boolean; heardAgentDuring: boolean; bargedIn: boolean }): boolean =>
  !s.bargedIn && (s.startedWhileAgentAudible || s.heardAgentDuring);

/** Root-mean-square of a frame of samples (-1..1). */
export function frameRms(frame: ArrayLike<number>): number {
  if (!frame.length) return 0;
  let sum = 0;
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
  return Math.sqrt(sum / frame.length);
}
