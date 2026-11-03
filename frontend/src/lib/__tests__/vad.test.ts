import {
  bargeThreshold,
  DEFAULT_SILERO,
  DEFAULT_VAD,
  frameRms,
  initSileroGate,
  initVad,
  isEchoSegment,
  speechThreshold,
  stepSileroGate,
  stepVad,
  type SileroGateState,
  type VadEvent,
  type VadState,
} from "@/lib/vad";

const TICK = 40;

/** Feeds `[rms, durationMs, playing?]` segments tick by tick and collects the events with their times. */
function run(segments: [number, number, boolean?][], start?: { state: VadState; now: number }) {
  let state = start?.state ?? initVad();
  let now = start?.now ?? 0;
  const events: { at: number; ev: VadEvent }[] = [];
  for (const [rms, ms, playing = false] of segments) {
    for (let t = 0; t < ms; t += TICK) {
      const step = stepVad(state, { rms, now, playing }, DEFAULT_VAD);
      state = step.state;
      step.events.forEach((ev) => events.push({ at: now, ev }));
      now += TICK;
    }
  }
  return { state, now, events, names: events.map((e) => e.ev) };
}

const QUIET = 0.004; // room noise
const SPEECH = 0.12;
const CALIBRATE: [number, number] = [QUIET, 600];

describe("VAD", () => {
  it("calibrates a noise floor and never reports an utterance for noise alone", () => {
    const r = run([CALIBRATE, [QUIET * 1.5, 5000], [QUIET, 5000]]);
    expect(r.names).toEqual([]);
    expect(r.state.phase).toBe("idle");
    expect(r.state.floor).toBeGreaterThan(0.001);
    expect(r.state.floor).toBeLessThan(0.01);
  });

  it("does not let a few early words inflate the noise floor", () => {
    const r = run([[QUIET, 440], [SPEECH, 160]]);
    expect(r.state.phase).toBe("idle");
    expect(r.state.floor).toBeLessThan(0.05);
  });

  it("speech then silence gives exactly one utterance, ended after the trailing silence", () => {
    const r = run([CALIBRATE, [SPEECH, 1500], [QUIET, 1500]]);
    expect(r.names).toEqual(["candidate", "start", "end"]);
    const [cand, start, end] = r.events;
    expect(start.at - cand.at).toBeGreaterThanOrEqual(DEFAULT_VAD.startMs);
    expect(start.at - cand.at).toBeLessThan(DEFAULT_VAD.startMs + 2 * TICK);
    const lastLoud = cand.at + 1500 - TICK;
    expect(end.at - lastLoud).toBeGreaterThanOrEqual(DEFAULT_VAD.endSilenceMs);
    expect(end.at - lastLoud).toBeLessThan(DEFAULT_VAD.endSilenceMs + 2 * TICK);
  });

  it("keeps one utterance together across short pauses and dips between words", () => {
    const r = run([CALIBRATE, [SPEECH, 600], [QUIET, 400], [SPEECH, 600], [QUIET, 100], [SPEECH, 400], [QUIET, 1500]]);
    expect(r.names.filter((n) => n === "end")).toHaveLength(1);
    expect(r.names.filter((n) => n === "start")).toHaveLength(1);
    expect(r.names).not.toContain("discard");
  });

  it("ignores a short blip: it never even becomes speech", () => {
    const r = run([CALIBRATE, [SPEECH, 80], [QUIET, 3000]]);
    expect(r.names).toEqual(["candidate", "abort"]);
  });

  it("ignores an utterance shorter than the minimum even though it was confirmed as speech", () => {
    const r = run([CALIBRATE, [SPEECH, 240], [QUIET, 2000]]);
    expect(r.names).toEqual(["candidate", "start", "discard"]);
  });

  it("can report a second utterance after the first", () => {
    const r = run([CALIBRATE, [SPEECH, 1000], [QUIET, 1500], [SPEECH, 1000], [QUIET, 1500]]);
    expect(r.names).toEqual(["candidate", "start", "end", "candidate", "start", "end"]);
  });

  it("adapts to a room that gets noisier slowly, without hearing speech in it", () => {
    const ramp: [number, number][] = Array.from({ length: 40 }, (_, k) => [QUIET + k * 0.0004, 400]);
    const r = run([CALIBRATE, ...ramp, [0.02, 3000]]);
    expect(r.names).toEqual([]);
    expect(r.state.floor).toBeGreaterThan(0.012);
  });

  describe("while the agent is talking", () => {
    const ECHO = 0.03; // the agent's own voice, as heard by the mic
    const playing = (rms: number, ms: number): [number, number, boolean] => [rms, ms, true];

    it("does not treat the agent's own audio as a new turn, even when it is louder than normal speech threshold", () => {
      expect(ECHO).toBeGreaterThan(speechThreshold(0.004));
      const r = run([CALIBRATE, playing(ECHO, 6000), [QUIET, 1500]]);
      expect(r.names).toEqual([]);
    });

    it("does not treat a loud burst of echo in the first moments of playback as speech", () => {
      const r = run([CALIBRATE, playing(0.2, 400), playing(ECHO, 3000), [QUIET, 1500]]);
      expect(r.names).toEqual([]);
    });

    it("ignores the room's reverb right after playback stops", () => {
      const r = run([CALIBRATE, playing(ECHO, 2000), [ECHO, 300], [QUIET, 2000]]);
      expect(r.names).toEqual([]);
    });

    it("barges in only when the user is clearly louder than the playback, for long enough", () => {
      const r = run([CALIBRATE, playing(ECHO, 2000), playing(0.2, 1000)]);
      expect(r.names).toEqual(["candidate", "barge-in"]);
      const [cand, barge] = r.events;
      expect(barge.at - cand.at).toBeGreaterThanOrEqual(DEFAULT_VAD.bargeMs);
      expect(r.state.phase).toBe("speech");
    });

    it("does not barge in on a quiet interjection, a cough or a short burst", () => {
      const quiet = run([CALIBRATE, playing(ECHO, 2000), playing(0.05, 1500), playing(ECHO, 1000)]);
      expect(quiet.names).toEqual([]);
      const cough = run([CALIBRATE, playing(ECHO, 2000), playing(0.25, 160), playing(ECHO, 1500)]);
      expect(cough.names).toEqual(["candidate", "abort"]);
    });

    it("keeps barge-in conservative when the speaker is loud: the bar rises with the playback level", () => {
      const loud = run([CALIBRATE, playing(0.1, 2000), playing(0.15, 1000)]);
      expect(loud.names).toEqual([]);
      expect(bargeThreshold({ floor: 0.004, echo: 0.1 })).toBeGreaterThan(0.2);
    });

    it("after a barge-in, finishes the user's utterance once the agent has gone quiet", () => {
      let r = run([CALIBRATE, playing(ECHO, 2000), playing(0.2, 800)]);
      r = run([[0.2, 600], [QUIET, 1500]], r);
      expect(r.names).toEqual(["end"]);
    });
  });
});

describe("Silero gate", () => {
  const FRAME = 32; // one Silero v5 frame

  /** Feeds `[probability, durationMs, playing?]` segments frame by frame and collects what the gate reported. */
  function gate(segments: [number, number, boolean?][], start?: { state: SileroGateState; now: number }) {
    let state = start?.state ?? initSileroGate();
    let now = start?.now ?? 0;
    const out: { at: number; kind: "candidate" | "barge-in" }[] = [];
    for (const [prob, ms, playing = false] of segments) {
      for (let t = 0; t < ms; t += FRAME) {
        const step = stepSileroGate(state, { prob, now, playing }, DEFAULT_SILERO);
        state = step.state;
        if (step.candidate && out.at(-1)?.kind !== "candidate") out.push({ at: now, kind: "candidate" });
        if (step.bargeIn) out.push({ at: now, kind: "barge-in" });
        now += FRAME;
      }
    }
    return { state, now, out, kinds: out.map((o) => o.kind) };
  }
  const playing = (prob: number, ms: number): [number, number, boolean] => [prob, ms, true];

  it("asks to start recording as soon as the voice is likely, and never barges in while the agent is quiet", () => {
    const r = gate([[0.05, 500], [0.9, 2000]]);
    expect(r.kinds).toEqual(["candidate"]);
  });

  it("does not react to low probabilities", () => {
    expect(gate([[0.2, 3000]]).kinds).toEqual([]);
  });

  it("needs ~300 ms of sustained high-probability speech to barge in on the agent", () => {
    const r = gate([playing(0.1, 1000), playing(0.95, 1000)]);
    const barge = r.out.find((o) => o.kind === "barge-in");
    expect(barge).toBeDefined();
    const started = r.out.find((o) => o.kind === "candidate")!.at;
    expect(barge!.at - started).toBeGreaterThanOrEqual(DEFAULT_SILERO.bargeMs);
    expect(barge!.at - started).toBeLessThan(DEFAULT_SILERO.bargeMs + 2 * FRAME);
    expect(r.kinds.filter((k) => k === "barge-in")).toHaveLength(1);
  });

  it("ignores a short burst or medium-probability speech (the agent's echo) while it talks", () => {
    expect(gate([playing(0.1, 800), playing(0.95, 200), playing(0.1, 800)]).kinds.includes("barge-in")).toBe(false);
    expect(gate([playing(0.1, 800), playing(0.7, 5000)]).kinds).toEqual([]);
  });

  it("survives a tiny dip but restarts the count after a longer break", () => {
    expect(gate([playing(0.1, 800), playing(0.95, 160), playing(0.1, 64), playing(0.95, 200)]).kinds).toContain("barge-in");
    expect(gate([playing(0.1, 800), playing(0.95, 160), playing(0.1, 300), playing(0.95, 160)]).kinds).not.toContain("barge-in");
  });

  it("only measures during the first moments of playback (the echo canceller settling)", () => {
    const r = gate([playing(0.95, DEFAULT_SILERO.echoGraceMs - FRAME * 2)]);
    expect(r.kinds).toEqual([]);
  });

  it("ignores the reverb tail for a moment after the agent stops", () => {
    let r = gate([playing(0.1, 1000), [0.9, 300]]);
    expect(r.kinds).toEqual([]);
    r = gate([[0.9, 400]], r);
    expect(r.kinds).toEqual(["candidate"]);
  });

  it("flags a Silero segment as the agent's own voice unless the user cut in", () => {
    expect(isEchoSegment({ startedWhileAgentAudible: true, heardAgentDuring: true, bargedIn: false })).toBe(true);
    expect(isEchoSegment({ startedWhileAgentAudible: false, heardAgentDuring: true, bargedIn: false })).toBe(true);
    expect(isEchoSegment({ startedWhileAgentAudible: true, heardAgentDuring: true, bargedIn: true })).toBe(false);
    expect(isEchoSegment({ startedWhileAgentAudible: false, heardAgentDuring: false, bargedIn: false })).toBe(false);
  });

  it("measures loudness of a frame", () => {
    expect(frameRms(new Float32Array(0))).toBe(0);
    expect(frameRms(new Float32Array(512).fill(0.5))).toBeCloseTo(0.5);
  });
});
