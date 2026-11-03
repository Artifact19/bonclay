/**
 * The web call page (backend/webcall/v1/call.js + index.html): the SSE parser, the ordered clip queue, the voice
 * activity state machine (same numbers as src/lib/vad.ts), the small helpers, and the page controller driven with a
 * fake microphone, fetch and audio element.
 */
import fs from "fs";
import path from "path";

// eslint-disable-next-line @typescript-eslint/no-require-imports
const w = require("../../../../backend/webcall/v1/call.js");
import { DEFAULT_VAD as TS_VAD, initVad as tsInit, stepVad as tsStep } from "@/lib/vad";

const LINK = "awc_abcdefghijklmnopqrstuvwx012345";
const tick = () => new Promise((r) => setTimeout(r, 0));

describe("SSEParser and AudioQueue", () => {
  it("parses events split across chunks", () => {
    const p = new w.SSEParser();
    expect(p.push('event: transcript\ndata: {"user_text":"hi"}\n\nevent: sent')).toEqual([{ event: "transcript", data: { user_text: "hi" } }]);
    expect(p.push('ence\ndata: {"index":0}\n\n')).toEqual([{ event: "sentence", data: { index: 0 } }]);
  });

  it("plays clips in order, reports failures without blocking, and stop() drops the rest", async () => {
    const order: string[] = [];
    const q = new w.AudioQueue({
      fetchClip: (i: { text: string }) => (i.text === "bad" ? Promise.reject(new Error("x")) : Promise.resolve(`u:${i.text}`)),
      play: (url: string) => {
        order.push(url);
        return Promise.resolve();
      },
      onStart: (i: { text: string }) => order.push(`start:${i.text}`),
      onError: (_e: unknown, i: { text: string }) => order.push(`error:${i.text}`),
    });
    q.add({ text: "a" });
    q.add({ text: "bad" });
    q.add({ text: "c" });
    expect(q.running).toBe(true);
    await q.idle();
    expect(order).toEqual(["start:a", "u:a", "error:bad", "start:bad", "start:c", "u:c"]);
    expect(q.running).toBe(false);
    q.add({ text: "d" });
    q.stop();
    await q.idle();
    expect(q.running).toBe(false);
  });
});

describe("voice activity (same machine as src/lib/vad.ts)", () => {
  const run = (step: typeof w.stepVad, init: () => unknown, cfg: unknown, readings: Array<[number, boolean]>) => {
    let s = init();
    const events: string[] = [];
    readings.forEach(([rms, playing], i) => {
      const out = step(s, { rms, now: i * 40, playing }, cfg);
      s = out.state;
      events.push(...out.events);
    });
    return events;
  };
  const quiet = Array.from({ length: 14 }, () => [0.002, false] as [number, boolean]);
  const talk = (ms: number, level = 0.1, playing = false) => Array.from({ length: ms / 40 }, () => [level, playing] as [number, boolean]);
  const silence = (ms: number, playing = false) => talk(ms, 0.002, playing);

  it("finds one utterance: candidate, start, end", () => {
    const ev = run(w.stepVad, w.initVad, w.DEFAULT_VAD, [...quiet, ...talk(1200), ...silence(1000)]);
    expect(ev).toEqual(["candidate", "start", "end"]);
  });

  it("ignores a short blip and a too-short utterance", () => {
    expect(run(w.stepVad, w.initVad, w.DEFAULT_VAD, [...quiet, ...talk(80), ...silence(1000)])).toEqual(["candidate", "abort"]);
    expect(run(w.stepVad, w.initVad, w.DEFAULT_VAD, [...quiet, ...talk(200), ...silence(1000)])).toEqual(["candidate", "start", "discard"]);
  });

  it("reports a barge-in only for speech clearly louder than the agent's own voice", () => {
    const echo = [...quiet, ...silence(40, true), ...talk(2000, 0.03, true)];
    expect(run(w.stepVad, w.initVad, w.DEFAULT_VAD, echo)).not.toContain("barge-in");
    const cut = [...quiet, ...talk(1200, 0.03, true), ...talk(600, 0.4, true)];
    expect(run(w.stepVad, w.initVad, w.DEFAULT_VAD, cut)).toContain("barge-in");
  });

  it("behaves exactly like the dashboard's TypeScript machine on the same readings", () => {
    const readings: Array<[number, boolean]> = [...quiet, ...talk(1000), ...silence(900), ...talk(1200, 0.03, true), ...talk(700, 0.4, true), ...silence(1000)];
    const js = run(w.stepVad, w.initVad, w.DEFAULT_VAD, readings);
    const ts = run(tsStep as typeof w.stepVad, tsInit, TS_VAD, readings);
    expect(js).toEqual(ts);
    expect(js.length).toBeGreaterThan(3);
  });

  it("computes loudness and thresholds", () => {
    expect(w.frameRms([])).toBe(0);
    expect(w.frameRms([0.5, -0.5, 0.5, -0.5])).toBeCloseTo(0.5);
    expect(w.speechThreshold(0.01)).toBeCloseTo(0.03);
    expect(w.bargeThreshold({ floor: 0.01, echo: 0.1 })).toBeCloseTo(0.25);
  });
});

describe("helpers", () => {
  it("formats the clock, picks file extensions and reads the link id", () => {
    expect(w.formatClock(0)).toBe("0:00");
    expect(w.formatClock(61)).toBe("1:01");
    expect(w.formatClock(599.2)).toBe("9:59");
    expect(w.formatClock(-5)).toBe("0:00");
    expect(w.extFor("audio/webm;codecs=opus")).toBe("webm");
    expect(w.extFor("audio/mp4")).toBe("m4a");
    expect(w.extFor("audio/ogg;codecs=opus")).toBe("ogg");
    expect(w.linkIdFromPath(`/call/${LINK}`)).toBe(LINK);
    expect(w.linkIdFromPath(`/call/${LINK}/`)).toBe(LINK);
    expect(w.linkIdFromPath("/call/other")).toBeNull();
    expect(w.linkIdFromPath("/api/voices")).toBeNull();
  });

  it("has a plain-language message for every code the server sends", () => {
    for (const code of ["CALL_LINK_INACTIVE", "CALL_DAY_LIMIT", "CALL_VISITOR_LIMIT", "CALL_RATE_LIMITED", "CALL_SESSION_EXPIRED", "CALL_SESSION_INVALID", "CALL_FORBIDDEN", "NO_SPEECH", "network", "mic", "nodevice", "nomic"]) {
      expect(w.friendlyError(code)).toMatch(/\w+ \w+ \w+/);
      expect(w.friendlyError(code)).not.toMatch(/CALL_|Traceback|undefined/);
    }
    expect(w.friendlyError("SOMETHING_NEW")).toBe(w.MESSAGES.generic);
  });
});

/* ------------------------------------------------------------------------------------------ the page controller */

const template = fs.readFileSync(path.join(__dirname, "../../../../backend/webcall/v1/index.html"), "utf8");
const bodyHtml = (name = "Ria") =>
  template
    .replace(/\{\{NAME\}\}/g, name)
    .replace(/\{\{LINK\}\}/g, LINK)
    .replace(/\{\{MINUTES\}\}/g, "10")
    .replace(/\{\{VERIFY\}\}/g, "bonclay.ayushchougula.in/verify")
    .replace(/^[\s\S]*<body>/, "")
    .replace(/<script[\s\S]*$/, "");

class FakeTrack {
  enabled = true;
  stopped = false;
  stop() {
    this.stopped = true;
  }
}
class FakeStream {
  track = new FakeTrack();
  getTracks() {
    return [this.track];
  }
  getAudioTracks() {
    return [this.track];
  }
}
class FakeAnalyser {
  fftSize = 0;
  connect() {}
  getFloatTimeDomainData(buf: Float32Array) {
    buf.fill(0);
  }
}
class FakeContext {
  state = "running";
  closed = false;
  createAnalyser() {
    return new FakeAnalyser();
  }
  createMediaStreamSource() {
    return { connect() {} };
  }
  close() {
    this.closed = true;
    return Promise.resolve();
  }
}
class FakeAudio {
  src = "";
  onended: (() => void) | null = null;
  onerror: (() => void) | null = null;
  paused = false;
  play() {
    if (this.src.startsWith("data:audio/wav;base64,UklGR") && this.src.length < 100) return Promise.resolve();
    setTimeout(() => this.onended?.(), 0);
    return Promise.resolve();
  }
  pause() {
    this.paused = true;
  }
}
class FakeRecorder {
  static isTypeSupported() {
    return true;
  }
  state = "inactive";
  mimeType = "audio/webm";
  start() {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
  }
}

const jsonRes = (status: number, body: unknown) => ({ ok: status < 400, status, json: () => Promise.resolve(body), text: () => Promise.resolve(JSON.stringify(body)), body: null });
const sseRes = (events: Array<[string, unknown]>) => ({
  ok: true,
  status: 200,
  json: () => Promise.resolve({}),
  body: null,
  text: () => Promise.resolve(events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join("")),
});

function setup(over: { getUserMedia?: jest.Mock; fetch?: jest.Mock; name?: string } = {}) {
  document.body.innerHTML = bodyHtml(over.name);
  const stream = new FakeStream();
  const ctx = new FakeContext();
  const audio = new FakeAudio();
  const getUserMedia = over.getUserMedia ?? jest.fn().mockResolvedValue(stream);
  const fetchMock =
    over.fetch ??
    jest.fn().mockImplementation((url: string) =>
      Promise.resolve(url.endsWith("/session") ? jsonRes(200, { token: "tok", expires_in: 600, max_session_s: 600, agent: { name: "Ria" } }) : jsonRes(200, { ok: true }))
    );
  const page = new w.CallPage({
    document,
    window,
    fetch: fetchMock,
    mediaDevices: { getUserMedia },
    Audio: function () {
      return audio;
    },
    AudioContext: function () {
      return ctx;
    },
    MediaRecorder: FakeRecorder,
  });
  const q = (id: string) => document.getElementById(id) as HTMLElement;
  return { page, stream, ctx, audio, getUserMedia, fetchMock, q };
}

describe("the call page", () => {
  beforeEach(() => {
    jest.useFakeTimers({ doNotFake: ["setTimeout"] });
  });
  afterEach(() => {
    jest.useRealTimers();
    document.body.innerHTML = "";
  });

  it("shows the AI disclosure and a labelled Start button, with the rest hidden", () => {
    const { q } = setup({ name: "Ria" });
    expect(q("agent-name").textContent).toBe("Ria");
    expect(document.body.textContent).toContain("You are talking to an AI, not a person");
    expect(q("start").textContent).toBe("Start talking");
    expect(q("v-live").hidden).toBe(true);
    expect(q("v-end").hidden).toBe(true);
    expect(document.body.textContent).toContain("Voice is AI-generated, watermarked and signed; check a clip at");
    const log = q("captions");
    expect(log.getAttribute("role")).toBe("log");
    expect(log.getAttribute("aria-live")).toBe("polite");
    expect(q("mute").getAttribute("aria-pressed")).toBe("false");
    expect(q("state").getAttribute("aria-live")).toBe("off");
  });

  it("asks for the microphone with echo cancellation, opens the session on its own origin and goes live", async () => {
    const { page, getUserMedia, fetchMock, q } = setup();
    await page.start();
    expect(getUserMedia).toHaveBeenCalledWith({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    expect(fetchMock).toHaveBeenCalledWith(`/call/api/${LINK}/session`, expect.objectContaining({ method: "POST" }));
    expect(q("v-start").hidden).toBe(true);
    expect(q("v-live").hidden).toBe(false);
    expect(q("clock").textContent).toBe("10:00 left");
    expect(q("state").textContent).toMatch(/Listening/);
    expect(q("captions").textContent).toContain("Say hello to begin");
    expect(document.activeElement).toBe(q("live-title"));
    page.end("user");
  });

  it("stays on the start screen with a plain message when the microphone is blocked or missing", async () => {
    const denied = setup({ getUserMedia: jest.fn().mockRejectedValue(Object.assign(new Error("x"), { name: "NotAllowedError" })) });
    await denied.page.start();
    expect(denied.q("start-error").hidden).toBe(false);
    expect(denied.q("start-error").textContent).toMatch(/microphone is blocked/);
    expect(denied.q("start").hasAttribute("disabled")).toBe(false);
    expect(denied.q("v-live").hidden).toBe(true);
    expect(denied.fetchMock).not.toHaveBeenCalled();   // no session is opened without a microphone

    const none = setup({ getUserMedia: jest.fn().mockRejectedValue(Object.assign(new Error("x"), { name: "NotFoundError" })) });
    await none.page.start();
    expect(none.q("start-error").textContent).toMatch(/No microphone was found/);
  });

  it("releases the microphone and says why when the link is no longer active or busy", async () => {
    for (const [status, code, text] of [
      [410, "CALL_LINK_INACTIVE", /no longer active/],
      [429, "CALL_DAY_LIMIT", /limit for today/],
      [429, "CALL_RATE_LIMITED", /Lots of people/],
    ] as const) {
      const fetchMock = jest.fn().mockResolvedValue(jsonRes(status, { code }));
      const { page, stream, q } = setup({ fetch: fetchMock });
      await page.start();
      expect(q("start-error").textContent).toMatch(text);
      expect(stream.track.stopped).toBe(true);
      expect(q("v-live").hidden).toBe(true);
    }
    const down = setup({ fetch: jest.fn().mockRejectedValue(new Error("offline")) });
    await down.page.start();
    expect(down.q("start-error").textContent).toMatch(/Could not reach the agent/);
  });

  it("sends a turn with the bearer token and shows the caller's and the agent's completed lines", async () => {
    const turnRes = sseRes([
      ["transcript", { user_text: "When do you open?", language: "en" }],
      ["sentence", { index: 0, text: "We open at nine.", audio_url: "data:audio/wav;base64,AAAA" }],
      ["sentence", { index: 1, text: "Anything else?", audio_url: "data:audio/wav;base64,AAAA" }],
      ["done", { turn_id: "t" }],
    ]);
    const fetchMock = jest.fn().mockImplementation((url: string) =>
      Promise.resolve(url.endsWith("/session") ? jsonRes(200, { token: "tok", max_session_s: 600, agent: { name: "Ria" } }) : turnRes)
    );
    const { page, q } = setup({ fetch: fetchMock });
    await page.start();
    const sent = page.sendTurn(new Blob(["x"], { type: "audio/webm" }), "webm");
    expect(q("state").getAttribute("data-s")).toBe("thinking");
    await sent;
    const turnCall = fetchMock.mock.calls.find((c) => String(c[0]).endsWith("/turn"));
    expect(turnCall[0]).toBe(`/call/api/${LINK}/turn`);
    expect(turnCall[1].headers).toEqual({ Authorization: "Bearer tok" });
    expect((turnCall[1].body as FormData).get("audio")).toBeInstanceOf(Blob);
    const lines = Array.from(q("captions").querySelectorAll("p.line")).map((p) => p.textContent);
    expect(lines).toEqual(["You: When do you open?", "Agent: We open at nine.", "Agent: Anything else?"]);
    expect(q("captions").querySelector(".hint")).toBeNull();
    expect(q("state").getAttribute("data-s")).toBe("listening");
    page.end("user");
  });

  it("only fetches clips that belong to its own link", async () => {
    const fetchMock = jest.fn().mockResolvedValue(jsonRes(200, {}));
    const { page } = setup({ fetch: fetchMock });
    page.session = { token: "tok" };
    expect(await page.fetchClip({ url: "https://evil.example/x.wav" })).toBeNull();
    expect(await page.fetchClip({ url: "/api/widget/clip/a/0" })).toBeNull();
    expect(await page.fetchClip({ url: `/call/api/awc_other/clip/a/0` })).toBeNull();
    expect(await page.fetchClip({ url: "" })).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(await page.fetchClip({ url: "data:audio/wav;base64,AAAA" })).toBe("data:audio/wav;base64,AAAA");
  });

  it("ends the call when the server says the time or the day's minutes are used up or the link is revoked", async () => {
    for (const [code, text] of [
      ["CALL_SESSION_EXPIRED", /time limit/],
      ["CALL_DAY_LIMIT", /limit for today/],
      ["CALL_LINK_INACTIVE", /no longer active/],
    ] as const) {
      const fetchMock = jest.fn().mockImplementation((url: string) =>
        Promise.resolve(url.endsWith("/session") ? jsonRes(200, { token: "tok", max_session_s: 600, agent: {} }) : url.endsWith("/turn") ? jsonRes(429, { code }) : jsonRes(200, {}))
      );
      const { page, stream, q } = setup({ fetch: fetchMock });
      await page.start();
      await page.sendTurn(new Blob(["x"]), "webm");
      expect(q("v-end").hidden).toBe(false);
      expect(q("v-live").hidden).toBe(true);
      expect(q("end-text").textContent).toMatch(text);
      expect(stream.track.stopped).toBe(true);
    }
  });

  it("shows a gentle error for a turn that failed and keeps the call going", async () => {
    const fetchMock = jest.fn().mockImplementation((url: string) =>
      Promise.resolve(url.endsWith("/session") ? jsonRes(200, { token: "tok", max_session_s: 600, agent: {} }) : sseRes([["error", { code: "NO_SPEECH" }]]))
    );
    const { page, q } = setup({ fetch: fetchMock });
    await page.start();
    await page.sendTurn(new Blob(["x"]), "webm");
    expect(q("live-error").hidden).toBe(false);
    expect(q("live-error").textContent).toMatch(/did not catch that/);
    expect(q("v-live").hidden).toBe(false);
    expect(q("state").getAttribute("data-s")).toBe("listening");
    page.end("user");
  });

  it("mutes and unmutes the microphone and says so", async () => {
    const { page, stream, q } = setup();
    await page.start();
    q("mute").click();
    expect(stream.track.enabled).toBe(false);
    expect(q("mute").getAttribute("aria-pressed")).toBe("true");
    expect(q("mute").textContent).toBe("Unmute microphone");
    expect(q("state").getAttribute("data-s")).toBe("muted");
    q("mute").click();
    expect(stream.track.enabled).toBe(true);
    expect(q("mute").getAttribute("aria-pressed")).toBe("false");
    expect(q("state").getAttribute("data-s")).toBe("listening");
    page.end("user");
  });

  it("End hangs up: releases the microphone, tells the server once, and offers a new call", async () => {
    const { page, stream, ctx, fetchMock, q } = setup();
    await page.start();
    q("end").click();
    expect(q("v-end").hidden).toBe(false);
    expect(q("end-text").textContent).toMatch(/Thanks for talking/);
    expect(stream.track.stopped).toBe(true);
    expect(ctx.closed).toBe(true);
    const ends = fetchMock.mock.calls.filter((c) => String(c[0]).endsWith("/end"));
    expect(ends).toHaveLength(1);
    expect(ends[0][1]).toMatchObject({ method: "POST", keepalive: true, headers: { Authorization: "Bearer tok" } });
    page.end("user");   // a second End (or the page closing) does nothing more
    page.hangUpQuietly();
    expect(fetchMock.mock.calls.filter((c) => String(c[0]).endsWith("/end"))).toHaveLength(1);
    q("again").click();
    expect(q("v-start").hidden).toBe(false);
    expect(q("v-end").hidden).toBe(true);
  });

  it("counts down, warns once with one minute left and ends at zero", async () => {
    const fetchMock = jest.fn().mockImplementation((url: string) =>
      Promise.resolve(url.endsWith("/session") ? jsonRes(200, { token: "tok", max_session_s: 90, agent: {} }) : jsonRes(200, {}))
    );
    const { page, q } = setup({ fetch: fetchMock });
    await page.start();
    expect(q("clock").textContent).toBe("1:30 left");
    jest.advanceTimersByTime(31_000);
    expect(q("captions").textContent).toContain("One minute left.");
    jest.advanceTimersByTime(5_000);
    expect(q("captions").querySelectorAll(".note")).toHaveLength(1);
    jest.advanceTimersByTime(60_000);
    expect(q("v-end").hidden).toBe(false);
    expect(q("end-text").textContent).toMatch(/time limit/);
    await tick();
  });

  it("writes names and captions as text, never as markup", async () => {
    const fetchMock = jest.fn().mockImplementation((url: string) =>
      Promise.resolve(
        url.endsWith("/session")
          ? jsonRes(200, { token: "tok", max_session_s: 600, agent: {} })
          : sseRes([["transcript", { user_text: "<img src=x onerror=alert(1)>" }], ["sentence", { index: 0, text: "<b>hi</b>", audio_url: "" }]])
      )
    );
    const { page, q } = setup({ fetch: fetchMock });
    await page.start();
    await page.sendTurn(new Blob(["x"]), "webm");
    expect(q("captions").querySelector("img")).toBeNull();
    expect(q("captions").querySelector("b")).toBeNull();
    expect(q("captions").textContent).toContain("<b>hi</b>");
    page.end("user");
  });
});
