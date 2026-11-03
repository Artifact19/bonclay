import "@testing-library/jest-dom";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import PlaygroundPage from "@/app/(app)/dashboard/playground/page";
import { api } from "@/lib/api";
import { toast } from "sonner";

jest.mock("next/link", () => ({ __esModule: true, default: ({ children, ...p }: { children: React.ReactNode }) => <a {...p}>{children}</a> }));
jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => {
  class ApiError extends Error {
    status = 0;
    code = "unknown_error";
  }
  return { api: { sse: jest.fn(), upload: jest.fn() }, ApiError, API_URL: "http://api.test" };
});
jest.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams() }));
jest.mock("@/hooks/useApi", () => ({
  useAgentTemplates: () => ({ data: [] }),
  useAgents: () => ({ data: [{ id: "a1", name: "Ria" }], isLoading: false, error: undefined }),
}));
jest.mock("@/components/audio/WaveformPlayer", () => ({ WaveformPlayer: () => null }));
jest.mock("@/components/dashboard/PerformancePanel", () => ({ PerformancePanel: () => null, PERF_SUBTITLE: "" }));
jest.mock("@/components/voices/Modal", () => ({ Modal: () => null }));
// Silero VAD (ONNX + wasm) cannot run in jsdom: the module is replaced. By default it fails to load, which is the fallback
// path (the built-in loudness detector) the older tests below drive; the "with Silero" tests plug in a controllable one.
const mockVadNew = jest.fn();
jest.mock("@ricky0123/vad-web", () => ({ MicVAD: { new: (opts: unknown) => mockVadNew(opts) } }));

// A microphone whose loudness the test controls: the analyser fills the buffer with a constant level.
let level = 0;
const trackStop = jest.fn();
const getUserMedia = jest.fn();
const recorders: FakeRecorder[] = [];
let ctxClosed = 0;

class FakeRecorder {
  state = "inactive";
  mimeType = "audio/webm";
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  constructor(public stream: unknown) {
    recorders.push(this);
  }
  static isTypeSupported = () => true;
  start() {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["x"]) });
    this.onstop?.();
  }
}

class FakeAudioContext {
  state = "running";
  createAnalyser() {
    return { fftSize: 0, getFloatTimeDomainData: (b: Float32Array) => b.fill(level) };
  }
  createMediaStreamSource() {
    return { connect: () => undefined };
  }
  resume() {
    return Promise.resolve();
  }
  close() {
    ctxClosed += 1;
    this.state = "closed";
    return Promise.resolve();
  }
}

class FakeAudio {
  static all: FakeAudio[] = [];
  preload = "";
  paused = false;
  constructor(public src: string) {
    FakeAudio.all.push(this);
  }
  play() {
    return Promise.resolve();
  }
  pause() {
    this.paused = true;
  }
  addEventListener() {
    /* nothing */
  }
}

beforeAll(() => {
  Object.assign(global, { Audio: FakeAudio, MediaRecorder: FakeRecorder });
  Object.defineProperty(window, "AudioContext", { value: FakeAudioContext, configurable: true });
  Object.defineProperty(navigator, "mediaDevices", { value: { getUserMedia }, configurable: true });
  Object.defineProperty(window, "matchMedia", { value: () => ({ matches: false, addEventListener: () => undefined, removeEventListener: () => undefined }), configurable: true });
  Element.prototype.scrollIntoView = jest.fn();
  let n = 0;
  Object.defineProperty(global.crypto, "randomUUID", { value: () => `id-${++n}`, configurable: true });
});

beforeEach(() => {
  jest.clearAllMocks();
  mockVadNew.mockRejectedValue(new Error("model not available in tests"));
  jest.useFakeTimers();
  level = 0;
  ctxClosed = 0;
  recorders.length = 0;
  FakeAudio.all = [];
  getUserMedia.mockResolvedValue({ getTracks: () => [{ stop: trackStop }] });
  (api.sse as jest.Mock).mockImplementation(() => new Promise<void>(() => undefined));
  try {
    localStorage.clear();
  } catch {
    /* ignore */
  }
});
afterEach(() => jest.useRealTimers());

const advance = (ms: number) => act(async () => void jest.advanceTimersByTime(ms));
const toggle = () => screen.getByRole("button", { name: /hands-free/i });

describe("Playground hands-free", () => {
  it("renders an off-by-default toggle that is a real button with aria-pressed", () => {
    render(<PlaygroundPage />);
    expect(toggle().tagName).toBe("BUTTON");
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: /hold to talk/i })).toBeEnabled();
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("switches on: asks for a cleaned-up mic, announces the state in a polite live region, and remembers the choice", async () => {
    render(<PlaygroundPage />);
    await act(async () => fireEvent.click(toggle()));
    expect(toggle()).toHaveAttribute("aria-pressed", "true");
    expect(getUserMedia).toHaveBeenCalledWith({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const live = await screen.findByText("Listening…");
    expect(live).toHaveAttribute("aria-live", "polite");
    expect(screen.getByRole("button", { name: /hold to talk/i })).toBeDisabled(); // one microphone user at a time
    expect(localStorage.getItem("bonclay.handsFree")).toBe("1");
  });

  it("starts on when it was left on, and survives a browser with blocked storage", async () => {
    localStorage.setItem("bonclay.handsFree", "1");
    const { unmount } = render(<PlaygroundPage />);
    await waitFor(() => expect(toggle()).toHaveAttribute("aria-pressed", "true"));
    unmount();
    const spy = jest.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    render(<PlaygroundPage />);
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    spy.mockRestore();
  });

  it("switching off releases the microphone and the audio context", async () => {
    render(<PlaygroundPage />);
    await act(async () => fireEvent.click(toggle()));
    await screen.findByText("Listening…");
    await act(async () => fireEvent.click(toggle()));
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    expect(trackStop).toHaveBeenCalled();
    expect(ctxClosed).toBe(1);
    expect(screen.getByText("Hold to talk", { selector: "p" })).toBeInTheDocument();
    expect(localStorage.getItem("bonclay.handsFree")).toBe("0");
  });

  it("leaving the page releases the microphone", async () => {
    const { unmount } = render(<PlaygroundPage />);
    await act(async () => fireEvent.click(toggle()));
    await screen.findByText("Listening…");
    unmount();
    expect(trackStop).toHaveBeenCalled();
  });

  it("sends a spoken utterance as a turn by itself, through the same streaming request", async () => {
    render(<PlaygroundPage />);
    await act(async () => fireEvent.click(toggle()));
    await screen.findByText("Listening…");
    await advance(700); // calibration on silence
    level = 0.15;
    await advance(700);
    expect(screen.getByText("Hearing you…")).toBeInTheDocument();
    expect(recorders).toHaveLength(1);
    level = 0;
    await advance(1000);
    expect(api.sse).toHaveBeenCalledTimes(1);
    const [path, body] = (api.sse as jest.Mock).mock.calls[0];
    expect(path).toBe("/api/conversations/a1/turn/stream");
    expect((body as FormData).get("audio")).toBeInstanceOf(Blob);
    expect(screen.getByText("Thinking…")).toBeInTheDocument();
  });

  it("does not send anything for noise or a short blip", async () => {
    render(<PlaygroundPage />);
    await act(async () => fireEvent.click(toggle()));
    await screen.findByText("Listening…");
    await advance(700);
    level = 0.15;
    await advance(80);
    level = 0;
    await advance(3000);
    expect(api.sse).not.toHaveBeenCalled();
  });

  it("tells the user when the microphone is blocked and turns itself off", async () => {
    getUserMedia.mockRejectedValue(new Error("denied"));
    render(<PlaygroundPage />);
    await act(async () => fireEvent.click(toggle()));
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(expect.stringMatching(/microphone access was blocked/i)));
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
  });

  describe("while the agent speaks", () => {
    /** Starts hands-free, says one thing and lets the agent begin a reply that keeps playing. */
    async function agentSpeaking() {
      let emit: (e: string, d: unknown) => void = () => undefined;
      (api.sse as jest.Mock).mockImplementation((_p: string, _b: FormData, opts: { onEvent: typeof emit; signal: AbortSignal }) => {
        emit = opts.onEvent;
        return new Promise<void>((_resolve, reject) => opts.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { code: "aborted" }))));
      });
      render(<PlaygroundPage />);
      await act(async () => fireEvent.click(toggle()));
      await screen.findByText("Listening…");
      await advance(700);
      level = 0.15;
      await advance(700);
      level = 0;
      await advance(1200);
      expect(api.sse).toHaveBeenCalledTimes(1);
      act(() => emit("transcript", { user_text: "What are your timings?" }));
      act(() => emit("sentence", { index: 0, text: "We open at nine.", audio_url: "/a/0" }));
      expect(screen.getByText("Speaking… (talk to interrupt)")).toBeInTheDocument();
      expect(FakeAudio.all).toHaveLength(1);
    }

    it("does not send the agent's own audio as a new turn", async () => {
      await agentSpeaking();
      level = 0.03; // the agent coming back through the speakers: above the normal threshold, but it is playback
      await advance(5000);
      expect(api.sse).toHaveBeenCalledTimes(1);
      expect(FakeAudio.all[0].paused).toBe(false);
      expect(screen.getByText("Speaking… (talk to interrupt)")).toBeInTheDocument();
    });

    it("stops the agent when the user clearly talks over it, and sends what they said", async () => {
      await agentSpeaking();
      level = 0.03;
      await advance(1500);
      level = 0.3; // well above the playback level
      await advance(700);
      expect(FakeAudio.all[0].paused).toBe(true);
      expect(screen.getByText("Hearing you…")).toBeInTheDocument();
      level = 0;
      await advance(1200);
      expect(api.sse).toHaveBeenCalledTimes(2);
      expect(screen.queryByText("Speaking… (talk to interrupt)")).toBeNull();
    });

    it("switching hands-free off stops the reply that was playing and the request behind it", async () => {
      await agentSpeaking();
      await act(async () => fireEvent.click(toggle()));
      expect(FakeAudio.all[0].paused).toBe(true);
      expect(((api.sse as jest.Mock).mock.calls[0][2] as { signal: AbortSignal }).signal.aborted).toBe(true);
      expect(trackStop).toHaveBeenCalled();
    });
  });
});

interface SileroOptions {
  baseAssetPath: string;
  onnxWASMBasePath: string;
  model: string;
  redemptionMs: number;
  minSpeechMs: number;
  positiveSpeechThreshold: number;
  getStream: () => Promise<MediaStream>;
  onFrameProcessed: (p: { isSpeech: number; notSpeech: number }, frame: Float32Array) => void;
  onSpeechStart: () => void;
  onSpeechEnd: (audio: Float32Array) => void;
  onVADMisfire: () => void;
}

describe("Playground hands-free with Silero VAD", () => {
  let silero: SileroOptions;
  const destroy = jest.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    destroy.mockClear();
    mockVadNew.mockImplementation(async (o: SileroOptions) => {
      silero = o;
      return { destroy };
    });
  });

  /** One Silero frame (32 ms) with the given speech probability, delivered the way the library does. */
  const frame = (prob: number, samples = 0.1) => silero.onFrameProcessed({ isSpeech: prob, notSpeech: 1 - prob }, new Float32Array(512).fill(samples));
  /** Feeds frames for `ms` of (fake) time. */
  const frames = async (prob: number, ms: number) => {
    for (let t = 0; t < ms; t += 32) {
      await advance(32);
      act(() => frame(prob));
    }
  };
  const start = async () => {
    render(<PlaygroundPage />);
    await act(async () => fireEvent.click(toggle()));
    await screen.findByText("Listening…");
  };

  it("loads only when hands-free is switched on, from the self-hosted files, with the cleaned-up microphone", async () => {
    render(<PlaygroundPage />);
    expect(mockVadNew).not.toHaveBeenCalled();
    await act(async () => fireEvent.click(toggle()));
    await screen.findByText("Listening…");
    expect(mockVadNew).toHaveBeenCalledTimes(1);
    expect(silero.baseAssetPath).toBe("/vad/");
    expect(silero.onnxWASMBasePath).toBe("/vad/");
    expect(silero.model).toBe("v5");
    expect(silero.redemptionMs).toBeGreaterThanOrEqual(600);
    expect(silero.redemptionMs).toBeLessThanOrEqual(800);
    expect(silero.minSpeechMs).toBe(400);
    expect(getUserMedia).toHaveBeenCalledTimes(1);
    expect(getUserMedia).toHaveBeenCalledWith({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    await expect(silero.getStream()).resolves.toBe(await getUserMedia.mock.results[0].value); // Silero listens to that same stream
  });

  it("sends an utterance as a turn when Silero reports the end of speech", async () => {
    await start();
    await frames(0.9, 96);
    act(() => silero.onSpeechStart());
    expect(screen.getByText("Hearing you…")).toBeInTheDocument();
    expect(recorders).toHaveLength(1);
    await frames(0.9, 640);
    await frames(0.05, 700);
    expect(api.sse).not.toHaveBeenCalled(); // Silero has not said it is over yet
    act(() => silero.onSpeechEnd(new Float32Array(16000)));
    expect(api.sse).toHaveBeenCalledTimes(1);
    const [path, body] = (api.sse as jest.Mock).mock.calls[0];
    expect(path).toBe("/api/conversations/a1/turn/stream");
    expect((body as FormData).get("audio")).toBeInstanceOf(Blob);
    expect(screen.getByText("Thinking…")).toBeInTheDocument();
  });

  it("sends nothing for a misfire (speech too short to count)", async () => {
    await start();
    await frames(0.9, 64);
    act(() => silero.onSpeechStart());
    act(() => silero.onVADMisfire());
    expect(screen.getByText("Listening…")).toBeInTheDocument();
    await frames(0.05, 2000);
    expect(api.sse).not.toHaveBeenCalled();
  });

  it("throws away a recording that was started early for a sound Silero never confirmed", async () => {
    await start();
    await frames(0.4, 64); // likely enough to start recording, never enough for Silero to call it speech
    expect(recorders).toHaveLength(1);
    await frames(0.05, 1500);
    expect(recorders[0].state).toBe("inactive");
    expect(api.sse).not.toHaveBeenCalled();
  });

  it("switching off tears Silero down and releases the microphone", async () => {
    await start();
    await act(async () => fireEvent.click(toggle()));
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(trackStop).toHaveBeenCalled();
    expect(ctxClosed).toBe(1);
  });

  describe("falls back to the built-in loudness detector", () => {
    it("when the model cannot be loaded: logs once, shows nothing scary, still works", async () => {
      const warn = jest.spyOn(console, "warn").mockImplementation(() => undefined);
      mockVadNew.mockRejectedValue(new Error("wasm 404"));
      await start();
      expect(toast.error).not.toHaveBeenCalled();
      await advance(700);
      level = 0.15;
      await advance(700);
      expect(screen.getByText("Hearing you…")).toBeInTheDocument();
      level = 0;
      await advance(1000);
      expect(api.sse).toHaveBeenCalledTimes(1);
      expect(getUserMedia).toHaveBeenCalledTimes(1); // the same microphone, not a second prompt
      await act(async () => fireEvent.click(toggle())); // off and on again: Silero fails again, quietly
      await act(async () => fireEvent.click(toggle()));
      await screen.findByText("Listening…");
      expect(toast.error).not.toHaveBeenCalled();
      expect(warn.mock.calls.length).toBeLessThanOrEqual(1); // logged at most once per page load
      warn.mockRestore();
    });

    it("when loading never finishes: gives up after the timeout and drops the late model", async () => {
      jest.spyOn(console, "warn").mockImplementation(() => undefined);
      let resolveLate: (v: unknown) => void = () => undefined;
      mockVadNew.mockImplementation(() => new Promise((resolve) => (resolveLate = resolve)));
      render(<PlaygroundPage />);
      await act(async () => fireEvent.click(toggle()));
      await advance(100);
      expect(screen.getByText("Starting the microphone…")).toBeInTheDocument();
      await advance(20000);
      await screen.findByText("Listening…");
      await advance(700); // calibration on silence
      level = 0.15;
      await advance(700);
      expect(screen.getByText("Hearing you…")).toBeInTheDocument();
      resolveLate({ destroy });
      await advance(10);
      expect(destroy).toHaveBeenCalledTimes(1);
    });
  });

  describe("while the agent speaks", () => {
    /** Silero hears one utterance; the agent starts a reply that keeps playing. */
    async function agentSpeaking() {
      let emit: (e: string, d: unknown) => void = () => undefined;
      (api.sse as jest.Mock).mockImplementation((_p: string, _b: FormData, opts: { onEvent: typeof emit; signal: AbortSignal }) => {
        emit = opts.onEvent;
        return new Promise<void>((_resolve, reject) => opts.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { code: "aborted" }))));
      });
      await start();
      await frames(0.9, 96);
      act(() => silero.onSpeechStart());
      await frames(0.9, 600);
      act(() => silero.onSpeechEnd(new Float32Array(16000)));
      expect(api.sse).toHaveBeenCalledTimes(1);
      act(() => emit("transcript", { user_text: "What are your timings?" }));
      act(() => emit("sentence", { index: 0, text: "We open at nine.", audio_url: "/a/0" }));
      expect(screen.getByText("Speaking… (talk to interrupt)")).toBeInTheDocument();
      expect(FakeAudio.all).toHaveLength(1);
    }

    it("does not treat the agent's own voice (Silero hears it as speech) as a new turn or an interruption", async () => {
      await agentSpeaking();
      await frames(0.1, 600); // the echo canceller settles
      act(() => silero.onSpeechStart());
      await frames(0.7, 5000); // speech-like, but not clearly the user
      act(() => silero.onSpeechEnd(new Float32Array(16000)));
      expect(api.sse).toHaveBeenCalledTimes(1);
      expect(FakeAudio.all[0].paused).toBe(false);
      expect(screen.getByText("Speaking… (talk to interrupt)")).toBeInTheDocument();
    });

    it("barge-in needs sustained speech: a short burst does nothing, ~300 ms stops the agent right away", async () => {
      await agentSpeaking();
      await frames(0.1, 1000);
      await frames(0.95, 192); // 200 ms of a cough or a click
      expect(FakeAudio.all[0].paused).toBe(false);
      await frames(0.1, 400);
      expect(FakeAudio.all[0].paused).toBe(false);

      act(() => silero.onSpeechStart());
      await frames(0.95, 256);
      expect(FakeAudio.all[0].paused).toBe(false);
      await frames(0.95, 128); // now more than 300 ms in a row
      expect(FakeAudio.all[0].paused).toBe(true);
      expect(screen.getByText("Hearing you…")).toBeInTheDocument();
      expect(screen.queryByText("Speaking… (talk to interrupt)")).toBeNull();
    });

    it("sends what the user said after cutting in", async () => {
      await agentSpeaking();
      await frames(0.1, 1000);
      act(() => silero.onSpeechStart());
      await frames(0.95, 640);
      expect(FakeAudio.all[0].paused).toBe(true);
      await frames(0.95, 640);
      act(() => silero.onSpeechEnd(new Float32Array(16000)));
      expect(api.sse).toHaveBeenCalledTimes(2);
    });
  });
});
