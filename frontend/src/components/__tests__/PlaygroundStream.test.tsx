import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import PlaygroundPage from "@/app/(app)/dashboard/playground/page";
import { api } from "@/lib/api";
import { toast } from "sonner";

jest.mock("next/link", () => ({ __esModule: true, default: ({ children, ...p }: { children: React.ReactNode }) => <a {...p}>{children}</a> }));
jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => {
  class ApiError extends Error {
    status: number;
    code: string;
    constructor(message: string, status = 0, code = "unknown_error") {
      super(message);
      this.status = status;
      this.code = code;
    }
  }
  return { api: { sse: jest.fn(), upload: jest.fn() }, ApiError, API_URL: "http://api.test" };
});
jest.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams() }));
jest.mock("@/hooks/useApi", () => ({
  useAgentTemplates: () => ({ data: [] }),
  useAgents: () => ({ data: [{ id: "a1", name: "Ria" }], isLoading: false, error: undefined }),
}));
jest.mock("@/components/audio/WaveformPlayer", () => ({
  WaveformPlayer: ({ src, autoPlay }: { src: string; autoPlay?: boolean }) => (
    <div data-testid="player" data-src={src} data-autoplay={String(!!autoPlay)} />
  ),
}));
jest.mock("@/components/dashboard/PerformancePanel", () => ({ PerformancePanel: () => null, PERF_SUBTITLE: "" }));
jest.mock("@/components/voices/Modal", () => ({ Modal: () => null }));

const sse = api.sse as jest.Mock;
const upload = api.upload as jest.Mock;

class FakeAudio {
  static all: FakeAudio[] = [];
  preload = "";
  played = false;
  paused = false;
  private handlers: Record<string, (() => void)[]> = {};
  constructor(public src: string) {
    FakeAudio.all.push(this);
  }
  play() {
    this.played = true;
    return Promise.resolve();
  }
  pause() {
    this.paused = true;
  }
  addEventListener(type: string, fn: () => void) {
    (this.handlers[type] ||= []).push(fn);
  }
  fire(type: string) {
    (this.handlers[type] || []).forEach((f) => f());
  }
}

class FakeRecorder {
  state = "inactive";
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  stream = { getTracks: () => [] };
  start() {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["x"]) });
    this.onstop?.();
  }
}

beforeAll(() => {
  Object.assign(global, { Audio: FakeAudio, MediaRecorder: FakeRecorder });
  Object.defineProperty(navigator, "mediaDevices", { value: { getUserMedia: async () => ({ getTracks: () => [] }) }, configurable: true });
  Object.defineProperty(window, "matchMedia", { value: () => ({ matches: false, addEventListener: () => undefined, removeEventListener: () => undefined }), configurable: true });
  Element.prototype.scrollIntoView = jest.fn();
  let n = 0;
  Object.defineProperty(global.crypto, "randomUUID", { value: () => `id-${++n}`, configurable: true });
});

beforeEach(() => {
  jest.clearAllMocks();
  FakeAudio.all = [];
});

async function talk() {
  render(<PlaygroundPage />);
  const mic = screen.getByRole("button", { name: /hold to talk/i });
  await act(async () => {
    fireEvent.mouseDown(mic);
  });
  await act(async () => {
    fireEvent.mouseUp(mic);
  });
}

const TIMINGS = { asr_ms: 600, retrieval_ms: 20, llm_first_token_ms: 100, llm_ms: 500, tts_ms: 2800, first_audio_ms: 1800, total_ms: 3600 };

describe("Playground streaming turn", () => {
  it("shows the transcript first, plays sentence clips back to back and ends with the badges", async () => {
    let emit: (e: string, d: unknown) => void = () => undefined;
    let finish: () => void = () => undefined;
    sse.mockImplementation((_path: string, _body: FormData, opts: { onEvent: typeof emit }) => {
      emit = opts.onEvent;
      return new Promise<void>((resolve) => (finish = resolve));
    });
    await talk();
    expect(sse).toHaveBeenCalledWith("/api/conversations/a1/turn/stream", expect.any(FormData), expect.anything());
    expect(screen.getByText("Listening and thinking")).toBeTruthy();

    act(() => emit("transcript", { user_text: "When do you open?", language: "en" }));
    expect(screen.getByText("When do you open?")).toBeTruthy();
    expect(screen.getByText("Thinking")).toBeTruthy();
    expect(FakeAudio.all).toHaveLength(0);

    act(() => emit("sentence", { index: 0, text: "This is an AI assistant.", audio_url: "/api/conversations/turns/t/sentences/0/audio" }));
    expect(screen.getByText("This is an AI assistant.")).toBeTruthy();
    expect(screen.getByText("Speaking")).toBeTruthy();
    expect(screen.getByTestId("ai-generated-tag")).toBeTruthy(); // agent speech is tagged while it streams
    expect(FakeAudio.all.map((a) => a.src)).toEqual(["http://api.test/api/conversations/turns/t/sentences/0/audio"]);
    expect(FakeAudio.all[0].played).toBe(true);
    expect(screen.queryByText("Thinking")).toBeNull();

    act(() => emit("sentence", { index: 1, text: "We open at nine.", audio_url: "/api/conversations/turns/t/sentences/1/audio" }));
    expect(screen.getByText("This is an AI assistant. We open at nine.")).toBeTruthy(); // the reply grows
    expect(FakeAudio.all[1].played).toBe(false); // queued behind the first clip
    act(() => FakeAudio.all[0].fire("ended"));
    expect(FakeAudio.all[1].played).toBe(true);
    expect(screen.queryByTestId("player")).toBeNull(); // the full-reply player waits for `done`

    act(() =>
      emit("done", {
        turn_id: "t", agent_text: "This is an AI assistant. We open at nine.", agent_audio_url: "/api/conversations/turns/t/audio",
        timings: TIMINGS, total_ms: 3600, engine_name: "Kokoro", llm_provider: "mlx", llm_model: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        conversation_id: "c1", grounded: true, sources_used: 2,
      })
    );
    await act(async () => finish());
    expect(screen.getByText("Replied in 3.6 s")).toBeTruthy(); // plain words first
    expect(screen.getByText("Details")).toBeTruthy(); // the technical numbers sit behind it
    expect(screen.getByText(/First sentence ready after 1\.8 s/)).toBeTruthy();
    expect(screen.getByText(/Whole reply 3\.6 s/)).toBeTruthy();
    expect(screen.getByText(/Answered from your documents \(2 sections\)/)).toBeTruthy();
    expect(screen.getByText("Kokoro")).toBeTruthy();
    const player = screen.getByTestId("player");
    expect(player.getAttribute("data-src")).toBe("http://api.test/api/conversations/turns/t/audio");
    expect(player.getAttribute("data-autoplay")).toBe("false"); // already heard sentence by sentence
    expect(screen.queryByText("Speaking")).toBeNull();
    expect(upload).not.toHaveBeenCalled();
  });

  it("shows a chip under the reply for each tool the agent used", async () => {
    sse.mockImplementation(async (_p: string, _b: FormData, opts: { onEvent: (e: string, d: unknown) => void }) => {
      opts.onEvent("transcript", { user_text: "Book me with the doctor at 10:30", language: "en" });
      opts.onEvent("sentence", { index: 0, text: "Your booking A12 is on Monday at 10:30.", audio_url: "/a/0" });
      opts.onEvent("done", {
        turn_id: "t", agent_text: "Your booking A12 is on Monday at 10:30.", timings: { ...TIMINGS, tools_ms: 40 },
        total_ms: 3000, engine_name: "Kokoro", conversation_id: "c1",
        tool_calls: [{ name: "book_appointment", arguments: { name: "Priya" }, ok: true, result: "booking A12 on Mon 10:30" }],
      });
    });
    await talk();
    await waitFor(() => expect(screen.getByTestId("tool-call-chip")).toBeTruthy());
    expect(screen.getByTestId("tool-call-chip").textContent).toBe("Used tool: book_appointment → booking A12 on Mon 10:30");
    expect(screen.getByText(/Tools 40 ms/)).toBeTruthy();
  });

  it("plays a filler clip first but keeps it out of the reply text", async () => {
    sse.mockImplementation(async (_p: string, _b: FormData, opts: { onEvent: (e: string, d: unknown) => void }) => {
      opts.onEvent("transcript", { user_text: "When do you open?", language: "en" });
      opts.onEvent("sentence", { index: 0, text: "Let me check.", audio_url: "/a/0", filler: true });
      opts.onEvent("sentence", { index: 1, text: "We open at nine.", audio_url: "/a/1" });
      opts.onEvent("done", { turn_id: "t", agent_text: "We open at nine.", timings: TIMINGS, total_ms: 3000, engine_name: "Kokoro", conversation_id: "c1" });
    });
    await talk();
    await waitFor(() => expect(screen.getByText("We open at nine.")).toBeTruthy());
    expect(FakeAudio.all.map((a) => a.src)).toEqual(["http://api.test/a/0", "http://api.test/a/1"]); // both are played, in order
    expect(FakeAudio.all[0].played).toBe(true);
    expect(screen.queryByText(/Let me check/)).toBeNull();
  });

  it("falls back to the original turn endpoint when the stream endpoint does not exist", async () => {
    const { ApiError } = jest.requireMock("@/lib/api");
    sse.mockRejectedValue(new ApiError("Not Found", 404, "http_404"));
    upload.mockResolvedValue({
      user_text: "Hello", agent_text: "Hi there.", latency_ms: 4200, agent_audio_url: "/api/conversations/turns/x/audio",
      engine_name: "Kokoro", conversation_id: "c9",
    });
    await talk();
    await waitFor(() => expect(screen.getByText("Hi there.")).toBeTruthy());
    expect(upload).toHaveBeenCalledWith("/api/conversations/a1/turn", expect.any(FormData));
    expect(screen.getByText("Hello")).toBeTruthy();
    expect(screen.getByTestId("player").getAttribute("data-autoplay")).toBe("true"); // not streamed: autoplay as before
    expect(toast.error).not.toHaveBeenCalled();
  });

  it("does not fall back for a real error such as an unknown agent", async () => {
    const { ApiError } = jest.requireMock("@/lib/api");
    sse.mockRejectedValue(new ApiError("Agent not found", 404, "AGENT_NOT_FOUND"));
    await talk();
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("Agent not found"));
    expect(upload).not.toHaveBeenCalled();
  });

  it("reports an error event and keeps what was already said", async () => {
    sse.mockImplementation(async (_p: string, _b: FormData, opts: { onEvent: (e: string, d: unknown) => void }) => {
      opts.onEvent("transcript", { user_text: "Hi", language: "en" });
      opts.onEvent("sentence", { index: 0, text: "One moment.", audio_url: "/a/0" });
      opts.onEvent("error", { code: "TURN_FAILED", message: "Failed to process conversation turn" });
    });
    await talk();
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("Failed to process conversation turn"));
    expect(screen.getByText("One moment.")).toBeTruthy();
    expect(screen.queryByText("Speaking")).toBeNull();
  });

  it("tells the user when the stream ends without a result", async () => {
    sse.mockImplementation(async (_p: string, _b: FormData, opts: { onEvent: (e: string, d: unknown) => void }) => {
      opts.onEvent("transcript", { user_text: "Hi", language: "en" });
    });
    await talk();
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("The reply was cut off. Try again."));
  });

  it("marks the question and the reply with the language the server heard, so screen readers switch voice", async () => {
    sse.mockImplementation(async (_p: string, _b: FormData, opts: { onEvent: (e: string, d: unknown) => void }) => {
      opts.onEvent("transcript", { user_text: "आप कब खुलते हैं?", language: "hi" });
      opts.onEvent("sentence", { index: 0, text: "हम नौ बजे खुलते हैं।", audio_url: "/a/0" });
      opts.onEvent("done", { turn_id: "t", agent_text: "हम नौ बजे खुलते हैं।", total_ms: 1000, conversation_id: "c1" });
    });
    await talk();
    await waitFor(() => expect(screen.getByText("हम नौ बजे खुलते हैं।")).toBeTruthy());
    expect(screen.getByText("आप कब खुलते हैं?").getAttribute("lang")).toBe("hi");
    expect(screen.getByText("हम नौ बजे खुलते हैं।").getAttribute("lang")).toBe("hi");
  });

  it("leaves lang off when the language is unknown and the agent speaks several", async () => {
    sse.mockImplementation(async (_p: string, _b: FormData, opts: { onEvent: (e: string, d: unknown) => void }) => {
      opts.onEvent("transcript", { user_text: "Hello there", language: "auto" });
      opts.onEvent("done", { turn_id: "t", agent_text: "Hi.", total_ms: 1000, conversation_id: "c1" });
    });
    await talk();
    await waitFor(() => expect(screen.getByText("Hi.")).toBeTruthy());
    expect(screen.getByText("Hello there").hasAttribute("lang")).toBe(false);
  });
});
