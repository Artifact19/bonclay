import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { TestCallDialog } from "@/components/agents/TestCallDialog";
import { api } from "@/lib/api";
import type { Agent } from "@/types/api";

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
  return { api: { post: jest.fn(), sse: jest.fn() }, ApiError, API_URL: "http://api.test" };
});

const post = api.post as jest.Mock;
const sse = api.sse as jest.Mock;

const AGENT = { id: "a1", name: "Sunrise Clinic", document_count: 2, voice_id: null } as unknown as Agent;

/** Every clip "plays" and ends on its own. */
class FakeAudio {
  static played: string[] = [];
  preload = "";
  private handlers: Record<string, (() => void)[]> = {};
  constructor(public src: string) {}
  play() {
    FakeAudio.played.push(this.src);
    setTimeout(() => (this.handlers.ended || []).forEach((f) => f()), 0);
    return Promise.resolve();
  }
  pause() {}
  addEventListener(type: string, fn: () => void) {
    (this.handlers[type] ||= []).push(fn);
  }
}

beforeAll(() => {
  Object.assign(global, { Audio: FakeAudio });
});

const callerLines = ["What are the clinic timings on Sunday?", "Is there anything else I should know?"];

function mockApi({ verdict = { verdict: "answered_from_documents", label: "Answered from your documents (retrieval used).", ok: true } } = {}) {
  let turn = 0;
  post.mockImplementation(async (path: string, body: Record<string, unknown>) => {
    if (path.endsWith("/test-call/caller")) {
      return { text: callerLines[body.turn as number], voice: "am_adam", audio_base64: btoa("RIFFfake"), turns_total: 2 };
    }
    if (path.endsWith("/test-call/verdict")) return verdict;
    throw new Error(`unexpected ${path}`);
  });
  sse.mockImplementation(async (_path: string, _fd: FormData, opts: { onEvent: (e: string, d: unknown) => void }) => {
    const n = turn++;
    opts.onEvent("transcript", { user_text: callerLines[n], language: "en" });
    opts.onEvent("sentence", { index: 0, text: n === 0 ? "This is an AI assistant." : "No, that is all.", audio_url: `/api/conversations/turns/t${n}/sentences/0/audio` });
    opts.onEvent("sentence", { index: 1, text: "We open at nine on Sunday.", audio_url: `/api/conversations/turns/t${n}/sentences/1/audio` });
    opts.onEvent("done", {
      turn_id: `t${n}`, agent_text: n === 0 ? "This is an AI assistant. We open at nine on Sunday." : "No, that is all. We open at nine on Sunday.",
      conversation_id: "c1", sources_used: 3, grounded: true,
    });
  });
}

beforeEach(() => {
  jest.clearAllMocks();
  FakeAudio.played = [];
});

describe("Test call dialog", () => {
  it("lists the scenarios and does nothing until started", () => {
    render(<TestCallDialog agent={AGENT} open onClose={() => undefined} />);
    const options = screen.getAllByRole("option").map((o) => o.textContent);
    expect(options).toEqual([
      "Ask a normal question from your document",
      "Ask something not in your document",
      "Try to trick the agent",
      "Speak Hinglish",
      "Custom",
    ]);
    expect(post).not.toHaveBeenCalled();
    expect(screen.queryByLabelText("What the caller says")).toBeNull();
    fireEvent.change(screen.getByLabelText("Scenario"), { target: { value: "custom" } });
    expect(screen.getByLabelText("What the caller says")).toBeTruthy();
    expect(screen.getByRole("button", { name: /start practice call/i })).toBeDisabled(); // custom needs a line
  });

  it("plays the caller, sends real audio as a test call, plays the reply with the AI label and shows the verdict", async () => {
    mockApi();
    render(<TestCallDialog agent={AGENT} open onClose={() => undefined} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /start practice call/i }));
    });
    await waitFor(() => expect(screen.getAllByText("Answered from your documents (retrieval used).")).toHaveLength(2));

    // two turns, the first caller line comes from the server for scenario "normal"
    expect(post.mock.calls.filter(([p]) => p.endsWith("/test-call/caller")).map(([, b]) => b)).toEqual([
      { scenario: "normal", turn: 0, custom_text: undefined },
      { scenario: "normal", turn: 1, custom_text: undefined },
    ]);
    // the caller's audio is played (a data URI), then the agent's sentence clips, in order
    expect(FakeAudio.played[0]).toMatch(/^data:audio\/wav;base64,/);
    expect(FakeAudio.played.slice(1, 3)).toEqual([
      "http://api.test/api/conversations/turns/t0/sentences/0/audio",
      "http://api.test/api/conversations/turns/t0/sentences/1/audio",
    ]);

    // the existing streaming turn endpoint is used with the audio and the test flag; turn 2 continues the conversation
    expect(sse).toHaveBeenCalledTimes(2);
    const [path0, fd0] = sse.mock.calls[0];
    expect(path0).toBe("/api/conversations/a1/turn/stream");
    expect((fd0 as FormData).get("test_call")).toBe("true");
    expect((fd0 as FormData).get("audio")).toBeInstanceOf(Blob);
    expect((fd0 as FormData).get("conversation_id")).toBeNull();
    expect((sse.mock.calls[1][1] as FormData).get("conversation_id")).toBe("c1");

    // transcript, AI-generated label on the agent's speech
    expect(screen.getByText("What are the clinic timings on Sunday?")).toBeTruthy();
    expect(screen.getAllByTestId("ai-generated-tag")).toHaveLength(2);
    expect(screen.getByText("This is an AI assistant. We open at nine on Sunday.")).toBeTruthy();
    const verdictCall = post.mock.calls.find(([p]) => p.endsWith("/test-call/verdict"))!;
    expect(verdictCall[1]).toEqual({
      scenario: "normal", caller_text: callerLines[0], agent_text: "This is an AI assistant. We open at nine on Sunday.", sources_used: 3,
    });
    expect(screen.getByRole("button", { name: /call again/i })).toBeEnabled();
  });

  it("a custom call is one turn and sends the typed line", async () => {
    mockApi({ verdict: { verdict: "answered_from_documents", label: "Answered from your documents (retrieval used).", ok: true } });
    post.mockImplementationOnce(async () => ({ text: "Can I reschedule?", voice: "am_adam", audio_base64: btoa("x"), turns_total: 1 }));
    render(<TestCallDialog agent={AGENT} open onClose={() => undefined} />);
    fireEvent.change(screen.getByLabelText("Scenario"), { target: { value: "custom" } });
    fireEvent.change(screen.getByLabelText("What the caller says"), { target: { value: "Can I reschedule?" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /start practice call/i }));
    });
    await waitFor(() => expect(screen.getByText("Answered from your documents (retrieval used).")).toBeTruthy());
    expect(post.mock.calls[0][1]).toEqual({ scenario: "custom", turn: 0, custom_text: "Can I reschedule?" });
    expect(sse).toHaveBeenCalledTimes(1);
  });

  it("shows a refused trick as a verdict and a failed call as an error", async () => {
    mockApi({ verdict: { verdict: "refused_trick", label: "Refused the trick and kept its rules.", ok: true } });
    const { rerender } = render(<TestCallDialog agent={AGENT} open onClose={() => undefined} />);
    fireEvent.change(screen.getByLabelText("Scenario"), { target: { value: "trick" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /start practice call/i }));
    });
    await waitFor(() => expect(screen.getAllByText("Refused the trick and kept its rules.")).toHaveLength(2));
    await waitFor(() => expect(screen.getByRole("button", { name: /call again/i })).toBeEnabled());

    sse.mockImplementation(async (_p: string, _fd: FormData, opts: { onEvent: (e: string, d: unknown) => void }) => {
      opts.onEvent("error", { code: "LLM_UNAVAILABLE", message: "No language model is available" });
    });
    rerender(<TestCallDialog agent={AGENT} open onClose={() => undefined} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /call again/i }));
    });
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("No language model is available"));
    expect(screen.getByRole("button", { name: /call again/i })).toBeEnabled();
  });

  it("marks the transcript text with its language: Hinglish as Hindi in Roman letters, else the agent's one language", async () => {
    mockApi();
    const { unmount } = render(<TestCallDialog agent={AGENT} open onClose={() => undefined} />);
    fireEvent.change(screen.getByLabelText("Scenario"), { target: { value: "hinglish" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /start practice call/i }));
    });
    await waitFor(() => expect(screen.getAllByText("Answered from your documents (retrieval used).")).toHaveLength(2));
    expect(screen.getAllByText(callerLines[0])[0].getAttribute("lang")).toBe("hi-Latn");
    expect(screen.getByText("This is an AI assistant. We open at nine on Sunday.").getAttribute("lang")).toBe("hi-Latn");
    unmount();

    mockApi();
    render(<TestCallDialog agent={{ ...AGENT, supported_languages: ["ta"] } as unknown as Agent} open onClose={() => undefined} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /start practice call/i }));
    });
    await waitFor(() => expect(screen.getAllByText("Answered from your documents (retrieval used).")).toHaveLength(2));
    expect(screen.getAllByText(callerLines[0])[0].getAttribute("lang")).toBe("ta");
  });

  it("warns when the agent has no documents for a document question", () => {
    render(<TestCallDialog agent={{ ...AGENT, document_count: 0 } as Agent} open onClose={() => undefined} />);
    expect(screen.getByText(/no documents yet/i)).toBeTruthy();
  });
});
