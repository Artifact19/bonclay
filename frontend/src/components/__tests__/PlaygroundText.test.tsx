import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import PlaygroundPage from "@/app/(app)/dashboard/playground/page";
import { api } from "@/lib/api";
import {
  suggestedQuestionsFor,
  DEFAULT_QUESTIONS,
} from "@/components/playground/suggestions";

let mockParams = new URLSearchParams();
let mockAgents: Record<string, unknown>[] = [];
let mockTemplates: Record<string, unknown>[] = [];
jest.mock("next/navigation", () => ({ useSearchParams: () => mockParams }));
jest.mock("next/link", () => ({
  __esModule: true,
  default: ({ children, ...p }: { children: React.ReactNode }) => (
    <a {...p}>{children}</a>
  ),
}));
jest.mock("sonner", () => ({
  toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() },
}));
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
  return {
    api: { post: jest.fn(), sse: jest.fn(), upload: jest.fn() },
    ApiError,
    API_URL: "http://api.test",
  };
});
jest.mock("@/hooks/useApi", () => ({
  useAgents: () => ({ data: mockAgents, isLoading: false, error: undefined }),
  useAgentTemplates: () => ({ data: mockTemplates }),
}));
jest.mock("@/components/audio/WaveformPlayer", () => ({
  WaveformPlayer: () => null,
}));
jest.mock("@/components/dashboard/PerformancePanel", () => ({
  PerformancePanel: () => null,
}));
jest.mock("@/components/voices/Modal", () => ({ Modal: () => null }));
jest.mock("@/components/agents/CallDialog", () => ({ CallDialog: () => null }));

const post = api.upload as jest.Mock;
const sentText = () => (post.mock.calls.at(-1)![1] as FormData).get("text");
const agent = (
  id: string,
  name: string,
  extra: Record<string, unknown> = {},
) => ({
  id,
  name,
  greeting: "",
  system_instructions: "",
  supported_languages: ["en"],
  voice_id: null,
  created_at: "2026-10-01T10:00:00Z",
  document_count: 1,
  ...extra,
});

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    value: () => ({
      matches: false,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
    }),
    configurable: true,
  });
  Element.prototype.scrollIntoView = jest.fn();
  let n = 0;
  Object.defineProperty(global.crypto, "randomUUID", {
    value: () => `id-${++n}`,
    configurable: true,
  });
});

beforeEach(() => {
  jest.clearAllMocks();
  mockParams = new URLSearchParams();
  mockAgents = [
    agent("a1", "Sunrise Family Clinic Receptionist"),
    agent("a2", "College Helper", { supported_languages: ["en", "hi"] }),
  ];
  mockTemplates = [
    {
      id: "clinic",
      name: "Sunrise Family Clinic Receptionist",
      role: "",
      languages: ["en"],
      greeting: "",
      document: "",
      suggested_questions: [
        "Is Dr. Meera in on Monday?",
        "How much is a consultation?",
      ],
    },
  ];
});

describe("Playground: talk to your agent", () => {
  it("has the plain-language title and the typed-question box next to the mic", () => {
    render(<PlaygroundPage />);
    expect(
      screen.getByRole("heading", { name: "Talk to your agent" }),
    ).toBeTruthy();
    expect(screen.getByLabelText("Type a question")).toBeTruthy();
    expect(screen.getByRole("button", { name: /hold to talk/i })).toBeTruthy();
    expect(
      screen.getAllByRole("button", { name: /usual reply time/i }).length,
    ).toBeTruthy();
    expect(
      screen.getAllByRole("button", { name: /phone someone/i }).length,
    ).toBeTruthy();
  });

  it("opens on the agent named by ?agent=<id>, else the newest", () => {
    mockParams = new URLSearchParams("agent=a2");
    const { unmount } = render(<PlaygroundPage />);
    expect((screen.getByLabelText("Talk to") as HTMLSelectElement).value).toBe(
      "a2",
    );
    unmount();
    mockParams = new URLSearchParams("agent=gone");
    render(<PlaygroundPage />);
    expect((screen.getByLabelText("Talk to") as HTMLSelectElement).value).toBe(
      "a1",
    );
  });

  it("tells same-named agents apart in the list", () => {
    mockAgents = [
      agent("a1", "Clinic", { created_at: "2026-10-01T10:00:00Z" }),
      agent("a2", "Clinic", { created_at: "2026-10-09T10:00:00Z" }),
    ];
    render(<PlaygroundPage />);
    const labels = Array.from(
      (screen.getByLabelText("Talk to") as HTMLSelectElement).options,
    ).map((o) => o.text);
    expect(new Set(labels).size).toBe(2);
    expect(labels[0]).toMatch(/made /);
  });

  it("sends a typed question as text (no audio) and shows the reply with its time in plain words", async () => {
    post.mockResolvedValueOnce({
      user_text: "When do you open?",
      agent_text: "We open at nine.",
      latency_ms: 1234,
      grounded: true,
      conversation_id: "c1",
    });
    render(<PlaygroundPage />);
    fireEvent.change(screen.getByLabelText("Type a question"), {
      target: { value: "When do you open?" },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    });
    expect(post.mock.calls[0][0]).toBe("/api/conversations/a1/turn");
    expect(sentText()).toBe("When do you open?");
    expect((post.mock.calls[0][1] as FormData).get("audio")).toBeNull();
    expect(screen.getByText("When do you open?")).toBeTruthy();
    await waitFor(() =>
      expect(screen.getByText("We open at nine.")).toBeTruthy(),
    );
    expect(screen.getByText("Replied in 1.2 s")).toBeTruthy();
    expect(
      (screen.getByLabelText("Type a question") as HTMLInputElement).value,
    ).toBe("");
    expect(api.sse).not.toHaveBeenCalled();
  });

  it("does not send an empty question", () => {
    render(<PlaygroundPage />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("shows the template's own suggested questions and sends one when clicked", async () => {
    post.mockResolvedValueOnce({
      user_text: "x",
      agent_text: "Yes, Dr. Meera is in.",
      latency_ms: 900,
    });
    render(<PlaygroundPage />);
    expect(
      screen.getByRole("button", { name: "How much is a consultation?" }),
    ).toBeTruthy();
    await act(async () => {
      fireEvent.click(
        screen.getByRole("button", { name: "Is Dr. Meera in on Monday?" }),
      );
    });
    expect(post.mock.calls[0][0]).toBe("/api/conversations/a1/turn");
    expect(sentText()).toBe("Is Dr. Meera in on Monday?");
  });

  it("falls back to sensible defaults for an agent with no template, and follows the selected agent", () => {
    mockParams = new URLSearchParams("agent=a2");
    render(<PlaygroundPage />);
    expect(
      screen.queryByRole("button", { name: "Is Dr. Meera in on Monday?" }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: DEFAULT_QUESTIONS.en[0] }),
    ).toBeTruthy();
    expect(
      screen.getByRole("button", { name: DEFAULT_QUESTIONS.hinglish[0] }),
    ).toBeTruthy(); // speaks Hindi too
  });
});

describe("suggestedQuestionsFor", () => {
  const tpl = [
    {
      id: "t",
      name: "Clinic Bot",
      role: "",
      languages: ["en"],
      greeting: "",
      document: "",
      suggested_questions: ["a", "b", "c", "d", "e"],
    },
  ];
  it("uses the template by id or by name, at most four", () => {
    expect(
      suggestedQuestionsFor(
        { name: "x", supported_languages: ["en"], template_id: "t" },
        tpl,
      ),
    ).toEqual(["a", "b", "c", "d"]);
    expect(
      suggestedQuestionsFor(
        { name: " clinic bot ", supported_languages: ["en"] },
        tpl,
      ),
    ).toHaveLength(4);
  });
  it("gives English, Hindi-only and mixed defaults", () => {
    expect(
      suggestedQuestionsFor({ name: "n", supported_languages: ["en"] }, []),
    ).toEqual([...DEFAULT_QUESTIONS.en]);
    expect(
      suggestedQuestionsFor({ name: "n", supported_languages: ["hi"] }, []),
    ).toEqual([...DEFAULT_QUESTIONS.hi]);
    const mixed = suggestedQuestionsFor(
      { name: "n", supported_languages: ["en", "hi"] },
      [],
    );
    expect(mixed).toHaveLength(4);
    expect(mixed).toContain(DEFAULT_QUESTIONS.hinglish[0]);
  });
  it("has nothing without an agent", () => {
    expect(suggestedQuestionsFor(null, [])).toEqual([]);
  });
});
