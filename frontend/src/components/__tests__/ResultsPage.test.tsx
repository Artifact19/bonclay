import { fireEvent, render, screen } from "@testing-library/react";
import ResultsPage from "@/app/(app)/dashboard/analytics/page";
import { languageName, seconds } from "@/components/dashboard/format";
import { useAgents } from "@/hooks/useApi";
import { useResults } from "@/components/analytics/results";

jest.mock("@/hooks/useApi", () => ({
  useAgents: jest.fn(),
  useAnalytics: jest.fn(() => ({ data: undefined, isLoading: true })),
  usePerfTurns: jest.fn(() => ({ data: undefined, isLoading: true })),
}));
jest.mock("@/components/analytics/results", () => ({
  ...jest.requireActual("@/components/analytics/results"),
  useResults: jest.fn(),
  useGaps: jest.fn(() => ({ data: { clusters: [], total_gap_turns: 0, total_turns: 0, window_days: 7 }, isLoading: false })),
}));
jest.mock("recharts", () => {
  const Stub = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>;
  return new Proxy({}, { get: () => Stub });
});

const mockAgents = useAgents as jest.Mock;
const mockResults = useResults as jest.Mock;

const results = (over = {}) => ({
  range: "7d", days: 7, agent_id: null, agent_name: null,
  conversations: 12, questions_asked: 40, questions_answered: 36, questions_unanswered: 4,
  reply_time: { count: 40, p50_ms: 1400, p95_ms: 3800, target_ms: 3000, within_target_pct: 92.5 },
  daily: [{ date: "2026-10-16", asked: 40, unanswered: 4 }],
  by_language: [{ language: "hi", count: 30 }, { language: "en", count: 10 }],
  bookings: { requested: 5, booked: 3, waiting: 2 },
  ...over,
});

beforeEach(() => {
  mockAgents.mockReturnValue({ data: [{ id: "a1", name: "Ria" }, { id: "a2", name: "Dev" }], isLoading: false });
  mockResults.mockReturnValue({ data: results(), isLoading: false });
});

describe("Results page", () => {
  it("answers 'is my agent working' in plain words", () => {
    render(<ResultsPage />);
    expect(screen.getByRole("heading", { name: "Results" })).toBeTruthy();
    expect(screen.getByTestId("verdict").textContent).toMatch(/answered 36 of 40 questions \(90%\)/);
    expect(screen.getByText("Conversations handled")).toBeTruthy();
    expect(screen.getByText("Questions it couldn't answer")).toBeTruthy();
    expect(screen.getByTestId("stat-p50").textContent).toMatch(/1\.4 s/);
    expect(screen.getByTestId("stat-p95").textContent).toMatch(/3\.8 s/);
    expect(screen.getByTestId("stat-bookings").textContent).toMatch(/3.*2 requests waiting/);
    expect(screen.getByText("Hindi")).toBeTruthy();
    expect(screen.queryByText("hi")).toBeNull();
  });

  it("puts the unanswered questions above the activity chart and hides engine charts in Advanced", () => {
    render(<ResultsPage />);
    const gaps = screen.getByRole("region", { name: /couldn.t answer/ });
    const activity = screen.getByRole("region", { name: "Questions per day" });
    expect(gaps.compareDocumentPosition(activity) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const adv = screen.getByRole("button", { name: /Advanced/ });
    expect(adv.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText(/Audio clips made, by voice engine/)).toBeNull();
    fireEvent.click(adv);
    expect(screen.getByText(/Audio clips made, by voice engine/)).toBeTruthy();
    expect(screen.getByText("Where reply time goes")).toBeTruthy();
  });

  it("asks for the chosen agent and range", () => {
    render(<ResultsPage />);
    fireEvent.change(screen.getByLabelText("Agent"), { target: { value: "a2" } });
    expect(mockResults).toHaveBeenLastCalledWith("7d", "a2");
    fireEvent.click(screen.getByRole("button", { name: "30 days" }));
    expect(mockResults).toHaveBeenLastCalledWith("30d", "a2");
  });

  it("does not fake figures the backend cannot give", () => {
    mockResults.mockReturnValue({
      data: results({ bookings: null, reply_time: { count: 0, p50_ms: null, p95_ms: null, target_ms: 3000, within_target_pct: null } }),
      isLoading: false,
    });
    render(<ResultsPage />);
    expect(screen.getByTestId("stat-bookings").textContent).toMatch(/Not tracked yet/);
    expect(screen.getByTestId("stat-p50").textContent).toMatch(/Not tracked yet/);
    expect(screen.getByTestId("stat-p95").textContent).toMatch(/Not tracked yet/);
  });

  it("shows a first-run empty state with no agents", () => {
    mockAgents.mockReturnValue({ data: [], isLoading: false });
    mockResults.mockReturnValue({ data: undefined, isLoading: true });
    render(<ResultsPage />);
    expect(screen.getByText(/appear once an agent takes its first conversation/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Create your first agent" }).getAttribute("href")).toBe("/dashboard/studio/agents");
    expect(screen.queryByLabelText("Agent")).toBeNull();
  });

  it("shows an empty state with next steps when an agent has no conversations yet", () => {
    mockResults.mockReturnValue({
      data: results({ conversations: 0, questions_asked: 0, questions_answered: 0, questions_unanswered: 0, by_language: [] }),
      isLoading: false,
    });
    render(<ResultsPage />);
    expect(screen.getByText(/No conversations in the last 7 days/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Talk to your agent" }).getAttribute("href")).toBe("/dashboard/playground");
    fireEvent.click(screen.getByRole("button", { name: "Look back 30 days" }));
    expect(mockResults).toHaveBeenLastCalledWith("30d", null);
  });
});

describe("format helpers", () => {
  it("names languages instead of showing codes", () => {
    expect(languageName("hi")).toBe("Hindi");
    expect(languageName("en")).toBe("English");
    expect(languageName("mr")).toBe("Marathi");
    expect(languageName("ta")).toBe("Tamil");
    expect(languageName("hi-IN")).toBe("Hindi");
    expect(languageName("auto")).toBe("Detected automatically");
    expect(languageName("")).toBe("Not recorded");
    expect(languageName(null)).toBe("Not recorded");
  });

  it("formats milliseconds as seconds", () => {
    expect(seconds(860)).toBe("0.9 s");
    expect(seconds(12400)).toBe("12 s");
    expect(seconds(null)).toBe("");
  });
});
