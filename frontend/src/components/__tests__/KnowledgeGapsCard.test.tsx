import { fireEvent, render, screen } from "@testing-library/react";
import { KnowledgeGapsCard } from "@/components/analytics/KnowledgeGapsCard";
import { useGaps } from "@/components/analytics/results";

jest.mock("@/components/analytics/results", () => ({ useGaps: jest.fn() }));
const mockHook = useGaps as jest.Mock;

beforeEach(() => localStorage.clear());

const cluster = (over = {}) => ({
  question: "What is the whitening fee?",
  examples: ["What is the whitening fee?", "how much is whitening", "whitening price please"],
  count: 4,
  last_asked: new Date().toISOString(),
  agent_id: "agent-1",
  agent_name: "Ria",
  language: "en",
  ...over,
});

describe("KnowledgeGapsCard", () => {
  it("lists each group with its count, agent, language name and an Add a document link", () => {
    mockHook.mockReturnValue({
      data: {
        clusters: [cluster(), cluster({ question: "Do you have parking?", examples: ["Do you have parking?"], count: 1, agent_id: "agent-2", agent_name: "Dev" })],
        total_gap_turns: 5,
        total_turns: 40,
        window_days: 30,
      },
      isLoading: false,
    });
    render(<KnowledgeGapsCard />);
    expect(screen.getByRole("heading", { name: /couldn.t answer/ })).toBeTruthy();
    expect(screen.getByText("What is the whitening fee?")).toBeTruthy();
    expect(screen.getByText(/Asked 4 times · Ria · English/)).toBeTruthy();
    expect(screen.getByText(/Asked 1 time · Dev/)).toBeTruthy();
    expect(screen.getByText(/5 of 40 replies/)).toBeTruthy();
    const link = screen.getByRole("link", { name: /Add a document: What is the whitening fee/ });
    expect(link.getAttribute("href")).toBe("/dashboard/studio/agents?agent=agent-1&tab=knowledge");
    expect(screen.getAllByRole("link")).toHaveLength(2);
  });

  it("shows which signal flagged each group", () => {
    mockHook.mockReturnValue({
      data: {
        clusters: [
          cluster({ reason: "said it didn't know" }),
          cluster({ question: "Do you have parking?", examples: ["Do you have parking?"], reason: "no matching document" }),
          cluster({ question: "Who won the match?", examples: ["Who won the match?"], reason: "weak match" }),
          cluster({ question: "Old turn", examples: ["Old turn"] }),
        ],
        total_gap_turns: 8,
        total_turns: 40,
        window_days: 30,
      },
      isLoading: false,
    });
    render(<KnowledgeGapsCard />);
    const chips = screen.getAllByTestId("gap-reason").map((c) => c.textContent);
    expect(chips).toEqual(["said it didn't know", "no matching document", "weak match"]);
    expect(screen.getByText("weak match").getAttribute("title")).toMatch(/weak match/);
  });

  it("expands the other ways a question was asked", () => {
    mockHook.mockReturnValue({ data: { clusters: [cluster()], total_gap_turns: 4, total_turns: 10, window_days: 30 }, isLoading: false });
    render(<KnowledgeGapsCard />);
    expect(screen.queryByText("how much is whitening")).toBeNull();
    const btn = screen.getByRole("button", { name: /other ways it was asked \(2\)/ });
    expect(btn.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(btn);
    expect(btn.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("how much is whitening")).toBeTruthy();
    expect(screen.getByText("whitening price please")).toBeTruthy();
  });

  it("shows the empty state when nothing went unanswered", () => {
    mockHook.mockReturnValue({ data: { clusters: [], total_gap_turns: 0, total_turns: 12, window_days: 30 }, isLoading: false });
    render(<KnowledgeGapsCard />);
    expect(screen.getByText("No unanswered questions in the last 30 days")).toBeTruthy();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("shows an error with retry when the request fails", () => {
    const mutate = jest.fn();
    mockHook.mockReturnValue({ data: undefined, error: new Error("boom"), isLoading: false, mutate });
    render(<KnowledgeGapsCard />);
    expect(screen.getByText(/Could not load unanswered questions/)).toBeTruthy();
  });

  it("hides a question marked as fine, remembers it, and lets you show it again", () => {
    mockHook.mockReturnValue({
      data: { clusters: [cluster(), cluster({ question: "Do you have parking?", examples: ["Do you have parking?"] })], total_gap_turns: 5, total_turns: 40, window_days: 30 },
      isLoading: false,
    });
    const { unmount } = render(<KnowledgeGapsCard />);
    fireEvent.click(screen.getByRole("button", { name: /Mark as fine: What is the whitening fee/ }));
    expect(screen.queryByText("What is the whitening fee?", { selector: "p" })).toBeNull();
    expect(screen.getByText("Do you have parking?")).toBeTruthy();
    unmount();
    render(<KnowledgeGapsCard />); // a fresh visit still hides it
    expect(screen.queryByRole("link", { name: /Add a document: What is the whitening fee/ })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Marked as fine \(1\)/ }));
    fireEvent.click(screen.getByRole("button", { name: "Show again" }));
    expect(screen.getByRole("link", { name: /Add a document: What is the whitening fee/ })).toBeTruthy();
  });

  it("asks for the chosen agent and window", () => {
    mockHook.mockReturnValue({ data: { clusters: [], total_gap_turns: 0, total_turns: 3, window_days: 7 }, isLoading: false });
    render(<KnowledgeGapsCard agentId="agent-9" days={7} />);
    expect(mockHook).toHaveBeenCalledWith(7, "agent-9");
    expect(screen.getByText("No unanswered questions in the last 7 days")).toBeTruthy();
  });
});
