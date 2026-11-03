import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { PossibleMistakes } from "@/components/analytics/PossibleMistakes";
import { useAnswerAudit } from "@/components/analytics/answerAudit";
import { api } from "@/lib/api";

jest.mock("@/components/analytics/answerAudit", () => ({ useAnswerAudit: jest.fn() }));
jest.mock("@/lib/api", () => ({
  api: { put: jest.fn(), post: jest.fn() },
  ApiError: class ApiError extends Error {
    status: number;
    constructor(message?: string, status = 0) {
      super(message);
      this.status = status;
    }
  },
}));

const hook = useAnswerAudit as jest.Mock;
const put = api.put as jest.Mock;
const post = api.post as jest.Mock;

const item = (over = {}) => ({
  turn_id: "t1",
  agent_id: "agent-1",
  agent_name: "Ria",
  conversation_id: "c1",
  channel: "phone",
  question: "Do you deliver?",
  reply: "We also offer free delivery across the whole country.",
  weakest_sentence: "We also offer free delivery across the whole country.",
  passage: "The refund window is thirty days from the date of purchase.",
  passage_source: "stored",
  score: 0.04,
  threshold: 0.5,
  feedback: null,
  at: new Date().toISOString(),
  ...over,
});
const audit = (over = {}) => ({
  flagged: [item()],
  counts: { checked: 5, flagged: 1, not_checked_language: 2, marked_fine: 0, marked_wrong: 0 },
  threshold: 0.5,
  days: 30,
  enabled: true,
  model_ready: true,
  note: "",
  run: null,
  ...over,
});
const serve = (data: unknown, mutate = jest.fn()) => hook.mockReturnValue({ data, isLoading: false, error: undefined, mutate });

beforeEach(() => {
  put.mockReset().mockResolvedValue({});
  post.mockReset().mockResolvedValue({});
});

describe("PossibleMistakes", () => {
  it("shows the question, what the agent said, what the document says and the honest language note", () => {
    serve(audit());
    render(<PossibleMistakes />);
    expect(screen.getByRole("heading", { name: "Possible mistakes" })).toBeTruthy();
    expect(screen.getByText("Do you deliver?")).toBeTruthy();
    expect(screen.getByText(/free delivery across the whole country/)).toBeTruthy();
    expect(screen.getByText(/thirty days from the date of purchase/)).toBeTruthy();
    expect(screen.getByText(/Checked automatically in English; Hindi and Hinglish answers are not checked yet\./)).toBeTruthy();
    expect(screen.getByText(/5 answers checked, 1 flagged, 2 not checked \(language\)/)).toBeTruthy();
    expect(screen.getByText(/Ria · /)).toBeTruthy();
  });

  it("says when the passage was found again afterwards", () => {
    serve(audit({ flagged: [item({ passage_source: "approx" })] }));
    render(<PossibleMistakes />);
    expect(screen.getByText(/closest passage, found again afterwards/)).toBeTruthy();
  });

  it("It was fine saves the owner's verdict and reloads the list", async () => {
    const mutate = jest.fn();
    serve(audit(), mutate);
    render(<PossibleMistakes />);
    fireEvent.click(screen.getByRole("button", { name: /It was fine: Do you deliver/ }));
    await waitFor(() => expect(put).toHaveBeenCalledWith("/api/answer-audit/t1", { feedback: "fine" }));
    await waitFor(() => expect(mutate).toHaveBeenCalled());
  });

  it("It was wrong links to the agent's documents and records the verdict", () => {
    serve(audit());
    render(<PossibleMistakes />);
    const link = screen.getByRole("link", { name: /It was wrong, add a document/ });
    expect(link.getAttribute("href")).toBe("/dashboard/studio/agents?agent=agent-1&tab=knowledge");
    link.addEventListener("click", (e) => e.preventDefault()); // jsdom cannot navigate
    fireEvent.click(link);
    expect(put).toHaveBeenCalledWith("/api/answer-audit/t1", { feedback: "wrong" });
  });

  it("Make an instant answer adds the question and points to the Phone tab; a duplicate is explained", async () => {
    serve(audit());
    render(<PossibleMistakes />);
    fireEvent.click(screen.getByRole("button", { name: /Make an instant answer: Do you deliver/ }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/agents/agent-1/instant-answers", { question: "Do you deliver?" }));
    expect(await screen.findByText(/Added to this agent's instant answers/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open Phone tab" }).getAttribute("href")).toBe("/dashboard/studio/agents?agent=agent-1&tab=phone");
  });

  it("Check recent answers starts a run for the chosen agent", async () => {
    const mutate = jest.fn();
    serve(audit(), mutate);
    render(<PossibleMistakes agentId="agent-9" />);
    fireEvent.click(screen.getByRole("button", { name: "Check recent answers" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/answer-audit/run", { agent_id: "agent-9" }));
    await waitFor(() => expect(mutate).toHaveBeenCalled());
  });

  it("explains the first check downloads a model when nothing has been checked", () => {
    serve(audit({ flagged: [], model_ready: false, counts: { checked: 0, flagged: 0, not_checked_language: 0, marked_fine: 0, marked_wrong: 0 } }));
    render(<PossibleMistakes />);
    expect(screen.getByText(/first check downloads a small checking model/)).toBeTruthy();
  });

  it("shows progress while a check runs and says when nothing was found", () => {
    serve(audit({ flagged: [], run: { id: "r", state: "running", total: 12, done: 4, checked: 3, flagged: 0, skipped: 1, language: 0, error: null } }));
    const first = render(<PossibleMistakes />);
    expect(screen.getByText(/Checking answers… 4 of 12/)).toBeTruthy();
    expect((screen.getByRole("button", { name: /Check recent answers/ }) as HTMLButtonElement).disabled).toBe(true);
    first.unmount();
    serve(audit({ flagged: [] }));
    render(<PossibleMistakes />);
    expect(screen.getByText("No possible mistakes found in the answers checked.")).toBeTruthy();
  });

  it("shows a failed run's reason and an error state when the list cannot load", () => {
    serve(audit({ flagged: [], run: { id: "r", state: "error", total: 1, done: 0, checked: 0, flagged: 0, skipped: 0, language: 0, error: "Not enough free memory" } }));
    const first = render(<PossibleMistakes />);
    expect(screen.getByText("Not enough free memory")).toBeTruthy();
    first.unmount();
    hook.mockReturnValue({ data: undefined, isLoading: false, error: new Error("boom"), mutate: jest.fn() });
    render(<PossibleMistakes />);
    expect(screen.getByText("Could not load the answer check")).toBeTruthy();
  });
});
