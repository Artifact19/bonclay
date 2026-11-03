import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { InstantAnswersPanel, type InstantAnswer, type InstantAnswersView } from "@/components/agents/InstantAnswersPanel";
import { api } from "@/lib/api";
import { toast } from "sonner";
import type { Agent } from "@/types/api";

jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({ api: { get: jest.fn(), put: jest.fn(), post: jest.fn(), del: jest.fn() } }));
jest.mock("@/components/agents/ConfirmDialog", () => ({
  ConfirmDialog: ({ open, title, confirmLabel, onConfirm, onCancel }: { open: boolean; title: string; confirmLabel: string; onConfirm: () => void; onCancel: () => void }) =>
    open ? (
      <div role="dialog" aria-label={title}>
        <button onClick={onConfirm}>{confirmLabel}</button>
        <button onClick={onCancel}>Keep it</button>
      </div>
    ) : null,
}));

const get = api.get as jest.Mock;
const put = api.put as jest.Mock;
const post = api.post as jest.Mock;
const del = api.del as jest.Mock;

const agent: Agent = {
  id: "a1",
  name: "Clinic",
  greeting: "Hello",
  system_instructions: "x",
  supported_languages: ["en", "hi"],
  voice_id: null,
  created_at: "2026-10-01",
  document_count: 2,
};

const row = (over: Partial<InstantAnswer> = {}): InstantAnswer => ({
  id: "r1",
  question: "What are the clinic timings on Sunday?",
  answer: "Open nine to one.",
  language: "en",
  status: "suggested",
  source: "template",
  paraphrases: [],
  times_asked: 0,
  ...over,
});

const view = (answers: InstantAnswer[] = [], over: Partial<InstantAnswersView> = {}): InstantAnswersView => ({
  answers,
  counts: {},
  build: null,
  document_count: 2,
  has_tools: false,
  threshold: 0.86,
  enabled: true,
  ...over,
});

beforeEach(() => {
  jest.clearAllMocks();
  put.mockResolvedValue({});
  post.mockResolvedValue({});
  del.mockResolvedValue({ deleted: true });
});

describe("InstantAnswersPanel", () => {
  it("explains the feature, loads the agent's answers and shows an empty state", async () => {
    get.mockResolvedValue(view());
    render(<InstantAnswersPanel agent={agent} />);
    expect(screen.getByText(/Common questions your agent answers immediately on calls/)).toBeInTheDocument();
    expect(await screen.findByTestId("instant-empty")).toHaveTextContent(/No instant answers yet/);
    expect(get).toHaveBeenCalledWith("/api/agents/a1/instant-answers");
    expect(screen.getByRole("button", { name: "Build suggestions" })).toBeEnabled();
  });

  it("lists each question with its answer, status, language and where it came from", async () => {
    get.mockResolvedValue(
      view(
        [
          row({ id: "r1", status: "approved", source: "calls", times_asked: 7, paraphrases: ["is the clinic open on sunday"] }),
          row({ id: "r2", question: "Where is the clinic?", answer: "Main Road.", status: "needs_review", source: "owner" }),
          row({ id: "r3", question: "क्लिनिक कब खुलता है", answer: "नौ बजे।", language: "hi", status: "disabled" }),
        ],
        { counts: { approved: 1, needs_review: 1, disabled: 1 } }
      )
    );
    render(<InstantAnswersPanel agent={agent} />);
    const list = await screen.findByTestId("instant-list");
    const items = within(list).getAllByRole("listitem").filter((li) => li.dataset.testid?.startsWith("instant-"));
    // what needs a look comes first
    expect(items.map((li) => li.dataset.testid)).toEqual(["instant-r2", "instant-r1", "instant-r3"]);
    const live = screen.getByTestId("instant-r1");
    expect(within(live).getByText("Live on calls")).toBeInTheDocument();
    expect(within(live).getByText("Asked by callers 7 times")).toBeInTheDocument();
    expect(within(live).getByLabelText("What the agent says")).toHaveValue("Open nine to one.");
    expect(within(live).getByText(/Other ways of asking this \(1\)/)).toBeInTheDocument();
    const review = screen.getByTestId("instant-r2");
    expect(within(review).getByText("Documents changed: check again")).toBeInTheDocument();
    expect(within(review).getByText(/not used on calls/)).toBeInTheDocument();
    expect(within(screen.getByTestId("instant-r3")).getByText("Hindi")).toBeInTheDocument();
    expect(screen.getByTestId("instant-summary")).toHaveTextContent("1 live, 1 waiting for you");
  });

  it("approves a suggestion, sending an edited answer with the approval", async () => {
    get.mockResolvedValue(view([row()]));
    render(<InstantAnswersPanel agent={agent} />);
    const box = await screen.findByLabelText("What the agent says");
    fireEvent.change(box, { target: { value: "Open nine to noon on Sunday." } });
    expect(screen.getByRole("button", { name: "Save answer" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));
    await waitFor(() => expect(put).toHaveBeenCalledWith("/api/agents/a1/instant-answers/r1", { status: "approved", answer: "Open nine to noon on Sunday." }));
    expect(toast.success).toHaveBeenCalledWith(expect.stringMatching(/Approved/));
    expect(get.mock.calls.length).toBeGreaterThan(1); // reloaded
  });

  it("approves an untouched suggestion with the status only", async () => {
    get.mockResolvedValue(view([row()]));
    render(<InstantAnswersPanel agent={agent} />);
    await screen.findByLabelText("What the agent says");
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));
    await waitFor(() => expect(put).toHaveBeenCalledWith("/api/agents/a1/instant-answers/r1", { status: "approved" }));
  });

  it("saves an edit on its own, and turns an approved answer off", async () => {
    get.mockResolvedValue(view([row({ status: "approved" })]));
    render(<InstantAnswersPanel agent={agent} />);
    fireEvent.change(await screen.findByLabelText("What the agent says"), { target: { value: "Open nine to two." } });
    fireEvent.click(screen.getByRole("button", { name: "Save answer" }));
    await waitFor(() => expect(put).toHaveBeenCalledWith("/api/agents/a1/instant-answers/r1", { answer: "Open nine to two." }));
    fireEvent.click(await screen.findByRole("button", { name: "Turn off" }));
    await waitFor(() => expect(put).toHaveBeenCalledWith("/api/agents/a1/instant-answers/r1", { status: "disabled" }));
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
  });

  it("explains why Approve is disabled when there is no answer yet", async () => {
    get.mockResolvedValue(view([row({ answer: "", source: "owner" })]));
    render(<InstantAnswersPanel agent={agent} />);
    const approve = await screen.findByRole("button", { name: "Approve" });
    expect(approve).toBeDisabled();
    expect(approve).toHaveAttribute("title", expect.stringMatching(/Write an answer first/));
    fireEvent.change(screen.getByLabelText("What the agent says"), { target: { value: "Yes." } });
    expect(screen.getByRole("button", { name: "Approve" })).toBeEnabled();
  });

  it("asks before deleting, then deletes", async () => {
    get.mockResolvedValue(view([row()]));
    render(<InstantAnswersPanel agent={agent} />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(del).not.toHaveBeenCalled();
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(del).toHaveBeenCalledWith("/api/agents/a1/instant-answers/r1"));
    expect(toast.success).toHaveBeenCalledWith("Deleted.");
  });

  it("builds suggestions, showing progress while it runs and the result when it is done", async () => {
    jest.useFakeTimers();
    try {
      const running = (done: number) =>
        view([], { build: { state: "running", total: 4, done, current: "Where is the clinic?", added: 0, updated: 0, skipped: [], error: null } });
      get
        .mockResolvedValueOnce(view())
        .mockResolvedValueOnce(running(1))
        .mockResolvedValueOnce(running(3))
        .mockResolvedValue(
          view([row()], {
            build: {
              state: "done", total: 4, done: 4, current: "", added: 3, updated: 0, error: null,
              skipped: [{ question: "What is the weather on mars", reason: "nothing in the documents is about this" }],
            },
          })
        );
      render(<InstantAnswersPanel agent={agent} />);
      await act(async () => {});
      fireEvent.click(screen.getByRole("button", { name: "Build suggestions" }));
      await act(async () => {});
      expect(post).toHaveBeenCalledWith("/api/agents/a1/instant-answers/build", {});
      expect(screen.getByTestId("build-progress")).toHaveTextContent("Writing suggestions: 1 of 4 done");
      expect(screen.getByTestId("build-progress")).toHaveTextContent("Now: Where is the clinic?");
      expect(screen.getByRole("button", { name: "Build suggestions" })).toBeDisabled();
      await act(async () => {
        jest.advanceTimersByTime(2000);
      });
      expect(screen.getByTestId("build-progress")).toHaveTextContent("3 of 4 done");
      await act(async () => {
        jest.advanceTimersByTime(2000);
      });
      expect(screen.queryByTestId("build-progress")).not.toBeInTheDocument();
      expect(screen.getByTestId("build-result")).toHaveTextContent("Added 3 new suggestions");
      expect(screen.getByText(/1 question was skipped/)).toBeInTheDocument();
      expect(screen.getByText(/nothing in the documents is about this/)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Build suggestions" })).toBeEnabled();
    } finally {
      jest.useRealTimers();
    }
  });

  it("disables Build suggestions with a reason when the agent has no documents", async () => {
    get.mockResolvedValue(view([], { document_count: 0 }));
    render(<InstantAnswersPanel agent={{ ...agent, document_count: 0 }} />);
    const button = await screen.findByRole("button", { name: "Build suggestions" });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("title", expect.stringMatching(/Upload a document first/));
    fireEvent.click(button);
    expect(post).not.toHaveBeenCalled();
  });

  it("can also write the answers in the agent's other languages", async () => {
    get.mockResolvedValue(view());
    render(<InstantAnswersPanel agent={agent} />);
    fireEvent.click(await screen.findByRole("checkbox", { name: /Also write each answer in Hindi/ }));
    fireEvent.click(screen.getByRole("button", { name: "Build suggestions" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/agents/a1/instant-answers/build", { languages: ["hi"] }));
  });

  it("shows why a build failed", async () => {
    get.mockResolvedValue(view([], { build: { state: "failed", total: 0, done: 0, current: "", added: 0, updated: 0, skipped: [], error: "The language model is not available right now." } }));
    render(<InstantAnswersPanel agent={agent} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("The language model is not available right now.");
  });

  it("adds a question, with an optional answer and language", async () => {
    get.mockResolvedValue(view());
    render(<InstantAnswersPanel agent={agent} />);
    const add = await screen.findByRole("button", { name: "Add question" });
    expect(add).toBeDisabled();
    expect(add).toHaveAttribute("title", "Type the question first.");
    fireEvent.change(screen.getByLabelText("A question callers ask"), { target: { value: "Is there parking?" } });
    expect(screen.getByRole("button", { name: "Add question" })).toBeEnabled();
    fireEvent.change(screen.getByLabelText("Answer (optional)"), { target: { value: "Yes, behind the building." } });
    fireEvent.change(screen.getByLabelText(/Language of the question/), { target: { value: "hi" } });
    fireEvent.click(screen.getByRole("button", { name: "Add question" }));
    await waitFor(() =>
      expect(post).toHaveBeenCalledWith("/api/agents/a1/instant-answers", { question: "Is there parking?", answer: "Yes, behind the building.", language: "hi" })
    );
    expect(toast.success).toHaveBeenCalledWith("Added and approved.");
    await waitFor(() => expect(screen.getByLabelText("A question callers ask")).toHaveValue(""));
  });

  it("shows the server's reason when a question cannot be added", async () => {
    get.mockResolvedValue(view());
    post.mockRejectedValue(new Error("This cannot be an instant answer: it asks more than one thing at once."));
    render(<InstantAnswersPanel agent={agent} />);
    fireEvent.change(await screen.findByLabelText("A question callers ask"), { target: { value: "Fee? And when?" } });
    fireEvent.click(screen.getByRole("button", { name: "Add question" }));
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(expect.stringMatching(/more than one thing/)));
    expect(screen.getByLabelText("A question callers ask")).toHaveValue("Fee? And when?"); // nothing was lost
  });

  it("says so when it cannot load, and offers to try again", async () => {
    get.mockRejectedValueOnce(new Error("Network down")).mockResolvedValue(view());
    render(<InstantAnswersPanel agent={agent} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Network down");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByTestId("instant-empty")).toBeInTheDocument();
  });
});
