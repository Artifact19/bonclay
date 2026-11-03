import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import AgentStudioPage from "@/app/(app)/dashboard/studio/agents/page";
import { toast } from "sonner";
import { useAgents } from "@/hooks/useApi";

let mockParams = new URLSearchParams();
jest.mock("next/navigation", () => ({ useSearchParams: () => mockParams }));
jest.mock("sonner", () => ({ toast: { error: jest.fn(), success: jest.fn() } }));
jest.mock("@/lib/api", () => {
  class ApiError extends Error {}
  return { api: { get: jest.fn().mockRejectedValue(new Error("offline")), post: jest.fn(), put: jest.fn(), del: jest.fn() }, ApiError, API_URL: "http://api.test" };
});
jest.mock("@/hooks/useApi", () => ({
  useAgents: jest.fn(),
  useVoices: () => ({ data: [{ id: "v9", name: "Asha", display_name: "Asha" }] }),
}));
jest.mock("@/components/agents/KnowledgePanel", () => ({ KnowledgePanel: () => <p>knowledge panel</p> }));
jest.mock("@/components/agents/TemplateGallery", () => ({ TemplateGallery: () => <p>gallery</p> }));
jest.mock("@/components/agents/TestCallDialog", () => ({ TestCallDialog: () => null }));
jest.mock("@/components/agents/CallDialog", () => ({ CallDialog: () => null }));
jest.mock("@/components/agents/ConfirmDialog", () => ({ ConfirmDialog: () => null }));
// The page only decides which agent to open and whether to focus Knowledge; the editor's own behaviour is tested below.
jest.mock("@/components/agents/AgentEditor", () => {
  const actual = jest.requireActual("@/components/agents/AgentEditor");
  return {
    ...actual,
    AgentEditor: (p: { agent: { name: string }; focusKnowledge?: boolean; tab?: string; onTabChange?: (t: string) => void }) => (
      <div data-testid="editor" data-focus={String(!!p.focusKnowledge)} data-tab={p.tab}>
        editing {p.agent.name}
        <button onClick={() => p.onTabChange?.("channels")}>go channels</button>
      </div>
    ),
  };
});

const mockAgents = useAgents as jest.Mock;
const agent = (id: string, name: string) => ({
  id,
  name,
  greeting: "",
  system_instructions: "",
  supported_languages: ["en"],
  voice_id: null,
  document_count: 0,
  created_at: "2026-10-01T00:00:00Z",
});
const withAgents = (data: unknown) =>
  mockAgents.mockReturnValue({ data, isLoading: false, error: undefined, mutate: jest.fn() });

beforeEach(() => {
  mockParams = new URLSearchParams();
  jest.clearAllMocks();
  window.history.replaceState(null, "", "/dashboard/studio/agents");
});

describe("Agent Studio deep link (?agent=<id>)", () => {
  it("opens that agent's editor straight away, set to focus its Documents with tab=knowledge", async () => {
    mockParams = new URLSearchParams("agent=a2&tab=knowledge");
    withAgents([agent("a1", "Ria"), agent("a2", "Dev")]);
    render(<AgentStudioPage />);
    const editor = await screen.findByTestId("editor");
    expect(editor.textContent).toContain("editing Dev");
    expect(editor.getAttribute("data-focus")).toBe("true");
  });

  it("shows the agent list without the parameter", () => {
    withAgents([agent("a1", "Ria")]);
    render(<AgentStudioPage />);
    expect(screen.queryByTestId("editor")).toBeNull();
    expect(screen.getByText("Ria")).toBeTruthy();
  });

  it("waits for the agents to load, then opens the editor", async () => {
    mockParams = new URLSearchParams("agent=a1");
    mockAgents.mockReturnValue({ data: undefined, isLoading: true, error: undefined, mutate: jest.fn() });
    const { rerender } = render(<AgentStudioPage />);
    expect(screen.queryByTestId("editor")).toBeNull();
    withAgents([agent("a1", "Ria")]);
    rerender(<AgentStudioPage />);
    expect((await screen.findByTestId("editor")).textContent).toContain("editing Ria");
  });

  it("tells the owner when the agent no longer exists and keeps the list", async () => {
    mockParams = new URLSearchParams("agent=gone");
    withAgents([agent("a1", "Ria")]);
    render(<AgentStudioPage />);
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("That agent no longer exists."));
    expect(screen.queryByTestId("editor")).toBeNull();
  });
});

describe("Agent page tabs (?agent=<id>&tab=...)", () => {
  it.each([
    ["basics", "basics", "false"],
    ["phone", "phone", "false"],
    ["channels", "channels", "false"],
    ["advanced", "advanced", "false"],
    ["knowledge", "basics", "true"], // the old link: Basics, scrolled to Documents
    [null, "basics", "false"], // the command palette's "Edit <agent>" link has no tab: no jump to documents
    ["nonsense", "basics", "false"],
  ])("tab=%s opens the %s tab (focus documents: %s)", async (param, expectedTab, focus) => {
    mockParams = new URLSearchParams(param ? `agent=a1&tab=${param}` : "agent=a1");
    withAgents([agent("a1", "Ria")]);
    render(<AgentStudioPage />);
    const editor = await screen.findByTestId("editor");
    expect(editor.getAttribute("data-tab")).toBe(expectedTab);
    expect(editor.getAttribute("data-focus")).toBe(focus);
  });

  it("changing the tab updates the address so the link can be shared, without leaving the agent", async () => {
    mockParams = new URLSearchParams("agent=a1&tab=basics");
    withAgents([agent("a1", "Ria")]);
    render(<AgentStudioPage />);
    fireEvent.click(await screen.findByText("go channels"));
    expect(window.location.search).toBe("?agent=a1&tab=channels");
    expect(screen.getByTestId("editor").getAttribute("data-tab")).toBe("channels");
  });
});

describe("Agents list", () => {
  it("shows what tells agents apart, with Talk first (to the Playground with that agent) and Edit", () => {
    withAgents([{ ...agent("a1", "Ria"), supported_languages: ["hi", "en"], document_count: 2, greeting: "Namaste" }]);
    render(<AgentStudioPage />);
    expect(screen.getByText("Hindi, English")).toBeTruthy();
    expect(screen.getByText("Default voice")).toBeTruthy();
    expect(screen.getByText(/2 documents/)).toBeTruthy();
    expect(screen.getByText("Ready")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Talk to Ria" }).getAttribute("href")).toBe("/dashboard/playground?agent=a1");
    expect(screen.getByRole("button", { name: "Edit Ria" })).toBeTruthy();
  });

  it("says when an agent needs documents", () => {
    withAgents([agent("a1", "Ria")]);
    render(<AgentStudioPage />);
    expect(screen.getByText("Needs documents")).toBeTruthy();
  });

  it("has no Delete button on the card: Delete is in the ... menu", () => {
    withAgents([agent("a1", "Ria")]);
    render(<AgentStudioPage />);
    expect(screen.queryByRole("button", { name: /^Delete/ })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "More actions for Ria" }));
    expect(screen.getByRole("menuitem", { name: "Delete agent" })).toBeTruthy();
    expect(screen.getByRole("menuitem", { name: "Practice call" })).toBeTruthy();
    expect(screen.getByRole("menuitem", { name: "Phone someone" })).toBeTruthy();
  });

  it("lists the agents with the templates in a side rail, and opens the create form on request", () => {
    withAgents([agent("a1", "Ria")]);
    render(<AgentStudioPage />);
    expect(screen.getByText("gallery")).toBeTruthy();
    expect(screen.queryByLabelText("Agent name")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Describe your own agent" }));
    expect(screen.getByLabelText("Agent name")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Describe your own agent" })).toBeNull();
  });

  it("shows the templates straight away when there are no agents", () => {
    withAgents([]);
    render(<AgentStudioPage />);
    expect(screen.getByText("gallery")).toBeTruthy();
  });

  it("Edit opens the agent on Basics and writes it into the address", async () => {
    withAgents([agent("a1", "Ria")]);
    render(<AgentStudioPage />);
    fireEvent.click(screen.getByRole("button", { name: "Edit Ria" }));
    const editor = await screen.findByTestId("editor");
    expect(editor.getAttribute("data-tab")).toBe("basics");
    expect(window.location.search).toBe("?agent=a1&tab=basics");
  });
});

describe("Voice sent from the Voices page (?voice=<id>)", () => {
  it("asks to pick an agent while none is open", () => {
    mockParams = new URLSearchParams("voice=v9");
    withAgents([agent("a1", "Ria")]);
    render(<AgentStudioPage />);
    expect(screen.getByRole("status").textContent).toContain("Pick an agent to use Asha with");
  });

  it("hands the voice to the agent that gets opened", async () => {
    mockParams = new URLSearchParams("voice=v9");
    withAgents([agent("a1", "Ria")]);
    render(<AgentStudioPage />);
    fireEvent.click(screen.getByRole("button", { name: "Edit Ria" }));
    await screen.findByTestId("editor");
    expect(screen.queryByText(/Pick an agent to use/)).toBeNull();
  });
});
