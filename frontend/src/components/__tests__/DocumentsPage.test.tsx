import { fireEvent, render, screen } from "@testing-library/react";
import DocumentsPage from "@/app/(app)/dashboard/knowledge/page";
import { useAgents } from "@/hooks/useApi";

let mockParams = new URLSearchParams();
const replace = jest.fn();
jest.mock("next/navigation", () => ({
  useSearchParams: () => mockParams,
  usePathname: () => "/dashboard/knowledge",
  useRouter: () => ({ replace, push: jest.fn() }),
}));
jest.mock("@/hooks/useApi", () => ({ useAgents: jest.fn() }));
jest.mock("@/components/agents/KnowledgePanel", () => ({
  KnowledgePanel: ({ agentId }: { agentId: string }) => <p data-testid="panel">panel for {agentId}</p>,
}));

const mockAgents = useAgents as jest.Mock;
const agent = (id: string, name: string, created_at: string) => ({
  id,
  name,
  greeting: "",
  system_instructions: "",
  supported_languages: ["en"],
  voice_id: null,
  document_count: 0,
  created_at,
});
const clinic = agent("a1", "Clinic desk", "2026-10-01T00:00:00Z");
const fees = agent("a2", "Fee reminder", "2026-10-10T00:00:00Z");
const state = (over: object) =>
  mockAgents.mockReturnValue({ data: undefined, isLoading: false, error: undefined, mutate: jest.fn(), ...over });

beforeEach(() => {
  mockParams = new URLSearchParams();
  jest.clearAllMocks();
  localStorage.clear();
});

test("shows the heading, the description and a picker with the user's agents", () => {
  state({ data: [clinic, fees] });
  render(<DocumentsPage />);
  expect(screen.getByRole("heading", { level: 1, name: "Documents" })).toBeInTheDocument();
  expect(screen.getByText("The files your agent answers from.")).toBeInTheDocument();
  const picker = screen.getByRole("combobox", { name: "Agent" });
  expect(Array.from((picker as HTMLSelectElement).options).map((o) => o.text)).toEqual(["Fee reminder", "Clinic desk"]);
});

test("opens on the newest agent and hands its id to the documents panel", () => {
  state({ data: [clinic, fees] });
  render(<DocumentsPage />);
  expect(screen.getByRole("combobox", { name: "Agent" })).toHaveValue("a2");
  expect(screen.getByTestId("panel")).toHaveTextContent("panel for a2");
  expect(replace).toHaveBeenCalledWith("/dashboard/knowledge?agent=a2", { scroll: false });
});

test("?agent= preselects that agent without rewriting the address", () => {
  mockParams = new URLSearchParams("agent=a1");
  state({ data: [clinic, fees] });
  render(<DocumentsPage />);
  expect(screen.getByRole("combobox", { name: "Agent" })).toHaveValue("a1");
  expect(screen.getByTestId("panel")).toHaveTextContent("panel for a1");
  expect(replace).not.toHaveBeenCalled();
});

test("an unknown ?agent= falls back to the newest agent", () => {
  mockParams = new URLSearchParams("agent=gone");
  state({ data: [clinic, fees] });
  render(<DocumentsPage />);
  expect(screen.getByTestId("panel")).toHaveTextContent("panel for a2");
});

test("choosing another agent switches the panel, updates the address and is remembered", () => {
  state({ data: [clinic, fees] });
  render(<DocumentsPage />);
  fireEvent.change(screen.getByRole("combobox", { name: "Agent" }), { target: { value: "a1" } });
  expect(screen.getByTestId("panel")).toHaveTextContent("panel for a1");
  expect(replace).toHaveBeenLastCalledWith("/dashboard/knowledge?agent=a1", { scroll: false });
  expect(localStorage.getItem("bonclay.documents.agent")).toBe("a1");
});

test("the last agent used is reopened when the link names none", () => {
  localStorage.setItem("bonclay.documents.agent", "a1");
  state({ data: [clinic, fees] });
  render(<DocumentsPage />);
  expect(screen.getByTestId("panel")).toHaveTextContent("panel for a1");
});

test("the link wins over the remembered agent", () => {
  localStorage.setItem("bonclay.documents.agent", "a1");
  mockParams = new URLSearchParams("agent=a2");
  state({ data: [clinic, fees] });
  render(<DocumentsPage />);
  expect(screen.getByTestId("panel")).toHaveTextContent("panel for a2");
});

test("blocked storage does not break the page", () => {
  const get = jest.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
    throw new Error("blocked");
  });
  const set = jest.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("blocked");
  });
  state({ data: [clinic, fees] });
  render(<DocumentsPage />);
  fireEvent.change(screen.getByRole("combobox", { name: "Agent" }), { target: { value: "a1" } });
  expect(screen.getByTestId("panel")).toHaveTextContent("panel for a1");
  get.mockRestore();
  set.mockRestore();
});

test("the next-step card opens the Playground and the agent editor for the chosen agent", () => {
  mockParams = new URLSearchParams("agent=a1");
  state({ data: [clinic, fees] });
  render(<DocumentsPage />);
  expect(screen.getByRole("link", { name: /try it/i })).toHaveAttribute("href", "/dashboard/playground?agent=a1");
  expect(screen.getByRole("link", { name: /edit agent/i })).toHaveAttribute("href", "/dashboard/studio/agents?agent=a1&tab=basics");
});

test("links to the quick try page", () => {
  state({ data: [clinic] });
  render(<DocumentsPage />);
  expect(screen.getByRole("link", { name: "Just trying a PDF?" })).toHaveAttribute("href", "/dashboard/knowledge/try");
});

test("with no agents it asks you to create one and links to the agents page", () => {
  state({ data: [] });
  render(<DocumentsPage />);
  expect(screen.getByText("Create an agent first")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Go to Agents" })).toHaveAttribute("href", "/dashboard/studio/agents");
  expect(screen.queryByTestId("panel")).not.toBeInTheDocument();
  expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Just trying a PDF?" })).toBeInTheDocument();
});

test("shows a loading state, then an error with a retry", () => {
  state({ isLoading: true });
  const { unmount } = render(<DocumentsPage />);
  expect(document.querySelector("[aria-busy='true']")).toBeInTheDocument();
  expect(screen.queryByTestId("panel")).not.toBeInTheDocument();
  unmount();

  const mutate = jest.fn();
  state({ error: new Error("down"), mutate });
  render(<DocumentsPage />);
  expect(screen.getByRole("alert")).toHaveTextContent("Could not load your agents");
  fireEvent.click(screen.getByRole("button", { name: /try again|retry/i }));
  expect(mutate).toHaveBeenCalled();
});
