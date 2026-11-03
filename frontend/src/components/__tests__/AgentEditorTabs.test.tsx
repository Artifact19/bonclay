import { fireEvent, render as rtlRender, screen, waitFor } from "@testing-library/react";
import type { ReactElement } from "react";
import { SWRConfig } from "swr";
import { AGENT_TABS, AgentEditor, parseTabParam } from "@/components/agents/AgentEditor";
import { api } from "@/lib/api";
import type { Agent } from "@/types/api";

jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({ api: { put: jest.fn().mockResolvedValue({}), get: jest.fn() } }));
jest.mock("@/hooks/useApi", () => ({ useAgents: () => ({ mutate: jest.fn().mockResolvedValue(undefined) }) }));
jest.mock("@/components/agents/KnowledgePanel", () => ({ KnowledgePanel: () => <p>documents panel</p> }));
jest.mock("@/components/agents/WebCallPanel", () => ({ WebCallPanel: () => <p>web call panel</p> }));
jest.mock("@/components/agents/EmbedPanel", () => ({ EmbedPanel: () => <p>website panel</p> }));
jest.mock("@/components/agents/WhatsAppPanel", () => ({ WhatsAppPanel: () => <p>whatsapp panel</p> }));
jest.mock("@/components/agents/IntegrationsPanel", () => ({ IntegrationsPanel: () => <p>integrations panel</p> }));

const agent: Agent = {
  id: "a1",
  name: "Reception",
  greeting: "Namaste",
  system_instructions: "Be kind.",
  supported_languages: ["en"],
  voice_id: null,
  disclose_ai: true,
  tools: [],
  created_at: "2026-01-01T00:00:00Z",
  document_count: 0,
};
// A fresh SWR cache per render, so the status lookups of one test never leak into the next.
const wrap = (ui: ReactElement) => <SWRConfig value={{ provider: () => new Map() }}>{ui}</SWRConfig>;
const render = (ui: ReactElement) => {
  const r = rtlRender(wrap(ui));
  return { ...r, rerender: (next: ReactElement) => r.rerender(wrap(next)) };
};
const noop = () => undefined;
const props = { agent, voices: [], onBack: noop, onDelete: noop };

beforeEach(() => {
  jest.clearAllMocks();
  (api.get as jest.Mock).mockRejectedValue(new Error("offline"));
});

describe("parseTabParam", () => {
  it("accepts the four tabs", () => {
    for (const t of AGENT_TABS) expect(parseTabParam(t.id)).toEqual({ tab: t.id, focusKnowledge: false });
  });
  it("keeps the knowledge link working: Basics, scrolled to Documents. No tab opens Basics without scrolling", () => {
    expect(parseTabParam("knowledge")).toEqual({ tab: "basics", focusKnowledge: true });
    expect(parseTabParam(null)).toEqual({ tab: "basics", focusKnowledge: false });
  });
  it("falls back to Basics for anything else", () => {
    expect(parseTabParam("billing")).toEqual({ tab: "basics", focusKnowledge: false });
  });
});

describe("AgentEditor tabs", () => {
  it("opens on Basics with name, greeting, voice and documents, and nothing from the other tabs", () => {
    render(<AgentEditor {...props} />);
    expect(screen.getAllByRole("tab").map((t) => t.textContent)).toEqual(["Basics", "Phone and meetings", "Channels", "Advanced"]);
    expect(screen.getByRole("tab", { name: "Basics" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("Name")).toHaveValue("Reception");
    expect(screen.getByLabelText("Greeting")).toBeInTheDocument();
    expect(screen.getByLabelText("Voice")).toBeInTheDocument();
    expect(screen.getByText("documents panel")).toBeInTheDocument();
    expect(screen.queryByLabelText("How your agent behaves")).toBeNull();
    expect(screen.queryByText("website panel")).toBeNull();
  });

  it("switches tabs by click and keeps unsaved edits across tabs", () => {
    const onTabChange = jest.fn();
    render(<AgentEditor {...props} onTabChange={onTabChange} />);
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Front desk" } });
    fireEvent.click(screen.getByRole("tab", { name: "Advanced" }));
    expect(onTabChange).toHaveBeenCalledWith("advanced");
    expect(screen.getByLabelText("How your agent behaves")).toHaveValue("Be kind.");
    expect(screen.getByText("integrations panel")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Basics" }));
    expect(screen.getByLabelText("Name")).toHaveValue("Front desk");
  });

  it("shows the tab it is told to (deep link) and puts Channels' web call link, website and WhatsApp there", () => {
    const { rerender } = render(<AgentEditor {...props} tab="channels" />);
    expect(screen.getByRole("tab", { name: "Channels" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("web call panel")).toBeInTheDocument();
    expect(screen.getByText("website panel")).toBeInTheDocument();
    expect(screen.getByText("whatsapp panel")).toBeInTheDocument();
    rerender(<AgentEditor {...props} tab="phone" />);
    expect(screen.getByLabelText(/Let the agent book appointments/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Approvals/ })).toHaveAttribute("href", "/dashboard/approvals");
  });

  it("moves between tabs with the arrow keys", () => {
    render(<AgentEditor {...props} />);
    fireEvent.keyDown(screen.getByRole("tab", { name: "Basics" }), { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Phone and meetings" })).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(screen.getByRole("tab", { name: "Phone and meetings" }), { key: "End" });
    expect(screen.getByRole("tab", { name: "Advanced" })).toHaveAttribute("aria-selected", "true");
  });

  it("still scrolls to and focuses Documents on Basics for the Add-to-knowledge link", async () => {
    render(<AgentEditor {...props} tab="basics" focusKnowledge />);
    await waitFor(() => expect(document.activeElement?.id).toBe("agent-knowledge"));
  });
});

describe("AgentEditor header", () => {
  it("has Talk (to the Playground with this agent), and Practice call / Phone someone when given", () => {
    const onPractice = jest.fn();
    const onPhone = jest.fn();
    render(<AgentEditor {...props} onPractice={onPractice} onPhone={onPhone} />);
    expect(screen.getByRole("link", { name: "Talk" })).toHaveAttribute("href", "/dashboard/playground?agent=a1");
    fireEvent.click(screen.getByRole("button", { name: "Practice call" }));
    fireEvent.click(screen.getByRole("button", { name: "Phone someone" }));
    expect(onPractice).toHaveBeenCalledWith(agent);
    expect(onPhone).toHaveBeenCalledWith(agent);
  });

  it("keeps Delete in the ... menu", () => {
    const onDelete = jest.fn();
    render(<AgentEditor {...props} onDelete={onDelete} />);
    expect(screen.queryByRole("button", { name: /Delete/ })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "More actions" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Delete agent" }));
    expect(onDelete).toHaveBeenCalledWith(agent);
  });

  it("counts only real checks: instructions, documents, voice, talked to, phone", async () => {
    (api.get as jest.Mock).mockImplementation((path: string) =>
      Promise.resolve(path.includes("/analytics/agents/") ? { total_turns: 3 } : { configured: true })
    );
    render(<AgentEditor {...props} agent={{ ...agent, document_count: 2, voice_id: "v1" }} voices={[{ id: "v1", name: "Asha" } as never]} />);
    // instructions + documents + voice are known at once; talked-to and phone arrive from the server
    await screen.findByText("5 of 5");
    expect(screen.getByText("Phone connected")).toBeInTheDocument();
    expect(screen.getByText("Ready")).toBeInTheDocument();
  });

  it("starts at 1 of 5 for a bare agent, and says what is missing", async () => {
    render(<AgentEditor {...props} />);
    expect(screen.getByText("1 of 5")).toBeInTheDocument();
    expect(screen.getByText("Needs documents")).toBeInTheDocument();
    expect(screen.getByText(/Add your documents/)).toBeInTheDocument();
    expect(screen.getByText(/Connect your phone line/)).toBeInTheDocument();
  });
});

describe("AgentEditor saving", () => {
  it("says why Save is disabled: nothing changed, then no name", () => {
    render(<AgentEditor {...props} />);
    const save = screen.getByRole("button", { name: /Save agent/ });
    expect(save).toBeDisabled();
    expect(screen.getAllByText("No changes to save yet").length).toBeGreaterThan(0);
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "  " } });
    expect(save).toBeDisabled();
    expect(screen.getAllByText("Add a name first").length).toBeGreaterThan(0);
  });

  it("picks languages by name and saves the same codes as before", async () => {
    render(<AgentEditor {...props} tab="advanced" />);
    expect(screen.getByRole("button", { name: "English" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "Hindi" }));
    expect(screen.getByRole("button", { name: "Hindi" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: /Save agent/ }));
    await waitFor(() => expect(api.put).toHaveBeenCalled());
    expect((api.put as jest.Mock).mock.calls[0][1]).toMatchObject({ supported_languages: ["en", "hi"] });
  });

  it("keeps a language code it does not know instead of dropping it", () => {
    render(<AgentEditor {...props} agent={{ ...agent, supported_languages: ["en", "xx"] }} tab="advanced" />);
    expect(screen.getByRole("button", { name: "XX" })).toHaveAttribute("aria-pressed", "true");
  });
});

describe("AgentEditor voice from the Voices page", () => {
  it("preselects the voice as an unsaved change and saves it", async () => {
    render(<AgentEditor {...props} voices={[{ id: "v9", name: "Asha" } as never]} presetVoiceId="v9" />);
    expect(screen.getByLabelText("Voice")).toHaveValue("v9");
    expect(screen.getByText("Voice chosen from the Voices page. Save to use it.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Save agent/ }));
    await waitFor(() => expect(api.put).toHaveBeenCalled());
    expect((api.put as jest.Mock).mock.calls[0][1]).toMatchObject({ voice_id: "v9" });
  });
});
