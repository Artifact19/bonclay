import { render, waitFor } from "@testing-library/react";
import { AgentEditor } from "@/components/agents/AgentEditor";
import { useAgents } from "@/hooks/useApi";

jest.mock("sonner", () => ({ toast: { error: jest.fn(), success: jest.fn() } }));
jest.mock("@/hooks/useApi", () => ({ useAgents: jest.fn() }));
jest.mock("@/components/agents/KnowledgePanel", () => ({ KnowledgePanel: () => <p>knowledge panel</p> }));

const agent = {
  id: "a1",
  name: "Ria",
  greeting: "",
  system_instructions: "",
  supported_languages: ["en"],
  voice_id: null,
  document_count: 0,
  created_at: "2026-10-01T00:00:00Z",
};
const props = { agent: agent as never, voices: [], onBack: jest.fn(), onDelete: jest.fn() };

beforeEach(() => (useAgents as jest.Mock).mockReturnValue({ mutate: jest.fn() }));

describe("AgentEditor focusKnowledge", () => {
  it("focuses the Knowledge panel when asked (the Add to knowledge deep link)", async () => {
    render(<AgentEditor {...props} focusKnowledge />);
    await waitFor(() => expect(document.activeElement?.id).toBe("agent-knowledge"));
  });

  it("leaves focus alone otherwise", () => {
    render(<AgentEditor {...props} />);
    expect(document.activeElement?.id).not.toBe("agent-knowledge");
  });
});
