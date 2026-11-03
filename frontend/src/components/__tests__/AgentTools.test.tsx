import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { AgentEditor } from "@/components/agents/AgentEditor";
import { ToolCallChips } from "@/components/dashboard/ToolCallChips";
import { api } from "@/lib/api";
import type { Agent } from "@/types/api";

jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({ api: { put: jest.fn().mockResolvedValue({}), get: jest.fn().mockRejectedValue(new Error("offline")) } }));
jest.mock("@/hooks/useApi", () => ({ useAgents: () => ({ mutate: jest.fn().mockResolvedValue(undefined) }) }));
jest.mock("@/components/agents/KnowledgePanel", () => ({ KnowledgePanel: () => null }));

const agent: Agent = {
  id: "a1",
  name: "Reception",
  greeting: "Hello",
  system_instructions: "Be kind.",
  supported_languages: ["en"],
  voice_id: null,
  disclose_ai: true,
  tools: [],
  created_at: "2026-01-01T00:00:00Z",
  document_count: 0,
};

describe("agent tools setting", () => {
  it("shows the Appointments checkbox, reflecting the agent, and saves the chosen tools", async () => {
    render(<AgentEditor agent={agent} voices={[]} tab="phone" onBack={() => undefined} onDelete={() => undefined} />);
    const box = screen.getByRole("checkbox", { name: /Let the agent book appointments/ });
    expect(box).not.toBeChecked();
    expect(screen.getByRole("button", { name: /Save agent/ })).toBeDisabled();

    fireEvent.click(box);
    expect(box).toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: /Save agent/ }));
    await waitFor(() => expect(api.put).toHaveBeenCalled());
    expect((api.put as jest.Mock).mock.calls[0][0]).toBe("/api/agents/a1");
    expect((api.put as jest.Mock).mock.calls[0][1]).toMatchObject({ tools: ["appointments"] });
  });

  it("starts checked when the agent already has the tool", () => {
    render(<AgentEditor agent={{ ...agent, tools: ["appointments"] }} voices={[]} tab="phone" onBack={() => undefined} onDelete={() => undefined} />);
    expect(screen.getByRole("checkbox", { name: /Let the agent book appointments/ })).toBeChecked();
  });
});

describe("ToolCallChips", () => {
  it("says which tool was used and what it did", () => {
    render(
      <ToolCallChips
        calls={[
          {
            name: "book_appointment",
            arguments: { name: "Priya" },
            ok: true,
            result: "booking A12 with Dr. Meera Kulkarni on Monday 2026-10-19 10:30",
          },
        ]}
      />
    );
    const chip = screen.getByTestId("tool-call-chip");
    expect(chip).toHaveTextContent("Used tool: book_appointment → booking A12 with Dr. Meera Kulkarni on Monday 2026-10-19 10:30");
  });

  it("marks a failed call as failed, and renders nothing without calls", () => {
    const { container, rerender } = render(
      <ToolCallChips calls={[{ name: "cancel_appointment", arguments: {}, ok: false, result: "failed: not confirmed" }]} />
    );
    expect(screen.getByTestId("tool-call-chip")).toHaveTextContent("Tool failed: cancel_appointment → failed: not confirmed");
    rerender(<ToolCallChips calls={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});
