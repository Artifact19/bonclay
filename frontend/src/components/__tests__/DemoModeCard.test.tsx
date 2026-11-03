import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { DemoModeCard } from "@/components/dashboard/DemoModeCard";
import { useDemoStatus, useHealth } from "@/hooks/useApi";
import { api } from "@/lib/api";

jest.mock("swr", () => ({ useSWRConfig: () => ({ mutate: jest.fn() }) }));
jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn() } }));
jest.mock("@/lib/api", () => ({ api: { post: jest.fn() }, ApiError: class extends Error {} }));
jest.mock("@/hooks/useApi", () => ({
  AGENTS_KEY: "/api/agents",
  DEMO_STATUS_KEY: "/api/demo/status",
  useHealth: jest.fn(),
  useDemoStatus: jest.fn(),
}));
const health = useHealth as jest.Mock;
const demo = useDemoStatus as jest.Mock;
const mutate = jest.fn();

describe("DemoModeCard", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    health.mockReturnValue({ data: { status: "ok", supabase: { configured: false } } });
    demo.mockReturnValue({ data: { loaded: false, agents: [], evaluation_ready: false }, mutate });
  });

  it("renders nothing when Supabase is configured (cloud mode)", () => {
    health.mockReturnValue({ data: { status: "ok", supabase: { configured: true } } });
    const { container } = render(<DemoModeCard />);
    expect(container.firstChild).toBeNull();
  });

  it("loads the demo with one click", async () => {
    (api.post as jest.Mock).mockResolvedValue({ loaded: true, agents: ["a", "b"], evaluation: "running" });
    render(<DemoModeCard />);
    fireEvent.click(screen.getByRole("button", { name: /Load sample data/ }));
    await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/demo/load"));
  });

  it("links to Agent Studio and Evaluation Lab once loaded", () => {
    demo.mockReturnValue({ data: { loaded: true, agents: ["a", "b"], evaluation_ready: true }, mutate });
    render(<DemoModeCard />);
    expect(screen.getByRole("link", { name: "Agents" }).getAttribute("href")).toBe("/dashboard/studio/agents");
    expect(screen.getByRole("link", { name: "Safety check" }).getAttribute("href")).toBe("/dashboard/eval");
    expect(screen.getByRole("button", { name: /Remove sample data/ })).toBeTruthy();
  });
});
