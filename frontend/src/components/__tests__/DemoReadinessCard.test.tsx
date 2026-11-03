import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { DemoReadinessCard } from "@/components/dashboard/DemoReadinessCard";
import { useWarmup } from "@/hooks/useApi";
import { api } from "@/lib/api";
import type { WarmupStatus } from "@/types/api";

jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn() } }));
jest.mock("@/lib/api", () => ({ api: { post: jest.fn() }, ApiError: class extends Error {} }));
jest.mock("@/hooks/useApi", () => ({ useWarmup: jest.fn() }));
const warm = useWarmup as jest.Mock;
const mutate = jest.fn();

const comp = (id: string, label: string, state: string, seconds: number | null, detail: string | null = null) =>
  ({ id, label, state, seconds, detail }) as WarmupStatus["components"][number];

describe("DemoReadinessCard", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("starts a warm-up from the idle state", async () => {
    warm.mockReturnValue({ data: { state: "idle", started_at: null, finished_at: null, components: [comp("llm", "Language model", "pending", null)] }, mutate });
    (api.post as jest.Mock).mockResolvedValue({ state: "running", started_at: 1, finished_at: null, components: [] });
    render(<DemoReadinessCard />);
    expect(screen.getByText("Make the first reply fast")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Get everything ready/ }));
    await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/warmup"));
  });

  it("shows the checklist with seconds while running", () => {
    warm.mockReturnValue({
      data: {
        state: "running", started_at: 1, finished_at: null,
        components: [comp("llm", "Language model", "ready", 3.8), comp("asr", "Speech recognition", "warming", null)],
      },
      mutate,
    });
    render(<DemoReadinessCard />);
    expect(screen.getByText("3.8 s")).toBeTruthy();
    expect(screen.getByText(/Speech recognition/).textContent).toContain("loading");
    expect((screen.getByRole("button", { name: /Get everything ready/ }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("says first replies will be fast once everything is ready", () => {
    warm.mockReturnValue({
      data: {
        state: "ready", started_at: 1, finished_at: 22,
        components: [comp("llm", "Language model", "ready", 3.8), comp("speaker", "Speaker check", "skipped", 0, "SpeechBrain is missing")],
      },
      mutate,
    });
    render(<DemoReadinessCard />);
    expect(screen.getByRole("status").textContent).toBe("Ready. The first replies will be fast.");
    expect(screen.getByText("SpeechBrain is missing")).toBeTruthy();
  });
});
