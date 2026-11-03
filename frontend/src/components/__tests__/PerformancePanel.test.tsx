import { render, screen } from "@testing-library/react";
import { PerformancePanel } from "@/components/dashboard/PerformancePanel";
import { usePerfTurns } from "@/hooks/useApi";

jest.mock("@/hooks/useApi", () => ({ usePerfTurns: jest.fn() }));
const mockHook = usePerfTurns as jest.Mock;

const turn = (id: string, total: number) => ({
  id,
  created_at: "2026-01-01T00:00:00Z",
  agent_id: "a",
  agent_name: "Ria",
  total_ms: total,
  timings: { asr_ms: 300, retrieval_ms: 50, llm_ms: total - 650, tts_ms: 300 },
  engine: "kokoro",
  llm_provider: "groq",
  language: "en",
  fallback: false,
});

describe("PerformancePanel", () => {
  it("shows the empty state with a Playground link", () => {
    mockHook.mockReturnValue({ data: { turns: [], count: 0, p50_ms: null, p95_ms: null, within_target_pct: null }, isLoading: false });
    render(<PerformancePanel />);
    expect(screen.getByText(/No conversations yet/)).toBeTruthy();
    expect(screen.getByRole("link", { name: /Playground/ }).getAttribute("href")).toBe("/dashboard/playground");
  });

  it("shows p50, p95, the share within target and one bar per turn", () => {
    mockHook.mockReturnValue({
      data: {
        turns: [turn("1", 1800), turn("2", 2400), turn("3", 3600)],
        count: 3,
        p50_ms: 2400,
        p95_ms: 3600,
        stage_p50: { asr_ms: 300, retrieval_ms: 50, llm_ms: 1750, tts_ms: 300 },
        target_ms: 3000,
        within_target_pct: 66.7,
      },
      isLoading: false,
    });
    render(<PerformancePanel agentId="a" />);
    expect(mockHook).toHaveBeenCalledWith(20, "a");
    expect(screen.getByText("Reply time")).toBeTruthy();
    expect(screen.getAllByText("2.4 s").length).toBeGreaterThan(0);
    expect(screen.getAllByText("3.6 s").length).toBeGreaterThan(0);
    expect(screen.getByText("66.7%")).toBeTruthy();
    expect(screen.getByLabelText("Recent turns, newest first").children).toHaveLength(3);
    expect(screen.getByText("Language model")).toBeTruthy();
  });

  it("shows first-audio p50 / p95 next to the full-reply numbers when turns were streamed", () => {
    mockHook.mockReturnValue({
      data: {
        turns: [{ ...turn("1", 3600), first_audio_ms: 1200 }, { ...turn("2", 2400), first_audio_ms: 900 }],
        count: 2,
        p50_ms: 2400,
        p95_ms: 3600,
        stage_p50: { asr_ms: 300, retrieval_ms: 50, llm_ms: 1750, tts_ms: 300 },
        target_ms: 3000,
        within_target_pct: 50,
        first_audio_count: 2,
        first_audio_p50_ms: 900,
        first_audio_p95_ms: 1200,
        first_audio_within_target_pct: 100,
      },
      isLoading: false,
    });
    render(<PerformancePanel agentId="a" />);
    expect(screen.getByText("First sound (usual)")).toBeTruthy();
    expect(screen.getByText("First sound (slowest)")).toBeTruthy();
    expect(screen.getByText("Full reply (usual)")).toBeTruthy();
    expect(screen.getByText("Full reply (slowest)")).toBeTruthy();
    expect(screen.getByText("0.9 s")).toBeTruthy();
    expect(screen.getByText("1.2 s")).toBeTruthy();
    expect(screen.getByText("100%")).toBeTruthy();
    expect(screen.getByText("50%")).toBeTruthy();
    expect(screen.getByLabelText(/3\.6 s total\. First sound 1\.2 s/)).toBeTruthy();
  });

  it("keeps the plain p50 / p95 labels when no turn was streamed", () => {
    mockHook.mockReturnValue({
      data: {
        turns: [turn("1", 1800)],
        count: 1,
        p50_ms: 1800,
        p95_ms: 1800,
        stage_p50: { asr_ms: 300, retrieval_ms: 50, llm_ms: 1150, tts_ms: 300 },
        target_ms: 3000,
        within_target_pct: 100,
        first_audio_count: 0,
        first_audio_p50_ms: null,
        first_audio_p95_ms: null,
        first_audio_within_target_pct: null,
      },
      isLoading: false,
    });
    render(<PerformancePanel />);
    expect(screen.getByText("Usual reply time")).toBeTruthy();
    expect(screen.queryByText("First sound (usual)")).toBeNull();
  });
});
