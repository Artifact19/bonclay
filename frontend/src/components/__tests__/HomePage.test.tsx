import { fireEvent, render, screen } from "@testing-library/react";
import { FirstRunChecklist } from "@/components/dashboard/FirstRunChecklist";
import { NeedsAttention } from "@/components/dashboard/NeedsAttention";
import { QuickActions } from "@/components/dashboard/QuickActions";
import { SystemStatus } from "@/components/dashboard/SystemStatus";
import { useHealth, useKnowledgeGaps, useOnboarding, useSetupHints, useWarmup } from "@/hooks/useApi";

jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: jest.fn(), replace: jest.fn(), prefetch: jest.fn() }),
  usePathname: () => "/dashboard",
}));

jest.mock("@/hooks/useApi", () => ({
  useOnboarding: jest.fn(),
  useKnowledgeGaps: jest.fn(),
  useWarmup: jest.fn(),
  useHealth: jest.fn(),
  useSetupHints: jest.fn(),
}));
// The heavy panels have their own tests; here only whether they are mounted matters.
jest.mock("@/components/dashboard/HealthStrip", () => ({ HealthStrip: () => <div>health-strip</div> }));
jest.mock("@/components/dashboard/SetupHints", () => ({ SetupHints: () => null }));
jest.mock("@/components/dashboard/StatCards", () => ({ StatCards: () => <div>stat-cards</div> }));
jest.mock("@/components/dashboard/PerformancePanel", () => ({ PerformancePanel: () => <div>perf-panel</div> }));
jest.mock("@/components/dashboard/EnginePanel", () => ({ EnginePanel: () => <div>engine-panel</div> }));
jest.mock("@/components/dashboard/DemoReadinessCard", () => ({ DemoReadinessCard: () => <div>readiness-card</div> }));
jest.mock("@/components/dashboard/DemoModeCard", () => ({ DemoModeCard: () => <div>sample-data-card</div> }));

const onboarding = useOnboarding as jest.Mock;
const gaps = useKnowledgeGaps as jest.Mock;
const warm = useWarmup as jest.Mock;
const health = useHealth as jest.Mock;
const hints = useSetupHints as jest.Mock;

const step = (id: string, title: string, done: boolean, minutes = 2) => ({
  id, title, done, minutes, why: `why ${id}`, cta: `cta ${id}`, href: `/dashboard/${id}`,
});
const SETUP_IDS = ["create_agent", "add_documents", "talk_to_it", "pick_voice", "connect_phone", "practice_call"];
const setup = (doneCount: number) => SETUP_IDS.map((id, i) => step(id, `Title ${id}`, i < doneCount));
const data = (over: object = {}) => ({
  steps: [], completed: 0, total: 5, setup: setup(2), agent_id: "a1", has_agent: true, phone_connected: false,
  approvals_waiting: 0, ...over,
});

beforeEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
  onboarding.mockReturnValue({ data: data(), isLoading: false });
  gaps.mockReturnValue({ data: { total_gap_turns: 0, clusters: [], total_turns: 0, window_days: 30 } });
  warm.mockReturnValue({ data: { state: "idle", components: [] } });
  health.mockReturnValue({ data: { status: "ok", llm: { active: "groq" }, supabase: { configured: false } } });
  hints.mockReturnValue({ data: { hints: [] } });
});

describe("QuickActions", () => {
  it("shows the four tasks with their destinations", () => {
    render(<QuickActions />);
    expect(screen.getByRole("link", { name: "Set up an agent" }).getAttribute("href")).toBe("/dashboard/studio/agents");
    expect(screen.getByRole("link", { name: "Give your agent a voice" }).getAttribute("href")).toBe("/dashboard/voices?new=1");
    expect(screen.getByRole("link", { name: "Connect my phone" }).getAttribute("href")).toBe(
      "/dashboard/studio/agents?agent=a1&tab=phone"
    );
    expect(screen.getByRole("link", { name: "See calls and approvals" }).getAttribute("href")).toBe("/dashboard/calls");
    expect(screen.getByRole("link", { name: "Approvals" }).getAttribute("href")).toBe("/dashboard/approvals");
  });

  it("opens the plain agents page for the phone card when there is no agent, and shows the waiting count", () => {
    onboarding.mockReturnValue({ data: data({ agent_id: null, has_agent: false, approvals_waiting: 2 }), isLoading: false });
    render(<QuickActions />);
    expect(screen.getByRole("link", { name: "Connect my phone" }).getAttribute("href")).toBe("/dashboard/studio/agents");
    expect(screen.getByRole("link", { name: "Approvals (2 waiting)" })).toBeTruthy();
  });
});

describe("FirstRunChecklist", () => {
  it("lists six steps with a button each, the next one first, and the time left", () => {
    render(<FirstRunChecklist />);
    expect(screen.getByText("2 of 6 done. About 8 minutes to go.")).toBeTruthy();
    expect(screen.getAllByRole("listitem")).toHaveLength(6);
    expect(screen.getByRole("link", { name: /cta talk_to_it/ }).getAttribute("href")).toBe("/dashboard/talk_to_it");
    expect(screen.queryByRole("link", { name: /cta create_agent/ })).toBeNull(); // done rows have no button
    expect(screen.getByText("Title create_agent").textContent).toContain("(done)");
    expect(screen.queryByRole("button", { name: "Dismiss" })).toBeNull(); // not dismissible until done
  });

  it("is dismissible only once every step is done, and stays dismissed", () => {
    onboarding.mockReturnValue({ data: data({ setup: setup(6) }), isLoading: false });
    const { container, rerender } = render(<FirstRunChecklist />);
    expect(screen.getByText("You are all set")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(container.firstChild).toBeNull();
    expect(localStorage.getItem("bonclay_setup_done_dismissed")).toBe("1");
    rerender(<FirstRunChecklist />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing against an older server that sends no setup list", () => {
    onboarding.mockReturnValue({ data: { steps: [], completed: 0, total: 5 }, isLoading: false });
    const { container } = render(<FirstRunChecklist />);
    expect(container.firstChild).toBeNull();
  });
});

describe("NeedsAttention", () => {
  it("says so when nothing needs attention", () => {
    onboarding.mockReturnValue({ data: data({ phone_connected: true }), isLoading: false });
    render(<NeedsAttention />);
    expect(screen.getByText("Nothing needs your attention right now.")).toBeTruthy();
  });

  it("lists approvals, unanswered questions and the phone line from real counts", () => {
    onboarding.mockReturnValue({ data: data({ approvals_waiting: 3 }), isLoading: false });
    gaps.mockReturnValue({ data: { total_gap_turns: 1, clusters: [], total_turns: 5, window_days: 30 } });
    render(<NeedsAttention />);
    expect(screen.getByText("3 meeting requests are waiting for your decision")).toBeTruthy();
    expect(screen.getByRole("link", { name: /Review/ }).getAttribute("href")).toBe("/dashboard/approvals");
    expect(screen.getByText("Your agent could not answer 1 question in the last 30 days")).toBeTruthy();
    expect(screen.getByRole("link", { name: /See them/ }).getAttribute("href")).toBe("/dashboard/analytics");
    expect(screen.getByText(/phone line is not connected/)).toBeTruthy();
    expect(screen.getByRole("link", { name: /Connect phone/ }).getAttribute("href")).toBe(
      "/dashboard/studio/agents?agent=a1&tab=phone"
    );
  });

  it("does not nag about the phone before an agent exists, and reports models still loading", () => {
    onboarding.mockReturnValue({ data: data({ has_agent: false, agent_id: null }), isLoading: false });
    warm.mockReturnValue({ data: { state: "running", components: [] } });
    render(<NeedsAttention />);
    expect(screen.queryByText(/phone line is not connected/)).toBeNull();
    expect(screen.getByText(/still loading/)).toBeTruthy();
  });
});

describe("SystemStatus", () => {
  it("is closed by default when everything works, and opens on click", () => {
    render(<SystemStatus />);
    const toggle = screen.getByRole("button", { name: /System status/ });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText("perf-panel")).toBeNull();
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    for (const t of ["health-strip", "stat-cards", "perf-panel", "engine-panel", "readiness-card", "sample-data-card"]) {
      expect(screen.getByText(t)).toBeTruthy();
    }
  });

  it("opens itself and names the problem when something is wrong", () => {
    health.mockReturnValue({ data: { status: "ok", llm: { active: null }, supabase: { configured: false } } });
    render(<SystemStatus />);
    expect(screen.getByRole("button", { name: /System status/ }).getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByRole("button", { name: /No language model is set up/ })).toBeTruthy();
    expect(screen.getByText("engine-panel")).toBeTruthy();
  });

  it("stays closed if the person closed it, even while a problem remains", () => {
    health.mockReturnValue({ data: { status: "ok", llm: { active: null }, supabase: { configured: false } } });
    render(<SystemStatus />);
    fireEvent.click(screen.getByRole("button", { name: /System status/ }));
    expect(screen.queryByText("engine-panel")).toBeNull();
  });
});
