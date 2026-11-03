import { render, screen, within } from "@testing-library/react";
import StudioHome from "@/app/(app)/dashboard/studio/home/page";
import { useStats, useVoices } from "@/hooks/useApi";

jest.mock("@/hooks/useApi", () => ({ useStats: jest.fn(), useVoices: jest.fn() }));
jest.mock("next/navigation", () => ({ usePathname: () => "/dashboard/studio/home", useRouter: () => ({ push: jest.fn() }) }));

const stats = useStats as jest.Mock;
const voices = useVoices as jest.Mock;
const noop = jest.fn();

const clip = { id: "g1", text: "Welcome to the show", voice_name: "Asha", duration: 12, audio_url: "/a.wav", created_at: new Date().toISOString() };

beforeEach(() => {
  stats.mockReturnValue({ data: { voices: 2, generations: 7, minutes: 3.5, recent: [clip] }, isLoading: false, error: undefined, mutate: noop });
  voices.mockReturnValue({
    data: [
      { id: "v1", name: "Asha", language: "hi", created_at: "2026-01-02T00:00:00Z" },
      { id: "v2", name: "Ravi", language: "en", created_at: "2026-01-01T00:00:00Z" },
    ],
    isLoading: false,
    error: undefined,
    mutate: noop,
  });
});

test("asks what you want to make and offers the four tasks with their destinations", () => {
  render(<StudioHome />);
  expect(screen.getByRole("heading", { level: 1, name: "What do you want to make?" })).toBeInTheDocument();
  const tasks = within(screen.getByRole("list", { name: "What do you want to make?" }));
  expect(tasks.getByRole("link", { name: "Turn text into speech" })).toHaveAttribute("href", "/dashboard/studio");
  expect(tasks.getByRole("link", { name: "Clone my voice" })).toHaveAttribute("href", "/dashboard/voices?new=1");
  expect(tasks.getByRole("link", { name: "Say it in several languages" })).toHaveAttribute("href", "/dashboard/say-it-once");
  expect(tasks.getByRole("link", { name: "Check if a clip is AI-made" })).toHaveAttribute("href", "/verify");
});

test("shows usage numbers, recent clips and your voices", () => {
  render(<StudioHome />);
  expect(screen.getByText("Audio clips made")).toBeInTheDocument();
  expect(screen.getByText("Minutes of audio")).toBeInTheDocument();
  expect(screen.getByText("Welcome to the show")).toBeInTheDocument();
  const strip = within(screen.getByRole("region", { name: "Your voices" }));
  expect(strip.getByText("Asha")).toBeInTheDocument();
  expect(strip.getByText("Hindi")).toBeInTheDocument();
  expect(strip.getByRole("link", { name: /Clone another voice/ })).toHaveAttribute("href", "/dashboard/voices?new=1");
});

test("has no agent content and no link to an agent page", () => {
  render(<StudioHome />);
  const hrefs = screen.getAllByRole("link").map((a) => a.getAttribute("href") ?? "");
  for (const agentPage of ["/dashboard/studio/agents", "/dashboard/calls", "/dashboard/approvals", "/dashboard/playground", "/dashboard/knowledge", "/dashboard/analytics", "/dashboard/eval"]) {
    expect(hrefs.some((h) => h === agentPage || h.startsWith(agentPage + "?"))).toBe(false);
  }
  expect(document.body.textContent).not.toMatch(/agent|phone|approval|booking/i);
});

test("friendly empty states when nothing has been made yet", () => {
  stats.mockReturnValue({ data: { voices: 0, generations: 0, minutes: 0, recent: [] }, isLoading: false, error: undefined, mutate: noop });
  voices.mockReturnValue({ data: [], isLoading: false, error: undefined, mutate: noop });
  render(<StudioHome />);
  expect(screen.getByText("No audio clips yet")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Open Studio" })).toHaveAttribute("href", "/dashboard/studio");
  expect(screen.getByText("No voices yet")).toBeInTheDocument();
});
