import { fireEvent, render, screen } from "@testing-library/react";
import StudioPage from "@/app/(app)/dashboard/studio/page";
import { useIndicSpeak, useKokoroVoices, useVoices } from "@/hooks/useApi";
import { api } from "@/lib/api";

jest.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams() }));
jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({
  api: { post: jest.fn(), get: jest.fn(), blob: jest.fn() },
  ApiError: class extends Error {},
}));
jest.mock("@/hooks/useApi", () => ({
  useVoices: jest.fn(),
  useKokoroVoices: jest.fn(),
  useIndicSpeak: jest.fn(),
  useModelStatus: jest.fn(() => ({ data: undefined })),
}));
jest.mock("@/components/audio/WaveformPlayer", () => ({ WaveformPlayer: () => null }));
jest.mock("@/components/studio/DialogueBuilder", () => ({ DialogueBuilder: () => <p>dialogue builder</p> }));

beforeEach(() => {
  jest.clearAllMocks();
  (useVoices as jest.Mock).mockReturnValue({ data: [], error: undefined, isLoading: false });
  (useKokoroVoices as jest.Mock).mockReturnValue({ data: [{ id: "af_heart", name: "Heart", language_code: "a" }], isLoading: false });
  (useIndicSpeak as jest.Mock).mockReturnValue({ data: { available: false, voices: [], max_chars: 400 } });
  (api.get as jest.Mock).mockResolvedValue({ samples: [] });
  (api.post as jest.Mock).mockResolvedValue({ language: "en", hinglish: false });
});

describe("Studio tabs", () => {
  it("keeps the typed script when the user opens Dialogue and comes back", () => {
    render(<StudioPage />);
    const script = screen.getByLabelText("Script") as HTMLTextAreaElement;
    fireEvent.change(script, { target: { value: "Welcome to the show" } });

    fireEvent.click(screen.getByRole("tab", { name: "Dialogue" }));
    expect(screen.getByText("dialogue builder")).toBeVisible();
    expect(document.getElementById("panel-speech")).not.toBeVisible();
    expect(document.getElementById("panel-speech")?.hasAttribute("hidden")).toBe(true);

    fireEvent.click(screen.getByRole("tab", { name: "Speech" }));
    expect(document.getElementById("panel-speech")?.hasAttribute("hidden")).toBe(false);
    expect(document.getElementById("panel-dialogue")?.hasAttribute("hidden")).toBe(true);
    // The very same field, with its text: nothing was unmounted.
    expect(screen.getByLabelText("Script")).toBe(script);
    expect(script.value).toBe("Welcome to the show");
  });

  it("labels each tabpanel by its tab and selects only the open one", () => {
    render(<StudioPage />);
    for (const id of ["speech", "dialogue"]) {
      const panel = document.getElementById(`panel-${id}`)!;
      expect(panel.getAttribute("role")).toBe("tabpanel");
      expect(panel.getAttribute("aria-labelledby")).toBe(`tab-${id}`);
      expect(document.getElementById(`tab-${id}`)?.getAttribute("aria-controls")).toBe(`panel-${id}`);
    }
    expect(screen.getByRole("tab", { name: "Speech" }).getAttribute("aria-selected")).toBe("true");
    fireEvent.click(screen.getByRole("tab", { name: "Dialogue" }));
    expect(screen.getByRole("tab", { name: "Dialogue" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("tab", { name: "Speech" }).getAttribute("aria-selected")).toBe("false");
  });

  it("still links to Document Q&A", () => {
    render(<StudioPage />);
    expect(screen.getByRole("link", { name: /Document Q&A/ }).getAttribute("href")).toBe("/dashboard/knowledge/try");
  });
});
