import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SpeechPanel } from "@/components/studio/SpeechPanel";
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

const post = api.post as jest.Mock;

const HINGLISH = "kal subah aana";
const DEVANAGARI = "कल सुबह आना";

beforeEach(() => {
  jest.clearAllMocks();
  (useVoices as jest.Mock).mockReturnValue({ data: [], error: undefined, isLoading: false });
  (useKokoroVoices as jest.Mock).mockReturnValue({
    data: [
      { id: "af_heart", name: "Heart", language_code: "a", gender: "female" },
      { id: "hf_alpha", name: "Alpha", language_code: "h", gender: "female" },
    ],
    isLoading: false,
  });
  (useIndicSpeak as jest.Mock).mockReturnValue({ data: { available: false, voices: [], max_chars: 400 } });
  (api.get as jest.Mock).mockResolvedValue({ samples: [] });
  post.mockImplementation(async (path: string, body: { text: string }) => {
    if (path === "/api/indic/detect") {
      const roman = /^[\x00-\x7f]+$/.test(body.text);
      return { language: roman ? "en" : "hi", hinglish: roman && body.text.includes("kal") };
    }
    if (path === "/api/indic/transliterate") return { text: DEVANAGARI, method: "llm" };
    throw new Error(`unexpected ${path}`);
  });
});

async function typeScript(value: string) {
  const ta = screen.getByLabelText("Script") as HTMLTextAreaElement;
  fireEvent.change(ta, { target: { value } });
  await act(async () => {
    await new Promise((r) => setTimeout(r, 700)); // the panel debounces language detection by 600 ms
  });
  return ta;
}

describe("SpeechPanel Hinglish suggestion", () => {
  it("suggests converting Roman Hindi, converts it, and can undo", async () => {
    render(<SpeechPanel />);
    const ta = await typeScript(HINGLISH);
    expect(screen.getByTestId("hinglish-suggestion").textContent).toContain("Looks like Hinglish");

    fireEvent.click(screen.getByRole("button", { name: "Convert to Devanagari" }));
    await waitFor(() => expect(ta.value).toBe(DEVANAGARI));
    expect(post).toHaveBeenCalledWith("/api/indic/transliterate", { text: HINGLISH, target: "devanagari" }, { retries: 0 });
    expect(screen.queryByTestId("hinglish-suggestion")).toBeNull();
    // Hindi detected right away: the Hindi Kokoro voice is picked by the panel's existing routing
    expect(await screen.findByText("Hindi detected")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(ta.value).toBe(HINGLISH);
  });

  it("does not suggest anything for English or Devanagari text", async () => {
    render(<SpeechPanel />);
    await typeScript("I will call you tomorrow");
    expect(screen.queryByTestId("hinglish-suggestion")).toBeNull();
    await typeScript(DEVANAGARI);
    expect(screen.queryByTestId("hinglish-suggestion")).toBeNull();
  });
});
