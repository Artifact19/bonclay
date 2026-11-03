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

const ATTRIBUTION = "Built with Indic-Speak from Bodhan AI / AI4Bharat.";
const status = (loaded: boolean) => ({ engines: [{ id: "indic_speak", loaded }] });

beforeEach(() => {
  jest.clearAllMocks();
  (useVoices as jest.Mock).mockReturnValue({ data: [], error: undefined, isLoading: false });
  (useKokoroVoices as jest.Mock).mockReturnValue({ data: [{ id: "af_heart", name: "Heart", language_code: "a", gender: "female" }], isLoading: false });
  (useIndicSpeak as jest.Mock).mockReturnValue({
    data: {
      available: true,
      reason: null,
      max_chars: 600,
      attribution: ATTRIBUTION,
      model_url: "https://huggingface.co/bodhan-ai/indic-speak",
      voices: [{ id: "Kavya", name: "Kavya", language: "Hindi", language_code: "hi", gender: "female", tier: "production" }],
    },
  });
  (api.post as jest.Mock).mockResolvedValue({ language: "hi", hinglish: false });
});

async function startIndicGeneration() {
  render(<SpeechPanel />);
  fireEvent.click(screen.getByText(/Advanced: choose the engine/));
  fireEvent.click(screen.getByRole("button", { name: /Indic-Speak/ }));
  fireEvent.change(screen.getByLabelText("Script"), { target: { value: "नमस्ते, कल सुबह आना" } });
  await act(async () => {
    await new Promise((r) => setTimeout(r, 700));
  });
  fireEvent.click(screen.getByRole("button", { name: /Generate speech/ }));
}

describe("SpeechPanel Indic-Speak cold start", () => {
  it("says the Indian-language voices are loading while the worker starts, then clears the note", async () => {
    let loaded = false;
    (api.get as jest.Mock).mockImplementation(async (path: string) => (path === "/api/models/status" ? status(loaded) : { samples: [] }));
    let finish: (b: Blob) => void = () => undefined;
    (api.blob as jest.Mock).mockReturnValue(new Promise<Blob>((r) => (finish = r)));

    await startIndicGeneration();
    const note = await screen.findByTestId("indic-loading");
    expect(note.textContent).toMatch(/Loading the Indian-language voices, about \d+ s/);

    loaded = true; // the status poll sees the worker up
    await waitFor(() => expect(screen.queryByTestId("indic-loading")).toBeNull(), { timeout: 4000 });

    await act(async () => finish(new Blob(["x"], { type: "audio/wav" })));
  });

  it("shows no loading note when the worker is already running", async () => {
    (api.get as jest.Mock).mockImplementation(async (path: string) => (path === "/api/models/status" ? status(true) : { samples: [] }));
    (api.blob as jest.Mock).mockReturnValue(new Promise<Blob>(() => undefined));
    await startIndicGeneration();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 100));
    });
    expect(screen.queryByTestId("indic-loading")).toBeNull();
  });

  it("shows the attribution wherever Indic-Speak is offered", async () => {
    (api.get as jest.Mock).mockResolvedValue({ samples: [] });
    render(<SpeechPanel />);
    fireEvent.click(screen.getByText(/Advanced: choose the engine/));
  fireEvent.click(screen.getByRole("button", { name: /Indic-Speak/ }));
    expect((await screen.findAllByText(new RegExp(ATTRIBUTION.replace(/[./]/g, "\\$&")))).length).toBeGreaterThan(0);
  });
});
