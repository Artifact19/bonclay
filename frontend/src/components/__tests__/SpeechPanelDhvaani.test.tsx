import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SpeechPanel } from "@/components/studio/SpeechPanel";
import { useIndicSpeak, useKokoroVoices, useModelStatus, useVoices } from "@/hooks/useApi";
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
  useModelStatus: jest.fn(),
}));
jest.mock("@/components/audio/WaveformPlayer", () => ({ WaveformPlayer: () => <div data-testid="player" /> }));

const CREDIT = "Voice cloning in Indian languages by DhVaani from ARTPARK-IISc.";
const LABEL = "Your voice (DhVaani · ARTPARK-IISc)";
const MR = "नमस्कार, तुमची भेट उद्या आहे.";

const engines = (dhvaani: Record<string, unknown> | null) => ({
  data: { device: "mps", dtype: "float32", engines: [{ id: "kokoro", name: "Kokoro", type: "preset", languages: [], loaded: false }, ...(dhvaani ? [dhvaani] : [])] },
});
const DHVAANI_READY = { id: "dhvaani", name: "DhVaani-0.5 (ARTPARK-IISc)", type: "clone", languages: ["mr"], loaded: false, available: true };
const DHVAANI_MISSING = { ...DHVAANI_READY, available: false, unavailable_reason: "DhVaani is not set up", setup: "./bonclay setup --dhvaani" };

beforeEach(() => {
  jest.clearAllMocks();
  global.URL.createObjectURL = jest.fn(() => "blob:x");
  global.URL.revokeObjectURL = jest.fn();
  (useVoices as jest.Mock).mockReturnValue({ data: [{ id: "v1", name: "My voice" }], error: undefined, isLoading: false });
  (useKokoroVoices as jest.Mock).mockReturnValue({ data: [{ id: "af_heart", name: "Heart", language_code: "a", gender: "female" }], isLoading: false });
  (useIndicSpeak as jest.Mock).mockReturnValue({ data: { available: false, voices: [], max_chars: 400 } });
  (api.get as jest.Mock).mockResolvedValue({ samples: [] });
  (api.post as jest.Mock).mockResolvedValue({ language: "mr", hinglish: false });
});

/** The exact engines sit under "Advanced". */
function openAdvanced() {
  fireEvent.click(screen.getByText(/Advanced: choose the engine/));
}

async function typeScript(value: string) {
  fireEvent.change(screen.getByLabelText("Script"), { target: { value } });
  await act(async () => {
    await new Promise((r) => setTimeout(r, 700)); // language detection is debounced by 600 ms
  });
}

describe("SpeechPanel: DhVaani, your voice in Indian languages", () => {
  it("lists DhVaani as 'Your voice (DhVaani · ARTPARK-IISc)' only when it is set up", () => {
    (useModelStatus as jest.Mock).mockReturnValue(engines(DHVAANI_READY));
    const { unmount } = render(<SpeechPanel />);
    openAdvanced();
    expect(screen.getByRole("button", { name: new RegExp(LABEL.replace(/[()·]/g, "\\$&")) })).toBeTruthy();
    unmount();

    (useModelStatus as jest.Mock).mockReturnValue(engines(DHVAANI_MISSING));
    render(<SpeechPanel />);
    openAdvanced();
    expect(screen.queryByRole("button", { name: /DhVaani/ })).toBeNull();
    expect(screen.queryByTestId("dhvaani-credit")).toBeNull();
  });

  it("shows the credit line next to the voice picker and sends the saved voice with the language", async () => {
    (useModelStatus as jest.Mock).mockReturnValue(engines(DHVAANI_READY));
    (api.blob as jest.Mock).mockResolvedValue(new Blob(["x"], { type: "audio/wav" }));
    render(<SpeechPanel />);
    fireEvent.click(screen.getByRole("button", { name: /^My voice/ }));
    expect(screen.getAllByTestId("dhvaani-credit")[0].textContent).toBe(CREDIT);
    await typeScript(MR);
    expect(screen.getByText(/Marathi is spoken in your voice by DhVaani/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Generate speech/ }));
    await waitFor(() => expect(api.blob).toHaveBeenCalled());
    const [path, body] = (api.blob as jest.Mock).mock.calls[0];
    expect(path).toBe("/api/voices/generate");
    expect((body as FormData).get("voice_id")).toBe("v1");
    expect((body as FormData).get("language")).toBe("mr");
    // the credit goes with the audio that was made
    await waitFor(() => expect(screen.getByTestId("player")).toBeTruthy());
    expect(screen.getAllByTestId("dhvaani-credit").length).toBeGreaterThanOrEqual(2);
  });

  it("does not credit DhVaani for audio it did not make", async () => {
    (useModelStatus as jest.Mock).mockReturnValue(engines(DHVAANI_READY));
    (api.post as jest.Mock).mockResolvedValue({ language: "en", hinglish: false });
    (api.blob as jest.Mock).mockResolvedValue(new Blob(["x"], { type: "audio/wav" }));
    render(<SpeechPanel />);
    await typeScript("Your appointment is confirmed.");
    fireEvent.click(screen.getByRole("button", { name: /Generate speech/ })); // Kokoro, English
    await waitFor(() => expect(screen.getByTestId("player")).toBeTruthy());
    expect(screen.queryByTestId("dhvaani-credit")).toBeNull();
    const output = screen.getByRole("region", { name: "Output" });
    expect(within(output).queryByText(CREDIT)).toBeNull();
  });

  it("offers two plain choices first and names the Indian languages My voice covers", () => {
    (useModelStatus as jest.Mock).mockReturnValue(
      engines({ ...DHVAANI_READY, languages: ["hi", "mr", "ta", "te", "bn"] })
    );
    render(<SpeechPanel />);
    const standard = screen.getByRole("button", { name: /^Standard voices/ });
    const mine = screen.getByRole("button", { name: /^My voice/ });
    expect(standard.getAttribute("aria-pressed")).toBe("true");
    expect(mine.textContent).toMatch(/English, plus Hindi, Marathi, Tamil and 2 more/);
    // no engine names in the plain choices
    expect(standard.textContent + mine.textContent).not.toMatch(/Kokoro|Indic-Speak|DhVaani/);
    fireEvent.click(mine);
    expect(mine.getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(standard);
    expect(standard.getAttribute("aria-pressed")).toBe("true");
  });

  it("keeps every engine selectable under Advanced", () => {
    (useModelStatus as jest.Mock).mockReturnValue(engines(DHVAANI_READY));
    (useIndicSpeak as jest.Mock).mockReturnValue({ data: { available: true, voices: [], max_chars: 400 } });
    render(<SpeechPanel />);
    openAdvanced();
    for (const name of [/Kokoro/, /Indic-Speak/, /Cloned voice/, /DhVaani/]) {
      expect(screen.getByRole("button", { name })).toBeTruthy();
    }
    fireEvent.click(screen.getByRole("button", { name: /Cloned voice/ }));
    expect(screen.getByRole("button", { name: /^My voice/ }).getAttribute("aria-pressed")).toBe("true");
  });

  it("says why Generate is off", async () => {
    (useModelStatus as jest.Mock).mockReturnValue(engines(null));
    render(<SpeechPanel />);
    expect((screen.getByRole("button", { name: /Generate speech/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByTestId("generate-reason").textContent).toBe("Type some text first.");
    fireEvent.change(screen.getByLabelText("Script"), { target: { value: "Hello there" } });
    expect(screen.queryByTestId("generate-reason")).toBeNull();
  });

  it("says to clone a voice first when My voice has none to use", () => {
    (useModelStatus as jest.Mock).mockReturnValue(engines(null));
    (useVoices as jest.Mock).mockReturnValue({ data: [], error: undefined, isLoading: false });
    render(<SpeechPanel />);
    fireEvent.click(screen.getByRole("button", { name: /^My voice/ }));
    fireEvent.change(screen.getByLabelText("Script"), { target: { value: "Hello there" } });
    expect(screen.getByTestId("generate-reason").textContent).toBe("Clone a voice first on the Voices page.");
  });
});
