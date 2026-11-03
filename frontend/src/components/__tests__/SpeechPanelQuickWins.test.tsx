import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { toast } from "sonner";
import { SpeechPanel } from "@/components/studio/SpeechPanel";
import { useGenerationStore } from "@/stores";
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
jest.mock("@/components/audio/WaveformPlayer", () => ({ WaveformPlayer: () => <div data-testid="player" /> }));

const blob = api.blob as jest.Mock;
const post = api.post as jest.Mock;

beforeAll(() => {
  Object.assign(URL, { createObjectURL: jest.fn(() => "blob:test"), revokeObjectURL: jest.fn() });
});

beforeEach(() => {
  jest.clearAllMocks();
  useGenerationStore.getState().reset();
  (useVoices as jest.Mock).mockReturnValue({ data: [], error: undefined, isLoading: false });
  (useKokoroVoices as jest.Mock).mockReturnValue({
    data: [{ id: "af_heart", name: "Heart", language_code: "a", gender: "female" }],
    isLoading: false,
  });
  (useIndicSpeak as jest.Mock).mockReturnValue({ data: { available: false, voices: [], max_chars: 400 } });
  (api.get as jest.Mock).mockResolvedValue({ samples: [] });
  post.mockResolvedValue({ language: "en", hinglish: false });
});

const script = () => screen.getByLabelText("Script") as HTMLTextAreaElement;
const type = (value: string) => fireEvent.change(script(), { target: { value } });
const generateButton = () => screen.getByRole("button", { name: /Generate speech/ });

/** A request that stays open until the caller's signal aborts it (like fetch does), or until `settle` is called. */
function pendingBlob() {
  const control: { settle: (b: Blob) => void; fail: (e: Error) => void } = { settle: () => undefined, fail: () => undefined };
  blob.mockImplementation(
    (_path: string, _body: unknown, opts: { signal: AbortSignal }) =>
      new Promise<Blob>((resolve, reject) => {
        control.settle = resolve;
        control.fail = reject;
        opts.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      })
  );
  return control;
}

describe("SpeechPanel over-limit text", () => {
  it("says how much of a long paste was kept, with the real limit and count, and clears it on the next edit", () => {
    render(<SpeechPanel />);
    type("a".repeat(3450));
    expect(script().value).toHaveLength(2000);
    const note = screen.getByTestId("script-trimmed");
    expect(note.getAttribute("role")).toBe("status");
    expect(note.textContent).toContain("Kept the first 2,000 of 3,450 characters");
    expect(script().getAttribute("aria-describedby")).toBe("script-trimmed");

    type("a".repeat(100));
    expect(screen.queryByTestId("script-trimmed")).toBeNull();
  });

  it("says nothing for text that fits", () => {
    render(<SpeechPanel />);
    type("a".repeat(2000));
    expect(script().value).toHaveLength(2000);
    expect(screen.queryByTestId("script-trimmed")).toBeNull();
  });

  it("also says so when typing past the limit (no silent maxLength)", () => {
    render(<SpeechPanel />);
    type("a".repeat(2000));
    type("a".repeat(2001));
    expect(script().value).toHaveLength(2000);
    expect(screen.getByTestId("script-trimmed").textContent).toContain("Kept the first 2,000 of 2,001 characters");
  });
});

describe("SpeechPanel generate: honest progress, Stop, errors, Make again", () => {
  it("shows a plain message, elapsed seconds and percentage, and keeps aria-live off the button", async () => {
    pendingBlob();
    render(<SpeechPanel />);
    type("Hello there");
    fireEvent.click(generateButton());

    expect(await screen.findByTestId("generate-hint")).toHaveTextContent("Making audio. Usually 10 to 40 seconds.");
    expect(screen.getByTestId("generate-elapsed").textContent).toMatch(/^\d+ s$/);
    expect(document.body.textContent).toMatch(/\d+%/);
    expect(document.querySelectorAll('button[aria-live]')).toHaveLength(0);
    expect(screen.getByTestId("generate-status")).toHaveTextContent("Making audio");
  });

  it("Stop aborts the request, restores the button and shows no error", async () => {
    pendingBlob();
    render(<SpeechPanel />);
    type("Hello there");
    fireEvent.click(generateButton());
    const stop = await screen.findByRole("button", { name: "Stop" });
    const signal = (blob.mock.calls[0][2] as { signal: AbortSignal }).signal;
    expect(signal.aborted).toBe(false);

    await act(async () => {
      fireEvent.click(stop);
    });

    expect(signal.aborted).toBe(true);
    await waitFor(() => expect(generateButton()).toBeEnabled());
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
    expect(screen.queryByTestId("generate-error")).toBeNull();
    expect(toast.error).not.toHaveBeenCalled();
    expect(screen.getByTestId("generate-status")).toBeEmptyDOMElement();
  });

  it("a failure shows an inline error with Retry (and still the toast); Retry tries again", async () => {
    const first = pendingBlob();
    render(<SpeechPanel />);
    type("Hello there");
    fireEvent.click(generateButton());
    await screen.findByRole("button", { name: "Stop" });
    await act(async () => first.fail(new Error("The voice engine is busy")));

    const error = await screen.findByTestId("generate-error");
    expect(error.textContent).toContain("The voice engine is busy");
    expect(toast.error).toHaveBeenCalledWith("The voice engine is busy");
    expect(screen.getByTestId("generate-status")).toHaveTextContent("Failed: The voice engine is busy");

    const second = pendingBlob();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    });
    expect(blob).toHaveBeenCalledTimes(2);
    expect(screen.queryByTestId("generate-error")).toBeNull();
    await act(async () => second.settle(new Blob(["x"], { type: "audio/wav" })));
    expect(await screen.findByRole("button", { name: "Play result" })).toBeTruthy();
    expect(screen.getByTestId("generate-status")).toHaveTextContent("Audio ready");
  });

  it("Make again regenerates with the same text and voice", async () => {
    blob.mockResolvedValue(new Blob(["x"], { type: "audio/wav" }));
    render(<SpeechPanel />);
    type("Hello there");
    fireEvent.click(generateButton());
    const again = await screen.findByRole("button", { name: "Make again" });
    expect(blob).toHaveBeenCalledTimes(1);

    await act(async () => {
      fireEvent.click(again);
    });
    await waitFor(() => expect(blob).toHaveBeenCalledTimes(2));
    expect(blob.mock.calls[1][1]).toEqual(blob.mock.calls[0][1]);
    expect(blob.mock.calls[1][0]).toBe("/api/kokoro/generate");
    expect(blob.mock.calls[1][1]).toEqual({ voice: "af_heart", text: "Hello there", speed: 1 });
  });
});

describe("SpeechPanel lang attributes", () => {
  async function detect() {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 700)); // detection is debounced by 600 ms
    });
  }

  it("sets lang on the script from the detected language", async () => {
    post.mockResolvedValue({ language: "hi", hinglish: false });
    render(<SpeechPanel />);
    expect(script().hasAttribute("lang")).toBe(false);
    type("नमस्ते, आप कैसे हैं");
    await detect();
    expect(script().getAttribute("lang")).toBe("hi");
  });

  it("uses hi-Latn for Hindi in Roman letters", async () => {
    post.mockResolvedValue({ language: "en", hinglish: true });
    render(<SpeechPanel />);
    type("kal subah aana");
    await detect();
    expect(script().getAttribute("lang")).toBe("hi-Latn");
  });

  it("sets lang on the script text shown with the finished audio", async () => {
    post.mockResolvedValue({ language: "mr", hinglish: false });
    blob.mockResolvedValue(new Blob(["x"], { type: "audio/wav" }));
    render(<SpeechPanel />);
    type("नमस्कार, तुमचे स्वागत आहे");
    await detect();
    fireEvent.click(generateButton());
    const shown = await screen.findByTestId("audio-text");
    expect(shown.textContent).toBe("नमस्कार, तुमचे स्वागत आहे");
    expect(shown.getAttribute("lang")).toBe("mr");
  });
});
