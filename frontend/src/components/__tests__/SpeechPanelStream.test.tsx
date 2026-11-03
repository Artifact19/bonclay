import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SpeechPanel } from "@/components/studio/SpeechPanel";
import { useIndicSpeak, useKokoroVoices, useModelStatus, useVoices } from "@/hooks/useApi";
import { api } from "@/lib/api";

jest.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams() }));
jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({
  API_URL: "http://api.test",
  api: { post: jest.fn(), get: jest.fn(), blob: jest.fn(), sse: jest.fn() },
  ApiError: class extends Error {},
}));
jest.mock("@/hooks/useApi", () => ({
  useVoices: jest.fn(),
  useKokoroVoices: jest.fn(),
  useIndicSpeak: jest.fn(),
  useModelStatus: jest.fn(),
}));
const players: { added: string[]; stopped: boolean; closed: boolean }[] = [];
jest.mock("@/lib/audioQueue", () => ({
  AudioQueue: class {
    added: string[] = [];
    stopped = false;
    closed = false;
    constructor() {
      players.push(this);
    }
    add(u: string) {
      this.added.push(u);
    }
    close() {
      this.closed = true;
    }
    stop() {
      this.stopped = true;
    }
  },
}));
jest.mock("@/components/audio/WaveformPlayer", () => ({
  WaveformPlayer: ({ autoPlay }: { autoPlay?: boolean }) => <div data-testid="player" data-autoplay={String(!!autoPlay)} />,
}));

type OnEvent = (event: string, data: unknown) => void;
const sentence = (index: number, total = 3) => ({ index, total, text: `Sentence ${index + 1}.`, audio_url: `/clip/${index}.wav` });
const DONE = { id: "g1", audio_url: "/api/generations/g1/audio", receipt_id: "r-123", duration_s: 1.2, engine: "qwen3_0_6b" };

beforeEach(() => {
  jest.clearAllMocks();
  players.length = 0;
  global.URL.createObjectURL = jest.fn(() => "blob:x");
  global.URL.revokeObjectURL = jest.fn();
  (useVoices as jest.Mock).mockReturnValue({ data: [{ id: "v1", name: "My voice" }], error: undefined, isLoading: false });
  (useKokoroVoices as jest.Mock).mockReturnValue({ data: [{ id: "af_heart", name: "Heart", language_code: "a", gender: "female" }], isLoading: false });
  (useIndicSpeak as jest.Mock).mockReturnValue({ data: { available: false, voices: [], max_chars: 400 } });
  (useModelStatus as jest.Mock).mockReturnValue({ data: { device: "mps", dtype: "float32", engines: [] } });
  (api.get as jest.Mock).mockResolvedValue({ samples: [] });
  (api.post as jest.Mock).mockResolvedValue({ language: "en", hinglish: false });
  (api.blob as jest.Mock).mockResolvedValue(new Blob(["x"], { type: "audio/wav" }));
});

async function startCloneRender() {
  render(<SpeechPanel />);
  fireEvent.click(screen.getByRole("button", { name: /^My voice/ }));
  fireEvent.change(screen.getByLabelText("Script"), { target: { value: "One. Two. Three." } });
  await act(async () => {
    await new Promise((r) => setTimeout(r, 700)); // language detection is debounced by 600 ms
  });
  fireEvent.click(screen.getByRole("button", { name: /Generate speech/ }));
}

describe("SpeechPanel: a cloned-voice render streams sentence by sentence", () => {
  it("plays each sentence as it arrives, shows the progress, then swaps in the final clip with its receipt", async () => {
    let emit: OnEvent = () => undefined;
    let finish: () => void = () => undefined;
    (api.sse as jest.Mock).mockImplementation(
      (_path: string, _body: FormData, opts: { onEvent: OnEvent }) =>
        new Promise<void>((resolve) => {
          emit = opts.onEvent;
          finish = resolve;
        })
    );
    await startCloneRender();
    await waitFor(() => expect(api.sse).toHaveBeenCalled());
    const [path, body] = (api.sse as jest.Mock).mock.calls[0];
    expect(path).toBe("/api/voices/generate/stream");
    expect((body as FormData).get("voice_id")).toBe("v1");
    expect((body as FormData).get("mode")).toBe("premium");
    expect(screen.getByTestId("stream-progress").textContent).toMatch(/Rendering sentence 1/);

    act(() => emit("sentence", sentence(0)));
    expect(players[0].added).toEqual(["http://api.test/clip/0.wav"]); // the first sentence plays at once
    expect(screen.getByTestId("stream-progress").textContent).toMatch(/Rendering sentence 2 of 3/);
    expect(screen.queryByTestId("player")).toBeNull(); // the final clip is not there yet

    act(() => emit("sentence", sentence(1)));
    act(() => emit("sentence", sentence(2)));
    expect(players[0].added).toHaveLength(3);
    expect(screen.getByTestId("stream-progress").textContent).toMatch(/Finishing the clip/);

    await act(async () => {
      emit("done", DONE);
      finish();
    });
    await waitFor(() => expect(screen.getByTestId("player")).toBeTruthy());
    expect(api.blob).toHaveBeenCalledWith("/api/generations/g1/audio", undefined, expect.anything());
    expect(players[0].closed).toBe(true);
    // the sentences were heard while rendering: the waveform must not play the clip again
    expect(screen.getByTestId("player").getAttribute("data-autoplay")).toBe("false");
    expect((screen.getByRole("link", { name: "receipt" }) as HTMLAnchorElement).href).toBe("http://api.test/api/receipts/r-123");
    expect(screen.queryByTestId("stream-progress")).toBeNull();
  });

  it("falls back to the one-response route when the stream sends an error event", async () => {
    (api.sse as jest.Mock).mockImplementation(async (_p: string, _b: FormData, opts: { onEvent: OnEvent }) => {
      opts.onEvent("sentence", sentence(0));
      opts.onEvent("error", { code: "GENERATION_FAILED", message: "engine died" });
    });
    await startCloneRender();
    await waitFor(() => expect(api.blob).toHaveBeenCalled());
    expect((api.blob as jest.Mock).mock.calls[0][0]).toBe("/api/voices/generate");
    expect(players[0].stopped).toBe(true); // what was playing stops before the fallback clip arrives
    await waitFor(() => expect(screen.getByTestId("player")).toBeTruthy());
    expect(screen.getByTestId("player").getAttribute("data-autoplay")).toBe("true");
    expect(screen.queryByRole("link", { name: "receipt" })).toBeNull();
  });

  it("falls back when the stream cannot be opened", async () => {
    (api.sse as jest.Mock).mockRejectedValue(new Error("Network error"));
    await startCloneRender();
    await waitFor(() => expect(api.blob).toHaveBeenCalled());
    expect((api.blob as jest.Mock).mock.calls[0][0]).toBe("/api/voices/generate");
    await waitFor(() => expect(screen.getByTestId("player")).toBeTruthy());
  });

  it("does not stream a preset-voice render", async () => {
    render(<SpeechPanel />);
    fireEvent.change(screen.getByLabelText("Script"), { target: { value: "Hello there friend." } });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 700));
    });
    fireEvent.click(screen.getByRole("button", { name: /Generate speech/ }));
    await waitFor(() => expect(api.blob).toHaveBeenCalled());
    expect((api.blob as jest.Mock).mock.calls[0][0]).toBe("/api/kokoro/generate");
    expect(api.sse).not.toHaveBeenCalled();
  });
});
