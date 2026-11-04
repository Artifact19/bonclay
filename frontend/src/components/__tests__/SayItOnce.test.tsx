import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SayItOnce } from "@/components/say-it-once/SayItOnce";
import { formatDuration, formatEta, ordinal } from "@/components/say-it-once/progress";
import { useVoices } from "@/hooks/useApi";
import { api } from "@/lib/api";

jest.mock("next/link", () => ({
  __esModule: true,
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));
jest.mock("@/lib/api", () => ({
  API_URL: "http://api.test",
  api: { get: jest.fn(), post: jest.fn(), blob: jest.fn() },
  ApiError: class extends Error {},
}));
jest.mock("@/hooks/useApi", () => ({ useVoices: jest.fn() }));
jest.mock("@/components/audio/WaveformPlayer", () => ({
  WaveformPlayer: ({ src, aiGenerated }: { src: unknown; aiGenerated?: boolean }) => (
    <div data-testid="player" data-ai={String(!!aiGenerated)}>
      {typeof src === "string" ? src : "blob"}
    </div>
  ),
}));

const get = api.get as jest.Mock;
const post = api.post as jest.Mock;
const blob = api.blob as jest.Mock;

const VOICES = [
  { id: "v-old", display_name: "Old clone", created_at: "2026-01-01" },
  { id: "v-user", display_name: "Studio Speaker", consent_id: "c1", consent_verified: true, created_at: "2026-10-14" },
];

const route = (voice_kind: string | null, engine: string | null, available = true, reason: string | null = null) => ({
  voice_kind, engine, engine_name: engine, available, reason, est_seconds: 10,
});
const cap = (language: string, language_name: string, own: ReturnType<typeof route>, fast: ReturnType<typeof route>) => ({
  language, language_name, ...own, fast,
});
const CAPS = {
  voice_id: "v-user",
  languages: [
    cap("hi", "Hindi", route("your_voice", "chatterbox_ml"), route("preset_voice", "kokoro")),
    cap("mr", "Marathi", route("preset_voice", "indic_speak"), route("preset_voice", "indic_speak")),
    cap("ta", "Tamil", route("preset_voice", "indic_speak", false, "not set up"), route("preset_voice", "indic_speak", false, "not set up")),
    cap("fr", "French", route("your_voice", "qwen3_0_6b"), route("preset_voice", "kokoro")),
    cap("bn", "Bengali", route("preset_voice", "indic_speak"), route("preset_voice", "indic_speak")),
    cap("te", "Telugu", route("preset_voice", "indic_speak"), route("preset_voice", "indic_speak")),
    cap("gu", "Gujarati", route("preset_voice", "indic_speak"), route("preset_voice", "indic_speak")),
    cap("en", "English", route("your_voice", "qwen3_0_6b"), route("preset_voice", "kokoro")),
    cap("es", "Spanish", route("your_voice", "qwen3_0_6b"), route("preset_voice", "kokoro")),
    cap("de", "German", route("your_voice", "qwen3_0_6b"), route(null, null, false, "No preset voice speaks German.")),
  ],
};

const estimateItem = (language: string, language_name: string, est_s: number, voice_kind = "your_voice") => ({
  language, language_name, voice_kind, engine: "x", engine_name: "x", available: true, reason: null, est_s,
});
const ESTIMATE = {
  speed: "own", total_s: 60,
  items: [estimateItem("fr", "French", 20), estimateItem("hi", "Hindi", 40)],
};

const item = (language: string, language_name: string, extra = {}) => ({
  language, language_name, status: "queued", stage: "waiting", progress: 0, eta_s: null, queue_position: null,
  started_at: null, est_s: null, text: null, voice_kind: null, engine: null, engine_name: null,
  attribution: null, audio_url: null, receipt_id: null, seconds: null, error: null, ...extra,
});
const job = (state: string, items: unknown[], extra = {}) => ({
  state, speed: "own", progress: 0, eta_s: 0, done: 0, failed: 0, total: items.length, elapsed_s: 0,
  cancelling: false, items, ...extra,
});

const FR_TEXT = "Votre rendez-vous est confirmé pour jeudi à seize heures.";
const MR_TEXT = "तुमची अपॉइंटमेंट गुरुवारी चार वाजता निश्चित आहे.";
const FR_DONE = item("fr", "French", {
  status: "done", stage: "done", progress: 100, text: FR_TEXT, voice_kind: "your_voice",
  engine: "qwen3_0_6b", engine_name: "Qwen3-TTS", audio_url: "/api/generations/g1/audio", receipt_id: "rcpt-fr", seconds: 9.6,
});
const MR_DONE = item("mr", "Marathi", {
  status: "done", stage: "done", progress: 100, text: MR_TEXT, voice_kind: "preset_voice",
  engine: "indic_speak", engine_name: "Indic-Speak", attribution: "Built with Indic-Speak from Bodhan AI / AI4Bharat.",
  audio_url: "/api/generations/g2/audio", receipt_id: "rcpt-mr", seconds: 58.2,
});

/** Answers the polls of one job one at a time, so a test can look at the page between two snapshots. */
function gate() {
  const waiting: ((v: unknown) => void)[] = [];
  return {
    call: () => new Promise((resolve) => waiting.push(resolve)),
    async push(snapshot: unknown) {
      await waitFor(() => expect(waiting.length).toBeGreaterThan(0));
      await act(async () => {
        waiting.shift()!(snapshot);
      });
    },
  };
}

type Handlers = Record<string, (body: Record<string, unknown>) => unknown>;
/** POSTs by path; the estimate always answers unless a test says otherwise. */
function onPost(handlers: Handlers) {
  post.mockImplementation(async (path: string, body: Record<string, unknown>) => {
    if (handlers[path]) return handlers[path](body);
    if (path === "/api/say-it-once/estimate") return ESTIMATE;
    throw new Error(`unexpected POST ${path}`);
  });
}
const startCalls = () => post.mock.calls.filter(([p]) => p === "/api/say-it-once");

beforeEach(() => {
  jest.clearAllMocks();
  (useVoices as jest.Mock).mockReturnValue({ data: VOICES, error: undefined, isLoading: false, mutate: jest.fn() });
  blob.mockResolvedValue(new Blob(["x"], { type: "audio/wav" }));
  get.mockImplementation(async (path: string) => {
    if (path.startsWith("/api/say-it-once/capabilities")) return CAPS;
    throw new Error(`unexpected GET ${path}`);
  });
  onPost({});
});

async function renderReady() {
  render(<SayItOnce pollMs={5} estimateDebounceMs={1} />);
  await waitFor(() => expect(screen.getByTestId("chip-kind-hi")).toBeTruthy());
}

/** French and Marathi picked (Hindi deselected), ready to say. */
async function pickFrenchAndMarathi() {
  await renderReady();
  fireEvent.click(screen.getByRole("button", { name: /Hindi/ }));
  fireEvent.click(screen.getByRole("button", { name: /Marathi/ }));
}

describe("Say it once page: the form", () => {
  it("offers only consented voices, labels each language chip and starts in review-free, your-voice mode", async () => {
    render(<SayItOnce pollMs={5} estimateDebounceMs={1} />);
    const select = screen.getByLabelText("Voice") as HTMLSelectElement;
    expect([...select.options].map((o) => o.text)).toEqual(["Studio Speaker"]);
    expect((screen.getByLabelText(/Sentence/) as HTMLTextAreaElement).value).toBe(
      "Your appointment is confirmed for Thursday at four."
    );
    await waitFor(() => expect(screen.getByTestId("chip-kind-hi").textContent).toBe("your voice"));
    expect(get).toHaveBeenCalledWith("/api/say-it-once/capabilities?voice_id=v-user", { retries: 0 });
    expect(screen.getByTestId("chip-kind-mr").textContent).toBe("preset voice");
    expect(screen.getByTestId("chip-kind-ta").textContent).toBe("not available");
    expect((screen.getByRole("button", { name: /Tamil/ }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByLabelText(/Your voice \(slower\)/) as HTMLInputElement).checked).toBe(true);
    expect((screen.getByLabelText(/Fast \(preset voices\)/) as HTMLInputElement).checked).toBe(false);
    expect((screen.getByLabelText("Review translations before speaking") as HTMLInputElement).checked).toBe(false);
    expect(screen.queryByRole("button", { name: "Translate" })).toBeNull(); // one action, not two
    expect(screen.getByRole("button", { name: "Say it" })).toBeTruthy();
  });

  it("tells you before you start how long each language takes on this laptop", async () => {
    await renderReady();
    const box = await screen.findByTestId("estimate");
    expect(box.textContent).toContain("Your voice in Hindi takes about 40 s on this laptop.");
    expect(box.textContent).toContain("Your voice in French takes about 20 s on this laptop.");
    expect(box.textContent).toContain("All together about 60 s, one language at a time.");
    expect(post).toHaveBeenCalledWith(
      "/api/say-it-once/estimate",
      { voice_id: "v-user", text: "Your appointment is confirmed for Thursday at four.", languages: ["hi", "fr"], source_language: "en", speed: "own" },
      { retries: 0 }
    );
  });

  it("Fast mode re-labels the languages for preset voices, re-estimates, and leaves out languages without a preset", async () => {
    onPost({
      "/api/say-it-once/estimate": (b) =>
        b.speed === "fast"
          ? { speed: "fast", total_s: 8, items: [estimateItem("fr", "French", 3, "preset_voice"), estimateItem("hi", "Hindi", 5, "preset_voice")] }
          : ESTIMATE,
    });
    await renderReady();
    fireEvent.click(screen.getByRole("button", { name: "More languages" }));
    expect((screen.getByRole("button", { name: /German/ }) as HTMLButtonElement).disabled).toBe(false); // your voice speaks German
    fireEvent.click(screen.getByLabelText(/Fast \(preset voices\)/));
    expect((screen.getByRole("button", { name: /German/ }) as HTMLButtonElement).disabled).toBe(true); // no preset does
    expect(screen.getByTestId("chip-kind-de").textContent).toBe("not available");
    expect(screen.getByTestId("chip-kind-hi").textContent).toBe("preset voice");
    expect(screen.getByTestId("chip-kind-fr").textContent).toBe("preset voice");
    await waitFor(() => expect(screen.getByTestId("estimate").textContent).toContain("A preset voice in Hindi takes about 5 s"));
    expect(post).toHaveBeenCalledWith(
      "/api/say-it-once/estimate",
      expect.objectContaining({ speed: "fast", languages: ["hi", "fr"] }),
      { retries: 0 }
    );
  });

  it("points to voice cloning when no voice has a consent", () => {
    (useVoices as jest.Mock).mockReturnValue({ data: [VOICES[0]], error: undefined, isLoading: false, mutate: jest.fn() });
    render(<SayItOnce />);
    expect(screen.getByRole("link", { name: "Clone a voice" }).getAttribute("href")).toBe("/voices/new");
    expect(screen.queryByRole("button", { name: "Say it" })).toBeNull();
  });
});

describe("Say it once page: DhVaani speaks the Indian languages in your voice", () => {
  const CREDIT = "Voice cloning in Indian languages by DhVaani from ARTPARK-IISc.";
  const DHVAANI_CAPS = {
    voice_id: "v-user",
    languages: CAPS.languages.map((c) =>
      ["hi", "mr", "bn", "te", "gu"].includes(c.language)
        ? { ...c, ...route("your_voice", "dhvaani"), engine_name: "DhVaani-0.5 (ARTPARK-IISc)", attribution: CREDIT }
        : c
    ),
  };

  it("labels the Indian languages 'your voice' and credits DhVaani on the card", async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith("/api/say-it-once/capabilities")) return DHVAANI_CAPS;
      if (path === "/api/say-it-once/job-d") {
        return job("done", [item("mr", "Marathi", { status: "done", stage: "done", progress: 100, text: MR_TEXT, voice_kind: "your_voice", engine: "dhvaani", engine_name: "DhVaani-0.5 (ARTPARK-IISc)", attribution: CREDIT, audio_url: "/api/generations/g3/audio", receipt_id: "rcpt-d", seconds: 6 })], { progress: 100, done: 1 });
      }
      throw new Error(`unexpected GET ${path}`);
    });
    onPost({ "/api/say-it-once": () => ({ job_id: "job-d", speed: "own", est_total_s: 8, items: [{ language: "mr", status: "queued" }] }) });
    await renderReady();
    expect(screen.getByTestId("chip-kind-mr").textContent).toBe("your voice");
    expect(screen.getByTestId("chip-kind-hi").textContent).toBe("your voice");
    expect(screen.getByTestId("chip-kind-ta").textContent).toBe("not available"); // not set up in this fixture
    fireEvent.click(screen.getByRole("button", { name: /Hindi/ })); // deselect Hindi and French, keep Marathi
    fireEvent.click(screen.getByRole("button", { name: /French/ }));
    fireEvent.click(screen.getByRole("button", { name: /Marathi/ }));
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    const card = await screen.findByTestId("result-mr");
    await waitFor(() => expect(within(card).getByText(/DhVaani-0.5 \(ARTPARK-IISc\)/)).toBeTruthy());
    expect(within(card).getByText(new RegExp(CREDIT.replace(/\./g, "\\.")))).toBeTruthy();
  });
});

describe("Say it once page: one click, one card per language", () => {
  it("translates and speaks each language in turn, with text, progress and audio in the same card", async () => {
    onPost({ "/api/say-it-once": () => ({ job_id: "job-1", speed: "own", est_total_s: 70, items: [{ language: "fr", status: "queued" }, { language: "mr", status: "queued" }] }) });
    const polls = gate();
    get.mockImplementation((path: string) => {
      if (path.startsWith("/api/say-it-once/capabilities")) return Promise.resolve(CAPS);
      if (path === "/api/say-it-once/job-1") return polls.call();
      return Promise.reject(new Error(`unexpected GET ${path}`));
    });
    await pickFrenchAndMarathi();
    expect(screen.queryByTestId("result-fr")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    // one request, straight from the sentence: no translate step, no translations, the speed is sent
    await waitFor(() => expect(startCalls()).toHaveLength(1));
    expect(startCalls()[0]).toEqual([
      "/api/say-it-once",
      { voice_id: "v-user", text: "Your appointment is confirmed for Thursday at four.", languages: ["fr", "mr"], source_language: "en", speed: "own" },
      { retries: 0 },
    ]);
    expect(post.mock.calls.some(([p]) => p === "/api/say-it-once/translate")).toBe(false);

    // right away: both cards, waiting, at 0 %, with their place in line
    const fr = await screen.findByTestId("result-fr");
    const mr = screen.getByTestId("result-mr");
    expect(within(fr).getByTestId("pct-fr").textContent).toBe("0%");
    expect(within(fr).getByRole("status").textContent).toBe("Waiting, up next");
    expect(within(mr).getByRole("status").textContent).toBe("Waiting, 2nd in line");

    // French is being translated: 10 %
    await polls.push(job("running", [
      item("fr", "French", { status: "translating", stage: "translating", progress: 10, eta_s: 35, started_at: 1, voice_kind: "your_voice", engine_name: "Qwen3-TTS" }),
      item("mr", "Marathi", { queue_position: 1, eta_s: 95, voice_kind: "preset_voice", engine_name: "Indic-Speak" }),
    ], { progress: 4, eta_s: 95 }));
    expect(within(fr).getByRole("status").textContent).toBe("Translating");
    expect(within(fr).getByTestId("pct-fr").textContent).toBe("10%");
    expect(within(screen.getByTestId("overall")).getByTestId("overall-pct").textContent).toBe("4%");
    expect(within(mr).getByRole("status").textContent).toBe("Waiting, up next");

    // French text appears as soon as it is ready, while its voice is still being generated: 45 %
    await polls.push(job("running", [
      item("fr", "French", { status: "speaking", stage: "speaking", progress: 45, eta_s: 20, text: FR_TEXT, voice_kind: "your_voice", engine_name: "Qwen3-TTS" }),
      item("mr", "Marathi", { queue_position: 1, eta_s: 80, voice_kind: "preset_voice", engine_name: "Indic-Speak" }),
    ], { progress: 18, eta_s: 80 }));
    expect(within(fr).getByTestId("text-fr").textContent).toBe(FR_TEXT);
    expect(within(fr).getByRole("status").textContent).toBe("Generating voice");
    expect(within(fr).getByTestId("pct-fr").textContent).toBe("45%");
    expect(within(fr).getByText("about 20 s left")).toBeTruthy();
    expect(within(fr).getByRole("progressbar").getAttribute("aria-valuenow")).toBe("45");
    expect(within(fr).queryByTestId("player")).toBeNull();
    expect(within(fr).queryByRole("textbox")).toBeNull(); // not editable until it is done
    const overall = screen.getByTestId("overall");
    expect(within(overall).getByTestId("overall-pct").textContent).toBe("18%");
    expect(within(overall).getByTestId("overall-detail").textContent).toContain("0 of 2 languages done");
    expect(within(overall).getByText("about 80 s left")).toBeTruthy();
    expect(within(overall).getByRole("progressbar", { name: "Overall progress" }).getAttribute("aria-valuenow")).toBe("18");

    // French done, Marathi under way
    await polls.push(job("running", [
      FR_DONE,
      item("mr", "Marathi", { status: "speaking", stage: "speaking", progress: 60, eta_s: 30, text: MR_TEXT, started_at: 5, voice_kind: "preset_voice", engine_name: "Indic-Speak", attribution: "Built with Indic-Speak from Bodhan AI / AI4Bharat." }),
    ], { progress: 70, eta_s: 30, done: 1 }));
    expect(within(fr).getByTestId("pct-fr").textContent).toBe("100%");
    expect(await within(fr).findByTestId("player")).toBeTruthy();
    expect(within(overall).getByTestId("overall-detail").textContent).toContain("1 of 2 languages done");
    // the attribution is on the card from the moment the voice is known
    expect(within(mr).getByText(/Built with Indic-Speak from Bodhan AI/)).toBeTruthy();

    // everything done: 100 %
    await polls.push(job("done", [FR_DONE, MR_DONE], { progress: 100, done: 2, elapsed_s: 71 }));
    expect(within(overall).getByTestId("overall-pct").textContent).toBe("100%");
    expect(within(overall).getByTestId("overall-detail").textContent).toContain("2 of 2 languages done");
    expect(within(overall).getByText("All done")).toBeTruthy();

    // exactly one card per language, each with its text, voice badge, audio and links together
    expect(screen.getAllByTestId(/^result-/)).toHaveLength(2);
    expect(screen.queryByRole("region", { name: "Review translations" })).toBeNull();
    expect(within(fr).getByTestId("badge-your-voice").textContent).toBe("Your voice");
    expect((within(fr).getByTestId("edit-fr") as HTMLTextAreaElement).value).toBe(FR_TEXT);
    expect(within(mr).getByTestId("badge-preset-voice").textContent).toBe("Preset voice");
    expect(within(mr).getByText(/Indic-Speak/, { selector: "p" })).toBeTruthy();
    expect((within(mr).getByTestId("edit-mr") as HTMLTextAreaElement).value).toBe(MR_TEXT);
    await waitFor(() => expect(within(mr).getByTestId("player").getAttribute("data-ai")).toBe("true"));
    expect(blob).toHaveBeenCalledWith("/api/generations/g2/audio", undefined, { retries: 1 });
    expect(within(mr).getByRole("link", { name: "Verify" }).getAttribute("href")).toBe("/verify");
    expect(within(mr).getByRole("link", { name: "Receipt" }).getAttribute("href")).toBe("http://api.test/api/receipts/rcpt-mr");
    expect(within(fr).getByRole("link", { name: "Receipt" }).getAttribute("href")).toBe("http://api.test/api/receipts/rcpt-fr");
    expect(within(mr).getByRole("status").textContent).toBe("Done in 58 s");
  });

  it("says so, and stops promising a time, when a language runs well past its estimate", async () => {
    onPost({ "/api/say-it-once": () => ({ job_id: "job-1", items: [{ language: "fr", status: "queued" }] }) });
    const polls = gate();
    get.mockImplementation((path: string) => {
      if (path.startsWith("/api/say-it-once/capabilities")) return Promise.resolve(CAPS);
      return polls.call();
    });
    await renderReady();
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    await polls.push(job("running", [item("fr", "French", { status: "speaking", stage: "speaking", progress: 95, eta_s: 30, slow: true, text: FR_TEXT })], { progress: 95, eta_s: 30, slow: true }));
    const fr = screen.getByTestId("result-fr");
    expect(within(fr).getByTestId("slow-fr").textContent).toContain("Taking longer than expected");
    expect(within(fr).getByTestId("pct-fr").textContent).toBe("95%");
    expect(within(fr).queryByText(/left/)).toBeNull();
    expect(screen.getByTestId("overall-detail").textContent).toContain("Taking longer than estimated");
    expect(screen.getByTestId("overall-detail").textContent).not.toContain("left");
    await polls.push(job("done", [FR_DONE], { progress: 100, done: 1 }));
    expect(screen.queryByTestId("slow-fr")).toBeNull();
  });

  it("shows a failed language inside its own card with a Retry button, and keeps the others", async () => {
    onPost({ "/api/say-it-once": () => ({ job_id: "job-2", items: [{ language: "fr", status: "queued" }] }) });
    get.mockImplementation(async (path: string) => {
      if (path.startsWith("/api/say-it-once/capabilities")) return CAPS;
      return job("done", [
        FR_DONE,
        item("hi", "Hindi", { status: "failed", stage: "failed", progress: 15, error: "The language model did not translate this into Hindi." }),
        item("mr", "Marathi", { status: "unavailable", stage: "unavailable", error: "Indic-Speak is not set up." }),
      ], { progress: 100, done: 1, failed: 1, total: 2 });
    });
    await renderReady();
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    const hi = await screen.findByTestId("result-hi");
    expect(within(hi).getByRole("alert").textContent).toContain("did not translate this into Hindi");
    expect(within(hi).getByRole("button", { name: "Retry" })).toBeTruthy();
    expect(within(hi).queryByTestId("player")).toBeNull();
    expect(within(hi).getByTestId("pct-hi").textContent).toBe("15%");
    expect(screen.getByTestId("result-mr").textContent).toContain("Indic-Speak is not set up.");
    expect(within(screen.getByTestId("result-mr")).queryByRole("button", { name: "Retry" })).toBeNull();
    expect(await within(screen.getByTestId("result-fr")).findByTestId("player")).toBeTruthy();
    expect(screen.getByTestId("overall-detail").textContent).toContain("1 of 2 languages done, 1 failed");
  });

  it("Retry runs just that language again and its card fills in when it is done", async () => {
    const HI_FAILED = item("hi", "Hindi", { status: "failed", stage: "failed", progress: 15, error: "The language model did not translate this into Hindi." });
    const HI_DONE = item("hi", "Hindi", { status: "done", stage: "done", progress: 100, text: "आपकी अपॉइंटमेंट गुरुवार को चार बजे की पक्की है।", voice_kind: "your_voice", engine_name: "Chatterbox", audio_url: "/api/generations/g9/audio", receipt_id: "rcpt-hi", seconds: 30 });
    onPost({
      "/api/say-it-once": (b) =>
        (b.languages as string[]).length === 1
          ? { job_id: "job-retry", items: [{ language: "hi", status: "queued" }] }
          : { job_id: "job-1", items: [{ language: "fr", status: "queued" }, { language: "hi", status: "queued" }] },
    });
    const retryPolls = gate();
    get.mockImplementation((path: string) => {
      if (path.startsWith("/api/say-it-once/capabilities")) return Promise.resolve(CAPS);
      if (path === "/api/say-it-once/job-1") return Promise.resolve(job("done", [FR_DONE, HI_FAILED], { progress: 100, done: 1, failed: 1 }));
      if (path === "/api/say-it-once/job-retry") return retryPolls.call();
      return Promise.reject(new Error(`unexpected GET ${path}`));
    });
    await renderReady();
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    const hi = await screen.findByTestId("result-hi");
    fireEvent.click(await within(hi).findByRole("button", { name: "Retry" }));

    await waitFor(() => expect(startCalls()).toHaveLength(2));
    expect(startCalls()[1]).toEqual([
      "/api/say-it-once",
      { voice_id: "v-user", text: "Your appointment is confirmed for Thursday at four.", languages: ["hi"], source_language: "en", speed: "own" },
      { retries: 0 },
    ]); // nothing was translated yet, so no text is sent and the model translates again
    expect(within(hi).queryByRole("alert")).toBeNull();
    await retryPolls.push(job("running", [item("hi", "Hindi", { status: "speaking", stage: "speaking", progress: 50, eta_s: 15, text: HI_DONE.text })]));
    expect(within(hi).getByTestId("pct-hi").textContent).toBe("50%");
    await retryPolls.push(job("done", [HI_DONE], { progress: 100, done: 1 }));
    expect(await within(hi).findByTestId("player")).toBeTruthy();
    expect(within(screen.getByTestId("result-fr")).getByTestId("player")).toBeTruthy(); // French untouched
    expect(screen.getAllByTestId(/^result-/)).toHaveLength(2);
  });

  it("lets you edit one finished translation and Re-speak only that language", async () => {
    const edited = "Votre rendez-vous est bien confirmé pour jeudi à seize heures.";
    const FR_AGAIN = { ...FR_DONE, text: edited, audio_url: "/api/generations/g7/audio", receipt_id: "rcpt-fr-2", seconds: 7.2 };
    onPost({
      "/api/say-it-once": (b) =>
        (b.languages as string[]).length === 1
          ? { job_id: "job-again", items: [{ language: "fr", status: "queued" }] }
          : { job_id: "job-1", items: [{ language: "fr", status: "queued" }, { language: "mr", status: "queued" }] },
    });
    const again = gate();
    get.mockImplementation((path: string) => {
      if (path.startsWith("/api/say-it-once/capabilities")) return Promise.resolve(CAPS);
      if (path === "/api/say-it-once/job-1") return Promise.resolve(job("done", [FR_DONE, MR_DONE], { progress: 100, done: 2 }));
      if (path === "/api/say-it-once/job-again") return again.call();
      return Promise.reject(new Error(`unexpected GET ${path}`));
    });
    await pickFrenchAndMarathi();
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    const fr = await screen.findByTestId("result-fr");
    const box = (await within(fr).findByTestId("edit-fr")) as HTMLTextAreaElement;
    expect((within(fr).getByRole("button", { name: "Re-speak" }) as HTMLButtonElement).disabled).toBe(false);

    fireEvent.change(box, { target: { value: edited } });
    fireEvent.click(within(fr).getByRole("button", { name: "Re-speak" }));

    await waitFor(() => expect(startCalls()).toHaveLength(2));
    expect(startCalls()[1]).toEqual([
      "/api/say-it-once",
      { voice_id: "v-user", text: "Your appointment is confirmed for Thursday at four.", languages: ["fr"], source_language: "en", speed: "own", translations: { fr: edited } },
      { retries: 0 },
    ]);
    // only French is busy again; Marathi keeps its audio
    expect(within(fr).queryByTestId("player")).toBeNull();
    expect(within(fr).getByTestId("pct-fr").textContent).toBe("0%");
    expect(within(screen.getByTestId("result-mr")).getByTestId("player")).toBeTruthy();
    await again.push(job("running", [item("fr", "French", { status: "speaking", stage: "speaking", progress: 40, eta_s: 9, text: edited, voice_kind: "your_voice" })]));
    expect(within(fr).getByTestId("pct-fr").textContent).toBe("40%");
    expect(within(fr).getByTestId("text-fr").textContent).toBe(edited);
    await again.push(job("done", [FR_AGAIN], { progress: 100, done: 1 }));
    await waitFor(() => expect(within(fr).getByRole("link", { name: "Receipt" }).getAttribute("href")).toBe("http://api.test/api/receipts/rcpt-fr-2"));
    expect((within(fr).getByTestId("edit-fr") as HTMLTextAreaElement).value).toBe(edited);
    expect(within(fr).getByTestId("player")).toBeTruthy();
    expect(screen.getAllByTestId(/^result-/)).toHaveLength(2);
  });

  it("Cancel stops the remaining languages and says so", async () => {
    onPost({
      "/api/say-it-once": () => ({ job_id: "job-1", items: [{ language: "fr", status: "queued" }, { language: "mr", status: "queued" }] }),
      "/api/say-it-once/job-1/cancel": () => job("running", [
        item("fr", "French", { status: "speaking", stage: "speaking", progress: 30, eta_s: 15, text: FR_TEXT }),
        item("mr", "Marathi", { queue_position: 1 }),
      ], { cancelling: true, progress: 10, eta_s: 15 }),
    });
    const polls = gate();
    get.mockImplementation((path: string) => {
      if (path.startsWith("/api/say-it-once/capabilities")) return Promise.resolve(CAPS);
      if (path === "/api/say-it-once/job-1") return polls.call();
      return Promise.reject(new Error(`unexpected GET ${path}`));
    });
    await pickFrenchAndMarathi();
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    await polls.push(job("running", [
      item("fr", "French", { status: "speaking", stage: "speaking", progress: 30, eta_s: 15, text: FR_TEXT }),
      item("mr", "Marathi", { queue_position: 1 }),
    ], { progress: 10, eta_s: 100 }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/say-it-once/job-1/cancel", {}, { retries: 0 }));
    expect(await screen.findByText("Stopping after the language being spoken")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement).disabled).toBe(true);

    await polls.push(job("cancelled", [FR_DONE, item("mr", "Marathi", { status: "cancelled", stage: "cancelled" })], { progress: 100, done: 1 }));
    expect(screen.getByText("Stopped")).toBeTruthy();
    const mr = screen.getByTestId("result-mr");
    expect(within(mr).getByRole("status").textContent).toBe("Cancelled");
    expect(within(mr).getByRole("button", { name: "Speak this language" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
  });

  it("locks the form while it works and sends the chosen speed", async () => {
    onPost({ "/api/say-it-once": () => ({ job_id: "job-1", items: [{ language: "hi", status: "queued" }] }) });
    const polls = gate();
    get.mockImplementation((path: string) => {
      if (path.startsWith("/api/say-it-once/capabilities")) return Promise.resolve(CAPS);
      return polls.call();
    });
    await renderReady();
    fireEvent.click(screen.getByLabelText(/Fast \(preset voices\)/));
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    await waitFor(() => expect(startCalls()).toHaveLength(1));
    expect(startCalls()[0][1]).toMatchObject({ speed: "fast", languages: ["hi", "fr"] });
    expect((screen.getByLabelText(/Sentence/) as HTMLTextAreaElement).disabled).toBe(true);
    expect((screen.getByLabelText(/Fast \(preset voices\)/) as HTMLInputElement).disabled).toBe(true);
    await polls.push(job("done", [FR_DONE], { progress: 100, done: 1 }));
    await waitFor(() => expect((screen.getByLabelText(/Sentence/) as HTMLTextAreaElement).disabled).toBe(false));
  });
});

describe("Say it once page: the optional review step", () => {
  const review = (language: string, language_name: string, translation: string, extra = {}) => ({
    language, language_name, translation, warnings: [], voice: "yours", engine: "qwen3_0_6b",
    engine_name: "Qwen3-TTS", available: true, reason: null, attribution: null, ...extra,
  });
  const REVIEW = {
    source_language: "en",
    items: [
      review("hi", "Hindi", "आपकी अपॉइंटमेंट शनिवार को चार बजे की पक्की है।", {
        engine: "chatterbox_ml", warnings: ["The Hindi translation names a different day than the sentence."],
      }),
      review("fr", "French", FR_TEXT),
    ],
  };

  it("is off by default: Say it goes straight to speaking, never to a review", async () => {
    onPost({ "/api/say-it-once": () => ({ job_id: "job-3", items: [{ language: "hi", status: "queued" }] }) });
    get.mockImplementation(async (path: string) => {
      if (path.startsWith("/api/say-it-once/capabilities")) return CAPS;
      return job("done", [FR_DONE], { progress: 100, done: 1 });
    });
    await renderReady();
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    await waitFor(() => expect(startCalls()).toHaveLength(1));
    expect(startCalls()[0][1]).not.toHaveProperty("translations");
    expect(screen.queryByTestId("review-hi")).toBeNull();
    expect(post.mock.calls.some(([p]) => p === "/api/say-it-once/translate")).toBe(false);
  });

  it("when switched on, translates first, shows editable translations with warnings, then speaks the approved text", async () => {
    const edited = "आपकी अपॉइंटमेंट गुरुवार को चार बजे की पक्की है।";
    onPost({
      "/api/say-it-once/translate": () => REVIEW,
      "/api/say-it-once": () => ({ job_id: "job-9", items: [{ language: "fr", status: "queued" }, { language: "hi", status: "queued" }] }),
    });
    get.mockImplementation(async (path: string) => {
      if (path.startsWith("/api/say-it-once/capabilities")) return CAPS;
      if (path === "/api/say-it-once/job-9") return job("done", [FR_DONE, item("hi", "Hindi", { status: "done", stage: "done", progress: 100, text: edited, voice_kind: "your_voice", audio_url: "/api/generations/g3/audio", receipt_id: "rcpt-hi", seconds: 30 })], { progress: 100, done: 2 });
      throw new Error(`unexpected GET ${path}`);
    });
    await renderReady();
    fireEvent.click(screen.getByLabelText("Review translations before speaking"));
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));

    const hi = await screen.findByTestId("review-hi");
    expect(post).toHaveBeenCalledWith(
      "/api/say-it-once/translate",
      { voice_id: "v-user", text: "Your appointment is confirmed for Thursday at four.", languages: ["hi", "fr"], source_language: "en", speed: "own" },
      { retries: 0 }
    );
    expect(within(hi).getByTestId("warning-hi").textContent).toContain("names a different day");
    expect(startCalls()).toHaveLength(0); // nothing is spoken yet
    fireEvent.change(within(hi).getByRole("textbox"), { target: { value: edited } });
    expect(screen.queryByTestId("warning-hi")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Speak all" }));
    await waitFor(() => expect(startCalls()).toHaveLength(1));
    expect(startCalls()[0][1]).toEqual({
      voice_id: "v-user", text: "Your appointment is confirmed for Thursday at four.", languages: ["hi", "fr"],
      source_language: "en", speed: "own", translations: { hi: edited, fr: FR_TEXT },
    });
    const result = await screen.findByTestId("result-hi");
    expect(await within(result).findByTestId("player")).toBeTruthy();
    expect((within(result).getByTestId("edit-hi") as HTMLTextAreaElement).value).toBe(edited);
  });

  it("blocks Speak all while a translation is empty, and Back to edit returns to the sentence", async () => {
    onPost({ "/api/say-it-once/translate": () => REVIEW });
    await renderReady();
    fireEvent.click(screen.getByLabelText("Review translations before speaking"));
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    const fr = await screen.findByTestId("review-fr");
    fireEvent.change(within(fr).getByRole("textbox"), { target: { value: "   " } });
    expect((screen.getByRole("button", { name: "Speak all" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Back to edit" }));
    expect(screen.queryByTestId("review-fr")).toBeNull();
    expect((screen.getByRole("button", { name: "Say it" }) as HTMLButtonElement).disabled).toBe(false);
    expect((screen.getByLabelText(/Sentence/) as HTMLTextAreaElement).disabled).toBe(false);
  });

  it("shows an error and stays on the sentence when translating fails", async () => {
    onPost({ "/api/say-it-once/translate": () => { throw new Error("No language model is available to translate."); } });
    await renderReady();
    fireEvent.click(screen.getByLabelText("Review translations before speaking"));
    fireEvent.click(screen.getByRole("button", { name: "Say it" }));
    expect((await screen.findByRole("alert")).textContent).toContain("No language model");
    expect(screen.queryByRole("button", { name: "Speak all" })).toBeNull();
  });
});

describe("progress wording", () => {
  it("rounds the time left so it does not flicker and never says 0 s while work remains", () => {
    expect(formatEta(null)).toBeNull();
    expect(formatEta(0)).toBe("a few seconds left");
    expect(formatEta(4)).toBe("a few seconds left");
    expect(formatEta(38)).toBe("about 40 s left");
    expect(formatEta(89)).toBe("about 90 s left");
    expect(formatEta(120)).toBe("about 2 min left");
    expect(formatEta(140)).toBe("about 2.5 min left");
    expect(formatDuration(3)).toBe("about 3 s");
    expect(formatDuration(42)).toBe("about 40 s");
    expect(formatDuration(60 + 20)).toBe("about 80 s");
    expect(formatDuration(200)).toBe("about 3.5 min");
  });

  it("names the place in line", () => {
    expect([1, 2, 3, 4, 11, 12, 21].map(ordinal)).toEqual(["1st", "2nd", "3rd", "4th", "11th", "12th", "21st"]);
  });
});
