import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SWRConfig } from "swr";
import { ResourceIndicator } from "@/components/resources/ResourceIndicator";
import { api } from "@/lib/api";
import type { ResourceModel, SystemResources } from "@/types/api";

jest.mock("@/lib/api", () => ({
  api: { get: jest.fn(), post: jest.fn() },
  ApiError: jest.requireActual("@/lib/apiBase").ApiError,
}));
const { ApiError } = jest.requireActual("@/lib/apiBase");

const get = api.get as jest.Mock;
const post = api.post as jest.Mock;

function model(over: Partial<ResourceModel> & Pick<ResourceModel, "key" | "label">): ResourceModel {
  return {
    kind: "voice",
    where: "worker process",
    device: "mlx",
    state: "loaded",
    memory_mb: 600,
    memory_source: "measured",
    idle_s: 5,
    pid: 123,
    can_unload: true,
    ...over,
  };
}

function snapshot(over: Partial<SystemResources["system"]> = {}, models?: ResourceModel[]): SystemResources {
  return {
    time: "2026-10-27T12:00:00+00:00",
    system: {
      total_mb: 16384,
      used_mb: 10035, // 9.8 GB
      available_mb: 6349,
      compressed_mb: 2048,
      swap_used_mb: 1024,
      swap_total_mb: 4096,
      pressure: "normal",
      platform: "darwin",
      device: "mps",
      gpu: null,
      ...over,
    },
    app: { api_rss_mb: 2500, workers_rss_mb: 3100, total_rss_mb: 5600 },
    models: models ?? [
      model({ key: "kokoro", label: "Standard voice (Kokoro)", memory_mb: 600, state: "loaded" }),
      model({ key: "qwen3_0_6b", label: "Voice cloning (Qwen3-TTS)", kind: "clone", memory_mb: 1800, state: "busy", can_unload: false }),
      model({ key: "indic_speak", label: "Indian-language voices (Indic-Speak)", memory_mb: 3500, state: "pinned (live call)", can_unload: false }),
      model({ key: "embedder:bge-m3", label: "Search model (bge-m3)", kind: "search", where: "in app", device: "cpu", memory_mb: 1100, memory_source: "estimated", pid: null, can_unload: false }),
      model({ key: "dhvaani", label: "Indian-language voice cloning (DhVaani)", kind: "clone", device: "cuda", memory_mb: 900, state: "idle", idle_s: 190 }),
    ],
    guard: { enabled: true, reserve_mb: 1500, estimates_mb: { kokoro: 600 } },
  };
}

function setVisibility(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", { configurable: true, get: () => state });
  document.dispatchEvent(new Event("visibilitychange"));
}

function renderIndicator(compact = false) {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ResourceIndicator compact={compact} />
    </SWRConfig>
  );
}

beforeEach(() => {
  get.mockReset().mockResolvedValue(snapshot());
  post.mockReset().mockResolvedValue({ unloaded: true });
  setVisibility("visible");
});

afterEach(() => {
  jest.useRealTimers();
});

async function openPanel() {
  const view = renderIndicator();
  await screen.findByText(/Memory 9\.8 \/ 16 GB/);
  screen.getByTestId("resource-indicator").focus();
  fireEvent.click(screen.getByTestId("resource-indicator"));
  return { view, dialog: await screen.findByRole("dialog", { name: "Models and memory" }) };
}

describe("sidebar footer indicator", () => {
  test("shows used and total memory and the model count, as a button named with the numbers", async () => {
    renderIndicator();
    expect(await screen.findByText("Memory 9.8 / 16 GB · 5 models")).toBeInTheDocument();
    const button = screen.getByRole("button", { name: /Memory: 9\.8 of 16 GB used, 5 models loaded/ });
    expect(button).toHaveAccessibleName(/2 GB compressed, 1 GB swapped, pressure normal/);
    expect(get).toHaveBeenCalledWith("/api/system/resources");
  });

  test.each([
    ["normal", "bg-accent-success"],
    ["warn", "bg-accent-warning"],
    ["critical", "bg-accent-error"],
  ] as const)("the status dot and bar follow pressure: %s", async (pressure, dot) => {
    get.mockResolvedValue(snapshot({ pressure }));
    renderIndicator();
    await screen.findByText(/Memory 9\.8/);
    expect(screen.getByTestId("resource-dot")).toHaveClass(dot);
    expect(screen.getByTestId("memory-bar").querySelector('[data-part="used"]')).toHaveClass(
      pressure === "normal" ? "bg-accent-primary" : dot
    );
    expect(screen.getByRole("button", { name: new RegExp(`pressure ${pressure === "warn" ? "busy" : pressure}`) })).toBeInTheDocument();
  });

  test("without a pressure reading (not macOS) the free memory stands in for it", async () => {
    get.mockResolvedValue(snapshot({ pressure: null, platform: "linux", compressed_mb: null, available_mb: 500 }));
    renderIndicator();
    await screen.findByText(/Memory 9\.8/);
    expect(screen.getByTestId("resource-dot")).toHaveClass("bg-accent-error");
  });

  test("the collapsed sidebar shows only the dot and the bar, with a tooltip and the same accessible name", async () => {
    renderIndicator(true);
    const button = await screen.findByRole("button", { name: /Memory: 9\.8 of 16 GB used/ });
    expect(screen.queryByText(/Memory 9\.8/)).toBeNull();
    expect(button).toHaveAttribute("title", expect.stringContaining("Memory 9.8 / 16 GB, 5 models"));
    expect(within(button).getByTestId("resource-dot")).toBeInTheDocument();
    expect(within(button).getByTestId("memory-bar")).toBeInTheDocument();
  });

  test("a server that cannot answer leaves a quiet, still-clickable button", async () => {
    get.mockRejectedValue(new ApiError("down", 503));
    renderIndicator();
    expect(await screen.findByText("Memory unavailable")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Memory use is unavailable/ })).toBeEnabled();
  });

  test("a body that is not a snapshot (an older server) is ignored", async () => {
    get.mockResolvedValue({ counts: { pending: 0 } });
    renderIndicator();
    await act(async () => {});
    expect(screen.getByText("Memory…")).toBeInTheDocument();
  });
});

describe("screen reader announcement", () => {
  test("only a change of pressure is announced, never an ordinary refresh", async () => {
    jest.useFakeTimers();
    renderIndicator();
    await act(async () => {});
    const announcer = screen.getByTestId("pressure-announcer");
    expect(announcer).toHaveTextContent("");

    get.mockResolvedValue(snapshot({ used_mb: 10100 })); // numbers move, pressure does not
    await act(async () => { jest.advanceTimersByTime(15000); });
    expect(get).toHaveBeenCalledTimes(2);
    expect(announcer).toHaveTextContent("");

    get.mockResolvedValue(snapshot({ pressure: "critical" }));
    await act(async () => { jest.advanceTimersByTime(15000); });
    expect(announcer).toHaveTextContent("Memory pressure is now critical. Critical: the Mac is short of memory; unload something.");
  });
});

describe("polling", () => {
  test("the footer re-reads every 15 s, pauses while the tab is hidden and reads again when it comes back", async () => {
    jest.useFakeTimers();
    renderIndicator();
    await act(async () => {});
    expect(get).toHaveBeenCalledTimes(1);

    await act(async () => { jest.advanceTimersByTime(15000); });
    expect(get).toHaveBeenCalledTimes(2);

    setVisibility("hidden");
    await act(async () => { jest.advanceTimersByTime(60000); });
    expect(get).toHaveBeenCalledTimes(2);

    setVisibility("visible");
    await act(async () => { jest.advanceTimersByTime(15000); });
    expect(get.mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  test("while the panel is open it re-reads every 3 s, and stops when it is closed", async () => {
    jest.useFakeTimers();
    renderIndicator();
    await act(async () => {});
    fireEvent.click(screen.getByTestId("resource-indicator"));
    await act(async () => {});
    const before = get.mock.calls.length;
    for (let i = 0; i < 3; i++) await act(async () => { jest.advanceTimersByTime(3000); });
    expect(get.mock.calls.length - before).toBe(3);

    fireEvent.click(screen.getByRole("button", { name: "Close dialog" }));
    await act(async () => {});
    const after = get.mock.calls.length;
    for (let i = 0; i < 3; i++) await act(async () => { jest.advanceTimersByTime(3000); });
    expect(get.mock.calls.length - after).toBeLessThanOrEqual(1); // at most the footer's own 15 s timer, not every 3 s
  });
});

describe("Models and memory panel", () => {
  test("shows the system, the app and each model", async () => {
    const { dialog } = await openPanel();
    const d = within(dialog);

    expect(d.getByText("Memory pressure")).toBeInTheDocument();
    expect(d.getByText("Normal: plenty of room.")).toBeInTheDocument();
    for (const [label, value] of [
      ["Total memory", "16 GB"],
      ["In use", "9.8 GB"],
      ["Available", "6.2 GB"],
      ["Compressed", "2 GB"],
      ["Swap", "1 GB of 4 GB"],
      ["The app (API)", "2.4 GB"],
      ["Model workers", "3 GB"],
    ]) {
      expect(d.getByText(label, { selector: "dt" }).closest("div")).toHaveTextContent(value);
    }

    const rows = d.getAllByTestId("model-row");
    expect(rows).toHaveLength(5);
    expect(rows[0]).toHaveTextContent("Indian-language voices (Indic-Speak)"); // largest first
    expect(rows[0]).toHaveTextContent("Pinned during a call");
    expect(rows[0]).toHaveTextContent("3.4 GB");
    expect(rows[0]).toHaveTextContent("measured");
    expect(within(rows[0]).getByText("Apple GPU")).toBeInTheDocument();

    const search = rows.find((r) => r.getAttribute("data-key") === "embedder:bge-m3")!;
    expect(search).toHaveTextContent("estimated");
    expect(search).toHaveTextContent("about 1.1 GB");
    expect(search).toHaveTextContent("inside the app");
    expect(within(search).getByText("CPU")).toBeInTheDocument();

    const idle = rows.find((r) => r.getAttribute("data-key") === "dhvaani")!;
    expect(idle).toHaveTextContent("Idle 3 min");
    expect(within(idle).getByText("NVIDIA")).toBeInTheDocument();
    expect(within(rows.find((r) => r.getAttribute("data-key") === "qwen3_0_6b")!).getByText("Busy")).toBeInTheDocument();

    expect(d.getByText(/What uses memory/)).toBeInTheDocument();
  });

  test.each([
    ["warn", "Busy: macOS is compressing memory."],
    ["critical", "Critical: the Mac is short of memory; unload something."],
  ] as const)("explains %s pressure in plain words", async (pressure, text) => {
    get.mockResolvedValue(snapshot({ pressure }));
    const { dialog } = await openPanel();
    expect(within(dialog).getByText(text)).toBeInTheDocument();
    expect(within(dialog).getByTestId("pressure-banner")).toHaveAttribute("data-level", pressure);
  });

  test("only models the server will unload get an Unload button", async () => {
    const { dialog } = await openPanel();
    expect(within(dialog).getAllByRole("button", { name: /^Unload / }).map((b) => b.getAttribute("aria-label"))).toEqual([
      "Unload Indian-language voice cloning (DhVaani)",
      "Unload Standard voice (Kokoro)",
    ]);
  });

  test("unloading asks first, then calls the API and re-reads", async () => {
    const { dialog } = await openPanel();
    fireEvent.click(within(dialog).getByRole("button", { name: "Unload Indian-language voice cloning (DhVaani)" }));
    expect(post).not.toHaveBeenCalled();
    const confirm = within(dialog).getByRole("group", { name: /Confirm unloading Indian-language voice cloning/ });
    expect(confirm).toHaveTextContent("loads again");

    const reads = get.mock.calls.length;
    fireEvent.click(within(confirm).getByRole("button", { name: /Confirm unload/ }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/system/models/dhvaani/unload"));
    await waitFor(() => expect(get.mock.calls.length).toBeGreaterThan(reads));
    expect(within(dialog).queryByRole("group", { name: /Confirm unloading/ })).toBeNull();
  });

  test("keeping the model loaded cancels the unload", async () => {
    const { dialog } = await openPanel();
    fireEvent.click(within(dialog).getByRole("button", { name: "Unload Standard voice (Kokoro)" }));
    fireEvent.click(within(dialog).getByRole("button", { name: "Keep it loaded" }));
    expect(post).not.toHaveBeenCalled();
    expect(within(dialog).queryByRole("group", { name: /Confirm unloading/ })).toBeNull();
  });

  test("a refusal from the server is shown in plain words", async () => {
    post.mockRejectedValue(new ApiError("This model is in use by a live call, so it cannot be unloaded now.", 409));
    const { dialog } = await openPanel();
    fireEvent.click(within(dialog).getByRole("button", { name: "Unload Standard voice (Kokoro)" }));
    fireEvent.click(within(dialog).getByRole("button", { name: /Confirm unload/ }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("This model is in use by a live call, so it cannot be unloaded now.");
  });

  test("an empty list says that models load on first use", async () => {
    get.mockResolvedValue(snapshot({}, []));
    const { dialog } = await openPanel();
    expect(within(dialog).getByText(/No models are loaded/)).toBeInTheDocument();
  });

  test("Escape closes the panel and focus returns to the footer button", async () => {
    const { dialog } = await openPanel();
    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(dialog).not.toBeInTheDocument());
    await waitFor(() => expect(screen.getByTestId("resource-indicator")).toHaveFocus());
  });
});
