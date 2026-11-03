import { act, fireEvent, render, screen } from "@testing-library/react";
import { VoiceReel } from "@/components/site/VoiceReel";
import { claimAudio } from "@/components/site/audioBus";
import { SCRIPT_FONT } from "@/components/site/fonts";

let reducedMotion = false;
jest.mock("motion/react", () => ({ ...jest.requireActual("motion/react"), useReducedMotion: () => reducedMotion }));
jest.mock("@/components/site/audioBus", () => ({ claimAudio: jest.fn() }));

class FakeAudio extends EventTarget {
  static last: FakeAudio | null = null;
  src = "";
  currentTime = 0;
  duration = 5;
  paused = true;
  ended = false;
  preload = "";
  constructor() {
    super();
    FakeAudio.last = this;
  }
  play() {
    this.paused = false;
    return Promise.resolve();
  }
  pause() {
    if (this.paused) return;
    this.paused = true;
    this.dispatchEvent(new Event("pause"));
  }
}

const created: unknown[] = [];
class FakeCtx {
  state = "running";
  constructor() {
    created.push(this);
  }
  resume() {
    return Promise.resolve();
  }
  close() {
    return Promise.resolve();
  }
  createMediaElementSource() {
    return { connect: jest.fn() };
  }
  createAnalyser() {
    return { fftSize: 0, smoothingTimeConstant: 0, frequencyBinCount: 128, connect: jest.fn(), getByteFrequencyData: jest.fn() };
  }
  destination = {};
}

const setReducedMotion = (on: boolean) => {
  reducedMotion = on;
};

const flush = () => act(async () => {});
const audio = () => FakeAudio.last as FakeAudio;

beforeEach(() => {
  created.length = 0;
  FakeAudio.last = null;
  (global as unknown as { Audio: unknown }).Audio = FakeAudio;
  (window as unknown as { AudioContext: unknown }).AudioContext = FakeCtx;
  setReducedMotion(false);
  jest.spyOn(window, "requestAnimationFrame").mockImplementation(() => 1);
  jest.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});
});
afterEach(() => {
  jest.restoreAllMocks();
  (claimAudio as jest.Mock).mockReset();
});

describe("VoiceReel", () => {
  it("makes the main play control at least 48 px and every voice row at least 48 px", () => {
    render(<VoiceReel />);
    const play = screen.getByRole("button", { name: /^Play Heart/ });
    expect(play.className).toContain("h-14"); // 56 px
    for (const row of screen.getAllByRole("button", { pressed: false }).concat(screen.getAllByRole("button", { pressed: true }))) {
      expect(row.className).toMatch(/min-h-\[3\.25rem\]/);
    }
  });

  it("does not create an AudioContext until the first click", async () => {
    render(<VoiceReel />);
    expect(created).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: /^Play Heart/ }));
    await flush();
    expect(created).toHaveLength(1);
    await act(async () => audio().dispatchEvent(new Event("playing")));
    fireEvent.click(screen.getByRole("button", { name: /^Pause Heart/ }));
    await act(async () => audio().pause());
    fireEvent.click(screen.getByRole("button", { name: /^Play Heart/ }));
    await flush();
    expect(created).toHaveLength(1); // never a second one
  });

  it("claims the audio bus before playing, and shows loading then playing", async () => {
    const order: string[] = [];
    (claimAudio as jest.Mock).mockImplementation(() => void order.push("claim"));
    render(<VoiceReel />);
    const original = FakeAudio.prototype.play;
    jest.spyOn(FakeAudio.prototype, "play").mockImplementation(function (this: FakeAudio) {
      order.push("play");
      return original.call(this);
    });
    fireEvent.click(screen.getByRole("button", { name: /^Play Heart/ }));
    await flush();
    expect(order).toEqual(["claim", "play"]);
    expect(screen.getByTestId("reel-status").textContent).toBe("Loading");
    expect(screen.getByRole("button", { name: /Loading Heart/ }).getAttribute("aria-busy")).toBe("true");
    await act(async () => audio().dispatchEvent(new Event("playing")));
    expect(screen.getByTestId("reel-status").textContent).toBe("Playing");
    expect(screen.getByRole("button", { name: /^Pause Heart/ })).toBeTruthy();
    expect(screen.getByTestId("reel-ring").getAttribute("data-playing")).toBe("true");
  });

  it("updates when another sample pauses this one", async () => {
    render(<VoiceReel />);
    fireEvent.click(screen.getByRole("button", { name: /^Play Heart/ }));
    await flush();
    await act(async () => audio().dispatchEvent(new Event("playing")));
    await act(async () => audio().pause()); // what claimAudio() does from another player
    expect(screen.getByTestId("reel-status").textContent).toBe("Tap play");
    expect(screen.getByRole("button", { name: /^Play Heart/ })).toBeTruthy();
  });

  it("plays normally when Web Audio is blocked or throws", async () => {
    (window as unknown as { AudioContext: unknown }).AudioContext = class {
      constructor() {
        throw new Error("blocked");
      }
    };
    render(<VoiceReel />);
    fireEvent.click(screen.getByRole("button", { name: /^Play Heart/ }));
    await flush();
    await act(async () => audio().dispatchEvent(new Event("playing")));
    expect(audio().paused).toBe(false);
    expect(screen.getByTestId("reel-status").textContent).toBe("Playing");
  });

  it("skips Web Audio under reduced motion and still shows a static playing indicator", async () => {
    setReducedMotion(true);
    render(<VoiceReel />);
    fireEvent.click(screen.getByRole("button", { name: /^Play Heart/ }));
    await flush();
    await act(async () => audio().dispatchEvent(new Event("playing")));
    expect(created).toHaveLength(0);
    expect(screen.getByTestId("reel-ring").getAttribute("data-playing")).toBe("true");
    expect(screen.getByTestId("reel-status").textContent).toBe("Playing");
  });
});

describe("VoiceReel script face", () => {
  const realIO = window.IntersectionObserver;
  const watchers: { cb: IntersectionObserverCallback; el: Element[] }[] = [];
  class FakeIntersectionObserver {
    readonly root = null;
    readonly rootMargin = "";
    readonly thresholds = [];
    private record: { cb: IntersectionObserverCallback; el: Element[] };
    constructor(cb: IntersectionObserverCallback) {
      this.record = { cb, el: [] };
      watchers.push(this.record);
    }
    observe(el: Element) {
      this.record.el.push(el);
    }
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  }
  beforeAll(() => {
    window.IntersectionObserver = FakeIntersectionObserver as unknown as typeof IntersectionObserver;
  });
  afterAll(() => {
    window.IntersectionObserver = realIO;
  });
  beforeEach(() => {
    watchers.length = 0;
  });

  it("gives the Hindi voice labels their script face only once the first Hindi label is on screen", () => {
    render(<VoiceReel />);
    const face = SCRIPT_FONT.hi;
    expect(face).toBeTruthy();
    const labels = screen.getAllByText("हिन्दी"); // two Hindi voices in the list
    expect(labels.length).toBe(2);
    expect(labels.every((l) => !l.classList.contains(face))).toBe(true);

    const watcher = watchers.find((w) => w.el.includes(labels[0]));
    expect(watcher).toBeTruthy();
    act(() => watcher!.cb([{ isIntersecting: true, target: labels[0] } as unknown as IntersectionObserverEntry], {} as IntersectionObserver));

    expect(labels.every((l) => l.classList.contains(face))).toBe(true);
  });
});
