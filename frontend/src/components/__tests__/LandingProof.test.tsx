import fs from "fs";
import path from "path";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { SCRIPT_FONT } from "@/components/site/fonts";
import { Comparison, HearIt, Measured } from "@/components/site/Proof";
import { claimAudio } from "@/components/site/audioBus";
import { INDIC_ATTRIBUTION, LANGUAGE_CLIPS, MEASURED_SPEED, MEASURED_TRUST, RESULTS_URL } from "@/components/site/proofData";

beforeAll(() => {
  jest.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(() => undefined); // jsdom does not implement media
});

const PUBLIC = path.join(__dirname, "..", "..", "..", "public");

describe("landing language cards", () => {
  it("has one clip per language and every clip file exists and is small", () => {
    expect(LANGUAGE_CLIPS.map((c) => c.code)).toEqual(["en", "hi", "mr", "ta", "bn", "te", "fr", "es"]);
    for (const c of LANGUAGE_CLIPS) {
      const file = path.join(PUBLIC, c.src);
      expect(fs.existsSync(file)).toBe(true);
      expect(fs.statSync(file).size).toBeLessThan(120_000);
    }
  });

  it("uses preset voices only: never a cloned voice on the public page", () => {
    const engines = new Set(LANGUAGE_CLIPS.map((c) => c.engine));
    expect([...engines].sort()).toEqual(["indic_speak", "kokoro"]); // neither engine id is a cloning engine
    expect(fs.readdirSync(path.join(PUBLIC, "samples", "languages")).filter((f) => f.endsWith(".m4a")).length).toBe(8);
  });

  it("shows the Indic-Speak attribution on exactly the Indic-Speak cards", () => {
    render(<HearIt />);
    const cards = screen.getAllByRole("listitem");
    expect(cards).toHaveLength(8);
    for (const [i, c] of LANGUAGE_CLIPS.entries()) {
      const has = cards[i].textContent?.includes(INDIC_ATTRIBUTION);
      expect(has).toBe(c.engine === "indic_speak");
      expect(cards[i].textContent).toContain(c.text);
    }
    expect(screen.getByText(/Preset voices only/)).toBeTruthy();
  });

  it("claims the shared audio bus, shows loading then playing, and pauses on a second press", async () => {
    const play = jest.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
    render(<HearIt />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Play the Tamil clip" }));
    });
    expect(play).toHaveBeenCalled();
    // the browser has not started the clip yet: loading state, busy
    const loading = screen.getByRole("button", { name: "Loading the Tamil clip" });
    expect(loading.getAttribute("aria-busy")).toBe("true");
    // the clip starts: playing state
    await act(async () => {
      const el = (play.mock.contexts[0] as HTMLAudioElement);
      el.dispatchEvent(new Event("playing"));
    });
    const playing = screen.getByRole("button", { name: "Pause the Tamil clip" });
    expect(playing.getAttribute("aria-pressed")).toBe("true");
    expect(playing.className).toContain("h-12"); // 48 px control
    play.mockRestore();
  });

  it("pauses a clip another player has claimed", async () => {
    const play = jest.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
    render(<HearIt />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Play the Hindi clip" }));
    });
    const el = play.mock.contexts[0] as HTMLAudioElement;
    Object.defineProperty(el, "paused", { value: false, configurable: true });
    const other = new Audio();
    Object.defineProperty(other, "paused", { value: false, configurable: true });
    // HearIt's element is the bus owner after its own claim; a different player claiming must pause it
    const pause = window.HTMLMediaElement.prototype.pause as jest.Mock;
    pause.mockClear();
    claimAudio(other);
    expect(pause).toHaveBeenCalledTimes(1);
    play.mockRestore();
  });

  it("states one language claim in the heading", () => {
    render(<HearIt />);
    expect(screen.getByRole("heading", { level: 2, name: /22 Indian languages/ })).toHaveTextContent("22 Indian languages. Hear five.");
  });
});

/** Keeps IntersectionObserver callbacks, so a test can report a section as near the screen. */
const observed: { cb: IntersectionObserverCallback; el: Element[] }[] = [];
class FakeIntersectionObserver {
  readonly root = null;
  readonly rootMargin = "";
  readonly thresholds = [];
  private record: { cb: IntersectionObserverCallback; el: Element[] };
  constructor(cb: IntersectionObserverCallback) {
    this.record = { cb, el: [] };
    observed.push(this.record);
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

describe("hear-it clips load their script faces near the screen", () => {
  const realIO = window.IntersectionObserver;
  beforeAll(() => {
    window.IntersectionObserver = FakeIntersectionObserver as unknown as typeof IntersectionObserver;
  });
  afterAll(() => {
    window.IntersectionObserver = realIO;
  });
  beforeEach(() => {
    observed.length = 0;
  });

  it("shows the Hindi clip's native name in the page font until the section nears the screen, then in its script face", () => {
    const { container } = render(<HearIt />);
    const face = SCRIPT_FONT.hi;
    expect(face).toBeTruthy();
    const native = screen.getByText("हिन्दी");
    expect(native.classList.contains(face)).toBe(false);

    const section = container.querySelector("section#hear")!;
    const watcher = observed.find((o) => o.el.includes(section));
    expect(watcher).toBeTruthy();
    act(() => watcher!.cb([{ isIntersecting: true, target: section } as IntersectionObserverEntry], {} as IntersectionObserver));

    expect(native.classList.contains(face)).toBe(true);
  });
});

describe("landing numbers", () => {
  it("does not link the measured strip to the private repository, and states the caveat", () => {
    render(<Measured />);
    expect(screen.queryByRole("link", { name: /Every number/ })).toBeNull();
    expect(screen.getByText(/Every number, method and the commands to rerun them are in the project's results file, shared on request/)).toBeTruthy();
    expect(RESULTS_URL).toBe("https://github.com/Artifact19/bonclay/blob/main/docs/RESULTS.md");
    expect(screen.getByText(/Hindi is slower than English/)).toBeTruthy();
  });

  it("leads with the first sentence of the answer and splits Speed from Trust", () => {
    render(<Measured />);
    expect(screen.getByRole("heading", { name: "Speed" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Trust" })).toBeTruthy();
    expect(MEASURED_SPEED[0].label).toMatch(/first sentence of the answer/);
    expect(MEASURED_SPEED[0].value).toBe("2.23 s");
    expect(MEASURED_TRUST.map((m) => m.value)).toEqual(["97.8%", "70 / 70", "733 / 733", "1,390 + 404"]);
    // every headline numeral is marked for the count-up, and compound values count only their first number
    const counted = document.querySelectorAll("[data-countup]");
    expect(counted).toHaveLength(MEASURED_SPEED.length + MEASURED_TRUST.length);
    expect(document.querySelector("[data-reveal]")).toBeTruthy();
  });

  it("only shows figures that docs/RESULTS.md contains", () => {
    const results = fs.readFileSync(path.join(__dirname, "..", "..", "..", "..", "docs", "RESULTS.md"), "utf8");
    for (const needle of ["**0.84 s**", "**2.23 s**", "**3.58 s**", "**3.02 s**", "5.03 s", "full reply 2.81 s", "**97.8%**", "70/70 detected", "1,390 backend", "Warm p50 (n=9) | 2.4", "733 of 733", "1,115 ms target", "$0.055", "$0.11 per minute"]) {
      expect(results).toContain(needle);
    }
    expect(results).toContain("0.076 mean"); // English WER 7.6% (history table, cited in the caveat)
    expect(results).toContain("0.071 mean"); // Hindi CER 7.1%
  });
});

describe("landing comparison", () => {
  it("compares three columns with cited external figures and no 'Varies by provider' filler", () => {
    render(<Comparison />);
    expect(screen.getAllByText("BonClay (local)", { selector: "span" })[0]).toBeTruthy();
    expect(screen.getAllByText("BonClay (+ Groq)", { selector: "span" })[0]).toBeTruthy();
    expect(screen.queryByText(/Varies by provider/)).toBeNull();
    expect(screen.queryByText(/open source/i)).toBeNull();
    for (const c of ["Vapi pricing, Oct 2026", "Retell AI pricing, Oct 2026"]) {
      const a = screen.getAllByRole("link", { name: c })[0] as HTMLAnchorElement;
      expect(a.href).toMatch(/^https:\/\//);
    }
    expect(screen.getByRole("link", { name: /Twilio, Core Latency in AI Voice Agents, Nov 2025/ })).toBeTruthy();
    expect(screen.getByText(/not like-for-like/)).toBeTruthy();
  });

  it("says what BonClay does not do yet, and no longer denies phone calls or Hindi-only cloning", () => {
    render(<Comparison />);
    expect(screen.getByText("WhatsApp is not live-tested")).toBeTruthy();
    expect(screen.getByText("Phone calls: trial account only")).toBeTruthy();
    expect(screen.getByText("Windows is untested")).toBeTruthy();
    expect(screen.queryByText("No phone channel yet")).toBeNull();
    expect(screen.queryByText("Indian-language cloning is Hindi only")).toBeNull();
  });
});
