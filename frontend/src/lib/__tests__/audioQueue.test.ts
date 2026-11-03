import { AudioQueue, type QueueAudio } from "@/lib/audioQueue";

class FakeAudio implements QueueAudio {
  preload = "";
  played = false;
  paused = false;
  private handlers: Record<string, (() => void)[]> = {};
  constructor(
    public src: string,
    private failPlay = false
  ) {}
  play() {
    this.played = true;
    return this.failPlay ? Promise.reject(new Error("blocked")) : Promise.resolve();
  }
  pause() {
    this.paused = true;
  }
  addEventListener(type: string, fn: () => void) {
    (this.handlers[type] ||= []).push(fn);
  }
  fire(type: string) {
    (this.handlers[type] || []).forEach((f) => f());
  }
}

const setup = (failing: string[] = []) => {
  const made: FakeAudio[] = [];
  const q = new AudioQueue((url) => {
    const a = new FakeAudio(url, failing.includes(url));
    made.push(a);
    return a;
  });
  return { q, made };
};

describe("AudioQueue", () => {
  it("plays the first clip at once and the rest one after another, in order", () => {
    const { q, made } = setup();
    q.add("a");
    expect(made.map((m) => m.played)).toEqual([true]);
    q.add("b");
    q.add("c");
    expect(made.map((m) => m.preload)).toEqual(["auto", "auto", "auto"]); // all preloaded
    expect(made.map((m) => m.played)).toEqual([true, false, false]); // but only one plays
    made[0].fire("ended");
    expect(made.map((m) => m.played)).toEqual([true, true, false]);
    made[1].fire("ended");
    expect(made.map((m) => m.played)).toEqual([true, true, true]);
    expect(q.busy).toBe(true);
    made[2].fire("ended");
    expect(q.busy).toBe(false);
  });

  it("continues with a clip that arrives after the queue went idle", () => {
    const { q, made } = setup();
    q.add("a");
    made[0].fire("ended");
    expect(q.busy).toBe(false);
    q.add("b");
    expect(made[1].played).toBe(true);
  });

  it("skips a clip that fails and counts it", async () => {
    const { q, made } = setup(["a"]);
    q.add("a");
    q.add("b");
    await Promise.resolve();
    await Promise.resolve();
    expect(made[1].played).toBe(true);
    made[1].fire("error");
    expect(q.failures).toBe(2);
    expect(q.busy).toBe(false);
  });

  it("finished() resolves after close() once the last clip has played", async () => {
    const { q, made } = setup();
    q.add("a");
    q.add("b");
    let done = false;
    const p = q.finished().then(() => (done = true));
    q.close();
    await Promise.resolve();
    expect(done).toBe(false);
    made[0].fire("ended");
    made[1].fire("ended");
    await p;
    expect(done).toBe(true);
  });

  it("stop() pauses everything, ignores later clips and releases waiters", async () => {
    const { q, made } = setup();
    q.add("a");
    q.add("b");
    q.close();
    const p = q.finished();
    q.stop();
    await p;
    expect(made.every((m) => m.paused)).toBe(true);
    q.add("c");
    expect(made).toHaveLength(2);
    made[0].fire("ended"); // a late event from a stopped clip does nothing
    expect(made[1].played).toBe(false);
    expect(q.busy).toBe(false);
  });
});
