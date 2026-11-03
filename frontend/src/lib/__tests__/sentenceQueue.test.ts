import { SentenceQueue, type StreamedSentence } from "@/lib/sentenceQueue";

const s = (index: number, total = 3): StreamedSentence => ({
  index,
  total,
  text: `Sentence ${index + 1}.`,
  audio_url: `/clip/${index}.wav`,
});

describe("SentenceQueue", () => {
  it("hands each sentence over as it arrives, in order", () => {
    const q = new SentenceQueue();
    expect(q.push(s(0)).map((x) => x.index)).toEqual([0]);
    expect(q.push(s(1)).map((x) => x.index)).toEqual([1]);
    expect(q.push(s(2)).map((x) => x.index)).toEqual([2]);
    expect(q.received).toBe(3);
  });

  it("holds a sentence that arrives before an earlier one, and releases both in order", () => {
    const q = new SentenceQueue();
    expect(q.push(s(1))).toEqual([]);
    expect(q.push(s(2))).toEqual([]);
    expect(q.push(s(0)).map((x) => x.index)).toEqual([0, 1, 2]);
  });

  it("ignores a repeated sentence event", () => {
    const q = new SentenceQueue();
    q.push(s(0));
    expect(q.push(s(0))).toEqual([]);
    expect(q.received).toBe(1);
  });

  it("ignores events without audio or with a bad index", () => {
    const q = new SentenceQueue();
    expect(q.push({ index: 0, text: "x", audio_url: "" })).toEqual([]);
    expect(q.push({ index: -1, text: "x", audio_url: "/a.wav" })).toEqual([]);
    expect(q.push({ index: 1.5, text: "x", audio_url: "/a.wav" })).toEqual([]);
    expect(q.received).toBe(0);
  });

  it("says which sentence is being rendered", () => {
    const q = new SentenceQueue();
    expect(q.label()).toBe("Rendering sentence 1");
    q.push(s(0));
    expect(q.label()).toBe("Rendering sentence 2 of 3");
    q.push(s(1));
    expect(q.label()).toBe("Rendering sentence 3 of 3");
    q.push(s(2));
    expect(q.label()).toBe("Finishing the clip");
  });

  it("is quiet once the final clip is in", () => {
    const q = new SentenceQueue();
    q.push(s(0, 1));
    q.finish();
    expect(q.label()).toBe("");
    expect(q.isFinished).toBe(true);
    expect(q.fraction()).toBe(1);
  });

  it("reports progress as a fraction of the sentences that arrived", () => {
    const q = new SentenceQueue();
    expect(q.fraction()).toBe(0);
    q.push({ ...s(0), total: undefined });
    expect(q.fraction()).toBe(0); // the total is not known yet
    q.push(s(1, 4));
    expect(q.total).toBe(4);
    expect(q.fraction()).toBe(0.5);
  });
});
