/**
 * Logic of the website widget bundle (backend/widget/v1/bonclay-agent.js): the incremental SSE parser, the ordered
 * clip queue, and the small helpers. The bundle is a plain script that also exports them under CommonJS.
 */
// eslint-disable-next-line @typescript-eslint/no-require-imports
const w = require("../../../../backend/widget/v1/bonclay-agent.js");

const tick = () => new Promise((r) => setTimeout(r, 0));

describe("SSEParser", () => {
  it("returns complete events and keeps a partial one until it finishes", () => {
    const p = new w.SSEParser();
    expect(p.push('event: transcript\ndata: {"user_text":"hi"}\n\nevent: sent')).toEqual([
      { event: "transcript", data: { user_text: "hi" } },
    ]);
    expect(p.push('ence\ndata: {"index":0,"text":"Hello.","audio_url":"/a"}\n\n')).toEqual([
      { event: "sentence", data: { index: 0, text: "Hello.", audio_url: "/a" } },
    ]);
  });

  it("handles a chunk split in the middle of a line, CRLF, comments and several events at once", () => {
    const p = new w.SSEParser();
    const all = 'event: a\r\ndata: {"n":1}\r\n\r\n: keep-alive\n\nevent: b\ndata: {"n":2}\n\nevent: c\ndata: {"n":';
    const out = [...p.push(all.slice(0, 20)), ...p.push(all.slice(20)), ...p.push("3}\n\n")];
    expect(out.map((e: { event: string }) => e.event)).toEqual(["a", "b", "c"]);
    expect(out[2].data).toEqual({ n: 3 });
  });

  it("keeps non-ASCII text and joins multi-line data", () => {
    const p = new w.SSEParser();
    const out = p.push('event: sentence\ndata: {"text":"नमस्ते, कैसे हैं?"}\n\nevent: x\ndata: line1\ndata: line2\n\n');
    expect(out[0].data.text).toBe("नमस्ते, कैसे हैं?");
    expect(out[1].data).toEqual({ raw: "line1\nline2" });
  });
});

describe("AudioQueue", () => {
  const make = (over: Record<string, unknown> = {}) => {
    const log: string[] = [];
    const started: string[] = [];
    const q = new w.AudioQueue({
      fetchClip: (i: { text: string }) => {
        log.push(`fetch:${i.text}`);
        return Promise.resolve(`u:${i.text}`);
      },
      play: (url: string) => {
        log.push(`play:${url}`);
        return Promise.resolve();
      },
      onStart: (i: { text: string }) => started.push(i.text),
      onError: (_e: unknown, i: { text: string }) => log.push(`error:${i.text}`),
      ...over,
    });
    return { q, log, started };
  };

  it("plays clips in the order they were added, one at a time", async () => {
    const order: string[] = [];
    let release: () => void = () => undefined;
    const { q, started } = make({
      play: (url: string) => {
        order.push(`start:${url}`);
        return new Promise<void>((r) => {
          release = () => {
            order.push(`end:${url}`);
            r();
          };
        });
      },
    });
    q.add({ text: "one" });
    q.add({ text: "two" });
    q.add({ text: "three" });
    await tick();
    expect(order).toEqual(["start:u:one"]); // the second does not start until the first has ended
    release();
    await tick();
    release();
    await tick();
    release();
    await q.idle();
    expect(order).toEqual(["start:u:one", "end:u:one", "start:u:two", "end:u:two", "start:u:three", "end:u:three"]);
    expect(started).toEqual(["one", "two", "three"]);
  });

  it("fetches later clips while an earlier one plays", async () => {
    let release: () => void = () => undefined;
    const { q, log } = make({ play: () => new Promise<void>((r) => (release = r)) });
    q.add({ text: "one" });
    q.add({ text: "two" });
    await tick();
    expect(log).toEqual(["fetch:one", "fetch:two"]);
    release();
    await tick();
    release();
    await q.idle();
  });

  it("keeps going after a clip fails to load or play, and still announces its text", async () => {
    const { q, log, started } = make({
      fetchClip: (i: { text: string }) => (i.text === "bad" ? Promise.reject(new Error("404")) : Promise.resolve(`u:${i.text}`)),
    });
    q.add({ text: "bad" });
    q.add({ text: "good" });
    await q.idle();
    expect(log).toEqual(["error:bad", "play:u:good"]);
    expect(started).toEqual(["bad", "good"]);

    const { q: q2, log: log2 } = make({ play: () => Promise.reject(new Error("blocked")) });
    q2.add({ text: "x" });
    await q2.idle();
    expect(log2).toContain("error:x");
  });

  it("does not play a clip without audio, but shows its caption", async () => {
    const { q, log, started } = make({ fetchClip: () => Promise.resolve(null) });
    q.add({ text: "silent" });
    await q.idle();
    expect(started).toEqual(["silent"]);
    expect(log).toEqual([]);
  });

  it("stop() drops what is queued and lets a later turn play normally", async () => {
    let release: () => void = () => undefined;
    const { q, log, started } = make({ play: (url: string) => (log0.push(url), new Promise<void>((r) => (release = r))) });
    const log0: string[] = [];
    q.add({ text: "one" });
    q.add({ text: "two" });
    await tick();
    q.stop();
    release();
    await q.idle();
    await tick();
    expect(log0).toEqual(["u:one"]);
    expect(started).toEqual(["one"]);
    q.add({ text: "next" });
    await tick();
    release();
    await q.idle();
    expect(log0).toEqual(["u:one", "u:next"]);
    expect(log).toContain("fetch:next");
  });

  it("is idle at once when nothing was added", async () => {
    await expect(make().q.idle()).resolves.toBeUndefined();
  });
});

describe("helpers", () => {
  it("maps error codes to plain sentences and never leaks unknown codes", () => {
    expect(w.friendlyError("WIDGET_RATE_LIMITED")).toMatch(/try again in a few minutes/i);
    expect(w.friendlyError("NO_SPEECH")).toMatch(/did not catch/i);
    expect(w.friendlyError("SOMETHING_ODD")).toBe(w.friendlyError("generic"));
  });

  it("picks the upload extension from the recorder's type", () => {
    expect(w.extFor("audio/webm;codecs=opus")).toBe("webm");
    expect(w.extFor("audio/ogg;codecs=opus")).toBe("ogg");
    expect(w.extFor("audio/mp4")).toBe("m4a");
    expect(w.extFor("")).toBe("webm");
  });

  it("only accepts a hex accent colour and an http(s) API address", () => {
    expect(w.safeColor("#12abEF")).toBe("#12abEF");
    expect(w.safeColor("red;background:url(x)")).toBe("#3b5bdb");
    expect(w.safeApi("https://api.example.com/path?x=1")).toBe("https://api.example.com");
    expect(w.safeApi("javascript:alert(1)")).toBe("");
    expect(w.safeApi("not a url")).toBe("");
  });
});

describe("custom element", () => {
  it("registers bonclay-agent and renders a labelled launcher in a shadow root without starting anything", () => {
    expect(customElements.get("bonclay-agent")).toBeDefined();
    const el = document.createElement("bonclay-agent");
    el.setAttribute("agent", "awk_test");
    document.body.appendChild(el);
    const root = el.shadowRoot!;
    const launcher = root.querySelector(".launcher")!;
    expect(launcher.getAttribute("aria-label")).toBe("Talk to our AI assistant");
    expect((root.querySelector(".panel") as HTMLElement).hidden).toBe(true);
    expect(root.querySelector(".ai")!.textContent).toBe("AI-generated voice");
    expect(root.querySelector("[aria-live]")).not.toBeNull();
    el.remove();
  });
});
