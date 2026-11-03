/**
 * @jest-environment node
 */
import { api, ApiError } from "@/lib/api";

jest.mock("@/lib/supabase", () => ({
  createClient: () => ({
    auth: { getSession: async () => ({ data: { session: { access_token: "tok" } } }) },
  }),
}));

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("api error handling", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });
  afterEach(() => {
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  it("normalizes backend {error,code,details}", async () => {
    global.fetch = jest.fn().mockResolvedValue(json(400, { error: "bad", code: "E_BAD", details: { f: 1 } }));
    await expect(api.post("/x", {})).rejects.toMatchObject({
      status: 400,
      code: "E_BAD",
      message: "bad",
      details: { f: 1 },
    });
  });

  it("normalizes FastAPI {detail}", async () => {
    global.fetch = jest.fn().mockResolvedValue(json(404, { detail: "Not found" }));
    const err = (await api.get("/x").catch((e: ApiError) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(404);
    expect(err.message).toBe("Not found");
  });

  it("sends bearer token", async () => {
    const f = jest.fn().mockResolvedValue(json(200, { ok: true }));
    global.fetch = f;
    const g = globalThis as unknown as Record<string, unknown>;
    g.window = {};
    g.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
    try {
      await api.get("/x");
    } finally {
      delete g.window;
      delete g.localStorage;
    }
    const h = f.mock.calls[0][1].headers;
    expect(h.Authorization).toBe("Bearer tok");
    expect(h["Storage-Mode"]).toBe("local");
  });

  it("retries GET on 5xx then succeeds", async () => {
    const f = jest
      .fn()
      .mockResolvedValueOnce(json(503, { detail: "down" }))
      .mockResolvedValueOnce(json(200, { ok: 1 }));
    global.fetch = f;
    const p = api.get("/x");
    await jest.runAllTimersAsync();
    await expect(p).resolves.toEqual({ ok: 1 });
    expect(f).toHaveBeenCalledTimes(2);
  });

  it("does not retry POST on 5xx", async () => {
    const f = jest.fn().mockResolvedValue(json(500, { detail: "boom" }));
    global.fetch = f;
    await expect(api.post("/x")).rejects.toMatchObject({ status: 500 });
    expect(f).toHaveBeenCalledTimes(1);
  });

  it("does not retry GET on 4xx", async () => {
    const f = jest.fn().mockResolvedValue(json(401, { detail: "no" }));
    global.fetch = f;
    await expect(api.get("/x")).rejects.toMatchObject({ status: 401 });
    expect(f).toHaveBeenCalledTimes(1);
  });

  it("wraps network failures with code network_error after 3 retries", async () => {
    const f = jest.fn().mockRejectedValue(new TypeError("fail"));
    global.fetch = f;
    const p = api.get("/x").catch((e: ApiError) => e);
    await jest.runAllTimersAsync();
    const err = (await p) as ApiError;
    expect(err.code).toBe("network_error");
    expect(f).toHaveBeenCalledTimes(4);
  });
});
