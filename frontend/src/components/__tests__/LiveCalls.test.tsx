import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import { LiveCalls } from "@/app/(app)/dashboard/calls/LiveCalls";
import { api } from "@/lib/api";
import { int16ToBase64 } from "@/lib/livePcm";

jest.mock("@/lib/api", () => ({ api: { get: jest.fn(), post: jest.fn() }, API_URL: "http://localhost:8000", ApiError: class ApiError extends Error {} }));

const played: unknown[][] = [];
const micStart = jest.fn();
const micStop = jest.fn();
jest.mock("@/lib/livePcm", () => {
  const real = jest.requireActual("@/lib/livePcm");
  return {
    ...real,
    PcmPlayer: class { async resume() {} play(...a: unknown[]) { played.push(a); } setMuted() {} close() {} },
    MicStreamer: class { async start(rate: number, cb: unknown) { return micStart(rate, cb); } stop() { micStop(); } },
  };
});

class MockWS {
  static OPEN = 1;
  static instances: MockWS[] = [];
  readyState = 1;
  sent: string[] = [];
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) { MockWS.instances.push(this); }
  send(d: string) { this.sent.push(d); }
  close() { this.readyState = 3; }
  emit(msg: object) { act(() => { this.onmessage?.({ data: JSON.stringify(msg) }); }); }
}

const call = { conversation_id: "conv-1", call_sid: "CA1", agent_id: "a1", agent_name: "Clinic desk", direction: "inbound", started_at: 1, seconds: 65, human_active: false, listeners: 0 };
const get = api.get as jest.Mock;
const post = api.post as jest.Mock;
const view = () => render(<SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}><LiveCalls /></SWRConfig>);

beforeEach(() => {
  MockWS.instances = [];
  played.length = 0;
  micStart.mockReset();
  micStop.mockReset();
  (globalThis as unknown as { WebSocket: unknown }).WebSocket = MockWS;
  get.mockReset().mockResolvedValue({ calls: [call] });
  post.mockReset().mockResolvedValue({ token: "tok123" });
});

test("shows nothing while no call is live", async () => {
  get.mockResolvedValue({ calls: [] });
  const { container } = view();
  await waitFor(() => expect(get).toHaveBeenCalled());
  expect(container).toBeEmptyDOMElement();
});

test("lists a live call with its length", async () => {
  view();
  expect(await screen.findByText("Clinic desk")).toBeInTheDocument();
  expect(screen.getByText(/Incoming call · 1:05/)).toBeInTheDocument();
});

test("listening in connects, plays the audio and shows the transcript", async () => {
  view();
  fireEvent.click(await screen.findByRole("button", { name: /Listen in/ }));
  await waitFor(() => expect(MockWS.instances).toHaveLength(1));
  expect(post).toHaveBeenCalledWith("/api/phone/exotel/live/conv-1/monitor-token", {});
  const ws = MockWS.instances[0];
  expect(ws.url).toBe("ws://localhost:8000/api/phone/exotel/monitor/tok123");
  ws.emit({ type: "hello", rate: 8000, human: false, call_sid: "CA1", agent: "Clinic desk" });
  expect(await screen.findByText("Listening")).toBeInTheDocument();
  ws.emit({ type: "transcript", role: "caller", text: "When does the clinic open?" });
  ws.emit({ type: "transcript", role: "agent", text: "We open at nine." });
  expect(screen.getByText("When does the clinic open?")).toBeInTheDocument();
  expect(screen.getByText("We open at nine.")).toBeInTheDocument();
  ws.emit({ type: "audio", from: "agent", rate: 8000, pcm: int16ToBase64(Int16Array.from([1, 2, 3])) });
  expect(played).toHaveLength(1);
  expect(played[0][0]).toBe("agent");
  expect(Array.from(played[0][1] as Int16Array)).toEqual([1, 2, 3]);
});

test("taking over starts the microphone, says so, and handing back asks the agent to return", async () => {
  view();
  fireEvent.click(await screen.findByRole("button", { name: /Listen in/ }));
  await waitFor(() => expect(MockWS.instances).toHaveLength(1));
  const ws = MockWS.instances[0];
  ws.emit({ type: "hello", rate: 8000, human: false });
  fireEvent.click(await screen.findByRole("button", { name: /Take over/ }));
  await waitFor(() => expect(micStart).toHaveBeenCalledWith(8000, expect.any(Function)));
  expect(JSON.parse(ws.sent[0])).toEqual({ type: "takeover" });
  ws.emit({ type: "state", human: true });
  expect(await screen.findByText("You have the call")).toBeInTheDocument();
  const sendMic = micStart.mock.calls[0][1] as (pcm: Int16Array) => void;
  sendMic(Int16Array.from([5, 6]));                                   // the person's voice goes to the call
  expect(JSON.parse(ws.sent[1])).toEqual({ type: "audio", pcm: int16ToBase64(Int16Array.from([5, 6])) });
  fireEvent.click(screen.getByRole("button", { name: /Hand back/ }));
  expect(JSON.parse(ws.sent[2])).toEqual({ type: "release" });
  ws.emit({ type: "state", human: false });
  await waitFor(() => expect(micStop).toHaveBeenCalled());
  expect(screen.queryByRole("button", { name: /Hand back/ })).toBeNull();
});

test("a refused microphone leaves the agent in charge and explains", async () => {
  micStart.mockRejectedValueOnce(new Error("denied"));
  view();
  fireEvent.click(await screen.findByRole("button", { name: /Listen in/ }));
  await waitFor(() => expect(MockWS.instances).toHaveLength(1));
  MockWS.instances[0].emit({ type: "hello", rate: 8000, human: false });
  fireEvent.click(await screen.findByRole("button", { name: /Take over/ }));
  expect(await screen.findByText("Allow the microphone to talk to the caller.")).toBeInTheDocument();
  expect(MockWS.instances[0].sent).toEqual([]);                       // no takeover was requested
});

test("when the call ends the panel says so", async () => {
  view();
  fireEvent.click(await screen.findByRole("button", { name: /Listen in/ }));
  await waitFor(() => expect(MockWS.instances).toHaveLength(1));
  MockWS.instances[0].emit({ type: "hello", rate: 8000, human: false });
  MockWS.instances[0].emit({ type: "ended" });
  expect(await screen.findByText("Call ended")).toBeInTheDocument();
});
