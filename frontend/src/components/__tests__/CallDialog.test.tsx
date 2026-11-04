import { act, fireEvent, render, screen } from "@testing-library/react";
import { CallDialog, looksLikeNumber, normaliseNumber } from "@/components/agents/CallDialog";
import { api } from "@/lib/api";
import type { ExotelStatus } from "@/types/api";

jest.mock("@/lib/api", () => {
  class ApiError extends Error {
    status: number;
    code: string;
    constructor(message: string, status = 0, code = "unknown_error") {
      super(message);
      this.status = status;
      this.code = code;
    }
  }
  return { api: { get: jest.fn(), post: jest.fn(), del: jest.fn() }, ApiError, API_URL: "http://api.test" };
});

const { ApiError } = jest.requireMock("@/lib/api") as { ApiError: new (m: string, s?: number, c?: string) => Error };
const get = api.get as jest.Mock;
const post = api.post as jest.Mock;
const del = api.del as jest.Mock;

const AGENT = { id: "a1", name: "Sunrise Clinic" };
const READY: ExotelStatus = {
  configured: true,
  missing: [],
  allowed_numbers: [
    { number: "+919876543210", masked: "+91•••••43210", label: "Arnav", source: "saved", id: "n1" },
    { number: "+919000000000", masked: "+91•••••00000", label: "Allowed in backend/.env", source: "env", id: null },
  ],
  allowed_numbers_set: true,
  caller_id_masked: "+91•••••91899",
  inbound_ready: true,
  public_base_url: "https://abc.ngrok.app",
  max_call_seconds: 600,
};

/** GET answers from `status` and, for the call, from the queue of states (the last one repeats). */
function mockGet(status: ExotelStatus, states: Record<string, unknown>[] = []) {
  const queue = [...states];
  get.mockImplementation(async (path: string) => {
    if (path === "/api/phone/exotel/status") return status;
    if (path.startsWith("/api/phone/exotel/call/")) return queue.length > 1 ? queue.shift() : queue[0];
    throw new Error(`unexpected ${path}`);
  });
}

async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

/** The dialog opens on "Receive calls"; most tests want the other tab. */
async function showPhoneTab() {
  fireEvent.click(await screen.findByRole("tab", { name: "Phone someone" }));
}

async function openReady(states: Record<string, unknown>[] = []) {
  mockGet(READY, states);
  render(<CallDialog agent={AGENT} open onClose={jest.fn()} />);
  await showPhoneTab();
  await screen.findByLabelText("Number to call");
}

const callButton = () => screen.getByRole("button", { name: /^(Call|Call again)$/ });
const CONSENT = "This person has agreed to get a call from an AI agent";
const tick = (ms: number) => act(async () => { await jest.advanceTimersByTimeAsync(ms); });

beforeEach(() => {
  jest.clearAllMocks();
});
afterEach(() => {
  jest.useRealTimers();
});

describe("CallDialog setup", () => {
  it("opens on Receive calls and shows plain wording, no developer words", async () => {
    mockGet(READY);
    render(<CallDialog agent={AGENT} open onClose={jest.fn()} />);
    expect(await screen.findByText("Phone line connected")).toBeInTheDocument();
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual(["Receive calls", "Phone someone"]);
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel")).toBeInTheDocument();
    const text = document.body.textContent ?? "";
    for (const word of ["backend/.env", "restart", "Exotel", "Voicebot", "ExoPhone", "applet"]) {
      expect(text).not.toContain(word);
    }
    fireEvent.click(screen.getByRole("tab", { name: "Phone someone" }));
    expect(screen.getByRole("tab", { name: "Phone someone" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("Number to call")).toBeInTheDocument();
    const text2 = document.body.textContent ?? "";
    for (const word of ["backend/.env", "restart", "Exotel", "Voicebot", "ExoPhone", "applet"]) {
      expect(text2).not.toContain(word);
    }
  });

  it("moves between the tabs with the arrow keys", async () => {
    mockGet(READY);
    render(<CallDialog agent={AGENT} open onClose={jest.fn()} />);
    const receive = await screen.findByRole("tab", { name: "Receive calls" });
    fireEvent.keyDown(receive, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Phone someone" })).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(screen.getByRole("tab", { name: "Phone someone" }), { key: "Home" });
    expect(screen.getByRole("tab", { name: "Receive calls" })).toHaveAttribute("aria-selected", "true");
  });

  it("says the phone line is not connected and keeps the missing settings behind Details", async () => {
    mockGet({ ...READY, configured: false, missing: ["EXOTEL_API_KEY", "PUBLIC_BASE_URL"], allowed_numbers: [], allowed_numbers_set: false });
    render(<CallDialog agent={AGENT} open onClose={jest.fn()} />);
    expect(await screen.findByText("Phone line not connected yet")).toBeInTheDocument();
    expect(screen.getByTestId("receive-ready")).toHaveTextContent("Not ready to receive calls yet");
    expect(screen.queryByText("EXOTEL_API_KEY")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Details/ }));
    const list = screen.getByRole("list", { name: "Missing settings" });
    expect(list).toHaveTextContent("EXOTEL_API_KEY");
    expect(list).toHaveTextContent("PUBLIC_BASE_URL");
    expect(screen.getByText(/backend\/\.env and restart/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Get the link" })).toBeDisabled();

    fireEvent.click(screen.getByRole("tab", { name: "Phone someone" }));
    expect(screen.getByText(/phone line is not connected yet/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Number to call")).not.toBeInTheDocument();
    expect(screen.queryByTestId("call-footer")).not.toBeInTheDocument();
  });

  it("opens the add form instead of a free-text field when the list is empty, and keeps the button off", async () => {
    mockGet({ ...READY, allowed_numbers: [], allowed_numbers_set: false });
    render(<CallDialog agent={AGENT} open onClose={jest.fn()} />);
    await showPhoneTab();
    expect(await screen.findByText("Add the number you want to call. Only saved numbers can be called.")).toBeInTheDocument();
    expect(screen.queryByLabelText("Number to call")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Name")).toBeInTheDocument();
    expect(screen.getByLabelText("Phone number")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Call/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByLabelText(CONSENT));
    expect(screen.getByRole("button", { name: "Save number" })).toBeDisabled();   // still needs a name and number
  });

  it("accepts the 0 and 91 forms of an Indian number", () => {
    for (const raw of ["9876543210", "09876543210", "919876543210", "+91 98765-43210"]) {
      expect(normaliseNumber(raw)).toBe("+919876543210");
    }
    expect(normaliseNumber("+1 415 555 0132")).toBe("+14155550132");
    expect(normaliseNumber("abc")).toBeNull();
    expect(normaliseNumber("+0123456789")).toBeNull();
  });

  it("has exactly one consent checkbox and a sticky Call footer", async () => {
    await openReady();
    expect(screen.getAllByRole("checkbox")).toHaveLength(1);
    expect(screen.getByLabelText(CONSENT)).toBeInTheDocument();
    const footer = screen.getByTestId("call-footer");
    expect(footer.className).toContain("sticky");
    expect(footer).toContainElement(callButton());
    expect(footer).toHaveTextContent("Tick the box above to confirm consent.");
  });

  it("offers the allowed numbers masked in a select, and keeps Call off until the person agreed", async () => {
    await openReady();
    const select = screen.getByLabelText("Number to call") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual(["Arnav, +91•••••43210", "Pre-approved number, +91•••••00000"]);
    expect(select.value).toBe("+919876543210");
    expect(screen.queryByText("+919876543210")).not.toBeInTheDocument();
    expect(screen.queryByText(/Allowed in backend/)).not.toBeInTheDocument();
    expect(callButton()).toBeDisabled();
    fireEvent.click(screen.getByLabelText(CONSENT));
    expect(callButton()).toBeEnabled();
    fireEvent.click(screen.getByLabelText(CONSENT));
    expect(callButton()).toBeDisabled();
  });
});

describe("CallDialog consent", () => {
  it("asks again when a different number is chosen", async () => {
    await openReady();
    fireEvent.click(screen.getByLabelText(CONSENT));
    expect(callButton()).toBeEnabled();
    fireEvent.change(screen.getByLabelText("Number to call"), { target: { value: "+919000000000" } });
    fireEvent.change(screen.getByLabelText("Number to call"), { target: { value: "+919876543210" } });
    expect(screen.getByLabelText(CONSENT)).not.toBeChecked();
    expect(callButton()).toBeDisabled();
  });

  it("treats a number from backend/.env as pre-approved: ticked, locked, callable at once", async () => {
    await openReady();
    fireEvent.change(screen.getByLabelText("Number to call"), { target: { value: "+919000000000" } });
    const box = screen.getByLabelText(CONSENT);
    expect(box).toBeChecked();
    expect(box).toBeDisabled();
    expect(screen.getByText(/Pre-approved: confirmed when the number was added to backend\/\.env/)).toBeInTheDocument();
    expect(callButton()).toBeEnabled();
  });

  it("never puts a full number from the server on screen", async () => {
    await openReady();
    expect(document.body.textContent).not.toMatch(/\d{10}/);
  });
});

describe("CallDialog calling", () => {
  it("places the call, then shows Calling, Ringing, Connected with a timer and Ended with the duration", async () => {
    jest.useFakeTimers();
    await openReady([
      { call_sid: "CA1", status: "ringing", duration_s: null },
      { call_sid: "CA1", status: "in-progress", duration_s: null },
      { call_sid: "CA1", status: "in-progress", duration_s: null },
      { call_sid: "CA1", status: "completed", duration_s: 75 },
    ]);
    post.mockResolvedValue({ status: "calling", to: "98*****3210", call_sid: "CA1", agent_id: "a1" });
    fireEvent.change(screen.getByLabelText("Number to call"), { target: { value: "+919000000000" } });
    expect(screen.getByLabelText(CONSENT)).toBeChecked();   // pre-approved in backend/.env: no tick needed
    fireEvent.click(callButton());
    await flush();
    expect(post).toHaveBeenCalledWith("/api/phone/exotel/call", { agent_id: "a1", to: "+919000000000", caller_agreed: true });
    const status = screen.getByTestId("call-status");
    expect(status).toHaveTextContent("Calling…");
    expect(status).toHaveAttribute("aria-live", "polite");

    await tick(2000);
    expect(status).toHaveTextContent("Ringing");
    expect(get).toHaveBeenCalledWith("/api/phone/exotel/call/CA1");

    await tick(2000);
    expect(status).toHaveTextContent("Connected 0:00");
    await tick(3000);
    expect(status).toHaveTextContent(/Connected 0:0[3-5]/);

    await tick(2000);
    expect(status).toHaveTextContent("Ended (1:15)");
    expect(callButton()).toHaveTextContent("Call again");
    expect(callButton()).toBeEnabled();            // a pre-approved (.env) number can be called again at once
    fireEvent.change(screen.getByLabelText("Number to call"), { target: { value: "+919876543210" } });
    expect(callButton()).toBeDisabled();           // a saved number: consent is per call, the box is ticked again
    fireEvent.click(screen.getByLabelText(CONSENT));
    expect(callButton()).toBeEnabled();
  });

  it("stops polling once the call has ended", async () => {
    jest.useFakeTimers();
    await openReady([{ call_sid: "CA1", status: "completed", duration_s: 5 }]);
    post.mockResolvedValue({ status: "calling", to: "x", call_sid: "CA1", agent_id: "a1" });
    fireEvent.click(screen.getByLabelText(CONSENT));
    fireEvent.click(callButton());
    await flush();
    await tick(2000);
    const polls = get.mock.calls.filter((c) => String(c[0]).includes("/call/")).length;
    await tick(10000);
    expect(get.mock.calls.filter((c) => String(c[0]).includes("/call/")).length).toBe(polls);
  });

  it("stops polling when the dialog closes", async () => {
    jest.useFakeTimers();
    mockGet(READY, [{ call_sid: "CA1", status: "ringing", duration_s: null }]);
    const { rerender } = render(<CallDialog agent={AGENT} open onClose={jest.fn()} />);
    await showPhoneTab();
    await screen.findByLabelText("Number to call");
    post.mockResolvedValue({ status: "calling", to: "x", call_sid: "CA1", agent_id: "a1" });
    fireEvent.click(screen.getByLabelText(CONSENT));
    fireEvent.click(callButton());
    await flush();
    await tick(2000);
    const polls = get.mock.calls.filter((c) => String(c[0]).includes("/call/")).length;
    expect(polls).toBeGreaterThan(0);
    rerender(<CallDialog agent={AGENT} open={false} onClose={jest.fn()} />);
    await tick(10000);
    expect(get.mock.calls.filter((c) => String(c[0]).includes("/call/")).length).toBe(polls);
  });

  it.each([
    ["busy", "Busy"],
    ["no-answer", "No answer"],
  ])("shows %s in plain words", async (state, words) => {
    jest.useFakeTimers();
    await openReady([{ call_sid: "CA1", status: state, duration_s: 0 }]);
    post.mockResolvedValue({ status: "calling", to: "x", call_sid: "CA1", agent_id: "a1" });
    fireEvent.click(screen.getByLabelText(CONSENT));
    fireEvent.click(callButton());
    await flush();
    await tick(2000);
    expect(screen.getByTestId("call-status")).toHaveTextContent(words);
  });

  it("shows the provider's message when the call failed", async () => {
    jest.useFakeTimers();
    await openReady([{ call_sid: "CA1", status: "failed", duration_s: 0, message: "The number is switched off" }]);
    post.mockResolvedValue({ status: "calling", to: "x", call_sid: "CA1", agent_id: "a1" });
    fireEvent.click(screen.getByLabelText(CONSENT));
    fireEvent.click(callButton());
    await flush();
    await tick(2000);
    expect(screen.getByTestId("call-status")).toHaveTextContent("Failed: The number is switched off");
  });
});

describe("CallDialog errors", () => {
  it.each([
    ["CONSENT_REQUIRED", "Only call someone who agreed"],
    ["NUMBER_NOT_ALLOWED", "That number is not saved. Add it first."],
    ["PHONE_NOT_CONFIGURED", "phone line is not connected yet"],
    ["RATE_LIMITED", "Too many calls this hour"],
    ["EXOTEL_REFUSED", "phone provider refused the call"],
    ["EXOTEL_UNREACHABLE", "Could not reach the phone provider"],
  ])("explains %s", async (code, words) => {
    await openReady();
    post.mockRejectedValue(new ApiError("server words", 400, code));
    fireEvent.click(screen.getByLabelText(CONSENT));
    fireEvent.click(callButton());
    expect(await screen.findByRole("alert")).toHaveTextContent(words);
    expect(callButton()).toBeEnabled();
  });
});

describe("CallDialog receive calls", () => {
  const openReceive = async (status: ExotelStatus = READY) => {
    mockGet(status);
    render(<CallDialog agent={AGENT} open onClose={jest.fn()} />);
    await screen.findByTestId("line-status");
  };

  it("states the line is ready, fetches the link on demand and copies it", async () => {
    await openReceive();
    const writeText = jest.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    post.mockResolvedValue({ stream_url: "wss://abc.ngrok.app/api/phone/exotel/stream/tok.en", valid_days: 30 });

    expect(screen.getByTestId("receive-ready")).toHaveTextContent("Ready to connect your number");
    expect(screen.getByRole("list", { name: "Steps to receive calls" }).querySelectorAll(":scope > li")).toHaveLength(3);
    expect(post).not.toHaveBeenCalled();           // nothing is created until asked
    expect(screen.getByText("+91•••••91899")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Get the link" }));
    const field = (await screen.findByLabelText("agent phone link")) as HTMLInputElement;
    expect(post).toHaveBeenCalledWith("/api/phone/exotel/inbound-link", { agent_id: "a1" });
    expect(field.value).toBe("wss://abc.ngrok.app/api/phone/exotel/stream/tok.en");

    fireEvent.click(screen.getByRole("button", { name: "Copy agent phone link" }));
    await flush();
    expect(writeText).toHaveBeenCalledWith("wss://abc.ngrok.app/api/phone/exotel/stream/tok.en");
    expect(screen.getByRole("button", { name: "Copy agent phone link" })).toHaveTextContent("Copied");

    // the link survives a trip to the other tab
    fireEvent.click(screen.getByRole("tab", { name: "Phone someone" }));
    fireEvent.click(screen.getByRole("tab", { name: "Receive calls" }));
    expect((screen.getByLabelText("agent phone link") as HTMLInputElement).value).toContain("tok.en");
  });

  it("keeps the vendor steps behind Details", async () => {
    await openReceive();
    expect(screen.queryByText(/Voicebot applet/)).not.toBeInTheDocument();
    const toggle = screen.getByRole("button", { name: /Details/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/Voicebot applet/)).toBeInTheDocument();
  });

  it("says it is not ready, and explains in Details, when the server has no public address", async () => {
    await openReceive({ ...READY, inbound_ready: false });
    expect(screen.getByTestId("receive-ready")).toHaveTextContent("Not ready to receive calls yet");
    expect(screen.getByRole("button", { name: "Get the link" })).toBeDisabled();
    expect(screen.queryByText("PUBLIC_BASE_URL")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Details/ }));
    expect(screen.getByText("PUBLIC_BASE_URL")).toBeInTheDocument();
  });
});

describe("CallDialog saved numbers", () => {
  const SAVED = { id: "n2", label: "Riya", number: "+919812345678", masked: "+91•••••45678", source: "saved" };
  const fill = (label: string, num: string, agree = true) => {
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: label } });
    fireEvent.change(screen.getByLabelText("Phone number"), { target: { value: num } });
    if (agree) fireEvent.click(screen.getByLabelText(CONSENT));
  };
  const save = () => screen.getByRole("button", { name: "Save number" });

  it("adds a number, refreshes the list and selects it", async () => {
    await openReady();
    fireEvent.click(screen.getByRole("button", { name: /Add a number/ }));
    fill(" Riya ", "98123 45678");
    post.mockResolvedValue(SAVED);
    const after = { ...READY, allowed_numbers: [...READY.allowed_numbers, { ...SAVED }] } as ExotelStatus;
    mockGet(after);
    fireEvent.click(save());
    await flush();
    await flush();
    expect(post).toHaveBeenCalledWith("/api/phone/exotel/numbers", { label: "Riya", number: "98123 45678", agreed: true });
    expect((screen.getByLabelText("Number to call") as HTMLSelectElement).value).toBe("+919812345678");
    expect(screen.queryByLabelText("Name")).not.toBeInTheDocument();
    expect(screen.getByTestId("numbers-status")).toHaveTextContent("Saved Riya.");
    expect(screen.getByTestId("numbers-status")).toHaveAttribute("aria-live", "polite");
  });

  it("needs the agreed box ticked before saving", async () => {
    await openReady();
    fireEvent.click(screen.getByRole("button", { name: /Add a number/ }));
    fill("Riya", "+91 98123 45678", false);
    expect(save()).toBeDisabled();
    fireEvent.click(screen.getByLabelText(CONSENT));
    expect(save()).toBeEnabled();
  });

  it("checks the name and number loosely", async () => {
    expect(looksLikeNumber("+91 (98123) 45-678")).toBe(true);
    expect(looksLikeNumber("98123.45678")).toBe(true);
    expect(looksLikeNumber("98123")).toBe(false);
    expect(looksLikeNumber("call me")).toBe(false);
    await openReady();
    fireEvent.click(screen.getByRole("button", { name: /Add a number/ }));
    fill("Riya", "12345");
    expect(save()).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Phone number"), { target: { value: "9812345678" } });
    expect(save()).toBeEnabled();
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "   " } });
    expect(save()).toBeDisabled();
  });

  it.each([
    ["CONSENT_REQUIRED", "agreed to get a call"],
    ["BAD_NUMBER", "doesn't look like a phone number"],
    ["NUMBER_EXISTS", "already saved"],
    ["TOO_MANY_NUMBERS", "up to 20 numbers"],
    ["BAD_LABEL", "1 to 40 characters"],
  ])("explains %s when saving", async (code, words) => {
    await openReady();
    fireEvent.click(screen.getByRole("button", { name: /Add a number/ }));
    fill("Riya", "9812345678");
    post.mockRejectedValue(new ApiError('{"error":"x"}', 400, code));
    fireEvent.click(save());
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(words);
    expect(alert).not.toHaveTextContent("{");
  });

  it("removes a saved number after confirming", async () => {
    await openReady();
    fireEvent.click(screen.getByRole("button", { name: "Remove Arnav" }));
    expect(screen.getByText("Remove Arnav?")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Keep" }));
    expect(screen.queryByText("Remove Arnav?")).not.toBeInTheDocument();
    expect(del).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Remove Arnav" }));
    del.mockResolvedValue({ removed: true });
    mockGet({ ...READY, allowed_numbers: [READY.allowed_numbers[1]] });
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    await flush();
    await flush();
    expect(del).toHaveBeenCalledWith("/api/phone/exotel/numbers/n1");
    const select = screen.getByLabelText("Number to call") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual(["Pre-approved number, +91•••••00000"]);
    expect(screen.getByTestId("numbers-status")).toHaveTextContent("Removed Arnav.");
  });

  it.each([
    ["NUMBER_NOT_FOUND", "already removed"],
    ["NUMBER_FROM_ENV", "can only be removed there"],
  ])("explains %s when removing", async (code, words) => {
    await openReady();
    fireEvent.click(screen.getByRole("button", { name: "Remove Arnav" }));
    del.mockRejectedValue(new ApiError("x", 400, code));
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(words);
  });

  it("gives pre-approved numbers no remove button", async () => {
    await openReady();
    fireEvent.change(screen.getByLabelText("Number to call"), { target: { value: "+919000000000" } });
    expect(screen.queryByRole("button", { name: /^Remove/ })).not.toBeInTheDocument();
    expect(screen.getByText(/Pre-approved in backend\/\.env/)).toBeInTheDocument();
  });
});
