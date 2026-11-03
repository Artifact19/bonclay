import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SWRConfig } from "swr";
import CallsPage from "@/app/(app)/dashboard/calls/page";
import { durationLabel, maskNumber, whenLabel } from "@/app/(app)/dashboard/calls/format";
import { api } from "@/lib/api";

jest.mock("@/lib/api", () => ({ api: { get: jest.fn() }, ApiError: class ApiError extends Error {} }));
jest.mock("@/hooks/useApi", () => ({ useAgents: () => ({ data: [{ id: "a1", name: "Clinic desk" }, { id: "a2", name: "HOD office" }] }) }));

const get = api.get as jest.Mock;
const now = new Date().toISOString();
const base = { agent_id: "a1", agent_name: "Clinic desk", practice: false, when: now, caller: null, duration_s: null, replies: 2, preview: "I want to see a doctor" };
const calls = [
  { ...base, id: "c1", channel: "phone", duration_s: 125, outcome: { key: "booked", label: "Booked", detail: "Booking A12 with Dr. Meera", request_id: null } },
  { ...base, id: "c2", channel: "whatsapp", caller: "••••3210", agent_id: "a2", agent_name: "HOD office", preview: "Can I meet the HOD",
    outcome: { key: "needs_approval", label: "Needs approval", detail: "Meeting request M10 is waiting for your decision.", request_id: "M10" } },
  { ...base, id: "c3", channel: "playground", practice: true, outcome: { key: "couldnt_answer", label: "Couldn't answer", detail: "The agent said it did not know: Do you treat dogs?", request_id: null } },
];
const transcript = [
  { role: "caller", text: "I want to see a doctor", at: now, actions: [] },
  { role: "agent", text: "Booked for Saturday.", at: now, actions: [{ name: "book_appointment", ok: true, summary: "booking A12 with Dr. Meera on Saturday 12:30" }] },
];

function serve(list: unknown[]) {
  get.mockReset().mockImplementation(async (path: string) => {
    if (path.startsWith("/api/phone/exotel/live")) return { calls: [] };      // nothing live right now
    if (path.startsWith("/api/calls/")) return { ...calls[0], transcript };
    return { calls: list, days: 30 };
  });
}
const view = () => render(<SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}><CallsPage /></SWRConfig>);

test("lists conversations with channel, masked caller, duration and a plain outcome", async () => {
  serve(calls);
  view();
  expect(await screen.findByText("Clinic desk · Number not recorded · 2 min 05 s", { exact: false })).toBeInTheDocument();
  expect(screen.getByText(/WhatsApp · HOD office · ••••3210/)).toBeInTheDocument();
  expect(screen.getByText("Practice call")).toBeInTheDocument();
  expect(screen.getByText("Meeting request M10 is waiting for your decision.")).toBeInTheDocument();
  expect(get).toHaveBeenCalledWith("/api/calls?days=30&limit=100");
});

test("the outcome chips count and filter", async () => {
  serve(calls);
  view();
  await screen.findByText(/WhatsApp · HOD office/);
  const chips = screen.getByRole("group", { name: "Filter by what happened" });
  expect(within(chips).getByRole("button", { name: /All 3/ })).toBeInTheDocument();
  fireEvent.click(within(chips).getByRole("button", { name: /Needs approval 1/ }));
  expect(screen.getAllByRole("listitem")).toHaveLength(1);
  expect(screen.getByText(/WhatsApp · HOD office/)).toBeInTheDocument();
  expect(screen.queryByText(/Clinic desk · /)).toBeNull();
});

test("picking an agent asks the server for that agent only", async () => {
  serve(calls);
  view();
  await screen.findByText(/WhatsApp · HOD office/);
  fireEvent.change(screen.getByLabelText("Agent"), { target: { value: "a2" } });
  await waitFor(() => expect(get).toHaveBeenCalledWith("/api/calls?days=30&limit=100&agent_id=a2"));
});

test("opening a conversation loads its transcript, with what the agent did", async () => {
  serve(calls);
  view();
  const row = (await screen.findAllByRole("button", { name: /Show transcript/ }))[0];
  expect(row).toHaveAttribute("aria-expanded", "false");
  fireEvent.click(row);
  expect(await screen.findByText("Booked for Saturday.")).toBeInTheDocument();
  expect(screen.getByText("booking A12 with Dr. Meera on Saturday 12:30")).toBeInTheDocument();
  expect(get).toHaveBeenCalledWith("/api/calls/c1");
  expect(row).toHaveAttribute("aria-expanded", "true");
});

test("a request made in a conversation links to Approvals", async () => {
  serve(calls);
  view();
  const rows = await screen.findAllByRole("button", { name: /Show transcript/ });
  fireEvent.click(rows[1]);
  expect(await screen.findByRole("link", { name: /Review this request in Approvals/ })).toHaveAttribute("href", "/dashboard/approvals");
});

test("empty state says what will appear here and offers a practice call and the agents", async () => {
  serve([]);
  view();
  expect(await screen.findByText("No conversations yet")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Make a practice call" })).toHaveAttribute("href", "/dashboard/playground");
  expect(screen.getByRole("link", { name: "Go to agents" })).toHaveAttribute("href", "/dashboard/studio/agents");
});

test("a failed load says so and can retry", async () => {
  let failed = false;
  get.mockReset().mockImplementation(async (path: string) => {
    if (path.startsWith("/api/phone/exotel/live")) return { calls: [] };
    if (!failed) { failed = true; throw new Error("offline"); }
    return { calls: [], days: 30 };
  });
  view();
  expect(await screen.findByText("Could not load the conversations")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /Try again|Retry/i }));
  expect(await screen.findByText("No conversations yet")).toBeInTheDocument();
});

test("format helpers", () => {
  expect(maskNumber("+91 98765 43210")).toBe("••••3210");
  expect(maskNumber("")).toBe("Number not given");
  expect(durationLabel(45)).toBe("45 s");
  expect(durationLabel(125)).toBe("2 min 05 s");
  expect(durationLabel(null)).toBeNull();
  const noon = new Date("2026-10-17T12:00:00");
  expect(whenLabel("2026-10-17T09:30:00", noon)).toMatch(/^Today, /);
  expect(whenLabel("2026-10-16T09:30:00", noon)).toMatch(/^Yesterday, /);
  expect(whenLabel(null)).toBe("Unknown time");
});
