import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SWRConfig } from "swr";
import CampaignsPage from "@/app/(app)/dashboard/campaigns/page";
import { progressPercent, windowLabel } from "@/app/(app)/dashboard/campaigns/format";
import { api } from "@/lib/api";

jest.mock("@/lib/api", () => ({ api: { get: jest.fn(), post: jest.fn(), del: jest.fn() }, ApiError: class ApiError extends Error {} }));
jest.mock("@/hooks/useApi", () => ({ useAgents: () => ({ data: [{ id: "a1", name: "Clinic desk" }, { id: "a2", name: "Fee reminder" }] }) }));
jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn() } }));

const get = api.get as jest.Mock;
const post = api.post as jest.Mock;
const del = api.del as jest.Mock;

const window_ = { days: [0, 1, 2, 3, 4, 5], start: "10:00", end: "18:00", timezone: "Asia/Kolkata" };
const camp = (over: object = {}) => ({
  id: "c1", name: "October fees", agent_id: "a2", status: "running", created_at: 1, finished_at: null, window: window_,
  max_attempts: 2, retry_minutes: 30, concurrency: 1, max_per_hour: 30, paused_reason: null, total: 10,
  counts: { pending: 4, calling: 1, done: 3, failed: 1, dnc: 1 }, ...over,
});
let campaigns: object[] = [];
let dnc: string[] = [];

function serve() {
  get.mockReset().mockImplementation(async (path: string) => {
    if (path === "/api/campaigns") return { campaigns };
    if (path === "/api/do-not-call") return { numbers: dnc };
    if (path.startsWith("/api/campaigns/")) return { ...camp(), contacts: [{ name: "Asha", number: "91••••••3210", status: "done", attempts: 1, last_status: "completed" }] };
    return {};
  });
  post.mockReset().mockResolvedValue({});
  del.mockReset().mockResolvedValue({});
}
const view = () => render(<SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}><CampaignsPage /></SWRConfig>);

beforeEach(() => { campaigns = []; dnc = []; serve(); });

test("progress and window helpers", () => {
  expect(progressPercent({ total: 10, counts: { pending: 4, calling: 1, done: 3, failed: 1, dnc: 1 } })).toBe(50);
  expect(progressPercent({ total: 0, counts: { pending: 0, calling: 0, done: 0, failed: 0, dnc: 0 } })).toBe(0);
  expect(windowLabel(window_)).toBe("Mon to Sat, 10:00 to 18:00 (Asia/Kolkata)");
});

test("shows an empty state and the do-not-call list", async () => {
  dnc = ["+919876543210"];
  view();
  expect(await screen.findByText("No campaigns yet")).toBeInTheDocument();
  expect(await screen.findByLabelText("Remove ••••3210 from the list")).toBeInTheDocument();
});

test("a campaign shows its progress in plain words and can be paused", async () => {
  campaigns = [camp()];
  view();
  expect(await screen.findByText("October fees")).toBeInTheDocument();
  expect(screen.getByText(/Fee reminder · 10 people/)).toBeInTheDocument();
  expect(screen.getByText(/3 reached · 5 still to call · 1 not reached · 1 on the do-not-call list · 1 on a call now/)).toBeInTheDocument();
  expect(screen.getByRole("progressbar", { name: "October fees progress" })).toHaveAttribute("aria-valuenow", "50");
  fireEvent.click(screen.getByRole("button", { name: /Pause/ }));
  await waitFor(() => expect(post).toHaveBeenCalledWith("/api/campaigns/c1/pause", {}));
});

test("a paused campaign says why and resumes", async () => {
  campaigns = [camp({ status: "paused", paused_reason: "Too many calls failed to connect. Check the phone setup, then resume." })];
  view();
  expect(await screen.findByText(/Too many calls failed to connect/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /Resume/ }));
  await waitFor(() => expect(post).toHaveBeenCalledWith("/api/campaigns/c1/start", {}));
});

test("stopping and deleting ask first", async () => {
  campaigns = [camp()];
  view();
  await screen.findByText("October fees");
  fireEvent.click(screen.getByRole("button", { name: /^Stop$/ }));
  expect(post).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Stop campaign" }));
  await waitFor(() => expect(post).toHaveBeenCalledWith("/api/campaigns/c1/stop", {}));
  fireEvent.click(screen.getByRole("button", { name: /^Delete$/ }));
  expect(del).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Delete campaign" }));
  await waitFor(() => expect(del).toHaveBeenCalledWith("/api/campaigns/c1"));
});

test("show people lists the masked contacts", async () => {
  campaigns = [camp()];
  view();
  await screen.findByText("October fees");
  fireEvent.click(screen.getByRole("button", { name: "Show people" }));
  const list = await screen.findByRole("list", { name: "People in this campaign" });
  expect(within(list).getByText(/Asha/)).toBeInTheDocument();
  expect(within(list).getByText(/91••••••3210/)).toBeInTheDocument();
});

test("a new campaign needs a name, an agent, numbers and the consent box before it can be created", async () => {
  view();
  fireEvent.click((await screen.findAllByRole("button", { name: /New campaign/ }))[0]);
  const create = screen.getByRole("button", { name: "Create campaign" });
  expect(create).toBeDisabled();
  fireEvent.change(screen.getByPlaceholderText("October fee reminders"), { target: { value: "Reminders" } });
  fireEvent.change(screen.getByLabelText(/Agent that makes the calls/), { target: { value: "a1" } });
  fireEvent.change(screen.getByLabelText(/Phone numbers/), { target: { value: "name,phone\nAsha,9876543210" } });
  expect(create).toBeDisabled();                                                    // still no consent
  fireEvent.click(screen.getByLabelText(/Everyone on this list agreed/));
  expect(create).toBeEnabled();
  post.mockResolvedValueOnce({ ...camp({ id: "c9", name: "Reminders", total: 1 }), import: { invalid: 0, duplicates: 0, over_limit: 0 } });
  fireEvent.click(create);
  await waitFor(() => expect(post).toHaveBeenCalled());
  const [path, body] = post.mock.calls[0];
  expect(path).toBe("/api/campaigns");
  expect(body).toMatchObject({ name: "Reminders", agent_id: "a1", caller_agreed: true, max_attempts: 2, concurrency: 1,
    window: { days: [0, 1, 2, 3, 4, 5], start: "10:00", end: "18:00", timezone: "Asia/Kolkata" } });
  expect(body.csv).toContain("9876543210");
});

test("adding numbers to the do-not-call list posts them", async () => {
  view();
  const input = await screen.findByLabelText("Numbers to add");
  fireEvent.change(input, { target: { value: "+919876543210, 9876543211" } });
  fireEvent.click(screen.getByRole("button", { name: "Add" }));
  await waitFor(() => expect(post).toHaveBeenCalledWith("/api/do-not-call", { numbers: ["+919876543210", "9876543211"] }));
});
