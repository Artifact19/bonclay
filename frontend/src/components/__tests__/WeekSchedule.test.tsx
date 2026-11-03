import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { WeekSchedule } from "@/components/dashboard/WeekSchedule";
import { api } from "@/lib/api";
import { toast } from "sonner";
import { useMeetingWeek } from "@/hooks/useApi";
import type { MeetingWeek } from "@/types/api";

jest.mock("@/lib/api", () => ({
  api: { post: jest.fn().mockResolvedValue({}), put: jest.fn().mockResolvedValue({}) },
  ApiError: class ApiError extends Error {},
}));
jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn() } }));
jest.mock("@/hooks/useApi", () => ({ useMeetingWeek: jest.fn() }));

const mutate = jest.fn();
const DAYS = [
  ["2026-10-19", "Monday"], ["2026-10-20", "Tuesday"], ["2026-10-21", "Wednesday"], ["2026-10-22", "Thursday"], ["2026-10-23", "Friday"],
  ["2026-10-24", "Saturday"], ["2026-10-25", "Sunday"],
] as const;
const closedDay = { "10:30": { state: "closed" }, "10:50": { state: "closed" }, "11:10": { state: "closed" } };
const free = () => ({ "10:30": { state: "free" }, "10:50": { state: "free" }, "11:10": { state: "free" } });

function makeWeek(over: Partial<MeetingWeek> = {}): MeetingWeek {
  return {
    start: "2026-10-19", today: "2026-10-19", max_days_ahead: 30, slot_minutes: 20, owner: "Prof. Jayashri Prasad", label: null,
    days: DAYS.map(([date, weekday], i) => ({ date, weekday, open: i < 5, full: date === "2026-10-23" })),
    times: ["10:30", "10:50", "11:10"],
    cells: {
      "2026-10-19": {
        "10:30": { state: "asked_for", request: { request_id: "M10", name: "Riya Patil", role: "student", on_calendar: false } },
        "10:50": { state: "busy", reason: "Lecture", source: "sample" },
        "11:10": { state: "free" },
      },
      "2026-10-20": {
        "10:30": { state: "booked", request: { request_id: "M11", name: "Sanjay Rao", role: "faculty", on_calendar: true } },
        "10:50": { state: "busy", reason: "Department meeting", source: "blocked" },
        "11:10": { state: "busy", reason: "Marked busy by you", source: "owner" },
      },
      "2026-10-21": { "10:30": { state: "past" }, "10:50": { state: "busy", reason: "Busy on Google Calendar", source: "calendar" }, "11:10": { state: "free", too_soon: true } },
      "2026-10-22": free(),
      "2026-10-23": free(),
      "2026-10-24": closedDay,
      "2026-10-25": closedDay,
    } as MeetingWeek["cells"],
    settings: { availability_check: true, sample_schedule: true },
    calendar: { configured: false, calendar_id_set: false, key_file_found: false, service_account_email: null, timezone: "Asia/Kolkata" },
    ...over,
  };
}

function show(data: MeetingWeek | undefined, extra: Record<string, unknown> = {}) {
  (useMeetingWeek as jest.Mock).mockReturnValue({ data, mutate, isLoading: false, error: undefined, ...extra });
}

beforeEach(() => {
  jest.clearAllMocks();
  show(makeWeek());
});

test("draws a real table with the days as column headers, the times as row headers and a cell for every state", () => {
  render(<WeekSchedule />);
  expect(screen.getByRole("heading", { name: "Head of department, this week" })).toBeInTheDocument();
  expect(screen.getByText("Sample schedule")).toBeInTheDocument();
  expect(screen.getByText(/20-minute slots/)).toBeInTheDocument();
  const table = screen.getByRole("table");
  expect(within(table).getAllByRole("columnheader")).toHaveLength(5);                    // Monday to Friday, no weekend
  expect(within(table).getByRole("columnheader", { name: /Friday.*full for the day/ })).toBeInTheDocument();
  expect(within(table).getAllByRole("rowheader").map((h) => h.textContent)).toEqual(["10:30", "10:50", "11:10"]);

  expect(screen.getByRole("button", { name: /^Monday 10:30, asked for by Riya, pending/ })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /^Tuesday 10:30, booked for Sanjay, on Google Calendar/ })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /^Monday 10:50, busy: Lecture \(sample\)/ })).toBeInTheDocument();   // a made-up slot can be changed
  expect(screen.getByRole("button", { name: /^Tuesday 11:10, busy: Marked busy by you/ })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /^Monday 11:10, free\. Press to mark busy/ })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /^Wednesday 11:10, free \(too soon for the agent to offer\)/ })).toBeInTheDocument();
  // the office rules and the calendar are not the owner's to change here: text, not buttons, with the reason in a tooltip
  expect(screen.queryByRole("button", { name: /Tuesday 10:50/ })).toBeNull();
  expect(screen.getByText("Tuesday 10:50, busy: Department meeting")).toBeInTheDocument();
  expect(screen.getByText("Wednesday 10:50, busy: Busy on Google Calendar")).toBeInTheDocument();
  expect(screen.getByText("Wednesday 10:30, already past")).toBeInTheDocument();
  expect(screen.getByText("Tuesday 10:50, busy: Department meeting").closest("span[title]")).toHaveAttribute("title", "Department meeting");
  // the legend
  const legend = screen.getByRole("list", { name: "Legend" });
  expect(within(legend).getAllByRole("listitem").map((l) => l.textContent)).toEqual(["Free", "Busy", "Asked for", "Booked"]);
});

test("the heading uses the office label when the settings carry one, and no pill shows without the sample", () => {
  show(makeWeek({ label: "Dean's office", settings: { availability_check: true, sample_schedule: false } }));
  render(<WeekSchedule />);
  expect(screen.getByRole("heading", { name: "Dean's office, this week" })).toBeInTheDocument();
  expect(screen.queryByText("Sample schedule")).toBeNull();
});

test("clicking a free cell marks it busy at once, saves it and refreshes", async () => {
  render(<WeekSchedule />);
  fireEvent.click(screen.getByRole("button", { name: /^Monday 11:10, free/ }));
  const patched = mutate.mock.calls[0][0] as MeetingWeek;
  expect(patched.cells["2026-10-19"]["11:10"]).toEqual({ state: "busy", reason: "Marked busy by you", source: "owner" });
  expect(mutate.mock.calls[0][1]).toEqual({ revalidate: false });
  await waitFor(() => expect(api.put).toHaveBeenCalledWith("/api/meetings/slots", { date: "2026-10-19", time: "11:10", state: "busy" }));
  await waitFor(() => expect(mutate).toHaveBeenCalledTimes(2));
});

test("clicking a sample or owner busy cell frees it", async () => {
  render(<WeekSchedule />);
  fireEvent.click(screen.getByRole("button", { name: /^Monday 10:50, busy: Lecture/ }));
  await waitFor(() => expect(api.put).toHaveBeenCalledWith("/api/meetings/slots", { date: "2026-10-19", time: "10:50", state: "free" }));
  expect((mutate.mock.calls[0][0] as MeetingWeek).cells["2026-10-19"]["10:50"]).toEqual({ state: "free" });
  fireEvent.click(screen.getByRole("button", { name: /^Tuesday 11:10, busy: Marked busy by you/ }));
  await waitFor(() => expect(api.put).toHaveBeenCalledWith("/api/meetings/slots", { date: "2026-10-20", time: "11:10", state: "free" }));
});

test("a refused change rolls the cell back and says why", async () => {
  const before = makeWeek();
  show(before);
  (api.put as jest.Mock).mockRejectedValueOnce(Object.assign(new Error("A meeting request holds this time. Approve or decline it first."), { status: 409 }));
  render(<WeekSchedule />);
  fireEvent.click(screen.getByRole("button", { name: /^Monday 11:10, free/ }));
  await waitFor(() => expect(toast.error).toHaveBeenCalledWith("A meeting request holds this time. Approve or decline it first."));
  expect(mutate).toHaveBeenLastCalledWith(before, { revalidate: false });
});

test("the arrow keys move between the cells that can be pressed", () => {
  render(<WeekSchedule />);
  screen.getByRole("button", { name: /^Monday 11:10, free/ }).focus();
  fireEvent.keyDown(document.activeElement as Element, { key: "ArrowRight" });
  expect(document.activeElement).toBe(screen.getByRole("button", { name: /^Tuesday 11:10, busy/ }));
  fireEvent.keyDown(document.activeElement as Element, { key: "ArrowUp" });          // Tuesday 10:50 is not pressable: skip to 10:30
  expect(document.activeElement).toBe(screen.getByRole("button", { name: /^Tuesday 10:30, booked/ }));
});

describe("the two switches", () => {
  test("the agent switch saves and shows what each state does", async () => {
    render(<WeekSchedule />);
    const sw = screen.getByRole("switch", { name: "Agent checks this schedule during calls" });
    expect(sw).toHaveAttribute("aria-checked", "true");
    expect(sw).toHaveAccessibleDescription(/On: the agent offers only free times and suggests others\. Off: the agent takes the caller's preferred time and tells them the office will confirm\./);
    fireEvent.click(sw);
    await waitFor(() => expect(api.put).toHaveBeenCalledWith("/api/meetings/settings", { availability_check: false }));
    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(toast.success).toHaveBeenCalledWith(expect.stringMatching(/preferred time/));
  });

  test("when it is off the switch is unchecked and turning it on saves true", async () => {
    show(makeWeek({ settings: { availability_check: false, sample_schedule: true } }));
    render(<WeekSchedule />);
    const sw = screen.getByRole("switch", { name: "Agent checks this schedule during calls" });
    expect(sw).toHaveAttribute("aria-checked", "false");
    fireEvent.click(sw);
    await waitFor(() => expect(api.put).toHaveBeenCalledWith("/api/meetings/settings", { availability_check: true }));
  });

  test("the sample switch saves too, and a failure is reported", async () => {
    render(<WeekSchedule />);
    const sw = screen.getByRole("switch", { name: "Fill the week with a sample schedule" });
    expect(sw).toHaveAttribute("aria-checked", "true");
    fireEvent.click(sw);
    await waitFor(() => expect(api.put).toHaveBeenCalledWith("/api/meetings/settings", { sample_schedule: false }));
    (api.put as jest.Mock).mockRejectedValueOnce(new Error("Could not reach the server."));
    fireEvent.click(sw);
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("Could not reach the server."));
  });
});

describe("a request in the grid", () => {
  test("an asked-for cell opens approve and decline, and approving refreshes the grid and the list", async () => {
    const onDecision = jest.fn();
    render(<WeekSchedule onDecision={onDecision} />);
    fireEvent.click(screen.getByRole("button", { name: /^Monday 10:30, asked for by Riya/ }));
    const panel = screen.getByRole("group", { name: "Selected request" });
    expect(within(panel).getByText(/Riya Patil/)).toBeInTheDocument();
    expect(within(panel).getByText(/Asked for Mon 19 Oct, 10:30/)).toBeInTheDocument();
    expect(within(panel).getByRole("button", { name: /Decline/ })).toBeInTheDocument();
    fireEvent.click(within(panel).getByRole("button", { name: /Approve/ }));
    await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/meetings/M10/approve", {}));
    await waitFor(() => expect(onDecision).toHaveBeenCalled());
    expect(mutate).toHaveBeenCalled();
    expect(screen.queryByRole("group", { name: "Selected request" })).toBeNull();
  });

  test("declining calls the decline route, and a refusal is shown as it is", async () => {
    render(<WeekSchedule />);
    fireEvent.click(screen.getByRole("button", { name: /^Monday 10:30, asked for/ }));
    (api.post as jest.Mock).mockRejectedValueOnce(Object.assign(new Error("Request M10 is already approved."), { status: 409 }));
    fireEvent.click(screen.getByRole("button", { name: /Decline/ }));
    await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/meetings/M10/decline", {}));
    expect(await screen.findByRole("alert")).toHaveTextContent("already approved");
  });

  test("a booked cell shows the request without approve or decline, and can open the full card", () => {
    const onShowRequest = jest.fn();
    render(<WeekSchedule onShowRequest={onShowRequest} />);
    fireEvent.click(screen.getByRole("button", { name: /^Tuesday 10:30, booked/ }));
    const panel = screen.getByRole("group", { name: "Selected request" });
    expect(within(panel).getByText(/Booked for Tue 20 Oct, 10:30/)).toBeInTheDocument();
    expect(within(panel).queryByRole("button", { name: /Approve/ })).toBeNull();
    fireEvent.click(within(panel).getByRole("button", { name: "Show the full request" }));
    expect(onShowRequest).toHaveBeenCalledWith("M11", "approved");
  });

  test("the full card of a waiting request is in the Waiting tab", () => {
    const onShowRequest = jest.fn();
    render(<WeekSchedule onShowRequest={onShowRequest} />);
    fireEvent.click(screen.getByRole("button", { name: /^Monday 10:30, asked for/ }));
    fireEvent.click(screen.getByRole("button", { name: "Show the full request" }));
    expect(onShowRequest).toHaveBeenCalledWith("M10", "pending");
  });
});

describe("moving between weeks", () => {
  test("next, previous and this week", () => {
    const week = () => (useMeetingWeek as jest.Mock).mock.calls.at(-1)?.[0] as string | undefined;
    const { rerender } = render(<WeekSchedule />);                                      // today is Monday 19 Oct
    expect(screen.getByRole("button", { name: "Previous week" })).toBeDisabled();       // the week that holds today is the first
    expect(screen.getByRole("button", { name: "This week" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Next week" }));
    expect(week()).toBe("2026-10-26");
    show(makeWeek({ start: "2026-10-26" }));                                            // what the server answers for that week
    rerender(<WeekSchedule />);
    expect(screen.getByRole("heading")).toHaveTextContent("week of 26 Oct");
    expect(screen.getByRole("button", { name: "Previous week" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Previous week" }));
    expect(week()).toBe("2026-10-19");
    fireEvent.click(screen.getByRole("button", { name: "This week" }));
    expect(week()).toBeUndefined();
  });

  test("it cannot go past the days the agent can book", () => {
    show(makeWeek({ start: "2026-11-16" }));             // 19 Oct + 30 days = 18 Nov: the week after starts on 23 Nov
    render(<WeekSchedule />);
    expect(screen.getByRole("button", { name: "Next week" })).toBeDisabled();
    show(makeWeek({ start: "2026-11-09" }));
    render(<WeekSchedule />);
    expect(screen.getAllByRole("button", { name: "Next week" })[1]).toBeEnabled();
  });
});

describe("the states around the grid", () => {
  test("loading", () => {
    show(undefined, { isLoading: true });
    const { container } = render(<WeekSchedule />);
    expect(screen.queryByRole("table")).toBeNull();
    expect(container.querySelector(".animate-pulse, [class*=skeleton]") || screen.getByText("Loading the schedule")).toBeTruthy();
  });

  test("an error with a way to try again", () => {
    show(undefined, { error: new Error("Check that the API is running.") });
    render(<WeekSchedule />);
    expect(screen.getByText("Could not load the schedule")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Try again|Retry/i }));
    expect(mutate).toHaveBeenCalled();
  });

  test("no office hours", () => {
    show(makeWeek({ times: [], days: DAYS.map(([date, weekday]) => ({ date, weekday, open: false, full: false })) }));
    render(<WeekSchedule />);
    expect(screen.getByText("No office hours are set")).toBeInTheDocument();
    expect(screen.queryByRole("table")).toBeNull();
    expect(screen.getByRole("switch", { name: /Agent checks this schedule/ })).toBeInTheDocument();   // the switches stay
  });

  test("a calendar that could not be read is said, not hidden", () => {
    const week = makeWeek();
    week.days[1] = { ...week.days[1], calendar_error: true };
    show(week);
    render(<WeekSchedule />);
    expect(screen.getByRole("status")).toHaveTextContent(/Google Calendar could not be read for Tue 20 Oct/);
  });
});
