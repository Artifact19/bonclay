import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import ApprovalsPage from "@/app/(app)/dashboard/approvals/page";
import { api } from "@/lib/api";

const mutate = jest.fn();
const pending = {
  request_id: "M10", name: "Riya Patil", role: "student", phone: "9876543210", class_or_ref: "TY AI-A",
  reason: "project guide approval", date: "2026-10-19", weekday: "Monday", time: "10:30", minutes: 20, status: "pending",
  on_calendar: false,
};

jest.mock("@/lib/api", () => ({ api: { post: jest.fn().mockResolvedValue({}), put: jest.fn().mockResolvedValue({}) }, ApiError: class ApiError extends Error {} }));
jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn() } }));
jest.mock("@/hooks/useApi", () => ({
  useMeetings: () => ({ data: { requests: [pending], counts: { pending: 1, approved: 0, declined: 0, cancelled: 0 } }, mutate }),
  // the week grid has its own tests (WeekSchedule.test.tsx): here it is still loading
  useMeetingWeek: () => ({ data: undefined, isLoading: true, error: undefined, mutate }),
  useMeetingSettings: () => ({ data: { settings: { owner: "Prof. Jayashri Prasad" }, calendar: { configured: false } } }),
}));

beforeEach(() => jest.clearAllMocks());

test("shows the pending request with who (number masked), why and when, in words that fit any agent", () => {
  render(<ApprovalsPage />);
  expect(screen.getByText(/Riya Patil/)).toBeInTheDocument();
  expect(screen.getByText(/••••3210/)).toBeInTheDocument();
  expect(screen.queryByText(/9876543210/)).toBeNull();
  expect(screen.getByText("project guide approval")).toBeInTheDocument();
  expect(screen.getByText(/10:30/)).toBeInTheDocument();
  expect(screen.getAllByText(/meeting or an appointment/).length).toBeGreaterThan(0);
  expect(screen.getByText("How requests get here")).toBeInTheDocument();
  expect(screen.queryByText(/HOD/)).toBeNull();
  expect(screen.queryByText(/Prof\./)).toBeNull();
  expect(screen.getByText(/Google Calendar is not connected/)).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: /Waiting for you \(1\)/ })).toBeInTheDocument();
});

test("approving sends the note and refreshes the list", async () => {
  render(<ApprovalsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Approve/ }));
  fireEvent.change(screen.getByLabelText(/Note for the caller/), { target: { value: "Bring your report." } });
  fireEvent.click(screen.getByRole("button", { name: "Confirm approval" }));
  await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/meetings/M10/approve", { note: "Bring your report.", date: undefined, time: undefined }));
  await waitFor(() => expect(mutate).toHaveBeenCalled());
});

test("declining can suggest another time, and a refusal is shown as it is", async () => {
  (api.post as jest.Mock).mockRejectedValueOnce(Object.assign(new Error("Request M10 is already approved."), { status: 409 }));
  render(<ApprovalsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Decline/ }));
  fireEvent.change(screen.getByLabelText(/Suggest another day/), { target: { value: "2026-10-20" } });
  fireEvent.change(screen.getByLabelText(/Suggested time/), { target: { value: "15:00" } });
  fireEvent.click(screen.getByRole("button", { name: "Confirm decline" }));
  await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/meetings/M10/decline",
    { note: "", suggest_date: "2026-10-20", suggest_time: "15:00" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("already approved");
  expect(mutate).not.toHaveBeenCalled();
});

test("the week schedule sits above the list of requests", () => {
  render(<ApprovalsPage />);
  const week = screen.getByRole("region", { name: /this week/ });
  const card = screen.getByText("Riya Patil").closest("[id='meeting-M10']") as HTMLElement;
  expect(card).toBeTruthy();
  expect(week.compareDocumentPosition(card) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
});
