import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SWRConfig } from "swr";
import { IntegrationsPanel, type IntegrationsView } from "@/components/agents/IntegrationsPanel";
import { CallRow } from "@/app/(app)/dashboard/calls/CallRow";
import { api } from "@/lib/api";
import { toast } from "sonner";

jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({ api: { get: jest.fn(), put: jest.fn(), post: jest.fn() }, ApiError: class ApiError extends Error {} }));
jest.mock("@/components/analytics/answerAudit", () => ({ useFlaggedByCall: () => ({}) }));

const get = api.get as jest.Mock;
const put = api.put as jest.Mock;
const post = api.post as jest.Mock;

const SHEET = "S".repeat(30);
const view = (over: Partial<IntegrationsView["caller_sheet"]> = {}): IntegrationsView => ({
  webhook: { enabled: false, url: "", include_test: false, redact_phones: false, events: ["turn.completed"], secret_set: false, secret_created_at: null },
  sheets: { enabled: false, spreadsheet_id: "", sheet_name: "", include_test: false, redact_phones: false, service_account_email: "", credentials_set: false },
  caller_sheet: {
    enabled: false,
    spreadsheet_id: "",
    sheet_name: "BonClay",
    include_phone: true,
    include_test: false,
    share_with: "logger@demo.iam.gserviceaccount.com",
    key_source: "server",
    last_write: null,
    columns: ["Time (IST)", "Agent", "Caller name", "Phone"],
    ...over,
  },
  privacy_note: "note",
  secrets_storage: { available: true, reason: null },
  local_webhooks_allowed: true,
  sheet_columns: ["timestamp"],
});

function open(v: IntegrationsView) {
  get.mockImplementation((path: string) => Promise.resolve(path.endsWith("/deliveries") ? { deliveries: [] } : v));
  render(<IntegrationsPanel agentId="a1" />);
  fireEvent.click(screen.getByRole("button", { name: /Integrations/ }));
}

beforeEach(() => jest.clearAllMocks());

describe("caller details in a Google Sheet (settings card)", () => {
  it("tells the owner which e-mail to share the sheet with and shows the defaults", async () => {
    open(view());
    const card = await screen.findByTestId("caller-sheet-card");
    expect(within(card).getByTestId("share-with")).toHaveTextContent("logger@demo.iam.gserviceaccount.com");
    expect(within(card).getByTestId("share-with")).toHaveTextContent("Editor");
    expect(within(card).getByLabelText("Tab name")).toHaveValue("BonClay");
    expect(within(card).getByRole("checkbox", { name: /Include phone numbers/ })).toBeChecked();
    expect(within(card).getByRole("checkbox", { name: /Send caller details/ })).not.toBeChecked();
    expect(within(card).getByRole("button", { name: "Test caller sheet" })).toBeDisabled();
    expect(card).toHaveTextContent("Time (IST), Agent, Caller name, Phone");
  });

  it("says so when no Google key is set up", async () => {
    open(view({ share_with: "", key_source: null }));
    const card = await screen.findByTestId("caller-sheet-card");
    expect(card).toHaveTextContent("GOOGLE_SERVICE_ACCOUNT_FILE");
    expect(within(card).queryByTestId("share-with")).toBeNull();
  });

  it("saves the link, the tab and the toggles", async () => {
    put.mockResolvedValue(view({ enabled: true, spreadsheet_id: SHEET, include_phone: false }));
    open(view());
    const link = await screen.findByLabelText("Sheet for caller details (link or id)");
    fireEvent.change(link, { target: { value: `https://docs.google.com/spreadsheets/d/${SHEET}/edit` } });
    fireEvent.change(screen.getByLabelText("Tab name"), { target: { value: "Bookings" } });
    fireEvent.click(screen.getByRole("checkbox", { name: /Send caller details/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Include phone numbers/ }));
    fireEvent.click(screen.getByRole("button", { name: "Save caller sheet" }));
    await waitFor(() => expect(put).toHaveBeenCalled());
    expect(put.mock.calls[0][0]).toBe("/api/agents/a1/integrations/caller-sheet");
    expect(put.mock.calls[0][1]).toEqual({
      spreadsheet: `https://docs.google.com/spreadsheets/d/${SHEET}/edit`,
      sheet_name: "Bookings",
      enabled: true,
      include_phone: false,
      include_test: false,
    });
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Caller details settings saved."));
    expect(await screen.findByLabelText("Sheet for caller details (link or id)")).toHaveValue(`https://docs.google.com/spreadsheets/d/${SHEET}`);
  });

  it("does not turn it on without a link", async () => {
    put.mockResolvedValue(view());
    open(view());
    fireEvent.click(await screen.findByRole("checkbox", { name: /Send caller details/ }));
    fireEvent.click(screen.getByRole("button", { name: "Save caller sheet" }));
    await waitFor(() => expect(put).toHaveBeenCalled());
    expect(put.mock.calls[0][1]).toMatchObject({ enabled: false, spreadsheet: "" });
  });

  it("tests the connection and shows Google's exact error with what to do", async () => {
    post.mockResolvedValueOnce({ ok: false, error: "Google Sheets said (403): The caller does not have permission", hint: "Share the spreadsheet with logger@demo.iam.gserviceaccount.com (Editor access) and try again.", status: 403, row: null });
    open(view({ spreadsheet_id: SHEET, enabled: true }));
    fireEvent.click(await screen.findByRole("button", { name: "Test caller sheet" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/agents/a1/integrations/caller-sheet/test"));
    const result = await screen.findByTestId("caller-sheet-result");
    expect(result).toHaveTextContent("does not have permission");
    expect(result).toHaveTextContent("Share the spreadsheet with logger@demo.iam.gserviceaccount.com");

    post.mockResolvedValueOnce({ ok: true, error: null, hint: "", status: 200, row: 2 });
    fireEvent.click(screen.getByRole("button", { name: "Test caller sheet" }));
    await waitFor(() => expect(screen.getByTestId("caller-sheet-result")).toHaveTextContent("a test row is in the sheet (row 2)"));
  });

  it("shows how the last write went", async () => {
    open(view({ spreadsheet_id: SHEET, last_write: { status: "failed", error: "Google Sheets said (404): Requested entity was not found.", at: "2026-10-27T10:00:00+00:00" } }));
    expect(await screen.findByTestId("caller-sheet-last")).toHaveTextContent("failed, Google Sheets said (404)");
  });
});

describe("Calls page: what happened to the sheet row", () => {
  const call = {
    id: "c1",
    agent_id: "a1",
    agent_name: "Clinic desk",
    channel: "phone" as const,
    practice: false,
    when: new Date().toISOString(),
    caller: null,
    duration_s: 60,
    replies: 2,
    preview: "Do you deliver?",
    outcome: { key: "answered" as const, label: "Answered", detail: null, request_id: null },
  };
  const row = (sheet?: { state: "sent" | "waiting" | "failed"; error?: string }, defaultOpen = false) =>
    render(
      <SWRConfig value={{ provider: () => new Map() }}>
        <CallRow call={{ ...call, sheet }} defaultOpen={defaultOpen} />
      </SWRConfig>
    );

  it("shows sent, waiting and failed, and nothing when the conversation never went to a sheet", () => {
    const sent = row({ state: "sent" });
    expect(screen.getByTestId("sheet-status")).toHaveTextContent("Sent to Google Sheets ✓");
    sent.unmount();
    const waiting = row({ state: "waiting" });
    expect(screen.getByTestId("sheet-status")).toHaveTextContent("Waiting to send to Google Sheets");
    waiting.unmount();
    const failed = row({ state: "failed", error: "the sheet is not shared" });
    expect(screen.getByTestId("sheet-status")).toHaveTextContent("Google Sheets failed: the sheet is not shared");
    failed.unmount();
    row();
    expect(screen.queryByTestId("sheet-status")).toBeNull();
  });

  it("opens the conversation a spreadsheet link points to", () => {
    get.mockResolvedValue({ ...call, transcript: [], details: {} });
    row(undefined, true);
    expect(screen.getByRole("button", { expanded: true })).toBeTruthy();
  });
});
