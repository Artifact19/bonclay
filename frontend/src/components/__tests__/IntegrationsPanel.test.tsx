import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { AgentEditor } from "@/components/agents/AgentEditor";
import { IntegrationsPanel, type IntegrationsView } from "@/components/agents/IntegrationsPanel";
import { api } from "@/lib/api";
import { toast } from "sonner";
import type { Agent } from "@/types/api";

jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({ api: { get: jest.fn(), put: jest.fn(), post: jest.fn() } }));
jest.mock("@/hooks/useApi", () => ({ useAgents: () => ({ mutate: jest.fn().mockResolvedValue(undefined) }) }));
jest.mock("@/components/agents/KnowledgePanel", () => ({ KnowledgePanel: () => null }));

const get = api.get as jest.Mock;
const put = api.put as jest.Mock;
const post = api.post as jest.Mock;

const view = (over: Partial<IntegrationsView> = {}): IntegrationsView => ({
  webhook: { enabled: false, url: "", include_test: false, redact_phones: false, events: ["turn.completed", "tool.called"], secret_set: false, secret_created_at: null },
  sheets: { enabled: false, spreadsheet_id: "", sheet_name: "", include_test: false, redact_phones: false, service_account_email: "", credentials_set: false },
  caller_sheet: { enabled: false, spreadsheet_id: "", sheet_name: "BonClay", include_phone: true, include_test: false, share_with: "", key_source: null, last_write: null, columns: ["Time (IST)", "Agent"] },
  privacy_note: "These integrations send what callers say to systems you choose.",
  secrets_storage: { available: true, reason: null },
  local_webhooks_allowed: true,
  sheet_columns: ["timestamp", "agent", "language"],
  ...over,
});

const delivery = { delivery_id: "d1", target: "webhook", event: "turn.completed", status: "failed", code: 500, latency_ms: 42, error: "timed out after 5 s", attempts: 4, at: "2026-10-15T10:00:00+00:00" };

function open() {
  fireEvent.click(screen.getByRole("button", { name: /Integrations/ }));
}

beforeEach(() => {
  jest.clearAllMocks();
  get.mockImplementation((path: string) => Promise.resolve(path.endsWith("/deliveries") ? { deliveries: [delivery] } : view()));
});

describe("IntegrationsPanel", () => {
  it("stays closed and makes no request until it is opened, then shows the privacy note and the log", async () => {
    render(<IntegrationsPanel agentId="a1" />);
    expect(get).not.toHaveBeenCalled();
    open();
    await screen.findByText(/send what callers say/);
    expect(get).toHaveBeenCalledWith("/api/agents/a1/integrations");
    expect(await screen.findByText("timed out after 5 s")).toBeInTheDocument();
    expect(screen.getByText("HTTP 500")).toBeInTheDocument();
    expect(screen.getByText(/4 tries/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send test event" })).toBeDisabled();
  });

  it("saves the webhook with its options and shows the signing secret once", async () => {
    put.mockResolvedValue(view({ webhook: { ...view().webhook, enabled: true, url: "https://hooks.example.com/x", secret_set: true, redact_phones: true }, secret: "whsec_abc123" }));
    render(<IntegrationsPanel agentId="a1" />);
    open();
    const url = await screen.findByLabelText("Webhook URL");
    fireEvent.change(url, { target: { value: "https://hooks.example.com/x" } });
    fireEvent.click(screen.getByRole("checkbox", { name: /Send events to this URL/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /^Hide phone numbers(?! \()/ }));
    fireEvent.click(screen.getByRole("button", { name: "Save webhook" }));
    await waitFor(() => expect(put).toHaveBeenCalled());
    expect(put.mock.calls[0][0]).toBe("/api/agents/a1/integrations/webhook");
    expect(put.mock.calls[0][1]).toMatchObject({
      url: "https://hooks.example.com/x",
      enabled: true,
      redact_phones: true,
      include_test: false,
      events: ["turn.completed", "tool.called"],
    });
    expect(await screen.findByTestId("signing-secret")).toHaveTextContent("whsec_abc123");
    expect(screen.getByTestId("signing-secret")).toHaveTextContent(/not shown again/);
    fireEvent.click(screen.getByRole("button", { name: "I copied it" }));
    expect(screen.queryByText("whsec_abc123")).not.toBeInTheDocument();
  });

  it("rotates the secret and sends a test event", async () => {
    const withSecret = view({ webhook: { ...view().webhook, url: "https://h.example.com", secret_set: true, enabled: true } });
    get.mockImplementation((path: string) => Promise.resolve(path.endsWith("/deliveries") ? { deliveries: [] } : withSecret));
    post.mockImplementation((path: string) =>
      Promise.resolve(path.endsWith("rotate-secret") ? { ...withSecret, secret: "whsec_new" } : { ok: true, delivery: { ...delivery, status: "delivered", code: 200, error: null } })
    );
    render(<IntegrationsPanel agentId="a1" />);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Send test event" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/agents/a1/integrations/webhook/test"));
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Test event delivered (HTTP 200)."));
    fireEvent.click(screen.getByRole("button", { name: "Rotate secret" }));
    expect(await screen.findByTestId("signing-secret")).toHaveTextContent("whsec_new");
  });

  it("never pre-fills the Google key, sends it only when typed, and reports the exact Sheets error", async () => {
    const withKey = view({ sheets: { ...view().sheets, credentials_set: true, service_account_email: "bot@p.iam.gserviceaccount.com", spreadsheet_id: "S".repeat(30) } });
    get.mockImplementation((path: string) => Promise.resolve(path.endsWith("/deliveries") ? { deliveries: [] } : withKey));
    put.mockResolvedValue(withKey);
    post.mockResolvedValue({ ok: false, error: "Google Sheets said (403): The caller does not have permission", hint: "Share the spreadsheet with bot@p.iam.gserviceaccount.com (Editor access) and try again.", status: 403 });
    render(<IntegrationsPanel agentId="a1" />);
    open();
    const key = (await screen.findByLabelText(/Service-account key/)) as HTMLTextAreaElement;
    expect(key.value).toBe("");
    expect(screen.getByText(/Share your spreadsheet with bot@p.iam.gserviceaccount.com/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Save Google Sheets" }));
    await waitFor(() => expect(put).toHaveBeenCalledTimes(1));
    expect(put.mock.calls[0][1]).not.toHaveProperty("service_account_json");

    fireEvent.change(key, { target: { value: '{"type":"service_account"}' } });
    fireEvent.click(screen.getByRole("button", { name: "Save Google Sheets" }));
    await waitFor(() => expect(put).toHaveBeenCalledTimes(2));
    expect(put.mock.calls[1][1]).toMatchObject({ service_account_json: '{"type":"service_account"}' });
    await waitFor(() => expect((screen.getByLabelText(/Service-account key/) as HTMLTextAreaElement).value).toBe(""));

    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));
    const result = await screen.findByTestId("sheets-result");
    expect(result).toHaveTextContent("The caller does not have permission");
    expect(result).toHaveTextContent("Share the spreadsheet with bot@p.iam.gserviceaccount.com");
  });

  it("explains when secrets cannot be stored", async () => {
    get.mockImplementation((path: string) =>
      Promise.resolve(path.endsWith("/deliveries") ? { deliveries: [] } : view({ secrets_storage: { available: false, reason: "BONCLAY_SECRETS_KEY is not set on the server." } }))
    );
    render(<IntegrationsPanel agentId="a1" />);
    open();
    expect(await screen.findByRole("alert")).toHaveTextContent("BONCLAY_SECRETS_KEY");
  });

  it("shows a load error", async () => {
    get.mockRejectedValue(new Error("Agent not found"));
    render(<IntegrationsPanel agentId="a1" />);
    open();
    expect(await screen.findByRole("alert")).toHaveTextContent("Agent not found");
  });
});

describe("AgentEditor", () => {
  it("has an Integrations section", () => {
    const agent: Agent = { id: "a1", name: "R", greeting: "", system_instructions: "", supported_languages: ["en"], voice_id: null, disclose_ai: true, tools: [], created_at: "2026-01-01T00:00:00Z", document_count: 0 };
    render(<AgentEditor agent={agent} voices={[]} tab="advanced" onBack={() => undefined} onDelete={() => undefined} />);
    expect(screen.getByRole("button", { name: /Integrations/ })).toHaveAttribute("aria-expanded", "false");
    // The header may look up phone status; nothing is requested for the integrations until the section is opened.
    expect(get.mock.calls.filter(([path]) => String(path).includes("/integrations"))).toHaveLength(0);
  });
});
