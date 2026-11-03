import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { WhatsAppPanel, webhookAddress, type WhatsAppSettings } from "@/components/agents/WhatsAppPanel";
import { api } from "@/lib/api";

jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({
  API_URL: "https://api.bonclay.test",
  api: { get: jest.fn(), put: jest.fn(), post: jest.fn(), del: jest.fn() },
}));

const off: WhatsAppSettings = {
  enabled: false,
  phone_number_id: "",
  handoff_text: "",
  voice_for_text: true,
  verify_token: "avt_verify123",
  access_token_set: false,
  access_token_hint: "",
  app_secret_set: false,
  webhook_url: "http://localhost:8000/api/whatsapp/webhook",
  webhook_path: "/api/whatsapp/webhook",
  ready: false,
  secrets_storage: { available: true, reason: null },
  default_handoff_text: "I'm an AI assistant, so I've passed your request to our team.",
};
const on: WhatsAppSettings = {
  ...off,
  enabled: true,
  phone_number_id: "109876543210987",
  access_token_set: true,
  access_token_hint: "••••9XYZ",
  app_secret_set: true,
  ready: true,
};

const get = api.get as jest.Mock;
const put = api.put as jest.Mock;
const post = api.post as jest.Mock;

/** GET answers by path: the settings, and the (initially empty) handoff list. */
const load = (settings: WhatsAppSettings, handoffs: unknown[] = []) =>
  get.mockImplementation((path: string) =>
    Promise.resolve(path.endsWith("/handoffs") ? { handoffs } : settings)
  );

beforeEach(() => {
  jest.clearAllMocks();
});

describe("WhatsAppPanel", () => {
  it("composes the webhook address from a tunnel address, else uses the server's", () => {
    expect(webhookAddress("", off)).toBe("http://localhost:8000/api/whatsapp/webhook");
    expect(webhookAddress(" https://abc.trycloudflare.com/ ", off)).toBe("https://abc.trycloudflare.com/api/whatsapp/webhook");
    expect(webhookAddress("not a url", off)).toBe("http://localhost:8000/api/whatsapp/webhook");
  });

  it("starts off, shows the webhook URL and verify token to paste into Meta", async () => {
    load(off);
    render(<WhatsAppPanel agentId="a1" />);
    expect(await screen.findByRole("switch", { name: "Answer on WhatsApp" })).not.toBeChecked();
    expect(get).toHaveBeenCalledWith("/api/agents/a1/whatsapp");
    expect(screen.getByLabelText("Webhook URL")).toHaveValue("http://localhost:8000/api/whatsapp/webhook");
    expect(screen.getByLabelText("Verify token")).toHaveValue("avt_verify123");
    fireEvent.change(screen.getByLabelText(/Public address/), { target: { value: "https://abc.trycloudflare.com" } });
    expect(screen.getByLabelText("Webhook URL")).toHaveValue("https://abc.trycloudflare.com/api/whatsapp/webhook");
  });

  it("keeps the token and secret write-only: masked fields, sent once, cleared after saving", async () => {
    load(off);
    put.mockResolvedValue({ ...off, phone_number_id: "109876543210987", access_token_set: true, access_token_hint: "••••9XYZ", app_secret_set: true });
    render(<WhatsAppPanel agentId="a1" />);
    const token = await screen.findByLabelText("Access token");
    expect(token).toHaveAttribute("type", "password");
    expect(screen.getByLabelText("App secret")).toHaveAttribute("type", "password");
    fireEvent.change(screen.getByLabelText("Phone number ID"), { target: { value: " 109876543210987 " } });
    fireEvent.change(token, { target: { value: "EAAGsecrettoken9XYZ" } });
    fireEvent.change(screen.getByLabelText("App secret"), { target: { value: "app-secret-value" } });
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    await waitFor(() => expect(put).toHaveBeenCalled());
    expect(put.mock.calls[0][0]).toBe("/api/agents/a1/whatsapp");
    expect(put.mock.calls[0][1]).toEqual({
      phone_number_id: "109876543210987",
      handoff_text: "",
      voice_for_text: true,
      access_token: "EAAGsecrettoken9XYZ",
      app_secret: "app-secret-value",
    });
    await waitFor(() => expect(screen.getByLabelText("Access token")).toHaveValue(""));
    expect(screen.getByLabelText("App secret")).toHaveValue("");
    expect(screen.getByLabelText("Access token")).toHaveAttribute("placeholder", expect.stringContaining("••••9XYZ"));
    expect(document.body.textContent).not.toContain("EAAGsecrettoken9XYZ");
  });

  it("does not send the token or secret when the fields are empty", async () => {
    load(on);
    put.mockResolvedValue({ ...on, handoff_text: "Call us" });
    render(<WhatsAppPanel agentId="a1" />);
    fireEvent.change(await screen.findByLabelText("Message when someone asks for a person"), { target: { value: "Call us" } });
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    await waitFor(() => expect(put).toHaveBeenCalled());
    expect(put.mock.calls[0][1]).not.toHaveProperty("access_token");
    expect(put.mock.calls[0][1]).not.toHaveProperty("app_secret");
    expect(put.mock.calls[0][1]).toMatchObject({ handoff_text: "Call us" });
  });

  it("turns WhatsApp on, and explains what is missing when the server refuses", async () => {
    const { toast } = jest.requireMock("sonner");
    load(off);
    put.mockRejectedValueOnce(Object.assign(new Error("Add the access token first."), { code: "ACCESS_TOKEN_REQUIRED" }));
    render(<WhatsAppPanel agentId="a1" />);
    fireEvent.click(await screen.findByRole("switch", { name: "Answer on WhatsApp" }));
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("Add the access token first."));
    expect(screen.getByRole("switch", { name: "Answer on WhatsApp" })).not.toBeChecked();

    put.mockResolvedValueOnce(on);
    fireEvent.click(screen.getByRole("switch", { name: "Answer on WhatsApp" }));
    await waitFor(() => expect(screen.getByRole("switch", { name: "Answer on WhatsApp" })).toBeChecked());
    expect(put.mock.calls[1][1]).toMatchObject({ enabled: true });
  });

  it("simulates a message and shows the voice note and the text without calling Meta", async () => {
    load(on);
    post.mockResolvedValue({
      dry_run: true,
      handoff: false,
      agent_text: "Refunds take thirty days.",
      messages: [
        { type: "audio", mime: "audio/ogg; codecs=opus", bytes: 20480, audio_url: "data:audio/wav;base64,AAAA" },
        { type: "text", body: "Refunds take thirty days." },
      ],
    });
    render(<WhatsAppPanel agentId="a1" />);
    fireEvent.change(await screen.findByLabelText("Message to simulate"), { target: { value: "Refund window?" } });
    fireEvent.click(screen.getByRole("button", { name: "Simulate a message" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/agents/a1/whatsapp/simulate", { text: "Refund window?" }));
    const replies = await screen.findByRole("list", { name: "Simulated replies" });
    expect(replies).toHaveTextContent("Voice note (20 KB, OGG/Opus)");
    expect(replies).toHaveTextContent("Refunds take thirty days.");
    expect(screen.getByLabelText("Voice note preview")).toHaveAttribute("src", "data:audio/wav;base64,AAAA");
  });

  it("checks the connection with Meta and shows the answer", async () => {
    load(on);
    post.mockResolvedValueOnce({ ok: true, display_phone_number: "+1 555 010 9999", verified_name: "Test Shop" });
    render(<WhatsAppPanel agentId="a1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Check connection/ }));
    expect(await screen.findByRole("status")).toHaveTextContent("Connected: Test Shop +1 555 010 9999");
    expect(post).toHaveBeenCalledWith("/api/agents/a1/whatsapp/check");

    post.mockResolvedValueOnce({ ok: false, error: "WhatsApp said: Invalid OAuth access token. (code 190)" });
    fireEvent.click(screen.getByRole("button", { name: /Check connection/ }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Invalid OAuth access token"));
  });

  it("lists callers waiting for a person and marks them followed up", async () => {
    load(on, [{ conversation_id: "c1", contact: "919876543210", requested_at: "2026-10-15T10:00:00+00:00" }]);
    post.mockResolvedValue({ conversation_id: "c1", needs_human: false });
    render(<WhatsAppPanel agentId="a1" />);
    expect(await screen.findByText("Waiting for a person (1)")).toBeInTheDocument();
    expect(screen.getByText("+919876543210")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Mark +919876543210 followed up" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/agents/a1/whatsapp/handoffs/c1/resolve"));
    await waitFor(() => expect(screen.queryByText("+919876543210")).not.toBeInTheDocument());
  });

  it("makes a new verify token", async () => {
    load(on);
    post.mockResolvedValue({ ...on, verify_token: "avt_new" });
    render(<WhatsAppPanel agentId="a1" />);
    fireEvent.click(await screen.findByRole("button", { name: /New token/ }));
    await waitFor(() => expect(screen.getByLabelText("Verify token")).toHaveValue("avt_new"));
    expect(post).toHaveBeenCalledWith("/api/agents/a1/whatsapp/verify-token");
  });

  it("warns when the server cannot store secrets, and shows a load error", async () => {
    load({ ...off, secrets_storage: { available: false, reason: "Set BONCLAY_SECRETS_KEY." } });
    const { unmount } = render(<WhatsAppPanel agentId="a1" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Set BONCLAY_SECRETS_KEY.");
    unmount();
    get.mockRejectedValue(new Error("boom"));
    render(<WhatsAppPanel agentId="a2" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("boom");
  });
});
