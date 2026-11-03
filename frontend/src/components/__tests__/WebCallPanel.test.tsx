import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { WebCallPanel, expiryText, type WebCallLink, type WebCallLinks } from "@/components/agents/WebCallPanel";
import { api } from "@/lib/api";
import { toast } from "sonner";

jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({
  API_URL: "https://api.bonclay.test",
  api: { get: jest.fn(), put: jest.fn(), post: jest.fn(), del: jest.fn() },
}));
jest.mock("@/components/agents/ConfirmDialog", () => ({
  ConfirmDialog: ({ open, confirmLabel, onConfirm }: { open: boolean; confirmLabel: string; onConfirm: () => void }) =>
    open ? <button onClick={onConfirm}>{confirmLabel}</button> : null,
}));

const link = (over: Partial<WebCallLink> = {}): WebCallLink => ({
  id: "awc_abcdefghijklmnopqrstuvwx012345",
  url: "https://phone.example.test/call/awc_abcdefghijklmnopqrstuvwx012345",
  label: "Clinic demo",
  status: "active",
  created_at: "2026-10-27T10:00:00+00:00",
  expires_at: "2026-11-03T10:00:00+00:00",
  revoked_at: null,
  last_used_at: null,
  max_session_minutes: 10,
  max_minutes_per_day: 60,
  max_minutes_per_visitor: null,
  sessions: 3,
  turns: 12,
  minutes_today: 4.5,
  minutes_total: 9,
  ...over,
});
const empty: WebCallLinks = {
  links: [],
  base_url: "https://phone.example.test",
  public: true,
  defaults: { expires_in: "7d", max_session_minutes: 10, max_minutes_per_day: 60 },
};
const withLinks = (...links: WebCallLink[]): WebCallLinks => ({ ...empty, links });

const get = api.get as jest.Mock;
const post = api.post as jest.Mock;
const del = api.del as jest.Mock;

beforeEach(() => {
  jest.clearAllMocks();
});

describe("WebCallPanel", () => {
  it("loads the agent's links with the defaults filled in and nothing to copy yet", async () => {
    get.mockResolvedValue(empty);
    render(<WebCallPanel agentId="a1" />);
    expect(await screen.findByText("No links yet. Create one above and share it.")).toBeInTheDocument();
    expect(get).toHaveBeenCalledWith("/api/agents/a1/web-call-links");
    expect(screen.getByLabelText("Link expires")).toHaveValue("7d");
    expect(screen.getByLabelText("Minutes per call")).toHaveValue(10);
    expect(screen.getByLabelText("Minutes per day, everyone together")).toHaveValue(60);
    expect(screen.getByLabelText("Minutes per visitor per day (optional)")).toHaveValue(null);
    expect(screen.queryByText(/only open on this computer/)).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Web call link" })).toBeInTheDocument();
  });

  it("warns when the address is not public yet", async () => {
    get.mockResolvedValue({ ...empty, public: false, base_url: "http://localhost:8010" });
    render(<WebCallPanel agentId="a1" />);
    expect(await screen.findByRole("status")).toHaveTextContent(/only open on this computer/);
    expect(screen.getByRole("status")).toHaveTextContent("PUBLIC_BASE_URL");
  });

  it("creates a link with the chosen expiry and limits, then shows it", async () => {
    get.mockResolvedValueOnce(empty).mockResolvedValueOnce(withLinks(link()));
    post.mockResolvedValue(link());
    render(<WebCallPanel agentId="a1" />);
    fireEvent.change(await screen.findByLabelText("Note for yourself (optional)"), { target: { value: " Clinic demo " } });
    fireEvent.change(screen.getByLabelText("Link expires"), { target: { value: "1d" } });
    fireEvent.change(screen.getByLabelText("Minutes per call"), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText("Minutes per day, everyone together"), { target: { value: "30" } });
    fireEvent.change(screen.getByLabelText("Minutes per visitor per day (optional)"), { target: { value: "8" } });
    fireEvent.click(screen.getByRole("button", { name: /Create link/ }));
    await waitFor(() => expect(post).toHaveBeenCalled());
    expect(post.mock.calls[0][0]).toBe("/api/agents/a1/web-call-links");
    expect(post.mock.calls[0][1]).toEqual({
      label: "Clinic demo",
      expires_in: "1d",
      max_session_minutes: 5,
      max_minutes_per_day: 30,
      max_minutes_per_visitor: 8,
    });
    const list = await screen.findByRole("list", { name: "Web call links" });
    expect(within(list).getByText("Clinic demo")).toBeInTheDocument();
    expect(within(list).getByDisplayValue("https://phone.example.test/call/awc_abcdefghijklmnopqrstuvwx012345")).toBeInTheDocument();
    expect(toast.success).toHaveBeenCalled();
  });

  it("sends no per-visitor cap and no label when they are left empty", async () => {
    get.mockResolvedValue(empty);
    post.mockResolvedValue(link());
    render(<WebCallPanel agentId="a1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Create link/ }));
    await waitFor(() => expect(post).toHaveBeenCalled());
    expect(post.mock.calls[0][1]).toEqual({
      label: undefined,
      expires_in: "7d",
      max_session_minutes: 10,
      max_minutes_per_day: 60,
      max_minutes_per_visitor: null,
    });
  });

  it("shows the counters, the expiry and the limits of each link", async () => {
    get.mockResolvedValue(withLinks(link({ max_minutes_per_visitor: 8 })));
    render(<WebCallPanel agentId="a1" />);
    const list = await screen.findByRole("list", { name: "Web call links" });
    expect(list).toHaveTextContent("3 sessions · 12 turns · 4.5 of 60 min used today · up to 10 min per call · 8 min per visitor per day");
    expect(within(list).getByText("Active")).toBeInTheDocument();
    expect(list).toHaveTextContent(/Expires /);
  });

  it("copies the full link", async () => {
    get.mockResolvedValue(withLinks(link()));
    const writeText = jest.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    render(<WebCallPanel agentId="a1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Copy link/ }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("https://phone.example.test/call/awc_abcdefghijklmnopqrstuvwx012345"));
    expect(await screen.findByRole("button", { name: /Copied/ })).toBeInTheDocument();
  });

  it("revokes only after the owner confirms, then reloads", async () => {
    get.mockResolvedValueOnce(withLinks(link())).mockResolvedValueOnce(withLinks(link({ status: "revoked", revoked_at: "2026-10-28T09:00:00+00:00" })));
    post.mockResolvedValue({});
    render(<WebCallPanel agentId="a1" />);
    fireEvent.click(await screen.findByRole("button", { name: /^Revoke$/ }));
    expect(post).not.toHaveBeenCalled();
    fireEvent.click(await screen.findByRole("button", { name: "Revoke link" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/agents/a1/web-call-links/awc_abcdefghijklmnopqrstuvwx012345/revoke"));
    expect(await screen.findByText("Revoked")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Copy link/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Remove$/ })).toBeInTheDocument();
  });

  it("removes a revoked or expired link from the list", async () => {
    get.mockResolvedValueOnce(withLinks(link({ status: "expired" }))).mockResolvedValueOnce(empty);
    del.mockResolvedValue({ ok: true });
    render(<WebCallPanel agentId="a1" />);
    expect(await screen.findByText("Expired")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Copy link/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^Remove$/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Remove link" }));
    await waitFor(() => expect(del).toHaveBeenCalledWith("/api/agents/a1/web-call-links/awc_abcdefghijklmnopqrstuvwx012345"));
    expect(await screen.findByText("No links yet. Create one above and share it.")).toBeInTheDocument();
  });

  it("says so when the links cannot be loaded or created", async () => {
    get.mockRejectedValueOnce(new Error("offline"));
    render(<WebCallPanel agentId="a1" />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/offline|Could not load/);
  });

  it("reports a failed create", async () => {
    get.mockResolvedValue(empty);
    post.mockRejectedValue(Object.assign(new Error("At most 20 active links per agent. Revoke one first."), { code: "TOO_MANY_LINKS" }));
    render(<WebCallPanel agentId="a1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Create link/ }));
    await waitFor(() => expect(toast.error).toHaveBeenCalled());
  });

  it("words the expiry of each status", () => {
    expect(expiryText({ status: "active", expires_at: null, revoked_at: null })).toBe("Never expires");
    expect(expiryText({ status: "active", expires_at: "2026-11-03T10:00:00+00:00", revoked_at: null })).toMatch(/^Expires /);
    expect(expiryText({ status: "expired", expires_at: "2026-11-03T10:00:00+00:00", revoked_at: null })).toMatch(/^Expired /);
    expect(expiryText({ status: "revoked", expires_at: null, revoked_at: "2026-10-28T09:00:00+00:00" })).toMatch(/^Revoked /);
  });
});
