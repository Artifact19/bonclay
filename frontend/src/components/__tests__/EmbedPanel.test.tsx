import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { EmbedPanel, embedSnippet, type WidgetSettings } from "@/components/agents/EmbedPanel";
import { api } from "@/lib/api";

jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({
  API_URL: "https://api.bonclay.test",
  api: { get: jest.fn(), put: jest.fn(), post: jest.fn(), del: jest.fn() },
}));
jest.mock("@/components/agents/ConfirmDialog", () => ({
  ConfirmDialog: ({ open, confirmLabel, onConfirm }: { open: boolean; confirmLabel: string; onConfirm: () => void }) =>
    open ? <button onClick={onConfirm}>{confirmLabel}</button> : null,
}));

const off: WidgetSettings = {
  enabled: false,
  key: null,
  origins: [],
  per_visitor_hour: 30,
  per_key_day: 300,
  business_name: "",
  script_path: "/widget/v1/bonclay-agent.js",
};
const on: WidgetSettings = { ...off, enabled: true, key: "awk_abc123", origins: ["https://shop.example.com"], business_name: "Shop" };

const get = api.get as jest.Mock;
const put = api.put as jest.Mock;
const post = api.post as jest.Mock;
const del = api.del as jest.Mock;

beforeEach(() => {
  jest.clearAllMocks();
});

describe("EmbedPanel", () => {
  it("builds the two-line snippet from the API host and the key", () => {
    expect(embedSnippet("awk_k", "/widget/v1/bonclay-agent.js", "https://api.bonclay.test/")).toBe(
      '<script src="https://api.bonclay.test/widget/v1/bonclay-agent.js" defer></script>\n<bonclay-agent agent="awk_k"></bonclay-agent>'
    );
  });

  it("starts unpublished with defaults and no code to copy", async () => {
    get.mockResolvedValue(off);
    render(<EmbedPanel agentId="a1" />);
    expect(await screen.findByRole("switch", { name: "Publish to website" })).not.toBeChecked();
    expect(get).toHaveBeenCalledWith("/api/agents/a1/widget");
    expect(screen.getByLabelText("Turns per visitor per hour")).toHaveValue(30);
    expect(screen.getByLabelText("Turns per day, all visitors")).toHaveValue(300);
    expect(screen.queryByRole("button", { name: /Copy code/ })).not.toBeInTheDocument();
  });

  it("publishes with the allowed sites typed in and then shows the snippet", async () => {
    get.mockResolvedValue(off);
    put.mockResolvedValue(on);
    render(<EmbedPanel agentId="a1" />);
    const box = await screen.findByLabelText("Allowed websites");
    fireEvent.change(box, { target: { value: "https://shop.example.com\nhttps://www.example.com" } });
    fireEvent.change(screen.getByLabelText("Business name"), { target: { value: "Shop" } });
    fireEvent.click(screen.getByRole("switch", { name: "Publish to website" }));
    await waitFor(() => expect(put).toHaveBeenCalled());
    expect(put.mock.calls[0][0]).toBe("/api/agents/a1/widget");
    expect(put.mock.calls[0][1]).toEqual({
      enabled: true,
      origins: ["https://shop.example.com", "https://www.example.com"],
      business_name: "Shop",
      per_visitor_hour: 30,
      per_key_day: 300,
    });
    const code = await screen.findByLabelText("Embed code");
    expect(code).toHaveTextContent('<script src="https://api.bonclay.test/widget/v1/bonclay-agent.js" defer></script>');
    expect(code).toHaveTextContent('<bonclay-agent agent="awk_abc123"></bonclay-agent>');
    expect(screen.getByRole("switch", { name: "Publish to website" })).toBeChecked();
  });

  it("copies the snippet", async () => {
    get.mockResolvedValue(on);
    const writeText = jest.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    render(<EmbedPanel agentId="a1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Copy code/ }));
    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    expect(writeText.mock.calls[0][0]).toContain('agent="awk_abc123"');
    expect(await screen.findByRole("button", { name: /Copied/ })).toBeInTheDocument();
  });

  it("saves the limits and unpublishes", async () => {
    get.mockResolvedValue(on);
    put.mockResolvedValue({ ...on, per_visitor_hour: 10, enabled: false });
    render(<EmbedPanel agentId="a1" />);
    fireEvent.change(await screen.findByLabelText("Turns per visitor per hour"), { target: { value: "10" } });
    fireEvent.click(screen.getByRole("switch", { name: "Publish to website" }));
    await waitFor(() => expect(put).toHaveBeenCalled());
    expect(put.mock.calls[0][1]).toMatchObject({ enabled: false, per_visitor_hour: 10, per_key_day: 300 });
    await waitFor(() => expect(screen.queryByLabelText("Embed code")).not.toBeInTheDocument());
    expect(screen.getByText(/Unpublished/)).toBeInTheDocument();
  });

  it("explains a missing site instead of publishing", async () => {
    const { toast } = jest.requireMock("sonner");
    get.mockResolvedValue(off);
    put.mockRejectedValue(Object.assign(new Error("Add at least one allowed site before publishing."), { code: "ORIGINS_REQUIRED" }));
    render(<EmbedPanel agentId="a1" />);
    fireEvent.click(await screen.findByRole("switch", { name: "Publish to website" }));
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("Add at least one allowed site before publishing."));
    expect(screen.getByLabelText("Allowed websites")).toHaveFocus();
    expect(screen.getByRole("switch", { name: "Publish to website" })).not.toBeChecked();
  });

  it("replaces the key after confirming, and removes the widget", async () => {
    get.mockResolvedValue(on);
    post.mockResolvedValue({ ...on, key: "awk_new999" });
    del.mockResolvedValue(off);
    render(<EmbedPanel agentId="a1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Replace key/ }));
    expect(post).not.toHaveBeenCalled();
    fireEvent.click(screen.getAllByRole("button", { name: "Replace key" }).pop()!);
    await waitFor(() => expect(post).toHaveBeenCalledWith("/api/agents/a1/widget/rotate"));
    await waitFor(() => expect(screen.getByLabelText("Embed code")).toHaveTextContent("awk_new999"));

    fireEvent.click(screen.getByRole("button", { name: /Remove widget/ }));
    fireEvent.click(screen.getAllByRole("button", { name: "Remove widget" }).pop()!);
    await waitFor(() => expect(del).toHaveBeenCalledWith("/api/agents/a1/widget"));
    await waitFor(() => expect(screen.queryByLabelText("Embed code")).not.toBeInTheDocument());
  });

  it("shows a load error", async () => {
    get.mockRejectedValue(new Error("boom"));
    render(<EmbedPanel agentId="a1" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("boom");
  });
});
