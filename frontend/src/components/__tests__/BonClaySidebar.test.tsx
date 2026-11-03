import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { SWRConfig } from "swr";
import { BonClaySidebar, systemState } from "@/components/BonClaySidebar";
import { api } from "@/lib/api";

let mockPath = "/dashboard";
const push = jest.fn();
jest.mock("next/navigation", () => ({ usePathname: () => mockPath, useRouter: () => ({ push, replace: jest.fn() }) }));
jest.mock("@/components/AuthProvider", () => ({ useAuth: () => ({ user: null, signOut: jest.fn() }) }));
jest.mock("@/components/ThemeToggle", () => ({ ThemeToggle: () => null }));
jest.mock("@/hooks/useApi", () => ({ useHealth: () => ({ data: { supabase: { configured: false } } }) }));
jest.mock("@/lib/api", () => ({ api: { get: jest.fn() } }));

const get = api.get as jest.Mock;
let modelStatus: () => Promise<unknown>;

beforeEach(() => {
  mockPath = "/dashboard";
  push.mockReset();
  localStorage.clear();
  get.mockReset().mockResolvedValue({ counts: { pending: 0 }, requests: [] });
  modelStatus = async () => ({ engines: [{ id: "kokoro", loaded: false }, { id: "dhvaani", loaded: false }] });
  global.fetch = jest.fn(async () => ({ ok: true, json: () => modelStatus() })) as unknown as typeof fetch;
});

async function open(path = "/dashboard") {
  mockPath = path;
  const view = render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <BonClaySidebar />
    </SWRConfig>
  );
  await act(async () => {});
  return view;
}

const desktop = () => document.querySelector("aside") as HTMLElement;

const mainNav = () => within(desktop()).getByRole("navigation", { name: "Main" });
const linkRows = (root: HTMLElement) =>
  within(root).getAllByRole("link").map((a) => [a.textContent?.replace(/\d+ waiting.*$/, "").trim(), a.getAttribute("href")]);

test("Agents area: its own groups, with Studio-only pages left out and API keys in the footer", async () => {
  await open("/dashboard");
  const groups = within(mainNav()).getAllByRole("group").map((g) => g.getAttribute("aria-label"));
  expect(groups).toEqual(["Agents home", "Set up", "Try and run", "How it is doing"]);
  expect(linkRows(mainNav())).toEqual([
    ["Agents home", "/dashboard"],
    ["Agents", "/dashboard/studio/agents"],
    ["Documents", "/dashboard/knowledge"],
    ["Voices", "/dashboard/voices"],
    ["Talk to your agent", "/dashboard/playground"],
    ["Calls and bookings", "/dashboard/calls"],
    ["Campaigns", "/dashboard/campaigns"],
    ["Approvals", "/dashboard/approvals"],
    ["Results", "/dashboard/analytics"],
    ["Safety check", "/dashboard/eval"],
  ]);
  expect(within(desktop()).getByRole("group", { name: "Developers" })).toHaveTextContent("API keys");
  expect(within(desktop()).getByRole("link", { name: "API keys" })).toHaveAttribute("href", "/dashboard/keys");
  expect(within(desktop()).queryByRole("link", { name: "Voice Studio" })).toBeNull();
});

test("Studio area: its own groups, with no agent pages anywhere in the sidebar", async () => {
  await open("/dashboard/studio/home");
  const groups = within(mainNav()).getAllByRole("group").map((g) => g.getAttribute("aria-label"));
  expect(groups).toEqual(["Studio home", "Make audio", "Voices and checks"]);
  expect(linkRows(mainNav())).toEqual([
    ["Studio home", "/dashboard/studio/home"],
    ["Voice Studio", "/dashboard/studio"],
    ["Say it once", "/dashboard/say-it-once"],
    ["Voices", "/dashboard/voices"],
    ["Verify a clip", "/verify"],
  ]);
  const all = within(desktop()).getAllByRole("link").map((a) => a.getAttribute("href") ?? "");
  for (const agentOnly of ["/dashboard/studio/agents", "/dashboard/calls", "/dashboard/campaigns", "/dashboard/approvals", "/dashboard/playground", "/dashboard/knowledge", "/dashboard/analytics", "/dashboard/eval"]) {
    expect(all).not.toContain(agentOnly);
  }
  expect(within(desktop()).getByRole("link", { name: "API keys" })).toBeInTheDocument();
});

const radios = () => within(desktop()).getAllByRole("radio");
const checked = () => radios().filter((r) => r.getAttribute("aria-checked") === "true").map((r) => r.textContent);

describe("the Studio | Agents switch", () => {
  test("is a radiogroup with two radios, exactly one checked, and only that one in the tab order", async () => {
    await open("/dashboard");
    expect(within(desktop()).getByRole("radiogroup", { name: "Area" })).toBeInTheDocument();
    expect(radios().map((r) => r.textContent)).toEqual(["Studio", "Agents"]);
    expect(checked()).toEqual(["Agents"]);
    expect(radios().map((r) => r.tabIndex)).toEqual([-1, 0]);
    radios().forEach((r) => expect(r.className).toMatch(/h-11/));
  });

  test("clicking an area remembers it and opens that area's home", async () => {
    await open("/dashboard");
    fireEvent.click(radios()[0]);
    expect(push).toHaveBeenCalledWith("/dashboard/studio/home");
    expect(localStorage.getItem("bonclay.area")).toBe("studio");
  });

  test("arrow keys, Home and End move to the other option, select it and move focus", async () => {
    await open("/dashboard/studio/home");
    expect(checked()).toEqual(["Studio"]);
    radios()[0].focus();
    fireEvent.keyDown(radios()[0], { key: "ArrowRight" });
    expect(push).toHaveBeenLastCalledWith("/dashboard");
    expect(localStorage.getItem("bonclay.area")).toBe("agents");
    expect(document.activeElement).toBe(radios()[1]);
    expect(checked()).toEqual(["Agents"]); // shown at once, before the new page has loaded
    // back to Studio: already on Studio home (the mocked route does not change), so only the choice is saved
    fireEvent.keyDown(radios()[1], { key: "ArrowLeft" });
    expect(checked()).toEqual(["Studio"]);
    expect(document.activeElement).toBe(radios()[0]);
    expect(localStorage.getItem("bonclay.area")).toBe("studio");
    fireEvent.keyDown(radios()[0], { key: "End" });
    expect(push).toHaveBeenLastCalledWith("/dashboard");
    expect(push).toHaveBeenCalledTimes(2);
    fireEvent.keyDown(radios()[1], { key: "Home" });
    expect(checked()).toEqual(["Studio"]);
    push.mockClear();
    fireEvent.keyDown(radios()[0], { key: "a" });
    expect(push).not.toHaveBeenCalled();
  });

  test("choosing the area you are already home in does not navigate", async () => {
    await open("/dashboard/studio/home");
    fireEvent.click(radios()[0]);
    expect(push).not.toHaveBeenCalled();
  });
});

describe("which area you are in", () => {
  test.each([
    ["/dashboard/studio/agents", "Agents"],
    ["/dashboard/calls", "Agents"],
    ["/dashboard/approvals", "Agents"],
    ["/dashboard/knowledge", "Agents"],
    ["/dashboard/knowledge/try", "Agents"],
    ["/dashboard/studio", "Studio"],
    ["/dashboard/studio/home", "Studio"],
    ["/dashboard/say-it-once", "Studio"],
  ])("%s is in %s", async (path, label) => {
    localStorage.setItem("bonclay.area", label === "Studio" ? "agents" : "studio"); // the route wins over what is remembered
    await open(path);
    expect(checked()).toEqual([label]);
  });

  test("a route is remembered once you are on it", async () => {
    await open("/dashboard/calls");
    expect(localStorage.getItem("bonclay.area")).toBe("agents");
    await open("/dashboard/say-it-once");
    expect(localStorage.getItem("bonclay.area")).toBe("studio");
  });

  test("a shared page (Voices, API keys) keeps the remembered area, and Voices is in both lists", async () => {
    localStorage.setItem("bonclay.area", "studio");
    await open("/dashboard/voices");
    expect(checked()).toEqual(["Studio"]);
    expect(within(mainNav()).getByRole("link", { name: "Voices" })).toHaveAttribute("aria-current", "page");
    document.body.innerHTML = "";
    localStorage.setItem("bonclay.area", "agents");
    await open("/dashboard/voices");
    expect(checked()).toEqual(["Agents"]);
    expect(within(mainNav()).getByRole("link", { name: "Voices" })).toHaveAttribute("aria-current", "page");
    document.body.innerHTML = "";
    await open("/dashboard/keys");
    expect(checked()).toEqual(["Agents"]);
    expect(within(desktop()).getByRole("link", { name: "API keys" })).toHaveAttribute("aria-current", "page");
  });

  test("/dashboard is Agents home when Agents (or nothing) is remembered", async () => {
    await open("/dashboard");
    expect(checked()).toEqual(["Agents"]);
    document.body.innerHTML = "";
    localStorage.setItem("bonclay.area", "agents");
    await open("/dashboard");
    expect(checked()).toEqual(["Agents"]);
  });

  test("nothing remembered on a shared page: Agents, the original dashboard", async () => {
    await open("/dashboard/voices");
    expect(checked()).toEqual(["Agents"]);
  });

  test("blocked storage does not break anything", async () => {
    const spy = jest.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    await open("/dashboard/voices");
    expect(checked()).toEqual(["Agents"]);
    spy.mockRestore();
  });

  test("a Studio person landing on /dashboard sees the Studio list while being sent to Studio home", async () => {
    localStorage.setItem("bonclay.area", "studio");
    await open("/dashboard");
    expect(checked()).toEqual(["Studio"]);
    expect(localStorage.getItem("bonclay.area")).toBe("studio");
  });
});

test("the most specific item is the current page (Agents is not Voice Studio)", async () => {
  await open("/dashboard/studio/agents");
  const current = within(mainNav()).getAllByRole("link").filter((a) => a.getAttribute("aria-current") === "page");
  expect(current.map((a) => a.getAttribute("href"))).toEqual(["/dashboard/studio/agents"]);
});

test("Studio home is current on its own page, not Voice Studio", async () => {
  await open("/dashboard/studio/home");
  const current = within(mainNav()).getAllByRole("link").filter((a) => a.getAttribute("aria-current") === "page");
  expect(current.map((a) => a.getAttribute("href"))).toEqual(["/dashboard/studio/home"]);
});

test.each(["/dashboard/knowledge", "/dashboard/knowledge/try"])("Documents is the current page on %s", async (path) => {
  await open(path);
  const current = within(mainNav()).getAllByRole("link").filter((a) => a.getAttribute("aria-current") === "page");
  expect(current.map((a) => a.getAttribute("href"))).toEqual(["/dashboard/knowledge"]);
});

test("the approvals count is only fetched in the Agents area", async () => {
  await open("/dashboard/studio/home");
  expect(get).not.toHaveBeenCalledWith("/api/meetings?status=pending");
  document.body.innerHTML = "";
  await open("/dashboard/calls");
  expect(get).toHaveBeenCalledWith("/api/meetings?status=pending");
});

test("the mobile drawer has the switch and the whole list of the current area", async () => {
  await open("/dashboard/studio/home");
  fireEvent.click(screen.getByRole("button", { name: "Open menu" }));
  const drawer = await screen.findByRole("dialog", { name: "Navigation" });
  expect(within(drawer).getByRole("radiogroup", { name: "Area" })).toBeInTheDocument();
  expect(within(drawer).getAllByRole("radio").filter((r) => r.getAttribute("aria-checked") === "true").map((r) => r.textContent)).toEqual(["Studio"]);
  expect(linkRows(within(drawer).getByRole("navigation", { name: "Main" }))).toEqual(linkRows(mainNav()));
  expect(within(drawer).getByRole("link", { name: "API keys" })).toBeInTheDocument();
});

test("Approvals shows how many requests are waiting, and nothing when none are", async () => {
  get.mockResolvedValue({ counts: { pending: 3 }, requests: [] });
  await open("/dashboard/approvals");
  const link = within(desktop()).getByRole("link", { name: /Approvals/ });
  expect(link).toHaveTextContent("3");
  expect(link).toHaveTextContent("3 waiting for your decision");
  expect(get).toHaveBeenCalledWith("/api/meetings?status=pending");
});

test("no badge for zero waiting, and a failing request just hides it", async () => {
  await open();
  expect(within(desktop()).getByRole("link", { name: "Approvals" })).toBeInTheDocument();
  get.mockRejectedValue(new Error("down"));
  await open();
  expect(within(desktop()).getAllByRole("link", { name: /^Approvals$/ })[0]).not.toHaveTextContent("waiting");
});

test("one status line replaces the per-engine dots", async () => {
  await open();
  const status = within(desktop()).getByRole("status");
  expect(status).toHaveTextContent("Ready");
  expect(within(desktop()).queryByText("Kokoro")).toBeNull();
});

test("a server that does not answer needs attention", async () => {
  global.fetch = jest.fn(async () => ({ ok: false, status: 500, json: async () => ({}) })) as unknown as typeof fetch;
  await open();
  expect(within(desktop()).getByRole("status")).toHaveTextContent("Needs attention");
});

test("system state: not loaded engines are ready, a loading one is starting, no answer is attention", () => {
  expect(systemState(false, [{ name: "A", state: "offline" }, { name: "B", state: "ready" }])).toBe("ready");
  expect(systemState(false, [{ name: "A", state: "loading" }])).toBe("starting");
  expect(systemState(false, undefined)).toBe("starting");
  expect(systemState(true, undefined)).toBe("attention");
});
