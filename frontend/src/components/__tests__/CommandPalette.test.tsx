import { act, fireEvent, render, screen } from "@testing-library/react";
import { LayoutWrapper } from "@/components/LayoutWrapper";
import { CommandPalette } from "@/components/CommandPalette";
import { useState } from "react";

const push = jest.fn();
let mockPath = "/dashboard";
jest.mock("next/navigation", () => ({ useRouter: () => ({ push }), usePathname: () => mockPath }));
jest.mock("@/components/BonClaySidebar", () => ({ BonClaySidebar: () => <nav aria-label="Navigation"><a href="/dashboard">Overview</a></nav> }));
jest.mock("@/hooks/useApi", () => ({
  useAgents: () => ({
    data: [
      { id: "a1", name: "Sunrise Clinic" },
      { id: "a2", name: "Campus Desk" },
    ],
    isLoading: false,
  }),
}));

beforeEach(() => {
  push.mockClear();
  mockPath = "/dashboard";
  Element.prototype.scrollIntoView = jest.fn();
});

const input = () => screen.getByRole("combobox");

test("skip link is the first focusable element and targets the main content", () => {
  render(
    <LayoutWrapper>
      <button>page button</button>
    </LayoutWrapper>
  );
  const skip = screen.getByRole("link", { name: "Skip to content" });
  expect(skip).toHaveAttribute("href", "#main-content");
  const focusable = Array.from(document.querySelectorAll<HTMLElement>("a[href], button, input, [tabindex]:not([tabindex='-1'])"));
  expect(focusable[0]).toBe(skip);
  const main = document.getElementById("main-content");
  expect(main?.tagName).toBe("MAIN");
});

test("Ctrl+K opens the palette, Escape closes it and focus returns", () => {
  render(
    <LayoutWrapper>
      <p>content</p>
    </LayoutWrapper>
  );
  const trigger = screen.getAllByRole("button", { name: "Search and commands" })[0];
  trigger.focus();
  expect(screen.queryByRole("dialog", { name: "Search and commands" })).toBeNull();
  fireEvent.keyDown(window, { key: "k", ctrlKey: true });
  expect(screen.getByRole("dialog", { name: "Search and commands" })).toBeInTheDocument();
  expect(input()).toHaveFocus();
  fireEvent.keyDown(input(), { key: "Escape" });
  expect(screen.queryByRole("dialog", { name: "Search and commands" })).toBeNull();
  expect(trigger).toHaveFocus();
});

test("Cmd+K works too and the palette is not offered outside the dashboard", () => {
  mockPath = "/";
  const { unmount } = render(
    <LayoutWrapper>
      <p>content</p>
    </LayoutWrapper>
  );
  fireEvent.keyDown(window, { key: "k", metaKey: true });
  expect(screen.queryByRole("dialog")).toBeNull();
  unmount();
  mockPath = "/dashboard";
  render(
    <LayoutWrapper>
      <p>content</p>
    </LayoutWrapper>
  );
  fireEvent.keyDown(window, { key: "k", metaKey: true });
  expect(screen.getByRole("dialog")).toBeInTheDocument();
});

function Harness() {
  const [open, setOpen] = useState(true);
  return <CommandPalette open={open} onClose={() => setOpen(false)} />;
}

test("lists pages, the user's agents and actions; arrows move and Enter goes", () => {
  render(<Harness />);
  const labels = screen.getAllByRole("option").map((o) => o.textContent);
  expect(labels.some((l) => l?.startsWith("Go to Agents home"))).toBe(true);
  expect(labels.some((l) => l?.startsWith("Go to Studio home"))).toBe(true);
  expect(labels.some((l) => l?.startsWith("Go to Voice Studio"))).toBe(true);
  expect(labels.some((l) => l?.startsWith("Go to Calls and bookings"))).toBe(true);
  expect(labels.some((l) => l?.startsWith("Talk to Sunrise Clinic"))).toBe(true);
  expect(labels.some((l) => l?.startsWith("Edit Campus Desk"))).toBe(true);
  expect(labels.some((l) => l?.startsWith("Clone a voice"))).toBe(true);
  expect(labels.some((l) => l?.startsWith("Approve bookings"))).toBe(true);

  const opts = () => screen.getAllByRole("option");
  expect(opts()[0]).toHaveAttribute("aria-selected", "true");
  fireEvent.keyDown(input(), { key: "ArrowDown" });
  expect(opts()[1]).toHaveAttribute("aria-selected", "true");
  expect(input()).toHaveAttribute("aria-activedescendant", opts()[1].id);
  fireEvent.keyDown(input(), { key: "ArrowUp" });
  fireEvent.keyDown(input(), { key: "ArrowUp" });
  expect(opts()[opts().length - 1]).toHaveAttribute("aria-selected", "true");
});

test("filters loosely, Enter navigates and closes", () => {
  render(<Harness />);
  fireEvent.change(input(), { target: { value: "talk camp" } });
  const opts = screen.getAllByRole("option");
  expect(opts[0]).toHaveTextContent("Talk to Campus Desk");
  fireEvent.keyDown(input(), { key: "Enter" });
  expect(push).toHaveBeenCalledWith("/dashboard/playground?agent=a2");
  expect(screen.queryByRole("dialog")).toBeNull();
});

test("typos in order still match, and no match says so", () => {
  render(<Harness />);
  fireEvent.change(input(), { target: { value: "apprvls" } });
  expect(screen.getAllByRole("option")[0]).toHaveTextContent("Go to Approvals");
  fireEvent.change(input(), { target: { value: "zzzzqq" } });
  expect(screen.queryAllByRole("option")).toHaveLength(0);
  expect(screen.getByRole("status")).toHaveTextContent(/nothing matches/i);
  fireEvent.keyDown(input(), { key: "Enter" });
  expect(push).not.toHaveBeenCalled();
});

test("Tab keeps focus inside the dialog", () => {
  render(<Harness />);
  const notPrevented = fireEvent.keyDown(input(), { key: "Tab" });
  expect(notPrevented).toBe(false);
  expect(input()).toHaveFocus();
});

test("clicking an option navigates", () => {
  render(<Harness />);
  act(() => {
    fireEvent.click(screen.getByRole("option", { name: /Clone a voice/ }));
  });
  expect(push).toHaveBeenCalledWith("/dashboard/voices?new=1");
});
