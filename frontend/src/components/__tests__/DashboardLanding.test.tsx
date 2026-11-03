import { render, screen } from "@testing-library/react";
import DashboardHome from "@/app/(app)/dashboard/page";

const replace = jest.fn();
jest.mock("next/navigation", () => ({ useRouter: () => ({ replace, push: jest.fn() }), usePathname: () => "/dashboard" }));
// The panels have their own tests; only whether the page shows them matters here.
jest.mock("@/components/dashboard/QuickActions", () => ({ QuickActions: () => <div>quick-actions</div> }));
jest.mock("@/components/dashboard/FirstRunChecklist", () => ({ FirstRunChecklist: () => null }));
jest.mock("@/components/dashboard/NeedsAttention", () => ({ NeedsAttention: () => null }));
jest.mock("@/components/dashboard/RecentActivity", () => ({ RecentActivity: () => null }));
jest.mock("@/components/dashboard/SystemStatus", () => ({ SystemStatus: () => null }));

beforeEach(() => {
  replace.mockReset();
  localStorage.clear();
});

test("Studio is remembered: nothing of Agents shows, and the person is sent to Studio home", () => {
  localStorage.setItem("bonclay.area", "studio");
  render(<DashboardHome />);
  expect(screen.queryByText("quick-actions")).toBeNull();
  expect(screen.queryByRole("heading", { name: "What do you want to do?" })).toBeNull();
  expect(screen.getByLabelText("Loading")).toBeInTheDocument();
  expect(replace).toHaveBeenCalledWith("/dashboard/studio/home");
});

test.each([["Agents is remembered", "agents"], ["nothing is remembered", null]])("%s: the Agents home shows and nobody is redirected", (_n, value) => {
  if (value) localStorage.setItem("bonclay.area", value);
  render(<DashboardHome />);
  expect(screen.getByRole("heading", { name: "What do you want to do?" })).toBeInTheDocument();
  expect(screen.getByText("quick-actions")).toBeInTheDocument();
  expect(replace).not.toHaveBeenCalled();
});

test("blocked storage: still the Agents home", () => {
  const spy = jest.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
    throw new Error("blocked");
  });
  render(<DashboardHome />);
  expect(screen.getByRole("heading", { name: "What do you want to do?" })).toBeInTheDocument();
  expect(replace).not.toHaveBeenCalled();
  spy.mockRestore();
});
