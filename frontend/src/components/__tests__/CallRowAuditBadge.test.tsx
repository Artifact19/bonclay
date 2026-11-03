import { render, screen } from "@testing-library/react";
import { SWRConfig } from "swr";
import { CallRow } from "@/app/(app)/dashboard/calls/CallRow";
import { useFlaggedByCall } from "@/components/analytics/answerAudit";

jest.mock("@/components/analytics/answerAudit", () => ({ useFlaggedByCall: jest.fn() }));
jest.mock("@/lib/api", () => ({ api: { get: jest.fn() }, ApiError: class ApiError extends Error {} }));

const flagged = useFlaggedByCall as jest.Mock;
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
const view = (id = "c1") =>
  render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <CallRow call={{ ...call, id }} />
    </SWRConfig>
  );

it("shows Check this answer on a call with one flagged answer", () => {
  flagged.mockReturnValue({ c1: 1 });
  view();
  expect(screen.getByText("Check this answer")).toBeTruthy();
});

it("counts several flagged answers and shows nothing for a clean call", () => {
  flagged.mockReturnValue({ c1: 3 });
  const first = view();
  expect(screen.getByText("Check 3 answers")).toBeTruthy();
  first.unmount();
  view("c2");
  expect(screen.queryByText(/Check .*answer/)).toBeNull();
});
