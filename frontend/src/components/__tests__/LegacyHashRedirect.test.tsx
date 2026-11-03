import { act, render } from "@testing-library/react";
import { LegacyHashRedirect } from "@/components/site/LegacyHashRedirect";
import { mapLegacyHash } from "@/components/site/routes";

const replace = jest.fn();
jest.mock("next/navigation", () => ({ useRouter: () => ({ replace }) }));

beforeEach(() => {
  replace.mockClear();
  window.history.replaceState(null, "", "/");
});

describe("mapLegacyHash", () => {
  it.each([
    ["#consent", "/how-it-works#consent"],
    ["#pipeline", "/how-it-works#how"],
    ["#how", "/how-it-works#how"],
    ["#channels", "/channels"],
    ["#proof", "/proof"],
    ["#numbers", "/proof#numbers"],
    ["#attacks", "/proof#attacks"],
    ["#compare", "/proof#compare"],
    ["#rig-check", "/run-it"],
    ["#rig", "/run-it"],
    ["#faq", "/faq"],
    ["#FAQ", "/faq"],
    ["faq", "/faq"],
  ])("sends %s to %s", (hash, target) => {
    expect(mapLegacyHash(hash)).toBe(target);
  });

  it.each(["", "#", "#why", "#products", "#languages", "#hear", "#start", "#top", "#voices", "#unknown", "#constructor", "#__proto__", "#%E0%A4%A"])(
    "leaves %j alone",
    (hash) => {
      expect(mapLegacyHash(hash)).toBeNull();
    },
  );
});

describe("LegacyHashRedirect", () => {
  it("replaces (not pushes) the URL for an old hash", () => {
    window.history.replaceState(null, "", "/#rig-check");
    render(<LegacyHashRedirect />);
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith("/run-it");
  });

  it("does nothing for a hash that still lives on Home, or no hash", () => {
    window.history.replaceState(null, "", "/#why");
    const { unmount } = render(<LegacyHashRedirect />);
    expect(replace).not.toHaveBeenCalled();
    unmount();
    window.history.replaceState(null, "", "/");
    render(<LegacyHashRedirect />);
    expect(replace).not.toHaveBeenCalled();
  });

  it("also follows a hash change while Home is open", () => {
    render(<LegacyHashRedirect />);
    window.history.replaceState(null, "", "/#compare");
    act(() => {
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });
    expect(replace).toHaveBeenCalledWith("/proof#compare");
  });

  it("renders nothing", () => {
    const { container } = render(<LegacyHashRedirect />);
    expect(container).toBeEmptyDOMElement();
  });
});
