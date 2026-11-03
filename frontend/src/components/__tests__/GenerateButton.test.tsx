import { act, fireEvent, render, screen } from "@testing-library/react";
import { GenerateButton } from "@/components/studio/GenerateButton";

const noop = () => undefined;

describe("GenerateButton", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it("counts seconds without changing any live text: one hidden status says only 'Making audio'", () => {
    const { container } = render(<GenerateButton state="working" onGenerate={noop} onPlay={noop} onStop={noop} />);
    const live = container.querySelectorAll("[aria-live]");
    expect(live).toHaveLength(1);
    expect(live[0]).toBe(screen.getByTestId("generate-status"));
    expect(live[0].className).toContain("sr-only");
    expect(live[0].textContent).toBe("Making audio");

    const buttons = screen.getAllByRole("button");
    buttons.forEach((b) => expect(b.hasAttribute("aria-live")).toBe(false));
    expect(screen.getByTestId("generate-hint").hasAttribute("aria-live")).toBe(false);
    expect(screen.getByTestId("generate-elapsed").textContent).toBe("0 s");

    // Ten ticks later the visible seconds moved; the announcement did not.
    const seen: string[] = [];
    const observer = new MutationObserver(() => seen.push(live[0].textContent ?? ""));
    observer.observe(live[0], { childList: true, characterData: true, subtree: true });
    act(() => {
      jest.advanceTimersByTime(10_000);
    });
    expect(screen.getByTestId("generate-elapsed").textContent).toBe("10 s");
    expect(live[0].textContent).toBe("Making audio");
    expect(seen).toEqual([]);
    observer.disconnect();
  });

  it("shows an aesthetic percentage progress bar while working", () => {
    render(<GenerateButton state="working" onGenerate={noop} onPlay={noop} onStop={noop} />);
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
    expect(screen.getByTestId("generate-progress-pct").textContent).toMatch(/%/);
    expect(screen.getByTestId("generate-hint").textContent).toBe("Making audio. Usually 10 to 40 seconds.");
  });

  it("announces Audio ready when done and Failed: reason on an error, nothing while idle", () => {
    const { rerender } = render(<GenerateButton state="idle" onGenerate={noop} onPlay={noop} />);
    expect(screen.getByTestId("generate-status")).toBeEmptyDOMElement();
    rerender(<GenerateButton state="done" onGenerate={noop} onPlay={noop} />);
    expect(screen.getByTestId("generate-status").textContent).toBe("Audio ready");
    rerender(<GenerateButton state="idle" error="No voice selected" onGenerate={noop} onPlay={noop} />);
    expect(screen.getByTestId("generate-status").textContent).toBe("Failed: No voice selected");
  });

  it("has a Stop button only when it can stop, wired to onStop", () => {
    const onStop = jest.fn();
    const { rerender } = render(<GenerateButton state="working" onGenerate={noop} onPlay={noop} />);
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
    rerender(<GenerateButton state="working" onGenerate={noop} onPlay={noop} onStop={onStop} />);
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("Retry and Make again call their handlers (falling back to onGenerate)", () => {
    const onGenerate = jest.fn();
    const onRetry = jest.fn();
    const { rerender } = render(<GenerateButton state="idle" error="Boom" onGenerate={onGenerate} onPlay={noop} onRetry={onRetry} />);
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledTimes(1);

    rerender(<GenerateButton state="done" onGenerate={onGenerate} onPlay={noop} />);
    fireEvent.click(screen.getByRole("button", { name: "Make again" }));
    expect(onGenerate).toHaveBeenCalledTimes(1);
  });
});
