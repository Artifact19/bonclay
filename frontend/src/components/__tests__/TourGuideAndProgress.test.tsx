import { fireEvent, render, screen } from "@testing-library/react";
import { TaskProgressBar } from "@/components/ui/TaskProgressBar";
import { TourGuide, TourTriggerButton } from "@/components/TourGuide";

describe("TaskProgressBar", () => {
  it("renders the clamped percentage, label, detail, and accessible progressbar role", () => {
    render(
      <TaskProgressBar
        value={68.4}
        label="Generating speech"
        detail="Synthesizing audio and embedding AudioSeal watermark..."
      />
    );
    const bar = screen.getByRole("progressbar", { name: "Generating speech" });
    expect(bar).toHaveAttribute("aria-valuenow", "68");
    expect(bar).toHaveAttribute("aria-valuemin", "0");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
    expect(screen.getByTestId("task-progress-bar-pct")).toHaveTextContent("68%");
    expect(screen.getByText(/Synthesizing audio/)).toBeInTheDocument();
  });
});

describe("TourGuide", () => {
  it("opens, steps through plain-English explanations, and highlights options", () => {
    render(
      <div>
        <section id="top">Hero section</section>
        <section id="products">Products section</section>
        <TourGuide mode="site" open={true} onClose={jest.fn()} />
      </div>
    );

    const dialog = screen.getByRole("dialog");
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText(/Step 1 of 5/i)).toBeInTheDocument();
    expect(screen.getByText(/Plain-English Tip/i)).toBeInTheDocument();
    expect(screen.getByTestId("tour-progress-pct")).toHaveTextContent("20%");

    fireEvent.click(screen.getByRole("button", { name: /Next option/i }));
    expect(screen.getByText(/Step 2 of 5/i)).toBeInTheDocument();
    expect(screen.getByTestId("tour-progress-pct")).toHaveTextContent("40%");
  });

  it("renders TourTriggerButton and fires onClick", () => {
    const onClick = jest.fn();
    render(<TourTriggerButton onClick={onClick} />);
    fireEvent.click(screen.getByRole("button", { name: /Start interactive tour guide/i }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
