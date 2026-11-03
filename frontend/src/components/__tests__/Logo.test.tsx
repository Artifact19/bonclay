import { render, screen } from "@testing-library/react";
import { Logo, LogoMark, Wordmark } from "../Logo";

describe("Wordmark", () => {
  it("is live text that reads exactly BonClay", () => {
    render(<Wordmark />);
    expect(screen.getByText("BonClay")).toBeInTheDocument();
  });

  it("sizes the type from the size prop and only shows VOICE STUDIO when asked", () => {
    const { container, rerender } = render(<Wordmark size={64} />);
    expect(container.firstElementChild).toHaveStyle({ fontSize: "64px" });
    expect(screen.queryByText("VOICE STUDIO")).not.toBeInTheDocument();
    rerender(<Wordmark size={64} subtitle />);
    expect(screen.getByText("VOICE STUDIO")).toBeInTheDocument();
  });

  it("applies the requested tone", () => {
    const { container, rerender } = render(<Wordmark tone="dark" />);
    expect(container.firstElementChild).toHaveClass("dark");
    rerender(<Wordmark tone="light" />);
    expect(container.firstElementChild).toHaveClass("light");
  });
});

describe("LogoMark", () => {
  it("is decorative by default and named when a label is given", () => {
    const { container, rerender } = render(<LogoMark size={40} />);
    const img = container.querySelector("img")!;
    expect(img).toHaveAttribute("alt", "");
    expect(img).toHaveAttribute("width", "40");
    expect(img.getAttribute("src")).toContain("bonclay-icon.svg");
    rerender(<LogoMark size={40} label="BonClay" />);
    expect(screen.getByAltText("BonClay")).toBeInTheDocument();
  });
});

describe("Logo lockup", () => {
  it("renders icon and wordmark at the 80:64 reference ratio", () => {
    const { container } = render(<Logo size={80} />);
    expect(container.querySelector("img")).toHaveAttribute("width", "80");
    expect(container.querySelector("img + span")).toHaveStyle({ fontSize: "64px" });
  });

  it("hides VOICE STUDIO at small sizes and shows it once the lockup can carry it", () => {
    const { rerender } = render(<Logo size={32} />);
    expect(screen.queryByText("VOICE STUDIO")).not.toBeInTheDocument();
    rerender(<Logo size={80} />);
    expect(screen.getByText("VOICE STUDIO")).toBeInTheDocument();
    rerender(<Logo size={32} subtitle />);
    expect(screen.getByText("VOICE STUDIO")).toBeInTheDocument();
  });

  it("supports mark-only and text-only use", () => {
    const { container, rerender } = render(<Logo showText={false} />);
    expect(screen.queryByText("BonClay")).not.toBeInTheDocument();
    expect(container.querySelector("img")).toBeInTheDocument();
    rerender(<Logo onlyText />);
    expect(screen.getByText("BonClay")).toBeInTheDocument();
    expect(container.querySelector("img")).not.toBeInTheDocument();
  });
});
