import { fireEvent, render, screen } from "@testing-library/react";
import { RecordingScript, RECORDING_SCRIPTS } from "@/components/voices/RecordingScript";

const words = (t: string) => t.trim().split(/\s+/).length;

test("shows the English script first and switches language", () => {
  render(<RecordingScript recording={false} />);
  expect(screen.getByTestId("recording-script")).toHaveTextContent("Good morning! Why did the quick brown fox");
  fireEvent.click(screen.getByRole("tab", { name: "हिन्दी" }));
  expect(screen.getByTestId("recording-script")).toHaveTextContent("नमस्ते! आज मौसम बहुत अच्छा है");
  expect(screen.getByTestId("recording-script")).toHaveAttribute("lang", "hi");
  fireEvent.click(screen.getByRole("tab", { name: "Hinglish" }));
  expect(screen.getByTestId("recording-script")).toHaveTextContent("Aap kaise hain? Kya aap mujhe Thursday ko");
  expect(screen.getByTestId("recording-script")).toHaveAttribute("lang", "hi-Latn");
  fireEvent.click(screen.getByRole("tab", { name: "मराठी" }));
  expect(screen.getByTestId("recording-script")).toHaveTextContent("नमस्कार! आज हवामान खूप छान आहे");
});

test("the text gets large and the heading changes while recording", () => {
  const { rerender } = render(<RecordingScript recording={false} />);
  expect(screen.getByText("Read this when you record")).toBeInTheDocument();
  expect(screen.getByTestId("recording-script").className).not.toMatch(/text-xl/);
  rerender(<RecordingScript recording />);
  expect(screen.getByText("Read this aloud now")).toBeInTheDocument();
  expect(screen.getByTestId("recording-script").className).toMatch(/text-xl/);
});

test("each script is 50 to 60 words", () => {
  for (const s of RECORDING_SCRIPTS) {
    expect(words(s.text)).toBeGreaterThanOrEqual(50);
    expect(words(s.text)).toBeLessThanOrEqual(60);
  }
});
