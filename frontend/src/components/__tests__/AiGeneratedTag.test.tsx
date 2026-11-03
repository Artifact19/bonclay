import { render, screen } from "@testing-library/react";
import { AiGeneratedTag, AI_GENERATED_TOOLTIP } from "@/components/audio/AiGeneratedTag";

describe("AiGeneratedTag", () => {
  it("labels synthetic audio and explains it in a tooltip", () => {
    render(<AiGeneratedTag />);
    const tag = screen.getByTestId("ai-generated-tag");
    expect(tag).toHaveTextContent("AI-generated");
    expect(tag).toHaveAttribute("title", AI_GENERATED_TOOLTIP);
    expect(AI_GENERATED_TOOLTIP).toBe("Synthetic speech made with BonClay");
  });
});
