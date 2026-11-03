import { render, screen } from "@testing-library/react";
import { Button } from "@/components/ui/Button";

test("a disabled button explains itself with a tooltip and an accessible description", () => {
  render(<Button disabled disabledReason="Type something to speak">Generate</Button>);
  const b = screen.getByRole("button", { name: "Generate" });
  expect(b).toBeDisabled();
  expect(b).toHaveAttribute("title", "Type something to speak");
  expect(b).toHaveAccessibleDescription("Type something to speak");
});

test("an enabled button shows no reason and keeps its own title", () => {
  render(<Button disabledReason="Not now" title="Hi">Go</Button>);
  const b = screen.getByRole("button", { name: "Go" });
  expect(b).toHaveAttribute("title", "Hi");
  expect(b).not.toHaveAttribute("aria-describedby");
  expect(screen.queryByText("Not now")).toBeNull();
});

test("existing aria-describedby is kept alongside the reason", () => {
  render(
    <>
      <p id="x">Extra</p>
      <Button disabled disabledReason="Reason" aria-describedby="x">Go</Button>
    </>
  );
  expect(screen.getByRole("button")).toHaveAccessibleDescription("Extra Reason");
});

test("a button without a reason is unchanged", () => {
  render(<Button disabled>Go</Button>);
  expect(screen.getByRole("button")).not.toHaveAttribute("title");
});
