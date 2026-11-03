import { render } from "@testing-library/react";
import { DevanagariKeyboard } from "@/components/studio/DevanagariKeyboard";
import { langAttr } from "@/lib/lang";

describe("langAttr", () => {
  it.each([
    ["hi", "hi"],
    ["mr", "mr"],
    ["bn", "bn"],
    ["ta", "ta"],
    ["te", "te"],
    ["gu", "gu"],
    ["kn", "kn"],
    ["ml", "ml"],
    ["pa", "pa"],
    ["en", "en"],
    ["en-IN", "en"],
    ["HI", "hi"],
    ["hinglish", "hi-Latn"],
    ["hi-en", "hi-Latn"],
  ])("%s -> %s", (code, want) => {
    expect(langAttr(code)).toBe(want);
  });

  it.each([[undefined], [null], [""], ["auto"], ["unknown"], ["xx"]])("gives no lang for %p", (code) => {
    expect(langAttr(code as string | null | undefined)).toBeUndefined();
  });
});

describe("DevanagariKeyboard", () => {
  it("marks every row of keys as Hindi, and not the English row labels", () => {
    const { container } = render(<DevanagariKeyboard onInsert={() => undefined} onBackspace={() => undefined} />);
    const keyRows = container.querySelectorAll('[lang="hi"]');
    expect(keyRows).toHaveLength(4);
    keyRows.forEach((row) => expect(row.querySelectorAll("button").length).toBeGreaterThan(5));
    const keys = Array.from(container.querySelectorAll("button")).filter((b) => /[ऀ-ॿ]/.test(b.textContent ?? ""));
    expect(keys.length).toBeGreaterThan(50);
    keys.forEach((k) => expect(k.closest("[lang]")?.getAttribute("lang")).toBe("hi"));
    container.querySelectorAll("p").forEach((label) => expect(label.closest("[lang]")).toBeNull());
  });
});
