import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import UploadZone from "@/components/knowledge/UploadZone";
import { api } from "@/lib/api";
import { PDF_TEXT_HINT, SCANNED_PDF_MESSAGE } from "@/lib/uploadErrors";

jest.mock("sonner", () => ({ toast: { error: jest.fn(), success: jest.fn() } }));
jest.mock("@/lib/api", () => ({ api: { upload: jest.fn() }, ApiError: class extends Error {} }));

test("the file input has a label and status messages are announced", () => {
  const { container } = render(<UploadZone onUploaded={() => {}} />);
  const input = container.querySelector("input[type=file]") as HTMLInputElement;
  expect(input).toBeInTheDocument();
  expect(screen.getByLabelText(/Choose a PDF or TXT document/)).toBe(input);
  expect(screen.getByRole("status")).toBeInTheDocument();
});

test("says before uploading that a PDF needs selectable text", () => {
  render(<UploadZone onUploaded={() => {}} />);
  expect(screen.getByTestId("pdf-text-hint").textContent).toBe(PDF_TEXT_HINT);
});

test("a scanned PDF gets plain next steps instead of the server's OCR message", async () => {
  (api.upload as jest.Mock).mockRejectedValue(
    Object.assign(new Error("No extractable text found"), { code: "NO_TEXT", status: 400 })
  );
  const { container } = render(<UploadZone onUploaded={() => {}} />);
  const input = container.querySelector("input[type=file]") as HTMLInputElement;
  fireEvent.change(input, { target: { files: [new File(["x"], "scan.pdf", { type: "application/pdf" })] } });
  await waitFor(() => expect(screen.getByRole("status").textContent).toContain(SCANNED_PDF_MESSAGE));
});
