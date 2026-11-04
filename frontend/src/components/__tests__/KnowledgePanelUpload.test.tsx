import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { toast } from "sonner";
import { KnowledgePanel } from "@/components/agents/KnowledgePanel";
import { api } from "@/lib/api";
import { PDF_TEXT_HINT, SCANNED_PDF_MESSAGE, friendlyUploadError } from "@/lib/uploadErrors";

jest.mock("next/link", () => ({ __esModule: true, default: ({ children, ...p }: { children: React.ReactNode }) => <a {...p}>{children}</a> }));
jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn(), info: jest.fn() } }));
jest.mock("@/lib/api", () => ({
  api: { upload: jest.fn(), del: jest.fn(), get: jest.fn() },
  ApiError: class extends Error {},
}));
jest.mock("@/hooks/useApi", () => ({
  useAgentDocuments: () => ({ data: [], isLoading: false, error: undefined, mutate: jest.fn() }),
  useAgents: () => ({ mutate: jest.fn() }),
  useExampleQuestions: () => ({ data: [] }),
}));

const upload = api.upload as jest.Mock;

/** What the backend sends for a PDF with no text layer (backend.py: AppError NO_TEXT_FOUND, HTTP 400). */
const scannedError = () =>
  Object.assign(new Error("No text found - OCR is not supported in this version."), { code: "NO_TEXT_FOUND", status: 400 });

const file = (name: string, type = "application/pdf") => new File(["x"], name, { type });
const choose = (files: File[]) => fireEvent.change(screen.getByLabelText("Upload documents"), { target: { files } });

beforeEach(() => jest.clearAllMocks());

describe("friendlyUploadError", () => {
  it("maps the backend's no-text answers on a PDF to the scanned-PDF next steps", () => {
    expect(friendlyUploadError(scannedError(), "a.pdf")).toBe(SCANNED_PDF_MESSAGE);
    expect(friendlyUploadError(Object.assign(new Error("No extractable text found"), { code: "NO_TEXT" }), "a.pdf")).toBe(SCANNED_PDF_MESSAGE);
    // by message alone, when the code is missing
    expect(friendlyUploadError(new Error("No text found - OCR is not supported in this version."), "A.PDF")).toBe(SCANNED_PDF_MESSAGE);
  });

  it("does not call a text file scanned, and passes other errors through", () => {
    expect(friendlyUploadError(scannedError(), "notes.txt")).toBe("This file has no text in it. Add some text and try again.");
    expect(friendlyUploadError(new Error("Network error"), "a.pdf")).toBe("Network error");
    expect(friendlyUploadError("weird", "a.pdf", "Could not upload a.pdf.")).toBe("Could not upload a.pdf.");
  });
});

describe("Agent documents upload", () => {
  it("says before uploading that a PDF needs selectable text", () => {
    render(<KnowledgePanel agentId="a1" />);
    expect(screen.getByTestId("pdf-text-hint").textContent).toBe(PDF_TEXT_HINT);
    expect(PDF_TEXT_HINT).toBe("PDF must have selectable text. Scanned pages are not supported.");
    expect(screen.getByLabelText("Upload documents").getAttribute("aria-describedby")).toBe(screen.getByTestId("pdf-text-hint").id);
  });

  it("shows plain next steps for a scanned PDF, keeps it in a list with Retry, and a retry that works clears it", async () => {
    upload.mockRejectedValueOnce(scannedError());
    render(<KnowledgePanel agentId="a1" />);
    choose([file("scan.pdf")]);

    const list = await screen.findByRole("list", { name: "Files that did not upload" });
    expect(within(list).getByText("scan.pdf")).toBeTruthy();
    expect(within(list).getByText(SCANNED_PDF_MESSAGE)).toBeTruthy();
    expect(toast.error).toHaveBeenCalledWith(`scan.pdf: ${SCANNED_PDF_MESSAGE}`);
    expect(screen.queryByTestId("upload-ready")).toBeNull();

    upload.mockResolvedValueOnce({ chunks_created: 4 });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Retry scan.pdf" }));
    });
    expect(upload).toHaveBeenCalledTimes(2);
    expect(((upload.mock.calls[1][1] as FormData).get("file") as File).name).toBe("scan.pdf");
    await waitFor(() => expect(screen.queryByRole("list", { name: "Files that did not upload" })).toBeNull());
    expect(screen.getByTestId("upload-ready").textContent).toContain("Ready: 4 sections. Try asking a question.");
  });

  it("keeps each failed file with its own Retry while the others go through", async () => {
    upload
      .mockResolvedValueOnce({ chunks_created: 1 })
      .mockRejectedValueOnce(new Error("Server unavailable"));
    render(<KnowledgePanel agentId="a1" />);
    choose([file("ok.txt", "text/plain"), file("bad.pdf")]);

    await screen.findByRole("button", { name: "Retry bad.pdf" });
    expect(screen.queryByRole("button", { name: "Retry ok.txt" })).toBeNull();
    expect(screen.getByText("Server unavailable")).toBeTruthy();
    expect(screen.getByTestId("upload-ready").textContent).toContain("Ready: 1 section. Try asking a question.");

    fireEvent.click(screen.getByRole("button", { name: "Dismiss bad.pdf" }));
    expect(screen.queryByRole("button", { name: "Retry bad.pdf" })).toBeNull();
  });

  it("lists an unsupported file type without a Retry (sending it again cannot help)", async () => {
    render(<KnowledgePanel agentId="a1" />);
    choose([file("photo.png", "image/png")]);
    const list = await screen.findByRole("list", { name: "Files that did not upload" });
    expect(within(list).getByText("Only PDF, TXT and MD files are supported.")).toBeTruthy();
    expect(within(list).queryByRole("button", { name: /Retry/ })).toBeNull();
    expect(upload).not.toHaveBeenCalled();
  });

  it("shows a progressbar with a value while bytes go up, an indeterminate one while indexing, and 'File 2 of 2'", async () => {
    const gates: { onProgress: (n: number) => void; finish: () => void }[] = [];
    upload.mockImplementation(
      (_p: string, _fd: FormData, opts: { onProgress: (n: number) => void }) =>
        new Promise((resolve) => gates.push({ onProgress: opts.onProgress, finish: () => resolve({ chunks_created: 3 }) }))
    );
    render(<KnowledgePanel agentId="a1" />);
    choose([file("a.txt", "text/plain"), file("b.md", "text/markdown")]);

    expect(await screen.findByText(/File 1 of 2: a\.txt/)).toBeTruthy();
    act(() => gates[0].onProgress(40));
    const bar = screen.getByRole("progressbar");
    expect(bar.getAttribute("aria-valuenow")).toBe("40");
    expect(bar.getAttribute("aria-valuemin")).toBe("0");
    expect(bar.getAttribute("aria-valuemax")).toBe("100");

    act(() => gates[0].onProgress(100)); // uploaded; the server is now reading and indexing: no percentage exists
    expect(screen.getByRole("progressbar").hasAttribute("aria-valuenow")).toBe(false);
    expect(screen.getByRole("progressbar").getAttribute("aria-valuetext")).toBe("Processing and indexing");

    await act(async () => gates[0].finish());
    expect(await screen.findByText(/File 2 of 2: b\.md/)).toBeTruthy();
    await waitFor(() => expect(gates).toHaveLength(2));
    await act(async () => gates[1].finish());

    const ready = await screen.findByTestId("upload-ready");
    expect(ready.textContent).toContain("Ready: 6 sections. Try asking a question.");
    expect(within(ready).getByRole("link").getAttribute("href")).toBe("/dashboard/playground?agent=a1");
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("says just 'Uploading <name>' for a single file", async () => {
    upload.mockReturnValue(new Promise(() => undefined));
    render(<KnowledgePanel agentId="a1" />);
    choose([file("one.pdf")]);
    expect(await screen.findByText("Uploading one.pdf")).toBeTruthy();
    expect(screen.queryByText(/File 1 of/)).toBeNull();
  });
});
