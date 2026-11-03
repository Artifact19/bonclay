import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import TryDocumentPage from "@/app/(app)/dashboard/knowledge/try/page";
import { api } from "@/lib/api";

jest.mock("sonner", () => ({ toast: { error: jest.fn(), success: jest.fn() } }));
jest.mock("@/lib/api", () => ({
  api: { get: jest.fn(), post: jest.fn(), upload: jest.fn(), del: jest.fn(), blob: jest.fn() },
  ApiError: class extends Error {},
  API_URL: "http://api.test",
}));

const get = api.get as jest.Mock;
const upload = api.upload as jest.Mock;

beforeEach(() => {
  get.mockReset().mockResolvedValue({ documents: [] });
  upload.mockReset().mockResolvedValue({ doc_id: "r1", filename: "menu.pdf", chunks: 3 });
});

const view = () =>
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <TryDocumentPage />
    </SWRConfig>
  );

test("is retitled, says it is not saved to an agent and links back to Documents", () => {
  view();
  expect(screen.getByRole("heading", { level: 1, name: "Try a document" })).toBeInTheDocument();
  expect(
    screen.getByText("A quick test: upload any PDF, ask it a question and hear the answer. This is not saved to an agent.")
  ).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /back to documents/i })).toHaveAttribute("href", "/dashboard/knowledge");
});

test("still uploads to the demo (rag) endpoint, not to an agent", async () => {
  const { container } = view();
  const input = container.querySelector("input[type='file']") as HTMLInputElement;
  const file = new File(["%PDF-1.4"], "menu.pdf", { type: "application/pdf" });
  fireEvent.change(input, { target: { files: [file] } });
  await waitFor(() => expect(upload).toHaveBeenCalled());
  expect(upload.mock.calls[0][0]).toBe("/api/rag/upload");
  expect(get).toHaveBeenCalledWith("/api/rag/documents");
});
