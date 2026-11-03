/** Shown before an upload, so nobody finds out after the fact. */
export const PDF_TEXT_HINT = "PDF must have selectable text. Scanned pages are not supported.";

export const SCANNED_PDF_MESSAGE = "This looks like a scanned PDF. Export it as a text PDF, or paste the text into a .txt file.";

/**
 * Plain next steps for a failed document upload. The backend answers a PDF with no text layer with NO_TEXT_FOUND
 * ("No text found. OCR is not supported in this version.", /api/knowledge/upload) or NO_TEXT ("No extractable text
 * found", /api/rag/upload); both mean a scanned PDF. Anything else keeps the server's own message.
 */
export function friendlyUploadError(e: unknown, filename: string, fallback = "Could not upload this file."): string {
  const code = (e as { code?: string } | null)?.code ?? "";
  const message = e instanceof Error ? e.message : "";
  const isPdf = /\.pdf$/i.test(filename);
  const noText = code === "NO_TEXT_FOUND" || code === "NO_TEXT" || /\bocr\b|no (extractable )?text/i.test(message);
  if (noText) return isPdf ? SCANNED_PDF_MESSAGE : "This file has no text in it. Add some text and try again.";
  if (code === "INVALID_PDF" || code === "PARSE_FAILED") {
    return "We could not read this file. Export it again as a text PDF, or paste the text into a .txt file.";
  }
  return message || fallback;
}
