// The parts of the API client that do not touch auth. Kept apart from ./api (which imports the Supabase client) so the
// public pages, /verify in particular, can use them without shipping the sign-in code.

export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
/** @deprecated use API_URL */
export const API_BASE_URL = API_URL;

export class ApiError extends Error {
  status: number;
  code: string;
  details: unknown;
  constructor(message: string, status = 0, code = "unknown_error", details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

/** Normalize backend {error,code,details} and FastAPI {detail} bodies. */
export async function normalizeError(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    const text = await res.text();
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  } catch {
    /* ignore */
  }
  let message = res.statusText || `Request failed (${res.status})`;
  let code = `http_${res.status}`;
  let details: unknown = undefined;
  if (body && typeof body === "object") {
    const b = body as Record<string, unknown>;
    if (typeof b.error === "string") message = b.error;
    else if (typeof b.detail === "string") message = b.detail;
    else if (Array.isArray(b.detail)) {
      message = b.detail
        .map((d) =>
          d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : String(d)
        )
        .join("; ");
      details = b.detail;
    } else if (typeof b.message === "string") message = b.message;
    if (typeof b.code === "string") code = b.code;
    if (b.details !== undefined) details = b.details;
  } else if (typeof body === "string" && body) {
    message = body.slice(0, 300);
  }
  return new ApiError(message, res.status, code, details);
}
