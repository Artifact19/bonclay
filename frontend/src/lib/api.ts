import { createClient } from "@/lib/supabase";
import { API_URL, ApiError, normalizeError } from "@/lib/apiBase";

export { API_URL, API_BASE_URL, ApiError, normalizeError } from "@/lib/apiBase";

export interface RequestOpts {
  headers?: Record<string, string>;
  signal?: AbortSignal;
  retries?: number;
  /** Skip auth header */
  anon?: boolean;
}

const MAX_RETRIES = 3;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const backoff = (attempt: number) => Math.min(250 * 2 ** attempt, 4000);

function storageHeader(): Record<string, string> {
  if (typeof window === "undefined") return {};
  try {
    const mode = localStorage.getItem("bonclay_storage_mode");
    return { "Storage-Mode": mode === "cloud" ? "cloud" : "local" };
  } catch {
    return { "Storage-Mode": "local" };
  }
}

async function authHeaders(anon?: boolean): Promise<Record<string, string>> {
  if (anon || typeof window === "undefined") return {};
  try {
    if (localStorage.getItem("bonclay_demo_mode") === "true") return {};
    const { data } = await createClient().auth.getSession();
    const token = data.session?.access_token;
    return token ? { Authorization: `Bearer ${token}` } : {};
  } catch {
    return {};
  }
}

async function baseHeaders(anon?: boolean): Promise<Record<string, string>> {
  return { ...storageHeader(), ...(await authHeaders(anon)) };
}

async function request(
  method: string,
  path: string,
  body: BodyInit | undefined,
  headers: Record<string, string>,
  opts: RequestOpts = {}
): Promise<Response> {
  const maxRetries = opts.retries ?? (method === "GET" ? MAX_RETRIES : 0);
  const url = path.startsWith("http") ? path : `${API_URL}${path}`;
  let lastErr: ApiError | undefined;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetch(url, {
        method,
        body,
        signal: opts.signal,
        headers: { ...(await baseHeaders(opts.anon)), ...headers, ...opts.headers },
      });
      if (res.ok) return res;
      const err = await normalizeError(res);
      if (res.status >= 500 && attempt < maxRetries) {
        lastErr = err;
        await sleep(backoff(attempt));
        continue;
      }
      throw err;
    } catch (e) {
      if (e instanceof ApiError) throw e;
      if (opts.signal?.aborted) throw new ApiError("Request aborted", 0, "aborted");
      lastErr = new ApiError(e instanceof Error ? e.message : "Network error", 0, "network_error");
      if (attempt < maxRetries) {
        await sleep(backoff(attempt));
        continue;
      }
      throw lastErr;
    }
  }
  throw lastErr ?? new ApiError("Request failed");
}

const JSON_HEADERS = { "Content-Type": "application/json" };

async function parse<T>(res: Response): Promise<T> {
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  if (!text) return undefined as T;
  try {
    return JSON.parse(text) as T;
  } catch {
    return text as unknown as T;
  }
}

const jsonBody = (b: unknown) => (b === undefined ? undefined : JSON.stringify(b));

export const api = {
  async get<T>(path: string, opts?: RequestOpts): Promise<T> {
    return parse<T>(await request("GET", path, undefined, {}, opts));
  },
  async post<T>(path: string, body?: unknown, opts?: RequestOpts): Promise<T> {
    return parse<T>(await request("POST", path, jsonBody(body), JSON_HEADERS, opts));
  },
  async put<T>(path: string, body?: unknown, opts?: RequestOpts): Promise<T> {
    return parse<T>(await request("PUT", path, jsonBody(body), JSON_HEADERS, opts));
  },
  async del<T>(path: string, opts?: RequestOpts): Promise<T> {
    return parse<T>(await request("DELETE", path, undefined, {}, opts));
  },
  /** multipart upload with progress (XHR, since fetch has no upload progress). */
  upload<T>(
    path: string,
    formData: FormData,
    {
      onProgress,
      signal,
      headers: extra,
    }: { onProgress?: (pct: number) => void; signal?: AbortSignal; headers?: Record<string, string> } = {}
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      baseHeaders().then((base) => {
        const headers = { ...base, ...extra };
        const xhr = new XMLHttpRequest();
        xhr.open("POST", path.startsWith("http") ? path : `${API_URL}${path}`);
        Object.entries(headers).forEach(([k, v]) => xhr.setRequestHeader(k, v));
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) onProgress?.(Math.round((e.loaded / e.total) * 100));
        };
        xhr.onerror = () => reject(new ApiError("Network error", 0, "network_error"));
        xhr.onabort = () => reject(new ApiError("Upload aborted", 0, "aborted"));
        xhr.onload = () => {
          const text = xhr.responseText;
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve((text ? JSON.parse(text) : undefined) as T);
            } catch {
              resolve(text as unknown as T);
            }
          } else {
            normalizeError(new Response(text, { status: xhr.status, statusText: xhr.statusText })).then(reject);
          }
        };
        signal?.addEventListener("abort", () => xhr.abort());
        xhr.send(formData);
      });
    });
  },
  /** Fetch binary (audio). GET when no body, POST otherwise. */
  async blob(path: string, body?: object | FormData, opts?: RequestOpts): Promise<Blob> {
    const isForm = typeof FormData !== "undefined" && body instanceof FormData;
    const res = await request(
      body === undefined ? "GET" : "POST",
      path,
      isForm ? (body as FormData) : jsonBody(body),
      body === undefined || isForm ? {} : JSON_HEADERS,
      opts
    );
    return res.blob();
  },
  /** POST Server-Sent-Events stream parser. The body is JSON, or multipart when it is a FormData. */
  async sse(
    path: string,
    body: unknown,
    { onEvent, signal }: { onEvent: (event: string, data: unknown) => void; signal?: AbortSignal }
  ): Promise<void> {
    const isForm = typeof FormData !== "undefined" && body instanceof FormData;
    const res = await request(
      "POST",
      path,
      isForm ? (body as FormData) : jsonBody(body),
      { ...(isForm ? {} : JSON_HEADERS), Accept: "text/event-stream" },
      { signal, retries: 0 }
    );
    if (!res.body) throw new ApiError("No response stream", res.status, "no_stream");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    const flush = (block: string) => {
      let event = "message";
      const dataLines: string[] = [];
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith(":")) continue;
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
      }
      if (!dataLines.length) return;
      const raw = dataLines.join("\n");
      let data: unknown = raw;
      try {
        data = JSON.parse(raw);
      } catch {
        /* keep raw string */
      }
      onEvent(event, data);
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let m: RegExpMatchArray | null;
      while ((m = buf.match(/\r?\n\r?\n/)) && m.index !== undefined) {
        flush(buf.slice(0, m.index));
        buf = buf.slice(m.index + m[0].length);
      }
    }
    if (buf.trim()) flush(buf);
  },
};
