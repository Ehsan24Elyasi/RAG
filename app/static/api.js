/* Small fetch layer shared by the customer and administrator interfaces. */
export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function cleanError(payload, fallback) {
  if (!payload) return fallback;
  if (typeof payload.detail === "string") return payload.detail;
  if (typeof payload.message === "string") return payload.message;
  return fallback;
}

export async function requestJson(path, options = {}) {
  const { token, body, headers = {}, signal, ...rest } = options;
  const requestHeaders = { Accept: "application/json", ...headers };
  if (token) requestHeaders.Authorization = `Bearer ${token}`;
  if (body !== undefined && !(body instanceof FormData)) requestHeaders["Content-Type"] = "application/json";

  let response;
  try {
    response = await fetch(path, {
      ...rest,
      signal,
      headers: requestHeaders,
      body: body instanceof FormData ? body : body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ApiError("امکان برقراری ارتباط با سرور وجود ندارد. اتصال خود را بررسی کنید.");
  }

  let payload = null;
  try { payload = await response.json(); } catch { /* An empty/non-JSON response still has a usable status. */ }
  if (!response.ok) throw new ApiError(cleanError(payload, "درخواست انجام نشد. لطفاً دوباره تلاش کنید."), response.status);
  return payload;
}

export const chatApi = (message, history, signal) => requestJson("/api/chat", {
  method: "POST",
  body: { message, history },
  signal,
});

export const adminApi = {
  status: (token) => requestJson("/api/admin/status", { token }),
  documents: (token) => requestJson("/api/admin/documents", { token }),
  crawl: (token, body) => requestJson("/api/admin/crawl", { method: "POST", token, body }),
  upload: (token, file) => {
    const form = new FormData();
    form.append("file", file, file.name);
    return requestJson("/api/admin/upload", { method: "POST", token, body: form });
  },
};
