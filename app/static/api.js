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

export const conversationApi = {
  create: (token, title, signal) => requestJson("/api/conversations", {
    method: "POST",
    token,
    body: title ? { title } : {},
    signal,
  }),
  sendMessage: (token, conversationId, message, clientMessageId, signal) => requestJson(
    `/api/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: "POST",
      token,
      body: { message, client_message_id: clientMessageId },
      signal,
    },
  ),
  requestHandoff: (token, conversationId, reason, signal) => requestJson(
    `/api/conversations/${encodeURIComponent(conversationId)}/handoff`,
    { method: "POST", token, body: { reason: reason || null }, signal },
  ),
};

export const configApi = (signal) => requestJson("/api/config", { signal });

export const localWidgetBootstrapApi = (signal) => requestJson("/api/dev/widget-bootstrap", {
  method: "POST",
  signal,
});

export const adminApi = {
  status: (token) => requestJson("/api/admin/status", { token }),
  metrics: (token, days = 30) => requestJson(`/api/admin/metrics?days=${encodeURIComponent(days)}`, { token }),
  conversations: (token, filters = {}) => {
    const query = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") query.set(key, value);
    });
    return requestJson(`/api/admin/conversations?${query}`, { token });
  },
  conversation: (token, id) => requestJson(`/api/admin/conversations/${encodeURIComponent(id)}`, { token }),
  handoffs: (token, filters = {}) => {
    const query = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") query.set(key, value);
    });
    return requestJson(`/api/admin/handoffs?${query}`, { token });
  },
  updateHandoff: (token, id, status) => requestJson(`/api/admin/handoffs/${encodeURIComponent(id)}`, {
    method: "PATCH", token, body: { status },
  }),
  documents: (token) => requestJson("/api/admin/documents", { token }),
  deleteDocument: (token, id) => requestJson(`/api/admin/documents/${encodeURIComponent(id)}`, { method: "DELETE", token }),
  crawl: (token, body) => requestJson("/api/admin/crawl", { method: "POST", token, body }),
  upload: (token, file) => {
    const form = new FormData();
    form.append("file", file, file.name);
    return requestJson("/api/admin/upload", { method: "POST", token, body: form });
  },
};
