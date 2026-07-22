from __future__ import annotations

import json
import logging
import uuid
from time import perf_counter

from fastapi import Request, Response

logger = logging.getLogger("app.requests")


async def request_context_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", "").strip()[:100] or str(uuid.uuid4())
    request.state.request_id = request_id
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round((perf_counter() - started) * 1000),
                },
                separators=(",", ":"),
            )
        )
