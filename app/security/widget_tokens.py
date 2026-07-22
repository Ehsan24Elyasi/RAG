from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any


class WidgetTokenError(ValueError):
    pass


@dataclass(frozen=True)
class WidgetIdentity:
    workspace_id: str
    external_user_id: str
    display_name: str | None = None
    email: str | None = None


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    try:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)
    except Exception as exc:
        raise WidgetTokenError("Invalid widget token encoding.") from exc


def _json_segment(value: dict[str, Any]) -> str:
    return _encode(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def create_widget_token(
    secret: str,
    *,
    workspace_id: str,
    external_user_id: str,
    audience: str = "rag-widget",
    expires_in_seconds: int = 300,
    display_name: str | None = None,
    email: str | None = None,
    now: int | None = None,
) -> str:
    issued_at = int(time.time()) if now is None else now
    payload: dict[str, Any] = {
        "workspace_id": workspace_id,
        "external_user_id": external_user_id,
        "aud": audience,
        "iat": issued_at,
        "exp": issued_at + expires_in_seconds,
    }
    if display_name:
        payload["display_name"] = display_name
    if email:
        payload["email"] = email
    header = _json_segment({"alg": "HS256", "typ": "JWT"})
    body = _json_segment(payload)
    signature = _encode(hmac.new(secret.encode("utf-8"), f"{header}.{body}".encode("ascii"), hashlib.sha256).digest())
    return f"{header}.{body}.{signature}"


def verify_widget_token(
    token: str,
    secret: str,
    *,
    audience: str = "rag-widget",
    clock_skew_seconds: int = 30,
    now: int | None = None,
) -> WidgetIdentity:
    if len(token) > 4096:
        raise WidgetTokenError("Widget token is too large.")
    parts = token.split(".")
    if len(parts) != 3:
        raise WidgetTokenError("Invalid widget token.")
    header_segment, payload_segment, signature_segment = parts
    expected = hmac.new(
        secret.encode("utf-8"),
        f"{header_segment}.{payload_segment}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(_decode(signature_segment), expected):
        raise WidgetTokenError("Invalid widget token signature.")
    try:
        header = json.loads(_decode(header_segment))
        payload = json.loads(_decode(payload_segment))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WidgetTokenError("Invalid widget token payload.") from exc
    if header != {"alg": "HS256", "typ": "JWT"} or not isinstance(payload, dict):
        raise WidgetTokenError("Unsupported widget token.")

    current = int(time.time()) if now is None else now
    exp = payload.get("exp")
    issued_at = payload.get("iat")
    if not isinstance(exp, int) or exp < current - clock_skew_seconds:
        raise WidgetTokenError("Widget token has expired.")
    if not isinstance(issued_at, int) or issued_at > current + clock_skew_seconds:
        raise WidgetTokenError("Widget token is not valid yet.")
    if payload.get("aud") != audience:
        raise WidgetTokenError("Invalid widget token audience.")

    workspace_id = payload.get("workspace_id")
    external_user_id = payload.get("external_user_id")
    if not isinstance(workspace_id, str) or not workspace_id or len(workspace_id) > 100:
        raise WidgetTokenError("Invalid workspace identity.")
    if not isinstance(external_user_id, str) or not external_user_id or len(external_user_id) > 200:
        raise WidgetTokenError("Invalid user identity.")
    display_name = payload.get("display_name")
    email = payload.get("email")
    return WidgetIdentity(
        workspace_id=workspace_id,
        external_user_id=external_user_id,
        display_name=display_name if isinstance(display_name, str) else None,
        email=email if isinstance(email, str) else None,
    )
