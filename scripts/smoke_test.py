"""Run a small health, status, and chat smoke test against a running API."""

from __future__ import annotations

import argparse
import os

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--admin-key", default=os.getenv("ADMIN_API_KEY", ""))
    parser.add_argument("--message", default="شرایط و قوانین پشتیبانی چیست؟")
    args = parser.parse_args()

    with httpx.Client(base_url=args.api_base, timeout=120) as client:
        health = client.get("/healthz")
        print("Health:", health.status_code, health.text)
        health.raise_for_status()

        if args.admin_key:
            status = client.get(
                "/api/admin/status",
                headers={"Authorization": f"Bearer {args.admin_key}"},
            )
            print("Admin status:", status.status_code, status.text)
            status.raise_for_status()

        chat = client.post("/api/chat", json={"message": args.message, "history": []})
        print("Chat:", chat.status_code, chat.text)
        chat.raise_for_status()


if __name__ == "__main__":
    main()
