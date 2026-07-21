"""Upload supported files from a directory through the admin API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx

SUPPORTED_SUFFIXES = {".txt", ".pdf", ".json"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="app/data/raw")
    parser.add_argument("--api-base", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--admin-key", default=os.getenv("ADMIN_API_KEY", ""))
    args = parser.parse_args()

    if not args.admin_key:
        raise SystemExit("Set ADMIN_API_KEY or pass --admin-key.")

    directory = Path(args.directory)
    files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise SystemExit(f"No supported files found under {directory}.")

    headers = {"Authorization": f"Bearer {args.admin_key}"}
    with httpx.Client(base_url=args.api_base, headers=headers, timeout=180) as client:
        for path in files:
            with path.open("rb") as source:
                response = client.post(
                    "/api/admin/upload",
                    files={"file": (path.name, source, "application/octet-stream")},
                )
            print(f"{path}: {response.status_code} {response.text}")
            response.raise_for_status()


if __name__ == "__main__":
    main()
