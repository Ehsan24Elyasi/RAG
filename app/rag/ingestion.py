from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

SUPPORTED_EXTENSIONS = {".txt", ".json", ".pdf"}


class DocumentParseError(ValueError):
    """Raised when an uploaded document is unsupported or unsafe to process."""


def parse_document_bytes(
    filename: str,
    content: bytes,
    *,
    max_extracted_chars: int | None = None,
    max_pdf_pages: int | None = None,
) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError("Only TXT, PDF, and JSON files are supported.")
    if not content:
        raise DocumentParseError("The uploaded document is empty.")

    try:
        if suffix == ".txt":
            text = content.decode("utf-8")
        elif suffix == ".json":
            value = json.loads(content.decode("utf-8"))
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        else:
            if not content.startswith(b"%PDF-"):
                raise DocumentParseError("The PDF signature is invalid.")
            reader = PdfReader(BytesIO(content))
            if max_pdf_pages is not None and len(reader.pages) > max_pdf_pages:
                raise DocumentParseError("The PDF exceeds the page limit.")
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except DocumentParseError:
        raise
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, OSError, PyPdfError, ValueError) as exc:
        raise DocumentParseError("The document could not be parsed.") from exc

    text = "\n".join(line.rstrip() for line in text.replace("\x00", "").splitlines()).strip()
    if not text:
        raise DocumentParseError("The document contains no extractable text.")
    if max_extracted_chars is not None and len(text) > max_extracted_chars:
        raise DocumentParseError("The extracted text exceeds the configured limit.")
    return text, suffix.lstrip(".")
