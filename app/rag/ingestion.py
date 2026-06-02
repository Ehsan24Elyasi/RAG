from pathlib import Path
import json

from pypdf import PdfReader


def _read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_raw_documents(raw_data_dir: str) -> list[dict]:
    base = Path(raw_data_dir)
    if not base.exists():
        return []

    docs: list[dict] = []
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".json", ".pdf"}:
            continue

        content = ""
        if path.suffix.lower() == ".json":
            content = json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False)
        elif path.suffix.lower() == ".pdf":
            content = _read_pdf_text(path)
        else:
            content = path.read_text(encoding="utf-8")

        docs.append(
            {
                "text": content,
                "source_path": str(path),
                "file_name": path.name,
                "doc_type": path.suffix.lower().lstrip("."),
            }
        )

    return docs
