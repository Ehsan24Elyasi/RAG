import pytest

from app.rag.ingestion import DocumentParseError, parse_document_bytes


def test_parses_txt_and_json():
    text, kind = parse_document_bytes("faq.txt", "سلام".encode(), max_pdf_pages=2, max_extracted_chars=100)
    assert text == "سلام"
    assert kind == "txt"

    text, kind = parse_document_bytes(
        "faq.json", b'{"b": 2, "a": 1}', max_pdf_pages=2, max_extracted_chars=100
    )
    assert '"a": 1' in text
    assert kind == "json"


def test_rejects_unsupported_empty_and_fake_pdf():
    with pytest.raises(DocumentParseError):
        parse_document_bytes("faq.md", b"hello", max_pdf_pages=2, max_extracted_chars=100)
    with pytest.raises(DocumentParseError):
        parse_document_bytes("faq.txt", b"", max_pdf_pages=2, max_extracted_chars=100)
    with pytest.raises(DocumentParseError):
        parse_document_bytes("faq.pdf", b"not a pdf", max_pdf_pages=2, max_extracted_chars=100)
    with pytest.raises(DocumentParseError):
        parse_document_bytes("faq.pdf", b"%PDF-", max_pdf_pages=2, max_extracted_chars=100)
