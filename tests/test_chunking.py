from app.rag.chunking import chunk_text


def test_chunk_text_returns_chunks():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=700, chunk_overlap=100)
    assert len(chunks) >= 2
    assert all(len(c) <= 700 for c in chunks)
