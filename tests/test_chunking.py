from app.rag.chunking import chunk_text


def test_chunk_text_returns_bounded_chunks():
    chunks = chunk_text("a" * 1200, chunk_size=700, chunk_overlap=100)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 700 for chunk in chunks)


def test_chunk_text_keeps_paragraphs_and_sentences_when_possible():
    text = "جمله اول. جمله دوم.\n\nپاراگراف دوم با اطلاعات مهم."
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=0)
    assert chunks[0] == "جمله اول. جمله دوم."
    assert chunks[1] == "پاراگراف دوم با اطلاعات مهم."
