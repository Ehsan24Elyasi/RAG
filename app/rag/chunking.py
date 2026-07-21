from __future__ import annotations

import re

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!؟!?])\s+|\n+")


def _split_oversized(text: str, chunk_size: int) -> list[str]:
    """Split oversized prose at sentence boundaries, then at a hard limit."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_BREAK.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                sentence[start : start + chunk_size] for start in range(0, len(sentence), chunk_size)
            )
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunk prose by paragraph and sentence, retaining bounded word overlap."""
    text = text.strip()
    if not text:
        return []

    units = [part.strip() for part in _PARAGRAPH_BREAK.split(text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        for piece in _split_oversized(unit, chunk_size):
            if not current:
                current = piece
            elif len(current) + 2 + len(piece) <= chunk_size:
                current = f"{current}\n\n{piece}"
            else:
                chunks.append(current)
                overlap = current[-chunk_overlap:].lstrip() if chunk_overlap else ""
                current = f"{overlap}\n\n{piece}" if overlap else piece
                if len(current) > chunk_size:
                    chunks.extend(_split_oversized(current, chunk_size)[:-1])
                    current = _split_oversized(current, chunk_size)[-1]
    if current:
        chunks.append(current)
    return chunks
