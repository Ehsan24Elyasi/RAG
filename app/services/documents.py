from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.rag.chunking import chunk_text
from app.rag.ingestion import parse_document_bytes
from app.services.metadata import MetadataStore, StoredDocument


class EmbeddingClient(Protocol):
    fingerprint: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorClient(Protocol):
    def upsert(
        self, ids: list[str], embeddings: list[list[float]], docs: list[str], metadatas: list[dict]
    ) -> None: ...

    def delete(self, ids: list[str]) -> None: ...


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    source_name: str
    source_type: str
    source_url: str | None
    chunks_created: int
    unchanged: bool


class DocumentService:
    def __init__(
        self,
        *,
        settings: Settings,
        metadata: MetadataStore,
        embeddings: EmbeddingClient,
        vectors: VectorClient,
    ):
        self.settings = settings
        self.metadata = metadata
        self.embeddings = embeddings
        self.vectors = vectors

    @property
    def index_fingerprint(self) -> str:
        """Version all settings that change indexed vectors or chunk boundaries."""
        return (
            f"{self.embeddings.fingerprint}:chunker-paragraph-sentence-v2:"
            f"size={self.settings.chunk_size}:overlap={self.settings.chunk_overlap}"
        )

    def ingest_upload(self, filename: str, content: bytes) -> IngestedDocument:
        source_name = Path(filename or "upload").name
        if source_name in {"", ".", ".."}:
            raise ValueError("A valid filename is required.")
        text, _ = parse_document_bytes(
            source_name,
            content,
            max_extracted_chars=self.settings.max_extracted_chars,
            max_pdf_pages=self.settings.max_pdf_pages,
        )
        return self.ingest_text(
            source_key=f"upload:{source_name.casefold()}",
            source_name=source_name,
            source_type="upload",
            source_url=None,
            text=text,
        )

    def ingest_crawl_page(self, url: str, title: str, text: str) -> IngestedDocument:
        if len(text) > self.settings.max_extracted_chars:
            raise ValueError("The crawled text exceeds the configured limit.")
        return self.ingest_text(
            source_key=f"web:{url}",
            source_name=title.strip()[:240] or url,
            source_type="web",
            source_url=url,
            text=text,
        )

    def ingest_text(
        self,
        *,
        source_key: str,
        source_name: str,
        source_type: str,
        source_url: str | None,
        text: str,
    ) -> IngestedDocument:
        content_hash = sha256(text.encode("utf-8")).hexdigest()
        existing = self.metadata.document_for_source(source_key)
        if (
            existing
            and existing.content_hash == content_hash
            and existing.embedding_fingerprint == self.index_fingerprint
        ):
            return IngestedDocument(
                existing.id, source_name, source_type, source_url, existing.chunk_count, True
            )

        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("The document contains no indexable text.")

        document_id = self.metadata.stable_document_id(source_key)
        fingerprint_hash = sha256(self.index_fingerprint.encode("utf-8")).hexdigest()[:12]
        vector_ids = [
            f"{document_id}:{fingerprint_hash}:{content_hash}:{index}" for index in range(len(chunks))
        ]
        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), self.settings.embedding_batch_size):
            batch = chunks[start : start + self.settings.embedding_batch_size]
            batch_embeddings = self.embeddings.embed(batch)
            if len(batch_embeddings) != len(batch):
                raise RuntimeError("Embedding provider returned an unexpected result.")
            embeddings.extend(batch_embeddings)

        metadatas = [
            {
                "document_id": document_id,
                "content_hash": content_hash,
                "embedding_fingerprint": self.index_fingerprint,
                "source_name": source_name,
                "source_type": source_type,
                "source_url": source_url or "",
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]
        previous_ids = self.metadata.active_vector_ids_for_source(source_key)
        self.vectors.upsert(vector_ids, embeddings, chunks, metadatas)
        try:
            self.metadata.replace_document(
                StoredDocument(
                    id=document_id,
                    source_key=source_key,
                    source_name=source_name,
                    source_type=source_type,
                    source_url=source_url,
                    content_hash=content_hash,
                    embedding_fingerprint=self.index_fingerprint,
                    chunk_count=len(chunks),
                    updated_at=datetime.now(timezone.utc),
                ),
                vector_ids,
            )
        except Exception:
            self.vectors.delete(vector_ids)
            raise
        self.vectors.delete(list(set(previous_ids) - set(vector_ids)))
        return IngestedDocument(document_id, source_name, source_type, source_url, len(chunks), False)

    def delete_upload(self, workspace_id: str, document_id: str) -> bool:
        candidate = self.metadata.document_deletion_candidate(workspace_id, document_id)
        if candidate is None:
            return False
        if candidate.document.source_type != "upload":
            raise ValueError("Only uploaded documents can be deleted.")
        self.vectors.delete(candidate.vector_ids)
        if not self.metadata.delete_document_metadata(workspace_id, document_id):
            raise RuntimeError("Document metadata could not be deleted.")
        return True
