import pytest

from app.services.documents import DocumentService
from app.services.metadata import MetadataStore
from tests.conftest import FakeEmbeddingProvider, FakeVectorStore


class FailingMetadataStore(MetadataStore):
    def replace_document(self, document, vector_ids):
        raise RuntimeError("database write failed")


def test_failed_metadata_write_cleans_new_vectors(test_settings):
    metadata = FailingMetadataStore(test_settings.metadata_db_path)
    metadata.initialize()
    vectors = FakeVectorStore()
    service = DocumentService(
        settings=test_settings,
        metadata=metadata,
        embeddings=FakeEmbeddingProvider(),
        vectors=vectors,
    )

    with pytest.raises(RuntimeError):
        service.ingest_upload("faq.txt", b"a" * 150)

    assert vectors.count() == 0


def test_embedding_fingerprint_controls_idempotency(test_settings):
    metadata = MetadataStore(test_settings.metadata_db_path)
    metadata.initialize()
    vectors = FakeVectorStore()
    first_embeddings = FakeEmbeddingProvider()
    service = DocumentService(
        settings=test_settings,
        metadata=metadata,
        embeddings=first_embeddings,
        vectors=vectors,
    )

    first = service.ingest_upload("faq.txt", b"a" * 150)
    first_vector_ids = set(vectors.records)
    repeated = service.ingest_upload("faq.txt", b"a" * 150)

    assert first.unchanged is False
    assert repeated.unchanged is True
    stored = metadata.document_for_source("upload:faq.txt")
    assert stored is not None
    assert stored.embedding_fingerprint == first_embeddings.fingerprint

    class NewEmbeddingProvider(FakeEmbeddingProvider):
        fingerprint = "test:embedding-v2"

    migrated_service = DocumentService(
        settings=test_settings,
        metadata=metadata,
        embeddings=NewEmbeddingProvider(),
        vectors=vectors,
    )
    migrated = migrated_service.ingest_upload("faq.txt", b"a" * 150)

    assert migrated.unchanged is False
    migrated_document = metadata.document_for_source("upload:faq.txt")
    assert migrated_document is not None
    assert migrated_document.embedding_fingerprint == "test:embedding-v2"
    assert first_vector_ids.isdisjoint(vectors.records)
