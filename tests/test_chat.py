from datetime import datetime, timezone

from app.services.chat import ChatService
from app.services.metadata import MetadataStore, StoredDocument
from tests.conftest import FakeChatProvider, FakeEmbeddingProvider


class StaleFirstVectorStore:
    def query(self, query_embedding, top_k):
        return {
            "ids": [["stale", "active"]],
            "documents": [["old text", "current text"]],
            "metadatas": [
                [
                    {"document_id": "old", "source_name": "old.txt", "source_type": "upload"},
                    {
                        "document_id": "doc-1",
                        "source_name": "faq.txt",
                        "source_type": "upload",
                        "source_url": "",
                        "chunk_index": 0,
                    },
                ]
            ],
        }


def test_chat_skips_stale_vector_candidates(test_settings):
    metadata = MetadataStore(test_settings.metadata_db_path)
    metadata.initialize()
    metadata.replace_document(
        StoredDocument(
            id="doc-1",
            source_key="upload:faq.txt",
            source_name="faq.txt",
            source_type="upload",
            source_url=None,
            content_hash="hash",
            embedding_fingerprint="test:embedding-v1",
            chunk_count=1,
            updated_at=datetime.now(timezone.utc),
        ),
        ["active"],
    )
    service = ChatService(
        metadata=metadata,
        embeddings=FakeEmbeddingProvider(),
        vectors=StaleFirstVectorStore(),
        chat=FakeChatProvider(),
    )

    answer, sources = service.answer("question", [], top_k=1)

    assert answer == "پاسخ آزمایشی [S1]"
    assert sources[0]["title"] == "faq.txt"
