from datetime import datetime, timezone

from app.services.chat import ChatService
from app.services.metadata import MetadataStore, StoredDocument
from tests.conftest import FakeChatProvider, FakeEmbeddingProvider


class CandidateVectorStore:
    def query(self, query_embedding, top_k):
        return {
            "ids": [["stale", "far", "active", "active-second"]],
            "documents": [["old text", "far text", "current text", "second text"]],
            "metadatas": [
                [
                    {"document_id": "old", "source_name": "old.txt", "source_type": "upload"},
                    {
                        "document_id": "doc-far",
                        "source_name": "far.txt",
                        "source_type": "upload",
                        "chunk_index": 0,
                    },
                    {
                        "document_id": "doc-1",
                        "source_name": "faq.txt",
                        "source_type": "upload",
                        "source_url": "",
                        "chunk_index": 0,
                    },
                    {
                        "document_id": "doc-1",
                        "source_name": "faq.txt",
                        "source_type": "upload",
                        "source_url": "",
                        "chunk_index": 1,
                    },
                ]
            ],
            "distances": [[0.1, 0.9, 0.2, 0.3]],
        }


def _service(test_settings, response="پاسخ آزمایشی [S1]"):
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
            chunk_count=2,
            updated_at=datetime.now(timezone.utc),
        ),
        ["active", "active-second"],
    )
    embeddings, chat = FakeEmbeddingProvider(), FakeChatProvider(response)
    return (
        ChatService(
            metadata=metadata,
            embeddings=embeddings,
            vectors=CandidateVectorStore(),
            chat=chat,
            settings=test_settings,
        ),
        embeddings,
        chat,
    )


def test_chat_filters_stale_far_and_uncited_candidates(test_settings):
    service, _, _ = _service(test_settings)
    answer, sources = service.answer("question", [], top_k=2)
    assert answer == "پاسخ آزمایشی [S1]"
    assert [source["id"] for source in sources] == ["S1"]
    assert sources[0]["title"] == "faq.txt"


def test_chat_uses_native_history_roles_and_policy(test_settings):
    service, _, chat = _service(test_settings)
    service.answer("question", [{"role": "assistant", "content": "پاسخ قبلی"}], top_k=1)
    assert chat.messages[0][0]["role"] == "system"
    assert test_settings.assistant_name in chat.messages[0][0]["content"]
    assert chat.messages[0][1] == {"role": "assistant", "content": "پاسخ قبلی"}


def test_short_referential_follow_up_expands_retrieval_query(test_settings):
    service, embeddings, _ = _service(test_settings)
    service.answer("شرایطش چیه؟", [{"role": "user", "content": "مهلت مرجوعی کالا چقدر است؟"}], top_k=1)
    assert "مهلت مرجوعی" in embeddings.calls[0][0]


def test_bare_noun_follow_up_uses_previous_turn(test_settings):
    service, embeddings, _ = _service(test_settings)
    service.answer(
        "یادداشت‌ها",
        [
            {"role": "user", "content": "اطلاعاتم در محصول امن است؟"},
            {"role": "assistant", "content": "اطلاعات به صورت محلی ذخیره می‌شود. [S1]"},
        ],
        top_k=1,
    )
    query = embeddings.calls[0][0]
    assert "اطلاعاتم در محصول امن است" in query
    assert "اطلاعات به صورت محلی" in query
    assert "یادداشت‌ها" in query


def test_persian_social_messages_are_deterministic_without_retrieval(test_settings):
    service, embeddings, chat = _service(test_settings)
    answer, sources = service.answer("سلام", [], top_k=1)
    assert test_settings.assistant_name in answer
    assert sources == []
    assert embeddings.calls == []
    assert chat.messages == []
