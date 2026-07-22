from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.llm.types import GenerationResult


class FakeEmbeddingProvider:
    fingerprint = "test:embedding-v1"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]


class FakeChatProvider:
    def __init__(self, response: str = "پاسخ آزمایشی [S1]"):
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        self.messages.append(messages)
        return GenerationResult(text=self.response, model="test-chat", latency_ms=1)


class FakeVectorStore:
    def __init__(self):
        self.records: dict[str, tuple[list[float], str, dict]] = {}

    def upsert(self, ids, embeddings, docs, metadatas) -> None:
        for vector_id, vector, text, metadata in zip(ids, embeddings, docs, metadatas, strict=True):
            self.records[vector_id] = (vector, text, metadata)

    def delete(self, ids: list[str]) -> None:
        for vector_id in ids:
            self.records.pop(vector_id, None)

    def query(self, query_embedding, top_k: int) -> dict:
        selected = list(self.records.items())[:top_k]
        return {
            "ids": [[item[0] for item in selected]],
            "documents": [[item[1][1] for item in selected]],
            "metadatas": [[item[1][2] for item in selected]],
            "distances": [[0.0 for _ in selected]],
        }

    def count(self) -> int:
        return len(self.records)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        DATA_DIR=tmp_path,
        SQLITE_PATH=tmp_path / "rag.sqlite3",
        CHROMA_PERSIST_DIR=tmp_path / "chroma",
        ADMIN_API_KEY="test-admin-key",
        WIDGET_TOKEN_SECRET="test-widget-secret-that-is-long-enough",
        CHAT_PROVIDER="ollama",
        CHAT_MODEL="test-chat",
        EMBEDDING_PROVIDER="sentence-transformers",
        EMBEDDING_MODEL="test-embedding",
        CHUNK_SIZE=100,
        CHUNK_OVERLAP=10,
        MAX_HISTORY_MESSAGES=4,
        RETRIEVAL_MAX_DISTANCE=0.65,
    )
