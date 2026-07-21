from __future__ import annotations

from typing import Protocol

from app.rag.prompting import build_prompt
from app.services.metadata import MetadataStore


class ChatClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorClient(Protocol):
    def query(self, query_embedding: list[float], top_k: int) -> dict: ...


class ChatService:
    def __init__(
        self, *, metadata: MetadataStore, embeddings: EmbeddingClient, vectors: VectorClient, chat: ChatClient
    ):
        self.metadata = metadata
        self.embeddings = embeddings
        self.vectors = vectors
        self.chat = chat

    def answer(self, message: str, history: list[dict[str, str]], top_k: int) -> tuple[str, list[dict]]:
        query_vector = self.embeddings.embed([message])
        if len(query_vector) != 1:
            raise RuntimeError("Embedding provider returned an unexpected result.")
        result = self.vectors.query(query_vector[0], min(top_k * 3, 50))
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        active_ids = self.metadata.active_vector_ids([str(item) for item in ids])

        contexts: list[str] = []
        sources: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for index, vector_id in enumerate(ids):
            if str(vector_id) not in active_ids or index >= len(documents):
                continue
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            document_id = str(metadata.get("document_id", ""))
            chunk_index = int(metadata.get("chunk_index", 0))
            if (document_id, chunk_index) in seen:
                continue
            seen.add((document_id, chunk_index))
            contexts.append(str(documents[index]))
            sources.append(
                {
                    "id": f"S{len(sources) + 1}",
                    "document_id": document_id,
                    "title": str(metadata.get("source_name", "Support document")),
                    "source_type": str(metadata.get("source_type", "upload")),
                    "url": str(metadata.get("source_url", "")) or None,
                    "chunk_index": chunk_index,
                }
            )
            if len(contexts) >= top_k:
                break

        if not contexts:
            return "بر اساس اطلاعات موجود در دانش‌نامه، پاسخ این پرسش را نمی‌دانم.", []
        normalized_history = [(item["role"], item["content"]) for item in history if item.get("content")]
        answer = self.chat.generate(build_prompt(message, contexts, normalized_history)).strip()
        return answer or "بر اساس اطلاعات موجود در دانش‌نامه، پاسخ این پرسش را نمی‌دانم.", sources
