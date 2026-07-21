from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "customer_support_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self, ids: list[str], embeddings: list[list[float]], docs: list[str], metadatas: list[dict[str, Any]]
    ) -> None:
        if ids:
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)

    def query(self, query_embedding: list[float], top_k: int) -> dict[str, Any]:
        if top_k < 1 or self.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()
