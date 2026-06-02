import chromadb
from chromadb.api.models.Collection import Collection


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "demo_rag_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, ids: list[str], embeddings: list[list[float]], docs: list[str], metadatas: list[dict]):
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)

    def query(self, query_embedding: list[float], top_k: int):
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def count(self) -> int:
        return self.collection.count()
