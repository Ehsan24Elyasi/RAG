from app.config import settings
from app.llm.provider import LLMProvider
from app.rag.chunking import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.generator import generate_answer
from app.rag.ingestion import load_raw_documents
from app.rag.retriever import to_retrieved_items
from app.rag.vectorstore import VectorStore


class RagPipeline:
    def __init__(self):
        self.embedding_model = EmbeddingModel(settings.embedding_model)
        self.vectorstore = VectorStore(settings.chroma_persist_dir)
        provider = settings.llm_provider.lower().strip()
        api_key = settings.llm_api_key.strip()
        if provider == "ollama":
            self.llm = LLMProvider(api_key or "dummy", settings.llm_model, settings.llm_base_url)
        elif api_key:
            self.llm = LLMProvider(api_key, settings.llm_model, settings.llm_base_url)
        else:
            self.llm = None

    def ingest(self) -> dict:
        docs = load_raw_documents(settings.raw_data_dir)
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        for doc_idx, doc in enumerate(docs):
            chunks = chunk_text(doc["text"], settings.chunk_size, settings.chunk_overlap)
            for chunk_idx, chunk in enumerate(chunks):
                ids.append(f"{doc_idx}_{chunk_idx}")
                texts.append(chunk)
                metadatas.append(
                    {
                        "source_path": doc["source_path"],
                        "file_name": doc["file_name"],
                        "doc_type": doc["doc_type"],
                        "chunk_index": chunk_idx,
                    }
                )

        if texts:
            vectors = self.embedding_model.encode(texts)
            self.vectorstore.upsert(ids=ids, embeddings=vectors, docs=texts, metadatas=metadatas)

        return {"files_processed": len(docs), "chunks_created": len(texts)}

    def query(self, question: str, top_k: int) -> dict:
        query_embedding = self.embedding_model.encode([question])[0]
        query_result = self.vectorstore.query(query_embedding, top_k)
        retrieved_items = to_retrieved_items(query_result)

        if not self.llm:
            answer = "LLM_API_KEY is missing. Add it to .env."
        elif not retrieved_items:
            answer = "I don't know based on the provided documents."
        else:
            try:
                answer = generate_answer(question, retrieved_items, self.llm)
            except Exception as exc:
                answer = f"LLM request failed: {exc}"

        sources = [
            {
                "file_name": item["metadata"].get("file_name", ""),
                "source_path": item["metadata"].get("source_path", ""),
                "chunk_index": int(item["metadata"].get("chunk_index", 0)),
            }
            for item in retrieved_items
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_context": [item["text"] for item in retrieved_items],
        }
