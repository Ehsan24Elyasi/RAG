from fastapi import FastAPI

from app.config import settings
from app.rag.pipeline import RagPipeline
from app.schemas import IngestResponse, QueryRequest, QueryResponse

app = FastAPI(title="RAG Demo")
pipeline = RagPipeline()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "collection_size": pipeline.vectorstore.count(),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    result = pipeline.ingest()
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = pipeline.query(req.question, req.top_k or settings.top_k)
    if not req.debug:
        result["retrieved_context"] = None
    return QueryResponse(**result)
