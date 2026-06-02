from pydantic import BaseModel


class IngestResponse(BaseModel):
    files_processed: int
    chunks_created: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    debug: bool = False


class SourceItem(BaseModel):
    file_name: str
    source_path: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    retrieved_context: list[str] | None = None
