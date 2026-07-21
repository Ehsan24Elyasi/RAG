from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content cannot be blank")
        return value


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be blank")
        return value


class SourceItem(BaseModel):
    id: str
    document_id: str
    title: str
    source_type: Literal["upload", "web"]
    url: str | None = None
    chunk_index: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]


class IngestionRunResponse(BaseModel):
    id: str
    kind: str
    source_label: str
    status: str
    documents_processed: int
    chunks_created: int
    error_message: str | None
    created_at: datetime
    finished_at: datetime | None


class AdminStatusResponse(BaseModel):
    status: Literal["ok", "degraded"]
    active_documents: int
    indexed_chunks: int
    chat_provider: str
    embedding_provider: str
    recent_runs: list[IngestionRunResponse]


class DocumentResponse(BaseModel):
    id: str
    source_type: Literal["upload", "web"]
    source_name: str
    source_url: str | None
    content_hash: str
    chunk_count: int
    updated_at: datetime


class DocumentsResponse(BaseModel):
    documents: list[DocumentResponse]


class IngestResponse(BaseModel):
    document_id: str
    source_name: str
    source_type: Literal["upload", "web"]
    chunks_created: int
    unchanged: bool = False
    run_id: str


class CrawlRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    max_pages: int | None = Field(default=None, ge=1, le=50)
    max_depth: int | None = Field(default=None, ge=0, le=5)


class CrawlResponse(BaseModel):
    pages_visited: int
    documents: list[IngestResponse]
    run_id: str
