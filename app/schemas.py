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


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationResponse(BaseModel):
    id: str
    title: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None


class ConversationsResponse(BaseModel):
    conversations: list[ConversationResponse]


class ConversationMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    client_message_id: str = Field(min_length=1, max_length=100)

    @field_validator("message", "client_message_id")
    @classmethod
    def normalize_message_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value


class ConversationMessageResponse(BaseModel):
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    answer: str
    sources: list[SourceItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]


class PublicConfigResponse(BaseModel):
    assistant_name: str
    company_name: str
    support_email: str | None = None
    support_phone: str | None = None
    support_url: str | None = None


class LocalWidgetBootstrapResponse(BaseModel):
    token: str
    expires_in_seconds: int


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


HandoffStatus = Literal["requested", "in_progress", "resolved", "cancelled"]


class AdminMetricsResponse(BaseModel):
    start: datetime
    end: datetime
    new_conversations: int
    user_messages: int
    assistant_messages: int
    successful_generations: int
    failed_generations: int
    average_latency_ms: float | None
    open_handoffs: int
    resolved_handoffs: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    lifetime_total_tokens: int
    unreported_runs: int


class AdminUserResponse(BaseModel):
    id: str
    external_id: str
    display_name: str | None
    email: str | None


class HandoffResponse(BaseModel):
    id: str
    conversation_id: str
    status: HandoffStatus
    reason: str | None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None


class HandoffCreateRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=1000)

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class HandoffCreateResponse(BaseModel):
    created: bool
    handoff: HandoffResponse


class HandoffUpdateRequest(BaseModel):
    status: Literal["in_progress", "resolved", "cancelled"]


class AdminConversationSummaryResponse(BaseModel):
    id: str
    title: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None
    user: AdminUserResponse
    last_message_preview: str | None
    message_count: int
    token_total: int
    active_handoff: HandoffResponse | None


class AdminConversationsResponse(BaseModel):
    conversations: list[AdminConversationSummaryResponse]
    total: int
    page: int
    page_size: int


class AdminCitationResponse(BaseModel):
    position: int
    document_id: str | None
    source_name: str | None
    source_url: str | None
    excerpt: str | None
    metadata: dict | None


class AdminGenerationResponse(BaseModel):
    provider: str | None
    model: str | None
    status: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: int | None
    finish_reason: str | None


class AdminTranscriptMessageResponse(BaseModel):
    id: str
    role: str
    status: str
    content: str | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None
    citations: list[AdminCitationResponse]
    generation: AdminGenerationResponse | None


class AdminConversationDetailResponse(BaseModel):
    conversation: ConversationResponse
    user: AdminUserResponse
    messages: list[AdminTranscriptMessageResponse]
    handoffs: list[HandoffResponse]
    messages_truncated: bool


class AdminHandoffsResponse(BaseModel):
    handoffs: list[HandoffResponse]
    total: int
    page: int
    page_size: int
