from __future__ import annotations

import hmac
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.llm.provider import ChatProvider, EmbeddingProvider, create_chat_provider, create_embedding_provider
from app.observability import request_context_middleware
from app.rag.vectorstore import VectorStore
from app.schemas import (
    AdminCitationResponse,
    AdminConversationDetailResponse,
    AdminConversationsResponse,
    AdminConversationSummaryResponse,
    AdminGenerationResponse,
    AdminHandoffsResponse,
    AdminMetricsResponse,
    AdminStatusResponse,
    AdminTranscriptMessageResponse,
    AdminUserResponse,
    ChatRequest,
    ChatResponse,
    ConversationCreateRequest,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ConversationResponse,
    ConversationsResponse,
    CrawlRequest,
    CrawlResponse,
    DocumentResponse,
    DocumentsResponse,
    HandoffCreateRequest,
    HandoffCreateResponse,
    HandoffResponse,
    HandoffUpdateRequest,
    HealthResponse,
    IngestionRunResponse,
    IngestResponse,
    LocalWidgetBootstrapResponse,
    PublicConfigResponse,
)
from app.security import WidgetIdentity, WidgetTokenError, create_widget_token, verify_widget_token
from app.services.chat import ChatGenerationError, ChatService
from app.services.crawler import CrawlError, WebCrawler
from app.services.documents import DocumentService, IngestedDocument
from app.services.metadata import IngestionRun, MetadataStore, User

STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class Runtime:
    settings: Settings
    metadata: MetadataStore
    documents: DocumentService
    chat: ChatService
    crawler: WebCrawler
    vectors: VectorStore
    ingestion_lock: threading.Lock


@dataclass(frozen=True)
class WidgetPrincipal:
    identity: WidgetIdentity
    user: User


def _http_error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def _effective_port(scheme: str, port: int | None) -> int | None:
    if port is not None:
        return port
    return {"http": 80, "https": 443}.get(scheme)


def _is_same_origin_loopback(request: Request) -> bool:
    origin = request.headers.get("Origin", "")
    if not origin or origin == "null":
        return False
    try:
        parsed = urlsplit(origin)
        origin_host = (parsed.hostname or "").lower()
        origin_port = _effective_port(parsed.scheme.lower(), parsed.port)
        request_host = (request.url.hostname or "").lower()
        request_port = _effective_port(request.url.scheme.lower(), request.url.port)
    except ValueError:
        return False
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    return bool(
        parsed.scheme in {"http", "https"}
        and not parsed.username
        and not parsed.password
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
        and origin_host in loopback_hosts
        and request_host in loopback_hosts
        and parsed.scheme.lower() == request.url.scheme.lower()
        and origin_host == request_host
        and origin_port == request_port
        and request.headers.get("Sec-Fetch-Site", "same-origin") in {"same-origin", "none"}
    )


def _run_response(run: IngestionRun) -> IngestionRunResponse:
    return IngestionRunResponse(**run.__dict__)


def _ingest_response(document: IngestedDocument, run_id: str) -> IngestResponse:
    return IngestResponse(
        document_id=document.document_id,
        source_name=document.source_name,
        source_type=document.source_type,
        chunks_created=document.chunks_created,
        unchanged=document.unchanged,
        run_id=run_id,
    )


def _conversation_response(conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        status=conversation.status,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_message_at=conversation.last_message_at,
    )


def _user_response(user) -> AdminUserResponse:
    return AdminUserResponse(
        id=user.id,
        external_id=user.external_id,
        display_name=user.display_name,
        email=user.email,
    )


def _handoff_response(handoff) -> HandoffResponse:
    return HandoffResponse(
        id=handoff.id,
        conversation_id=handoff.conversation_id,
        status=handoff.status,
        reason=handoff.reason,
        created_at=handoff.created_at,
        updated_at=handoff.updated_at,
        resolved_at=handoff.resolved_at,
    )


def create_app(
    settings: Settings | None = None,
    *,
    chat_provider: ChatProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    crawler: WebCrawler | None = None,
) -> FastAPI:
    configured = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configured.data_dir.mkdir(parents=True, exist_ok=True)
        metadata = MetadataStore(configured.metadata_db_path)
        metadata.initialize()
        metadata.recover_stale_generating_messages(configured.generation_stale_after_seconds)
        vectors = vector_store or VectorStore(
            str(configured.chroma_persist_dir),
            configured.chroma_collection_name,
        )
        embeddings = embedding_provider or create_embedding_provider(configured)
        chat_client = chat_provider or create_chat_provider(configured)
        documents = DocumentService(
            settings=configured, metadata=metadata, embeddings=embeddings, vectors=vectors
        )
        app.state.runtime = Runtime(
            settings=configured,
            metadata=metadata,
            documents=documents,
            chat=ChatService(
                metadata=metadata,
                embeddings=embeddings,
                vectors=vectors,
                chat=chat_client,
                settings=configured,
            ),
            crawler=crawler
            or WebCrawler(
                timeout_seconds=configured.crawl_timeout_seconds,
                max_response_bytes=configured.crawl_max_response_bytes,
                allowed_origins=configured.crawl_allowed_origins,
            ),
            vectors=vectors,
            ingestion_lock=threading.Lock(),
        )
        yield

    api = FastAPI(title=configured.app_name, lifespan=lifespan)
    api.middleware("http")(request_context_middleware)
    if configured.cors_allowed_origins:
        api.add_middleware(
            CORSMiddleware,
            allow_origins=configured.cors_allowed_origins,
            allow_credentials=configured.cors_allow_credentials,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @api.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Invalid request."})

    @api.exception_handler(CrawlError)
    async def crawl_error_handler(_: Request, exc: CrawlError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def runtime(request: Request) -> Runtime:
        return request.app.state.runtime

    def require_admin(request: Request, active: Runtime = Depends(runtime)) -> None:
        secret = active.settings.admin_api_key
        if not secret:
            raise _http_error(503, "Administrative API is not configured.")
        expected = f"Bearer {secret.get_secret_value()}"
        if not hmac.compare_digest(request.headers.get("Authorization", ""), expected):
            raise _http_error(401, "Authentication required.")

    def require_widget_user(request: Request, active: Runtime = Depends(runtime)) -> WidgetPrincipal:
        if not active.settings.server_conversations_enabled:
            raise _http_error(404, "Server conversations are not enabled.")
        secret = active.settings.widget_token_secret
        if not secret:
            raise _http_error(503, "Widget authentication is not configured.")
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer "):
            raise _http_error(401, "Widget authentication required.")
        try:
            identity = verify_widget_token(
                authorization.removeprefix("Bearer ").strip(),
                secret.get_secret_value(),
                audience=active.settings.widget_token_audience,
                clock_skew_seconds=active.settings.widget_token_clock_skew_seconds,
            )
        except WidgetTokenError:
            raise _http_error(401, "Invalid or expired widget token.") from None
        if active.metadata.workspace(identity.workspace_id) is None:
            raise _http_error(401, "Invalid widget workspace.")
        user = active.metadata.upsert_user(
            identity.workspace_id,
            identity.external_user_id,
            display_name=identity.display_name,
            email=identity.email,
        )
        return WidgetPrincipal(identity=identity, user=user)

    @api.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.get("/api/config", response_model=PublicConfigResponse)
    def public_config() -> PublicConfigResponse:
        return PublicConfigResponse(
            assistant_name=configured.assistant_name,
            company_name=configured.company_name,
            support_email=configured.support_email,
            support_phone=configured.support_phone,
            support_url=configured.support_url,
        )

    @api.post("/api/dev/widget-bootstrap", response_model=LocalWidgetBootstrapResponse)
    def local_widget_bootstrap(
        request: Request,
        response: Response,
        active: Runtime = Depends(runtime),
    ) -> LocalWidgetBootstrapResponse:
        if not active.settings.local_demo_widget_available:
            raise _http_error(404, "Local widget bootstrap is not available.")
        if not _is_same_origin_loopback(request):
            raise _http_error(403, "Local widget bootstrap requires a same-origin loopback request.")
        secret = active.settings.widget_token_secret
        if not secret:
            raise _http_error(404, "Local widget bootstrap is not available.")
        token = create_widget_token(
            secret.get_secret_value(),
            workspace_id=active.metadata.default_workspace().id,
            external_user_id=active.settings.local_demo_widget_external_user_id,
            display_name=active.settings.local_demo_widget_display_name,
            audience=active.settings.widget_token_audience,
            expires_in_seconds=active.settings.local_demo_widget_token_ttl_seconds,
        )
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Vary"] = "Origin"
        return LocalWidgetBootstrapResponse(
            token=token,
            expires_in_seconds=active.settings.local_demo_widget_token_ttl_seconds,
        )

    @api.post("/api/chat", response_model=ChatResponse)
    def chat(request_body: ChatRequest, active: Runtime = Depends(runtime)) -> ChatResponse:
        if len(request_body.message) > active.settings.max_message_chars:
            raise _http_error(422, "The message exceeds the configured limit.")
        if len(request_body.history) > active.settings.max_history_messages:
            raise _http_error(422, "The conversation history exceeds the configured limit.")
        if any(len(item.content) > active.settings.max_message_chars for item in request_body.history):
            raise _http_error(422, "A history message exceeds the configured limit.")
        try:
            answer, sources = active.chat.answer(
                request_body.message,
                [item.model_dump() for item in request_body.history],
                active.settings.top_k,
            )
        except Exception:
            raise _http_error(503, "سرویس پاسخ‌گویی موقتاً در دسترس نیست.") from None
        return ChatResponse(answer=answer, sources=sources)

    @api.post("/api/conversations", response_model=ConversationResponse, status_code=201)
    def create_conversation(
        request_body: ConversationCreateRequest,
        principal: WidgetPrincipal = Depends(require_widget_user),
        active: Runtime = Depends(runtime),
    ) -> ConversationResponse:
        conversation = active.metadata.create_conversation(
            principal.identity.workspace_id,
            principal.user.id,
            title=request_body.title,
        )
        return _conversation_response(conversation)

    @api.get("/api/conversations", response_model=ConversationsResponse)
    def list_conversations(
        principal: WidgetPrincipal = Depends(require_widget_user),
        active: Runtime = Depends(runtime),
    ) -> ConversationsResponse:
        conversations = active.metadata.list_authorized_conversations(
            principal.identity.workspace_id,
            principal.user.id,
        )
        return ConversationsResponse(conversations=[_conversation_response(item) for item in conversations])

    @api.post(
        "/api/conversations/{conversation_id}/messages",
        response_model=ConversationMessageResponse,
    )
    def create_conversation_message(
        conversation_id: str,
        request_body: ConversationMessageRequest,
        request: Request,
        principal: WidgetPrincipal = Depends(require_widget_user),
        active: Runtime = Depends(runtime),
    ) -> ConversationMessageResponse:
        if len(request_body.message) > active.settings.max_message_chars:
            raise _http_error(422, "The message exceeds the configured limit.")
        workspace_id = principal.identity.workspace_id
        conversation = active.metadata.get_conversation(
            workspace_id,
            conversation_id,
            user_id=principal.user.id,
        )
        if conversation is None:
            raise _http_error(404, "Conversation not found.")
        user_message = active.metadata.append_user_message(
            workspace_id,
            conversation_id,
            principal.user.id,
            request_body.message,
            client_message_id=request_body.client_message_id,
        )
        assistant_message, claimed = active.metadata.claim_assistant_placeholder(
            workspace_id,
            conversation_id,
            user_id=principal.user.id,
            parent_message_id=user_message.id,
        )
        if assistant_message.status == "completed" and assistant_message.content is not None:
            citations = active.metadata.citations_for_message(workspace_id, assistant_message.id)
            return ConversationMessageResponse(
                conversation_id=conversation_id,
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
                answer=assistant_message.content,
                sources=[
                    {
                        "id": (item.metadata or {}).get("source_id", f"S{item.position + 1}"),
                        "document_id": item.document_id or "",
                        "title": item.source_name or "Support document",
                        "source_type": (item.metadata or {}).get("source_type", "upload"),
                        "url": item.source_url,
                        "chunk_index": (item.metadata or {}).get("chunk_index", 0),
                    }
                    for item in citations
                    if item.document_id
                ],
            )
        if not claimed:
            raise _http_error(409, "This message is already being processed or requires a new retry ID.")
        persisted_history = active.metadata.load_conversation_history(
            workspace_id,
            conversation_id,
            user_id=principal.user.id,
            limit=active.settings.max_history_messages + 2,
        )
        history = [
            {"role": item.role, "content": item.content}
            for item in persisted_history
            if item.id not in {user_message.id, assistant_message.id}
            and item.role in {"user", "assistant"}
            and item.status == "completed"
            and item.content
        ][-active.settings.max_history_messages :]
        try:
            result = active.chat.answer_detailed(
                request_body.message,
                history,
                active.settings.top_k,
            )
            active.metadata.complete_assistant_message(
                workspace_id,
                assistant_message.id,
                result.answer,
            )
            active.metadata.record_citations(
                workspace_id,
                assistant_message.id,
                [
                    {
                        "document_id": source.get("document_id"),
                        "source_name": source.get("title"),
                        "source_url": source.get("url"),
                        "metadata": {
                            "source_id": source.get("id"),
                            "source_type": source.get("source_type"),
                            "chunk_index": source.get("chunk_index"),
                        },
                    }
                    for source in result.sources
                ],
            )
            generation = result.generation
            if generation is not None:
                active.metadata.record_generation_usage(
                    workspace_id,
                    conversation_id,
                    assistant_message.id,
                    status="completed",
                    provider=active.settings.chat_provider,
                    model=generation.model,
                    prompt_tokens=generation.usage.prompt_tokens,
                    completion_tokens=generation.usage.completion_tokens,
                    total_tokens=generation.usage.total_tokens,
                    latency_ms=generation.latency_ms,
                    finish_reason=generation.finish_reason,
                    request_id=request.state.request_id,
                    metadata={
                        "context_count": len(result.sources),
                        "provider_request_id": generation.provider_request_id,
                    },
                )
        except ChatGenerationError as error:
            active.metadata.fail_assistant_message(
                workspace_id,
                assistant_message.id,
                "سرویس پاسخ‌گویی موقتاً در دسترس نیست.",
            )
            active.metadata.record_generation_usage(
                workspace_id,
                conversation_id,
                assistant_message.id,
                status="failed",
                provider=active.settings.chat_provider,
                model=active.settings.chat_model,
                latency_ms=error.latency_ms,
                request_id=request.state.request_id,
                metadata={"failure_phase": "generation"},
                error_message="Chat provider generation failed.",
            )
            raise _http_error(503, "سرویس پاسخ‌گویی موقتاً در دسترس نیست.") from None
        except Exception:
            active.metadata.fail_assistant_message(
                workspace_id,
                assistant_message.id,
                "سرویس پاسخ‌گویی موقتاً در دسترس نیست.",
            )
            raise _http_error(503, "سرویس پاسخ‌گویی موقتاً در دسترس نیست.") from None
        return ConversationMessageResponse(
            conversation_id=conversation_id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            answer=result.answer,
            sources=result.sources,
        )

    @api.post(
        "/api/conversations/{conversation_id}/handoff",
        response_model=HandoffCreateResponse,
    )
    def create_handoff(
        conversation_id: str,
        request_body: HandoffCreateRequest,
        principal: WidgetPrincipal = Depends(require_widget_user),
        active: Runtime = Depends(runtime),
    ) -> HandoffCreateResponse:
        try:
            handoff, created = active.metadata.create_or_get_active_handoff(
                principal.identity.workspace_id,
                conversation_id,
                principal.user.id,
                request_body.reason,
            )
        except ValueError:
            raise _http_error(404, "Conversation not found.") from None
        return HandoffCreateResponse(created=created, handoff=_handoff_response(handoff))

    @api.get("/api/admin/status", response_model=AdminStatusResponse, dependencies=[Depends(require_admin)])
    def admin_status(active: Runtime = Depends(runtime)) -> AdminStatusResponse:
        documents, chunks = active.metadata.counts()
        return AdminStatusResponse(
            status="ok",
            active_documents=documents,
            indexed_chunks=chunks,
            chat_provider=active.settings.chat_provider,
            embedding_provider=active.settings.embedding_provider,
            recent_runs=[_run_response(run) for run in active.metadata.recent_runs()],
        )

    @api.get("/api/admin/metrics", response_model=AdminMetricsResponse, dependencies=[Depends(require_admin)])
    def admin_metrics(days: int = 30, active: Runtime = Depends(runtime)) -> AdminMetricsResponse:
        if days not in {7, 30, 90}:
            raise _http_error(422, "The metrics period must be 7, 30, or 90 days.")
        end = datetime.now(timezone.utc)
        metrics = active.metadata.usage_metrics(
            active.metadata.default_workspace().id,
            end - timedelta(days=days),
            end,
        )
        return AdminMetricsResponse(**metrics.__dict__)

    @api.get(
        "/api/admin/conversations",
        response_model=AdminConversationsResponse,
        dependencies=[Depends(require_admin)],
    )
    def admin_conversations(
        search: str | None = None,
        status: str | None = None,
        handoff: str | None = None,
        page: int = 1,
        page_size: int = 20,
        active: Runtime = Depends(runtime),
    ) -> AdminConversationsResponse:
        if page < 1 or page_size < 1 or page_size > 100:
            raise _http_error(422, "Invalid pagination.")
        items, total = active.metadata.list_admin_conversations(
            active.metadata.default_workspace().id,
            search=search.strip()[:200] if search else None,
            status=status,
            handoff=handoff,
            page=page,
            page_size=page_size,
        )
        return AdminConversationsResponse(
            conversations=[
                AdminConversationSummaryResponse(
                    id=item.conversation.id,
                    title=item.conversation.title,
                    status=item.conversation.status,
                    created_at=item.conversation.created_at,
                    updated_at=item.conversation.updated_at,
                    last_message_at=item.conversation.last_message_at,
                    user=_user_response(item.user),
                    last_message_preview=item.last_message_preview,
                    message_count=item.message_count,
                    token_total=item.token_total,
                    active_handoff=(_handoff_response(item.active_handoff) if item.active_handoff else None),
                )
                for item in items
            ],
            total=total,
            page=page,
            page_size=page_size,
        )

    @api.get(
        "/api/admin/conversations/{conversation_id}",
        response_model=AdminConversationDetailResponse,
        dependencies=[Depends(require_admin)],
    )
    def admin_conversation_detail(
        conversation_id: str, active: Runtime = Depends(runtime)
    ) -> AdminConversationDetailResponse:
        detail = active.metadata.admin_conversation_detail(
            active.metadata.default_workspace().id, conversation_id
        )
        if detail is None:
            raise _http_error(404, "Conversation not found.")
        messages = []
        for message in detail.messages:
            generation = detail.generations.get(message.id)
            messages.append(
                AdminTranscriptMessageResponse(
                    id=message.id,
                    role=message.role,
                    status=message.status,
                    content=message.content,
                    error_message=message.error_message,
                    created_at=message.created_at,
                    completed_at=message.completed_at,
                    citations=[
                        AdminCitationResponse(
                            position=item.position,
                            document_id=item.document_id,
                            source_name=item.source_name,
                            source_url=item.source_url,
                            excerpt=item.excerpt,
                            metadata=item.metadata,
                        )
                        for item in detail.citations.get(message.id, [])
                    ],
                    generation=(
                        AdminGenerationResponse(
                            provider=generation.provider,
                            model=generation.model,
                            status=generation.status,
                            prompt_tokens=generation.prompt_tokens,
                            completion_tokens=generation.completion_tokens,
                            total_tokens=generation.total_tokens,
                            latency_ms=generation.latency_ms,
                            finish_reason=generation.finish_reason,
                        )
                        if generation
                        else None
                    ),
                )
            )
        return AdminConversationDetailResponse(
            conversation=_conversation_response(detail.conversation),
            user=_user_response(detail.user),
            messages=messages,
            handoffs=[_handoff_response(item) for item in detail.handoffs],
            messages_truncated=detail.messages_truncated,
        )

    @api.get(
        "/api/admin/handoffs",
        response_model=AdminHandoffsResponse,
        dependencies=[Depends(require_admin)],
    )
    def admin_handoffs(
        status: str | None = None,
        page: int = 1,
        page_size: int = 50,
        active: Runtime = Depends(runtime),
    ) -> AdminHandoffsResponse:
        if status and status not in {"requested", "in_progress", "resolved", "cancelled"}:
            raise _http_error(422, "Invalid handoff status.")
        if page < 1 or page_size < 1 or page_size > 100:
            raise _http_error(422, "Invalid pagination.")
        items, total = active.metadata.list_handoffs(
            active.metadata.default_workspace().id,
            status=status,
            page=page,
            page_size=page_size,
        )
        return AdminHandoffsResponse(
            handoffs=[_handoff_response(item) for item in items],
            total=total,
            page=page,
            page_size=page_size,
        )

    @api.patch(
        "/api/admin/handoffs/{handoff_id}",
        response_model=HandoffResponse,
        dependencies=[Depends(require_admin)],
    )
    def update_admin_handoff(
        handoff_id: str,
        request_body: HandoffUpdateRequest,
        active: Runtime = Depends(runtime),
    ) -> HandoffResponse:
        try:
            handoff = active.metadata.update_handoff_status(
                active.metadata.default_workspace().id, handoff_id, request_body.status
            )
        except ValueError:
            raise _http_error(409, "Invalid handoff status transition.") from None
        if handoff is None:
            raise _http_error(404, "Handoff request not found.")
        return _handoff_response(handoff)

    @api.get("/api/admin/documents", response_model=DocumentsResponse, dependencies=[Depends(require_admin)])
    def list_documents(active: Runtime = Depends(runtime)) -> DocumentsResponse:
        return DocumentsResponse(
            documents=[
                DocumentResponse(
                    id=item.id,
                    source_type=item.source_type,
                    source_name=item.source_name,
                    source_url=item.source_url,
                    content_hash=item.content_hash,
                    chunk_count=item.chunk_count,
                    updated_at=item.updated_at,
                )
                for item in active.metadata.list_documents()
            ]
        )

    @api.delete("/api/admin/documents/{document_id}", status_code=204, dependencies=[Depends(require_admin)])
    def delete_document(document_id: str, active: Runtime = Depends(runtime)) -> None:
        if not active.ingestion_lock.acquire(blocking=False):
            raise _http_error(409, "Another ingestion is already running.")
        try:
            deleted = active.documents.delete_upload(active.metadata.default_workspace().id, document_id)
            if not deleted:
                raise _http_error(404, "Document not found.")
        except HTTPException:
            raise
        except ValueError as exc:
            raise _http_error(409, str(exc)) from None
        except Exception:
            raise _http_error(503, "The document could not be deleted.") from None
        finally:
            active.ingestion_lock.release()

    @api.post("/api/admin/upload", response_model=IngestResponse, dependencies=[Depends(require_admin)])
    async def upload_document(
        file: UploadFile = File(...), active: Runtime = Depends(runtime)
    ) -> IngestResponse:
        if not file.filename or len(file.filename) > 255:
            raise _http_error(422, "A valid filename is required.")
        if not active.ingestion_lock.acquire(blocking=False):
            raise _http_error(409, "Another ingestion is already running.")
        run: IngestionRun | None = None
        try:
            run = active.metadata.create_run("upload", file.filename)
            chunks = []
            received = 0
            while piece := await file.read(64 * 1024):
                received += len(piece)
                if received > active.settings.max_upload_bytes:
                    raise _http_error(413, "The uploaded document exceeds the size limit.")
                chunks.append(piece)
            document = active.documents.ingest_upload(file.filename, b"".join(chunks))
            active.metadata.finish_run(
                run.id,
                status="succeeded",
                documents_processed=1,
                chunks_created=document.chunks_created,
            )
            return _ingest_response(document, run.id)
        except HTTPException:
            if run is not None:
                active.metadata.finish_run(run.id, status="failed", error_message="Upload rejected.")
            raise
        except ValueError as exc:
            if run is not None:
                active.metadata.finish_run(run.id, status="failed", error_message=str(exc))
            raise _http_error(422, str(exc)) from None
        except Exception:
            if run is not None:
                active.metadata.finish_run(run.id, status="failed", error_message="Indexing failed.")
            raise _http_error(503, "The document could not be indexed.") from None
        finally:
            active.ingestion_lock.release()

    @api.post("/api/admin/crawl", response_model=CrawlResponse, dependencies=[Depends(require_admin)])
    def crawl_site(request_body: CrawlRequest, active: Runtime = Depends(runtime)) -> CrawlResponse:
        if not active.ingestion_lock.acquire(blocking=False):
            raise _http_error(409, "Another ingestion is already running.")
        run: IngestionRun | None = None
        try:
            run = active.metadata.create_run("crawl", request_body.url)
            pages = active.crawler.crawl(
                request_body.url,
                max_pages=min(
                    request_body.max_pages or active.settings.crawl_max_pages, active.settings.crawl_max_pages
                ),
                max_depth=min(
                    request_body.max_depth
                    if request_body.max_depth is not None
                    else active.settings.crawl_max_depth,
                    active.settings.crawl_max_depth,
                ),
            )
            documents = [
                active.documents.ingest_crawl_page(page.url, page.title, page.text) for page in pages
            ]
            chunks = sum(document.chunks_created for document in documents)
            active.metadata.finish_run(
                run.id,
                status="succeeded",
                documents_processed=len(documents),
                chunks_created=chunks,
            )
            return CrawlResponse(
                pages_visited=len(pages),
                documents=[_ingest_response(document, run.id) for document in documents],
                run_id=run.id,
            )
        except CrawlError as exc:
            if run is not None:
                active.metadata.finish_run(run.id, status="failed", error_message=str(exc))
            raise
        except ValueError as exc:
            if run is not None:
                active.metadata.finish_run(run.id, status="failed", error_message=str(exc))
            raise _http_error(422, str(exc)) from None
        except Exception:
            if run is not None:
                active.metadata.finish_run(run.id, status="failed", error_message="Crawl indexing failed.")
            raise _http_error(503, "The crawled pages could not be indexed.") from None
        finally:
            active.ingestion_lock.release()

    if STATIC_DIR.exists():
        api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @api.get("/", include_in_schema=False)
        def frontend() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

        @api.get("/admin", include_in_schema=False)
        def admin_frontend() -> FileResponse:
            return FileResponse(STATIC_DIR / "admin.html")

    return api


app = create_app()
