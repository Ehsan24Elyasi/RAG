from __future__ import annotations

import hmac
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.llm.provider import ChatProvider, EmbeddingProvider, create_chat_provider, create_embedding_provider
from app.rag.vectorstore import VectorStore
from app.schemas import (
    AdminStatusResponse,
    ChatRequest,
    ChatResponse,
    CrawlRequest,
    CrawlResponse,
    DocumentResponse,
    DocumentsResponse,
    HealthResponse,
    IngestionRunResponse,
    IngestResponse,
)
from app.services.chat import ChatService
from app.services.crawler import CrawlError, WebCrawler
from app.services.documents import DocumentService, IngestedDocument
from app.services.metadata import IngestionRun, MetadataStore

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


def _http_error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


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
            chat=ChatService(metadata=metadata, embeddings=embeddings, vectors=vectors, chat=chat_client),
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
    if configured.cors_allowed_origins:
        api.add_middleware(
            CORSMiddleware,
            allow_origins=configured.cors_allowed_origins,
            allow_credentials=configured.cors_allow_credentials,
            allow_methods=["GET", "POST"],
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

    @api.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

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
            raise _http_error(503, "The language service is temporarily unavailable.") from None
        return ChatResponse(answer=answer, sources=sources)

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
