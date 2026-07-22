from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration for the bounded support-RAG service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = "Customer Support RAG"
    assistant_name: str = Field(
        default="پاسخ‌یار", validation_alias="ASSISTANT_NAME", min_length=1, max_length=100
    )
    company_name: str = Field(default="باسلام", validation_alias="COMPANY_NAME", min_length=1, max_length=100)
    environment: Literal["development", "test", "production"] = Field(
        default="development", validation_alias=AliasChoices("APP_ENV", "ENVIRONMENT")
    )
    data_dir: Path = Field(default=Path("app/data/runtime"), validation_alias="DATA_DIR")
    metadata_db_path: Path = Field(
        default=Path("app/data/runtime/rag.sqlite3"), validation_alias="SQLITE_PATH"
    )
    chroma_persist_dir: Path = Field(
        default=Path("app/data/runtime/chroma"), validation_alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        default="customer_support_multilingual_minilm_normalized_paragraph_v2",
        validation_alias="CHROMA_COLLECTION_NAME",
        min_length=1,
    )

    chat_provider: str = Field(
        default="gapgpt", validation_alias="CHAT_PROVIDER", min_length=1, max_length=50
    )
    chat_api_key: SecretStr | None = Field(default=None, validation_alias="CHAT_API_KEY")
    chat_model: str = Field(default="gpt-4o", validation_alias="CHAT_MODEL", min_length=1, max_length=200)
    chat_base_url: str | None = Field(default="https://api.gapgpt.app/v1", validation_alias="CHAT_BASE_URL")

    embedding_provider: Literal["sentence-transformers"] = Field(
        default="sentence-transformers", validation_alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        validation_alias="EMBEDDING_MODEL",
        min_length=1,
        max_length=300,
    )
    embedding_batch_size: int = Field(default=64, validation_alias="EMBEDDING_BATCH_SIZE", ge=1, le=256)

    admin_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ADMIN_API_KEY", "ADMIN_TOKEN")
    )
    server_conversations_enabled: bool = Field(
        default=True, validation_alias="SERVER_CONVERSATIONS_ENABLED"
    )
    widget_token_secret: SecretStr | None = Field(default=None, validation_alias="WIDGET_TOKEN_SECRET")
    widget_token_audience: str = Field(
        default="rag-widget", validation_alias="WIDGET_TOKEN_AUDIENCE", min_length=1, max_length=100
    )
    widget_token_clock_skew_seconds: int = Field(
        default=30, validation_alias="WIDGET_TOKEN_CLOCK_SKEW_SECONDS", ge=0, le=300
    )
    cors_allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=list, validation_alias=AliasChoices("CORS_ALLOWED_ORIGINS", "CORS_ORIGINS")
    )
    cors_allow_credentials: bool = False

    top_k: int = Field(default=4, validation_alias="TOP_K", ge=1, le=20)
    retrieval_max_distance: float = Field(default=0.65, validation_alias="RETRIEVAL_MAX_DISTANCE", ge=0, le=2)
    chunk_size: int = Field(default=900, validation_alias="CHUNK_SIZE", ge=100, le=4000)
    chunk_overlap: int = Field(default=150, validation_alias="CHUNK_OVERLAP", ge=0, le=1000)
    max_upload_bytes: int = Field(
        default=10 * 1024 * 1024,
        validation_alias=AliasChoices("MAX_UPLOAD_BYTES", "UPLOAD_MAX_BYTES"),
        ge=1024,
        le=25 * 1024 * 1024,
    )
    max_extracted_chars: int = Field(
        default=1_000_000, validation_alias="MAX_EXTRACTED_CHARS", ge=1_000, le=5_000_000
    )
    max_pdf_pages: int = Field(default=200, validation_alias="MAX_PDF_PAGES", ge=1, le=1_000)
    max_history_messages: int = Field(default=12, validation_alias="MAX_HISTORY_MESSAGES", ge=0, le=100)
    generation_stale_after_seconds: int = Field(
        default=600, validation_alias="GENERATION_STALE_AFTER_SECONDS", ge=60, le=86400
    )
    max_message_chars: int = Field(
        default=4_000,
        validation_alias=AliasChoices("MAX_MESSAGE_CHARS", "MAX_QUESTION_CHARS"),
        ge=100,
        le=20_000,
    )

    crawl_allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    crawl_max_pages: int = Field(
        default=5, validation_alias=AliasChoices("MAX_CRAWL_PAGES", "CRAWL_MAX_PAGES"), ge=1, le=100
    )
    crawl_max_depth: int = Field(
        default=1, validation_alias=AliasChoices("MAX_CRAWL_DEPTH", "CRAWL_MAX_DEPTH"), ge=0, le=5
    )
    crawl_max_response_bytes: int = Field(
        default=2 * 1024 * 1024,
        validation_alias=AliasChoices("MAX_CRAWL_RESPONSE_BYTES", "CRAWL_MAX_RESPONSE_BYTES"),
        ge=1024,
        le=10 * 1024 * 1024,
    )
    crawl_timeout_seconds: float = Field(default=15.0, validation_alias="CRAWL_TIMEOUT_SECONDS", gt=0, le=60)

    @field_validator("chat_provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("provider must not be empty")
        return value

    @field_validator("chat_base_url")
    @classmethod
    def validate_provider_url(cls, value: str | None) -> str | None:
        if not value:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("provider base URL must be an absolute http(s) URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_runtime_settings(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.environment == "production" and not self.admin_api_key:
            raise ValueError("ADMIN_API_KEY is required in production")
        if self.environment == "production" and not self.chat_api_key:
            raise ValueError("CHAT_API_KEY is required in production")
        if (
            self.environment == "production"
            and self.server_conversations_enabled
            and not self.widget_token_secret
        ):
            raise ValueError("WIDGET_TOKEN_SECRET is required when server conversations are enabled")
        if self.chat_provider not in {"openai", "ollama"} and not self.chat_base_url:
            raise ValueError("CHAT_BASE_URL is required for custom providers")
        return self

    @field_validator("cors_allowed_origins", "crawl_allowed_origins", mode="before")
    @classmethod
    def parse_origin_list(cls, value: str | list[str] | None) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        if not isinstance(value, list):
            raise ValueError("must be a comma-separated string or a list")
        origins: list[str] = []
        for item in value:
            parsed = urlsplit(str(item))
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("origins must be absolute http(s) URLs")
            if parsed.username or parsed.password or parsed.path not in ("", "/"):
                raise ValueError("origins must not contain credentials or a path")
            origins.append(f"{parsed.scheme.lower()}://{parsed.netloc.lower()}")
        return list(dict.fromkeys(origins))

    def provider_base_url(self, kind: Literal["chat"]) -> str | None:
        if self.chat_base_url:
            return self.chat_base_url
        if self.chat_provider == "ollama":
            return "http://127.0.0.1:11434/v1"
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
