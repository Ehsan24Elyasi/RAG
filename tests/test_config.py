import pytest

from app.config import Settings


def test_comma_separated_cors_origins_parse(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "gapgpt")
    monkeypatch.setenv("CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("CHAT_BASE_URL", "https://api.gapgpt.app/v1")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )

    settings = Settings(_env_file=None)

    assert settings.cors_allowed_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


def test_gapgpt_requires_api_key_in_production():
    with pytest.raises(ValueError, match="CHAT_API_KEY"):
        Settings(
            _env_file=None,
            APP_ENV="production",
            ADMIN_API_KEY="admin-secret",
            CHAT_PROVIDER="gapgpt",
            CHAT_API_KEY=None,
            CHAT_BASE_URL="https://api.gapgpt.app/v1",
        )


def test_custom_chat_provider_requires_base_url():
    with pytest.raises(ValueError, match="CHAT_BASE_URL"):
        Settings(
            _env_file=None,
            CHAT_PROVIDER="custom",
            CHAT_API_KEY="secret",
            CHAT_BASE_URL=None,
        )


def test_local_multilingual_embedding_defaults():
    settings = Settings(_env_file=None)

    assert settings.embedding_provider == "sentence-transformers"
    assert settings.embedding_model == ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    assert settings.chroma_collection_name == "customer_support_multilingual_minilm_v1"
