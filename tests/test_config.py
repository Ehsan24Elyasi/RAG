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


def test_production_server_conversations_require_widget_secret():
    with pytest.raises(ValueError, match="WIDGET_TOKEN_SECRET"):
        Settings(
            _env_file=None,
            APP_ENV="production",
            ADMIN_API_KEY="admin-secret",
            CHAT_PROVIDER="gapgpt",
            CHAT_API_KEY="chat-secret",
            CHAT_BASE_URL="https://api.gapgpt.app/v1",
            WIDGET_TOKEN_SECRET=None,
        )


def test_local_demo_widget_defaults_to_available_with_development_secret():
    settings = Settings(
        _env_file=None,
        APP_ENV="development",
        WIDGET_TOKEN_SECRET="local-widget-secret",
    )

    assert settings.local_demo_widget_enabled is None
    assert settings.local_demo_widget_available is True
    assert settings.local_demo_widget_external_user_id == "local-demo-user"
    assert settings.local_demo_widget_token_ttl_seconds == 300


def test_local_demo_widget_rejects_unsafe_configuration():
    with pytest.raises(ValueError, match="must not be enabled in production"):
        Settings(
            _env_file=None,
            APP_ENV="production",
            ADMIN_API_KEY="admin-secret",
            CHAT_API_KEY="chat-secret",
            WIDGET_TOKEN_SECRET="widget-secret",
            LOCAL_DEMO_WIDGET_ENABLED=True,
        )
    with pytest.raises(ValueError, match="requires WIDGET_TOKEN_SECRET"):
        Settings(_env_file=None, LOCAL_DEMO_WIDGET_ENABLED=True)
    with pytest.raises(ValueError, match="requires SERVER_CONVERSATIONS_ENABLED"):
        Settings(
            _env_file=None,
            LOCAL_DEMO_WIDGET_ENABLED=True,
            SERVER_CONVERSATIONS_ENABLED=False,
            WIDGET_TOKEN_SECRET="widget-secret",
        )


def test_local_demo_widget_identity_and_ttl_are_validated():
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            LOCAL_DEMO_WIDGET_EXTERNAL_USER_ID="demo\nuser",
        )
    with pytest.raises(ValueError):
        Settings(_env_file=None, LOCAL_DEMO_WIDGET_TOKEN_TTL_SECONDS=30)


def test_custom_chat_provider_requires_base_url():
    with pytest.raises(ValueError, match="CHAT_BASE_URL"):
        Settings(
            _env_file=None,
            CHAT_PROVIDER="custom",
            CHAT_API_KEY="secret",
            CHAT_BASE_URL=None,
        )


def test_support_contacts_are_optional_and_validated():
    settings = Settings(
        _env_file=None,
        SUPPORT_EMAIL=" help@example.test ",
        SUPPORT_PHONE="+98 21 1234-5678",
        SUPPORT_URL="https://example.test/support",
    )

    assert settings.support_email == "help@example.test"
    assert settings.support_phone == "+98 21 1234-5678"
    assert settings.support_url == "https://example.test/support"
    assert Settings(_env_file=None, SUPPORT_EMAIL="").support_email is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("SUPPORT_EMAIL", "bad-address"),
        ("SUPPORT_PHONE", "123<script>"),
        ("SUPPORT_URL", "javascript:alert(1)"),
        ("SUPPORT_URL", "https://user:pass@example.test/support"),
        ("ASSISTANT_NAME", "یار\nدستور تازه"),
    ],
)
def test_public_brand_and_contact_values_reject_unsafe_input(field, value):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: value})


def test_local_multilingual_embedding_defaults():
    settings = Settings(_env_file=None)

    assert settings.embedding_provider == "sentence-transformers"
    assert settings.embedding_model == ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    assert settings.chroma_collection_name == "customer_support_multilingual_minilm_normalized_paragraph_v2"
    assert settings.retrieval_max_distance == 0.65
