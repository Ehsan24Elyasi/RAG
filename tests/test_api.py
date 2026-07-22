from fastapi.testclient import TestClient

from app.main import create_app
from app.security import create_widget_token
from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


def create_client(test_settings):
    app = create_app(
        test_settings,
        chat_provider=FakeChatProvider(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )
    return TestClient(app)


def test_static_pages_and_health(test_settings):
    with create_client(test_settings) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/").status_code == 200
        assert "پاسخ‌یار" in client.get("/").text
        assert client.get("/admin").status_code == 200

        widget = client.get("/static/widget.js")
        api_client = client.get("/static/api.js")
        styles = client.get("/static/styles.css")
        font = client.get("/static/fonts/Vazirmatn%5Bwght%5D.woff2")
        license_file = client.get("/static/fonts/OFL.txt")

        assert widget.status_code == 200
        assert api_client.status_code == 200
        assert styles.status_code == 200
        assert font.status_code == 200
        assert license_file.status_code == 200
        assert "RagSupportWidget" in widget.text
        assert "/api/conversations" in api_client.text
        assert "WIDGET_TOKEN_SECRET" not in widget.text
        assert "WIDGET_TOKEN_SECRET" not in api_client.text
        assert "Vazirmatn" in styles.text


def test_public_config_exposes_only_branding(test_settings):
    with create_client(test_settings) as client:
        response = client.get("/api/config")
        assert response.status_code == 200
        assert response.json() == {
            "assistant_name": test_settings.assistant_name,
            "company_name": test_settings.company_name,
        }


def test_admin_requires_bearer_key(test_settings):
    with create_client(test_settings) as client:
        assert client.get("/api/admin/status").status_code == 401
        response = client.get(
            "/api/admin/status",
            headers={"Authorization": "Bearer test-admin-key"},
        )
        assert response.status_code == 200
        assert response.json()["active_documents"] == 0


def test_upload_then_chat_returns_safe_source(test_settings):
    headers = {"Authorization": "Bearer test-admin-key"}
    with create_client(test_settings) as client:
        upload = client.post(
            "/api/admin/upload",
            headers=headers,
            files={"file": ("returns.txt", "مرجوعی کالا تا هفت روز امکان‌پذیر است.".encode(), "text/plain")},
        )
        assert upload.status_code == 200
        assert upload.json()["source_type"] == "upload"

        chat = client.post("/api/chat", json={"message": "مهلت مرجوعی چقدر است؟", "history": []})
        assert chat.status_code == 200
        payload = chat.json()
        assert payload["answer"] == "پاسخ آزمایشی [S1]"
        assert payload["sources"][0]["title"] == "returns.txt"
        assert "source_path" not in payload["sources"][0]


def test_upload_is_idempotent(test_settings):
    headers = {"Authorization": "Bearer test-admin-key"}
    files = {"file": ("faq.json", b'{"answer": "yes"}', "application/json")}
    with create_client(test_settings) as client:
        first = client.post("/api/admin/upload", headers=headers, files=files)
        second = client.post(
            "/api/admin/upload",
            headers=headers,
            files={"file": ("faq.json", b'{"answer": "yes"}', "application/json")},
        )
        assert first.status_code == 200
        assert second.json()["unchanged"] is True


def _widget_headers(test_settings, workspace_id: str, external_user_id: str) -> dict[str, str]:
    token = create_widget_token(
        test_settings.widget_token_secret.get_secret_value(),
        workspace_id=workspace_id,
        external_user_id=external_user_id,
        audience=test_settings.widget_token_audience,
    )
    return {"Authorization": f"Bearer {token}"}


def test_server_conversation_persists_messages_and_is_idempotent(test_settings):
    with create_client(test_settings) as client:
        workspace_id = client.app.state.runtime.metadata.default_workspace().id
        headers = _widget_headers(test_settings, workspace_id, "customer-1")
        created = client.post("/api/conversations", headers=headers, json={"title": "پشتیبانی"})
        assert created.status_code == 201
        conversation_id = created.json()["id"]

        first = client.post(
            f"/api/conversations/{conversation_id}/messages",
            headers=headers,
            json={"message": "سلام", "client_message_id": "client-1"},
        )
        duplicate = client.post(
            f"/api/conversations/{conversation_id}/messages",
            headers=headers,
            json={"message": "متن تکراری", "client_message_id": "client-1"},
        )

        assert first.status_code == 200
        assert duplicate.status_code == 200
        assert duplicate.json()["user_message_id"] == first.json()["user_message_id"]
        assert duplicate.json()["assistant_message_id"] == first.json()["assistant_message_id"]
        assert duplicate.json()["answer"] == first.json()["answer"]
        listed = client.get("/api/conversations", headers=headers)
        assert listed.json()["conversations"][0]["id"] == conversation_id


def test_server_conversation_rejects_other_user(test_settings):
    with create_client(test_settings) as client:
        workspace_id = client.app.state.runtime.metadata.default_workspace().id
        owner_headers = _widget_headers(test_settings, workspace_id, "owner")
        other_headers = _widget_headers(test_settings, workspace_id, "other")
        conversation_id = client.post(
            "/api/conversations", headers=owner_headers, json={}
        ).json()["id"]

        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            headers=other_headers,
            json={"message": "سلام", "client_message_id": "foreign-1"},
        )

        assert response.status_code == 404


def test_server_conversation_rejects_invalid_widget_token(test_settings):
    with create_client(test_settings) as client:
        response = client.post(
            "/api/conversations",
            headers={"Authorization": "Bearer invalid-token"},
            json={},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired widget token."


def test_server_conversation_can_be_disabled(test_settings):
    disabled_settings = test_settings.model_copy(
        update={"server_conversations_enabled": False}
    )
    with create_client(disabled_settings) as client:
        response = client.post("/api/conversations", json={})

        assert response.status_code == 404
