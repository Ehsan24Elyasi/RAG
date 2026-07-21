from fastapi.testclient import TestClient

from app.main import create_app
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
