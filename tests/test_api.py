from fastapi.testclient import TestClient

from app.llm.types import GenerationResult, GenerationUsage
from app.main import create_app
from app.security import create_widget_token, verify_widget_token
from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


def create_client(test_settings, chat_provider=None, *, base_url="http://testserver"):
    app = create_app(
        test_settings,
        chat_provider=chat_provider or FakeChatProvider(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )
    return TestClient(app, base_url=base_url)


def test_static_pages_and_health(test_settings):
    with create_client(test_settings) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/").status_code == 200
        assert "پاسخ‌یار" in client.get("/").text
        admin = client.get("/admin")
        assert admin.status_code == 200

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
        assert "window.prompt" not in widget.text
        assert "گفت‌وگوی زنده نیست" in client.get("/").text
        assert "/api/conversations" in api_client.text
        assert "/api/dev/widget-bootstrap" in api_client.text
        assert "localWidgetBootstrapApi" in widget.text
        assert "WIDGET_TOKEN_SECRET" not in widget.text
        assert "WIDGET_TOKEN_SECRET" not in api_client.text
        assert "Vazirmatn" in styles.text
        assert 'class="admin-sidebar"' in admin.text
        for section_id in (
            "overview-section",
            "conversations-section",
            "handoffs-section",
            "documents-section",
            "ingestion-section",
        ):
            assert f'id="{section_id}"' in admin.text
        assert "پردازش‌های اخیر" not in admin.text
        assert "نتیجه بارگذاری و خزش‌های اخیر" not in admin.text
        assert "operation-list" not in admin.text


def test_public_config_exposes_only_branding(test_settings):
    with create_client(test_settings) as client:
        response = client.get("/api/config")
        assert response.status_code == 200
        assert response.json() == {
            "assistant_name": test_settings.assistant_name,
            "company_name": test_settings.company_name,
            "support_email": None,
            "support_phone": None,
            "support_url": None,
        }


def test_local_widget_bootstrap_is_loopback_same_origin_only(test_settings):
    with create_client(test_settings, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/api/dev/widget-bootstrap",
            headers={"Origin": "http://127.0.0.1:8000", "Sec-Fetch-Site": "same-origin"},
        )
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store, private"
        assert set(response.json()) == {"token", "expires_in_seconds"}
        identity = verify_widget_token(
            response.json()["token"],
            test_settings.widget_token_secret.get_secret_value(),
            audience=test_settings.widget_token_audience,
        )
        assert identity.workspace_id == client.app.state.runtime.metadata.default_workspace().id
        assert identity.external_user_id == test_settings.local_demo_widget_external_user_id

        assert client.post("/api/dev/widget-bootstrap").status_code == 403
        assert client.post(
            "/api/dev/widget-bootstrap",
            headers={"Origin": "http://localhost:8000"},
        ).status_code == 403

    with create_client(test_settings, base_url="http://example.test") as public_client:
        assert public_client.post(
            "/api/dev/widget-bootstrap",
            headers={"Origin": "http://example.test"},
        ).status_code == 403


def test_local_widget_bootstrap_can_be_disabled(test_settings):
    for disabled in (
        test_settings.model_copy(update={"local_demo_widget_enabled": False}),
        test_settings.model_copy(update={"environment": "production"}),
    ):
        with create_client(disabled, base_url="http://127.0.0.1:8000") as client:
            response = client.post(
                "/api/dev/widget-bootstrap",
                headers={"Origin": "http://127.0.0.1:8000"},
            )
            assert response.status_code == 404


def test_local_widget_flow_updates_admin_metrics_and_handoffs(test_settings):
    class UsageChatProvider:
        def generate(self, messages):
            return GenerationResult(
                text="پاسخ آزمایشی [S1]",
                model="usage-test",
                usage=GenerationUsage(prompt_tokens=12, completion_tokens=7, total_tokens=19),
                latency_ms=25,
            )

    admin_headers = {"Authorization": "Bearer test-admin-key"}
    with create_client(
        test_settings,
        UsageChatProvider(),
        base_url="http://127.0.0.1:8000",
    ) as client:
        bootstrap = client.post(
            "/api/dev/widget-bootstrap",
            headers={"Origin": "http://127.0.0.1:8000"},
        )
        widget_headers = {"Authorization": f"Bearer {bootstrap.json()['token']}"}
        upload = client.post(
            "/api/admin/upload",
            headers=admin_headers,
            files={"file": ("local.txt", "راهنمای پشتیبانی محلی".encode(), "text/plain")},
        )
        assert upload.status_code == 200
        conversation = client.post("/api/conversations", headers=widget_headers, json={}).json()
        message = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            headers=widget_headers,
            json={"message": "راهنمای پشتیبانی چیست؟", "client_message_id": "local-flow-1"},
        )
        handoff = client.post(
            f"/api/conversations/{conversation['id']}/handoff",
            headers=widget_headers,
            json={"reason": "بررسی انسانی"},
        )

        metrics = client.get("/api/admin/metrics?days=30", headers=admin_headers).json()
        conversations = client.get("/api/admin/conversations", headers=admin_headers).json()
        queue = client.get("/api/admin/handoffs", headers=admin_headers).json()
        transcript = client.get(
            f"/api/admin/conversations/{conversation['id']}", headers=admin_headers
        ).json()

        assert message.status_code == 200
        assert handoff.status_code == 200
        assert metrics["new_conversations"] == 1
        assert metrics["user_messages"] == 1
        assert metrics["assistant_messages"] == 1
        assert metrics["successful_generations"] == 1
        assert metrics["total_tokens"] == 19
        assert metrics["open_handoffs"] == 1
        assert conversations["conversations"][0]["user"]["external_id"] == "local-demo-user"
        assert len(transcript["messages"]) == 2
        assert queue["total"] == 1


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
        conversation_id = client.post("/api/conversations", headers=owner_headers, json={}).json()["id"]

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
    disabled_settings = test_settings.model_copy(update={"server_conversations_enabled": False})
    with create_client(disabled_settings) as client:
        response = client.post("/api/conversations", json={})

        assert response.status_code == 404


def test_handoff_admin_metrics_and_transcript(test_settings):
    admin_headers = {"Authorization": "Bearer test-admin-key"}
    with create_client(test_settings) as client:
        workspace_id = client.app.state.runtime.metadata.default_workspace().id
        widget_headers = _widget_headers(test_settings, workspace_id, "handoff-user")
        conversation = client.post("/api/conversations", headers=widget_headers, json={}).json()
        message = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            headers=widget_headers,
            json={"message": "کمک می‌خواهم", "client_message_id": "handoff-message"},
        )
        assert message.status_code == 200

        first = client.post(
            f"/api/conversations/{conversation['id']}/handoff",
            headers=widget_headers,
            json={"reason": "نیاز به کارشناس"},
        )
        duplicate = client.post(
            f"/api/conversations/{conversation['id']}/handoff",
            headers=widget_headers,
            json={"reason": "تکرار"},
        )
        assert first.json()["created"] is True
        assert duplicate.json()["created"] is False
        assert duplicate.json()["handoff"]["id"] == first.json()["handoff"]["id"]

        metrics = client.get("/api/admin/metrics?days=30", headers=admin_headers)
        conversations = client.get("/api/admin/conversations", headers=admin_headers)
        transcript = client.get(f"/api/admin/conversations/{conversation['id']}", headers=admin_headers)
        queue = client.get("/api/admin/handoffs", headers=admin_headers)
        assert metrics.status_code == 200
        assert metrics.json()["new_conversations"] == 1
        assert metrics.json()["successful_generations"] == 0
        assert metrics.json()["unreported_runs"] == 0
        assert conversations.json()["total"] == 1
        assert len(transcript.json()["messages"]) == 2
        assert queue.json()["total"] == 1

        updated = client.patch(
            f"/api/admin/handoffs/{first.json()['handoff']['id']}",
            headers=admin_headers,
            json={"status": "in_progress"},
        )
        assert updated.status_code == 200
        assert updated.json()["status"] == "in_progress"


def test_failed_provider_generation_is_recorded_in_metrics(test_settings):
    class FailingChatProvider:
        def generate(self, messages):
            raise RuntimeError("provider details must stay private")

    admin_headers = {"Authorization": "Bearer test-admin-key"}
    with create_client(test_settings, FailingChatProvider()) as client:
        upload = client.post(
            "/api/admin/upload",
            headers=admin_headers,
            files={"file": ("support.txt", "پشتیبانی سفارش".encode(), "text/plain")},
        )
        assert upload.status_code == 200
        workspace_id = client.app.state.runtime.metadata.default_workspace().id
        widget_headers = _widget_headers(test_settings, workspace_id, "failed-user")
        conversation = client.post("/api/conversations", headers=widget_headers, json={}).json()

        response = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            headers=widget_headers,
            json={"message": "پشتیبانی سفارش چگونه است؟", "client_message_id": "failure-1"},
        )
        metrics = client.get("/api/admin/metrics?days=30", headers=admin_headers).json()

        assert response.status_code == 503
        assert response.json()["detail"] == "سرویس پاسخ‌گویی موقتاً در دسترس نیست."
        assert metrics["successful_generations"] == 0
        assert metrics["failed_generations"] == 1
        assert "provider details" not in response.text


def test_admin_can_delete_uploaded_document(test_settings):
    headers = {"Authorization": "Bearer test-admin-key"}
    with create_client(test_settings) as client:
        upload = client.post(
            "/api/admin/upload",
            headers=headers,
            files={"file": ("delete.txt", "متن قابل حذف".encode(), "text/plain")},
        )
        document_id = upload.json()["document_id"]
        deleted = client.delete(f"/api/admin/documents/{document_id}", headers=headers)
        assert deleted.status_code == 204
        assert client.get("/api/admin/status", headers=headers).json()["active_documents"] == 0
