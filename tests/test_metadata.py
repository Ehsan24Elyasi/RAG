import sqlite3
from datetime import datetime, timezone

from app.db.migrations import DEFAULT_WORKSPACE_ID
from app.services.metadata import MetadataStore, StoredDocument


def test_initialize_migrates_legacy_catalogue_without_data_loss(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                source_key TEXT NOT NULL UNIQUE,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_url TEXT,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE document_chunks (
                vector_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                content_hash TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                UNIQUE(document_id, chunk_index)
            );
            CREATE TABLE ingestion_runs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                source_label TEXT NOT NULL,
                status TEXT NOT NULL,
                documents_processed INTEGER NOT NULL DEFAULT 0,
                chunks_created INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                created_at TEXT NOT NULL,
                finished_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("doc-1", "faq", "FAQ", "text", None, "hash", 1, now),
        )
        connection.execute(
            "INSERT INTO document_chunks VALUES ('vec-1', 'doc-1', 'hash', 0)"
        )
        connection.execute(
            """INSERT INTO ingestion_runs
               VALUES ('run-1', 'file', 'faq.txt', 'completed', 1, 1, NULL, ?, ?)""",
            (now, now),
        )

    store = MetadataStore(database)
    store.initialize()
    store.initialize()

    with sqlite3.connect(database) as connection:
        document = connection.execute(
            "SELECT embedding_fingerprint, workspace_id FROM documents WHERE id = 'doc-1'"
        ).fetchone()
        run_workspace = connection.execute(
            "SELECT workspace_id FROM ingestion_runs WHERE id = 'run-1'"
        ).fetchone()[0]
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        chunk = connection.execute(
            "SELECT vector_id FROM document_chunks WHERE document_id = 'doc-1'"
        ).fetchone()[0]

    assert document == (None, DEFAULT_WORKSPACE_ID)
    assert run_workspace == DEFAULT_WORKSPACE_ID
    assert versions == [(1,), (2,), (3,), (4,), (5,), (6,)]
    assert chunk == "vec-1"
    assert store.document_for_source("faq") is not None


def test_default_workspace_and_new_catalogue_rows_are_scoped(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    workspace = store.default_workspace()
    document = StoredDocument(
        id=store.stable_document_id("guide"),
        source_key="guide",
        source_name="Guide",
        source_type="text",
        source_url=None,
        content_hash="hash",
        embedding_fingerprint=None,
        chunk_count=1,
        updated_at=datetime.now(timezone.utc),
    )
    store.replace_document(document, ["vector-1"])
    run = store.create_run("file", "guide.txt")

    with sqlite3.connect(store.database_path) as connection:
        document_workspace = connection.execute(
            "SELECT workspace_id FROM documents WHERE id = ?", (document.id,)
        ).fetchone()[0]
        run_workspace = connection.execute(
            "SELECT workspace_id FROM ingestion_runs WHERE id = ?", (run.id,)
        ).fetchone()[0]

    assert workspace.id == DEFAULT_WORKSPACE_ID
    assert document_workspace == workspace.id
    assert run_workspace == workspace.id


def test_user_upsert_is_stable_and_updates_profile(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    workspace_id = store.default_workspace().id

    first = store.upsert_user(
        workspace_id, "customer-1", display_name="First", metadata={"tier": "basic"}
    )
    second = store.upsert_user(
        workspace_id,
        "customer-1",
        display_name="Updated",
        email="customer@example.com",
        metadata={"tier": "plus"},
    )

    assert second.id == first.id
    assert second.display_name == "Updated"
    assert second.email == "customer@example.com"
    assert second.metadata == {"tier": "plus"}


def test_conversation_access_is_scoped_to_workspace_and_owner(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    workspace_id = store.default_workspace().id
    owner = store.upsert_user(workspace_id, "owner")
    other = store.upsert_user(workspace_id, "other")
    conversation = store.create_conversation(workspace_id, owner.id, title="Support")

    assert store.get_conversation(workspace_id, conversation.id, user_id=owner.id) == conversation
    assert store.get_conversation(workspace_id, conversation.id, user_id=other.id) is None
    assert store.get_conversation("not-this-workspace", conversation.id, user_id=owner.id) is None
    assert store.list_authorized_conversations(workspace_id, other.id) == []


def test_message_idempotency_and_bounded_history(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    workspace_id = store.default_workspace().id
    user = store.upsert_user(workspace_id, "customer")
    conversation = store.create_conversation(workspace_id, user.id)

    first = store.append_user_message(
        workspace_id,
        conversation.id,
        user.id,
        "hello",
        client_message_id="client-1",
    )
    duplicate = store.append_user_message(
        workspace_id,
        conversation.id,
        user.id,
        "ignored retry body",
        client_message_id="client-1",
    )
    assistant = store.create_assistant_placeholder(
        workspace_id, conversation.id, user_id=user.id
    )
    completed = store.complete_assistant_message(workspace_id, assistant.id, "welcome")
    store.append_user_message(
        workspace_id,
        conversation.id,
        user.id,
        "follow-up",
        client_message_id="client-2",
    )

    history = store.load_conversation_history(
        workspace_id, conversation.id, user_id=user.id, limit=2
    )

    assert duplicate.id == first.id
    assert duplicate.content == "hello"
    assert completed is not None and completed.status == "completed"
    assert [message.content for message in history] == ["welcome", "follow-up"]
    assert store.load_conversation_history(
        "wrong-workspace", conversation.id, user_id=user.id
    ) == []


def test_generation_usage_accepts_nullable_fields(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    workspace_id = store.default_workspace().id
    user = store.upsert_user(workspace_id, "customer")
    conversation = store.create_conversation(workspace_id, user.id)
    assistant = store.create_assistant_placeholder(workspace_id, conversation.id)

    generation = store.record_generation_usage(
        workspace_id,
        conversation.id,
        assistant.id,
        status="completed",
        provider="test-provider",
        model=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        metadata={"cached": False},
    )

    assert generation.model is None
    assert generation.prompt_tokens is None
    assert generation.completion_tokens is None
    assert generation.total_tokens is None
    assert generation.metadata == {"cached": False}
    assert generation.finished_at is not None
