from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone

DEFAULT_WORKSPACE_ID = "00000000-0000-5000-8000-000000000001"
DEFAULT_WORKSPACE_SLUG = "default"

Migration = Callable[[sqlite3.Connection], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _migration_001_catalogue(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source_key TEXT NOT NULL UNIQUE,
            source_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_url TEXT,
            content_hash TEXT NOT NULL,
            embedding_fingerprint TEXT,
            chunk_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS document_chunks (
            vector_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            content_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            UNIQUE(document_id, chunk_index)
        )"""
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id)")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS ingestion_runs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            source_label TEXT NOT NULL,
            status TEXT NOT NULL,
            documents_processed INTEGER NOT NULL DEFAULT 0,
            chunks_created INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT
        )"""
    )


def _migration_002_embedding_fingerprint(connection: sqlite3.Connection) -> None:
    if "embedding_fingerprint" not in _columns(connection, "documents"):
        connection.execute("ALTER TABLE documents ADD COLUMN embedding_fingerprint TEXT")


def _migration_003_workspaces(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            slug TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )
    now = _utc_now()
    connection.execute(
        """INSERT OR IGNORE INTO workspaces (id, slug, name, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (DEFAULT_WORKSPACE_ID, DEFAULT_WORKSPACE_SLUG, "Default Workspace", now, now),
    )
    if "workspace_id" not in _columns(connection, "documents"):
        connection.execute("ALTER TABLE documents ADD COLUMN workspace_id TEXT REFERENCES workspaces(id)")
    connection.execute(
        "UPDATE documents SET workspace_id = ? WHERE workspace_id IS NULL",
        (DEFAULT_WORKSPACE_ID,),
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace_id)")
    connection.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_workspace_source
           ON documents(workspace_id, source_key)"""
    )
    if "workspace_id" not in _columns(connection, "ingestion_runs"):
        connection.execute(
            "ALTER TABLE ingestion_runs ADD COLUMN workspace_id TEXT REFERENCES workspaces(id)"
        )
    connection.execute(
        "UPDATE ingestion_runs SET workspace_id = ? WHERE workspace_id IS NULL",
        (DEFAULT_WORKSPACE_ID,),
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingestion_runs_workspace ON ingestion_runs(workspace_id)"
    )


def _migration_004_conversations(connection: sqlite3.Connection) -> None:
    statements = (
        """CREATE TABLE users (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            external_id TEXT NOT NULL,
            display_name TEXT,
            email TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(workspace_id, external_id)
        )""",
        """CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            user_id TEXT NOT NULL REFERENCES users(id),
            title TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_message_at TEXT
        )""",
        """CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(id),
            role TEXT NOT NULL,
            status TEXT NOT NULL,
            content TEXT,
            client_message_id TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            UNIQUE(conversation_id, client_message_id)
        )""",
        """CREATE TABLE message_citations (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
            vector_id TEXT,
            source_name TEXT,
            source_url TEXT,
            excerpt TEXT,
            position INTEGER NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(message_id, position)
        )""",
        """CREATE TABLE generation_runs (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            provider TEXT,
            model TEXT,
            status TEXT NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            latency_ms INTEGER,
            finish_reason TEXT,
            request_id TEXT,
            metadata_json TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT
        )""",
        """CREATE TABLE message_feedback (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(id),
            rating INTEGER,
            comment TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(message_id, user_id)
        )""",
        """CREATE TABLE message_reports (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(id),
            reason TEXT NOT NULL,
            details TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TEXT NOT NULL,
            resolved_at TEXT
        )""",
        """CREATE TABLE handoff_requests (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(id),
            status TEXT NOT NULL DEFAULT 'requested',
            reason TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT
        )""",
        """CREATE TABLE admins (
            id TEXT PRIMARY KEY,
            external_id TEXT NOT NULL UNIQUE,
            email TEXT,
            display_name TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE admin_workspace_memberships (
            admin_id TEXT NOT NULL REFERENCES admins(id) ON DELETE CASCADE,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            role TEXT NOT NULL DEFAULT 'member',
            created_at TEXT NOT NULL,
            PRIMARY KEY(admin_id, workspace_id)
        )""",
    )
    for statement in statements:
        connection.execute(statement)
    indexes = (
        "CREATE INDEX idx_users_workspace ON users(workspace_id)",
        "CREATE INDEX idx_conversations_workspace_user ON conversations(workspace_id, user_id, updated_at)",
        "CREATE INDEX idx_messages_conversation ON messages(workspace_id, conversation_id, created_at, id)",
        "CREATE INDEX idx_citations_message ON message_citations(workspace_id, message_id, position)",
        "CREATE INDEX idx_generation_message ON generation_runs(workspace_id, message_id)",
        "CREATE INDEX idx_handoffs_workspace_status ON handoff_requests(workspace_id, status)",
    )
    for statement in indexes:
        connection.execute(statement)


def _migration_005_message_parent(connection: sqlite3.Connection) -> None:
    if "parent_message_id" not in _columns(connection, "messages"):
        connection.execute("ALTER TABLE messages ADD COLUMN parent_message_id TEXT REFERENCES messages(id)")
    connection.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS uq_assistant_parent_message
           ON messages(parent_message_id)
           WHERE role = 'assistant' AND parent_message_id IS NOT NULL"""
    )


def _migration_006_workspace_catalogue_keys(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE documents_new (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            source_key TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_url TEXT,
            content_hash TEXT NOT NULL,
            embedding_fingerprint TEXT,
            chunk_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(workspace_id, source_key)
        )"""
    )
    connection.execute(
        """INSERT INTO documents_new
           (id, workspace_id, source_key, source_name, source_type, source_url,
            content_hash, embedding_fingerprint, chunk_count, updated_at)
           SELECT id, workspace_id, source_key, source_name, source_type, source_url,
                  content_hash, embedding_fingerprint, chunk_count, updated_at
           FROM documents"""
    )
    connection.execute(
        """CREATE TABLE document_chunks_new (
            vector_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents_new(id) ON DELETE CASCADE,
            content_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            UNIQUE(document_id, chunk_index)
        )"""
    )
    connection.execute(
        """INSERT INTO document_chunks_new
           (vector_id, document_id, content_hash, chunk_index)
           SELECT vector_id, document_id, content_hash, chunk_index FROM document_chunks"""
    )
    connection.execute("DROP TABLE document_chunks")
    connection.execute("DROP TABLE documents")
    connection.execute("ALTER TABLE documents_new RENAME TO documents")
    connection.execute("ALTER TABLE document_chunks_new RENAME TO document_chunks")
    connection.execute("CREATE INDEX idx_documents_workspace ON documents(workspace_id)")
    connection.execute("CREATE INDEX idx_chunks_document ON document_chunks(document_id)")


def _migration_007_admin_operations(connection: sqlite3.Connection) -> None:
    active = connection.execute(
        """SELECT conversation_id, GROUP_CONCAT(id) AS ids
           FROM handoff_requests WHERE status IN ('requested', 'in_progress')
           GROUP BY conversation_id HAVING COUNT(*) > 1"""
    ).fetchall()
    now = _utc_now()
    for row in active:
        ids = str(row[1]).split(",")
        keep = connection.execute(
            f"SELECT id FROM handoff_requests WHERE id IN ({','.join('?' for _ in ids)}) "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            ids,
        ).fetchone()[0]
        connection.execute(
            f"UPDATE handoff_requests SET status = 'cancelled', updated_at = ?, resolved_at = ? "
            f"WHERE id IN ({','.join('?' for _ in ids)}) AND id <> ?",
            (now, now, *ids, keep),
        )
    indexes = (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_handoffs_active_conversation ON handoff_requests(conversation_id) WHERE status IN ('requested', 'in_progress')",
        "CREATE INDEX IF NOT EXISTS idx_conversations_workspace_created ON conversations(workspace_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_conversations_workspace_activity ON conversations(workspace_id, last_message_at DESC, updated_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_messages_workspace_created ON messages(workspace_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_generation_workspace_created ON generation_runs(workspace_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_handoffs_workspace_queue ON handoff_requests(workspace_id, status, created_at, id)",
    )
    for statement in indexes:
        connection.execute(statement)


MIGRATIONS: tuple[tuple[int, str, Migration], ...] = (
    (1, "catalogue", _migration_001_catalogue),
    (2, "embedding_fingerprint", _migration_002_embedding_fingerprint),
    (3, "workspaces", _migration_003_workspaces),
    (4, "conversations", _migration_004_conversations),
    (5, "message_parent", _migration_005_message_parent),
    (6, "workspace_catalogue_keys", _migration_006_workspace_catalogue_keys),
    (7, "admin_operations", _migration_007_admin_operations),
)


def run_migrations(connection: sqlite3.Connection) -> None:
    """Apply each migration exactly once in its own transaction."""
    connection.execute(
        """CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )"""
    )
    applied = {int(row[0]) for row in connection.execute("SELECT version FROM schema_migrations")}
    for version, name, migration in MIGRATIONS:
        if version in applied:
            continue
        connection.execute("BEGIN IMMEDIATE")
        try:
            migration(connection)
            connection.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (version, name, _utc_now()),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
